//
//  Copyright © 2026 Yehor Smoliakov <egorsmkv@gmail.com>. All rights reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//

import Foundation
import MLX
import ReuseFastSwift

enum CLIError: Error, LocalizedError {
    case missingValue(String)
    case unknownFlag(String)
    case missingRequired(String)

    var errorDescription: String? {
        switch self {
        case .missingValue(let flag): return "Missing value for \(flag)"
        case .unknownFlag(let flag): return "Unknown flag: \(flag)"
        case .missingRequired(let name): return "Missing required argument: \(name)"
        }
    }
}

struct Options {
    var weightsPath: String?
    var inputPath: String?
    var outputPath: String?
    var chunkSizeSeconds: Float = 1.0
    var hopPortion: Float = 0.5
    var strict: Bool = false
    var dtype: DType = .bfloat16
    var printModelKeys: Bool = false
    var printCheckpointKeys: Bool = false
    var benchmarkScan: Bool = false
    var verifyScan: Bool = false
    var compareScan: Bool = false
    var benchmarkIterations: Int = 50
    var benchmarkBatch: Int = 82
    var benchmarkDim: Int = 256
    var benchmarkLength: Int = 51
    var benchmarkDState: Int = 16
}

func printUsage() {
    print("""
    reuse-fast-cli — fast MLX/Metal inference for RE-USE / SEMamba

    Required inference arguments:
      --weights PATH             Directory containing model_mlx.safetensors, or the .safetensors file itself
      --input noisy.wav          Input WAV. PCM 16/24/32-bit or IEEE float32, mono/stereo/multichannel
      --output clean.wav         Output mono float32 WAV

    Common options:
      --chunk-size-s 1.0         Chunk size in seconds for long files. Default: 1.0
      --hop-portion 0.5          Chunk overlap hop ratio. Default: 0.5
      --dtype bf16               Compute dtype: bf16 (default, faster) | fp32 (exact) | fp16 (fastest, unstable)
      --strict                   Verify all checkpoint keys and shapes while loading

    Diagnostics:
      --print-model-keys         Print expected SEMamba parameter keys and exit
      --print-checkpoint-keys    Print keys from --weights and exit
      --verify-scan              Check the fused selective-scan kernel against a reference and exit
      --benchmark-scan           Run the fused selective-scan microbenchmark and exit
      --iterations N             Benchmark iterations. Default: 50
      --scan-batch N             Benchmark batch/lane batch. Default: 82
      --scan-dim N               Benchmark dInner. Default: 256
      --scan-length N            Benchmark sequence length. Default: 51
      --scan-dstate N            Benchmark state size. Default: 16

    Example:
      reuse-fast-cli --weights re-use-mlx --input noisy.wav --output clean.wav

    Download weights:
      huggingface-cli download --local-dir re-use-mlx faraday/re-use-mlx
    """)
}

func parseOptions(_ args: [String]) throws -> Options {
    var options = Options()
    var i = 1

    func takeValue(_ flag: String) throws -> String {
        guard i + 1 < args.count else { throw CLIError.missingValue(flag) }
        i += 1
        return args[i]
    }

    while i < args.count {
        let arg = args[i]
        switch arg {
        case "-h", "--help":
            printUsage()
            exit(0)
        case "--weights":
            options.weightsPath = try takeValue(arg)
        case "--input":
            options.inputPath = try takeValue(arg)
        case "--output":
            options.outputPath = try takeValue(arg)
        case "--chunk-size-s":
            options.chunkSizeSeconds = Float(try takeValue(arg)) ?? options.chunkSizeSeconds
        case "--hop-portion":
            options.hopPortion = Float(try takeValue(arg)) ?? options.hopPortion
        case "--strict":
            options.strict = true
        case "--dtype":
            switch try takeValue(arg).lowercased() {
            case "fp32", "float32", "f32": options.dtype = .float32
            case "fp16", "float16", "f16": options.dtype = .float16
            case "bf16", "bfloat16": options.dtype = .bfloat16
            default: throw CLIError.missingValue("--dtype (expected fp32|fp16|bf16)")
            }
        case "--print-model-keys":
            options.printModelKeys = true
        case "--print-checkpoint-keys":
            options.printCheckpointKeys = true
        case "--benchmark-scan":
            options.benchmarkScan = true
        case "--verify-scan":
            options.verifyScan = true
        case "--compare-scan":
            options.compareScan = true
        case "--iterations":
            options.benchmarkIterations = Int(try takeValue(arg)) ?? options.benchmarkIterations
        case "--scan-batch":
            options.benchmarkBatch = Int(try takeValue(arg)) ?? options.benchmarkBatch
        case "--scan-dim":
            options.benchmarkDim = Int(try takeValue(arg)) ?? options.benchmarkDim
        case "--scan-length":
            options.benchmarkLength = Int(try takeValue(arg)) ?? options.benchmarkLength
        case "--scan-dstate":
            options.benchmarkDState = Int(try takeValue(arg)) ?? options.benchmarkDState
        default:
            throw CLIError.unknownFlag(arg)
        }
        i += 1
    }
    return options
}

func printExpectedModelKeys() {
    for key in ReuseModelLoader.modelParameterKeys() {
        print(key)
    }
}

func printCheckpointKeys(weightsPath: String) throws {
    let url = try ReuseModelLoader.resolveWeights(URL(fileURLWithPath: weightsPath))
    let arrays = try loadArrays(url: url)
    for key in arrays.keys.sorted() {
        let value = arrays[key]!
        print("\(key) \(value.shape) \(value.dtype)")
    }
}

/// Naive reference selective scan in pure MLX (channels-last), used to validate the
/// fused Metal kernel numerically.
func referenceScan(u: MLXArray, delta: MLXArray, A: MLXArray, Bvar: MLXArray, Cvar: MLXArray, D: MLXArray, z: MLXArray, bias: MLXArray) -> MLXArray {
    let batch = u.shape[0], length = u.shape[1], dim = u.shape[2], dState = A.shape[1]
    var state = MLXArray.zeros([batch, dim, dState], dtype: .float32)
    var outCols: [MLXArray] = []
    for t in 0 ..< length {
        let ut = u[0..., t, 0...]                        // [B, dim]
        var dt = delta[0..., t, 0...] + bias.reshaped(1, dim)
        dt = log(1 + exp(dt))                            // softplus
        let bt = Bvar[0..., t, 0...]                     // [B, dState]
        let ct = Cvar[0..., t, 0...]                     // [B, dState]
        let aBar = exp(expandedDimensions(dt, axis: -1) * A.reshaped(1, dim, dState))
        let bBar = expandedDimensions(dt, axis: -1) * expandedDimensions(bt, axis: 1) * expandedDimensions(ut, axis: -1)
        state = aBar * state + bBar
        var yt = (state * expandedDimensions(ct, axis: 1)).sum(axis: -1) + D.reshaped(1, dim) * ut
        let zt = z[0..., t, 0...]
        yt = yt * (zt / (1 + exp(-zt)))
        outCols.append(expandedDimensions(yt, axis: 1))
    }
    return concatenated(outCols, axis: 1)
}

func verifyScan() {
    let batch = 3, length = 17, dim = 8, dState = 16
    var rngState: UInt64 = 0x2545F4914F6CDD1D
    func rnd() -> Float {  // deterministic xorshift in [-1, 1]
        rngState ^= rngState << 13; rngState ^= rngState >> 7; rngState ^= rngState << 17
        return Float(rngState % 20001) / 10000.0 - 1.0
    }
    func arr(_ shape: [Int], _ scale: Float) -> MLXArray {
        let n = shape.reduce(1, *)
        return MLXArray((0 ..< n).map { _ in rnd() * scale }, shape)
    }
    let u = arr([batch, length, dim], 1.0)
    let delta = arr([batch, length, dim], 0.5)
    let A = -abs(arr([dim, dState], 1.0))
    let Bvar = arr([batch, length, dState], 0.2)
    let Cvar = arr([batch, length, dState], 0.3)
    let D = arr([dim], 1.0)
    let z = arr([batch, length, dim], 1.0)
    let bias = arr([dim], 0.1)

    let fused = ReuseSelectiveScan.fused(u: u, delta: delta, A: A, Bvar: Bvar, Cvar: Cvar, D: D, z: z, deltaBias: bias, deltaSoftplus: true, outputDType: .float32)
    let ref = referenceScan(u: u, delta: delta, A: A, Bvar: Bvar, Cvar: Cvar, D: D, z: z, bias: bias)
    eval(fused, ref)
    let diff = abs(fused - ref).max().item(Float.self)
    let denom = abs(ref).max().item(Float.self)
    print(String(format: "verify-scan: max|fused-ref|=%.3e  max|ref|=%.3e  rel=%.3e", diff, denom, diff / denom))

    // Reverse scan must equal flip(scan(flip(input))) along the time axis.
    func flipT(_ x: MLXArray) -> MLXArray {
        let idx = MLXArray(Array((0 ..< x.dim(1)).reversed()))
        return contiguous(x[0..., idx, 0...])
    }
    let rev = ReuseSelectiveScan.fused(u: u, delta: delta, A: A, Bvar: Bvar, Cvar: Cvar, D: D, z: z, deltaBias: bias, deltaSoftplus: true, reverse: true, outputDType: .float32)
    let revRef = flipT(ReuseSelectiveScan.fused(u: flipT(u), delta: flipT(delta), A: A, Bvar: flipT(Bvar), Cvar: flipT(Cvar), D: D, z: flipT(z), deltaBias: bias, deltaSoftplus: true, outputDType: .float32))
    eval(rev, revRef)
    let rdiff = abs(rev - revRef).max().item(Float.self)
    print(String(format: "verify-scan reverse: max|rev - flip(fwd(flip))|=%.3e", rdiff))
}

/// Parallel selective scan via a Hillis-Steele associative scan.
///
/// The recurrence `state_t = aBar_t * state_{t-1} + bBar_t` is an affine map, and
/// affine maps compose associatively: combine((a_l,b_l),(a_r,b_r)) = (a_r*a_l, a_r*b_l+b_r).
/// So an inclusive scan over time can run in O(log L) depth instead of O(L).
///
/// The catch: unlike the register kernel (which keeps `state[dState]` in registers per
/// lane), this must materialize aBar/bBar for every (b, t, d, n) — a `[B, L, dim, dState]`
/// tensor. For RE-USE's time-Mamba that is 41*2000*256*16 ≈ 1.3 GB per tensor.
func parallelScan(u: MLXArray, delta: MLXArray, A: MLXArray, Bvar: MLXArray, Cvar: MLXArray, D: MLXArray, z: MLXArray, bias: MLXArray) -> MLXArray {
    let batch = u.shape[0], length = u.shape[1], dim = u.shape[2], dState = A.shape[1]

    let dt = log(1 + exp(delta + bias.reshaped(1, 1, dim)))          // softplus, [B,L,dim]
    let dtE = expandedDimensions(dt, axis: -1)                       // [B,L,dim,1]
    var aCum = exp(dtE * A.reshaped(1, 1, dim, dState))              // [B,L,dim,dState]
    var bCum = dtE * expandedDimensions(u, axis: -1) * expandedDimensions(Bvar, axis: 2) // [B,L,dim,dState]

    // Inclusive Hillis-Steele scan over the time axis (1).
    var shift = 1
    while shift < length {
        let idA = MLXArray.ones([batch, shift, dim, dState], dtype: aCum.dtype)
        let idB = MLXArray.zeros([batch, shift, dim, dState], dtype: bCum.dtype)
        let aPrev = concatenated([idA, aCum[0..., 0 ..< (length - shift), 0..., 0...]], axis: 1)
        let bPrev = concatenated([idB, bCum[0..., 0 ..< (length - shift), 0..., 0...]], axis: 1)
        bCum = aCum * bPrev + bCum
        aCum = aCum * aPrev
        shift *= 2
    }

    // state_t = bCum (transform applied to state_0 = 0).
    var y = (bCum * expandedDimensions(Cvar, axis: 2)).sum(axis: -1) + D.reshaped(1, 1, dim) * u
    y = y * (z / (1 + exp(-z)))
    return y
}

func compareScan(_ options: Options) {
    var rngState: UInt64 = 0x2545F4914F6CDD1D
    func rnd() -> Float {
        rngState ^= rngState << 13; rngState ^= rngState >> 7; rngState ^= rngState << 17
        return Float(rngState % 20001) / 10000.0 - 1.0
    }
    func arr(_ shape: [Int], _ scale: Float) -> MLXArray {
        MLXArray((0 ..< shape.reduce(1, *)).map { _ in rnd() * scale }, shape)
    }

    // 1) Correctness on a small shape: parallel vs fused kernel.
    do {
        let (b, l, dim, n) = (3, 33, 8, 16)
        let u = arr([b, l, dim], 1.0), delta = arr([b, l, dim], 0.5)
        let A = -abs(arr([dim, n], 1.0)), Bvar = arr([b, l, n], 0.2), Cvar = arr([b, l, n], 0.3)
        let D = arr([dim], 1.0), z = arr([b, l, dim], 1.0), bias = arr([dim], 0.1)
        let fused = ReuseSelectiveScan.fused(u: u, delta: delta, A: A, Bvar: Bvar, Cvar: Cvar, D: D, z: z, deltaBias: bias, deltaSoftplus: true, outputDType: .float32)
        let par = parallelScan(u: u, delta: delta, A: A, Bvar: Bvar, Cvar: Cvar, D: D, z: z, bias: bias)
        eval(fused, par)
        let diff = abs(fused - par).max().item(Float.self)
        let denom = abs(fused).max().item(Float.self)
        print(String(format: "compare-scan correctness: max|parallel-fused|=%.3e rel=%.3e", diff, diff / denom))
    }

    // 2) Timing at RE-USE's real time-Mamba shape.
    let batch = options.benchmarkBatch, dim = options.benchmarkDim
    let length = options.benchmarkLength, dState = options.benchmarkDState
    let iters = max(1, options.benchmarkIterations)
    print("timing shape: [B=\(batch), L=\(length), dim=\(dim), dState=\(dState)]  (materialized aBar ≈ \(batch*length*dim*dState*4/1_000_000) MB/tensor)")

    let u = arr([batch, length, dim], 0.1), delta = arr([batch, length, dim], -1.0)
    let A = -abs(arr([dim, dState], 0.1)), Bvar = arr([batch, length, dState], 0.1), Cvar = arr([batch, length, dState], 0.1)
    let D = arr([dim], 1.0), z = arr([batch, length, dim], 0.1), bias = arr([dim], 0.1)

    func time(_ label: String, _ f: () -> MLXArray) {
        eval(f())
        let start = DispatchTime.now().uptimeNanoseconds
        for _ in 0 ..< iters { eval(f()) }
        let ms = Double(DispatchTime.now().uptimeNanoseconds - start) / Double(iters) / 1_000_000.0
        print(String(format: "  %@: %.2f ms/run", label, ms))
    }
    time("fused register kernel", { ReuseSelectiveScan.fused(u: u, delta: delta, A: A, Bvar: Bvar, Cvar: Cvar, D: D, z: z, deltaBias: bias, deltaSoftplus: true, outputDType: .float32) })
    time("parallel (Hillis-Steele)", { parallelScan(u: u, delta: delta, A: A, Bvar: Bvar, Cvar: Cvar, D: D, z: z, bias: bias) })
}

func benchmarkScan(_ options: Options) {
    let batch = options.benchmarkBatch
    let dim = options.benchmarkDim
    let length = options.benchmarkLength
    let dState = options.benchmarkDState
    let iterations = max(1, options.benchmarkIterations)

    precondition(dState <= ReuseSelectiveScan.maxRegisterState, "dState is too large for register kernel")

    // Channels-last: u/delta [B, L, dInner], Bvar/Cvar [B, L, dState].
    let u = MLXArray.ones([batch, length, dim], dtype: .float32) * 0.01
    let delta = MLXArray.ones([batch, length, dim], dtype: .float32) * -4.0
    let A = -MLXArray.ones([dim, dState], dtype: .float32) * 0.05
    let Bvar = MLXArray.ones([batch, length, dState], dtype: .float32) * 0.02
    let Cvar = MLXArray.ones([batch, length, dState], dtype: .float32) * 0.03
    let Dvec = MLXArray.ones([dim], dtype: .float32)

    let warmup = ReuseSelectiveScan.fused(
        u: u,
        delta: delta,
        A: A,
        Bvar: Bvar,
        Cvar: Cvar,
        D: Dvec,
        z: nil,
        deltaBias: nil,
        deltaSoftplus: true
    )
    eval(warmup)

    let start = DispatchTime.now().uptimeNanoseconds
    var y = warmup
    for _ in 0 ..< iterations {
        y = ReuseSelectiveScan.fused(
            u: u,
            delta: delta,
            A: A,
            Bvar: Bvar,
            Cvar: Cvar,
            D: Dvec,
            z: nil,
            deltaBias: nil,
            deltaSoftplus: true
        )
        eval(y)
    }
    let end = DispatchTime.now().uptimeNanoseconds

    let ms = Double(end - start) / Double(iterations) / 1_000_000.0
    print("scan shape: u=[\(batch), \(dim), \(length)], state=\(dState)")
    print(String(format: "fused selective scan: %.3f ms/run", ms))
    print("output: shape=\(y.shape), dtype=\(y.dtype)")
}

func runInference(_ options: Options) throws {
    guard let weightsPath = options.weightsPath else { throw CLIError.missingRequired("--weights") }
    guard let inputPath = options.inputPath else { throw CLIError.missingRequired("--input") }
    guard let outputPath = options.outputPath else { throw CLIError.missingRequired("--output") }

    let inputURL = URL(fileURLWithPath: inputPath)
    let outputURL = URL(fileURLWithPath: outputPath)
    let weightsURL = URL(fileURLWithPath: weightsPath)

    print("loading wav: \(inputURL.path)")
    let audio = try WAVIO.readMono(url: inputURL)
    print("audio: \(audio.samples.count) samples @ \(audio.sampleRate) Hz")

    print("loading weights: \(try ReuseModelLoader.resolveWeights(weightsURL).path)")
    let model = try ReuseModelLoader.loadSEMamba(weightsAt: weightsURL, strict: options.strict, dtype: options.dtype)
    let enhancer = REUSEEnhancer(model: model)

    print("running enhancement")
    let output = enhancer.enhance(
        waveform: audio.mlxMonoBatch,
        sampleRate: audio.sampleRate,
        chunkSizeSeconds: options.chunkSizeSeconds,
        hopPortion: options.hopPortion
    )
    eval(output)

    print("writing wav: \(outputURL.path)")
    try WAVIO.writeMonoFloat32(url: outputURL, array: output, sampleRate: audio.sampleRate)
    print("done")
}

do {
    let options = try parseOptions(CommandLine.arguments)

    if options.printModelKeys {
        printExpectedModelKeys()
        exit(0)
    }

    if options.printCheckpointKeys {
        guard let weightsPath = options.weightsPath else { throw CLIError.missingRequired("--weights") }
        try printCheckpointKeys(weightsPath: weightsPath)
        exit(0)
    }

    if options.verifyScan {
        verifyScan()
        exit(0)
    }

    if options.compareScan {
        compareScan(options)
        exit(0)
    }

    if options.benchmarkScan {
        benchmarkScan(options)
        exit(0)
    }

    try runInference(options)
} catch {
    fputs("error: \(error.localizedDescription)\n\n", stderr)
    printUsage()
    exit(1)
}
