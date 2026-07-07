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
import MLXFFT

private let stftEPS: Float = 1e-10
private let baseSR = 8000
private let baseNFFT = 320
private let baseHop = 40
private let baseWin = 320
private let realtimeBaseHop = 160

public struct STFTParams: Sendable {
    public let nFFT: Int
    public let hop: Int
    public let win: Int
}

private func makeEven(_ value: Int) -> Int {
    value % 2 == 0 ? value : value + 1
}

public func stftParams(for sampleRate: Int) -> STFTParams {
    STFTParams(
        nFFT: baseNFFT,
        hop: baseHop,
        win: baseWin
    )
}

public func realtimeSTFTParams(for sampleRate: Int) -> STFTParams {
    STFTParams(
        nFFT: makeEven(baseNFFT * sampleRate / baseSR),
        hop: makeEven(realtimeBaseHop * sampleRate / baseSR),
        win: makeEven(baseWin * sampleRate / baseSR)
    )
}

public func hannWindow(_ winSize: Int) -> MLXArray {
    let n = arange(winSize, dtype: .float32)
    return 0.5 - 0.5 * cos(2.0 * Float.pi * n / Float(winSize))
}

private func reflectPad(_ wave: MLXArray, pad: Int) -> MLXArray {
    guard pad > 0 else { return wave }
    let total = wave.dim(-1)
    precondition(total > pad + 1, "input too short for reflect padding; use a longer chunk")

    let leftIdx = MLXArray(Array((1 ... pad).reversed()))
    let rightIdx = MLXArray(Array(((total - pad - 1) ..< (total - 1)).reversed()))
    let left = wave[0..., leftIdx]
    let right = wave[0..., rightIdx]
    return concatenated([left, wave, right], axis: -1)
}

private func frame(_ signal: MLXArray, frameLength: Int, hop: Int) -> MLXArray {
    let total = signal.dim(-1)
    precondition(total >= frameLength, "signal is shorter than one STFT frame")
    let nFrames = 1 + (total - frameLength) / hop
    let starts = expandedDimensions(arange(nFrames), axis: 1) * hop
    let offsets = expandedDimensions(arange(frameLength), axis: 0)
    let idx = starts + offsets
    return signal[0..., idx]
}

private func windowForFFT(nFFT: Int, win: Int) -> MLXArray {
    var window = hannWindow(win)
    if win < nFFT {
        let left = (nFFT - win) / 2
        let right = nFFT - win - left
        window = concatenated([
            MLXArray.zeros([left], dtype: .float32),
            window,
            MLXArray.zeros([right], dtype: .float32),
        ])
    }
    return window
}

private func compressMag(_ mag: MLXArray, compressFactor: String) -> MLXArray {
    switch compressFactor {
    case "log1p", "relu_log1p", "signed_log1p":
        return log1p(mag)
    default:
        return mag
    }
}

private func expandMag(_ mag: MLXArray, compressFactor: String) -> MLXArray {
    switch compressFactor {
    case "log1p":
        return expm1(mag)
    case "signed_log1p":
        return sign(mag) * expm1(abs(mag))
    case "relu_log1p":
        return expm1(maximum(mag, 0.0))
    default:
        return maximum(mag, 0.0)
    }
}

/// Forward STFT for RE-USE. Returns compressed magnitude and phase, both `[B,F,T]`.
public func magPhaseSTFT(
    wave input: MLXArray,
    nFFT: Int,
    hop: Int,
    win: Int,
    compressFactor: String = "relu_log1p",
    center: Bool = true,
    addEps: Bool = false
) -> (MLXArray, MLXArray) {
    let squeeze = input.ndim == 1
    var wave = squeeze ? expandedDimensions(input, axis: 0) : input
    wave = wave.asType(.float32)

    if center {
        wave = reflectPad(wave, pad: nFFT / 2)
    }

    let window = windowForFFT(nFFT: nFFT, win: win)
    let frames = frame(wave, frameLength: nFFT, hop: hop)
    let windowed = frames * window.reshaped(1, 1, -1)

    var spec = rfft(windowed, n: nFFT, axis: -1) // [B, frames, freqs]
    spec = spec.transposed(0, 2, 1)              // [B, freqs, frames]

    let real = spec.realPart()
    let imag = spec.imaginaryPart()
    let mag: MLXArray
    let pha: MLXArray
    if addEps {
        mag = sqrt(real * real + imag * imag + stftEPS)
        pha = atan2(imag + stftEPS, real + stftEPS)
    } else {
        mag = sqrt(real * real + imag * imag)
        pha = atan2(imag, real)
    }

    var cmag = compressMag(mag, compressFactor: compressFactor)
    var outPha = pha
    if squeeze {
        cmag = cmag[0]
        outPha = outPha[0]
    }
    return (cmag, outPha)
}

private func overlapAdd(frames: MLXArray, hop: Int, outLen: Int) -> MLXArray {
    let b = frames.dim(0)
    let nFrames = frames.dim(1)
    let frameLen = frames.dim(2)
    
    // Evaluate frames to pull them to the CPU
    eval(frames)
    let framesData = frames.asArray(Float.self)
    
    var outData = [Float](repeating: 0.0, count: b * outLen)
    
    for batch in 0 ..< b {
        for i in 0 ..< nFrames {
            let outOffset = batch * outLen + i * hop
            let frameOffset = batch * nFrames * frameLen + i * frameLen
            
            for j in 0 ..< frameLen {
                outData[outOffset + j] += framesData[frameOffset + j]
            }
        }
    }
    
    return MLXArray(outData, [b, outLen])
}

/// Inverse STFT for RE-USE. Inputs are magnitude/phase `[B,F,T]`.
public func magPhaseISTFT(
    mag inputMag: MLXArray,
    phase inputPhase: MLXArray,
    nFFT: Int,
    hop: Int,
    win: Int,
    compressFactor: String = "relu_log1p",
    center: Bool = true
) -> MLXArray {
    let squeeze = inputMag.ndim == 2
    var mag = squeeze ? expandedDimensions(inputMag, axis: 0) : inputMag
    var pha = squeeze ? expandedDimensions(inputPhase, axis: 0) : inputPhase
    mag = expandMag(mag.asType(.float32), compressFactor: compressFactor)
    pha = pha.asType(.float32)

    let real = mag * cos(pha)
    let imag = mag * sin(pha)
    var spec = real + imag.asImaginary() // [B,F,T]
    spec = spec.transposed(0, 2, 1)      // [B,T,F]

    var frames = MLXFFT.irfft(spec, n: nFFT, axis: -1) // [B,T,nFFT]
    let window = windowForFFT(nFFT: nFFT, win: win)
    frames = frames * window.reshaped(1, 1, -1)

    // Calculate NOLA (Non-Zero Overlap Add) normalization factor for Hann window
    // PyTorch's istft divides by the sum of squared windows.
    // For a Hann window, the sum of shifted squares is exactly (win / hop) * 0.375
    let windowSqSum = Float(win) / Float(hop) * 0.375

    let nFrames = frames.dim(1)
    let outLen = nFFT + hop * (nFrames - 1)
    var signal = overlapAdd(frames: frames, hop: hop, outLen: outLen)
    signal = signal / windowSqSum

    if center {
        let pad = nFFT / 2
        signal = signal[0..., pad ..< (outLen - pad)]
    }
    return squeeze ? signal[0] : signal
}

/// Zero out spectral frames dominated by exact zero amplitude in the expanded domain.
public func sweepArtifactFilter(_ amp: MLXArray) -> MLXArray {
    let squeeze = amp.ndim == 2
    var x = squeeze ? expandedDimensions(amp, axis: 0) : amp
    let mag = expm1(maximum(x, 0.0))
    let nFreq = Float(mag.dim(1))
    let zeroPortion = (mag .== 0).asType(.float32).sum(axis: 1) / nFreq
    let keep = (zeroPortion .<= 0.5).asType(x.dtype)
    x = x * keep.reshaped(keep.dim(0), 1, keep.dim(1))
    return squeeze ? x[0] : x
}

private func padOrTrimToMatch(reference: MLXArray, target: MLXArray, padValue: Float = 1e-8) -> MLXArray {
    let refLen = reference.dim(-1)
    let tgtLen = target.dim(-1)
    if tgtLen == refLen { return target }
    if tgtLen > refLen { return target[0..., 0 ..< refLen] }
    let pad = MLXArray.ones([target.dim(0), refLen - tgtLen], dtype: target.dtype) * padValue
    return concatenated([target, pad], axis: -1)
}

/// Chunked Hann-window OLA over a long waveform.
public func chunkedHannOLA(
    wave input: MLXArray,
    chunkSize: Int,
    hopPortion: Float = 0.5,
    padValue: Float = 1e-8,
    batchSize: Int = 4,
    process: (MLXArray) -> MLXArray
) -> MLXArray {
    let squeeze = input.ndim == 1
    var wave = squeeze ? expandedDimensions(input, axis: 0) : input
    wave = wave.asType(.float32)

    let total = wave.dim(-1)
    if total <= chunkSize {
        var chunk = wave
        if total < chunkSize {
            let pad = MLXArray.zeros([wave.dim(0), chunkSize - total], dtype: wave.dtype)
            chunk = concatenated([chunk, pad], axis: -1)
        }
        var out = process(chunk)
        if total < chunkSize {
            out = out[0..., 0 ..< total]
        }
        eval(out)
        return squeeze ? out[0] : out
    }

    let hopLength = max(1, Int(Float(chunkSize) * hopPortion))
    let nChunks = max(1, Int(ceil(Float(total - chunkSize) / Float(hopLength))) + 1)
    let window = hannWindow(chunkSize)

    var enhanced = MLXArray.zeros(wave.shape, dtype: wave.dtype)
    var windowSum = MLXArray.zeros(wave.shape, dtype: wave.dtype)

    var i = 0
    while i < nChunks {
        var batchedInputs: [MLXArray] = []
        var chunkStarts: [Int] = []
        var chunkLengths: [Int] = []
        
        let batchEnd = min(i + batchSize, nChunks)
        for b in i ..< batchEnd {
            let start = b * hopLength
            let end = min(start + chunkSize, total)
            let length = end - start
            guard length >= 2 else { continue }
            
            var chunk = wave[0..., start ..< end]
            if length < chunkSize {
                let pad = MLXArray.zeros([wave.dim(0), chunkSize - length], dtype: wave.dtype)
                chunk = concatenated([chunk, pad], axis: -1)
            }
            // Squeeze to 1D: [chunkSize]
            batchedInputs.append(chunk[0])
            chunkStarts.append(start)
            chunkLengths.append(length)
        }
        
        if batchedInputs.isEmpty {
            i = batchEnd
            continue
        }
        
        // Process batch
        let batchInput = stacked(batchedInputs, axis: 0) // [B, chunkSize]
        let batchOut = process(batchInput) // [B, chunkSize]
        eval(batchOut)
        
        for b in 0 ..< batchedInputs.count {
            let start = chunkStarts[b]
            let length = chunkLengths[b]
            
            var out = expandedDimensions(batchOut[b], axis: 0) // [1, chunkSize]
            if length < chunkSize {
                out = out[0..., 0 ..< length]
            }
            
            let refChunk = wave[0..., start ..< (start + length)]
            out = padOrTrimToMatch(reference: refChunk, target: out, padValue: padValue)
            
            let seg = out.dim(-1)
            let wSlice = window[0 ..< seg]
            let cols = arange(seg)
            enhanced[0..., start + cols] += out * wSlice.reshaped(1, -1)
            windowSum[0..., start + cols] += wSlice.reshaped(1, -1)
        }
        
        i = batchEnd
    }

    let mask = windowSum .> 1e-8
    enhanced = which(mask, enhanced / maximum(windowSum, stftEPS), enhanced)
    return squeeze ? enhanced[0] : enhanced
}
