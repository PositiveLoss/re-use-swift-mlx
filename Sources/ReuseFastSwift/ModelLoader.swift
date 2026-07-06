import Foundation
import MLX
import MLXNN

public enum ReuseLoadError: Error, LocalizedError {
    case noSafetensorsFile(URL)

    public var errorDescription: String? {
        switch self {
        case .noSafetensorsFile(let url):
            return "No model_mlx.safetensors or model.safetensors found in \(url.path)"
        }
    }
}

public enum ReuseModelLoader {
    public static func resolveWeights(_ url: URL) throws -> URL {
        var isDirectory: ObjCBool = false
        if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue {
            let candidates = [
                url.appendingPathComponent("model_mlx.safetensors"),
                url.appendingPathComponent("model.safetensors"),
            ]
            for candidate in candidates where FileManager.default.fileExists(atPath: candidate.path) {
                return candidate
            }
            throw ReuseLoadError.noSafetensorsFile(url)
        }
        return url
    }

    /// Key compatibility for common converted names. The Faraday checkpoint is already
    /// documented as MLX-Swift-compatible; this function keeps the loader tolerant of
    /// Python-style or camelCase variants.
    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var out = [String: MLXArray]()

        func add(_ key: String, _ value: MLXArray) {
            out[key] = value
        }

        for (rawKey, value) in weights {
            var k = rawKey
            if k.hasPrefix("model.") { k.removeFirst("model.".count) }
            if k.hasPrefix("module.") { k.removeFirst("module.".count) }

            // CamelCase -> Python/checkpoint style aliases used by this package.
            let replacements: [(String, String)] = [
                ("denseEncoder.conv1.", "dense_encoder.dense_conv_1."),
                ("denseEncoder.conv2.", "dense_encoder.dense_conv_2."),
                ("denseEncoder.", "dense_encoder."),
                ("denseBlock.layers.", "dense_block.dense_block."),
                ("maskDecoder.", "mask_decoder."),
                ("phaseDecoder.", "phase_decoder."),
                ("upConv1.", "up_conv1."),
                ("upConv2.", "up_conv2."),
                ("finalConv.", "final_conv."),
                ("phaseConvR.", "phase_conv_r."),
                ("phaseConvI.", "phase_conv_i."),
                ("timeMamba.forward.", "time_mamba.forward_blocks."),
                ("timeMamba.backward.", "time_mamba.backward_blocks."),
                ("freqMamba.forward.", "freq_mamba.forward_blocks."),
                ("freqMamba.backward.", "freq_mamba.backward_blocks."),
                ("timeMamba.", "time_mamba."),
                ("freqMamba.", "freq_mamba."),
                ("outputProj.", "output_proj."),
                ("inProj.", "in_proj."),
                ("xProj.", "x_proj."),
                ("dtProj.", "dt_proj."),
                ("outProj.", "out_proj."),
                ("ALog", "A_log"),
                ("A_Log", "A_log"),
                ("tfMamba.", "TSMamba."),
                
                // Map PyTorch ConvNormAct Sequential layers to MLX property names
                (".layers.0.", ".conv."),
                (".layers.1.", ".norm."),
                (".layers.2.", ".act."),
            ]
            for (from, to) in replacements {
                k = k.replacingOccurrences(of: from, with: to)
            }

            add(k, value)
        }

        return out
    }

    public static func loadSEMamba(
        weightsAt url: URL,
        strict: Bool = false,
        dtype: DType = .float32
    ) throws -> SEMamba {
        let weightsURL = try resolveWeights(url)
        let rawWeights = try loadArrays(url: weightsURL)
        var weights = sanitize(weights: rawWeights)

        // Keep compute numerics simple and stable. The Faraday artifact is fp32-sized;
        // if a future half checkpoint is used, this casts model arrays on load.
        if dtype != .float32 {
            weights = weights.mapValues { value in
                value.dtype.isFloatingPoint ? value.asType(dtype) : value
            }
        }

        let model = SEMamba()
        model.computeDType = dtype

        let modelKeys = Set(model.parameters().flattened().map { $0.0 })
        let weightKeys = Set(weights.keys)
        let missing = modelKeys.subtracting(weightKeys)
        let extra = weightKeys.subtracting(modelKeys)
        
        if !missing.isEmpty {
            print("[warning] Model has missing keys: \(missing.count) keys (e.g. \(missing.prefix(5)))")
        }
        if !extra.isEmpty {
            print("[warning] Checkpoint has extra keys: \(extra.count) keys (e.g. \(extra.prefix(5)))")
        }

        let flatModel = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        for (k, v) in weights {
            if let mv = flatModel[k], mv.shape != v.shape {
                print("[error] Shape mismatch for \(k): model \(mv.shape), weights \(v.shape)")
            }
        }

        try model.update(
            parameters: ModuleParameters.unflattened(weights),
            verify: strict ? .all : .shapeMismatch
        )
        model.train(false)
        eval(model)
        return model
    }

    public static func modelParameterKeys() -> [String] {
        let model = SEMamba()
        return model.parameters().flattened().map { $0.0 }.sorted()
    }
}
