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

    public static func sanitizeRealtime(weights: [String: MLXArray]) -> [String: MLXArray] {
        var out = [String: MLXArray]()

        func convert(_ key: String, _ value: MLXArray) -> MLXArray {
            if key.hasSuffix(".conv1d.weight"), value.shape.count == 3, value.shape[1] == 1 {
                return value.transposed(0, 2, 1)
            }
            if key.hasSuffix(".weight"), value.shape.count == 4 {
                return value.transposed(0, 2, 3, 1)
            }
            return value
        }

        for (rawKey, value) in weights {
            var k = rawKey
            if k.hasPrefix("model.") { k.removeFirst("model.".count) }
            if k.hasPrefix("module.") { k.removeFirst("module.".count) }

            for i in 0 ..< 4 {
                k = k.replacingOccurrences(of: ".dense_block.dense_block.\(i).1.", with: ".dense_block.dense_block.\(i).conv.")
                k = k.replacingOccurrences(of: ".dense_block.dense_block.\(i).2.norm.", with: ".dense_block.dense_block.\(i).norm.")
                k = k.replacingOccurrences(of: ".dense_block.dense_block.\(i).3.", with: ".dense_block.dense_block.\(i).act.")
                k = k.replacingOccurrences(of: ".dense_block.dense_block.\(i).norm.weight", with: ".dense_block.dense_block.\(i).norm.norm.weight")
                k = k.replacingOccurrences(of: ".dense_block.dense_block.\(i).norm.bias", with: ".dense_block.dense_block.\(i).norm.norm.bias")
            }

            let branchPrefixes = [
                "dense_encoder.dense_conv_1_1",
                "dense_encoder.dense_conv_1_2",
                "dense_encoder.dense_conv_1_3",
                "dense_encoder.dense_conv_2",
            ]
            for prefix in branchPrefixes {
                k = k.replacingOccurrences(of: "\(prefix).0.", with: "\(prefix).conv.")
                k = k.replacingOccurrences(of: "\(prefix).1.norm.", with: "\(prefix).norm.")
                k = k.replacingOccurrences(of: "\(prefix).2.", with: "\(prefix).act.")
                k = k.replacingOccurrences(of: "\(prefix).norm.weight", with: "\(prefix).norm.norm.weight")
                k = k.replacingOccurrences(of: "\(prefix).norm.bias", with: "\(prefix).norm.norm.bias")
            }

            let upPrefixes = ["up_conv1", "up_conv2"]
            for prefix in upPrefixes {
                k = k.replacingOccurrences(of: ".\(prefix).0.conv.", with: ".\(prefix).conv.conv.")
                k = k.replacingOccurrences(of: ".\(prefix).1.norm.", with: ".\(prefix).norm.")
                k = k.replacingOccurrences(of: ".\(prefix).2.", with: ".\(prefix).act.")
                k = k.replacingOccurrences(of: ".\(prefix).norm.weight", with: ".\(prefix).norm.norm.weight")
                k = k.replacingOccurrences(of: ".\(prefix).norm.bias", with: ".\(prefix).norm.norm.bias")
            }

            // SPUp uses `conv` for the sub-pixel wrapper, then another `conv` for
            // the inner Conv2d. Keep decoder up-conv keys aligned with that nesting.
            k = k.replacingOccurrences(of: ".up_conv1.conv.weight", with: ".up_conv1.conv.conv.weight")
            k = k.replacingOccurrences(of: ".up_conv1.conv.bias", with: ".up_conv1.conv.conv.bias")
            k = k.replacingOccurrences(of: ".up_conv2.conv.weight", with: ".up_conv2.conv.conv.weight")
            k = k.replacingOccurrences(of: ".up_conv2.conv.bias", with: ".up_conv2.conv.conv.bias")

            out[k] = convert(k, value)
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

    public static func loadRealtimeSEMamba(
        weightsAt url: URL,
        strict: Bool = false,
        dtype: DType = .float32
    ) throws -> RealtimeSEMamba {
        let weightsURL = try resolveWeights(url)
        let rawWeights = try loadArrays(url: weightsURL)
        var weights = sanitizeRealtime(weights: rawWeights)

        if dtype != .float32 {
            weights = weights.mapValues { value in
                value.dtype.isFloatingPoint ? value.asType(dtype) : value
            }
        }

        let model = RealtimeSEMamba()
        model.computeDType = dtype

        let modelKeys = Set(model.parameters().flattened().map { $0.0 })
        let weightKeys = Set(weights.keys)
        let missing = modelKeys.subtracting(weightKeys)
        let extra = weightKeys.subtracting(modelKeys)

        if !missing.isEmpty {
            print("[warning] Realtime model has missing keys: \(missing.count) keys (e.g. \(missing.prefix(5)))")
        }
        if !extra.isEmpty {
            print("[warning] Realtime checkpoint has extra keys: \(extra.count) keys (e.g. \(extra.prefix(5)))")
        }

        let flatModel = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        for (k, v) in weights {
            if let mv = flatModel[k], mv.shape != v.shape {
                print("[error] Realtime shape mismatch for \(k): model \(mv.shape), weights \(v.shape)")
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

    public static func realtimeModelParameterKeys() -> [String] {
        let model = RealtimeSEMamba()
        return model.parameters().flattened().map { $0.0 }.sorted()
    }
}
