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

public struct WAVAudio: Sendable {
    public let samples: [Float]
    public let sampleRate: Int

    public var mlxMonoBatch: MLXArray {
        MLXArray(samples).reshaped(1, -1)
    }
}

public enum WAVError: Error, LocalizedError {
    case invalidHeader
    case unsupportedFormat(String)
    case missingChunk(String)

    public var errorDescription: String? {
        switch self {
        case .invalidHeader: return "Invalid WAV/RIFF header"
        case .unsupportedFormat(let s): return "Unsupported WAV format: \(s)"
        case .missingChunk(let s): return "Missing WAV chunk: \(s)"
        }
    }
}

private extension Data {
    func u16(_ offset: Int) -> UInt16 {
        self.withUnsafeBytes { raw in
            UInt16(littleEndian: raw.loadUnaligned(fromByteOffset: offset, as: UInt16.self))
        }
    }
    func u32(_ offset: Int) -> UInt32 {
        self.withUnsafeBytes { raw in
            UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: offset, as: UInt32.self))
        }
    }
    func i16(_ offset: Int) -> Int16 {
        self.withUnsafeBytes { raw in
            Int16(littleEndian: raw.loadUnaligned(fromByteOffset: offset, as: Int16.self))
        }
    }
    func i32(_ offset: Int) -> Int32 {
        self.withUnsafeBytes { raw in
            Int32(littleEndian: raw.loadUnaligned(fromByteOffset: offset, as: Int32.self))
        }
    }
    func f32(_ offset: Int) -> Float {
        let bits = u32(offset)
        return Float(bitPattern: bits)
    }
    func ascii(_ range: Range<Int>) -> String {
        String(bytes: self[range], encoding: .ascii) ?? ""
    }
}

private extension Data {
    mutating func appendASCII(_ s: String) {
        append(contentsOf: s.utf8)
    }
    mutating func appendLE(_ value: UInt16) {
        var v = value.littleEndian
        Swift.withUnsafeBytes(of: &v) { append(contentsOf: $0) }
    }
    mutating func appendLE(_ value: UInt32) {
        var v = value.littleEndian
        Swift.withUnsafeBytes(of: &v) { append(contentsOf: $0) }
    }
    mutating func appendLEFloat(_ value: Float) {
        appendLE(value.bitPattern)
    }
}

public enum WAVIO {
    public static func readMono(url: URL) throws -> WAVAudio {
        let data = try Data(contentsOf: url)
        guard data.count >= 44,
              data.ascii(0 ..< 4) == "RIFF",
              data.ascii(8 ..< 12) == "WAVE"
        else { throw WAVError.invalidHeader }

        var offset = 12
        var fmtOffset: Int?
        var fmtSize: Int = 0
        var dataOffset: Int?
        var dataSize: Int = 0

        while offset + 8 <= data.count {
            let id = data.ascii(offset ..< offset + 4)
            let size = Int(data.u32(offset + 4))
            let payload = offset + 8
            if id == "fmt " {
                fmtOffset = payload
                fmtSize = size
            } else if id == "data" {
                dataOffset = payload
                dataSize = size
            }
            offset = payload + size + (size % 2)
        }

        guard let fmt = fmtOffset else { throw WAVError.missingChunk("fmt") }
        guard let start = dataOffset else { throw WAVError.missingChunk("data") }
        guard fmtSize >= 16 else { throw WAVError.invalidHeader }

        let audioFormat = data.u16(fmt)
        let channels = Int(data.u16(fmt + 2))
        let sampleRate = Int(data.u32(fmt + 4))
        let bitsPerSample = Int(data.u16(fmt + 14))
        guard channels > 0 else { throw WAVError.invalidHeader }

        let bytesPerSample = bitsPerSample / 8
        let frameBytes = bytesPerSample * channels
        guard frameBytes > 0 else { throw WAVError.invalidHeader }
        let frames = dataSize / frameBytes
        var samples = [Float]()
        samples.reserveCapacity(frames)

        for frame in 0 ..< frames {
            var acc: Float = 0
            for ch in 0 ..< channels {
                let p = start + frame * frameBytes + ch * bytesPerSample
                let v: Float
                switch (audioFormat, bitsPerSample) {
                case (1, 16):
                    v = Float(data.i16(p)) / 32768.0
                case (1, 24):
                    let b0 = Int32(data[p])
                    let b1 = Int32(data[p + 1]) << 8
                    let b2 = Int32(data[p + 2]) << 16
                    var raw = b0 | b1 | b2
                    if (raw & 0x800000) != 0 { raw |= ~0xFFFFFF }
                    v = Float(raw) / 8_388_608.0
                case (1, 32):
                    v = Float(data.i32(p)) / 2_147_483_648.0
                case (3, 32):
                    v = data.f32(p)
                default:
                    throw WAVError.unsupportedFormat("format=\(audioFormat), bits=\(bitsPerSample)")
                }
                acc += v
            }
            samples.append(acc / Float(channels))
        }

        return WAVAudio(samples: samples, sampleRate: sampleRate)
    }

    /// Writes mono Float32 IEEE WAV.
    public static func writeMonoFloat32(url: URL, samples: [Float], sampleRate: Int) throws {
        var data = Data()
        let channels: UInt16 = 1
        let bitsPerSample: UInt16 = 32
        let bytesPerSample = UInt32(bitsPerSample / 8)
        let byteRate = UInt32(sampleRate) * UInt32(channels) * bytesPerSample
        let blockAlign = channels * (bitsPerSample / 8)
        let dataBytes = UInt32(samples.count) * bytesPerSample
        let riffSize = UInt32(4 + (8 + 16) + (8 + dataBytes))

        data.appendASCII("RIFF")
        data.appendLE(riffSize)
        data.appendASCII("WAVE")

        data.appendASCII("fmt ")
        data.appendLE(UInt32(16))
        data.appendLE(UInt16(3)) // IEEE float
        data.appendLE(channels)
        data.appendLE(UInt32(sampleRate))
        data.appendLE(byteRate)
        data.appendLE(blockAlign)
        data.appendLE(bitsPerSample)

        data.appendASCII("data")
        data.appendLE(dataBytes)
        for s in samples {
            data.appendLEFloat(max(-1.0, min(1.0, s)))
        }
        try data.write(to: url)
    }

    public static func writeMonoFloat32(url: URL, array: MLXArray, sampleRate: Int) throws {
        var x = array
        if x.ndim == 2 { x = x[0] }
        x = x.asType(.float32)
        eval(x)
        let samples = x.asArray(Float.self)
        try writeMonoFloat32(url: url, samples: samples, sampleRate: sampleRate)
    }
}
