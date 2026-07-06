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

public final class REUSEEnhancer {
    public let model: SEMamba
    public let compressFactor: String

    public init(model: SEMamba, compressFactor: String = "relu_log1p") {
        self.model = model
        self.compressFactor = compressFactor
    }

    /// Set `REUSE_TIMING=1` in the environment to print per-stage timings to stderr.
    private static let timingEnabled = ProcessInfo.processInfo.environment["REUSE_TIMING"] == "1"

    public func enhanceChunk(_ chunk: MLXArray, sampleRate: Int) -> MLXArray {
        let params = stftParams(for: sampleRate)
        let timing = Self.timingEnabled
        let t0 = DispatchTime.now().uptimeNanoseconds

        let (mag, pha) = magPhaseSTFT(
            wave: chunk,
            nFFT: params.nFFT,
            hop: params.hop,
            win: params.win,
            compressFactor: compressFactor,
            center: true,
            addEps: false
        )
        if timing { eval(mag, pha) }
        let t1 = DispatchTime.now().uptimeNanoseconds

        let (amp, estPha, _) = model(noisyMag: mag, noisyPhase: pha)
        if timing { eval(amp, estPha) }
        let t2 = DispatchTime.now().uptimeNanoseconds

        let filteredAmp = sweepArtifactFilter(amp)
        let out = magPhaseISTFT(
            mag: filteredAmp,
            phase: estPha,
            nFFT: params.nFFT,
            hop: params.hop,
            win: params.win,
            compressFactor: compressFactor,
            center: true
        )
        let result = clip(out, min: -1.0, max: 1.0)
        if timing {
            eval(result)
            let t3 = DispatchTime.now().uptimeNanoseconds
            let ms = { (a: UInt64, b: UInt64) in Double(b - a) / 1_000_000.0 }
            FileHandle.standardError.write("  [timing] stft=\(String(format: "%.0f", ms(t0, t1)))ms model=\(String(format: "%.0f", ms(t1, t2)))ms istft=\(String(format: "%.0f", ms(t2, t3)))ms\n".data(using: .utf8)!)
        }
        return result
    }

    public func enhance(
        waveform: MLXArray,
        sampleRate: Int,
        chunkSizeSeconds: Float = 1.0,
        hopPortion: Float = 0.5
    ) -> MLXArray {
        let wave2D = waveform.ndim == 1 ? expandedDimensions(waveform, axis: 0) : waveform
        let chunkSize = Int(Float(sampleRate) * chunkSizeSeconds)
        let out2D = chunkedHannOLA(
            wave: wave2D,
            chunkSize: chunkSize,
            hopPortion: hopPortion,
            batchSize: 4
        ) { [weak self] chunk in
            guard let self = self else { return chunk }
            return self.enhanceChunk(chunk, sampleRate: sampleRate)
        }
        eval(out2D)
        return out2D
    }
}
