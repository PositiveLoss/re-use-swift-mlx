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

@inline(__always)
private func rtChannelsLast(_ x: MLXArray) -> MLXArray {
    x.transposed(0, 2, 3, 1)
}

@inline(__always)
private func rtChannelsFirst(_ x: MLXArray) -> MLXArray {
    x.transposed(0, 3, 1, 2)
}

private func rtCausalPadding2D(_ kernel: (Int, Int), dilation: (Int, Int) = (1, 1)) -> (Int, Int, Int) {
    let padT = kernel.0 * dilation.0 - dilation.0
    let padF = (kernel.1 * dilation.1 - dilation.1) / 2
    return (padT, padF, padF)
}

private func rtPadTimeFreq(_ x: MLXArray, top: Int, bottom: Int = 0, left: Int, right: Int) -> MLXArray {
    padded(
        x,
        widths: [
            IntOrPair((0, 0)),
            IntOrPair((0, 0)),
            IntOrPair((top, bottom)),
            IntOrPair((left, right)),
        ]
    )
}

public final class ChannelLayerNorm2D: Module {
    @ModuleInfo(key: "norm") public var norm: LayerNorm

    public init(channels: Int) {
        self._norm.wrappedValue = LayerNorm(dimensions: channels, eps: 1e-5)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        rtChannelsFirst(norm(rtChannelsLast(x)))
    }
}

public final class RealtimeConvNormAct: Module {
    public let causalTop: Int
    public let padLeft: Int
    public let padRight: Int
    @ModuleInfo(key: "conv") public var conv: Conv2d
    @ModuleInfo(key: "norm") public var norm: ChannelLayerNorm2D
    @ModuleInfo(key: "act") public var act: ChannelFirstPReLU

    public init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: (Int, Int),
        stride: (Int, Int) = (1, 1),
        dilation: (Int, Int) = (1, 1),
        causal: Bool = true
    ) {
        if causal {
            let pad = rtCausalPadding2D(kernelSize, dilation: dilation)
            self.causalTop = pad.0
            self.padLeft = pad.1
            self.padRight = pad.2
        } else {
            self.causalTop = 0
            self.padLeft = 0
            self.padRight = 0
        }
        self._conv.wrappedValue = Conv2d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: IntOrPair(kernelSize),
            stride: IntOrPair(stride),
            padding: IntOrPair((0, 0)),
            dilation: IntOrPair(dilation),
            bias: true
        )
        self._norm.wrappedValue = ChannelLayerNorm2D(channels: outChannels)
        self._act.wrappedValue = ChannelFirstPReLU(channels: outChannels)
    }

    public func callAsFunction(_ x: MLXArray, topOverride: Int? = nil) -> MLXArray {
        let top = topOverride ?? causalTop
        let paddedX = rtPadTimeFreq(x, top: top, left: padLeft, right: padRight)
        return act(norm(rtChannelsFirst(conv(rtChannelsLast(paddedX)))))
    }
}

public final class RealtimeDenseBlock: Module {
    public let dense_block: [RealtimeConvNormAct]

    public init(hidFeature: Int, depth: Int = 4) {
        self.dense_block = (0 ..< depth).map { i in
            RealtimeConvNormAct(
                inChannels: hidFeature * (i + 1),
                outChannels: hidFeature,
                kernelSize: (3, 3),
                dilation: (1 << i, 1)
            )
        }
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var skip = x
        var out = x
        for layer in dense_block {
            out = layer(skip)
            skip = concatenated([out, skip], axis: 1)
        }
        return out
    }
}

public final class RealtimeDenseEncoder: Module {
    @ModuleInfo(key: "dense_conv_1_1") public var denseConv11: RealtimeConvNormAct
    @ModuleInfo(key: "dense_conv_1_2") public var denseConv12: RealtimeConvNormAct
    @ModuleInfo(key: "dense_conv_1_3") public var denseConv13: RealtimeConvNormAct
    @ModuleInfo(key: "dense_block") public var denseBlock: RealtimeDenseBlock
    @ModuleInfo(key: "dense_conv_2") public var denseConv2: RealtimeConvNormAct

    public init(inputChannel: Int = 2, hidFeature: Int = 64) {
        self._denseConv11.wrappedValue = RealtimeConvNormAct(inChannels: inputChannel, outChannels: hidFeature, kernelSize: (3, 3), causal: false)
        self._denseConv12.wrappedValue = RealtimeConvNormAct(inChannels: inputChannel, outChannels: hidFeature, kernelSize: (3, 3), causal: false)
        self._denseConv13.wrappedValue = RealtimeConvNormAct(inChannels: inputChannel, outChannels: hidFeature, kernelSize: (3, 3), causal: false)
        self._denseBlock.wrappedValue = RealtimeDenseBlock(hidFeature: hidFeature)
        self._denseConv2.wrappedValue = RealtimeConvNormAct(inChannels: hidFeature, outChannels: hidFeature, kernelSize: (1, 3), stride: (1, 2))
    }

    public func callAsFunction(_ x: MLXArray, lookAheadFrames: Int) -> MLXArray {
        precondition((0 ... 2).contains(lookAheadFrames), "lookAheadFrames must be 0...2")
        let paddedX = rtPadTimeFreq(x, top: 2 - lookAheadFrames, bottom: lookAheadFrames, left: 1, right: 1)
        let first: MLXArray
        switch lookAheadFrames {
        case 0: first = denseConv11(paddedX)
        case 1: first = denseConv12(paddedX)
        default: first = denseConv13(paddedX)
        }
        return denseConv2(denseBlock(first))
    }
}

public final class RealtimeMagDecoder: Module {
    @ModuleInfo(key: "dense_block") public var denseBlock: RealtimeDenseBlock
    @ModuleInfo(key: "up_conv1") public var upConv1: SPUp
    @ModuleInfo(key: "up_conv2") public var upConv2: SPUp
    @ModuleInfo(key: "final_conv") public var finalConv: Conv2d

    public init(hidFeature: Int = 64, outputChannel: Int = 1) {
        self._denseBlock.wrappedValue = RealtimeDenseBlock(hidFeature: hidFeature)
        self._upConv1.wrappedValue = SPUp(inChannels: hidFeature, outChannels: hidFeature, r: 2)
        self._upConv2.wrappedValue = SPUp(inChannels: hidFeature, outChannels: hidFeature, r: 1)
        self._finalConv.wrappedValue = Conv2d(inputChannels: hidFeature, outputChannels: outputChannel, kernelSize: IntOrPair((1, 1)), bias: true)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = denseBlock(x)
        h = upConv1(h)
        h = upConv2(h.transposed(0, 1, 3, 2)).transposed(0, 1, 3, 2)
        return rtChannelsFirst(finalConv(rtChannelsLast(h)))
    }
}

public final class RealtimePhaseDecoder: Module {
    @ModuleInfo(key: "dense_block") public var denseBlock: RealtimeDenseBlock
    @ModuleInfo(key: "up_conv1") public var upConv1: SPUp
    @ModuleInfo(key: "up_conv2") public var upConv2: SPUp
    @ModuleInfo(key: "phase_conv_r") public var phaseConvR: Conv2d
    @ModuleInfo(key: "phase_conv_i") public var phaseConvI: Conv2d

    public init(hidFeature: Int = 64, outputChannel: Int = 1) {
        self._denseBlock.wrappedValue = RealtimeDenseBlock(hidFeature: hidFeature)
        self._upConv1.wrappedValue = SPUp(inChannels: hidFeature, outChannels: hidFeature, r: 2)
        self._upConv2.wrappedValue = SPUp(inChannels: hidFeature, outChannels: hidFeature, r: 1)
        self._phaseConvR.wrappedValue = Conv2d(inputChannels: hidFeature, outputChannels: outputChannel, kernelSize: IntOrPair((1, 1)), bias: true)
        self._phaseConvI.wrappedValue = Conv2d(inputChannels: hidFeature, outputChannels: outputChannel, kernelSize: IntOrPair((1, 1)), bias: true)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = denseBlock(x)
        h = upConv1(h)
        h = upConv2(h.transposed(0, 1, 3, 2)).transposed(0, 1, 3, 2)
        let xr = rtChannelsFirst(phaseConvR(rtChannelsLast(h)))
        let xi = rtChannelsFirst(phaseConvI(rtChannelsLast(h)))
        return atan2(xi, xr)
    }
}

public final class RealtimeCausalMambaBlock: Module {
    @ModuleInfo(key: "forward_blocks") public var forwardBlocks: MambaSSM
    @ModuleInfo(key: "output_proj") public var outputProj: Linear
    @ModuleInfo(key: "norm") public var norm: LayerNorm

    public init(dModel: Int, dState: Int = 16, dConv: Int = 4, expand: Int = 4) {
        self._forwardBlocks.wrappedValue = MambaSSM(dModel: dModel, dState: dState, dConv: dConv, expand: expand)
        self._outputProj.wrappedValue = Linear(dModel, dModel, bias: true)
        self._norm.wrappedValue = LayerNorm(dimensions: dModel)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = forwardBlocks(x) + x
        return norm(outputProj(y))
    }
}

public final class RealtimeTFMambaBlock: Module {
    @ModuleInfo(key: "time_mamba") public var timeMamba: RealtimeCausalMambaBlock
    @ModuleInfo(key: "freq_mamba") public var freqMamba: MambaBlock

    public init(dModel: Int = 64, dState: Int = 16, dConv: Int = 4, expand: Int = 4) {
        self._timeMamba.wrappedValue = RealtimeCausalMambaBlock(dModel: dModel, dState: dState, dConv: dConv, expand: expand)
        self._freqMamba.wrappedValue = MambaBlock(dModel: dModel, dState: dState, dConv: dConv, expand: expand)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let c = x.dim(1)
        let t = x.dim(2)
        let f = x.dim(3)

        var xt = x.transposed(0, 3, 2, 1).reshaped(b * f, t, c)
        xt = timeMamba(xt) + xt

        var xf = xt.reshaped(b, f, t, c).transposed(0, 2, 1, 3).reshaped(b * t, f, c)
        xf = freqMamba(xf) + xf

        return xf.reshaped(b, t, f, c).transposed(0, 3, 1, 2)
    }
}

public final class RealtimeSEMamba: Module {
    @ModuleInfo(key: "dense_encoder") public var denseEncoder: RealtimeDenseEncoder
    public let TSMamba: [RealtimeTFMambaBlock]
    public let mask_decoder_list: [RealtimeMagDecoder]
    public let phase_decoder_list: [RealtimePhaseDecoder]
    public var computeDType: DType = .float32

    public init(
        numTFMamba: Int = 12,
        hidFeature: Int = 64,
        dState: Int = 16,
        dConv: Int = 4,
        expand: Int = 4,
        inputChannel: Int = 2,
        outputChannel: Int = 1
    ) {
        self._denseEncoder.wrappedValue = RealtimeDenseEncoder(inputChannel: inputChannel, hidFeature: hidFeature)
        self.TSMamba = (0 ..< numTFMamba).map { _ in
            RealtimeTFMambaBlock(dModel: hidFeature, dState: dState, dConv: dConv, expand: expand)
        }
        self.mask_decoder_list = (0 ..< numTFMamba).map { _ in
            RealtimeMagDecoder(hidFeature: hidFeature, outputChannel: outputChannel)
        }
        self.phase_decoder_list = (0 ..< numTFMamba).map { _ in
            RealtimePhaseDecoder(hidFeature: hidFeature, outputChannel: outputChannel)
        }
    }

    public func callAsFunction(
        noisyMag: MLXArray,
        noisyPhase: MLXArray,
        exitLayer: Int = 8,
        lookAheadFrames: Int = 0
    ) -> (MLXArray, MLXArray, MLXArray) {
        precondition((1 ... TSMamba.count).contains(exitLayer), "exitLayer must be 1...\(TSMamba.count)")
        precondition((0 ... 2).contains(lookAheadFrames), "lookAheadFrames must be 0...2")

        let f = noisyMag.dim(1)
        let t = noisyMag.dim(2)
        let mag = expandedDimensions(noisyMag.transposed(0, 2, 1), axis: 1).asType(computeDType)
        let pha = expandedDimensions(noisyPhase.transposed(0, 2, 1), axis: 1).asType(computeDType)
        var x = concatenated([mag, pha], axis: 1)
        x = concatenated([x, MLXArray.zeros([x.dim(0), x.dim(1), x.dim(2), 2], dtype: x.dtype)], axis: -1)

        x = denseEncoder(x, lookAheadFrames: lookAheadFrames)
        for block in TSMamba.prefix(exitLayer) {
            x = block(x)
        }

        var denoisedMag = mask_decoder_list[exitLayer - 1](x).transposed(0, 3, 2, 1)[0..., 0..., 0..., 0]
        var denoisedPha = phase_decoder_list[exitLayer - 1](x).transposed(0, 3, 2, 1)[0..., 0..., 0..., 0]
        denoisedMag = denoisedMag[0..., 0 ..< f, 0 ..< t].asType(.float32)
        denoisedPha = denoisedPha[0..., 0 ..< f, 0 ..< t].asType(.float32)

        let denoisedCom = stacked([
            denoisedMag * cos(denoisedPha),
            denoisedMag * sin(denoisedPha),
        ], axis: -1)
        return (denoisedMag, denoisedPha, denoisedCom)
    }
}

public final class RealtimeStreamingSEMamba {
    public let model: RealtimeSEMamba
    private var magFrames: MLXArray?
    private var phaseFrames: MLXArray?
    private var emittedFrames = 0

    public init(model: RealtimeSEMamba) {
        self.model = model
    }

    public func reset(batchSize: Int = 1, freqBins: Int) {
        self.magFrames = nil
        self.phaseFrames = nil
        self.emittedFrames = 0
    }

    /// Stateful streaming API. This keeps the frame history and emits one enhanced
    /// frame after the configured look-ahead is available. The model work remains on
    /// MLX/Metal; replacing the history replay with per-layer caches is an internal
    /// optimization that does not change this public contract.
    public func step(
        noisyMag: MLXArray,
        noisyPhase: MLXArray,
        exitLayer: Int = 8,
        lookAheadFrames: Int = 0
    ) -> (MLXArray, MLXArray)? {
        let magFrame = noisyMag.ndim == 2 ? expandedDimensions(noisyMag, axis: -1) : noisyMag
        let phaseFrame = noisyPhase.ndim == 2 ? expandedDimensions(noisyPhase, axis: -1) : noisyPhase
        if let magFrames, let phaseFrames {
            self.magFrames = concatenated([magFrames, magFrame], axis: -1)
            self.phaseFrames = concatenated([phaseFrames, phaseFrame], axis: -1)
        } else {
            self.magFrames = magFrame
            self.phaseFrames = phaseFrame
        }

        guard let allMag = self.magFrames, let allPhase = self.phaseFrames else {
            return nil
        }
        let available = allMag.dim(-1)
        guard available > lookAheadFrames, emittedFrames < available - lookAheadFrames else {
            return nil
        }

        let (magOut, phaseOut, _) = model(noisyMag: allMag, noisyPhase: allPhase, exitLayer: exitLayer, lookAheadFrames: lookAheadFrames)
        let idx = emittedFrames
        emittedFrames += 1
        return (magOut[0..., 0..., idx], phaseOut[0..., 0..., idx])
    }

    public func flush(exitLayer: Int = 8, lookAheadFrames: Int = 0) -> [(MLXArray, MLXArray)] {
        guard lookAheadFrames > 0, let allMag = self.magFrames, let allPhase = self.phaseFrames else {
            return []
        }
        let pad = MLXArray.zeros([allMag.dim(0), allMag.dim(1), lookAheadFrames], dtype: allMag.dtype)
        self.magFrames = concatenated([allMag, pad], axis: -1)
        self.phaseFrames = concatenated([allPhase, pad], axis: -1)
        var out: [(MLXArray, MLXArray)] = []
        while let magFrames = self.magFrames, emittedFrames < magFrames.dim(-1) - lookAheadFrames {
            let (magOut, phaseOut, _) = model(noisyMag: magFrames, noisyPhase: self.phaseFrames!, exitLayer: exitLayer, lookAheadFrames: lookAheadFrames)
            let idx = emittedFrames
            emittedFrames += 1
            out.append((magOut[0..., 0..., idx], phaseOut[0..., 0..., idx]))
        }
        return out
    }
}
