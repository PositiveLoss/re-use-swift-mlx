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
        self._denseConv2.wrappedValue = RealtimeConvNormAct(inChannels: hidFeature, outChannels: hidFeature, kernelSize: (1, 3), stride: (1, 2), causal: false)
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

public final class RealtimeSPConvTranspose2d: Module {
    public let outChannels: Int
    public let r: Int
    public let padding: (left: Int, right: Int, top: Int, bottom: Int)
    @ModuleInfo(key: "conv") public var conv: Conv2d

    public init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: (Int, Int),
        padding: (Int, Int, Int, Int),
        r: Int = 1
    ) {
        self.outChannels = outChannels
        self.r = r
        self.padding = (padding.0, padding.1, padding.2, padding.3)
        self._conv.wrappedValue = Conv2d(
            inputChannels: inChannels,
            outputChannels: outChannels * r,
            kernelSize: IntOrPair(kernelSize),
            stride: IntOrPair((1, 1)),
            padding: IntOrPair((0, 0)),
            bias: true
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let paddedX = padded(
            x,
            widths: [
                IntOrPair((0, 0)),
                IntOrPair((0, 0)),
                IntOrPair((padding.top, padding.bottom)),
                IntOrPair((padding.left, padding.right)),
            ]
        )
        var out = rtChannelsFirst(conv(rtChannelsLast(paddedX)))
        let b = out.dim(0)
        let nch = out.dim(1)
        let h = out.dim(2)
        let w = out.dim(3)
        out = out.reshaped(b, r, nch / r, h, w)
        out = out.transposed(0, 2, 3, 4, 1)
        return out.reshaped(b, nch / r, h, -1)
    }
}

public final class RealtimeSPUp: Module {
    @ModuleInfo(key: "conv") public var conv: RealtimeSPConvTranspose2d
    @ModuleInfo(key: "norm") public var norm: ChannelLayerNorm2D
    @ModuleInfo(key: "act") public var act: ChannelFirstPReLU

    public init(inChannels: Int, outChannels: Int, padding: (Int, Int, Int, Int), r: Int) {
        self._conv.wrappedValue = RealtimeSPConvTranspose2d(
            inChannels: inChannels,
            outChannels: outChannels,
            kernelSize: (1, 3),
            padding: padding,
            r: r
        )
        self._norm.wrappedValue = ChannelLayerNorm2D(channels: outChannels)
        self._act.wrappedValue = ChannelFirstPReLU(channels: outChannels)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        act(norm(conv(x)))
    }
}

public final class RealtimeMagDecoder: Module {
    @ModuleInfo(key: "dense_block") public var denseBlock: RealtimeDenseBlock
    @ModuleInfo(key: "up_conv1") public var upConv1: RealtimeSPUp
    @ModuleInfo(key: "up_conv2") public var upConv2: RealtimeSPUp
    @ModuleInfo(key: "final_conv") public var finalConv: Conv2d

    public init(hidFeature: Int = 64, outputChannel: Int = 1) {
        self._denseBlock.wrappedValue = RealtimeDenseBlock(hidFeature: hidFeature)
        self._upConv1.wrappedValue = RealtimeSPUp(inChannels: hidFeature, outChannels: hidFeature, padding: (1, 1, 0, 0), r: 2)
        self._upConv2.wrappedValue = RealtimeSPUp(inChannels: hidFeature, outChannels: hidFeature, padding: (2, 0, 0, 0), r: 1)
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
    @ModuleInfo(key: "up_conv1") public var upConv1: RealtimeSPUp
    @ModuleInfo(key: "up_conv2") public var upConv2: RealtimeSPUp
    @ModuleInfo(key: "phase_conv_r") public var phaseConvR: Conv2d
    @ModuleInfo(key: "phase_conv_i") public var phaseConvI: Conv2d

    public init(hidFeature: Int = 64, outputChannel: Int = 1) {
        self._denseBlock.wrappedValue = RealtimeDenseBlock(hidFeature: hidFeature)
        self._upConv1.wrappedValue = RealtimeSPUp(inChannels: hidFeature, outChannels: hidFeature, padding: (1, 1, 0, 0), r: 2)
        self._upConv2.wrappedValue = RealtimeSPUp(inChannels: hidFeature, outChannels: hidFeature, padding: (2, 0, 0, 0), r: 1)
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

fileprivate final class StreamingConv2DState {
    var cache: MLXArray?
}

fileprivate final class StreamingConv2D {
    let layer: RealtimeConvNormAct
    let cacheLen: Int
    let freqPadLeft: Int
    let freqPadRight: Int

    init(layer: RealtimeConvNormAct, freqPadLeft: Int? = nil, freqPadRight: Int? = nil) {
        self.layer = layer
        let kt = layer.conv.weight.shape[1]
        self.cacheLen = (kt - 1) * layer.conv.dilation.0
        self.freqPadLeft = freqPadLeft ?? layer.padLeft
        self.freqPadRight = freqPadRight ?? layer.padRight
    }

    func makeState() -> StreamingConv2DState {
        StreamingConv2DState()
    }

    func step(_ x: MLXArray, state: StreamingConv2DState, initialTop: Int? = nil) -> MLXArray {
        if cacheLen == 0 {
            return layer(x)
        }

        let b = x.dim(0)
        let c = x.dim(1)
        let f = x.dim(3)
        let cache = state.cache ?? MLXArray.zeros([b, c, initialTop ?? cacheLen, f], dtype: x.dtype)
        let xCat = concatenated([cache, x], axis: 2)
        let paddedX = rtPadTimeFreq(xCat, top: 0, left: freqPadLeft, right: freqPadRight)
        let y = layer.act(layer.norm(rtChannelsFirst(layer.conv(rtChannelsLast(paddedX)))))

        let totalT = xCat.dim(2)
        state.cache = xCat[0..., 0..., (totalT - cacheLen) ..< totalT, 0...]
        return y
    }
}

fileprivate final class StreamingDenseBlockState {
    let layerStates: [StreamingConv2DState]

    init(layers: [StreamingConv2D]) {
        self.layerStates = layers.map { $0.makeState() }
    }
}

fileprivate final class StreamingDenseBlock {
    let layers: [StreamingConv2D]

    init(_ block: RealtimeDenseBlock) {
        self.layers = block.dense_block.map { StreamingConv2D(layer: $0) }
    }

    func makeState() -> StreamingDenseBlockState {
        StreamingDenseBlockState(layers: layers)
    }

    func step(_ x: MLXArray, state: StreamingDenseBlockState) -> MLXArray {
        var skip = x
        var out = x
        for i in 0 ..< layers.count {
            out = layers[i].step(skip, state: state.layerStates[i])
            skip = concatenated([out, skip], axis: 1)
        }
        return out
    }
}

fileprivate final class StreamingDenseEncoderState {
    let conv11: StreamingConv2DState
    let conv12: StreamingConv2DState
    let conv13: StreamingConv2DState
    let denseBlock: StreamingDenseBlockState

    init(encoder: StreamingDenseEncoder) {
        self.conv11 = encoder.conv11.makeState()
        self.conv12 = encoder.conv12.makeState()
        self.conv13 = encoder.conv13.makeState()
        self.denseBlock = encoder.denseBlock.makeState()
    }
}

fileprivate final class StreamingDenseEncoder {
    let conv11: StreamingConv2D
    let conv12: StreamingConv2D
    let conv13: StreamingConv2D
    let denseBlock: StreamingDenseBlock
    let conv2: RealtimeConvNormAct

    init(_ encoder: RealtimeDenseEncoder) {
        self.conv11 = StreamingConv2D(layer: encoder.denseConv11, freqPadLeft: 1, freqPadRight: 1)
        self.conv12 = StreamingConv2D(layer: encoder.denseConv12, freqPadLeft: 1, freqPadRight: 1)
        self.conv13 = StreamingConv2D(layer: encoder.denseConv13, freqPadLeft: 1, freqPadRight: 1)
        self.denseBlock = StreamingDenseBlock(encoder.denseBlock)
        self.conv2 = encoder.denseConv2
    }

    func makeState() -> StreamingDenseEncoderState {
        StreamingDenseEncoderState(encoder: self)
    }

    func step(_ x: MLXArray, state: StreamingDenseEncoderState, lookAheadFrames: Int, initial: Bool) -> MLXArray {
        let top = initial ? 2 - lookAheadFrames : nil
        let first: MLXArray
        switch lookAheadFrames {
        case 0:
            first = conv11.step(x, state: state.conv11, initialTop: top)
        case 1:
            first = conv12.step(x, state: state.conv12, initialTop: top)
        default:
            first = conv13.step(x, state: state.conv13, initialTop: top)
        }
        return conv2(denseBlock.step(first, state: state.denseBlock))
    }
}

fileprivate final class StreamingFTConvState {
    var cache: MLXArray?
}

fileprivate final class StreamingFTConv {
    let up: RealtimeSPUp
    let cacheLen: Int

    init(_ up: RealtimeSPUp) {
        self.up = up
        let kw = up.conv.conv.weight.shape[2]
        self.cacheLen = kw - 1
    }

    func makeState() -> StreamingFTConvState {
        StreamingFTConvState()
    }

    func step(_ x: MLXArray, state: StreamingFTConvState) -> MLXArray {
        let b = x.dim(0)
        let c = x.dim(1)
        let f = x.dim(2)
        let cache = state.cache ?? MLXArray.zeros([b, c, f, cacheLen], dtype: x.dtype)
        let xCat = concatenated([cache, x], axis: 3)
        let convOut = rtChannelsFirst(up.conv.conv(rtChannelsLast(xCat)))
        let y = up.act(up.norm(convOut))
        let totalT = xCat.dim(3)
        state.cache = xCat[0..., 0..., 0..., (totalT - cacheLen) ..< totalT]
        return y
    }
}

fileprivate final class StreamingMagDecoderState {
    let denseBlock: StreamingDenseBlockState
    let upConv2: StreamingFTConvState

    init(decoder: StreamingMagDecoder) {
        self.denseBlock = decoder.denseBlock.makeState()
        self.upConv2 = decoder.upConv2.makeState()
    }
}

fileprivate final class StreamingMagDecoder {
    let decoder: RealtimeMagDecoder
    let denseBlock: StreamingDenseBlock
    let upConv2: StreamingFTConv

    init(_ decoder: RealtimeMagDecoder) {
        self.decoder = decoder
        self.denseBlock = StreamingDenseBlock(decoder.denseBlock)
        self.upConv2 = StreamingFTConv(decoder.upConv2)
    }

    func makeState() -> StreamingMagDecoderState {
        StreamingMagDecoderState(decoder: self)
    }

    func step(_ x: MLXArray, state: StreamingMagDecoderState, outputFreqBins: Int) -> MLXArray {
        var h = denseBlock.step(x, state: state.denseBlock)
        h = decoder.upConv1(h)
        h = upConv2.step(h.transposed(0, 1, 3, 2), state: state.upConv2).transposed(0, 1, 3, 2)
        let y = rtChannelsFirst(decoder.finalConv(rtChannelsLast(h)))
        return y[0..., 0, 0, 0 ..< outputFreqBins]
    }
}

fileprivate final class StreamingPhaseDecoderState {
    let denseBlock: StreamingDenseBlockState
    let upConv2: StreamingFTConvState

    init(decoder: StreamingPhaseDecoder) {
        self.denseBlock = decoder.denseBlock.makeState()
        self.upConv2 = decoder.upConv2.makeState()
    }
}

fileprivate final class StreamingPhaseDecoder {
    let decoder: RealtimePhaseDecoder
    let denseBlock: StreamingDenseBlock
    let upConv2: StreamingFTConv

    init(_ decoder: RealtimePhaseDecoder) {
        self.decoder = decoder
        self.denseBlock = StreamingDenseBlock(decoder.denseBlock)
        self.upConv2 = StreamingFTConv(decoder.upConv2)
    }

    func makeState() -> StreamingPhaseDecoderState {
        StreamingPhaseDecoderState(decoder: self)
    }

    func step(_ x: MLXArray, state: StreamingPhaseDecoderState, outputFreqBins: Int) -> MLXArray {
        var h = denseBlock.step(x, state: state.denseBlock)
        h = decoder.upConv1(h)
        h = upConv2.step(h.transposed(0, 1, 3, 2), state: state.upConv2).transposed(0, 1, 3, 2)
        let xr = rtChannelsFirst(decoder.phaseConvR(rtChannelsLast(h)))
        let xi = rtChannelsFirst(decoder.phaseConvI(rtChannelsLast(h)))
        let y = atan2(xi, xr)
        return y[0..., 0, 0, 0 ..< outputFreqBins]
    }
}

fileprivate final class StreamingCausalMambaBlockState {
    let forward: MambaSSMState

    init(block: RealtimeCausalMambaBlock, batchSize: Int, dtype: DType) {
        self.forward = MambaSSMState(
            batchSize: batchSize,
            dInner: block.forwardBlocks.dInner,
            dConv: block.forwardBlocks.dConv,
            dState: block.forwardBlocks.dState,
            dtype: dtype
        )
    }
}

fileprivate final class StreamingTFMambaBlockState {
    let timeMamba: StreamingCausalMambaBlockState

    init(block: RealtimeTFMambaBlock, batchSize: Int, dtype: DType) {
        self.timeMamba = StreamingCausalMambaBlockState(block: block.timeMamba, batchSize: batchSize, dtype: dtype)
    }
}

extension RealtimeCausalMambaBlock {
    fileprivate func step(_ x: MLXArray, state: StreamingCausalMambaBlockState) -> MLXArray {
        let y = forwardBlocks.step(x, state: state.forward) + x
        return norm(outputProj(y))
    }
}

extension RealtimeTFMambaBlock {
    fileprivate func step(_ x: MLXArray, state: StreamingTFMambaBlockState) -> MLXArray {
        let b = x.dim(0)
        let c = x.dim(1)
        let f = x.dim(3)

        var xt = x.transposed(0, 3, 2, 1).reshaped(b * f, 1, c)
        xt = timeMamba.step(xt, state: state.timeMamba) + xt
        let xTime = xt.reshaped(b, f, 1, c).transposed(0, 3, 2, 1)

        var xf = xTime.transposed(0, 2, 3, 1).reshaped(b, f, c)
        xf = freqMamba(xf) + xf
        return xf.reshaped(b, 1, f, c).transposed(0, 3, 1, 2)
    }
}

public final class RealtimeStreamingSEMamba {
    public let model: RealtimeSEMamba
    private let encoder: StreamingDenseEncoder
    private let magDecoders: [StreamingMagDecoder]
    private let phaseDecoders: [StreamingPhaseDecoder]
    private var state: State?

    public final class State {
        public let batchSize: Int
        public let freqBins: Int
        public let encodedFreqBins: Int
        public let exitLayer: Int
        public let lookAheadFrames: Int
        public let dtype: DType
        fileprivate let encoder: StreamingDenseEncoderState
        fileprivate let mamba: [StreamingTFMambaBlockState]
        fileprivate let magDecoder: StreamingMagDecoderState
        fileprivate let phaseDecoder: StreamingPhaseDecoderState
        fileprivate var pendingMag: [MLXArray] = []
        fileprivate var pendingPhase: [MLXArray] = []
        fileprivate var initialized = false

        fileprivate init(
            model: RealtimeSEMamba,
            encoder: StreamingDenseEncoder,
            magDecoder: StreamingMagDecoder,
            phaseDecoder: StreamingPhaseDecoder,
            batchSize: Int,
            freqBins: Int,
            exitLayer: Int,
            lookAheadFrames: Int,
            dtype: DType
        ) {
            self.batchSize = batchSize
            self.freqBins = freqBins
            // The model appends two frequency bins, then dense_conv_2 uses kernel=3,
            // stride=2, and no padding: floor(((freqBins + 2) - 3) / 2) + 1.
            let encoded = freqBins / 2 + 1
            self.encodedFreqBins = encoded
            self.exitLayer = exitLayer
            self.lookAheadFrames = lookAheadFrames
            self.dtype = dtype
            self.encoder = encoder.makeState()
            self.mamba = model.TSMamba.map {
                StreamingTFMambaBlockState(block: $0, batchSize: batchSize * encoded, dtype: dtype)
            }
            self.magDecoder = magDecoder.makeState()
            self.phaseDecoder = phaseDecoder.makeState()
        }
    }

    public init(model: RealtimeSEMamba) {
        self.model = model
        self.encoder = StreamingDenseEncoder(model.denseEncoder)
        self.magDecoders = model.mask_decoder_list.map { StreamingMagDecoder($0) }
        self.phaseDecoders = model.phase_decoder_list.map { StreamingPhaseDecoder($0) }
    }

    public func reset(batchSize: Int = 1, freqBins: Int) {
        reset(batchSize: batchSize, freqBins: freqBins, exitLayer: 8, lookAheadFrames: 0, dtype: model.computeDType)
    }

    public func reset(batchSize: Int = 1, freqBins: Int, exitLayer: Int, lookAheadFrames: Int, dtype: DType) {
        precondition((1 ... model.TSMamba.count).contains(exitLayer), "exitLayer must be 1...\(model.TSMamba.count)")
        precondition((0 ... 2).contains(lookAheadFrames), "lookAheadFrames must be 0...2")
        self.state = State(
            model: model,
            encoder: encoder,
            magDecoder: magDecoders[exitLayer - 1],
            phaseDecoder: phaseDecoders[exitLayer - 1],
            batchSize: batchSize,
            freqBins: freqBins,
            exitLayer: exitLayer,
            lookAheadFrames: lookAheadFrames,
            dtype: dtype
        )
    }

    public func step(noisyMag: MLXArray, noisyPhase: MLXArray) -> (MLXArray, MLXArray)? {
        guard let state else {
            preconditionFailure("RealtimeStreamingSEMamba.reset(...) must be called before step(noisyMag:noisyPhase:)")
        }
        let magFrame = (noisyMag.ndim == 3 ? noisyMag[0..., 0..., 0] : noisyMag).asType(state.dtype)
        let phaseFrame = (noisyPhase.ndim == 3 ? noisyPhase[0..., 0..., 0] : noisyPhase).asType(state.dtype)

        if !state.initialized {
            state.pendingMag.append(magFrame)
            state.pendingPhase.append(phaseFrame)
            guard state.pendingMag.count >= state.lookAheadFrames + 1 else {
                return nil
            }
            let magSeq = stacked(state.pendingMag, axis: -1)
            let phaseSeq = stacked(state.pendingPhase, axis: -1)
            state.pendingMag.removeAll(keepingCapacity: true)
            state.pendingPhase.removeAll(keepingCapacity: true)
            state.initialized = true
            return process(magSeq: magSeq, phaseSeq: phaseSeq, initial: true, state: state)
        }

        return process(
            magSeq: expandedDimensions(magFrame, axis: -1),
            phaseSeq: expandedDimensions(phaseFrame, axis: -1),
            initial: false,
            state: state
        )
    }

    public func step(
        noisyMag: MLXArray,
        noisyPhase: MLXArray,
        exitLayer: Int = 8,
        lookAheadFrames: Int = 0
    ) -> (MLXArray, MLXArray)? {
        let frame = noisyMag.ndim == 3 ? noisyMag[0..., 0..., 0] : noisyMag
        if state == nil ||
            state!.batchSize != frame.dim(0) ||
            state!.freqBins != frame.dim(1) ||
            state!.exitLayer != exitLayer ||
            state!.lookAheadFrames != lookAheadFrames ||
            state!.dtype != model.computeDType {
            reset(batchSize: frame.dim(0), freqBins: frame.dim(1), exitLayer: exitLayer, lookAheadFrames: lookAheadFrames, dtype: model.computeDType)
        }
        return step(noisyMag: noisyMag, noisyPhase: noisyPhase)
    }

    public func flush(exitLayer: Int = 8, lookAheadFrames: Int = 0) -> [(MLXArray, MLXArray)] {
        guard let state, state.lookAheadFrames > 0 else {
            return []
        }
        var out: [(MLXArray, MLXArray)] = []
        for _ in 0 ..< state.lookAheadFrames {
            let zero = MLXArray.zeros([state.batchSize, state.freqBins], dtype: state.dtype)
            if let frame = step(noisyMag: zero, noisyPhase: zero) {
                out.append(frame)
            }
        }
        return out
    }

    private func process(magSeq: MLXArray, phaseSeq: MLXArray, initial: Bool, state: State) -> (MLXArray, MLXArray) {
        let mag = expandedDimensions(magSeq.transposed(0, 2, 1), axis: 1)
        let pha = expandedDimensions(phaseSeq.transposed(0, 2, 1), axis: 1)
        var x = concatenated([mag, pha], axis: 1)
        x = concatenated([x, MLXArray.zeros([x.dim(0), x.dim(1), x.dim(2), 2], dtype: x.dtype)], axis: -1)

        x = encoder.step(x, state: state.encoder, lookAheadFrames: state.lookAheadFrames, initial: initial)
        for i in 0 ..< state.exitLayer {
            x = model.TSMamba[i].step(x, state: state.mamba[i])
        }

        let magOut = magDecoders[state.exitLayer - 1].step(x, state: state.magDecoder, outputFreqBins: state.freqBins).asType(.float32)
        let phaseOut = phaseDecoders[state.exitLayer - 1].step(x, state: state.phaseDecoder, outputFreqBins: state.freqBins).asType(.float32)
        return (magOut, phaseOut)
    }
}
