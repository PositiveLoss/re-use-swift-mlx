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
public func mlxSilu(_ x: MLXArray) -> MLXArray {
    x / (1 + exp(-x))
}

@inline(__always)
private func toChannelsLast(_ x: MLXArray) -> MLXArray {
    // [B, C, H, W] -> [B, H, W, C]
    x.transposed(0, 2, 3, 1)
}

@inline(__always)
private func toChannelsFirst(_ x: MLXArray) -> MLXArray {
    // [B, H, W, C] -> [B, C, H, W]
    x.transposed(0, 3, 1, 2)
}

@inline(__always)
private func samePadding2D(_ kernel: (Int, Int), dilation: (Int, Int) = (1, 1)) -> IntOrPair {
    let ph = (kernel.0 * dilation.0 - dilation.0) / 2
    let pw = (kernel.1 * dilation.1 - dilation.1) / 2
    return IntOrPair((ph, pw))
}

/// PReLU for channel-first 4-D tensors. Stored key is `weight`, matching PyTorch/MLX ports.
public final class ChannelFirstPReLU: Module {
    @ParameterInfo(key: "weight") public var weight: MLXArray

    public init(channels: Int, initValue: Float = 0.25) {
        self._weight.wrappedValue = MLXArray.ones([channels], dtype: .float32) * initValue
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let w = weight.reshaped(1, -1, 1, 1)
        return maximum(x, 0) + w * minimum(x, 0)
    }
}

/// Conv2d -> InstanceNorm -> PReLU on public channel-first `[B,C,H,W]` tensors.
public final class ConvNormAct: Module {
    @ModuleInfo(key: "conv") public var conv: Conv2d
    @ModuleInfo(key: "norm") public var norm: InstanceNorm
    @ModuleInfo(key: "act") public var act: ChannelFirstPReLU

    public init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: (Int, Int),
        stride: (Int, Int) = (1, 1),
        dilation: (Int, Int) = (1, 1),
        padding: (Int, Int) = (0, 0)
    ) {
        self._conv.wrappedValue = Conv2d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: IntOrPair(kernelSize),
            stride: IntOrPair(stride),
            padding: IntOrPair(padding),
            dilation: IntOrPair(dilation),
            bias: true
        )
        self._norm.wrappedValue = InstanceNorm(dimensions: outChannels, eps: 1e-5, affine: true)
        self._act.wrappedValue = ChannelFirstPReLU(channels: outChannels)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = toChannelsFirst(norm(conv(toChannelsLast(x))))
        return act(h)
    }
}

/// Sub-pixel transpose convolution along the frequency axis.
public final class SPConvTranspose2d: Module {
    public let outChannels: Int
    public let r: Int
    @ModuleInfo(key: "conv") public var conv: Conv2d

    public init(inChannels: Int, outChannels: Int, kernelSize: (Int, Int), r: Int = 1) {
        self.outChannels = outChannels
        self.r = r
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
        // Pad W/frequency by one bin on both sides.
        let paddedX = padded(
            x,
            widths: [
                IntOrPair((0, 0)),
                IntOrPair((0, 0)),
                IntOrPair((0, 0)),
                IntOrPair((1, 1)),
            ]
        )
        var out = toChannelsFirst(conv(toChannelsLast(paddedX)))
        let b = out.dim(0)
        let nch = out.dim(1)
        let h = out.dim(2)
        let w = out.dim(3)
        out = out.reshaped(b, r, nch / r, h, w)
        out = out.transposed(0, 2, 3, 4, 1)
        out = out.reshaped(b, nch / r, h, -1)
        return out
    }
}

/// SPConvTranspose2d -> InstanceNorm -> PReLU.
public final class SPUp: Module {
    @ModuleInfo(key: "conv") public var conv: SPConvTranspose2d
    @ModuleInfo(key: "norm") public var norm: InstanceNorm
    @ModuleInfo(key: "act") public var act: ChannelFirstPReLU

    public init(inChannels: Int, outChannels: Int, r: Int) {
        self._conv.wrappedValue = SPConvTranspose2d(
            inChannels: inChannels,
            outChannels: outChannels,
            kernelSize: (1, 3),
            r: r
        )
        self._norm.wrappedValue = InstanceNorm(dimensions: outChannels, eps: 1e-5, affine: true)
        self._act.wrappedValue = ChannelFirstPReLU(channels: outChannels)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = toChannelsFirst(norm(toChannelsLast(conv(x))))
        return act(h)
    }
}

public final class DenseBlock: Module {
    public let depth: Int
    public let dense_block: [ConvNormAct]

    public init(hidFeature: Int, kernelSize: (Int, Int) = (3, 3), depth: Int = 4) {
        self.depth = depth
        self.dense_block = (0 ..< depth).map { i in
            let dilation = (1 << i, 1)
            return ConvNormAct(
                inChannels: hidFeature * (i + 1),
                outChannels: hidFeature,
                kernelSize: kernelSize,
                dilation: dilation,
                padding: ((kernelSize.0 * dilation.0 - dilation.0) / 2,
                          (kernelSize.1 * dilation.1 - dilation.1) / 2)
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

public final class DenseEncoder: Module {
    @ModuleInfo(key: "dense_conv_1") public var denseConv1: ConvNormAct
    @ModuleInfo(key: "dense_block") public var denseBlock: DenseBlock
    @ModuleInfo(key: "dense_conv_2") public var denseConv2: ConvNormAct

    public init(inputChannel: Int = 2, hidFeature: Int = 64) {
        self._denseConv1.wrappedValue = ConvNormAct(inChannels: inputChannel, outChannels: hidFeature, kernelSize: (1, 1))
        self._denseBlock.wrappedValue = DenseBlock(hidFeature: hidFeature, depth: 4)
        self._denseConv2.wrappedValue = ConvNormAct(
            inChannels: hidFeature,
            outChannels: hidFeature,
            kernelSize: (1, 3),
            stride: (4, 2)
        )
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        denseConv2(denseBlock(denseConv1(x)))
    }
}

public final class MagDecoder: Module {
    @ModuleInfo(key: "dense_block") public var denseBlock: DenseBlock
    @ModuleInfo(key: "up_conv1") public var upConv1: SPUp
    @ModuleInfo(key: "up_conv2") public var upConv2: SPUp
    @ModuleInfo(key: "final_conv") public var finalConv: Conv2d

    public init(hidFeature: Int = 64, outputChannel: Int = 1) {
        self._denseBlock.wrappedValue = DenseBlock(hidFeature: hidFeature, depth: 4)
        self._upConv1.wrappedValue = SPUp(inChannels: hidFeature, outChannels: hidFeature, r: 2)
        self._upConv2.wrappedValue = SPUp(inChannels: hidFeature, outChannels: hidFeature, r: 4)
        self._finalConv.wrappedValue = Conv2d(inputChannels: hidFeature, outputChannels: outputChannel, kernelSize: IntOrPair((1, 1)), bias: true)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = denseBlock(x)
        h = upConv1(h)
        h = upConv2(h.transposed(0, 1, 3, 2)).transposed(0, 1, 3, 2)
        return toChannelsFirst(finalConv(toChannelsLast(h)))
    }
}

public final class PhaseDecoder: Module {
    @ModuleInfo(key: "dense_block") public var denseBlock: DenseBlock
    @ModuleInfo(key: "up_conv1") public var upConv1: SPUp
    @ModuleInfo(key: "up_conv2") public var upConv2: SPUp
    @ModuleInfo(key: "phase_conv_r") public var phaseConvR: Conv2d
    @ModuleInfo(key: "phase_conv_i") public var phaseConvI: Conv2d

    public init(hidFeature: Int = 64, outputChannel: Int = 1) {
        self._denseBlock.wrappedValue = DenseBlock(hidFeature: hidFeature, depth: 4)
        self._upConv1.wrappedValue = SPUp(inChannels: hidFeature, outChannels: hidFeature, r: 2)
        self._upConv2.wrappedValue = SPUp(inChannels: hidFeature, outChannels: hidFeature, r: 4)
        self._phaseConvR.wrappedValue = Conv2d(inputChannels: hidFeature, outputChannels: outputChannel, kernelSize: IntOrPair((1, 1)), bias: true)
        self._phaseConvI.wrappedValue = Conv2d(inputChannels: hidFeature, outputChannels: outputChannel, kernelSize: IntOrPair((1, 1)), bias: true)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = denseBlock(x)
        h = upConv1(h)
        h = upConv2(h.transposed(0, 1, 3, 2)).transposed(0, 1, 3, 2)
        let xr = toChannelsFirst(phaseConvR(toChannelsLast(h)))
        let xi = toChannelsFirst(phaseConvI(toChannelsLast(h)))
        return atan2(xi, xr)
    }
}

/// Depthwise **causal** 1-D convolution, drop-in for the Mamba `conv1d`.
///
/// MLX's generic grouped `Conv1d` is pathologically slow here (~87 ms/call for a
/// kernel-4 depthwise conv). The same op is just `K` shifted, per-channel scaled adds,
/// which run as cheap broadcasted elementwise ops. Parameter keys (`weight` `[C,K,1]`,
/// `bias` `[C]`) match MLX `Conv1d` so the checkpoint loads unchanged. Numerically
/// identical to `Conv1d(padding: K-1)` trimmed to causal length.
public final class DepthwiseCausalConv1d: Module {
    public let channels: Int
    public let kernel: Int
    @ParameterInfo(key: "weight") public var weight: MLXArray  // [C, K, 1]
    @ParameterInfo(key: "bias") public var bias: MLXArray      // [C]

    public init(channels: Int, kernel: Int) {
        self.channels = channels
        self.kernel = kernel
        self._weight.wrappedValue = MLXArray.zeros([channels, kernel, 1], dtype: .float32)
        self._bias.wrappedValue = MLXArray.zeros([channels], dtype: .float32)
    }

    /// `x`: `[B, L, C]` channels-last. Returns `[B, L, C]`, optionally with SiLU fused in.
    /// `reverse` runs the conv anti-causally (for the backward Mamba direction).
    public func callAsFunction(_ x: MLXArray, applySilu: Bool = false, reverse: Bool = false) -> MLXArray {
        FusedDepthwiseConv.callAsFunction(
            x: x,
            weight: weight.reshaped(channels, kernel), // [C, K]
            bias: bias,
            applySilu: applySilu,
            reverse: reverse
        )
    }
}

public final class MambaSSM: Module {
    public let dModel: Int
    public let dState: Int
    public let dConv: Int
    public let dInner: Int
    public let dtRank: Int

    @ModuleInfo(key: "in_proj") public var inProj: Linear
    @ModuleInfo(key: "conv1d") public var conv1d: DepthwiseCausalConv1d
    @ModuleInfo(key: "x_proj") public var xProj: Linear
    @ModuleInfo(key: "dt_proj") public var dtProj: Linear
    @ModuleInfo(key: "out_proj") public var outProj: Linear
    @ParameterInfo(key: "A_log") public var ALog: MLXArray
    @ParameterInfo(key: "D") public var D: MLXArray

    public init(dModel: Int, dState: Int = 16, dConv: Int = 4, expand: Int = 4) {
        self.dModel = dModel
        self.dState = dState
        self.dConv = dConv
        self.dInner = expand * dModel
        self.dtRank = (dModel + 15) / 16

        self._inProj.wrappedValue = Linear(dModel, dInner * 2, bias: false)
        self._conv1d.wrappedValue = DepthwiseCausalConv1d(channels: dInner, kernel: dConv)
        self._xProj.wrappedValue = Linear(dInner, dtRank + dState * 2, bias: false)
        self._dtProj.wrappedValue = Linear(dtRank, dInner, bias: true)
        self._outProj.wrappedValue = Linear(dInner, dModel, bias: false)
        self._ALog.wrappedValue = MLXArray.zeros([dInner, dState], dtype: .float32)
        self._D.wrappedValue = MLXArray.ones([dInner], dtype: .float32)
    }

    /// `reverse` computes `flip(mamba(flip(x)))` in place for the backward direction:
    /// all projections are per-token (flip-invariant), so only the conv (anti-causal)
    /// and the scan (reverse-time) change — no sequence copies needed.
    public func callAsFunction(_ x: MLXArray, reverse: Bool = false) -> MLXArray {
        let xz = ReuseProfiler.measure("in_proj") { inProj(x) }
        let parts = xz.split(parts: 2, axis: -1)
        let xIn = parts[0]
        let z = parts[1]

        let xConv = ReuseProfiler.measure("conv1d+silu") {
            conv1d(xIn, applySilu: true, reverse: reverse)
        }

        let splitBC = ReuseProfiler.measure("x_proj+split") {
            xProj(xConv)
        }
        let bc = MLX.split(splitBC, indices: [dtRank, dtRank + dState], axis: -1)
        let dt = bc[0]
        let Bvar = bc[1]
        let Cvar = bc[2]

        let delta = ReuseProfiler.measure("dt_matmul") { dt.matmul(dtProj.weight.T) }
        let A = -exp(ALog.asType(.float32))

        // Channels-last [B, L, *] arrays feed the scan directly — no transposes.
        let y = ReuseProfiler.measure("scan") {
            ReuseSelectiveScan.fused(
                u: xConv,
                delta: delta,
                A: A,
                Bvar: Bvar,
                Cvar: Cvar,
                D: D,
                z: z,
                deltaBias: dtProj.bias,
                deltaSoftplus: true,
                reverse: reverse,
                outputDType: x.dtype
            )
        }
        return ReuseProfiler.measure("out_proj") { outProj(y) }
    }
}

public final class MambaBlock: Module {
    @ModuleInfo(key: "forward_blocks") public var forwardBlocks: MambaSSM
    @ModuleInfo(key: "backward_blocks") public var backwardBlocks: MambaSSM
    @ModuleInfo(key: "output_proj") public var outputProj: Linear
    @ModuleInfo(key: "norm") public var norm: LayerNorm

    public init(dModel: Int, dState: Int = 16, dConv: Int = 4, expand: Int = 4) {
        self._forwardBlocks.wrappedValue = MambaSSM(dModel: dModel, dState: dState, dConv: dConv, expand: expand)
        self._backwardBlocks.wrappedValue = MambaSSM(dModel: dModel, dState: dState, dConv: dConv, expand: expand)
        self._outputProj.wrappedValue = Linear(2 * dModel, dModel, bias: true)
        self._norm.wrappedValue = LayerNorm(dimensions: dModel)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let outFw = forwardBlocks(x) + x
        // Backward direction runs in place: `backwardBlocks(x, reverse: true)` equals
        // flip(backward(flip(x))), so the two sequence flips (and their gather+copy)
        // vanish. The residual is `+ x` because flip(flip(x)) == x.
        let outBw = backwardBlocks(x, reverse: true) + x
        return ReuseProfiler.measure("output_proj+norm") { norm(outputProj(concatenated([outFw, outBw], axis: -1))) }
    }
}

public final class TFMambaBlock: Module {
    @ModuleInfo(key: "time_mamba") public var timeMamba: MambaBlock
    @ModuleInfo(key: "freq_mamba") public var freqMamba: MambaBlock

    public init(dModel: Int = 64, dState: Int = 16, dConv: Int = 4, expand: Int = 4) {
        self._timeMamba.wrappedValue = MambaBlock(dModel: dModel, dState: dState, dConv: dConv, expand: expand)
        self._freqMamba.wrappedValue = MambaBlock(dModel: dModel, dState: dState, dConv: dConv, expand: expand)
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let c = x.dim(1)
        let t = x.dim(2)
        let f = x.dim(3)

        // [B,C,T,F] -> [B,F,T,C] -> [B*F,T,C]
        var xt = ReuseProfiler.measure("tf_reshape") { x.transposed(0, 3, 2, 1).reshaped(b * f, t, c) }
        xt = timeMamba(xt) + xt

        // [B*F,T,C] -> [B,T,F,C] -> [B*T,F,C]
        var xf = ReuseProfiler.measure("tf_reshape") { xt.reshaped(b, f, t, c).transposed(0, 2, 1, 3).reshaped(b * t, f, c) }
        xf = freqMamba(xf) + xf

        return ReuseProfiler.measure("tf_reshape") { xf.reshaped(b, t, f, c).transposed(0, 3, 1, 2) }
    }
}

public final class SEMamba: Module {
    @ModuleInfo(key: "dense_encoder") public var denseEncoder: DenseEncoder
    public let TSMamba: [TFMambaBlock]
    @ModuleInfo(key: "mask_decoder") public var maskDecoder: MagDecoder
    @ModuleInfo(key: "phase_decoder") public var phaseDecoder: PhaseDecoder

    /// Dtype used for the dense/Mamba compute graph. The selective-scan kernel always
    /// accumulates in float32 internally regardless of this; only the surrounding
    /// matmul/conv/elementwise ops run in this dtype. `.float32` is exact; `.bfloat16`
    /// trades a little accuracy for speed. `.float16` is fastest but numerically
    /// unstable across this model's 30 blocks (overflow) — avoid it.
    public var computeDType: DType = .float32

    public init(
        numTFMamba: Int = 30,
        hidFeature: Int = 64,
        dState: Int = 16,
        dConv: Int = 4,
        expand: Int = 4,
        inputChannel: Int = 2,
        outputChannel: Int = 1
    ) {
        self._denseEncoder.wrappedValue = DenseEncoder(inputChannel: inputChannel, hidFeature: hidFeature)
        self.TSMamba = (0 ..< numTFMamba).map { _ in
            TFMambaBlock(dModel: hidFeature, dState: dState, dConv: dConv, expand: expand)
        }
        self._maskDecoder.wrappedValue = MagDecoder(hidFeature: hidFeature, outputChannel: outputChannel)
        self._phaseDecoder.wrappedValue = PhaseDecoder(hidFeature: hidFeature, outputChannel: outputChannel)
    }

    /// Inputs are channel-first STFT magnitude/phase `[B, F, T]`.
    public func callAsFunction(noisyMag: MLXArray, noisyPhase: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let f = noisyMag.dim(1)
        let t = noisyMag.dim(2)

        let mag = expandedDimensions(noisyMag.transposed(0, 2, 1), axis: 1).asType(computeDType)
        let pha = expandedDimensions(noisyPhase.transposed(0, 2, 1), axis: 1).asType(computeDType)
        var x = concatenated([mag, pha], axis: 1) // [B,2,T,F]

        let b = x.dim(0)
        let c = x.dim(1)
        let time = x.dim(2)
        let freq = x.dim(3)
        x = concatenated([x, MLXArray.zeros([b, c, time, 2], dtype: x.dtype)], axis: -1)
        x = concatenated([x, MLXArray.zeros([b, c, 2, freq + 2], dtype: x.dtype)], axis: -2)

        ReuseProfiler.reset()
        x = ReuseProfiler.measure("dense_encoder") { denseEncoder(x) }
        for block in TSMamba {
            x = block(x)
        }

        var denoisedMag = ReuseProfiler.measure("decoders") { maskDecoder(x) }.transposed(0, 3, 2, 1)[0..., 0..., 0..., 0]
        var denoisedPha = ReuseProfiler.measure("decoders") { phaseDecoder(x) }.transposed(0, 3, 2, 1)[0..., 0..., 0..., 0]
        ReuseProfiler.dump("SEMamba forward (per-section, sequential eval)")
        denoisedMag = denoisedMag[0..., 0 ..< f, 0 ..< t].asType(.float32)
        denoisedPha = denoisedPha[0..., 0 ..< f, 0 ..< t].asType(.float32)

        let denoisedCom = stacked([
            denoisedMag * cos(denoisedPha),
            denoisedMag * sin(denoisedPha),
        ], axis: -1)
        return (denoisedMag, denoisedPha, denoisedCom)
    }
}
