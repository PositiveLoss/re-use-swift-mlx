import Foundation
import MLX
import MLXFast

/// Single-pass depthwise **causal** conv1d fused with an optional SiLU activation.
///
/// One GPU thread per output element `(b, t, c)` of a channels-last `[B, L, C]` tensor.
/// Each thread reads `K` causal input taps, applies the per-channel weights + bias, and
/// (optionally) SiLU, writing one value. Adjacent threads walk adjacent channels, so
/// device reads/writes coalesce. This replaces ~6 elementwise passes (K shifted adds +
/// bias + silu) with a single pass over the tensor.
public enum FusedDepthwiseConv {
    private static let kernel = MLXFast.metalKernel(
        name: "reuse_depthwise_causal_conv1d_silu",
        inputNames: ["x", "weight", "bias", "params"],
        outputNames: ["out"],
        source: """
        uint gid = thread_position_in_grid.x;

        uint C = uint(params[0]);
        uint L = uint(params[1]);
        uint K = uint(params[2]);
        bool applySilu = uint(params[3]) != 0;
        uint total = uint(params[4]);
        bool reverse = uint(params[5]) != 0;
        if (gid >= total) { return; }

        uint c = gid % C;
        uint bt = gid / C;          // = b * L + t
        uint t = bt % L;
        uint bBase = (bt - t) * C;  // (b * L) * C

        // Causal taps look back (t-(K-1)..t); the reverse (anti-causal) branch looks
        // forward (t..t+(K-1)) so that flip(causalConv(flip(x))) is produced directly.
        float acc = float(bias[c]);
        for (uint k = 0; k < K; ++k) {
            int srcT = reverse ? (int(t) + int(K - 1) - int(k)) : (int(t) - int(K - 1) + int(k));
            if (srcT >= 0 && srcT < int(L)) {
                acc += float(weight[c * K + k]) * float(x[bBase + uint(srcT) * C + c]);
            }
        }
        if (applySilu) {
            acc = acc / (1.0f + exp(-acc));
        }
        out[gid] = static_cast<T>(acc);
        """
    )

    /// `x`: `[B, L, C]`; `weight`: `[C, K]`; `bias`: `[C]`. Returns `[B, L, C]`.
    /// `reverse` selects the anti-causal (forward-looking) taps for the backward Mamba.
    public static func callAsFunction(
        x: MLXArray, weight: MLXArray, bias: MLXArray, applySilu: Bool, reverse: Bool = false
    ) -> MLXArray {
        let b = x.shape[0], l = x.shape[1], c = x.shape[2]
        let k = weight.shape[1]
        let total = b * l * c

        let xc = contiguous(x)
        let wc = contiguous(weight.asType(x.dtype))
        let bc = contiguous(bias.asType(x.dtype))
        let params = MLXArray([Float(c), Float(l), Float(k), applySilu ? 1.0 : 0.0, Float(total), reverse ? 1.0 : 0.0]).asType(.float32)

        let threadGroup = min(256, max(1, total))
        return kernel(
            [xc, wc, bc, params],
            template: [("T", x.dtype)],
            grid: (total, 1, 1),
            threadGroup: (threadGroup, 1, 1),
            outputShapes: [x.shape],
            outputDTypes: [x.dtype]
        )[0]
    }
}
