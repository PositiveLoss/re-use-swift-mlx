import Foundation
import MLX
import MLXFast

/// Fused selective scan for the RE-USE / SEMamba Mamba block.
///
/// Layout mirrors `mlx_speech.models.reuse.mamba.scan.selective_scan`:
///
///   u, delta, z: [B, dInner, L]
///   A:           [dInner, dState]
///   Bvar, Cvar:  [B, dState, L]
///   D, bias:     [dInner]
///
/// The Metal kernel fuses:
///
///   delta = softplus(delta + deltaBias)
///   state[n] = exp(delta * A[d,n]) * state[n] + delta * B[b,n,t] * u[b,d,t]
///   out = (sum_n state[n] * C[b,n,t] + D[d] * u) * silu(z)
///
/// It intentionally uses one GPU thread per (batch, dInner) lane and keeps the
/// small Mamba state vector in registers. This is the fastest practical path
/// for RE-USE's small `dState` on Apple GPUs without writing a full hierarchical
/// prefix-scan implementation.
public enum ReuseSelectiveScan {
    /// Maximum state dimension stored in thread registers by the Metal kernel.
    /// RE-USE / SEMamba uses dState = 16.
    public static let maxRegisterState = 64

    private static let fusedKernel = MLXFast.metalKernel(
        name: "reuse_semamba_selective_scan_fused",
        inputNames: ["u", "delta", "A", "Bvar", "Cvar", "Dvec", "z", "deltaBias", "params"],
        outputNames: ["out"],
        source: """
        uint lane = thread_position_in_grid.x;

        uint dim = uint(params[0]);
        uint length = uint(params[1]);
        uint dstate = uint(params[2]);
        bool hasD = uint(params[3]) != 0;
        bool hasZ = uint(params[4]) != 0;
        bool hasDeltaBias = uint(params[5]) != 0;
        bool deltaSoftplus = uint(params[6]) != 0;
        bool reverse = uint(params[7]) != 0;

        uint b = lane / dim;
        uint d = lane - b * dim;

        float state[64];
        float Aloc[64];
        for (uint n = 0; n < dstate; ++n) {
            state[n] = 0.0f;
            // A[d,n] is invariant across time; hoist the lane's row into registers
            // so the t-loop never re-reads it from device memory.
            Aloc[n] = float(A[d * dstate + n]);
        }

        // Channels-last layout:
        //   u, delta, z, out: [B, L, dim]   -> index (b*length + t)*dim + d
        //   Bvar, Cvar:       [B, L, dstate]-> index (b*length + t)*dstate + n
        //   A:                [dim, dstate] -> index d*dstate + n
        // Adjacent lanes (d, d+1) touch adjacent memory, so device reads coalesce.
        // `reverse` walks time backwards (t = L-1..0) for the backward Mamba direction,
        // which is exactly flip(scan(flip(input))) with no materialized flips.
        for (uint i = 0; i < length; ++i) {
            uint t = reverse ? (length - 1 - i) : i;
            uint rowU = (b * length + t) * dim;
            uint rowS = (b * length + t) * dstate;
            uint udx = rowU + d;

            float uVal = float(u[udx]);
            float dt = float(delta[udx]);

            if (hasDeltaBias) {
                dt += float(deltaBias[d]);
            }
            if (deltaSoftplus) {
                // Numerically stable softplus(x) = log(1 + exp(x)).
                // This branch avoids overflow for large positive x.
                if (dt > 20.0f) {
                    // approximately identity
                } else if (dt < -20.0f) {
                    dt = fast::exp(dt);
                } else {
                    dt = fast::log(1.0f + fast::exp(dt));
                }
            }

            float acc = 0.0f;
            float dt_log2e = dt * 1.4426950408889634f;

            for (uint n = 0; n < dstate; ++n) {
                uint bn = rowS + n;

                float aBar = fast::exp2(dt_log2e * Aloc[n]);
                float bBar = dt * float(Bvar[bn]) * uVal;

                state[n] = aBar * state[n] + bBar;
                acc += state[n] * float(Cvar[bn]);
            }

            if (hasD) {
                acc += float(Dvec[d]) * uVal;
            }

            if (hasZ) {
                float zVal = float(z[udx]);
                // silu(z) = z * sigmoid(z)
                acc *= zVal / (1.0f + fast::exp(-zVal));
            }

            out[udx] = static_cast<T>(acc);
        }
        """
    )

    /// Fused selective scan on **channels-last** tensors:
    ///
    ///   u, delta, z: [B, L, dInner]
    ///   A:           [dInner, dState]
    ///   Bvar, Cvar:  [B, L, dState]
    ///   D, bias:     [dInner]
    ///
    /// Inputs are consumed in their native dtype (fp16/fp32); the kernel reads them
    /// through `float(...)` and always accumulates the recurrence in float32. This
    /// avoids the transpose + float32 materialization the old channels-first layout
    /// required, which is the dominant per-call overhead for RE-USE.
    public static func fused(
        u: MLXArray,
        delta: MLXArray,
        A: MLXArray,
        Bvar: MLXArray,
        Cvar: MLXArray,
        D: MLXArray?,
        z: MLXArray?,
        deltaBias: MLXArray?,
        deltaSoftplus: Bool = true,
        reverse: Bool = false,
        outputDType: DType = .float32
    ) -> MLXArray {
        precondition(u.shape.count == 3, "u must be [B, L, dInner]")
        precondition(delta.shape == u.shape, "delta shape must match u")
        precondition(z == nil || z!.shape == u.shape, "z shape must match u")
        precondition(A.shape.count == 2, "A must be [dInner, dState]")
        precondition(Bvar.shape.count == 3, "Bvar must be [B, L, dState]")
        precondition(Cvar.shape == Bvar.shape, "Cvar shape must match Bvar")

        let batch = u.shape[0]
        let length = u.shape[1]
        let dim = u.shape[2]
        let dState = A.shape[1]

        precondition(A.shape[0] == dim, "A.shape[0] must equal dInner")
        precondition(Bvar.shape[0] == batch, "Bvar batch must match u")
        precondition(Bvar.shape[1] == length, "Bvar length must match u")
        precondition(Bvar.shape[2] == dState, "Bvar dState must match A")
        precondition(dState <= maxRegisterState, "dState \(dState) exceeds maxRegisterState \(maxRegisterState)")
        precondition(D == nil || D!.shape == [dim], "D must be [dInner]")
        precondition(deltaBias == nil || deltaBias!.shape == [dim], "deltaBias must be [dInner]")

        // Native dtype; only ensure row-major contiguity (a no-op for arrays that
        // already flow channels-last out of the preceding matmuls).
        let u32 = contiguous(u)
        let delta32 = contiguous(delta)
        let A32 = contiguous(A)
        let B32 = contiguous(Bvar)
        let C32 = contiguous(Cvar)

        let dVec = contiguous(D ?? MLXArray.zeros([dim], dtype: u.dtype))
        let zVal = contiguous(z ?? MLXArray.zeros(u.shape, dtype: u.dtype))
        let bias = contiguous(deltaBias ?? MLXArray.zeros([dim], dtype: u.dtype))

        let params = MLXArray([
            Float(dim),
            Float(length),
            Float(dState),
            D == nil ? 0.0 : 1.0,
            z == nil ? 0.0 : 1.0,
            deltaBias == nil ? 0.0 : 1.0,
            deltaSoftplus ? 1.0 : 0.0,
            reverse ? 1.0 : 0.0
        ]).asType(.float32)

        let lanes = batch * dim
        let threadGroup = min(256, max(1, lanes))

        return fusedKernel(
            [u32, delta32, A32, B32, C32, dVec, zVal, bias, params],
            template: [("T", outputDType)],
            grid: (lanes, 1, 1),
            threadGroup: (threadGroup, 1, 1),
            outputShapes: [u.shape],
            outputDTypes: [outputDType]
        )[0]
    }
}
