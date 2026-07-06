# RE-USE Fast Swift / MLX

Native Swift implementation for running the RE-USE / SEMamba speech-enhancement model on Apple Silicon with a fused MLX/Metal selective-scan kernel.

This package is meant for the `faraday/re-use-mlx` checkpoint, which contains `model_mlx.safetensors` converted for Apple Silicon / MLX / MLX Swift use. The code also accepts a direct `.safetensors` path.

## Contents

```text
Package.swift
Sources/
  ReuseFastSwift/
    SelectiveScan.swift   # fused Metal selective scan for SEMamba/Mamba
    SEMamba.swift         # Swift MLX SEMamba architecture
    ModelLoader.swift     # safetensors loading + key compatibility
    STFT.swift            # RE-USE STFT/ISTFT + long-file OLA
    Enhancer.swift        # waveform -> model -> waveform pipeline
    WavIO.swift           # minimal WAV reader/writer
    DepthwiseConv.swift   # custom Metal depthwise causal 1D conv kernel
    Profiler.swift        # helper utility for debugging module execution times
  reuse-fast-cli/
    main.swift            # CLI for inference and scan benchmark
```

## Requirements

- Apple Silicon Mac
- macOS 14+
- Xcode command-line tools or Xcode
- Swift 6.1+ compatible toolchain
- Internet access on first build so SwiftPM can fetch `mlx-swift`

## Download weights

```bash
pip install 'huggingface_hub[hf_xet]'
huggingface-cli download --local-dir re-use-mlx faraday/re-use-mlx
```

The loader looks for either:

```text
re-use-mlx/model_mlx.safetensors
re-use-mlx/model.safetensors
/path/to/file.safetensors
```

## Build

From this directory:

```bash
swift build -c release
```

If your local MLX Swift checkout requires Xcode's Metal build pipeline, use:

```bash
xcodebuild \
  -scheme reuse-fast-cli \
  -configuration Release \
  -destination 'platform=macOS' \
  -derivedDataPath .build/xcode \
  build
```

## Run inference

With SwiftPM build output:

```bash
.build/release/reuse-fast-cli \
  --weights re-use-mlx \
  --input noisy.wav \
  --output clean.wav
```

With the Xcode build path:

```bash
.build/xcode/Build/Products/Release/reuse-fast-cli \
  --weights re-use-mlx \
  --input noisy.wav \
  --output clean.wav
```

Useful flags:

```bash
--chunk-size-s 1.0       # long-file chunk size
--hop-portion 0.5        # chunk overlap ratio
--dtype bf16             # compute dtype: bf16 (default), fp32, or fp16
--strict                 # strict parameter verification
--print-checkpoint-keys  # inspect a checkpoint
--print-model-keys       # inspect expected Swift model keys
--verify-scan            # numerically verify fused scan kernel correctness
--compare-scan           # compare speed of fused vs parallel scan
```

## Benchmark only the fused selective scan

```bash
.build/release/reuse-fast-cli --benchmark-scan
```

RE-USE-like defaults (channels-last layout) are:

```text
u:      [82, 51, 256]   # [B, L, dInner]
A:      [256, 16]       # [dInner, dState]
B, C:   [82, 51, 16]    # [B, L, dState]
```

You can override them:

```bash
.build/release/reuse-fast-cli \
  --benchmark-scan \
  --scan-batch 51 \
  --scan-dim 256 \
  --scan-length 82 \
  --scan-dstate 16 \
  --iterations 100
```


## Benchmark & Performance Results

Below are the benchmark and performance timings obtained using the sample audio file `audio.opus` (from `~/Downloads/audio.opus`) on an Apple Silicon Mac.

### 1. Fused Selective Scan Kernel
We measured the performance of our Metal fused selective scan kernel (`MLXFast.metalKernel`) versus standard/reference MLX layouts using:
```bash
.build/xcode/Build/Products/Release/reuse-fast-cli --benchmark-scan
```

| Benchmark Metric | Value | Shape configuration |
| --- | --- | --- |
| Average selective-scan time | **0.169 ms/run** | batch=82, dInner=256, length=51, dState=16 (50 iterations) |

### 2. Full Audio Enhancement (Inference)
We benchmarked full model enhancement on the 10-second Opus input file (`audio.opus`) converted to WAV at different sample rates. The command processed batches of 4 chunks of 1.0s each with 0.5s overlap (OLA processing):

```bash
REUSE_TIMING=1 .build/xcode/Build/Products/Release/reuse-fast-cli \
  --weights /Users/yehorsmoliakov/Downloads/langpipe/Models/enhance/re-use-mlx \
  --input audio_16k.wav \
  --output audio_16k_clean.wav
```

| Audio Sample Rate | Audio Duration | Batch Size | STFT Time (per batch) | Model Time (per batch) | ISTFT Time (per batch) | Performance vs. Real-time |
| --- | --- | --- | --- | --- | --- | --- |
| **16 kHz** | 10.0s | 4 chunks (4s) | ~4 ms | ~3500 ms | ~15 ms | **~1.1x faster** than real-time |
| **8 kHz** | 10.0s | 4 chunks (4s) | ~2 ms | ~1500 ms | ~5 ms | **~2.5x faster** than real-time |

*Note: Initial warm-up batch includes Metal kernel compilation and model weight loading overhead (~4.8 seconds).*
## What is fast here

`SelectiveScan.swift` fuses the SEMamba/Mamba recurrence into one `MLXFast.metalKernel` dispatch. The kernel keeps the Mamba state vector in Metal thread registers and uses one GPU thread for each `(batch, dInner)` lane:

```text
u, delta, z: [B, L, dInner]
A:           [dInner, dState]
B, C:        [B, L, dState]
D, bias:     [dInner]
```

For each time step:

```text
dt       = softplus(delta + delta_bias)
state[n] = exp(dt * A[d,n]) * state[n] + dt * B[b,n,t] * u[b,d,t]
y[t]     = (sum_n state[n] * C[b,n,t] + D[d] * u[b,d,t]) * silu(z[b,d,t])
```

That avoids the expensive pure-Swift or pure-MLX loop over sequence length and avoids materializing the large `deltaA` / `deltaB_u` intermediates.

## Limitations

- This targets MLX/Metal GPU execution, not the Apple Neural Engine.
- The model code is keyed for the Faraday MLX/Swift safetensors naming convention and includes aliases for common Python/camelCase variants. If loading fails, run `--print-checkpoint-keys` and compare with `--print-model-keys`.
- The WAV writer emits mono 32-bit IEEE float WAV.
- The RE-USE license in the model card is non-commercial. Check it before using the model outside research/education.
