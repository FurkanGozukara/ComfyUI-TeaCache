# MiniMax H3 SOL update

Checked 2026-09-13. Baseline: `a197f9eaee0b9088e12e047b6a6eeb073b152481`.

## Upstream findings

| Source inspected | Finding and decision |
| --- | --- |
| [Comfy Kitchen #117](https://github.com/Comfy-Org/comfy-kitchen/pull/117), [#150](https://github.com/Comfy-Org/comfy-kitchen/pull/150), [#156](https://github.com/Comfy-Org/comfy-kitchen/pull/156) | Merged August 29, September 3 and September 5: native INT8 SOL, improved exact-branch probability quantization/FP16 support, and extra token routing. Reuse the released public API; keep 256 extra tokens and full-precision prefix queries. |
| [Kijai](https://github.com/kijai/ComfyUI-SolAttn_triton/tree/26d816ebd4f1e43a2c6e4d4759be3137f10a7a73) | September 7 HEAD explicitly deprecates the custom node in favor of native ComfyUI/Comfy Kitchen. Its newer-looking Chroma01 and Goldlionren forks have the same HEAD. |
| [KingGore](https://github.com/KingGore/ComfyUI_sol-attn_Blackwell/tree/a8a9584e1ed700f2ce3b7569048cab0071bbf58a) | August 4 HEAD; one branch, neither listed fork newer. Its dense trailing steps are useful. The hard-mask flex implementation drops unrouted blocks, so its speed is not equivalent to SOL with approximate tail correction. Retain the correction and add trailing dense steps. |
| [Saganaki22](https://github.com/Saganaki22/ComfyUI-sol-attn/tree/930a4d6e432ff8b8ed5e30ff2f72519b92d69bdf) | August 13 HEAD; strided QKV, residual INT8, scheduled tau and broader GPU support. Native Kitchen supplies the relevant strided execution and newer quality correction without adding another vendored kernel tree. |
| [NVlabs sol-engine](https://github.com/NVlabs/Sana/tree/8e0db4fa562d727ea28b8d63015c196db7d97cae) | September 11 HEAD. The latest CUDA change in the shared SOL subtree is the August 15 SM89 dispatch already present here; the subsequent Metal backend is unrelated to NVIDIA execution. |
| [NVlabs Sol-H3](https://github.com/NVlabs/Sana/blob/8e0db4fa562d727ea28b8d63015c196db7d97cae/models/minimax_h3/Sol-H3/README.md) | September 7 runtime plus September 9 MXFP8 work. Reported large gains combine four-forward distillation and multi-GPU B300 execution; MXFP8 compute is SM100-family only. They do not establish an RTX 5090 attention-only gain. No checkpoint, adapter, quantization or multi-GPU recipe replacement. |
| [NVlabs Sol-H3 Spark](https://github.com/NVlabs/Sana/blob/8e0db4fa562d727ea28b8d63015c196db7d97cae/models/minimax_h3/Sol-H3-Spark/README.md) | September 10/11 two-stage H3-to-LTX-2.5 pipeline for Linux aarch64 GB10/SM121. Requires several additional models and changes the generation pipeline. It is a separate project, not a replacement attention backend for existing H3 presets. |
| Open Kitchen PRs [#146](https://github.com/Comfy-Org/comfy-kitchen/pull/146), [#168](https://github.com/Comfy-Org/comfy-kitchen/pull/168), [#171](https://github.com/Comfy-Org/comfy-kitchen/pull/171) | Respectively a no-tail exact-kernel optimization, optional routing telemetry, and chunked key-bias support. None is needed for this unmasked, tail-preserving integration. No unreleased wheel or pending patch is required. |

Branches and fork listings were inspected for NVlabs, Kijai, KingGore and Saganaki22. This is a bounded survey, not a claim to have reviewed every GitHub fork. Kitchen source HEAD was `21003fa97bf3b180393446d729ae630ceb6c2a52`; the installed released package was 0.2.33.

## Implementation

- `sparse_backend=auto` verifies and benchmarks native Kitchen and bundled SOL, then selects the fastest qualifying path. `sparse_attention=auto` still requires at least a 5% win over the incumbent backend; `enabled` bypasses the speed requirement, not the numerical check.
- `sparse_extra_tokens=256` enables Kitchen's additional token routing. The approximate contribution of unrouted blocks remains enabled. `0` is an explicit speed/accuracy tradeoff.
- Text, reference and audio prefix KV blocks remain fully routed. Their query outputs are recomputed with full-precision SDPA; this retains the pre-existing conditioning-query arithmetic even with an INT8 kernel.
- The benchmark includes input conversions, sparse execution and dense prefix recomputation. NaN/Inf cannot pass the correctness check. Decisions are specific to device, dtype, shape/head count, strides, prefix, score scale and token budget; a failed shape does not disable other shapes.
- `sparse_dense_last_steps=1` protects the final detail step. FirstBlockCache always computes the final step, including short schedules; previously the fractional `0.95` end bound did not ensure this. Setting the new control to `0` restores the previous sparse-attention end window while retaining the cache correction.
- Preserve the attention score scale, FP16/BF16 output dtype and `skip_output_reshape` contract. New optional widgets follow all original widgets, including the master switch; existing preset JSON requires no rewiring.
- Use `comfy-kitchen>=0.2.33`. Existing Windows, RunPod and Massed Compute installers already target this fork and install requirements. The workspace's shared installer requirements also specify this minimum.

## Validation

Windows; physical GPU 0, RTX 5090; torch 2.13.0+cu130, comfy-kitchen 0.2.33, triton-windows 3.7.1.post27, comfy-aimdo 0.5.3; ComfyUI `7dac1d2512c20c3c2cc201b24180cde18e41b31a`. CUDA was explicitly restricted to GPU UUID `GPU-21350b86-3099-1af0-58ef-f6432e60d9d0`.

An isolated native ComfyUI execution used the existing `minimax_h3_fl2va_pruned_int8_convrot.safetensors` and `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`, 1344x768, 124 frames and five Euler steps. FirstBlockCache was disabled for this attention comparison. The test output checked finite video/audio latent tensors without decoding or exporting media. Allocation compilation remained enabled: the run completed with zero rogue allocations and no AIMDO error.

Each comparison below uses the **same real H3 QKV tensors**: 37,741 tokens, 56 heads, 128 head dimension, 445 prefix tokens. Timings are medians of five warmed CUDA-event measurements of the complete attention path. Relative-L2 errors compare video query rows sampled every 145 tokens with dense SDPA over all keys/values and all heads.

| Step / block (zero-based) | Bundled SOL ms | New SOL + 256 tokens ms | Attention speed ratio | Bundled error | New error |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 / 2 | 111.23 | 44.85 | 2.48x | 0.06392 | 0.04966 |
| 1 / 25 | 110.28 | 43.19 | 2.55x | 0.09283 | 0.06922 |
| 3 / 2 | 85.04 | 40.57 | 2.10x | 0.04862 | 0.03688 |
| 3 / 25 | 121.33 | 46.10 | 2.63x | 0.09716 | 0.06106 |

The new path has 22–37% lower sampled attention error than bundled SOL in these four calls. Prefix-query error is exactly zero for both. The full all-block correctness gate measured relative L2 0.00670 for Kitchen and 0.00047 for bundled SOL, below the unchanged 0.02 limit. The gate selected Kitchen at 43.30 ms versus incumbent SageAttention at 81.86 ms. Bundled SOL, including its copies/prefix cost, measured 110.40 ms and would have been rejected.

Additional checks:

- 12 CPU unit tests: backend dispatch, legacy imports, final-step schedules, prefix/scale/token plumbing, zero-copy views, head output layout, workload-specific fallback, nonfinite rejection and old widget order.
- Actual CUDA FP16/BF16, batch 2, 1,025 ragged tokens, nondefault score scale: all-block relative L2 0.01133 / 0.01164; exact prefix equality and dtype/shape preserved.
- Actual CUDA 103,237-token, 56-head fused-QKV strided inputs: native output bit-identical to contiguous inputs, finite, with 256 extra tokens. This exercises addressing beyond signed-int32 fused-buffer offsets.
- Real `cudaMallocAsync`/AIMDO FirstBlockCache regression for GPU and CPU cache storage: eight steps, four computed / four reused, 20 original block calls, exact expected tensors, final step computed.
- `git diff --check` passes. Both shipped backends remain installed; no media, model or preset assets were replaced.

Limits: the five-step run validates execution and attention numerics, not a recommended generation recipe. No end-to-end speed ratio, decoded video-quality rating, audio listening result, Ref2VA validation, Linux execution or other-GPU result is claimed. Sparse attention remains approximate; the incumbent dense backend is numerically closer to SDPA. Synthetic-input timings are not substituted for real-model measurements. Kernel checks and token routing do not guarantee identical or better artistic output for every prompt.

Local reproducibility receipts, raw measurements, prompts and test helpers are in `G:/ComfyUI_V92/SOL_Attention_Upgrade/`. The first isolated run encountered the existing ComfyUI database lock; database initialization was skipped and the prompt completed successfully. The reusable launcher now supplies a separate test database. The main ComfyUI process was not restarted.
