# Native MiniMax H3 SOL attention

Updated 2026-10-08. The duplicate bundled SOL kernel tree and the custom QKV/projection implementation have been removed. `MiniMaxH3SpeedOptimizer` now delegates to ComfyUI's `SparseAttnPatch`, `h3_eligible` and `h3_sparse_attention`. ComfyUI core is unchanged.

## Controls and compatibility

- The existing node class, positional widgets and saved workflow IDs remain compatible. The master switch disables the optimizer. SOL and FirstBlockCache remain independent choices.
- `sparse_attention=disabled` preserves the incoming attention path. `enabled` requests native SOL without the startup speed benchmark. `auto` compares complete projected-attention calls and requires at least a 5% speed gain and finite output; route-all numerical validation is recorded separately in the audit.
- The historical backend values `auto`, `comfy_kitchen`, and `vendored` remain accepted aliases for `native`; none loads a bundled implementation.
- Native SOL keeps the approximate tail contribution, 256 additional exact tokens and native H3 conditioning sinks. Dense first/last sampling steps and dense transformer blocks 0 and 1 remain conservative defaults. Existing sparse/SLA patches compose with FirstBlockCache.
- SOL and cache reuse are approximate. Routing checks detect broken computation; they do not prove identical images, imperceptible error or better lip sync.
- Requires a ComfyUI version exposing the native H3 sparse APIs (tested commit `52f98af2`) and `comfy-kitchen>=0.2.37`. The existing SECourses installers already update this fork and install its requirements.

## End-to-end measurements

RTX 5090, physical GPU 0; Windows; torch 2.13.0+cu130, Kitchen 0.2.37, triton-windows 3.7.1.post27. FL2VA INT8 ConvRot, LightX2V Turbo 4 steps, Euler/simple, CFG 1, 832x1248, 10-second locked narration, same portrait and seed 20261010. FirstBlockCache off. Times include executed prompt work and final video export; model/compile state was warm. Single runs, not a confidence interval.

| Path | Seconds | Speed vs dense |
| --- | ---: | ---: |
| Dense | 110.935 | 1.00x |
| Updated compatibility node, protected first/last steps, tau 1 | 99.235 | 1.12x |
| Native BlockSparseAttention, every step, tau 1 | 83.074 | 1.34x |
| Native BlockSparseAttention, every step, tau 1.3 | 67.008 | 1.66x |

A matched warm 512x768 repeat at seed 20261010 measured 30.792 s dense and 29.051 s protected native SOL (1.06x). Smaller sequences give a smaller end-to-end benefit.

Sampled-frame caveat: the dense and protected-SOL seed-20261010 timing outputs show a ghosted close-up at the ending; they are not accepted tutorial deliveries. The tau-1.3 sample keeps its framing. The defect also occurs with dense attention; its cause is not established. Speed measurements do not certify production quality.

An earlier matched-seed 832x1248 run took 115.951 s dense, 207.108 s with the retired integration and 80.671 s with native SOL. The old wrapper spent too much time in extra projection/prefix work. Its kernel was not necessarily mathematically broken: its route-all check passed. The replacement removes that overhead and delegates eligibility, partial RoPE, projections and conditioning behavior to upstream.

The new complete projected-attention benchmark measured 168.94 ms sparse vs 345.73 ms dense (2.05x) on the real model. Sampled relative L2 was 0.05567 for approximate SOL. Synthetic route-all tests at 1,025, 4,097 and 12,295 tokens, including a ragged conditioning prefix and 96-dimensional partial RoPE, were finite with relative L2 0.0092-0.0126. Synthetic approximate-SOL errors are not avatar quality scores.

There is no general 4x attention-only or end-to-end claim. The linked Veda/Sol-H3 comparison used different models/adapters, schedules, dimensions and an unpublished integration branch. SLA comparisons also change the distilled LoRA, so they are separate recipes rather than an isolated SOL comparison.

## Verification and limits

CPU coverage verifies native dispatch, disabled-path identity, step gates, patch composition, failure fallback and historical widget ordering. The existing JS serialization suite passes. Native GPU tests and generated videos cover the shared backend; no core files or accepted tutorial preset defaults were changed. The long-avatar comparison uses a separate SECoursesAudioTools controller and preserves its own receipts.

The local detailed report and comparison media are under `G:/ComfyUI_V92/Reports/SOL_H3_20261008/`; reproducibility records are under `G:/ComfyUI_V92/dont_upload_dont_depend_on_test_folder_use_here_for_local_tests_development_temporary/SOL_H3_Audit_20261008/`. The [September measurements](sol_attention_20260913_historical.md) describe the retired wrapper and are retained only as historical evidence.
