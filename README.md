# hf-int4-g16-xpu

An out-of-tree vLLM quantization plugin for `HF-INT4-G16` (`Had128NoPerm`) weights on Intel Arc Pro B70 (`bmg_g31`, Xe2). It includes the BF16 converter, Triton and native Level Zero kernels, and the tests. It is a community plugin and is not part of upstream vLLM.

## Requirements

- `vllm/vllm-openai-xpu:v0.31.0` (`torch 2.14.0+xpu`, `triton 3.8.0+xpu`)
- One Intel Arc Pro B70. Other GPUs have not been tested.
- Dense or MoE checkpoints. MoE support is single-GPU only; see [MoE](#moe-routed-experts) for what has been tested.

## Install

```bash
pip install --no-deps --no-build-isolation -e .
```

This registers the `vllm.general_plugins` entry point, so the API server and the engine workers both load the quantization method `hf_int4_g16`. On first use, the native extension compiles into `hf_int4_g16_xpu/.torch_ext_build/`.

## Convert a BF16 Checkpoint

The input and output folders are positional arguments.

```bash
python3 convert_hf_int4_g16.py \
  /path/to/Qwen3.8-27B-bf16 \
  /path/to/Qwen3.8-27B-HF-INT4-G16-BC \
  --biascorr --cal-mode mixed_train --biascorr-tokens 1024 \
  --max-shard-gib 3.0
```

`--cal-mode mixed_train` needs the `datasets` package and the `Salesforce/wikitext` dataset. On one B70, converting `Qwen3.8-27B` took 691.7 s with 10.9 GiB peak host RSS and 2.38 GiB peak calibration VRAM.

For MoE checkpoints the converter quantizes the routed experts too. It reads three source layouts: per expert (`experts.{e}.{gate,up,down}_proj.weight`), fused 3D (`experts.gate_up_proj`, `experts.down_proj`, with or without `.weight`), and GraniteMoE `block_sparse_moe.{input,output}_linear.weight`. It stops on an expert tensor it does not recognize, on a missing expert, and when the expert count differs from `config.json`. Routers stay in the source dtype. To keep the experts in BF16, append `experts` to `--skip-patterns`. The opt-out covers a whole expert bank: a pattern that matches any of the bank's keys, or its `.experts` namespace (also for GraniteMoE's `block_sparse_moe` names), keeps every tensor of that bank unchanged. Opted-out banks do not need widths on the Hadamard grid.

| `--biascorr-scope` | Corrections written |
|---|---|
| `none` (default without `--biascorr`) | None |
| `shared_only` (default with `--biascorr`) | Every dense linear, including shared MLPs. No expert. |
| `all_experts` | Dense linears and every routed expert. An expert that received no calibration token gets zeros. |

**Warning for instruct (`-it`) models:** the default raw-text BiasCorr calibration failed our out-of-domain chat KL gate on `google/gemma-4-26B-A4B-it` (`+0.044` OOD KL, CI crossing zero), even though it passed WT2 and Code. Gate instruct checkpoints on chat-format evaluation before shipping BiasCorr.

`--cal-chat-template` (off by default, needs BiasCorr) renders each calibration text block as one user turn: `tokenizer.apply_chat_template([{"role": "user", "content": block}], tokenize=False, add_generation_prompt=True)`. The template writes its own special tokens. Position 0 (the BOS, where the template has one) is excluded from the means as always. The other template tokens are included. With `mixed_train` the two rows must have one length, so the converter may trim up to 64 content tokens from the end of a block. It fails if the tokenizer has no `chat_template`. `config.json` records `quantization_config.calibration.cal_chat_template`, and the conversion report records `cal_chat_template`. In direct served-kernel gating on `google/gemma-4-26B-A4B-it` (`Exp 21`), `--cal-chat-template` fixed the raw-text OOD chat failure (`-22.9%` PASS) and passed WikiText-2 (`-20.3%` PASS) and `-b` controls, but missed the Code 95% CI upper bound by `+0.00073` on the served GPU kernels (`[-0.04884, +0.00073]`, while simulated math passed `[-0.05385, -0.00723]`). Always gate any `--cal-chat-template` checkpoint on the actual served GPU kernels before publishing.

Streaming calibration covers `full_attention`, `linear_attention` and `sliding_attention` layers, including Gemma-4 text. It does not cover Mamba layers or per-layer embeddings, so GraniteMoeHybrid converts with scope `none` only.

## Block Size and Format Version

The Hadamard block follows from each projection's input width `K`: 128 when `K % 128 == 0`, otherwise 64 when `K % 64 == 0`. Every block of a Had64 projection is 64 wide. Other widths cannot be represented: dense linears stay in BF16, and the converter refuses such experts.

| Format | Contents | Loads in |
|---|---|---|
| `hf_int4_g16/1` | Dense, Had128 only. Same bytes as 0.1.0 writes. | 0.1.0, 0.2.0 |
| `hf_int4_g16/2` | Any Had64 projection or packed experts. | 0.2.0 |

A `/2` manifest must carry exactly `"had_block_rule": "per projection: 128 if K % 128 == 0 else 64"` and no top-level `hadamard_block`. Per-module entries carry `had_block` only. Integer fields must be JSON integers: `64.0` and `true` are rejected. Another block schedule, such as Had128 blocks with a Had64 tail, needs a new format version.

Expert tensors are `<ns>.{e}.{gate_proj,up_proj,down_proj}.{qweight,scales,scale_super,bias_corr}`. `qweight`, `scales` and `scale_super` hold the same data as `q_pack`, `u_pack` and `d_row`. The names follow vLLM's `RoutedExperts` parameter mapping. `scale_super` is not called `super_scales` because vLLM 0.31's GraniteMoeHybrid loader only remaps expert names from a fixed suffix list.

## Gemma-4 and GraniteMoeHybrid Dense Layers

- Gemma-4 with `attention_k_eq_v` ships no `v_proj` in its full-attention layers. vLLM delivers `k_proj` twice, for the K and V partitions, and the plugin loads both from the one packed tensor. It does this only for layers the model config declares as `attention_k_eq_v` full attention, and there the manifest must hold exactly `q_proj` and `k_proj`. Anywhere else a missing `v_proj` is an error.
- vLLM 0.31 asks for GraniteMoeHybrid Mamba layers under the prefix `mixer`, but the checkpoint says `mamba`. The plugin applies this alias only when the model type is `granitemoehybrid`, and rejects a manifest that has both names.

## MoE Routed Experts

vLLM keeps routing and shared experts. `HFInt4G16MoEMethod` runs the routed experts:

- Routing is read on the device, never on the host, so the forward stays correct when vLLM replays an XPU graph with new routing. Batches of up to 15 tokens run one (token, slot) kernel launch per GEMM. From 16 tokens on, native BF16 batches bucket the pairs by expert on the device, rotate each activation row once, and apply each unpacked 64-wide weight panel to up to 16 token rows. Both paths give bit-identical outputs, and the choice depends only on the token count. The bucket buffers live in an opaque descriptor bound to the pair count, expert count and device, and the grouped GEMM rejects it for any other batch.
- Expert ids outside `[0, E)` contribute zero. `int64` ids are masked before they are narrowed to `int32`. Activations and router weights must be BF16, FP16 or FP32.
- The gate/up output is rounded to BF16. The activation (`silu`, `gelu` or `gelu_tanh`) is computed in FP32. Each expert output stays FP32 until the weighted sum, which is cast to BF16 once.
- Experts always run W4A16. `HF_INT4_G16_MODE=w4a8` is an error for a MoE model. `auto` applies W4A8 to dense layers only.
- vLLM does not track every missing weight for quantized models, so the plugin checks coverage itself. An expert layer with no `moe_modules` entry must have a complete full-precision bank in the checkpoint. GraniteMoE's `input_linear.weight` and `output_linear.weight` count only under their raw `block_sparse_moe` names, never under an `.experts` namespace. A packed bank may hold nothing but packed tensors, including tensors that only a shard header lists. Before the first forward, every packed dense and expert module in the manifest must have been bound to a layer and processed. The one exception is the K/V tensors that vLLM itself drops for Gemma-4 KV-shared layers.

The method raises an error for tensor, data, pipeline-context or expert parallelism above 1, EPLB, expert biases, non-gated activations (NemotronH), SwiGLU clamp variants and `apply_router_weight_on_input`.

## Serve

```bash
HF_INT4_G16_MODE=auto \
vllm serve /path/to/Qwen3.8-27B-HF-INT4-G16-BC \
  --served-model-name default \
  --host 127.0.0.1 --port 8091 \
  --max-model-len 36864 --max-num-seqs 96 \
  --gpu-memory-utilization 0.90 --trust-remote-code
```

| Variable | Values | Effect |
|---|---|---|
| `HF_INT4_G16_MODE` | `w4a16` (default), `auto`, `w4a8` | `auto` uses native W4A16 for batch sizes 1 to 4 and native INT8 DPAS W4A8 from 5 up. W4A8 rounds activations to INT8. |
| `HF_INT4_G16_BACKEND` | `auto` (default), `native`, `triton` | `triton` runs the reference kernels. |
| `HF_INT4_G16_BIASCORR_MODE` | `default`, `off`, `neg_zero` | Load the stored `.bias_corr` tensors, skip them, or load `-0.0f` for a bit-identity check. |

For hybrid attention and linear-attention models on a 32 GiB card, set `--max-num-seqs 96` or lower.

## Test

```bash
python3 -m pytest tests/ -q
```

The GPU tests skip without an XPU, and the plugin tests need vLLM. The full release verification chain is 699 logged GPU passes plus 1 added Granite prefix-collision regression test (17/17 targeted rerun = 700 tests) on one B70 inside the image above. `tests/conftest.py` caps the test process at 0.55 of VRAM.

`bench_moe.py` measures one routed-expert layer, and `bench_moe_grouped.py` compares the pair path, the grouped path and vLLM's BF16 experts. `e2e_vllm_moe.py` converts tiny random MoE models, runs them in a full `vllm.LLM` engine with XPU graphs and with `--enforce-eager`, and scores both against an FP32 transformers model holding the dequantized weights. It needs the package installed.

To rebuild the shipped kernel binary, run `hf_int4_g16_xpu/native/build_kernels.sh` inside the same image.

## Measured on Qwen3.8-27B (2026-11-25)

Live `vllm serve` against the shipped `GPTQ-Int4-sym-G128` checkpoint with `HF_INT4_G16_MODE=auto`. Both arms used identical flags (`--max-model-len 36864 --max-num-seqs 96 --gpu-memory-utilization 0.90 --no-enable-prefix-caching`), one B70 at a 230 W cap, XPU graphs on, interleaved ABABAB, `n=6`, thinking off, `T=0.7`, `top_p=0.8`, `top_k=20`, 256 output tokens.

| Measurement | This plugin | GPTQ in stock vLLM |
|---|---:|---:|
| 8K prefill | 2,364 tok/s | 1,849 tok/s |
| 32K prefill | 2,054 tok/s | 1,634 tok/s |
| Decode, one request, 0K / 8K / 32K | 29.48 / 28.38 / 26.26 tok/s | 33.42 / 31.95 / 29.20 tok/s |
| Total decode, 4 requests (`n=3`) | 69.06 tok/s | 97.93 tok/s |

Prefill is faster and single-user decode is slower (about 10 to 12% vs shipped GPTQ on `Qwen3.8-27B`). On quiet-host `FP8` KV cold turns from `8K` to `128K` (`Exp 22`, 256 output tokens), faster prefill wins full-turn wall clock and energy at `32K` (`+2.25 s`, `-7.0%` J), `64K` (`+4.51 s`, `-6.7%` J), and `128K` (`+5.36 s`, `-3.0%` J) up to break-even reply lengths (`~791` to `~1,455` tokens), while warm prefix-cached multi-turn workloads still favor decode speed. The full V11 method, quality measurements, and 49 sanitized receipts are in the [quantization study V11](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/master/docs/qwen38-27b/quantization-study-v11-202611/README.md) (V10 remains immutable under `docs/qwen38-27b/quantization-study-202611/`).

## Limits

- MoE prefill is still slower than vLLM's BF16 experts at large batches. One layer through vLLM's MoE runner at 512 tokens, random weights, quiet-host Gate 5 (`432` cooled samples `<= 55 C`, `230 W / 2,800 MHz`):

  | Expert shape | Pair path | Grouped (default) | vLLM BF16 |
  |---|---:|---:|---:|
  | Gemma-4-26B eager (128 experts, top 8) | 24.45 ms | 7.17 ms | 3.97 ms |
  | Gemma-4-26B graph (128 experts, top 8) | 24.19 ms | 6.88 ms | 3.73 ms |
  | Qwen3.6-35B eager (256 experts, top 8) | 15.05 ms | 4.59 ms | 3.36 ms |

  At 64 tokens the grouped runner took 1.61 ms and 1.45 ms, against 2.91 ms and 2.70 ms for BF16. At 16 tokens grouping does not help in eager mode, because there are about as many experts as routed pairs: the routed experts took 1.01 ms grouped against 1.02 ms paired (Gemma-4 shape), and 0.85 ms against 0.74 ms (Qwen shape, grouped is slower; its runner call took 0.94 ms against 0.83 ms). Under XPU graph replay at 16 tokens it was 0.70 ms against 0.83 ms, and 0.53 ms against 0.55 ms.
- Real MoE serving scope: `google/gemma-4-26B-A4B-it` (`14.64 GiB` on disk, `hf_int4_g16/2`, full multimodal `Gemma4ForConditionalGeneration`) was converted deterministically and served live in `vllm serve` with XPU graphs enabled on a verified quiet host (`Exp 21`, `host_quiet: true`, `230 W / 2,800 MHz`, `52 to 55 C`, `n=3`, model-card sampling: `84.76 tok/s` at 51 prompt tokens, `70.80 tok/s` at 1,988 tokens, `55.60 tok/s` at 8,133 tokens; `8/8` text and code canaries, live vision pass, live vs offline WikiText-2 PPL `25.983` vs `25.982`). Neither the raw-text nor the default `--cal-chat-template` Gemma sidecar is published (raw-text failed OOD chat; default `--cal-chat-template` passed WT2 `-20.3%` and OOD chat `-22.9%` but missed the served-kernel Code 95% CI upper bound by `+0.00073`). `Qwen3_5MoeForConditionalGeneration` and other MoE families have been exercised in `vllm.LLM` only on tiny random weights so far.
- NemotronH experts do not load: they are non-gated, and the MoE method supports gated activations only. Had64 removes the `K=1856` width limit, not this one.
- Had64 rotates every block of a projection at width 64. On `google/gemma-4-26B-A4B-it` (`K=704` and `K=2112`, `E=128`), uniform Had64 with chat-calibrated `shared_only` BiasCorr passed both the 3-check and 4-check OOD chat KL gates (`0.138 -> 0.097` WT2, `0.193 -> 0.138` Code, `1.167 -> 0.982` OOD), was not worse than mixed `128+64` (`Exp 17B`), and stacked additively with BiasCorr without cancelling (`Had64 -0.022`, `BiasCorr -0.058`, both `-0.070`, interaction CI `[-0.008, +0.027]`). Uniform Had64 has not yet been KL-gated on other `K % 128 == 64` MoE architectures (`Qwen3.6-35B-A3B` has every quantized `K` divisible by `128`).
- With tensor parallelism above 1, row-parallel layers use W4A16 and add `bias_corr` on rank 0 only. Live two-GPU vLLM collectives have not been benchmarked.
- The native kernels use Intel 2D block I/O, so they need the effective base address of `q_pack` aligned to 64 bytes, `N >= 16` and `N % 16 == 0`. The W4A8 DPAS path also needs `N >= 32`. They need `K % 64 == 0`, and dense kernels need `K <= 17408`. Under `auto`, other shapes and views run on Triton; `native` raises. W4A8 rejects `K > 17408`.
- One known layer falls off the native path under this rule: the Mamba `in_proj` of the tiny GraniteMoeHybrid smoke model is `[1064, 256]` (`1064 % 16 == 8`), so its single-token decode runs on Triton. 0.1.0's `N % 8` rule would have kept it native. Hybrid models whose `in_proj` width follows the same formula are likely affected too (not checked on a real checkpoint). Every dense layer of the Qwen3.8-27B `/1` checkpoint has `N % 16 == 0`.
- The on-disk layout `xe2_dpas_native_v1` is specific to this plugin. It is not a GPTQ or AWQ interchange format.
