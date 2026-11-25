# hf-int4-g16-xpu

An out-of-tree vLLM quantization plugin for `HF-INT4-G16` (`Had128NoPerm`) weights on Intel Arc Pro B70 (`bmg_g31`, Xe2). It includes the BF16 converter, Triton and native Level Zero kernels, and the tests. It is a community plugin and is not part of upstream vLLM.

## Requirements

- `vllm/vllm-openai-xpu:v0.31.0` (`torch 2.14.0+xpu`, `triton 3.8.0+xpu`)
- One Intel Arc Pro B70. Other GPUs have not been tested.
- A dense (non-MoE) checkpoint. Packed MoE expert keys are rejected at load time.

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
| `HF_INT4_G16_BIASCORR_MODE` | `default`, `off`, `neg_zero` | Load the stored `.bias_corr` tensors, skip them, or load `-0.0f` for a bit-identity check. |

For hybrid attention and linear-attention models on a 32 GiB card, set `--max-num-seqs 96` or lower.

## Test

```bash
python3 -m pytest tests/ -q
```

The GPU tests skip without an XPU, and the plugin tests need vLLM. The full suite is 130 tests on one B70 inside the image above.

To rebuild the shipped kernel binary, run `hf_int4_g16_xpu/native/build_kernels.sh` inside the same image.

## Measured on Qwen3.8-27B (2026-11-25)

Live `vllm serve` against the shipped `GPTQ-Int4-sym-G128` checkpoint with `HF_INT4_G16_MODE=auto`. Both arms used identical flags (`--max-model-len 36864 --max-num-seqs 96 --gpu-memory-utilization 0.90 --no-enable-prefix-caching`), one B70 at a 230 W cap, XPU graphs on, interleaved ABABAB, `n=6`, thinking off, `T=0.7`, `top_p=0.8`, `top_k=20`, 256 output tokens.

| Measurement | This plugin | GPTQ in stock vLLM |
|---|---:|---:|
| 8K prefill | 2,364 tok/s | 1,849 tok/s |
| 32K prefill | 2,054 tok/s | 1,634 tok/s |
| Decode, one request, 0K / 8K / 32K | 29.48 / 28.38 / 26.26 tok/s | 33.42 / 31.95 / 29.20 tok/s |
| Total decode, 4 requests (`n=3`) | 69.06 tok/s | 97.93 tok/s |

Prefill is faster and decode is slower. The method, the quality measurements and the receipts are in the [quantization study](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook) in the cookbook repository, under `docs/qwen38-27b/quantization-study-202611/`.

## Limits

- Dense models only.
- With tensor parallelism above 1, row-parallel layers use W4A16 and add `bias_corr` on rank 0 only. Live two-GPU vLLM collectives have not been benchmarked.
- Native kernels require `K % 128 == 0` and `K <= 17408`. Wider layers run on Triton in W4A16. W4A8 rejects them.
- The on-disk layout `xe2_dpas_native_v1` is specific to this plugin. It is not a GPTQ or AWQ interchange format.
