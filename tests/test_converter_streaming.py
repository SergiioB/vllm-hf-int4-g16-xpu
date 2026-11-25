"""End-to-end streaming-converter tests on a tiny Llama checkpoint."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

pytest.importorskip("transformers")
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast, Qwen3_5ForCausalLM, Qwen3_5TextConfig

from hf_int4_g16_xpu.calibration import CalibrationData, stream_calibration_means
from hf_int4_g16_xpu.converter import convert, dry_run_layer_mapping, load_index


def _tiny_tokenizer(out: Path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    words = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
             "nu xi omicron pi rho sigma tau upsilon phi chi psi omega calibration streaming").split()
    vocab = {"<unk>": 0, "<pad>": 1}
    vocab.update({w: i + 2 for i, w in enumerate(words)})
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>")
    fast.save_pretrained(out)


def _make_tiny_llama(root: Path):
    cfg = LlamaConfig(
        vocab_size=64, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=512, tie_word_embeddings=False,
    )
    torch.manual_seed(123)
    model = LlamaForCausalLM(cfg).to(torch.bfloat16).eval()
    model.save_pretrained(root, safe_serialization=True)
    _tiny_tokenizer(root)
    corpus = root / "corpus.txt"
    corpus.write_text(" ".join((("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
                                 "nu xi omicron pi rho sigma tau upsilon phi chi psi omega ") * 24).split()))
    return corpus


def test_streaming_biascorr_conversion_tiny_llama(tmp_path):
    src = tmp_path / "tiny-src"
    out = tmp_path / "tiny-out"
    src.mkdir()
    corpus = _make_tiny_llama(src)
    report = convert(
        src, out, device="xpu:0" if torch.xpu.is_available() else "cpu",
        biascorr_corpus=str(corpus), biascorr_tokens=256, max_shard_bytes=512 * 1024**2,
    )
    assert report["stats"]["quantized_linears"] == 14
    assert report["stats"]["biascorr_linears"] == 14
    assert report["calibration"]["streaming"]["linears_seen"] == 14
    assert report["calibration"]["streaming"]["pos0_excluded"] is True
    assert report["calibration"]["cal_token_sha256"]

    qc = json.loads((out / "quantization_config.json").read_text())
    assert qc["calibration"]["mode"] == "corpus_file"
    assert qc["calibration"]["streaming"]["linears_seen"] == 14
    assert qc["calibration"]["streaming"]["attn_implementation"] == "sdpa"
    assert qc["cal_token_sha256"] == report["calibration"]["cal_token_sha256"]
    assert qc["format_version"] == "hf_int4_g16/1"
    assert qc["layout"] == "xe2_dpas_native_v1"
    # Every global K used by this tiny model has a raw int8 sign digest.  This
    # is part of the new conversion contract even though legacy checkpoints may
    # omit it for backwards compatibility.
    from hf_int4_g16_xpu.format import sign_sha256_for_k
    assert qc["sign_sha256_by_k"] == {
        "128": sign_sha256_for_k(128),
        "256": sign_sha256_for_k(256),
    }
    assert not (out.parent / (out.name + ".hf-int4-stage")).exists()

    idx = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    bc_keys = sorted(k for k in idx if k.endswith(".bias_corr"))
    q_keys = sorted(k for k in idx if k.endswith(".q_pack"))
    assert len(bc_keys) == len(q_keys) == 14
    shard = next(iter(idx.values()))
    with safe_open(str(out / shard), framework="pt", device="cpu") as f:
        sample = f.get_tensor("model.layers.0.mlp.gate_proj.bias_corr")
        assert sample.dtype == torch.float32
        assert sample.shape == (256,)
        assert torch.isfinite(sample).all()
        # Quantization is lossy but coherent; a nonzero correction must survive.
        assert sample.abs().max() > 0


def test_dry_run_mapping_tiny_llama(tmp_path):
    src = tmp_path / "tiny-src"
    src.mkdir()
    _make_tiny_llama(src)
    rep = dry_run_layer_mapping(src)
    assert rep["decoder_linear_candidates"] == 14
    assert rep["layers_covered"] == 2
    assert rep["excluded"]["k_not_multiple_128"] == 0
    assert not any("lm_head.weight" == k for k in rep["keys"])



def _make_tiny_qwen35_conditional(root: Path):
    cfg = Qwen3_5TextConfig(
        vocab_size=64, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, linear_key_head_dim=32, linear_num_key_heads=2,
        linear_value_head_dim=32, linear_num_value_heads=4,
        linear_conv_kernel_dim=3, full_attention_interval=2,
        layer_types=["linear_attention", "full_attention"],
        max_position_embeddings=512,
    )
    torch.manual_seed(321)
    model = Qwen3_5ForCausalLM(cfg).to(torch.bfloat16).eval()
    model.save_pretrained(root, safe_serialization=True)
    # Make the on-disk config look like the production conditional wrapper.  The
    # streaming mapper deliberately resolves real checkpoint key namespaces from
    # the safetensors index rather than assuming one prefix.
    raw = json.loads((root / "config.json").read_text())
    wrapped = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": raw,
        "tie_word_embeddings": False,
    }
    (root / "config.json").write_text(json.dumps(wrapped, indent=2))
    _tiny_tokenizer(root)
    corpus = root / "corpus.txt"
    corpus.write_text(" ".join((("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
                                 "nu xi omicron pi rho sigma tau upsilon phi chi psi omega ") * 24).split()))
    return corpus


def test_streaming_biascorr_conversion_tiny_qwen35_conditional(tmp_path):
    src = tmp_path / "qwen-src"
    out = tmp_path / "qwen-out"
    src.mkdir()
    corpus = _make_tiny_qwen35_conditional(src)
    report = convert(
        src, out, device="xpu:0" if torch.xpu.is_available() else "cpu",
        biascorr_corpus=str(corpus), biascorr_tokens=256, max_shard_bytes=512 * 1024**2,
    )
    # Layer 0 linear attention has in_proj_{qkv,z,b,a}+out+mlp 3 = 8; layer 1
    # full attention has q/k/v/o+mlp 3 = 7, total 15 decoder linears.
    assert report["stats"]["quantized_linears"] == 15
    assert report["stats"]["biascorr_linears"] == 15
    assert report["calibration"]["streaming"]["linears_seen"] == 15
    qc = json.loads((out / "quantization_config.json").read_text())
    assert qc["calibration"]["streaming"]["peak_device_gib"] < 10.0
    assert qc["calibration"]["streaming"]["peak_rss_gib"] < 15.0


def test_streaming_calibration_is_causal_bit_identical_prefix(tmp_path):
    src = tmp_path / "tiny-src"
    src.mkdir()
    _make_tiny_llama(src)
    wmap, _ = load_index(src)
    row = [(i % 60) + 2 for i in range(64)]
    cal = CalibrationData([row], "causality-fixture", {"mode": "unit"})
    _, meta, hidden = stream_calibration_means(src, wmap, cal, "cpu", return_hidden=True)
    mutated = list(row)
    # Mutate only future tokens.  A causal SDPA mask makes every prefix hidden
    # state bitwise unchanged; a full/unmasked contract would leak these tokens
    # into earlier positions.
    for i in range(32, 64):
        mutated[i] = 2 + ((mutated[i] + 17) % 60)
    cal2 = CalibrationData([mutated], "causality-fixture-mutated", {"mode": "unit"})
    _, meta2, hidden2 = stream_calibration_means(src, wmap, cal2, "cpu", return_hidden=True)
    assert meta["attn_implementation"] == "sdpa"
    assert meta2["attn_implementation"] == "sdpa"
    assert torch.equal(hidden[:, :32], hidden2[:, :32])
    assert not torch.equal(hidden[:, 32:], hidden2[:, 32:])


def test_streaming_calibration_rejects_unknown_layer_type(tmp_path):
    src = tmp_path / "tiny-src"
    src.mkdir()
    _make_tiny_llama(src)
    cfg_path = src / "config.json"
    raw = json.loads(cfg_path.read_text())
    raw["layer_types"] = ["mystery_attention", "full_attention"]
    cfg_path.write_text(json.dumps(raw))
    wmap, _ = load_index(src)
    row = [(i % 60) + 2 for i in range(64)]
    cal = CalibrationData([row], "unknown-layer-fixture", {"mode": "unit"})
    with pytest.raises(ValueError, match="unsupported layer_types"):
        stream_calibration_means(src, wmap, cal, "cpu")
