"""Regressions from the Wave 9E reviews, and chat-template calibration."""
from __future__ import annotations

import datetime
import json

import pytest
import torch

from hf_int4_g16_xpu import calibration as calibration_module
from hf_int4_g16_xpu import native_l0
from hf_int4_g16_xpu.calibration import load_corpus_calibration, load_mixed_train_calibration
from hf_int4_g16_xpu.converter import convert, main
from hf_int4_g16_xpu.moe import moe_grouped_gemm, moe_pair_gemm

from test_hf_int4_g16_had64 import _x
from test_hf_int4_g16_moe import (
    E,
    H,
    I,
    _bits,
    _build_moe,
    _gen,
    _random_stack,
    _to_device,
    vllm_distributed,  # noqa: F401  (fixture)
    vllm_env,  # noqa: F401  (fixture)
)
from test_hf_int4_g16_wave9d import NS1, _banks, _checkpoint, two_layer  # noqa: F401  (fixture)

pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="requires XPU")
DEV = "xpu"


# ===========================================================================
# 1. The grouped tile operator only accepts the descriptor of its own batch
# ===========================================================================
def _grouped_case(experts=24, k=192, n=80, slots=8, rows=(16, 17)):
    raw, _ = _random_stack(experts, n, k, seed=31, bias=True)
    stack = _to_device(raw, k)
    batches = []
    for m in rows:
        x = _x(m, k, m).to(DEV)
        ids = torch.randint(0, experts, (m * slots,), generator=_gen(m), dtype=torch.int32).to(DEV)
        batches.append((x, ids))
    return stack, slots, batches


def test_mismatched_bucket_descriptor_is_rejected_without_a_host_sync():
    """M=16 and M=17 need the same tile capacity, so no buffer size tells their
    buckets apart.  The rejections run while a graph records, where Level Zero
    refuses a host sync (UR_RESULT_ERROR_UNSUPPORTED_FEATURE), so a check that
    synced would raise that instead of the expected message.  Do not add a sync
    here as a control: the refused call leaves the runtime broken and the
    process segfaults about half the time."""
    stack, slots, ((x_small, ids_small), (x_big, ids_big)) = _grouped_case()
    small = native_l0.moe_bucket_native(ids_small, stack.num_experts)
    big = native_l0.moe_bucket_native(ids_big, stack.num_experts)
    capacity = {native_l0.moe_bucket_snapshot(b)[1].shape[0] for b in (small, big)}
    assert capacity == {33}, "the two batches must be indistinguishable by tile capacity"
    fewer, _ = _random_stack(stack.num_experts - 1, stack.n, stack.k, seed=32, bias=True)
    fewer = _to_device(fewer, stack.k)
    want = moe_pair_gemm(x_small, ids_small, stack, slots, torch.float32, "native")
    for _ in range(2):
        moe_grouped_gemm(x_small, small, stack, slots, torch.float32)
    torch.xpu.synchronize()

    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        captured = moe_grouped_gemm(x_small, small, stack, slots, torch.float32)
        # Smaller descriptor first: this direction stays in bounds even if the check is lost.
        with pytest.raises(RuntimeError, match="built for 128 pairs, this batch has rows 17"):
            moe_grouped_gemm(x_big, small, stack, slots, torch.float32)
        with pytest.raises(RuntimeError, match="built for 136 pairs, this batch has rows 16"):
            moe_grouped_gemm(x_small, big, stack, slots, torch.float32)
        with pytest.raises(RuntimeError, match="built for E=24 experts, the expert table has E=23"):
            moe_grouped_gemm(x_small, small, fewer, slots, torch.float32)
    graph.replay()
    torch.xpu.synchronize()
    assert torch.equal(_bits(captured), _bits(want))


def test_bucket_buffers_cannot_be_supplied_or_replaced():
    stack, slots, ((x, ids), (_, other_ids)) = _grouped_case()
    buckets = native_l0.moe_bucket_native(ids, stack.num_experts)
    other = native_l0.moe_bucket_native(other_ids, stack.num_experts)
    rotated = native_l0.moe_rotate_native(x, stack.signs)
    tile = lambda b: native_l0.moe_tile_native(
        rotated, stack.q_pack, stack.u_pack, stack.d_row, stack.bias_corr, b, slots, True)
    mixed = (native_l0.moe_bucket_snapshot(buckets)[0], *native_l0.moe_bucket_snapshot(other)[1:])
    with pytest.raises(TypeError):
        tile(mixed)
    with pytest.raises(RuntimeError, match="must be the GroupedBucketDescriptor"):
        tile(datetime.datetime_CAPI)  # a capsule from somewhere else
    with pytest.raises((TypeError, AttributeError)):
        buckets[0] = mixed[0]
    with pytest.raises(AttributeError):
        buckets.perm = mixed[0]
    before = tile(buckets)
    for copy in native_l0.moe_bucket_snapshot(buckets):
        copy.fill_(1 << 30)
    assert torch.equal(_bits(tile(buckets)), _bits(before))


# ===========================================================================
# 2. GraniteMoE's fused member names are scoped to its raw namespace
# ===========================================================================
def _granite_named(prefix, e, h, i):
    return {f"{prefix}.input_linear.weight": ((e, 2 * i, h), "BF16"),
            f"{prefix}.output_linear.weight": ((e, h, i), "BF16")}


@pytest.mark.parametrize("namespace", ["model.layers.0.mlp.experts", "model.layers.0.block_sparse_moe.experts"])
def test_granite_member_names_under_an_experts_namespace_are_rejected(namespace):
    pytest.importorskip("vllm")
    _banks(_granite_named("model.layers.0.block_sparse_moe", 3, 256, 192)).require(
        "model.layers.0.block_sparse_moe", 3, 256, 192, True)  # control: raw GraniteMoE
    with pytest.raises(ValueError, match="unrecognized expert layout"):
        _banks(_granite_named(namespace, 3, 256, 192)).require(namespace, 3, 256, 192, True)


def test_qwen_layer_rejects_a_granite_named_bf16_bank(tmp_path, vllm_env, two_layer):
    tensors, manifest, admit = _checkpoint(tmp_path, two_layer)
    for key in [k for k in tensors if k.startswith(NS1 + ".")]:
        del tensors[key]
    del manifest["moe_modules"]["layers.1.mlp.experts"]
    for name, (shape, _) in _granite_named(NS1, E, H, I).items():
        tensors[name] = torch.zeros(shape, dtype=torch.bfloat16)
    config = admit()
    with pytest.raises(ValueError, match="unrecognized expert layout"):
        _build_moe(config, NS1)


def test_literal_block_sparse_moe_prefix_member_cannot_spoof_granite_bank():
    pytest.importorskip("vllm")
    _banks(_granite_named("model.layers.0.block_sparse_moe", 3, 256, 192)).require(
        "model.layers.0.block_sparse_moe", 3, 256, 192, True)  # positive control: genuine raw Granite
    spoofed = {
        "model.layers.0.mlp.experts.block_sparse_moe:input_linear.weight": ((3, 384, 256), "BF16"),
        "model.layers.0.mlp.experts.block_sparse_moe:output_linear.weight": ((3, 256, 192), "BF16"),
    }
    with pytest.raises(ValueError, match="unrecognized expert layout"):
        _banks(spoofed).require("model.layers.0.mlp.experts", 3, 256, 192, True)


# ===========================================================================
# 3. The grouped helper never coerces an output dtype
# ===========================================================================
@pytest.mark.parametrize("out_dtype", [torch.float16, torch.int32, torch.float64, torch.int8, None])
def test_grouped_gemm_rejects_other_output_dtypes_before_rotation(monkeypatch, out_dtype):
    stack, slots, ((x, ids), _) = _grouped_case()
    buckets = native_l0.moe_bucket_native(ids, stack.num_experts)
    for name in ("moe_rotate_native", "moe_tile_native"):
        monkeypatch.setattr(native_l0, name, lambda *a, _n=name: pytest.fail(f"{_n} ran"))
    with pytest.raises(ValueError, match="out_dtype must be torch.bfloat16 or torch.float32"):
        moe_grouped_gemm(x, buckets, stack, slots, out_dtype)


# ===========================================================================
# 4. --cal-chat-template
# ===========================================================================
_TEMPLATE = ("{{ bos_token }}{% for m in messages %} <turn> {{ m['role'] }} {{ m['content'] }} <end>{% endfor %}"
             "{% if add_generation_prompt %} <turn> assistant{% endif %}")
_WORDS = "alpha beta gamma delta epsilon zeta eta theta".split()


def _chat_tokenizer(out, template=_TEMPLATE):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {w: i for i, w in enumerate(["<unk>", "<pad>", "user", "assistant", *_WORDS])}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>", bos_token="<bos>",
                                   additional_special_tokens=["<turn>", "<end>"])
    fast.chat_template = template
    out.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(out)
    return fast


def _corpus(path, words=200):
    path.write_text(" ".join(_WORDS[i % len(_WORDS)] for i in range(words)))
    return path


def test_cal_chat_template_wraps_the_corpus_block(tmp_path):
    tok = _chat_tokenizer(tmp_path / "tok")
    corpus = _corpus(tmp_path / "corpus.txt")
    raw = load_corpus_calibration(tok, corpus, max_tokens=24)
    chat = load_corpus_calibration(tok, corpus, max_tokens=24, chat_template=True)
    ids = tok.convert_tokens_to_ids
    prefix = [ids("<bos>"), ids("<turn>"), ids("user")]
    suffix = [ids("<end>"), ids("<turn>"), ids("assistant")]
    assert chat.rows == [prefix + raw.rows[0] + suffix]
    assert chat.rows[0].count(tok.bos_token_id) == 1 and tok.bos_token_id not in raw.rows[0]
    assert (raw.info["cal_chat_template"], chat.info["cal_chat_template"]) == (False, True)
    assert chat.info["cal_content_tokens"] == [24] and chat.info["cal_row_tokens"] == 30
    assert chat.token_sha256 != raw.token_sha256


class _UnevenTokenizer:
    """Whitespace tokenizer with a growing vocabulary.  Text holding the word
    ``uneven`` re-encodes one token longer, as a real tokenizer can at a seam."""

    chat_template = "test"
    bos_token_id = None

    def __init__(self):
        self.words = []
        self.index = {}

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens, "the chat template owns the special tokens"
        words = text.split() + (["<seam>"] if "<bos>" in text and "uneven" in text else [])
        for word in words:
            if word not in self.index:
                self.index[word] = len(self.words)
                self.words.append(word)
        return [self.index[word] for word in words]

    def decode(self, ids, clean_up_tokenization_spaces=False):
        return " ".join(self.words[i] for i in ids)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        assert not tokenize and add_generation_prompt and [m["role"] for m in messages] == ["user"]
        return f"<bos> <user> {messages[0]['content']} <end> <assistant>"


def test_cal_chat_template_gives_mixed_train_rows_one_length(monkeypatch):
    datasets = pytest.importorskip("datasets")
    splits = {"train": {"text": [" ".join(["uneven", *(f"train{i}" for i in range(400))])]},
              "test": {"text": [" ".join(f"test{i}" for i in range(400))]}}
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: splits)
    tok = _UnevenTokenizer()
    raw = load_mixed_train_calibration(tok, tokens_per_split=64)
    chat = load_mixed_train_calibration(tok, tokens_per_split=64, chat_template=True)
    assert [len(r) for r in raw.rows] == [64, 64]
    # The prose row renders one token longer, so it gives up one content token.
    assert chat.info["cal_content_tokens"] == [63, 64] and [len(r) for r in chat.rows] == [68, 68]
    for row, source, content in zip(chat.rows, raw.rows, chat.info["cal_content_tokens"]):
        assert tok.decode(row[:2]) == "<bos> <user>" and row[2:2 + content] == source[:content]
        assert "<end> <assistant>" in tok.decode(row), "the generation prompt is never trimmed"
    assert chat.info["wt2_test_head_6gram_overlap"] == chat.info["difflib_py_6gram_overlap"] == 0

    monkeypatch.setattr(calibration_module, "_CHAT_MAX_TRIM", 0)
    with pytest.raises(ValueError, match="cannot give all calibration rows one length"):
        load_mixed_train_calibration(tok, tokens_per_split=64, chat_template=True)


@pytest.mark.parametrize("template", [None, ""])
def test_cal_chat_template_fails_closed_without_a_chat_template(tmp_path, template):
    from test_converter_streaming import _make_tiny_llama

    src = tmp_path / "src"
    corpus = _make_tiny_llama(src)  # its tokenizer has no chat_template
    tok = _chat_tokenizer(tmp_path / "tok", template=template)
    with pytest.raises(ValueError, match="needs a tokenizer with a chat_template"):
        load_corpus_calibration(tok, corpus, max_tokens=24, chat_template=True)
    with pytest.raises(ValueError, match="needs a tokenizer with a chat_template"):
        convert(src, tmp_path / "out", device="xpu:0", biascorr_corpus=str(corpus), biascorr_tokens=64,
                cal_chat_template=True)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["src", "tok"], "nothing may be written"


def test_cal_chat_template_without_calibration_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="does nothing without BiasCorr calibration"):
        convert(tmp_path / "src", tmp_path / "out", cal_chat_template=True)


def test_cli_records_cal_chat_template_in_config_and_report(tmp_path):
    from safetensors.torch import load_file

    from test_converter_streaming import _make_tiny_llama

    src = tmp_path / "src"
    corpus = _make_tiny_llama(src)
    _chat_tokenizer(src)
    corpus.write_text(" ".join(_WORDS[i % len(_WORDS)] for i in range(200)))
    results = {}
    for flag in (False, True):
        out = tmp_path / f"out-{flag}"
        main([str(src), str(out), "--device", "xpu:0", "--biascorr-corpus", str(corpus), "--biascorr-tokens", "64",
              *(["--cal-chat-template"] if flag else [])])
        report = json.loads((out / "hf_int4_g16_conversion_report.json").read_text())
        recorded = json.loads((out / "config.json").read_text())["quantization_config"]["calibration"]
        assert report["cal_chat_template"] is flag and recorded["cal_chat_template"] is flag
        assert recorded["streaming"]["pos0_excluded"] is True
        assert recorded["streaming"]["batch_shape"] == [1, 70 if flag else 64]
        corrections = {k: v for shard in out.glob("*.safetensors") for k, v in load_file(str(shard)).items()
                       if k.endswith(".bias_corr")}
        results[flag] = (recorded["cal_token_sha256"], corrections)
    assert results[False][0] != results[True][0]
    assert results[False][1].keys() == results[True][1].keys()
    assert any(not torch.equal(results[False][1][k], results[True][1][k]) for k in results[True][1])

    out = tmp_path / "out-uncalibrated"
    report = convert(src, out, device="xpu:0")
    assert report["cal_chat_template"] is False
