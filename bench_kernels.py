"""GPU0 cold-L2 kernel benchmark: HF-INT4-G16 vs vendor oneDNN/INC baselines.

Methodology: four independently allocated weight-buffer copies are rotated on
back-to-back launches to defeat L2 reuse.  Each cell reports (a) synchronized
single-call latency and (b) steady-state amortized latency from sync -> 20
launches -> sync, repeated three times, all with the same perf_counter timer in
one GPU0 process.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch

from hf_int4_g16_xpu.format import pack_native_words, signs_for_k
from hf_int4_g16_xpu.ops import hf_int4_g16_linear

SHAPES = {
    "attn_q": (12288, 5120),
    "mlp_up": (17408, 5120),
    "mlp_down": (5120, 17408),
}
MS = [1, 2, 4, 8, 16, 128, 512, 2048, 4096]
NBUF = 4
LAUNCHES = 20


def sync():
    torch.xpu.synchronize()


def read_text(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except Exception:
        return None


def gpu0_hwmon() -> str | None:
    try:
        pci = Path("/sys/class/drm/card0/device").resolve().name
        for h in Path("/sys/class/hwmon").glob("hwmon*"):
            if (h / "name").exists() and (h / "name").read_text().strip() == "xe":
                try:
                    if (h / "device").resolve().name == pci:
                        return str(h)
                except Exception:
                    pass
    except Exception:
        pass
    return None


def telemetry(energy0=None, energy1=None, elapsed=None) -> dict:
    hw = gpu0_hwmon()
    out = {}
    if hw:
        t = read_text(hw + "/temp2_input")
        cap = read_text(hw + "/power1_cap")
        if t is not None:
            out["temp_c"] = int(t) / 1000.0
        if cap is not None:
            out["power_cap_w"] = int(cap) / 1e6
        if energy0 is not None and energy1 is not None and elapsed and elapsed > 0:
            out["power_w"] = max(0.0, (energy1 - energy0) * 1e-6 / elapsed)
    act = read_text("/sys/class/drm/card0/device/tile0/gt0/freq0/act_freq")
    cur = read_text("/sys/class/drm/card0/device/tile0/gt0/freq0/cur_freq")
    if act is not None:
        out["act_mhz"] = float(act)
    if cur is not None:
        out["cur_mhz"] = float(cur)
    return out


def temp_c() -> float:
    hw = gpu0_hwmon()
    if not hw:
        return -1.0
    v = read_text(hw + "/temp2_input")
    return int(v) / 1000.0 if v is not None else -1.0


def cool_down(limit_c=55.0) -> None:
    # Host measurement protocol: every measured cell starts at or below 55 C.
    t0 = time.perf_counter()
    while temp_c() > limit_c and time.perf_counter() - t0 < 240.0:
        time.sleep(2.0)


def energy_uj() -> int | None:
    hw = gpu0_hwmon()
    if not hw:
        return None
    v = read_text(hw + "/energy1_input")
    return int(v) if v is not None else None


def pack_hf(n, k):
    g = torch.Generator().manual_seed(1234 + n + k)
    q = torch.randint(0, 16, (n, k), generator=g, dtype=torch.uint8)
    u = torch.randint(0, 16, (n, k // 16), generator=g, dtype=torch.uint8)
    return pack_native_words(q, u)


def pack_inc(n, k):
    # GPTQ sequential nibble order [K//8,N], then the physical NT layout required
    # by the stock oneDNN operators.  Symmetric zero point is the scalar value 8.
    g = torch.Generator().manual_seed(2234 + n + k)
    vals = torch.randint(0, 16, (k // 8, 8, n), generator=g, dtype=torch.int32)
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)[None, :, None]
    qw = (vals << shifts).sum(dim=1, dtype=torch.int32).contiguous()
    return qw.t().contiguous().t()


def clone4(obj):
    if isinstance(obj, dict):
        return [{k: v.clone() for k, v in obj.items()} for _ in range(NBUF)]
    return [obj.clone() for _ in range(NBUF)]


def ramp_clocks(ms=35):
    # Discarded busy loop immediately before a cell.  Fixed short ramps sometimes
    # leave the first decode cell at 400-650 MHz, so continue until the Xe
    # frequency controller reports a high clock (or a bounded fallback elapsed).
    a = torch.randn((1024, 1024), device="xpu", dtype=torch.bfloat16)
    b = torch.randn((1024, 1024), device="xpu", dtype=torch.bfloat16)
    freq = Path("/sys/class/drm/card0/device/tile0/gt0/freq0/cur_freq")
    t0 = time.perf_counter()
    while True:
        c = a @ b
        elapsed_ms = (time.perf_counter() - t0) * 1000
        try:
            cur = int(freq.read_text().strip())
        except Exception:
            cur = 0
        if (elapsed_ms >= ms and cur >= 2400) or elapsed_ms >= max(150, ms * 4):
            break
    torch.xpu.synchronize()
    del a, b, c


def measure(call, bundles, launches=LAUNCHES, cool_to_c=55.0) -> dict:
    cool_down(cool_to_c)
    ramp_clocks()
    # Warm every buffer repeatedly; this is discarded and includes compile/import
    # plus enough work for the Xe power management clock ramp.
    for _ in range(3):
        for b in bundles:
            call(b)
    sync()

    single = []
    for i in range(5):
        sync()
        t0 = time.perf_counter()
        call(bundles[i % NBUF])
        sync()
        single.append(time.perf_counter() - t0)

    loop_times = []
    powers = []
    for rep in range(3):
        sync()
        e0 = energy_uj()
        t0 = time.perf_counter()
        for i in range(launches):
            call(bundles[(i + rep) % NBUF])
        sync()
        dt = time.perf_counter() - t0
        e1 = energy_uj()
        loop_times.append(dt / launches)
        if e0 is not None and e1 is not None:
            powers.append(max(0.0, (e1 - e0) * 1e-6 / dt))
    tele = telemetry()
    if powers:
        tele["power_w"] = statistics.median(powers)
    return {
        "single_us": statistics.median(single) * 1e6,
        "steady_us": statistics.median(loop_times) * 1e6,
        "single_all_us": [round(x * 1e6, 3) for x in single],
        "steady_repeats_us": [round(x * 1e6, 3) for x in loop_times],
        **tele,
    }


def make_cell(shape_name, n, k, m, quick=False):
    dev = "xpu"
    x = torch.randn((m, k), generator=torch.Generator().manual_seed(3000 + m), dtype=torch.float32).to(dev).bfloat16()
    xq = torch.randint(-127, 128, (m, k), generator=torch.Generator().manual_seed(3100 + m), dtype=torch.int8).to(dev)
    signs = signs_for_k(k, device=dev)
    mode = "w4a16" if m in (1, 2, 3, 4) else "w4a8"

    # HF-INT4-G16 native bundles.
    qp, up = pack_hf(n, k)
    d = (torch.rand(n, generator=torch.Generator().manual_seed(3200)) * 0.001 + 0.0001)
    hf0 = {"qp": qp.to(dev), "up": up.to(dev), "d": d.to(dev)}
    hf_b = clone4(hf0)

    # BF16 oneDNN reference.
    wb = (torch.randn((n, k), generator=torch.Generator().manual_seed(3300), dtype=torch.float32) * 0.01).to(dev).bfloat16()
    bf_b = clone4(wb)

    # oneDNN integer reference through torch._int_mm.
    wi8 = torch.randint(-127, 128, (n, k), generator=torch.Generator().manual_seed(3400), dtype=torch.int8).to(dev)
    i8_b = clone4(wi8)

    # Stock vLLM-XPU INC oneDNN INT4 bundles (bit-identical weights for W4A16/W4A8).
    group = 128
    qw = pack_inc(n, k).to(dev)
    scales = (torch.rand((k // group, n), generator=torch.Generator().manual_seed(3500), dtype=torch.float32) * 0.001 + 0.0001).to(dev).half()
    qzeros = torch.tensor([8], dtype=torch.int8, device=dev)
    inc_b = clone4(qw)

    from vllm._xpu_ops import xpu_ops

    cells = []
    weight_bytes = {
        "hf_int4_g16_native": n * k / 2 + n * k / 32 + 4 * n,
        "onednn_bf16_mm": n * k * 2,
        "onednn_int8_intmm": n * k,
        "inc_w4a16": n * k / 2 + (k // group) * n * 2,
        "inc_w4a8_dynact": n * k / 2 + (k // group) * n * 2,
    }

    def add(name, fn, bundles):
        cells.append((name, fn, bundles, weight_bytes[name] + m * k * 2 + m * n * 2))

    add("hf_int4_g16_native", lambda b: hf_int4_g16_linear(x, b["qp"], b["up"], b["d"], signs, backend="native", mode=mode), hf_b)
    add("onednn_bf16_mm", lambda w: torch.mm(x, w.t()), bf_b)
    add("onednn_int8_intmm", lambda w: torch._int_mm(xq, w.t()), i8_b)
    add("inc_w4a16", lambda w: torch.ops._xpu_C.int4_gemm_w4a16(x, w, None, scales, qzeros, group, None), inc_b)

    def inc_w4a8(w):
        qx, xs, xz = xpu_ops.dynamic_per_token_int8_quant_ref(x, True, 8)
        return torch.ops._xpu_C.int4_gemm_w4a8(qx, xs.to(torch.float16), xz, w, scales, qzeros, group, None, None)
    add("inc_w4a8_dynact", inc_w4a8, inc_b)
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/work/results/kernel_bench_gpu0_v2.json")
    ap.add_argument("--quick", action="store_true", help="single shape/reduced M smoke run")
    ap.add_argument("--launches", type=int, default=LAUNCHES)
    ap.add_argument("--shapes", default="", help="comma subset of attn_q,mlp_up,mlp_down")
    ap.add_argument("--ms", default="", help="comma subset of benchmark M values")
    ap.add_argument("--cool-to-c", type=float, default=55.0)
    args = ap.parse_args()
    torch.manual_seed(0)
    # Global one-time power-management warmup (discarded).  Without this, the
    # first small decode cells measure the low-clock ramp rather than the kernel.
    if torch.xpu.is_available():
        a = torch.randn((1024, 1024), device="xpu", dtype=torch.bfloat16)
        b = torch.randn((1024, 1024), device="xpu", dtype=torch.bfloat16)
        for _ in range(80):
            c = a @ b
        torch.xpu.synchronize()
        del a, b, c
        torch.xpu.empty_cache()
    rows = []
    shapes = {"attn_q": SHAPES["attn_q"]} if args.quick else SHAPES
    ms = [1, 8, 512] if args.quick else MS
    wanted_shapes = {x.strip() for x in args.shapes.split(",") if x.strip()}
    if wanted_shapes:
        shapes = {k: v for k, v in shapes.items() if k in wanted_shapes}
    wanted_ms = {int(x) for x in args.ms.split(",") if x.strip()}
    if wanted_ms:
        ms = [m for m in ms if m in wanted_ms]
    tstart = time.perf_counter()
    tele0 = telemetry()
    for shape_name, (n, k) in shapes.items():
        print(f"=== {shape_name} N={n} K={k} ===", flush=True)
        for m in ms:
            try:
                cells = make_cell(shape_name, n, k, m, args.quick)
            except Exception as e:
                print(f"setup failed M={m}: {e}", flush=True)
                raise
            for backend, fn, bundles, bytes_tot in cells:
                r = measure(fn, bundles, launches=args.launches, cool_to_c=args.cool_to_c)
                sec = r["steady_us"] * 1e-6
                flops = 2.0 * m * n * k
                row = {
                    "shape": shape_name, "N": n, "K": k, "M": m, "backend": backend,
                    "mode": "w4a16" if m in (1, 2, 3, 4) else "w4a8",
                    "cold_l2_buffers": NBUF, "launches_per_steady_repeat": args.launches,
                    "gbps": bytes_tot / sec / 1e9,
                    "tops": flops / sec / 1e12,
                    **r,
                }
                rows.append(row)
                print(
                    f"{shape_name:8s} M={m:4d} {backend:18s} steady={r['steady_us']:9.1f}us "
                    f"single={r['single_us']:9.1f}us {row['gbps']:8.1f}GB/s {row['tops']:7.2f}T",
                    flush=True,
                )
            del cells
            torch.xpu.empty_cache()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "device": "Intel Arc Pro B70 GPU0",
        "method": "cold-L2 four-weight-buffer rotation; sync-single plus sync-loop-sync steady state",
        "torch": torch.__version__,
        "telemetry_start": tele0,
        "telemetry_end": telemetry(),
        "wall_s": round(time.perf_counter() - tstart, 3),
        "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print("WROTE", out, flush=True)


if __name__ == "__main__":
    main()
