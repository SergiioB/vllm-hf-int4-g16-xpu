#!/bin/bash
# Reproducible Xe2/BMG-G31 zebin build for hf_int4_kernels.cl.
# Intended to run inside vllm/vllm-openai-xpu:v0.31.0 (ocloc 26.27.39122.11).
# The artifact is validated before atomic publication; a failed build never
# replaces the previous loadable zebin.
set -euo pipefail
cd "$(dirname "$0")"
SRC=hf_int4_kernels.cl
OUTDIR=build
BIN=hf_int4_kernels_bmg.bin
mkdir -p "$OUTDIR"
rm -f "$OUTDIR/hf_int4_kernels_bmg.bin" "$OUTDIR/hf_int4_kernels_bmg.spv" "$OUTDIR/ocloc.log"
ocloc compile \
  -file "$SRC" \
  -device bmg_g31 \
  -64 \
  --format zebin \
  -output hf_int4_kernels_bmg \
  -output_no_suffix \
  -out_dir "$OUTDIR" \
  -options "-cl-std=CL3.0 -cl-mad-enable -cl-intel-256-GRF-per-thread" \
  >"$OUTDIR/ocloc.log" 2>&1
if [ ! -s "$OUTDIR/$BIN" ]; then
  ls -la "$OUTDIR" >&2
  echo "ocloc did not produce $OUTDIR/$BIN" >&2
  exit 2
fi
python3 - <<'PY'
from pathlib import Path
import hashlib, json, os, re, subprocess
root = Path('.')
build_bin = root / 'build' / 'hf_int4_kernels_bmg.bin'
data = build_bin.read_bytes()
try:
    txt = subprocess.check_output(['readelf', '-p', '.ze_info', str(build_bin)], text=True, stderr=subprocess.DEVNULL)
except Exception as e:
    raise SystemExit(f'readelf .ze_info failed: {e}')
names = set(re.findall(r'- name:\s+([A-Za-z0-9_]+)', txt))
expected = {
    'fused_fwht128_actq', 'fused_fwht64_actq', 'reduce_gsk2_epilogue',
    'hf_decode_fused_had64_m1', 'hf_decode_fused_had64_m2', 'hf_decode_fused_had64_m4',
    'hf_moe_pair_had128_bf16', 'hf_moe_pair_had128_f32',
    'hf_moe_pair_had64_bf16', 'hf_moe_pair_had64_f32',
    'hf_moe_bucket_zero', 'hf_moe_bucket_count', 'hf_moe_bucket_scan', 'hf_moe_bucket_scatter',
    'hf_moe_rotate_had128', 'hf_moe_rotate_had64',
    'hf_moe_tile_had128_bf16', 'hf_moe_tile_had128_f32',
    'hf_moe_tile_had64_bf16', 'hf_moe_tile_had64_f32',
    'hf_dpas_m8_sk4', 'hf_dpas_m16_sk4', 'hf_dpas_m32_sk4',
    'hf_dpas_m32_n16', 'hf_dpas_m64_n16', 'hf_dpas_m64_n16_wg16',
    'hf_dpas_m128_n16', 'hf_dpas_m128_n16_wg8', 'hf_dpas_m128_n16_wg8_p1',
    'hf_dpas_m128_n16_p1', 'hf_dpas_m128_n16_p4', 'hf_dpas_m128_n16_gsk2',
    'bf16_m8_sk4', 'bf16_m16_sk4', 'bf16_m32_n16', 'bf16_m128_n16',
    'hf_decode_fused_m1', 'hf_decode_fused_m2', 'hf_decode_fused_m4',
    'mlx_decode_m1', 'mlx_decode_m2', 'mlx_decode_m4',
}
if names != expected:
    raise SystemExit(f'exported kernel set mismatch: missing={sorted(expected-names)} extra={sorted(names-expected)}')
grfs = [int(x) for x in re.findall(r'grf_count:\s+(\d+)', txt)]
meta = {
    'source_sha256': hashlib.sha256((root/'hf_int4_kernels.cl').read_bytes()).hexdigest(),
    'binary_sha256': hashlib.sha256(data).hexdigest(),
    'binary_bytes': len(data),
    'build_options': '-cl-std=CL3.0 -cl-mad-enable -cl-intel-256-GRF-per-thread',
    'device': 'bmg_g31',
    'kernel_count': len(names),
    'exported_kernels': sorted(names),
    'grf_count_min': min(grfs) if grfs else None,
    'grf_count_max': max(grfs) if grfs else None,
    'spill_metadata_present': 'spill' in txt.lower(),
    'required_kernels_present': True,
}
if not grfs or min(grfs) != 256 or max(grfs) != 256:
    raise SystemExit(f'zebin is not uniformly 256-GRF: {meta}')
(root/'build'/'build_metadata.json').write_text(json.dumps(meta, indent=2) + '\n')

def atomic_write(path: Path, payload: bytes):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_bytes(payload)
    os.replace(tmp, path)

# Validation is complete before either publication target is replaced.
atomic_write(root/'hf_int4_kernels_bmg.bin', data)
mirror = (root / '..' / '..' / 'native').resolve()
if mirror.is_dir():
    atomic_write(mirror/'hf_int4_kernels_bmg.bin', data)
    atomic_write(mirror/'hf_int4_kernels.cl', (root/'hf_int4_kernels.cl').read_bytes())
print(json.dumps(meta, indent=2))
PY
