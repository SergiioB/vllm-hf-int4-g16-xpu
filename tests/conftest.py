"""Session-wide XPU safety rails.

An XPU allocation larger than VRAM spills into host shared memory that docker
``--memory`` does not bound, and that has rebooted the host.  Cap this process
before any test allocates, and ask for deterministic kernels because XPU float
matmuls are otherwise run-to-run different.
"""
import torch

VRAM_FRACTION = 0.55  # ~18 GiB of a 32 GiB Arc Pro B70

if torch.xpu.is_available():
    torch.xpu.memory.set_per_process_memory_fraction(VRAM_FRACTION)

# warn_only: an op with no deterministic implementation warns instead of failing.
torch.use_deterministic_algorithms(True, warn_only=True)
