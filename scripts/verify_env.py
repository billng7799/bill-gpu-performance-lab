#!/usr/bin/env python3
"""Environment smoke test: driver -> torch -> Triton JIT -> sm_120 codegen."""
import torch, triton, triton.language as tl

print(f"torch {torch.__version__} | CUDA runtime {torch.version.cuda} | triton {triton.__version__}")
assert torch.cuda.is_available(), "CUDA not visible to PyTorch"
p = torch.cuda.get_device_properties(0)
print(f"device     : {p.name}")
print(f"capability : sm_{p.major}{p.minor}   (RTX 5090 should be sm_120)")
print(f"VRAM       : {p.total_memory / 2**30:.1f} GiB")
print(f"SMs        : {p.multi_processor_count}")
print(f"arch list  : {torch.cuda.get_arch_list()}")

# cuBLAS path
a = torch.randn(4096, 4096, device="cuda"); b = torch.randn_like(a)
torch.cuda.synchronize(); c = a @ b; torch.cuda.synchronize()
print("matmul OK  :", c.shape)

# Triton path: exercises the full JIT -> PTX -> ptxas(sm_120) chain
@triton.jit
def add_k(x, y, o, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(o + i, tl.load(x + i, mask=m) + tl.load(y + i, mask=m), mask=m)

n = 1 << 20
x = torch.randn(n, device="cuda"); y = torch.randn(n, device="cuda"); o = torch.empty_like(x)
add_k[(triton.cdiv(n, 1024),)](x, y, o, n, BLOCK=1024)
torch.cuda.synchronize()
assert torch.allclose(o, x + y)
print("triton JIT OK")
