# Module 1: Strided vs. Contiguous VRAM Reads (RTX 5090 Laptop, 24 GB)

## Hypothesis (written BEFORE measuring)
- Peak: 896 GB/s theoretical (256-bit x 28 Gbps / 8). nvidia-smi reports a 14001 MHz
  memory clock; 28 Gbps is assumed from the GDDR7 spec, not independently verified.
- Contiguous copy reaches ~720-800 GB/s effective (80-90% of peak).
- Strided reads waste 7/8 of each 32B sector at stride >= 8. If DRAM-bound,
  slowdown is ~1.5x at stride 2, ~2.5x at stride 4, plateau ~4.5x from stride 8.
- Possible divergence at stride 32: LSU/L1-bound or latency-bound before DRAM-bound.

## Method
- N = 64Mi fp32 outputs, cold L2 (512 MiB flush), CUDA-event timing, median of 50.
- Env: torch 2.11.0+cu128, triton 3.6.0, Windows driver 616.56 (WSL user-mode 615.65.06),
  WSL2 Ubuntu 24.04.5, 82 SMs, sm_120.
