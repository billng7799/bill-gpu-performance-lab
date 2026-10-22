#!/usr/bin/env python3
"""
01_vram_saturation.py: Contiguous vs. strided global-memory reads on RTX 5090 Laptop.

Experiment
----------
out[i] = src[i * stride]    for i in [0, N)

Same N, same bytes of *useful* payload, but the read stride changes how many
32-byte L2 sectors the memory system must fetch to deliver it.

Systems analogy (for the DB/kernel veteran)
-------------------------------------------
  sequential scan w/ readahead   <->  coalesced warp loads
  one-row-per-page index probes  <->  strided loads (read amplification)
  page / buffer-pool granularity <->  32B sector / 128B cache line
  I/O queue depth (Little's law) <->  warps in flight hiding ~600-800ns DRAM latency
  RAID stripe / NUMA channels    <->  256-bit bus = 8 x 32-bit GDDR7 channels
"""
import argparse, datetime, json, pathlib, statistics
import torch

# ── Hardware constants ─────────────────────────────────────────────────────
RTX5090_PEAK_GBPS = 896.0      # laptop: 256-bit * 28 Gbps / 8 (override with --peak-gbps)
SECTOR_BYTES      = 32         # L2 <-> DRAM transfer quantum on recent NVIDIA parts
LINE_BYTES        = 128        # L1/L2 cache line = 4 sectors
ELEM_BYTES        = 4          # float32
GB                = 1e9        # decimal GB to match how vendors quote bandwidth


def parse_args():
    root = pathlib.Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-elems", type=int, default=64 * 1024 * 1024,
                    help="output elements (default 64Mi = 256 MiB fp32)")
    ap.add_argument("--strides", type=str, default="1,2,4,8,16,32")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--flush-mb", type=int, default=512,
                    help="scratch read between iterations to evict L2 (must be >> L2)")
    ap.add_argument("--peak-gbps", type=float, default=RTX5090_PEAK_GBPS)
    ap.add_argument("--out", type=pathlib.Path,
                    default=root / "benchmarks/results/01_vram_saturation.json")
    ap.add_argument("--plot", action="store_true")
    return ap.parse_args()


def bench_stride(src, out, stride, n, flush, iters, warmup):
    """Return per-iteration kernel times in ms, measured with CUDA events."""
    # A strided *view*: no data movement here, just (shape=(n,), stride=(stride,)).
    # The copy kernel must gather through this view.
    view = src[: n * stride : stride]

    for _ in range(warmup):                     # clocks ramp, JIT/allocator settle
        out.copy_(view)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        # Cold-cache methodology: read 512 MiB (>> 64 MiB L2) so the timed kernel
        # starts with no residency. The flush is stream-ordered *before* start.record().
        flush.sum()

        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)

        start.record()   # event is timestamped on the GPU timeline, so there is no
                         # host launch latency or Python overhead in the measurement
        with torch.cuda.nvtx.range(f"copy_stride_{stride}"):   # visible in nsys timeline
            # stride == 1: vectorized_elementwise_kernel, 128-bit LDG/STG per thread,
            #              a warp's 32 lanes cover 512 contiguous bytes.
            # stride >= 2: generic elementwise_kernel with per-element index math.
            #              A warp's lanes hit addresses `stride*4` bytes apart.
            out.copy_(view)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))   # milliseconds
    return times


def main():
    a = parse_args()
    assert torch.cuda.is_available(), "CUDA not available"
    dev = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)
    strides = [int(s) for s in a.strides.split(",")]
    n, max_stride = a.n_elems, max(strides)

    # ── Capacity check (VRAM is finite, and WDDM/Windows holds some) ────────
    need = (n * max_stride + n) * ELEM_BYTES + a.flush_mb * 2**20
    free, total = torch.cuda.mem_get_info()
    print(f"Device : {props.name} | sm_{props.major}{props.minor} | {props.multi_processor_count} SMs | "
          f"{total/2**30:.1f} GiB total, {free/2**30:.1f} GiB free")
    l2 = getattr(props, "L2_cache_size", None)
    if l2:
        print(f"L2     : {l2/2**20:.0f} MiB  (flush buffer {a.flush_mb} MiB)")
    print(f"Peak   : {a.peak_gbps:.0f} GB/s theoretical | need {need/2**30:.1f} GiB")
    if need > free * 0.95:
        raise SystemExit("Insufficient free VRAM: lower --n-elems or --strides")

    # ── Allocation: ONE big source buffer; every stride is a view into it ───
    # Large allocation => PyTorch uses cudaMalloc'd 2MiB-aligned segments, so
    # GPU large pages keep TLB misses off the critical path. Strides up to 32
    # floats (128B) still walk monotonically through memory => DRAM row-buffer
    # locality remains decent, and any slowdown is overwhelmingly sector waste.
    src   = torch.empty(n * max_stride, dtype=torch.float32, device=dev).uniform_()
    out   = torch.empty(n, dtype=torch.float32, device=dev)
    flush = torch.empty(a.flush_mb * 2**20 // ELEM_BYTES, dtype=torch.float32, device=dev)
    flush.uniform_()

    # ── Correctness first: a fast wrong kernel is worthless ─────────────────
    for s in strides:
        out.copy_(src[: n * s : s])
        assert torch.equal(out, src[: n * s : s]), f"mismatch at stride {s}"
    torch.cuda.synchronize()

    # ── Timed region. cudaProfilerStart/Stop gate nsys/ncu so they capture ──
    # ── ONLY the benchmark, not allocation/init/verification kernels.     ──
    results = []
    torch.cuda.profiler.start()
    for s in strides:
        times = bench_stride(src, out, s, n, flush, a.iters, a.warmup)
        med, best = statistics.median(times), min(times)

        # Useful payload: n floats read + n floats written
        useful_bytes = 2 * n * ELEM_BYTES
        eff_gbps = useful_bytes / (med * 1e-3) / GB

        # MODELED bus traffic. A load of 4B drags in a whole 32B sector. When
        # stride*4B < 32B, neighbouring elements share sectors (partial reuse);
        # at stride >= 8 floats every element costs a *private* sector => 8x read
        # amplification. Writes stay coalesced (4B/elem). ncu will give the real
        # number (dram__bytes_read.sum); treat this as a hypothesis to falsify.
        read_bytes_per_elem = min(s * ELEM_BYTES, SECTOR_BYTES)
        bus_bytes = n * (read_bytes_per_elem + ELEM_BYTES)
        bus_gbps = bus_bytes / (med * 1e-3) / GB

        results.append(dict(
            stride=s, stride_bytes=s * ELEM_BYTES, median_ms=med, min_ms=best,
            effective_gbps=eff_gbps, effective_pct_peak=100 * eff_gbps / a.peak_gbps,
            modeled_bus_gbps=bus_gbps, modeled_bus_pct_peak=100 * bus_gbps / a.peak_gbps,
        ))
    torch.cuda.profiler.stop()

    base = results[0]["median_ms"]
    for r in results:
        r["slowdown_vs_contig"] = r["median_ms"] / base

    # ── Report ───────────────────────────────────────────────────────────────
    hdr = (f"{'stride':>6} {'B/step':>6} {'med ms':>8} {'min ms':>8} {'eff GB/s':>9} "
           f"{'%peak':>6} {'bus GB/s':>9} {'%peak':>6} {'slowdn':>7}")
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for r in results:
        print(f"{r['stride']:>6} {r['stride_bytes']:>6} {r['median_ms']:>8.3f} {r['min_ms']:>8.3f} "
              f"{r['effective_gbps']:>9.1f} {r['effective_pct_peak']:>5.1f}% "
              f"{r['modeled_bus_gbps']:>9.1f} {r['modeled_bus_pct_peak']:>5.1f}% "
              f"{r['slowdown_vs_contig']:>6.2f}x")
    print("\neff GB/s = useful payload / time   (what your application sees)")
    print("bus GB/s = modeled sector traffic / time (what the memory system works)")
    print("If bus GB/s sits near the contiguous eff GB/s, you are DRAM-bandwidth-bound\n"
          "by read amplification, not by arithmetic or launch overhead.")

    # ── Persist evidence ─────────────────────────────────────────────────────
    a.out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
        device=props.name, torch=torch.__version__, cuda=torch.version.cuda,
        n_elems=n, iters=a.iters, peak_gbps=a.peak_gbps, results=results)
    a.out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {a.out}")

    if a.plot:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [r["stride"] for r in results]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(xs, [r["effective_gbps"] for r in results], "o-", label="effective (useful)")
        ax.plot(xs, [r["modeled_bus_gbps"] for r in results], "s--", label="modeled bus traffic")
        ax.axhline(a.peak_gbps, color="r", ls=":", label=f"theoretical peak {a.peak_gbps:.0f}")
        ax.set_xscale("log", base=2); ax.set_xlabel("read stride (float32 elements)")
        ax.set_ylabel("GB/s"); ax.set_title(f"{props.name}: strided-read bandwidth")
        ax.legend(); ax.grid(alpha=.3); fig.tight_layout()
        png = a.out.with_suffix(".png"); fig.savefig(png, dpi=150); print(f"wrote {png}")


if __name__ == "__main__":
    main()
