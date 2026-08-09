# Energy Efficiency: GFLOPS/watt

Hardware: AMD Ryzen 7 5800H (8-core/16-thread), NVIDIA GeForce RTX 3050 Laptop. Both laptop-class,
power-limited parts — see `results/summary.md` Stage 3/5.6 for how that shapes the rest of this
project's findings.

Measured via `scripts/energy_bench.sh mandelbrot 1920 1080 1000 200` (both rows from a single run):
CPU side reads Intel RAPL's package-energy counter before and after a `--runs 200` batch; GPU side
background-polls `nvidia-smi --query-gpu=power.draw` for the duration of a `--cuda-loop 200` run.
Both sides use the *same* real iteration count (`total_iters` from the CPU run) for the FLOP
calculation — not a `pixels × max_iter` worst-case bound, since most pixels escape long before
`max_iter` and that bound would overstate true GFLOPS substantially. This is valid because CPU and
GPU are now verified bit-identical on this codebase (see the "math must be perfect" correctness
fixes referenced in `results/summary.md`).

Fractal: Mandelbrot, 1920×1080, `max_iter=1000`, 200 renders per backend.

| Backend | Wall time | Avg power | GFLOPS | GFLOPS/watt |
|---|---:|---:|---:|---:|
| CPU (16 threads) | 1.877s | 33.82 W | 100.35 | **2.97** |
| GPU (CUDA)        | 5.341s | 13.49 W |  54.37 | **4.03** |

## Honest finding — the GPU is more energy-efficient per FLOP despite being slower overall

This flips the naive expectation. Every other benchmark in this project (Stage 2a, Stage 3) found
this GPU *slower* than the CPU in absolute terms — a modest, power-limited RTX 3050 Laptop losing to
an already well-optimized 16-thread CPU kernel. But on **GFLOPS per watt, the GPU wins by ~36%**
(4.03 vs 2.97), because its absolute power draw (13.49W) is less than half the CPU's (33.82W) under
full load — low enough that even at less than half the CPU's raw throughput, it still comes out
ahead per watt.

This is consistent with the two backends' fundamentally different design points: the CPU is running
16 threads flat-out on a general-purpose core design (higher power ceiling, higher raw throughput
when the workload suits it, as it clearly does here per Stage 2a); the GPU, even underutilized by
this workload's fixed per-frame transfer overhead (Stage 2a's finding), still spends much less power
doing it. For a battery-powered/thermally-constrained context, this is a real, practical argument for
preferring the GPU path even where it's not the fastest option in wall-clock terms — a genuinely
useful, non-obvious result for this thesis's hardware, and a reminder that "faster" and "more
efficient" are different questions, not two names for the same measurement.
