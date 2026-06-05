# CustomAwid — Benchmark Results (KM2)

Single source of truth for the IEEE article (`article/main.typ`).

## Environment

| Item | Value |
|---|---|
| Language / runtime | Julia 1.12 |
| Machine | 6 physical / 12 logical CPU cores |
| Batch size | 20 |
| Hyper-parameters | 3 epochs, η = 0.01, plain SGD, dropout 0.4 |
| Network | conv(3×3,1→6,pad1) → maxpool(2) → conv(3×3,6→16,pad1) → maxpool(2) → flatten → dense(784→84,relu) → dropout(0.4) → dense(84→10) → logit-cross-entropy |
| Parameters | 67,708 |
| Library micro-benchmarks | `bench.jl`, `bench_conv.jl` — `BenchmarkTools.@btime`, fixed random batch, `Random.seed!(0)`, warm-up before timing |
| Accuracy / wall-clock | `train.jl` (Julia), `reference/reference_pytorch.py`, `reference/reference_flux.jl` — FashionMNIST, mean ± std over 4 seeds (0–3) |
| Threads | single CPU thread for the per-step tables; BLAS=6 for the Julia library by default |

Step = one full training step (`zerograd!` → `forward!` → `backward!` → `optimize!`)
on a 28×28×1×20 batch, single thread.

## Table I — Per-step time and allocations across optimization stages

| Stage | fwd (ms) | bwd (ms) | step (ms) | alloc / step |
|---|---:|---:|---:|---:|
| KM1 baseline | 10.33 | 7.17 | 17.29 | 1.84 MiB |
| + type stability | 10.33 | 7.23 | 17.28 | 1.84 MiB |
| + in-place ops | 10.17 | 7.21 | 17.11 | 12.95 KiB |
| + `@turbo` conv | 0.60 | 5.56 | 6.13 | 12.97 KiB |
| + `Float32` (whole net) | 0.58 | 5.46 | 5.91 | 8.58 KiB |

## Table II — Isolated convolution kernels (ms)

Dense-gradient worst case. Numerical equality vs. the naive kernels verified
(forward / dW / dx, rtol 1e-3).

| Kernel | naive | `@turbo` |
|---|---:|---:|
| conv1 forward | 1.42 | 0.076 |
| conv1 backward | 2.85 | 2.64 |
| conv2 forward | 5.73 | 0.191 |
| conv2 backward | 11.39 | 10.60 |

## Table III — Accuracy progression (initialization)

`train.jl`, FashionMNIST, 3 epochs. He = `sqrt(2/fan_in)`.

| | conv `·0.1`, dense `·0.01` | + He conv | + He dense (final) |
|---|---:|---:|---:|
| epoch 1 | 75.38% | 79.59% | — |
| epoch 2 | 80.01% | 84.89% | — |
| epoch 3 | 83.26% | 85.38% | — |
| 4-seed mean | ~83% | 85.9 ± 0.36% | 86.2 ± 0.24% |

(`+ He dense` column measured in Float64.)

**Single precision (Float32, final library), 4 seeds:** 86.42 ± 0.46% — per seed
85.89 / 86.85 / 86.76 / 86.18, all ≥ 85%. Indistinguishable from the Float64 result
(86.2 ± 0.24%); Float32 is sufficient precision for this task.

## Table IV — Final library vs. identical references

CPU, identical architecture/hyper-parameters, 4 seeds, 3 epochs, run one at a time.

| Metric | CustomAwid | Flux (Julia) | PyTorch |
|---|---:|---:|---:|
| Test accuracy (%) | 86.4 ± 0.46 | 85.2 ± 0.48 | 84.0 ± 0.13 |
| Time / 3 epochs (s) | ≈ 88 | ≈ 63 | ≈ 22.9 |
| Alloc. / step | 8.6 KiB | 2145 KiB | — |
| BLAS threads | 6 | 12 | 6 |
| Process RSS | light | — | ≈ 942 MiB |

## Headline summary

| Metric | KM1 baseline | KM2 final | gain |
|---|---:|---:|---:|
| step time | 17.29 ms | 5.91 ms | 2.9× |
| forward pass | 10.33 ms | 0.58 ms | 17× |
| alloc / step | 1.84 MiB | 8.6 KiB | −99% |
| test accuracy (4 seeds) | ~83% | 86.4 ± 0.46% | > 85% target |
| 3 epochs (full training) | — | ≈ 88 s | ≪ 15 min limit |
