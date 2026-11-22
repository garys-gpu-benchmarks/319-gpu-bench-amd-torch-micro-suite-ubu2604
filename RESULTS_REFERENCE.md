# Results: 319 PyTorch Microkernel Suite

**Main goal:** Measure PyTorch GEMM and Conv2d microkernel time, TFLOPS, and bandwidth  
**Tools:** Run the PyTorch microkernel suite through scripts/gemm.py and scripts/conv2d.py

Reference results from the validation run on the system shown in each section (recorded 2026-11-22). Your own runs write `results/RESULTS_ACTUAL.md` in the same format.

> Smoke runs use deliberately small sizes. They check that the workload installs and runs end to end; their numbers are not performance results. Baseline and extended are.

<!-- profile:smoke -->
## smoke

- **System:** 1× AMD Instinct MI300X VF (196288 MB) · INTEL(R) XEON(R) PLATINUM 8568Y+ · Ubuntu 26.04.1 LTS · kernel 7.0.0-43-generic · driver 7.1.9.31600000
- **Run:** 2026-11-22T14:41:36Z · status **PASS** (exit 0)
- **Time (mm:ss):** setup 01:13 · benchmark 00:07 · whole run 01:27
- **Source:** run 319_20261122_144136
- **Command:** `bash run_benchmark.sh --smoke --device-id 0 --dtype fp16 --input-shape 1,3,32,32 --M 64 --N 64 --K 64 --warmup-iters 2 --num-iterations 5 --seed 42`

| # | Metric | Result |
|---|---|---|
| 1 | GEMM execution time, us (gemm_time_usec) | 23.3962 |
| 2 | GEMM throughput, TFLOPS (gemm_tflops) | 0.022409 |
| 3 | GEMM minimum traffic, GB/s (gemm_bandwidth_gb_s) | 1.050427 |
| 4 | Conv2d execution time, us (conv_time_usec) | 40.4726 |
| 5 | Conv2d throughput, TFLOPS (conv_tflops) | 0.01093 |
| 6 | Conv2d minimum traffic, GB/s (conv_bandwidth_gb_s) | 0.567297 |
<!-- /profile:smoke -->
