# PyTorch Microkernel Suite Benchmark

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![CI](https://img.shields.io/badge/CI-host--safe-green.svg)](.github/workflows/ci.yml)

Target: Ubuntu 26.04 · AMD · see Hardware Requirements. This is a host benchmark, not a laptop `pip install` project.

## Quick Start

```bash
git clone https://github.com/garys-gpu-benchmarks/319-gpu-bench-amd-torch-micro-suite-ubu2604.git
cd 319-gpu-bench-amd-torch-micro-suite-ubu2604
sudo bash setup.sh --assume-yes
bash run_benchmark.sh --profile smoke --validate
```
Results are written to `results/benchmark.db` and `results/summary.json`.

This workload is executed on the validation host after the repository is copied there. `setup.sh` and `run_benchmark.sh` do not open an outbound SSH session.

Prerequisites: Ubuntu 26.04; AMD; Python 3.14.4; root or sudo for `setup.sh`. Framework: Bash, SQLite, Python, PyYAML, ROCm Runtime, PyTorch-ROCm, HIP/ROCm. This is a host benchmark, not a laptop `pip install` project.

```mermaid
flowchart LR
  setup.sh --> run_benchmark.sh --> parse_results.py --> results/benchmark.db
```

## 1. Overview

Runs repository-local scripts/gemm.py and scripts/conv2d.py (PyTorch on ROCm; the same scripts as 219/419) through scripts/collect_torch_micro.py. device_id, dtype, M/N/K, input_shape, warmup_iters, num_iterations, and seed come from yaml; batch_size and kernel_type do not change what runs (both kernels always run). This is not TorchBench. output_format: csv Sweep dimensions: device_id, kernel_type, dtype, input_shape, M, N, K, batch_size.

## 2. What It Validates

- Validates local gemm.py and conv2d.py RESULT parse. Official TorchBench runners are not executed
- #1: GEMM execution time, us (gemm_time_usec); is present and physically sensible.
- #2: GEMM throughput, TFLOPS (gemm_tflops); is present and physically sensible.
- #3: GEMM minimum traffic, GB/s (gemm_bandwidth_gb_s); is present and physically sensible.
- #4: Conv2d execution time, us (conv_time_usec); is present and physically sensible.
- #5: Conv2d throughput, TFLOPS (conv_tflops); is present and physically sensible.
- #6: Conv2d minimum traffic, GB/s (conv_bandwidth_gb_s) is present and physically sensible.

## 3. Metrics Captured

- **#1: GEMM execution time, us** — stored as `gemm_time_usec`.
- **#2: GEMM throughput, TFLOPS** — stored as `gemm_tflops`.
- **#3: GEMM minimum traffic, GB/s** — stored as `gemm_bandwidth_gb_s`.
- **#4: Conv2d execution time, us** — stored as `conv_time_usec`.
- **#5: Conv2d throughput, TFLOPS** — stored as `conv_tflops`.
- **#6: Conv2d minimum traffic, GB/s** — stored as `conv_bandwidth_gb_s`.

## 4. Hardware Requirements

### Supported environment

- OS: Ubuntu 26.04
- GPU vendor: AMD
- Framework family: Bash, SQLite, Python, PyYAML, ROCm Runtime, PyTorch-ROCm, HIP/ROCm
- Python: Python 3.14.4

### Reference validation environment

The tables below describe the machine used to generate the reference results. They are not a requirement that every user buy that exact cloud instance.

### System

Ubuntu 26.04 / AMD / Bash, SQLite, Python, PyYAML, ROCm Runtime, PyTorch-ROCm, HIP/ROCm

### GPU

Ubuntu 26.04 / AMD / Bash, SQLite, Python, PyYAML, ROCm Runtime, PyTorch-ROCm, HIP/ROCm

## 5. Software Requirements

| Component | Version |
|---|---|
| OS | Ubuntu 26.04 |
| Kernel | kernel 7.0.0 |
| Python | Python 3.14.4 |
| ROCm | ROCm 7.14 |
| rocBLAS | rocBLAS 5.2.0 |

Runs repository-local scripts/gemm.py and scripts/conv2d.py (PyTorch on ROCm; the same scripts as 219/419) through scripts/collect_torch_micro.py. device_id, dtype, M/N/K, input_shape, warmup_iters, num_iterations, and seed come from yaml; batch_size and kernel_type do not change what runs (both kernels always run). This is not TorchBench.

## 6. Installation

```bash
Run scripts/gemm.py and scripts/conv2d.py. Official TorchBench model runners are not executed
```

## 7. Running the Benchmark

```bash
Run scripts/gemm.py and scripts/conv2d.py. Official TorchBench model runners are not executed
```

**Validating results separately:**

```bash
export BENCHMARK_PYTHON=/usr/bin/python3.13  # optional
python3 -m venv .venv
source ".venv/bin/activate"
".venv/bin/python" scripts/validate_results.py
```

## 8. Output

### `results/benchmark.db` (SQLite)

raw_results.csv, raw_results.jsonl, and gemm plus conv2d text in raw_output.txt. collect_workload.py copies raw_results.csv to the harness raw file

check,device_id,dtype,kernel_type,status,gemm_time_usec,gemm_tflops,gemm_bandwidth_gb_s,conv_time_usec,conv_tflops,conv_bandwidth_gb_s
summary,0,fp16,all,ok,20,10,400,30,8,300

```bash
Run scripts/gemm.py and scripts/conv2d.py. Official TorchBench model runners are not executed
```

### `results/summary.json`

Consolidated metrics from the most recent run — suitable for CI artifact upload or dashboard ingestion.

### `results/raw/<timestamp>.txt`

raw_results.csv, raw_results.jsonl, and gemm plus conv2d text in raw_output.txt. collect_workload.py copies raw_results.csv to the harness raw file

check,device_id,dtype,kernel_type,status,gemm_time_usec,gemm_tflops,gemm_bandwidth_gb_s,conv_time_usec,conv_tflops,conv_bandwidth_gb_s
summary,0,fp16,all,ok,20,10,400,30,8,300

## 9. Baselines / Thresholds

Expected ranges and gates live in `config/benchmark_config.yaml` under `baselines:` or `thresholds:`. To update them, edit that file — never edit validation code directly.

## 10. Troubleshooting

**`setup.sh` missing collector**
Create cannot finish without `scripts/collect_workload.py`.

**`self_check` overlay rewritten**
Do not overwrite files listed in `results/overlay_lock.json`.

**Remote SSH drop during setup**
Reconnect and resume `bash setup.sh --assume-yes`. Do not wipe `.venv` or `.cache`.

## 11. NVIDIA H100 Coding Differences

Primary target is AMD ROCm. NVIDIA notes in this section are reference only and are not the execution path.

## Repository layout

```text
.
├── setup.sh
├── run_benchmark.sh
├── benchmark_specification.json
├── config/
├── scripts/
├── src/
├── tests/
├── docs/
├── results/
└── LICENSE
```
