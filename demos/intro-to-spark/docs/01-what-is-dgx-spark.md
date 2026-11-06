# What is DGX Spark?

A desktop-sized AI computer built on the **GB10 Grace Blackwell Superchip**: an Arm CPU and a Blackwell GPU
on one package sharing a single pool of memory. It runs the same NVIDIA software stack (CUDA, PyTorch,
NGC containers) as data-center systems, so what you build here moves to larger systems unchanged.

## Key specs (verify with Lab 01 on your unit)

| | |
|---|---|
| Chip | NVIDIA GB10 Grace Blackwell Superchip |
| CPU | 20 Arm cores (10 Cortex-X925 + 10 Cortex-A725) — **aarch64, not x86** |
| GPU | Blackwell, 5th-gen Tensor Cores, up to ~1 PFLOP FP4 (with sparsity) |
| Memory | 128 GB LPDDR5x **unified** between CPU and GPU, ~273 GB/s |
| Storage | 1–4 TB NVMe (config dependent) |
| Network | 10 GbE, Wi-Fi 7, ConnectX-7 200 Gb/s for linking two units |
| OS | DGX OS (Ubuntu-based) with CUDA toolkit and Docker + NVIDIA Container Toolkit preinstalled |

## Three things that differ from a typical GPU server

1. **Unified memory.** There is no separate "VRAM". A model can use most of the 128 GB, so large models
   that would not fit on a 24–48 GB GPU load fine. The trade-off: bandwidth (~273 GB/s) is far below HBM
   (multi TB/s), so large *memory-bound* workloads run slower than on data-center GPUs.
2. **aarch64.** Any pip wheel or container must be built for Arm64. Pure-Python packages are fine; CUDA
   extensions (natten, flash-attn, earth2grid) are where problems show up. This is the main risk when moving x86 workflows to Spark.
3. **Blackwell compute capability 12.1 (`sm_121`).** Older PyTorch builds without Blackwell kernels fail with
   "no kernel image is available". Use NGC PyTorch containers or recent CUDA 13 wheels.

## Where it fits

| Good fit | Poor fit |
|---|---|
| Developing and testing workflows before scaling out | Training large models from scratch |
| Inference with large models that need lots of memory | Throughput-critical production serving |
| Teaching: one box per team, SSH access, full stack | Many concurrent heavy users on one unit |
| Fine-tuning / experiments on modest datasets | Petabyte-scale reanalysis processing |
| Data sovereignty: forecasts generated on-premises | |

Next: [capabilities](02-capabilities.md).
