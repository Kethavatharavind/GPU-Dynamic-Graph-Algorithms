# GPU-Accelerated Dynamic Graph Algorithms

Course project for CS5658: GPU Programming (Group 16).

### Team Members
- Arun Shyam (142301007)
- D. Lakshmi Harsha (142301008)
- K. Aravind (112301017)
- B. Lepakshi Kalyan (112301003)

---

## Overview

In real-world graphs such as communication networks and social platforms, edges are frequently added or deleted over time. Recomputing metrics from scratch across massive graphs after every change is computationally expensive.

This project focuses on implementing dynamic graph algorithms on NVIDIA GPUs using CUDA C/C++ to accelerate these updates. Specifically, we study and implement:

1. **Dynamic Triangle Counting**
   - Static triangle counting baseline using degree-based ordering to avoid double counting.
   - Batch-dynamic updates tracking created and destroyed triangles across edge insertion and deletion batches.

2. **Dynamic Connected Components**
   - Parallel connected components using Shiloach-Vishkin style hooking and pointer jumping.
   - Dynamic update handling for edge additions and edge removals.

We evaluate both algorithms against sequential CPU reference baselines to measure execution speedup and update throughput.

## Project Structure

```
gpu-dynamic-graph-algorithms/
├── src/
│   ├── triangle.cuh      # CUDA kernels for static and batch-dynamic triangle counting
│   └── components.cuh    # CUDA kernels for parallel hooking & pointer jumping (CC)
└── README.md
```

*(Note: Graph loader, CPU reference baselines, benchmark runner, and Slurm scripts are currently being staged and will be pushed in upcoming commits.)*

## Environment & Prerequisites

- **Language:** C++17 / CUDA C++
- **Target GPU Architecture:** sm_75 (Turing) or higher
- **HPC Environment:** Bhavani Cluster (Slurm workload manager)
- **Compiler:** `nvcc` (CUDA 11.0+) / `g++` (C++17)
