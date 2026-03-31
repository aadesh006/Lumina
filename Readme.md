# Lumina

**GPU-Accelerated Image Processing Engine**  
High-performance C++/CUDA system for exploring parallel compute, memory hierarchies, and GPU offloading.

---

## Overview

**Lumina** is a research-oriented GPU-accelerated image processing prototype built using **C++17** and **CUDA**.  
The project focuses on understanding *how* performance is achieved on modern GPUs rather than building a consumer-facing library.

Lumina explores:
- Host ↔ Device separation
- Explicit GPU memory management (`cudaMalloc`, `cudaMemcpy`, `cudaFree`)
- Tiled parallel execution with shared memory
- Performance bottlenecks such as PCIe transfers

This project is intended as a **systems and performance engineering learning artifact**.

---

## Current Status

**Phase 2 Complete — Shared Memory Box Blur**

### Completed
- Stable CUDA development environment under **WSL2**
- Robust **CMake** build system for C++ + CUDA
- Explicit GPU memory allocation and transfer pipeline
- Tiled **Box Blur** CUDA kernel using `__shared__` memory
  - 16×16 thread blocks (`TILE_SIZE = 16`)
  - Blur radius of 3 (`RADIUS = 3`), yielding a 7×7 blur window
  - Halo loading: each block cooperatively loads its tile plus surrounding border pixels into shared memory before computing
- End-to-end execution: load → GPU blur → write

### Planned
- Gaussian blur kernel
- Sobel edge detection
- Performance benchmarking vs. naive (global memory) implementation
- Support for multi-channel (RGB) images

---

## Usage

```bash
./lumina_app path/to/input.jpg
```

The processed image is written to `blurred_output.jpg` in the working directory.

> **Note:** Input is loaded as a single-channel (grayscale) image via `stb_image`. Multi-channel RGB support is planned.

---

## Repository Structure

```
Lumina/
├── CMakeLists.txt           # Build configuration (C++17 + CUDA, native arch)
├── include/
│   ├── stb_image.h          # Single-header image loader
│   └── stb_image_write.h    # Single-header image writer
├── src/
│   ├── main.cpp             # Host-side orchestration: load → blur → save
│   ├── kernels.cu           # CUDA device kernels
│   └── kernels.cuh          # Kernel launch declarations
├── Basics/                  # Standalone CUDA learning exercises
│   ├── hello.cu             # Hello World: basic kernel launch and threadIdx
│   ├── sqArray.cu           # Array squaring: explicit cudaMalloc/cudaMemcpy pattern
│   └── vectorAdd.cu         # Vector addition: Unified Memory + grid-stride loops
└── README.md
```

---

## Build & Run

### Prerequisites

- Linux (native or WSL2)
- NVIDIA GPU with CUDA support
- CUDA Toolkit (tested on CUDA 12.9)
- CMake ≥ 3.18
- GCC / Clang compatible with CUDA

### Build

```bash
mkdir build
cd build
cmake ..
make
```

### Run

```bash
./lumina_app path/to/input.jpg
# Output saved to: blurred_output.jpg
```

### Build and run a Basics exercise

```bash
nvcc Basics/hello.cu -o hello && ./hello
nvcc Basics/sqArray.cu -o sqArray && ./sqArray
nvcc Basics/vectorAdd.cu -o vectorAdd && ./vectorAdd
```

---

## Kernel Details

### `blurKernelShared` (`src/kernels.cu`)

A tiled box blur kernel that uses shared memory to avoid redundant global memory reads.

| Parameter | Value |
|---|---|
| Thread block size | 16 × 16 |
| Blur radius | 3 (compile-time constant `RADIUS`) |
| Blur window | 7 × 7 (49 pixels averaged) |
| Shared tile size | 22 × 22 (`TILE_SIZE + 2 * RADIUS`) |

**How it works:**

1. Each block is responsible for a 16×16 output tile.
2. All threads in the block cooperatively load a 22×22 region (the tile plus a 3-pixel halo on each side) into `__shared__` memory using a strided loop.
3. `__syncthreads()` ensures the load is complete before any thread reads from shared memory.
4. Each thread computes the average of the 7×7 neighborhood around its pixel entirely from shared memory, with zero additional global reads.
5. Boundary pixels default to `0` (black padding) when the halo extends outside the image.

---

## Basics Exercises

The `Basics/` directory contains self-contained CUDA programs used to build foundational GPU programming intuition:

**`hello.cu`** — Launches a kernel across 5 threads and prints each thread's `threadIdx.x`. Demonstrates the minimal kernel launch syntax and `cudaDeviceSynchronize`.

**`sqArray.cu`** — Squares each element of a float array on the GPU. Demonstrates the explicit `cudaMalloc` → `cudaMemcpy` (H→D) → kernel → `cudaMemcpy` (D→H) → `cudaFree` pattern.

**`vectorAdd.cu`** — Adds two 1M-element float vectors using Unified Memory (`cudaMallocManaged`) and a grid-stride loop, so the kernel handles arrays larger than the total thread count. Verifies correctness and prints max error.

---

## Author

**Aadesh Chaudhari**  
GitHub: [@aadesh006](https://github.com/aadesh006)