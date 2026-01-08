# Lumina

**GPU-Accelerated Image Processing Engine**  
High-performance C++/CUDA system for exploring parallel compute, memory hierarchies, and GPU offloading.

---

## Overview

**Lumina** is a research-oriented GPU-accelerated image processing prototype built using **C++17** and **CUDA**.  
The project focuses on understanding *how* performance is achieved on modern GPUs rather than building a consumer-facing library.

Lumina explores:
- Host ↔ Device separation
- Explicit GPU memory management
- Parallel execution models
- Performance bottlenecks such as PCIe transfers

This project is intended as a **systems and performance engineering learning artifact**, not a replacement for existing libraries like OpenCV.

---

## Current Status

**Phase 1 Complete — Functional Prototype**

### Completed
- Stable CUDA development environment under **WSL2**
- Robust **CMake** build system for C++ + CUDA
- Explicit GPU memory allocation and transfer pipeline
- Parallel **RGB → Grayscale** CUDA kernel
- End-to-end execution: load → compute → write

### In Progress / Planned
- **Phase 2:** Box Blur (Convolution)
- **Phase 3:** Shared Memory Optimization & Performance Benchmarking

You can currently execute:

```bash
./lumina_app input.jpg output.jpg
```
---

### Repository Structure
Lumina/
├── CMakeLists.txt        # Build configuration
├── .gitignore
├── include/
│   ├── stb_image.h
│   └── stb_image_write.h
├── src/
│   ├── main.cpp          # Host-side orchestration
│   └── kernels.cu        # CUDA device kernels
└── README.md             # Project documentation

---

### Build & Run Instructions
**Prerequisites**

- Linux (native or WSL2)
- NVIDIA GPU with CUDA support
- CUDA Toolkit (tested on CUDA 12.9)
- CMake ≥ 3.18
- GCC / Clang compatible with CUDA

```bash
mkdir build
cd build
cmake ..
make
```
**Run**
```bash
./lumina_app path/to/input.jpg path/to/output.jpg
```

