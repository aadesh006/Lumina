# Lumina

**GPU-Accelerated Image Processing Engine**  
High-performance C++/CUDA library for experimenting with parallel image compute.

---

## Overview

**Lumina** is a research-oriented, GPU-accelerated image processing prototype built in C++ and CUDA. Its purpose is to explore the fundamentals of heterogeneous computing — moving compute from CPU to GPU, managing memory explicitly, and optimizing parallel workloads for performance.

It is *not* a production-ready library. Instead, it serves as a learning artifact and platform for benchmarking memory hierarchies, thread coordination, and data movement.

---

## Current Status

**Phase 1 Complete — Functional Prototype**

Environment standardization under WSL2 with NVIDIA GPU drivers  
CMake build system linking C++ host and CUDA device code  
Basic GPU kernel for **RGB → Grayscale conversion**  
Host pipeline for image load → GPU compute → result write

Phase 2: Box Blur (Convolution) — *in progress*  
Phase 3: Shared Memory Optimization — *planned*

You can currently run:

```bash
./lumina_app input.jpg output.jpg
