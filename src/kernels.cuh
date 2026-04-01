#pragma once
#include <cuda_runtime.h>

void launchBlurKernelAsync(const unsigned char* d_input, unsigned char* d_output, int width, int height, int blurSize, cudaStream_t stream);