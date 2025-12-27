#include <stdio.h>

// Kernel function to add two vectors
__global__ void add(int n, float *x, float *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  
  // Use a grid-stride loop so we can handle arrays 
  // larger than the number of threads launched
  for (int i = index; i < n; i += stride)
      y[i] = x[i] + y[i];
}

int main(void) {
  int N = 1 << 20; // 1M elements
  float *x, *y;

  // 1. Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&x, N * sizeof(float));
  cudaMallocManaged(&y, N * sizeof(float));

  // 2. Initialize data on the CPU
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // 3. Run kernel on 1M elements on the GPU
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  add<<<numBlocks, blockSize>>>(N, x, y);

  // 4. Wait for GPU to finish before CPU accesses the memory
  // This is REQUIRED when using Unified Memory
  cudaDeviceSynchronize();

  // 5. Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    if (abs(y[i] - 3.0f) > maxError) maxError = abs(y[i] - 3.0f);
  
  printf("Max error: %f\n", maxError);

  // 6. Free memory
  cudaFree(x);
  cudaFree(y);

  return 0;
}