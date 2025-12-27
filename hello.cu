#include <stdio.h>

__global__ void helloFromGPU() {
    printf("Hello World from GPU thread %d!\n", threadIdx.x);
}

int main() {
    printf("Hello World from CPU!\n");

    // Launch the kernel with 1 block and 5 threads
    helloFromGPU<<<1, 5>>>();
        printf("Hello from GPU!");

    // Wait for GPU to finish before exiting
    cudaDeviceSynchronize();

    return 0;
}