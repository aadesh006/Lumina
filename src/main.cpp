#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "kernels.cuh"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./lumina_app <video_file.mp4>" << std::endl;
        return -1;
    }

    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) return -1;

    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    size_t numBytes = width * height * sizeof(unsigned char);

    cv::VideoWriter writer("blurred_output.mp4", cv::VideoWriter::fourcc('m','p','4','v'), fps, cv::Size(width, height), false);

    std::cout << "--- Lumina Asynchronous Engine ---" << std::endl;

    // 1. PRE-ALLOCATION PHASE
    // A. Allocate Pinned Memory on the CPU (Page-Locked for fast PCIe transfers)
    unsigned char *h_pinned_in, *h_pinned_out;
    cudaMallocHost((void**)&h_pinned_in, numBytes);
    cudaMallocHost((void**)&h_pinned_out, numBytes);

    // B. Allocate VRAM on the GPU
    unsigned char *d_in, *d_out;
    cudaMalloc((void**)&d_in, numBytes);
    cudaMalloc((void**)&d_out, numBytes);

    // C. Create the CUDA Stream (Our custom assembly line lane)
    cudaStream_t stream1;
    cudaStreamCreate(&stream1);

    cv::Mat frame, grayFrame;
    cv::Mat blurredFrame(height, width, CV_8UC1, h_pinned_out); // Point OpenCV directly to our pinned output memory!
    int frameCount = 0;

    // 2. THE ASYNC VIDEO LOOP
    while (cap.read(frame)) {
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

        // 1. CPU copies OpenCV data into the Pinned Memory buffer
        std::memcpy(h_pinned_in, grayFrame.data, numBytes);

        // 2. ASYNC Transfer: CPU tells DMA controller to move data, then immediately moves to next line
        cudaMemcpyAsync(d_in, h_pinned_in, numBytes, cudaMemcpyHostToDevice, stream1);

        // 3. ASYNC Compute: Fire the kernel into the stream
        launchBlurKernelAsync(d_in, d_out, width, height, 3, stream1);

        // 4. ASYNC Transfer: Send data back to Pinned Memory
        cudaMemcpyAsync(h_pinned_out, d_out, numBytes, cudaMemcpyDeviceToHost, stream1);

        // 5. Sync Point: Wait for THIS specific frame to finish its round trip
        cudaStreamSynchronize(stream1);

        // 6. Write to hard drive
        writer.write(blurredFrame);

        frameCount++;
        if (frameCount % 10 == 0) std::cout << "Processed " << frameCount << " frames...\r" << std::flush;
    }

    std::cout << "\nDone! Cleaning up memory..." << std::endl;

    // 3. CLEANUP PHASE
    cudaStreamDestroy(stream1);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFreeHost(h_pinned_in);
    cudaFreeHost(h_pinned_out);
    
    cap.release();
    writer.release();
    return 0;
}