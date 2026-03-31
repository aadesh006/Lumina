#include <iostream>
#include <opencv2/opencv.hpp>
#include "kernels.cuh"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./lumina_app <video_file.mp4>" << std::endl;
        return -1;
    }

    // Open the Input Video
    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file. Check the path!" << std::endl;
        return -1;
    }

    // Get video metadata
    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    int totalFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);

    std::cout << "--- Lumina Video Engine ---" << std::endl;
    std::cout << "Resolution: " << width << "x" << height << " @ " << fps << " FPS" << std::endl;
    std::cout << "Total Frames: " << totalFrames << std::endl;

    // Create the Output Video Writer
    cv::VideoWriter writer("blurred_output.mp4", cv::VideoWriter::fourcc('m','p','4','v'), fps, cv::Size(width, height), false);

    // Matrix objects to hold our image data in CPU RAM
    cv::Mat frame, grayFrame;
    
    // pre-allocate the output frame to hold the data coming back from the GPU
    cv::Mat blurredFrame(height, width, CV_8UC1); 

    int frameCount = 0;

    // The Main Processing Loop
    while (cap.read(frame)) {
        // Convert the BGR video frame to Grayscale
        cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

        launchBlurKernel(grayFrame.data, blurredFrame.data, width, height, 3);

        // processed frame to the new video file
        writer.write(blurredFrame);

        frameCount++;
        
        if (frameCount % 10 == 0) {
            std::cout << "Processed " << frameCount << " / " << totalFrames << " frames...\r" << std::flush;
        }
    }

    std::cout << "\nDone! Successfully rendered 'blurred_output.mp4'" << std::endl;

    // Clean up
    cap.release();
    writer.release();
    
    return 0;
}