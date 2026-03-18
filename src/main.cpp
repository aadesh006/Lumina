#include <iostream>
#include <vector>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "kernels.cuh" 

int main(int argc, char* argv[]) {
    // Check if the user passed an image name
    if (argc < 2) {
        std::cerr << "Usage: ./lumina_app <path_to_image>" << std::endl;
        return -1;
    }

    std::string inputPath = argv[1];
    int width, height, channels;

    //Load Image
    std::cout << "Loading image: " << inputPath << "..." << std::endl;
    unsigned char* imgData = stbi_load(inputPath.c_str(), &width, &height, &channels, 1);

    if (!imgData) {
        std::cerr << "FAILED to load image. Check the file name and path!" << std::endl;
        return -1;
    }

    std::cout << "Success! Image is " << width << "x" << height << " pixels." << std::endl;

    //Prepare Output Memory
    std::vector<unsigned char> blurredData(width * height);

    //Launch GPU Blur (Radius 5 creates an 11x11 blur box)
    int blurRadius = 5;
    launchBlurKernel(imgData, blurredData.data(), width, height, blurRadius);

    std::string outputPath = "blurred_output.jpg";
    stbi_write_jpg(outputPath.c_str(), width, height, 1, blurredData.data(), 100);
    
    std::cout << "Done! Saved processed image to: " << outputPath << std::endl;

    //Clean up CPU memory
    stbi_image_free(imgData);

    return 0;
}