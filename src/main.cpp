#include <iostream>
#include <vector>

void runCudaProcess(const unsigned char* h_input, unsigned char* h_output, int width, int height);

int main() {
    int width = 4;
    int height = 4;
    
    // Create a tiny 4x4 image
    std::vector<unsigned char> h_input(width * height * 3);
    std::vector<unsigned char> h_output(width * height);

    // Color it RED
    for (int i = 0; i < width * height; i++) {
        h_input[i * 3 + 0] = 255; 
        h_input[i * 3 + 1] = 0;   
        h_input[i * 3 + 2] = 0;   
    }

    runCudaProcess(h_input.data(), h_output.data(), width, height);

    std::cout << "Input Pixel (Red): 255" << std::endl;
    std::cout << "Output Pixel (Gray): " << (int)h_output[0] << std::endl;
    
    return 0;
}