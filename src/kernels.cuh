#pragma once

void launchGrayscaleKernel(const unsigned char* h_input, unsigned char* h_output, int width, int height);

void launchBlurKernel(const unsigned char* h_input, unsigned char* h_output, int width, int height, int blurSize);