cmake_minimum_required(VERSION 3.18)
project(Lumina LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)

# Search OpenCV on the system
find_package(OpenCV REQUIRED)

add_executable(lumina_app
    src/main.cpp
    src/kernels.cu
)

#Link the OpenCV headers and binaries
target_include_directories(lumina_app PRIVATE include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(lumina_app PRIVATE cuda ${OpenCV_LIBS})