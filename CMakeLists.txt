cmake_minimum_required(VERSION 3.18)
project(Lumina LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(lumina_app
    src/main.cpp
    src/kernels.cu
)

target_include_directories(lumina_app PRIVATE include)

target_link_libraries(lumina_app PRIVATE cuda)