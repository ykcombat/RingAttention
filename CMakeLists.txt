cmake_minimum_required(VERSION 3.22)
project(cutlass CUDA CXX)

# set environment PATH for cmake
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
set(CMAKE_COLOR_DIAGNOSTICS ON)

find_package(CUDAToolkit REQUIRED)
include_directories(
    "deps/cutlass/include"
    "include"
    "/home/whcui/nvshmem_src_3.0.6-4/install/include"
)
link_directories("/home/whcui/nvshmem_src_3.0.6-4/install/lib")

file(GLOB CUDA_SOURCE_FILES "./csrc/*.cu")
foreach(CUDA_SOURCE_FILE ${CUDA_SOURCE_FILES})
    # NOTE: NAME_WE: name without extension
    # Extract the filename ${CUDA_SOURCE_FILE} without the extension to EXECUTABLE_NAME
    get_filename_component(EXECUTABLE_NAME ${CUDA_SOURCE_FILE} NAME_WE)

    # Create an executable for each source file
    add_executable(${EXECUTABLE_NAME} ${CUDA_SOURCE_FILE})
    set_target_properties(${EXECUTABLE_NAME} PROPERTIES CXX_STANDARD 17 CUDA_ARCHITECTURES 80 )
    set_target_properties(${EXECUTABLE_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    set_target_properties(${EXECUTABLE_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(${EXECUTABLE_NAME} -lnvshmem)

    # target_compile_options(${EXECUTABLE_NAME} PRIVATE -G -g)
    target_compile_options(${EXECUTABLE_NAME} PRIVATE -O3 -lineinfo)
endforeach()
