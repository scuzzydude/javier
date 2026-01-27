cmake_minimum_required(VERSION 3.24)

project(javier
  VERSION 0.1.0
  LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Good defaults
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# If you want a specific GPU arch, set it here.
# Common: 75 (Turing), 80/86 (Ampere), 89 (Ada), 90 (Hopper)
# You can also pass via preset cacheVariables.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

add_executable(javier
  src/main.cu
  src/flash_controller.cu
)

target_include_directories(javier PRIVATE include)

# Useful compile options (optional)
target_compile_options(javier PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

# Make separable compilation easy if you grow into multiple .cu files
set_target_properties(javier PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
