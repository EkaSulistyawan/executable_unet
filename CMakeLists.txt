cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(executable_unet)

# Define the paths as variables
set(LIBTORCH_PATH "C:/Users/Eka Sulistyawan/Documents/GitHub/cpptorch/libtorch/share/cmake/Torch")
set(OpenCV_DIR "C:/opencv/build")  # Adjust this path to where OpenCV is installed
set(EIGEN3_INCLUDE_DIR "C:/Users/Eka Sulistyawan/Documents/GitHub/cpptorch/eigen-3.4.0")
# set(ANACONDA_TORCH_PATH "C:/Users/Eka Sulistyawan/anaconda3/Lib/site-packages/torch/share/cmake")

# Find OpenCV package
find_package(OpenCV REQUIRED)

# add Torch directory
set(CMAKE_PREFIX_PATH "${LIBTORCH_PATH}" ${CMAKE_PREFIX_PATH})
# set(CMAKE_PREFIX_PATH "${LIBTORCH_PATH};${ANACONDA_TORCH_PATH}" ${CMAKE_PREFIX_PATH})

find_package(Torch REQUIRED)
# Append the paths to CMAKE_PREFIX_PATH
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


# add the util directory
include_directories(${CMAKE_SOURCE_DIR}/utils)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
link_directories(${OpenCV_DIR}/x64/vc16/lib)  # Adjust this path based on your OpenCV version and configuration


# modify this to add more and more cpp
add_executable(executable_unet executable_unet.cpp utils/vrs_loader.cpp utils/beamforming.cpp utils/INIReader.cpp utils/ini.c)

target_link_libraries(executable_unet "${OpenCV_LIBS}" "${TORCH_LIBRARIES}")
set_property(TARGET executable_unet PROPERTY CXX_STANDARD 17)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET executable_unet
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:executable_unet>)
endif (MSVC)
