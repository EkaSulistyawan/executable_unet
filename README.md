Hi!

# Setup
Follow  the step here,
1. OpenCV:
    - Install opencv, [here](https://opencv.org/releases/), the Windows one typically gives you pre-built ones. 
    - Copy the opencv folder to `C:`
    - **IMPORTANT** Add `C:\opencv\build\x64\vc16\bin` or wherever your `x64\vc16\bin` located to the system's PATH. Please make any changes on the CMakeLists accordingly.
2. LibTorch:
    - Download the source for libtorch [Here](https://pytorch.org/get-started/locally/). Please follow  your system's specs for the type. **IMPORTANT** make sure to download the Debug version as it will allow you to trace any error later.
    - Extract the `.zip` files
    - Copy the `libtorch` folder wherever you want and specify the location on the CMakeLists accordingly.
3. CMake:
    - Just download and install CMake [Here](https://cmake.org/download/)
    - Make folder named `build`

# Compilation & Execution
Compilation

```
cd build
cmake ..
cmake --build . --clean-first
```

Execution

```
cd Debug
executable_unet.exe <PATH TO config.ini>
```

# Notes
1. Only tested on windows
2. Set the data accordingly, it should have
    - sensor_pos and it's interpolated version,  this one exported from actual data. Should have dimension of (Nsensor, 3) for (x, y, z) coordinate of the sensors.
    - The data is Vantage's `vrs` format. We made `load_vrs` to stream the data and load it. If you have no `vrs` file, then please customize the load data.