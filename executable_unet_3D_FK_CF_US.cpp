/**
 * @file executable_unet.cpp
 * @author sulis347@gmail.com
 * @brief this is 3D reconstruction + envelope + FK-filter + CF + MIP + US
 * @version 0.1
 * @date 2025-01-20
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#include <iostream>
#include <memory>
#include <stdexcept>  // for std::exception>
#include <cmath>
#include <filesystem>
#include <string>
#include <regex>
#include <chrono>

// required libraries
#include <opencv2/opencv.hpp>
#include <torch/script.h> // For TorchScript
#include <torch/torch.h> // For TorchScript
#include <torch/fft.h> 

// custom
#include "utils/vrs_loader.h"
#include "utils/INIReader.h"


std::string getLatestFile(const std::string& folderPath) {
    std::string latestFile;
    int maxIndex = -1;

    // Regular expression to extract the number from filenames like TestFile000.vrs
    std::regex filenamePattern("TestFile(\\d+).vrs");

    // Iterate through all files in the folder
    for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
        const auto& filename = entry.path().filename().string();
        
        // Check if the filename matches the pattern
        std::smatch match;
        if (std::regex_match(filename, match, filenamePattern)) {
            int index = std::stoi(match[1].str());  // Convert the matched number to integer

            // Update the latest file if the current one has a larger index
            if (index > maxIndex) {
                maxIndex = index;
                latestFile = entry.path().string();  // Get the full path
            }
        }
    }

    return latestFile;
}

int getLatestFileIndex(const std::string& folderPath) {
    std::string latestFile;
    int maxIndex = -1;

    // Regular expression to extract the number from filenames like TestFile000.vrs
    std::regex filenamePattern("TestFile(\\d+).vrs");

    // Iterate through all files in the folder
    for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
        const auto& filename = entry.path().filename().string();
        
        // Check if the filename matches the pattern
        std::smatch match;
        if (std::regex_match(filename, match, filenamePattern)) {
            int index = std::stoi(match[1].str());  // Convert the matched number to integer

            // Update the latest file if the current one has a larger index
            if (index > maxIndex) {
                maxIndex = index;
                latestFile = entry.path().string();  // Get the full path
            }
        }
    }

    return maxIndex;
}


cv::Mat applyColormapWithIntensity(const torch::Tensor& depth_normalized, const torch::Tensor& intensity_normalized) {
    // Convert tensors to CPU for OpenCV
    auto depth_normalized_cpu = depth_normalized.to(torch::kCPU);
    auto intensity_normalized_cpu = intensity_normalized.to(torch::kCPU);

    // Convert tensors to OpenCV Mat
    cv::Mat depth_normalized_cv(depth_normalized_cpu.size(0), depth_normalized_cpu.size(1), CV_32F, depth_normalized_cpu.data_ptr());
    cv::Mat intensity_normalized_cv(intensity_normalized_cpu.size(0), intensity_normalized_cpu.size(1), CV_32F, intensity_normalized_cpu.data_ptr());

    // Convert to 8-bit for colormap application
    depth_normalized_cv.convertTo(depth_normalized_cv, CV_8U, 255); // Scale to [0, 255]
    intensity_normalized_cv.convertTo(intensity_normalized_cv, CV_8U, 255); // Scale to [0, 255]

    // Apply the 'inferno' colormap
    cv::Mat depth_colored;
    cv::applyColorMap(depth_normalized_cv, depth_colored, cv::COLORMAP_INFERNO);

    // Scale the colormap by intensity
    cv::Mat intensity_normalized_color;
    cv::cvtColor(intensity_normalized_cv, intensity_normalized_color, cv::COLOR_GRAY2BGR); // Convert grayscale to 3 channels
    cv::Mat final_colored;
    cv::multiply(depth_colored, intensity_normalized_color, final_colored, 1.0 / 255.0); // Element-wise scaling

    return final_colored;
}

int main(int argc, char* argv[]) {

    INIReader reader(argv[1]);
    if (reader.ParseError() < 0) {
        std::cout << "Can't load 'config.ini'\n";
        return 1;
    }

    int nt = reader.GetInteger("general", "nt", 3072);
    int ntx = reader.GetInteger("general", "ntx", 58);
    bool verbose = reader.GetBoolean("general", "verbose", false);
    int dim3_start = reader.GetInteger("general", "dim3_start", 1144);
    int dim3_stop = reader.GetInteger("general", "dim3_stop", 1400);
    int imsz = reader.GetInteger("general", "imsz", 128);
    int idx   = reader.GetInteger("general", "slice", 50);
    int idxUS = reader.GetInteger("general", "sliceUS", 1);
    int slice3dup = reader.GetInteger("general", "slice3dup", 0);
    int slice3ddown = reader.GetInteger("general", "slice3ddown", 64);
    double c     = reader.GetReal("general", "cwater", 1475);
    double Fs  = reader.GetReal("general", "Fs", 62.5e6);
    double rs  = reader.GetReal("general", "rs", 1.5); // in mm
    int sleepdur = reader.GetInteger("general", "sleep_duration", 10);
    std::string model_path = reader.Get("general", "model_path", "../../model_traced.pt");
    std::string pathName = reader.Get("general", "pathName", "../../");
    std::string distpath3d = reader.Get("general", "distpath3d", "../../utils/pre_computed_distance/c1475_3D_64.pt");
    std::string distpathUS = reader.Get("general", "distpathUS", "../../utils/pre_computed_distance/USZslice_c1475_64.pt");
    
    // if (model_path.empty()) {
    //     std::cerr << "Model path is empty!" << std::endl;
    // } else {
    //     std::cout << "Model path: " << model_path << std::endl;
    // }
    std::cout << "Raw model path: " << model_path << std::endl; // Check the raw value
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;


    /////////////////////////////////////////////////////////////////////////// load model
    std::cout << "PyTorch Version: " << TORCH_VERSION << std::endl;
    std::cout << "Loading Model ..." << std::endl;
    torch::jit::script::Module model = torch::jit::load(model_path);

    if (torch::cuda::is_available()){
        model.to(torch::kCUDA);
        std::cout << "CUDA exist" << std::endl;
    }
    /////////////////////////////////////////////////////////////////////////// load distance matrix
    // US
    auto moduleUS = torch::jit::load(distpathUS);
    torch::Tensor distmatUS = moduleUS.attr("tensor").toTensor();
    if (torch::cuda::is_available()){
        distmatUS = distmatUS.to(torch::kCUDA);
    }
    distmatUS = distmatUS.to(torch::kLong);
    verbose ? std::cout << "US " << distmatUS.sizes() << std::endl : void();

    // PA
    auto module3d = torch::jit::load(distpath3d);
    torch::Tensor distmat3d = module3d.attr("tensor").toTensor();
    if (torch::cuda::is_available()){
        distmat3d = distmat3d.to(torch::kCUDA);
    }
    distmat3d = torch::transpose(distmat3d,1,0).to(torch::kLong);
    verbose ? std::cout << distmat3d.sizes() << std::endl : void();


    cv::namedWindow("US", cv::WINDOW_NORMAL);
    cv::resizeWindow("US", 400, 400);
    cv::namedWindow("XZ", cv::WINDOW_NORMAL);
    cv::resizeWindow("XZ", 400, 400);
    cv::namedWindow("YZ", cv::WINDOW_NORMAL);
    cv::resizeWindow("YZ", 400, 400);
    cv::namedWindow("MIPZ", cv::WINDOW_NORMAL);
    cv::resizeWindow("MIPZ", 400, 400);
    cv::namedWindow("DEPTH", cv::WINDOW_NORMAL);
    cv::resizeWindow("DEPTH", 400, 400);
    cv::namedWindow("RAW", cv::WINDOW_NORMAL);
    cv::resizeWindow("RAW", 200, 400);

    /////////////////////////////////////////////////////////////////////////// load vrs
    while (true){
        start = std::chrono::high_resolution_clock::now();
        // try to load the data

        torch::Tensor tensor;
        std::string fileName;
        int fileNameIndex;

        try{
            fileName = getLatestFile(pathName);
            fileNameIndex = getLatestFileIndex(pathName);
            verbose ? std::cout << "Loading Data ..." << fileName << std::endl : void();
            tensor = load_vrs_torch(fileName, nt, ntx, verbose);
        }catch(...){
            fileNameIndex = -1;
            verbose ? std::cout << "Loading error, set all zeros" << fileName << std::endl : void();
            tensor = torch::zeros({256,ntx,nt});
        }

        if (torch::cuda::is_available()){
            tensor = tensor.to(torch::kCUDA);
        }
        //============================================================= US processing
        // compute depth, use the last sensor
        verbose ? std::cout <<"Flag"<< std::endl : void();
        torch::Tensor usRF = tensor.index({255, idxUS, torch::indexing::Slice(1000, nt-1)});
        verbose ? std::cout <<"Flag"<< std::endl : void();
        usRF = usRF.abs();
        auto [maxRF, argmaxRF] = usRF.max(0);
        double argmaxRFscalar = argmaxRF.item<double>() ;
        
        verbose ? std::cout <<"Sizes : " << usRF.sizes() << std::endl : void();
        double depth = 30.0 - ((argmaxRFscalar + 1000.0) / 2.0) * c * 1e3 / Fs; // in mm
        verbose ? std::cout << depth << std::endl : void();
        // get even data, shape is (256, Ndata, 3072), so we select the [0, 1, 3, 4, etc]
        // tensor is (256, 30, 3072) --> get (256, 20, 3072) where third element is removed
        // it will give you 20 data
        // change the 20 and % 2 as the structure of the data
        torch::Tensor indices = torch::arange(30, torch::kInt64).index(
            {torch::arange(30, torch::kInt64) % 3 != 2}
        );
        verbose ? std::cout << "indices " << indices.sizes() << std::endl : void();
    
        // Index the tensor along the second dimension, should be (256, 20, 3072)
        torch::Tensor usonly = tensor.index({torch::indexing::Slice(), indices, torch::indexing::Slice()});
        verbose ? std::cout << "usonly " << usonly.sizes() << usonly.min() << usonly.max() << std::endl : void();
        // hilbert
        torch::Tensor fftvalUS = torch::fft::fft(usonly); // by default along last dimension
        fftvalUS.narrow(2, nt/2, nt/2).zero_(); // Zero out the negative frequencies (right half)
        torch::Tensor hilbertUS = torch::fft::ifft(fftvalUS);
        verbose ? std::cout << "hilbertUS " << hilbertUS.sizes() << hilbertUS.device() << std::endl : void();

        // beamform
        verbose ? std::cout << "distmatUS " << distmatUS.device() << std::endl : void();
        torch::Tensor usslice = torch::gather(hilbertUS, 2, distmatUS); 
        verbose ? std::cout << "usslice " << usslice.sizes() << std::endl : void();
        usslice = usslice.sum(0);
        verbose ? std::cout << "flag sum " << std::endl : void();
        usslice = usslice.sum(0);
        verbose ? std::cout << "flag sum 2" << std::endl : void();
        usslice = usslice.view({imsz,imsz});
        usslice = usslice.abs();
        verbose ? std::cout << "flag sum 2 " << usslice.max() << " " << usslice.min() << std::endl : void();
        try{
            usslice = usslice / usslice.max();
            usslice = 20 * torch::log10(usslice);
            verbose ? std::cout << "LogComp " << usslice.max() << " " << usslice.min() << std::endl : void();
            usslice = torch::clamp(usslice, -60, 0);
            verbose ? std::cout << "LogComp " << usslice.max() << " " << usslice.min() << std::endl : void();
            usslice = usslice / -60; // this is for visualization
            usslice = 1 - usslice;
        }catch(...){
            usslice = usslice;
        }
        verbose ? std::cout << "flag reshape" << usslice.sizes() << std::endl : void();

        usslice = usslice.to(torch::kCPU).contiguous();
        verbose ? std::cout << "flag" << std::endl : void();
        cv::Mat imageUS(usslice.size(0), usslice.size(1), CV_32F, usslice.data_ptr<float>());
        verbose ? std::cout << "flag" << std::endl : void();
        cv::Mat displaySliceUS;
        verbose ? std::cout << "flag" << std::endl : void();
        imageUS.convertTo(displaySliceUS, CV_8U, 255.0);
        verbose ? std::cout << "flag" << std::endl : void();
        cv::Mat rotatedImageUS;
        verbose ? std::cout << "flag" << std::endl : void();
        cv::rotate(displaySliceUS, rotatedImageUS, cv::ROTATE_90_COUNTERCLOCKWISE);
        verbose ? std::cout << "flag" << std::endl : void();
        cv::imshow("US", rotatedImageUS);
        verbose ? std::cout << "US end" << std::endl : void();
        //============================================================= PA processing
        // get one slice
        tensor = tensor.slice(1, idx, idx+1);
        verbose ? std::cout <<"Sizes : " << tensor.sizes() << std::endl : void();

        tensor = tensor.to(torch::kFloat);
        verbose ? std::cout <<"Sizes : " << tensor.sizes() << std::endl: void();
        
        tensor = tensor.slice(2, dim3_start, dim3_stop);
        verbose ? std::cout <<"Sizes : " << tensor.sizes() << std::endl: void();
        
        tensor = tensor.unsqueeze(0);
        verbose ? std::cout <<"Sizes : " << tensor.sizes() << std::endl: void();

        torch::Tensor demo = tensor.permute({0,2,1,3});
        verbose ? std::cout <<"Sizes : " << demo.sizes() << std::endl: void();
        ///////////////////////////////////////////////////////////////////////// fkfilter
        demo = demo.squeeze(0).squeeze(0);
        int quarter_size = demo.size(0) / 4;

        for (int i = 0; i < 4; ++i) {
            int start_row = i * quarter_size;
            int end_row = (i + 1) * quarter_size;

            // Slice the tensor for the current quarter
            torch::Tensor quarter = demo.slice(0, start_row, end_row);

            // FFT process on the quarter
            torch::Tensor fftval2 = torch::fft::fft2(quarter);
            fftval2.narrow(0, 0, 1).zero_(); // Remove the central frequency
            quarter = torch::real(torch::fft::ifft2(fftval2));

            // Assign the processed quarter back to the original tensor
            demo.slice(0, start_row, end_row).copy_(quarter);
        }
        demo = demo.unsqueeze(0).unsqueeze(0);

        ///////////////////////////////////////////////////////////////////////// Interpolation 
        verbose ? std::cout << "Bilinear interpolation ..." << std::endl : void();
        torch::Tensor interpolated_tensor = torch::nn::functional::interpolate(
            demo,
            torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{512,256})
            .mode(torch::kBilinear)
            .align_corners(false)
        );
        verbose ? std::cout <<"Sizes : " << interpolated_tensor.sizes() << std::endl: void();
        
        verbose ? std::cout << "after interpolation" << std::endl : void();

        ///////////////////////////////////////////////////////////////////////// Inference
        verbose ? std::cout << "Inference ..." << std::endl : void();
        // Run the model on the input tensor
        interpolated_tensor = interpolated_tensor / interpolated_tensor.max();
        at::Tensor output_raw = model.forward({interpolated_tensor}).toTensor();
        output_raw = output_raw.squeeze();
        output_raw = output_raw.narrow(0, 0, output_raw.size(0) - 1); // removes last row

        ///////////////////////////////////////////////////////////////////////// pabeamforming

        /// hilbert transform
        torch::Tensor fftval = torch::fft::fft(output_raw); // by default along last dimension
        fftval.narrow(1, 128, 128).zero_(); // Zero out the negative frequencies (right half)
        torch::Tensor output = torch::fft::ifft(fftval);


        verbose ? std::cout << "Beamforming ..." << std::endl : void();

        torch::Tensor gathered3d = torch::gather(output, 1, distmat3d);  // Gather along columns
        
        torch::Tensor summed3d      = gathered3d.sum(0);  // Shape will be (511,)
        torch::Tensor summed3dabs   = gathered3d.abs().sum(0);  // Shape will be (511,)

        gathered3d.reset();

        torch::Tensor beamformed3d      = summed3d.view({imsz,imsz,imsz});
        torch::Tensor beamformed3dabs   = summed3dabs.view({imsz,imsz,imsz});

        beamformed3d = beamformed3d.abs();
        beamformed3d = beamformed3d * beamformed3d * beamformed3d / (511 * beamformed3dabs);

        torch::Tensor beamformedx = beamformed3d.index({imsz/2, torch::indexing::Slice(), torch::indexing::Slice()});
        torch::Tensor beamformedy = beamformed3d.index({torch::indexing::Slice(), imsz/2, torch::indexing::Slice()});
        // torch::Tensor beamformedz = beamformed3d.index({torch::indexing::Slice(), torch::indexing::Slice(), imsz/2});
        torch::Tensor mipz = std::get<0>(beamformed3d.max(2)); // dim 2 corresponds to z-dimension

        try{
            // catch if all beamform is zero 
            beamformedx = (beamformedx - beamformedx.min()) / (beamformedx.max() - beamformedx.min());
            beamformedy = (beamformedy - beamformedy.min()) / (beamformedy.max() - beamformedy.min());
            // beamformedz = (beamformedz - beamformedz.min()) / (beamformedz.max() - beamformedz.min());
            mipz = (mipz - mipz.min()) / (mipz.max() - mipz.min());
        }catch(...){
            beamformedx = beamformedx;
            beamformedy = beamformedy;
            // beamformedz = beamformedz;
            mipz = mipz;
        }
        beamformedx = beamformedx.to(torch::kCPU).contiguous();
        beamformedy = beamformedy.to(torch::kCPU).contiguous();
        // beamformedz = beamformedz.to(torch::kCPU).contiguous();
        mipz = mipz.to(torch::kCPU).contiguous();
        
        ///////////////////////////////////////////////////////////////////////// time 
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

        ///////////////////////////////////////////////////////////////////////// visualize 
        // compute depth
        int depth_idx = std::floor(depth / (2.0 * rs / double(imsz)));
        verbose ? std::cout << depth_idx << std::endl : void();
        int depth_cv = imsz/2 - depth_idx;
        verbose ? std::cout << depth_cv << std::endl : void();
        if (depth_cv < 0){
            depth_cv = 0;
        }else{
            if (depth_cv > imsz){
                depth_cv = imsz-1;
            }
        }

        verbose ? std::cout << "View ..." << std::endl : void();
        cv::Mat imagex(beamformedx.size(0), beamformedx.size(1), CV_32F, beamformedx.data_ptr<float>());
        cv::Mat displaySlicex;
        imagex.convertTo(displaySlicex, CV_8U, 255.0);
        cv::Mat colorMappedx;
        cv::applyColorMap(displaySlicex, colorMappedx, cv::COLORMAP_HOT);
        cv::Mat rotatedImagex;
        cv::rotate(colorMappedx, rotatedImagex, cv::ROTATE_90_COUNTERCLOCKWISE);
        cv::line(rotatedImagex, cv::Point(0, depth_cv), cv::Point(rotatedImagex.cols, depth_cv), cv::Scalar(0, 255, 0), 1);  // Black lines with thickness 2
        cv::imshow("YZ", rotatedImagex);

        cv::Mat imagey(beamformedy.size(0), beamformedy.size(1), CV_32F, beamformedy.data_ptr<float>());
        cv::Mat displaySlicey;
        imagey.convertTo(displaySlicey, CV_8U, 255.0);
        cv::Mat colorMappedy;
        cv::applyColorMap(displaySlicey, colorMappedy, cv::COLORMAP_HOT);
        cv::Mat rotatedImagey;
        cv::rotate(colorMappedy, rotatedImagey, cv::ROTATE_90_COUNTERCLOCKWISE);
        cv::line(rotatedImagey, cv::Point(0, depth_cv), cv::Point(rotatedImagex.cols, depth_cv), cv::Scalar(0, 255, 0), 1);  // Black lines with thickness 2
        cv::imshow("XZ", rotatedImagey);

        cv::Mat imagez(mipz.size(0), mipz.size(1), CV_32F, mipz.data_ptr<float>());
        cv::Mat displaySlicez;
        imagez.convertTo(displaySlicez, CV_8U, 255.0);
        cv::Mat colorMappedz;
        cv::applyColorMap(displaySlicez, colorMappedz, cv::COLORMAP_HOT);
        cv::Mat rotatedImagez;
        cv::rotate(colorMappedz, rotatedImagez, cv::ROTATE_90_COUNTERCLOCKWISE);
        std::string text = std::to_string(fileNameIndex);
        int imageHeight = rotatedImagez.rows;
        cv::Point position(1, imageHeight-1); // 10px from left, 10px from bottom
        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 0.3;
        cv::Scalar textColor(0, 255, 0);  // Green color
        int thickness = 1;
        cv::putText(rotatedImagez, text, position, fontFace, fontScale, textColor, thickness);
        cv::imshow("MIPZ", rotatedImagez);

        torch::Tensor view_raw = output_raw.to(torch::kCPU).contiguous();
        view_raw = (view_raw - view_raw.min()) / (view_raw.max() - view_raw.min());
        cv::Mat imageraw(view_raw.size(0), view_raw.size(1), CV_32F, view_raw.data_ptr<float>());
        cv::Mat displayRaw;
        imageraw.convertTo(displayRaw, CV_8U, 255.0);
        cv::imshow("RAW", imageraw);

        ////////////////////////// view 3D
        // slice the beamformed3d
        beamformed3d = beamformed3d.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(slice3dup,slice3ddown)});
        torch::Tensor depth_map = std::get<1>(torch::max(beamformed3d, 2)); // Argmax along depth axis
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min());
        torch::Tensor intensity_map = torch::amax(beamformed3d, 2); // Max along depth axis
        intensity_map = (intensity_map - intensity_map.min()) / (intensity_map.max() - intensity_map.min());
        cv::Mat final_colored = applyColormapWithIntensity(depth_map, intensity_map);
        cv::Mat rotatedImage3d;
        cv::rotate(final_colored, rotatedImage3d, cv::ROTATE_90_COUNTERCLOCKWISE);
        cv::imshow("DEPTH", rotatedImage3d);

        cv::waitKey(10);

        std::this_thread::sleep_for(std::chrono::microseconds(sleepdur));
        
    }
    return 0;
}
