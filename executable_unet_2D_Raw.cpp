/**
 * @file executable_unet_2D.cpp
 * @author sulis347@gmail.com
 * @brief this is 2D reconstruction
 * @version 0.1
 * @date 2025-01-14
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

int main(int argc, char* argv[]) {
    /////////////////////////////////////////////////////////////////////////// important parameters
    // int nt          = 3072;  // Default 4096 if not provided
    // int ntx         = 58;  // Default 28 if not provided
    // bool verbose    = false;  // Default false if not provided
    // // int dim1_start  = 0;
    // // int dim1_stop   = 256;
    // int dim3_start  = 1144;
    // int dim3_stop   = 1400;
    // int imsz        = 128;
    // std::string model_path = "../../model_traced.pt";
    // std::string fileName   = "../../22102024.vrs";
    // std::string distpath = "../../utils/pre_computed_distance/c1475_z-0.40.pt";
    
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
    double c     = reader.GetReal("general", "cwater", 1475);
    double Fs  = reader.GetReal("general", "Fs", 62.5e6);
    double rs  = reader.GetReal("general", "rs", 1.5); // in mm
    int sleepdur = reader.GetInteger("general", "sleep_duration", 10);
    std::string model_path = reader.Get("general", "model_path", "../../model_traced.pt");
    std::string pathName = reader.Get("general", "pathName", "../../");
    std::string distpathx = reader.Get("general", "distpathx", "../../utils/pre_computed_distance/c1475_x0.00.pt");
    std::string distpathy = reader.Get("general", "distpathy", "../../utils/pre_computed_distance/c1475_y0.00.pt");
    std::string distpathz = reader.Get("general", "distpathz", "../../utils/pre_computed_distance/c1475_z0.00.pt");

    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;


    /////////////////////////////////////////////////////////////////////////// load model
    std::cout << "PyTorch Version: " << TORCH_VERSION << std::endl;
    std::cout << "Loading Model ..." << std::endl;
    torch::jit::script::Module model = torch::jit::load(model_path);

    if (torch::cuda::is_available()){
        model.to(torch::kCUDA);
    }
    /////////////////////////////////////////////////////////////////////////// load distance matrix
    auto modulex = torch::jit::load(distpathx);
    auto moduley = torch::jit::load(distpathy);
    auto modulez = torch::jit::load(distpathz);

    torch::Tensor distmatx = modulex.attr("tensor").toTensor();
    torch::Tensor distmaty = moduley.attr("tensor").toTensor();
    torch::Tensor distmatz = modulez.attr("tensor").toTensor();

    if (torch::cuda::is_available()){
        distmatx = distmatx.to(torch::kCUDA);
        distmaty = distmaty.to(torch::kCUDA);
        distmatz = distmatz.to(torch::kCUDA);
    }
    distmatx = torch::transpose(distmatx,1,0).to(torch::kLong);
    distmaty = torch::transpose(distmaty,1,0).to(torch::kLong);
    distmatz = torch::transpose(distmatz,1,0).to(torch::kLong);
    
    verbose ? std::cout << distmatx.size(0) << " " << distmatx.size(1) << std::endl : void();
    verbose ? std::cout << distmaty.size(0) << " " << distmaty.size(1) << std::endl : void();
    verbose ? std::cout << distmatz.size(0) << " " << distmatz.size(1) << std::endl : void();


    cv::namedWindow("XY", cv::WINDOW_NORMAL);
    cv::resizeWindow("XY", 400, 400);
    cv::namedWindow("XZ", cv::WINDOW_NORMAL);
    cv::resizeWindow("XZ", 400, 400);
    cv::namedWindow("YZ", cv::WINDOW_NORMAL);
    cv::resizeWindow("YZ", 400, 400);

    /////////////////////////////////////////////////////////////////////////// load vrs
    while (true){
        start = std::chrono::high_resolution_clock::now();
        // try to load the data

        torch::Tensor tensor;
        std::string fileName = getLatestFile(pathName);
        verbose ? std::cout << "Loading Data ..." << fileName << std::endl : void();
        tensor = load_vrs_torch(fileName, nt, ntx, verbose);
        if (torch::cuda::is_available()){
            tensor = tensor.to(torch::kCUDA);
        }

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
        at::Tensor output = model.forward({interpolated_tensor}).toTensor();
        output = output.squeeze();
        output = output.narrow(0, 0, output.size(0) - 1); // removes last row

        ///////////////////////////////////////////////////////////////////////// beamforming
        verbose ? std::cout << "Beamforming ..." << std::endl : void();

        torch::Tensor gatheredx = torch::gather(output, 1, distmatx);  // Gather along columns
        torch::Tensor gatheredy = torch::gather(output, 1, distmaty);  // Gather along columns
        torch::Tensor gatheredz = torch::gather(output, 1, distmatz);  // Gather along columns
        
        torch::Tensor summedx = gatheredx.sum(0);  // Shape will be (511,)
        torch::Tensor summedy = gatheredy.sum(0);  // Shape will be (511,)
        torch::Tensor summedz = gatheredz.sum(0);  // Shape will be (511,)

        torch::Tensor beamformedx = -summedx.view({imsz,imsz});
        torch::Tensor beamformedy = -summedy.view({imsz,imsz});
        torch::Tensor beamformedz = -summedz.view({imsz,imsz});

        try{
            // catch if all beamform is zero 
            beamformedx = (beamformedx - beamformedx.min()) / (beamformedx.max() - beamformedx.min());
            beamformedy = (beamformedy - beamformedy.min()) / (beamformedy.max() - beamformedy.min());
            beamformedz = (beamformedz - beamformedz.min()) / (beamformedz.max() - beamformedz.min());
        }catch(...){
            beamformedx = beamformedx;
            beamformedy = beamformedy;
            beamformedz = beamformedz;
        }
        beamformedx = beamformedx.to(torch::kCPU).contiguous();
        beamformedy = beamformedy.to(torch::kCPU).contiguous();
        beamformedz = beamformedz.to(torch::kCPU).contiguous();
        
        ///////////////////////////////////////////////////////////////////////// time 
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

        ///////////////////////////////////////////////////////////////////////// visualize 
        // compute depth
        int depth_idx = std::floor(depth / (2.0 * rs / double(imsz)));
        verbose ? std::cout << depth_idx << std::endl : void();
        int depth_cv = 64 - depth_idx;
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
        cv::line(rotatedImagex, cv::Point(0, depth_cv), cv::Point(rotatedImagex.cols, depth_cv), cv::Scalar(0, 0, 0), 1);  // Black lines with thickness 2
        cv::imshow("YZ", rotatedImagex);

        cv::Mat imagey(beamformedy.size(0), beamformedy.size(1), CV_32F, beamformedy.data_ptr<float>());
        cv::Mat displaySlicey;
        imagey.convertTo(displaySlicey, CV_8U, 255.0);
        cv::Mat colorMappedy;
        cv::applyColorMap(displaySlicey, colorMappedy, cv::COLORMAP_HOT);
        cv::Mat rotatedImagey;
        cv::rotate(colorMappedy, rotatedImagey, cv::ROTATE_90_COUNTERCLOCKWISE);
        cv::line(rotatedImagey, cv::Point(0, depth_cv), cv::Point(rotatedImagex.cols, depth_cv), cv::Scalar(0, 0, 0), 1);  // Black lines with thickness 2
        cv::imshow("XZ", rotatedImagey);

        cv::Mat imagez(beamformedz.size(0), beamformedz.size(1), CV_32F, beamformedz.data_ptr<float>());
        cv::Mat displaySlicez;
        imagez.convertTo(displaySlicez, CV_8U, 255.0);
        cv::Mat colorMappedz;
        cv::applyColorMap(displaySlicez, colorMappedz, cv::COLORMAP_HOT);
        cv::Mat rotatedImagez;
        cv::rotate(colorMappedz, rotatedImagez, cv::ROTATE_90_COUNTERCLOCKWISE);
        cv::imshow("XY", rotatedImagez);

        cv::waitKey(10);

        std::this_thread::sleep_for(std::chrono::microseconds(sleepdur));
        
    }
    return 0;
}
