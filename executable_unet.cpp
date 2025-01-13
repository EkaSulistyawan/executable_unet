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
// #include <Eigen/Dense>

// custom
#include "utils/vrs_loader.h"
#include "utils/INIReader.h"
// #include "utils/beamforming.h"


// torch::Tensor convert_to_tensor(const std::vector<std::vector<std::vector<int16_t>>>& raw_data) {
//     // Flatten the 3D vector into a 1D vector of float (casting int16_t to float)
//     std::vector<float> flattened_data;
//     for (const auto& slice : raw_data) {
//         for (const auto& row : slice) {
//             flattened_data.insert(flattened_data.end(), row.begin(), row.end());
//         }
//     }

//     // Ensure that the dimensions are cast to int64_t to avoid narrowing conversion issues
//     torch::Tensor tensor = torch::from_blob(flattened_data.data(),
//         {static_cast<int64_t>(raw_data.size()), 
//          static_cast<int64_t>(raw_data[0].size()), 
//          static_cast<int64_t>(raw_data[0][0].size())},
//         torch::kFloat);

//     // Ensure the tensor is contiguous in memory
//     tensor = tensor.clone();
    
//     return tensor;
// }

torch::Tensor convert_to_tensor(const std::vector<std::vector<std::vector<int16_t>>>& raw_data) {
    // Calculate the total size of the flattened data
    size_t total_size = 0;
    for (const auto& slice : raw_data) {
        for (const auto& row : slice) {
            total_size += row.size();
        }
    }

    // Reserve space in a vector to store the flattened data
    std::vector<float> flattened_data;
    flattened_data.reserve(total_size);

    // Flatten the 3D vector into a 1D vector of float (casting int16_t to float)
    for (const auto& slice : raw_data) {
        for (const auto& row : slice) {
            flattened_data.insert(flattened_data.end(), row.begin(), row.end());
        }
    }

    // Convert the flattened vector directly to a torch tensor
    torch::Tensor tensor = torch::from_blob(flattened_data.data(),
        {static_cast<int64_t>(raw_data.size()), 
         static_cast<int64_t>(raw_data[0].size()), 
         static_cast<int64_t>(raw_data[0][0].size())},
        torch::kFloat);

    // Ensure the tensor is contiguous in memory
    tensor = tensor.clone();
    
    return tensor;
}

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
    int idx = reader.GetInteger("general", "slice", 50);
    std::string model_path = reader.Get("general", "model_path", "../../model_traced.pt");
    std::string pathName = reader.Get("general", "pathName", "../../");
    std::string distpath = reader.Get("general", "distpath", "../../utils/pre_computed_distance/c1475_x0.00.pt");

    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;


    /////////////////////////////////////////////////////////////////////////// load model
    std::cout << "PyTorch Version: " << TORCH_VERSION << std::endl;
    std::cout << "Loading Model ..." << std::endl;
    torch::jit::script::Module model = torch::jit::load(model_path);
    cv::namedWindow("Tensor Image", cv::WINDOW_NORMAL);
    cv::resizeWindow("Tensor Image", 200, 200);

    /////////////////////////////////////////////////////////////////////////// load vrs
    while (true){
        // try to load the data
        torch::Tensor tensor;
        try{
            std::string fileName = getLatestFile(pathName);
            std::cout << "Loading Data ..." << fileName << std::endl;

            start = std::chrono::high_resolution_clock::now();
            tensor = load_vrs_torch(fileName, nt, ntx, verbose);
            end = std::chrono::high_resolution_clock::now();
            elapsed = end - start;
            std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;

        }catch(...){
            std::cout << "Loading data failed, use all zeros" << std::endl;
            tensor = torch::zeros({256,ntx,nt});
        }

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
        std::cout << "Bilinear interpolation ..." << std::endl;
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
        std::cout << "Inference ..." << std::endl;
        // Run the model on the input tensor
        interpolated_tensor = interpolated_tensor / interpolated_tensor.max();
        at::Tensor output = model.forward({interpolated_tensor}).toTensor();
        output = output.squeeze();
        output = output.narrow(0, 0, output.size(0) - 1);

        verbose ? std::cout << "after inference" << std::endl: void();

        auto sizes = output.sizes();

        ///////////////////////////////////////////////////////////////////////// beamforming
        std::cout << "Beamforming ..." << std::endl;

        auto module = torch::jit::load(distpath);
        torch::Tensor distmat = module.attr("tensor").toTensor();

        distmat = torch::transpose(distmat,1,0).to(torch::kLong);
        verbose ? std::cout << distmat.size(0) << " " << distmat.size(1) << std::endl: void();

        torch::Tensor gathered = torch::gather(output, 1, distmat);  // Gather along columns
        torch::Tensor summed = gathered.sum(0);  // Shape will be (511,)
        torch::Tensor beamformed = -summed.view({imsz,imsz});
        beamformed = (beamformed - beamformed.min()) / (beamformed.max() - beamformed.min());
        beamformed = beamformed.to(torch::kCPU).contiguous();
        verbose ? std::cout << beamformed.sizes() << std::endl: void();

        ///////////////////////////////////////////////////////////////////////// visualize 
        std::cout << "View ..." << std::endl;
        cv::Mat image(beamformed.size(0), beamformed.size(1), CV_32F, beamformed.data_ptr<float>());
        // cv::Mat image(result.rows(), result.cols(), CV_64F, result.data()); // Use double (CV_64F) for Eigen::MatrixXd
        cv::Mat displaySlice;
        image.convertTo(displaySlice, CV_8U, 255.0); // Scale from [0, 1] to [0, 255

        // // Step 5: Display the image using OpenCV
        cv::Mat colorMapped;
        cv::applyColorMap(displaySlice, colorMapped, cv::COLORMAP_HOT);
        cv::Mat rotatedImage;
        cv::rotate(colorMapped, rotatedImage, cv::ROTATE_90_COUNTERCLOCKWISE);
        cv::imshow("Tensor Image", rotatedImage);
        cv::waitKey(10); // Wait for a key press before closing the window

        std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
    return 0;
}
