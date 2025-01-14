#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <torch/torch.h>

int main() {
    // Load an image using OpenCV
    cv::Mat img = cv::imread("your_image.jpg", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Could not open the image!" << std::endl;
        return -1;
    }

    // Convert the image to a tensor (1xH x W) where H and W are the image height and width
    torch::Tensor tensor = torch::from_blob(img.data, {img.rows, img.cols}, torch::kByte);

    // Print the size of the tensor
    std::cout << "Tensor size: " << tensor.sizes() << std::endl;

    // Convert the tensor to float (optional, depending on your needs)
    tensor = tensor.to(torch::kFloat);

    // Print tensor info (for debugging purposes)
    std::cout << "Tensor after conversion: " << tensor << std::endl;

    return 0;
}
