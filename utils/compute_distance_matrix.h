#ifndef COMPUTE_DISTANCE_MATRIX_H
#define COMPUTE_DISTANCE_MATRIX_H

#include <torch/script.h> // For TorchScript
#include <torch/torch.h> // For TorchScript
#include <torch/fft.h> 
#include <cmath>

torch::Tensor create_time_dist_PA(
    torch::Tensor sensor_pos,
    double r,
    int imsz,
    double Fs,
    double cPA,
    int start_time_PA
);

torch::Tensor create_time_dist_US(
    torch::Tensor sensor_pos,
    torch::Tensor UseTXElements,
    int dim,
    double r,
    int imsz,
    double Fs,
    double cUS
);

#endif