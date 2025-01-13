#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <cmath>
#include <fstream>

constexpr double r = 1.5; // Define your radius
constexpr int imsz = 128;
constexpr int sensor_count = 511;
constexpr double cPA = 1475; // Speed of sound
constexpr double Fs = 4 * 15.625e6; // Sampling frequency
constexpr double start_time = 1144;

// Load sensor positions from a text file
Eigen::MatrixXd loadSensorPositions(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    Eigen::MatrixXd sensor_pos(sensor_count, 3);
    for (int i = 0; i < sensor_count; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (!(file >> sensor_pos(i, j))) {
                throw std::runtime_error("Error reading file");
            }
        }
    }

    return sensor_pos;
}

// Create meshgrid for rix and riy
void createMeshgrid(Eigen::MatrixXd &rix, Eigen::MatrixXd &riy) {
    Eigen::VectorXd linspace = Eigen::VectorXd::LinSpaced(imsz, -r, r);
    rix.resize(imsz, imsz);
    riy.resize(imsz, imsz);

    for (int i = 0; i < imsz; ++i) {
        rix.row(i) = linspace;
        riy.col(i) = linspace;
    }
}

// Compute Euclidean distance
Eigen::MatrixXd computeDistances(const Eigen::MatrixXd &ri, const Eigen::MatrixXd &sensor_pos) {
    Eigen::MatrixXd distances(ri.rows(), sensor_pos.rows());
    for (int i = 0; i < ri.rows(); ++i) {
        for (int j = 0; j < sensor_pos.rows(); ++j) {
            distances(i, j) = (ri.row(i) - sensor_pos.row(j)).norm();
        }
    }
    return distances;
}

Eigen::MatrixXd recon_z(double zidx, const Eigen::MatrixXd &sensor_pos, const Eigen::MatrixXd &outp) {
    Eigen::MatrixXd rix, riy;
    createMeshgrid(rix, riy);

    std::cout<< "meshgrid construction" << std::endl;

    Eigen::MatrixXd riz = Eigen::MatrixXd::Ones(imsz, imsz) * zidx;
    Eigen::MatrixXd ri(imsz * imsz, 3);
    for (int i = 0; i < imsz; ++i) {
        for (int j = 0; j < imsz; ++j) {
            int idx = i * imsz + j;
            ri(idx, 0) = rix(i, j);
            ri(idx, 1) = riz(i, j);
            ri(idx, 2) = riy(i, j);
        }
    }

    std::cout<< "computeDistance" << std::endl;

    Eigen::MatrixXd phys_dist_rcv = computeDistances(ri, sensor_pos);
    Eigen::MatrixXd time_points_distPA = ((phys_dist_rcv * 1e-3 / cPA) * Fs).array() - start_time;
    Eigen::MatrixXd linear_factor = time_points_distPA.array() - time_points_distPA.array().floor();

    std::cout<< "beamform" << std::endl;
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(imsz * imsz, 1);
    for (int i = 0; i < sensor_count; ++i) {
        for (int j = 0; j < imsz * imsz; ++j) {
            int lower_idx = static_cast<int>(std::floor(time_points_distPA(j, i)));
            int upper_idx = static_cast<int>(std::ceil(time_points_distPA(j, i)));

            double resultlower = outp(i, lower_idx);
            double resultupper = outp(i, upper_idx);

            result(j, 0) += resultlower + linear_factor(j, i) * (resultupper - resultlower);
        }
    }

    Eigen::Map<Eigen::MatrixXd> reshaped(result.data(), imsz, imsz);

    return reshaped;
}

// int main() {
//     try {
//         // Load sensor positions
//         Eigen::MatrixXd sensor_pos = loadSensorPositions("sensor_pos.txt");

//         // Define output matrix (dummy data, replace with actual data)
//         Eigen::MatrixXd outp = Eigen::MatrixXd::Random(sensor_count, 256); // Assuming 256 time steps

//         // Perform reconstruction for a specific z index
//         int zidx = 10; // Example value
//         Eigen::MatrixXd result = recon_z(zidx, sensor_pos, outp);

//         // Output result
//         std::cout << "Reconstruction completed for zidx: " << zidx << std::endl;
//     } catch (const std::exception &e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//     }

//     return 0;
// }
