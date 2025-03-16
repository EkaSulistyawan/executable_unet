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
) {
    // Create a meshgrid using linspace
    auto rix = torch::linspace(-r, r, imsz).reshape({imsz, 1, 1}).expand({imsz, imsz, imsz});
    auto riy = torch::linspace(-r, r, imsz).reshape({1, imsz, 1}).expand({imsz, imsz, imsz});
    auto riz = torch::linspace(-r, r, imsz).reshape({1, 1, imsz}).expand({imsz, imsz, imsz});

    // Rotation matrix for 45 degrees in xy-plane
    double theta = -M_PI / 4;
    double cos_theta = std::cos(theta);
    double sin_theta = std::sin(theta);

    auto rix_rot = cos_theta * rix - sin_theta * riy;
    auto riy_rot = sin_theta * rix + cos_theta * riy;

    // Flatten and stack into (N, 3) shape
    auto ri = torch::stack({rix_rot.flatten(), riy_rot.flatten(), riz.flatten()}, -1);

    // Compute Euclidean distance between ri and sensor_pos.T
    auto phys_dist_rcv = torch::cdist(ri, sensor_pos.transpose(0, 1));

    // Compute start_time_PA
    auto time_points_distPA = (phys_dist_rcv * 1e-3 / cPA) * Fs - start_time_PA;

    return time_points_distPA;
}


torch::Tensor create_time_dist_US(
    torch::Tensor sensor_pos,
    torch::Tensor UseTXElements,
    int dim,
    double r,
    int imsz,
    double Fs,
    double cUS
) {
    // Create a meshgrid using linspace
    auto rix = torch::linspace(-r, r, imsz).reshape({imsz, 1, 1}).expand({imsz, imsz, imsz});
    auto riy = torch::linspace(-r, r, imsz).reshape({1, imsz, 1}).expand({imsz, imsz, imsz});
    auto riz = torch::linspace(-r, r, imsz).reshape({1, 1, imsz}).expand({imsz, imsz, imsz});

    // Rotation matrix for 45 degrees in the xy-plane
    double theta = -M_PI / 4;
    double cos_theta = std::cos(theta);
    double sin_theta = std::sin(theta);

    auto rix_rot = cos_theta * rix - sin_theta * riy;
    auto riy_rot = sin_theta * rix + cos_theta * riy;

    // Convert rotated coordinates back to tensors
    rix = rix_rot;
    riy = riy_rot;

    // The z-coordinates remain unchanged
    // Select mid slices based on `dim`
    torch::Tensor ri;
    if (dim == 0) {
        ri = torch::stack({
            rix.index({imsz / 2, torch::indexing::Slice(),torch::indexing::Slice() }).flatten(),
            riy.index({imsz / 2, torch::indexing::Slice(),torch::indexing::Slice() }).flatten(),
            riz.index({imsz / 2, torch::indexing::Slice(),torch::indexing::Slice() }).flatten()
        }, -1);
    } else if (dim == 1) {
        ri = torch::stack({
            rix.index({torch::indexing::Slice(), imsz / 2, torch::indexing::Slice()}).flatten(),
            riy.index({torch::indexing::Slice(), imsz / 2, torch::indexing::Slice()}).flatten(),
            riz.index({torch::indexing::Slice(), imsz / 2, torch::indexing::Slice()}).flatten()
        }, -1);
    } else if (dim == 2) {
        std::cout << " select 2" << std::endl;
        ri = torch::stack({
            rix.index({torch::indexing::Slice(), torch::indexing::Slice(), imsz / 2}).flatten(),
            riy.index({torch::indexing::Slice(), torch::indexing::Slice(), imsz / 2}).flatten(),
            riz.index({torch::indexing::Slice(), torch::indexing::Slice(), imsz / 2}).flatten()
        }, -1);
    }

    // Create output tensor for time distances
    auto time_points_distUS_all = torch::zeros({256, static_cast<int>(UseTXElements.size(0)), imsz * imsz});

    for (int i = 0; i < UseTXElements.size(0); ++i) {
        auto tx = sensor_pos.index({torch::indexing::Slice(), i}).unsqueeze(0);
        auto phys_dist_rcv = torch::cdist(ri, sensor_pos.t()) + 
                             torch::cdist(ri, tx.repeat({256, 1}));
        time_points_distUS_all.index_put_({torch::indexing::Slice(), i, torch::indexing::Slice()},
                                          (phys_dist_rcv.t() * 1e-3 / cUS) * Fs);
    }

    return time_points_distUS_all;
}
