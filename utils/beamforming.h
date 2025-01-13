#ifndef BEAMFORMING_H
#define BEAMFORMING_H

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <cmath>
#include <fstream>

Eigen::MatrixXd loadSensorPositions(const std::string &filename);

void createMeshgrid(Eigen::MatrixXd &rix, Eigen::MatrixXd &riy) ;

Eigen::MatrixXd computeDistances(const Eigen::MatrixXd &ri, const Eigen::MatrixXd &sensor_pos);

Eigen::MatrixXd recon_z(double zidx, const Eigen::MatrixXd &sensor_pos, const Eigen::MatrixXd &outp);


#endif // VRS_LOADER_H
