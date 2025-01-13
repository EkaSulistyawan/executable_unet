#ifndef VRS_LOADER_H
#define VRS_LOADER_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <stdint.h> // For uint8_t, uint16_t, etc.

#include <torch/script.h> // For TorchScript
#include <torch/torch.h> // For TorchScript

// Struct to store the header info, update according to your structure
struct HeaderInfo {
    uint16_t version;
    uint8_t compression;
    uint8_t timetagflag;
    std::vector<uint8_t> time_tag;
    uint64_t studyidlength;
    std::string studyid;
    uint64_t sampleidlength;
    std::string sampleid;
    uint64_t commentlength;
    std::string comment;
    std::vector<uint64_t> dim;
    uint64_t numdatapoints;
    uint8_t datatypes;
};

// Function prototype
std::vector<std::vector<std::vector<int16_t>>> load_vrs(const std::string& fileName, int nt = 4096, int ntx = 28, bool verbose = false);
torch::Tensor load_vrs_torch(const std::string& fileName, int nt, int ntx = 28, bool verbose = false);

#endif // VRS_LOADER_H
