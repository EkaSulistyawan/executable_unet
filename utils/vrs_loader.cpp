#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <stdint.h> // For uint8_t, uint16_t, etc.

#include <torch/script.h> // For TorchScript
#include <torch/torch.h> // For TorchScript

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

std::vector<std::vector<std::vector<int16_t>>> load_vrs(const std::string& fileName, int nt, int ntx = 28, bool verbose = false) {
    HeaderInfo headerInfo;
    std::vector<std::vector<std::vector<int16_t>>> data2(
    256, 
    std::vector<std::vector<int16_t>> (
        ntx, std::vector<int16_t>(nt)
        )
        );

    std::ifstream file_obj(fileName, std::ios::binary);
    if (!file_obj.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return data2;
    }

    // Read Version (uint16)
    file_obj.read(reinterpret_cast<char*>(&headerInfo.version), 3);
    
    // Read Compression (uint8)
    file_obj.read(reinterpret_cast<char*>(&headerInfo.compression), 1);
    
    // Read Timetag flag (uint8)
    file_obj.read(reinterpret_cast<char*>(&headerInfo.timetagflag), 1);
    

    if (headerInfo.timetagflag == 1) {
        // Read time tag (6 bytes: Sec, Min, Hour, Day, Month, Year)
        headerInfo.time_tag.resize(6);
        file_obj.read(reinterpret_cast<char*>(headerInfo.time_tag.data()), 6);
    }
    
    
    // Read studyidlength (uint64)
    file_obj.read(reinterpret_cast<char*>(&headerInfo.studyidlength), 8);
    
    headerInfo.studyid.resize(headerInfo.studyidlength);
    file_obj.read(&headerInfo.studyid[0], headerInfo.studyidlength);
    
    // Read sampleidlength (uint64)
    file_obj.read(reinterpret_cast<char*>(&headerInfo.sampleidlength), 8);
    headerInfo.sampleid.resize(headerInfo.sampleidlength);
    file_obj.read(&headerInfo.sampleid[0], headerInfo.sampleidlength);
    
    
    // Read commentlength (uint64)
    file_obj.read(reinterpret_cast<char*>(&headerInfo.commentlength), 8);
    headerInfo.comment.resize(headerInfo.commentlength);
    file_obj.read(&headerInfo.comment[0], headerInfo.commentlength);
    
    
    // Read dim (4 uint64 values)
    headerInfo.dim.resize(4);
    file_obj.read(reinterpret_cast<char*>(headerInfo.dim.data()), 32);
    
    
    // Read numdatapoints (uint64)
    file_obj.read(reinterpret_cast<char*>(&headerInfo.numdatapoints), 8);
    
    
    // Read datatypes (uint8)
    file_obj.read(reinterpret_cast<char*>(&headerInfo.datatypes), 1);

    // Read data (numdatapoints * 2 bytes for int16)
    std::vector<int16_t> data(headerInfo.numdatapoints);
    file_obj.read(reinterpret_cast<char*>(data.data()), headerInfo.numdatapoints * 2);

    // reshape data to 3D
    int idx = 0;
    for (int i = 0; i < 256; ++i) {
        for (int j = 0; j < ntx; ++j) {
            // Copy each slice of data directly
            std::memcpy(&data2[i][j][0], &data[idx], nt * sizeof(int16_t));
            idx += nt;
        }
    }

    



    if (verbose) {
        std::cout << "Header Info:" << std::endl;
        std::cout << "Version: " << headerInfo.version << std::endl;
        std::cout << "Compression: " << (int)headerInfo.compression << std::endl;
        std::cout << "Timetag Flag: " << (int)headerInfo.timetagflag << std::endl;
        std::cout << "Study ID: " << headerInfo.studyid << std::endl;
        std::cout << "Sample ID: " << headerInfo.sampleid << std::endl;
        std::cout << "Comment: " << headerInfo.comment << std::endl;
        std::cout << "Dimensions: ";
        for (auto& dim_val : headerInfo.dim) {
            std::cout << dim_val << " ";
        }
        std::cout << std::endl;
    }

    return data2;
}

torch::Tensor load_vrs_torch(const std::string& fileName, int nt, int ntx = 28, bool verbose = false) {
    HeaderInfo headerInfo;

    std::ifstream file_obj(fileName, std::ios::binary);
    if (!file_obj.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return torch::zeros({256,ntx,nt});
    }

    // Read Version (uint16)
    file_obj.read(reinterpret_cast<char*>(&headerInfo.version), 3);
    
    // Read Compression (uint8)
    file_obj.read(reinterpret_cast<char*>(&headerInfo.compression), 1);
    
    // Read Timetag flag (uint8)
    file_obj.read(reinterpret_cast<char*>(&headerInfo.timetagflag), 1);
    

    if (headerInfo.timetagflag == 1) {
        // Read time tag (6 bytes: Sec, Min, Hour, Day, Month, Year)
        headerInfo.time_tag.resize(6);
        file_obj.read(reinterpret_cast<char*>(headerInfo.time_tag.data()), 6);
    }
    
    
    // Read studyidlength (uint64)
    file_obj.read(reinterpret_cast<char*>(&headerInfo.studyidlength), 8);
    
    headerInfo.studyid.resize(headerInfo.studyidlength);
    file_obj.read(&headerInfo.studyid[0], headerInfo.studyidlength);
    
    // Read sampleidlength (uint64)
    file_obj.read(reinterpret_cast<char*>(&headerInfo.sampleidlength), 8);
    headerInfo.sampleid.resize(headerInfo.sampleidlength);
    file_obj.read(&headerInfo.sampleid[0], headerInfo.sampleidlength);
    
    
    // Read commentlength (uint64)
    file_obj.read(reinterpret_cast<char*>(&headerInfo.commentlength), 8);
    headerInfo.comment.resize(headerInfo.commentlength);
    file_obj.read(&headerInfo.comment[0], headerInfo.commentlength);
    
    
    // Read dim (4 uint64 values)
    headerInfo.dim.resize(4);
    file_obj.read(reinterpret_cast<char*>(headerInfo.dim.data()), 32);
    
    
    // Read numdatapoints (uint64)
    file_obj.read(reinterpret_cast<char*>(&headerInfo.numdatapoints), 8);
    
    
    // Read datatypes (uint8)
    file_obj.read(reinterpret_cast<char*>(&headerInfo.datatypes), 1);

    // Read data (numdatapoints * 2 bytes for int16)
    std::vector<int16_t> raw_data(headerInfo.numdatapoints);
    file_obj.read(reinterpret_cast<char*>(raw_data.data()), headerInfo.numdatapoints * 2);

    // Now we will reshape the data into a 3D tensor (256, ntx, nt)
    // First, flatten the data into a 1D vector of floats (casting int16_t to float)
    std::vector<float> flattened_data(raw_data.size());
    std::transform(raw_data.begin(), raw_data.end(), flattened_data.begin(),
                   [](int16_t val) { return static_cast<float>(val); });

    // Create a tensor from the flattened data
    torch::Tensor tensor = torch::from_blob(flattened_data.data(),
                                            {256, ntx, nt}, torch::kFloat);

    // Ensure the tensor is contiguous in memory
    tensor = tensor.clone();

    if (verbose) {
        std::cout << "Header Info:" << std::endl;
        std::cout << "Version: " << headerInfo.version << std::endl;
        std::cout << "Compression: " << (int)headerInfo.compression << std::endl;
        std::cout << "Timetag Flag: " << (int)headerInfo.timetagflag << std::endl;
        std::cout << "Study ID: " << headerInfo.studyid << std::endl;
        std::cout << "Sample ID: " << headerInfo.sampleid << std::endl;
        std::cout << "Comment: " << headerInfo.comment << std::endl;
        std::cout << "Dimensions: ";
        for (auto& dim_val : headerInfo.dim) {
            std::cout << dim_val << " ";
        }
        std::cout << std::endl;
    }

    return tensor;
}

// int main(int argc, char* argv[]) {
//     if (argc < 2) {
//         std::cerr << "Usage: " << argv[0] << " <file_name> [nt] [ntx] [verbose]" << std::endl;
//         return 1;
//     }

//     std::string fileName = argv[1];
//     int nt = (argc > 2) ? std::stoi(argv[2]) : 4096;  // Default nt is 4096
//     int ntx = (argc > 3) ? std::stoi(argv[3]) : 28; // Default ntx is 28
//     bool verbose = (argc > 4) ? std::stoi(argv[4]) : true; // Default verbose is true

//     auto data = load_vrs(fileName, nt, ntx, verbose);

//     return 0;
// }
