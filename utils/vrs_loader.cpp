#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <stdint.h> // For uint8_t, uint16_t, etc.

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
    for (int i=0; i < 256;++i){
        for (int j=0; j < ntx; ++j){
            for (int k=0; k < nt; ++k){
                data2[i][j][k] = data[idx++];
            }
        }
    }
    std::cout << idx << " " << headerInfo.numdatapoints << std::endl;

    // // Reshape data (dim[1], dim[0])
    // std::vector<std::vector<int16_t>> reshaped_data(headerInfo.dim[1], std::vector<int16_t>(headerInfo.dim[0]));
    // int idx = 0;
    // for (size_t i = 0; i < headerInfo.dim[1]; ++i) {
    //     for (size_t j = 0; j < headerInfo.dim[0]; ++j) {
    //         reshaped_data[i][j] = data[idx++];
    //     }
    // }

    // // Further reshape data to (dim[1], ntx, nt)
    // data2.resize(headerInfo.dim[1], std::vector<std::vector<int16_t>>(nt, std::vector<int16_t>(ntx)));
    // for (size_t i = 0; i < headerInfo.dim[1]; ++i) {
    //     for (int j = 0; j < nt; ++j) {
    //         for (int k = 0; k < ntx; ++k) {
    //             data2[i][j][k] = reshaped_data[i][k * nt + j];
    //         }
    //     }
    // }

    // data3.resize(headerInfo.dim[1], std::vector<std::vector<int16_t>>(ntx, std::vector<int16_t>(nt)));
    // for (size_t i = 0; i < headerInfo.dim[1]; ++i) {
    //     for (int j = 0; j < ntx*nt; ++j) {
    //             data3[i][j] = reshaped_data[i][j];
    //     }
    // }

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
