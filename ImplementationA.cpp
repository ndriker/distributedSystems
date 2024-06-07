#include <cstdlib>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cctype>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <mpi.h>

#define INITIATOR_NODE 2
#define INITIATOR_RANK INITIATOR_NODE - 1

int image_height;
int image_width;
int image_maxShades;
int *inputImage;

int outputIntensity[256];

int adj_size;
int* adjMatrix;

int row_num;

void fill_adjMatrix(std::vector<int> row_vec) {
    int j = 0;
    for (int val : row_vec) {
        adjMatrix[row_num * adj_size + j] = val;
        j++;
    }
    row_num++;
}

std::vector<int> read_adjMatrix_line(std::string row) {
    std::stringstream ss(row);
    int count = 0;
    std::string token;
    std::vector<int> row_vec;
    while (!ss.eof()) {
        std::getline(ss, token, '\t');
        row_vec.push_back(stoi(token));
    }
    return row_vec;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int num_processes, rank;

    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    // check processes
    if (num_processes < 2) {
        std::cout << "Need at least two processes" << std::endl;
        MPI_Finalize();
        return 0;
    }


    /*
        =============================
        Start Read Image Parameters
        =============================
    */

    std::string workString;

    std::ifstream file(argv[1]);
    if (!file.is_open()) {
        std::cout << "ERROR: Could not open file " << argv[1] << std::endl;
        MPI_Finalize();
        return 0;
    }
    
    // Remove comments '#' and check image format
    while (std::getline(file, workString)) {
        if (workString.at(0) != '#') {
            if (workString.at(1) != '2') {
                std::cout << "Input image is not a valid PGM image" << std::endl;
                MPI_Finalize();
                return 0;
            } else {
                break;
            }
        } else {
            continue;
        }
    }

    // Check image size
    while (std::getline(file, workString)) {
        if (workString.at(0) != '#') {
            std::stringstream stream(workString);
            int n;
            stream >> n;
            image_width = n;
            stream >> n;
            image_height = n;
            break;
        } else {
            continue;
        }
    }



    // Check image max shades
    while (std::getline(file, workString)) {
        if (workString.at(0) != '#') {
            std::stringstream stream(workString);
            stream >> image_maxShades;
            break;
        } else {
            continue;
        }

    }

    inputImage = (int*) malloc(image_height * image_width * sizeof(int) );

    /*
        =============================
        End Read Image Parameters
        =============================
    */

    /*
        =============================
        Start Find Adj Matrix Size
        =============================
    */
    
    std::ifstream adj_file(argv[2]);
    if (!adj_file.is_open()) {
        std::cout << "ERROR: Could not open file " << argv[2] << std::endl;
        MPI_Finalize();
        return 0;
    }

    std::string adjString;
    row_num = 0;

    std::vector<int> temp;

    std::getline(adj_file, adjString, '\n');
    std::vector<int> row_vec = read_adjMatrix_line(adjString);
    adj_size = row_vec.size();

    adjMatrix = (int*) malloc(adj_size * adj_size * sizeof(int) );


    if (rank == INITIATOR_RANK) {

        // Fill input image matrix
        int pixel_val;
        for (int i = 0; i < image_height; i++) {
            if (std::getline(file, workString) && workString.at(0) != '#') {
                std::stringstream stream(workString);
                for (int j = 0; j < image_width; j++) {
                    if (!stream) {
                        break;
                    }
                    stream >> pixel_val;
                    inputImage[i*image_width + j] = pixel_val;
                }
            } else {
                continue;
            }
        }

        fill_adjMatrix(row_vec);
        while (std::getline(adj_file, adjString, '\n')) {
            row_vec = read_adjMatrix_line(adjString);
            fill_adjMatrix(row_vec);

        }

        //for (int i = 0; i < image_height * image_width; i++) {
        //    std::cout << inputImage[i] << " ";
        //}
        //std::cout << std::endl;

    }

    int data_per_process = (image_height * image_width) / num_processes;
    int* partial_img = (int*) malloc(sizeof(int) * data_per_process);

    MPI_Scatter(inputImage, data_per_process, MPI_INT, partial_img, data_per_process, MPI_INT, INITIATOR_RANK, MPI_COMM_WORLD);


    int* partial_adj = (int*) malloc(adj_size * sizeof(int));
    MPI_Scatter(adjMatrix, adj_size, MPI_INT, partial_adj, adj_size, MPI_INT, INITIATOR_RANK, MPI_COMM_WORLD);

    for (int i = 0; i < data_per_process; i++) {
        int intensity_val = partial_img[i];
        outputIntensity[intensity_val] += 1;
    }


    //std::cout << "Rank: " << rank << std::endl;
    //for (int intensity : outputIntensity) {
    //    std::cout << intensity << " ";
    //}
    //std::cout << std::endl;
    //
    //std::cout << "Rank: " << rank << std::endl;
    //for (int adj_ind = 0; adj_ind < adj_size; adj_ind++) {
    //    std::cout << partial_adj[adj_ind] << " ";
    //}
    //std::cout << std::endl;


    int parent = -1;
    int* sent_to = (int*) calloc(adj_size, sizeof(int));
    bool has_accumulated = false;

    if (rank == INITIATOR_RANK) {
        for (int adj_ind = 0; adj_ind < adj_size; adj_ind++) {
            if (partial_adj[adj_ind] == 1) {
                MPI_Send(&outputIntensity, 256, MPI_INT, adj_ind, 0, MPI_COMM_WORLD);
                std::cout << "Sent to node " << adj_ind + 1 << " from node" << rank + 1<< std::endl;
                sent_to[adj_ind] = 1;
                has_accumulated = true;
                break;
            }
        }
    }


    int* cum_o_intensity = (int*) calloc(256, sizeof(int));
    
    bool sent_to_parent = false;
    MPI_Status status;
    while (!sent_to_parent) {

        MPI_Recv(cum_o_intensity, 256, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        std::cout << "Node " << rank + 1 << " received from node " << status.MPI_SOURCE  + 1<< std::endl;
        if (parent == -1 && rank != INITIATOR_RANK) {
            parent = status.MPI_SOURCE;
        }
        int send_to;
        bool found_one_not_parent = false;
    

        for (int node_ind = 0; node_ind < adj_size; node_ind++) {
            if (partial_adj[node_ind]) {
                if (!sent_to[node_ind]) {
                    if (node_ind != parent) {
                        found_one_not_parent = true;
                        send_to = node_ind;
                        break;
                    }
                }
            }
        }
        if (!found_one_not_parent) {
            if (rank == INITIATOR_RANK) {
                std::cout << "should be final iter" << std::endl;
                break;
            }
            send_to = parent;
            sent_to_parent = true;
        }

        sent_to[send_to] = 1;

        if (has_accumulated) {
            // forward
            MPI_Send(cum_o_intensity, 256, MPI_INT, send_to, 0, MPI_COMM_WORLD);
            std::cout << "Forwarded to node " << send_to + 1<< std::endl;
            
        } else {
            for (int oidx = 0; oidx < 256; oidx++) {
                cum_o_intensity[oidx] = cum_o_intensity[oidx] + outputIntensity[oidx];
            }
            MPI_Send(cum_o_intensity, 256, MPI_INT, send_to, 0, MPI_COMM_WORLD);
            std::cout << "Sent to node " << send_to + 1<< std::endl;
            has_accumulated = true;

        }

    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == INITIATOR_RANK) {
        std::cout << "writing to file" << std::endl;
        // Write output intensities to file
        std::ofstream ofile(argv[3]);
        if (ofile.is_open()) {
            for (int oidx = 0; oidx < 256; oidx++) {
                ofile << cum_o_intensity[oidx] << std::endl;
            }

        } else {
            std::cout << "ERROR: Could not open output file " << argv[3] << std::endl;
            MPI_Finalize();
            return 0;

        }
    }

    MPI_Finalize();
    return 0;
}
