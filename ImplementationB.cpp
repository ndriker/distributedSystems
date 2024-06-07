#include "mpi.h"
#include <algorithm>
#include <functional>
#include <cstdlib>
#include <ctime>
#include <cctype>
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
const static int ARRAY_SIZE = 130000;
using Lines = char[ARRAY_SIZE][16];

/************* Feel free to change code below (even main() function), add functions, etc.. But do not change CL arguments *************/


struct letter_only: std::ctype<char> 
{
    letter_only(): std::ctype<char>(get_table()) {}

    static std::ctype_base::mask const* get_table()
    {
        static std::vector<std::ctype_base::mask> 
            rc(std::ctype<char>::table_size,std::ctype_base::space);

        std::fill(&rc['A'], &rc['z'+1], std::ctype_base::alpha);
        return &rc[0];
    }
};

void DoOutput(std::string word, int result)
{
    std::cout << "Word Frequency: " << word << " -> " << result << std::endl;
}


int countFrequency( std::vector<std::string> data, std::string word)
{
	int freq = 0;
	for(auto workString : data) {
		if( !workString.compare(word) )
			freq++;
	}
	return freq;
}

int main(int argc, char* argv[])
{
    int processId;
    int numberOfProcesses;
    int *to_return = NULL;
    double start_time, end_time;
 
    // Setup MPI
    MPI_Init( &argc, &argv );
    MPI_Comm_rank( MPI_COMM_WORLD, &processId);
    MPI_Comm_size( MPI_COMM_WORLD, &numberOfProcesses);
 
    // Two arguments, the program name and the input file. The second should be the input file
    if(argc != 4)
    {
        if(processId == 0)
        {
            std::cout << "ERROR: Incorrect number of arguments. Format is: <path to search file> <search word> <b1/b2>" << std::endl;
        }
        MPI_Finalize();
        return 0;
    }
	std::string word = argv[2];
 
    Lines lines;
    if (processId == 0) {
        std::ifstream file;
		file.imbue(std::locale(std::locale(), new letter_only()));
		file.open(argv[1]);
		std::string workString;
		int i = 0;
		while(file >> workString){
			memset(lines[i], '\0', 16);
			memcpy(lines[i++], workString.c_str(), workString.length());
		}
    }
	
    char buf[(ARRAY_SIZE / numberOfProcesses) * 16];

    MPI_Scatter(lines, ARRAY_SIZE  * 16/ numberOfProcesses, MPI_CHAR, buf, ARRAY_SIZE * 16/ numberOfProcesses, MPI_CHAR, 0, MPI_COMM_WORLD);
    
    start_time=MPI_Wtime();
	
    std::vector<std::string> data;
    for (int i = 0; i < (ARRAY_SIZE / numberOfProcesses) * 16; i += 16) {
        char s[16];
        memcpy(s, &buf[i], 16);
        std::string s2(s);
        data.push_back(s2);
    }

	int temp_result = countFrequency(std::ref(data), word);
	
	std::string opt = argv[3];	
	if( !opt.compare("b1") )
	{
		std::cout << "Using reduction" << std::endl;

        int global_result;
        MPI_Reduce(&temp_result, &global_result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        to_return = &global_result;

	} else {
		std::cout << "Using ring" << std::endl;
        
        int temp;
        int sum = 0;
        for (int rank = 0; rank < numberOfProcesses; rank++) {
            temp = sum + temp_result ;
            MPI_Send(&temp, 1, MPI_INT, (processId + 1) % numberOfProcesses, 0, MPI_COMM_WORLD);
            MPI_Recv(&sum, 1, MPI_INT, (processId + numberOfProcesses - 1) % numberOfProcesses, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        }        
		
		if (processId == 0){
			to_return = &sum;
		}
	}
    // ... Eventually.. 
    if(processId == 0)
    {
        DoOutput(word, *to_return);
		end_time=MPI_Wtime();
        std::cout << "time: " << ((double)end_time-start_time) << std::endl;
    }
 
    MPI_Finalize();
 
    return 0;
}
