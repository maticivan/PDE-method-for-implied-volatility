//*************************************************************************************************
//*************************************************************************************************
//* The MIT License (MIT)                                                                         *
//* Copyright (C) 2019 Ivan Matic, Rados Radoicic, and Dan Stefanica                              *
//*                                                                                               *
//* Permission is hereby granted, free of charge, to any person obtaining a copy of this          *
//* software and associated documentation files (the "Software"), to deal in the Software         *
//* without restriction, including without limitation the rights to use, copy, modify, merge,     *
//* publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons    *
//* to whom the Software is furnished to do so, subject to the following conditions:              *
//*                                                                                               *
//* The above copyright notice and this permission notice shall be included in all copies or      *
//* substantial portions of the Software.                                                         *
//*                                                                                               *
//* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,           *
//* INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR      *
//* PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE     *
//* FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR          *
//* OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        *
//* DEALINGS IN THE SOFTWARE.                                                                     *
//*************************************************************************************************

#ifndef GRAPHICCARDOPENCL_H
#define GRAPHICCARDOPENCL_H



#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <map>
#include <set>
#include <iostream>
#include <fstream>
#include <sstream>
#include <ctime>
#include <iterator>
#include <regex>
#include <string>
#include <streambuf>
#include <random>
#include <chrono>
#include<vector>
#include <omp.h>
#include "linear_algebra_tnt/jama_eig.h"


typedef long int mysint;
typedef long int myint;
typedef double mydouble;



class Strin_gList{
public:
    std::string content;
    Strin_gList* next;
};



class GraphicCard{
public:
    GraphicCard(std::string="parallel.cl",mysint =0);
     ~GraphicCard();
    mysint findAddKernel(std::string,mysint=0);
    //Arguments: kernelName, preventAddition
    //If the kernel with given name exists, the
    //function will return the kernel's number.
    //Otherwise, the function will return -1.
    //If the second parameter is set to 0, the function will add the kernel with the given name.

    

    
    
    template <typename int_doub> mysint deviceMemoryAccess(std::string,int_doub *, myint, mysint=0, myint=0);
    
    //The arguments are: std::string memBlockName, myint * memorySequence,
    //                      myint sLength, mysint action, myint writingShift=0
    //First the function will determine the index of the element of memObjectsGC that
    //corresponds to the memBlockName.
    //If it does not exist then the index will be announced to be -1.
    //action==0 - Locating the memory block named memBlockName, or creating a new one, if it
    //            does not exist. The content will be copied or overwritten from memorySequence.
    //            The first writingShift members of memoryBlock will be skipped.
    //action==1 - Reading the memory block of length sLength (starting from writingShift)
    //              and storing it into the sequence
    //action==2 - Updating only the length of the memory block. The new length is provided in sLength.
    //action==3 - Just determine the id number of the memory block with name memBlockName.
    //action==4 - Delete memory block;
    

    myint getPrefferedWorkGroupSize() const;
    cl_double * convertVectToSeq(const std::vector<mydouble>&);
    myint * convertVectToSeq(const std::vector<myint>&);
    std::vector<mydouble> convertSeqToVect(cl_double *, const myint &);
    std::vector<myint> convertSeqToVect(myint *, const myint &);
    
    template <typename int_doub> mysint writeDeviceMemory(std::string, int_doub *, myint);
    template <typename int_doub> mysint readDeviceMemory(std::string, int_doub *, myint);
    
    mysint writeDeviceMemory(std::string, const std::vector<mydouble> &, myint);
    mysint writeDeviceMemory(std::string, const std::vector<myint> &, myint);
    mysint readDeviceMemory(std::string, std::vector<mydouble> &, myint);
    mysint readDeviceMemory(std::string, std::vector<myint> &, myint);

    
    mysint freeDeviceMemory(std::string);
    
    mysint setKernelArguments(std::string,std::string*,mysint);
    // The arguments are: kernelName, listOfParameters, numberOfParametersInTheList
    mysint executeKernel(std::string,myint);
    
    
    mysint generateRandomNumbers(myint,myint,mysint=0);
    //Runs random number generator (in the first run, it will initialize them as well).
    //Arguments:    N - Size of each axis in the seed
    //              r - length of the numbers in binary
    //              indicator
    // - If indicator is 1, then the permutations will be created again.
    // - Note that permutations will be created the first time the generator is executed.
   
    mysint setLocationNameForRandomNumbers(std::string);
    // The location on device where the random numbers will be stored is
    // "randSampGCCL" by default.
    // However, it does not have to be that way. You can set a new location, especially if
    // you want to store more random numbers than one algorithm can generate.
    
    mysint generateNormalBeasleySpringerMoro(myint, myint, myint, myint*,
                                             cl_double =0.0, cl_double =1.0, mysint=0);
    
    //Generates normal distribution
    //Arguments:    N - Size of each axis in the seed
    //              r - length of the numbers in binary
    //              precisionRequest - the number of integers from the discrete distribution that will be used
    //                                  to generate uniform [0,1] distribution.
    //              sampleLength - (myint*) contains the size of the generated sample
    //                              sampleLength = N^2 /  precisionRequest
    //              parameter1 - used for normal and denotes the mean
    //              parameter2 - for normal it denotes sigma, for exponential, it denotes lambda.
    //              indicator - If the indicator is 1, then the permutations will be created again.
    
    
    mysint generateExponential(myint, myint, myint, myint*,
                               cl_double =1.0, mysint=0);

    
    myint createPSS(std::string, std::string, std::string, std::string, myint,myint);
    // creates PSS (prefix sums sequence)
    // for a given sequence originalSeq creates pSS
    // pSS[k] = \sum_{i=0}^{k}
    // Arguments are:
    // 1) pSSNameWithOrigSeq    (string)
    // 2) pSSHelper             (string)
    // 3) lengthName            (string)
    // 4) blockSizeName         (string)
    // 5) length                (myint)
    // 6) blockSize             (myint)
    
    myint setParametersPSS(std::string, std::string, std::string, std::string,myint,myint);
    // sets the names for parameters in PSS
    // 1) pSSNameWithOrigSeq    (string)
    // 2) pSSHelper             (string)
    // 3) lengthName            (string)
    // 4) blockSizeName         (string)
    // 5) length                (myint)
    // 6) blockSize             (myint)

    myint createPSS();
    // creates PSS
    // This one should be used if the parameters are already set with setParametersPSS
    

    myint partialSumsPSS(std::string, std::string, std::string, std::string, std::string ,myint);
    // creates partial sums based on PSS that is already in device memory.
    // Parameters are:
    // 1) nameOfPSS             (string)
    // 2) lenghtName            (string)
    // 3) fromSequenceName      (string)
    // 4) toSequenceName        (string)
    // 5) resultingSequenceName (string)
    // 6) length (myint)
    
    template <typename int_doub> myint createFenwickTree(std::string,std::string, int_doub* ,myint);
    // creates Fenwick tree in device memory.
    // Arguments are
    // 1) treeName (string)
    // 2) nameOfTheParameterThatHoldsTheLength (string)
    // 3) sequence of origin (int_doub)
    // 4) length (myint)
    // The function returns the length of the Fenwick tree. The length is the smallest
    // power of 2 that is greater than or equal to length.
    
    
    cl_double justSummation(std::string,std::string, myint, myint =0);
    // Sums the sequence using Fenwick tree.
    // Arguments are
    // 1) sequenceName (string)
    // 2) nameOfTheParameterThatHoldsTheLength (string)
    // 3) length (myint)
    // 4) avInd (myint) - indicator for average. If it is 0, the sum is calculated. Otherwise, the average.
    // The function returns the sum.
    
    
    myint partialSumsFTree(std::string, std::string, std::string, std::string, std::string ,myint);
    // creates partial sums based on Fenwick tree that is already in device memory.
    // Parameters are:
    // 1) treeName (string)
    // 2) nameOfTheParameterThatHoldsTheLength (string)
    // 3) fromSequenceName (string)
    // 4) toSequenceName (string)
    // 5) resultingSequenceName (string)
    // 6) length (myint)
    
    
    myint mergeSort(std::string, myint,myint=0, std::string="not provided");
    // Performs a parallel merge sort.
    // Parameters are:
    // 1) sequenceToSort (string)
    // 2) length(myint)
    // 3) direction(myint) - 0 for increasing, 1 for decreasing
    // 4) name of helper sequence (string) - if not provided, one will be created
    //    There may be an advantage of providing a helper sequence already allocated on
    //    device memory if it is anticipated that mergeSort will be frequently called.
    //    In that case memory can be saved by not allocating a helper sequence every time.

    
    myint mergeSortWithFollower(std::string, std::string, myint,myint=0, std::string="not provided", std::string="not provided");
    // Performs a parallel merge sort with additional sequence called follower
    // that contains the permutation that keeps track of the original position of the elements.
    // Parameters are:
    // 1) sequenceToSort (string)
    // 2) followerSequence (string)
    // 3) length(myint)
    // 4) direction(myint) - 0 for increasing, 1 for decreasing
    // 5) name of helper sequence (string) - if not provided, one will be created
    //    There may be an advantage of providing a helper sequence already allocated on
    //    device memory if it is anticipated that mergeSort will be frequently called.
    //    In that case time and memory can be saved by not allocating a helper sequence every time.
    // 6) name of helper for the follower sequence(string)
    mysint overrideAntitetic(myint );
    
protected:
    cl_context contextGC;
    cl_command_queue commandQueueGC;
    cl_program programGC;
    cl_device_id deviceGC;
    cl_kernel* kernflsGC;
    cl_mem *memObjectsGC;
    mysint numMemObjectsGC;
    mysint numberOfKernelsGC;
    std::map<std::string,myint> kernelNamesGC;
    std::map<std::string,myint> memObjNamesGC;
    size_t* preferred_workgroup_sizeGC;
    
    std::set<myint>* kernelResponsibilitiesGC;
    
    mysint *variablesCorrectlySetInKernelGC;
    
    //For random number generators
    myint *axisSizeGC;
    myint *xAxisGC;
    myint *yAxisGC;
    myint *lengthInBinaryGC;
    myint *balancedNumbersGC;
    myint *numBalancedNumbersGC;
    myint *shufflingPrimeGC;
    myint *xPermutationsGC;
    myint *yPermutationsGC;
    myint *pascalTriangleGC;
    myint *sizePascalTrGC;
    myint *highestPrecisionGC;
    myint *precisionRequestGC;
    
    myint *exponentKGC;
    myint *sizeForRejectionSamplingGC;
    myint *sampleLengthGC;
    myint *inspectorRejSampGC;
    
    myint antitheticGC;
    myint overrideAntiteticGC;
    cl_double *parameter1GC;
    cl_double *parameter2GC;
    
    cl_double *randomSampleGC;

    
    
    myint indFirstRNInitGC;
    
    myint inspectorExecutedGC;
    myint sampleGenKernExecutedGC;
    
    myint currentSeedGC;
    myint uniformLimitGC;
    
    myint lastRandAlgUsedGC; //each random distribution has a code. Normal is 7; exponential is 8
                             //when the same algorithm with same sample sizes is used repeatedly, some
                             //savings in time are possible.
                             //This variable is updated to contain the label of the last algorithm used.
    
    myint blockSizePSSGC;
    myint lengthValuePSSGC;
    myint *randNumbersGC;
    myint *powersOfTwoGC;
    
    myint numberOfDistributionsGC;
    std::string normalBSMNameGC;
    std::string randSampGC;
    std::string exponentialDistNameGC;
    std::string kernelFileGC;
    std::string kernelPreLoadedGC;
    std::string createFenwickGC;
    std::string justSummationFenwickGC;
    std::string partialSumsFTreeGC;
    std::string partialSumsPSSGC;
    std::string shiftCalculationMergSortIncGC;
    std::string shiftCalculationMergSortDecGC;
    std::string copyResultToStartGC;
    std::string shiftCalculationMergSortFollIncGC;
    std::string shiftCalculationMergSortFollDecGC;
    std::string copyResultToStartFollGC;
    std::string origNamePSSGC;
    std::string namePSSGC;
    std::string lengthNamePSSGC;
    std::string nameBlockSizePSSGC;
    std::string kernelNameStage1PSSGC;
    std::string kernelNameStage2PSSGC;
    std::string posVariableNamePSSGC;
    std::string normalBSMAntNameGC;
    cl_context CreateContext(mysint =0);
    cl_command_queue CreateCommandQueue(cl_context, cl_device_id*);
    cl_program CreateProgram(cl_context, cl_device_id , const char* );
    myint treatError(cl_int, cl_context, cl_command_queue,
                     cl_program, cl_kernel *, myint,cl_mem *, myint);
    
    
    void Cleanup();
    mysint generatePermutationsGC(myint*,myint,myint);
    myint createBalancedNumbersGC(myint *,  myint *, myint* , myint *, myint*);
    myint createPascalTriangleGC(myint);
};




#endif


