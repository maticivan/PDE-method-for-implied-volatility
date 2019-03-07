
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

#include"GCOpenCL.h"

void deleteStrin_gListInGC(Strin_gList* first){
    if(first->next!=0){
        deleteStrin_gListInGC(first->next);
    }
    delete first;
}



GraphicCard::GraphicCard(std::string programName,mysint forceCPU){
    contextGC=nullptr;
    commandQueueGC=nullptr;
    programGC=nullptr;
    deviceGC=0;
    kernflsGC=nullptr;
    numMemObjectsGC=0;
    memObjectsGC=nullptr;
    numberOfKernelsGC=0;
    preferred_workgroup_sizeGC=new size_t;
    *preferred_workgroup_sizeGC=128;
    variablesCorrectlySetInKernelGC=nullptr;
    
    kernelPreLoadedGC="gc_parallel_rnum.cl";
    axisSizeGC=new myint;
    *axisSizeGC=0;
    xAxisGC=nullptr;
    yAxisGC=nullptr;
    randNumbersGC=nullptr;
    lengthInBinaryGC=new myint;
    *lengthInBinaryGC=0;
    shufflingPrimeGC=new myint;
    *shufflingPrimeGC=3;
    
    numBalancedNumbersGC=new myint;
    *numBalancedNumbersGC=0;
    
    xPermutationsGC=nullptr;
    yPermutationsGC=nullptr;
    indFirstRNInitGC=0;
    inspectorExecutedGC=0;
    sampleGenKernExecutedGC=0;
    uniformLimitGC=10000;
    currentSeedGC=std::chrono::high_resolution_clock::now().time_since_epoch().count();
    overrideAntiteticGC=0;
    powersOfTwoGC=nullptr;
    balancedNumbersGC=nullptr;
    contextGC=CreateContext(forceCPU);
    kernelFileGC=programName;
    
    pascalTriangleGC=nullptr;
    sizePascalTrGC=new myint;
    *sizePascalTrGC=0;
    
    highestPrecisionGC=new myint;
    *highestPrecisionGC=100000;
    
    precisionRequestGC=new myint;
    *precisionRequestGC=0;
    exponentKGC=new myint;
    *exponentKGC=3;
    sizeForRejectionSamplingGC=new myint;
    *sizeForRejectionSamplingGC=5;
    
    sampleLengthGC=new myint;
    *sampleLengthGC=0;
    antitheticGC=1;
    parameter1GC=new cl_double;
    *parameter1GC=0;
    parameter2GC=new cl_double;
    *parameter2GC=1;
    randomSampleGC=nullptr;
    
    inspectorRejSampGC=nullptr;
    kernelResponsibilitiesGC=nullptr;
    
    randSampGC="randSampGCCL";
    
    normalBSMNameGC="normalBSMGC";

    
    exponentialDistNameGC="exponentialDistGC";
    createFenwickGC="createFenwickGC";
    justSummationFenwickGC="justSummationFenwickGC";
    partialSumsFTreeGC="partialSumsFTreeGC";
    partialSumsPSSGC="partialSumsPSSGC";
    shiftCalculationMergSortIncGC="shiftCalculationMergSortIncGC";
    shiftCalculationMergSortDecGC="shiftCalculationMergSortDecGC";
    copyResultToStartGC="copyResultToStartGC";
    shiftCalculationMergSortFollIncGC="shiftCalculationMergSortFollIncGC";
    shiftCalculationMergSortFollDecGC="shiftCalculationMergSortFollDecGC";
    copyResultToStartFollGC="copyResultToStartFollGC";
    normalBSMAntNameGC="normalBSMAntitheticGC";
    origNamePSSGC="origNamePSSGC";
    namePSSGC="namePSSGC";
    lengthNamePSSGC="lengthNamePSSGC";
    nameBlockSizePSSGC="nameBlockSizePSSGC";
    kernelNameStage1PSSGC="stage1PSSGC";
    kernelNameStage2PSSGC="stage2PSSGC";
    posVariableNamePSSGC="positionNamePSSGC";
    blockSizePSSGC=*preferred_workgroup_sizeGC;
    lengthValuePSSGC=-1;
    lastRandAlgUsedGC=-1;
    mysint shouldQuit=0;
    
    if (contextGC == nullptr)
    {
        std::cerr << "Failed to create OpenCL context." << std::endl;
        shouldQuit=1;
    }
    if(shouldQuit==0){
        commandQueueGC = CreateCommandQueue(contextGC, &deviceGC);
    }
    if((shouldQuit==0)&& (commandQueueGC == nullptr))
    {
        Cleanup();
        shouldQuit= 1;
    }
    if(shouldQuit==0){
        const char * cChar = kernelFileGC.c_str();
        
        
        programGC = CreateProgram(contextGC, deviceGC, cChar);
    }
    if((shouldQuit==0)&&(programGC == nullptr))
    {
        Cleanup();
        shouldQuit=1;
    }
    
    
    
}
mysint GraphicCard::overrideAntitetic(myint ov){
    overrideAntiteticGC=ov;
    return 1;
}
mysint GraphicCard::findAddKernel(std::string nameToAdd, mysint preventAddition){
    mysint i=0;
    mysint foundName=-1;
    std::map<std::string, myint>::iterator it;
    it=kernelNamesGC.find(nameToAdd);
    if(it!=kernelNamesGC.end()){
        foundName=it->second;
    }
    
    if((foundName==-1)&&(preventAddition==0)){
        cl_kernel *newKernfls;
        mysint *newVarCS;
        newKernfls=new cl_kernel[numberOfKernelsGC+1];
        newVarCS=new mysint[numberOfKernelsGC+1];
        for(i=0;i<numberOfKernelsGC;++i){
            newKernfls[i]=kernflsGC[i];
            newVarCS[i]=variablesCorrectlySetInKernelGC[i];
            
        }
        if(numberOfKernelsGC>0){
            delete[] kernflsGC;
            delete[] variablesCorrectlySetInKernelGC;
        }
        kernflsGC=newKernfls;
        variablesCorrectlySetInKernelGC=newVarCS;
        kernelNamesGC.insert(std::make_pair(nameToAdd,numberOfKernelsGC));;
        const char * cChar = nameToAdd.c_str();
        cl_int error_ret;
        kernflsGC[numberOfKernelsGC]=clCreateKernel(programGC,cChar,&error_ret);
        myint jErr=0;
        while((jErr<1000)&&(error_ret!=0)){
            kernflsGC[numberOfKernelsGC]=clCreateKernel(programGC,cChar,&error_ret);
            
            ++jErr;
        }
        if(numberOfKernelsGC<3){
            clGetKernelWorkGroupInfo (kernflsGC[numberOfKernelsGC],
                                      deviceGC,
                                      CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                                      sizeof(size_t),
                                      preferred_workgroup_sizeGC,
                                      nullptr);
        }
        variablesCorrectlySetInKernelGC[numberOfKernelsGC]=0;
        numberOfKernelsGC++;
    }
    return foundName;
}









cl_context GraphicCard::CreateContext(mysint forceCPU)
{
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id firstPlatformId;
    cl_context context = nullptr;
    
    
    errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
    if (errNum != CL_SUCCESS || numPlatforms <= 0)
    {
        std::cerr << "No OpenCL platforms." << std::endl;
        return nullptr;
    }
    
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)firstPlatformId,
        0
    };
    
    //Select GPU:
    
    
    if(forceCPU==0){
        context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,nullptr, nullptr, &errNum);
        if (errNum != CL_SUCCESS)
        {
            std::cout << "No GPU context. Trying CPU." << std::endl;
            context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
                                              nullptr, nullptr, &errNum);
            if (errNum != CL_SUCCESS)
            {
                std::cerr << "Failed to create an OpenCL GPU or CPU context." << std::endl;
                return nullptr;
            }
        }
    }
    //Comparison: Select CPU:
    
    if(forceCPU==1){
        context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_CPU,
                                          nullptr, nullptr, &errNum);
        if (errNum != CL_SUCCESS)
        {
            std::cerr << "Failed to create an OpenCL CPU context." << std::endl;
            return nullptr;
        }
    }
    
    
    
    return context;
}


cl_command_queue GraphicCard::CreateCommandQueue(cl_context context, cl_device_id *device)
{
    cl_int errNum;
    cl_device_id *devices;
    cl_command_queue commandQueue = nullptr;
    size_t deviceBufferSize = -1;
    
    // First get the size of the devices buffer
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, nullptr, &deviceBufferSize);
    
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)";
        return nullptr;
    }
    
    if (deviceBufferSize <= 0)
    {
        std::cerr << "No devices available.";
        return nullptr;
    }
    
    // Allocate memory for the devices buffer
    myint numDevices=(myint)(deviceBufferSize / sizeof(cl_device_id));
    devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, nullptr);
    if (errNum != CL_SUCCESS)
    {
        delete [] devices;
        std::cerr << "Failed to get device IDs";
        return nullptr;
    }
    
    
    
    myint deviceToChoose=numDevices-1;

    
    commandQueue = clCreateCommandQueue(context, devices[deviceToChoose], 0, nullptr);

    if (commandQueue == nullptr)
    {
        delete [] devices;
        std::cerr << "Failed to create commandQueue for device 0";
        return nullptr;
    }
    
    *device = devices[deviceToChoose];
    delete [] devices;
    return commandQueue;
}

cl_program GraphicCard::CreateProgram(cl_context context, cl_device_id device, const char* fileName)
{
    cl_int errNum;
    cl_program program;
    
    
    
    const char * fileName0 = kernelPreLoadedGC.c_str();
    std::ifstream kernelFile0(fileName0, std::ios::in);
    if (!kernelFile0.is_open())
    {
        std::cerr << "Failed to open file for reading: " << fileName0 << std::endl;
        return nullptr;
    }
    
    std::ostringstream oss0;
    oss0 << kernelFile0.rdbuf();
    
    std::string srcStdStr0 = oss0.str();
    
    std::ifstream kernelFile(fileName, std::ios::in);
    if (!kernelFile.is_open())
    {
        std::cerr << "Failed to open file for reading: " << fileName << std::endl;
        return nullptr;
    }
    
    std::ostringstream oss;
    oss << kernelFile.rdbuf();
    
    std::string srcStdStr = srcStdStr0;
    srcStdStr+=oss.str();
    
    const char *srcStr = srcStdStr.c_str();

    program = clCreateProgramWithSource(context, 1,
                                        (const char**)&srcStr,
                                        nullptr, nullptr);
    if (program == nullptr)
    {
        std::cerr << "Failed to create CL program from source." << std::endl;
        return nullptr;
    }
    
    errNum = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
    if (errNum != CL_SUCCESS)
    {

        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), buildLog, nullptr);
        
        std::cerr << "Error in kernel: " << std::endl;
        
        
        std::cerr << buildLog;
        clReleaseProgram(program);
        return nullptr;
    }
    
    return program;
}



void GraphicCard::Cleanup()
{
    for (myint i = 0; i < numMemObjectsGC; ++i)
    {
        
        if (memObjectsGC[i] != nullptr)
            clReleaseMemObject(memObjectsGC[i]);
    }
    
    if (commandQueueGC != nullptr)
        clReleaseCommandQueue(commandQueueGC);
    
    for(myint k=0;k<numberOfKernelsGC;++k){
        if(kernflsGC[k]!=nullptr){
            clReleaseKernel(kernflsGC[k]);
        }
    }
    
    
    
    
    if (programGC != nullptr)
        clReleaseProgram(programGC);
    
    if (contextGC != nullptr)
        clReleaseContext(contextGC);
    
    delete[] kernflsGC;
    
    delete[] memObjectsGC;
    delete[] kernelResponsibilitiesGC;
    delete preferred_workgroup_sizeGC;
    delete axisSizeGC;
    delete lengthInBinaryGC;
    delete shufflingPrimeGC;
    delete[] variablesCorrectlySetInKernelGC;
    delete[] powersOfTwoGC;
    
    
    delete[] xAxisGC;
    
    delete[] yAxisGC;
    
    delete[] xPermutationsGC;
    
    delete[] yPermutationsGC;
    
    delete[] balancedNumbersGC;
    
    delete numBalancedNumbersGC;
    
    delete[] pascalTriangleGC;
    
    delete sizePascalTrGC;
    
    delete highestPrecisionGC;
    delete precisionRequestGC;
    
    delete exponentKGC;
    delete sizeForRejectionSamplingGC;
    delete sampleLengthGC;
    
    delete parameter1GC;
    delete parameter2GC;
    
    delete[] randomSampleGC;
    
    delete[] inspectorRejSampGC;


    
    delete randNumbersGC;
    
}


template <typename int_doub>
mysint GraphicCard::deviceMemoryAccess(std::string memBlockName,int_doub *memorySequence,
                                       myint sLength, mysint action, myint writingShift){
    mysint i=0;
    mysint foundName=-1;
    cl_int errNum;
    cl_event eventE;
    myint *pSLength;
    pSLength=new myint;
    *pSLength=sLength;
    
    std::map<std::string, myint>::iterator it;
    it=memObjNamesGC.find(memBlockName);
    if(it!=memObjNamesGC.end()){
        foundName=it->second;
    }
    if((foundName==-1)&&(action==0)){
        
        cl_mem *newMemObjects;
        std::set<myint>*newKernelResp;
        
        newKernelResp=new std::set<myint>[numMemObjectsGC+1];
        newMemObjects=new cl_mem[2*(numMemObjectsGC+1)];
        for(i=0;i<numMemObjectsGC;++i){
            newKernelResp[i]=kernelResponsibilitiesGC[i];
            newMemObjects[2*i]=memObjectsGC[2*i];
            newMemObjects[2*i+1]=memObjectsGC[2*i+1];
        }
        if(numMemObjectsGC>0){
            delete[] memObjectsGC;
            delete[] kernelResponsibilitiesGC;
        }
        
        
        memObjectsGC=newMemObjects;
        kernelResponsibilitiesGC=newKernelResp;
        memObjNamesGC.insert(std::make_pair(memBlockName,numMemObjectsGC));
        //kernelResponsibilitiesGC[numMemObjectsGC] is empty set without us having to do anything
        
        
        memObjectsGC[2*numMemObjectsGC] = clCreateBuffer(contextGC, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                                         sizeof(myint) , pSLength, nullptr);
        memObjectsGC[2*numMemObjectsGC+1] = clCreateBuffer(contextGC, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                                           sizeof(int_doub) * sLength , memorySequence, nullptr);
        
        ++numMemObjectsGC;
        
    }
    if((foundName!=-1)&&(action==1)){
        errNum = clEnqueueReadBuffer(commandQueueGC, memObjectsGC[2*foundName+1], CL_TRUE,
                                     sizeof(int_doub) * writingShift,   sizeof(int_doub)* sLength, memorySequence,
                                     0, nullptr, &eventE);
        clWaitForEvents(1, &eventE);
    }
    
    if((foundName!=-1)&&(action==0)){
        clEnqueueWriteBuffer(commandQueueGC, memObjectsGC[2*foundName+1], CL_TRUE, sizeof(int_doub) * writingShift,   sizeof(int_doub)* sLength,memorySequence,0,nullptr, &eventE);
        clWaitForEvents(1, &eventE);
    }
    
    if((foundName!=-1)&&(action==2)){
        clEnqueueWriteBuffer(commandQueueGC, memObjectsGC[2*foundName], CL_TRUE, 0,     sizeof(myint),pSLength,0,nullptr, &eventE);
        clWaitForEvents(1, &eventE);
    }
    if((foundName!=-1)&&(action==4)){
        if (memObjectsGC[2*foundName] != nullptr){
            clReleaseMemObject(memObjectsGC[2*foundName]);
        }
        if (memObjectsGC[2*foundName+1] != nullptr){
            clReleaseMemObject(memObjectsGC[2*foundName+1]);
        }
        --numMemObjectsGC;
        memObjNamesGC.erase(memBlockName);
        
        std::map<std::string, myint>::iterator it1;
        for(it1=memObjNamesGC.begin();it1!=memObjNamesGC.end();++it1){
            if((*it1).second>foundName){
                --((*it1).second);
            }
        }
        
        for(myint i=foundName;i<numMemObjectsGC;++i){
            memObjectsGC[2*i]=memObjectsGC[2*(i+1)];
            memObjectsGC[2*i+1]=memObjectsGC[2*(i+1)+1];
            kernelResponsibilitiesGC[i]=kernelResponsibilitiesGC[i+1];
        }
        
    }
    
    
    delete pSLength;
    return foundName;
}






mysint GraphicCard::setKernelArguments(std::string kernelName,std::string* parNames,mysint numberOfParameters){
    mysint kernelNumber=findAddKernel(kernelName);
    mysint forReturn=0;
    myint *memSequence=nullptr;
    mysint helpNum;
    cl_int errNum;
    
    if(kernelNumber==-1){
        kernelNumber=findAddKernel(kernelName);
    }
    
    mysint *parNumbers;
    parNumbers=new mysint[numberOfParameters];
     
    for(mysint i=0;i<numberOfParameters;++i){
        parNumbers[i]=deviceMemoryAccess(parNames[i],memSequence , 0, 3, 0);
        if(parNumbers[i]==-1){
            forReturn=-1;
            
        }
    }
    if(forReturn!=-1){
        for(mysint i=0;i<numberOfParameters;++i){
            kernelResponsibilitiesGC[parNumbers[i]].insert(kernelNumber);
            helpNum=2*parNumbers[i];
            ++helpNum;
            errNum=clSetKernelArg(kernflsGC[kernelNumber], i, sizeof(cl_mem), &memObjectsGC[helpNum]);
            if (errNum != CL_SUCCESS)
            {
                std::cerr << "Error setting kernel argument " << i<<" "<<errNum<<std::endl;
                std::cerr << "Kernel number: " << kernelNumber<<", kernel code:"<<kernflsGC[kernelNumber]<<std::endl;
                
                if(errNum==CL_INVALID_KERNEL){
                    std::cout<<"CL_INVALID_KERNEL"<<std::endl;
                }
                if(errNum==CL_INVALID_ARG_INDEX){
                    std::cout<<"CL_INVALID_ARG_INDEX"<<std::endl;
                }
                if(errNum==CL_INVALID_ARG_VALUE){
                    std::cout<<"CL_INVALID_ARG_VALUE"<<std::endl;
                }
                if(errNum==CL_INVALID_MEM_OBJECT){
                    std::cout<<"CL_INVALID_MEM_OBJECT"<<std::endl;
                }
                if(errNum==CL_INVALID_SAMPLER){
                    std::cout<<"CL_INVALID_SAMPLER"<<std::endl;
                }
                if(errNum==CL_INVALID_ARG_SIZE){
                    std::cout<<"CL_INVALID_ARG_SIZE"<<std::endl;
                }
                if(errNum==CL_OUT_OF_RESOURCES){
                    std::cout<<"CL_OUT_OF_RESOURCES"<<std::endl;
                }
                if(errNum==CL_OUT_OF_HOST_MEMORY){
                    std::cout<<"CL_OUT_OF_HOST_MEMORY"<<std::endl;
                }
                forReturn=-2;
            }
            
            if(forReturn!=-2){
                
                variablesCorrectlySetInKernelGC[kernelNumber]=1;
            }
            
            
        }
    }
    
    delete[] parNumbers;
    return forReturn;
    
}






mysint GraphicCard::executeKernel(std::string kernelName, myint numberOfProcessingElements){
    mysint kernelNumber=findAddKernel(kernelName, 1);
    cl_int errNum;
    cl_event eventE;
    size_t *globalWorkSizeP;
    size_t *localWorkSizeP;
    globalWorkSizeP=new size_t;
    localWorkSizeP=new size_t;
    size_t pws;
    
    if(kernelNumber==-1){
        
        kernelNumber=findAddKernel(kernelName);
        if(kernelNumber==-1){
            kernelNumber=findAddKernel(kernelName,1);
        }
    }
    
    if(kernelNumber!=-1){
        
        if(variablesCorrectlySetInKernelGC[kernelNumber]==0){
            //  The variables were not set correctly
            //  We need to set them up.
            
            
            std::ifstream t(kernelFileGC);
            std::string str;
            
            t.seekg(0, std::ios::end);
            str.reserve(t.tellg());
            t.seekg(0, std::ios::beg);
            
            str.assign((std::istreambuf_iterator<char>(t)),
                       std::istreambuf_iterator<char>());
            
            replace (str.begin(), str.end(), '*' , ' ');
            replace (str.begin(), str.end(), ',' , ' ');
            replace (str.begin(), str.end(), ';' , ' ');
            replace (str.begin(), str.end(), ')' , ' ');
            replace (str.begin(), str.end(), '&' , ' ');
            replace (str.begin(), str.end(), '(' , ' ');
            
            size_t index;
            while ((index = str.find("const")) != std::string::npos)
                str.replace(index, 5, " ");
            
            while ((index = str.find("void")) != std::string::npos)
                str.replace(index, 4, " ");
            while ((index = str.find("__local")) != std::string::npos)
                str.replace(index, 7, " __global ");
            
            
            std::regex word_regex("(\\S+)");
            std::sregex_iterator words_begin =
            std::sregex_iterator(str.begin(), str.end(), word_regex);
            std::sregex_iterator  words_end = std::sregex_iterator();
            
            std::sregex_iterator cWord;
            std::string wordNext;
            mysint ik=0;
            mysint jk=0;
            cWord=words_begin;
            
            Strin_gList* firstListEl, *currentListEl;
            firstListEl=new Strin_gList;
            currentListEl=firstListEl;
            (*firstListEl).next=0;
            while(cWord!=words_end){
                wordNext=(*cWord).str();
                if(wordNext=="__kernel"){
                    ++cWord;
                    if(cWord!=words_end){
                        wordNext=(*cWord).str();
                        if(wordNext==kernelName){
                            ++cWord;
                            while(cWord!=words_end){
                                wordNext=(*cWord).str();
                                if(wordNext=="__global"){
                                    ++cWord;
                                    if(cWord!=words_end){
                                        ++cWord;
                                        if(cWord!=words_end){
                                            ++ik;
                                            (*currentListEl).content=(*cWord).str();
                                            (*currentListEl).next=new Strin_gList;
                                            currentListEl=(*currentListEl).next;
                                            (*currentListEl).next=0;
                                        }
                                    }
                                }
                                if(cWord!=words_end){++cWord;}
                                if(cWord!=words_end){
                                    if(((*cWord).str())=="__kernel"){
                                        cWord=words_end;
                                    }
                                }
                                
                            }
                        }
                    }
                }
                if(cWord!=words_end){++cWord;}
            }
            std::string* myStList;
            myStList=new std::string[ik];
            currentListEl=firstListEl;
            for(myint i=0;i<ik;++i){
                myStList[i]=currentListEl->content;
                currentListEl=currentListEl->next;
            }
            
            
            deleteStrin_gListInGC(firstListEl);
            
            setKernelArguments(kernelName,myStList,ik);
            
            delete[] myStList;
            
        }
        
        pws= (size_t) numberOfProcessingElements;
        pws= ( (pws/ (*preferred_workgroup_sizeGC))+1) * (*preferred_workgroup_sizeGC);
        *globalWorkSizeP=pws;
        *localWorkSizeP=*preferred_workgroup_sizeGC;
        
        
        errNum=clEnqueueNDRangeKernel(commandQueueGC, kernflsGC[kernelNumber], 1, nullptr,
                                      globalWorkSizeP, localWorkSizeP,
                                      0, nullptr, &eventE);
        
        if (errNum != CL_SUCCESS){
            std::cout<<"Error in kernel "<<kernelNumber<<std::endl;
            
            std::map<std::string, myint>::iterator it;
            
            for(it=kernelNamesGC.begin();it!=kernelNamesGC.end();++it){
                if(it->second==kernelNumber){
                    std::cout<<it->first<<std::endl;
                }
            }
            
            treatError(errNum,contextGC, commandQueueGC, programGC, kernflsGC, numberOfKernelsGC, memObjectsGC, numMemObjectsGC);
            return 1;
        }
        clWaitForEvents(1, &eventE);
        
    }
    delete globalWorkSizeP;
    delete localWorkSizeP;
    
    
    return kernelNumber;
    
}


myint GraphicCard::treatError(cl_int errNum, cl_context context, cl_command_queue commandQueue,
                              cl_program program, cl_kernel *kernfls, myint numKernels,cl_mem *memObjects, myint numMemObjects){
    std::cerr << "Error queuing kernel for execution! " << " "<<errNum<< std::endl;
    if(errNum==CL_INVALID_PROGRAM_EXECUTABLE){std::cout<<"CL_INVALID_PROGRAM_EXECUTABLE"<<std::endl;}
    if(errNum==CL_INVALID_COMMAND_QUEUE){std::cout<<"CL_INVALID_COMMAND_QUEUE"<<std::endl;}
    if(errNum==CL_INVALID_KERNEL){std::cout<<"CL_INVALID_KERNEL"<<std::endl;}
    if(errNum==CL_INVALID_CONTEXT){std::cout<<"CL_INVALID_CONTEXT"<<std::endl;}
    if(errNum==CL_INVALID_KERNEL_ARGS){std::cout<<"CL_INVALID_KERNEL_ARGS"<<std::endl;}
    if(errNum==CL_INVALID_WORK_DIMENSION){std::cout<<"CL_INVALID_WORK_DIMENSION"<<std::endl;}
    if(errNum==CL_INVALID_GLOBAL_WORK_SIZE){std::cout<<"CL_INVALID_GLOBAL_WORK_SIZE"<<std::endl;}
    if(errNum==CL_INVALID_GLOBAL_OFFSET){std::cout<<"CL_INVALID_GLOBAL_OFFSET"<<std::endl;}
    if(errNum==CL_INVALID_WORK_GROUP_SIZE){std::cout<<"CL_INVALID_WORK_GROUP_SIZE"<<std::endl;}
    if(errNum==CL_INVALID_WORK_ITEM_SIZE){std::cout<<"CL_INVALID_WORK_ITEM_SIZE"<<std::endl;}
    if(errNum==CL_MISALIGNED_SUB_BUFFER_OFFSET){std::cout<<"CL_MISALIGNED_SUB_BUFFER_OFFSET"<<std::endl;}
    if(errNum==CL_OUT_OF_RESOURCES){std::cout<<"CL_OUT_OF_RESOURCES"<<std::endl;}
    if(errNum==CL_MEM_OBJECT_ALLOCATION_FAILURE){std::cout<<"CL_MEM_OBJECT_ALLOCATION_FAILURE"<<std::endl;}
    if(errNum==CL_INVALID_EVENT_WAIT_LIST){std::cout<<"CL_INVALID_EVENT_WAIT_LIST"<<std::endl;}
    if(errNum==CL_OUT_OF_HOST_MEMORY){std::cout<<"CL_OUT_OF_HOST_MEMORY"<<std::endl;}
    
    
    Cleanup();
    return 1;
}

template <typename int_doub> mysint GraphicCard::writeDeviceMemory(std::string memBlockName,int_doub *memorySequence, myint sLength){
    mysint memId=deviceMemoryAccess(memBlockName,memorySequence,sLength,3);
 
    if(memId!=-1){
        // We will check whether the new length is bigger than the allocated length.
        // If that is the case, we need to reallocate more space.
        myint currentLength;
        myint *clp;
        clp=&currentLength;
        cl_int errNum;
        cl_event eventE;
        
        
        errNum = clEnqueueReadBuffer(commandQueueGC, memObjectsGC[2*memId], CL_TRUE,
                                     0,   sizeof(myint) , clp,
                                     0, nullptr, &eventE);
        
        clWaitForEvents(1, &eventE);
        
        if(errNum!=CL_SUCCESS){
            std::cout<<"Error in reading "<<memBlockName<<" "<<errNum<< std::endl;
        
            if(errNum==CL_INVALID_COMMAND_QUEUE){
                std::cout<<"CL_INVALID_COMMAND_QUEUE"<<std::endl;
            }
            if(errNum==CL_INVALID_CONTEXT){
                std::cout<<"CL_INVALID_CONTEXT"<<std::endl;
            }
            if(errNum==CL_INVALID_MEM_OBJECT){
                std::cout<<"CL_INVALID_MEM_OBJECT"<<std::endl;
            }
            if(errNum==CL_INVALID_VALUE){
                std::cout<<"CL_INVALID_VALUE"<<std::endl;
            }
            if(errNum== CL_INVALID_EVENT_WAIT_LIST ){
                std::cout<<"CL_INVALID_EVENT_WAIT_LIST"<<std::endl;
            }
            if(errNum== CL_MEM_OBJECT_ALLOCATION_FAILURE){
                std::cout<<"CL_MEM_OBJECT_ALLOCATION_FAILURE"<<std::endl;
            }
            if(errNum==CL_OUT_OF_HOST_MEMORY){
                std::cout<<"CL_OUT_OF_HOST_MEMORY"<<std::endl;
            }
            if(errNum==CL_OUT_OF_RESOURCES){
                std::cout<<"CL_OUT_OF_RESOURCES"<<std::endl;
            }
        }
        
        if(currentLength<sLength){
            std::set<myint>::iterator it;
            for(it=(kernelResponsibilitiesGC[memId]).begin();it!=(kernelResponsibilitiesGC[memId]).end();++it){
                variablesCorrectlySetInKernelGC[*it]=0;
            }
            deviceMemoryAccess(memBlockName,memorySequence,sLength,4);
        }
        
    }
    
    
    return deviceMemoryAccess(memBlockName,memorySequence,sLength);
}
template <typename int_doub> mysint GraphicCard::readDeviceMemory(std::string memBlockName,int_doub *memorySequence, myint sLength){
    return deviceMemoryAccess(memBlockName,memorySequence,sLength,1);
}

cl_double * GraphicCard::convertVectToSeq(const std::vector<mydouble> & _v){
    myint len=_v.size();
    cl_double * fR;
    fR=new cl_double[len];
    
    myint nthreads;
#pragma omp parallel
    {
        if(omp_get_thread_num()==0){
            nthreads=omp_get_num_threads();
        }
    }
#pragma omp barrier
#pragma omp parallel num_threads(nthreads)
    {
        myint myId=omp_get_thread_num();
        myint numberOfMyJobs=(len/nthreads);
        
        if(len % nthreads!=0){
            ++numberOfMyJobs;
        }
        myint i;
        for(myint mI=0;mI<numberOfMyJobs;++mI){
            
            i=mI*nthreads+ myId;
            if(i<len){
                fR[i] = (cl_double)(_v[i]);
            }
        }
        
    }
    
    
#pragma omp barrier
    
    
    
    return fR;
}

myint * GraphicCard::convertVectToSeq(const std::vector<myint> & _v){
    myint len=_v.size();
    myint * fR;
    fR=new myint[len];
    
    myint nthreads;
#pragma omp parallel
    {
        if(omp_get_thread_num()==0){
            nthreads=omp_get_num_threads();
        }
    }
#pragma omp barrier
#pragma omp parallel num_threads(nthreads)
    {
        myint myId=omp_get_thread_num();
        myint numberOfMyJobs=(len/nthreads);
        
        if(len % nthreads!=0){
            ++numberOfMyJobs;
        }
        myint i;
        for(myint mI=0;mI<numberOfMyJobs;++mI){
            i=mI*nthreads+ myId;
            if(i<len){fR[i] = _v[i];}
        }
        
    }
    
    
#pragma omp barrier
    
    
    
    return fR;
}



std::vector<mydouble> GraphicCard::convertSeqToVect(cl_double * _s, const myint &len){
    std::vector<mydouble> fR;
    
    fR.resize(len);
    for(myint i=0;i<len;++i){
        fR[i]=(mydouble)(_s[i]);
    }
    return fR;
}

std::vector<myint> GraphicCard::convertSeqToVect(myint * _s, const myint &len){
    std::vector<myint> fR;
    fR.resize(len);
    for(myint i=0;i<len;++i){
        fR[i]= _s[i] ;
    }
    return fR;
}


mysint GraphicCard::writeDeviceMemory(std::string memBlockName,
                                      const std::vector<mydouble> & _v,
                                      myint sLength){
    cl_double * _ts;
    _ts=convertVectToSeq(_v);
    
    mysint fR=writeDeviceMemory(memBlockName,_ts,sLength);
    delete[] _ts;
    return fR;
}

mysint GraphicCard::writeDeviceMemory(std::string memBlockName,
                                      const std::vector<myint> & _v,
                                      myint sLength){
    myint * _ts;
    _ts=convertVectToSeq(_v);
    
    mysint fR=writeDeviceMemory(memBlockName,_ts,sLength);
    delete[] _ts;
    return fR;
}

mysint GraphicCard::readDeviceMemory(std::string memBlockName,
                                     std::vector<mydouble> & _v,
                                     myint sLength){
    cl_double * recV;
    recV=new cl_double[sLength];
    mysint fR=readDeviceMemory(memBlockName,recV,sLength);
    _v=convertSeqToVect(recV,sLength);
    
    delete[] recV;
    return fR;
}
mysint GraphicCard::readDeviceMemory(std::string memBlockName,
                                     std::vector<myint> & _v,
                                     myint sLength){
    myint * recV;
    recV=new myint[sLength];
    mysint fR=readDeviceMemory(memBlockName,recV,sLength);
    _v=convertSeqToVect(recV,sLength);
    
    delete[] recV;
    return fR;
}



mysint GraphicCard::freeDeviceMemory(std::string memBlockName){
    myint*memorySequence, sLength;
    return deviceMemoryAccess(memBlockName,memorySequence,sLength,4);
}



GraphicCard::~GraphicCard(){
    
    Cleanup();
}













class TripleGC{
public:
    myint a;
    myint b;
    myint c ;
    myint sorting;
    TripleGC();
    TripleGC(myint, myint, myint);
    TripleGC(myint, myint, myint,myint);
    mysint operator<(const TripleGC&);
    mysint operator>(const TripleGC&);
    mysint operator=(const TripleGC&);
};
TripleGC::TripleGC(){
    a=0;
    b=0;
    c =0;
    sorting=-1;
}
TripleGC::TripleGC(myint p, myint q, myint r){
    a=p;
    b=q;
    c=r;
}
TripleGC::TripleGC(myint p, myint q, myint r,myint s ){
    a=p;
    b=q;
    c=r;
    sorting=s;
}
mysint TripleGC::operator=(const TripleGC& t){
    a=t.a;
    b=t.b;
    c=t.c;
    sorting=t.sorting;
    return 1;
}

mysint TripleGC::operator<(const TripleGC& t){
    if(a<t.a){
        return 1;
    }
    if(a>t.a){
        return 0;
    }
    if(b<t.b){
        return 1;
    }
    if(b>t.b){
        return 0;
    }
    if(c<t.c){
        return 1;
    }
    if(c>t.c){
        return 0;
    }
    return 0;
}

mysint TripleGC::operator>(const TripleGC& t){
    if(a>t.a){
        return 1;
    }
    if(a<t.a){
        return 0;
    }
    if(b>t.b){
        return 1;
    }
    if(b<t.b){
        return 0;
    }
    if(c>t.c){
        return 1;
    }
    if(c<t.c){
        return 0;
    }
    return 0;
}
myint mergeSortTGC(TripleGC *seq, myint lS,  TripleGC *sSeq=nullptr, myint lSS=0){
    //First job is to sort the first sequence seq
    // and the second job is to add the sorted sequence sSeq into seq
    // If the length of the first sequence is 1 or 0, there is no need for sorting.
    if(lS>1){
        //The sorting will be done by dividing the sequence in two parts.
        myint middle=(lS+1)/2;
        mergeSortTGC(seq+middle,lS-middle);
        mergeSortTGC(seq,middle,seq+middle,lS-middle);
    }
    if(lSS>0){
        myint length=lS+lSS;
        TripleGC* help;
        help=new TripleGC[length];
        myint r=0, rL=0,rR=0;
        while(r<length){
            if(rL>=lS){
                help[r]=sSeq[rR];
                ++rR;
            }
            else{
                if(rR>=lSS){
                    help[r]=seq[rL];
                    ++rL;
                }
                else{
                    if(seq[rL]<sSeq[rR]){
                        help[r]=seq[rL];
                        ++rL;
                    }
                    else{
                        help[r]=sSeq[rR];
                        ++rR;
                    }
                }
            }
            ++r;
        }
        for(r=0;r<length;++r){
            seq[r]=help[r];
        }
        delete[] help;
    }
    return 0;
}



myint powerGC(myint base, myint exponent, myint modulo=0){
    if(exponent==1){
        if(modulo!=0){
            return base%modulo;
        }
        else{
            return base;
        }
    }
    if(exponent==0){
        return 1;
    }
    myint forReturn=1;
    myint forReturnH,exp1;
    if(exponent%2==1){
        forReturn=base;
    }
    exp1=exponent/2;
    forReturnH= powerGC(base, exp1, modulo);
    forReturn*=forReturnH;
    forReturn*=forReturnH;
    if(modulo!=0){
        forReturn%=modulo;
    }
    return forReturn;
    
}





mysint GraphicCard::generatePermutationsGC(myint *seqToFill, myint N, myint r){
    if(uniformLimitGC<r){
        uniformLimitGC=r;
    }
    std::uniform_int_distribution<myint> uInt(0,uniformLimitGC);
    myint i,j;
    TripleGC *permuts;
    permuts=new TripleGC[r];
    
    myint seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    seed+=currentSeedGC;
    std::mt19937 mt_randIF(seed);
    
    for(i=0;i<r;++i){
        permuts[i].a=0;
        permuts[i].b=0;
        permuts[i].c=0;
    }
    
    for(i=0;i<N;++i){
        for(j=0;j<r;++j){
            permuts[j].a=uInt(mt_randIF);
            permuts[j].sorting=j;
        }
        mergeSortTGC(permuts,r);
        for(j=0;j<r;++j){
            seqToFill[i*r+j]=permuts[j].sorting;
        }
    }
    
    currentSeedGC+=uInt(mt_randIF);
    delete[] permuts;
    return 1;
    
    
}


myint GraphicCard::createPascalTriangleGC(myint sizeM ){
    if(pascalTriangleGC!=nullptr){
        delete[] pascalTriangleGC;
    }
    
    myint size=sizeM;
    if(size<5){size=5;}
    *sizePascalTrGC=( (size +1)*(size +2) )/2;
    
    pascalTriangleGC=new myint[*sizePascalTrGC];
    myint index1,index2,index3;
    pascalTriangleGC[*sizePascalTrGC-1]=1;
    pascalTriangleGC[0]=1;
    pascalTriangleGC[1]=1;
    
    for(myint n=2;n <=size;++n){
        index1=( n*(n+1) )/2;
        pascalTriangleGC[index1]=1;
        pascalTriangleGC[index1-1]=1;
        index2=index1-n;
        for(myint k =1;k <n; ++k){
            //calculating  n choose k
            //Its index in the sequence is k+ n(n+1)/2
            // We will use that \binom nk=\binom{n-1}{k}+\binom{n-1}{k-1}
            
            pascalTriangleGC[index1+k]=pascalTriangleGC[index2+k]+pascalTriangleGC[index2+k-1];
        }
    }
    
    return 1;
    
}

myint GraphicCard::createBalancedNumbersGC(myint *seqBalNum,  myint *cNumber, myint* remZero, myint *remOne, myint *cCounter){
    if((*remZero==0) && (*remOne==0)){
        seqBalNum[*cCounter]=*cNumber;
        *cCounter+=1;
        
        return 1;
    }
    
    if(*remZero>0){
        *cNumber= (*cNumber) * 2;
        *remZero-=1;
        createBalancedNumbersGC(seqBalNum,cNumber,remZero,remOne,cCounter);
        *remZero+=1;
        *cNumber=(*cNumber)/2;
        
    }
    
    if(*remOne>0){
        *cNumber= (*cNumber) * 2 +1 ;
        *remOne-=1;
        createBalancedNumbersGC(seqBalNum,cNumber,remZero,remOne,cCounter);
        *remOne+=1;
        *cNumber=(*cNumber)/2;
        
        
    }
    
    return 1;
}




mysint GraphicCard::generateRandomNumbers(myint N,myint r,mysint inputIndPGen){
    myint indPGen=inputIndPGen;
    
    myint padding;
    padding=*preferred_workgroup_sizeGC;

   
    std::string *lArgRNK;
    myint numArg=12;
    lArgRNK=new std::string[numArg];
    lArgRNK[0]="rNumGCCL";
    lArgRNK[1]="xAxGCCL";
    lArgRNK[2]="yAxGCCL";
    lArgRNK[3]="permutationsXGCCL";
    lArgRNK[4]="permutationsYGCCL";
    lArgRNK[5]="powersOfTwoGCCL";
    lArgRNK[6]="balancedNumbersGCCL";
    lArgRNK[7]="pascalTGCCL";
    lArgRNK[8]="axisSizeGCCL";
    lArgRNK[9]="binaryLengthGCCL";
    lArgRNK[10]="numBalancedNumbersGCCL";
    lArgRNK[11]="shufflingPrimeGCCL";



    if((indPGen==1)||(N!=*axisSizeGC)||(r!=*lengthInBinaryGC)){
        indPGen=1;
        indFirstRNInitGC=0;
        if(r!=*lengthInBinaryGC){
            delete[]balancedNumbersGC;
            balancedNumbersGC=nullptr;
            *numBalancedNumbersGC=0;
            delete[]pascalTriangleGC;
            pascalTriangleGC=nullptr;
            *sizePascalTrGC=0;
        }
        
        for(myint i=0;i<numArg;++i){
            freeDeviceMemory(lArgRNK[i]);
        }
        
        
        *lengthInBinaryGC=r;
        *axisSizeGC=N;
        if(xAxisGC!=nullptr){
            delete[] xAxisGC;
        }
        xAxisGC=new myint[*axisSizeGC+padding];
        if(yAxisGC!=nullptr){
            delete[] yAxisGC;
        }
        yAxisGC=new myint[*axisSizeGC+padding];
        
        uniformLimitGC=*numBalancedNumbersGC-1;
        if(randNumbersGC==nullptr){
            delete[] randNumbersGC;
        }
        randNumbersGC=new myint[ (*axisSizeGC+padding)*(*axisSizeGC+padding)];
        for(myint i=0;i<(*axisSizeGC+padding)*(*axisSizeGC+padding);++i){
            randNumbersGC[i]=17;
        }
        if(xPermutationsGC!=nullptr){
            delete[] xPermutationsGC;
        }
        xPermutationsGC=new myint[(*axisSizeGC+padding)* (*lengthInBinaryGC)];
        if(yPermutationsGC!=nullptr){
            delete[] yPermutationsGC;
        }
        yPermutationsGC=new myint[(*axisSizeGC+padding)* (*lengthInBinaryGC)];
        
        if(powersOfTwoGC!=nullptr){
            delete[] powersOfTwoGC;
        }
        powersOfTwoGC=new myint[*lengthInBinaryGC];
        powersOfTwoGC[0]=1;
        for(myint k=1;k<*lengthInBinaryGC;++k){
            powersOfTwoGC[k]=2 * powersOfTwoGC[k-1];
        }
        
        
        generatePermutationsGC(xPermutationsGC,*axisSizeGC+padding,*lengthInBinaryGC);
        generatePermutationsGC(yPermutationsGC,*axisSizeGC+padding,*lengthInBinaryGC);
        
    }
    if(balancedNumbersGC==nullptr){
        myint **binMatrix;
        myint bigInt=*lengthInBinaryGC+2;
        binMatrix=new myint*[bigInt];
        for(myint i=0;i<bigInt;++i){
            binMatrix[i]=new myint[bigInt];
            for(myint j=0;j<bigInt;++j){
                binMatrix[i][j]=-1;
            }
        }
        
        createPascalTriangleGC(*lengthInBinaryGC+3);
        
        *numBalancedNumbersGC=pascalTriangleGC[ ((*lengthInBinaryGC)*(*lengthInBinaryGC+1))/2+ (*lengthInBinaryGC)/2];
        
        uniformLimitGC=*numBalancedNumbersGC-1;
        for(myint i=0;i<bigInt;++i){
            delete[] binMatrix[i];
        }
        delete[] binMatrix;
        balancedNumbersGC=new myint[*numBalancedNumbersGC];
        myint cNumber=0;
        myint remZero=(*lengthInBinaryGC)/2;
        myint remOne=remZero;
        myint cCounter=0;
        createBalancedNumbersGC(balancedNumbersGC,  &cNumber, &remZero, &remOne, &cCounter);
        
    }
    
    
    
    
    
    std::uniform_int_distribution<myint> uInt(0,uniformLimitGC);
    myint seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    seed+=currentSeedGC;
    std::mt19937 mt_randIF(seed);
    for(myint i=0;i<*axisSizeGC+padding;++i){
        
        xAxisGC[i]= uInt(mt_randIF);
        yAxisGC[i]= uInt(mt_randIF);
        
    }

    currentSeedGC+=uInt(mt_randIF);
    
    writeDeviceMemory("xAxGCCL",xAxisGC,*axisSizeGC+padding);
    writeDeviceMemory("yAxGCCL",yAxisGC,*axisSizeGC+padding);
    
    if(indPGen==1){
        writeDeviceMemory("permutationsXGCCL",xPermutationsGC,(*axisSizeGC+padding)*(*lengthInBinaryGC));
        writeDeviceMemory("permutationsYGCCL",yPermutationsGC,(*axisSizeGC+padding)*(*lengthInBinaryGC));
        writeDeviceMemory("powersOfTwoGCCL",powersOfTwoGC,*lengthInBinaryGC);
        writeDeviceMemory("balancedNumbersGCCL",balancedNumbersGC,*numBalancedNumbersGC);
        
        writeDeviceMemory("binaryLengthGCCL",lengthInBinaryGC,1);
        writeDeviceMemory("shufflingPrimeGCCL",shufflingPrimeGC,1);
        writeDeviceMemory("numBalancedNumbersGCCL",numBalancedNumbersGC,1);
        writeDeviceMemory("pascalTGCCL",pascalTriangleGC,*sizePascalTrGC);
        
        
        writeDeviceMemory("axisSizeGCCL",axisSizeGC,1);
        
    }
    
    
    
    
    
    if(indFirstRNInitGC==0){
        indPGen=1;
        indFirstRNInitGC=1;
        writeDeviceMemory("rNumGCCL",randNumbersGC,(*axisSizeGC+padding)*(*axisSizeGC+padding));
        
        findAddKernel("genMainRandomMatrixGC");
        

        setKernelArguments("genMainRandomMatrixGC",lArgRNK,numArg);
        
        
    }
    delete[] lArgRNK;
    
    
    executeKernel("genMainRandomMatrixGC", (*axisSizeGC)*(*axisSizeGC));
    
    return 1;
}

mysint GraphicCard::setLocationNameForRandomNumbers(std::string newName){
    randSampGC=newName;
    sampleGenKernExecutedGC=0;
    lastRandAlgUsedGC=-1;
    return 1;
}

mysint GraphicCard::generateNormalBeasleySpringerMoro(myint N,myint r,  myint prec1ReqInput,
                                                      myint *sampleLength,
                                                      cl_double par1, cl_double par2, mysint inputIndPGen){
   
    if((antitheticGC==1)||(overrideAntiteticGC==0)){
        
        generateRandomNumbers(N,r,inputIndPGen);
        
        
        mysint inspResFun=0;
        myint precReq=  prec1ReqInput;
        
        myint type=7;
        
        
        *highestPrecisionGC=powerGC(*numBalancedNumbersGC, prec1ReqInput)-1;
        
        *precisionRequestGC=precReq;
        
        *sizeForRejectionSamplingGC=precReq;
        
        *sampleLength= (N * N) / (*sizeForRejectionSamplingGC);
        if((*sampleLength!=*sampleLengthGC)||(lastRandAlgUsedGC!=type)){
            lastRandAlgUsedGC=type;
            delete[] randomSampleGC;
            inspectorExecutedGC=0;
            sampleGenKernExecutedGC=0;
            *sampleLengthGC=*sampleLength;
            randomSampleGC=new cl_double[*sampleLengthGC];
            
            writeDeviceMemory(randSampGC,randomSampleGC,*sampleLengthGC);
            writeDeviceMemory("sampleLengthGCCL",sampleLengthGC,1);
            writeDeviceMemory("sizeRejectionSamplingGCCL",sizeForRejectionSamplingGC,1);
            writeDeviceMemory("precReqGCCL",precisionRequestGC,1);
            writeDeviceMemory("par1GCCL",&par1,1);
            writeDeviceMemory("par2GCCL",&par2,1);
            writeDeviceMemory("numBalancedNumbersGCCL",numBalancedNumbersGC,1);
            writeDeviceMemory("biggestNumGCCL",highestPrecisionGC,1);
            
            cl_double *a,*b,*c;
            myint numAB=4, numC=9;
            a=new cl_double[numAB];
            b=new cl_double[numAB];
            c=new cl_double[numC];
            a[0]=2.50662823884;
            a[1]=-18.61500062529;
            a[2]=41.39119773534;
            a[3]=-25.44106049637;
            
            b[0]=-8.47351093090;
            b[1]=23.08336743743;
            b[2]=-21.06224101826;
            b[3]=3.13082909833;
            
            c[0]=0.3374754822726147;
            c[1]=0.9761690190917186;
            c[2]=0.1607979714918209;
            c[3]=0.0276438810333863;
            c[4]=0.0038405729373609;
            c[5]=0.0003951896511919;
            c[6]=0.0000321767881768;
            c[7]=0.0000002888167364;
            c[8]=0.0000003960315187;
            writeDeviceMemory("seqAGCCL",a,numAB);
            writeDeviceMemory("seqBGCCL",b,numAB);
            writeDeviceMemory("seqCGCCL",c,numC);
            writeDeviceMemory("numABGCCL",&numAB,1);
            writeDeviceMemory("numCGCCL",&numC,1);
            delete[]a;
            delete[]b;
            delete[]c;
        }
        
        
        std::string *lArgRNK;
        myint numArg=14;
        lArgRNK=new std::string[numArg];
        lArgRNK[0]=randSampGC;
        lArgRNK[1]="rNumGCCL";
        lArgRNK[2]="sampleLengthGCCL";
        lArgRNK[3]="sizeRejectionSamplingGCCL";
        lArgRNK[4]="precReqGCCL";
        lArgRNK[5]="par1GCCL";
        lArgRNK[6]="par2GCCL";
        lArgRNK[7]="numBalancedNumbersGCCL";
        lArgRNK[8]="biggestNumGCCL";
        lArgRNK[9]="seqAGCCL";
        lArgRNK[10]="seqBGCCL";
        lArgRNK[11]="seqCGCCL";
        lArgRNK[12]="numABGCCL";
        lArgRNK[13]="numCGCCL";
        if(sampleGenKernExecutedGC==0){
            sampleGenKernExecutedGC=1;
            findAddKernel(normalBSMNameGC);
            setKernelArguments(normalBSMNameGC,lArgRNK,numArg);
            
        }
        
        executeKernel(normalBSMNameGC,*sampleLengthGC);
        
        
        delete[] lArgRNK;
        antitheticGC=-1;
    }
    else{
        std::string *lArgRNK;
        myint numArg=14;
        lArgRNK=new std::string[numArg];
        lArgRNK[0]=randSampGC;
        lArgRNK[1]="rNumGCCL";
        lArgRNK[2]="sampleLengthGCCL";
        lArgRNK[3]="sizeRejectionSamplingGCCL";
        lArgRNK[4]="precReqGCCL";
        lArgRNK[5]="par1GCCL";
        lArgRNK[6]="par2GCCL";
        lArgRNK[7]="numBalancedNumbersGCCL";
        lArgRNK[8]="biggestNumGCCL";
        lArgRNK[9]="seqAGCCL";
        lArgRNK[10]="seqBGCCL";
        lArgRNK[11]="seqCGCCL";
        lArgRNK[12]="numABGCCL";
        lArgRNK[13]="numCGCCL";
        findAddKernel(normalBSMAntNameGC);
        setKernelArguments(normalBSMAntNameGC,lArgRNK,numArg);
        executeKernel(normalBSMAntNameGC,*sampleLengthGC);
        
        delete[] lArgRNK;
        antitheticGC=1;
    }
    
    return 1;
    
}





mysint GraphicCard::generateExponential(myint N,myint r,  myint prec1ReqInput, myint *sampleLength,
                                        cl_double lambda, mysint inputIndPGen){
    generateRandomNumbers(N,r,inputIndPGen);
    mysint inspResFun=0;
    myint precReq=  prec1ReqInput;
    
    myint type=8;
    
    
    *highestPrecisionGC=powerGC(*numBalancedNumbersGC, prec1ReqInput)-1;
    
    *precisionRequestGC=precReq;
    
    *sizeForRejectionSamplingGC=precReq;
    
    *sampleLength= (N * N) / (*sizeForRejectionSamplingGC);
    if((*sampleLength!=*sampleLengthGC)||(lastRandAlgUsedGC!=type)){
        lastRandAlgUsedGC=type;
        
        delete[] randomSampleGC;
        sampleGenKernExecutedGC=0;
        *sampleLengthGC=*sampleLength;
        randomSampleGC=new cl_double[*sampleLengthGC];
        
        writeDeviceMemory(randSampGC,randomSampleGC,*sampleLengthGC);
        writeDeviceMemory("sampleLengthGCCL",sampleLengthGC,1);
        writeDeviceMemory("sizeRejectionSamplingGCCL",sizeForRejectionSamplingGC,1);
        writeDeviceMemory("precReqGCCL",precisionRequestGC,1);
        writeDeviceMemory("par2GCCL",&lambda,1);
 
        writeDeviceMemory("numBalancedNumbersGCCL",numBalancedNumbersGC,1);
        writeDeviceMemory("biggestNumGCCL",highestPrecisionGC,1);
        
    }
    
    
    std::string *lArgRNK;
    myint numArg=8;
    lArgRNK=new std::string[numArg];
    lArgRNK[0]=randSampGC;
    lArgRNK[1]="rNumGCCL";
    lArgRNK[2]="sampleLengthGCCL";
    lArgRNK[3]="sizeRejectionSamplingGCCL";
    lArgRNK[4]="precReqGCCL";
    lArgRNK[5]="par2GCCL";
    lArgRNK[6]="numBalancedNumbersGCCL";
    lArgRNK[7]="biggestNumGCCL";
    if(sampleGenKernExecutedGC==0){
        sampleGenKernExecutedGC=1;
        findAddKernel(exponentialDistNameGC);
        setKernelArguments(exponentialDistNameGC,lArgRNK,numArg);
        
    }
    
    executeKernel(exponentialDistNameGC,*sampleLengthGC);
    
    
    delete[] lArgRNK;
    
    
    return 1;
    
}

template <typename int_doub> myint GraphicCard::createFenwickTree(std::string treeName,
                                                                   std::string lengthKeeper,
                                                                   int_doub *inputSeq, myint length){
    myint i;
    myint powerOfTwoGreaterThanLength=1;
    while(powerOfTwoGreaterThanLength<length){
        powerOfTwoGreaterThanLength*=2;
    }
    
    int_doub *seqToCopy;
    if(powerOfTwoGreaterThanLength>length){
        seqToCopy=new int_doub[powerOfTwoGreaterThanLength];
        for(i=0;i<length;++i){
            seqToCopy[i]=inputSeq[i];
        }
        while(i<powerOfTwoGreaterThanLength){
            seqToCopy[i]=int_doub(0);
            ++i;
        }
    }
    else{
        seqToCopy=inputSeq;
    }
    std::string *lArgRNK;
    myint numArg=3;
    lArgRNK=new std::string[numArg];
    lArgRNK[0]=treeName;
    lArgRNK[1]=lengthKeeper;
    lArgRNK[2]="layerSkipGCCL";
    

    
    
    myint layer=0, layerSkip=1, layerSkip2;
    
    writeDeviceMemory(treeName,seqToCopy,powerOfTwoGreaterThanLength);
    writeDeviceMemory(lengthKeeper,&powerOfTwoGreaterThanLength,1);
    writeDeviceMemory("layerSkipGCCL",&layerSkip,1);
    
    
    findAddKernel(createFenwickGC);
    
    
    setKernelArguments(createFenwickGC,lArgRNK,numArg);

    layerSkip2=2*layerSkip;
    while(layerSkip2<=powerOfTwoGreaterThanLength){
        
        
        writeDeviceMemory("layerSkipGCCL",&layerSkip,1);
        executeKernel(createFenwickGC,powerOfTwoGreaterThanLength/layerSkip2);
        layerSkip=layerSkip2;
        layerSkip2*=2;
    }
    
    
    if(powerOfTwoGreaterThanLength>length){
        delete[] seqToCopy;
    }
    delete[] lArgRNK;
    

    return powerOfTwoGreaterThanLength;
}




cl_double GraphicCard::justSummation(std::string seqName,
                                     std::string lengthKeeper,
                                     myint length,
                                     myint avInd){
    myint i;
    myint powerOfTwoGreaterThanLength=1;
    while(powerOfTwoGreaterThanLength<length){
        powerOfTwoGreaterThanLength*=2;
    }
    
    
    std::string *lArgRNK;
    myint numArg=4;
    lArgRNK=new std::string[numArg];
    lArgRNK[0]=seqName;
    lArgRNK[1]=lengthKeeper;
    lArgRNK[2]="layerSkipGCCL";
    lArgRNK[3]="multiplierFnwGCCL";
    
    
    
    myint layer=0, layerSkip=1, layerSkip2;
    cl_double mult=1.0;
    
    if(avInd==1){
        mult*=cl_double(powerOfTwoGreaterThanLength)/cl_double(2*length);
    }
    writeDeviceMemory(lengthKeeper,&length,1);
    
    writeDeviceMemory("layerSkipGCCL",&layerSkip,1);
    writeDeviceMemory("multiplierFnwGCCL",&mult,1);
    
    findAddKernel(justSummationFenwickGC);
    
    
    setKernelArguments(justSummationFenwickGC,lArgRNK,numArg);
    
    layerSkip2=2*layerSkip;
    while(layerSkip2<=powerOfTwoGreaterThanLength){
        
        writeDeviceMemory("layerSkipGCCL",&layerSkip,1);

        executeKernel(justSummationFenwickGC,powerOfTwoGreaterThanLength/layerSkip2);
        layerSkip=layerSkip2;
        if((avInd==1)&&(layerSkip2==2)){
            mult=0.5;
            writeDeviceMemory("multiplierFnwGCCL",&mult,1);
        }
        layerSkip2*=2;
    }
    
    cl_double fR;
    
    readDeviceMemory(seqName,&fR,1);
 
    
    delete[] lArgRNK;
    
    
    return fR;
}



myint GraphicCard::partialSumsFTree(std::string treeName,
                                     std::string lengthKeeper,
                                     std::string fromSeqN,
                                     std::string toSeqN,
                                     std::string resultN,
                                     myint length){
    findAddKernel(partialSumsFTreeGC);
    myint numArg=5;
    std::string *lArgRNK;
    
    lArgRNK=new std::string[numArg];
    lArgRNK[0]=treeName;
    lArgRNK[1]=lengthKeeper;
    lArgRNK[2]=fromSeqN;
    lArgRNK[3]=toSeqN;
    lArgRNK[4]=resultN;
    writeDeviceMemory(lengthKeeper,&length,1);
    setKernelArguments(partialSumsFTreeGC,lArgRNK,numArg);
    executeKernel(partialSumsFTreeGC,length);
    delete[] lArgRNK;
    
    return 1;
}


myint GraphicCard::mergeSort(std::string stSeqName, myint nRAM,
                             myint direction, std::string resSeqName){
    myint potentialMemoryLeak=0;
    if(resSeqName=="not provided"){
        
        resSeqName="resSeqMergSortGCCL";
        if(resSeqName==stSeqName){
            resSeqName="resSeqMergSortGCCLRB";
        }
        
        cl_double *auxSeq;
        auxSeq=new cl_double[nRAM];
        writeDeviceMemory(resSeqName,auxSeq,nRAM);
        delete[] auxSeq;
        potentialMemoryLeak=1;
        
    }
    
    std::string lengthName="lenMergSortGCCL";
    std::string sortedSizeN="sortedSizeMergSortGCCL";
    std::string correctKernel=shiftCalculationMergSortIncGC;
    if(direction==1){
        correctKernel=shiftCalculationMergSortDecGC;
    }
    std::string switcher;
    myint numArg=4;
    std::string *lArgRNK;
    
    lArgRNK=new std::string[numArg];
    lArgRNK[0]=stSeqName;
    lArgRNK[1]=lengthName;
    lArgRNK[2]=resSeqName;
    lArgRNK[3]=sortedSizeN;
    findAddKernel(correctKernel);
    findAddKernel(copyResultToStartGC);
    writeDeviceMemory(lengthName,&nRAM,1);
    
    myint sortSize=1;
    while(sortSize<nRAM){
        writeDeviceMemory(sortedSizeN,&sortSize,1);
        setKernelArguments(correctKernel,lArgRNK,numArg);
        executeKernel(correctKernel,nRAM);
        switcher=lArgRNK[0];
        lArgRNK[0]=lArgRNK[2];
        lArgRNK[2]=switcher;
        
        sortSize*=2;
    }
    
    if(lArgRNK[2]==stSeqName){
        myint numArg2=3;
        
        std::string *lArgRNK2;
        lArgRNK2=new std::string[numArg2];
        lArgRNK2[0]=stSeqName;
        lArgRNK2[1]=lengthName;
        lArgRNK2[2]=resSeqName;
        setKernelArguments(copyResultToStartGC,lArgRNK2,numArg2);
        executeKernel(copyResultToStartGC,nRAM);
        delete[] lArgRNK2;
    }

    
    delete[] lArgRNK;
    if(potentialMemoryLeak==1){
        freeDeviceMemory(resSeqName);
    }
    return 1;
    
}

myint GraphicCard::mergeSortWithFollower(std::string stSeqName,std::string followerName,
                                         myint nRAM,
                                         myint direction,
                                         std::string resSeqName, std::string resFollName){
    myint potentialMemoryLeak=0;
    if(resSeqName=="not provided"){
        
        resSeqName="resSeqMergSortGCCL";
        resFollName="resSeqFollGCCL";
        if(resSeqName==stSeqName){
            resSeqName="resSeqMergSortGCCLRB";
        }
        if(resFollName==followerName){
            resFollName="resSeqFollGCCLRB";
        }
        
        cl_double *auxSeq;
        myint *auxSeqFoll;
        auxSeq=new cl_double[nRAM];
        auxSeqFoll=new myint[nRAM];
        writeDeviceMemory(resSeqName,auxSeq,nRAM);
        writeDeviceMemory(resFollName,auxSeqFoll,nRAM);
        delete[] auxSeq;
        delete[] auxSeqFoll;
        potentialMemoryLeak=1;
        
    }
    
    std::string lengthName="lenMergSortGCCL";
    std::string sortedSizeN="sortedSizeMergSortGCCL";
    std::string correctKernel=shiftCalculationMergSortFollIncGC;
    if(direction==1){
        correctKernel=shiftCalculationMergSortFollDecGC;
    }
    std::string switcher;
    myint numArg=6;
    std::string *lArgRNK;
    
    lArgRNK=new std::string[numArg];
    lArgRNK[0]=stSeqName;
    lArgRNK[1]=lengthName;
    lArgRNK[2]=resSeqName;
    lArgRNK[3]=sortedSizeN;
    lArgRNK[4]=followerName;
    lArgRNK[5]=resFollName;
    
    findAddKernel(correctKernel);
    findAddKernel(copyResultToStartGC);
    writeDeviceMemory(lengthName,&nRAM,1);
    
    myint sortSize=1;
    while(sortSize<nRAM){
        writeDeviceMemory(sortedSizeN,&sortSize,1);
        setKernelArguments(correctKernel,lArgRNK,numArg);
        executeKernel(correctKernel,nRAM);
        switcher=lArgRNK[0];
        lArgRNK[0]=lArgRNK[2];
        lArgRNK[2]=switcher;
        switcher=lArgRNK[4];
        lArgRNK[4]=lArgRNK[5];
        lArgRNK[5]=switcher;
        sortSize*=2;
    }
    
    if(lArgRNK[2]==stSeqName){
        myint numArg2=5;
        
        std::string *lArgRNK2;
        lArgRNK2=new std::string[numArg2];
        lArgRNK2[0]=stSeqName;
        lArgRNK2[1]=lengthName;
        lArgRNK2[2]=resSeqName;
        lArgRNK2[3]=followerName;
        lArgRNK2[4]=resFollName;
        setKernelArguments(copyResultToStartFollGC,lArgRNK2,numArg2);
        executeKernel(copyResultToStartFollGC,nRAM);
        delete[] lArgRNK2;
    }
    
    
    delete[] lArgRNK;
    if(potentialMemoryLeak==1){
        freeDeviceMemory(resSeqName);
        freeDeviceMemory(resFollName);
    }
    return 1;
    
}
myint GraphicCard::setParametersPSS(std::string origSeqN, std::string pSSN,
                                    std::string lengthN, std::string blSizN,
                                    myint length,myint blSize){
    origNamePSSGC=origSeqN;
    namePSSGC=pSSN;
    lengthNamePSSGC=lengthN;
    nameBlockSizePSSGC=blSizN;
    blockSizePSSGC=blSize;
    lengthValuePSSGC=length;
    writeDeviceMemory(lengthNamePSSGC,&lengthNamePSSGC,1);
    writeDeviceMemory(nameBlockSizePSSGC,&blockSizePSSGC,1);
    
    return 1;
}

myint GraphicCard::createPSS(){
    
    if(lengthValuePSSGC>0){
        //STAGE 1
        
        findAddKernel(kernelNameStage1PSSGC);
        myint counter=0;
        writeDeviceMemory(posVariableNamePSSGC,&counter,1);
        myint numArg=5;
        std::string *lArgRNK;
        
        lArgRNK=new std::string[numArg];
        lArgRNK[0]=origNamePSSGC;
        lArgRNK[1]=namePSSGC;
        lArgRNK[2]=lengthNamePSSGC;
        lArgRNK[3]=nameBlockSizePSSGC;
        lArgRNK[4]=posVariableNamePSSGC;
        
        setKernelArguments(kernelNameStage1PSSGC,lArgRNK,numArg);
        myint kernelsNeededStage1= lengthValuePSSGC/blockSizePSSGC;
        if(lengthValuePSSGC%blockSizePSSGC>0){
            kernelsNeededStage1+=1;
        }
        for(counter=0;counter<blockSizePSSGC;++counter){
            writeDeviceMemory(posVariableNamePSSGC,&counter,1);
            executeKernel(kernelNameStage1PSSGC,kernelsNeededStage1);
        }
        
        
        //STAGE 2
        findAddKernel(kernelNameStage2PSSGC);
        setKernelArguments(kernelNameStage2PSSGC,lArgRNK,numArg);
        for(counter=0;counter<kernelsNeededStage1;++counter){
            writeDeviceMemory(posVariableNamePSSGC,&counter,1);
            executeKernel(kernelNameStage2PSSGC,blockSizePSSGC);
        }
        
        delete[] lArgRNK;
    }
    else{
        std::cout<<"Set-up not performed correctly. Could not create PSS"<<std::endl;
        return 0;
    }
    return 1;
}

myint GraphicCard::createPSS(std::string origSeqN, std::string pSSN, std::string lengthN, std::string blSizN, myint length, myint blSize){
    setParametersPSS(origSeqN, pSSN, lengthN,blSizN, length,blSize);
    return createPSS();
}


myint GraphicCard::partialSumsPSS(std::string pssName,
                                  std::string lengthKeeper,
                                  std::string fromSeqN,
                                  std::string toSeqN,
                                  std::string resultN,
                                  myint length){
    
    findAddKernel(partialSumsPSSGC);
    myint numArg=5;
    std::string *lArgRNK;
    
    lArgRNK=new std::string[numArg];
    lArgRNK[0]=pssName;
    lArgRNK[1]=lengthKeeper;
    lArgRNK[2]=fromSeqN;
    lArgRNK[3]=toSeqN;
    lArgRNK[4]=resultN;
    writeDeviceMemory(lengthKeeper,&length,1);
    setKernelArguments(partialSumsPSSGC,lArgRNK,numArg);
    executeKernel(partialSumsPSSGC,length);
    delete[] lArgRNK;
    
    return 1;
}
myint GraphicCard::getPrefferedWorkGroupSize() const{
    return static_cast<myint>( *preferred_workgroup_sizeGC);
}
 
