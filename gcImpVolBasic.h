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
#ifndef GCIMPVOLBASIC_H
#define GCIMPVOLBASIC_H

#include <cmath>
#include <vector>
#include <iostream>
#include <set>
#include <map>
#include <iomanip>
#include <random>
#include <chrono>
#include <fstream>
#include "indexStructure.h"
#include "cImpVolBasic.h"
#include "GCOpenCL.h"

typedef long int myint;
typedef double mydouble;


class GCImpVol:public CImpVol{
    
    
public:
    
    GCImpVol();
    ~GCImpVol();
    
    cl_double* ivGC(cl_double * &,
                    cl_double * &,
                    cl_double * &,
                    cl_double * &,
                    cl_double * &,
                    myint * &,
                    myint);
    
    
    myint* prepareIndStructureForGC(const IndexStructure & );
   
    cl_double* evaluateGC(const myint &);
    // assumes that k_Fukasawa and c_Fukasawa are already on graphic card
    
private:
    
    GraphicCard *gCard;
    myint graphicCardInitialized;
    
    myint initializeGraphicCard();
    myint initializeGC();
    
    
    template <typename int_doub> myint extendSeq(int_doub * &, const myint &, const myint &);
    
   
    
    myint lastCalculationSize=0;
    
    std::string name_numberOfSamplesGC;
    std::string name_FFGC;
    std::string name_pxGC;
    std::string name_KKGC;
    std::string name_rrGC;
    std::string name_texpGC;
    std::string name_PCGC;
    std::string name_PCGCCL;
    std::string name_k_FukasawaGC;
    std::string name_c_FukasawaGC;
    std::string nameFractionsGC;
    
    std::string nameLengthIndStruKGC;
    std::string nameIndStruKGC;
    
    std::string nameLengthIndicesKGC;
    std::string nameIndicesKGC;
    std::string nameIndicesXGC;
    
    std::string nameSeqShiftXGC;
    std::string nameSeqXTotalGC;
    std::string nameLenSeqShiftXGC;
    std::string nameLenSeqXTotalGC;
    
    
    std::string substitutionKernelNameGC;
    std::string getIndexStructureForKKernelGC;
    std::string getIndicesKernelGC;
    std::string approxPolsKernelGC;
    
    std::string namePolCoeffsGC;
    std::string namePartKLengthGC;
    std::string namePartKSeqGC;
    std::string namePartXSeqGC;
    std::string nameShiftSeqPartXGC;
    std::string nameShiftSequenceKXGC;
    std::string nameDegKGC;
    std::string nameDegXGC;
    std::string nameVectorOfResultsGC;
    std::string nameNumCoeffsInPolGC;
    std::string nameMaxElReductionKGC;
    std::string nameMaxElMultiplierKGC;
    std::string nameMaxElReductionSeqXGC;
    std::string nameMaxElMultiplierSeqXGC;
    
};
#endif
