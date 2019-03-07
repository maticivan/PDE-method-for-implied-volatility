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
#ifndef CIMPVOLBASIC_H
#define CIMPVOLBASIC_H

#include <cmath>
#include <vector>
#include <iostream>
#include <set>
#include <map>
#include <iomanip>
#include <random>
#include <chrono>
#include <fstream>
#include <omp.h>
#include "indexStructure.h"

 


typedef long int myint;
typedef double mydouble;

const std::string CIMPV_CONST_DEFAULT_FILE_WITH_PARTITION = "loadPartition.txt";

const myint       CIMPV_CONST_INT_START_OF_PARTITION_DATA = 314159265;

const mydouble    CIMPV_CONST_DEFAULT_PRECISION_FOR_QNORM = 0.00000000000001;
const mydouble    CIMPV_CONST_DEFAULT_PRECISION_FOR_YBS   = 0.0000000001;

const mydouble    CIMPV_CONST_DEFAULT_K_THRESHOLD_SIGMA_SR   = 0.00000000001;

const mydouble    CIMPV_CONST_DEFAULT_CL_THRESHOLD_SIGMA_SR   = 0.01;
const mydouble    CIMPV_CONST_DEFAULT_CU_THRESHOLD_SIGMA_SR   = 0.031;
class CImpVol{
    friend class GCImpVol;
    
public:
    
    CImpVol(const myint & = 0);
    virtual ~CImpVol();
    mydouble evaluate(const mydouble &, const mydouble &);
    
    //  Arguments:
    //      1) const mydouble &k, 2) const mydouble &x
    
    std::vector<mydouble> evaluate(const std::vector<mydouble> &, const std::vector<mydouble> &);
    
    //  Arguments:
    //      1) const mydouble &k, 2) const mydouble &x
    
    
    mydouble sigma_call(const mydouble & , const mydouble & , const mydouble & , const mydouble & , const mydouble & );
    //  Arguments:
    //  1) FF,   2) C0,   3) KK,   4) rr,   5) texp
    
    mydouble  iv(const mydouble & ,
                 const mydouble & ,
                 const mydouble & ,
                 const mydouble & ,
                 const mydouble & ,
                 const myint    & = 1,
                 const myint & = 0);
    
    
    //  Arguments:
    //  1) FF,       2) px,       3) KK,       4) rr,       5) texp,     6) PC (1=call, 0=put)
    

    
    std::vector<mydouble>  iv(const std::vector<mydouble> & ,
                              const std::vector<mydouble> & ,
                              const std::vector<mydouble> & ,
                              const std::vector<mydouble> & ,
                              const std::vector<mydouble> & ,
                              const std::vector<myint>   & ,
                              const myint & =0 );

    
    std::vector<mydouble> getIntervalBounds() const;
    // Returns the vector with 4 components that are respectively:
    // cMin, cMax, kMin, kMax
    
    mydouble CBS(const mydouble &, const mydouble &,
                 const mydouble & =-1.0) const;
    
    myint ii_numberOfElementsInK() const;
    
    void ii_createIndexStructures();
    
    std::vector<myint> ii_getIndices(const mydouble &, const mydouble &) const;
    void loadData();
private:
    
    
    
    
    myint getNextInteger(char * , std::streampos , std::streampos *);
    mydouble getNextReal(char * , std::streampos , std::streampos *);
    void getBigString(const std::string & , char * & , std::streampos & );
    
    void totalPartition(mydouble *, myint);
    void createPolynomials2D(mydouble *, myint);
    
    
    std::vector<mydouble> createAlpha(mydouble *, myint *, const myint &, const myint &);
    mydouble evaluateYPol2D(const mydouble &,const mydouble &,const mydouble &,
                            const mydouble &,const std::vector<mydouble> &) const;
    mydouble evaluateYPol2D(const mydouble &,const mydouble &,
                            const std::vector<mydouble> &) const;
    mydouble square_md(const mydouble &) const;
    mydouble Af(const mydouble &) const;
    mydouble evaluateSigmaSR(const mydouble &, const mydouble &,
                             const mydouble &,const  mydouble &) const;

    myint degreeK;
    myint degreeX;
    mydouble yMin;
    mydouble yMax;
    mydouble kMin;
    mydouble kMax;
    mydouble errorA;
    myint dKP1;
    myint dKP1M;
    
    
    mydouble randomY_optional;
    mydouble randomK_optional;
    
    std::vector<mydouble> partK;
    
    std::vector<std::vector<mydouble> > pKX;
    std::vector<std::vector<mydouble> > pKY;
    
    std::vector< std::vector<std::vector<mydouble > > >  coeff;
    
    
    
    IndexStructure indStruForK;
    std::vector<IndexStructure> indStruForX;
    
    
    
};
#endif
