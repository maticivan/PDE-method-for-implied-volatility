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
#ifndef INDEX_STRUCTURE_H
#define INDEX_STRUCTURE_H

#include <cmath>
#include <vector>
#include <iostream>
#include <set>
#include <map>
#include <iomanip>
#include <random>
#include <chrono>
#include <fstream>
 


typedef long int myint;
typedef double mydouble;

myint getOrderOfMagnitude(const mydouble & );
myint getOrderOfMagnitudeAndModifyMe(mydouble & );
class IndexStructure{
private:
    std::vector<myint> resolutions;
    std::vector<std::vector<myint> > allInd;
    int allIndSize;
    mydouble maxElReduction=1.0;
    mydouble maxElMultiplier=1.0;
    mydouble maxEl=1.0;
    myint getOrderOfMagnitude(const mydouble & ) const;
    int getOrderOfMagnitudeAndModifyMe(mydouble & ) const;
    
    myint oMMultSize;
    std::vector<myint> oMMultipliersInt;
    std::vector<myint> exponents;
    std::vector<myint> resolutionsInt;
    mydouble getMaxEl(const std::vector<mydouble> & ) const;
    void createIndicesOfSeqWhoseTermsAreSmallerThanOne(const std::vector<mydouble> &);
    
public:
    IndexStructure();
    myint getResolution(const myint &) const;
    myint getElAllInd(const myint &, const myint &) const;
    
    void setResolution(const myint &, const myint &);
    void setElAllInd(const myint &, const myint &, const myint &);
    
    void createIndices(const std::vector<mydouble> &);
    std::vector<myint> getExponents() const;
    std::vector<std::vector<myint> > getAllInd() const;
    myint getIndexOfSeq(const mydouble &) const;
    mydouble getMaxElReduction() const;
    mydouble getMaxElMultiplier() const;
    myint size() const;
    void clear();
};
#endif
