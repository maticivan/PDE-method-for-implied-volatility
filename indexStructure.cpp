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

#include "indexStructure.h"

myint IndexStructure::getOrderOfMagnitude(const mydouble & d) const{
    int fR;
    
    frexp(d,&fR);
    fR=1-fR;
    return fR;
    
    
}

int IndexStructure::getOrderOfMagnitudeAndModifyMe(mydouble & d1) const{
    int fR;
    
    frexp(d1,&fR);
    fR=1-fR;
    d1=ldexp(d1,exponents[fR]);
    return fR;
    
    
}

mydouble IndexStructure::getMaxElReduction() const{
    return maxElReduction;
}
mydouble IndexStructure::getMaxElMultiplier() const{
    return maxElMultiplier;
}
IndexStructure::IndexStructure(){
    resolutions.clear();
    allInd.clear();
    oMMultSize=32;
    
    oMMultipliersInt.resize(oMMultSize);
    
    oMMultipliersInt[0]=1;
    
    
    for(myint i=1;i<oMMultSize;++i){
        oMMultipliersInt[i] = oMMultipliersInt[i-1]+oMMultipliersInt[i-1];
        
    }
    
    
}

myint IndexStructure::getResolution(const myint & i) const{
    return resolutions[i];
}
myint IndexStructure::getElAllInd(const myint & i,
                                  const myint & j) const{
    return allInd[i][j];
}

void IndexStructure::setResolution(const myint & i,
                                   const myint & _s){
    resolutions[i]=_s;
}
void IndexStructure::setElAllInd(const myint & i,
                                 const myint & j,
                                 const myint & _s){
    allInd[i][j]=_s;
}
mydouble IndexStructure::getMaxEl(const std::vector<mydouble> &a) const{
    mydouble maxEl=-1.0;
    myint len=a.size();
    if(len>0){
        maxEl=a[0];
        myint i=1;
        while(i<len){
            if(a[i]>maxEl){
                maxEl=a[i];
            }
            ++i;
        }
    }
    return maxEl;
}
void IndexStructure::createIndicesOfSeqWhoseTermsAreSmallerThanOne(const std::vector<mydouble> & a){
    myint iA;
    myint sA=a.size();
    
    //ORDERS OF MAGNITUDE
    myint currentOM, myOM, maxOM, startingIndex;
    
    
    
    mydouble tempVal;
    maxOM=getOrderOfMagnitude(a[0]);
    resolutions.resize(maxOM+1);
    allInd.resize(maxOM+1);
    currentOM=maxOM;
    
    myint currentResolution=1;
    myint currentLen,cou;
    myint cIndex,tempIntPart;
    startingIndex=0;
    mydouble multiplierOM=1.0, resMult=1.0,diff,topOM;
    myint myOMNext,countNext;
    myint xWait;
    for(myint in=0;in<currentOM;++in){
        multiplierOM+=multiplierOM;
    }
    for(iA=1;iA<sA;++iA){
        myOM=getOrderOfMagnitude(a[iA]);
        if(myOM==currentOM){
            // order of magnitude is the same as before
            //
            
            diff=a[iA]-a[iA-1];
            diff*= multiplierOM;
            diff*= resMult;
            while(static_cast<myint>(diff)==0){
                diff+=diff;
                resMult+=resMult;
                currentResolution+=currentResolution;
            }
            
            if(iA<sA-1){
                // check whether we are at the end of order of magnitude
                myOMNext=getOrderOfMagnitude(a[iA+1]);
                
                if(myOMNext<myOM){
                    topOM=1.0;
                    for(countNext=0;countNext<myOM-1;++countNext){
                        topOM*=0.5;
                    }
                    diff=topOM- a[iA];
                    diff*= multiplierOM;
                    diff*= resMult;
                    
                    while(static_cast<myint>(diff)==0){
                        diff+=diff;
                        resMult+=resMult;
                        currentResolution+=currentResolution;
                        
                    }
                }
            }
            
            
        }
        if((myOM!=currentOM)||(iA==sA-1)){
            // order of magnitude is changing.
            
            
            
            currentResolution+=currentResolution;
            
            resMult+=resMult;
            resolutions[currentOM]=currentResolution;
            if(currentOM!=0){
                currentLen=resolutions[currentOM];
            }
            else{
                tempIntPart= static_cast<myint>(a[sA-1]) +1;
                tempIntPart *= resolutions[currentOM];
                currentLen=tempIntPart;
            }
            if((currentLen>0)&&(resolutions[currentOM]>0)){
                
                
                allInd[currentOM].resize(currentLen);
                cIndex=startingIndex;
                for(cou=0;cou<currentLen;++cou){
                    tempVal=static_cast<mydouble>(cou);
                    tempVal/= multiplierOM * static_cast<mydouble>(resolutions[currentOM]);
                    
                    tempVal += tempVal;
                    
                    if((cIndex<sA)&&(tempVal>=a[cIndex])){
                        ++cIndex;
                    }
                    allInd[currentOM][cou]=-1;
                    if((tempVal<a[cIndex])&&((cIndex==0)||(tempVal>=a[cIndex-1]))){
                        allInd[currentOM][cou]=cIndex-1;
                        if(cIndex==0){
                            allInd[currentOM][cou]=0;
                        }
                    }
                    
                    if((currentOM==0)&&(tempVal>=a[sA-1])){
                        allInd[currentOM][cou]=sA-1;
                    }
                    
             
                    
                }

                
                
            }
            else{
                
                currentLen=2;
                allInd[currentOM].resize(currentLen);
                resolutions[currentOM]=2;
                
                allInd[currentOM][0]=-19;
                if(currentOM+1<allInd.size()){
                    myint lastInd=allInd[currentOM+1].size();
                    if(lastInd>0){
                        allInd[currentOM][0]=allInd[currentOM+1][lastInd-1];
                        allInd[currentOM][1]=allInd[currentOM][0];
                    }
                }
             }
            currentOM=myOM;
            startingIndex=iA;
            currentResolution=1;
            resMult=1.0;
            multiplierOM=1.0;
            for(myint in=0;in<currentOM;++in){
                multiplierOM+=multiplierOM;
            }
        }
    }
    myint numRes=resolutions.size();
    for(iA=numRes-2;iA>-1;--iA){
        if(resolutions[iA]==0){
            resolutions[iA]=2;
            allInd[iA].resize(2);
            allInd[iA][0]=allInd[iA+1][ allInd[iA+1].size() -1];
            allInd[iA][1]=allInd[iA][0];
        }
    }
    allIndSize=allInd.size() ;
    for(iA=0;iA<numRes;++iA){
        resolutions[iA] /= 2;
        if(resolutions[iA]==0){
            resolutions[iA]=1;
        }
    }
    
    exponents.resize(resolutions.size());
    myint tempNum;
    for(myint i=0;i<resolutions.size();++i){
        if(i!=0){
            if(oMMultipliersInt[i]==0){
                oMMultipliersInt[i]=1;
            }
            if(resolutions[i]==0){
                resolutions[i]=1;
            }
        }
        oMMultipliersInt[i]*=  resolutions[i] ;
        tempNum=oMMultipliersInt[i];
        exponents[i]=-1;
        while(tempNum>0){
            tempNum/=2;
            exponents[i]+=1;
        }
        if(exponents[i]<0){
            exponents[i]=0;
        }
    }
}
void IndexStructure::createIndices(const std::vector<mydouble> & a){
    myint iA;
    myint sA=a.size();
    maxEl=getMaxEl(a);
    
    if(maxEl<1.0){
        createIndicesOfSeqWhoseTermsAreSmallerThanOne(a);
    }
    else{
        std::vector<mydouble> b;
        b.resize(sA);
        while(maxEl > maxElMultiplier){
            maxElReduction/=2.0;
            maxElMultiplier*=2.0;
        }
        for(myint i=0;i<sA;++i){
            b[i]=a[i]/maxElMultiplier;
        }
        createIndicesOfSeqWhoseTermsAreSmallerThanOne(b);
    }
    
}

myint IndexStructure::getIndexOfSeq(const mydouble & num) const{
    myint fR=-17;
    mydouble prepNum=num*maxElReduction;
    int oM=getOrderOfMagnitudeAndModifyMe(prepNum);
    myint prepNumInt= static_cast<myint>(prepNum);
    if(((allIndSize-oM)*(oM+1)>0)&&(prepNumInt<allInd[oM].size())){
        fR=allInd[oM][prepNumInt];
     }
    
    return fR;
}

myint IndexStructure::size() const{
    return resolutions.size();
}

std::vector<myint> IndexStructure::getExponents() const{
    return exponents;
}
std::vector<std::vector<myint> > IndexStructure::getAllInd() const{
    return allInd;
}


void IndexStructure::clear(){
    resolutions.clear();
    allInd.clear();
}
