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

#include "gcImpVolBasic.h"


GCImpVol::GCImpVol(){
    
    initializeGC();
   
    initializeGraphicCard();
    
    
}


myint GCImpVol::initializeGC(){
    
    name_numberOfSamplesGC="numberOfSamplesGCCL";
    name_FFGC="FFGCCL";
    name_pxGC="pxGCCL";
    name_KKGC="KKGCCL";
    name_rrGC="rrGCCL";
    name_texpGC="texpGCCL";
    name_PCGC="PCGCCL";
    name_k_FukasawaGC="k_FukasawaGCCL";
    name_c_FukasawaGC="c_FukasawaGCCL";
    nameFractionsGC="fractionsGCCL";
    substitutionKernelNameGC="prepareSubstitution";
    
    getIndexStructureForKKernelGC="getIndStForK";
    getIndicesKernelGC="getIndices";
    
    nameLengthIndStruKGC="lengthIndStructureKGCCL";
    nameIndStruKGC="indStructureKGCCL";
    
    nameLengthIndicesKGC="lengthIndicesKGCCL";
    nameIndicesKGC="indicesKGCCL";
    nameIndicesXGC="indicesXGCCL";
    
    nameSeqShiftXGC="seqShiftXGCCL";
    nameSeqXTotalGC="seqXTotalGCCL";
    nameLenSeqShiftXGC="lenSeqShiftXGCCL";
    nameLenSeqXTotalGC="lenSeqXTotalGCCL";
    
    namePolCoeffsGC="polCoefficientsGCCL";
    nameDegKGC="degreeKGCCL";
    nameDegXGC="degreeXGCCL";
    namePartKLengthGC="partKLengthGCCL";
    namePartKSeqGC="partKSeqGCCL";
    namePartXSeqGC="partXSeqGCCL";
    nameShiftSeqPartXGC="shiftSeqPartXGCCL";
    nameShiftSequenceKXGC="shiftSequenceKXGCCL";
    nameVectorOfResultsGC="vectorOfResultsGCCL";
    nameNumCoeffsInPolGC="numCoeffPolGCCL";
    nameMaxElReductionKGC="maxElReductionKGCCL";
    nameMaxElMultiplierKGC="maxElMultiplierKGCCL";
    nameMaxElReductionSeqXGC="maxElReductionSeqXGCCL";
    nameMaxElMultiplierSeqXGC="maxElMultiplierSeqXGCCL";
    
    approxPolsKernelGC="approxPolsGC";
    
    graphicCardInitialized=0;
    return 1;
}

myint* GCImpVol::prepareIndStructureForGC(const IndexStructure & iS){
    
    
    //Index structure is put on graphic card as a single sequence
    // First 3 elements are parameters:
    // [0] is total length
    // [1] is the length of the sequence of exponenets
    // [2] is numOMs the number of orders of magnitude
    // The next numberOfExponents terms are
    //      the sequence of exponents
    // The next numOMs terms are the shifts for the vectors
    // The remaining terms are the vectors of indices
    std::vector<myint> exps=iS.getExponents();
    std::vector<std::vector<myint> > allInds=iS.getAllInd();
    myint numParameters=3;
    myint expSeqLen=exps.size();
    myint numOMs=allInds.size();
    myint totalIndSize=0;
    for(myint i=0;i<numOMs;++i){
        totalIndSize+= allInds[i].size();
    }
    myint lenOfSeqForGC= numParameters + expSeqLen + numOMs + totalIndSize;
    myint* fR;
    fR=new myint[lenOfSeqForGC];
    fR[0]=lenOfSeqForGC;
    fR[1]=expSeqLen;
    fR[2]=numOMs;

    myint shift=numParameters;
    for(myint i=0;i<expSeqLen;++i){
        fR[shift+i]=exps[i];
    }
    shift+=expSeqLen;
    fR[shift]=0;
    ++shift;
    for(myint i=1;i<numOMs;++i){
        fR[shift]=fR[shift-1]+allInds[i-1].size();
        ++shift;
    }
    
    for(myint i=0;i<numOMs;++i){
        for(myint j=0;j<allInds[i].size();++j){
            fR[shift]=allInds[i][j];
            ++shift;
        }
    }
    return fR;
    
}

myint GCImpVol::initializeGraphicCard(){
    mysint forceCPU=0;
    
    gCard=new GraphicCard("parallelImpVol.cl",forceCPU);
    graphicCardInitialized=1;
    

    
    myint * indSKGC=prepareIndStructureForGC(indStruForK);
    myint lenIndStruKGC=indSKGC[0];
    gCard->writeDeviceMemory(nameLengthIndStruKGC, &lenIndStruKGC,1);
    gCard->writeDeviceMemory(nameIndStruKGC, indSKGC,lenIndStruKGC);
    gCard->writeDeviceMemory(nameDegKGC,&degreeK,1);
    gCard->writeDeviceMemory(nameDegXGC,&degreeX,1);
    myint ncip=(degreeK+1)*(degreeX+1);
    
    gCard->writeDeviceMemory(nameNumCoeffsInPolGC,&ncip,1);
    delete[] indSKGC;
    
    myint nIndSX=indStruForX.size();
    myint** indSXGCSeq;
    indSXGCSeq=new myint*[nIndSX];
    myint* lenSeqISX, *shiftSeqISX;
    mydouble *maxElReductionSeqXGC, *maxElMultiplierSeqXGC;

    maxElReductionSeqXGC= new mydouble[nIndSX];
    maxElMultiplierSeqXGC= new mydouble[nIndSX];
    
    
    gCard->writeDeviceMemory(nameLengthIndStruKGC, &lenIndStruKGC,1);
    
    
    mydouble tempHolderMaxEl=indStruForK.getMaxElReduction();
    gCard->writeDeviceMemory(nameMaxElReductionKGC, &tempHolderMaxEl,1);
    
    tempHolderMaxEl=indStruForK.getMaxElMultiplier();

    
    lenSeqISX=new myint[nIndSX];
    shiftSeqISX=new myint[nIndSX];
    
    
    myint totalLenISX=0;

    for(myint i=0;i<nIndSX;++i){

        indSXGCSeq[i]=prepareIndStructureForGC(indStruForX[i]);
        maxElReductionSeqXGC[i]=indStruForX[i].getMaxElReduction();
        maxElMultiplierSeqXGC[i]=indStruForX[i].getMaxElMultiplier();
        lenSeqISX[i]=indSXGCSeq[i][0];
        totalLenISX+=lenSeqISX[i];
        
    }
    shiftSeqISX[0]=0;

    for(myint i=1;i<nIndSX;++i){
        shiftSeqISX[i]=shiftSeqISX[i-1]+lenSeqISX[i-1];
    }
    myint* indSXGC;
    indSXGC=new myint[totalLenISX];
    myint posInIndS=0;
    for(myint i=0;i<nIndSX;++i){
        for(myint j=0;j<lenSeqISX[i];++j){
            indSXGC[posInIndS]=indSXGCSeq[i][j];
            ++posInIndS;
        }
    }

    
    gCard->writeDeviceMemory(nameLenSeqShiftXGC, &nIndSX,1);
    gCard->writeDeviceMemory(nameLenSeqXTotalGC,&totalLenISX,1);
    gCard->writeDeviceMemory(nameSeqShiftXGC,shiftSeqISX,nIndSX);
    gCard->writeDeviceMemory(nameSeqXTotalGC,indSXGC,totalLenISX);
    
    gCard->writeDeviceMemory(nameMaxElReductionSeqXGC,maxElReductionSeqXGC ,nIndSX);

    
    delete[] lenSeqISX;
    delete[] shiftSeqISX;
    for(myint i=0;i<nIndSX;++i){
        delete[] indSXGCSeq[i];
    }
    delete[] indSXGCSeq;
    
    myint pKLength=partK.size();
    gCard->writeDeviceMemory(namePartKLengthGC,&pKLength,1);
    gCard->writeDeviceMemory(namePartKSeqGC,partK,pKLength);
    
    myint* shiftSeqPX;
    shiftSeqPX=new myint[pKLength];
    myint totalLenPX=0;
    totalLenPX+=pKX[0].size();
    shiftSeqPX[0]=0;
    for(myint i=1;i<pKLength;++i){
        shiftSeqPX[i]=totalLenPX;
        totalLenPX+=pKX[i].size();
        
    }
    
    cl_double* partXSeqGC;
    partXSeqGC=new cl_double[totalLenPX];
    cl_double* coeffsGC;
    
    coeffsGC=new cl_double[totalLenPX*ncip];
    
    myint counter=0,counter2=0;
    for(myint i=0;i<pKLength;++i){
        for(myint j=0;j<pKX[i].size();++j){
            partXSeqGC[counter]=(cl_double)( pKX[i][j]);
            ++counter;
            for(myint kC=0;kC<ncip;++kC){
                coeffsGC[counter2]=(cl_double)(coeff[i][j][kC]);
                ++counter2;
            }
        }
    }
    gCard->writeDeviceMemory(nameShiftSeqPartXGC,shiftSeqPX,pKLength);
    gCard->writeDeviceMemory(namePartXSeqGC,partXSeqGC,totalLenPX);

    totalLenPX*=ncip;
    
    gCard->writeDeviceMemory(namePolCoeffsGC,coeffsGC,totalLenPX);

    
    delete[] shiftSeqPX;
    delete[] partXSeqGC;
    

    
    delete[] coeffsGC;
 
    delete[] maxElMultiplierSeqXGC;
    delete[] maxElReductionSeqXGC;

    
    return 1;
}


GCImpVol::~GCImpVol(){
 
  
    if(graphicCardInitialized==1){
        delete gCard;
    }
    
}

cl_double* GCImpVol::evaluateGC(const myint &_lengths){

    
    myint lengths=_lengths;
    
    lengths=gCard->getPrefferedWorkGroupSize() * (lengths /gCard->getPrefferedWorkGroupSize() );
     
    
    
    myint numKsL = partK.size();
    cl_double* fR;
    fR=new cl_double[lengths];
    
    
    std::string *lArgRNK;
    myint lens2=lengths;
    if(lastCalculationSize!=lengths){
        myint *indTempEmpty;
        indTempEmpty=new myint[lengths];
        gCard->writeDeviceMemory(nameIndicesKGC, indTempEmpty, lens2);
        gCard->writeDeviceMemory(nameIndicesXGC,indTempEmpty,lens2);
        delete[] indTempEmpty;
    }
    
    
    gCard->writeDeviceMemory(nameLengthIndicesKGC, &lens2,1);
    

    myint numArg=10;
    numArg=11;
    lArgRNK=new std::string[numArg];
    lArgRNK[0]=nameIndicesKGC;
    lArgRNK[1]=nameIndicesXGC;
    lArgRNK[2]=nameIndStruKGC;
    lArgRNK[3]=nameSeqXTotalGC;
    lArgRNK[4]=nameSeqShiftXGC;
    lArgRNK[5]=name_k_FukasawaGC;
    lArgRNK[6]=name_c_FukasawaGC;
    lArgRNK[7]=nameLengthIndicesKGC;
    lArgRNK[8]=nameMaxElReductionKGC;
    lArgRNK[9]=nameMaxElReductionSeqXGC;
    lArgRNK[10]=name_KKGC;
    gCard->findAddKernel(getIndicesKernelGC);

    gCard->setKernelArguments(getIndicesKernelGC,lArgRNK,numArg);

    gCard->executeKernel(getIndicesKernelGC, lengths );

    delete[] lArgRNK;

    
    numArg=16;
    lArgRNK=new std::string[numArg];
    lArgRNK[0]=nameVectorOfResultsGC;
    lArgRNK[1]=name_c_FukasawaGC;
    lArgRNK[2]=name_k_FukasawaGC;
    
    lArgRNK[3]=namePartXSeqGC;
    lArgRNK[4]=namePartKSeqGC;
    lArgRNK[5]=nameShiftSeqPartXGC;
    lArgRNK[6]=nameIndicesXGC;
    lArgRNK[7]=nameIndicesKGC;
    lArgRNK[8]=nameLengthIndicesKGC;
    lArgRNK[9]=namePartKLengthGC;
    lArgRNK[10]=nameDegXGC;
    lArgRNK[11]=nameDegKGC;
    lArgRNK[12]=namePolCoeffsGC;
    lArgRNK[13]=nameNumCoeffsInPolGC;
    lArgRNK[14]=name_texpGC;
    lArgRNK[15]=name_KKGC;
    
    
    gCard->findAddKernel(approxPolsKernelGC);
    
    gCard->setKernelArguments(approxPolsKernelGC,lArgRNK,numArg);

    delete[] lArgRNK;
    gCard->executeKernel(approxPolsKernelGC,lens2);
    

    
    
    gCard->readDeviceMemory(nameVectorOfResultsGC,fR,lengths);
  
    return (fR);
    
    
    
}
template <typename int_doub>
myint GCImpVol::extendSeq(int_doub  * &seq, const myint & oldN, const myint & newN){
    int_doub* newSeq=new int_doub[newN];
    myint i=0;
    if(newN>oldN){
        while(i<oldN){
            newSeq[i]=seq[i];
            ++i;
        }
        while(i<newN){
            newSeq[i]=newSeq[i-1];
            ++i;
        }
        delete[] seq;
        seq=newSeq;
        return 1;
    }
    return 0;
    
}


cl_double* GCImpVol::ivGC(cl_double * & FF,
                         cl_double * & px,
                         cl_double * &KK,
                         cl_double * &rr,
                         cl_double * &texp,
                         myint * &PC,
                         myint  lengthsSupplied) {

    
    myint numWG=gCard->getPrefferedWorkGroupSize();
    myint lengths=lengthsSupplied;
    if (lengths%numWG!=0){
        
        lengths= (lengths/numWG + 1)*numWG;
        extendSeq(FF,lengthsSupplied,lengths);
        extendSeq(px,lengthsSupplied,lengths);
        extendSeq(KK,lengthsSupplied,lengths);
        extendSeq(rr,lengthsSupplied,lengths);
        extendSeq(texp,lengthsSupplied,lengths);
        extendSeq(PC,lengthsSupplied,lengths);
    }

    cl_double *temp_seq;
    temp_seq=new cl_double[lengths];
    
    
    
    
    myint lns2=lengths;
    
    myint *pLens=&lns2;
    gCard->writeDeviceMemory(name_numberOfSamplesGC,pLens,1);
    
    gCard->writeDeviceMemory(name_PCGC,PC,lengths);
    gCard->writeDeviceMemory(name_texpGC,texp,lengths);
    gCard->writeDeviceMemory(name_rrGC,rr,lengths);
    gCard->writeDeviceMemory(name_FFGC,FF,lengths);
    gCard->writeDeviceMemory(name_KKGC,KK,lengths);
    gCard->writeDeviceMemory(name_pxGC,px,lengths);
    
    
    
    if(lastCalculationSize!=lengths){
        gCard->writeDeviceMemory(name_k_FukasawaGC,temp_seq,lengths);
        gCard->writeDeviceMemory(name_c_FukasawaGC,temp_seq,lengths);
        gCard->writeDeviceMemory(nameVectorOfResultsGC,temp_seq,lengths);
    }
    
    delete[] temp_seq;
    
    std::string *lArgRNK;
    myint numArg=10;
    numArg=9;
    lArgRNK=new std::string[numArg];
    lArgRNK[0]=name_numberOfSamplesGC;
    lArgRNK[1]=name_FFGC;
    lArgRNK[2]=name_pxGC;
    lArgRNK[3]=name_KKGC;
    lArgRNK[4]=name_rrGC;
    lArgRNK[5]=name_texpGC;
    lArgRNK[6]=name_PCGC;
    lArgRNK[7]=name_k_FukasawaGC;
    lArgRNK[8]=name_c_FukasawaGC;
    
    gCard->findAddKernel(substitutionKernelNameGC);
    
    gCard->setKernelArguments(substitutionKernelNameGC,lArgRNK,numArg);
    
    gCard->executeKernel(substitutionKernelNameGC, lengths );
    
    delete[] lArgRNK;
    

    
    
    cl_double* fR= evaluateGC(lengths);
   
    lastCalculationSize=lengths;
    return fR;
}



