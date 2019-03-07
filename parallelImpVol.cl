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

__constant double CONST_K_MIN=0.00000001;
__constant double CONST_K_MIN_MULT101=0.0000000101;
myint getIndex(__global myint *entireIS,double k,
               double maxRedu){
    myint fR=-1;
 
    myint entireLen=entireIS[0];
    myint expSeqLen=entireIS[1];
    myint numOMs=entireIS[2];
    myint numParameters=3;
    myint shift;
    
    
    int oMInt;
    k*=maxRedu;
    frexp(k,&oMInt);
    oMInt=1-oMInt;
    myint oM;
    oM=(myint)(oMInt);
    k=ldexp(k,entireIS[numParameters+oM]);
    
    myint prepNumInt= (myint)(k);
    shift=numParameters+expSeqLen;
    myint relevantSize=entireIS[shift+oM+1]-entireIS[shift+oM];
    
    
    if((oM<numOMs)&&(prepNumInt<relevantSize)){
        fR=entireIS[shift+numOMs+entireIS[shift+oM]+prepNumInt];
        
    }
    return fR;
}

__kernel void getIndices(__global myint *resIndSeqK,
                         __global myint *resIndSeqX,
                         __global myint *entireISK,
                         __global myint *entireISX,
                         __global myint *shiftSeqX,
                         __global double *main_seqK,
                         __global double *main_seqX,
                         __global myint *numSamples,
                         __global double *maxElReduK,
                         __global double *maxElReduSeqX,
                         __global double *KK){
    
    
    myint gid = get_global_id(0);
    double k=main_seqK[gid];
    double c=main_seqX[gid];
    myint kInd=getIndex(entireISK,k, *maxElReduK);
    resIndSeqK[gid]=kInd;
    
    
    myint cInd=-1;
    if(kInd>-1){
        cInd=getIndex(entireISX+shiftSeqX[kInd],c, maxElReduSeqX[kInd]);
    }
    resIndSeqX[gid]=cInd;
    
}

__kernel void prepareSubstitution(__global myint *lengthOfVectors,
                                  __global double *FF,
                                  __global double *px,
                                  __global double *KK,
                                  __global double *rr,
                                  __global double *texp,
                                  __global myint  *PC,
                                  __global double *k_Fukasawa,
                                  __global double *c_Fukasawa)
{
    myint gid = get_global_id(0);
    double rMultT=rr[gid]*texp[gid];
    double recDisc=exp(rMultT);
    double C0 = PC[gid] ? px[gid] : px[gid] +  FF[gid] - KK[gid]/recDisc;
    double FRec=1.0/FF[gid];
    double kRecF=KK[gid]*FRec-rMultT;
    double k=log(kRecF);
    double x=C0 * FRec;
    if(k<CONST_K_MIN){
        if(k<-CONST_K_MIN){
            x=((x-1.0)/kRecF) * recDisc+1.0;
            k=0.0-k;
        }
        else{
            k=CONST_K_MIN_MULT101;
        }
    }
    k_Fukasawa[gid]=k;
    c_Fukasawa[gid] =x;
    
}


__kernel void approxPolsGC(__global double *seqRes,
                           __global double *seqX,
                           __global double *seqK,
                           __global double *partKX,
                           __global double *partK,
                           __global myint *shiftSeqX,
                           __global myint *seqIndXS,
                           __global myint *seqIndKS,
                           __global myint *numSamples,
                           __global myint *numKsL,
                           __global myint *degX,
                           __global myint *degK,
                           __global double *coeff,
                           __global myint *polNumC,
                           __global double *texp,
                           __global double *KK){
    myint gid = get_global_id(0);
    double expk=KK[gid];
    double x=seqX[gid];
    double k=seqK[gid];
    
    double fR=-111.0;
    
    if((k>=partK[0])&&(k<=partK[(*numKsL)-1])&&(seqIndKS[gid]>-1)&&(seqIndXS[gid]>-1)&&(seqIndKS[gid]<*numKsL)){
        myint posInPol=shiftSeqX[seqIndKS[gid]]+seqIndXS[gid];
        myint posInCoeff=posInPol* (*polNumC);
        
        x-=partKX[posInPol];
        k-=partK[seqIndKS[gid]];
        double helpCoeff;
        
        myint j;
        
    //LOOPS ARE UNROLLED: WORKS ONLY IF DEGREES ARE D=8
j=posInCoeff+ 80;
helpCoeff = (((((((coeff[j]*x+coeff[j-9])*x+coeff[j-18])*x+coeff[j-27])*x+coeff[j-36])*x+coeff[j-45])*x+coeff[j-54])*x+coeff[j-63])*x+coeff[j-72];
fR = helpCoeff*k;
--j;
helpCoeff = (((((((coeff[j]*x+coeff[j-9])*x+coeff[j-18])*x+coeff[j-27])*x+coeff[j-36])*x+coeff[j-45])*x+coeff[j-54])*x+coeff[j-63])*x+coeff[j-72];
fR += helpCoeff;
fR *= k;
--j;
helpCoeff = (((((((coeff[j]*x+coeff[j-9])*x+coeff[j-18])*x+coeff[j-27])*x+coeff[j-36])*x+coeff[j-45])*x+coeff[j-54])*x+coeff[j-63])*x+coeff[j-72];
fR += helpCoeff;
fR *= k;
--j;
helpCoeff = (((((((coeff[j]*x+coeff[j-9])*x+coeff[j-18])*x+coeff[j-27])*x+coeff[j-36])*x+coeff[j-45])*x+coeff[j-54])*x+coeff[j-63])*x+coeff[j-72];
fR += helpCoeff;
fR *= k;
--j;
helpCoeff = (((((((coeff[j]*x+coeff[j-9])*x+coeff[j-18])*x+coeff[j-27])*x+coeff[j-36])*x+coeff[j-45])*x+coeff[j-54])*x+coeff[j-63])*x+coeff[j-72];
fR += helpCoeff;
fR *= k;
--j;
helpCoeff = (((((((coeff[j]*x+coeff[j-9])*x+coeff[j-18])*x+coeff[j-27])*x+coeff[j-36])*x+coeff[j-45])*x+coeff[j-54])*x+coeff[j-63])*x+coeff[j-72];
fR += helpCoeff;
fR *= k;
--j;
helpCoeff = (((((((coeff[j]*x+coeff[j-9])*x+coeff[j-18])*x+coeff[j-27])*x+coeff[j-36])*x+coeff[j-45])*x+coeff[j-54])*x+coeff[j-63])*x+coeff[j-72];
fR += helpCoeff;
fR *= k;
--j;
helpCoeff = (((((((coeff[j]*x+coeff[j-9])*x+coeff[j-18])*x+coeff[j-27])*x+coeff[j-36])*x+coeff[j-45])*x+coeff[j-54])*x+coeff[j-63])*x+coeff[j-72];
fR += helpCoeff;
fR *= k;
--j;
helpCoeff = (((((((coeff[j]*x+coeff[j-9])*x+coeff[j-18])*x+coeff[j-27])*x+coeff[j-36])*x+coeff[j-45])*x+coeff[j-54])*x+coeff[j-63])*x+coeff[j-72];
fR += helpCoeff;

    }
    
    seqRes[gid]=fR/sqrt(texp[gid]);
    
    
}


    