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
typedef long int mysint;
typedef long int myint;


myint invBalNum(__global myint *seqPascTr, myint needle){
    
    myint numZeroes=0;
    myint numOnes=0;
    myint needleHelp=needle;
    myint currentDigit;
    myint fR=0; myint helpVar;
    myint m=0;

    while(needleHelp>0){
        currentDigit=needleHelp % 2;
        needleHelp = needleHelp/2;
        
        numZeroes+=(1-currentDigit);
        numOnes+=currentDigit;

        // m is a 0-1 indicator that is 0 until the first digit 0 appears. Once
        // the first 0 appears, m becomes 1 and stays 1.
        m+=(1-currentDigit)*(1-m);
        

        // If currentDigit==1 then we have to add \varpsi(numOnes, numZeroes)
        // \varpsi(numOnes, numZeroes)  = \binom{numOnes+numZeroes-1}{numZeroes-1}
        // \binom(n,k)=seqPascTr( n*(n+1)/2+k)
        helpVar=numOnes+numZeroes-1;
        
        fR+= currentDigit* m * seqPascTr[ numZeroes-1 +( helpVar*(helpVar+1)/2)];
            
       
    }
    
    return fR;

    
}


myint properlyShuffleInsideGC(__global myint *permutationX,
                              __global myint *permutationY,
                              __global myint *powersOfTwo,
                              __global myint *seqPascTr,
                              __global myint *seqBalNum,
                              myint numSeqBalNum,
                              myint r, myint p,
                              myint i, myint j, myint xA, myint yA){
    myint fR;
    
    myint xShuffled=0;
    myint yShuffled=0;
    myint nextDigitX,nextDigitY;
    myint k=0;myint positionForNextDigitX, positionForNextDigitY;
    myint biggestPowerOfTwo=powersOfTwo[r-1];
    biggestPowerOfTwo*=2;
    myint x=seqBalNum[xA];myint y=seqBalNum[yA];
    myint jr=j*r; myint ir=i*r;
    for(k=0;k<r;k++){

        nextDigitX=x % 2;
        nextDigitY=y % 2;
        positionForNextDigitX=permutationY[jr+k];
        positionForNextDigitY=permutationX[ir+k];


        xShuffled+=nextDigitX * powersOfTwo[positionForNextDigitX];
        yShuffled+=nextDigitY * powersOfTwo[positionForNextDigitY];
         x=x/2;
        y=y/2;
    }

    xShuffled=invBalNum(seqPascTr, xShuffled);
    yShuffled=invBalNum(seqPascTr,yShuffled);
    
    fR= (xShuffled+yShuffled)% numSeqBalNum;
    
    return fR;
    
}

__kernel void genMainRandomMatrixGC(__global myint *rNumGCCL,
                __global  myint *xAxGCCL,
                __global  myint *yAxGCCL,
                __global  myint *permutationsXGCCL,
                __global  myint *permutationsYGCCL,
                __global myint *powersOfTwoGCCL,
                __global myint *balancedNumbersGCCL,
                __global myint *pascalTGCCL,
                __global myint *axisSizeGCCL,
                __global myint *binaryLengthGCCL,
                __global myint *numBalancedNumbersGCCL,
                __global myint *shufflingPrimeGCCL)
{   myint gid = get_global_id(0);

        myint xCoord=gid % (*axisSizeGCCL);
        myint yCoord=gid / (*axisSizeGCCL);
        rNumGCCL[gid]=*binaryLengthGCCL;
        

        
        myint xH, yH;
        xH=yAxGCCL[0];
        yH=xAxGCCL[0];

        
        rNumGCCL[gid]=properlyShuffleInsideGC(permutationsXGCCL, permutationsYGCCL,powersOfTwoGCCL,
                                              pascalTGCCL, balancedNumbersGCCL,
                                              *numBalancedNumbersGCCL,
                                              *binaryLengthGCCL, *shufflingPrimeGCCL,
                                              xCoord, yCoord, xAxGCCL[xCoord],yAxGCCL[yCoord]);

}



__kernel void inspectorKernelGC(__global myint *inspectorGCCL,
                                    __global  myint *sampleLengthGCCL,
                                    __global  myint *oldLayerGCCL)
{
    myint gid=get_global_id(0);
    myint writingPosition= gid * (*oldLayerGCCL)*2;
    if(writingPosition+*oldLayerGCCL<*sampleLengthGCCL){
        myint theFirst= inspectorGCCL[writingPosition];
        myint theSecond=inspectorGCCL[writingPosition+*oldLayerGCCL];
        theFirst%=2;
        theSecond%=2;
        inspectorGCCL[writingPosition]=1;
        if((theFirst==0)||(theSecond==0)){
            inspectorGCCL[writingPosition]=0;
        }
    }
}


__kernel void rejectionSamplerExponentialGC(__global double *randSampGCCL,
                                            __global myint *rNumGCCL,
                                            __global myint *inspectorGCCL,
                                            __global myint *sampleLengthGCCL,
                                            __global myint *sizeRejectionSamplingGCCL,
                                            __global myint *precReqGCCL,
                                            __global myint *par1GCCL,
                                            __global myint *par2GCCL,
                                            __global myint *numBalancedNumbersGCCL,
                                            __global myint *biggestNumGCCL)
{
    myint gid=get_global_id(0);
    if(gid<*sampleLengthGCCL){
        myint sL=*sizeRejectionSamplingGCCL;
        myint base=*numBalancedNumbersGCCL;
        myint readPos=gid * sL;
        myint lastReadPos=(gid+1)*sL;
        myint oneNumSize= (*precReqGCCL)/2;
        myint success=0;
        double x,y;
        myint j;
        while((success==0)&&(readPos<lastReadPos)){
            x=0.0;y=0.0;
            for(j=0;j<oneNumSize;j++){
                x=x * ((double) base)+((double) rNumGCCL[readPos+1+j]);
                y=y * ((double) base)+((double) rNumGCCL[readPos+1+j+oneNumSize]);
            }
            x= x/ ((double)*biggestNumGCCL);
            y= y/ ((double)*biggestNumGCCL);
            readPos+=sL;
            success=1;
            j=2;
            x=(double)rNumGCCL[readPos+1+j] +0.1;
        }
        randSampGCCL[gid]=x;
        inspectorGCCL[gid]=success;
        
    }
}




__kernel void rejectionSamplerNormalGC(__global double *randSampGCCL,
                                       __global myint *rNumGCCL,
                                       __global myint *inspectorGCCL,
                                       __global myint *sampleLengthGCCL,
                                       __global myint *sizeRejectionSamplingGCCL,
                                       __global myint *precReqGCCL,
                                       __global myint *par1GCCL,
                                       __global myint *par2GCCL,
                                       __global myint *numBalancedNumbersGCCL,
                                       __global myint *biggestNumGCCL)
{
    myint gid=get_global_id(0);
    if(gid<*sampleLengthGCCL){
        myint sL=*sizeRejectionSamplingGCCL;
        myint base=*numBalancedNumbersGCCL;
        myint readPos=gid * sL;
        myint lastReadPos=(gid+1)*sL;
        myint oneNumSize= (*precReqGCCL)/2;
        myint success=0;
        double x,y;
        myint j;
        while((success==0)&&(readPos<lastReadPos)){
            x=0.0;y=0.0;
            for(j=0;j<oneNumSize;j++){
                x=x * ((double) base)+((double) rNumGCCL[readPos+1+j]);
                y=y * ((double) base)+((double) rNumGCCL[readPos+1+j+oneNumSize]);
            }
            x= x/ ((double)*biggestNumGCCL);
            y= y/ ((double)*biggestNumGCCL);
            readPos+=sL;
            success=1;
            
            
        }
        randSampGCCL[gid]=x;
        inspectorGCCL[gid]=success;
        
    }
}








__kernel void normalBSMAntitheticGC(__global double *randSampGCCL,
                          __global myint *rNumGCCL,
                          __global myint *sampleLengthGCCL,
                          __global myint *sizeRejectionSamplingGCCL,
                          __global myint *precReqGCCL,
                          __global double *par1GCCL,
                          __global double *par2GCCL,
                          __global myint *numBalancedNumbersGCCL,
                          __global myint *biggestNumGCCL,
                          __global double *seqAGCCL,
                          __global double *seqBGCCL,
                          __global double *seqCGCCL,
                          __global myint* numABGCCL,
                          __global myint* numCGCCL)
{
    myint gid=get_global_id(0);
    if(gid<*sampleLengthGCCL){
 
        randSampGCCL[gid]*=-1.0;
        
        
        
    }
}








__kernel void uniformDebuggingBSMGC(__global double *randSampGCCL,
                          __global myint *rNumGCCL,
                          __global myint *sampleLengthGCCL,
                          __global myint *sizeRejectionSamplingGCCL,
                          __global myint *precReqGCCL,
                          __global double *par1GCCL,
                          __global double *par2GCCL,
                          __global myint *numBalancedNumbersGCCL,
                          __global myint *biggestNumGCCL,
                          __global double *seqAGCCL,
                          __global double *seqBGCCL,
                          __global double *seqCGCCL,
                          __global myint* numABGCCL,
                          __global myint* numCGCCL)
{
    myint gid=get_global_id(0);
    if(gid<*sampleLengthGCCL){
        myint sL=*sizeRejectionSamplingGCCL;
        myint base=*numBalancedNumbersGCCL;
        myint readPos=gid * sL;
        
        double x;
        myint i ;
        
        x=0.0;
        for(i=0;i<sL;i++){
            x=x * ((double) base)+((double) rNumGCCL[readPos+i]);
            
        }
        x= x/ ((double)*biggestNumGCCL);
        
        randSampGCCL[gid]=x;
        
        
        
    }
}






__kernel void normalBSMGC(__global double *randSampGCCL,
                          __global myint *rNumGCCL,
                          __global myint *sampleLengthGCCL,
                          __global myint *sizeRejectionSamplingGCCL,
                          __global myint *precReqGCCL,
                          __global double *par1GCCL,
                          __global double *par2GCCL,
                          __global myint *numBalancedNumbersGCCL,
                          __global myint *biggestNumGCCL,
                          __global double *seqAGCCL,
                          __global double *seqBGCCL,
                          __global double *seqCGCCL,
                          __global myint* numABGCCL,
                          __global myint* numCGCCL)
{
    myint gid=get_global_id(0);
    if(gid<*sampleLengthGCCL){
        myint sL=*sizeRejectionSamplingGCCL;
        myint base=*numBalancedNumbersGCCL;
        myint readPos=gid * sL;
        
        double x;
        myint i ;
        
        x=0.0;
        for(i=0;i<sL;i++){
            x=x * ((double) base)+((double) rNumGCCL[readPos+i]);
            
        }
        x= x/ ((double)*biggestNumGCCL);
        
        
        double fInvX;
        
        myint numAB=*numABGCCL, numC=*numCGCCL;
        __global double *a,*b,*c;
        a=seqAGCCL;
        b=seqBGCCL;
        c=seqCGCCL;
 
        double y=x - 0.5;
 
        double absy= y,signy=1.0;
        if(y<0.0){
            absy=-y;
            signy=-1.0;
        }
        
        double r;
        double numerator=0.0, denominator=0.0;
        
        if(absy<0.42){
            
            r= absy * absy;
            for(i=0;i<numAB;i++){
                numerator*= r;
                numerator+=a[numAB-i-1];
            }
            numerator*=y;
            for( i =0;i<numAB;i++){
                denominator*=r;
                denominator+=b[numAB-i-1];
            }
            denominator*=r;
            denominator+=1;
            fInvX=0.0;
            if(denominator!=0.0){
                fInvX= numerator/denominator;
            }
            
        }
        else{
            r=x;
            if(x>0.5){
                r=1-x;
            }
            fInvX=0.0;
            if(r!=0.0){
                r=log(-(log(r)));
                fInvX=0.0;
                for(i=0;i<numC;i++){
                    fInvX*=r;
                    fInvX+=c[numC-i-1];
                }
                fInvX*=signy;
            }
        }
        

        fInvX*= (*par2GCCL);
        fInvX+=*par1GCCL;
 
        randSampGCCL[gid]=fInvX;
        


    }
}




__kernel void exponentialDistGC(__global double *randSampGCCL,
                          __global myint *rNumGCCL,
                          __global myint *sampleLengthGCCL,
                          __global myint *sizeRejectionSamplingGCCL,
                          __global myint *precReqGCCL,
                          __global double *par2GCCL,
                          __global myint *numBalancedNumbersGCCL,
                          __global myint *biggestNumGCCL)
{
    myint gid=get_global_id(0);
    if(gid<*sampleLengthGCCL){
        myint sL=*sizeRejectionSamplingGCCL;
        myint base=*numBalancedNumbersGCCL;
        myint readPos=gid * sL;
        
        double x;
        myint i ;
        
        x=0.0;
        for(i=0;i<sL;i++){
            x=x * ((double) base)+((double) rNumGCCL[readPos+i]);
            
        }
        x= x/ ((double)*biggestNumGCCL);
        
        double fInvX=0;
        
        if(x!=1){
            fInvX=-log(1-x);
            fInvX/= *par2GCCL;
        }
  
        
        randSampGCCL[gid]=fInvX;
        
        
        
    }
}

//MERGE SORT BEGIN
__kernel void copyResultToStartGC(__global  double *startingSeq,
                                __global myint *length,
                                __global  double *resultingSeq
                                )
{   myint gid = get_global_id(0);
    if(gid<*length){
        startingSeq[gid]=resultingSeq[gid];
    }
}


__kernel void shiftCalculationMergSortIncGC(__global  double *startingSeq,
                                            __global myint *length,
                                            __global  double *resultingSeq,
                                            __global  myint *sortedSize
                                            )
{   myint gid = get_global_id(0);
    if(gid<*length){
        
        myint newSize=2 * (*sortedSize);
        
        myint absoluteStart=gid/newSize;
        myint friendlyBlockStart=absoluteStart * newSize;
        absoluteStart=friendlyBlockStart;
        myint myLocation=0;//left
        myint myPosition=gid%newSize;
        
        
        if( myPosition>= *sortedSize){
            myPosition-= (*sortedSize);
            myLocation=1;//right
        }
        else{
            friendlyBlockStart+=*sortedSize;
        }
        if(friendlyBlockStart<*length){
            myint rightPosition=-1;
            if(startingSeq[friendlyBlockStart]>startingSeq[gid]){
                rightPosition=0;
            }
            if((startingSeq[friendlyBlockStart]==startingSeq[gid])&&(myLocation==0)){
                rightPosition=0;
            }
            if(rightPosition!=0){
                myint leftPosition=0;
                rightPosition= *sortedSize;
                myint middlePosition;
                
                while (leftPosition+1<rightPosition){
                    middlePosition=(leftPosition+rightPosition)/2;
                    if(middlePosition + friendlyBlockStart<*length){
                        if(startingSeq[middlePosition+friendlyBlockStart]>startingSeq[gid]){
                            rightPosition=middlePosition;
                        }
                        else{
                            if(startingSeq[middlePosition+friendlyBlockStart]<startingSeq[gid]){
                                leftPosition=middlePosition;
                            }
                            else{
                                if(myLocation==0){
                                    rightPosition=middlePosition;
                                }
                                else{
                                    leftPosition=middlePosition;
                                }
                            }
                        }
                    }
                    else{
                        rightPosition=middlePosition;
                    }
                }
            }
            resultingSeq[absoluteStart+myPosition+rightPosition]=startingSeq[gid];
        }
        else{
            resultingSeq[gid]=startingSeq[gid];
            
        }
    }
}



__kernel void shiftCalculationMergSortDecGC(__global  double *startingSeq,
                                            __global myint *length,
                                            __global  double *resultingSeq,
                                            __global  myint *sortedSize
                                            )
{   myint gid = get_global_id(0);
    if(gid<*length){
        
        myint newSize=2 * (*sortedSize);
        myint absoluteStart=gid/newSize;
        myint friendlyBlockStart=absoluteStart * newSize;
        absoluteStart=friendlyBlockStart;
        myint myLocation=0;//left
        myint myPosition=gid%newSize;
        if( myPosition>= *sortedSize){
            myPosition-= (*sortedSize);
            myLocation=1;//right
        }
        else{
            friendlyBlockStart+=*sortedSize;
        }
        if(friendlyBlockStart<*length){
            myint rightPosition=-1;
            if(startingSeq[friendlyBlockStart]<startingSeq[gid]){
                rightPosition=0;
            }
            if((startingSeq[friendlyBlockStart]==startingSeq[gid])&&(myLocation==0)){
                rightPosition=0;
            }
            if(rightPosition!=0){
                myint leftPosition=0;
                rightPosition= *sortedSize;
                myint middlePosition;
                
                while (leftPosition+1<rightPosition){
                    middlePosition=(leftPosition+rightPosition)/2;
                    if(middlePosition + friendlyBlockStart<*length){
                        if(startingSeq[middlePosition+friendlyBlockStart]<startingSeq[gid]){
                            rightPosition=middlePosition;
                        }
                        else{
                            if(startingSeq[middlePosition+friendlyBlockStart]>startingSeq[gid]){
                                leftPosition=middlePosition;
                            }
                            else{
                                if(myLocation==0){
                                    rightPosition=middlePosition;
                                }
                                else{
                                    leftPosition=middlePosition;
                                }
                            }
                        }
                    }
                    else{
                        rightPosition=middlePosition;
                    }
                }
            }
            resultingSeq[absoluteStart+myPosition+rightPosition]=startingSeq[gid];
        }
        else{
            resultingSeq[gid]=startingSeq[gid];
            
        }
    }
}



//MERGE SORT END


//MERGE SORT WITH FOLLOWER BEGIN
__kernel void copyResultToStartFollGC(__global  double *startingSeq,
                                  __global myint *length,
                                  __global  double *resultingSeq,
                                  __global myint *followerSt,
                                  __global myint *followerEnd
                                  )
{   myint gid = get_global_id(0);
    if(gid<*length){
        startingSeq[gid]=resultingSeq[gid];
        followerSt[gid]=followerEnd[gid];
    }
}


__kernel void shiftCalculationMergSortFollIncGC(__global  double *startingSeq,
                                                __global myint *length,
                                                __global  double *resultingSeq,
                                                __global  myint *sortedSize,
                                                __global myint *followerSt,
                                                __global myint *followerRes
                                                )
{   myint gid = get_global_id(0);
    if(gid<*length){
        
        myint newSize=2 * (*sortedSize);
        
        myint absoluteStart=gid/newSize;
        myint friendlyBlockStart=absoluteStart * newSize;
        absoluteStart=friendlyBlockStart;
        myint myLocation=0;//left
        myint myPosition=gid%newSize;
        
        
        if( myPosition>= *sortedSize){
            myPosition-= (*sortedSize);
            myLocation=1;//right
        }
        else{
            friendlyBlockStart+=*sortedSize;
        }
        if(friendlyBlockStart<*length){
            myint rightPosition=-1;
            if(startingSeq[friendlyBlockStart]>startingSeq[gid]){
                rightPosition=0;
            }
            if((startingSeq[friendlyBlockStart]==startingSeq[gid])&&(myLocation==0)){
                rightPosition=0;
            }
            if(rightPosition!=0){
                myint leftPosition=0;
                rightPosition= *sortedSize;
                myint middlePosition;
                
                while (leftPosition+1<rightPosition){
                    middlePosition=(leftPosition+rightPosition)/2;
                    if(middlePosition + friendlyBlockStart<*length){
                        if(startingSeq[middlePosition+friendlyBlockStart]>startingSeq[gid]){
                            rightPosition=middlePosition;
                        }
                        else{
                            if(startingSeq[middlePosition+friendlyBlockStart]<startingSeq[gid]){
                                leftPosition=middlePosition;
                            }
                            else{
                                if(myLocation==0){
                                    rightPosition=middlePosition;
                                }
                                else{
                                    leftPosition=middlePosition;
                                }
                            }
                        }
                    }
                    else{
                        rightPosition=middlePosition;
                    }
                }
            }
            myint writePos=absoluteStart+myPosition+rightPosition;
            resultingSeq[writePos]=startingSeq[gid];
            followerRes[writePos]=followerSt[gid];
        }
        else{
            resultingSeq[gid]=startingSeq[gid];
            followerRes[gid]=followerSt[gid];
        }
    }
}



__kernel void shiftCalculationMergSortFollDecGC(__global  double *startingSeq,
                                                __global myint *length,
                                                __global  double *resultingSeq,
                                                __global  myint *sortedSize,
                                                __global myint *followerSt,
                                                __global myint *followerRes
                                            )
{   myint gid = get_global_id(0);
    if(gid<*length){
        
        myint newSize=2 * (*sortedSize);
        myint absoluteStart=gid/newSize;
        myint friendlyBlockStart=absoluteStart * newSize;
        absoluteStart=friendlyBlockStart;
        myint myLocation=0;//left
        myint myPosition=gid%newSize;
        if( myPosition>= *sortedSize){
            myPosition-= (*sortedSize);
            myLocation=1;//right
        }
        else{
            friendlyBlockStart+=*sortedSize;
        }
        if(friendlyBlockStart<*length){
            myint rightPosition=-1;
            if(startingSeq[friendlyBlockStart]<startingSeq[gid]){
                rightPosition=0;
            }
            if((startingSeq[friendlyBlockStart]==startingSeq[gid])&&(myLocation==0)){
                rightPosition=0;
            }
            if(rightPosition!=0){
                myint leftPosition=0;
                rightPosition= *sortedSize;
                myint middlePosition;
                
                while (leftPosition+1<rightPosition){
                    middlePosition=(leftPosition+rightPosition)/2;
                    if(middlePosition + friendlyBlockStart<*length){
                        if(startingSeq[middlePosition+friendlyBlockStart]<startingSeq[gid]){
                            rightPosition=middlePosition;
                        }
                        else{
                            if(startingSeq[middlePosition+friendlyBlockStart]>startingSeq[gid]){
                                leftPosition=middlePosition;
                            }
                            else{
                                if(myLocation==0){
                                    rightPosition=middlePosition;
                                }
                                else{
                                    leftPosition=middlePosition;
                                }
                            }
                        }
                    }
                    else{
                        rightPosition=middlePosition;
                    }
                }
            }
            myint writePos=absoluteStart+myPosition+rightPosition;
            resultingSeq[writePos]=startingSeq[gid];
            followerRes[writePos]=followerSt[gid];
        }
        else{
            resultingSeq[gid]=startingSeq[gid];
            followerRes[gid]=followerSt[gid];
        }
    }
}



//MERGE SORT WITH FOLLOWER END



//FENWICK SUMMATION BEGIN
__kernel void createFenwickGC(__global double *fSeqGCCL,
                              __global myint *fSeqLengthGCCL,
                              __global myint *layerSkipGCCL){
    myint readingPosition=get_global_id(0) * (*layerSkipGCCL) * 2  + (*layerSkipGCCL)-1;
    myint writingPosition=readingPosition+(*layerSkipGCCL);
    if(writingPosition<*fSeqLengthGCCL){
        fSeqGCCL[writingPosition]+=fSeqGCCL[readingPosition];
        
    }
    
    
}
__kernel void justSummationFenwickGC(__global double *fSeqGCCL,
                                     __global myint *fSeqLengthGCCL,
                                     __global myint *layerSkipGCCL,
                                     __global double *multiplier){
    myint writingPosition=get_global_id(0) * (*layerSkipGCCL) * 2;
    myint readingPosition=writingPosition+(*layerSkipGCCL);
    if(readingPosition<*fSeqLengthGCCL){
        fSeqGCCL[writingPosition]+=fSeqGCCL[readingPosition];
    }

    if(writingPosition<*fSeqLengthGCCL){
        fSeqGCCL[writingPosition]*=*multiplier;
    }

}


double inversionFenwick(__global double *fSeqGCCL,
                        myint k, myint twoToK, myint Q){
    double fR=fSeqGCCL[Q];
    myint fEnding=0;
    myint helpQ=Q;
    while(helpQ%2==1){
        fEnding+=1;
        helpQ/=2;
    }
    myint powerKeep=twoToK;
    myint i=k;
    while(i<fEnding){
        fR-=fSeqGCCL[Q-powerKeep];
        powerKeep*=2;
        i+=1;
    }
    return fR;
}

__kernel void partialSumsFTreeGC(__global double *fSeqGCCL,
                                 __global myint *resSeqLengthGCCL,
                                 __global myint *fromSeqGCCL,
                                 __global myint *toSeqGCCL,
                                 __global double *psumSeqGCCL){
    myint i=get_global_id(0);
    if(i<*resSeqLengthGCCL){
        double overflow=0.0;
        
        myint Left=fromSeqGCCL[i];
        myint Right=toSeqGCCL[i];
        
        myint k=0;myint twoToK=1;
        myint pHelp,pHelpOverTwoToK;
        
        while(Left!=Right){
            pHelp=Left-twoToK+1;
            pHelpOverTwoToK=pHelp/twoToK;
            if(pHelpOverTwoToK%2==0){
                // Left runner is the left child
                // There is no overflow
                Left=2 * Left-pHelp+1;
            }
            else{
                // Left runner is the right child
                // There is overflow.
                overflow+=inversionFenwick(fSeqGCCL,k,twoToK,Left-twoToK);
            }
            pHelp=Right-twoToK+1;
            pHelpOverTwoToK=pHelp/twoToK;
            if(pHelpOverTwoToK%2==0){
                // Right runner is the left child
                // There is overflow
                
                overflow+=inversionFenwick(fSeqGCCL,k,twoToK,Right+twoToK);
                Right=2 * Right-pHelp+1;
            }
            twoToK*=2;
            k+=1;
        }
        psumSeqGCCL[i]=inversionFenwick(fSeqGCCL,k,twoToK, Left)-overflow;
        

        
    }
    
}
//FENWICK SUMMATION END

//PREFIX SUMS BEGIN
__kernel void stage1PSSGC(__global double *orig,
                          __global double *dest,
                          __global myint *N,
                          __global myint *A,
                          __global myint *Pos){
    myint i=get_global_id(0);
    myint writingPosition=i* (*A)+ (*Pos);
    if(writingPosition<*N){
        if(*Pos!=0){
            dest[writingPosition]=dest[writingPosition-1]+orig[writingPosition];
        }
        else{
            dest[writingPosition]=orig[writingPosition];
        }
    }
}

__kernel void stage2PSSGC(__global double *dest,
                          __global double *orig,
                          __global myint *N,
                          __global myint *A,
                          __global myint *Pos){
    myint i=get_global_id(0);
    myint firstPos=(*A) * (*Pos);
    myint writingPosition=firstPos +i;
    double toAdd;
    if(*Pos==0){
        toAdd=0;
    }
    else{
        toAdd=dest[firstPos-1];
    }
    if((writingPosition<*N)&&(writingPosition<firstPos+(*A))){
        dest[writingPosition]=toAdd+orig[writingPosition];
    }
}

__kernel void partialSumsPSSGC(__global double *fSeqGCCL,
                                 __global myint *length,
                                 __global myint *fromSeqGCCL,
                                 __global myint *toSeqGCCL,
                                 __global double *psumSeqGCCL){
    myint i=get_global_id(0);
    if(i<*length){
        double leftPart=0;
        myint fPos=fromSeqGCCL[i];
        if(fPos>0){
            leftPart=fSeqGCCL[fPos-1];
        }
        
        psumSeqGCCL[i]=fSeqGCCL[toSeqGCCL[i]]-leftPart;
    }
}
    
    
    //PREFIX SUMS END




