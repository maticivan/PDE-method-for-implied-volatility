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
#include "cImpVolBasic.h"


CImpVol::CImpVol(const myint &_selectConstructor){
    indStruForX.clear();
    indStruForK.clear();
    if(_selectConstructor==0){
        loadData();
        ii_createIndexStructures();
    }
 
}
CImpVol::~CImpVol(){
    //
}
mydouble CImpVol::evaluateYPol2D(const mydouble &x,
                                 const mydouble &k,
                                 const mydouble &x0,
                                 const mydouble &k0,
                                 const std::vector<mydouble> &alpha) const {
    
    
    
    mydouble fR=0.0;
    
    
    
    mydouble f= x-x0;
    
    mydouble s= k-k0;
    mydouble helpCoeff;
    

    
    myint t,j;
  
    for(t= degreeX;t>-1;--t){
        fR *= s ;
        helpCoeff=0.0;
        
        
        for(j= dKP1M+t;j>-1;j-=dKP1){
            helpCoeff*= f;
            
            
            helpCoeff+= alpha[j];
            
        }
        fR+=helpCoeff;
    }
    return fR;
    
    
}
mydouble CImpVol::square_md(const mydouble &x)const {
    return x*x;
}
mydouble CImpVol::Af(const mydouble &x) const {
    mydouble sgn=1.0;
    if(x<0.0){
        sgn=-1.0;
    }
    return 0.5*(1.0+sgn
                *std::sqrt(1.0-std::exp(-M_2_PI*x*x)));
}
mydouble CImpVol::evaluateSigmaSR(const mydouble &K, const mydouble &F,
                                  const mydouble &marketValue,const  mydouble &df) const{
        

    
    const mydouble ey = F/K;
    const mydouble ey2 = ey*ey;
    const mydouble y = std::log(ey);
    const mydouble alpha = marketValue/(K*df);

    const mydouble R = 2*alpha -ey+1.0 ;
    const mydouble R2 = R*R;
    
    const mydouble a = std::exp((1.0-M_2_PI)*y);
    const mydouble A = square_md(a - 1.0/a);
    const mydouble b = std::exp(M_2_PI*y);
    const mydouble B = 4.0*(b + 1/b)
    - 2*K/F*(a + 1.0/a)*(ey2 + 1 - R2);
    const mydouble C = (R2-square_md(ey-1))*(square_md(ey+1)-R2)/ey2;
    
    const mydouble beta = 2*C/(B+std::sqrt(B*B+4*A*C));
    const mydouble gamma = -M_PI_2*std::log(beta);
    
    if (y >= 0.0) {

        const mydouble M0 = K*df*(  ey*Af(std::sqrt(2*y)) - 0.5 );
        if (marketValue <= M0)
            return std::sqrt(gamma+y)-std::sqrt(gamma-y);
        else
            return std::sqrt(gamma+y)+std::sqrt(gamma-y);
    }
    else {

        const mydouble M0 = K*df*( 0.5*ey - Af(-std::sqrt(-2*y)) );
        if (marketValue <= M0)
            return std::sqrt(gamma-y)-std::sqrt(gamma+y);
        else
            return std::sqrt(gamma+y)+std::sqrt(gamma-y);
    }
    
    

}

mydouble CImpVol::evaluateYPol2D(const mydouble &f,
                                 const mydouble &s,
                                 const std::vector<mydouble> &alpha) const {
    
    
    
    
    //LOOPS ARE UNROLLED: WORKS ONLY IF DEGREES ARE D=8
mydouble helpCoeff = (((((((alpha[80]*f+alpha[71])*f+alpha[62])*f+alpha[53])*f+alpha[44])*f+alpha[35])*f+alpha[26])*f+alpha[17])*f+alpha[8];
mydouble fR = helpCoeff*s;
helpCoeff = (((((((alpha[79]*f+alpha[70])*f+alpha[61])*f+alpha[52])*f+alpha[43])*f+alpha[34])*f+alpha[25])*f+alpha[16])*f+alpha[7];
fR += helpCoeff;
fR *= s;
helpCoeff = (((((((alpha[78]*f+alpha[69])*f+alpha[60])*f+alpha[51])*f+alpha[42])*f+alpha[33])*f+alpha[24])*f+alpha[15])*f+alpha[6];
fR += helpCoeff;
fR *= s;
helpCoeff = (((((((alpha[77]*f+alpha[68])*f+alpha[59])*f+alpha[50])*f+alpha[41])*f+alpha[32])*f+alpha[23])*f+alpha[14])*f+alpha[5];
fR += helpCoeff;
fR *= s;
helpCoeff = (((((((alpha[76]*f+alpha[67])*f+alpha[58])*f+alpha[49])*f+alpha[40])*f+alpha[31])*f+alpha[22])*f+alpha[13])*f+alpha[4];
fR += helpCoeff;
fR *= s;
helpCoeff = (((((((alpha[75]*f+alpha[66])*f+alpha[57])*f+alpha[48])*f+alpha[39])*f+alpha[30])*f+alpha[21])*f+alpha[12])*f+alpha[3];
fR += helpCoeff;
fR *= s;
helpCoeff = (((((((alpha[74]*f+alpha[65])*f+alpha[56])*f+alpha[47])*f+alpha[38])*f+alpha[29])*f+alpha[20])*f+alpha[11])*f+alpha[2];
fR += helpCoeff;
fR *= s;
helpCoeff = (((((((alpha[73]*f+alpha[64])*f+alpha[55])*f+alpha[46])*f+alpha[37])*f+alpha[28])*f+alpha[19])*f+alpha[10])*f+alpha[1];
fR += helpCoeff;
fR *= s;
helpCoeff = (((((((alpha[72]*f+alpha[63])*f+alpha[54])*f+alpha[45])*f+alpha[36])*f+alpha[27])*f+alpha[18])*f+alpha[9])*f+alpha[0];
fR += helpCoeff;
return fR;

    
}

void CImpVol::totalPartition(mydouble * fromSeq, myint len){
    
    
    
    myint nthreads;
    nthreads=1;
   
    
    myint pos=0;

    randomY_optional=fromSeq[pos];++pos;
    randomK_optional=fromSeq[pos];++pos;

    partK.clear();
    myint lK=myint(fromSeq[pos]+0.001);++pos;
    partK.resize(lK);
 
    for(myint i=0;i<lK;++i){
        partK[i]=fromSeq[pos];++pos;
    }
    
    
    
    
    pKX.resize(lK);
    pKY.resize(lK);
    
    
    std::vector<myint> vectorSizesSequence(lK);
    
    for(int i=0;i<lK;++i){
        vectorSizesSequence[i]=myint(fromSeq[pos]+0.001);++pos;
    }
    
    
    myint xDim=10000;
    
    {
        
        myint myId=0;
        myint numberOfMyJobs=(lK/nthreads);
        if(lK%nthreads!=0){
            ++numberOfMyJobs;
        }
        std::vector<mydouble> ppX,ppY;
        
        myint i,j;
        myint lX;
        for(myint iCounter=0;iCounter<numberOfMyJobs;++iCounter){
            i= iCounter*nthreads+myId;
            if(i<lK){
                ppX.clear();
                ppY.clear();
                
                ppX.resize(vectorSizesSequence[i]);
                ppY.resize(vectorSizesSequence[i]);
                for(j=0;j<vectorSizesSequence[i];++j){
                    ppX[j]=fromSeq[pos];++pos;
                    ppY[j]=fromSeq[pos];++pos;
                    
                }
                pKX[i]=ppX;
                pKY[i]=ppY;
                
                
                
            }
        }
        
    }
    
    
}

std::vector<mydouble> CImpVol::createAlpha(mydouble * fromSeq,
                                           myint *pos,
                                           const myint &uMax,
                                           const myint &vMax){
    
    
    std::vector<mydouble> alpha;
    alpha.resize((vMax+1)*(uMax+1));
    
    for (myint i = 0;i < uMax+1;++i){
        for(myint j = 0; j < vMax+1;++j){
            alpha[(vMax+1)*i+j]=fromSeq[*pos];
            *pos+=1;
        }
    }
    return(alpha);
}

void CImpVol::createPolynomials2D(mydouble* fromSeq, myint len){
    
    
    
    myint nthreads;
    nthreads=1;
    
    myint pos=0;
 
    myint numKsL=partK.size();
    
    
    coeff.resize(numKsL);
    
    
    {
        
        myint myId=0;
        myint numberOfMyJobs=(numKsL/nthreads);
        if(numKsL%nthreads!=0){
            ++numberOfMyJobs;
        }
        myint kC;
        mydouble k0,x0,Y0;
        myint cNX,xC;
        for(myint kCCounter=0; kCCounter<numberOfMyJobs;++kCCounter){
            kC=kCCounter*nthreads+myId;
            if(kC<numKsL){
                k0= partK[kC];
                cNX = pKX[kC].size();
                
                coeff[kC].resize(cNX);
                
                
                for ( xC=0;xC< cNX;++xC){
                    
                    x0=pKX[kC][xC];
                    Y0=pKY[kC][xC];
                    
                    std::vector<mydouble> helperSeq=createAlpha(fromSeq,&pos,degreeX, degreeK);
                    
                    coeff[kC][xC]=helperSeq;
                }
            }
        }
        
    }
    
}
mydouble CImpVol::evaluate(const mydouble& _k,const mydouble & _x){
    
    mydouble k=_k;
    mydouble x=_x;
    if(k<0.0){
        k*= -1.0;
        x= (x-1.0)*exp(k)+1.0;
        
    }
    
    myint numKsL = partK.size();
    
    myint indK= -1;
    myint indX= -1;
    mydouble fR= -1.0;
    
    
 
    indK = indStruForK.getIndexOfSeq(k);
    
    if(indK>=0){
        indX=indStruForX[indK].getIndexOfSeq(x);
    }
    if((k>=partK[0])&&(k<=partK[numKsL-1])&&(indK>=0)&&(indX>=0)){
        fR= evaluateYPol2D(x,k,pKX[indK][indX],partK[indK], coeff[indK][indX]);

    }
   
    return (fR);
    
    
    
}


std::vector<mydouble> CImpVol::evaluate(const std::vector<mydouble>& _k,
                                        const std::vector<mydouble>& _x){
    
    myint lengths=_k.size();
 
    myint numKsL = partK.size();
    std::vector<mydouble> fR;
    fR.resize(lengths);
    
    myint nthreads;
#pragma omp parallel
    {
        if(omp_get_thread_num()==0){
            nthreads=omp_get_num_threads();
        }
    }
#pragma omp parallel num_threads(nthreads)
    {
        myint myId=omp_get_thread_num();
 
        mydouble k,x;
 
        myint indKSI,indXSI;
 
            while(myId<lengths){
                
                k=_k[myId];
                x=_x[myId];
                if(k<0.0){
                    k= 0.0-k;
                    x= (x-1.0)*exp(k)+1.0;
                    
                }
                
                fR[myId]=-1.0;
                
                
                indKSI = indStruForK.getIndexOfSeq(k);
              
                if(indKSI>=0){
                    indXSI=indStruForX[indKSI].getIndexOfSeq(x);
                    if((k>=partK[0])&&(k<=partK[numKsL-1])&&(indXSI>=0)){
                      
                        fR[myId]= evaluateYPol2D(x,k,pKX[indKSI][indXSI],partK[indKSI],
                                              coeff[indKSI][indXSI]);
                        
                    }
                }
                
                myId+=nthreads;
            }
        
        
    
    }
    
#pragma omp barrier
    
    return (fR);
    
    
    
}



myint CImpVol::getNextInteger(char * text, std::streampos size, std::streampos *pos){
    
    myint current=0;
    myint mult=1;
    
    while ((*pos<size) &&(text[*pos]!='-')&&(( text[*pos]<'0')||(text[*pos]>'9'))){
        *pos= (*pos)+((std::streampos)1);
    }
    if(text[*pos]=='-'){
        mult=-1;
        *pos= (*pos)+((std::streampos)1);
    }
    while ((*pos<size) &&(( text[*pos]>='0')&&(text[*pos]<='9'))){
        current= 10 * current;
        current+= (int)(text[*pos]-'0');
        *pos=*pos+((std::streampos)1);
    }
    current=current*mult;
    return current;
}


mydouble CImpVol::getNextReal(char * text, std::streampos size, std::streampos *pos){
    
    myint current=0;
    myint mult=1;
    
    while ((*pos<size) &&(text[*pos]!='-')&&(( text[*pos]<'0')||(text[*pos]>'9'))){
        *pos= (*pos)+((std::streampos)1);
    }
    if(text[*pos]=='-'){
        mult=-1;
        *pos= (*pos)+((std::streampos)1);
    }
    while ((*pos<size) &&(( text[*pos]>='0')&&(text[*pos]<='9'))){
        current= 10 * current;
        current+= (int)(text[*pos]-'0');
        *pos=*pos+((std::streampos)1);
    }
    
    mydouble dCurrent= mydouble(current);
    if((*pos<size) && (text[*pos]=='.')){
        *pos+=((std::streampos)1);
        mydouble decimal=0.1;
        
        myint currentDigit;
        while ((*pos<size) &&(( text[*pos]>='0')&&(text[*pos]<='9'))){
            currentDigit= (int)(text[*pos]-'0');
            dCurrent+= mydouble(currentDigit)*decimal;
            *pos=*pos+((std::streampos)1);
            decimal*=0.1;
        }
        
    }
    if((*pos<size)&&(text[*pos]=='e')){
        myint exponent=getNextInteger(text,size,pos);
        if(exponent<0){
            exponent*=(-1);
            for(int i=0;i<exponent;++i){
                dCurrent *= 0.1;
            }
        }
        else{
            for(int i=0;i<exponent;++i){
                dCurrent *= 10.0;
            }
        }
        
    }
    dCurrent=mydouble(mult)* dCurrent;
    return dCurrent;
}



void CImpVol::getBigString(const std::string &filename, char * &memblock, std::streampos &size){
    
    std::ifstream ifile(filename,std::ios::in|std::ios::binary|std::ios::ate);

    if (ifile.is_open())
    {
        size = ifile.tellg();
        memblock = new char [size];
        ifile.seekg (0, std::ios::beg);
        ifile.read (memblock, size);
        ifile.close();
        
    }
}

void CImpVol::loadData(){
    char *memblock;
    
    
    std::streampos *position, size;
    getBigString(CIMPV_CONST_DEFAULT_FILE_WITH_PARTITION, memblock,size);
    myint i;
     position=new std::streampos;
    *position=0;
    
    myint numTerms, numReal;
    myint nextInt;
    mydouble nextmydouble;
    numTerms=-1;
    while(numTerms!=CIMPV_CONST_INT_START_OF_PARTITION_DATA){
        numTerms = getNextInteger(memblock,size,position);
    }
    
    numTerms = getNextInteger(memblock,size,position);
    degreeK=getNextInteger(memblock,size,position);
    
    degreeX=getNextInteger(memblock,size,position);
    dKP1=degreeK+1;
    dKP1M=dKP1*degreeK;
    yMin=getNextReal(memblock,size,position);
    
    
    yMax=getNextReal(memblock,size,position);
    
    kMin=getNextReal(memblock,size,position);
    
    kMax=getNextReal(memblock,size,position);
    
    
    errorA=getNextReal(memblock,size,position);
    
    nextInt=getNextInteger(memblock,size,position);
    
    mydouble *partitionSeq;
    --nextInt;
    partitionSeq=new mydouble[nextInt];
    for( i=0;i<nextInt;++i){
        partitionSeq[i]=getNextReal(memblock,size,position);
    }
    
    totalPartition(partitionSeq,nextInt);
    delete[] partitionSeq;
    
    
    nextInt=getNextInteger(memblock,size,position);
    
    mydouble *polsSeq;
    --nextInt;
    polsSeq=new mydouble[nextInt];
    
    for(  i=0;i<nextInt;++i){
        polsSeq[i]=getNextReal(memblock,size,position);
    }
    createPolynomials2D(polsSeq,nextInt);
    delete[] polsSeq;
    
    
    
    delete position;
    delete[] memblock;
    
}


mydouble CImpVol::sigma_call(const mydouble &FF,
                                 const mydouble &C0,
                                 const mydouble &KK,
                                 const mydouble &rr,
                                 const mydouble &texp) {
    mydouble disc_fact = exp(-rr*texp);
    mydouble S0 = FF * disc_fact;
    
    mydouble k_Fukasawa = log(disc_fact * KK / S0);
    mydouble c_Fukasawa = C0 / S0;
    
    return evaluate(k_Fukasawa, c_Fukasawa);
}


mydouble CImpVol::iv(const mydouble &FF,
                     const mydouble &px,
                     const mydouble &KK,
                     const mydouble &rr,
                     const mydouble &texp,
                     const myint    &PC,
                     const myint &neverSR) {
    myint fakeVectorLength=128;
    std::vector<mydouble> vFF,vpx,vKK,vrr,vtexp,fR;
    std::vector<myint> vPC;
    vFF.resize(fakeVectorLength);
    vpx.resize(fakeVectorLength);
    vKK.resize(fakeVectorLength);
    vrr.resize(fakeVectorLength);
    vtexp.resize(fakeVectorLength);
    vPC.resize(fakeVectorLength);
    for(myint i=0;i<fakeVectorLength;++i){
        vFF[i]=FF;
        vpx[i]=px;
        vKK[i]=KK;
        vrr[i]=rr;
        vtexp[i]=texp;
        vPC[i]=PC;
    }
    fR=iv(vFF,vpx,vKK,vrr,vtexp,vPC,neverSR);
    return fR[0];
    
}

std::vector<mydouble> CImpVol::iv(const std::vector<mydouble> &FF,
                                  const std::vector<mydouble> &px,
                                  const std::vector<mydouble> &KK,
                                  const std::vector<mydouble> &rr,
                                  const std::vector<mydouble> &texp,
                                  const std::vector<myint>    &PC,
                                  const myint &neverSR) {
    myint lengths=rr.size();
    mydouble kMinMult101= kMin * 1.01;
    std::vector<mydouble> fR;
    fR.resize(lengths);
    myint numKsL = partK.size();
    mydouble leftK=partK[0];
    mydouble rightK=partK[numKsL-1];
    
    myint nthreads;
#pragma omp parallel
    {
        if(omp_get_thread_num()==0){
            nthreads=omp_get_num_threads();
        }
    }
#pragma omp parallel num_threads(nthreads)
    {
        
        mydouble  C0,k,x;
        
        myint i=omp_get_thread_num();
        myint indK,indX;
        mydouble reciprocalF;
        mydouble rMultT;
        mydouble kRecF;
        while(i<lengths){
           
            rMultT=rr[i]*texp[i];
            
  
            C0 = PC[i] ? px[i] : px[i] + FF[i] - exp(-rMultT)  * KK[i] ;
            
            reciprocalF=1.0/FF[i];
            
            kRecF=KK[i]*reciprocalF;
            k= log(kRecF) - rMultT;
            x = C0 *reciprocalF;
            
            
            
            
           
            if(k<kMin){
                if(k<-kMin){
                    x=((x-1.0)/kRecF) *exp(rMultT)+1.0;
                    
                    k=0.0-k;
                }
                else{
                    k=kMinMult101;
                }
            }
            
            
           
            
            
          
            
            if((k-leftK)*(rightK-k)>0.0){
                
                indK = indStruForK.getIndexOfSeq(k);
                if(indK>=0){
                    indX=indStruForX[indK].getIndexOfSeq(x);
                    if(indX>=0){
                        fR[i]= evaluateYPol2D(x-pKX[indK][indX],k-partK[indK], coeff[indK][indX])/sqrt(texp[i]);
                    }
                }
            }
            
            
            i+=nthreads;
        }
    }
 
    
    
  
#pragma omp barrier
 
    return fR;
}


std::vector<mydouble> CImpVol::getIntervalBounds() const{
    std::vector<mydouble> forReturn;
    forReturn.clear();
    
    mydouble cMin=0.0;
    mydouble cMax=100.0;
    myint numKs=partK.size();
    myint i,j,partXSize;
    std::vector<mydouble> vP;
    for(i=0;i<numKs;++i){
        vP=pKX[i];
        partXSize= (vP).size();
        if( (vP)[0]>cMin){
            cMin=(vP)[0];
        }
        if( (vP)[partXSize-1]<cMax){
            cMax=(vP)[partXSize-1];
        }
    }
    forReturn.push_back(cMin);
    forReturn.push_back(cMax);
    forReturn.push_back(kMin);
    forReturn.push_back(kMax);
    return forReturn;
}

////////////////////
////////////////////
// IMPROVING INDEXING
myint CImpVol::ii_numberOfElementsInK() const{
    return partK.size();
}


void CImpVol::ii_createIndexStructures(){
    indStruForK.createIndices(partK);
    
    myint numKs=partK.size();
    indStruForX.resize(numKs);
    for(myint i=0;i<numKs;++i){
        indStruForX[i].createIndices(pKX[i]);
    }
}
std::vector<myint> CImpVol::ii_getIndices(const mydouble & _k,
                                          const mydouble & _c) const{
    std::vector<myint> fR(2,-1);
    fR[0]=indStruForK.getIndexOfSeq(_k);
 
    if(fR[0]>0){
        fR[1]=indStruForX[fR[0]].getIndexOfSeq(_c);
    }
    
    return fR;
    
}
    