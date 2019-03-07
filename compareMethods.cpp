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



#include<iostream>


#include "cImpVolBasic.h"
#include "gcImpVolBasic.h"
#include "lets_be_rational.h"

using namespace std;

typedef long int myint;
typedef double mydouble;


const myint CONST_COMP_EXAMPLES_COMPARE_WITH_LI=0;
const myint CONST_SAMPLE_SIZE=1000000;




void GF_substitution(const mydouble &FF,
                     const mydouble &C0,
                     const mydouble &KK,
                     const mydouble &rr,
                     const mydouble &texp,
                     mydouble &c_GF,
                     mydouble &k_GF){
    mydouble disc_fact = exp(-rr*texp);
    mydouble S0 = FF * disc_fact;
    k_GF = log(disc_fact * KK / S0);
    c_GF = C0 / S0;
}




myint randSeq(std::vector<myint> &seq, const myint &length,  myint uniformLimit=3431){
    
    seq.resize(length);
    std::uniform_int_distribution<myint> uInt(0,uniformLimit);
    myint seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::mt19937 mt_randIF(seed);
    for(myint i=0;i<length;i++){
        
        seq[i]= uInt(mt_randIF);
        
        
    }
    return 1;
    
}

mydouble normalCDF(mydouble x){
    mydouble sqrt12=0.707106781186547524401;
    return 0.5+0.5*erf(x*sqrt12 );
}

mydouble rescale(const mydouble &minV,
                 const mydouble &maxV,
                 const myint &num,
                 const mydouble &uld){
    return minV+(maxV-minV)* static_cast<mydouble>(num/uld);
}


mydouble price_call_BS(const mydouble &FF,
                       const mydouble &KK,
                       const mydouble &rr,
                       const mydouble &sigma,
                       const mydouble &texp){
    mydouble fR;
    mydouble frac=FF/KK;
    mydouble ssq=sigma*sigma;
    mydouble ssqrtt=sigma *sqrt(texp);
    mydouble dplus=(log(frac)+ ssq*0.5*texp)/(ssqrtt);
    mydouble dminus=dplus-ssqrtt;
    fR=FF * normalCDF(dplus)- KK*normalCDF(dminus);
    fR*= exp(-rr*texp);
    return fR;
    
    
}

void randomizeInput(const mydouble &kMin, const mydouble &kMax,
                    const mydouble &cMin, const mydouble &cMax,
                    const mydouble &FFMin, const mydouble &FFMax,
                    const mydouble &pxMin, const mydouble &pxMax,
                    const mydouble &KKMin, const mydouble &KKMax,
                    const mydouble &rrMin, const mydouble &rrMax,
                    const mydouble &texpMin, const mydouble &texpMax,
                    std::vector<mydouble> & _FF,
                    std::vector<mydouble> & _px,
                    std::vector<mydouble> & _KK,
                    std::vector<mydouble> & _rr,
                    std::vector<mydouble> & _texp,
                    std::vector<myint>    & _pc){
    myint lens=_FF.size();
    
    myint uniformLimit=10000;
    mydouble uLD=static_cast<mydouble>(uniformLimit);
    std::vector<myint> FFNum,pxNum,KKNum,rrNum,texpNum,pcNum,CNum,KNum;
    mydouble c_GF,k_GF;
    
    randSeq(CNum, lens, uniformLimit);
    randSeq(KNum, lens, uniformLimit);
    
    randSeq(FFNum, lens, uniformLimit);
    KKNum.resize(lens);
    pxNum.resize(lens);
    randSeq(rrNum, lens, uniformLimit);
    randSeq(texpNum, lens, uniformLimit);
    randSeq(pcNum, lens, uniformLimit);
    
    mydouble tempC,tempK,disc_fact,S0;
     
    for(myint i=0;i<lens;++i){
        tempC=rescale(cMin,cMax,CNum[i],uLD);
        tempK=rescale(kMin,kMax,KNum[i],uLD);
        
        _FF[i]=rescale(FFMin,FFMax,FFNum[i],uLD);
        _rr[i]=rescale(rrMin,rrMax,rrNum[i],uLD);
        _rr[i]=0.0;
        _texp[i]=rescale(texpMin,texpMax,texpNum[i],uLD);
        disc_fact=exp(- _rr[i] * _texp[i]);
        
        S0=_FF[i] * disc_fact;
        _px[i]= S0 * tempC;
        _KK[i] = exp(tempK) * S0/disc_fact;
        _pc[i]=pcNum[i]%2;
        _pc[i]=1;
        GF_substitution(_FF[i],
                        _px[i],
                        _KK[i],
                        _rr[i],
                        _texp[i],
                        c_GF,
                        k_GF);
   
        
        
    }
   
}

std::vector<mydouble> ivBisection(const std::vector<mydouble> &FF,
                                  const std::vector<mydouble> &px,
                                  const std::vector<mydouble> &KK,
                                  const std::vector<mydouble> &rr,
                                  const std::vector<mydouble> &texp,
                                  const std::vector<myint>    &PC,
                                  const mydouble &tolerance=0.00000001) {
    myint lengths=rr.size();
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
        
        mydouble disc_fact, C0,k,x;
        
        myint i=omp_get_thread_num();
        myint indK,indX;
        mydouble frac;
        
        while(i<lengths){
            
            
            disc_fact= exp(-rr[i]*texp[i]);
            C0 = PC[i] ? px[i] : px[i] + disc_fact  * (FF[i] - KK[i]);
            
            frac=KK[i]/FF[i];
            
            k = log( frac);
            x = C0 / (FF[i] * disc_fact);
            
            
            
            
            mydouble leftEndpoint=0.0;
            mydouble rightEndpoint=10.0;
            mydouble middlePoint,needle,dplus,dminus;
            
            while(rightEndpoint-leftEndpoint>tolerance){
                
                middlePoint=(leftEndpoint+rightEndpoint)*0.5;
                dplus=-(k/middlePoint)+middlePoint*0.5;
                dminus=dplus-middlePoint;
                needle=normalCDF(dplus)-normalCDF(dminus)*frac;
                if(needle>x){
                    rightEndpoint=middlePoint;
                }
                else{
                    leftEndpoint=middlePoint;
                }
            }
            middlePoint=(leftEndpoint+rightEndpoint)*0.5;
            fR[i]=middlePoint/sqrt(texp[i]);
            
            
            
            
            i+=nthreads;
        }
    }
    
    
    
    
#pragma omp barrier
    
    return fR;
}


mydouble impVol_LBR(const mydouble &FF,
                            const mydouble &px,
                            const mydouble &KK,
                            const mydouble &rr,
                            const mydouble &texp,
                            const myint    &PC) {
   
    
    return implied_volatility_from_a_transformed_rational_guess(px, FF, KK, texp, 1.0);
}
std::vector<mydouble> impVol_LBR(const std::vector<mydouble> &FF,
                                 const std::vector<mydouble> &px,
                                 const std::vector<mydouble> &KK,
                                 const std::vector<mydouble> &rr,
                                 const std::vector<mydouble> &texp,
                                 const std::vector<myint>    &PC) {
    std::vector<mydouble> fR;
    myint lengths=rr.size();
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
        myint numberOfMyJobs=(lengths/nthreads);
        
        if(lengths%nthreads!=0){
            ++numberOfMyJobs;
        }
        myint i=myId;
        while(i<lengths){
            if(PC[i]==1){
                fR[i]=implied_volatility_from_a_transformed_rational_guess(px[i], FF[i], KK[i], texp[i], 1.0);
            }
            i+=nthreads;
        }

    }
    
    
    
    
#pragma omp barrier

    return fR;
}

mydouble Af(mydouble x) {
    mydouble sgn=1.0;
    if(x<0.0){
        sgn=-1.0;
    }
    return 0.5*(1.0+sgn
                *std::sqrt(1.0-std::exp(-M_2_PI*x*x)));
}
mydouble inverseNormalBSM(const mydouble& quantile)   {
    // This is the Beasley-Springer-Moro algorithm which can
    // be found in Glasserman [2004]. We won't go into the
    // details here, so have a look at the reference for more info
    if((quantile<0.00000000001)||(quantile>1.000000001)){
        return -70.0;
    }
    static mydouble a[4] = {   2.50662823884,
        -18.61500062529,
        41.39119773534,
        -25.44106049637};
    
    static mydouble b[4] = {  -8.47351093090,
        23.08336743743,
        -21.06224101826,
        3.13082909833};
    
    static mydouble c[9] = {0.3374754822726147,
        0.9761690190917186,
        0.1607979714918209,
        0.0276438810333863,
        0.0038405729373609,
        0.0003951896511919,
        0.0000321767881768,
        0.0000002888167364,
        0.0000003960315187};
    
    if (quantile >= 0.5 && quantile <= 0.92) {
        
        mydouble num = 0.0;
        mydouble denom=0.0;
        mydouble q=quantile-0.5;
        mydouble r=q*q;
        for (int i=0; i<4; i++) {
            num*=r;
            denom*=r;
            num+=a[3-i];
            denom+=b[3-i];
        }
        denom*=r;
        num*=q;
        denom+=1.0;
        return num/denom;
        
    } else if (quantile > 0.92 && quantile < 1) {
        mydouble num = 0.0;
        mydouble r=log(-log(1-quantile));
        for (int i=0; i<9; i++) {
            num *= r;
            num += c[8-i];
        }
        return num;
        
    } else {
        return -1.0*inverseNormalBSM(1-quantile);
    }
}

mydouble square_md(mydouble x){
    return x*x;
}
mydouble Np(mydouble x, mydouble v) {
    return normalCDF(x/v + 0.5*v);
}
mydouble Nm(mydouble x, mydouble v) {
    return std::exp(-x)*normalCDF(x/v - 0.5*v);
}
mydouble phi(mydouble x, mydouble v) {
    const mydouble ax = 2*std::fabs(x);
    const mydouble v2 = v*v;
    return (v2-ax)/(v2+ax);
}
mydouble F(mydouble v, mydouble x, mydouble cs, mydouble w) {
    return cs+Nm(x,v)+w*Np(x,v);
}

mydouble cbsF(mydouble x,mydouble v){
    return Np(x,v)-Nm(x,v);
}
mydouble FBad(mydouble v, mydouble x, mydouble vs, mydouble w) {
    mydouble cs=cbsF(x,vs);
    return cs+Nm(x,v)+w*Np(x,v);
}

mydouble G(mydouble v, mydouble x, mydouble cs, mydouble w) {
    const mydouble q = F(v,x,cs,w)/(1+w);
    
    // Acklam's inverse w/o Halley's refinement step
    // does not provide enough accuracy. But both together are
    // slower than the boost replacement.
    const mydouble k = inverseNormalBSM(q);
    
    return k + std::sqrt(k*k + 2*std::fabs(x));
}


// type for call=1, for put something else
mydouble blackFormulaImpliedStdDevApproximationRS(myint type, mydouble K, mydouble F,
                                                  mydouble marketValue, mydouble df, mydouble displacement) {
    
 
    F = F + displacement;
    K = K + displacement;
    
    const mydouble ey = F/K;
    const mydouble ey2 = ey*ey;
    const mydouble y = std::log(ey);
    const mydouble alpha = marketValue/(K*df);
    const mydouble R = 2*alpha + ((type == 1) ? -ey+1.0 : ey-1.0);
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
        const mydouble M0 = K*df*(
                                  (type == 1) ? ey*Af(std::sqrt(2*y)) - 0.5
                                  : 0.5-ey*Af(-std::sqrt(2*y)));
        
        if (marketValue <= M0)
            return std::sqrt(gamma+y)-std::sqrt(gamma-y);
        else
            return std::sqrt(gamma+y)+std::sqrt(gamma-y);
    }
    else {
        const mydouble M0 = K*df*(
                                  (type == 1) ? 0.5*ey - Af(-std::sqrt(-2*y))
                                  : Af(std::sqrt(-2*y)) - 0.5*ey);
        
        if (marketValue <= M0)
            return std::sqrt(gamma-y)-std::sqrt(gamma+y);
        else
            return std::sqrt(gamma+y)+std::sqrt(gamma-y);
    }
}

mydouble blackFormulaImpliedStdDevLiRS(
                                       myint optionType,
                                       mydouble strike,
                                       mydouble forward,
                                       mydouble blackPrice,
                                       mydouble discount,
                                       mydouble displacement,
                                       mydouble guess,
                                       mydouble relaxationParameter,
                                       mydouble accuracy,
                                       myint maxIterations) {
    
 
    strike = strike + displacement;
    forward = forward + displacement;
    if (guess < 0.000001) {
        guess = blackFormulaImpliedStdDevApproximationRS(
                                                         optionType, strike, forward,
                                                         blackPrice, discount, displacement);
    }
    
    mydouble x = std::log(forward/strike);
    mydouble cs = (optionType == 1)
    ? blackPrice / (forward*discount)
    : (blackPrice/ (forward*discount) + 1.0 - strike/forward);
    
    
    if (x > 0) {
        cs = forward/strike*cs + 1.0 - forward/strike;
        x = -x;
    }
    
    myint nIter = 0;
    mydouble dv, vk, vkp1 = guess;
    
    do {
        vk = vkp1;
        const mydouble alphaK = (1.0+relaxationParameter)/(1.0+phi(x,vk));
        vkp1 = alphaK*G(vk,x,cs,relaxationParameter) + (1.0-alphaK)*vk;
        
        dv = std::fabs(vkp1 - vk);
        
        
    } while (dv > accuracy && ++nIter < maxIterations);
    
    return vk;
}


mydouble impVol_LiRS(const mydouble &FF,
                    const mydouble &px,
                    const mydouble &KK,
                    const mydouble &rr,
                    const mydouble &texp,
                    const myint    &PC) {
    
    
    mydouble fR= blackFormulaImpliedStdDevLiRS(PC,
                                               KK,
                                               FF,
                                               px,
                                               1.0,
                                               0.0,
                                               0.0,
                                               1.0,
                                               0.000001,
                                               10000);
    
    return fR/sqrt(texp);
}


std::vector<mydouble> impVol_LiRS(const std::vector<mydouble> &FF,
                                 const std::vector<mydouble> &px,
                                 const std::vector<mydouble> &KK,
                                 const std::vector<mydouble> &rr,
                                 const std::vector<mydouble> &texp,
                                 const std::vector<myint>    &PC) {
    std::vector<mydouble> fR;
    myint lengths=rr.size();
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
        myint numberOfMyJobs=(lengths/nthreads);
        
        if(lengths%nthreads!=0){
            ++numberOfMyJobs;
        }
        
        myint i=myId;
        while(i<lengths){
            fR[i]=impVol_LiRS(FF[i], px[i], KK[i], rr[i], texp[i], PC[i]);
            i+=nthreads;
        }
        
    }
    
    
    
    
#pragma omp barrier
    
    return fR;
}




int main() {
    

 
    std::string choiceAgain;
    myint realWorldData=0;

    mydouble testImpVol;
    myint tBegin, tEnd;
    mydouble totalTime;
    mydouble totalTimeLiRS;
    mydouble totalTimeOld;
    mydouble totalTimePreparation;
    std::cout<<"Preparation of polynomials and partitions. Please wait ... "<<std::endl;
    tBegin=std::chrono::high_resolution_clock::now().time_since_epoch().count();
    CImpVol calc3;
    GCImpVol calc3GC;
    tEnd=std::chrono::high_resolution_clock::now().time_since_epoch().count();
    totalTime=(static_cast<mydouble>(tEnd-tBegin) )/(static_cast<mydouble>(1000000000));
    std::cout<<"Thank you for waiting "<<totalTime<<" seconds."<<std::endl;
 
    
    std::cout<<"IMPLIED VOLATILITIES"<<std::endl;
    mydouble F0=38.985;  // FORWARD PRICE OF THE UNDERLYING SECURITY
    mydouble C0=2.215;   // PRICE OF THE DERIVED SECURITY
    mydouble T=0.122011; // TIME TO EXPIRATION
    mydouble r=0.0;    // INTEREST RATE
    mydouble K=39.5;     // STRIKE
    
    
    
    mydouble kMin,kMax,cMin,cMax;
    
    cMin=0.000001;
    cMax=0.94488;
    kMin=0.000001;
    kMax=5.0;
    
    
    

    myint numTrials,sampleSize;
    numTrials=1;
    std::vector<mydouble> vect_FF,vect_px,vect_KK,vect_rr, vect_texp, vect_impv;
    std::vector<mydouble> vect_impvOther;
    std::vector<myint> vect_pc;
    
    
    cl_double *seq_FF,*seq_px,*seq_KK,*seq_rr, *seq_texp, *seq_impv;
     
    myint* seq_pc;
    
    mydouble FFMin,FFMax,pxMin,pxMax,KKMin,KKMax,rrMin,rrMax,tMin,tMax, c_GF,k_GF;
    
    FFMin=0.0001;
    FFMax=5.0;
    FFMax=50.0;
    pxMin=0.0001;
    pxMax=50.0;
    KKMin=0.0001;
    KKMax=5.0;
    rrMin=0.001;
    rrMax=2.0;
    tMin=0.1;
    tMax=10.0;
    
    mydouble kThatMaximizesTheError,kThatMaximizesRepricingError;

    mydouble maxError, currentError, errorImproved,SRF,noSRF,noSRFError,discount;
    mydouble repricingErrorC,repricingErrorMax,reprice;
    mydouble smallestIntVol, largestIntVol,currentIntVol;
    mydouble smallestC;
    mydouble smallestKOverIntVol, largestKOverIntVol,currentKOverIntVol;
    myint numBad,noSR;
    while(numTrials > 0){
        
        
        maxError=0.0;numBad=0;repricingErrorMax=0.0;
        if(numTrials>1){
            std::cout<<"Do you want to run another experiment? [Y/N] ";
            std::cin>>choiceAgain;
            if((choiceAgain=="n")||(choiceAgain=="N")){
                numTrials=-1;
            }
        }
        if(numTrials>0){
            sampleSize=CONST_SAMPLE_SIZE;
            
            vect_FF.resize(sampleSize);
            vect_px.resize(sampleSize);
            vect_KK.resize(sampleSize);
            vect_rr.resize(sampleSize);
            vect_FF.resize(sampleSize);
            vect_texp.resize(sampleSize);
            vect_pc.resize(sampleSize);
            vect_impv.resize(sampleSize);
            vect_impvOther.resize(sampleSize);
            
            
            std::cout<<"Randomizing input..."<<std::endl;
            randomizeInput(kMin, kMax,
                           cMin, cMax,
                           FFMin,FFMax,pxMin,pxMax,KKMin,KKMax,rrMin,rrMax,tMin,tMax,
                           vect_FF,vect_px,vect_KK,vect_rr,vect_texp,vect_pc);
            
            
            
            
            
            seq_impv=nullptr;
            seq_FF=new cl_double[sampleSize];
            seq_px=new cl_double[sampleSize];
            seq_KK=new cl_double[sampleSize];
            seq_rr=new cl_double[sampleSize];
            seq_texp=new cl_double[sampleSize];
            seq_pc=new myint[sampleSize];
            for(myint i=0;i<sampleSize;++i){
                seq_FF[i]=vect_FF[i];
                seq_px[i]=vect_px[i];
                seq_KK[i]=vect_KK[i];
                seq_rr[i]=vect_rr[i];
                seq_texp[i]=vect_texp[i];
                seq_pc[i]=vect_pc[i];
            }
            
            if(numTrials==1){
                numTrials=2;
                seq_impv=calc3GC.ivGC(seq_FF,seq_px,seq_KK,seq_rr, seq_texp,seq_pc,sampleSize);
                if(seq_impv!=nullptr){delete[] seq_impv;}
                seq_impv=nullptr;
                
                delete[] seq_FF;
                delete[] seq_px;
                delete[] seq_KK;
                delete[] seq_rr;
                delete[] seq_texp;
                delete[] seq_pc;
                
                seq_FF=new cl_double[sampleSize];
                seq_px=new cl_double[sampleSize];
                seq_KK=new cl_double[sampleSize];
                seq_rr=new cl_double[sampleSize];
                seq_texp=new cl_double[sampleSize];
                seq_pc=new myint[sampleSize];
                for(myint i=0;i<sampleSize;++i){
                    seq_FF[i]=vect_FF[i];
                    seq_px[i]=vect_px[i];
                    seq_KK[i]=vect_KK[i];
                    seq_rr[i]=vect_rr[i];
                    seq_texp[i]=vect_texp[i];
                    seq_pc[i]=vect_pc[i];
                }
                
                
                
            }
            
            std::cout<<"Finished preparing the input.\n";
            std::cout<<"Jaeckel's method:\t"<<std::flush;
            
            
            
            
            tBegin=std::chrono::high_resolution_clock::now().time_since_epoch().count();
            vect_impvOther=impVol_LBR(vect_FF,vect_px,vect_KK,vect_rr,vect_texp, vect_pc);
            tEnd=std::chrono::high_resolution_clock::now().time_since_epoch().count();
            totalTime=(static_cast<mydouble>(tEnd-tBegin) )/(static_cast<mydouble>(1000000000));
            
            std::cout<<" "<<totalTime<<std::endl;
            
            totalTimeOld=totalTime;
            if(CONST_COMP_EXAMPLES_COMPARE_WITH_LI==1){
            
                std::cout<<"Li's method:\t"<<std::flush;
                tBegin=std::chrono::high_resolution_clock::now().time_since_epoch().count();
                
                vect_impv=impVol_LiRS(vect_FF,vect_px,vect_KK,vect_rr,
                                      vect_texp,vect_pc);
                
                tEnd=std::chrono::high_resolution_clock::now().time_since_epoch().count();
                totalTimeLiRS=(static_cast<mydouble>(tEnd-tBegin) )/(static_cast<mydouble>(1000000000));
                std::cout<<" "<<totalTimeLiRS<<std::endl;
            }
            
            std::cout<<"PDE method on CPU:\t"<<std::flush;
            tBegin=std::chrono::high_resolution_clock::now().time_since_epoch().count();
            noSR=0;
            vect_impv=calc3.iv(vect_FF,vect_px,vect_KK,vect_rr,
                               vect_texp,vect_pc,noSR);

            tEnd=std::chrono::high_resolution_clock::now().time_since_epoch().count();
            totalTime=(static_cast<mydouble>(tEnd-tBegin) )/(static_cast<mydouble>(1000000000));
            std::cout<<" "<<totalTime;
            std::cout<<"\t"<<100.0*(totalTime/totalTimeOld)<<"% of LBR"<<std::endl;

            
            std::cout<<"PDE on graphic card:\t"<<std::flush;
            tBegin=std::chrono::high_resolution_clock::now().time_since_epoch().count();
            
            
            seq_impv=calc3GC.ivGC(seq_FF,seq_px,seq_KK,seq_rr, seq_texp,seq_pc,sampleSize);
            tEnd=std::chrono::high_resolution_clock::now().time_since_epoch().count();
            totalTime=(static_cast<mydouble>(tEnd-tBegin) )/(static_cast<mydouble>(1000000000));
            std::cout<<" "<<totalTime;
            std::cout<<"\t"<<100.0*(totalTime/totalTimeOld)<<"% of LBR"<<std::endl;
            
 
            for(myint i=0;i<sampleSize;++i){
                currentIntVol=vect_impv[i] * sqrt(vect_texp[i]);
                GF_substitution(vect_FF[i],
                                vect_px[i],
                                vect_KK[i],
                                vect_rr[i],
                                vect_texp[i],
                                c_GF,
                                k_GF);
                currentKOverIntVol=k_GF/currentIntVol;
                if(i==0){
                    smallestIntVol=currentIntVol;
                    largestIntVol=currentIntVol;
                    smallestKOverIntVol=currentKOverIntVol;
                    largestKOverIntVol=currentKOverIntVol;
                    smallestC=c_GF;
                }
                else{
                    if(c_GF<smallestC){
                        smallestC=c_GF;
                    }
                    if(currentIntVol<smallestIntVol){
                        smallestIntVol=currentIntVol;
                    }
                    if(currentIntVol>largestIntVol){
                        largestIntVol=currentIntVol;
                    }
                    if(currentKOverIntVol<smallestKOverIntVol){
                        smallestKOverIntVol=currentKOverIntVol;
                    }
                    if(currentKOverIntVol>largestKOverIntVol){
                        largestKOverIntVol=currentKOverIntVol;
                    }
                }
                currentError=vect_impvOther[i]-seq_impv[i];
                reprice=price_call_BS(vect_FF[i],vect_KK[i],
                                      vect_rr[i],vect_impv[i],vect_texp[i]);
                repricingErrorC=vect_px[i]-reprice;
                if(repricingErrorC<0.0){
                    repricingErrorC*=-1.0;
                }
                if(currentError<0.0){
                    currentError*=-1.0;
                }
                if(currentError>maxError){
                    maxError=currentError;
                    kThatMaximizesTheError=k_GF;
                }
                if(repricingErrorMax<repricingErrorC){
                    repricingErrorMax=repricingErrorC;
                    kThatMaximizesRepricingError=k_GF;
                }
                
                
                
                if(currentError>0.000001){
                    if(currentError>0.000001){
                        ++numBad;
                        std::cout<<"***** ";
                    }
                    
                    
                    std::cout<<"F="<<vect_FF[i]<<" ";
                    std::cout<<"K="<<vect_KK[i]<<" ";
                    std::cout<<"T="<<vect_texp[i]<<" ";
                    GF_substitution(vect_FF[i],
                                    vect_px[i],
                                    vect_KK[i],
                                    vect_rr[i],
                                    vect_texp[i],
                                    c_GF,
                                    k_GF);
                    std::cout<<" c="<<c_GF<<" k="<<k_GF<<" ";
                    discount=std::exp(-vect_rr[i]*vect_texp[i]);
                    
                    std::cout<<seq_impv[i]<<" vs "<<vect_impvOther[i]<<" Error="<<currentError;
                    
                    
                    
                    std::cout<<std::endl;
                }
                
            }
            std::cout<<"Max error = "<<maxError<<" at k="<<kThatMaximizesTheError<<std::endl;
            std::cout<<"PDE Repricing error = "<<repricingErrorMax<<" at k="<<kThatMaximizesRepricingError<<std::endl;
            std::cout<<"Sigma sqrt (T) belongs to ["<<smallestIntVol<<","<<largestIntVol;
            std::cout<<"]"<<std::endl;
            std::cout<<"The smallest c is "<<smallestC<<std::endl;
            std::cout<<"k/(sigma sqrt(t)) ["<<smallestKOverIntVol;
            std::cout<<","<<largestKOverIntVol;
            std::cout<<"]"<<std::endl;
            
            
            
            
            if(seq_impv!=nullptr){delete[] seq_impv;}
            seq_impv=nullptr;
            
            delete[] seq_FF;
            delete[] seq_px;
            delete[] seq_KK;
            delete[] seq_rr;
            delete[] seq_texp;
            delete[] seq_pc;
        }
    }

    return 0;
}
