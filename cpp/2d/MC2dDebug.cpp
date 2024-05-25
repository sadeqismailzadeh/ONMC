//
// Created by Sadeq Ismailzadeh on ۱۱/۰۲/۲۰۲۳.
//
#include "MC2d.h"
#include "WalkerAlias.h"
#include <highfive/H5Easy.hpp>
#ifdef _OPENMP // Use omp.h if -fopenmp is used in g++
#include <omp.h>
#endif


double MC2d::consensusProb(int i){
    double conProb = 1;
//    for (int j = 0; j < N; ++j) {
//        if (j != i){
//            conProb *= exp(-max(dEij(i,j), 0.)/Temperature);
//        }
//    }

    assert(almostEquals(dENN(i), dErot(i)));
    conProb *= exp(-max(dENN(i), 0.)/Temperature);
    const int xi = Lat.XPbyIndex(i);
    const int yi = Lat.YPbyIndex(i);
    for (int jrej = 1; jrej < Lat.XPByDistanceFar.size(); ++jrej){
        const int xjper = Lat.XPByDistanceFar(jrej);
        const int yjper = Lat.YPByDistanceFar(jrej);
        const int xj0 = xjper + xi;
        const int yj0 = yjper + yi;
        const int xj = (xj0 >= Lat.L1)? xj0 - Lat.L1 : xj0;
        const int yj = (yj0 >= Lat.L1)? yj0 - Lat.L1 : yj0;
        int j = yj * Lat.L1 + xj;
//        cout << "i = " << i <<endl;
//        cout << "j = " << j <<endl;
        assert(i != j);
        conProb *= exp(-max(dEij(i,j), 0.)/Temperature);
    }


//    for (int j = 0; j < N; ++j) {
//        if (j != i){
//            conProb *= exp(-max(dEij(i,j), 0.)/Temperature);
//        }
//    }
    return conProb;
}

void MC2d::run1ClockParticleExact(int i) {
//    cout << "run1ClockParticle" <<endl;
    Debug_totali++;
//    int i = intDis(gen); //sorted by Index
//    cout << "concensusProb(i) = " << concensusProb(i) << endl;
    if (realDis(gen) < consensusProb(i)) {
        updateField(i);
        updateState(i);
        Debug_Iaccepted++;
    }
}

double MC2d::getEnergyDirect() {
    VectorXd HnewX;
    VectorXd HnewY;
    HnewX.setZero(Hx.size());
    HnewY.setZero(Hy.size());
    if (Prop.isMacIsaacMethod || Prop.isClockMethod || Prop.isSCOMethod || Prop.isTomitaMethod){
        for (int i = 0; i < N; ++i) {
            int rowspin = i / Lat.L1;
            int colspin = i % Lat.L1;
            const int L1 = Lat.L1;
            const int L2 = Lat.L2;
            for (int j = 0; j < L2; ++j) {
                int Fn = 2 * L1 * (L2 + j - rowspin) - colspin + L1;
                int j1 = L1 * j;
                for (int k = 0; k < L1; ++k) {
                    int pnum2 = j1 + k;
                    if (i != pnum2){
                        HnewX(i) += Lat.Wxx(Fn + k) * Ux(pnum2) + Lat.Wxy(Fn + k) * Uy(pnum2);
                        HnewY(i) += Lat.Wyy(Fn + k) * Uy(pnum2) + Lat.Wxy(Fn + k) * Ux(pnum2);
                    }
                }
            }
        }
    } else {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                if (i != j){
                    HnewX(i) += Jxx(j,i) * Ux(j) + Jxy(j,i) * Uy (j);
                    HnewY(i) += Jxy(j,i) * Ux(j) + Jyy(j,i) * Uy (j);
//                    HnewX(j) += Jxx(i,j) * Ux(i) + Jxy(i,j) * Uy (i);
//                    HnewY(j) += Jxy(i,j) * Ux(i) + Jyy(i,j) * Uy (i);
                }
            }
        }
    }
    double E = 0;
    E += Ux.dot(HnewX);
    E += Uy.dot(HnewY);

    if(Prop.isFieldOn){
        E -= Ux.dot(hxAll);   //TODO reuse MVect for this purpose
        E -= Uy.dot(hyAll);
    }
    for (int i = 0; i < N; i++) { // TODO self energy
//        E += (Uxy.segment<2>(2 * i).transpose() * J.block<2,2>(2 * i, 2 * i) * Uxy.segment<2>(2 * i)).value();

        E += (Ux(i) * (Jself(0,0) * Ux(i) + Jself(0,1) * Uy(i))) +
             (Uy(i) * (Jself(0,1) * Ux(i) + Jself(1,1) * Uy(i)));
    }
    return E;
}




void MC2d::validiatePjSCOPreset(int jrej, double invOneMinusPHatOld, SCOData& scoData) {
    complexitySum += 1;
    assert(jrej >=0 && jrej < N-1);
    assert(Lat.XPByDistance(jrej) + Lat.YPByDistance(jrej) * Lat.L1 == Lat.PByDistance(jrej));
    const int xjper = Lat.XPByDistance(jrej);
    const int yjper = Lat.YPByDistance(jrej);
    const int xj0 = xjper + scoData.xi;
    const int yj0 = yjper + scoData.yi;
    const int xj = (xj0 >= Lat.L1)? xj0 - Lat.L1 : xj0;
    const int yj = (yj0 >= Lat.L1)? yj0 - Lat.L1 : yj0;
    int j2 = yj * Lat.L1 + xj;
    int jper =  yjper * Lat.L1 + xjper;
    assert(jper >=0 && jper < N);
    int i = scoData.i;

    double Hxij = Lat.Jxx1p(jper) * Ux(j2) + Lat.Jxy1p(jper) * Uy(j2);
    double Hyij = Lat.Jxy1p(jper) * Ux(j2) + Lat.Jyy1p(jper) * Uy(j2);
    double Eij = 2*(Ux(i) * Hxij + Uy(i) * Hyij);
    assert(almostEquals(InvTemperature, 1/Temperature));
    double dVStar = dVstarVec(jrej);
    double OneMinusPj = - expm1(Eij * InvTemperature - dVStar);   // - expm1 = 1 - e^x
    OneMinusPj = (OneMinusPj < 0.)? 0. : OneMinusPj;
    if (1 - ProbClock(jrej) <= OneMinusPj){
        cout << "1 - ProbClock(jrej) = " << 1 - ProbClock(jrej)<<  endl;
        cout << "OneMinusPj = " << OneMinusPj << endl;
        cout << "jrej = " << jrej << endl;
    }
    assert(1 - ProbClock(jrej) >= OneMinusPj);
    assert(OneMinusPj >= 0. && OneMinusPj <= 1.);
    assert(almostEquals(exp(Eij * InvTemperature - dVStar),  PjSCO(jrej, i, scoData.xi, scoData.yi)));
    double prel = OneMinusPj * InVOneMinusProbClock(jrej);
    assert(prel>=0 && prel <=1);
    RNDGenerationSum += 1;
    if (realDis(gen) <= prel) {
        double EijNew = 2 * (UiNewX * Hxij + UiNewY * Hyij);
        double OneMinusPjnew = - expm1(EijNew * InvTemperature - dVStar);

        scoData.OneMinusPjOldProd *= OneMinusPj;
        scoData.OneMinusPjNewProd *= OneMinusPjnew;
        scoData.Hxi += Hxij;
        scoData.Hyi += Hyij;
        if (Prop.isSCOMethodOverrelaxationBuiltIn) {
            scoData.Hxij.push_back(Hxij);
            scoData.Hyij.push_back(Hyij);
            scoData.jrejVec.push_back(jrej);
        }
    }

}

double MC2d::dESCOPreset(SCOData& scoData){
    const int i = scoData.i;
    if (Prop.isSCOMethodOverrelaxationBuiltIn) {
        if (isOverrelaxing && Prop.isSCOMethodNearGroupBuiltIn){
            overRelaxateUiNew(i, scoData.Hxi + Hx(i), scoData.Hyi + Hy(i));
        } else if (isOverrelaxing && !Prop.isSCOMethodNearGroupBuiltIn){
            overRelaxateUiNew(i, scoData.Hxi, scoData.Hyi);
        }
        scoData.OneMinusPjNewProd = 1;
        for (int j = 0; j < scoData.jrejVec.size(); ++j) {
            double dVStar = dVstarVec(scoData.jrejVec[j]);
            double Enew = 2 * (UiNewX * scoData.Hxij[j] + UiNewY * scoData.Hyij[j]);
            double Pjnew = exp(Enew * InvTemperature - dVStar);
            scoData.OneMinusPjNewProd *= 1 - Pjnew;
        }
    }
    const double dMx = UiNewX - Ux(i);
    const double dMy = UiNewY - Uy(i);
    double dE =  2 * (dMx * scoData.Hxi + dMy * scoData.Hyi);
    dE += Temperature * log(scoData.OneMinusPjOldProd/scoData.OneMinusPjNewProd);
    if(Prop.isSCOMethodNearGroupBuiltIn){
        assert(almostEquals(dErot(i),dENN(i)));
        dE += dErot(i);
    }
    return dE;
}

void MC2d::run1SCOPresetParicle(int i) {
    Debug_totali++;
    TotalRegularParticleSteps        += (!isOverrelaxing)? 1 : 0;
    TotalOverrelaxedParticleSteps    += ( isOverrelaxing)? 1 : 0;
    if (Prop.isNearGroup){
        if (!Prop.isSCOMethodNearGroupBuiltIn){
            assert(almostEquals(dErot(i),dENN(i)));
            double dE = dErot(i);
            RNDGenerationSum += 1;
            if (realDis(gen) > exp(-dE * InvTemperature)){
                return;   // reject move
            }
        }
    }
    const int xi = Lat.XPbyIndex(i);
    const int yi = Lat.YPbyIndex(i);
    SCOData scoData(i, xi, yi);
    assert(xi == i % Lat.L1);
    assert(yi == i / Lat.L1);
    vector<int> jrejVec;
    vector<double> PjVec;

    for (int jrej: jrejMat[i]) { //TODO make jreMat to vector<vector<int>>
        validiatePjSCOPreset(jrej, 0, scoData);
    }



    double dE = dESCOPreset(scoData);
    RNDGenerationSum += 1;
    if (dE < 0 || (realDis(gen) < exp(-dE * InvTemperature))){
        AcceptedRegularParticleSteps     += (!isOverrelaxing)? 1 : 0;
        AcceptedOverrelaxedParticleSteps += ( isOverrelaxing)? 1 : 0;
        Debug_Iaccepted++;    // move accepted
        if (Prop.isNearGroup){
            updateField(i);
        }
        updateState(i);
    }
}

void MC2d::run1SCOStepPreset() {
    static const double PI = acos(-1);
    #ifdef NDEBUG
    static const long ReCalcPeriod = 1'000'000;
    #else
    static const long ReCalcPeriod = 10'000;
    #endif
    int CounterAccceptence = 0;
    if (Prop.isTakeSnapshot && (SnapshotTakingIndex++ % TakeSnapshotStep == 0)){
        takeSnapshot();
    }
    if (++iReCalc % ReCalcPeriod == 0){
        calcAllFields();
    }

    SCOReshuffleCounter += 1;
    if (SCOReshuffleCounter >= Prop.SCOPresetReshuffleInterval){
        run1SCOShuffle();
        SCOReshuffleCounter = 0;
    }

    if (Prop.isOverRelaxationMethod) {
        if (Prop.isSCOMethodOverrelaxationBuiltIn && !Prop.isNearGroup){
            isOverrelaxing = true;
            for (int j = 0; j < Prop.OverrelaxationSteps; ++j) {
                for (int i = 0; i < N; ++i) {
                    randomizeUiNew();
                    run1SCOPresetParicle(i);
                }
            }
            isOverrelaxing = false;
            for (int j = 0; j < Prop.MetropolisSteps; ++j) {
                for (int i = 0; i < N; ++i) {
                    randomizeUiNew();
                    run1SCOPresetParicle(i);
                }
            }
        } else if (Prop.isNearGroup) {
            isOverrelaxing = true;
            for (int j = 0; j < Prop.OverrelaxationSteps; ++j) {
                for (int i = 0; i < N; ++i) {
                    overRelaxateUiNew(i);
                    run1SCOPresetParicle(i);
                }
            }
            isOverrelaxing = false;
            for (int j = 0; j < Prop.MetropolisSteps; ++j) {
                for (int i = 0; i < N; ++i) {
                    randomizeUiNew();
                    run1SCOPresetParicle(i);
                }
            }
        }
    } else {
        isOverrelaxing = false;
        for (int j = 0; j < Prop.MetropolisSteps; ++j) {
            for (int i = 0; i < N; ++i) {
                randomizeUiNew();
                run1SCOPresetParicle(i);
            }
        }
    }
}



void MC2d::validiatePjSCOShuffle(int jrej, double invOneMinusPHatOld, SCOData &scoData) {
    complexitySum += 1;
//    assert(jrej >=0 && jrej < N-1);
//    assert(Lat.XPByDistance(jrej) + Lat.YPByDistance(jrej) * Lat.L1 == Lat.PByDistance(jrej));
//    const int xjper = Lat.XPByDistance(jrej);
//    const int yjper = Lat.YPByDistance(jrej);
//    const int xj0 = xjper + scoData.xi;
//    const int yj0 = yjper + scoData.yi;
//    const int xj = (xj0 >= Lat.L1)? xj0 - Lat.L1 : xj0;
//    const int yj = (yj0 >= Lat.L1)? yj0 - Lat.L1 : yj0;
//    int j2 = yj * Lat.L1 + xj;
//    int jper =  yjper * Lat.L1 + xjper;
//    assert(jper >=0 && jper < N);
    int i = scoData.i;
//
//    double Hxij = Lat.Jxx1p(jper) * Ux(j2) + Lat.Jxy1p(jper) * Uy(j2);
//    double Hyij = Lat.Jxy1p(jper) * Ux(j2) + Lat.Jyy1p(jper) * Uy(j2);
//    double Eij = 2*(Ux(i) * Hxij + Uy(i) * Hyij);
//    assert(almostEquals(InvTemperature, 1/Temperature));
//    double dVStar = dVstarVec(jrej);
////    double Pj = exp(Eij * InvTemperature - dVStar);
//    assert(invOneMinusPHatOld >= (1 - Pj));
//    assert(Pj >=0 && Pj <=1);
//    assert(almostEquals(Pj,  PjSCO(jrej, i, scoData.xi, scoData.yi)));
//    double Pj = ProbClock(jrej);
//    double prel = (1 - Pj) * invOneMinusPHatOld;
//    if (!(prel >= 0 && prel <= 1)){
//        cout <<  " 1 / (1 - Pj) = " << 1 / (1 - Pj) << endl;
//        cout <<  "invOneMinusPHatOld = " << invOneMinusPHatOld << endl;
//        cout <<  "jrej = " << jrej << endl;
//        cout <<  "prel = " << prel << endl;
//    }
//    assert(prel >= 0 && prel <= 1);
//    RNDGenerationSum += 1;
//    if (realDis(gen) <= prel) {
        jrejMat[i].push_back(jrej);
//    }
}



void MC2d::run1SCOShuffle(){
    for (int i = 0; i < N; ++i) {
        jrejMat[i].clear();
        run1SCOShuffleParticle(i);
    }
}


void MC2d::run1SCOShuffleParticle(int i) {
//    cout << "it is walker!"<<endl;
    int jrej = 0;     //jrej is sorted by distance and is periodic
    int shift = 0;
    if (Prop.isNearGroup){
        shift = Lat.Neighbours.size();
    }
    Debug_totali++;
    TotalRegularParticleSteps     += (!isOverrelaxing)? 1 : 0;
    TotalOverrelaxedParticleSteps += ( isOverrelaxing)? 1 : 0;


    const int xi = Lat.XPbyIndex(i);
    const int yi = Lat.YPbyIndex(i);
    assert(xi == i % Lat.L1);
    assert(yi == i / Lat.L1);

    SCOData scoData(i, xi, yi);

    vector<int> jrejVec;
    vector<double> PjVec;


    vector<int> jrejVecSmall;
    double RandomPoisson = PoissonDis(gen);
    RNDGenerationSum += 1;
    for (int j = 0; j < RandomPoisson; ++j) {
        jrej = WalkerSampler(gen) + shift;
        RNDGenerationSum += 1;
        if (jrejVecBool[jrej]){
            continue;
        }
        jrejVecBool[jrej] = true;
        jrejVecSmall.push_back(jrej);

        double invOneMinusPhat = InVOneMinusProbClock(jrej);
        validiatePjSCOShuffle(jrej, invOneMinusPhat, scoData);
        NumResampled++;
    }
    cleanjrejVecBool(jrejVecSmall);
}


//void MC2d::run1SCOShuffleParticle(int i) {
//    int jrej = 0;     //jrej is sorted by distance and is periodic
//    if (Prop.isNearGroup){
//        jrej = Lat.Neighbours.size();
//    }
//    const int xi = Lat.XPbyIndex(i);
//    const int yi = Lat.YPbyIndex(i);
//    SCOData scoData(i, xi, yi);
//    assert(xi == i % Lat.L1);
//    assert(yi == i / Lat.L1);
//    vector<int> jrejVec;
//    vector<double> PjVec;
//    RNDGenerationSum += 1;
//    double nu = realDis(gen);
//    double invOneMinusPHatOld = InVOneMinusProbClock(jrej);
//    assert(int(floor(log(nu)/log(ProbClock(jrej)))) == int(log(nu)/log(ProbClock(jrej))));
//    assert(almostEquals(InVLogProbClock(jrej), 1 / log(ProbClock(jrej))));
//    jrej = jrej + (log(nu) * InVLogProbClock(jrej));
//
//    while(true){
//        if (jrej > Lat.PByDistance.size() - 1){ // there is totally N-1 bouds indexed from 0. so the index of the last bond is N-2
//            break;
//        } else  {
//            validiatePjSCOShuffle(jrej, invOneMinusPHatOld, scoData);
//        }
//
//        NumResampled++;
//        assert(jrej >=0 && jrej < N-1);
//        RNDGenerationSum += 1;
//        nu = realDis(gen);
//        invOneMinusPHatOld = InVOneMinusProbClock(jrej);
//        assert(almostEquals(InVLogProbClock(jrej), 1 / log(ProbClock(jrej))));
//        assert(int(floor(log(nu)/log(ProbClock(jrej)))) == int(log(nu)/log(ProbClock(jrej))));
//        jrej = jrej + (1 + log(nu) * InVLogProbClock(jrej));
//    }
//}


void MC2d::setProbTomita() {
    double ratio = 1;
    double lambdaTot = 0;
    alphaTilde = Prop.TomitaAlphaTilde;

    kappa = alphaTilde / (1 + alphaTilde);
    invKappa = 1 / kappa;

    lambdaVec.clear();
    gammaVec.clear();
    alphaVec.clear();
    invAlphaVec.clear();
    invLambdaVec.clear();
    for (int j = 0; j < ProbClock.size(); ++j) {
        double r0j = Lat.R0jBydistance(j);
        assert(r0j > 0);        // TODO set max prob using J interactions eigenvalues
        double gamma = 4*ratio*Lat.JmaxByDistance(j) /Temperature ;
        double alpha = gamma * alphaTilde;
        gammaVec.push_back(gamma);
        alphaVec.push_back(alpha);
        invAlphaVec.push_back(1/alpha);
        lambdaVec.push_back(gamma + alpha);
        invLambdaVec.push_back(1/(gamma + alpha));
    }



    if (Prop.isNearGroup) {
        lambdaVec.erase(lambdaVec.begin(), lambdaVec.begin() + Lat.Neighbours.size());
        gammaVec.erase(gammaVec.begin(), gammaVec.begin() + Lat.Neighbours.size());
        alphaVec.erase(alphaVec.begin(), alphaVec.begin() + Lat.Neighbours.size());
        invAlphaVec.erase(invAlphaVec.begin(), invAlphaVec.begin() + Lat.Neighbours.size());
        invLambdaVec.erase(invLambdaVec.begin(), invLambdaVec.begin() + Lat.Neighbours.size());
    }
    WalkerSampler.set(lambdaVec);
    lambdaTot = std::accumulate(lambdaVec.begin(), lambdaVec.end(),
                                decltype(lambdaVec)::value_type(0));
    PoissonDis = std::poisson_distribution<int>(lambdaTot); //TODO make this work for near neighbours as well
    cout << "lambdaTot = " << lambdaTot  <<endl;
}

void MC2d::run1TomitaParticle(int i) {
    int shift = 0;
    double pflip = 1;
    Debug_totali++;
    TotalUpdateSum++;
    TotalRegularParticleSteps     += (!isOverrelaxing)? 1 : 0;
    TotalOverrelaxedParticleSteps += ( isOverrelaxing)? 1 : 0;
    if (Prop.isNearGroup){
        assert(almostEquals(dErot(i),dENN(i)));
        double dE = dErot(i);
        double pNear = exp(-dE * InvTemperature);
        pflip *= (Prop.isTomitaMethodNearGroupBuiltIn)? pNear : 1;
        RNDGenerationSum += 1;
        if (!isOverrelaxing && !Prop.isTomitaMethodNearGroupBuiltIn && dE > 0 && realDis(gen) > pNear){
            return;   // reject move
        }
        shift = Lat.Neighbours.size();
    }
    const int xi = Lat.XPbyIndex(i);
    const int yi = Lat.YPbyIndex(i);
    assert(xi == i % Lat.L1);
    assert(yi == i / Lat.L1);

    vector<int> jrejVecSmall;
    RNDGenerationSum += 1;
    double RandomPoisson = PoissonDis(gen);
    for (int j = 0; j < RandomPoisson; ++j) {
        RNDGenerationSum += 1;
        int jrejMinusShift = WalkerSampler(gen);
        int jrej = jrejMinusShift + shift;
        if (jrejVecRepeated[jrej] == 0){
            jrejVecSmall.push_back(jrej);
        }
        jrejVecRepeated[jrej] += 1;
//    }                                                //uncomment for no group
//    for (int jrej : jrejVecSmall) {                  //uncomment for no group
//        int jrejMinusShift = jrej - shift;           //uncomment for no group
//        const int power = jrejVecRepeated[jrej];     //uncomment for no group
//        constexpr int power = 1;                       //comment for no group
        const double Jtilde = Jtildeijrej(jrej, i, xi, yi);
        const double alpha = alphaVec[jrejMinusShift];
        const double invAlpha = invAlphaVec[jrejMinusShift];
        const double invLambda = invLambdaVec[jrejMinusShift];
        const double Beta = InvTemperature;
        if (Jtilde >= 0.){
            double prel = (2*Beta*abs(Jtilde) + alpha) * invLambda;
            assert(prel>=0);
//            for (int k = 0; k < power; ++k) {           //uncomment for no group
                RNDGenerationSum += 1;
                if (realDis(gen) <= prel){
                    pflip *= alpha / (2*Beta*abs(Jtilde) + alpha);
                }
//            }                                           //uncomment for no group
        } else {
            double prel = alpha * invLambda;
            assert(prel>=0);
//            for (int k = 0; k < power; ++k) {           //uncomment for no group
                RNDGenerationSum += 1;
                if (realDis(gen) <= prel) {
                    pflip *= (2*Beta*abs(Jtilde) + alpha) * invAlpha;
                }
//            }                                           //uncomment for no group
        }

    }
    RNDGenerationSum += 1;
    if (realDis(gen) <= pflip){
        Debug_Iaccepted++;    // move accepted
        AcceptedRegularParticleSteps     += (!isOverrelaxing)? 1 : 0;
        AcceptedOverrelaxedParticleSteps += ( isOverrelaxing)? 1 : 0;
        if (Prop.isNearGroup){
            updateField(i);
        }
        updateState(i);
    }
    cleanjrejVecRepeated(jrejVecSmall);
}

void MC2d::run1TomitaStep() {
    static const double PI = acos(-1);
    #ifdef NDEBUG
    static const long ReCalcPeriod = 1'000'000;
    #else
    static const long ReCalcPeriod = 10'000;
    #endif
    int CounterAccceptence = 0;
    if (Prop.isTakeSnapshot && (SnapshotTakingIndex++ % TakeSnapshotStep == 0)){
        takeSnapshot();
    }
    if ((++iReCalc % ReCalcPeriod == 0)){
        calcAllFields();
    }

    if (Prop.isOverRelaxationMethod && Prop.isNearGroup) {
        isOverrelaxing = true;
        for (int j = 0; j < Prop.OverrelaxationSteps; ++j) {
            for (int i = 0; i < N; ++i) {
                overRelaxateUiNew(i);
                run1TomitaParticle(i);
            }
        }
        isOverrelaxing = false;
        for (int j = 0; j < Prop.MetropolisSteps; ++j) {
            for (int i = 0; i < N; ++i) {
                randomizeUiNew();
                run1TomitaParticle(i);
            }
        }
    } else {
        isOverrelaxing = false;
        for (int j = 0; j < Prop.MetropolisSteps; ++j) {
            for (int i = 0; i < N; ++i) {
                randomizeUiNew();
                run1TomitaParticle(i);
            }
        }
    }

}

double MC2d::dEijrej(int jrej, int i, int xi, int yi){  // jrej is sorted by distance and is periodic
    assert(jrej >=0 && jrej < N-1);
    complexitySum += 1;
    assert(Lat.XPByDistance(jrej) + Lat.YPByDistance(jrej) * Lat.L1 == Lat.PByDistance(jrej));
    const int xjper = Lat.XPByDistance(jrej);
    const int yjper = Lat.YPByDistance(jrej);
    const int xj0 = xjper + xi;
    const int yj0 = yjper + yi;
    const int xj = (xj0 >= Lat.L1)? xj0 - Lat.L1 : xj0;
    const int yj = (yj0 >= Lat.L1)? yj0 - Lat.L1 : yj0;
    int j2 = yj * Lat.L1 + xj;
    int jper =  yjper * Lat.L1 + xjper;
    const double dMx = UiNewX - Ux(i);
    const double dMy = UiNewY - Uy(i);
    assert(jper >=0 && jper < N);
    double dE = 2*(dMx * (Lat.Jxx1p(jper) * Ux(j2) + Lat.Jxy1p(jper) * Uy(j2)) +
                   dMy * (Lat.Jxy1p(jper) * Ux(j2) + Lat.Jyy1p(jper) * Uy(j2)));
    return dE;
}

double MC2d::Jtildeijrej(int jrej, int i, int xi, int yi){  // jrej is sorted by distance and is periodic
    assert(jrej >=0 && jrej < N-1);
    complexitySum += 1;
    assert(Lat.XPByDistance(jrej) + Lat.YPByDistance(jrej) * Lat.L1 == Lat.PByDistance(jrej));
    const int xjper = Lat.XPByDistance(jrej);
    const int yjper = Lat.YPByDistance(jrej);
    const int xj0 = xjper + xi;
    const int yj0 = yjper + yi;
    const int xj = (xj0 >= Lat.L1)? xj0 - Lat.L1 : xj0;
    const int yj = (yj0 >= Lat.L1)? yj0 - Lat.L1 : yj0;
    int j2 = yj * Lat.L1 + xj;
    int jper =  yjper * Lat.L1 + xjper;
    const double sigmax = (Ux(i) - UiNewX) * 0.5; //TODO make setters and getters for UiNew to avoid unintentional rewrite
    const double sigmay = (Uy(i) - UiNewY) * 0.5;
    assert(jper >=0 && jper < N);
    double Jtilde = -2*(sigmax * (Lat.Jxx1p(jper) * Ux(j2) + Lat.Jxy1p(jper) * Uy(j2)) +
                        sigmay * (Lat.Jxy1p(jper) * Ux(j2) + Lat.Jyy1p(jper) * Uy(j2)));
    return Jtilde;
}