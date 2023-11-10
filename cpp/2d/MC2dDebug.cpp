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
    assert(jrej >=0 && jrej < N-1);
//    cout << "jrej from PRel = " << jrej << endl;
//    int jper = Lat.PByDistance(jrej);    //jper is periodic
//    int j = Lat.unperiodicParticle(jper, i);
//    assert(j >=0 && j < N);
//    assert(jper >=0 && jper < N);
//    int jperback = Lat.periodicParticle(j, i);
//    assert(jper == jperback);
//    assert(almostEquals(Lat.getMinDistance(i,j), Lat.getMinDistance(0, jper)));
//    double Phatj = ProbClock(jrej);
    assert(Lat.XPByDistance(jrej) + Lat.YPByDistance(jrej) * Lat.L1 == Lat.PByDistance(jrej));
    const int xjper = Lat.XPByDistance(jrej);
    const int yjper = Lat.YPByDistance(jrej);
    const int xj0 = xjper + scoData.xi;
    const int yj0 = yjper + scoData.yi;
    const int xj = (xj0 >= Lat.L1)? xj0 - Lat.L1 : xj0;
    const int yj = (yj0 >= Lat.L1)? yj0 - Lat.L1 : yj0;
    int j2 = yj * Lat.L1 + xj;
    int jper =  yjper * Lat.L1 + xjper;
//    assert(j2 == j);
//    int jper = Lat.periodicParticle(j, i);
    assert(jper >=0 && jper < N);
    int i = scoData.i;

    double Hxij = Lat.Jxx1p(jper) * Ux(j2) + Lat.Jxy1p(jper) * Uy(j2);
    double Hyij = Lat.Jxy1p(jper) * Ux(j2) + Lat.Jyy1p(jper) * Uy(j2);
    double Eij = 2*(Ux(i) * Hxij + Uy(i) * Hyij);
//    double Eij2 = 2*(Ux(i) * (Lat.Jxx1p(jper) * Ux(j2) + Lat.Jxy1p(jper) * Uy(j2)) +
//                   Uy(i) * (Lat.Jxy1p(jper) * Ux(j2) + Lat.Jyy1p(jper) * Uy(j2)));
//    assert(almostEquals(Eij, Eij2));
//    assert(almostEquals(dE, dEij(i,j)));

//    double Phatj = InvOneMinusPHatOld;
//    double Pj = exp(-max(dEij(i,j), 0.)/Temperature);
    assert(almostEquals(InvTemperature, 1/Temperature));
    double dVStar = dVstarVec(jrej);
    double Pj = exp(Eij * InvTemperature - dVStar);
//    assert(invOneMinusPHatOld >= (1 - Pj));
    assert(Pj >=0 && Pj <=1);
    assert(almostEquals(Pj,  PjSCO(jrej, i, scoData.xi, scoData.yi)));


    double EijNew = 2*(UiNewX * Hxij + UiNewY * Hyij);
//        double EijNew2 = 2*(UiNewX * (Lat.Jxx1p(jper) * Ux(j2) + Lat.Jxy1p(jper) * Uy(j2)) +
//                         UiNewY* (Lat.Jxy1p(jper) * Ux(j2) + Lat.Jyy1p(jper) * Uy(j2)));
//        assert(almostEquals(EijNew, EijNew2));
    double Pjnew = exp(EijNew * InvTemperature - dVStar);
    scoData.OneMinusPjOldProd *= 1 - Pj;
    scoData.OneMinusPjNewProd *= 1 - Pjnew;
    scoData.Hxi += Hxij;
    scoData.Hyi += Hyij;
    if (Prop.isSCOMethodOverrelaxationBuiltIn){
        scoData.Hxij.push_back(Hxij);
        scoData.Hyij.push_back(Hyij);
        scoData.jrejVec.push_back(jrej);
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
//    cout << "run1ClockParticle" <<endl;
    Debug_totali++;
    TotalRegularParticleSteps        += (!isOverrelaxing)? 1 : 0;
    TotalOverrelaxedParticleSteps    += ( isOverrelaxing)? 1 : 0;
    if (Prop.isNearGroup){
        if (!Prop.isSCOMethodNearGroupBuiltIn){
            //        cout << "here" << endl;
            assert(almostEquals(dErot(i),dENN(i)));
            double dE = dErot(i);
            //        cout << "here " << endl;
            //         if (dE > 0 && realDis(gen) > exp(-dE * InvTemperature)){
            if (realDis(gen) > exp(-dE * InvTemperature)){
                return;   // reject move
            }
        }
//        cout << "jrej = " << jrej << endl;
    }
//    const int i = intDis(gen); //sorted by Index
//    const int i = Random(N); //sorted by Index
//    const int xi = i % Lat.L1;
//    const int yi = i / Lat.L1;
    const int xi = Lat.XPbyIndex(i);
    const int yi = Lat.YPbyIndex(i);
    SCOData scoData(i, xi, yi);
//    scoData.i = i;
//    scoData.xi = xi;
//    scoData.yi = yi;
    assert(xi == i % Lat.L1);
    assert(yi == i / Lat.L1);
//    randomizeUiNew();
//    randomAdaptive(i);
//    cout << "concensusProb(i) = " << concensusProb(i) << endl;
    vector<int> jrejVec;
    vector<double> PjVec;

    for (int jrej: jrejMat[i]) { //TODO make jreMat to vector<vector<int>>
        validiatePjSCOPreset(jrej, 0, scoData);
    }



//    cout << "jrejVec = " <<endl;
//    for_each(jrejVec.begin(), jrejVec.end(), [](const auto& elem) { cout << elem << " "; });
//    cout << endl;
//    double dE = dESCO(jrejVec, i, xi, yi, PjVec);
    double dE = dESCOPreset(scoData);
//    assert(almostEquals(dE, dE2));
    if (dE < 0 || (realDis(gen) < exp(-dE * InvTemperature))){
        AcceptedRegularParticleSteps     += (!isOverrelaxing)? 1 : 0;
        AcceptedOverrelaxedParticleSteps += ( isOverrelaxing)? 1 : 0;
        Debug_Iaccepted++;    // move accepted
        updateField(i);
        updateState(i);
//        cout << "i = " << i << " accepted" <<endl;
//        JrejSum(jrej)++;
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

    if (Prop.isOverRelaxationMethod) {
        if (Prop.isSCOMethodOverrelaxationBuiltIn && !Prop.isNearGroup){
            run1SCOShuffle();
            for (int j = 0; j < 10; ++j) {
//                for (int c = 0; c < N; ++c) {
//                    int i = intDis(gen);
                for (int i = 0; i < N; ++i) {
                    randomizeUiNew();
                    run1SCOPresetParicle(i);
                }
            }
        } else if (Prop.isNearGroup) {
            isOverrelaxing = true;

            for (int j = 0; j < 10; ++j) {
                run1SCOShuffle();
//            for (int i : ParticleListSorted) {
                for (int i = 0; i < N; ++i) {
                    overRelaxateUiNew(i);
                    run1SCOPresetParicle(i);
                }
            }
            isOverrelaxing = false;
            for (int j = 0; j < 1; ++j) {
                run1SCOShuffle();
                for (int c = 0; c < N; ++c) {
                    int i = intDis(gen);
//            for (int i = 0; i < N; ++i) {
                    randomizeUiNew();
                    run1SCOPresetParicle(i);
                }
            }
        }
    } else {
        run1SCOShuffle();
        for (int j = 0; j < 10; ++j) {
//            for (int c = 0; c < N; ++c) {
//                int i = intDis(gen);
            for (int i = 0; i < N; ++i) {
                randomizeUiNew();
                run1SCOPresetParicle(i);
            }
        }
    }

//    shuffle(ParticleList.begin(), ParticleList.end(), gen);
//    for (int i : ParticleList) {
//        randomizeUiNew();
//        run1ClockParticle(i);
//    }

//    double RateAccepted = AdapticeCounterAcceptance / (cmax + 0.0);
//    double sigma_temp = sigma * (0.9/(1.0 - RateAccepted));
//    if(sigma_temp > 500.0) {
//        sigma_temp = 500.0;
//    }
//    sigma = sigma_temp;

}



void MC2d::validiatePjSCOShuffle(int jrej, double invOneMinusPHatOld, SCOData &scoData) {
    assert(jrej >=0 && jrej < N-1);
//    cout << "jrej from PRel = " << jrej << endl;
//    int jper = Lat.PByDistance(jrej);    //jper is periodic
//    int j = Lat.unperiodicParticle(jper, i);
//    assert(j >=0 && j < N);
//    assert(jper >=0 && jper < N);
//    int jperback = Lat.periodicParticle(j, i);
//    assert(jper == jperback);
//    assert(almostEquals(Lat.getMinDistance(i,j), Lat.getMinDistance(0, jper)));
//    double Phatj = ProbClock(jrej);
    assert(Lat.XPByDistance(jrej) + Lat.YPByDistance(jrej) * Lat.L1 == Lat.PByDistance(jrej));
    const int xjper = Lat.XPByDistance(jrej);
    const int yjper = Lat.YPByDistance(jrej);
    const int xj0 = xjper + scoData.xi;
    const int yj0 = yjper + scoData.yi;
    const int xj = (xj0 >= Lat.L1)? xj0 - Lat.L1 : xj0;
    const int yj = (yj0 >= Lat.L1)? yj0 - Lat.L1 : yj0;
    int j2 = yj * Lat.L1 + xj;
    int jper =  yjper * Lat.L1 + xjper;
//    assert(j2 == j);
//    int jper = Lat.periodicParticle(j, i);
    assert(jper >=0 && jper < N);
    int i = scoData.i;

    double Hxij = Lat.Jxx1p(jper) * Ux(j2) + Lat.Jxy1p(jper) * Uy(j2);
    double Hyij = Lat.Jxy1p(jper) * Ux(j2) + Lat.Jyy1p(jper) * Uy(j2);
    double Eij = 2*(Ux(i) * Hxij + Uy(i) * Hyij);
//    double Eij2 = 2*(Ux(i) * (Lat.Jxx1p(jper) * Ux(j2) + Lat.Jxy1p(jper) * Uy(j2)) +
//                   Uy(i) * (Lat.Jxy1p(jper) * Ux(j2) + Lat.Jyy1p(jper) * Uy(j2)));
//    assert(almostEquals(Eij, Eij2));
//    assert(almostEquals(dE, dEij(i,j)));

//    double Phatj = InvOneMinusPHatOld;
//    double Pj = exp(-max(dEij(i,j), 0.)/Temperature);
    assert(almostEquals(InvTemperature, 1/Temperature));
    double dVStar = dVstarVec(jrej);
    double Pj = exp(Eij * InvTemperature - dVStar);
    assert(invOneMinusPHatOld >= (1 - Pj));
    assert(Pj >=0 && Pj <=1);
    assert(almostEquals(Pj,  PjSCO(jrej, i, scoData.xi, scoData.yi)));
    double prel = (1 - Pj) * invOneMinusPHatOld;
    if (realDis(gen) <= prel) {
        jrejMat[i].push_back(jrej);
    }
}



void MC2d::run1SCOShuffle(){
    for (int i = 0; i < N; ++i) {
        jrejMat[i].clear();
        run1SCOShuffleParticle(i);
    }
//    print(jrejMat);
}

void MC2d::run1SCOShuffleParticle(int i) {
//    cout << "run1ClockParticle" <<endl;
    int jrej = 0;     //jrej is sorted by distance and is periodic
//    Debug_totali++;
    if (Prop.isNearGroup){
        jrej = Lat.Neighbours.size();
//        cout << "jrej = " << jrej << endl;
    }
//    const int i = intDis(gen); //sorted by Index
//    const int i = Random(N); //sorted by Index
//    const int xi = i % Lat.L1;
//    const int yi = i / Lat.L1;
    const int xi = Lat.XPbyIndex(i);
    const int yi = Lat.YPbyIndex(i);
    SCOData scoData(i, xi, yi);
//    scoData.i = i;
//    scoData.xi = xi;
//    scoData.yi = yi;
    assert(xi == i % Lat.L1);
    assert(yi == i / Lat.L1);
//    randomizeUiNew();
//    randomAdaptive(i);
//    cout << "concensusProb(i) = " << concensusProb(i) << endl;
    vector<int> jrejVec;
    vector<double> PjVec;
    double nu = realDis(gen);
//    double nu = Random();
    double invOneMinusPHatOld = InVOneMinusProbClock(jrej);
    assert(int(floor(log(nu)/log(ProbClock(jrej)))) == int(log(nu)/log(ProbClock(jrej))));
//    jrej = jrej + floor(log(nu)/log(ProbClock(jrej)));
    assert(almostEquals(InVLogProbClock(jrej), 1 / log(ProbClock(jrej))));
    jrej = jrej + (log(nu) * InVLogProbClock(jrej));

    while(true){
//        jrej = jrej + (1 + log(nu)/log(ProbClock(jrej)));
//        cout << "jrej = " <<  jrej <<endl;
//        assert((jrej >= 0 && jrej < N-2));
//        assert(ProbOld <= ProbClock(jrej));
        if (jrej > Lat.PByDistance.size() - 1){ // there is totally N-1 bouds indexed from 0. so the index of the last bond is N-2
//            Debug_Iaccepted++;    // move accepted
//            updateState(i);
//            AdapticeCounterAcceptance++;
//            cout << "i = " << i << " accepted" <<endl;
            break;
        } else  {
//            double Pj = PjSCO(jrej, i, xi, yi);
//            assert(invOneMinusPHatOld >= (1 - Pj));
//            assert(Pj >=0 && Pj <=1);
//            double prel = (1 - Pj) * invOneMinusPHatOld;
            validiatePjSCOShuffle(jrej, invOneMinusPHatOld, scoData);
//            if (realDis(gen) <= prel){
//                jrejVec.push_back(jrej);
//                PjVec.push_back(Pj);
//            }
        }

        NumResampled++;
        assert(jrej >=0 && jrej < N-1);
        nu = realDis(gen);
//        nu = Random();
//        ProbOld = ProbClock(jrej);
        invOneMinusPHatOld = InVOneMinusProbClock(jrej);
//        assert(almostEquals(InVOneMinusProbClock(jrej), 1/(1 - ProbOld)));
        assert(almostEquals(InVLogProbClock(jrej), 1 / log(ProbClock(jrej))));
        assert(int(floor(log(nu)/log(ProbClock(jrej)))) == int(log(nu)/log(ProbClock(jrej))));
//        jrej = jrej + floor(1 + log(nu) * InVLogProbClock(jrej));
        jrej = jrej + (1 + log(nu) * InVLogProbClock(jrej));
    }
}


void MC2d::setProbTomita() {
//    cout << "setProbClock" <<endl;
//    ProbClock.setZero();  //TODO rename ProbClock to ProbMax
//    InVLogProbClock.setZero();
//    InVOneMinusProbClock.setZero();
//    dVstarVec.setZero();
    double ratio = 1;
    double lambdaTot = 0;
    alphaTilde = Prop.TomitaAlphaTilde;

    kappa = alphaTilde / (1 + alphaTilde);
    invKappa = 1 / kappa;

    if (Prop.isHavingExchangeInteraction){
        ratio = Prop.DipolarStrengthToExchangeRatio;
    }
    lambdaVec.clear();
    gammaVec.clear();
    alphaVec.clear();
    invAlphaVec.clear();
    invLambdaVec.clear();
    for (int j = 0; j < ProbClock.size(); ++j) {
        double r0j = Lat.R0jBydistance(j);
        assert(r0j > 0);        // TODO set max prob using J interactions eigenvalues
//        double gamma = 8.6 * ratio / (Temperature * pow(r0j, 3));
        double gamma = 4*ratio*Lat.JmaxByDistance(j) /Temperature ;
//        double gammaNew = 2*Lat.JmaxByDistance(j) /Temperature ;
//        assert(gammaNew < gamma);
//        assert(gammaNew > gamma/2);

        double alpha = gamma * alphaTilde;
//        ProbClock(j) = exp(- lambda);  //exp(-2*beta*2/r^3)
//        InVLogProbClock(j) = 1 / log(ProbClock(j));  //exp(-2*beta*2/r^3)
//        InVOneMinusProbClock(j) = 1 / (1 - ProbClock(j));  //exp(-2*beta*2/r^3)
//        lambdaTot += lambda;
        gammaVec.push_back(gamma);
        alphaVec.push_back(alpha);
        invAlphaVec.push_back(1/alpha);
        lambdaVec.push_back(gamma + alpha);
        invLambdaVec.push_back(1/(gamma + alpha));
//        gammaVec[j] = (gamma);
//        alphaVec[j] = (alpha);
//        invAlphaVec[j] = (1/alpha);
//        lambdaVec[j] = (gamma + alpha);
//        invLambdaVec[j] = (1/(gamma + alpha));
    }
//    cout << "ProbClockNN = " << ProbClock(0) << endl <<endl;
//    cout << "ProbClock = " << ProbClock(seq(1,min(5L, ProbClock.size()-1)))  << endl <<endl;
//    cout << "ProbClock.prod() = " << ProbClock(seq(1, last)).prod() << endl;
//    cout << "p(0) = " << ProbClock(0) << endl;
//    cout << "lambda(0) = " << -log(ProbClock(0)) << endl;
//    cout << "pow(p(1), Nfar) = " << pow(ProbClock(1), ProbClock.size() - 1) << endl;



    if (Prop.isNearGroup) {
        lambdaVec.erase(lambdaVec.begin(), lambdaVec.begin() + Lat.Neighbours.size());
        gammaVec.erase(gammaVec.begin(), gammaVec.begin() + Lat.Neighbours.size());
        alphaVec.erase(alphaVec.begin(), alphaVec.begin() + Lat.Neighbours.size());
        invAlphaVec.erase(invAlphaVec.begin(), invAlphaVec.begin() + Lat.Neighbours.size());
        invLambdaVec.erase(invLambdaVec.begin(), invLambdaVec.begin() + Lat.Neighbours.size());
    }
    WalkerSampler.set(lambdaVec);
//    lambdaTot = std::reduce(lambdaVec.begin(), lambdaVec.end());
    lambdaTot = std::accumulate(lambdaVec.begin(), lambdaVec.end(),
                                decltype(lambdaVec)::value_type(0));
    PoissonDis = std::poisson_distribution<int>(lambdaTot); //TODO make this work for near neighbours as well
    cout << "lambdaTot = " << lambdaTot  <<endl;
//    cout << "Random poisson = " << PoissonDis(gen)  <<endl;
}

void MC2d::run1TomitaParticle(int i) {
//    cout << "run1ClockParticle" <<endl;
//    int jrej = 0;     //jrej is sorted by distance and is periodic
    int shift = 0;
    double pflip = 1;
    Debug_totali++;
    TotalRegularParticleSteps     += (!isOverrelaxing)? 1 : 0;
    TotalOverrelaxedParticleSteps += ( isOverrelaxing)? 1 : 0;
    if (Prop.isNearGroup){
        //        cout << "here" << endl;
//        cout << "dErot(i) = " << dErot(i) <<endl;
//        cout << "dENN(i) = " << dENN(i) <<endl;
        assert(almostEquals(dErot(i),dENN(i)));
        double dE = dErot(i);
        double pNear = exp(-dE * InvTemperature);
        pflip *= (Prop.isTomitaMethodNearGroupBuiltIn)? pNear : 1;
//         if (dE > 0 && realDis(gen) > exp(-dE * InvTemperature)){
        if (!isOverrelaxing && !Prop.isTomitaMethodNearGroupBuiltIn && dE > 0 && realDis(gen) > pNear){
            return;   // reject move
        }
        shift = Lat.Neighbours.size();
    }
//    const int i = intDis(gen); //sorted by Index
//    const int i = Random(N); //sorted by Index
//    const int xi = i % Lat.L1;
//    const int yi = i / Lat.L1;
    const int xi = Lat.XPbyIndex(i);
    const int yi = Lat.YPbyIndex(i);
    assert(xi == i % Lat.L1);
    assert(yi == i / Lat.L1);

    vector<int> jrejVecSmall;
    double RandomPoisson = PoissonDis(gen);
    for (int j = 0; j < RandomPoisson; ++j) {
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
                if (realDis(gen) <= prel){
                    pflip *= alpha / (2*Beta*abs(Jtilde) + alpha);
                }
//            }                                           //uncomment for no group
        } else {
            double prel = alpha * invLambda;
            assert(prel>=0);
//            for (int k = 0; k < power; ++k) {           //uncomment for no group
                if (realDis(gen) <= prel) {
                    pflip *= (2*Beta*abs(Jtilde) + alpha) * invAlpha;
                }
//            }                                           //uncomment for no group
        }

    }

//    if (Prop.isTomitaMethodNearGroupBuiltIn){
//        //        cout << "here" << endl;
//        assert(almostEquals(dErot(i),dENN(i)));
//        double dEnear = dErot(i);
//        pflip *= exp(-dEnear * InvTemperature);
//    }

    if (realDis(gen) <= pflip){
        Debug_Iaccepted++;    // move accepted
        AcceptedRegularParticleSteps     += (!isOverrelaxing)? 1 : 0;
        AcceptedOverrelaxedParticleSteps += ( isOverrelaxing)? 1 : 0;
        if (Prop.isNearGroup){
            updateField(i);
        }
        updateState(i);
//        cout << "i = " << i << " accepted" <<endl;
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
//            for (int i : ParticleListSorted) {
            for (int i = 0; i < N; ++i) {
                overRelaxateUiNew(i);
                run1TomitaParticle(i);
            }
        }
        isOverrelaxing = false;
        for (int j = 0; j < Prop.MetropolisSteps; ++j) {
//            for (int c = 0; c < N; ++c) {
//                int i = intDis(gen);
            for (int i = 0; i < N; ++i) {
                randomizeUiNew();
                run1TomitaParticle(i);
            }
        }
    } else {
        isOverrelaxing = false;
        for (int j = 0; j < Prop.MetropolisSteps; ++j) {
//            for (int c = 0; c < N; ++c) {
//                int i = intDis(gen);
            for (int i = 0; i < N; ++i) {
                randomizeUiNew();
                run1TomitaParticle(i);
            }
        }
    }

//    shuffle(ParticleList.begin(), ParticleList.end(), gen);
//    for (int i : ParticleList) {
//        randomizeUiNew();
//        run1ClockParticle(i);
//    }

//    double RateAccepted = AdapticeCounterAcceptance / (cmax + 0.0);
//    double sigma_temp = sigma * (0.9/(1.0 - RateAccepted));
//    if(sigma_temp > 500.0) {
//        sigma_temp = 500.0;
//    }
//    sigma = sigma_temp;

}

double MC2d::dEijrej(int jrej, int i, int xi, int yi){  // jrej is sorted by distance and is periodic
//    cout << "PRel" <<endl;
    assert(jrej >=0 && jrej < N-1);
//    cout << "jrej from PRel = " << jrej << endl;
//    int jper = Lat.PByDistance(jrej);    //jper is periodic
//    int j = Lat.unperiodicParticle(jper, i);
//    assert(j >=0 && j < N);
//    assert(jper >=0 && jper < N);
//    int jperback = Lat.periodicParticle(j, i);
//    assert(jper == jperback);
//    assert(almostEquals(Lat.getMinDistance(i,j), Lat.getMinDistance(0, jper)));
//    double Phatj = ProbClock(jrej);
    assert(Lat.XPByDistance(jrej) + Lat.YPByDistance(jrej) * Lat.L1 == Lat.PByDistance(jrej));
    const int xjper = Lat.XPByDistance(jrej);
    const int yjper = Lat.YPByDistance(jrej);
    const int xj0 = xjper + xi;
    const int yj0 = yjper + yi;
    const int xj = (xj0 >= Lat.L1)? xj0 - Lat.L1 : xj0;
    const int yj = (yj0 >= Lat.L1)? yj0 - Lat.L1 : yj0;
    int j2 = yj * Lat.L1 + xj;
    int jper =  yjper * Lat.L1 + xjper;
//    assert(j2 == j);
    const double dMx = UiNewX - Ux(i);
    const double dMy = UiNewY - Uy(i);
//    int jper = Lat.periodicParticle(j, i);
    assert(jper >=0 && jper < N);
    double dE = 2*(dMx * (Lat.Jxx1p(jper) * Ux(j2) + Lat.Jxy1p(jper) * Uy(j2)) +
                   dMy * (Lat.Jxy1p(jper) * Ux(j2) + Lat.Jyy1p(jper) * Uy(j2)));
    return dE;
}

double MC2d::Jtildeijrej(int jrej, int i, int xi, int yi){  // jrej is sorted by distance and is periodic
//    cout << "PRel" <<endl;
    assert(jrej >=0 && jrej < N-1);
//    cout << "jrej from PRel = " << jrej << endl;
//    int jper = Lat.PByDistance(jrej);    //jper is periodic
//    int j = Lat.unperiodicParticle(jper, i);
//    assert(j >=0 && j < N);
//    assert(jper >=0 && jper < N);
//    int jperback = Lat.periodicParticle(j, i);
//    assert(jper == jperback);
//    assert(almostEquals(Lat.getMinDistance(i,j), Lat.getMinDistance(0, jper)));
//    double Phatj = ProbClock(jrej);
    assert(Lat.XPByDistance(jrej) + Lat.YPByDistance(jrej) * Lat.L1 == Lat.PByDistance(jrej));
    const int xjper = Lat.XPByDistance(jrej);
    const int yjper = Lat.YPByDistance(jrej);
    const int xj0 = xjper + xi;
    const int yj0 = yjper + yi;
    const int xj = (xj0 >= Lat.L1)? xj0 - Lat.L1 : xj0;
    const int yj = (yj0 >= Lat.L1)? yj0 - Lat.L1 : yj0;
    int j2 = yj * Lat.L1 + xj;
    int jper =  yjper * Lat.L1 + xjper;
//    assert(j2 == j);
    const double sigmax = (Ux(i) - UiNewX) * 0.5; //TODO make setters and getters for UiNew to avoid unintentional rewrite
    const double sigmay = (Uy(i) - UiNewY) * 0.5;
//    int jper = Lat.periodicParticle(j, i);
    assert(jper >=0 && jper < N);
    double Jtilde = -2*(sigmax * (Lat.Jxx1p(jper) * Ux(j2) + Lat.Jxy1p(jper) * Uy(j2)) +
                        sigmay * (Lat.Jxy1p(jper) * Ux(j2) + Lat.Jyy1p(jper) * Uy(j2)));
    return Jtilde;
}