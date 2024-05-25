//
// Created by Sadeq Ismailzadeh on ۱۷/۰۱/۲۰۲۲.
//

#include "MC2d.h"
#include "WalkerAlias.h"
#include <highfive/H5Easy.hpp>
#ifdef _OPENMP // Use omp.h if -fopenmp is used in g++
#include <omp.h>
#endif

MC2d::MC2d(const Lattice2d &lat, const int64_t &seed, const MCProperties  &properties) :
        Lat(lat),
        Prop(properties),
        gen(seed),
        realDis(0, 1),
        realDissym(-1, 1),
        intDis(0, Lat.N - 1),
        gaussianDis(0.0, 1.0),
        N(Lat.N),
//        J(Lat.J),
        Jxx(Lat.Jxx),
        Jyy(Lat.Jyy),
        Jxy(Lat.Jxy),
        Plabel(Lat.PLabel),
        invN(1./(N + 0.0))
    {
    Ux.setZero(N);
    Uy.setZero(N);

    Hx.setZero(N);
    Hy.setZero(N);

//    indices.setZero(2 * N - 2, N);

    MVectConfig.setZero();
    MFEVectConfig.setZero();

    hxAll.setZero(N);
    hyAll.setZero(N);
//    Jself = Lat.J.block<2,2>(0,0);
//    Jself << Jxx(0,0), Jxy(0,0), Jxy(0,0), Jyy(0,0);

    //taking data
    StatisticalData.setZero(1000, 32);
    LastTempIndex = 0;

    //eigenvectors
    OPmakerX.setZero(N);
    OPmakerY.setZero(N);

    OPmakerXX.setZero(N);
    OPmakerXY.setZero(N);
    OPmakerYX.setZero(N);
    OPmakerYY.setZero(N);



    Debug_Iaccepted = 0;
    Debug_totali = 0;
    isSerial=true;
    Nthread = 1;
//    properties.isFiniteDiffSimulation = false;
//    properties.ControlParamType = 'T';
    hHat.setZero();
    hx = 0;
    hy = 0;

    //specific use
    iReCalc = 0;
    SnapshotTakingIndex = 0;
    Jself.setZero();
    Jself = Lat.Jself;

    UTx.setZero(N);
    UTy.setZero(N);
    GVecVsTemperature.setZero(1000, Lat.DistanceVec.size());
    GConVecVsTemperature.setZero(1000, Lat.DistanceVec.size());



    sigma = 60;
    ParticleList.setLinSpaced(N, 0, N-1);
    ParticleListSorted = ParticleList;
    sort(ParticleListSorted.begin(), ParticleListSorted.end());

    ProbClock.setZero(Lat.PByDistance.size());
    InVLogProbClock.setZero(Lat.PByDistance.size());
    InVOneMinusProbClock.setZero(Lat.PByDistance.size());

    ProbClockFar.setZero(Lat.PByDistanceFar.size());
    InVLogProbClockFar.setZero(Lat.PByDistanceFar.size());
    InVOneMinusProbClockFar.setZero(Lat.PByDistanceFar.size());

    JrejSum.setZero(Lat.PByDistance.size());
    JrejProb.setZero(Lat.PByDistance.size());
    JrejProbCumul.setZero(Lat.PByDistance.size());

    dVstarVec.setZero(Lat.PByDistance.size());

    ExchangeFactor = 0.5;
    NumResampled = 0;

    // acceptance ratio
    TotalRegularParticleSteps = 0;
    AcceptedRegularParticleSteps = 0;
    TotalOverrelaxedParticleSteps = 0;
    AcceptedOverrelaxedParticleSteps = 0;
    complexitySum = 0;
    RNDGenerationSum = 0;

    complexityAcceptSum = 0;
    RNDGenerationAcceptSum = 0;

    AcceptedBondsSum.setZero(N - 1);
    SelectedBondsSum.setZero(N - 1);
    SelectedBondsLocal.setZero(N - 1);


    UAntiAlignedWithHSum = 0;
    UAlignedWithHSum = 0;
    EIncreaserReducerTotalSum = 0;
    attemptSum = 0;
    FineTuneNum = 0;


    SCOReshuffleCounter = 0;

    jrejMat.resize(N, std::vector<double>(1, 0));
    jrejVecBool.resize(N, false);
    jrejVecRepeated.resize(N, 0);
}

void MC2d::init() {
    Randomize();
    randomizeAllSpins();
    calcAllFields();
    generateOPmaker();
//    saveLocations();
    setSimulationParams();
    setFieldOffSimulationType();
    if (Prop.isFieldOn) {
        setFieldOnSimulationType();
    }
    StopWatchTemperature.reset();
    StopWatchTotal.reset();
    UpdatehAll();
}

void MC2d::setId(){
    #pragma omp critical (IndexNaming)
    {
        InstanceIndex = LastInstanceIndex++;
        cout << "setId " << InstanceIndex <<endl;
    }
//    setFFT();
//    cout << "setFFT inside setid " << InstanceIndex <<endl;
}

void MC2d::resetLastInstanceIndex() {
    LastInstanceIndex = 0;
}


void MC2d::randomizeAllSpins(){
//    const double PI = acos(-1);
//    double Theta = PI/3;
    double Theta = Prop.InitialAlignment;
    for (int i = 0; i < N; i++) {
        if (Prop.isDipolesInitializedOrdered){
            Ux(i) = cos(Theta);
            Uy(i) = sin(Theta);
        } else {
            randomizeUiNew();
            Ux(i) = UiNewX;
            Uy(i) = UiNewY;
        }
    }
}

void MC2d::loadEquilibriumState(){
    #pragma omp critical (Hdf5LoadFiles)
    {
        string filePath = ParentFolderPath + "/.."s + "/EquilibriumState_L="s + to_string(Lat.L1) + ".h5"s;
        H5Easy::File EQStateHdf5Ens(filePath, H5Easy::File::OpenOrCreate);
        string dataPath ="/"s + to_string(Temperature) + "/"s + to_string(InstanceIndex);
        Ux = H5Easy::load<Eigen::VectorXd>(EQStateHdf5Ens, dataPath +  "/EqulibriumState/Ux"s);
        Uy = H5Easy::load<Eigen::VectorXd>(EQStateHdf5Ens, dataPath +  "/EqulibriumState/Uy"s);
    }

}



double MC2d::dErot(int i) {
    double dE1 = 0; //// assert
    const double dMx = UiNewX - Ux(i);
    const double dMy = UiNewY - Uy(i);


    //another version using Jxx, Jxy, Jyy
    const double sMx = UiNewX + Ux(i);
    const double sMy = UiNewY + Uy(i);
//    dE += (dM(0) * (Jxx(i,i) * sM(0) + Jxy(i,i) * sM(1))) + (dM(1) * (Jxy(i,i) * sM(0) + Jyy(i,i) * sM(1)));
//    dE += (dMz)* Jzz(i, i)* (UiNewZ + Uz(i));
//TODO BUG its not valid for honeycomb
    dE1 += (dMx * (Jself(0,0) * sMx + Jself(1,0) * sMy)) + (dMy * (Jself(1,0) * sMx + Jself(1,1) * sMy));

    if (Prop.isFieldOn){
        dE1 -= dMx * hx + dMy * hy;
    }

    dE1 += 2 * (dMx* Hx(i) + dMy * Hy(i));
    return dE1;
}

void MC2d::run1MetropolisStep() {
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

//    shuffle(ParticleList.begin(), ParticleList.end(), gen);
//    for (int i : ParticleList) {
    for (int i = 0; i < N; i++) {
        TotalUpdateSum++;
        complexityLocalSum = 0;
        RNDGenerationLocalSum = 0;
//    for (int c = 0; c < N; c++) {
//        int i = intDis(gen);
        Debug_totali++;
        TotalRegularParticleSteps += 1;
        randomizeUiNew();
//        random3dAngle(i);
//        randomAdaptive(i);
        double dE = dErot(i);
//        assert(almostEquals(dErot(i), dENN(i)));
        assert(almostEquals(1/Temperature, InvTemperature));
        RNDGenerationSum += 1;
        if (dE < 0 || (realDis(gen) < exp(-dE * InvTemperature))) {
//        if (realDis(gen) < 0.5*(1-tanh(0.5*dE * InvTemperature))) {
            updateField(i);
            updateState(i);
            Debug_Iaccepted++;
            AcceptedRegularParticleSteps += 1;
            CounterAccceptence++;
            AcceptSum++;
//        } else {
//            overRelaxateUiNew(i);
//            updateField(i);
//            updateState(i);
//            Debug_Iaccepted++;
//            CounterAccceptence++;
        }
    }

//    shuffle(ParticleList.begin(), ParticleList.end(), gen);
//    for (int i : ParticleList) {
//        Debug_totali++;
//        randomizeUiNew();
//        double dE = dErot(i);
//        assert(almostEquals(1/Temperature, InvTemperature));
//        if (dE < 0 || (Random() < exp(-dE * InvTemperature))) {
//            updateField(i);
//            updateState(i);
//            Debug_Iaccepted++;
//            CounterAccceptence++;
//        }
//    }

//    double RateAccepted = CounterAccceptence / (N + 0.0);
//    double sigma_temp = sigma * (0.5/(1.0 - RateAccepted));
//    if(sigma_temp > 500.0) {
//        sigma_temp = 500.0;
//    }
//    sigma = sigma_temp;

//    if(sigma_temp > PI){
//        sigma_temp = PI;
//    } else if (sigma_temp < 1e-15){
//        sigma_temp = 1e-15;
//    }
//    sigma = sigma_temp;
}

void MC2d::run1OverRelaxationStep() {
    static const double PI = acos(-1);
//    int CounterAccceptence = 0;
    if (Prop.isTakeSnapshot && (SnapshotTakingIndex++ % TakeSnapshotStep == 0)){
        takeSnapshot();
    }
    if (++iReCalc % 1'000'000 == 0){
        calcAllFields();
    }

//    for (int c = 0; c < N; c++) {
////        Debug_totali++;

//        double theta_i = atan2(Uy(i), Ux(i));
//        double theta_H = atan2(Hy(i), Hx(i));
//        double theta_inew = theta_i + 2 * (theta_H - theta_i);
//        UiNewX = cos(theta_inew);
//        UiNewY = sin(theta_inew);
////        cout << "theta_inew - theta_H = " << theta_inew - theta_H << endl;
////        cout << "theta_i - theta_H = " << theta_i - theta_H << endl;
////        cout << "dErot(i) = " << dErot(i) << endl;
//        assert(almostEquals(dErot(i),0));
//        updateField(i);
//        updateState(i);
//    }


//    shuffle(ParticleList.begin(), ParticleList.end(), gen);
//    for (int i : ParticleListSorted) {
    for (int i = 0; i < N; ++i) {
        TotalUpdateSum++;
//    for (int c = 0; c < N; c++) {
//        int i = intDis(gen);
//        double theta_i = atan2(Uy(i), Ux(i));
//        double theta_H = atan2(Hy(i), Hx(i));
//        double theta_inew = theta_i + 2 * (theta_H - theta_i);
        // TODO overrelaxation using well known formula in literature
        double m = Hy(i) / Hx(i);
        double m2 = m*m;
        double div1pm2 = 1/(1+m2);
        UiNewX = div1pm2*((1-m2)*Ux(i) +    2*m*Uy(i));
        UiNewY = div1pm2*(   2*m*Ux(i) - (1-m2)*Uy(i));

//        double dotProd = Hx(i) * Ux(i) + Hy(i) * Uy(i);
//        double absH = sqrt(Hx(i)*Hx(i) + Hy(i)*Hy(i));
//        double cosHiUi = dotProd/absH;
//        //AntiAlign Reduces energy
//        if (cosHiUi > -0.9) {
//            UAlignedWithHSum++;
//        } else {
//            UAntiAlignedWithHSum++;
//        }
//        EIncreaserReducerTotalSum++;

//        double abstheta_diff = abs(theta_inew - theta_i);
//        double theta_diff = abstheta_diff;
//        while(theta_diff > 2 * PI){
//            theta_diff -= 2* PI;
//        }
//        if (theta_diff < PI * 0.1){
//            continue;
//        }

//        UiNewX2 = cos(theta_inew);
//        UiNewY2 = sin(theta_inew);
//        assert(almostEquals(UiNewX,UiNewX2));
//        assert(almostEquals(UiNewY,UiNewY2));
//        cout << "theta_inew - theta_H = " << theta_inew - theta_H << endl;
//        cout << "theta_i - theta_H = " << theta_i - theta_H << endl;
//        cout << "dErot(i) = " << dErot(i) << endl;
//        if (realDis(gen) < 0.5)  {
            assert(almostEquals(dErot(i), 0));
            Debug_Iaccepted++;
            Debug_totali++;
            TotalOverrelaxedParticleSteps    += 1;
            AcceptedOverrelaxedParticleSteps += 1;
            updateField(i);
            updateState(i);
//        }
    }
}


void MC2d::run1HybridStep() {
    if (Prop.isOverRelaxationMethod && Prop.isMacIsaacMethod) {
        for (int i = 0; i < Prop.OverrelaxationSteps; ++i) {
            run1OverRelaxationStep();
        }
        for (int i = 0; i < Prop.MetropolisSteps; ++i) {
            run1MetropolisStep();
        }
    } else if (Prop.isClockMethod) {
        run1ClockStep();
    } else if (Prop.isSCOMethod) {
        if (Prop.isSCOMethodPreset) {
            run1SCOStepPreset();
        } else {
            run1SCOStep();
        }
    } else if (Prop.isTomitaMethod){
        run1TomitaStep();
    } else {
        for (int i = 0; i < Prop.MetropolisSteps; ++i) {
            run1MetropolisStep();
        }
    }
}

void MC2d::updateField(const int i) {
    const double dMx = UiNewX - Ux(i);
    const double dMy = UiNewY - Uy(i);
    if (!(Prop.isNeighboursMethod || Prop.isClockMethod || Prop.isSCOMethod || Prop.isTomitaMethod)) {
        Hx(i) -= Jself(0, 0) * dMx + Jself(0, 1) * dMy;
        Hy(i) -= Jself(0, 1) * dMx + Jself(1, 1) * dMy;
    }
    // due to last measurement 1401.05.11  this code is 0.08% faster
    if (Prop.isNeighboursMethod || Prop.isClockMethod || Prop.isSCOMethod || Prop.isTomitaMethod){
//        cout << "updateField NeighboursMethod" <<endl;
//    if (false){
//        Hx(i)  += Jself(0,0) * dMx + Jself(0, 1) * dMy;
//        Hy(i)  += Jself(0,1) * dMx + Jself(1, 1) * dMy;
//        assert(almostEquals(Jself(0,0), 0));
        const int xi = Lat.XPbyIndex(i);
        const int yi = Lat.YPbyIndex(i);
        assert(xi == i % Lat.L1);
        assert(yi == i / Lat.L1);
        const int L1 = Lat.L1;
        const int L2 = Lat.L2;
        if (Prop.isStoreNeighboursForAllPariticles){
            complexitySum += Lat.Neighbours.size();
            complexityLocalSum += Lat.Neighbours.size();
            for (int j = 0; j < Lat.Neighbours.size(); ++j) {
                const int pj = Lat.NeighboursMat(j, i);
//                assert(almostEquals(Lat.getMinDistance(0,Lat.Neighbours(j)), Lat.getMinDistance(i, pj)));
                Hx(pj) += Lat.Jxxn(j) * dMx + Lat.Jxyn(j) * dMy; //TODO if possible: optimize: simd
                Hy(pj) += Lat.Jxyn(j) * dMx + Lat.Jyyn(j) * dMy;
//                Hx(pj) += Lat.JxxMatn(j, i) * dMx + Lat.JxyMatn(j, i) * dMy;
//                Hy(pj) += Lat.JxyMatn(j, i) * dMx + Lat.JyyMatn(j, i) * dMy;
            }
        } else {
            complexitySum += Lat.Neighbours.size();
            complexityLocalSum += Lat.Neighbours.size();
            for (int j = 0; j < Lat.Neighbours.size(); ++j) {
                const int xj0 = Lat.Xn0(j) + xi;
                const int yj0 = Lat.Yn0(j) + yi;
                const int xj = (xj0 >= L1)? xj0 - L1 : xj0;
                const int yj = (yj0 >= L2)? yj0 - L2 : yj0;
//            int xj = (Lat.Xn0(j) + xi) % L1;
//            int yj = (Lat.Yn0(j) + yi) % L2;
                const int pj = yj * L1 + xj;
                assert(almostEquals(Lat.getMinDistance(0,Lat.Neighbours(j)), Lat.getMinDistance(i, pj)));
//            assert(almostEquals(Lat.getMinDistanceVec(0,Lat.Neighbours(j)), Lat.getMinDistanceVec(i, pj)));
//            assert(almostEquals(Lat.Jxxn(j), Lat.Jxx1p(Lat.Neighbours(j))));
//            assert(almostEquals(Lat.Jxxn(j), Lat.Jxx(0,Lat.Neighbours(j))));
//            assert(almostEquals(Lat.Jxxn(j), Lat.Jxx(i,pj)));
                Hx(pj) += Lat.Jxxn(j) * dMx + Lat.Jxyn(j) * dMy;
                Hy(pj) += Lat.Jxyn(j) * dMx + Lat.Jyyn(j) * dMy;
            }
        }
    } else if (Prop.LatticeType != 'h' && Prop.isMacIsaacMethod) {
        complexitySum += N;
        complexityLocalSum += N;
        int rowspin = i / Lat.L1;
        int colspin = i % Lat.L1;
        const int L1 = Lat.L1;
        const int L2 = Lat.L2;
        for (int j = 0; j < L2; ++j) {
            int Fn = 2 * L1 * (L2 + j - rowspin) - colspin + L1;
            int j1 = L1 * j;
            for (int k = 0; k < L1; ++k) {
                int pnum2 = j1 + k;
                Hx(k + j1) += Lat.Wxx(Fn + k) * dMx + Lat.Wxy(Fn + k) * dMy;
                Hy(k + j1) += Lat.Wyy(Fn + k) * dMy + Lat.Wxy(Fn + k) * dMx;
            }
        }
    } else if (Prop.LatticeType == 'h' && Prop.isMacIsaacMethod && Lat.isHoneycombFromTriangular) {
        complexitySum += N;
        complexityLocalSum += N;
//        cout << "here";
        const int L1 = Lat.L1;
        const int L2 = Lat.L2;
        int rowspin = (i/2) / Lat.L1;
        int colspin = (i/2) % Lat.L1;
        if (Lat.PSubLattice(i) == 0){
            assert(Lat.PSubLattice(i) == 0);
            for (int j = 0; j < L2; ++j) {
                int Fn = 2*(L1)*(L2+j-rowspin)-colspin+(L1);
                int j1 = (2*L1)*j;
                for (int k = 0; k < (L1); ++k) {
                    assert(j1 + 2*k < N);
                    assert(j1 + 2*k+1 < N);
                    assert(Fn+k < 4*Lat.Nbase);
                    assert(i < N);
                    Hx(2*k + j1) += Lat.Wxx11(Fn + k) * dMx + Lat.Wxy11(Fn + k) * dMy;
                    Hy(2*k + j1) += Lat.Wyy11(Fn + k) * dMy + Lat.Wxy11(Fn + k) * dMx;
                    Hx(2*k+1 + j1) += Lat.Wxx12(Fn + k) * dMx + Lat.Wxy12(Fn + k) * dMy;
                    Hy(2*k+1 + j1) += Lat.Wyy12(Fn + k) * dMy + Lat.Wxy12(Fn + k) * dMx;
                }
            }
        } else {
            assert(Lat.PSubLattice(i) == 1);
            for (int j = 0; j < L2; ++j) {
                int Fn = 2*(L1)*(L2+j-rowspin)-colspin+(L1);
                int j1 = (2*L1)*j;
                for (int k = 0; k < (L1); ++k) {
                    assert(j1 + 2*k < N);
                    assert(j1 + 2*k+1 < N);
                    assert(Fn+k < 4*Lat.Nbase);
                    assert(i < N);
                    Hx(2*k + j1) += Lat.Wxx21(Fn + k) * dMx + Lat.Wxy21(Fn + k) * dMy;
                    Hy(2*k + j1) += Lat.Wyy21(Fn + k) * dMy + Lat.Wxy21(Fn + k) * dMx;
                    Hx(2*k+1 + j1) += Lat.Wxx22(Fn + k) * dMx + Lat.Wxy22(Fn + k) * dMy;
                    Hy(2*k+1 + j1) += Lat.Wyy22(Fn + k) * dMy + Lat.Wxy22(Fn + k) * dMx;
                }
            }
        }

    } else {
        complexitySum += N;
        complexityLocalSum += N;
        int a = 0;
        for (int j = 0; j < N; j++) {
            Hx(j) += Jxx(j, i) * dMx +  Jxy(j, i) * dMy;
            Hy(j) += Jxy(j, i) * dMx +  Jyy(j, i) * dMy;
        }
    }

    if (Prop.isHavingExchangeInteraction && !Prop.isExchangeInteractionCombinedWithDij){
        //        cout << "updateField NeighboursMethod" <<endl;
//    if (false){
//        Hx(i)  += Jself(0,0) * dMx + Jself(0, 1) * dMy;
//        Hy(i)  += Jself(0,1) * dMx + Jself(1, 1) * dMy;
//        assert(almostEquals(Jself(0,0), 0));
        int xi = i % Lat.L1; // TODO optimize by XPbyInedx
        int yi = i / Lat.L1;
        const int L1 = Lat.L1;
        const int L2 = Lat.L2;
        for (int j = 0; j < Lat.NearestNeighbours.size(); ++j) {
            const int xj0 = Lat.NearestNeighbours(j) % L1 + xi;
            const int yj0 = Lat.NearestNeighbours(j) / L1 + yi;
            const int xj = (xj0 >= L1)? xj0 - L1 : xj0;
            const int yj = (yj0 >= L2)? yj0 - L2 : yj0;

//            int xj = (Lat.Xn0(j) + xi) % L1;
//            int yj = (Lat.Yn0(j) + yi) % L2;
            const int pj = yj * L1 + xj;
//            assert(minDisi < Lat.DistanceVec1p(n) || almostEquals(minDisi, Lat.DistanceVec1p(n)));
            assert(pj >= 0 && pj < Ux.size());
            assert(i != pj);
//            cout << "Lat.getMinDistance(i, pj) = " <<Lat.getMinDistance(i, pj) <<endl;
            assert(almostEquals(Lat.getMinDistance(0,Lat.NearestNeighbours(j)), Lat.getMinDistance(i, pj)));
            assert(almostEquals(Lat.getMinDistanceVec(0,Lat.NearestNeighbours(j)), Lat.getMinDistanceVec(i, pj)));
//            assert(almostEquals(Lat.Jxxn(j), Lat.Jxx1p(Lat.Neighbours(j))));
//            assert(almostEquals(Lat.Jxxn(j), Lat.Jxx(0,Lat.Neighbours(j))));
//            assert(almostEquals(Lat.Jxxn(j), Lat.Jxx(i,pj)));
            Hx(pj) += - dMx * ExchangeFactor;
            Hy(pj) += - dMy * ExchangeFactor;
        }
    }
}

void MC2d::stabilize() {
    EtimeEq.clear();
    EtimeEq.shrink_to_fit();
    mtimeEq.clear();
    mtimeEq.shrink_to_fit();
    if (Prop.isLoadEquilibriumState) {
        loadEquilibriumState();
        calcAllFields();
    }
    if (Prop.isClockMethod || Prop.isSCOMethod){
        if (Prop.isBoxes){
            setProbClockBoxes();
        } else {
            setProbClock();
        }
    } else if (Prop.isTomitaMethod){
        setProbTomita();
    }

    calcOrderParam();
    double E0 = getEnergy();
    double m0 = MVectConfig.norm() / N;
    cout << "m0 = " << m0 << endl;

    StopWatchTemperature.reset();
    for (int i = 0; i < Prop.NStabilize - 1; i++) {
        if (Prop.isSaveEquilibrationTimeSeries){
            calcOrderParam();
            EtimeEq.push_back(getEnergy());
            mtimeEq.push_back(MVectConfig.norm());
        }
        run1HybridStep();
    }
    if (Prop.isSaveEquilibrationTimeSeries){
        calcOrderParam();
        EtimeEq.push_back(getEnergy());
        mtimeEq.push_back(MVectConfig.norm());
    }
//    cout << "sigma = " << sigma << endl;
    if (Prop.isSaveEquilibrationTimeSeries) {
        if (Prop.isHistogramSimulation){
            #pragma omp critical (Hdf5DumpFiles)
            {
                if (Prop.isSaveSamplesToSepareteFolders) {
                    H5Easy::File timeSeriesHdf5Ens(FolderPath + "/EnsembleResults.h5"s,
                                                   H5Easy::File::OpenOrCreate);
                    H5Easy::dump(timeSeriesHdf5Ens, "/TimeSeries/Equilibration/Energy"s, EtimeEq,
                                 H5Easy::DumpOptions(H5Easy::Compression(9)));
                    H5Easy::dump(timeSeriesHdf5Ens, "/TimeSeries/Equilibration/Magnetization"s, mtimeEq,
                                 H5Easy::DumpOptions(H5Easy::Compression(9)));
                } else {
                    H5Easy::File timeSeriesHdf5Ens(ParentFolderPath + "/Results.h5"s,
                                                   H5Easy::File::OpenOrCreate);
                    H5Easy::dump(timeSeriesHdf5Ens, ("/" + to_string(InstanceIndex) + "/TimeSeries/Equilibration/Energy"s), EtimeEq,
                                 H5Easy::DumpOptions(H5Easy::Compression(9)));
                    H5Easy::dump(timeSeriesHdf5Ens, ("/" + to_string(InstanceIndex) + "/TimeSeries/Equilibration/Magnetization"s), mtimeEq,
                                 H5Easy::DumpOptions(H5Easy::Compression(9)));
                }
            }
        }
        if (Prop.isFiniteDiffSimulation){
            #pragma omp critical (Hdf5DumpFiles)
            {
                H5Easy::File timeSeriesHdf5Ens(ParentFolderPath + "/Results.h5"s,
                                               H5Easy::File::OpenOrCreate);
                string dataPath = "/"s + to_string(Temperature) + "/"s + to_string(InstanceIndex);
                H5Easy::dump(timeSeriesHdf5Ens, (dataPath + "/TimeSeries/Equilibration/Energy"s), EtimeEq,
                             H5Easy::DumpOptions(H5Easy::Compression(9)));
                H5Easy::dump(timeSeriesHdf5Ens, (dataPath + "/TimeSeries/Equilibration/Magnetization"s), mtimeEq,
                             H5Easy::DumpOptions(H5Easy::Compression(9)));

            }
        }

        EtimeEq.clear();
        EtimeEq.shrink_to_fit();
        mtimeEq.clear();
        mtimeEq.shrink_to_fit();
    }

    if (Prop.isSaveEquilibriumState) {
        #pragma omp critical (Hdf5DumpFiles)
        {
            string filePath = ParentFolderPath + "/.."s + "/EquilibriumState_L="s + to_string(Lat.L1) + ".h5"s;
            H5Easy::File EQStateHdf5Ens(filePath, H5Easy::File::OpenOrCreate);
            string dataPath = "/"s + to_string(Temperature) + "/"s + to_string(InstanceIndex);
            H5Easy::dump(EQStateHdf5Ens, dataPath + "/EqulibriumState/Ux"s, Ux,
                         H5Easy::DumpOptions(H5Easy::Compression(9)));
            H5Easy::dump(EQStateHdf5Ens, dataPath + "/EqulibriumState/Uy"s, Uy,
                         H5Easy::DumpOptions(H5Easy::Compression(9)));
            H5Easy::dump(EQStateHdf5Ens, dataPath + "/Temperature"s, Temperature,
                         H5Easy::DumpOptions(H5Easy::Compression(9)));
        }
    }

    if (Prop.isHistogramSimulation){
        StopWatchTemperature.printElapsedSecondsPrecise("Equilibration"s);
    }
}

void MC2d::stabilizeMinor() {
    if (Prop.isClockMethod || Prop.isSCOMethod){
        if (Prop.isBoxes){
            setProbClockBoxes();
        } else {
            setProbClock();
        }
//        cout << "setProbClock from stabilizeMinor" <<endl;
    } else if (Prop.isTomitaMethod){
        setProbTomita();
    }
    for (int i = 0; i < Prop.NStabilizeMinor; i++) {
        run1HybridStep();
        if (Prop.isSaveEquilibrationTimeSeries && Prop.isHistogramSimulation){
            calcOrderParam();
            EtimeEq.push_back(getEnergy());
            mtimeEq.push_back(MVectConfig.norm());
        }
    }
//    cout << "sigma = " << sigma << endl;

}

void MC2d::getStatisticalData() {
    double SumE = 0;
    double SumE2 = 0;
    double SumM = 0;
    double SumM2 = 0;
    double SumM4 = 0;
    double SumMPlane = 0;
    double SumMPlane2 = 0;
    double SumMPlane4 = 0;
    double SumMFE = 0;
    double SumMFE2 = 0;
    double SumMFE4 = 0;
    double SumMFEPlane = 0;
    double SumMFEPlane2 = 0;
    double SumMFEPlane4 = 0;
    double SumMT = 0;
    Vector2d SumMvect;
    Vector2d SumMFEvect;
    SumMvect.setZero();
    SumMFEvect.setZero();
    int N_MCstep = Prop.NData * Prop.dataTakingInterval;
    VectorXd mtime;
    VectorXd Etime;
    NumResampled = 0; // for clock method
    complexitySum = 0;
    RNDGenerationSum = 0;
    complexityAcceptSum = 0;
    RNDGenerationAcceptSum = 0;
    SelectedBondsSum.setZero();
    AcceptedBondsSum.setZero();
    AcceptSum = 0;
    TotalUpdateSum = 0;
    StopWatchTemperature.reset();

//    if (Prop.isHistogramMethod){
//        EtimeSeries.setZero(Prop.NData);
//        mtimeSeries.setZero(Prop.NData);
//    }

    if (Prop.isComputingAutoCorrelationTime) {
        mtime.setZero(Prop.NData);
        Etime.setZero(Prop.NData);
    }

    if (Prop.isComputingCorrelationLength){
        GVec.setZero(Lat.DistanceVec.size());
        GConVec.setZero(Lat.DistanceVec.size());
//        GMat.setZero(N,N);
    } else {
        GVec.setZero(0);
        GConVec.setZero(0);
//        GMat.setZero(0,0);
    }

    for (int i = 0; i < N_MCstep; i++) {
        run1HybridStep();
//        run1NewOverRelaxationStep();
//        run1MetropolisStep();

        if (i%(std::max({N_MCstep / 20, 1})) == 0 && Prop.isHistogramSimulation && i > 0){
            std::stringstream stream;
            stream << std::setprecision(2) << i/(N_MCstep+0.0);
            cout << "id " << InstanceIndex << ": progress  " <<  stream.str() << "%  "  <<endl;
            StopWatchTemperature.printElapsedSecondsPrecise();
        }

        if (i % Prop.dataTakingInterval == 0) {
            calcOrderParam();
            int j = i / Prop.dataTakingInterval;
            assert(j * Prop.dataTakingInterval == i);
            assert(j < Prop.NData);
            assert(j >= 0);
            double E_config = getEnergy();
            double m_config = MVectConfig.norm();
            if (Prop.isHistogramSimulation) {
                EtimeSeries.push_back(E_config);
                mtimeSeries.push_back(m_config);
                if (i%2 == 0){
                    EtimeSeries2step.push_back(E_config);
                    mtimeSeries2step.push_back(m_config);
                }
            }

            if (Prop.isComputingAutoCorrelationTime){
                mtime(i) = m_config;
                Etime(i) = E_config;
            }

            SumE += E_config;
            SumE2 += E_config * E_config;
            if (Prop.isComputingCorrelationLength){
                UpdateGVec();
            }
            SumMT += MTVectConfig.norm();    //TODO check if overflow
            double m2_config = m_config * m_config;
            double m4_config = m2_config * m2_config;
            SumM += m_config;
            SumM2 += m2_config;
            SumM4 += m4_config;
            double mPlane_config = sqrt(square(MVectConfig(0)) + square(MVectConfig(1)));
            double mPlane2_config = mPlane_config * mPlane_config;
            double mPlane4_config = mPlane2_config * mPlane2_config;
            SumMPlane += mPlane_config;
            SumMPlane2 += mPlane2_config;
            SumMPlane4 += mPlane4_config;
            SumMvect += MVectConfig.cwiseAbs();
            if (Prop.LatticeType != 't'){
                double mFE_config = MFEVectConfig.norm();
                double mFE2_config = mFE_config * mFE_config;
                double mFE4_config = mFE2_config * mFE2_config;
                SumMFE += mFE_config;
                SumMFE2 += mFE2_config;
                SumMFE4 += mFE4_config;
                double mFEPlane_config = sqrt(square(MFEVectConfig(0)) + square(MFEVectConfig(1)));
                double mFEPlane2_config = mFEPlane_config * mFEPlane_config;
                double mFEPlane4_config = mFEPlane2_config * mFEPlane2_config;
                SumMFEPlane += mFEPlane_config;
                SumMFEPlane2 += mFEPlane2_config;
                SumMFEPlane4 += mFEPlane4_config;
                SumMFEvect += MFEVectConfig;
            }
        }
    }


    double MeanM = SumM / Prop.NData;
    double MeanM2 = SumM2 / Prop.NData;
    double MeanM4 = SumM4 / Prop.NData;
    double MeanMPlane = SumMPlane / Prop.NData;
    double MeanMPlane2 = SumMPlane2 / Prop.NData;
    double MeanMPlane4 = SumMPlane4 / Prop.NData;
    double MeanMFE = SumMFE / Prop.NData;
    double MeanMFE2 = SumMFE2 / Prop.NData;
    double MeanMFE4 = SumMFE4 / Prop.NData;
    double MeanMFEPlane = SumMFEPlane / Prop.NData;
    double MeanMFEPlane2 = SumMFEPlane2 / Prop.NData;
    double MeanMFEPlane4 = SumMFEPlane4 / Prop.NData;
    double MeanE = SumE / Prop.NData;
    double MeanE2 = SumE2 / Prop.NData;
    Vector2d MeanMvectPP = SumMvect / (Prop.NData * N); // MeanMvect per particle
    // lattice specific order parameter
    mpp = MeanM / N;  //m per particle
    double mTpp = SumMT / (N * Prop.NData);  //m per particle
    double Xpp = (MeanM2 - square(MeanM)) / (N * Temperature);  // susceptibility
    double Binder = (MeanM4 / (square(MeanM2)));
    mPlanepp = MeanMPlane / N;
    double XPlanepp = (MeanMPlane2 - square(MeanMPlane)) / (N * Temperature);
    double BinderPlane = (MeanMPlane4 / (square(MeanMPlane2)));
    // ferromagnetic order parameter
    double mFEpp = MeanMFE / N;  //m per particle
    double XFEpp = (MeanMFE2 - square(MeanMFE)) / (N * Temperature);  // susceptibility
    double BinderFE = (MeanMFE4 / (square(MeanMFE2)));
    double mFEPlanepp = MeanMFEPlane / N;
    double XFEPlanepp = (MeanMFEPlane2 - square(MeanMFEPlane)) / (N * Temperature);
    double BinderFEPlane = (MeanMFEPlane4 / (square(MeanMFEPlane2)));

    double Epp = MeanE / N;
    double cpp = (MeanE2 - square(MeanE)) / (N * Temperature * Temperature);  // heat capacity

    // external field
    double hNorm = sqrt(square(hx) + square(hy));
    // runtime
    StopWatchTemperature.printElapsedSecondsPrecise("Data Taking"s);
    double runtimeTemperature = StopWatchTemperature.elapsedSecondsPrecise();
    StopWatchTemperature.reset();
    // Correlation time
    double tauM =  (Prop.isComputingAutoCorrelationTime) ? getAutoCorrelationTime(mtime) : 0;
    double tauE = (Prop.isComputingAutoCorrelationTime) ? getAutoCorrelationTime(Etime) : 0;
    double tau = max(tauM, tauE);
    if (Prop.isComputingCorrelationLength){
        calcGVec();
//        calcGVecOld();
    }
    if (Prop.isClockMethod && Prop.isNearGroup){
        double ResampleMean = NumResampled / (N * 10 * N_MCstep + 0.);
        JrejProb.array() = JrejSum.array() / JrejSum.sum();
        JrejProbCumul(0) = JrejProb(0);
        for (int i = 1; i < JrejProbCumul.size(); ++i) {
            JrejProbCumul(i) = JrejProbCumul(i - 1) + JrejProb(i);
        }
//        cout << "JrejProb = " << JrejProb(seq(0,min(20L, JrejProb.size()-1))) <<endl <<endl;
//        cout << "JrejProbCumul = " << JrejProbCumul(seq(0,min(20L, JrejProb.size()-1))) <<endl <<endl;
        cout << "JrejProb(0) = " << JrejProb(0) << endl;
        cout << "ResampleMean = " << ResampleMean <<endl;
//        cout << "JrejProbCumul = " << JrejProbCumul(seq(0,min(20L, JrejProb.size()-1))) <<endl <<endl;
//        cout << "JrejProb.sum() = " << JrejProb.sum()  <<endl;
    }
    VectorXd SelectedBondsProbability(N);
    VectorXd AcceptedBondsProbability(N);
    if (Prop.isCalcSelectedBondsProbability){
        SelectedBondsProbability = SelectedBondsSum / (N * Prop.NDataTotal  + 0.);
        AcceptedBondsProbability = AcceptedBondsSum / (AcceptSum  + 0.);
        #pragma omp critical (Hdf5DumpFiles)
        {
            H5Easy::File SelectedBonds(FolderPath + "/SelectedBondsProbability.h5"s,
                                        H5Easy::File::OpenOrCreate);
            H5Easy::dump(SelectedBonds, "/"s + to_string(Temperature) + "/SelectedBondsProbability"s, SelectedBondsProbability,
                         H5Easy::DumpOptions(H5Easy::Compression(9)));
            H5Easy::dump(SelectedBonds, "/"s + to_string(Temperature) + "/AcceptedBondsProbability"s, AcceptedBondsProbability,
                         H5Easy::DumpOptions(H5Easy::Compression(9)));
        }
    }
//    cout << "EIncreaserProb = " << UAntiAlignedWithHSum / (EIncreaserReducerTotalSum + 0.) << endl;
//    cout << "EReducerProb = " << UAlignedWithHSum / (EIncreaserReducerTotalSum + 0.) << endl;
//    cout << "attempt mean = " << attemptSum / (FineTuneNum + 0.0) << endl;
//    double Binder = 1 - (MeanM4 / (3*square(MeanM2)));
//    double phiMvect = atan2(MeanMvectPP(0), MeanMvectPP(2)); // phi = atan2(x,z)
//    double thetaMvect = atan2(hypot(MeanMvectPP(0), MeanMvectPP(2)), MeanMvectPP(1)); // theta = atan2(hypot(x,z), y)
    double complexity = 0;
    double RNDperStep = 0;
    if (Prop.isComplexityAccept){
        complexity = complexityAcceptSum / (AcceptSum + 0.);
        RNDperStep = RNDGenerationAcceptSum / (AcceptSum + 0.);
    } else {
        complexity = complexitySum / (TotalUpdateSum  + 0.);
        RNDperStep = RNDGenerationSum / (TotalUpdateSum  + 0.);
    }
    double ProbAccepted = (Debug_Iaccepted + 0.) / Debug_totali;
//    double Efficiency = (Prop.NData/(2*tau)) / runtimeTemperature; // number of uncorrelated data produced in simulation runtime
    double runtimePerMCStep = runtimeTemperature / (Prop.NDataTotal); // number of uncorrelated data produced in simulation runtime
    double t_eff = tau * runtimePerMCStep; // number of uncorrelated data produced in simulation runtime
    double ProbabilityRegularAccepted     = AcceptedRegularParticleSteps     / (TotalRegularParticleSteps     + 0.);
    double ProbabilityOverrelaxedAccepted = AcceptedOverrelaxedParticleSteps / (TotalOverrelaxedParticleSteps + 0.);
    StatisticalData.row(LastTempIndex++) <<
            Temperature, hNorm, Epp, cpp, mpp , Xpp, Binder,
            mPlanepp, XPlanepp, BinderPlane,
            MeanMvectPP.norm(), MeanMvectPP(0), MeanMvectPP(1), 0,
            mFEpp, XFEpp, BinderFE, mFEPlanepp, XFEPlanepp, BinderFEPlane, mTpp, tauM, tauE, tau,
            ProbAccepted, ProbabilityRegularAccepted, ProbabilityOverrelaxedAccepted,
            runtimeTemperature, t_eff, runtimePerMCStep, complexity, RNDperStep;

}

void MC2d::setSimulationParams() {
    assert(Prop.NStabilize > 0);
    assert(Prop.NStabilizeMinor > 0);
    assert(Prop.NData > 0);
    assert(Prop.dataTakingInterval > 0);

    TakeSnapshotStep = (Prop.NStabilize + Prop.NData * Prop.dataTakingInterval) / 5;
}

void MC2d::setFieldOffSimulationType() {
    Temperature = Prop.Pstart;
}

void MC2d::simulate() {
    init();
    LastTempIndex = 0;
    StatisticalData.setZero();
    setFFT();
    setControlParam(Prop.Pstart);

    if (Prop.isHistogramSimulation){
        simulateHistogram();
    } else if (Prop.isFiniteDiffSimulation) {
        simulateFiniteDiff();
    } else if (Prop.isEquidistantTemperatureInCriticalRegionSimulation) {
        simulateCriticalRegionEquidistant();
    } else {
        simulateCriticalRegionNonEquidistant();
    }

    StatisticalData = (StatisticalData(seq(0, LastTempIndex - 1), all)).eval();
    GVecVsTemperature = (GVecVsTemperature(seq(0, LastTempIndex - 1), all)).eval();
    GConVecVsTemperature = (GConVecVsTemperature(seq(0, LastTempIndex - 1), all)).eval();
    if (Prop.isSaveSampleStatData && Prop.isSaveSamplesToSepareteFolders) {
        ofstream SaveSampleStatData(FolderPath + "/SampleStatData.csv"s);
        SaveSampleStatData << StatisticalData.format(CSVFormat);
    }
    TimeMCTotal = StopWatchTotal.elapsedSeconds();
    destrtoyFFT();
    cout << "done id=" << InstanceIndex << endl;
}

void MC2d::setControlParam(const double Pset) {
    // TODO  setControlParam(CtrlParamAbs) --> setControlParam()     bring CtrlParamAbs
    if (Prop.ControlParamType == 'T'){
        Temperature = Pset;
        InvTemperature = 1/Temperature;
    } else if (Prop.ControlParamType == 'h') {
        hx = hHat(0) * Pset;
        hy = hHat(1) * Pset;
        UpdatehAll();
    }
}
void MC2d::simulateHistogram() {
    if (Prop.isSlowlyCoolingEquilibration){
        for (CtrlParamAbs = Prop.Pstart;
             CtrlParamAbs > Prop.PcEstimate && Temperature > 0; CtrlParamAbs -= Prop.Pdecrement){
            setControlParam(CtrlParamAbs);
            stabilizeMinor();
            cout << "id " << InstanceIndex << ": ControlParam = " << CtrlParamAbs << endl;
        }
    }

    CtrlParamAbs = Prop.PcEstimate;
    setControlParam(CtrlParamAbs);
    stabilizeAndGetStatisticalData();
    if (Prop.isSaveDataTakingTimeSeriesInHistogramSimulation){
        SaveTimeSeries();
    }
}

void MC2d::simulateFiniteDiff() {
//    for (CtrlParamAbs = Prop.Pstart; (CtrlParamAbs > Prop.Pend && Temperature > 0); CtrlParamAbs -= Prop.Pdecrement) {
    for (auto Param : Prop.TemperatureList) {
        CtrlParamAbs = Param;
        setControlParam(CtrlParamAbs);
        stabilizeAndGetStatisticalData();
    }
}

void MC2d::simulateCriticalRegionNonEquidistant() {
    double NearCriticalRange = 0.5;
    double PmaxCritical = Prop.PcEstimate + NearCriticalRange/2;
    double PminCritical = Prop.PcEstimate - NearCriticalRange/2;
    for (CtrlParamAbs = Prop.Pstart; (CtrlParamAbs > PmaxCritical && Temperature > 0); CtrlParamAbs -= Prop.Pdecrement){
        setControlParam(CtrlParamAbs);
        stabilizeMinor();
    }
    int a = 0;
    while(CtrlParamAbs > PminCritical) {
        if (a % 5 == 0) {
            stabilizeAndGetStatisticalData();
        } else {
            stabilizeMinor();
        }
        double dP = max(0.03*abs(CtrlParamAbs - Prop.PcEstimate), 0.002);
        CtrlParamAbs -= dP;
        setControlParam(CtrlParamAbs);
        a++;
    }
}

void MC2d::simulateCriticalRegionEquidistant() {
    double NearCriticalRange = 0.3;
//    double PmaxCritical = Prop.PcEstimate + NearCriticalRange/2;
//    double PminCritical = Prop.PcEstimate - NearCriticalRange/2;
    double PmaxCritical = Prop.PmaxCritical;
    double PminCritical = Prop.PminCritical;

    if (Prop.isSlowlyCoolingEquilibration) {
        for (CtrlParamAbs = Prop.Pstart;
             CtrlParamAbs > PmaxCritical && Temperature > 0; CtrlParamAbs -= Prop.Pdecrement) {
            setControlParam(CtrlParamAbs);
            stabilizeMinor();
            cout << "ControlParam = " << CtrlParamAbs << endl;
        }
    }

    double dP = 0.01;
    setControlParam(PmaxCritical);
    for(CtrlParamAbs = PmaxCritical; CtrlParamAbs > PminCritical; CtrlParamAbs -= dP) {
        setControlParam(CtrlParamAbs);
        stabilizeAndGetStatisticalData();
    }
}

void MC2d::stabilizeAndGetStatisticalData() {
    Debug_Iaccepted = 0;
    Debug_totali = 0;
    AcceptedRegularParticleSteps=0;
    TotalRegularParticleSteps=0;
    AcceptedOverrelaxedParticleSteps=0;
    TotalOverrelaxedParticleSteps=0;
    stabilize();
    getStatisticalData();
    cout << "ControlParam = " << CtrlParamAbs << endl;
    cout << "Debug_Iaccepted = " << Debug_Iaccepted << endl;
    cout << "probability accepted = " << (Debug_Iaccepted + 0.) / Debug_totali << "\n";
    StopWatchTotal.printElapsedSecondsPrecise();
    cout << endl;
}

void MC2d::calcAllFields() {
    // line be low equals to H = J * Uxy;
    VectorXd HnewX;
    VectorXd HnewY;
    VectorXd HnewZ;
    HnewX.setZero(Hx.size());
    HnewY.setZero(Hy.size());
//    H.setZero();
    if (Prop.isNeighboursMethod || Prop.isClockMethod || Prop.isSCOMethod || Prop.isTomitaMethod){
        cout << "calcAllFields NeighboursMethod" <<endl;
        for (int i = 0; i < N; ++i) {
            int xi = i % Lat.L1;
            int yi = i / Lat.L1;
            const int L1 = Lat.L1;
            const int L2 = Lat.L2;
//            if (i == N/2+L1/2) {
//                cout << "i = " << i <<endl;
//            }
            for (int j = 0; j < Lat.Neighbours.size(); ++j) {
//                cout << "Lat.Neighbours.size() " << Lat.Neighbours.size() << endl;
//                cout << "Lat.Jxxn.size() " << Lat.Jxxn.size() << endl;
                assert(Lat.Neighbours.size() == Lat.Jxxn.size());
                int xj = (Lat.Xn0(j) + xi) % L1;
                int yj = (Lat.Yn0(j) + yi) % L2;

//                int xj2 = ((Lat.Xn0(j) + xi) >= L1)? (Lat.Xn0(j) + xi) - L1 : (Lat.Xn0(j) + xi);
//                int yj2 = ((Lat.Yn0(j) + yi) >= L2)? (Lat.Yn0(j) + yi) - L2 : (Lat.Yn0(j) + yi);
//                cout << "xj = " << xj << endl;
//                cout << "xj2 = " << xj2 << endl;
//                assert(xj == xj2);
//                assert(yj == yj2);
                int pj = yj * L1 + xj;
                double minDisi = Lat.getMinDistance(i, pj);
                int n = Prop.NthNeighbour;
//                if (i == N/2+L1/2) {
//                    cout << "pj = " << pj <<endl;
//                }
//                assert(minDisi < Lat.DistanceVec1p(n) || almostEquals(minDisi, Lat.DistanceVec1p(n)));
                assert(almostEquals(Lat.getMinDistance(0,Lat.Neighbours(j)), Lat.getMinDistance(i, pj)));
                assert(almostEquals(Lat.getMinDistance(Lat.Neighbours(j),0), Lat.getMinDistance(pj, i)));
                assert(almostEquals(Lat.getMinDistanceVec(0,Lat.Neighbours(j)), Lat.getMinDistanceVec(i, pj)));
                assert(almostEquals(Lat.getMinDistanceVec(Lat.Neighbours(j), 0), Lat.getMinDistanceVec(pj, i)));
//                cout << "after assert" <<endl;
//                cout << "Lat.getMinDistanceVec(pj, i) = " << Lat.getMinDistanceVec(pj, i).transpose() <<endl;
//                cout << "Lat.getMinDistanceVec(i, pj) = " << Lat.getMinDistanceVec(i, pj).transpose() <<endl;
//                assert(almostEquals(Lat.getMinDistanceVec(pj, i), -1 * Lat.getMinDistanceVec(i, pj)));
//                assert(almostEquals(Lat.Jxxn(j), Lat.Jxx1p(Lat.Neighbours(j))));
//                assert(almostEquals(Lat.Jxxn(j), Lat.Jxx(0,Lat.Neighbours(j))));
//                assert(almostEquals(Lat.Jxxn(j), Lat.Jxx(i,pj)));
//                assert(almostEquals(Lat.Jxyn(j), Lat.Jxy(i,pj)));
//                assert(almostEquals(Lat.Jyyn(j), Lat.Jyy(i,pj)));
//                assert(almostEquals(Lat.Jxxn(j), Lat.Jxx(pj,i)));
//                assert(almostEquals(Lat.Jxyn(j), Lat.Jxy(pj,i)));
//                assert(almostEquals(Lat.Jyyn(j), Lat.Jyy(pj,i)));
                assert(pj != i);
//                HnewX(i) += Lat.Jxxn(j) * Ux(pj) + Lat.Jxyn(j) * Uy(pj);
//                HnewY(i) += Lat.Jxyn(j) * Ux(pj) + Lat.Jyyn(j) * Uy(pj);
                assert(pj >=0 && pj < N);
                assert(j >=0 && j < Lat.Jxxn.size());
                assert(i >=0 && i < Uy.size());
                HnewX(pj) += Lat.Jxxn(j) * Ux(i) + Lat.Jxyn(j) * Uy(i);
                HnewY(pj) += Lat.Jxyn(j) * Ux(i) + Lat.Jyyn(j) * Uy(i);

//                    HnewX(j) += Jxx(i,j) * Ux(i) + Jxy(i,j) * Uy (i);
//                    HnewY(j) += Jxy(i,j) * Ux(i) + Jyy(i,j) * Uy (i);

            }
        }
    } else if (Prop.isMacIsaacMethod){
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
//
//                        HnewX(pnum2) += Lat.Wxx(Fn + k) * Ux(i) + Lat.Wxy(Fn + k) * Uy(i);
//                        HnewY(pnum2) += Lat.Wyy(Fn + k) * Uy(i) + Lat.Wxy(Fn + k) * Ux(i);

//                        Hx(k + j1) += Lat.Wxx(Fn + k) * dMx + Lat.Wxy(Fn + k) * dMy;
//                        Hy(k + j1) += Lat.Wyy(Fn + k) * dMy + Lat.Wxy(Fn + k) * dMx;
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

    if (Prop.isHavingExchangeInteraction && !Prop.isExchangeInteractionCombinedWithDij
        && (Prop.isMacIsaacMethod || Prop.isLRONMethod)){
//        cout << "calcAllFields Exchange" <<endl;
        for (int i = 0; i < N; ++i) {
            int xi = i % Lat.L1;
            int yi = i / Lat.L1;
            const int L1 = Lat.L1;
            const int L2 = Lat.L2;
//            if (i == N/2+L1/2) {
//                cout << "i = " << i <<endl;
//            }
            for (int j = 0; j < Lat.NearestNeighbours.size(); ++j) {
//                cout << "Lat.Neighbours.size() " << Lat.Neighbours.size() << endl;
//                cout << "Lat.Jxxn.size() " << Lat.Jxxn.size() << endl;
//                assert(Lat.Neighbours.size() == Lat.Jxxn.size());
                int xj0 = Lat.NearestNeighbours(j) % L1;
                int yj0 = Lat.NearestNeighbours(j) / L1;
                int xj = (xj0 + xi) % L1;
                int yj = (yj0 + yi) % L2;

//                int xj2 = ((Lat.Xn0(j) + xi) >= L1)? (Lat.Xn0(j) + xi) - L1 : (Lat.Xn0(j) + xi);
//                int yj2 = ((Lat.Yn0(j) + yi) >= L2)? (Lat.Yn0(j) + yi) - L2 : (Lat.Yn0(j) + yi);
//                cout << "xj = " << xj << endl;
//                cout << "xj2 = " << xj2 << endl;
//                assert(xj == xj2);
//                assert(yj == yj2);
                int pj = yj * L1 + xj;
                double minDisi = Lat.getMinDistance(i, pj);
//                int n = Prop.NthNeighbour;
//                if (i == N/2+L1/2) {
//                    cout << "pj = " << pj <<endl;
//                }
//                assert(minDisi < Lat.DistanceVec1p(n) || almostEquals(minDisi, Lat.DistanceVec1p(n)));
                assert(almostEquals(Lat.getMinDistance(0,Lat.NearestNeighbours(j)), Lat.getMinDistance(i, pj)));
                assert(almostEquals(Lat.getMinDistance(Lat.NearestNeighbours(j),0), Lat.getMinDistance(pj, i)));
                assert(almostEquals(Lat.getMinDistanceVec(0,Lat.NearestNeighbours(j)), Lat.getMinDistanceVec(i, pj)));
                assert(almostEquals(Lat.getMinDistanceVec(Lat.NearestNeighbours(j), 0), Lat.getMinDistanceVec(pj, i)));
//                cout << "after assert" <<endl;
//                cout << "Lat.getMinDistanceVec(pj, i) = " << Lat.getMinDistanceVec(pj, i).transpose() <<endl;
//                cout << "Lat.getMinDistanceVec(i, pj) = " << Lat.getMinDistanceVec(i, pj).transpose() <<endl;
//                assert(almostEquals(Lat.getMinDistanceVec(pj, i), -1 * Lat.getMinDistanceVec(i, pj)));
//                assert(almostEquals(Lat.Jxxn(j), Lat.Jxx1p(Lat.Neighbours(j))));
//                assert(almostEquals(Lat.Jxxn(j), Lat.Jxx(0,Lat.Neighbours(j))));
//                assert(almostEquals(Lat.Jxxn(j), Lat.Jxx(i,pj)));
//                assert(almostEquals(Lat.Jxyn(j), Lat.Jxy(i,pj)));
//                assert(almostEquals(Lat.Jyyn(j), Lat.Jyy(i,pj)));
//                assert(almostEquals(Lat.Jxxn(j), Lat.Jxx(pj,i)));
//                assert(almostEquals(Lat.Jxyn(j), Lat.Jxy(pj,i)));
//                assert(almostEquals(Lat.Jyyn(j), Lat.Jyy(pj,i)));
                assert(pj != i);
//                HnewX(i) += Lat.Jxxn(j) * Ux(pj) + Lat.Jxyn(j) * Uy(pj);
//                HnewY(i) += Lat.Jxyn(j) * Ux(pj) + Lat.Jyyn(j) * Uy(pj);
                assert(pj >=0 && pj < N);
//                assert(j >=0 && j < Lat.Jxxn.size());
                assert(i >=0 && i < Uy.size());
                HnewX(pj) += - Ux(i) * ExchangeFactor;
                HnewY(pj) += - Uy(i) * ExchangeFactor;

//                    HnewX(j) += Jxx(i,j) * Ux(i) + Jxy(i,j) * Uy (i);
//                    HnewY(j) += Jxy(i,j) * Ux(i) + Jyy(i,j) * Uy (i);

            }
        }
    }
    //// assertion
//    cout << "HnewX = " << HnewX <<endl;
    if (iReCalc >0){
//        cout << "iReCalc " << iReCalc  << endl;
//        cout << "(HnewXY - H).norm() = " << (HnewXY - H).norm() << endl;
//#ifndef NDEBUG
//        cout << "(HnewX - Hx).norm() = " << (HnewX - Hx).norm() << endl;
//        cout << "(HnewY - Hy).norm() = " << (HnewY - Hy).norm() << endl;
//#endif
        assert(almostEquals(HnewX,Hx));
        assert(almostEquals(HnewY,Hy));
    }
    Hx = HnewX;
    Hy = HnewY;
//    cout << "All fields recalculated!"<< endl;
}

double MC2d::getEnergyByFields() {
    double E = 0;
    E += Ux.dot(Hx);
    E += Uy.dot(Hy);

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

double MC2d::getEnergyByFFT() {
    double E = 0;

    int nx = Lat.L1;
    int ny = Lat.L2;
    int Pindex = 0;
    VectorXd UxSuffle = Ux;
    std::shuffle(UxSuffle.begin(), UxSuffle.end(), gen);
    for(unsigned int i=0; i < nx; i++) {
        for (unsigned int j = 0; j < ny; j++) {
            Ux2dreal(i, j) = Ux(Pindex);
            Uy2dreal(i, j) = Uy(Pindex);
            Pindex++;
        }
    }

    ForwardFFT2d->fft0(Ux2dreal,Ux2dfourier);
    ForwardFFT2d->fft0(Uy2dreal,Uy2dfourier);
    int iLast = Ux2dfourier.Nx() -1;
    int jLast = Ux2dfourier.Ny() -1;

    for (int i = 0; i < Ux2dfourier.Nx(); ++i) {
        for (int j = 0; j < Ux2dfourier.Ny(); ++j) {
            assert(iLast-i < Ux2dfourier.Nx());
            assert(jLast-j < Ux2dfourier.Ny());
            double factor = 2;
            if (j == 0 || j == jLast){   // j == jLast is only for cases which Ux2dreal.Ny() is even
                factor = 1;
            }
            E +=(Jxx2dfourier(i, j) * Ux2dfourier(i, j) * conj(Ux2dfourier(i, j)) +
                 Jxy2dfourier(i, j) * Uy2dfourier(i, j) * conj(Ux2dfourier(i, j)) +
                 Jxy2dfourier(i, j) * Ux2dfourier(i, j) * conj(Uy2dfourier(i, j)) +
                 Jyy2dfourier(i, j) * Uy2dfourier(i, j) * conj(Uy2dfourier(i, j))).real() * factor;
        }
    }
    E /= (N);
    return E;
}



double MC2d::getEnergy() {
    if (Prop.isClockMethod || Prop.isSCOMethod || Prop.isTomitaMethod || Prop.isNeighboursMethod){
        assert(almostEquals(getEnergyByFFT(), getEnergyDirect()));
        return getEnergyByFFT();
    } else if (Prop.isClockMethod ){
        assert(almostEquals(getEnergyByFields(), getEnergyByFFT()));
        assert(almostEquals(getEnergyByFields(), getEnergyDirect()));
        return getEnergyByFields();
    } else {
        return getEnergyByFields();
    }
}

void MC2d::calcOrderParam(){
    if (Prop.LatticeType != 'h'){
        MVectConfig(0) = Ux.dot(OPmakerX);
        MVectConfig(1) = Uy.dot(OPmakerY);

    } else {
        MVectConfig(0) = Ux.dot(OPmakerXX) + Uy.dot(OPmakerXY);
        MVectConfig(1) = Ux.dot(OPmakerYX) + Uy.dot(OPmakerYY);
    }

    if (Prop.LatticeType != 't'){
        MFEVectConfig.setZero();
        for (int i = 0; i < N; ++i) {
            MFEVectConfig(0) += Ux(i);
            MFEVectConfig(1) += Uy(i);
        }
    }

    MTVectConfig.setZero();
    for (int i = 0; i < N; ++i) {
        MTVectConfig(0) += UTx(i);
        MTVectConfig(1) += UTy(i);
    }

    if (Prop.isHavingExchangeInteraction && Prop.LatticeType != 't') {
        MVectConfig = MFEVectConfig;
    }
}

void MC2d::setRNGseed(int32_t seednum) {
    gen.seed(seednum);
}


void MC2d::generateOPmaker() {
    if (Prop.LatticeType == 's') {
        for (int i = 0; i < N; ++i) {
            switch (Plabel(i)) {
                case 1:
                    OPmakerX(i) = 1;
                    OPmakerY(i) = 1;
                    break;
                case 2:
                    OPmakerX(i) = -1;
                    OPmakerY(i) = 1;
                    break;
                case 3:
                    OPmakerX(i)  = -1;
                    OPmakerY(i)  = -1;
                    break;
                case 4:
                    OPmakerX(i)  = 1;
                    OPmakerY(i)  = -1;
                    break;
            }
        }
    } else if (Prop.LatticeType == 't') {
        for (int i = 0; i < N; ++i) {
            OPmakerX(i) = 1;
            OPmakerY(i) = 1;
        }
    } else if (Prop.LatticeType == 'h'){
        double pi = acos(-1);
        Matrix2d Rp, Rm, Rm2, D, T1, T2, T3, T4, T5, T6;
        Rp << cos(2*pi/3),  -sin(2*pi/3),  // rotation by +2*pi/3
              sin(2*pi/3),   cos(2*pi/3);

        Rm << cos(2*pi/3),   sin(2*pi/3),   // rotation by -2*pi/3
              -sin(2*pi/3),  cos(2*pi/3);

        Rm2 << cos(pi/3),   sin(pi/3),   // rotation by -pi/3
                -sin(pi/3),  cos(pi/3);

        D << 1,   0,            // theta -> -theta
             0,  -1;

//        T1 = Rm;
//        T2 = D * Rm;
//        T3.setIdentity();
//        T4 = D;
//        T5 = Rp;
//        T6 = D * Rp;

        T1 = Rm;
        T2 = D * Rm * Rm2;
        T3.setIdentity();
        T4 = D * Rm2;
        T5 = Rp;
        T6 = D * Rp * Rm2;
        assert(almostEquals(abs(T1.determinant()), 1));
        assert(almostEquals(abs(T2.determinant()), 1));
        assert(almostEquals(abs(T3.determinant()), 1));
        assert(almostEquals(abs(T4.determinant()), 1));
        assert(almostEquals(abs(T5.determinant()), 1));
        assert(almostEquals(abs(T6.determinant()), 1));
        vector<Matrix2d> T;
        T.push_back(T1);
        T.push_back(T2);
        T.push_back(T3);
        T.push_back(T4);
        T.push_back(T5);
        T.push_back(T6);

        for (int i = 0; i < N; ++i) {
            int j = Plabel(i) - 1;
            assert(j >=0 && j<=5);
            assert(Plabel(i) >= 1 && Plabel(i)<=6);
            OPmakerXX(i) = T[j](0, 0);
            OPmakerXY(i) = T[j](0, 1);
            OPmakerYX(i) = T[j](1, 0);
            OPmakerYY(i) = T[j](1, 1);
        }
    } else {
        throw std::invalid_argument("Lattice type must be either s or t or h");
    }
}

void MC2d::setFieldOnSimulationType() {
/**
 *  @brief  set simulation type.
 *  @param  hfield                      fixed field at varying temperature simulation
 *  @param  properties.ControlParamType            varying temperature @c ('T') or varying field @c ('h') simulation
 *  @param  hHat                        orientation of field in varying field simulation. ControlParam will be magnitude of field.
 *  @param  Temperature                fixed Temperature at varying field simulation
*/

//    properties.isFieldOn = true;
//    this->properties.ControlParamType = properties.ControlParamType;

    if (Prop.ControlParamType == 'T'){
        hx = Prop.hfield(0);
        hy = Prop.hfield(1);
        Temperature = Prop.Pstart;
        this->hHat.setZero();   // TODO hHat is used for two things and should be changed
    } else if (Prop.ControlParamType == 'h') {
        this->hHat = Prop.hHat;
        Temperature = Prop.FixedTemperatureInVaryingField;
        hx = hHat(0) * Prop.Pstart;
        hy = hHat(1) * Prop.Pstart;
        UpdatehAll();
    }
}

void MC2d::UpdatehAll() {
    for (int i = 0; i < N; ++i) {
        hxAll(i) = hx;
        hyAll(i) = hy;
    }
}
double MC2d::getAutoCorrelationTime(const VectorXd &mtime){
    assert(!FolderPath.empty());

    VectorXd mtimeNormalized = mtime.array() - mtime.mean();
    unsigned int n=mtimeNormalized.size();

    for(unsigned int i=0; i < freal1dAC.Nx(); i++) {
        freal1dAC[i] = mtimeNormalized[i];
    }
    ForwardFFT1dAC->fft(freal1dAC, Ffourier1dAC);
    for(unsigned int i=0; i < Ffourier1dAC.Nx(); i++) {
        Ffourier1dAC[i] = square(abs(Ffourier1dAC[i]));
    }
    BackwardFFT1dAC->fftNormalized(Ffourier1dAC, freal1dAC);
    VectorXd Xtimefft;
    Xtimefft.setZero(mtimeNormalized.size());
    for(unsigned int i=0; i < freal1dAC.Nx(); i++) {
        Xtimefft[i] = freal1dAC[i];
    }

    double tau = 1;
    for (int i = 0; i < 30; ++i) {
//        cout << "tau(recursion) = " << tau <<endl;
        assert(tau*6 > 0 && tau*6 < Xtimefft.size());
        tau = Xtimefft(seqN(0,6*tau)).sum() / Xtimefft(0);
        if (6*tau >= Xtimefft.size()-5){
            break;
        } else if (6*tau  <= 0) {
            tau = 0./0.;
//            tau = std::numeric_limits<double>::quiet_NaN();
            assert(tau != tau);
            break;
        }
    }
    tau += 0.5;

//    if (Prop.isOverRelaxationMethod){
//        tau *= (Prop.OverrelaxationSteps + Prop.MetropolisSteps);
//    } else {
//        tau *= Prop.MetropolisSteps;
//    }

    tau *= Prop.TotalSteps;
    return tau;

//   OLDER METHOD ----------------------------------
//    long Size = mtime.size();
//    auto tmax = (Size-1L)/5L;
//    VectorXd Xtime(tmax), XtimeDiff(tmax);
//    Xtime.setZero();
//    XtimeDiff.setZero();
//    for (int i = 0; i < Xtime.size(); ++i) {
//        Xtime(i) = calcXt(mtime, i);
//        XtimeDiff(i) = calcXtDiff(mtime, i);
//    }
//    VectorXd XtimeNormalized;
//    XtimeNormalized = Xtime / Xtime(0);
//    cout << "tau2 = " << tau2 << endl;
//    double tau = 1;
//    for (int i = 0; i < 30; ++i) {
//        tau = XtimeNormalized(seqN(0,6*tau)).sum();
//        if (6*tau >= XtimeNormalized.size()-5){
//            break;
//        }
//    }
//    SaveXtime << Xtime << endl;
//    SaveXtimeNormalized << XtimeNormalized<< endl;
//    ofstream SaveMtime(FolderPath + "/mtime.txt"s,ofstream::out | ofstream::app);
//    ofstream SaveMtimeNormalized(FolderPath + "/mtimeNormalized.txt"s, ofstream::out | ofstream::app);
//    SaveMtime << mtime.head(max(10000L, mtime.size()-1)).transpose() << endl;
//    SaveMtimeNormalized << (mtime.head(max(10000L, mtime.size()-1)).array() - mtime.mean()) << endl;

//    ofstream SaveXtimeDiff(FolderPath + "/XtimeDiff.txt"s,ofstream::out | ofstream::app);
//    SaveXtimeDiff << XtimeDiff << endl;
//    cout << "Temperature = "  << Temperature << endl
//         << "Xtime : " << setprecision(3) << (Xtime.head(50).transpose()) <<endl<< endl;
//   END OF OLDER METHOD -----------------------------
}






void MC2d::setParentFolderPath(const string& ParentFolderPath1) {
    this->ParentFolderPath = ParentFolderPath1;
    FolderPath = ParentFolderPath1 + "/"s + to_string(InstanceIndex);

    if (Prop.isSaveSamplesToSepareteFolders) {
        fs::create_directories(FolderPath);
    }
//    SaveXtime.open(FolderPath + "/Xtime.txt"s);
//    SaveXtimeNormalized.open(FolderPath + "/XtimeNormalized.txt"s);

}

void MC2d::takeSnapshot() {
    assert(!FolderPath.empty());
    if (InstanceIndex == 0){
        takeSnapshotXY();
    }
}

void MC2d::takeSnapshotXY(){
    rassert(Prop.isSaveSamplesToSepareteFolders);
    ofstream Snapshot(FolderPath + "/snapshot.txt"s, ofstream::out | ofstream::app);
    Snapshot << CtrlParamAbs << endl;
    Snapshot << Ux.transpose() <<endl;
    Snapshot << Uy.transpose() <<endl;

    ofstream SnapshotTransformed(FolderPath + "/SnapshotTransformed.txt"s, ofstream::out | ofstream::app);
    SnapshotTransformed << CtrlParamAbs << endl;
    calcUT();
    SnapshotTransformed << UTx.transpose() <<endl;
    SnapshotTransformed << UTy.transpose() <<endl;
}

void MC2d::saveLocations() {
    rassert(Prop.isSaveSamplesToSepareteFolders);
    ofstream Locations(FolderPath + "/locations.txt"s);
    ofstream Distances(FolderPath + "/Distances.txt"s);
    Locations << Lat.D;
}



void MC2d::UpdateGVec() {
    if (Prop.isMacIsaacMethod){
        UpdateGVecMacIsaac();
    } else {
        UpdateGVecRegular();
    }

}

void MC2d::UpdateGVecRegular() {
    if (Prop.LatticeType == 't') {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                GVec(Lat.MapR(i, j)) += Ux(i) * Ux(j) + Uy(i) * Uy(j);
            }
        }
    } else if (Prop.LatticeType == 's'){
        calcUT();
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                GVec(Lat.MapR(i,j)) += UTx(i) * UTx(j) + UTy(i) * UTy(j);
            }
        }
    } else if (Prop.LatticeType == 'h') {
        calcUT();
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                GVec(Lat.MapR(i,j)) += UTx(i) * UTx(j) + UTy(i) * UTy(j);
            }
        }
    }
}

void MC2d::UpdateGVecMacIsaac() {
    if (Prop.LatticeType == 't') {
        const int L1 = Lat.L1;
        const int L2 = Lat.L2;
        for (int rowspin = 0; rowspin < L2; ++rowspin) {
            for (int colspin = 0; colspin < L1; ++colspin) {
                int pnum = rowspin*L1 + colspin;
                for (int j = 0; j < L2; ++j) {
                    int Fn = 2*L1*(L2+j-rowspin)-colspin+L1;
                    int j1 = L1*j;
                    for (int k = 0; k < L1; ++k) {
                        assert(Fn+k < 4*N);
                        assert(pnum < N);
                        assert(j1+k < N);
                        assert(Lat.MapR0(Fn+k) < GVec.size());
                        GVec(Lat.MapR0(Fn+k)) += Ux(pnum) * Ux(j1+k) + Uy(pnum) * Uy(j1+k);
                    }
                }
            }
        }
    } else if (Prop.LatticeType == 's'){
        calcUT();
        const int L1 = Lat.L1;
        const int L2 = Lat.L2;
        for (int rowspin = 0; rowspin < L2; ++rowspin) {
            for (int colspin = 0; colspin < L1; ++colspin) {
                int pnum = rowspin*L1 + colspin;
                for (int j = 0; j < L2; ++j) {
                    int Fn = 2*L1*(L2+j-rowspin)-colspin+L1;
                    int j1 = L1*j;
                    for (int k = 0; k < L1; ++k) {
                        assert(Fn+k < 4*N);
                        assert(pnum < N);
                        assert(j1+k < N);
                        assert(Lat.MapR0(Fn+k) < GVec.size());
                        GVec(Lat.MapR0(Fn+k)) += UTx(pnum) * UTx(j1+k) + UTy(pnum) * UTy(j1+k);
                    }
                }
            }
        }
    } else if (Prop.LatticeType == 'h') {
        calcUT();
        const int L1 = Lat.L1;
        const int L2 = Lat.L2;
        // for sublatice 0
        for (int rowspin = 0; rowspin < L2; ++rowspin) {
            for (int colspin = 0; colspin < L1; ++colspin) {
                int pnum = rowspin*(2*L1) + 2*colspin;
                assert(Lat.PSubLattice(pnum) == 0);
                for (int j = 0; j < L2; ++j) {
                    int Fn = 2*(L1)*(L2+j-rowspin)-colspin+(L1);
                    int j1 = (2*L1)*j;
                    for (int k = 0; k < (L1); ++k) {
                        int pnum2 = j1 + 2*k;
                        assert(j1 + 2*k < N);
                        assert(j1 + 2*k+1 < N);
                        assert(Fn+k < 4*Lat.Nbase);
                        assert(pnum < N);
                        GVec(Lat.MapR0Sub11(Fn+k)) += UTx(pnum) * UTx(j1 + 2*k)   + UTy(pnum) * UTy(j1 + 2*k);
                        GVec(Lat.MapR0Sub12(Fn+k)) += UTx(pnum) * UTx(j1 + 2*k+1) + UTy(pnum) * UTy(j1 + 2*k+1);
                    }
                }
            }
        }
        // for sublatice 1
        for (int rowspin = 0; rowspin < L2; ++rowspin) {
            for (int colspin = 0; colspin < L1; ++colspin) {
                int pnum = rowspin*(2*L1) + 2*colspin+1;
                assert(Lat.PSubLattice(pnum) == 1);
                for (int j = 0; j < L2; ++j) {
                    int Fn = 2*(L1)*(L2+j-rowspin)-colspin+(L1);
                    int j1 = (2*L1)*j;
                    for (int k = 0; k < (L1); ++k) {
                        int pnum2 = j1 + 2*k;
                        assert(j1 + 2*k < N);
                        assert(j1 + 2*k+1 < N);
                        assert(Fn+k < 4*Lat.Nbase);
                        assert(pnum < N);
                        GVec(Lat.MapR0Sub21(Fn+k)) += UTx(pnum) * UTx(j1 + 2*k)   + UTy(pnum) * UTy(j1 + 2*k);
                        GVec(Lat.MapR0Sub22(Fn+k)) += UTx(pnum) * UTx(j1 + 2*k+1) + UTy(pnum) * UTy(j1 + 2*k+1);
                    }
                }
            }
        }
    }
}



void MC2d::calcGVec() {
    rassert(Prop.isSaveSamplesToSepareteFolders);
    GConVec = GVec;

    GVec.array() /= Prop.NData;
    GVec.array() /= Lat.CounterR.array().cast<double>();
    ofstream GVecOut(FolderPath + "/GVec.txt"s, ofstream::out | ofstream::app);
    GVecOut << GVec.transpose() << endl;
    GVecVsTemperature.row(LastTempIndex) = GVec.transpose() ;

    GConVec.array() /= Prop.NData;
    GConVec.array() /= Lat.CounterR.array().cast<double>();
    if(Prop.LatticeType != 's'){
        GConVec.array() -=  mpp*mpp;
    } else {
        GConVec.array() -=  mPlanepp*mPlanepp;
    }
    ofstream GConVecOut(FolderPath + "/GConVec.txt"s, ofstream::out | ofstream::app);
    GConVecOut << GConVec.transpose() << endl;
    GConVecVsTemperature.row(LastTempIndex) = GConVec.transpose();
}

void MC2d::calcUT() {
    if (Prop.LatticeType == 's'){
        for (int i = 0; i < N; ++i) {
            UTx(i) = OPmakerX(i) * Ux(i);
            UTy(i) = OPmakerY(i) * Uy(i);
            assert(almostEquals(sqrt(square(UTx(i)) + square(UTy(i))), 1));
        }
    } else if (Prop.LatticeType == 'h'){
        for (int i = 0; i < N; ++i) {
            UTx(i) = OPmakerXX(i) * Ux(i) + OPmakerXY(i) * Uy(i);
            UTy(i) = OPmakerYX(i) * Ux(i) + OPmakerYY(i) * Uy(i);
            assert(almostEquals(sqrt(square(UTx(i)) + square(UTy(i))), 1));
        }
    }
    ////assertion
}



void MC2d::SaveTimeSeries() {
//    ofstream EtimeSeriesOut(FolderPath + "/EtimeSeries.csv"s, ofstream::out);
//    ofstream mtimeSeriesOut(FolderPath + "/mtimeSeries.csv"s, ofstream::out);
//    EtimeSeriesOut << EtimeSeries.format(CSVFormat);
//    mtimeSeriesOut << mtimeSeries.format(CSVFormat);



    #pragma omp critical (Hdf5DumpFiles)
    {//TODO it seems to become too slow. change feom vecor to eigen vector  to test its problem
        if (Prop.isSaveSamplesToSepareteFolders) {
            H5Easy::File timeSeriesHdf5Ens(FolderPath + "/EnsembleResults.h5"s,
                                           H5Easy::File::OpenOrCreate);
            H5Easy::dump(timeSeriesHdf5Ens, "/TimeSeries/Energy"s, EtimeSeries,
                         H5Easy::DumpOptions(H5Easy::Compression(9)));
            H5Easy::dump(timeSeriesHdf5Ens, "/TimeSeries/Magnetization"s, mtimeSeries,
                         H5Easy::DumpOptions(H5Easy::Compression(9)));
        } else {
            H5Easy::File timeSeriesHdf5Ens(ParentFolderPath + "/Results.h5"s,
                                           H5Easy::File::OpenOrCreate);
            H5Easy::dump(timeSeriesHdf5Ens, ("/" + to_string(InstanceIndex) + "/TimeSeries/Energy"s), EtimeSeries,
                         H5Easy::DumpOptions(H5Easy::Compression(9)));
            H5Easy::dump(timeSeriesHdf5Ens, ("/" + to_string(InstanceIndex) + "/TimeSeries/Magnetization"s), mtimeSeries,
                         H5Easy::DumpOptions(H5Easy::Compression(9)));
        }


//        H5Easy::dump(timeSeriesHdf5Ens, "/TimeSeries/Energy2step"s, EtimeSeries2step,
//                     H5Easy::DumpOptions(H5Easy::Compression(9)));
//        H5Easy::dump(timeSeriesHdf5Ens, "/TimeSeries/Magnetization2step"s, mtimeSeries2step,
//                     H5Easy::DumpOptions(H5Easy::Compression(9)));
//        H5Easy::File timeSeriesHdf5(ParentFolderPath + "/results.h5"s,
//                                    H5Easy::File::OpenOrCreate);
//        H5Easy::dump(timeSeriesHdf5, "/timeSeries/" + to_string(InstanceIndex) + "/EtimeSeries"s, EtimeSeries,
//                     H5Easy::DumpOptions(H5Easy::Compression(9)));
//        H5Easy::dump(timeSeriesHdf5, "/timeSeries/" + to_string(InstanceIndex) + "/mtimeSeries"s, mtimeSeries,
//                     H5Easy::DumpOptions(H5Easy::Compression(9)));
    }
    EtimeSeries.clear();
    EtimeSeries.shrink_to_fit();
    mtimeSeries.clear();
    mtimeSeries.shrink_to_fit();
}



void MC2d::setProbClock() {
    ProbClock.setZero();  //TODO rename ProbClock to ProbMax
    InVLogProbClock.setZero();
    InVOneMinusProbClock.setZero();
    dVstarVec.setZero();
    double ratio = 1;
    double lambdaTot = 0;

//    if (Prop.isHavingExchangeInteraction){
//        ratio = Prop.DipolarStrengthToExchangeRatio;
//    }
    lambdaVec.clear();
    for (int j = 0; j < ProbClock.size(); ++j) {
        double r0j = Lat.R0jBydistance(j);
        assert(r0j > 0);        // TODO set max prob using J interactions eigenvalues
        double lambdaOld = 8.6 * ratio/(Temperature * pow(r0j, 3));

        double lambdaNew;
        if (Prop.isWalkerAliasMethod){
            lambdaNew = 4*ratio*Lat.JmaxByDistance(j) /Temperature; //TODO investigate more about this difference between Walker and Bernoulli
        } else {
            lambdaNew = 4*ratio*Lat.JmaxByDistance(j) /Temperature; //TODO ALERT: test more for D/J-0.1 to check its working properly
        }
        double lambda = lambdaNew;
//        lambda *= (Prop.isSCOMethod)? Prop.SCOMethodJmaxIncreaseFactor : 1;
        ProbClock(j) = exp(- lambda);  //exp(-2*beta*2/r^3)
        InVLogProbClock(j) = 1 / (- lambda);  //exp(-2*beta*2/r^3) //TODO log(exp(x)) = x
        InVOneMinusProbClock(j) = 1 / (-expm1(-lambda));  //exp(-2*beta*2/r^3)
        lambdaVec.push_back(lambda);
        assert(ProbClock(j) > 0);
        dVstarVec(j) = lambda / 2;
    }
    if (InstanceIndex == 0) {
        cout << "ProbClock = " << endl << ProbClock(seq(0, min(20L,  ProbClock.size() -1)))  << endl;
        cout << "ProbClock.prod() = " << ProbClock.prod() << endl;
    }
//    cout << "p(far0) = " << ProbClock(Lat.Neighbours.size()) << endl;



    if (Prop.isNearGroup || Prop.isNearGroup) {
        lambdaVec.erase(lambdaVec.begin(), lambdaVec.begin() + Lat.Neighbours.size());
    }
    WalkerSampler.set(lambdaVec);
    lambdaTot = std::accumulate(lambdaVec.begin(), lambdaVec.end(),
                                decltype(lambdaVec)::value_type(0));
    PoissonDis = std::poisson_distribution<int>(lambdaTot); //TODO make this work for near neighbours as well
    cout << "lambdaTot = " << lambdaTot  <<endl;
    int NumResampledii = 0;
    int Nloop = 300;
    for (int i = 0; i < Nloop; ++i) {
        int jrej = (Prop.isNearGroup || Prop.isNearGroup) ? Lat.Neighbours.size() : 0;
        double nu;
        double invOneMinusPHatOld = InVOneMinusProbClock(jrej);
        while(true){
            if (jrej > Lat.PByDistance.size() - 1){
                break;
            }
            NumResampledii++;
            nu = realDis(gen);
            const int IntMax = std::numeric_limits<int>::max();
            double jrejTemp = jrej + (1 + log(nu) * InVLogProbClock(jrej));
            jrej = (jrejTemp<= IntMax)? jrejTemp : IntMax;
        }
    }
    cout << "lambda bernoulli = " << NumResampledii/(Nloop + 0.) << endl;
}



void MC2d::setProbClockBoxes() {
    ProbClock.setZero();  //TODO rename ProbClock to ProbMax
    InVLogProbClock.setZero();
    InVOneMinusProbClock.setZero();
    double ratio = 1;
    double lambdaTot = 0;

//    if (Prop.isHavingExchangeInteraction){
//        ratio = Prop.DipolarStrengthToExchangeRatio;
//    }
    VectorXi Indices;
    Lat.getIndicesInsideBox(63);
    if (InstanceIndex == 0) {
        cout << "Indices = " << endl << Lat.BoxIndices(seq(0, min(20L,  Lat.BoxIndices.size() -1 )))  <<endl;
    }
    for (int j = 0; j < Lat.PByDistance.size(); ++j) {
        Lat.getIndicesInsideBox(Lat.PByDistance(j));
        ProbClock(j) = 1;
        for (int ii = 0; ii < Lat.BoxIndices.size(); ++ii) {
            int PinBoxidx = Lat.BoxIndices(ii);
            double lambda = 4*ratio*Lat.Jmax1p(PinBoxidx) /Temperature;
            double ProbClockPraticle = exp(-lambda);
            ProbClock(j) *= ProbClockPraticle;
        }
        assert(ProbClock(j) > 0);

//        InVLogProbClock(j) = 1 / log(ProbClock(j));  //exp(-2*beta*2/r^3)
//        InVOneMinusProbClock(j) = 1 / (1 - ProbClock(j));  //exp(-2*beta*2/r^3)

//        cout << "R0jBydistance(j) = " << endl << Lat.R0jBydistance(j) <<endl;
//        cout << "ProbClock(j) = " << endl << ProbClock(j) <<endl;
    }

    // sorting by ProbClock
    for (int i = 0; i < ProbClock.size(); ++i) {
        for (int j = i; j < ProbClock.size(); ++j) {
            if (ProbClock(j) < ProbClock(i)) {
                swap(Lat.R0jBydistance(i), Lat.R0jBydistance(j));
                swap(Lat.PByDistance(i),   Lat.PByDistance(j));
                swap(Lat.XPByDistance(i),  Lat.XPByDistance(j));
                swap(Lat.YPByDistance(i),  Lat.YPByDistance(j));
                swap(ProbClock(i), ProbClock(j));
            }
        }
    }

    for (int j = 0; j < ProbClock.size(); ++j) {
        InVLogProbClock(j) = 1 / log(ProbClock(j));  //exp(-2*beta*2/r^3)
        InVOneMinusProbClock(j) = 1 / (1 - ProbClock(j));  //exp(-2*beta*2/r^3)
    }

    if (Prop.isMergeNearBoxes){
        Lat.getIndicesInsideBox(0);
        for (int ii = 0; ii < Lat.BoxIndices.size(); ++ii) {
            int PinBoxidx = Lat.BoxIndices(ii);
            if (PinBoxidx == 0) {
                continue;
            }
            NearIndices.push_back(PinBoxidx);
        }

        int LastMergeIndex = Prop.NumberOfMerges;
        for (int j = 0; j < LastMergeIndex ;++j) {
            Lat.getIndicesInsideBox(Lat.PByDistance(j));
            for (int ii = 0; ii < Lat.BoxIndices.size(); ++ii) {
                int PinBoxidx = Lat.BoxIndices(ii);
                if (PinBoxidx == 0) {
                    continue;
                }
                NearIndices.push_back(PinBoxidx);
            }
        }
        Lat.R0jBydistance   = (Lat.R0jBydistance(seq(LastMergeIndex, last))).eval();
        Lat.PByDistance     = (Lat.PByDistance(seq(LastMergeIndex, last))).eval();
        Lat.XPByDistance    = (Lat.XPByDistance(seq(LastMergeIndex, last))).eval();
        Lat.YPByDistance    = (Lat.YPByDistance(seq(LastMergeIndex, last))).eval();
        ProbClock           = (ProbClock(seq(LastMergeIndex, last))).eval();
        InVLogProbClock     = (InVLogProbClock(seq(LastMergeIndex, last))).eval();
        InVOneMinusProbClock= (InVOneMinusProbClock(seq(LastMergeIndex, last))).eval();

    }

    if (InstanceIndex == 0) {
        cout << "ProbClock = " <<endl << ProbClock(seq(0, min(20L, ProbClock.size() -1 )))  <<endl;
        cout << "R0jBydistance = " <<endl << Lat.R0jBydistance(seq(0, min(20L, Lat.R0jBydistance.size()-1))) <<endl;
        cout << "ProbClock.prod() = " << ProbClock.prod() << endl;
    }
}


double MC2d::dEij(int i, int j) {
//    cout << "dEij" <<endl;
    assert(i >=0 && i < N);
    assert(j >=0 && j < N);
    const double dMx = UiNewX - Ux(i);
    const double dMy = UiNewY - Uy(i);
    int jper = Lat.periodicParticle(j, i);
    assert(jper >=0 && jper < N);
    double dE = dMx * (Lat.Jxx1p(jper) * Ux(j) + Lat.Jxy1p(jper) * Uy(j)) +
                dMy * (Lat.Jxy1p(jper) * Ux(j) + Lat.Jyy1p(jper) * Uy(j));
//    double dE1 = dMx * (Lat.Jxx(i, j) * Ux(j) + Lat.Jxy(i, j) * Uy(j)) +
//                 dMy * (Lat.Jxy(i, j) * Ux(j) + Lat.Jyy(i, j) * Uy(j));
//    assert(almostEquals(dE, dE1));
    return 2*dE;
}





void MC2d::run1ClockStep() {
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
                run1ClockParticle(i);
            }
        }
        isOverrelaxing = false;
        for (int j = 0; j < Prop.MetropolisSteps; ++j) {
            for (int i = 0; i < N; ++i) {
                randomizeUiNew();
                run1ClockParticle(i);
            }
        }
    } else {
        isOverrelaxing = false;
        for (int j = 0; j < Prop.MetropolisSteps; ++j) {
            for (int i = 0; i < N; ++i) {
                randomizeUiNew();
                run1ClockParticle(i);
            }
        }
    }
}


double MC2d::dENN(int i) {
    const int xi = Lat.XPbyIndex(i);
    const int yi = Lat.YPbyIndex(i);
    assert(xi == i % Lat.L1);
    assert(yi == i / Lat.L1);
    const int L1 = Lat.L1;
    const int L2 = Lat.L2;
    double Hxn = 0;
    double Hyn = 0;
    for (int j = 0; j < Lat.Neighbours.size(); ++j) {
        const int xj0 = Lat.Xn0(j) + xi;
        const int yj0 = Lat.Yn0(j) + yi;
        const int xj = (xj0 >= L1)? xj0 - L1 : xj0;
        const int yj = (yj0 >= L2)? yj0 - L2 : yj0;

//            int xj = (Lat.Xn0(j) + xi) % L1;
//            int yj = (Lat.Yn0(j) + yi) % L2;
//        int nn = Prop.CalcInteractionsUpToNthNeighbour;
        const int pj = yj * L1 + xj;
//        cout << "Lat.getMinDistance(i,pj) = " << Lat.getMinDistance(i,pj) <<endl;
//            assert(Lat.getMinDistance(i,pj) < Lat.DistanceVec1p(nn) ||
//                    almostEquals(Lat.getMinDistance(i,pj), Lat.DistanceVec1p(nn)));
            assert(almostEquals(Lat.getMinDistance(0,Lat.Neighbours(j)), Lat.getMinDistance(i, pj)));
//            assert(almostEquals(Lat.getMinDistanceVec(0,Lat.Neighbours(j)), Lat.getMinDistanceVec(i, pj)));
//            assert(almostEquals(Lat.Jxxn(j), Lat.Jxx1p(Lat.Neighbours(j))));
//            assert(almostEquals(Lat.Jxxn(j), Lat.Jxx(0,Lat.Neighbours(j))));
//            assert(almostEquals(Lat.Jxxn(j), Lat.Jxx(i,pj)));
        assert(j  >= 0 && j  < Lat.Jxxn.size());
        assert(pj >= 0 && pj < Ux.size());
        assert(i != pj);
        Hxn += Lat.Jxxn(j) * Ux(pj) + Lat.Jxyn(j) * Uy(pj);
        Hyn += Lat.Jxyn(j) * Ux(pj) + Lat.Jyyn(j) * Uy(pj);
    }
    const double dMx = UiNewX - Ux(i);
    const double dMy = UiNewY - Uy(i);
    const double sMx = UiNewX + Ux(i);
    const double sMy = UiNewY + Uy(i);
    double dE = 2 * (dMx* Hxn + dMy * Hyn);
    dE += (dMx * (Lat.Jself(0,0) * sMx + Lat.Jself(1,0) * sMy)) +
            (dMy * (Lat.Jself(1,0) * sMx + Lat.Jself(1,1) * sMy));
//    cout << "dEnn = " << dE << endl;
    return dE;
}
void MC2d::overRelaxateUiNew(int i){
    overRelaxateUiNew(i,  Hx(i),  Hy(i));
    assert(almostEquals(dErot(i),0));
}
void MC2d::overRelaxateUiNew(int i, double Hxi, double Hyi){
    double m = Hyi / Hxi;
    double m2 = m*m;
    double div1pm2 = 1/(1+m2);
    UiNewX = div1pm2*((1-m2)*Ux(i) +    2*m*Uy(i));
    UiNewY = div1pm2*(   2*m*Ux(i) - (1-m2)*Uy(i));

//        double abstheta_diff = abs(theta_inew - theta_i);
//        double theta_diff = abstheta_diff;
//        while(theta_diff > 2 * PI){
//            theta_diff -= 2* PI;
//        }
//        if (theta_diff < PI * 0.1){
//            continue;
//        }

//        UiNewX2 = cos(theta_inew);
//        UiNewY2 = sin(theta_inew);
//        assert(almostEquals(UiNewX,UiNewX2));
//        assert(almostEquals(UiNewY,UiNewY2));
//        cout << "theta_inew - theta_H = " << theta_inew - theta_H << endl;
//        cout << "theta_i - theta_H = " << theta_i - theta_H << endl;
//        cout << "dErot(i) = " << dErot(i) << endl;

//    assert(almostEquals(dErot(i),0));
}



double MC2d::PRelOriginal(int jrej, int i, int xi, int yi, double invOneMinusPHatOld) {  // jrej is sorted by distance and is periodic
    complexitySum += 1;
    complexityLocalSum += 1;
    assert(Lat.XPByDistance(jrej) + Lat.YPByDistance(jrej) * Lat.L1 == Lat.PByDistance(jrej));
    const int xjper = Lat.XPByDistance(jrej);  //jper is periodic
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
    if (dE <= 0.){
        return 0.;
    }

    assert(almostEquals(InvTemperature, 1/Temperature));
    double OneMinusPj = - expm1(- dE * InvTemperature);
    assert(1 - ProbClock(jrej) >= OneMinusPj);

    double prel = OneMinusPj * invOneMinusPHatOld;
    return prel;
}

void MC2d::run1ClockParticleOriginal(int i) {
    if (Prop.isCalcSelectedBondsProbability){
        SelectedBondsLocal.setZero();
    }
    int jrej = 0;     //jrej is sorted by distance and is periodic
    Debug_totali++;
    complexityLocalSum = 0;
    RNDGenerationLocalSum = 0;
    TotalRegularParticleSteps     += (!isOverrelaxing)? 1 : 0;
    TotalOverrelaxedParticleSteps += ( isOverrelaxing)? 1 : 0;
    if (Prop.isNearGroup){
        assert(almostEquals(dErot(i),dENN(i)));
         double dE = dErot(i);
        RNDGenerationSum += 1;
        RNDGenerationLocalSum += 1;
         if (!isOverrelaxing && realDis(gen) > exp(-dE * InvTemperature)){
             return;   // reject move
         }
         jrej = Lat.Neighbours.size();
    }
    const int xi = Lat.XPbyIndex(i);
    const int yi = Lat.YPbyIndex(i);
    assert(xi == i % Lat.L1);
    assert(yi == i / Lat.L1);
    RNDGenerationLocalSum += 1;
    RNDGenerationSum += 1;
    double nu = realDis(gen);
    double invOneMinusPHatOld = InVOneMinusProbClock(jrej);
    assert(int(floor(log(nu)/log(ProbClock(jrej)))) == int(log(nu)/log(ProbClock(jrej))));
    assert(almostEquals(InVLogProbClock(jrej), 1 / log(ProbClock(jrej))));
    const int IntMax = std::numeric_limits<int>::max();
    double jrejTemp = jrej + (log(nu) * InVLogProbClock(jrej));
    jrej = (jrejTemp<= IntMax)? jrejTemp : IntMax;
    while(true){
        assert(Lat.PByDistance.size() - 1 == N - 2);
        if (jrej > Lat.PByDistance.size() - 1){ // there is totally N-1 bouds indexed from 0. so the index of the last bond is N-2
            Debug_Iaccepted++;    // move accepted
            AcceptedRegularParticleSteps     += (!isOverrelaxing)? 1 : 0;
            AcceptedOverrelaxedParticleSteps += ( isOverrelaxing)? 1 : 0;
            if (Prop.isNearGroup){
                updateField(i);
            }
            updateState(i);
            if (Prop.isCalcSelectedBondsProbability){
                SelectedBondsSum += SelectedBondsLocal;
                AcceptedBondsSum += SelectedBondsLocal;
            }
            AcceptSum++;
            complexityAcceptSum += complexityLocalSum;
            RNDGenerationAcceptSum += RNDGenerationLocalSum;
            break;
        } else  {
            if (Prop.isCalcSelectedBondsProbability){
                SelectedBondsLocal(jrej)++;
            }
            double prel = PRelOriginal(jrej, i, xi, yi, invOneMinusPHatOld);
            RNDGenerationLocalSum += 1;
            RNDGenerationSum += 1;
            if (prel > 0. && realDis(gen) <= prel){
                JrejSum(jrej)++;
                if (Prop.isCalcSelectedBondsProbability){
                    SelectedBondsSum += SelectedBondsLocal;
                }
                break;   // move rejected
            }
        }
        NumResampled++;
        assert(jrej >=0 && jrej < N-1);
        RNDGenerationLocalSum += 1;
        RNDGenerationSum += 1;
        nu = realDis(gen);
        invOneMinusPHatOld = InVOneMinusProbClock(jrej);
        assert(almostEquals(InVLogProbClock(jrej), 1 / log(ProbClock(jrej))));
        assert(int(floor(log(nu)/log(ProbClock(jrej)))) == int(log(nu)/log(ProbClock(jrej))));
        jrejTemp = jrej + (1 + log(nu) * InVLogProbClock(jrej));
        jrej = (jrejTemp<= IntMax)? jrejTemp : IntMax;
    }
}


double MC2d::PRelBoxes(int jrej, int i, int xi, int yi, double invOneMinusPHatOld) {  // jrej is sorted by distance and is periodic
    assert(Lat.XPByDistance(jrej) + Lat.YPByDistance(jrej) * Lat.L1 == Lat.PByDistance(jrej));
    const int jperBox = Lat.PByDistance(jrej);  //jper is periodic
    assert(jperBox >=0 && jperBox < N);
    Lat.getIndicesInsideBox(jperBox);
    double dELocal = 0;
    for (int ii = 0; ii < Lat.BoxIndices.size(); ++ii) {
        complexitySum += 1;
        int PinBoxidx = Lat.BoxIndices(ii);
        if (PinBoxidx == 0) { // TODO this is an assertion
            cout << "jperBox = " << jperBox <<endl;
            rassert(false);
        }
        const int xjper = PinBoxidx % Lat.L1;  //jper is periodic
        const int yjper = PinBoxidx / Lat.L1;
        const int xj0 = xjper + xi;
        const int yj0 = yjper + yi;
        const int xj = (xj0 >= Lat.L1)? xj0 - Lat.L1 : xj0;
        const int yj = (yj0 >= Lat.L1)? yj0 - Lat.L1 : yj0;
        int j2 = yj * Lat.L1 + xj;
//        int jper =  yjper * Lat.L1 + xjper;
        assert(PinBoxidx ==  yjper * Lat.L1 + xjper);
        int jper =  PinBoxidx;
        const double dMx = UiNewX - Ux(i);
        const double dMy = UiNewY - Uy(i);
        assert(jper >=0 && jper < N);
        dELocal  += 2*(dMx * (Lat.Jxx1p(jper) * Ux(j2) + Lat.Jxy1p(jper) * Uy(j2)) +
                       dMy * (Lat.Jxy1p(jper) * Ux(j2) + Lat.Jyy1p(jper) * Uy(j2)));
    }

    double dE = dELocal;
    if (dE <= 0.){
        return 0.;
    }

    assert(almostEquals(InvTemperature, 1/Temperature));
    double OneMinusPj = - expm1(- dE * InvTemperature);
    assert(1 - ProbClock(jrej) >= OneMinusPj);
//    assert(InVLogProbClock(jrej) >= OneMinusPj);

    double prel = OneMinusPj * invOneMinusPHatOld;
    return prel;
}

void MC2d::run1ClockParticleBoxes(int i) {
    if (Prop.isCalcSelectedBondsProbability){
        SelectedBondsLocal.setZero();
    }
    int jrej = 0;     //jrej is sorted by distance and is periodic
    Debug_totali++;
    TotalRegularParticleSteps     += (!isOverrelaxing)? 1 : 0;
    TotalOverrelaxedParticleSteps += ( isOverrelaxing)? 1 : 0;
//    if (Prop.isNearGroup){
//        assert(almostEquals(dErot(i),dENN(i)));
//        double dE = dErot(i);
//        RNDGenerationSum += 1;
//        if (!isOverrelaxing && realDis(gen) > exp(-dE * InvTemperature)){
//            return;   // reject move
//        }
//        jrej = Lat.Neighbours.size();
//    }
    const int xi = Lat.XPbyIndex(i);
    const int yi = Lat.YPbyIndex(i);
    assert(xi == i % Lat.L1);
    assert(yi == i / Lat.L1);

    double dELocal = 0;

    if (Prop.isMergeNearBoxes){
        for (int PinBoxidx : NearIndices) {
            complexitySum += 1;
            const int xjper = PinBoxidx % Lat.L1;  //jper is periodic
            const int yjper = PinBoxidx / Lat.L1;
            const int xj0 = xjper + xi;
            const int yj0 = yjper + yi;
            const int xj = (xj0 >= Lat.L1)? xj0 - Lat.L1 : xj0;
            const int yj = (yj0 >= Lat.L1)? yj0 - Lat.L1 : yj0;
            int j2 = yj * Lat.L1 + xj;
            int jper =  yjper * Lat.L1 + xjper;
            const double dMx = UiNewX - Ux(i);
            const double dMy = UiNewY - Uy(i);
            assert(jper >=0 && jper < N);
            dELocal  += 2*(dMx * (Lat.Jxx1p(jper) * Ux(j2) + Lat.Jxy1p(jper) * Uy(j2)) +
                           dMy * (Lat.Jxy1p(jper) * Ux(j2) + Lat.Jyy1p(jper) * Uy(j2)));
        }
    } else {
        Lat.getIndicesInsideBox(0);
        for (int ii = 0; ii < Lat.BoxIndices.size(); ++ii) {
            int PinBoxidx = Lat.BoxIndices(ii);
            if (PinBoxidx == 0) {
                continue;
            }
            const int xjper = PinBoxidx % Lat.L1;  //jper is periodic
            const int yjper = PinBoxidx / Lat.L1;
            const int xj0 = xjper + xi;
            const int yj0 = yjper + yi;
            const int xj = (xj0 >= Lat.L1)? xj0 - Lat.L1 : xj0;
            const int yj = (yj0 >= Lat.L1)? yj0 - Lat.L1 : yj0;
            int j2 = yj * Lat.L1 + xj;
            int jper =  yjper * Lat.L1 + xjper;
            const double dMx = UiNewX - Ux(i);
            const double dMy = UiNewY - Uy(i);
            assert(jper >=0 && jper < N);
            dELocal  += 2*(dMx * (Lat.Jxx1p(jper) * Ux(j2) + Lat.Jxy1p(jper) * Uy(j2)) +
                           dMy * (Lat.Jxy1p(jper) * Ux(j2) + Lat.Jyy1p(jper) * Uy(j2)));
        }
    }
    if (realDis(gen) > exp(-dELocal * InvTemperature)){
            return;   // reject move
    }


    double nu = realDis(gen);
    RNDGenerationSum += 1;
    double invOneMinusPHatOld = InVOneMinusProbClock(jrej);
    assert(int(floor(log(nu)/log(ProbClock(jrej)))) == int(log(nu)/log(ProbClock(jrej))));
    assert(almostEquals(InVLogProbClock(jrej), 1 / log(ProbClock(jrej))));
    const int IntMax = std::numeric_limits<int>::max();
    double jrejTemp = jrej + (log(nu) * InVLogProbClock(jrej));
    jrej = (jrejTemp<= IntMax)? jrejTemp : IntMax;
    while(true){
//        assert(Lat.PByDistance.size() - 1 == N - 2);
        if (jrej > Lat.PByDistance.size() - 1){ // there is totally N-1 bouds indexed from 0. so the index of the last bond is N-2
            Debug_Iaccepted++;    // move accepted
            AcceptedRegularParticleSteps     += (!isOverrelaxing)? 1 : 0;
            AcceptedOverrelaxedParticleSteps += ( isOverrelaxing)? 1 : 0;
            if (Prop.isNearGroup){
                updateField(i);
            }
            updateState(i);
            if (Prop.isCalcSelectedBondsProbability){
                SelectedBondsSum += SelectedBondsLocal;
                AcceptedBondsSum += SelectedBondsLocal;
                AcceptSum++;
            }
            break;
        } else  {
            if (Prop.isCalcSelectedBondsProbability){
                SelectedBondsLocal(jrej)++;
            }
            double prel = PRelBoxes(jrej, i, xi, yi, invOneMinusPHatOld);
            RNDGenerationSum += 1;
            if (prel > 0. && realDis(gen) <= prel){
                JrejSum(jrej)++;
                if (Prop.isCalcSelectedBondsProbability){
                    SelectedBondsSum += SelectedBondsLocal;
                }
                break;   // move rejected
            }
        }
        NumResampled++;
        assert(jrej >=0 && jrej < N-1);
        nu = realDis(gen);
        RNDGenerationSum += 1;
        invOneMinusPHatOld = InVOneMinusProbClock(jrej);
        assert(almostEquals(InVLogProbClock(jrej), 1 / log(ProbClock(jrej))));
        assert(int(floor(log(nu)/log(ProbClock(jrej)))) == int(log(nu)/log(ProbClock(jrej))));
        jrejTemp = jrej + (1 + log(nu) * InVLogProbClock(jrej));
        jrej = (jrejTemp<= IntMax)? jrejTemp : IntMax;
    }
}

void MC2d::cleanjrejVecBool(const vector<int> &jrejVecSmall){
    for (int index : jrejVecSmall) {
        jrejVecBool[index] = false;
    }
    assert(jrejVecBool == vector<bool>(N,false));
}

void MC2d::cleanjrejVecRepeated(const vector<int> &jrejVecSmall){
    for (int index : jrejVecSmall) {
        jrejVecRepeated[index] = 0;
    }
    assert(jrejVecRepeated == vector<int>(N,0));
}


void MC2d::run1ClockParticleWalker(int i) {
//    cout << "run1ClockParticle" <<endl;
    int jrej = 0;     //jrej is sorted by distance and is periodic
    int shift = 0;
    Debug_totali++;
    TotalRegularParticleSteps        += (!isOverrelaxing)? 1 : 0;
    TotalOverrelaxedParticleSteps    += ( isOverrelaxing)? 1 : 0;
    complexityLocalSum = 0;
    RNDGenerationLocalSum = 0;
    if (Prop.isNearGroup){
//        cout << "here" << endl;
        assert(almostEquals(dErot(i),dENN(i)));
        double dE = dErot(i);
//         if (dE > 0 && realDis(gen) > exp(-dE * InvTemperature)){
        RNDGenerationLocalSum += 1;
        RNDGenerationSum += 1;
        if (!isOverrelaxing && realDis(gen) > exp(-dE * InvTemperature)){
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
    RNDGenerationLocalSum += 1;
    RNDGenerationSum += 1;
    double RandomPoisson = PoissonDis(gen);
    for (int j = 0; j < RandomPoisson; ++j) {
        RNDGenerationLocalSum += 1;
        RNDGenerationSum += 1;
        jrej = WalkerSampler(gen) + shift;
//        while (jrejVecBool[jrej]){
//            jrej = WalkerSampler(gen) + shift;
//        }

        if (jrejVecBool[jrej]){
            continue;
        }
        jrejVecBool[jrej] = true;
        jrejVecSmall.push_back(jrej);

//        double invOneMinusPhat = 1/(1-ProbClock[jrej]);
        double invOneMinusPhat = InVOneMinusProbClock(jrej);

        double prel = PRelOriginal(jrej, i, xi, yi, invOneMinusPhat);
        RNDGenerationLocalSum += 1;
        RNDGenerationSum += 1;
        if (prel > 0. && realDis(gen) <= prel){
//            cout << "i = " << i << " rejected" <<endl;
//            cout << "jrej = " << jrej  <<endl;
            JrejSum(jrej)++;
//            break;   // move rejected
            cleanjrejVecBool(jrejVecSmall);
            return;   // move rejected
        }
        NumResampled++;
    }

    Debug_Iaccepted++;    // move accepted
    AcceptedRegularParticleSteps     += (!isOverrelaxing)? 1 : 0;
    AcceptedOverrelaxedParticleSteps += ( isOverrelaxing)? 1 : 0;
    AcceptSum++;
    complexityAcceptSum += complexityLocalSum;
    RNDGenerationAcceptSum += RNDGenerationLocalSum;
    if (Prop.isNearGroup){
        updateField(i);
    }
    updateState(i);
    cleanjrejVecBool(jrejVecSmall);
}

void MC2d::run1ClockParticle(int i){
    TotalUpdateSum++;
    if (Prop.isBoxes){
        run1ClockParticleBoxes(i);
    } else if(Prop.isWalkerAliasMethod){
        run1ClockParticleWalker(i);
    } else {
        run1ClockParticleOriginal(i);
    }
}


double MC2d::dESCO(const vector<int> &jrejVec, int i, int xi, int yi,
                    const vector<double> &PjVec){
    //    cout << "PRel" <<endl;
    double dE = 0;
    assert(PjVec.size() == jrejVec.size());
    const double dMx = UiNewX - Ux(i);
    const double dMy = UiNewY - Uy(i);
    double oneMinusPjOldProd = 1;
    double oneMinusPjNewProd = 1;
    for (int k = 0; k < jrejVec.size(); ++k) {
        int jrej = jrejVec[k];
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
//    int jper = Lat.periodicParticle(j, i);
        assert(jper >=0 && jper < N);
        dE += 2*(dMx * (Lat.Jxx1p(jper) * Ux(j2) + Lat.Jxy1p(jper) * Uy(j2)) +
                 dMy * (Lat.Jxy1p(jper) * Ux(j2) + Lat.Jyy1p(jper) * Uy(j2)));
        double Pj = PjVec[k];
        double Enew = 2*(UiNewX * (Lat.Jxx1p(jper) * Ux(j2) + Lat.Jxy1p(jper) * Uy(j2)) +
                         UiNewY* (Lat.Jxy1p(jper) * Ux(j2) + Lat.Jyy1p(jper) * Uy(j2)));
        double dVStar = dVstarVec(jrej);
        double Pjold = Pj;
        double Pjnew = exp(Enew * InvTemperature - dVStar);
        oneMinusPjOldProd *= 1 - Pj;
        oneMinusPjNewProd *= 1 - Pjnew;
//        dE += Temperature * log(1 - Pj) - Temperature * log(1 - Pjnew);
    }
//    cout << "log E / field  E = " << Temperature * log(oneMinusPjOldProd/oneMinusPjNewProd) / dE <<endl;
    dE += Temperature * log(oneMinusPjOldProd/oneMinusPjNewProd);
    return dE;
}


double MC2d::PjSCO(int jrej, int i, int xi, int yi) {  // jrej is sorted by distance and is periodic
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
//    int jper = Lat.periodicParticle(j, i);
    assert(jper >=0 && jper < N);
    double dV = 2*(Ux(i) * (Lat.Jxx1p(jper) * Ux(j2) + Lat.Jxy1p(jper) * Uy(j2)) +
                   Uy(i) * (Lat.Jxy1p(jper) * Ux(j2) + Lat.Jyy1p(jper) * Uy(j2)));
//    assert(almostEquals(dE, dEij(i,j)));

//    double Phatj = InvOneMinusPHatOld;
//    double Pj = exp(-max(dEij(i,j), 0.)/Temperature);

    assert(almostEquals(InvTemperature, 1/Temperature));
    double dVStar = dVstarVec(jrej);
    double Pj = exp(dV * InvTemperature - dVStar);
//    double Pj = exp(-max(dEij(i,j2), 0.) * InvTemperature);

//    double prel = (1 - Pj) * invOneMinusPHatOld;
    return Pj;
}

void MC2d::validiatePjSCONew(int jrej, double invOneMinusPHatOld, SCOData& scoData) {
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
    complexitySum += 1;
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
    double OneMinusPj = - expm1(Eij * InvTemperature - dVStar);   // - expm1 = 1 - e^x
//    double OneMinusPj = 1 - (exp(Eij * InvTemperature) *  exp(- dVStar));   // - expm1 = 1 - e^x
    OneMinusPj = (OneMinusPj < 0.)? 0. : OneMinusPj;
//    double Pj = exp(Eij * InvTemperature - dVStar);
    if (1 - ProbClock(jrej) <= OneMinusPj){
        cout << "1 - ProbClock(jrej) = " << 1 - ProbClock(jrej)<<  endl;
        cout << "OneMinusPj = " << OneMinusPj << endl;
        cout << "jrej = " << jrej << endl;
    }
    assert(1 - ProbClock(jrej) >= OneMinusPj);
//    assert(invOneMinusPHatOld >= OneMinusPj);
    assert(OneMinusPj >= 0. && OneMinusPj <= 1.);
    assert(almostEquals(exp(Eij * InvTemperature - dVStar),  PjSCO(jrej, i, scoData.xi, scoData.yi)));
    double prel = OneMinusPj * invOneMinusPHatOld;
    RNDGenerationSum += 1;
    if (realDis(gen) <= prel) {
        double EijNew = 2*(UiNewX * Hxij + UiNewY * Hyij);
//        double EijNew2 = 2*(UiNewX * (Lat.Jxx1p(jper) * Ux(j2) + Lat.Jxy1p(jper) * Uy(j2)) +
//                         UiNewY* (Lat.Jxy1p(jper) * Ux(j2) + Lat.Jyy1p(jper) * Uy(j2)));
//        assert(almostEquals(EijNew, EijNew2));
//        double Pjnew = exp(EijNew * InvTemperature - dVStar);
        double OneMinusPjnew = - expm1(EijNew * InvTemperature - dVStar);
        scoData.OneMinusPjOldProd *= OneMinusPj;
        scoData.OneMinusPjNewProd *= OneMinusPjnew;
        scoData.Hxi += Hxij;
        scoData.Hyi += Hyij;
        if (Prop.isSCOMethodOverrelaxationBuiltIn){
            scoData.Hxij.push_back(Hxij);
            scoData.Hyij.push_back(Hyij);
            scoData.jrejVec.push_back(jrej);
        }
    }
}

double MC2d::dESCONew(SCOData& scoData){
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
            double OneMinusPjnew = -expm1(Enew * InvTemperature - dVStar);
            assert(OneMinusPjnew > 0 && OneMinusPjnew <= 1);
            scoData.OneMinusPjNewProd *= OneMinusPjnew;
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

void MC2d::run1SCOParticleOriginal(int i) {
//    cout << "run1SCOParticleOriginal" <<endl;
    if (Prop.isCalcSelectedBondsProbability){
        SelectedBondsLocal.setZero();
    }
    int jrej = 0;     //jrej is sorted by distance and is periodic
    Debug_totali++;
    TotalRegularParticleSteps     += (!isOverrelaxing)? 1 : 0;
    TotalOverrelaxedParticleSteps += ( isOverrelaxing)? 1 : 0;
    if (Prop.isNearGroup){
        if (!Prop.isSCOMethodNearGroupBuiltIn){
            //        cout << "here" << endl;
            assert(almostEquals(dErot(i),dENN(i)));
            double dE = dErot(i);
            //        cout << "here " << endl;
            //         if (dE > 0 && realDis(gen) > exp(-dE * InvTemperature)){
            RNDGenerationSum += 1;
            if (!isOverrelaxing && realDis(gen) > exp(-dE * InvTemperature)){
                return;   // reject move
            }
        }
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
    RNDGenerationSum += 1;
//    double nu = Random();
    double invOneMinusPHatOld = InVOneMinusProbClock(jrej);
    assert(int(floor(log(nu)/log(ProbClock(jrej)))) == int(log(nu)/log(ProbClock(jrej))));
//    jrej = jrej + floor(log(nu)/log(ProbClock(jrej)));
    assert(almostEquals(InVLogProbClock(jrej), 1 / log(ProbClock(jrej))));
    const int IntMax = std::numeric_limits<int>::max();
    double jrejTemp = jrej + (log(nu) * InVLogProbClock(jrej));
    jrej = (jrejTemp<= IntMax)? jrejTemp : IntMax;

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
            if (Prop.isCalcSelectedBondsProbability){
                SelectedBondsLocal(jrej)++;
            }
//            double Pj = PjSCO(jrej, i, xi, yi);
//            assert(invOneMinusPHatOld >= (1 - Pj));
//            assert(Pj >=0 && Pj <=1);
//            double prel = (1 - Pj) * invOneMinusPHatOld;

            validiatePjSCONew(jrej, invOneMinusPHatOld, scoData);
//            if (realDis(gen) <= prel){
//                jrejVec.push_back(jrej);
//                PjVec.push_back(Pj);
//            }
        }

        NumResampled++;
        assert(jrej >=0 && jrej < N-1);
        nu = realDis(gen);
        RNDGenerationSum += 1;
//        nu = Random();
//        ProbOld = ProbClock(jrej);
        invOneMinusPHatOld = InVOneMinusProbClock(jrej);
//        assert(almostEquals(InVOneMinusProbClock(jrej), 1/(1 - ProbOld)));
        assert(almostEquals(InVLogProbClock(jrej), 1 / log(ProbClock(jrej))));
        assert(int(floor(log(nu)/log(ProbClock(jrej)))) == int(log(nu)/log(ProbClock(jrej))));
//        jrej = jrej + floor(1 + log(nu) * InVLogProbClock(jrej));
        jrejTemp = jrej + (1 + log(nu) * InVLogProbClock(jrej));
        jrej = (jrejTemp<= IntMax)? jrejTemp : IntMax;
    }

//    cout << "jrejVec = " <<endl;
//    for_each(jrejVec.begin(), jrejVec.end(), [](const auto& elem) { cout << elem << " "; });
//    cout << endl;
//    double dE = dESCO(jrejVec, i, xi, yi, PjVec);
    double dE = dESCONew(scoData);
//    assert(almostEquals(dE, dE2));
    RNDGenerationSum += 1;
    if (Prop.isCalcSelectedBondsProbability){
        SelectedBondsSum += SelectedBondsLocal;
    }
    if (dE < 0 || (realDis(gen) < exp(-dE * InvTemperature))){
        AcceptedRegularParticleSteps     += (!isOverrelaxing)? 1 : 0;
        AcceptedOverrelaxedParticleSteps += ( isOverrelaxing)? 1 : 0;
        Debug_Iaccepted++;    // move accepted
        AcceptSum++;
        if (Prop.isNearGroup){
            updateField(i);
        };
        updateState(i);
        if (Prop.isCalcSelectedBondsProbability){
            AcceptedBondsSum += SelectedBondsLocal;
        }
//        cout << "i = " << i << " accepted" <<endl;
//        JrejSum(jrej)++;
    }
}

void MC2d::run1SCOParticleWalker(int i) {
//    cout << "it is walker!"<<endl;
    int jrej = 0;     //jrej is sorted by distance and is periodic
    int shift = 0;
    Debug_totali++;
    TotalRegularParticleSteps     += (!isOverrelaxing)? 1 : 0;
    TotalOverrelaxedParticleSteps += ( isOverrelaxing)? 1 : 0;
    complexityLocalSum = 0;
    RNDGenerationLocalSum = 0;
    if (Prop.isNearGroup){
        if (!Prop.isSCOMethodNearGroupBuiltIn){
            assert(almostEquals(dErot(i),dENN(i)));
            double dE = dErot(i);
            RNDGenerationSum += 1;
            if (!isOverrelaxing && realDis(gen) > exp(-dE * InvTemperature)){
                return;   // reject move
            }
        }
        shift = Lat.Neighbours.size();
    }


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
        validiatePjSCONew(jrej, invOneMinusPhat, scoData);
        NumResampled++;
    }

    double dE = dESCONew(scoData);
    RNDGenerationSum += 1;
    if (dE < 0 || (realDis(gen) < exp(-dE * InvTemperature))){
        AcceptedRegularParticleSteps     += (!isOverrelaxing)? 1 : 0;
        AcceptedOverrelaxedParticleSteps += ( isOverrelaxing)? 1 : 0;
        Debug_Iaccepted++;    // move accepted
        updateField(i);
        updateState(i);
    }
    cleanjrejVecBool(jrejVecSmall);
}

void MC2d::run1SCOParticle(int i){
    TotalUpdateSum++;
    if(Prop.isWalkerAliasMethod){
        run1SCOParticleWalker(i);
    } else {
        run1SCOParticleOriginal(i);
    }
}

void MC2d::run1SCOStep() {
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
            isOverrelaxing = true;
            for (int j = 0; j < Prop.OverrelaxationSteps; ++j) {
                for (int i = 0; i < N; ++i) {
                    randomizeUiNew();
                    run1SCOParticle(i);
                }
            }
            isOverrelaxing = false;
            for (int j = 0; j < Prop.MetropolisSteps; ++j) {
                for (int i = 0; i < N; ++i) {
                    randomizeUiNew();
                    run1SCOParticle(i);
                }
            }
        } else if (Prop.isNearGroup) {
            isOverrelaxing = true;
            for (int j = 0; j < Prop.OverrelaxationSteps; ++j) {
                for (int i = 0; i < N; ++i) {
                    overRelaxateUiNew(i);
                    run1SCOParticle(i);
                }
            }
            isOverrelaxing = false;
            for (int j = 0; j < Prop.MetropolisSteps; ++j) {
                for (int i = 0; i < N; ++i) {
                    randomizeUiNew();
                    run1SCOParticle(i);
                }
            }
        }
    } else {
        isOverrelaxing = false;
        for (int j = 0; j < Prop.MetropolisSteps; ++j) {
            for (int i = 0; i < N; ++i) {
                randomizeUiNew();
                run1SCOParticle(i);
            }
        }
    }

}

void MC2d::setFFT() {
    using namespace utils;
    using namespace fftwpp;
    fftw::maxthreads=1;
    int size = Prop.NData;
    unsigned int n=Prop.NData;
    unsigned int np=n/2+1;
    size_t align=sizeof(Complex);
//    Ffourier1dAC.Reallocate(np, align);
//    freal1dAC.Reallocate(n, align);
    Ffourier1dAC.Allocate(np, align);
    freal1dAC.Allocate(n, align);
    #pragma omp critical (FFT)
    {
        ForwardFFT1dAC = make_unique<fftwpp::rcfft1d>(freal1dAC.Nx(), freal1dAC, Ffourier1dAC);
        BackwardFFT1dAC = make_unique<fftwpp::crfft1d>(freal1dAC.Nx(), Ffourier1dAC, freal1dAC);
    }

    unsigned int nx=Lat.L1;
    unsigned int ny=Lat.L2;
    unsigned int nyp=ny/2+1;
//    size_t align=sizeof(Complex);
    Ux2dfourier.Allocate(nx,nyp,align);
    Uy2dfourier.Allocate(nx,nyp,align);
    Jxx2dfourier.Allocate(nx,nyp,align);
    Jxy2dfourier.Allocate(nx,nyp,align);
    Jyy2dfourier.Allocate(nx,nyp,align);

    Ux2dreal.Allocate(nx, ny, align);
    Uy2dreal.Allocate(nx, ny, align);
    Jxx2dreal.Allocate(nx, ny, align);
    Jxy2dreal.Allocate(nx, ny, align);
    Jyy2dreal.Allocate(nx, ny, align);
    #pragma omp critical (FFT)
    {
        ForwardFFT2d = make_unique<fftwpp::rcfft2d>(Ux2dreal.Nx(), Ux2dreal.Ny(), Ux2dreal, Ux2dfourier);
        BackwardFFT2d = make_unique<fftwpp::crfft2d>(Ux2dreal.Nx(), Ux2dreal.Ny(), Ux2dfourier, Ux2dreal);
    }
    int Pindex = 0;
    assert(Lat.Jxx1p.size() == nx * ny);
    for(unsigned int i=0; i < nx; i++) {
        for (unsigned int j = 0; j < ny; j++) {
            assert(Pindex < Lat.Jxx1p.size());
            Jxx2dreal(i, j) = Lat.Jxx1p(Pindex);
            Jxy2dreal(i, j) = Lat.Jxy1p(Pindex);
            Jyy2dreal(i, j) = Lat.Jyy1p(Pindex);
            Pindex++;
        }
    }
    ForwardFFT2d->fft0(Jxx2dreal,Jxx2dfourier);
    ForwardFFT2d->fft0(Jxy2dreal,Jxy2dfourier);
    ForwardFFT2d->fft0(Jyy2dreal,Jyy2dfourier);

//    Forward.fft0(f,F);
//    Backward.fft0Normalized(F,f);
    cout << "setFFT for instace " << InstanceIndex << " done" <<endl;
}

void MC2d::destrtoyFFT() {
    #pragma omp critical (FFT)
    {
        ForwardFFT1dAC.reset();
        BackwardFFT1dAC.reset();
        ForwardFFT2d.reset();
        BackwardFFT2d.reset();
    }
    cout << "destrtoyFFT for instace " << InstanceIndex << " done" <<endl;

}




