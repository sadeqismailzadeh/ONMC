//
// Created by Sadeq Ismailzadeh on ۱۷/۰۱/۲۰۲۲.
//

#include "MC2d.h"
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
        Jzz(Lat.Jzz),
        Plabel(Lat.PLabel),
        invN(1./(N + 0.0))
    {
    Ux.setZero(N);
    Uy.setZero(N);
    Uz.setZero(N);

    Hx.setZero(N);
    Hy.setZero(N);
    Hz.setZero(N);

//    indices.setZero(2 * N - 2, N);

    MVectConfig.setZero(3);
    MFEVectConfig.setZero(3);

    hxAll.setZero(N);
    hyAll.setZero(N);
    hzAll.setZero(N);
//    Jself = Lat.J.block<2,2>(0,0);
//    Jself << Jxx(0,0), Jxy(0,0), Jxy(0,0), Jyy(0,0);

    //taking data
    StatisticalData.setZero(1000, 23);
    LastTempIndex = 0;

    //eigenvectors
    OPmakerX.setZero(N);
    OPmakerY.setZero(N);
    OPmakerZ.setZero(N);

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
    hz = 0;
    #pragma omp critical
    {
        InstanceIndex = LastInstanceIndex++;
    }

    //specific use
    iReCalc = 0;
    SnapshotTakingIndex = 0;
    Jself.setZero();
    Jself = Lat.Jself;

    UTx.setZero(N);
    UTy.setZero(N);
    UTz.setZero(N);
    GVecVsTemperature.setZero(1000, Lat.DistanceVec.size());
    GConVecVsTemperature.setZero(1000, Lat.DistanceVec.size());

    if (Prop.isHistogramSimulation){
        EtimeSeries.setZero(properties.NData);
        mtimeSeries.setZero(properties.NData);
        mPlanetimeSeries.setZero(properties.NData);
    }

    sigma = 60;

    init();
    UpdatehAll();
}

void MC2d::init() {
    randomizeAllSpins();
    calcAllFields();
    generateOPmaker();
    saveLocations();
    setSimulationParams();
    setFieldOffSimulationType();
    if (Prop.isFieldOn) {
        setFieldOnSimulationType();
    }
}

void MC2d::randomizeAllSpins(){
    for (int i = 0; i < N; i++) {
        randomizeUiNew();
        Ux(i) = UiNewX;
        Uy(i) = UiNewY;
        Uz(i) = UiNewZ;
    }
}



double MC2d::dErot(int i) {
    double dE1 = 0; //// assert
    const double dMx = UiNewX - Ux(i);
    const double dMy = UiNewY - Uy(i);
    const double dMz = UiNewZ - Uz(i);


    //another version using Jxx, Jxy, Jyy
    const double sMx = UiNewX + Ux(i);
    const double sMy = UiNewY + Uy(i);
//    dE += (dM(0) * (Jxx(i,i) * sM(0) + Jxy(i,i) * sM(1))) + (dM(1) * (Jxy(i,i) * sM(0) + Jyy(i,i) * sM(1)));
//    dE += (dMz)* Jzz(i, i)* (UiNewZ + Uz(i));

    dE1 += (dMx * (Jself(0,0) * sMx + Jself(1,0) * sMy)) + (dMy * (Jself(1,0) * sMx + Jself(1,1) * sMy));
    dE1 += (dMz)* Jself(2,2)* (UiNewZ + Uz(i));  //TODO BUG its not valid for honeycomb

    if (Prop.isFieldOn){
        dE1 -= dMx * hx + dMy * hy;
        dE1 -= dMz * hz;
    }

    dE1 += 2 * (dMx* Hx(i) + dMy * Hy(i) + dMz * Hz(i));
    return dE1;
}

void MC2d::run1MetropolisStep() {
    static const double PI = acos(-1);
    int CounterAccceptence = 0;
    if (Prop.isTakeSnapshot && (SnapshotTakingIndex++ % TakeSnapshotStep == 0)){
        takeSnapshot();
    }
    if (++iReCalc % 1'000'000 == 0){
        calcAllFields();
    }

    for (int c = 0; c < N; c++) {
        Debug_totali++;
        int i = intDis(gen);
        randomizeUiNew();
//        random3dAngle(i);
//        randomAdaptive(i);
        double dE = dErot(i);
        if (dE < 0 || (Random() < exp(-dE / Temperature))) {
            updateField(i);
            updateState(i);
            Debug_Iaccepted++;
            CounterAccceptence++;
        }
    }

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

void MC2d::updateField(const int i) {
    const double dMx = UiNewX - Ux(i);
    const double dMy = UiNewY - Uy(i);
    const double dMz = UiNewZ - Uz(i);

    Hx(i)  -= Jself(0,0) * dMx + Jself(0, 1) * dMy;
    Hy(i)  -= Jself(0,1) * dMx + Jself(1, 1) * dMy;
    Hz(i)  -= Jself(2,2) * dMz;

    // due to last measurement 1401.05.11  this code is 0.08% faster
    if (Prop.LatticeType != 'h' && Prop.isMacIsaacMethod) {
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
                Hz(k + j1) += Lat.Wzz(Fn + k) * dMz;
            }
        }
    } else if (Prop.LatticeType == 'h' && Prop.isMacIsaacMethod && Lat.isHoneycombFromTriangular) {
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
                    Hz(2*k + j1) += Lat.Wzz11(Fn + k) * dMz;
                    Hx(2*k+1 + j1) += Lat.Wxx12(Fn + k) * dMx + Lat.Wxy12(Fn + k) * dMy;
                    Hy(2*k+1 + j1) += Lat.Wyy12(Fn + k) * dMy + Lat.Wxy12(Fn + k) * dMx;
                    Hz(2*k+1 + j1) += Lat.Wzz12(Fn + k) * dMz;
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
                    Hz(2*k + j1) += Lat.Wzz21(Fn + k) * dMz;
                    Hx(2*k+1 + j1) += Lat.Wxx22(Fn + k) * dMx + Lat.Wxy22(Fn + k) * dMy;
                    Hy(2*k+1 + j1) += Lat.Wyy22(Fn + k) * dMy + Lat.Wxy22(Fn + k) * dMx;
                    Hz(2*k+1 + j1) += Lat.Wzz22(Fn + k) * dMz;
                }
            }
        }

    } else {
        int a = 0;
        for (int j = 0; j < N; j++) {
            Hx(j) += Jxx(j, i) * dMx +  Jxy(j, i) * dMy;
            Hy(j) += Jxy(j, i) * dMx +  Jyy(j, i) * dMy;
            Hz(j) += Jzz(j,i) * dMz;
        }
    }

}

void MC2d::stabilize() {
    for (int i = 0; i < Prop.NStabilize; i++) {
        run1MetropolisStep();
    }
}

void MC2d::stabilizeMinor() {
    for (int i = 0; i < Prop.NStabilizeMinor; i++) {
        run1MetropolisStep();
    }
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
    Vector3d SumMvect;
    Vector3d SumMFEvect;
    SumMvect.setZero();
    SumMFEvect.setZero();
    int N_MCstep = Prop.NData * Prop.dataTakingInterval;
    VectorXd mtime;

    if (Prop.isComputingAutoCorrelationTime) {
        mtime.setZero(N_MCstep);
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
        run1MetropolisStep();
        if (Prop.isComputingAutoCorrelationTime){
            calcOrderParam();
            double m_config = MVectConfig.norm();
            mtime(i) = m_config * invN;
        }

        if (i%(max({N_MCstep / 20, 1})) == 0 && Prop.isHistogramSimulation){
            cout << "done  " << i/(N_MCstep+0.0) << "%  "  <<endl;
            toc();
        }

        if (i % Prop.dataTakingInterval == 0) {
            if (Prop.isHistogramSimulation) {
                calcOrderParam();  //TODO this line is duplicated
                EtimeSeries(i) = getEnergy();
                mtimeSeries(i) = MVectConfig.norm();
                mPlanetimeSeries(i) = sqrt(square(MVectConfig(0)) + square(MVectConfig(1)));
            }
            double E_config = getEnergy();
            SumE += E_config;
            SumE2 += E_config * E_config;
            if (Prop.isComputingCorrelationLength){
                UpdateGVec();
//                UpdateGVecMacIsaacTest();
//                UpdateGMatOld();
            }
            calcOrderParam();
            SumMT += MTVectConfig.norm();    //TODO check if overflow
            double m_config = MVectConfig.norm();
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
    Vector3d MeanMvectPP = SumMvect / (Prop.NData * N); // MeanMvect per particle
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
    double hNorm = sqrt(square(hx) + square(hy) + square(hz));
    // Correlation time
    double tau = (Prop.isComputingAutoCorrelationTime) ? getAutoCorrelationTime(mtime) : 0;
    if (Prop.isComputingCorrelationLength){
        calcGVec();
//        calcGVecOld();
    }
//    double Binder = 1 - (MeanM4 / (3*square(MeanM2)));
//    double phiMvect = atan2(MeanMvectPP(0), MeanMvectPP(2)); // phi = atan2(x,z)
//    double thetaMvect = atan2(hypot(MeanMvectPP(0), MeanMvectPP(2)), MeanMvectPP(1)); // theta = atan2(hypot(x,z), y)
    double ProbAccepted = (Debug_Iaccepted + 0.) / Debug_totali;
    StatisticalData.row(LastTempIndex++) <<
            Temperature, hNorm, Epp, cpp, mpp , Xpp, Binder,
            mPlanepp, XPlanepp, BinderPlane,
            MeanMvectPP.norm(), MeanMvectPP(0), MeanMvectPP(1), MeanMvectPP(2),
            mFEpp, XFEpp, BinderFE, mFEPlanepp, XFEPlanepp, BinderFEPlane, mTpp, tau, ProbAccepted;

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
    ofstream SaveSampleStatData(FolderPath + "/SampleStatData.txt"s);
    SaveSampleStatData << StatisticalData;

}

void MC2d::setControlParam(const double Pset) {
    if (Prop.ControlParamType == 'T'){
        Temperature = Pset;
    } else if (Prop.ControlParamType == 'h') {
        hx = hHat(0) * Pset;
        hy = hHat(1) * Pset;
        hz = hHat(2) * Pset;
        UpdatehAll();
    }
}
void MC2d::simulateHistogram() {
    for (CtrlParamAbs = Prop.Pstart;
         CtrlParamAbs > Prop.PcEstimate && Temperature > 0; CtrlParamAbs -= Prop.Pdecrement){
        setControlParam(CtrlParamAbs);
        stabilizeMinor();
        cout << "ControlParam = " << CtrlParamAbs << endl;
    }

    setControlParam(Prop.PcEstimate);
    stabilizeAndGetStatisticalData();
    SaveTimeSeries();
    cout << "done id=" << this_thread::get_id() << endl;
}

void MC2d::simulateFiniteDiff() {
    for (CtrlParamAbs = Prop.Pstart; (CtrlParamAbs > Prop.Pend && Temperature > 0); CtrlParamAbs -= Prop.Pdecrement) {
        setControlParam(CtrlParamAbs);
        stabilizeAndGetStatisticalData();
    }
    cout << "done id=" << this_thread::get_id() << endl;
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
    cout << "done id=" << this_thread::get_id() << endl;
}

void MC2d::simulateCriticalRegionEquidistant() {
    double NearCriticalRange = 0.3;
//    double PmaxCritical = Prop.PcEstimate + NearCriticalRange/2;
//    double PminCritical = Prop.PcEstimate - NearCriticalRange/2;
    double PmaxCritical = Prop.PmaxCritical;
    double PminCritical = Prop.PminCritical;
    for (CtrlParamAbs = Prop.Pstart;
         CtrlParamAbs > PmaxCritical && Temperature > 0; CtrlParamAbs -= Prop.Pdecrement){
        setControlParam(CtrlParamAbs);
        stabilize();
        cout << "ControlParam = " << CtrlParamAbs << endl;
    }

    double dP = 0.01;
    setControlParam(PmaxCritical);
    for(CtrlParamAbs = PmaxCritical; CtrlParamAbs > PminCritical; CtrlParamAbs -= dP) {
        setControlParam(CtrlParamAbs);
        stabilizeAndGetStatisticalData();
    }
    cout << "done id=" << this_thread::get_id() << endl;
}

void MC2d::stabilizeAndGetStatisticalData() {
    Debug_Iaccepted = 0;
    Debug_totali = 0;
    stabilize();
    getStatisticalData();
    cout << "ControlParam = " << CtrlParamAbs << endl;
    cout << "Debug_ii = " << Debug_Iaccepted << endl;
    cout << "probability accepted = " << (Debug_Iaccepted + 0.) / Debug_totali << "\n";
    toc();
    cout << endl;
}

void MC2d::calcAllFields() {
    // line be low equals to H = J * Uxy;
    VectorXd HnewX;
    VectorXd HnewY;
    VectorXd HnewZ;
    HnewX.setZero(Hx.size());
    HnewY.setZero(Hy.size());
    HnewZ.setZero(Hz.size());
//    H.setZero();

    if (Prop.isMacIsaacMethod){
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
                        HnewZ(i) += Lat.Wzz(Fn + k) * Uz(pnum2);
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
                    HnewZ(i) += Jzz(j,i) * Uz(j);
                }
            }
        }
    }


    //// assertion
    Hx = HnewX;
    Hy = HnewY;
    Hz = HnewZ;
    if (iReCalc >0){
//        cout << "iReCalc " << iReCalc  << endl;
//        cout << "(HnewXY - H).norm() = " << (HnewXY - H).norm() << endl;
//        cout << "(HnewX - Hx).norm() = " << (HnewX - Hx).norm() << endl;
//        cout << "(HnewY - Hy).norm() = " << (HnewY - Hy).norm() << endl;
        assert(almostEquals(HnewX,Hx));
        assert(almostEquals(HnewY,Hy));
    }
}

double MC2d::getEnergy() {
    double E = 0;
    double E1 = 0;
    double E2 = 0;
    E += Ux.dot(Hx);
    E += Uy.dot(Hy);
    E += Uz.dot(Hz);

    if(Prop.isFieldOn){
        E -= Ux.dot(hxAll);   //TODO reuse MVect for this purpose
        E -= Uy.dot(hyAll);
        E -= Uz.dot(hzAll);
    }
    for (int i = 0; i < N; i++) { // TODO self energy
//        E += (Uxy.segment<2>(2 * i).transpose() * J.block<2,2>(2 * i, 2 * i) * Uxy.segment<2>(2 * i)).value();

        E += (Ux(i) * (Jself(0,0) * Ux(i) + Jself(0,1) * Uy(i))) +
             (Uy(i) * (Jself(0,1) * Ux(i) + Jself(1,1) * Uy(i)));
        E += Uz(i) * Jself(2,2) * Uz(i);
    }
    return E;
}

void MC2d::calcOrderParam(){
    if (Prop.LatticeType != 'h'){
        MVectConfig(0) = Ux.dot(OPmakerX);
        MVectConfig(1) = Uy.dot(OPmakerY);
        MVectConfig(2) = Uz.dot(OPmakerZ);

    } else {
        MVectConfig(0) = Ux.dot(OPmakerXX) + Uy.dot(OPmakerXY);
        MVectConfig(1) = Ux.dot(OPmakerYX) + Uy.dot(OPmakerYY);
        MVectConfig(2) = 0;
    }

    if (Prop.LatticeType != 't'){
        MFEVectConfig.setZero();
        for (int i = 0; i < N; ++i) {
            MFEVectConfig(0) += Ux(i);
            MFEVectConfig(1) += Uy(i);
            MFEVectConfig(2) += Uz(i);
        }
    }

    MTVectConfig.setZero();
    for (int i = 0; i < N; ++i) {
        MTVectConfig(0) += UTx(i);
        MTVectConfig(1) += UTy(i);
        MTVectConfig(2) += UTz(i);
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
                    OPmakerZ(i) = 1;
                    break;
                case 2:
                    OPmakerX(i) = -1;
                    OPmakerY(i) = 1;
                    OPmakerZ(i) = -1;
                    break;
                case 3:
                    OPmakerX(i)  = -1;
                    OPmakerY(i)  = -1;
                    OPmakerZ(i) = 1;
                    break;
                case 4:
                    OPmakerX(i)  = 1;
                    OPmakerY(i)  = -1;
                    OPmakerZ(i) = -1;
                    break;
            }
        }
    } else if (Prop.LatticeType == 't') {
        for (int i = 0; i < N; ++i) {
            OPmakerX(i) = 1;
            OPmakerY(i) = 1;
            OPmakerZ(i) = 1;
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
        hz = Prop.hfield(2);
        Temperature = Prop.Pstart;
        this->hHat.setZero();   // TODO hHat is used for two things and should be changed
    } else if (Prop.ControlParamType == 'h') {
        this->hHat = Prop.hHat;
        Temperature = Prop.FixedTemperatureInVaryingField;
        hx = hHat(0) * Prop.Pstart;
        hy = hHat(1) * Prop.Pstart;
        hz = hHat(2) * Prop.Pstart;
        UpdatehAll();
    }
}

void MC2d::UpdatehAll() {
    for (int i = 0; i < N; ++i) {
        hxAll(i) = hx;
        hyAll(i) = hy;
        hzAll(i) = hz;
    }
}
double MC2d::getAutoCorrelationTime(const VectorXd &mtime){
    assert(!FolderPath.empty());
//    auto tmax = min(((mtime.size()-1L)/5L), 5000L);
//    auto tmax = mtime.size()/4L;
    long Size = mtime.size();
//    auto tmax = min(((Size-1L)/5L), 10L*N);
    auto tmax = (Size-1L)/5L;
    VectorXd Xtime(tmax), XtimeDiff(tmax);
    Xtime.setZero();
    XtimeDiff.setZero();
    for (int i = 0; i < Xtime.size(); ++i) {
        Xtime(i) = calcXt(mtime, i);
        XtimeDiff(i) = calcXtDiff(mtime, i);
    }
    VectorXd XtimeNormalized;
    XtimeNormalized = Xtime / Xtime(0);
    double tau = XtimeNormalized.sum();
    SaveXtime << Xtime.transpose() << endl;
    SaveXtimeNormalized << XtimeNormalized.transpose() << endl;

    ofstream SaveMtime(FolderPath + "/mtime.txt"s,ofstream::out | ofstream::app);
    ofstream SaveMtimeNormalized(FolderPath + "/mtimeNormalized.txt"s, ofstream::out | ofstream::app);
    SaveMtime << mtime.head(10000).transpose() << endl;
    SaveMtimeNormalized << (mtime.head(10000).array() - mtime.mean()).transpose() << endl;

    ofstream SaveXtimeDiff(FolderPath + "/XtimeDiff.txt"s,ofstream::out | ofstream::app);
    SaveXtimeDiff << XtimeDiff.transpose() << endl;
//    cout << "Temperature = "  << Temperature << endl
//         << "Xtime : " << setprecision(3) << (Xtime.head(50).transpose()) <<endl<< endl;
    return tau;
}

double MC2d::calcXt(const VectorXd &mtime, int tstep){
    auto tmax = mtime.size() - 1;
    auto deltat = tmax - tstep;
    double invDelta = 1 / (deltat +0.0);
//    double Xt = pow(deltat, -1) * mtime.head(deltat).dot(mtime.tail(deltat));
//    double Xt = invDelta * mtime.head(deltat).dot(mtime.tail(deltat));
    double Xt = 0;
    double mean1 = mtime.head(deltat).mean();
    double mean2 = mtime.tail(deltat).mean();
    for (int i = 0; i < deltat; ++i) {
        Xt += (mtime(i)-mean1)*(mtime(i+tstep)-mean2);
    }
    Xt *= invDelta;
//    Xt -= pow(deltat, -2) * mtime.head(deltat).sum() * mtime.tail(deltat).sum();
//    Xt -= invDelta * invDelta * mtime.head(deltat).sum() * mtime.tail(deltat).sum();
//    Xt -= pow (1/(tmax+0.0) * mtime.sum(), 2);
    return Xt;
}

double MC2d::calcXtDiff(const VectorXd &mtime, int tstep){
    int tmax = mtime.size() - 1;
    int deltat = tmax - tstep;
    double invDelta = 1 / (deltat +0.0);
    double Xt = 0;
    for (int i = 0; i < deltat; ++i) {
        Xt += square(mtime(i)- mtime(i+tstep));
    }
    Xt *= invDelta;
    return Xt;
}


void MC2d::setParentFolderPath(const string& ParentFolderPath1) {
    this->ParentFolderPath = ParentFolderPath1;
    FolderPath = ParentFolderPath1 + "/"s + to_string(InstanceIndex);
    fs::create_directories(FolderPath);

    SaveXtime.open(FolderPath + "/Xtime.txt"s);
    SaveXtimeNormalized.open(FolderPath + "/XtimeNormalized.txt"s);

}

void MC2d::takeSnapshot() {
    assert(!FolderPath.empty());
    if (InstanceIndex == 0){
        takeSnapshotXY();
    }
}

void MC2d::takeSnapshotXY(){
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
    ofstream Locations(FolderPath + "/locations.txt"s);
    ofstream Distances(FolderPath + "/Distances.txt"s);
    Locations << Lat.D;
}

void MC2d::UpdateGMatOld() {
    if (Prop.LatticeType == 't') {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                GMat(i, j) += Ux(i) * Ux(i) * Ux(j) + Uy(i) * Uy(j) + Uz(i) * Uz(j);
            }
        }
    } else if (Prop.LatticeType == 's'){
        calcUT();
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
//                GMat(i, j) += UTxy(2 * i) * UTxy(2 * j) + UTxy(2 * i + 1) * UTxy(2 * j + 1) + UTz(i) * UTz(j);
                GMat(i, j) += UTx(i) * UTx(j) + UTy(i) * UTy(j);
            }
        }
    } else if (Prop.LatticeType == 'h') {
        calcUT();
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                GMat(i, j) += UTx(i) * UTx(j) + UTy(i) * UTy(j);
            }
        }
    }

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
                GVec(Lat.MapR(i, j)) += Ux(i) * Ux(j) + Uy(i) * Uy(j) + Uz(i) * Uz(j);
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
                        GVec(Lat.MapR0(Fn+k)) += Ux(pnum) * Ux(j1+k) + Uy(pnum) * Uy(j1+k) + Uz(pnum) * Uz(j1+k);
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

void MC2d::calcGVecOld() {
    GMat.array() /= Prop.NData;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            GVec(Lat.MapR(i,j)) += GMat(i,j);
        }
    }
//    ofstream GVecOut(FolderPath + "/GVec.txt"s, ofstream::out | ofstream::app);
//    GVecOut << GVec.transpose() << endl;
    GVec.array() /= Lat.CounterR.array().cast<double>();
    GVecVsTemperature.row(LastTempIndex) = GVec.transpose() ;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if(Prop.LatticeType != 's'){
                GConVec(Lat.MapR(i,j)) += GMat(i,j) - mpp*mpp;
            } else {
                GConVec(Lat.MapR(i,j)) += GMat(i,j) - mPlanepp*mPlanepp;
            }
        }
    }
//    ofstream GConVecOut(FolderPath + "/GConVec.txt"s, ofstream::out | ofstream::app);
//    GConVecOut << GConVec.transpose() << endl;
    GConVec.array() /= Lat.CounterR.array().cast<double>();
    GConVecVsTemperature.row(LastTempIndex) = GConVec.transpose();
}

void MC2d::calcGVec() {
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
            UTz(i) = OPmakerZ(i) * Uz(i);
            assert(almostEquals(sqrt(square(UTx(i)) + square(UTy(i)) + square(UTz(i))), 1));
        }
    } else if (Prop.LatticeType == 'h'){
        for (int i = 0; i < N; ++i) {
            UTx(i) = OPmakerXX(i) * Ux(i) + OPmakerXY(i) * Uy(i);
            UTy(i) = OPmakerYX(i) * Ux(i) + OPmakerYY(i) * Uy(i);
            UTz(i) = Uz(i);
            assert(almostEquals(sqrt(square(UTx(i)) + square(UTy(i)) + square(UTz(i))), 1));
        }
    }
    ////assertion
}

void MC2d::UpdateGVecMacIsaacTest() {
    assert(Prop.LatticeType == 'h');
    const int L1 = Lat.L1;
    const int L2 = Lat.L2;
    VectorXd GVectemp1;
    VectorXd GVectemp2;
    GVectemp1.setZero(GVec.size());
    GVectemp2.setZero(GVec.size());

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            GVectemp2(Lat.MapR(i,j)) += UTx(i) * UTx(j) + UTy(i) * UTy(j);
        }
    }


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
                    GVectemp1(Lat.MapR0Sub11(Fn+k)) += UTx(pnum) * UTx(j1 + 2*k)   + UTy(pnum) * UTy(j1 + 2*k);
                    GVectemp1(Lat.MapR0Sub12(Fn+k)) += UTx(pnum) * UTx(j1 + 2*k+1) + UTy(pnum) * UTy(j1 + 2*k+1);
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
                    GVectemp1(Lat.MapR0Sub21(Fn+k)) += UTx(pnum) * UTx(j1 + 2*k)   + UTy(pnum) * UTy(j1 + 2*k);
                    GVectemp1(Lat.MapR0Sub22(Fn+k)) += UTx(pnum) * UTx(j1 + 2*k+1) + UTy(pnum) * UTy(j1 + 2*k+1);
                }
            }
        }
    }

    assert(almostEquals(GVectemp1, GVectemp2));
}

void MC2d::SaveTimeSeries() {
    ofstream EtimeSeriesOut(FolderPath + "/EtimeSeries.csv"s, ofstream::out);
    ofstream mtimeSeriesOut(FolderPath + "/mtimeSeries.csv"s, ofstream::out);
    ofstream mPlanetimeSeriesOut(FolderPath + "/mPlanetimeSeries.csv"s, ofstream::out);
    EtimeSeriesOut << EtimeSeries.format(CSVFormat);
    mtimeSeriesOut << mtimeSeries.format(CSVFormat);
    mPlanetimeSeriesOut << mPlanetimeSeries.format(CSVFormat);
}
