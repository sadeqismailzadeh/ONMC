#ifndef MC_H
#define MC_H

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <cstdlib>
#include <ctime>
#include <chrono>
//#include <thread>
#include <Eigen/Dense>
#include "Lattice2d.h"
#include "utils.h"
#include "MCProperties.h"
#include "prng.h"
#include "WalkerAlias.h"

#ifdef NDEBUG
#undef NDEBUG
#include "ArrayFFT.h"   //TODO check with NDEBUG enabled
#define NDEBUG
#else
#include "ArrayFFT.h"
#endif

#include "fftw++.h"

#if  __GNUC__ < 9
#define BOOST_NO_CXX11_SCOPED_ENUMS
    #include <boost/filesystem.hpp>
    #undef BOOST_NO_CXX11_SCOPED_ENUMS
    namespace fs = boost::filesystem;
#elif  __GNUC__ >= 9
#include <filesystem>
namespace fs = std::filesystem;
//#elif defined(__clang__)
//    #include <filesystem>
//    namespace fs = std::filesystem;
#endif

#ifdef _OPENMP // Use omp.h if -fopenmp is used in g++
#include <omp.h>
#endif

using namespace std;
using namespace Eigen;

//std::random_device rd;
//static thread_local mt19937 gen(time(nullptr) + clock() + hash<thread::id>()(this_thread::get_id()));
//static mt19937 gen;
//static thread_local std::mt19937 gen{ std::random_device{}()};
//static thread_local default_random_engine gen(time(nullptr) + clock() + hash<thread::id>()(this_thread::get_id()));
//std::mt19937 gen(static_cast<long unsigned int>(time(0)));  //set seed
//static thread_local std::mt19937 gen{ std::random_device{}() };
// TODO a define a struct to share data between MC2d and simulation class
//   and remove defining parameter both in MC2d and Simulation like NStabilize
class MC2d {
private:
    double getAutoCorrelationTime(const VectorXd &mtime);
    double calcXt(const VectorXd &m_time, int tstep);
    double calcXtDiff(const VectorXd &mtime, int tstep);
    string ParentFolderPath;
    string FolderPath;
    SimpleStopWatch StopWatchTemperature;
    SimpleStopWatch StopWatchTotal;
    inline static size_t LastInstanceIndex = 0;


    void saveLocations();
    void takeSnapshot();
    void takeSnapshotXY();

    //for specific function
    int64_t iReCalc;
    int64_t SnapshotTakingIndex;
    int64_t TakeSnapshotStep;
    ofstream SaveXtime;
    ofstream SaveXtimeNormalized;
public:
    explicit MC2d(const Lattice2d &lat, const int64_t &seed, const MCProperties &properties);
    MC2d(const MC2d&) = delete;
//    MC2d(const MC2d&) = default;
    MC2d(MC2d &&) = default;

    size_t InstanceIndex;

    const Lattice2d Lat; //underlying lattice

    //
    VectorXd Ux;
    VectorXd Uy;
//VectorXd H = VectorXd::Zero(3*N);
    VectorXd Hx;
    VectorXd Hy;
    MatrixXd indices;
    Vector2d MVectConfig;
    Vector2d MTVectConfig;   //transformed UTxy
    Vector2d MFEVectConfig;  // ferromagnetic order param
    const VectorXi & Plabel;
//    const MatrixXd & J; //TODO remove J
    Matrix3d Jself;
    const MatrixXd & Jxx;
    const MatrixXd & Jyy;
    const MatrixXd & Jxy;
    double UiNewX;
    double UiNewY;

    const int& N;
    const double invN;

    VectorXi ParticleList;
    VectorXi ParticleListSorted;


    // external field
    double hx;
    double hy;
    VectorXd hxAll;
    VectorXd hyAll;


    // simulation
    VectorXd ProbClock;
    VectorXd InVLogProbClock;
    VectorXd InVOneMinusProbClock;
    VectorXd ProbClockFar;
    VectorXd InVLogProbClockFar;
    VectorXd InVOneMinusProbClockFar;
    vector<bool> jrejVecBool;
    WalkerAlias WalkerSampler;
    std::poisson_distribution<int> PoissonDis;
    double ProbClockNN;
    VectorXd JrejSum;
    VectorXd JrejProb;
    VectorXd JrejProbCumul;
    // stochastic cutoff
    VectorXd dVstarVec;
    int64_t TimeMCTotal;
    double ExchangeFactor;
    int64_t NumResampled;
    bool isOverrelaxing;
    vector<vector<double>> jrejMat;
    //Tomita
    vector<double> gammaVec;
    vector<double> alphaVec;
    vector<double> invAlphaVec;
    vector<double> lambdaVec;
    vector<double> invLambdaVec;
    vector<int> jrejVecRepeated;
    double alphaTilde;
    double kappa;
    double invKappa;
//taking data
    MatrixXd StatisticalData;
    int LastTempIndex;


    VectorXd OPmakerX;  // to construct order parameter
    VectorXd OPmakerY;  // to construct order parameter
    VectorXd OPmakerXX;  // to construct order parameter for honeycomb
    VectorXd OPmakerXY;  // to construct order parameter for honeycomb
    VectorXd OPmakerYX;  // to construct order parameter for honeycomb
    VectorXd OPmakerYY;  // to construct order parameter for honeycomb
    double CtrlParamAbs;

    mt19937 gen;
    std::uniform_real_distribution<double> realDis;
    std::uniform_real_distribution<double> realDissym;
    std::uniform_int_distribution<int> intDis;

    int Nthread;
    bool isSerial;

    // Regular and Overrelaxed acceptance rates
    long TotalRegularParticleSteps;
    long AcceptedRegularParticleSteps;
    long TotalOverrelaxedParticleSteps;
    long AcceptedOverrelaxedParticleSteps;

    // control parameters

    double Temperature;
    double InvTemperature;
    Vector3d hHat;

    // correlation function
//    MatrixXd GMat;
    MatrixXd GMat;
    VectorXd GVec;
    VectorXd GConVec;
    MatrixXd GVecVsTemperature;
    MatrixXd GConVecVsTemperature;
    double mpp;
    double mPlanepp;
    VectorXd UTx;
    VectorXd UTy;

    //Histogram method
//    VectorXf EtimeSeries;
//    VectorXf mtimeSeries;
    vector<float> EtimeSeries;
    vector<float> mtimeSeries;
    vector<double> EtimeSeries2step;
    vector<double> mtimeSeries2step;
    vector<double> EtimeEq;
    vector<double> mtimeEq;
    unique_ptr<fftwpp::rcfft1d> ForwardFFT1dAC;
    unique_ptr<fftwpp::crfft1d> BackwardFFT1dAC;
    unique_ptr<fftwpp::rcfft2d> ForwardFFT2d;
    unique_ptr<fftwpp::crfft2d> BackwardFFT2d;

    ArrayFFT::array1<Complex> Ffourier1dAC;
    ArrayFFT::array1<double> freal1dAC;

//    ArrayFFT::array2<Complex> Ffourier2d;
    ArrayFFT::array2<double> Ux2dreal;
    ArrayFFT::array2<double> Uy2dreal;
    ArrayFFT::array2<double> Jxx2dreal;
    ArrayFFT::array2<double> Jxy2dreal;
    ArrayFFT::array2<double> Jyy2dreal;
    ArrayFFT::array2<Complex> Ux2dfourier;
    ArrayFFT::array2<Complex> Uy2dfourier;
    ArrayFFT::array2<Complex> Jxx2dfourier;
    ArrayFFT::array2<Complex> Jxy2dfourier;
    ArrayFFT::array2<Complex> Jyy2dfourier;



    // Debug
    int64_t Debug_Iaccepted;
    int64_t Debug_totali;
    int64_t UAlignedWithHSum;
    int64_t UAntiAlignedWithHSum;
    int64_t EIncreaserReducerTotalSum;
    int64_t attemptSum;
    int64_t FineTuneNum;
    double Eold;
    // TODO save snapshot

    // adaptive sampling
    double sigma;
    std::normal_distribution<double> gaussianDis;
    int64_t AdapticeCounterAcceptance;
//    int64_t Debug_totali;
//    A(const T &&) = delete;  // prevents rvalue binding for not accepting temporary
//    throw std::invalid_argument( "received negative value" );

    // MC properties
    const MCProperties Prop;

//    MC2d(const Lattice2d &lat, const int32_t & seed);

    void init();
    void setId();
    void static resetLastInstanceIndex();
    void randomizeAllSpins();
    double dErot(int i);
    void run1MetropolisStep();
    void run1OverRelaxationStep();
    void run1GlobalOverRelaxationStep();
    void run1NewOverRelaxationStep();
    void run1FineTuningStep();
    void run1RandomizerStep();
    void run1UltraFineTuningStep();
    void run1HybridStep();
    void stabilize();
    void stabilizeMinor();
    void calcAllFields();
    void updateField(const int i);
    void getStatisticalData();
    void setRNGseed(int32_t seednum);
    double getEnergy();
    double getEnergyByFFT();
    double getEnergyDirect();
    double getEnergyByFields();
    void calcOrderParam();
    void setProbClockFar();
    void setProbClock();
    double dEij(int i, int j);
    double dENN(int i);
    double PRelFar(int jrej, int i, int xi, int yi, double InvOneMinusPHatOld);
    void run1ClockParticleFar(int i);
    double PRelOriginal(int jrej, int i, int xi, int yi, double InvOneMinusPHatOld);
    void run1ClockParticle(int i);
    void run1ClockParticleOriginal(int i);
    void run1ClockParticleWalker(int i);
    void cleanjrejVecBool(const vector<int> &jrejVecSmall);
    void cleanjrejVecRepeated(const vector<int> &jrejVecSmall);
    double PRelAddUp(const vector<int> &jrejVecAddUp, int i, int xi, int yi,
                     vector<double> &PHatOldVecAddUp);
    void run1ClockParticleAddUp(int i);
    void run1SCOParticleOriginal(int i);
    void run1SCOParticleWalker(int i);
    void run1SCOParticle(int i);
    void run1SCOParticleOverrelaxation(int i);
    void run1SCOStep();
    void run1SCOShuffle();
    void run1SCOShuffleParticle(int i);
    void run1SCOStepPreset();
    void run1SCOPresetParicle(int i);
    double PjSCO(int jrej, int i, int xi, int yi);
    double dESCO(const vector<int> &jrejVecAddUp, int i, int xi, int yi,
                 const vector<double> &PHatOldVecAddUp);
    void validiatePjSCONew(int jrej, double invOneMinusPHatOld, SCOData& scoData);
    void validiatePjSCOPreset(int jrej, double invOneMinusPHatOld, SCOData& scoData);
    void validiatePjSCOShuffle(int jrej, double invOneMinusPHatOld, SCOData& scoData);
    double dESCONew(SCOData& scoData);
    double dESCOPreset(SCOData& scoData);
    void run1ClockParticleExact(int i);
    void run1ClockStep();
    void run1ClockStepOriginal();
    void setProbTomita();
    void run1TomitaStep();
    void run1TomitaParticle(int i);
    double dEijrej(int jrej, int i, int xi, int yi);
    double Jtildeijrej(int jrej, int i, int xi, int yi);
    double consensusProb(int i);
    void overRelaxateUiNew(int i);
    void overRelaxateUiNew(int i, double Hxi, double Hyi);
    void setFFT();
    void destrtoyFFT();

    void simulate();
    void simulateFiniteDiff();
    void simulateHistogram();
    void simulateCriticalRegionNonEquidistant();
    void simulateCriticalRegionEquidistant();

    [[maybe_unused]] [[deprecated]] void decrementControlParam(const double dP);
    void stabilizeAndGetStatisticalData();
    void SaveTimeSeries();
    void setControlParam(const double Pset);
    void generateOPmaker();
    void setFieldOnSimulationType();
    void setParentFolderPath(const string& ParentFolderPath1);
    void UpdatehAll();

    // correlation function
    void calcGVecOld();
    void UpdateGMatOld();
    void UpdateGVec();
    void UpdateGVecRegular();
    void UpdateGVecMacIsaac();
    void UpdateGVecMacIsaacTest();
    void calcGVec();
    void calcUT();


//    __attribute__ ((noinline)) void updateField(int i);    // Eigen problem with inline
//    [[gnu::noinline]] void updateField(int i);


    [[deprecated]] void makeIndices();
    void setSimulationParams();
    void setFieldOffSimulationType();

    void setMultithreadingParams(bool isSerial, int Nthread){
        this->isSerial = isSerial;
        this->Nthread = Nthread;
    }

    inline void updateState(int i){
        Ux(i) = UiNewX;
        Uy(i) = UiNewY;
    }


    inline void randomizeUiNew (){
        double x, y;
        x = realDissym(gen);
        y = realDissym(gen);
        double r2 = x*x +y*y;

        while (r2 > 1.0){
            x = realDissym(gen);
            y = realDissym(gen);
            r2 = x*x +y*y;
        }
        double invr = 1.0/sqrt(r2);
        UiNewX = x * invr;
        UiNewY = y * invr;
//        cout << "random variables norm = " << sqrt(x*x + y*y + z*z) << endl;
//        std::this_thread::sleep_for(2000ms);
    }

    inline void random3dAngle (int i){ //TODO  this method gives wrong results
//        static const double PI = acos(-1);
//        static const double PId4 = PI / 4;
//        static const double PId8 = PI / 8;
//        double &xi = Ux(i);
//        double &yi = Uy(i);
//        double phiiNew= atan2(yi,xi) + realDissym(gen) * PId4;
//        double thetaiNew = atan2(sqrt(xi*xi+yi*yi),zi) + realDissym(gen) * PId8;
//        double sintheta = sin(thetaiNew);
//        UiNewX = sintheta * cos(phiiNew);
//        UiNewY = sintheta * sin(phiiNew);
//        UiNewZ = cos(thetaiNew);

//        cout << "random variables norm = " << sqrt(x*x + y*y + z*z) << endl;
//        std::this_thread::sleep_for(2000ms);
    }

    inline void randomAdaptive (int i){
//        double xt, yt, zt;
//        xt = realDissym(gen);
//        yt = realDissym(gen);
//        zt = realDissym(gen);
//        double r2 = xt*xt +yt*yt +zt*zt;
//
//        while (r2 > 1.0){
//            xt = realDissym(gen);
//            yt = realDissym(gen);
//            zt = realDissym(gen);
//            r2 = xt*xt +yt*yt +zt*zt;
//        }
//        double x = Uxy(2*i) + sigma * xt;
//        double y = Uxy(2*i + 1) + sigma * yt;
//        double z = Uz(i) + sigma * zt;
//        double invr = 1.0 / sqrt(x*x +y*y +z*z);
//        mi_new(0) = x * invr;
//        mi_new(1) = y * invr;
//        mi_newz   = z * invr;



//        static const double PI = acos(-1);
//        static const double PId4 = PI / 4;
//        static const double PId8 = PI / 8;
//        double &xi = Uxy(2*i);
//        double &yi = Uxy(2*i + 1);
//        double &zi = Uz(i);
//        double phiiNew   = atan2(yi,xi) + realDissym(gen)  * sigma;
//        double thetaiNew = atan2(sqrt(xi*xi+yi*yi),zi) + realDissym(gen) * sigma * 0.5;
//        double sintheta = sin(thetaiNew);
//        mi_new(0) = sintheta * cos(phiiNew);
//        mi_new(1) = sintheta * sin(phiiNew);
//        mi_newz = cos(thetaiNew);

        double x = Ux(i) + sigma * gaussianDis(gen);
        double y = Uy(i) + sigma * gaussianDis(gen);
        double invr = 1.0 / sqrt(x*x +y*y);
        UiNewX = x * invr;
        UiNewY = y * invr;
    }

};

#endif // MC_H
