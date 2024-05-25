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
#include <iomanip>
#include <sstream>
#include "Eigen/Dense"

#include <exception>
//#include <cblas.h>


//#include <blis.h>
#ifdef _OPENMP // Use omp.h if -fopenmp is used in g++
#include <omp.h>
#endif

#include "2d/MC2d.h"
#include "2d/Simulation.h"
#include "2d/Param.h"

using namespace std;
using namespace Eigen;

void paralell();
void serial(int Nthreads );
void runSize(int FunctionParamLatticeSize, int FunctionParamDataTaking, char FunctionParamMethodType = 'm',
             double FunctionParamTemperature = 0.849,  bool FunctionParamIsOverrelaxing = false, bool FunctionParamIsNearGroup = true);

char ParamLatticeType;
int ParamSystemSize;
double ParamTHistogram;
int  ParamNthreads;
int  ParamNthNeighbour;

//int realRand() {
//    static thread_local mt19937 generator(clock() + hash<thread::id>()(this_thread::get_id()));
//    uniform_real_distribution<double> distribution(0, 1);
//    return distribution(generator);
//}
//TODO assert hdf5 is working before hand by writting a dummy file
//TODO assert h5easy  is using eigen
//TODO make hdf5 dynamic link
void runSize(int FunctionParamLatticeSize, int FunctionParamDataTaking, char FunctionParamMethodType,
             double FunctionParamTemperature, bool FunctionParamIsOverrelaxing, bool FunctionParamIsNearGroup) {
//    Eigen::initParallel();
#ifdef H5_USE_EIGEN  //TODO make it assert
    cout << "highFive using eigen" <<endl;
#endif
//    if (H5Zfilter_avail(H5Z_FILTER_DEFLATE)){
//        cout << "hdf5 has deflate" <<endl;
//    }
//    if (H5Zfilter_avail(H5Z_FILTER_SZIP)){
//        cout << "hdf5 has Szip" <<endl;
//    }

//    #ifdef PARAMS_ARE_SET_IN_COMPILE_TIME
    ParamLatticeType = 't';
    ParamSystemSize = FunctionParamLatticeSize;
    ParamTHistogram = FunctionParamTemperature;
    ParamNthreads = 8;
    ParamNthNeighbour = 0;

    const int NensembleExpected = 100;
//    #else
//        cout << "Latticetype (t: triangular, s: square, h: honeycomb) = ";
//        cin >> ParamLatticeType;
//        cout << "Lattice size:  ";
//        cin >> ParamSystemSize;
//        cout << "T histogram = ";
//        cin >> ParamTHistogram;
//        cout << "Num threads =  ";
//        cin >> ParamNthreads;
//        cout << "Nth Neighbour =  ";
//        cin >> ParamNthNeighbour;
//    #endif

//    ParamLatticeType = 't';
    rassert(ParamLatticeType == 't' || ParamLatticeType == 's' || ParamLatticeType == 'h');
    int np1max, np2max;

    if (ParamLatticeType == 't'){
        np2max = np1max = ParamSystemSize;
    } else if (ParamLatticeType == 's') {
        np2max = np1max = ParamSystemSize/2;
    }

    cout << "Lattice size = " << ParamSystemSize <<endl;
    cout << "Lattice type = " << ParamLatticeType <<endl;
    cout << "T histogram  = " << ParamTHistogram <<endl;
    cout << "Num threads  = " << ParamNthreads <<endl;

    int alpha ;
    Matrix2d t;
    MatrixXd d;
    VectorXi PrimitivePLabel;
    const double pi = acos(-1);
    bool isHoneycombFromTriangular = true;
    if (ParamLatticeType == 't'){
        // set lattice
        alpha = 1;
        t << 1., 1./2.,
                0, sqrt(3)/2.;
        d.setZero(2, alpha);
        d << 0,
                0;
        PrimitivePLabel.setZero(alpha);
        PrimitivePLabel << 1;

    } else if (ParamLatticeType == 's') {
        // set lattice
        alpha = 4;
        t << 2., 0.,
                0., 2.;
        d.setZero(2, alpha);
        d << 0., 0., 1., 1.,
                0., 1., 1., 0.;
        PrimitivePLabel.setZero(alpha);
        PrimitivePLabel << 1, 2, 3, 4;
    } else if (ParamLatticeType == 'h') {
        // set lattice
        if (isHoneycombFromTriangular){
            alpha = 9;
            double a = 3 / sqrt(3);
            Vector2d a1, a2;
            a1.setZero();
            a2.setZero();
            a1(0) = a;
            a2(0) = a / 2;
            a2(1) = a * sqrt(3) / 2;

            t(all,0) = 3 * a1;
            t(all,1) = 3 * a2;

            d.setZero(2, alpha);

            d(all,0) = (a1 + a2) / 2.;
            d(all,1) = d(all,0) + a1;
            d(all,2) = d(all,0) + 2*a1;
            d(all,3) = d(all,0) + a2;
            d(all,4) = d(all,0) + a2 + a1;
            d(all,5) = d(all,0) + a2 + 2*a1;
            d(all,6) = d(all,0) + 2*a2;
            d(all,7) = d(all,0) + 2*a2 + a1;
            d(all,8) = d(all,0) + 2*a2 + 2*a1;

            PrimitivePLabel.setZero(alpha);
            PrimitivePLabel <<
                            1, 2, 3,
                    3, 1, 2,
                    2, 3, 1;
            PrimitivePLabel.array() -= 1;
        } else {
//            alpha = 6;
//            t << 3.,         3./2.,
//                    0.,     -3*sqrt(3)/2.;
//            d.setZero(2, alpha);
//            d <<
//              1./2.,       1.,   0.5,           -0.5,          -1.,           -0.5,
//                    sqrt(3)/2.,  0.,   -sqrt(3)/2.,  -sqrt(3)/2.,    0.,       sqrt(3)/2.;
//            PrimitivePLabel.setZero(alpha);
//            PrimitivePLabel << 1, 2, 3, 4, 5, 6;

            alpha = 6;
            double a = 1;
            Vector2d d1, d2;
            d1 << 0., a;
            d2 << a*cos(pi/6), a*sin(pi/6);

            d.setZero(2, alpha);
            d(all,0) = d1;
            d(all,1) = d2;
            d(all,2) = d2 - d1;
            d(all,3) = - d1;
            d(all,4) = - d2;
            d(all,5) = - d2 + d1;

            t(all,0) = 3*d(all,1);
            t(all,1) = 3*d(all,2);
            PrimitivePLabel.setZero(alpha);
            PrimitivePLabel << 1, 2, 3, 4, 5, 6;

        }

    }

    MCProperties Prop;
    if (ParamLatticeType == 't'){
//        Prop.PcEstimate = 0.849;
        Prop.PcEstimate = ParamTHistogram;
        Prop.PmaxCritical = 0.99;
        Prop.PminCritical = 0.8;
//        PcEstimate = 0.39;   // field on
    } else if (ParamLatticeType == 's'){
//        Prop.PcEstimate = 0.76;
        Prop.PcEstimate = ParamTHistogram;       // neighbours method
        Prop.PmaxCritical = 1.2;
        Prop.PminCritical = 0.8;
//        properties.PcEstimate = 0.45;  //field on

    } else if (ParamLatticeType == 'h'){
        Prop.PcEstimate = ParamTHistogram;
        Prop.PmaxCritical = 0.68;
        Prop.PminCritical = 0.21;
    }

    const double PI = acos(-1);
    // lattice properties
    Prop.LatticeType = ParamLatticeType;



    //field and temperature properties
    Prop.hfield << 0.1,0,0;
    Prop.hHat << 1,0,0;
    Prop.ControlParamType = 'T';
    Prop.Pstart = 0.85;
    Prop.Pend = 0.55;
    Prop.Pdecrement = 0.02;
    Prop.FixedTemperatureInVaryingField = 0.1;

    // field on or off
    Prop.isFieldOn = false;
    //simulation type (simulate on a range of temperatures or only one temperature)
    Prop.isFiniteDiffSimulation = false ;  //TODO to simlation type string variable
    Prop.isEquidistantTemperatureInCriticalRegionSimulation = false;
    Prop.isHistogramSimulation = true;
    // optionals
    Prop.isSlowlyCoolingEquilibration = false;    //TODO fix setProbclock Problem in stabilize minor
    Prop.isComputingAutoCorrelationTime = false;
    Prop.isComputingCorrelationLength = false;
    Prop.isTakeSnapshot = false;
    Prop.isSaveEquilibrationTimeSeries = true;
    Prop.isSaveDataTakingTimeSeriesInHistogramSimulation = true;
    Prop.isSaveSampleStatData = true;
    Prop.isSaveSamplesToSepareteFolders = true;
    Prop.isDipolesInitializedOrdered = false;
    Prop.InitialAlignment = PI/3;
    Prop.isNoSelfEnergy = true; // TODO self energy is not working for LRON methods yet. it neads  considering ith particle at the end of nieghboursVec
    Prop.isStoreNeighboursForAllPariticles = true;
    Prop.isFixedSeed = false;

    Prop.seed = 24;
    //Prop.isGroundStateFerromagnetic = true; //for simulation of exchange with dipolar with D/J =0.1 on square lattice
    //monte carlo methods (overrelaxation is optional for MacIssac and Clock, doesnt work on clack original)
    Prop.isOverRelaxationMethod = false;
    Prop.isNearGroup = true;
    Prop.isWalkerAliasMethod = true;  // TODO Non walker is not working correctly right npw

    Prop.isMacIsaacMethod = false;

    Prop.isNeighboursMethod = false;

    Prop.isClockMethod = false;

    Prop.isSCOMethod = false;
    Prop.isSCOMethodOverrelaxationBuiltIn = false;
    Prop.isSCOMethodNearGroupBuiltIn = false;
    Prop.isSCOMethodPreset= false;
    Prop.SCOMethodJmaxIncreaseFactor = 1; //TODO

    Prop.isTomitaMethod = false;
    Prop.isTomitaMethodNearGroupBuiltIn = false;
    Prop.TomitaAlphaTilde = 0.7;

    Prop.NthNeighbour = ParamNthNeighbour;
    Prop.isNearGroupByMaxEnergyProportion = true;  // TODO CRITICAL it fails on assertion. fix this
    Prop.NearGroupMaxEnergyProportion = 0.95;
    // optional exchange interaction
    Prop.isHavingExchangeInteraction = false;    // ALERT: for now only use with metroplois or method that use walker alias
    Prop.isExchangeInteractionCombinedWithDij = false;
    Prop.DipolarStrengthToExchangeRatio = 0.1;

    if (FunctionParamMethodType == 'm') {
        Prop.isMacIsaacMethod = true;
        Prop.isClockMethod = false;
        Prop.isSCOMethod = false;
        Prop.isTomitaMethod = false;
    } else if (FunctionParamMethodType == 'c'){
        Prop.isMacIsaacMethod = false;
        Prop.isClockMethod = true;
        Prop.isSCOMethod = false;
        Prop.isTomitaMethod = false;
    } else if (FunctionParamMethodType == 's'){
        Prop.isMacIsaacMethod = false;
        Prop.isClockMethod = false;
        Prop.isSCOMethod = true;
        Prop.isTomitaMethod = false;
    } else if (FunctionParamMethodType == 't'){
        Prop.isMacIsaacMethod = false;
        Prop.isClockMethod = false;
        Prop.isSCOMethod = false;
        Prop.isTomitaMethod = true;
    } else {
        rassert((false, "no matching method"));
    }

    if (Prop.isNearGroup && (Prop.isTomitaMethod ||  Prop.isSCOMethod)){
        Prop.isTomitaMethodNearGroupBuiltIn = true;
        Prop.isSCOMethodNearGroupBuiltIn = true;
    }

    if (Prop.isTomitaMethodNearGroupBuiltIn || Prop.isSCOMethodNearGroupBuiltIn){
        Prop.isNearGroup = true;
    }
    Prop.isLRONMethod = Prop.isNeighboursMethod || Prop.isClockMethod || Prop.isTomitaMethod;

    // data taking
    if (Prop.isMacIsaacMethod){
        Prop.MetropolisSteps = 1;
        Prop.OverrelaxationSteps = 10;
    } else {  // Overrelaxed (Clock, SCO, Tomita)
        Prop.MetropolisSteps = 1;
        Prop.OverrelaxationSteps = 10;
    }

    if (Prop.isOverRelaxationMethod){
        Prop.TotalSteps = Prop.MetropolisSteps + Prop.OverrelaxationSteps;
    } else {
        Prop.MetropolisSteps = 1;  // TODO print metropolis steps in info when not over-relaxing
        Prop.TotalSteps = Prop.MetropolisSteps;
    }
//    Prop.NStabilize = max((FunctionParamDataTaking / 10) / Prop.TotalSteps, 1);
//    Prop.NData = max(FunctionParamDataTaking / Prop.TotalSteps, 1);
    Prop.NStabilize = FunctionParamDataTaking;
    Prop.NData = 4;           // N_MC_Step = NData * dataTakingInterval
    Prop.NStabilizeMinor = 1000 / Prop.TotalSteps;
    Prop.NDataTotal = Prop.NData * Prop.TotalSteps;
    Prop.NStabilizeTotal = Prop.NStabilize * Prop.TotalSteps;
//    Prop.NData = ipow(2,20);           // N_MC_Step = NData * dataTakingInterval
    Prop.dataTakingInterval = 1;  // free to choose this interval in whatever way the
    // most convenient (Newman p.69)

//    Prop.CalcInteractionsUpToNthNeighbour = ceil(4*sqrt(ParamSystemSize));

    // Ensemble

    Prop.Nthreads = ParamNthreads;

    Lattice2d Lat1(alpha, t, d, np1max, np2max, ParamLatticeType , PrimitivePLabel,
                   isHoneycombFromTriangular, Prop);
    Simulation sim(Lat1, Prop);
    sim.NensembleExpected = NensembleExpected;
    sim.setupSimulation();

}

int main(){
    vector<char> methodVec{'t'};
    for (char method: methodVec){
        // vector<int> sizeVec{16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 88, 96, 112, 128, 144, 160, 196, 226};
        vector<int> sizeVec{16,20,24};
        // vector<double> TempVec(sizeVec.size(), 0.849);
        // vector<double> TempVec{
        // 0.9251,
        // 0.9081,
        // 0.8952,
        // 0.8887,
        // 0.8845,
        // 0.8818,
        // 0.8806,
        // 0.875,
        // 0.8709,
        // 0.8668,
        // 0.8698,
        // 0.8646,
        // 0.8643,
        // 0.8624,
        // 0.8593,
        // 0.858,
        // 0.8577,
        // 0.8565,
        // 0.8556,
        // 0.8543};

        vector<double> TempVec{
                0.848,
                0.849,
                0.850};
//        vector<double> TempVec{
//                0.847};
//        std::reverse(TempVec.begin(), TempVec.end());
        //        rassert(sizeVec.size() == TempVec.size());
        vector<int> DataTakingVec(sizeVec.size(), 100);
        // for (int i = 0; i < DataTakingVec.size(); ++i) {
        //       DataTakingVec[i] = bounded((pow(sizeVec[0],4.5) / pow(sizeVec[i],4.5)) * 30'000'000.,  1'000'000., 1000.);
        // }
        cout << "DataTakingVec = " <<endl;
        for (int i : DataTakingVec) {
            cout << i <<endl;
        }
        for (auto Temp : TempVec){
            for (int i = 0; i < DataTakingVec.size(); ++i) {
                runSize(sizeVec[i], DataTakingVec[i], method, Temp);
            }
        }
    }
}

int main2(int argc, char** argv) {
    cout << almostEquals(0, 1.0001) << endl;
    long double a;
    return 0;
}

//void paralell(int Nthreads){
//TODO
//}

