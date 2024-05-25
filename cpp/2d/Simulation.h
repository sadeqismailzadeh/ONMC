//
// Created by Sadeq Ismailzadeh on ۱۷/۰۲/۲۰۲۲.
//

#ifndef SRC_SIMULATION_H
#define SRC_SIMULATION_H

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
#include "MCProperties.h"
#include <highfive/H5Easy.hpp>
#include <exception>
//#include <cblas.h>

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


#include "2d/MC2d.h"
//#include "Save.h"

class Simulation{
private:
    string folderpath;
    string simulationInfoOnFolderPath;
    string LatticeTypeName;
    SimpleStopWatch StopWatch;
    void createFolder();
    void parallel();
    void setNensemble();

    [[maybe_unused]] void serial();
    void save(long elapsedCPU, const vector<ArrayXXd> &StatDataMeanErr, vector<ArrayXXd> VectorStatDatas,
              int64_t elapsed);
    void saveData(vector<ArrayXXd> data, const string& Name);
    inline Lattice2d &computeInteractions(){
        return Lat1.setSupercell().computeInteractionTensorEwald();
    }
public:
    int Nthreads;
    int Nensemble;
    int NensembleExpected;
    int NensembleDiv;

    // MC properties
    const MCProperties Prop;

    Lattice2d & Lat1;
    explicit Simulation(Lattice2d &Lat1, const MCProperties &properties);
    void setupSimulation();

    static vector<ArrayXXd> getMeanErr(vector<ArrayXXd> &datas);

    void makeLatticeTypeName();
    const string&  makeFolderPath();

};
#endif //SRC_SIMULATION_H
