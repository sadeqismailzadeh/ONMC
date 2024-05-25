//
// Created by Sadeq Ismailzadeh on ۱۷/۰۲/۲۰۲۲.
//

#include "Simulation.h"
Simulation::Simulation(Lattice2d &Lat1, const MCProperties &properties) :
        Lat1(Lat1),
        Prop(properties)
{

}

void Simulation::setupSimulation(){
    assert(NensembleExpected > 0);
    if (Prop.isNeighboursMethod){
//        rassert(properties.LatticeType == 's');
    }
//    Randomize();
    if (Prop.isLRONMethod){
        cout << "CalcInteractionsUpToNthNeighbour = " << Prop.NthNeighbour << endl;
    }
    cout << "total threads of system:" << std::thread::hardware_concurrency() << endl
         << "Num particles = " << Lat1.N <<endl
         << "NensembleExpected = " << NensembleExpected <<endl;
    Nthreads = Prop.Nthreads;
    rassert(Prop.LatticeType == 't' || Prop.LatticeType == 's' || Prop.LatticeType == 'h');
    rassert(Prop.ControlParamType == 'T' || Prop.ControlParamType == 'h');
    rassert(Nthreads >= 1);
    rassert(Nthreads <= NensembleExpected); // make this check also in Release mode

    rassert(Prop.NStabilize >= 1);
    rassert(Prop.NData >= 1);
    if (Prop.ControlParamType == 'h'){
        rassert(Prop.isFieldOn);
    }


    int simultaneusMethods = Prop.isMacIsaacMethod +
                             Prop.isNeighboursMethod +
                             Prop.isClockMethod +
                             Prop.isSCOMethod +
                             Prop.isTomitaMethod;
//    cout << "simultaneusMethods = " << simultaneusMethods << endl;
    rassert(simultaneusMethods <= 1); // only one of them must be run at the same time
    rassert(Prop.isFiniteDiffSimulation +
    Prop.isEquidistantTemperatureInCriticalRegionSimulation +
    Prop.isHistogramSimulation == 1);

    rassert(Prop.isSaveEquilibriumState + Prop.isLoadEquilibriumState <= 1);
    rassert(Prop.isNearGroupByMaxEnergyProportion + Prop.isNearGroupByDistance <= 1);

    if (Prop.isCalcSelectedBondsProbability){
        rassert(!Prop.isNearGroup);
        rassert(!Prop.isBoxes);
        rassert(!Prop.isMacIsaacMethod);
    }

    if (Prop.isNearGroup && Prop.isLRONMethod) {
        rassert(Prop.isClockMethod || Prop.isSCOMethod || Prop.isTomitaMethod);
    }

    if (Prop.isOverRelaxationMethod && Prop.isLRONMethod) {
        if (Prop.isClockMethod || Prop.isTomitaMethod){
            rassert(Prop.isNearGroup);
        } else if (Prop.isSCOMethod) {
            rassert(Prop.isNearGroup || Prop.isSCOMethodOverrelaxationBuiltIn);
        }
    }

    if (Prop.isClockMethod){
        rassert(!Prop.isNeighboursMethod);
        rassert(!Prop.isTomitaMethod);
        rassert(!Prop.isSCOMethod);
        rassert(!Prop.isMacIsaacMethod);
//        rassert(Prop.isClockMethodNearGroup || Prop.isClockMethodOriginal);
//        rassert(Prop.isClockMethodOriginal ^ Prop.isClockMethodNearGroup); // != act as XOR operator for boolean
//        if (Prop.isClockMethodWalker){
//            rassert(!Prop.isClockMethodOriginal);
//        }

    }


    if (Prop.isSCOMethod){
        rassert(!Prop.isNeighboursMethod);
        rassert(!Prop.isClockMethod);
        rassert(!Prop.isTomitaMethod);
        rassert(!Prop.isMacIsaacMethod);
        if (Prop.isSCOMethodOverrelaxationBuiltIn && Prop.isNearGroup) {
            rassert(Prop.isSCOMethodNearGroupBuiltIn);
        }
//        rassert(Prop.isSCOMethodOverrelaxationBuiltIn != Prop.isSCOMethodNearGroup);
        if (Prop.isSCOMethodNearGroupBuiltIn){
            rassert(Prop.isNearGroup);
        }
//        if (Prop.isWalkerAliasMethod){
//            rassert(!Prop.isSCOMethodOverrelaxationBuiltIn);
//        }
        rassert(Prop.SCOMethodJmaxIncreaseFactor >= 1);
    }

    if (Prop.isTomitaMethod){
        rassert(!Prop.isNeighboursMethod);
        rassert(!Prop.isClockMethod);
        rassert(!Prop.isSCOMethod);
        rassert(!Prop.isMacIsaacMethod);
//        rassert(Prop.isSCOMethodOverrelaxationBuiltIn != Prop.isSCOMethodNearGroup);
        if (Prop.isTomitaMethodNearGroupBuiltIn){
            rassert(Prop.isNearGroup);
        }
    }

    if (Prop.isSaveSampleStatData){
        rassert(Prop.isSaveSamplesToSepareteFolders);
    }


    Lat1.setSupercell();
    setNensemble();
    createFolder();

    parallel();

//    #ifdef NDEBUG
//        parallel();
//    #else
//        if (Nthreads == 1){
//            serial();
//        } else {
//            parallel();
//        }
//    #endif
}

void Simulation::parallel() {
//    tic();
    StopWatch.reset();

    cout << "Nensemble = " << Nensemble << endl;
    omp_set_num_threads(Nthreads);
    omp_set_dynamic(0);
    Eigen::setNbThreads(1);

    vector<ArrayXXd> StatDatas;
    vector<ArrayXXd> GVecs;
    vector<ArrayXXd> GConVecs;
    std::vector<MC2d> mcs;
    long elapsedCPU = 0;

    bool isParallel = false;
    bool isParallelNew = true;

    if (Prop.isNearGroup && (Prop.isNearGroupByMaxEnergyProportion || Prop.isNearGroupByDistance)
        && !Prop.isMacIsaacMethod) {
        H5Easy::File infoFile(folderpath + "/info.h5"s,
                              H5Easy::File::OpenOrCreate);
        H5Easy::dump(infoFile, "/NearGroupDistance"s, Lat1.DistanceVec1p_n,
                     H5Easy::DumpOptions(H5Easy::Compression(9)));
        H5Easy::dump(infoFile, "/Neighbours.size()"s, Lat1.Neighbours.size(),
                     H5Easy::DumpOptions(H5Easy::Compression(9)));
        H5Easy::dump(infoFile, "/R0j"s, Lat1.R0jBydistance,
                     H5Easy::DumpOptions(H5Easy::Compression(9)));
        H5Easy::dump(infoFile, "/JmaxVecCumulNormalized"s, Lat1.JmaxCumulNormalizedVecToSave,
                     H5Easy::DumpOptions(H5Easy::Compression(9)));
        H5Easy::dump(infoFile, "/JmaxByDistance"s, Lat1.JmaxByDistance,
                     H5Easy::DumpOptions(H5Easy::Compression(9)));
    }

    if (!Prop.isSaveSamplesToSepareteFolders) {
        H5Easy::File timeSeriesHdf5Ens(folderpath + "/Results.h5"s,
                                       H5Easy::File::OpenOrCreate);
        H5Easy::dump(timeSeriesHdf5Ens, "/Nensemble"s, Nensemble,
                     H5Easy::DumpOptions(H5Easy::Compression(9)));
    }

    if (Prop.isSaveEquilibriumState) {
        string filePath = folderpath + "/.."s + "/EquilibriumState_L="s + to_string(Lat1.L1) + ".h5"s;
        H5Easy::File EQStateHdf5Ens(filePath, H5Easy::File::Overwrite);
        H5Easy::dump(EQStateHdf5Ens, "/TemperatureList"s, Prop.TemperatureList,
                     H5Easy::DumpOptions(H5Easy::Compression(9)));
    }

    if (Prop.isCalcSelectedBondsProbability){
        H5Easy::File BondsProbFile(folderpath + "/NeighborsDistancesSorted.h5"s,
                                    H5Easy::File::OpenOrCreate);
        H5Easy::dump(BondsProbFile, "/R0j"s, Lat1.R0jBydistance,
                     H5Easy::DumpOptions(H5Easy::Compression(9)));
    }

//    #ifndef NDEBUG
//        if (Nthreads == 1){
//            isParallel = false;
//            isParallelNew = false;
//        }
//    #endif

    if (isParallel) {
//        for (int i = 0; i < NensembleDiv; i++) {  //TODO thread pool
//            // line below uses copy constructor which is not compatible with ofstream
////        std::vector<MC2d> mcs(Nthreads, MC2d(Lat1, Random(std::numeric_limits<int32_t>::max()), LatticeType));
//            std::vector<MC2d> mcs;
//            int64_t DefaultSeed = Prop.isFixedSeed ? Prop.seed : time(nullptr);
////            int64_t DefaultSeed = 1;
//            for (int j = 0; j < Nthreads; ++j) {
//                // line below uses move constructor
//                int64_t seed = DefaultSeed + j;
//                mcs.emplace_back(MC2d(Lat1, seed, Prop));
//                mcs[j].setParentFolderPath(folderpath);
//            }
//            #pragma omp parallel for default(none) shared(Nthreads, mcs, StatDatas, GVecs, GConVecs)
//            for (int j = 0; j < Nthreads; ++j) {
//                mcs[j].simulate();
////                #pragma omp critical
////                {
////                    StatDatas.emplace_back(mcs[j].StatisticalData.array());
////                    GVecs.emplace_back(mcs[j].GVecVsTemperature.array());
////                    GConVecs.emplace_back(mcs[j].GConVecVsTemperature.array());
////                }
//            }
//            for (int j = 0; j < Nthreads; ++j) {
//                StatDatas.emplace_back(mcs[j].StatisticalData.array());
//                GVecs.emplace_back(mcs[j].GVecVsTemperature.array());
//                GConVecs.emplace_back(mcs[j].GConVecVsTemperature.array());
//            }
//        }
    } else if (isParallelNew) {
//        std::vector<MC2d> mcs;
//        int64_t DefaultSeed = time(nullptr);
        int64_t DefaultSeed = Prop.isFixedSeed ? Prop.seed : time(nullptr);
        //TODO time(nullptr) OR chrono?
//        int64_t DefaultSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        cout << "DefaultSeed  = " << DefaultSeed << endl;
//        for (int j = 0; j < Nensemble; ++j) {
            // line below uses move constructor
//            mcs.emplace_back(MC2d(Lat1, Random(std::numeric_limits<int32_t>::max()), properties));
//            mcs[j].setParentFolderPath(folderpath);
//        }
//        #pragma omp parallel for default(none) shared(Nensemble, mcs, StatDatas, GVecs, GConVecs)
        std::vector<int> NumberIDs(Nensemble);
        std::iota (std::begin(NumberIDs), std::end(NumberIDs), 1);
        std::vector<int64_t> Seeds(Nensemble);
        std::iota (std::begin(Seeds), std::end(Seeds), DefaultSeed);
//        for (int j = 0; j < Nensemble; ++j) {
//            int64_t seed = DefaultSeed + j;
//            // TODO Or move initialization of constructor to init() function and call init() later
//            mcs.emplace_back(MC2d(Lat1, seed, Prop));
//            cout << "seed  at j = " << j <<  " = " << seed << endl;
//        }
        #pragma omp parallel for default(none) shared(Nensemble, StatDatas, DefaultSeed, GVecs, GConVecs, \
                                                      cout, mcs, Seeds,elapsedCPU)
        for (int j = 0; j < Nensemble; ++j) {
            int64_t seed = Seeds[j];
            auto MC2dSample = make_unique<MC2d>(Lat1, seed, Prop);
            #pragma omp critical (setID)
            {
                MC2dSample->setId();
                MC2dSample->init();
                MC2dSample->setParentFolderPath(folderpath);
//                mcs[j].init();
//                mcs[j].setId();
//                mcs[j].setParentFolderPath(folderpath);
            }
//            mcs[j].simulate();
            MC2dSample->simulate();
            #pragma omp critical (emplaceBackStats)
            {
//                StatDatas.emplace_back(mcs[j].StatisticalData.array());
//                GVecs.emplace_back(mcs[j].GVecVsTemperature.array());
//                GConVecs.emplace_back(mcs[j].GConVecVsTemperature.array());
                StatDatas.emplace_back(MC2dSample->StatisticalData.array());
                GVecs.emplace_back(MC2dSample->GVecVsTemperature.array());
                GConVecs.emplace_back(MC2dSample->GConVecVsTemperature.array());
                elapsedCPU += MC2dSample->TimeMCTotal;
            }
        }

    } else {
//        int64_t seed = time(nullptr);
//        for (int i = 0; i < Nensemble; i++) {
//            MC2d mc1 = MC2d(Lat1, seed, Prop);
//            mc1.setParentFolderPath(folderpath);
//            mc1.simulate();
//            cout << "run = " << i << endl;
//            StatDatas.emplace_back(mc1.StatisticalData.array());
//            GVecs.emplace_back(mc1.GVecVsTemperature.array());
//            GConVecs.emplace_back(mc1.GConVecVsTemperature.array());
//            StopWatch.elapsedSeconds();
//        }
    }

    auto StatDataMeanErr = getMeanErr(StatDatas);
    StatDataMeanErr[1](all, seq(0, 1)) = StatDataMeanErr[0](all, seq(0, 1)); //tempetature and field
    auto GVecMeanErr = getMeanErr(GVecs);  //TODO combine this and save line
    auto GConVecDataMeanErr = getMeanErr(GConVecs);
    MC2d::resetLastInstanceIndex();
    cout << "Total ";  // not very efficient way of printing total elpsed time
    StopWatch.printElapsedSecondsPrecise();

    int64_t elapsed = StopWatch.elapsedSeconds();
    save(elapsedCPU, StatDataMeanErr, StatDatas, elapsed);
    saveData(GVecMeanErr, "GVec"s);
    saveData(GConVecDataMeanErr, "GConVec"s);
}


[[maybe_unused]] void Simulation::serial() {
//    cout <<  hash<thread::id>()(this_thread::get_id()) <<endl;
//    std::mt19937 gen(clock() + hash<thread::id>()(this_thread::get_id()));
//    this_thread::sleep_for(chrono::milliseconds(2000));

//    tic();
    Nensemble = NensembleExpected;
    vector<ArrayXXd> StatDatas;
    vector<ArrayXXd> GVecs;
    vector<ArrayXXd> GConVecs;


//    mc1.setMultithreadingParams(true, Nthreads);
/*
    for (int i = 0; i < Nensemble; i++) {
        MC2d mc1 = MC2d(Lat1,time(nullptr), LatticeType);
        mc1.setSimulationParams(NStabilize, NStabilizeMinor, NData, NDecorrelate);
        mc1.setSimulationType(isFiniteDiffSimulation, Pstart, Pend, Pdecrement, PcEstimate);
        mc1.setOptionalFlags(isComputingCorrelationTime, isTakeSnapshot,isComputingCorrelationLength);
        mc1.setParentFolderPath(folderpath);
        if(isFieldOn){
            mc1.setFieldSimulationType(hfield, ControlParamType, hHat, FixedTemperatureInVaryingField);
        }


        mc1.simulate();
        cout << "run = " << i << endl;
        StatDatas.emplace_back(mc1.StatisticalData.array());
        GVecs.emplace_back(mc1.GVec.array());
        GConVecs.emplace_back(mc1.GConVec.array());
        toc();
    }
    ArrayXXd MeanX, errorX;
   auto StatDataMeanErr = getMeanErr(StatDatas);

    int64_t elapsed = static_cast<int64_t>(toc());
    save(StatDataMeanErr, StatDatas, elapsed);
    */

}

void Simulation::save(long elapsedCPU, const vector<ArrayXXd> &StatDataMeanErr, vector<ArrayXXd> VectorStatDatas,
                      int64_t elapsed) {
    string MeanXpath = folderpath + "/MeanX.csv"s;
    ofstream MeanXout(MeanXpath.c_str(), ios::out | ios::trunc);
    if (MeanXout) {
        MeanXout << StatDataMeanErr[0].format(CSVFormat);
    }
    MeanXout.close();

    string errorXpath = folderpath + "/errorX.csv"s;
    ofstream errorXout(errorXpath.c_str(), ios::out | ios::trunc);
    if (errorXout) {
        errorXout << StatDataMeanErr[1].format(CSVFormat);
    }
    errorXout.close();

    string Samplepath = folderpath + "/samples.txt"s;
    ofstream SampleXout(Samplepath.c_str(), ios::out | ios::trunc);
    if (SampleXout) {
        for (int i = 0; i < VectorStatDatas.size(); ++i) {
            SampleXout << "Sample #" << i+1 << " :" << endl << VectorStatDatas.at(i) << endl << endl;
        }
    }
    SampleXout.close();

    string DistanceVecPath = folderpath + "/DistanceVec.txt"s;
    ofstream DistanceVecout(DistanceVecPath.c_str(), ios::out | ios::trunc);
    if (DistanceVecout) {
        DistanceVecout << Lat1.DistanceVec;
    }
    DistanceVecout.close();

    string PlabelPath = folderpath + "/Plabel.txt"s;
    ofstream Plabelout(PlabelPath.c_str(), ios::out | ios::trunc);
    if (Plabelout) {
        Plabelout << Lat1.PLabel;
    }
    Plabelout.close();


    // save info
    auto [days, hours, minutes, secs] = toDHMS(elapsed);
//    long elapsedCPU = std::accumulate (begin(elapsedCPU), end(elapsedCPU), 0L,
//                                       [](int i, const MC2d& MC){ return MC.TimeMCTotal + i; });
    auto [daysCPU, hoursCPU,
    minutesCPU, secsCPU] = toDHMS(elapsedCPU);

    string filepathinfo = folderpath + "/info.txt";
    ofstream InfoOut(filepathinfo.c_str(), ios::out | ios::app);
    if (InfoOut) {
        InfoOut << "Latttice Type: " << LatticeTypeName << endl
                << "Num Paricles = " << Lat1.N << endl
                << "ensemble = " << Nensemble << endl
                << "threads = " << Nthreads << endl
                << "Stabilize = " << Prop.NStabilizeTotal << endl
                << "Data = " << Prop.NDataTotal << endl
                << "Decorrelate = " << Prop.dataTakingInterval << endl
                << "runtime = " << days << " days and " << std::setfill('0') << std::setw(2) << hours << ":"
                << std::setfill('0') << std::setw(2) << minutes << ":"
                << std::setfill('0') << std::setw(2) << secs << endl
                << "CPU time = " << daysCPU << " days and " << std::setfill('0') << std::setw(2) << hoursCPU << ":"
                << std::setfill('0') << std::setw(2) << minutesCPU << ":"
                << std::setfill('0') << std::setw(2) << secsCPU << endl
                << "CPU seconds = " << ((elapsedCPU)) << endl
                << "CPU hour = " << ((elapsedCPU)/3600) << endl
                << "runtime in seconds = " << elapsed << endl
                << "info on folder path = " << endl << simulationInfoOnFolderPath <<endl;
        if (Prop.isFieldOn && (Prop.ControlParamType == 'T')){
            InfoOut << "h = " << endl << Prop.hfield;
            cout << endl << "h = " << endl << Prop.hfield << endl;
        }
    }  else {
        cout << "file not created" <<endl;
    }
    InfoOut.close();
    // TODO save simulation type  metropolis, neighbours, Clock, Overrelaxation
    // TODO save default seed

    auto mysetw = setw(15);
    cout << std::right << setprecision(5) << std::fixed << endl;

    ostringstream MeanXWithNamesoutSStream;
    MeanXWithNamesoutSStream << std::right << setprecision(5) << std::fixed << endl;

    cout << endl;
    for (int i = 0; i < StatDataMeanErr[0].cols(); ++i) {
        MeanXWithNamesoutSStream << mysetw << i+1 ;
    }
    MeanXWithNamesoutSStream << endl;
    MeanXWithNamesoutSStream << mysetw << "T" << mysetw << "h" << mysetw << "E" << mysetw << "c" << mysetw <<
         "m" << mysetw << "X" << mysetw << "U" << mysetw <<
         "mPlane" << mysetw << "XPlane" << mysetw << "UPlane" << mysetw <<
         "Mvect" << mysetw << "mx" << mysetw << "my" << mysetw << "mz"
         << mysetw <<  "mFE" << mysetw << "XFE" << mysetw << "UFE"
         << mysetw <<  "mFEPlane" << mysetw << "XFEPlane" << mysetw << "UFEPlane"
         << mysetw << "mTpp" << mysetw << "tauM" << mysetw << "tauE" << mysetw << "tau" << mysetw << "Pacc_TOT"
         << mysetw << "Pacc_MC" << mysetw << "Pacc_OR"<< mysetw << "runtime"
         << mysetw << "t_eff"<< mysetw << "run/Step"<< mysetw << "complexity/N" << mysetw << "RND/step/N" << endl;
    // TODO print info in column (table like) fashion to be able show more info more easily
    for (int i = 0; i < StatDataMeanErr[0].rows(); ++i) {
        for (int j = 0; j < StatDataMeanErr[0].cols(); ++j) {
            MeanXWithNamesoutSStream << mysetw << StatDataMeanErr[0](i, j);
        }
        MeanXWithNamesoutSStream << endl;
    }

    MeanXWithNamesoutSStream << "error = " << endl;
    for (int i = 0; i < StatDataMeanErr[1].rows(); ++i) {
        for (int j = 0; j < StatDataMeanErr[1].cols(); ++j) {
            MeanXWithNamesoutSStream << mysetw << StatDataMeanErr[1](i, j);
        }
        MeanXWithNamesoutSStream << endl;
    }

    cout << MeanXWithNamesoutSStream.str();
    string MeanXWithNamespath = folderpath + "/MeanXWithNames.txt"s;
    ofstream MeanXWithNamesout(MeanXWithNamespath.c_str(), ios::out | ios::trunc);
    if (MeanXWithNamesout) {
        MeanXWithNamesout << MeanXWithNamesoutSStream.str();
    }
    MeanXWithNamesout.close();


//    cout << "error = " << endl<< StatDataMeanErr[1] << endl << endl;  // TODO Print this clearly similar to MeanX

    if (Prop.isHistogramSimulation){
        cout << "valid Temperature range = " << Prop.PcEstimate / (sqrt(Lat1.N * StatDataMeanErr[0](0, 4))) << endl;
    }
}

void Simulation::createFolder() {
    makeLatticeTypeName();
    makeFolderPath();
    fs::create_directories(folderpath.c_str());
    string filepathinfo = folderpath + "/infoRaw.txt";
    ofstream InfoRawOut(filepathinfo.c_str(), ios::out | ios::app);
    InfoRawOut << LatticeTypeName + " N="s + to_string(Lat1.N);
}

const string& Simulation::makeFolderPath() {
    auto tt = std::time(nullptr);
    auto tm = *std::localtime(&tt);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y.%m.%d_%H-%M-%S");
    auto DateTimeStr = oss.str();
    auto relpath = "./results/"s;

    double MCs = Prop.NDataTotal * Prop.dataTakingInterval;
    ostringstream MCsSStream;
    MCsSStream << setprecision(1) << scientific;
    MCsSStream << MCs;
    folderpath = relpath + DateTimeStr;

    simulationInfoOnFolderPath.clear();
    if (Prop.isMacIsaacMethod){
        simulationInfoOnFolderPath += "_MacIsaac"s;
    } else if (Prop.isClockMethod){
        simulationInfoOnFolderPath += "_Clock"s;
        if (Prop.isWalkerAliasMethod){
            simulationInfoOnFolderPath += "_Walker"s;
        }
    } else if (Prop.isSCOMethod){
        simulationInfoOnFolderPath += "_SCO"s;
        if (Prop.isWalkerAliasMethod){
            simulationInfoOnFolderPath += "_Walker"s;
        }
        if (Prop.isSCOMethodOverrelaxationBuiltIn){
            simulationInfoOnFolderPath += "_builtInOR"s;
        }
        if (Prop.isSCOMethodPreset){
            simulationInfoOnFolderPath += "_PS="s + to_string(Prop.SCOPresetReshuffleInterval);
        }
        std::stringstream stream;
        stream << std::fixed << std::setprecision(2) << Prop.SCOMethodJmaxIncreaseFactor;
        std::string IncreaseFactor = stream.str();
        simulationInfoOnFolderPath += "_IF="s + IncreaseFactor;
    } else if (Prop.isTomitaMethod){
        simulationInfoOnFolderPath += "_Tomita"s ;
        std::stringstream stream;
        stream << std::fixed << std::setprecision(2) << Prop.TomitaAlphaTilde;
        std::string alphaTilde = stream.str();
        simulationInfoOnFolderPath += "_a="s + alphaTilde;
    } else if (Prop.isNeighboursMethod){
        simulationInfoOnFolderPath += "_Neighbours"s;
    }

    if (Prop.isNearGroup && (Prop.isClockMethod || Prop.isSCOMethod || Prop.isTomitaMethod)) {
        if((Prop.isSCOMethod || Prop.isTomitaMethod) &&
           (Prop.isSCOMethodNearGroupBuiltIn || Prop.isTomitaMethodNearGroupBuiltIn)){
            simulationInfoOnFolderPath += "_BuiltIn"s;
        }
        simulationInfoOnFolderPath += "_NearGroup="s;
        if (Prop.isNearGroupByMaxEnergyProportion) {
            std::stringstream stream;
            stream << std::fixed << std::setprecision(3) << Prop.NearGroupMaxEnergyProportion;
            std::string NearGroupMaxEnergyProportionStr = stream.str();
            simulationInfoOnFolderPath += NearGroupMaxEnergyProportionStr;
        } else if (Prop.isNearGroupByDistance) {
            std::stringstream stream;
            stream << std::fixed << std::setprecision(3) << Prop.NearGroupDistance;
            std::string NearGroupMaxEnergyProportionStr = stream.str();
            simulationInfoOnFolderPath += NearGroupMaxEnergyProportionStr;
        } else{
            simulationInfoOnFolderPath += to_string(Prop.NthNeighbour);
        }
    }

    if (Prop.isHavingExchangeInteraction){
        std::stringstream stream;
        stream << std::fixed << std::setprecision(3) << Prop.DipolarStrengthToExchangeRatio;
        std::string DtoJRatio = stream.str();
        simulationInfoOnFolderPath += "_DtoJ="s + DtoJRatio;
    }

    if (Prop.isOverRelaxationMethod){
        simulationInfoOnFolderPath += "_OR"s + to_string(Prop.OverrelaxationSteps) + ";" + to_string(Prop.MetropolisSteps);
    }

    simulationInfoOnFolderPath += "_"s + LatticeTypeName + "_L="s + to_string(Lat1.L1) +
                 "_thread="s + to_string(Nthreads) +"_MCs="s + MCsSStream.str() + "_ens=" + to_string(Nensemble);

    if (Prop.ControlParamType == 'T'){
//        simulationInfoOnFolderPath += "_Type=T"s;
    } else if (Prop.ControlParamType == 'h'){
        simulationInfoOnFolderPath += "_Type=h"s;
    }
    if (Prop.isFieldOn && (Prop.ControlParamType == 'T')){
        simulationInfoOnFolderPath += "_FieldOn"s;
    }

    if (Prop.isFiniteDiffSimulation){
        simulationInfoOnFolderPath += "_diff"s;
    } else if (Prop.isEquidistantTemperatureInCriticalRegionSimulation){
        simulationInfoOnFolderPath += "_CEqui"s;
    } else if (Prop.isHistogramSimulation){
        simulationInfoOnFolderPath += "_Hist"s;
        std::stringstream stream;
        stream << std::fixed << std::setprecision(6) << Prop.PcEstimate;
        std::string THistogram = stream.str();
        simulationInfoOnFolderPath += "_T="s + THistogram;
    }

    if (Prop.isNeighboursMethod) {
        simulationInfoOnFolderPath += "_Cutoff"s;
    }
    folderpath += simulationInfoOnFolderPath;
    return folderpath;
}

void Simulation::makeLatticeTypeName() {
    switch (Prop.LatticeType) {
        case 't':
            LatticeTypeName = "Tr";
            break;
        case 's':
            LatticeTypeName = "Sq";
            break;
        case 'h':
            LatticeTypeName = "Ho";
            break;
        default:
            LatticeTypeName = "Other";
            break;
    }
}


vector<ArrayXXd>  Simulation::getMeanErr(vector<ArrayXXd> &datas) {
    if (datas.empty()){
//        return vector<ArrayXXd>(ArrayXXd(0));
    }

    auto Nensemble = datas.size();
    assert (Nensemble > 0);
    ArrayXXd SumX, SumX2, MeanX, MeanX2, errorX;
//    SumX = SumX2 = MeanX = MeanX2 = MatrixXd(VectorStatDatas[0]).setZero();
//    SumX = MatrixXd(datas[0]).setZero();
    SumX.setZero(datas[0].rows(), datas[0].cols());
    errorX = SumX2 = MeanX = MeanX2 = SumX;

    for (int i = 0; i < Nensemble; ++i) {
        SumX += datas.at(i);
        SumX2 += datas.at(i).square();
    }

    MeanX = SumX / Nensemble;
    MeanX2 = SumX2 / Nensemble;
    double factor = (1.0/(Nensemble-1.0));   //standard error of mean
    if (Nensemble == 1) {
        // TODO verify this
//        errorX = sqrt(MeanX2 - MeanX.square());
        errorX.setZero();
    } else {
        // Bessel's correction
        errorX = sqrt(factor * (MeanX2 - MeanX.square()));
    }

    vector<ArrayXXd> dataMeanErr;
    dataMeanErr.push_back(MeanX);
    dataMeanErr.push_back(errorX);
    return dataMeanErr;

}

void Simulation::saveData(vector<ArrayXXd> data, const string& Name) {
    string MeanXpath = folderpath + "/Mean_"s + Name + ".txt"s;
    ofstream MeanXout(MeanXpath.c_str(), ios::out | ios::trunc);
    if (MeanXout) {
        MeanXout << data[0];
    }
    MeanXout.close();

    string errorXpath = folderpath  + "/error_"s + Name + ".txt"s;
    ofstream errorXout(errorXpath.c_str(), ios::out | ios::trunc);
    if (errorXout) {
        errorXout << data[1];
    }
    errorXout.close();
}

void Simulation::setNensemble() {
    // to avoid unused threads
    if (Nthreads <= 0 || Nthreads > NensembleExpected){ //set thread num to max
        Nthreads = Nensemble = NensembleExpected;
        NensembleDiv = 1;
    } else {
        NensembleDiv = NensembleExpected / Nthreads;
        Nensemble = NensembleDiv * Nthreads;
    }
}