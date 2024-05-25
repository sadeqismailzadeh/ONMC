//
// Created by Sadeq Ismailzadeh on ۳۰/۰۷/۲۰۲۲.
//

#ifndef SRC_MCPROPERTIES_H
#define SRC_MCPROPERTIES_H
#include "Eigen/Dense"

class MCProperties {
private:

public:
//    MCProperties() = default;
//    MCProperties(MCProperties const &) = default;
//    MCProperties & operator= (MCProperties const &) = default;


    // lattice properties
    char LatticeType;

    // data taking
    int NStabilize;
    int NStabilizeMinor;
    int NData;
    int dataTakingInterval;
    int NStabilizeTotal;
    int NDataTotal;

    //field and temperature properties
    Eigen::Vector3d hfield;
    Eigen::Vector3d hHat;
    char ControlParamType;
    double PcEstimate;          // for non equidistant temperature in critical region
    double PmaxCritical;        // for equidistant temperature in critical region
    double PminCritical;
    double Pstart;
    double Pend;
    double Pdecrement;
    double FixedTemperatureInVaryingField;

    // simulation options
    bool isFieldOn; //TODO organize variables like the ones in prodMC
    bool isFiniteDiffSimulation;
    Eigen::VectorXd TemperatureList;
    bool isEquidistantTemperatureInCriticalRegionSimulation;
    bool isHistogramSimulation;
    bool isSlowlyCoolingEquilibration;
    bool isComputingAutoCorrelationTime;
    bool isComputingCorrelationLength;
    bool isTakeSnapshot;
    bool isSaveEquilibrationTimeSeries;
    bool isSaveDataTakingTimeSeriesInHistogramSimulation;
    bool isSaveSampleStatData;
    bool isMacIsaacMethod;
    bool isNeighboursMethod;
    bool isOverRelaxationMethod;
    bool isClockMethod;
    bool isNearGroup;
    bool isBoxes;
    bool isMergeNearBoxes;
    int NumberOfMerges;
    int LBox;
    bool isCalcSelectedBondsProbability;
    bool isClockMethodOriginal;
    bool isWalkerAliasMethod;
    bool isSCOMethod;
    bool isSCOMethodOverrelaxationBuiltIn;
    bool isSCOPureOverrelaxation;
    bool isSCOMethodNearGroupBuiltIn;
    bool isSCOMethodPreset;
    int SCOPresetReshuffleInterval;
    double SCOMethodJmaxIncreaseFactor;
    bool isTomitaMethod;
    bool isTomitaMethodNearGroupBuiltIn;
    bool isSaveEquilibriumState;
    bool isLoadEquilibriumState;
    bool isComplexityAccept;
    double TomitaAlphaTilde;
//    bool isNearGroup;
    bool isLRONMethod;  //  long ranged O(N) method = click or sco or tomita
    bool isDipolesInitializedOrdered;
    double InitialAlignment;
    bool isNoSelfEnergy;
    bool isStoreNeighboursForAllPariticles;
    bool isFixedSeed;
    bool isSaveSamplesToSepareteFolders;
    long seed;
    bool isHavingExchangeInteraction;
    bool isExchangeInteractionCombinedWithDij;
    double DipolarStrengthToExchangeRatio;
    int NthNeighbour;
    bool isNearGroupByMaxEnergyProportion;
    double NearGroupMaxEnergyProportion;
    bool isNearGroupByDistance;
    double NearGroupDistance;
    int MetropolisSteps;
    int OverrelaxationSteps;
    int TotalSteps;

    // ensemble an threads
    int Nthreads;
    int Nensemble;
};


#endif //SRC_MCPROPERTIES_H
