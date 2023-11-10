//
// Created by Sadeq Ismailzadeh on ۰۵/۰۱/۲۰۲۲.
//

#ifndef SRC_LATTICE2D_H
#define SRC_LATTICE2D_H

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
#include <thread>
#include <iomanip>
#include <sstream>
#include <cfloat>
//#include <quadmath.h>
//#include <omp.h>

#include "utils.h"
#include "Eigen/Dense"
#include "MCProperties.h"

using namespace std;
using namespace Eigen;

class Lattice2d {
private:
    // ------------input----------
    int alpha; // num particle in primitive cell  TODO remove alpha nd use d.row(o).size()
    Matrix2d t; // (size = 2*2) primitive translation vectors
    MatrixXd d; // (size = 2*alpha) position of particles in primitive cell
    int np1Max;
    int np2Max; // primitive cells ar translated as r = np1*t1 + np2*t2 to construct supercell
    VectorXi PrimitivePLabel; // particle labels in primitive cell (size = alpha)
    const char LatticeType;
    const MCProperties Prop;

public:
    //-----------output-----------
    Matrix2d T; // supercell translation vectors  e.g.  T1 = np1Max *t1    (size=2*2)
    int N; // num particles in supercell  TODO make const
    int Nbase; // num particles in supercell TODO make const
    int L1; //
    int L2; //
    MatrixXd D; //position of particles in supercell (size = 2*N)
    MatrixXd Dbase; //position of particles in supercell (size = 2*N)
    MatrixXd G; // supercell reciprocal vectors
    double A; // supercell area
    VectorXi PLabel; //label of particles in supercell to organize them as sublattices (size = N)
    VectorXi PSubLattice; //label of particles in supercell to organize them as sublattices (size = N)
    VectorXi PLabelbase; //label of particles in supercell to organize them as sublattices (size = N)
//    MatrixXd J; //interaction tensor in XY plane
    MatrixXd Jxx; // Jxx element of interaction tensor. other elments are zero e.g. Jxz = Jxy = 0
    MatrixXd Jyy; // Jxx element of interaction tensor. other elments are zero e.g. Jxz = Jxy = 0
    MatrixXd Jxy; // Jxx element of interaction tensor. other elments are zero e.g. Jxz = Jxy = 0
    MatrixXd Jzz; // Jxx element of interaction tensor. other elments are zero e.g. Jxz = Jxy = 0

    VectorXd Jxx1p; // Jxx element of interaction tensor of 0th particle with others.
    VectorXd Jyy1p; // Jxx element of interaction tensor of 0th particle with others.
    VectorXd Jxy1p; // Jxx element of interaction tensor of 0th particle with others.
    VectorXd Jzz1p; // Jxx element of interaction tensor of 0th particle with others.

    VectorXd Jxxn; // Jxx element of interaction tensor of neighbors of 0th particle
    VectorXd Jyyn; // Jxx element of interaction tensor of neighbors of 0th particle
    VectorXd Jxyn; // Jxx element of interaction tensor of neighbors of 0th particle
    VectorXd Jzzn; // Jxx element of interaction tensor of neighbors of 0th particle
    MatrixXd JxxMatn; // Jxx element of interaction tensor of neighbors of 0th particle
    MatrixXd JyyMatn; // Jxx element of interaction tensor of neighbors of 0th particle
    MatrixXd JxyMatn; // Jxx element of interaction tensor of neighbors of 0th particle

    VectorXd Wxx; // Macisaac version of J
    VectorXd Wyy; // Macisaac version of J
    VectorXd Wxy; // Macisaac version of J
    VectorXd Wzz; // Macisaac version of J


    VectorXd Wxx1p; // Macisaac version of J
    VectorXd Wyy1p; // Macisaac version of J
    VectorXd Wxy1p; // Macisaac version of J
    VectorXd Wzz1p; // Macisaac version of J

    VectorXd Wxx11; // Macisaac version of J
    VectorXd Wyy11; // Macisaac version of J
    VectorXd Wxy11; // Macisaac version of J
    VectorXd Wzz11; // Macisaac version of J

    VectorXd Wxx12; // Macisaac version of J
    VectorXd Wyy12; // Macisaac version of J
    VectorXd Wxy12; // Macisaac version of J
    VectorXd Wzz12; // Macisaac version of J

    VectorXd Wxx21; // Macisaac version of J
    VectorXd Wyy21; // Macisaac version of J
    VectorXd Wxy21; // Macisaac version of J
    VectorXd Wzz21; // Macisaac version of J

    VectorXd Wxx22; // Macisaac version of J
    VectorXd Wyy22; // Macisaac version of J
    VectorXd Wxy22; // Macisaac version of J
    VectorXd Wzz22; // Macisaac version of J
//    MatrixXd J3d;
//    Matrix<int, Dynamic, Dynamic, RowMajor> J3d;
//    MatrixXf J3df;
    VectorXd DistanceVec;   // store distances between particles
    VectorXd DistanceVec1p;   // store distances between particles
    VectorXi Neighbours;   // store distances between particles
    MatrixXi NeighboursMat;   // store distances between particles
    VectorXi NearestNeighbours;   // store distances between particles

    VectorXd R0jBydistanceFar;   // store distances between particles
    VectorXi PByDistanceFar;   // store distances between particles
    VectorXi XPByDistanceFar;   // store distances between particles
    VectorXi YPByDistanceFar;   // store distances between particles
    VectorXd R0jBydistance;   // store distances between particles
    VectorXi PByDistance;   // store distances between particles
    VectorXi XPByDistance;   // store distances between particles
    VectorXi YPByDistance;   // store distances between particles
    VectorXd Jmax1p;   // store distances between particles
    VectorXd JmaxByDistance;   // store distances between particles


    VectorXi XPbyIndex;   // store distances between particles
    VectorXi YPbyIndex;   // store distances between particles
    VectorXi Xn0;
    VectorXi Yn0;

    MatrixXi MapR;          // map(i,j) = k maps distance of particles i,j to DistanceVec(k)
    VectorXi MapR0;          // MacIsaac version of map(0,j) = k maps distance of particles 0,j to DistanceVec(k)
    VectorXi MapR0Sub11;          // MacIsaac version of map(0,j) = k for honeycomb
    VectorXi MapR0Sub12;          // MacIsaac version of map(0,j) = k for honeycomb
    VectorXi MapR0Sub21;          // MacIsaac version of map(0,j) = k for honeycomb
    VectorXi MapR0Sub22;          // MacIsaac version of map(0,j) = k for honeycomb
    VectorXi CounterR;      // counts number of repeated distances in DistanceVec
    Matrix3d Jself;
    const bool isHoneycombFromTriangular;

    explicit Lattice2d(const int alpha, const Matrix2d &t,
                       const MatrixXd &d, const int np1Max, const int np2Max,
                       const char LatticeType,
                       const VectorXi &primitivePLabel,
                       const bool isHoneycombFromTriangular,
                       const MCProperties Prop);
    Lattice2d(const Lattice2d&) = default;
    Lattice2d(Lattice2d &&) = delete;
    Lattice2d& setSupercell();

    static tuple<int, int, bool> get_in2Extremes(const MatrixXd& T, double Rcut, int in1);
    static ArrayXXi get_in2MinMaxArray(const MatrixXd &T, double Rcut);
    Lattice2d& computeInteractionTensorEwald();
    Lattice2d& computeInteractionTensorEwald1p();
    void computeInteractionTensorDirect();
    double getLatticeConstant();
    MatrixXd getMadelung();
    MatrixXd getMadelung1p();
    void checkJfinite();
    void sortLocations(MatrixXd &D1, VectorXi &PLabel1);
    void printD();
    void setWvectors();
    void setWvectors1p();
    void setWvectorsHoneycomb();
    std::pair<int, int> getL1L2(MatrixXd &D1);
    void howSymmetric();

    void calcDistanceVec();
    void calcDistanceVec1p();
    void calcNeighbours(int n);
    void calcNeighboursMat();
    void setJNeighbors();
    void setJNeighborsEwald();
    double getMinDistance(int i, int j) const;
    void setMapR0();
    std::pair<int, int> periodicParticle(int x, int y, int x0, int y0) const;
    int periodicParticle(int p, int pBase) const;
    std::pair<int, int> unperiodicParticle(int x, int y, int x0, int y0) const;
    int unperiodicParticle(int p, int pBase) const;
    Vector2d getMinDistanceVec(int i, int j) const;

    void generateHoneycomb();
    void setPAndRijAndJmaxByDistance();
    void setPAndRijByDistanceFar();
    void setXYPByIndex();
    void decreaseDipolarStrength();
    void calcNearestNeighbours();
    void setJmax1p();
    void calcNeighboursbyNearGroupEnergyProportion(double proportion);
};


#endif //SRC_LATTICE2D_H
