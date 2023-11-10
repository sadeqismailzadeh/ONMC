//
// Created by Sadeq Ismailzadeh on ۰۵/۰۱/۲۰۲۲.
//
#include <limits>
#include "2d/Lattice2d.h"
#include "utils.h"

int main() {
//    int alpha = 4;
//    Matrix2d t;
//    t << 1, 0,
//            0, 1;
//    MatrixXd d;
//    d.setZero(2, alpha);
//    d << 0.25, 0.25, 0.75, 0.75,
//            0.25, 0.75, 0.25, 0.75;
//
//    int np1Max = 16;
//    int np2Max = 16;
//    t *= 2;
//    d *= 2;
//    VectorXi PrimitivePLabel;
//    PrimitivePLabel.setLinSpaced(4, 0, 3);

    int alpha = 1;
    Matrix2d t;
    t << 1., 1./2.,
         0, sqrt(3)/2.;

    MatrixXd d;
    d.setZero(2, alpha);
    d << 0,
         0;

    int np1Max = 64;
    int np2Max = 64;
    VectorXi PrimitivePLabel(1);
    PrimitivePLabel << 1;

    Lattice2d inter1(alpha, t, d, np1Max, np2Max, PrimitivePLabel);
    inter1.setSupercell();
//    tie(in2Min, in2Max, Is_Any_in2_found) = Interactions::get_in2Extremes(inter1.T,100, 21);
//    ArrayXXd in2MinMaxArray = Interactions::get_in2MinMaxArray(inter1.T, 30);
    tic();
    inter1.computeInteractionTensorEwald();
//    inter1.computeInteractionTensorDirect();
    toc();
    cout << inter1.T << endl << endl;
    cout << inter1.N << endl << endl;
//    cout << inter1.D << endl << endl;
//    cout << "here" << endl;
    cout.precision(20);
    cout << inter1.getMadelung() << endl << endl;
    cout << inter1.getMadelung1p() << endl << endl;


    const Lattice2d& inter2(inter1);
    cout.precision(3);
    cout << inter2.J(seqN(0,9), seqN(0,9))/2. << endl;
    cout.precision(15);
//    cout << exp(-80) <<endl;

//    ofstream JoutStream("J.txt", ios::out | ios::trunc);
//    JoutStream.precision(4);
//    JoutStream << inter1.J;
//    inter1.checkJfinite();
    toc();
//    cin.get();
}
