//
// Created by Sadeq Ismailzadeh on ۰۵/۰۱/۲۰۲۲.
//

#include "2d/direct/Interactions.h"
#include "utils.h"

int main() {
    int alpha = 4;
    Matrix2d t;
    t << 1, 0,
            0, 1;
    MatrixXd d;
    d.setZero(2, alpha);
    d << 0.25, 0.25, 0.75, 0.75,
            0.25, 0.75, 0.25, 0.75;

    int np1Max = 16;
    int np2Max = 16;
    t *= 2;
    d *= 2;
    VectorXi PrimitivePLabel;
    PrimitivePLabel.setLinSpaced(4, 0, 3);

//    int alpha = 1;
//    Matrix2d t;
//    t << 1/2., 1,
//         sqrt(3)/2., 0;
//    MatrixXd d;
//    d.setZero(2, alpha);
//    d << 0,
//         0;
//
//    int np1Max = 5;
//    int np2Max = 5;
//    VectorXi PrimitivePLabel;
//    PrimitivePLabel.setLinSpaced(1, 0, 0);

    Interactions inter1(alpha, t, d, np1Max, np2Max, PrimitivePLabel);
    inter1.setSupercell();
//    tie(in2Min, in2Max, Is_Any_in2_found) = Interactions::get_in2Extremes(inter1.T,100, 21);
//    ArrayXXd in2MinMaxArray = Interactions::get_in2MinMaxArray(inter1.T, 30);
    tic();
    inter1.computeInteractionTensor();
    toc();
    cout << inter1.CtrlParamAbs << endl << endl;
    cout << inter1.N << endl << endl;
    cout << inter1.D << endl << endl;
//    cout << "here" << endl;
    cout << inter1.getMadelung() << endl << endl;
//    cout << inter1.J(seqN(0,9), seqN(0,9))/2. << endl;
    toc();

}