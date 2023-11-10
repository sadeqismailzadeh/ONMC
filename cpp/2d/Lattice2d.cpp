//
// Created by Sadeq Ismailzadeh on ۰۵/۰۱/۲۰۲۲.
//

#include "Lattice2d.h"


Lattice2d::Lattice2d(const int alpha, const Matrix2d &t,
                     const MatrixXd &d, const int np1Max, const int np2Max,
                     const char LatticeType,
                     const VectorXi &primitivePLabel,
                     const bool isHoneycombFromTriangular,
                     const MCProperties Prop) :
                       alpha(alpha), t(t), d(d), np1Max(np1Max), np2Max(np2Max),
                       PrimitivePLabel(primitivePLabel), N(alpha*np1Max*np2Max),
                       LatticeType(LatticeType),
                       isHoneycombFromTriangular(isHoneycombFromTriangular),
                       Prop(Prop)
{
    if (isHoneycombFromTriangular && LatticeType == 'h'){
        Nbase = alpha*np1Max*np2Max;
        N = 0;
    } else {
        Nbase = alpha*np1Max*np2Max;
        N = alpha*np1Max*np2Max;
    }

}

Lattice2d& Lattice2d::setSupercell() {
//    N = np1Max * np2Max * alpha;
    rassert(LatticeType == 't' || LatticeType == 's' || LatticeType == 'h');
    if (LatticeType != 'h'){
//        assert(L*L == N);
    }
    T = t * (Matrix2d() << np1Max, 0,
            0, np2Max).finished();

    if (isHoneycombFromTriangular && LatticeType == 'h'){
        Dbase.setZero(2, Nbase);
        PLabelbase.setZero(Nbase);
        int ips = 0; //index of particles in supercell
        for (int np1 = 0; np1 < np1Max; ++np1) {
            for (int np2 = 0; np2 < np2Max; ++np2) {
                Vector2d r;
                r = np1 * t(all, 0) + np2 * t(all, 1);
                for (int ipp = 0; ipp < alpha; ++ipp) {
                    Dbase(all, ips) = d(all, ipp) + r;
                    PLabelbase(ips) = PrimitivePLabel(ipp);
                    ips++;
                }
            }
        }
        sortLocations(Dbase, PLabelbase);
        tie(L1, L2) = getL1L2(Dbase);
        generateHoneycomb();
    } else {
        D.setZero(2, N);
        PLabel.setZero(N);
        int ips = 0; //index of particles in supercell
        for (int np1 = 0; np1 < np1Max; ++np1) {
            for (int np2 = 0; np2 < np2Max; ++np2) {
                Vector2d r;
                r = np1 * t(all, 0) + np2 * t(all, 1);
                for (int ipp = 0; ipp < alpha; ++ipp) {
                    D(all, ips) = d(all, ipp) + r;
                    PLabel(ips) = PrimitivePLabel(ipp);
                    ips++;
                }
            }
        }
        sortLocations(D, PLabel);
        tie(L1, L2) = getL1L2(D);
    }

    double pi = acos(-1);
    G = 2 * pi * T.inverse().transpose();
    A = ((Vector3d() << T(all,0) , 0).finished().cross(
            (Vector3d() << T(all,1), 0).finished())).norm();
    cout << "A = " << A <<endl;
    Jself.setZero();
    if (Prop.isNeighboursMethod || Prop.isClockMethod || Prop.isSCOMethod || Prop.isTomitaMethod){
        setXYPByIndex();
    }
    if (Prop.isNeighboursMethod){
        calcDistanceVec1p();
        calcNeighbours(Prop.NthNeighbour);
    }
    if(Prop.isComputingCorrelationLength){
        calcDistanceVec();
        setMapR0();
    }
    if (Prop.isClockMethod || Prop.isSCOMethod || Prop.isTomitaMethod){
        computeInteractionTensorEwald1p();
        setJmax1p();
        calcDistanceVec1p();
        setPAndRijAndJmaxByDistance();
        if (Prop.isNearGroupByMaxEnergyProportion){
            calcNeighboursbyNearGroupEnergyProportion(Prop.NearGroupMaxEnergyProportion);
        } else {
            calcNeighbours(Prop.NthNeighbour);
        }
        setPAndRijByDistanceFar();
//        computeInteractionTensorEwald1p();
//        setJNeighborsEwald();
//        computeInteractionTensorEwald();

    }
    if (Prop.isHavingExchangeInteraction){
        calcDistanceVec1p();
        calcNearestNeighbours();
    }

    if (Prop.isNeighboursMethod) {
        setJNeighbors();
//        Lat1.computeInteractionTensorEwald();
    } else if (Prop.isClockMethod || Prop.isSCOMethod || Prop.isTomitaMethod){
        setJNeighborsEwald();
    } else if (Prop.isMacIsaacMethod && Prop.LatticeType != 'h'){
        computeInteractionTensorEwald1p();
    } else {
        computeInteractionTensorEwald();
    }
    return *this;
}

tuple<int, int, bool> Lattice2d::get_in2Extremes(const MatrixXd &T, double Rcut, int in1) {
// finds maximum and minimum of in2 for fixxed in1 such that |in1*T1 + in2*T2| < Rcut
// if in1 is so large such that |in1*T1 + in2*T2| > Rcut for every in2, thirs return value will be false
    double T1T2 = T(all, 0).dot(T(all, 1));
    double T1norm2 = T(all, 0).squaredNorm();
    double T2norm2 = T(all, 1).squaredNorm();

    double SqrtDelta = sqrt(square(2 * in1 * T1T2) - 4 * (T2norm2 * (square(in1) * T1norm2 - square(Rcut))));
    if (isnan(SqrtDelta)) {
//        cout << SqrtDelta << endl;
        return make_tuple(0, 0, false);
    }

    int in21 = (-2. * in1 * T1T2 + SqrtDelta) / (2. * T2norm2);
    int in22 = (-2. * in1 * T1T2 - SqrtDelta) / (2. * T2norm2);

    int in2Min = 0;
    int in2Max = 0;

    if (in21 > in22) {
        in2Max = in21;
        in2Min = in22;
    } else {
        in2Max = in22;
        in2Min = in21;
    }

    return make_tuple(in2Min, in2Max, true);
}

ArrayXXi Lattice2d::get_in2MinMaxArray(const MatrixXd &T, double Rcut) {
    // mmake an array of minimum and maximum of in2 at fixed in1 such that |in1*T1 + in2*T2| <Rcut
    // array columns are in order = in1, in2Min, in2Max

    int NN = 5 * Rcut / min(T(all, 0).norm(), T(all, 1).norm());
    if (NN == 0){
        return (ArrayXXi(1,3) << 0, 0 ,0).finished();
    }

    ArrayXXi in2MinMaxArray(NN, 3);
    int irow = 0;
    int in2Min;
    int in2Max;
    bool Is_Any_in2_Found;

    // probe from 0 to inf
    int in1 = 0;
    tie(in2Min, in2Max, Is_Any_in2_Found) = get_in2Extremes(T, Rcut, in1);

    while (Is_Any_in2_Found) {
        in2MinMaxArray.row(irow++) << in1, in2Min, in2Max;

        in1++;
        tie(in2Min, in2Max, Is_Any_in2_Found) = get_in2Extremes(T, Rcut, in1);
    }

    // probe from -1 to -inf
    in1 = -1;
    tie(in2Min, in2Max, Is_Any_in2_Found) = get_in2Extremes(T, Rcut, in1);

    while (Is_Any_in2_Found) {
        in2MinMaxArray.row(irow++) << in1, in2Min, in2Max;

        in1--;
        tie(in2Min, in2Max, Is_Any_in2_Found) = get_in2Extremes(T, Rcut, in1);
    }

    if (irow == 0){
        return (ArrayXXi(1,3) << 0, 0 ,0).finished();
    }

    return (in2MinMaxArray(seq(0, irow - 1), all));
}

Lattice2d& Lattice2d::computeInteractionTensorEwald() {
    Jxx.setZero(N,N);
    Jyy.setZero(N,N);
    Jxy.setZero(N,N);
    Jzz.setZero(N,N);

    double pi = acos(-1);
    double invsPi = 1/sqrt(pi);
    // minimmum distance between two dipoles in supercell
    double r0 = getLatticeConstant();
    cout << " r0 = " << (r0) << endl;
    double Tmax = max(T(all,0).norm(), T(all,1).norm());
    double Gmax = max(G(all,0).norm(), G(all,1).norm());
//    double tol = 1e-30; // tolerance
    double prec = 100;
//    double prec = -log(tol);
    cout << "prec = " << (prec)<<endl;
    //kappa parameter sets the weight of real space sum vs reciprocal space sum. if we set this
    // parameter large enough , the real space sum will be negligible and all summation wight will be
    // in fourier space. we set kappa such that strongest interaction in real space  becoms of the
    // order of tolerance
    double Rcut = 0;
    double Gcut = 0;
    double kappa =0;
    Rcut = 1.*sqrt(prec/pi) * Tmax* pow(N,-1/4);
    kappa = 1.*sqrt(prec)/Rcut;
    Gcut = 1*2*kappa*sqrt(prec);

    cout << "T = " << endl <<  T << endl << endl;
    cout << "N = " << endl <<  N << endl << endl;
    cout << "kappa = " << (kappa) << endl;
    cout << "Gcut = " << (Gcut) << endl;
    cout << "Rcut = " << (Rcut) << endl;
    cout << "G = " << endl <<  G << endl << endl;
    ArrayXXi im2MinMaxArray = get_in2MinMaxArray(G, Gcut+0*Gmax);
    ArrayXXi in2MinMaxArray = get_in2MinMaxArray(T, Rcut+0*Tmax);
    cout << "rows G = " << im2MinMaxArray.rows() << endl;
    cout << "rows T = " << in2MinMaxArray.rows() << endl;

    double Residue = -2 * pow(kappa, 3) / (3 * sqrt(pi));
    double ResidueZ = 2 * sqrt(pi) * kappa / A;
    double inv2kappa = 1 /(2*kappa);
    cout << "Residue = " << (Residue) << endl;
    cout << "ResidueZ = " << (ResidueZ) << endl;
    for (int i = 0; i < N; ++i) {
        if (i%(max({N / 10, 1})) == 0){
            cout << "particle  " << i << "  of  "  << N << endl;
        }
        for (int j = i; j < N; ++j) {
            double rijx = D(0,i)-D(0,j);
            double rijy = D(1,i)-D(1,j);
            double JxxijRec = 0; //reciprocal
            double JyyijRec = 0;
            double JzzijRec = 0;
            double JxyijRec = 0;
            for (int index = 0; index < im2MinMaxArray.rows(); ++index) {
                int im1 = im2MinMaxArray(index, 0);
                int im2Min = im2MinMaxArray(index, 1);
                int im2Max = im2MinMaxArray(index, 2);
                for (int im2 = 0; im2 <= im2Max; ++im2) {
                    double factor =  2;
                    if(im2 == 0){
                        factor = 1;
                    }
                    if((im1 == 0) && (im2 == 0))
                        continue;
                    double Gmx = im1 * G(0, 0) + im2 * G(0, 1);
                    double Gmy = im1 * G(1, 0) + im2 * G(1, 1);
                    double Gm2 = Gmx*Gmx +Gmy*Gmy;
                    double Gmnorm = sqrt(Gm2);
                    double cosGr = cos(Gmx*rijx + Gmy*rijy);
                    double erfcGk = erfc(Gmnorm * inv2kappa);
                    double eGc = erfcGk / Gmnorm * cosGr;
                    JxxijRec += factor * square(Gmx) * eGc;
                    JyyijRec += factor * square(Gmy) * eGc;
                    JxyijRec += factor * Gmx * Gmy * eGc;
                    JzzijRec += factor * (2 * kappa *invsPi * exp(-square(Gmnorm * inv2kappa))
                            - Gmnorm * erfcGk) * cosGr;
                }
            }

            JxxijRec *= pi / A;
            JyyijRec *= pi / A;
            JxyijRec *= pi / A;
            JzzijRec *= pi / A;

            JzzijRec += ResidueZ;
            if (i == j){
                JxxijRec += Residue;
                JyyijRec += Residue;
                JzzijRec += Residue;
            }

            //real part
            double JxxijReal = 0;
            double JyyijReal = 0;
            double JxyijReal = 0;
            double JzzijReal = 0;
            for (int index = 0; index < in2MinMaxArray.rows(); ++index) {
                int in1 = in2MinMaxArray(index, 0);
                int in2Min = in2MinMaxArray(index, 1);
                int in2Max = in2MinMaxArray(index, 2);
                for (int in2 = in2Min; in2 <= in2Max; ++in2) {
                    if ((in1 == 0) && (in2 == 0) && (i == j))
                        continue;
                    double Rnx = rijx + in1 * T(0, 0) + in2 * T(0, 1);
                    double Rny = rijy + in1 * T(1, 0) + in2 * T(1, 1);
                    double Rn2 = Rnx * Rnx + Rny * Rny;
                    double Rn = sqrt(Rn2);
                    double invRn = 1. / Rn;
                    double invRn2 = invRn * invRn;
                    double invRn3 = invRn2 * invRn;
                    double invRn5 = invRn2 * invRn3;
                    double ec = erfc(kappa * Rn);
                    double ex = exp(-square((kappa * Rn)));
                    double ecR = ec * invRn2 * invRn;
                    double exR = ex * invRn2 * 2 * kappa * invsPi;
                    double B = (ecR + exR);
                    double C = 3 * invRn2 * ecR + (3 * invRn2 + 2 * kappa * kappa) * exR;

                    JxxijReal += B - Rnx * Rnx *  C;
                    JyyijReal += B - Rny * Rny *  C;
                    JxyijReal += -Rnx * Rny *  C;
                    JzzijReal += B;
                }
            }
            JxxijReal *= 0.5;
            JyyijReal *= 0.5;
            JxyijReal *= 0.5;
            JzzijReal *= 0.5;

            double Jxxij = JxxijRec + JxxijReal;
            double Jyyij = JyyijRec + JyyijReal;
            double Jxyij = JxyijRec + JxyijReal;
            double Jzzij = JzzijRec + JzzijReal;

            Matrix2d Jplane;
            Jplane << Jxxij, Jxyij,
                    Jxyij, Jyyij;


            Jzz(i,j) += Jzzij;
            Jxx(i,j) += Jplane(0,0);
            Jyy(i,j) += Jplane(1,1);
            Jxy(i,j) += Jplane(0,1);
            if (i != j){
                Jxx(j,i) += Jplane(0,0);
                Jyy(j,i) += Jplane(1,1);
                Jxy(j,i) += Jplane(0,1);
                Jzz(j,i) += Jzzij;
            }



        }
    }
    Jself(0,0) = Jxx(0,0);
    Jself(0,1) = Jxy(0,0);
    Jself(1,0) = Jxy(0,0);
    Jself(1,1) = Jyy(0,0);
    Jself(2,2) = Jzz(0,0);
//    checkJfinite();
//    howSymmetric();
    if (LatticeType != 'h'){
//        setWvectors();
    } else if (LatticeType == 'h' && isHoneycombFromTriangular){
        setWvectorsHoneycomb();
    }
    cout << "Madelung:" << endl;
    cout << getMadelung() << endl << endl;
    computeInteractionTensorEwald1p();
    return *this;
}

Lattice2d& Lattice2d::computeInteractionTensorEwald1p() {
    Jxx1p.setZero(N);
    Jyy1p.setZero(N);
    Jxy1p.setZero(N);
    Jzz1p.setZero(N);

    double pi = acos(-1);
    double invsPi = 1/sqrt(pi);
    // minimmum distance between two dipoles in supercell
    double r0 = getLatticeConstant();
    cout << " r0 = " << (r0) << endl;
    double Tmax = max(T(all,0).norm(), T(all,1).norm());
    double Gmax = max(G(all,0).norm(), G(all,1).norm());
//    double tol = 1e-30; // tolerance
    double prec = 100;
//    double prec = -log(tol);
    cout << "prec = " << (prec)<<endl;
    //kappa parameter sets the weight of real space sum vs reciprocal space sum. if we set this
    // parameter large enough , the real space sum will be negligible and all summation wight will be
    // in fourier space. we set kappa such that strongest interaction in real space  becoms of the
    // order of tolerance
//    double kappa = sqrt(-log(tol)) / r0;
//    double kappa = sqrt(pi) / T0;
//    double kappa = 5;
    double Rcut = 0;
    double Gcut = 0;
    double kappa =0;

    Rcut = 1.*sqrt(prec/pi) * Tmax* pow(N,-1/4);
    kappa = 1.*sqrt(prec)/Rcut;
    Gcut = 1*2*kappa*sqrt(prec);

    cout << "T = " << endl <<  T << endl << endl;
    cout << "N = " << endl <<  N << endl << endl;
    cout << "kappa = " << (kappa) << endl;
    cout << "Gcut = " << (Gcut) << endl;
    cout << "Rcut = " << (Rcut) << endl;
    cout << "G = " << endl <<  G << endl << endl;
    ArrayXXi im2MinMaxArray = get_in2MinMaxArray(G, Gcut+0*Gmax);
    ArrayXXi in2MinMaxArray = get_in2MinMaxArray(T, Rcut+0*Tmax);
    cout << "rows G = " << im2MinMaxArray.rows() << endl;
    cout << "rows T = " << in2MinMaxArray.rows() << endl;

    double Residue = -2 * pow(kappa, 3) / (3 * sqrt(pi));
    double ResidueZ = 2 * sqrt(pi) * kappa / A;
    double inv2kappa = 1 /(2*kappa);
    cout << "Residue = " << (Residue) << endl;
    cout << "ResidueZ = " << (ResidueZ) << endl;

    int i = 0;
    for (int j = i; j < N; ++j) {
        double rijx = D(0,i)-D(0,j);
        double rijy = D(1,i)-D(1,j);
        double JxxijRec = 0; //reciprocal
        double JyyijRec = 0;
        double JzzijRec = 0;
        double JxyijRec = 0;
        for (int index = 0; index < im2MinMaxArray.rows(); ++index) {
            int im1 = im2MinMaxArray(index, 0);
            int im2Min = im2MinMaxArray(index, 1);
            int im2Max = im2MinMaxArray(index, 2);
            for (int im2 = 0; im2 <= im2Max; ++im2) {
                double factor =  2;
                if(im2 == 0){
                    factor = 1;
                }
                if((im1 == 0) && (im2 == 0))
                    continue;
                double Gmx = im1 * G(0, 0) + im2 * G(0, 1);
                double Gmy = im1 * G(1, 0) + im2 * G(1, 1);
                double Gm2 = Gmx*Gmx +Gmy*Gmy;
                double Gmnorm = sqrt(Gm2);
                double cosGr = cos(Gmx*rijx + Gmy*rijy);
                double erfcGk = erfc(Gmnorm * inv2kappa);
                double eGc = erfcGk / Gmnorm * cosGr;
                JxxijRec += factor * square(Gmx) * eGc;
                JyyijRec += factor * square(Gmy) * eGc;
                JxyijRec += factor * Gmx * Gmy * eGc;
                JzzijRec += factor * (2 * kappa *invsPi * exp(-square(Gmnorm * inv2kappa))
                                      - Gmnorm * erfcGk) * cosGr;
            }
        }

        JxxijRec *= pi / A;
        JyyijRec *= pi / A;
        JxyijRec *= pi / A;
        JzzijRec *= pi / A;

        JzzijRec += ResidueZ;
        if (i == j){
            JxxijRec += Residue;
            JyyijRec += Residue;
            JzzijRec += Residue;
        }

        //real part
        double JxxijReal = 0;
        double JyyijReal = 0;
        double JxyijReal = 0;
        double JzzijReal = 0;
        for (int index = 0; index < in2MinMaxArray.rows(); ++index) {
            int in1 = in2MinMaxArray(index, 0);
            int in2Min = in2MinMaxArray(index, 1);
            int in2Max = in2MinMaxArray(index, 2);
            for (int in2 = in2Min; in2 <= in2Max; ++in2) {
                if ((in1 == 0) && (in2 == 0) && (i == j))
                    continue;
                double Rnx = rijx + in1 * T(0, 0) + in2 * T(0, 1);
                double Rny = rijy + in1 * T(1, 0) + in2 * T(1, 1);
                double Rn2 = Rnx * Rnx + Rny * Rny;
                double Rn = sqrt(Rn2);
                double invRn = 1. / Rn;
                double invRn2 = invRn * invRn;
                double invRn3 = invRn2 * invRn;
                double invRn5 = invRn2 * invRn3;
                double ec = erfc(kappa * Rn);
                double ex = exp(-square((kappa * Rn)));
                double ecR = ec * invRn2 * invRn;
                double exR = ex * invRn2 * 2 * kappa * invsPi;
                double B = (ecR + exR);
                double C = 3 * invRn2 * ecR + (3 * invRn2 + 2 * kappa * kappa) * exR;

                JxxijReal += B - Rnx * Rnx *  C;
                JyyijReal += B - Rny * Rny *  C;
                JxyijReal += -Rnx * Rny *  C;
                JzzijReal += B;
            }
        }
        JxxijReal *= 0.5;
        JyyijReal *= 0.5;
        JxyijReal *= 0.5;
        JzzijReal *= 0.5;

        double Jxxij = JxxijRec + JxxijReal;
        double Jyyij = JyyijRec + JyyijReal;
        double Jxyij = JxyijRec + JxyijReal;
        double Jzzij = JzzijRec + JzzijReal;

        Matrix2d Jplane;
        Jplane << Jxxij, Jxyij,
                Jxyij, Jyyij;


        Jzz1p(j) += Jzzij;
        Jxx1p(j) += Jplane(0,0);
        Jyy1p(j) += Jplane(1,1);
        Jxy1p(j) += Jplane(0,1);

    }

    if (Prop.isHavingExchangeInteraction){
        calcDistanceVec1p();
        decreaseDipolarStrength();
    }

    if (Prop.isNoSelfEnergy){
        Jxx1p(0) = 0;
        Jxy1p(0) = 0;
        Jxy1p(0) = 0;
        Jyy1p(0) = 0;
        Jzz1p(0) = 0;
    }
    Jself(0,0) = Jxx1p(0);
    Jself(0,1) = Jxy1p(0);
    Jself(1,0) = Jxy1p(0);
    Jself(1,1) = Jyy1p(0);
    Jself(2,2) = Jzz1p(0);


//    checkJfinite();
//    if (LatticeType != 'h'){

//    } else if (LatticeType == 'h' && isHoneycombFromTriangular){
//        setWvectorsHoneycomb();
//    }
//    cout << "Jxx1p = " << Jxx1p(seq(0,min(50L, Jxx1p.size()-1))) <<endl <<endl;

    setWvectors1p();
    cout << "Madelung 1 paricle:" << endl;
    cout << getMadelung1p() << endl << endl;
    return *this;
}


void Lattice2d::computeInteractionTensorDirect() {
    MatrixXd J;
    J.setZero(3 * N, 3* N);
    Jzz.setZero(N,N);

    double tol = 1e-11;
    cout << "N = " << N << endl;
    double Rcut = pow(tol, -1./3) * getLatticeConstant();
    cout << "Rcut = " << Rcut << endl;
    ArrayXXi in2MinMaxArray = get_in2MinMaxArray(T, Rcut);
    cout << "rows = " << in2MinMaxArray.rows() << endl;

    for (int i = 0; i < N; ++i) {
//        if (i%(max({N / 10, 1})) == 0){
//            cout << "particle  " << i << "  of  "  << N << endl;
//        }
        for (int j = i; j < N; ++j) {
            double Jxxij = 0;
            double Jyyij = 0;
            double Jzzij = 0;
            double Jxyij = 0;
            for (int index = 0; index < in2MinMaxArray.rows(); ++index) {
                int in1 = in2MinMaxArray(index, 0);
                int in2Min = in2MinMaxArray(index, 1);
                int in2Max = in2MinMaxArray(index, 2);
                for (int in2 = in2Min; in2 <= in2Max; ++in2) {
                    if((in1 == 0) && (in2 == 0) && (i == j))
                        continue;
//                    Vector2d rn = in1 * T(all, 0) + in2 * T(all, 1);
                    Vector2d R = D(all, i) - D(all, j) + in1 * T(all, 0) + in2 * T(all, 1);
                    double x = R(0);
                    double y = R(1);
                    double Rnorm = R.norm();
                    double Rnorm2 = Rnorm * Rnorm;
                    double invR5 = 1. / (Rnorm2 * Rnorm2 * Rnorm);
                    double x2 = x * x;
                    double y2 = y * y;
                    Jxxij += (-2 * x2 + y2) * invR5;
                    Jyyij += (-2 * y2 + x2) * invR5;
                    Jzzij += (x2 + y2) * invR5;
                    Jxyij += (-3 * x * y) * invR5;
                }
            }

            Matrix3d Jij;
            Jij << Jxxij, Jxyij,     0,
                    Jxyij, Jyyij,     0,
                    0    ,     0, Jzzij;

            J(seqN(3*i,3), seqN(3*j,3)) += Jij;
            if (i != j){
                J(seqN(3*j,3), seqN(3*i,3)) += Jij;
            }
        }
    }
    // times by 1/2 because we summed j > i
    J /= 2 ;
    checkJfinite();

}

MatrixXd Lattice2d::getMadelung(){
    MatrixXd J1(3,3);
    J1.setZero();

    for (int j = 0; j < N; ++j) {
        J1(seq(0,1), seq(0,1)) += (Matrix2d() << Jxx(0,j), Jxy(0,j), Jxy(0,j), Jyy(0,j)).finished();
        J1(2,2) += Jzz(0,j);
    }
    return J1;
}

double Lattice2d::getLatticeConstant() {
    double dmin = DBL_MAX;

    for (int i = 0; i < N; ++i) {
        for (int j = i+1; j < N; ++j) {
            double dij = (D(all,i) - D(all,j)).norm();
            if (dmin > dij){
                dmin = dij;
            }
        }
    }
    return dmin;
}

void Lattice2d::checkJfinite() {
#ifndef NDEBUG
    for (int i = 0; i < N; ++i) {
        assert(("J has non finite element!",isfinite(Jxx1p(i))));
        for (int j = 0; j < N; ++j) {
            assert(("J has non finite element!",isfinite(Jxx(i,j))));
        }
    }
#else  //noting
#endif
}

MatrixXd Lattice2d::getMadelung1p() {
    MatrixXd J1(3,3);
    J1.setZero();

    for (int j = 0; j < N; ++j) {
        J1(seq(0,1), seq(0,1)) += (Matrix2d() << Jxx1p(j), Jxy1p(j), Jxy1p(j), Jyy1p(j)).finished();
        J1(2,2) += Jzz1p(j);
    }
    return J1;
}

void Lattice2d::calcDistanceVec() {
    DistanceVec.setZero(10*N);
    CounterR.setZero(10*N);
    MapR.setZero(N,N);
    MatrixXd DistanceMat;
    DistanceMat.setZero(N,N);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == j){
                DistanceMat(i,j) = 0.;
                continue;
            }
            DistanceMat(i,j) = getMinDistance(i,j);
            assert(!isnan(DistanceMat(i,j)));
            assert((DistanceMat(i,j) > 0.001));
        }
    }

    int a = 0;
    DistanceVec(0) = DistanceMat(0,0);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            bool HasThisR = false;
            for (int k = 0; k <= a; ++k) {
                if (almostEquals(DistanceVec(k), DistanceMat(i, j))){
                    HasThisR = true;
                    break;
                }
            }
            if (!HasThisR){
                a++;
                DistanceVec(a) = DistanceMat(i,j);
            }
        }
    }
    assert(a > 0);
    DistanceVec = DistanceVec(seq(0,a)).eval();
    std::sort(DistanceVec.begin(), DistanceVec.end());
    CounterR.setZero(DistanceVec.size());
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < DistanceVec.size(); ++k) {
                if (almostEquals(DistanceMat(i, j), DistanceVec(k))){
                    MapR(i,j) = k;
                    CounterR(k)++;
                    break;
                }
            }
        }
    }
    cout << endl;
    cout << "done calculating distances" <<endl;
    assert(DistanceVec.size() > 0);
    assert(CounterR.size() > 0);
    cout << "CounterR = " << endl << CounterR.transpose()(seqN(0,min(CounterR.size(), 5L))) << endl;
    cout << "DistanceVec = " << endl << DistanceVec.transpose()(seqN(0,min(CounterR.size(), 5L))) << endl;
    cout << "size of DistanceVec = " << DistanceVec.size() << endl;
    cout << "N*N = " << N*N <<endl;
    cout << "sum CounterR = " << CounterR.sum() << endl;

}

void Lattice2d::calcDistanceVec1p() {
    DistanceVec1p.setZero(10*N);
    VectorXd DistanceMat1p;
    DistanceMat1p.setZero(N);

    const int i = 0;
    for (int j = 0; j < N; ++j) {
        if (i == j){
            DistanceMat1p(j) = 0.;
            continue;
        }
        DistanceMat1p(j) = getMinDistance(i,j);
        assert(!isnan(DistanceMat1p(j)));
        assert((DistanceMat1p(j) > 0.001));
    }

    int a = 0;
    DistanceVec1p(0) = DistanceMat1p(0,0);
    for (int j = 0; j < N; ++j) {
        bool HasThisR = false;
        for (int k = 0; k <= a; ++k) {
            if (almostEquals(DistanceVec1p(k), DistanceMat1p(j))){
                HasThisR = true;
                break;
            }
        }
        if (!HasThisR){
            a++;
            DistanceVec1p(a) = DistanceMat1p(j);
        }
    }

    assert(a > 0);
    DistanceVec1p = DistanceVec1p(seq(0,a)).eval();
    std::sort(DistanceVec1p.begin(), DistanceVec1p.end());

    cout << endl;
    cout << "done calculating distances 1p" <<endl;
    assert(DistanceVec1p.size() > 0);
    cout << "DistanceVec = " << endl << DistanceVec1p.transpose()(seqN(0,min(DistanceVec1p.size(), 5L))) << endl;
    cout << "size of DistanceVec = " << DistanceVec1p.size() << endl;

}

double Lattice2d::getMinDistance(int i, int j) const {
    Vector2d rtemp;
    Vector2d Dij;
    Dij =  D(all, i) - D(all, j);
    double MinDis = DBL_MAX;

    for (int k = -5; k < 5; ++k) {
        for (int l = -5; l < 5; ++l) {
            rtemp = Dij  + k * T(all, 0) + l * T(all, 1);
            MinDis = min(MinDis, rtemp.norm());
        }
    }
    return MinDis;
}

void Lattice2d::sortLocations(MatrixXd &D1, VectorXi &PLabel1) {
    assert(D1.cols() == PLabel1.size());
    auto N1 = PLabel1.size();
    for (int i = 0; i < N1; ++i) {
        for (int j = i; j < N1; ++j) {
            if ((D1(1,i) > D1(1,j) && !almostEquals(D1(1,i),D1(1,j))) ||
                (D1(0,i) > D1(0,j) &&  almostEquals(D1(1,i),D1(1,j)))) {
                swap(PLabel1(i), PLabel1(j));
                D1.col(i).swap(D1.col(j));
            }
        }
    }
//    printD();
}

void Lattice2d::printD() {
    cout << endl << "D: " <<endl;
    int a = 0;
    for (int i = 0; i < L2; ++i) {
        for (int j = 0; j < L1; ++j) {
            cout << a << '\t';
            a++;
        }
        cout <<endl;
    }
    cout <<endl;
    for (int i = 0; i < L2; ++i) {
        cout << D(all, seqN(i*L1, L1));
        cout <<endl <<endl;
    }
}

void Lattice2d::setWvectors() {
    assert(LatticeType != 'h');
    Wxx.setZero(4*N);
    Wxy.setZero(4*N);
    Wyy.setZero(4*N);
    Wzz.setZero(4*N);
    assert(L1 == L2);
    for (int rowspin = 0; rowspin < L2; ++rowspin) {
        for (int colspin = 0; colspin < L1; ++colspin) {
            int pnum = rowspin*L1 + colspin;
            for (int j = 0; j < L2; ++j) {
                int Fn = 2*L1*(L2+j-rowspin)-colspin+L1;
                int j1 = L1*j;
                for (int k = 0; k < L1; ++k) {
                    int pnum2 = j1 +k;
                    Wxx(Fn+k) = Jxx(pnum,pnum2);
                    Wxy(Fn+k) = Jxy(pnum,pnum2);
                    Wyy(Fn+k) = Jyy(pnum,pnum2);
                    Wzz(Fn+k) = Jzz(pnum,pnum2);
                }
            }
        }
    }

    for (int rowspin = 0; rowspin < L2; ++rowspin) {
        for (int colspin = 0; colspin < L1; ++colspin) {
            int pnum = rowspin*L1 + colspin;
            for (int j = 0; j < L2; ++j) {
                int Fn = 2*L1*(L2+j-rowspin)-colspin+L1;
                int j1 = L1*j;
                for (int k = 0; k < L1; ++k) {
                    int pnum2 = j1 +k;
//                    cout << "Wxx(Fn+k),Jxx(pnum,pnum2), dif = " << Wxx(Fn+k) << '\t'
//                         <<Jxx(pnum,pnum2) << '\t' << Wxx(Fn+k) -Jxx(pnum,pnum2) << endl;
                    assert(almostEquals(Wxx(Fn+k),Jxx(pnum,pnum2)));  // TODO make this assertion active
                    assert(almostEquals(Wxy(Fn+k),Jxy(pnum,pnum2)));
                    assert(almostEquals(Wyy(Fn+k),Jyy(pnum,pnum2)));
                    assert(almostEquals(Wzz(Fn+k),Jzz(pnum,pnum2)));
                }
            }
        }
    }
    cout << "set W vectors done" <<endl;
}

void Lattice2d::setWvectors1p() {
    assert(LatticeType != 'h');
    Wxx.setZero(4*N);
    Wxy.setZero(4*N);
    Wyy.setZero(4*N);
    Wzz.setZero(4*N);
    assert(L1 == L2);
    for (int rowspin = 0; rowspin < L2; ++rowspin) {
        for (int colspin = 0; colspin < L1; ++colspin) {
            int pnum = rowspin*L1 + colspin;
            for (int j = 0; j < L2; ++j) {
                int Fn = 2*L1*(L2+j-rowspin)-colspin+L1;
                int j1 = L1*j;
                for (int k = 0; k < L1; ++k) {
                    int pnum2 = j1 +k;
                    auto [x3,y3] = periodicParticle(rowspin, colspin, j, k);
                    int pnum3 = x3*L1 + y3;

                    Wxx(Fn+k) = Jxx1p(pnum3);
                    Wxy(Fn+k) = Jxy1p(pnum3);
                    Wyy(Fn+k) = Jyy1p(pnum3);
                    Wzz(Fn+k) = Jzz1p(pnum3);
                }
            }
        }
    }
//    assert(almostEquals(Wxx1p,Wxx));
//    assert(almostEquals(Wxy1p,Wxy));
//    assert(almostEquals(Wyy1p,Wyy));
//    assert(almostEquals(Wzz1p,Wzz));

    for (int rowspin = 0; rowspin < L2; ++rowspin) {
        for (int colspin = 0; colspin < L1; ++colspin) {
            int pnum = rowspin*L1 + colspin;
            for (int j = 0; j < L2; ++j) {
                int Fn = 2*L1*(L2+j-rowspin)-colspin+L1;
                int j1 = L1*j;
                for (int k = 0; k < L1; ++k) {
                    int pnum2 = j1 +k;
                    auto [x3,y3] = periodicParticle(rowspin, colspin, j, k);
                    int pnum3 = x3*L1 + y3;
                    assert(almostEquals(Wxx(Fn+k),Jxx1p(pnum3)));  // TODO make this assertion active
                    assert(almostEquals(Wxy(Fn+k),Jxy1p(pnum3)));
                    assert(almostEquals(Wyy(Fn+k),Jyy1p(pnum3)));
                    assert(almostEquals(Wzz(Fn+k),Jzz1p(pnum3)));
                }
            }
        }
    }
    cout << "set W vectors done" <<endl;
}

std::pair<int, int> Lattice2d::getL1L2(MatrixXd &D1) {
//    vector<double> L1Vec;
    vector<double> YiVec;
    auto NN = D1.cols();
    int LL1 = 0;
    int LL2 = 0;
    for (int i = 0; i < NN; ++i) {
//        if (none_of(L1Vec.begin(), L1Vec.end(),
//                   [this, i](double num){ return almostEquals(D(0,i), num);})){
//            L1++;
//            L1Vec.push_back(D(0,i));
//        }
        if (none_of(YiVec.begin(), YiVec.end(),
                    [=](double num){ return almostEquals(D1(1,i), num);})){
            LL2++;
            YiVec.push_back(D1(1, i));
        }
    }
    LL1 = NN / LL2;
//    cout << "YiVec = "  <<endl;
//    for (auto i :YiVec) {
//        cout << i <<endl;
//    }
    cout << "L1 = " << LL1 << endl;
    cout << "L2 = " << LL2 << endl;
    return make_pair(LL1, LL2);
}

void Lattice2d::howSymmetric() {
    vector<double> JxxUnique;
    for (auto i : Jxx.reshaped()) {
        if (none_of(JxxUnique.begin(), JxxUnique.end(),
                    [i](double num){ return almostEquals(i, num);})){
            JxxUnique.push_back(i);
//            cout << "JxxUnique.back() = " << JxxUnique.back() <<endl;
        }
    }
    sort(JxxUnique.begin(), JxxUnique.end());
    cout << "Jxx.size() =" << Jxx.size() <<endl;
    cout << "symmetry param =" << Jxx.size() / (JxxUnique.size()+0.0) <<endl;
//    cout << "JxxUnique =" <<endl;
//    for (auto i :JxxUnique) {
//        cout << i <<endl;
//    }
    cout << "JxxUnique.size() = " << JxxUnique.size() <<endl;
}

void Lattice2d::generateHoneycomb() {
    double a = (Dbase(all,1) - Dbase(all,0)).norm();
    assert (almostEquals(a,3/sqrt(3)));
    Vector2d a1, a2;
    a1.setZero();
    a2.setZero();
    a1(0) = a;
    a2(0) = a / 2;
    a2(1) = a * sqrt(3) / 2;

    double Babs = 1;
    assert (almostEquals(Babs,1));
    Vector2d b1 = -(a1 + a2) / 6.;
    Vector2d b2 = -b1;
    cout << "(b1-b2).norm() = " << (b1-b2).norm() << endl;
    cout << "Babs = " << Babs << endl;
    assert(almostEquals(((a1+a2)/3.).norm(), 1));
    assert(almostEquals((b1-b2).norm(), Babs));

    MatrixXi ToHoneycombLabel(3,2);
    ToHoneycombLabel <<
    4, 3,
    6, 1,
    2, 5;
    N = 2 * Nbase;
    D.setZero(2, N);
    PLabel.setZero(N);
    PSubLattice.setZero(N);
    for (int i = 0; i < Nbase; ++i) {
        D(all,2*i) = Dbase(all,i) + b1;
        D(all,2*i+1) = Dbase(all,i) + b2;
        PLabel(2*i)   = ToHoneycombLabel(PLabelbase(i),0);
        PLabel(2*i+1) = ToHoneycombLabel(PLabelbase(i),1);
        PSubLattice(2*i)   = 0;
        PSubLattice(2*i+1) = 1;
    }
//    cout << "Dbase = " <<endl << Dbase.transpose() << endl;
//    cout << "PLabel = " <<endl << PLabel<< endl;
//    cout << "D = " <<endl << D.transpose() << endl;
    cout << "N = " << N << endl;
}

void Lattice2d::setWvectorsHoneycomb() {
    assert(LatticeType == 'h');

    Wxx11.setZero(4*Nbase);
    Wxy11.setZero(4*Nbase);
    Wyy11.setZero(4*Nbase);
    Wzz11.setZero(4*Nbase);

    Wxx12.setZero(4*Nbase);
    Wyy12.setZero(4*Nbase);
    Wxy12.setZero(4*Nbase);
    Wzz12.setZero(4*Nbase);

    Wxx21.setZero(4*Nbase);
    Wxy21.setZero(4*Nbase);
    Wyy21.setZero(4*Nbase);
    Wzz21.setZero(4*Nbase);

    Wxx22.setZero(4*Nbase);
    Wxy22.setZero(4*Nbase);
    Wyy22.setZero(4*Nbase);
    Wzz22.setZero(4*Nbase);
    assert(L1 == L2);

    // for sublatice 0
    for (int rowspin = 0; rowspin < L2; ++rowspin) {
        for (int colspin = 0; colspin < L1; ++colspin) {
            int pnum = rowspin*(2*L1) + 2*colspin;
            assert(PSubLattice(pnum) == 0);
            for (int j = 0; j < L2; ++j) {
                int Fn = 2*(L1)*(L2+j-rowspin)-colspin+(L1);
                int j1 = (2*L1)*j;
                for (int k = 0; k < (L1); ++k) {
                    int pnum2 = j1 + 2*k;
                    assert(j1 + 2*k < N);
                    assert(j1 + 2*k+1 < N);
                    assert(Fn+k < 4*Nbase);
                    assert(pnum < N);
                    Wxx11(Fn+k) = Jxx(pnum,j1 + 2*k);
                    Wxy11(Fn+k) = Jxy(pnum,j1 + 2*k);
                    Wyy11(Fn+k) = Jyy(pnum,j1 + 2*k);
                    Wzz11(Fn+k) = Jzz(pnum,j1 + 2*k);

                    Wxx12(Fn+k) = Jxx(pnum,j1 + 2*k+1);
                    Wxy12(Fn+k) = Jxy(pnum,j1 + 2*k+1);
                    Wyy12(Fn+k) = Jyy(pnum,j1 + 2*k+1);
                    Wzz12(Fn+k) = Jzz(pnum,j1 + 2*k+1);
                }
            }
        }
    }

    for (int rowspin = 0; rowspin < L2; ++rowspin) {
        for (int colspin = 0; colspin < L1; ++colspin) {
            int pnum = rowspin*(2*L1) + 2*colspin;
            for (int j = 0; j < L2; ++j) {
                int Fn = 2*(L1)*(L2+j-rowspin)-colspin+(L1);
                int j1 = (2*L1)*j;
                for (int k = 0; k < (L1); ++k) {
                    int pnum2 = j1 + 2*k;
//                    cout << "Wxx(Fn+k),Jxx(pnum,pnum2), dif = " << Wxx11(Fn+k) << '\t'
//                       <<Jxx(pnum,j1 + 2*k) << '\t' << Wxx11(Fn+k) -Jxx(pnum,j1 + 2*k) << endl;
                    assert(almostEquals(Wxx11(Fn+k),Jxx(pnum,j1 + 2*k)));
                    assert(almostEquals(Wxy11(Fn+k),Jxy(pnum,j1 + 2*k)));
                    assert(almostEquals(Wyy11(Fn+k),Jyy(pnum,j1 + 2*k)));
                    assert(almostEquals(Wzz11(Fn+k),Jzz(pnum,j1 + 2*k)));

//                    cout << "Wxx(Fn+k),Jxx(pnum,pnum2), dif = " << Wxx12(Fn+k) << '\t'
//                         <<Jxx(pnum,j1 + 2*k+1) << '\t' << Wxx12(Fn+k) -Jxx(pnum,j1 + 2*k+1) << endl;
                    assert(almostEquals(Wxx12(Fn+k),Jxx(pnum,j1 + 2*k+1)));
                    assert(almostEquals(Wxy12(Fn+k),Jxy(pnum,j1 + 2*k+1)));
                    assert(almostEquals(Wyy12(Fn+k),Jyy(pnum,j1 + 2*k+1)));
                    assert(almostEquals(Wzz12(Fn+k),Jzz(pnum,j1 + 2*k+1)));
                }
            }
        }
    }


    // for sublatice 1
    for (int rowspin = 0; rowspin < L2; ++rowspin) {
        for (int colspin = 0; colspin < L1; ++colspin) {
            int pnum = rowspin*(2*L1) + 2*colspin+1;
            assert(PSubLattice(pnum) == 1);
            for (int j = 0; j < L2; ++j) {
                int Fn = 2*(L1)*(L2+j-rowspin)-colspin+(L1);
                int j1 = (2*L1)*j;
                for (int k = 0; k < (L1); ++k) {
                    int pnum2 = j1 + 2*k;
                    assert(j1 + 2*k < N);
                    assert(j1 + 2*k+1 < N);
                    assert(Fn+k < 4*Nbase);
                    assert(pnum < N);
                    Wxx21(Fn+k) = Jxx(pnum,j1 + 2*k);
                    Wxy21(Fn+k) = Jxy(pnum,j1 + 2*k);
                    Wyy21(Fn+k) = Jyy(pnum,j1 + 2*k);
                    Wzz21(Fn+k) = Jzz(pnum,j1 + 2*k);

                    Wxx22(Fn+k) = Jxx(pnum,j1 + 2*k+1);
                    Wxy22(Fn+k) = Jxy(pnum,j1 + 2*k+1);
                    Wyy22(Fn+k) = Jyy(pnum,j1 + 2*k+1);
                    Wzz22(Fn+k) = Jzz(pnum,j1 + 2*k+1);
                }
            }
        }
    }

    for (int rowspin = 0; rowspin < L2; ++rowspin) {
        for (int colspin = 0; colspin < L1; ++colspin) {
            int pnum = rowspin*(2*L1) + 2*colspin+1;
            assert(PSubLattice(pnum) == 1);
            for (int j = 0; j < L2; ++j) {
                int Fn = 2*(L1)*(L2+j-rowspin)-colspin+(L1);
                int j1 = (2*L1)*j;
                for (int k = 0; k < (L1); ++k) {
                    int pnum2 = j1 + 2*k;
//                    cout << "Wxx(Fn+k),Jxx(pnum,pnum2), dif = " << Wxx21(Fn+k) << '\t'
//                         <<Jxx(pnum,j1 + 2*k) << '\t' << Wxx21(Fn+k) -Jxx(pnum,j1 + 2*k) << endl;
                    assert(almostEquals(Wxx21(Fn+k),Jxx(pnum,j1 + 2*k)));
                    assert(almostEquals(Wxy21(Fn+k),Jxy(pnum,j1 + 2*k)));
                    assert(almostEquals(Wyy21(Fn+k),Jyy(pnum,j1 + 2*k)));
                    assert(almostEquals(Wzz21(Fn+k),Jzz(pnum,j1 + 2*k)));

//                    cout << "Wxx(Fn+k),Jxx(pnum,pnum2), dif = " << Wxx22(Fn+k) << '\t'
//                         <<Jxx(pnum,j1 + 2*k+1) << '\t' << Wxx22(Fn+k) -Jxx(pnum,j1 + 2*k+1) << endl;
                    assert(almostEquals(Wxx22(Fn+k),Jxx(pnum,j1 + 2*k+1)));
                    assert(almostEquals(Wxy22(Fn+k),Jxy(pnum,j1 + 2*k+1)));
                    assert(almostEquals(Wyy22(Fn+k),Jyy(pnum,j1 + 2*k+1)));
                    assert(almostEquals(Wzz22(Fn+k),Jzz(pnum,j1 + 2*k+1)));
                }
            }
        }
    }
    cout << "set W vectors done" <<endl;
}

void Lattice2d::setMapR0() {
    if (LatticeType != 'h'){
        assert(LatticeType != 'h');
        MapR0.setZero(4*N);
        assert(L1 == L2);
        for (int rowspin = 0; rowspin < L2; ++rowspin) {
            for (int colspin = 0; colspin < L1; ++colspin) {
                int pnum = rowspin*L1 + colspin;
                for (int j = 0; j < L2; ++j) {
                    int Fn = 2*L1*(L2+j-rowspin)-colspin+L1;
                    int j1 = L1*j;
                    for (int k = 0; k < L1; ++k) {
                        int pnum2 = j1 +k;
                        assert(Fn+k < 4*N);
                        assert(j1+k < N);
                        MapR0(Fn+k) = MapR(pnum,pnum2);
                    }
                }
            }
        }

        for (int rowspin = 0; rowspin < L2; ++rowspin) {
            for (int colspin = 0; colspin < L1; ++colspin) {
                int pnum = rowspin*L1 + colspin;
                for (int j = 0; j < L2; ++j) {
                    int Fn = 2*L1*(L2+j-rowspin)-colspin+L1;
                    int j1 = L1*j;
                    for (int k = 0; k < L1; ++k) {
                        int pnum2 = j1 +k;
                        assert(almostEquals(MapR0(Fn+k),MapR(pnum,pnum2)));  // TODO make this assertion active
                    }
                }
            }
        }
    } else {
        assert(LatticeType == 'h');

        MapR0Sub11.setZero(4*Nbase);
        MapR0Sub12.setZero(4*Nbase);
        MapR0Sub21.setZero(4*Nbase);
        MapR0Sub22.setZero(4*Nbase);
        assert(L1 == L2);

        // for sublatice 0
        for (int rowspin = 0; rowspin < L2; ++rowspin) {
            for (int colspin = 0; colspin < L1; ++colspin) {
                int pnum = rowspin*(2*L1) + 2*colspin;
                assert(PSubLattice(pnum) == 0);
                for (int j = 0; j < L2; ++j) {
                    int Fn = 2*(L1)*(L2+j-rowspin)-colspin+(L1);
                    int j1 = (2*L1)*j;
                    for (int k = 0; k < (L1); ++k) {
                        int pnum2 = j1 + 2*k;
                        assert(j1 + 2*k < N);
                        assert(j1 + 2*k+1 < N);
                        assert(Fn+k < 4*Nbase);
                        assert(pnum < N);
                        MapR0Sub11(Fn+k) = MapR(pnum,j1 + 2*k);
                        MapR0Sub12(Fn+k) = MapR(pnum,j1 + 2*k+1);
                    }
                }
            }
        }

        for (int rowspin = 0; rowspin < L2; ++rowspin) {
            for (int colspin = 0; colspin < L1; ++colspin) {
                int pnum = rowspin*(2*L1) + 2*colspin;
                for (int j = 0; j < L2; ++j) {
                    int Fn = 2*(L1)*(L2+j-rowspin)-colspin+(L1);
                    int j1 = (2*L1)*j;
                    for (int k = 0; k < (L1); ++k) {
                        int pnum2 = j1 + 2*k;
//                    cout << "Wxx(Fn+k),Jxx(pnum,pnum2), dif = " << Wxx11(Fn+k) << '\t'
//                       <<Jxx(pnum,j1 + 2*k) << '\t' << Wxx11(Fn+k) -Jxx(pnum,j1 + 2*k) << endl;
                        assert(almostEquals(MapR0Sub11(Fn+k),MapR(pnum,j1 + 2*k)));

//                    cout << "Wxx(Fn+k),Jxx(pnum,pnum2), dif = " << Wxx12(Fn+k) << '\t'
//                         <<Jxx(pnum,j1 + 2*k+1) << '\t' << Wxx12(Fn+k) -Jxx(pnum,j1 + 2*k+1) << endl;
                        assert(almostEquals(MapR0Sub12(Fn+k),MapR(pnum,j1 + 2*k+1)));
                    }
                }
            }
        }


        // for sublatice 1
        for (int rowspin = 0; rowspin < L2; ++rowspin) {
            for (int colspin = 0; colspin < L1; ++colspin) {
                int pnum = rowspin*(2*L1) + 2*colspin+1;
                assert(PSubLattice(pnum) == 1);
                for (int j = 0; j < L2; ++j) {
                    int Fn = 2*(L1)*(L2+j-rowspin)-colspin+(L1);
                    int j1 = (2*L1)*j;
                    for (int k = 0; k < (L1); ++k) {
                        int pnum2 = j1 + 2*k;
                        assert(j1 + 2*k < N);
                        assert(j1 + 2*k+1 < N);
                        assert(Fn+k < 4*Nbase);
                        assert(pnum < N);
                        MapR0Sub21(Fn+k) = MapR(pnum,j1 + 2*k);
                        MapR0Sub22(Fn+k) = MapR(pnum,j1 + 2*k+1);
                    }
                }
            }
        }

        for (int rowspin = 0; rowspin < L2; ++rowspin) {
            for (int colspin = 0; colspin < L1; ++colspin) {
                int pnum = rowspin*(2*L1) + 2*colspin+1;
                assert(PSubLattice(pnum) == 1);
                for (int j = 0; j < L2; ++j) {
                    int Fn = 2*(L1)*(L2+j-rowspin)-colspin+(L1);
                    int j1 = (2*L1)*j;
                    for (int k = 0; k < (L1); ++k) {
                        int pnum2 = j1 + 2*k;
//                    cout << "Wxx(Fn+k),Jxx(pnum,pnum2), dif = " << Wxx21(Fn+k) << '\t'
//                         <<Jxx(pnum,j1 + 2*k) << '\t' << Wxx21(Fn+k) -Jxx(pnum,j1 + 2*k) << endl;
                        assert(almostEquals(MapR0Sub21(Fn+k),MapR(pnum,j1 + 2*k)));

//                    cout << "Wxx(Fn+k),Jxx(pnum,pnum2), dif = " << Wxx22(Fn+k) << '\t'
//                         <<Jxx(pnum,j1 + 2*k+1) << '\t' << Wxx22(Fn+k) -Jxx(pnum,j1 + 2*k+1) << endl;
                        assert(almostEquals(MapR0Sub22(Fn+k),MapR(pnum,j1 + 2*k+1)));
                    }
                }
            }
        }

    }

    cout << "set MapR0 done" <<endl;
}


std::pair<int, int> Lattice2d::periodicParticle(int x, int y, int x0, int y0) const{
    int xt = (x-x0+L1)%L1;
    int yt = (y-y0+L2)%L2;
    return make_pair(xt,yt);
}

int Lattice2d::periodicParticle(int p, int pBase) const {
    int x = p % L1;
    int y = p / L1;
    int x0 = pBase % L1;
    int y0 = pBase / L1;
    int xt = (x-x0+L1)%L1;
    int yt = (y-y0+L2)%L2;
    int pend = xt + yt * L1;
    return pend;
}

std::pair<int, int> Lattice2d::unperiodicParticle(int x, int y, int x0, int y0) const{
    int xt = (x-x0+L1)%L1;
    int yt = (y-y0+L2)%L2;
    return make_pair(xt,yt);
}

int Lattice2d::unperiodicParticle(int p, int pBase) const{
    int x = p % L1;
    int y = p / L1;
    int x0 = pBase % L1;
    int y0 = pBase / L1;
    int xt = (x+x0+L1)%L1;
    int yt = (y+y0+L2)%L2;
    int pend = xt + yt * L1;
    return pend;
}



void Lattice2d::calcNeighbours(int n) {
    vector<int> NeighboursVector;
    if (n < 1) {
        n = DistanceVec1p.size() - 1 + n;  // n=-1  -> n= DistanceVec1p.size() - 2
    }
    assert(n > 0 && n < DistanceVec1p.size());
    rassert(n > 0 && n < DistanceVec1p.size());
    for (int i = 1; i < N; ++i) {
        double minDisi = getMinDistance(0, i);
        if (minDisi < DistanceVec1p(n) || almostEquals(minDisi, DistanceVec1p(n))){
            NeighboursVector.push_back(i);
        }
    }
    Neighbours.setZero(NeighboursVector.size());
    for (int i = 0; i < NeighboursVector.size(); ++i) {
        Neighbours(i) = NeighboursVector[i];
    }
//    Neighbours = Map<VectorXd> (NeighboursVector.data(), NeighboursVector.size());
    std::sort(Neighbours.begin(), Neighbours.end());   // it's sorted by the way but just  in case....
    cout << "DistanceVec1p(n) = "<< endl << DistanceVec1p(n)<< endl;
//    cout << "Neighbours = "<< endl << Neighbours<< endl;
    cout << "Neighbours.size() = "<< endl << Neighbours.size()<< endl;
//    cout << "Neighbours = "<< endl << Neighbours<< endl;

//    for (int i = 0; i < Neighbours.size(); ++i) {
//        cout <<  "Distance "<< Neighbours(i) << " = " << getMinDistance(0, Neighbours(i)) << endl;
//    }

    Xn0.setZero(Neighbours.size());
    Yn0.setZero(Neighbours.size());
    for (int j = 0; j < Neighbours.size(); ++j) {
        int pj = Neighbours(j);
        rassert(!almostEquals(getMinDistance(0, pj), L1)  && !almostEquals(getMinDistance(0, pj), L2));
        rassert(getMinDistance(0, pj) <= L1);
        Xn0(j) = Neighbours(j) % L1;
        Yn0(j) = Neighbours(j) / L1;
    }
    if (Prop.isStoreNeighboursForAllPariticles){
        calcNeighboursMat();
    }

//    cout << "size of Neighbours = " << Neighbours.size() << endl;
//    cout << "Neighbours = " << Neighbours << endl;
}

void Lattice2d::calcNeighboursMat() {
    NeighboursMat.setZero(Neighbours.size(), N);
//    JxxMatn.setZero(Neighbours.size(), N);
//    JyyMatn.setZero(Neighbours.size(), N);
//    JxyMatn.setZero(Neighbours.size(), N);
    for (int i = 0; i < N; ++i) {
        int xi = i % L1; // TODO optimize by XPbyInedx
        int yi = i / L1;
        for (int j = 0; j < Neighbours.size(); ++j) {
            const int xj0 = Xn0(j) + xi;
            const int yj0 = Yn0(j) + yi;
            const int xj = (xj0 >= L1)? xj0 - L1 : xj0;
            const int yj = (yj0 >= L2)? yj0 - L2 : yj0;

//            int xj = (Lat.Xn0(j) + xi) % L1;
//            int yj = (Lat.Yn0(j) + yi) % L2;
            const int pj = yj * L1 + xj;
//            assert(minDisi < Lat.DistanceVec1p(n) || almostEquals(minDisi, Lat.DistanceVec1p(n)));
            assert(almostEquals(getMinDistance(0,Neighbours(j)), getMinDistance(i, pj)));
//            assert(almostEquals(Lat.getMinDistanceVec(0,Lat.Neighbours(j)), Lat.getMinDistanceVec(i, pj)));
//            assert(almostEquals(Lat.Jxxn(j), Lat.Jxx1p(Lat.Neighbours(j))));
//            assert(almostEquals(Lat.Jxxn(j), Lat.Jxx(0,Lat.Neighbours(j))));
//            assert(almostEquals(Lat.Jxxn(j), Lat.Jxx(i,pj)));
            NeighboursMat(j, i) = pj;
        }
    }

//    for(auto col : NeighboursMat.colwise()){
//        std::sort(col.begin(), col.end());
//    }
//    for (int i = 0; i < NeighboursMat.cols(); ++i) {
//        for (int j = 0; j < NeighboursMat.rows(); ++j) {
//            int pj = NeighboursMat(j, i);
//            int pjPeriodic = periodicParticle(pj, i);
//            JxxMatn(j, i) = Jxx1p(pjPeriodic);
//            JyyMatn(j, i) = Jyy1p(pjPeriodic);
//            JxyMatn(j, i) = Jxy1p(pjPeriodic);
//        }
//    }
//    cout << "NeighboursMat = " << endl << NeighboursMat <<endl;


//    for(auto col : NeighboursMat.colwise()){  //TODO it need to be jxxn, jxyn, jyyn to Matrix and sortedd the same way
//        std::sort(col.begin(), col.end());
//    }
    cout << "calcNeighboursMat done!" <<endl;

}

void Lattice2d::setJNeighbors() {
    rassert(Jxxn.size() == 0);
    Jxxn.setZero(Neighbours.size());
    Jxyn.setZero(Neighbours.size());
    Jyyn.setZero(Neighbours.size());
    Jzzn.setZero(Neighbours.size());
    assert(Neighbours.size() == Jxxn.size());
    cout << "Jxxn.size() = " << Jxxn.size() <<endl;

//    computeInteractionTensorEwald1p();
//    computeInteractionTensorEwald();
//    for (int i = 0; i < Neighbours.size(); ++i) {
//        Jxxn(i) = Jxx1p(Neighbours(i));
//        Jxyn(i) = Jxy1p(Neighbours(i));
//        Jyyn(i) = Jyy1p(Neighbours(i));
//        Jzzn(i) = Jzz1p(Neighbours(i));
//    }
//    Jself(0,0) = Jxx1p(0);
//    Jself(0,1) = Jxy1p(0);
//    Jself(1,0) = Jxy1p(0);
//    Jself(1,1) = Jyy1p(0);
//    Jself(2,2) = Jzz1p(0);

    int i = 0;
    for (int j = 0; j < Neighbours.size(); ++j) {
        int pj = Neighbours(j);
        double Jxxij = 0;
        double Jyyij = 0;
        double Jzzij = 0;
        double Jxyij = 0;
        Vector2d R = getMinDistanceVec(i, pj);
        rassert(!almostEquals(getMinDistance(i, pj), L1)  && !almostEquals(getMinDistance(i, pj), L2));
        rassert(getMinDistance(i, pj) <= L1);
//        cout << "R = " << R.transpose() << endl;
        double x = R(0);
        double y = R(1);
        double Rnorm = R.norm();
        double Rnorm2 = Rnorm * Rnorm;
        double invR5 = 1. / (Rnorm2 * Rnorm2 * Rnorm);
        double x2 = x * x;
        double y2 = y * y;
        Jxxn(j) += (-2 * x2 + y2) * invR5 /2;
        Jyyn(j) += (-2 * y2 + x2) * invR5 /2;
        Jzzn(j) += (x2 + y2)      * invR5 /2;
        Jxyn(j) += (-3 * x * y)   * invR5 /2;
    }
//    Jself(0,0) = 0;
//    Jself(0,1) = 0;
//    Jself(1,0) = 0;
//    Jself(1,1) = 0;
//    Jself(2,2) = 0;

    cout << "setJNeighbors() done!" <<endl;
}

Vector2d Lattice2d::getMinDistanceVec(int i, int j) const {
    Vector2d rtemp;
    Vector2d Dij;
    Dij =  D(all, i) - D(all, j);
    double MinDisX = DBL_MAX;
    double MinDisY = DBL_MAX;
    double MinDis = DBL_MAX;

    for (int k = -5; k < 5; ++k) {
        for (int l = -5; l < 5; ++l) {
            rtemp = Dij  + k * T(all, 0) + l * T(all, 1);
            if (rtemp.norm() < MinDis) {
                MinDis = rtemp.norm();
                MinDisX = rtemp(0);
                MinDisY = rtemp(1);
            }
//            if (abs(rtemp(0)) < abs(MinDisX)) {
//                MinDisX = rtemp(0);
//            }
//            if (abs(rtemp(1)) < abs(MinDisY)) {
//                MinDisY = rtemp(1);
//            }
        }
    }
    Vector2d DijMin;
    DijMin(0) = MinDisX;
    DijMin(1) = MinDisY;
    assert(!almostEquals(MinDis, L1) && !almostEquals(MinDis, L2));
//    cout << "DijMin.norm() = " << DijMin.norm() << endl;
//    cout << "getMinDistance(i,j) = " << getMinDistance(i,j) << endl;
    assert(almostEquals(DijMin.norm(), getMinDistance(i,j)));
    return DijMin;
}

void Lattice2d::setPAndRijAndJmaxByDistance() {
    PByDistance.setZero(N - 1);
    XPByDistance.setZero(N - 1);
    YPByDistance.setZero(N - 1);
    R0jBydistance.setZero(N - 1);
    JmaxByDistance.setZero(N - 1);
    for (int j = 1; j < N; ++j) {
        PByDistance(j - 1) = j;
        XPByDistance(j - 1) = j % L1;
        YPByDistance(j - 1) = j / L1;
        R0jBydistance(j - 1) = getMinDistance(0, j);
        JmaxByDistance(j - 1) = Jmax1p(j);
    }
    // sorting by distance from close to far
    for (int i = 0; i < PByDistance.size(); ++i) {
        for (int j = i; j < PByDistance.size(); ++j) {
            if (R0jBydistance(j) < R0jBydistance(i)) {
                swap(R0jBydistance(i), R0jBydistance(j));
                swap(PByDistance(i), PByDistance(j));
                swap(XPByDistance(i), XPByDistance(j));
                swap(YPByDistance(i), YPByDistance(j));
                swap(JmaxByDistance(i), JmaxByDistance(j));
            }
        }
    }


}

void Lattice2d::setJNeighborsEwald() {
    rassert(Jxxn.size() == 0);
    Jxxn.setZero(Neighbours.size());
    Jxyn.setZero(Neighbours.size());
    Jyyn.setZero(Neighbours.size());
    Jzzn.setZero(Neighbours.size());
    assert(Neighbours.size() == Jxxn.size());
    cout << "Jxxn.size() = " << Jxxn.size() <<endl;

    computeInteractionTensorEwald1p();
//    computeInteractionTensorEwald();
    for (int i = 0; i < Neighbours.size(); ++i) {
        Jxxn(i) = Jxx1p(Neighbours(i));
        Jxyn(i) = Jxy1p(Neighbours(i));
        Jyyn(i) = Jyy1p(Neighbours(i));
        Jzzn(i) = Jzz1p(Neighbours(i));
    }
    //TODO insert Jself if needed
//    Jself(0,0) = Jxx1p(0);
//    Jself(0,1) = Jxy1p(0);
//    Jself(1,0) = Jxy1p(0);
//    Jself(1,1) = Jyy1p(0);
//    Jself(2,2) = Jzz1p(0);


//    Jself(0,0) = 0;
//    Jself(0,1) = 0;
//    Jself(1,0) = 0;
//    Jself(1,1) = 0;
//    Jself(2,2) = 0;


    cout << "setJNeighborsEwald() done!" <<endl;
}

void Lattice2d::setXYPByIndex() {
    XPbyIndex.setZero(N);
    YPbyIndex.setZero(N);
    for (int i = 0; i < N; ++i) {
        XPbyIndex(i) = i % L1;
        YPbyIndex(i) = i / L1;
    }
}

void Lattice2d::decreaseDipolarStrength() {
    double ratio = Prop.DipolarStrengthToExchangeRatio;
    cout << "DipolarStrengthToExchangeRatio = " << ratio <<endl;
//    cout << "Jxx1p = "  << endl << Jxx1p <<endl;
    Jxx1p = (Jxx1p.array() * ratio).eval();
    Jxy1p = (Jxx1p.array() * ratio).eval();
    Jyy1p = (Jxx1p.array() * ratio).eval();
    Jzz1p = (Jxx1p.array() * ratio).eval();

    vector<int> NNVector;
    int n = 1;
    assert(n > 0 && n < DistanceVec1p.size());
    for (int i = 1; i < N; ++i) {
        double minDisi = getMinDistance(0, i);
        if (minDisi < DistanceVec1p(n) || almostEquals(minDisi, DistanceVec1p(n))){
            NNVector.push_back(i);
        }
    }
    //ferromagnetic interacion
    if (Prop.isExchangeInteractionCombinedWithJ) {
        for (int i: NNVector) {
            Jxx1p(i) -= 1. / 2;  // TODO give reason why it gives correct results
            Jyy1p(i) -= 1. / 2;
            Jzz1p(i) -= 1. / 2;
        }
    }

    VectorXi NNVectorEigen = Map<VectorXi> (NNVector.data(), NNVector.size());
//    cout << "Number of Nearest Neighbours  = " << NNVector.size()<< endl;
//    cout << "NNVector = "<< endl << NNVectorEigen << endl;

}


void Lattice2d::calcNearestNeighbours() {
    vector<int> NeighboursVector;
    int n = 1;
    if (n < 1) {
        n = DistanceVec1p.size() - 1 + n;  // n=-1  -> n= DistanceVec1p.size() - 2
    }
    assert(n > 0 && n < DistanceVec1p.size());
    for (int i = 1; i < N; ++i) {
        double minDisi = getMinDistance(0, i);
        if (minDisi < DistanceVec1p(n) || almostEquals(minDisi, DistanceVec1p(n))){
            NeighboursVector.push_back(i);
        }
    }
    NearestNeighbours.setZero(NeighboursVector.size());
    for (int i = 0; i < NeighboursVector.size(); ++i) {
        NearestNeighbours(i) = NeighboursVector[i];
    }
//    Neighbours = Map<VectorXd> (NeighboursVector.data(), NeighboursVector.size());
    std::sort(NearestNeighbours.begin(), NearestNeighbours.end());   // it's sorted  already by the way but just  in case....
//    cout << "DistanceVec1p(n) = "<< endl << DistanceVec1p(n)<< endl;
    cout << "NearestNeighbours.size() = "<< NearestNeighbours.size()<< endl;
    cout << "NearestNeighbours = "<< endl << NearestNeighbours<< endl;
//    cout << "Neighbours = "<< endl << Neighbours<< endl;

//    for (int i = 0; i < Neighbours.size(); ++i) {
//        cout <<  "Distance "<< Neighbours(i) << " = " << getMinDistance(0, Neighbours(i)) << endl;
//    }

//    Xn0.setZero(Neighbours.size());
//    Yn0.setZero(Neighbours.size());
//    for (int j = 0; j < Neighbours.size(); ++j) {
//        int pj = Neighbours(j);
//        rassert(!almostEquals(getMinDistance(0, pj), L1)  && !almostEquals(getMinDistance(0, pj), L2));
//        rassert(getMinDistance(0, pj) <= L1);
//        Xn0(j) = Neighbours(j) % L1;
//        Yn0(j) = Neighbours(j) / L1;
//    }
}

void Lattice2d::setJmax1p() {
    Jmax1p.setZero(Jxx1p.size());

    for (int i = 1; i < Jxx1p.size(); ++i) {
        Eigen::Matrix2d A;
        A << Jxx1p(i), Jxy1p(i),
             Jxy1p(i), Jyy1p(i);
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigensolver(A);
        if (eigensolver.info() != Eigen::Success) abort();
//        cout << "eigensolver.eigenvalues() = " << eigensolver.eigenvalues() <<endl;
        double Jmax = eigensolver.eigenvalues().array().abs().maxCoeff();

        double ratio = 1;
        if (Prop.isHavingExchangeInteraction){
            ratio = Prop.DipolarStrengthToExchangeRatio;
        }
        double r0j = getMinDistance(0, i);
        assert(r0j > 0);        // TODO set max prob using J interactions eigenvalues
        double lambda = 4.6 * ratio/(pow(r0j, 3)) / 4;
        double powr0j = ratio/(pow(r0j, 3));
//        assert(lambda > Jmax);
//        assert(Jmax > lambda/2);
        Jmax1p(i) = Jmax;
    }
}

void Lattice2d::setPAndRijByDistanceFar() {
    auto nn = Neighbours.size()-1;
    R0jBydistanceFar = (R0jBydistance(seq(nn, last))).eval();   // another candidate
    PByDistanceFar = (PByDistance(seq(nn, last))).eval();   // another candidate
    XPByDistanceFar = (XPByDistance(seq(nn, last))).eval();   // another candidate
    YPByDistanceFar = (YPByDistance(seq(nn, last))).eval();   // another candidate


//    PByDistance = (PByDistance(seq(nn, last))).eval();   // another candidate

    R0jBydistanceFar(0) = 0;
    PByDistanceFar(0) = 0;
    XPByDistanceFar(0) = 0;
    YPByDistanceFar(0) = 0;
    //    std::reverse(PByDistance.begin(), PByDistance.end());
//    std::reverse(R1jBydistance.begin(), R1jBydistance.end());
//    cout << "PByDistance = " << PByDistance<< endl <<endl;
//    cout << "R1jBydistance = " << R0jBydistance << endl <<endl;
    cout << "R1jBydistance.size() = " << R0jBydistanceFar.size() << endl << endl;
}

void Lattice2d::calcNeighboursbyNearGroupEnergyProportion(double proportion) {
    VectorXd JmaxCumulVec;
    JmaxCumulVec.setZero(JmaxByDistance.size());
    JmaxCumulVec(0) = JmaxByDistance(0);
    for (int i = 1; i < JmaxByDistance.size(); ++i) {
        JmaxCumulVec(i) = JmaxCumulVec(i - 1) + JmaxByDistance(i);
    }
//    cout << "JmaxCumulVec = " << JmaxCumulVec <<endl;
    VectorXd JmaxCumulNormalizedVec;
    const int lastJmaxCumulVecIndex = JmaxCumulVec.size()-1;
    JmaxCumulNormalizedVec = JmaxCumulVec / JmaxCumulVec(lastJmaxCumulVecIndex);
    const auto lastNeighbourIterator = std::upper_bound(JmaxCumulNormalizedVec.begin(),
                                                    JmaxCumulNormalizedVec.end(),
                                                    proportion);
    const int lastNeighbourIndex = std::distance(JmaxCumulNormalizedVec.begin(),
                                                 lastNeighbourIterator);
    assert(lastNeighbourIndex > 0 && lastNeighbourIndex < JmaxCumulNormalizedVec.size());


    double minDisi = getMinDistance(0, PByDistance(lastNeighbourIndex));
    const auto lastNeighbourByDistanceIterator = std::upper_bound(DistanceVec1p.begin(),
                                                        DistanceVec1p.end(),
                                                        minDisi);
    const int lastNeighbourByDistanceIndex = std::distance(DistanceVec1p.begin(),
                                                           lastNeighbourByDistanceIterator);
    assert(lastNeighbourByDistanceIndex > 0 && lastNeighbourByDistanceIndex < DistanceVec1p.size());
    calcNeighbours(lastNeighbourByDistanceIndex);


// TODO  assertion assert(almostEquals(dErot(i),dENN(i))); fails when lastNeighbourIndex does not contain
// all particles of same distance. e.g. it dows not fail when number of neghbours is 4, 8 for square lattice but fails
// when its 7, 6
//    Neighbours = PByDistance(seq(0, lastNeighbourIndex));
//    std::sort(Neighbours.begin(), Neighbours.end());
//    Xn0.setZero(Neighbours.size());
//    Yn0.setZero(Neighbours.size());
//    for (int j = 0; j < Neighbours.size(); ++j) {
//        int pj = Neighbours(j);
//        rassert(!almostEquals(getMinDistance(0, pj), L1)  && !almostEquals(getMinDistance(0, pj), L2));
//        rassert(getMinDistance(0, pj) <= L1);
//        Xn0(j) = Neighbours(j) % L1;
//        Yn0(j) = Neighbours(j) / L1;
//    }
//    if (Prop.isStoreNeighboursForAllPariticles){
//        calcNeighboursMat();
//    }

    cout << "size of Neighbours = " << Neighbours.size() << endl;
//    cout << "Neighbours = " << Neighbours << endl;


}
