// a collection of unused or faulty or algorithmically wrong code
// Created by Sadeq Ismailzadeh on ۱۱/۰۲/۲۰۲۳.
//

#include "MC2d.h"
#include "WalkerAlias.h"
#include <highfive/H5Easy.hpp>
#ifdef _OPENMP // Use omp.h if -fopenmp is used in g++
#include <omp.h>
#endif


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


double MC2d::PRelAddUp(const vector<int> &jrejVecAddUp, int i, int xi, int yi,
                       vector<double> &PHatOldVecAddUp) {  // jrej is sorted by distance and is periodic
//    cout << "PRel" <<endl;
    double dE = 0;
    double PHatOldprod = 1;
    assert(PHatOldVecAddUp.size() > jrejVecAddUp.size());
    const double dMx = UiNewX - Ux(i);
    const double dMy = UiNewY - Uy(i);
    for (int k = 0; k < jrejVecAddUp.size(); ++k) {
        int jrej = jrejVecAddUp[k];
        PHatOldprod *= PHatOldVecAddUp[k];
        assert(jrej >=0 && jrej < N-1);
//    cout << "jrej from PRel = " << jrej << endl;
//    int jper = Lat.PByDistance(jrej);    //jper is periodic
//    int j = Lat.unperiodicParticle(jper, i);
//    assert(j >=0 && j < N);
//    assert(jper >=0 && jper < N);
//    int jperback = Lat.periodicParticle(j, i);
//    assert(jper == jperback);
//    assert(almostEquals(Lat.getMinDistance(i,j), Lat.getMinDistance(0, jper)));
//    double Phatj = ProbClock(jrej);
        assert(Lat.XPByDistance(jrej) + Lat.YPByDistance(jrej) * Lat.L1 == Lat.PByDistance(jrej));
        const int xjper = Lat.XPByDistance(jrej);
        const int yjper = Lat.YPByDistance(jrej);
        const int xj0 = xjper + xi;
        const int yj0 = yjper + yi;
        const int xj = (xj0 >= Lat.L1)? xj0 - Lat.L1 : xj0;
        const int yj = (yj0 >= Lat.L1)? yj0 - Lat.L1 : yj0;
        int j2 = yj * Lat.L1 + xj;
        int jper =  yjper * Lat.L1 + xjper;
//    assert(j2 == j);
//    int jper = Lat.periodicParticle(j, i);
        assert(jper >=0 && jper < N);
        dE += 2*(dMx * (Lat.Jxx1p(jper) * Ux(j2) + Lat.Jxy1p(jper) * Uy(j2)) +
                 dMy * (Lat.Jxy1p(jper) * Ux(j2) + Lat.Jyy1p(jper) * Uy(j2)));
    }



//    assert(almostEquals(dE, dEij(i,j)));

//    double Phatj = InvOneMinusPHatOld;
//    double Pj = exp(-max(dEij(i,j), 0.)/Temperature);
    if (dE <= 0.){
        return 0.;
    }

    assert(almostEquals(InvTemperature, 1/Temperature));
    double Pj = exp(- dE * InvTemperature);
//    double Pj = exp(-max(dEij(i,j2), 0.) * InvTemperature);
    assert((1 - PHatOldprod) >= (1 - Pj));
//    double prel = 1- (Pj - Phatj);
//    double prel = (1 - Pj) / (1 - Phatj);
    double prel = (1 - Pj) / (1 - PHatOldprod);
//    double prel = (Pj - Phatj) / (1 - Phatj);
//    cout << "prel = " <<prel << endl;
    return prel;
}

// TODO chack that what is run1ClockParticleAddUp?
void MC2d::run1ClockParticleAddUp(int i) {
//    cout << "run1ClockParticle" <<endl;
    int jrej = 0;     //jrej is sorted by distance and is periodic
    Debug_totali++;
//    const int i = intDis(gen); //sorted by Index
//    const int i = Random(N); //sorted by Index
//    const int xi = i % Lat.L1;
//    const int yi = i / Lat.L1;
    const int xi = Lat.XPbyIndex(i);
    const int yi = Lat.YPbyIndex(i);
    assert(xi == i % Lat.L1);
    assert(yi == i / Lat.L1);
//    randomizeUiNew();
//    randomAdaptive(i);
//    cout << "concensusProb(i) = " << concensusProb(i) << endl;
    vector<int> jrejVecAddUp;
    vector<double> PHatOldVecAddUp;
    double nu = realDis(gen);
//    double nu = Random();
    double invOneMinusPHatOld = InVOneMinusProbClock(jrej);
    PHatOldVecAddUp.push_back(ProbClock(jrej));
    assert(int(floor(log(nu)/log(ProbClock(jrej)))) == int(log(nu)/log(ProbClock(jrej))));
//    jrej = jrej + floor(log(nu)/log(ProbClock(jrej)));
    assert(almostEquals(InVLogProbClock(jrej), 1 / log(ProbClock(jrej))));
    jrej = jrej + (log(nu) * InVLogProbClock(jrej));

    while(true){
//        jrej = jrej + (1 + log(nu)/log(ProbClock(jrej)));
//        cout << "jrej = " <<  jrej <<endl;
//        assert((jrej >= 0 && jrej < N-2));
//        assert(ProbOld <= ProbClock(jrej));
        if (jrej > Lat.PByDistance.size() - 1){ // there is totally N-1 bouds indexed from 0. so the index of the last bond is N-2
//            Debug_Iaccepted++;    // move accepted
//            updateState(i);
//            AdapticeCounterAcceptance++;
//            cout << "i = " << i << " accepted" <<endl;
            break;
        }
        jrejVecAddUp.push_back(jrej);
        NumResampled++;
        assert(jrej >=0 && jrej < N-1);
        nu = realDis(gen);
//        nu = Random();
//        ProbOld = ProbClock(jrej);
        invOneMinusPHatOld = InVOneMinusProbClock(jrej);
        PHatOldVecAddUp.push_back(ProbClock(jrej));
//        assert(almostEquals(InVOneMinusProbClock(jrej), 1/(1 - ProbOld)));
        assert(almostEquals(InVLogProbClock(jrej), 1 / log(ProbClock(jrej))));
        assert(int(floor(log(nu)/log(ProbClock(jrej)))) == int(log(nu)/log(ProbClock(jrej))));
//        jrej = jrej + floor(1 + log(nu) * InVLogProbClock(jrej));
        jrej = jrej + (1 + log(nu) * InVLogProbClock(jrej));
    }

//    cout << "jrejVecAddUp = " <<endl;
//    for (int k : jrejVecAddUp){
//        cout << k << "  " ;
//    }
//    for_each(jrejVecAddUp.begin(), jrejVecAddUp.end(),
//             [](const auto& elem) { cout << elem << " "; });
//    cout << endl;
    double prel = PRelAddUp(jrejVecAddUp, i, xi, yi, PHatOldVecAddUp);
    if (!(prel > 0. && realDis(gen) <= prel)){
        Debug_Iaccepted++;    // move accepted
        updateState(i);
//            cout << "i = " << i << " rejected" <<endl;
//        JrejSum(jrej)++;
    }
}

void MC2d::run1ClockParticleFar(int i) {
//jrej is sorted by distance and is periodic, jrej is index of the first bond to be rejected.
// bonds indeices start from zero
// initially no bond has rejected move. so jrej=-1
// in the paper bonds start from 1 and irej is initially zero;
//but we need value of ProbClock(jrej) at jrej = 0.
// so we modified recurtion relation  { jrej = jrej + (1 + log(nu) * InVLogProbClock(jrej))  }
// to fit our purposes
    int jrej = 0;
    Debug_totali++;
//    const int i = intDis(gen); //sorted by Index
//    const int i = Random(N); //sorted by Index
//    const int xi = i % Lat.L1;
//    const int yi = i / Lat.L1;
    const int xi = Lat.XPbyIndex(i);
    const int yi = Lat.YPbyIndex(i);
    assert(xi == i % Lat.L1);
    assert(yi == i / Lat.L1);
    double nu;
//    nu = realDis(gen);   // initial jrej disabled
    double invOneMinusPHatOld = InVOneMinusProbClockFar(jrej);
    assert(int(floor(log(nu)/log(ProbClockFar(jrej)))) == int(log(nu)/log(ProbClock(jrej))));
    assert(almostEquals(InVLogProbClockFar(jrej), 1 / log(ProbClockFar(jrej))));
//    jrej = jrej + (log(nu) * InVLogProbClock(jrej)); // initial jrej disabled
    while(true){
//        jrej = jrej + (1 + log(nu)/log(ProbClock(jrej)));
//        assert((jrej >= 0 && jrej < N-2));
//        assert(ProbOld <= ProbClock(jrej));
        if (jrej > Lat.PByDistanceFar.size() - 1){
            // there is totally N-1 bouds indexed from 0. so the index of the last bond is N-2
            updateField(i);
            updateState(i);
            Debug_Iaccepted++;    // move accepted
//            AdapticeCounterAcceptance++;
            break;
        } else  {
            double prel = PRelFar(jrej, i, xi, yi, invOneMinusPHatOld);
            if (prel > 0. && realDis(gen) <= prel){
//                if (jrej != 0) {
//                    JrejSum(jrej)++;
//                    break;   // move rejected
//                } else {
//                    overRelaxateUiNew(i);
//                }
                JrejSum(jrej)++;
                break;   // move rejected
            }
        }
        NumResampled++;
        assert(jrej >=0 && jrej < Lat.PByDistanceFar.size());
        nu = realDis(gen);
//        nu = Random();
//        ProbOld = ProbClock(jrej);
        invOneMinusPHatOld = InVOneMinusProbClockFar(jrej);
//        assert(almostEquals(InVOneMinusProbClock(jrej), 1/(1 - ProbOld)));
        assert(almostEquals(InVLogProbClockFar(jrej), 1 / log(ProbClockFar(jrej))));
        assert(int(floor(log(nu)/log(ProbClockFar(jrej)))) == int(log(nu)/log(ProbClockFar(jrej))));
//        jrej = jrej + floor(1 + log(nu) * InVLogProbClock(jrej));
        jrej = jrej + (1 + log(nu) * InVLogProbClockFar(jrej));
    }
}
void MC2d::setProbClockFar() {
    setProbClock();
//    cout << "setProbClock" <<endl;
    ProbClockFar.setZero();
    InVLogProbClockFar.setZero();
    InVOneMinusProbClockFar.setZero();
    double ratio = 1;
    double lambdaTot = 0;

    if (Prop.isHavingExchangeInteraction){
        ratio = Prop.DipolarStrengthToExchangeRatio;
    }
    for (int j = 1; j < ProbClockFar.size(); ++j) {
        double r0j = Lat.R0jBydistanceFar(j);
        assert(r0j > 0);        // TODO set max prob using J interactions
        double lambda = 8.6 * ratio/(Temperature * pow(r0j, 3));
        ProbClockFar(j) = exp(- lambda);  //exp(-2*beta*2/r^3)
        lambdaTot += lambda;
        assert(ProbClockFar(j) > 0);
    }
//    std::reverse(ProbClock.begin(), ProbClock.end());

//    ProbClock(0) = 1;
//    for (int i = 0; i < Lat.Neighbours.size(); ++i) {
//        double r0j = Lat.getMinDistance(0, Lat.Neighbours(i));
//        ProbClock(0) *= exp(-8.5/(Temperature * pow(r0j, 3)));
////        cout << "ProbClock(0) = " <<ProbClock(0) <<endl;
//    }

    ProbClockFar(0) = 0;
    InVLogProbClockFar(0) = 0;
    InVOneMinusProbClockFar(0) = 1 / (1 - ProbClockFar(0));
//    assert(ProbClock(0) > 0.);

    for (int j = 1; j < ProbClockFar.size(); ++j) {
        InVLogProbClockFar(j) = 1 / log(ProbClockFar(j));  //exp(-2*beta*2/r^3)
        InVOneMinusProbClockFar(j) = 1 / (1 - ProbClockFar(j));  //exp(-2*beta*2/r^3)
    }
//    cout << "ProbClockNN = " << ProbClock(0) << endl <<endl;
//    cout << "ProbClock = " << ProbClock(seq(1,min(5L, ProbClock.size()-1)))  << endl <<endl;
    cout << "ProbClock.prod() = " << ProbClockFar(seq(1, last)).prod() << endl;
    cout << "p(1) = " << ProbClockFar(1) << endl;
    cout << "lambda(1) = " << -log(ProbClockFar(1)) << endl;
    cout << "pow(p(1), Nfar) = " << pow(ProbClockFar(1), ProbClockFar.size() - 1) << endl;
    cout << "lambdaTot = " << lambdaTot  <<endl;
    std::poisson_distribution<int> PoissonDis;
    PoissonDis = std::poisson_distribution<int>(lambdaTot);
    cout << "Random poisson = " << PoissonDis(gen)  <<endl;

    int NumResampledii = 0;
    int Nloop = 100;
    for (int i = 0; i < Nloop; ++i) {
        int jrej = 0;
        double nu;
        double invOneMinusPHatOld = InVOneMinusProbClockFar(jrej);
        while(true){
            if (jrej > Lat.PByDistanceFar.size() - 1){
                break;
            }
            NumResampledii++;
            nu = realDis(gen);
            invOneMinusPHatOld = InVOneMinusProbClockFar(jrej);
            jrej = jrej + (1 + log(nu) * InVLogProbClockFar(jrej));
        }
    }
    cout << "lambda bernoulli = " << NumResampledii/(Nloop + 0.) << endl;
}

double MC2d::PRelFar(int jrej, int i, int xi, int yi, double invOneMinusPHatOld) {  // jrej is sorted by distance and is periodic
//    cout << "PRel" <<endl;
//    cout << jrej <<endl;
    assert(jrej >=0 && jrej < Lat.PByDistanceFar.size());
//    cout << "jrej from PRel = " << jrej << endl;
//    int jper = Lat.PByDistance(jrej);    //jper is periodic
//    int j = Lat.unperiodicParticle(jper, i);
//    assert(j >=0 && j < N);
//    assert(jper >=0 && jper < N);
//    int jperback = Lat.periodicParticle(j, i);
//    assert(jper == jperback);
//    assert(almostEquals(Lat.getMinDistance(i,j), Lat.getMinDistance(0, jper)));
//    double Phatj = ProbClock(jrej);
    double dE = 0;
    if (jrej == 0){
//        cout << "i = " << i <<endl;
//        cout << "dErot(i) = " << dErot(i) <<endl;
//        cout << "dENN(i) = " << dENN(i) <<endl;
        assert(almostEquals(dErot(i),dENN(i)));
//        dE = dENN(i);
        dE = dErot(i);
//        assert(almostEquals(InvTemperature, 1/Temperature));
//        cout << "exp(- dE * InvTemperature) = " <<exp(- dE * InvTemperature)<<endl;
    } else {
        assert(Lat.XPByDistanceFar(jrej) + Lat.YPByDistanceFar(jrej) * Lat.L1 == Lat.PByDistanceFar(jrej));
        const int xjper = Lat.XPByDistanceFar(jrej);
        const int yjper = Lat.YPByDistanceFar(jrej);
        const int xj0 = xjper + xi;
        const int yj0 = yjper + yi;
        const int xj = (xj0 >= Lat.L1)? xj0 - Lat.L1 : xj0;
        const int yj = (yj0 >= Lat.L1)? yj0 - Lat.L1 : yj0;
        int j = yj * Lat.L1 + xj;
        int jper =  yjper * Lat.L1 + xjper;
//    assert(j2 == j);
        const double dMx = UiNewX - Ux(i);
        const double dMy = UiNewY - Uy(i);
//    int jper = Lat.periodicParticle(j, i);
        assert(jper >=0 && jper < N);
        assert(i != j);
        dE = 2*(dMx * (Lat.Jxx1p(jper) * Ux(j) + Lat.Jxy1p(jper) * Uy(j)) +
                dMy * (Lat.Jxy1p(jper) * Ux(j) + Lat.Jyy1p(jper) * Uy(j)));
    }

//    assert(almostEquals(dE, dEij(i,j)));

//    double Phatj = InvOneMinusPHatOld;
//    double Pj = exp(-max(dEij(i,j), 0.)/Temperature);

    if (dE <= 0.){
        return 0.;
    }

    assert(almostEquals(InvTemperature, 1/Temperature));
    double Pj = exp(- dE * InvTemperature);
//    double Pj = exp(-max(dEij(i,j2), 0.) * InvTemperature);
    assert(invOneMinusPHatOld >= (1 - Pj));
    assert((1 - ProbClockFar(jrej)) >= (1 - Pj));
//    double prel = 1- (Pj - Phatj);
//    double prel = (1 - Pj) / (1 - Phatj);
    double prel = (1 - Pj) * invOneMinusPHatOld;
//    double prel = (Pj - Phatj) / (1 - Phatj);
//    cout << "prel = " <<prel << endl;
//    if(jrej == 0){
//        cout << "prel = " << prel << endl;
//    }
    return prel;
}

void MC2d::run1FineTuningStep() {
    static const double PI = acos(-1);

//    shuffle(ParticleList.begin(), ParticleList.end(), gen);
//    for (int i : ParticleListSorted) {
//    cout << "E before Fine Tune= " << getEnergy()  <<endl;
    double Econfig = getEnergy();
    int attempt = 0;
    double factor = 1;
    if (Econfig - Eold > 0){
        factor = 1;
    } else {
        factor = -1;
    }
    while ((Econfig - Eold) * factor > 0) {
        shuffle(ParticleList.begin(), ParticleList.end(), gen);
        for (int i : ParticleList) {
//        for (int i = 0; i < N; ++i) {
//    for (int c = 0; c < N; c++) {
//        int i = intDis(gen);
//        Debug_totali++;

            double InvabsH = 1/sqrt(Hx(i)*Hx(i) + Hy(i)*Hy(i));
//            UiNewX = (-Hx(i) *InvabsH);
//            UiNewY = (-Hy(i) *InvabsH);
            UiNewX = ((-Hx(i) *InvabsH)*factor + 4*Ux(i));
            UiNewY = ((-Hy(i) *InvabsH)*factor + 4*Uy(i));
            double InvabsUiNew = 1/sqrt(UiNewX*UiNewX + UiNewY*UiNewY);
            UiNewX *= InvabsUiNew;
            UiNewY *= InvabsUiNew;
            Econfig += dErot(i);
            if ((Econfig - Eold)*factor < 0){
                Econfig -= dErot(i);
                double Echange = Eold - Econfig;
                double Eprime = Echange/2 + Ux(i)*Hx(i)+ Uy(i)*Hy(i);
                double R = sqrt(Hx(i)*Hx(i) + Hy(i)*Hy(i));
                double alpha = atan2(Hy(i), Hx(i));
//                cout << "abs(Eprime/R) = "<< abs(Eprime/R) <<endl;
                rassert(abs(Eprime/R) <= 1);
                double theta = acos(Eprime/R) + alpha;
                UiNewX = cos(theta);
                UiNewY = sin(theta);
                Econfig += dErot(i);
                updateField(i);
                updateState(i);
                break;
            }
            updateField(i);
            updateState(i);
            attempt++;
        }
        if (almostEquals(Econfig, Eold)){
            break;
        }
    }
    FineTuneNum++;
    attemptSum += attempt;
//    cout << "fine tuned after attempt " << attempt <<endl;
//    cout << "Econfig = " << Econfig <<endl;
//    cout << "Eold = " << Eold <<endl;
//    cout << "Ereal = " << getEnergy() <<endl;
//    cout << "Efft = " << getEnergyByFFT() <<endl;
    if (!almostEquals(Eold, getEnergy())) {
        cout << "fine tuned after attempt " << attempt <<endl;
        cout << "Econfig = " << Econfig <<endl;
        cout << "Eold = " << Eold <<endl;
        cout << "Ereal = " << getEnergy() <<endl;
        cout << "Efft = " << getEnergyByFFT() <<endl;
    }
    rassert(almostEquals(Eold, getEnergy()));
}



void MC2d::run1NewOverRelaxationStep() {
    run1RandomizerStep();
    run1FineTuningStep();
    for (int i = 0; i < 10; ++i) {
        run1MetropolisStep();
    }
    for (int j = 0; j < 100; ++j) {
        for (int i = 0; i < 1; ++i) {
            run1MetropolisStep();
        }

        for (int i = 0; i < 4; ++i) {
            run1OverRelaxationStep();
        }
    }
}

void MC2d::run1RandomizerStep(){
    static const double PI = acos(-1);

//    shuffle(ParticleList.begin(), ParticleList.end(), gen);
//    for (int i : ParticleListSorted) {
    Eold = getEnergy();
    for (int c = 0; c < N; c++) {
        int i = intDis(gen);
        randomizeUiNew();
        double dE = dErot(i);
//        if (dE < 0 || (realDis(gen) < exp(-dE /(Temperature*5)))) {
        updateField(i);
        updateState(i);
//        }
    }
    calcAllFields();
}

void MC2d::UpdateGMatOld() {
    if (Prop.LatticeType == 't') {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                GMat(i, j) += Ux(i) * Ux(i) * Ux(j) + Uy(i) * Uy(j);
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

void MC2d::run1ClockStepOriginal() {
    for (int j = 0; j < 10; ++j) {
        for (int i = 0; i < N; ++i) {
            randomizeUiNew();
            run1ClockParticleOriginal(i);
//            run1ClockParticleAddUp(i);
//            run1SCOParticle(i);
        }
    }
}