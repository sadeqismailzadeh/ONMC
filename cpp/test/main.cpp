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
#include "Eigen/Dense"
#include <filesystem>
#include "include/prng.h"


#include "2d/MC2d.h"

using namespace std;
using namespace Eigen;
namespace fs = std::filesystem;


int N = 16;
VectorXd U(3 * N);
//VectorXd H = VectorXd::Zero(3*N);
VectorXd H(3 * N);
MatrixXd J(3 * N, 3 * N);
MatrixXd indices(3 * N - 3, N);
VectorXd mi_new(3); //TODO pass into function

//taking data
MatrixXd StatisticalData(1000, 5);
int LastTIndex = 0;

//eigenvectors
MatrixXd Ueig = MatrixXd::Zero(3 * N, 2);


double T = 3;

int Debug_i = 0;
bool Debug_T_flag = false;


//std::mt19937 gen{std::random_device{}()};
std::mt19937 gen(time(nullptr));
//thread_local mt19937 gen(time(NULL) + hash<thread::id>()(this_thread::get_id()));
std::uniform_real_distribution<double> realDis(0, 1);
std::uniform_real_distribution<double> realDissym(-1, 1);
std::uniform_int_distribution<int> intDis(0, N - 1);
//std::random_device rd;
//default_random_engine gen(rd());



void init();

void calcAllFields();

void updateField(int);

double dErot(int);

double dErot_old(int);

void run1step();

void run1step_old();

void stabilize(int);

void stabilize_old(int);

double calcEnergy();

void makeIndices();

void getStatisticalData(int N_MCstep, int NStepToDecorrelate = 1);

inline double square(const double &a) { return a * a; }

void paralell();

void serial();




//std::random_device rd;
////std::mt19937 gen(rd());
//static thread_local mt19937 gen(clock() + hash<thread::id>()(this_thread::get_id()));
////std::mt19937 gen(clock() + this_thread::get_id().hash());
////std::mt19937 gen(static_cast<long unsigned int>(time(0)));  //set seed
//std::uniform_real_distribution<double> realDis(0, 1);
//std::uniform_int_distribution<int> intDis(0, N-1);
//std::uniform_real_distribution<double> dis2(0, 1);


long tic(int mode = 0);

long toc();

//int intRand(const int & max) {
//    static thread_local mt19937 generator(clock() + hash<thread::id>()(this_thread::get_id()));
//    uniform_int_distribution<int> distribution(0, max);
//    return distribution(generator);
//}
//
//int realRand() {
//    static thread_local mt19937 generator(clock() + hash<thread::id>()(this_thread::get_id()));
//    uniform_real_distribution<double> distribution(0, 1);
//    return distribution(generator);
//}

int main() {

//    cout << isnan(sqrt(-1));
    testInteractions();
    return 0;
}

int main2() {
    cout << "single thread = 1, multi thread = 2 \nwhich one? ";
    char Choice = '0';
    cin >> Choice;

    Randomize();
    cout << Random(std::numeric_limits<int32_t>::max())
         << endl;
    switch (Choice) {
        case '1' : {
            cout << "choosed single thread \n\n";
            serial();
            break;
        }
        case '2' : {
            cout << "choosed multi thread \n\n";
            paralell();
            break;
        }
        default :
            cout << "\nBad Input. Must be 1-2.";
    }
    return 0;
}

void paralell() {
    tic();
//    Eigen::initParallel();
    int N = 16;
    int NensembleDiv = 3;
    int Nthreads = 7;
//    int Nthreads =  std::thread::hardware_concurrency();
    int Nensemble = Nthreads * NensembleDiv;
    int NStabilize = 50000;
    int NData = 100000;
    int NDecorrelate = 10;
    double Tstart = 2;
    double Tend = 0.01;
    double Tdecrement = 0.05;

    MatrixXd EnsembleAveragedStatisticalData(1000, 5);

    MC sim1(N);
    MC sim2(N);
    MC sim3(N);
    MC sim4(N);
    MC sim5(N);
    MC sim6(N);
    MC sim7(N);
//        MC sim8(N);

    sim1.setRNGseed(Random(std::numeric_limits<int32_t>::max()));
    sim2.setRNGseed(Random(std::numeric_limits<int32_t>::max()));
    sim3.setRNGseed(Random(std::numeric_limits<int32_t>::max()));
    sim4.setRNGseed(Random(std::numeric_limits<int32_t>::max()));
    sim5.setRNGseed(Random(std::numeric_limits<int32_t>::max()));
    sim6.setRNGseed(Random(std::numeric_limits<int32_t>::max()));
    sim7.setRNGseed(Random(std::numeric_limits<int32_t>::max()));
//    sim8.setRNGseed(Random(std::numeric_limits<int32_t>::max()));

//    sim1.setRNGseed(1);
//    sim2.setRNGseed(1);
//    sim3.setRNGseed(1);
//    sim4.setRNGseed(1);
//    sim5.setRNGseed(1);
//    sim6.setRNGseed(1);
//    sim7.setRNGseed(1);

//    sim8.setRNGseed(Random(std::numeric_limits<int32_t>::max()));



    sim1.setSimulationParams(NStabilize, NData, NDecorrelate, Tstart, Tend, Tdecrement);
    sim2.setSimulationParams(NStabilize, NData, NDecorrelate, Tstart, Tend, Tdecrement);
    sim3.setSimulationParams(NStabilize, NData, NDecorrelate, Tstart, Tend, Tdecrement);
    sim4.setSimulationParams(NStabilize, NData, NDecorrelate, Tstart, Tend, Tdecrement);
    sim5.setSimulationParams(NStabilize, NData, NDecorrelate, Tstart, Tend, Tdecrement);
    sim6.setSimulationParams(NStabilize, NData, NDecorrelate, Tstart, Tend, Tdecrement);
    sim7.setSimulationParams(NStabilize, NData, NDecorrelate, Tstart, Tend, Tdecrement);
//        sim8.setSimulationParams(NStabilize,NData,NDecorrelate,Tstart,Tend,Tdecrement);
    for (int i = 0; i < NensembleDiv; i++) {
        std::thread t1(&MC::simulate, &sim1);
        std::thread t2(&MC::simulate, &sim2);
        std::thread t3(&MC::simulate, &sim3);
        std::thread t4(&MC::simulate, &sim4);
        std::thread t5(&MC::simulate, &sim5);
        std::thread t6(&MC::simulate, &sim6);
        std::thread t7(&MC::simulate, &sim7);
//        std::thread t8(&MC::simulate, &sim8);

        t1.join();
        t2.join();
        t3.join();
        t4.join();
        t5.join();
        t6.join();
        t7.join();
//        t8.join();

        //    sim1.simulate();

        EnsembleAveragedStatisticalData += (sim1.StatisticalData).eval();
        EnsembleAveragedStatisticalData += (sim2.StatisticalData).eval();
        EnsembleAveragedStatisticalData += (sim3.StatisticalData).eval();
        EnsembleAveragedStatisticalData += (sim4.StatisticalData).eval();
        EnsembleAveragedStatisticalData += (sim5.StatisticalData).eval();
        EnsembleAveragedStatisticalData += (sim6.StatisticalData).eval();
        EnsembleAveragedStatisticalData += (sim7.StatisticalData).eval();
//        EnsembleAveragedStatisticalData += (sim8.StatisticalData).eval();

        LastTIndex = sim1.LastTempIndex;
    }
    EnsembleAveragedStatisticalData /= (Nensemble);
    int64_t elapsed = static_cast<int64_t>(toc());

    cout << EnsembleAveragedStatisticalData(seq(0, LastTIndex - 1), all) << endl;

    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y.%m.%d %H-%M-%S");
    auto str = oss.str();
    auto relpath = "../../results/cpp/";
    string folderpath = relpath + str + " multi";
    string filepath = folderpath + "/StatisticalData.txt";
    fs::create_directories(folderpath.c_str());
    ofstream fout(filepath.c_str(), ios::out | ios::trunc);
//    fout.open("StatisticalData.txt", ios::trunc);
    if (fout) {
        fout << EnsembleAveragedStatisticalData(seq(0, LastTIndex - 1), all);
    }
    fout.close();

    // save info
    int64_t days = elapsed / (3600 * 24);
    int64_t hour = (elapsed % (3600 * 24)) / 3600;
    int64_t minute = (elapsed % 3600) / 60;
    int64_t sec = elapsed % 60;

    string filepathinfo = folderpath + "/info.txt";
    ofstream InfoOut(filepathinfo.c_str(), ios::out | ios::trunc);
    if (InfoOut) {
        InfoOut << "Num Paricles = " << N << endl;
        InfoOut << "ensemble = " << Nensemble << endl;
        InfoOut << "threads = " << Nthreads << endl;
        InfoOut << "Stabilize = " << NStabilize << endl;
        InfoOut << "Data = " << NData << endl;
        InfoOut << "Decorrelate = " << NDecorrelate << endl;
        InfoOut << "runtime = " << days << " days and " << std::setfill('0') << std::setw(2) << hour << ":"
                << std::setfill('0') << std::setw(2) << minute << ":"
                << std::setfill('0') << std::setw(2) << sec << endl;
        InfoOut << "runtime in seconds = " << elapsed << endl;
        InfoOut << "Tstart = " << Tstart << endl
                << "Tend = " << Tend << endl
                << "Tdecrement = " << Tdecrement << endl;
    }
    InfoOut.close();
}

void serial() {
//    cout <<  hash<thread::id>()(this_thread::get_id()) <<endl;
//    std::mt19937 gen(clock() + hash<thread::id>()(this_thread::get_id()));
//    this_thread::sleep_for(chrono::milliseconds(2000));

    tic();
    init();
    MatrixXd EnsembleAveragedStatisticalData(1000, 5);


    int Nthreads = 1;
    int NStabilize = 50000;
    int NData = 100000;
    int NDecorrelate = 10;
    int Nensemble = 35;
    double Tstart = 2;
    double Tend = 0.01;
    double Tdecrement = 0.05;
    for (int i = 0; i < Nensemble; i++) {
        LastTIndex = 0;
        StatisticalData.setZero(1000, 5);
        int a = 0;
        for (T = Tstart; T > Tend; T -= Tdecrement) {
            Debug_i = 0;
            stabilize(NStabilize);
            getStatisticalData(NData, NDecorrelate);
            if (a % 10 == 0) {
                cout << "T = " << T << endl;
                cout << "debugi = " << Debug_i << "\n\n";
            }
            a++;
        }
        cout << "run = " << i << endl;
        EnsembleAveragedStatisticalData += StatisticalData;
    }
    EnsembleAveragedStatisticalData /= Nensemble;
    int64_t elapsed = static_cast<int64_t>(toc());
    cout << EnsembleAveragedStatisticalData(seq(0, LastTIndex - 1), all) << endl;

//    cout << endl;
//    cout << U << endl;
//    double E = (U.transpose() * J * U).value();
//    double E2 = calcEnergy();
//    E/=N;
//    cout << "E = " << E << endl;
//    cout << "E2 = " << E2/N << endl;


    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y.%m.%d %H-%M-%S");
    auto str = oss.str();
    auto relpath = "../../results/cpp/";
    string folderpath = relpath + str + " single";
    string filepath = folderpath + "/StatisticalData.txt";
    fs::create_directories(folderpath.c_str());
    ofstream fout(filepath.c_str(), ios::out | ios::trunc);
//    fout.open("StatisticalData.txt", ios::trunc);
    if (fout) {
        fout << EnsembleAveragedStatisticalData(seq(0, LastTIndex - 1), all);
    }
    fout.close();


    int64_t days = elapsed / (3600 * 24);
    int64_t hour = (elapsed % (3600 * 24)) / 3600;
    int64_t minute = (elapsed % 3600) / 60;
    int64_t sec = elapsed % 60;

    string filepathinfo = folderpath + "/info.txt";
    ofstream InfoOut(filepathinfo.c_str(), ios::out | ios::trunc);
    if (InfoOut) {
        InfoOut << "Num Paricles = " << N << endl;
        InfoOut << "ensemble = " << Nensemble << endl;
        InfoOut << "threads = " << Nthreads << endl;
        InfoOut << "Stabilize = " << NStabilize << endl;
        InfoOut << "Data = " << NData << endl;
        InfoOut << "Decorrelate = " << NDecorrelate << endl;
        InfoOut << "runtime = " << days << " days and " << std::setfill('0') << std::setw(2) << hour << ":"
                << std::setfill('0') << std::setw(2) << minute << ":"
                << std::setfill('0') << std::setw(2) << sec << endl;
        InfoOut << "runtime in seconds = " << elapsed << endl;
        InfoOut << "Tstart = " << Tstart << endl
                << "Tend = " << Tend << endl
                << "Tdecrement = " << Tdecrement << endl;
    }
    InfoOut.close();
}

void init() {
//    auto rand_fn = [&](){return realRand();};
    for (int i = 0; i < N; i++) {
        VectorXd tempvec(3);
        tempvec.setRandom().normalize();
        U(seqN(3 * i, 3)) = tempvec;
    }

    ifstream file("J.txt");

    for (int i = 0; i < 3 * N; i++) {
        for (int j = 0; j < 3 * N; j++) {
            file >> J(i, j);
        }
    }

    J *= -1;
    makeIndices();
    calcAllFields();

    for (int i = 0; i < N; i++) {
        Ueig(3 * i, 0) = 1;
        Ueig(3 * i + 1, 1) = 1;
    }
    Ueig(all, 0).normalize();
    Ueig(all, 1).normalize();
}

double dErot_old(int i) {
    //VectorXd dE1 = -(mi_new - U(seqN(3*i,3))).transpose() * J(seqN(3*i,3)),seqN(3*i,3))) *(mi_new + U(seqN(3*i,3)));
    double dE = (mi_new - U(seqN(3 * i, 3))).transpose() * J(seqN(3 * i, 3), seqN(3 * i, 3))
                * (mi_new + U(seqN(3 * i, 3)));
    VectorXd dM = mi_new - U(seqN(3 * i, 3));

//    makeIndices(i);
//    dE -= (2*dM.transpose() * J(seqN(3*i,3), indices)) * U(indices);

    for (int j = 0; j < N; j++) {
        if (j != i) {
            dE += 2 * dM.transpose() * J(seqN(3 * i, 3), seqN(3 * j, 3)) * U(seqN(3 * j, 3));
        }
    }
    return dE;
}

double dErot(int i) {
    double dE = 0;
    VectorXd dM = mi_new - U(seqN(3 * i, 3));
    dE += (dM.transpose() * J(seqN(3 * i, 3), seqN(3 * i, 3))
           * (mi_new + U(seqN(3 * i, 3)))).value();
//    double E1 = 0;
//    double E2 = 0;
//    cout << "dE = " << dE << endl;
//    cout << "E1 = " << (E1 = (mi_new.transpose() * J(seqN(3*i,3),seqN(3*i,3))
//                        *(mi_new)).value()) << endl;
//    cout << "E2 = " << (E2 = (U(seqN(3*i,3)).transpose() * J(seqN(3*i,3),seqN(3*i,3))
//                        *(U(seqN(3*i,3)))).value()) << endl;
//    cout << "E2 - E1 = " << E2 - E1 << endl ;
//
//
//    this_thread::sleep_for(chrono::milliseconds(500));



//    dE -= (dM.transpose() * H(seqN(3*i,3))).value();
    dE += 2 * dM.dot(H(seqN(3 * i, 3)));
//    cout << "dEtot = " << dE << endl << endl;
    return dE;
}

void run1step_old() {
    for (int c = 0; c < N; c++) {
        int i = Random(N);
//        int i = intDis(gen);
        //       cout << "i = " << i << endl;
//        mi_new = Eigen::Vector3d::NullaryExpr(rand_fn);  //thread safe
//        mi_new.normalize();
//        cout << mi_new << "\n\n";
        mi_new.setRandom().normalize();
        double dE = dErot_old(i);
        if (dE < 0) {
            U(seqN(3 * i, 3)) = mi_new;
        } else if (Random() < exp(-dE / T)) {
            U(seqN(3 * i, 3)) = mi_new;
            Debug_i++;
        }
    }

}

void run1step() {

    static int irecalc = 0;
    if (irecalc++ % 100000 == 0)
        calcAllFields();
//    double a1;
//    double a2;
//    double a3;
    auto rand_fn = [&]() { return realDissym(gen); };
    for (int c = 0; c < N; c++) {
        int i = intDis(gen);
//        int i = Random(N);
        //       cout << "i = " << i << endl;
//        mi_new.setRandom().normalize();
        mi_new = Eigen::VectorXd::NullaryExpr(3, rand_fn);
//        mi_new.setRandom();
        mi_new.normalize();
//        cout << mi_new << "\n\n";
        double dE = dErot(i);
//        cout << "older = " << dErot(i) << endl;
//        cout << "older = " << dErot_new(i) << endl;
        if (dE < 0) {
            updateField(i);
            U(seqN(3 * i, 3)) = mi_new;
//            Debug_i++;
        } else if (realDis(gen) < exp(-dE / T)) {
            updateField(i);
            U(seqN(3 * i, 3)) = mi_new;
            Debug_i++;

        }

    }
//    double E = (-U.transpose() * J) * U;
//    double E2 = calcEnergy();
}

void stabilize_old(int num) {
    for (int i = 0; i < num; i++)
        run1step_old();
}

void stabilize(int num) {
//    calcAllFields();
    for (int i = 0; i < num; i++)
        run1step();
}


void calcAllFields() {
    H.setZero();
    for (int i = 0; i < N; i++) {
        H(seqN(3 * i, 3)).noalias() += J(seqN(3 * i, 3), indices(all, i)) * U(indices(all, i));
    }

//    ArrayXi indices(3*N-3);
//    for(int i = 0; i < N; i++){
//        H = J(seqN(3*i,3), all) * U;
////        int a = 0;
//        for (int j = 0; j < N; j++){
//            if(j != i){
////                indices(a++) = 3*j;
////                indices(a++) = 3*j+1;
////                indices(a++) = 3*j+2;
//                H(seqN(3*i,3)) += J(seqN(3*i,3), seqN(3*j,3)) * U(seqN(3*j,3));
//            }
//        }
//        cout << indices <<endl;
//        cout << indices <<endl;
//         = J(indices, seqN(3*i,3)) * U(indices);
//        cout << U(indices) <<endl;
//        H(seqN(3*i,3)) = J(seqN(3*i,3), indices) * U(indices);

//    }
}

void updateField(int i) {
    VectorXd dM = mi_new - U(seqN(3 * i, 3));

//    H(indices(all,i)).noalias() += J(indices(all,i), seqN(3*i,3)) * dM;
//

    H.noalias() += J(all, seqN(3 * i, 3)) * dM;
    H(seqN(3 * i, 3)).noalias() -= J(seqN(3 * i, 3), seqN(3 * i, 3)) * dM;
//
//    for(int j=0; j<3*N; j++){
//            for(int k=0; k<3; k++){
//                H(j) += J(j,3*i+k)*dM(k);
//            }
//    }

//    VectorXd dM = mi_new - U(seqN(3*i,3));
//    for(int j = 0; j < N; j++){
//        if( j!=i){
//            H(seqN(3*j,3)) += J(seqN(3*j,3), seqN(3*i,3)) * dM;
//        }
//    }
}

long tic(int mode) {
    static std::chrono::_V2::system_clock::time_point t_start;

    if (mode == 0) {
        t_start = std::chrono::high_resolution_clock::now();
        return 0;
    } else {
        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed = (t_end - t_start).count() * 1E-9;
        std::cout << "Elapsed time is " << elapsed << "\n";
        return elapsed;
    }
}

long toc() { return tic(1); }

double calcEnergy() { //slow
    double E;
    E = U.transpose() * H;
    for (int i = 0; i < N; i++) {
        E += (U(seqN(3 * i, 3)).transpose() * J(seqN(3 * i, 3), seqN(3 * i, 3)) * U(seqN(3 * i, 3))).value();
    }
    return E;
}


void makeIndices() {
    int a = 0;
    for (int i = 0; i < N; i++) {
        a = 0;
        for (int j = 0; j < N; j++) {
            if (j != i) {
                indices(a++, i) = 3 * j;
                indices(a++, i) = 3 * j + 1;
                indices(a++, i) = 3 * j + 2;
            }
        }
    }
//    cout << indices(all,1) << endl;
}

void getStatisticalData(int N_MCstep, int NStepToDecorrelate) {
    double SumE = 0;
    double SumE2 = 0;
    double SumOrderparam = 0;
    double SumOrderparam2 = 0;

    for (int i = 0; i < N_MCstep; i++) {
        run1step();
        if (i % NStepToDecorrelate == 0) {
            double EConfig = (U.transpose() * J * U).value(); //faster
//            double EConfig = calcEnergy();
//            EConfig /=2; //TODO get rid of this
//            double EConfig = - U.transpose() * H;
            double OrderparamConfig = square(U.dot(Ueig(all, 0))) + square(U.dot(Ueig(all, 1)));

            SumE += EConfig;
            SumE2 += square(EConfig);
            SumOrderparam += OrderparamConfig;
            SumOrderparam2 += square(OrderparamConfig);
        }
    }

    int NDataTaking = N_MCstep / NStepToDecorrelate;
//    cout << "NdataTaking = " << NDataTaking<< endl;
    double MeanOrderparam = SumOrderparam / NDataTaking;
    double MeanOrderparam2 = SumOrderparam2 / NDataTaking;
    double MeanE = SumE / NDataTaking;
    double MeanE2 = SumE2 / NDataTaking;

    double XPerParicle = (MeanOrderparam2 - square(MeanOrderparam)) / (N * T);
    double CPerParicle = (MeanE2 - square(MeanE)) / (N * T * T);
    double MeanEPerParticle = MeanE / N;
    double MeanOrderparamPerParticle = MeanOrderparam / N;

    StatisticalData.row(LastTIndex++) << T, MeanEPerParticle, MeanOrderparamPerParticle,
            CPerParicle, XPerParicle;
}


/* Thread-safe function that returns a random number between min and max (inclusive).
This function takes ~142% the time that calling rand() would take. For this extra
cost you get a better uniform distribution and thread-safety. */
//int intRand(const int & min, const int & max) {
//    static thread_local mt19937* generator = nullptr;
//    if (!generator) generator = new mt19937(clock() + this_thread::get_id().hash());
//    uniform_int_distribution<int> distribution(min, max);
//    return distribution(*generator);
//}
