//
// Created by Sadeq Ismailzadeh on ۰۵/۰۱/۲۰۲۲.
//

#ifndef SRC_UTILS_H
#define SRC_UTILS_H
#include <cmath>
#include <chrono>
#include <iostream>
#include <tuple>
#include <algorithm>
#include <cassert>
#include <vector>
#include <Eigen/Dense>
#include "omp.h"

using namespace  std;
using namespace  Eigen;
#define mAssert(exp, msg) assert(((void)msg, exp))

//#define STR(x) #x
#define rassert(condition) \
do { \
        if (! (condition)) { \
            std::cerr << "Assertion `" #condition "` failed in " << std::endl \
            << " function " <<   __PRETTY_FUNCTION__ << std::endl  \
            << " file " << __FILE__ << std::endl \
            << " line " << __LINE__ << std::endl; \
            std::terminate(); \
        } \
    } while (false)

extern const  IOFormat CSVFormat;
//long tic(int mode = 0);
//long toc();
tuple<int64_t,int64_t,int64_t,int64_t> toDHMS (int64_t elapsed);
bool almostEquals(double x, double y);
//bool almostEquals(const VectorXd& v1, const VectorXd& v2);

void swapCol(MatrixXd &M, int i, int j);
bool contains(vector<double> &vec, const double num);
template <typename T>
T square (T value){
    return value * value;
}

template<typename DerivedA, typename DerivedB>
bool almostEquals(const Eigen::DenseBase<DerivedA>& a,
                  const Eigen::DenseBase<DerivedB>& b,
                  const typename DerivedA::RealScalar& rtol
              = 1e-05,
//              = Eigen::NumTraits<typename DerivedA::RealScalar>::dummy_precision(),
              const typename DerivedA::RealScalar& atol
              = 1e-08)
//              = Eigen::NumTraits<typename DerivedA::RealScalar>::epsilon())
{
    if (a.rows() != b.rows() || a.cols() != b.cols()){
        return false;
    }
    return ((a.derived() - b.derived()).array().abs()
            <= (atol + rtol *a.derived().array().abs().max(b.derived().array().abs()))).all();
//    <= (atol + rtol * b.derived().array().abs())).all();
}
int64_t ipow(int64_t base, uint8_t exp);
void print(const vector<vector<double>> &vec);

template<typename Type>
Type bounded(Type Value, Type UpperBound, Type LowerBound){
    if (Value > UpperBound){
        return UpperBound;
    } else if (Value < LowerBound) {
        return LowerBound;
    }
    return Value;
}

class SimpleStopWatch {
public:  //  TODO add pause and resume
    typedef chrono::high_resolution_clock Clock;
    void reset()  { t_start = Clock::now(); }
    int64_t elapsedSeconds() const{
        return chrono::duration_cast<chrono::seconds>(Clock::now() - t_start).count();
    }
    void printElapsedSeconds() const {
        cout << "Elapsed time is " << elapsedSeconds() << "\n";
    }
    double elapsedSecondsPrecise() const{
        return chrono::duration_cast<chrono::nanoseconds>(Clock::now() - t_start).count() * 1E-9;
    }
    void printElapsedSecondsPrecise() const {
        cout << "Elapsed time is " << elapsedSecondsPrecise() << "\n";
    }
    void printElapsedSecondsPrecise(const string& s) const {
        cout << s << " time is " << elapsedSecondsPrecise() << "\n";
    }
private:
    Clock::time_point t_start;
};

struct SCOData {
    SCOData(const int i,const  int xi ,const int yi): i(i), xi(xi), yi(yi){
        Hxi = 0;
        Hyi = 0;
        OneMinusPjOldProd = 1;
        OneMinusPjNewProd = 1;
    }
    double Hxi;
    double Hyi;
    const int i;
    const int xi;
    const int yi;
    double OneMinusPjOldProd;
    double OneMinusPjNewProd;
    vector<double> Hxij;
    vector<double> Hyij;
    vector<int>    jrejVec;
};



//std::string operator ""_s(const char * str, std::size_t len) {
//    return std::string(str, len);
//}


#endif //SRC_UTILS_H
