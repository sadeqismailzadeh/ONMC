//
// Created by Sadeq Ismailzadeh on ۰۷/۰۱/۲۰۲۲.
//
#include "utils.h"
const  IOFormat CSVFormat(FullPrecision, DontAlignCols, ", ", "\n");

//long tic(int mode) {
//    static thread_local std::chrono::time_point<std::chrono::high_resolution_clock> t_start;
//
//    if (mode == 0) {
//        t_start = std::chrono::high_resolution_clock::now();
//        return 0;
//    } else {
//        auto t_end = std::chrono::high_resolution_clock::now();
//        long elapsed =  chrono::duration_cast<chrono::seconds>(t_end - t_start).count();
//        std::cout << "Elapsed time is " << elapsed << "\n";
//        return elapsed;
//    }
//}
//
//long toc() { return tic(1); }
//
tuple<int64_t,int64_t,int64_t,int64_t> toDHMS (int64_t elapsed) {
    int64_t days = elapsed / (3600 * 24);
    int64_t hours = (elapsed % (3600 * 24)) / 3600;
    int64_t minutes = (elapsed % 3600) / 60;
    int64_t secs = elapsed % 60;
    return make_tuple(days, hours, minutes, secs);
}

bool almostEquals(double x, double y) {
    double EPSILON = 1e-9;
//    return (fabs(x - y) <= EPSILON );
    return (fabs(x - y) <= EPSILON * max({1.0, fabs(x), fabs(y)}));  // best version
//    return (fabs(x - y) <= EPSILON * min({fabs(x), fabs(y)}));
//    return (fabs(x - y) / fabs(x + y)) <= EPSILON;  // BEWARE division by zero
}

void swapCol(MatrixXd &M, int i, int j) {
    VectorXd temp(M.rows());
    temp = M.col(i);
    M.col(i) = M.col(j);
    M.col(j) = temp;
}

bool almostEquals(const VectorXd &v1, const VectorXd& v2){
    if( v1.size() != v2.size()){
        return false;
    }
    for (int i = 0; i < v1.size(); ++i) {
        if (!almostEquals(v1(i),v2(i))){
            return false;
        }
    }
    return true;
}

bool contains(vector<double> &vec, const double num) {
    for (auto &&i : vec) {
        if (almostEquals(i,num)){
            return true;
        }
    }
    return false;
}


int64_t ipow(int64_t base, uint8_t exp) {
    static const uint8_t highest_bit_set[] = {
            0, 1, 2, 2, 3, 3, 3, 3,
            4, 4, 4, 4, 4, 4, 4, 4,
            5, 5, 5, 5, 5, 5, 5, 5,
            5, 5, 5, 5, 5, 5, 5, 5,
            6, 6, 6, 6, 6, 6, 6, 6,
            6, 6, 6, 6, 6, 6, 6, 6,
            6, 6, 6, 6, 6, 6, 6, 6,
            6, 6, 6, 6, 6, 6, 6, 255, // anything past 63 is a guaranteed overflow with base > 1
            255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255,
            255, 255, 255, 255, 255, 255, 255, 255,
    };

    int64_t result = 1;

    switch (highest_bit_set[exp]) {
        case 255: // we use 255 as an overflow marker and return 0 on overflow/underflow
            if (base == 1) {
                return 1;
            }

            if (base == -1) {
                return 1 - 2 * (exp & 1);
            }

            return 0;
        case 6:
            if (exp & 1) result *= base;
            exp >>= 1;
            base *= base;
        case 5:
            if (exp & 1) result *= base;
            exp >>= 1;
            base *= base;
        case 4:
            if (exp & 1) result *= base;
            exp >>= 1;
            base *= base;
        case 3:
            if (exp & 1) result *= base;
            exp >>= 1;
            base *= base;
        case 2:
            if (exp & 1) result *= base;
            exp >>= 1;
            base *= base;
        case 1:
            if (exp & 1) result *= base;
        default:
            return result;
    }
}


void print(const vector<vector<double>> &vec) {
    for (int i = 0; i < vec.size(); i++)
    {
        for (int j = 0; j < vec[i].size(); j++)
        {
            cout << vec[i][j] << "   ";
        }
        cout << endl;
    }
}