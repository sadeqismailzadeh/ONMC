//------------------------------------------------------------------------//
// Name: Jan Koziol                                                       //
// Email: jankoziol@gmx.de                                                //
// GitHub: https://github.com/Jankoziol/discrete-sample                   //
//------------------------------------------------------------------------//

// Modified by Sadeq Ismailzadeh on ۰۸/۰۱/۲۰۲۳.

#ifndef SRC_WALKERALIAS_H
#define SRC_WALKERALIAS_H
#include <vector>
#include <numeric>
#include <cstdint>
#include <stdexcept>
#include <random>



class WalkerAlias {

private :
    std::vector<double> prob;
    std::vector<int> inx;
    std::uniform_real_distribution<double> realDis;
    std::uniform_int_distribution<int>     intDis;

    int draw ( const double u, const int i );
    int draw ( double u ); // This is dirty, here no const possible

public :
    WalkerAlias (  );
    explicit WalkerAlias (const std::vector<double> & w );
    void set ( const std::vector<double> & w );
    void set (  );

    template <typename RNE>
    inline auto operator()(RNE& gen)
    // Return a random number according to the given probabilities
    // at initialization using a Random Number Engine.
    {
        auto u = realDis(gen);
        auto i = intDis(gen);
        return draw(u, i);
//        return draw(u);
    };

};


#endif //SRC_WALKERALIAS_H
