//
// Created by Sadeq Ismailzadeh on ۱۶/۰۲/۲۰۲۳.
//

#include <cassert>
#include "WalkerAlias.h"


WalkerAlias::WalkerAlias (  ) {
    set (  );
}

WalkerAlias::WalkerAlias (const std::vector<double> & w ) {
    set ( w );
}

void WalkerAlias::set (const std::vector<double> & w ) {
    int n = w.size();
    prob = std::vector<double> ( n, 0.0 );
    inx = std::vector<int> ( n, -1 );
    realDis = std::uniform_real_distribution<double> (0.0, 1.0);
    intDis  = std::uniform_int_distribution<int> (0, n - 1);
    double sumw = std::accumulate ( w.begin(), w.end(), 0.0 );
    bool onlyzero = true;
    for ( int i = 0; i < n; i++ ) {
        if ( w[i] < 0 ) throw std::domain_error( "Error walker_sampler::set: Negative entries in argument w." );
        else if ( ( onlyzero ) && ( w[i] > 0 ) ) onlyzero = false;
        prob[i] = (w[i]*n)/sumw;
    }
    if ( onlyzero ) throw std::domain_error( "Error walker_sampler::set: Only zeros as entries in argument w." );
    std::vector<int> vshort;
    std::vector<int> vlong;
    for ( int i = 0; i < n; i++ ) {
        if ( prob[i] < 1 ) vshort.push_back ( i );
        if ( prob[i] > 1 ) vlong.push_back ( i );
    }
    while ( !vshort.empty() && !vlong.empty() ) {
        int j = vshort.back();
        vshort.pop_back();
        int k = vlong.back();
        inx[j] = k;
        prob[k] -= ( 1 - prob[j] );
        if ( prob[k] < 1 ) {
            vshort.push_back( k );
            vlong.pop_back();
        }
    }
}

void WalkerAlias::set (  ) {
    set ( {1} );
}

int WalkerAlias::draw (const double u, const int i ) {
    assert( !(( u > 1 ) || ( u < 0 )) );
    assert ( !(( i < 0) || ( (unsigned)i >= inx.size() )) );
    if ( u <= prob[i] ) return i;
    return inx[i];
}

int WalkerAlias::draw (double u ) {
    assert( ( u > 1 ) || ( u < 0 ) );
    int64_t i = ( ( reinterpret_cast<int64_t&>(u) & (((int64_t)1 << 53) - 1) ) % inx.size() );
    return draw ( u, i );
}