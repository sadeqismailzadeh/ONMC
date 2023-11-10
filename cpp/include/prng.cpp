/******************************************************************************/
/***                                                                        ***/
/*** Thread Safe Random generator                                           ***/    
/*** Ver 1.45                                                               ***/
/*** Date: 13990525                                                         ***/
/*** Code is implemented in GNU C++ compiler (v. 4.6.3) in 64-bit UBUNTU 12 ***/
/*** Running under an Intel® Core™ i3 CPU × 4 machine with 2 GB of RAM      ***/
/***                                                                        ***/
/******************************************************************************/

/* These function and routines are based on the random number generator 

   double ran2(long *idum);

which are implemented in the Numerical Recipes in C, chapter 7 (ran2)

Long period (> 2 × 10^{18}) random number generator of L. Ecuyer with Bays-Durham shuffle
and added safeguards. Returns a uniform random deviate between 0.0 and 1.0 (exclusive of
the endpoint values). 

***!!! Call with idum a negative integer to initialize; !!!*** thereafter, do not alter
idum between successive deviates in a sequence. RNMX should approximate the largest floating
value that is less than 1.

Visit www.nr.com for the licence.*/

/* OpenMP Note:
 * code should be compiled with -fopenmp switch in openmp implementation.
*/

#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cmath>
#ifdef _OPENMP // Use omp.h if -fopenmp is used in g++
  #include <omp.h>
#endif
#include "prng.h"

using namespace std;

//------------------------------------------------------------------------------

// Seed of the random number generator
long iseed = -36;
#ifdef _OPENMP // This line makes iseed threadsafe variable; each thread has its own copy of variable.
  #pragma omp threadprivate(iseed)
#endif

// Note: idum in routines argument is the seed of the random number generator. If it is not availible, default seed (iseed) is used
// The seed of the random generator can be initialized by using the following routine
void Randomize() { // Threadsafe version of Randomize
  #ifdef _OPENMP 
    #pragma omp parallel
    { iseed = -abs(time(NULL) + omp_get_thread_num()); }
  #else
    iseed = -abs(time(NULL));  
  #endif
}
void Randomize(int seed) {                  // Renew the iseed in multi-thread mode to seed.
  #ifdef _OPENMP 
    #pragma omp parallel
    { iseed = -abs(seed) + omp_get_thread_num(); }
  #else
    iseed = -abs(seed);  
  #endif
}

/* Returns a normally distributed deviate with zero mean and unit variance, using ran2(idum)
 * as the source of uniform deviates. gasdev() implemented in the Numerical recipes in C, chapter 7
 * 
 * idum is the seed of ran2() random number generator.
 * 
 * Visit www.nr.com for the licence.*/
double gasdev(long *idum) {
  static int iset = 0;
  static double gset;
  #ifdef _OPENMP // This line makes iset, gset threadsafe variables; each thread has its own copy of these variables
    #pragma omp threadprivate(iset, gset)
  #endif

  if (*idum < 0) iset = 0;                  // Reinitialize
  
  if  (iset == 0) {                         // We don’t have an extra deviate handy, so
    double fac, rsq, v1, v2;

    do {
      v1 = 2.0*ran2(idum) - 1.0;            // pick two uniform numbers in the square extending 
      v2 = 2.0*ran2(idum) - 1.0;            // from -1 to +1 in each direction,
      rsq = v1*v1 + v2*v2;                  // see if they are in the unit circle,
    } while (rsq >= 1.0 || rsq == 0.0);     // and if they are not, try again.

    fac = sqrt(-2.0*log(rsq)/rsq);          // Now make the Box-Muller transformation to get two normal deviates. Return
    gset = v1*fac;                          // one and save the other for next time in gset.
    iset = 1;
    return v2*fac;
  } else {                                  // We have an extra deviate handy, so
    iset = 0;                               // unset the flag,
    return gset;                            // and return it.
  }
}

//------------------------------------------------------------------------------

/* This is a 32 bit random number generator with uniform distribution in range [0..1),
 * implemented in the  Numerical recipes in C, chapter 7 (ran2)
 * 
 * Long period (> 2 × 10^{18}) random number generator of L. Ecuyer with Bays-Durham shuffle
 * and added safeguards. Returns a uniform random deviate between 0.0 and 1.0 (exclusive of
 * the endpoint values). 
 * 
 * Note: ***!!! Call with idum a negative integer to initialize; !!!*** thereafter, do not alter
 * idum between successive deviates in a sequence. RNMX should approximate the largest floating
 * value that is less than 1. 
 * 
 * Visit www.nr.com for the licence.*/
/* Note: following definitions #undef's after the ran2() routine */
#define IM1 2147483563
#define IM2 2147483399
#define AM (1.0/IM1)
#define IMM1 (IM1-1)
#define IA1 40014
#define IA2 40692
#define IQ1 53668
#define IQ2 52774
#define IR1 12211
#define IR2 3791
#define NTAB 32
#define NDIV (1+IMM1/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)

double ran2(long *idum) {
    int j;
    long k;
    static long idum2=123456789;
    static long iy=0;
    static long iv[NTAB];
    double temp;
    #ifdef _OPENMP // this line makes idum2, iy, iv threadsafe variables; each thread has its own copy of these variables
        #pragma omp threadprivate(idum2, iy, iv)
    #endif

    if (*idum <= 0) {                       // Initialize
        if (-(*idum) < 1) *idum=1;          // Be sure to prevent idum = 0
        else *idum = -(*idum);
        idum2=(*idum);
        for (j=NTAB+7;j>=0;j--) {           // Load the shuffle table (after 8 warm-ups)
            k=(*idum)/IQ1;
            *idum=IA1*(*idum-k*IQ1)-k*IR1;
            if (*idum < 0) *idum += IM1;
            if (j < NTAB) iv[j] = *idum;
        }
        iy=iv[0];
    }

    k=(*idum)/IQ1;                          // Start here when not initializing
    *idum=IA1*(*idum-k*IQ1)-k*IR1;
    if (*idum < 0) *idum += IM1;
    k=idum2/IQ2;
    idum2=IA2*(idum2-k*IQ2)-k*IR2;
    if (idum2 < 0) idum2 += IM2;
    j=iy/NDIV;
    iy=iv[j]-idum2;
    iv[j] = *idum;
    if (iy < 1) iy += IMM1;
    if ((temp=AM*iy) > RNMX) return RNMX;   // Because users don’t expect endpoint values
    else return temp;
}
#undef IM1
#undef IM2
#undef AM
#undef IMM1
#undef IA1
#undef IA2
#undef IQ1
#undef IQ2
#undef IR1
#undef IR2
#undef NTAB
#undef NDIV
#undef EPS
#undef RNMX

//------------------------------------------------------------------------------
