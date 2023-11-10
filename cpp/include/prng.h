/******************************************************************************/
/***                                                                        ***/
/*** Thread Safe Random generator                                           ***/    
/*** Ver 1.44                                                               ***/
/*** Date: 13990315                                                         ***/
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

#ifndef RANDOM_H

#define RANDOM_H

#include "utils.h"

//------------------------------------------------------------------------------

// Seed of the random number generator
extern long iseed;
#ifdef _OPENMP
  #pragma omp threadprivate(iseed)  // this line makes iseed threadsafe variable; each thread has its own copy of variable
#endif

/* This is a 32 bit random number generator with uniform distribution in range [0..1),
 * which is implemented in the Numerical recipes in C, chapter 7 (ran2)
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
double ran2(long *idum);

/* Returns a normally distributed deviate with zero mean and unit variance, using ran2(idum)
 * as the source of uniform deviates. gasdev() implemented in the Numerical recipes in C, chapter 7
 * 
 * idum is the seed of ran2() random number generator.
 * 
 * Visit www.nr.com for the licence.*/
double gasdev(long *idum);

// Note: idum in routines argument is the seed of the random number generator. If it is not availible, default seed (iseed) is used
// The seed of the random generator can be initialized by using the following routine
void Randomize();
void Randomize(int seed);                   // Renew the iseed in multi-thread mode to seed.

// Random number generator with Gausssian distribution (Normal)
inline double GRand() { return gasdev(&iseed); }

// Random generator with uniform distribution in range [0..1)
[[deprecated]] inline double Random(long *idum) { return ran2(idum); }
inline double Random() { return ran2(&iseed); }

// Random generator with uniform distribution in integer range [0..N)
[[deprecated]] inline int Random(int N, long *idum) { return int(ran2(idum)*N); }
inline int Random(int N) { return int(ran2(&iseed)*N); }

// Random generator with uniform distribution in range [min..max]
// Note: for selecting floating-point version of function you can use "<double>" after the name of function. e.g.,
// double r = Random<double>(1., 10.);
[[deprecated]] inline double Random(float min, float max, long *idum) { return min + ran2(idum)*(max-min); }
inline double Random(float min, float max) { return min + ran2(&iseed)*(max-min); }
[[deprecated]] inline int Random(int min, int max, long *idum) { return int(min + ran2(idum)*(max-min+1)); }
inline int Random(int min, int max) { return int(min + ran2(&iseed)*(max-min+1)); }

/* Note: use of idum in the interface function of the random generator is deprecated.
 * Only use Randomise() routine or iseed.*/

//------------------------------------------------------------------------------

#endif
