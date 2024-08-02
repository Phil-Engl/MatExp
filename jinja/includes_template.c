#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>
#include <assert.h>
{% if dgemm %}
#include <cblas.h>
{%- endif %}
{% if flop_count %}
#include "matrix_exponential.h"
{%- endif %}

#define DEBUG {{ dbg }}


const double pade_coefs[14] =
{
    1.0,
    0.5,
    0.12,
    1.833333333333333333333e-2,
    1.992753623188405797101e-3,
    1.630434782608695652174e-4,
    1.035196687370600414079e-5,
    5.175983436853002070393e-7,
    2.043151356652500817261e-8,
    6.306022705717595115002e-10,
    1.483770048404140027059e-11,
    2.529153491597965955215e-13,
    2.810170546219962172461e-15,
    1.544049750670308885967e-17
};

const double theta[15] =
{
    0.0, 
    0.0, 
    0.0,
    1.495585217958292e-2, // theta_3
    0.0,
    2.539398330063230e-1, // theta_5
    0.0,
    9.504178996162932e-1, // theta_7
    0,
    2.097847961257068e0,  // theta_9
    0.0,
    0.0,
    0.0,
    4.25,                 // theta_13 for alg 5.1
    5.371920351148152e0  // theta_13 for alg 3.1
};

const double theta3_exp_10 = 10.150682505756677;
const double theta5_exp_10 = 12.890942410094057;
const double theta7_exp_10 = 25.867904522060577;
const double theta9_exp_10 = 81.486148948487397;
const double theta13_exp_10 = 4416.640977841335371;
const double theta13_inv = 0.235294117647059;

const double coeffs[14] =
{
    0.0,
    0.0,
    0.0,
    1.0/100800.0, //m = 3
    0.0,
    1.0/10059033600.0, //m = 5
    0.0,
    1.0/4487938430976000.0, //m = 7
    0.0,
    1.0/5914384781877411840000.0, //m = 9
    0.0,
    0.0,
    0.0,
    1.0/113250775606021113483283660800000000.0 //m = 13
};

const int64_t ur_int = 0x3ca0000000000000; //hex representation of 2^-53
const double *unit_roundoff = (double*)&ur_int; //2^-53


{% if dgemm %}
const double alpha = 1.0;
const double beta = 0.0;
const CBLAS_LAYOUT layout = CblasColMajor;
{%- endif %}


