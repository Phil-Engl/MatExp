#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "matrix_operations.h"
#include "onenorm.h"
#include "matrix_exponential.h"
#include <cassert>

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

void print_constants(){
    printf("10*exp(theta[3]) = %.15lf\n", 10*exp(theta[3]));
    printf("10*exp(theta[5]) = %.15lf\n", 10*exp(theta[5]));
    printf("10*exp(theta[7]) = %.15lf\n", 10*exp(theta[7]));
    printf("10*exp(theta[9]) = %.15lf\n", 10*exp(theta[9]));
    printf("(10+s_max)*exp(theta[13]) = %.15lf\n", (10+53)*exp(theta[13]));
    printf("1.0/theta[13] = %.15lf\n", 1.0/theta[13]);
}

int rnd(int lower, int upper){
    return (rand() % (upper - lower + 1)) + lower; 
}

double* creatematrix(int m, int n){
    double* M = (double*)malloc(m * n * sizeof(double));
    return M;
}

void freematrix(double* M){
    free(M);
}

void get_test_matrix1(double* M, int m, int n, double diag, double non_diag){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            M[i * n + j] = (i==j)? diag : non_diag;
        }
    }
}

void fillmatrix(double* M, int m, int n, int lower, int upper){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            M[i*n + j] = (double)rnd(lower,upper); /// (double)rnd(1, upper);
        }
    }
}

int is_equal(double *A, double *B, int m, int n){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            if(fabs(A[i*n + j] - B[i*n+j]) > 1e-8){
                return 0;
            }
        }
    }
    return 1;
}

void test_transpose(int n, int t){
    printf("Test transpose with n=%d, t=%d: \n", n, t);
    double* B = creatematrix(n, t);
    double* BT = creatematrix(t, n);
    double* BTT = creatematrix(n, t);

    fillmatrix(B, n, t, 0, 10);

    transpose(B, BT, n, t);
    transpose(BT, BTT, t, n);

    // printf("\n\nB:\n");
    // printmatrix(B, n, t);
    // printf("B_Transpose:\n");
    // printmatrix(BT, t, n);

    assert(is_equal(B, BTT, n, t));
    printf("    Passed!\n");

    freematrix(B);
    freematrix(BT);
    freematrix(BTT);
}

void test_mmm(int m, int n, int t){
    printf("\nTest mmm with m=%d, n=%d, t=%d: \n", m, n, t);

    double* A  = creatematrix(m, n);
    double* B = creatematrix(n, t);
    double* C = creatematrix(m, t);

    fillmatrix(A, m, n, 0, 10);
    fillmatrix(B, n, t, 0, 10);

    mmm(A, B, C, m, n, t);

    printf("A:\n");
    printmatrix(A, m, n);
    printf("\nB:\n");
    printmatrix(B, n, t);
    printf("\nC = A*B:\n");
    printmatrix(C, m, t);
    printf("\n");

    freematrix(A);
    freematrix(B);
    freematrix(C);
}

void test_normest(){
    printf("\nTest one-norm and onenormest: \n");
    int m = 3;
    int n = 3;
    int t = 6;
    int ind_bestA;
    int ind_bestB;
    srand(time(0));
    double* A  = creatematrix(m, n);
    double* B = creatematrix(n, t);
    fillmatrix(A, m, n, 0, 10);
    fillmatrix(B, n, t, 0, 10);

    double Anorm = onenorm_best(A, m, n, &ind_bestA);
    double Bnorm = onenorm_best(B, n, t, &ind_bestB);
    double* v = (double*)malloc(n * sizeof(double));
    double* w = (double*)malloc(n * sizeof(double));
    double Anormest = onenormest(A,n,14,5,1,1,v,w);
    //v[0] = -1.0;
    //w[0] = -1.0;
    printf("A:\n");
    printmatrix(A, m, n);
    printf("One norm of A: %f with best index %d\n", Anorm, ind_bestA);
    printf("One norm est of A: %f\n ", Anormest);
    /*for(int i = 0; i < n; i++){
        printf("%f, ", v[i]);
    }
    printf("\nw: ");
    for(int i = 0; i < n; i++){
        printf("%f, ", w[i]);
    }*/

    printf("\nB:\n");
    printmatrix(B, n, t);
    printf("One norm of b: %f with best index %d\n\n", Bnorm, ind_bestB);

    freematrix(A);
    freematrix(B);
    freematrix(v);
    freematrix(w);
}

void test_eval3_4(int n){
    const double b[14] =
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
    printf("Test eval 3.4 for m=5 and n=%d: \n", n);

    double* I = (double*)malloc(n * n * sizeof(double));
    double* A = (double*)malloc(n * n * sizeof(double));
    double* A2 = (double*)malloc(n * n * sizeof(double));
    double* A3 = (double*)malloc(n * n * sizeof(double));
    double* A4 = (double*)malloc(n * n * sizeof(double));
    double* A5 = (double*)malloc(n * n * sizeof(double));
    double* A6 = (double*)malloc(n * n * sizeof(double));
    double* P = (double*)malloc(n * n * sizeof(double));
    double* Q = (double*)malloc(n * n * sizeof(double));
    double* res = (double*)malloc(n * n * sizeof(double));

    fill_diagonal_matrix(I, 1.0, n);
    fillmatrix(A, n, n, -10, 20);


    mmm(A, A, A2, n, n, n);
    mmm(A, A2, A3, n, n, n);
    mmm(A2, A2, A4, n, n, n);
    mmm(A, A4, A5, n, n, n);
    mmm(A2, A4, A6, n, n, n);

    eval3_4(A, A2, A4, A6, n, 5, P, Q);

    // check P

    scalar_matrix_mult(b[0], I, res, n, n);
    mm_add(b[1], A, res, res, n, n);
    mm_add(b[2], A2, res, res, n, n);
    mm_add(b[3], A3, res, res, n, n);
    mm_add(b[4], A4, res, res, n, n);
    mm_add(b[5], A5, res, res, n, n);

    assert(is_equal(res, P, n, n) == 1);
    printf("   P is correct\n");

    // check Q
    scalar_matrix_mult(-1.0, A, A, n, n);
    mmm(A, A, A2, n, n, n);
    mmm(A, A2, A3, n, n, n);
    mmm(A2, A2, A4, n, n, n);
    mmm(A, A4, A5, n, n, n);

    scalar_matrix_mult(b[0], I, res, n, n);
    mm_add(b[1], A, res, res, n, n);
    mm_add(b[2], A2, res, res, n, n);
    mm_add(b[3], A3, res, res, n, n);
    mm_add(b[4], A4, res, res, n, n);
    mm_add(b[5], A5, res, res, n, n);

    assert(is_equal(res, Q, n, n) == 1);
    printf("   Q is correct\n");
    printf("   Passed!\n");

    free(I);
    free(A);
    free(A2);
    free(A3);
    free(A4);
    free(A5);
    free(P);
    free(Q);
    free(res);
}

void test_repeated_squaring(int n){
    printf("Test repeated squaring: \n");
    int b = 6;

    double* A = (double*)malloc(n * n * sizeof(double));
    double* P_rs = (double*)malloc(n * n * sizeof(double));
    double* P_base = (double*)malloc(n * n * sizeof(double));
    double* Temp = (double*)malloc(n * n * sizeof(double));
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            A[i*n + j] = i;
        }
    }
    // printf("\n");
    // printmatrix(A,n,n);
    // printf("\n");
    matpow_by_squaring(A, n, b, P_rs);
    // printmatrix(P_rs,n,n);
    // printf("\n");

    mmm(A, A, P_base, n, n, n); 
    mmm(A, P_base, Temp, n, n, n);
    mmm(A, Temp, P_base, n, n, n);
    mmm(A, P_base, Temp, n, n, n);
    mmm(A, Temp, P_base, n, n, n);
    // printmatrix(P_base,n,n);
    // printf("\n");

    assert(is_equal(P_base, P_rs, n, n) == 1);
    printf("   Passed!\n");

    free(A);
    free(P_rs);
    free(P_base);
    free(Temp);
}

void test_eval3_6(int n){
    printf("Test eval3_6: \n");
    double* P = (double*)malloc(n * n * sizeof(double));
    double* Q = (double*)malloc(n * n * sizeof(double));
    double* R = (double*)malloc(n * n * sizeof(double));
    double* QR = (double*)malloc(n * n * sizeof(double));

    fillmatrix(P, n, n, 0, 10);
    fillmatrix(Q, n, n, 0, 10);

    eval3_6(P, Q, n, R, 0);
    // printf("Evaluated:\n");
    // printmatrix(R,n,n);

    mmm(Q, R, QR, n, n, n);
    // printf("QR:\n");
    // printmatrix(QR,n,n);
    // printf("P:\n");
    // printmatrix(P,n,n);
    assert(is_equal(QR, P, n, n)); //this probably won't work with doubles...
    printf("   Passed!\n");

    free(P);
    free(Q);
    free(R);
    free(QR);
}

void test_mat_col_sum(){
    printf("Test mat_col_sum: \n");
    int n = 3;
    double* A = (double*)malloc(n * n * sizeof(double));
    double* res = (double*)malloc(n * sizeof(double));
    get_test_matrix1(A, n, n, 2.0, -1.0);
    A[1] = -5.0;
    A[n] = 3.0;
    mat_col_sum(A, n, res);

    assert(res[0] == 4.0);
    assert(res[1] == -4.0);
    assert(res[2] == 0.0);
    printf("   Passed!\n");

    free(A);
    free(res);
}

void test_triangular_matrix(){
    printf("Test triangular matrix: \n");
    int n = 100;
    double* A = (double*)malloc(n * n * sizeof(double));
    fill_diagonal_matrix(A, 1.0, n);
    assert(is_upper_triangular(A,n,n) == 1);
    assert(is_lower_triangular(A,n,n) == 1);
    assert(is_triangular(A,n,n));
    printf("   Diagonal matrix passed!\n");
    
    A[n-1] = 1.0;
    assert(is_upper_triangular(A,n,n) == 1);
    assert(is_lower_triangular(A,n,n) == 0);
    assert(is_triangular(A,n,n));
    printf("   Upper triangular matrix passed!\n");
    
    A[n-1] = 0.0;
    A[n*(n-1)] = 1.0;
    assert(is_upper_triangular(A,n,n) == 0);
    assert(is_lower_triangular(A,n,n) == 1);
    assert(is_triangular(A,n,n));
    printf("   Lower triangular matrix passed!\n");

    A[n-1] = 1.0;
    assert(is_upper_triangular(A,n,n) == 0);
    assert(is_lower_triangular(A,n,n) == 0);
    assert(!is_triangular(A,n,n));
    printf("   Non-triangular matrix passed!\n");

    printf("   Passed!\n");

    free(A);
}

void test_ell(){
    printf("Test ell: \n");
    int n = 3;
    int m = 9;
    const double A[9] =
    {
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1
    };

    int res  = ell(A,n,m);
    printf("   Ell result is : %d \n", res);
}

void test_det(){
    printf("Test det: \n");
    int n = 4;
    double* A = (double*)malloc(n * n * sizeof(double));

    get_test_matrix1(A, n, n, 2.0, -1.0);
    A[1] = -5.0;
    A[n] = 3.0;

    double det_ = det(A, n);
    assert(det_ == 21.0);

    printf("   Passed!\n");
    free(A);
}

void test_inverse_matrix(int n){
    printf("Test inverse matrix with n=%d: \n", n);

    double* A = (double*)malloc(n * n * sizeof(double));
    double* A_inv = (double*)malloc(n * n * sizeof(double));
    double* AA_inv = (double*)malloc(n * n * sizeof(double));
    double* I = (double*)malloc(n * n * sizeof(double));

    fillmatrix(A, n, n, -10, 20);
    fill_diagonal_matrix(I, 1.0, n);

    inverse_matrix(A, A_inv, n);

    mmm(A, A_inv, AA_inv, n, n, n);
    assert(is_equal(AA_inv, I, n, n) == 1); // A*A^-1 == I
    printf("   Passed!\n");

    free(A);
    free(A_inv);
    free(AA_inv);
    free(I);
}

void test_mat_exp(double* A, double* E_expected, int n){
    double* E = (double*)malloc(n * n * sizeof(double));

    mat_exp(A, n, E);

    // printf("Expected:\n");
    // printmatrix(E_expected,n,n);
    // printf("Result:\n");
    // printmatrix(E,n,n);
    assert(is_equal(E, E_expected, n, n)==1);
    printf("   Passed!\n");

    free(E);
}

void test_mat_exp_I(int n){
    if(n < 2) return;
    printf("Test mat_exp_I with n=%d: \n", n);
    double* A = (double*)malloc(n * n * sizeof(double));
    double* E_expected = (double*)malloc(n * n * sizeof(double));

    fill_diagonal_matrix(A, 1.0, n);
    fill_diagonal_matrix(E_expected, M_E, n);
    test_mat_exp(A, E_expected, n);

    free(A);
    free(E_expected);
}

void test_mat_exp1(){
    printf("Test mat_exp1: \n");
    int n = 4;
    double* A = (double*)malloc(n * n * sizeof(double));
    double* E_expected = (double*)malloc(n * n * sizeof(double));

    get_test_matrix1(A, n, n, 3.0, -2.0);
    get_test_matrix1(E_expected, n, n, ((1+3*pow(M_E, 8))/(4*pow(M_E, 3))), ((1-pow(M_E, 8))/(4*pow(M_E, 3))));

    test_mat_exp(A, E_expected, n);

    free(A);
    free(E_expected);
}

void test_mat_exp_triangular(){
    printf("Test mat_exp_triangular: \n");
    int n = 3;
    double* A = (double*)malloc(n * n * sizeof(double));
    double* E_expected = (double*)malloc(n * n * sizeof(double));

    get_test_matrix1(A, n, n, 1.0, 0.0);
    A[3] = 1.0;
    A[6] = -1.0;
    A[7] = -1.0;
    A[8] = 2.0;

    double e2 = M_E * M_E;
    get_test_matrix1(E_expected, n, n, M_E, 0.0);
    E_expected[3] = M_E;
    E_expected[6] = 3*M_E - 2*e2;
    E_expected[7] = M_E - e2;
    E_expected[8] = e2;

    test_mat_exp(A, E_expected, n);

    free(A);
    free(E_expected);
}

void test_mat_exp_PI(){
    printf("\nTest mat_exp_PI: \n");
    double* N_ONE = creatematrix(2,2);
    double* PI = creatematrix(2,2);
    PI[0] = 0;
    PI[1] = -M_PI;
    PI[2] = M_PI;
    PI[3] = 0;

    mat_exp(PI,2,N_ONE);

    printf("A:\n");
    printmatrix(PI,2,2);
    printf("\n");
    printf("mat_exp(A):\n");
    printmatrix(N_ONE, 2, 2);

    freematrix(PI);
    freematrix(N_ONE);
}

int main(){
    srand(time(0));
    test_transpose(3, 6);
    test_mmm(3, 3, 6);
    test_triangular_matrix();

    test_normest();
    test_ell();
    test_mat_col_sum();

    test_det();
    test_inverse_matrix(4);
    test_inverse_matrix(8);
    test_eval3_6(5);

    test_eval3_4(16);
    test_repeated_squaring(5);

    test_mat_exp_I(4);
    test_mat_exp_I(10);
    test_mat_exp1();
    test_mat_exp_triangular();
    test_mat_exp_PI();

    print_constants();

    return 0;

}