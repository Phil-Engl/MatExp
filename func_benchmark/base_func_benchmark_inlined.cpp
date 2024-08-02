#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>
#include <assert.h>



#define DEBUG 0


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





/* --- Matrix operations template  --- */

/** 
 *@brief copy the values of matrix A to the matrix B
 *@param A Input matrix
 *@param B Output matrix
 *@param m number of rows
 *@param n number of columns
 */

void copy_matrix(const double* A, double* B, int m, int n){
    memcpy(B, A, m*n*sizeof(double));
}


void mmm(const double *A, const double *B, double *C, int common, int rowsA, int colsB){



    
    if(common % 4 == 0 && rowsA % 4 == 0 && colsB % 4 == 0){
        for(int i=0; i<colsB; i+=4){
            for(int j = 0; j<rowsA; j+=4){
                __m256d c_0 = _mm256_set1_pd(0.0);
                __m256d c_1 = _mm256_set1_pd(0.0);
                __m256d c_2 = _mm256_set1_pd(0.0);
                __m256d c_3 = _mm256_set1_pd(0.0);
                for(int k=0; k<common;k+=4){
                    __m256d a_0 = _mm256_load_pd(&A[rowsA * k + j]);
                    __m256d a_1 = _mm256_load_pd(&A[rowsA * (k+1) + j]);
                    __m256d a_2 = _mm256_load_pd(&A[rowsA * (k+2) + j]);
                    __m256d a_3 = _mm256_load_pd(&A[rowsA * (k+3) + j]);

                    __m256d b_00 = _mm256_set1_pd(B[common * i + k]);
                    __m256d b_10 = _mm256_set1_pd(B[common * i + k + 1]);
                    __m256d b_20 = _mm256_set1_pd(B[common * i + k + 2]);
                    __m256d b_30 = _mm256_set1_pd(B[common * i + k + 3]);

                    __m256d b_01 = _mm256_set1_pd(B[common * (i+1) + k ]);
                    __m256d b_11 = _mm256_set1_pd(B[common * (i+1) + k + 1]);
                    __m256d b_21 = _mm256_set1_pd(B[common * (i+1) + k + 2]);
                    __m256d b_31 = _mm256_set1_pd(B[common * (i+1) + k + 3]);

                    __m256d b_02 = _mm256_set1_pd(B[common * (i+2) + k]);
                    __m256d b_12 = _mm256_set1_pd(B[common * (i+2) + k + 1]);
                    __m256d b_22 = _mm256_set1_pd(B[common * (i+2) + k + 2]);
                    __m256d b_32 = _mm256_set1_pd(B[common * (i+2) + k + 3]);

                    __m256d b_03 = _mm256_set1_pd(B[common * (i+3) + k]);
                    __m256d b_13 = _mm256_set1_pd(B[common * (i+3) + k + 1]);
                    __m256d b_23 = _mm256_set1_pd(B[common * (i+3) + k + 2]);
                    __m256d b_33 = _mm256_set1_pd(B[common * (i+3) + k + 3]);

                    c_0 = _mm256_fmadd_pd(a_0, b_00, c_0);
                    c_0 = _mm256_fmadd_pd(a_1, b_10, c_0);
                    c_0 = _mm256_fmadd_pd(a_2, b_20, c_0);
                    c_0 = _mm256_fmadd_pd(a_3, b_30, c_0);

                    c_1 = _mm256_fmadd_pd(a_0, b_01, c_1);
                    c_1 = _mm256_fmadd_pd(a_1, b_11, c_1);
                    c_1 = _mm256_fmadd_pd(a_2, b_21, c_1);
                    c_1 = _mm256_fmadd_pd(a_3, b_31, c_1);

                    c_2 = _mm256_fmadd_pd(a_0, b_02, c_2);
                    c_2 = _mm256_fmadd_pd(a_1, b_12, c_2);
                    c_2 = _mm256_fmadd_pd(a_2, b_22, c_2);
                    c_2 = _mm256_fmadd_pd(a_3, b_32, c_2);

                    c_3 = _mm256_fmadd_pd(a_0, b_03, c_3);
                    c_3 = _mm256_fmadd_pd(a_1, b_13, c_3);
                    c_3 = _mm256_fmadd_pd(a_2, b_23, c_3);
                    c_3 = _mm256_fmadd_pd(a_3, b_33, c_3);
                }
                _mm256_store_pd(&C[i * rowsA + j], c_0);
                _mm256_store_pd(&C[(i+1) * rowsA + j], c_1);
                _mm256_store_pd(&C[(i+2) * rowsA + j], c_2);
                _mm256_store_pd(&C[(i+3) * rowsA + j], c_3);
            }
        }
    }else{ // fallback, only used for onenorm, can still be optimized
        for(int i = 0;i < colsB; i++){
            for(int j = 0; j < rowsA; j++){
                C[i * rowsA + j] = 0.0;
                for(int k = 0;k < common; k++){
                    C[i * rowsA + j] += A[k * rowsA + j] * B[k + common * i];
                }
            }
        }
    }
}

// TODO: inline
void matpow_by_squaring(const double *A, int n, long b, double *P){
    // FLOP_COUNT_INC(0, "matpow_by_squaring");
    if(b == 1){
        for(int i = 0; i < n*n; i++){
            P[i] = A[i];
        }
        return;
    }

    //P = I
    if(b == 0){
        for(int i = 0; i<n*n; i++){
            P[i] = 0.0;
        }

        for(int i=0; i<n; i++){
            P[i * n + i] = 1.0;  
        }
        return;
    }

    double *curr_P = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *tmp_P = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *cpy_A = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *tmp_A = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *tmp_pointer;
    
    //create a copy of A
    copy_matrix(A,cpy_A,n,n);

    //set curr_P = I
    for(int i = 0; i<n*n; i++){
        curr_P[i] = 0.0;
    }

    for(int i=0; i<n; i++){
        curr_P[i * n + i] = 1.0;  
    }
    
    while(b){
        if(b&1){
            mmm(curr_P, cpy_A, tmp_P, n, n, n);
            tmp_pointer = curr_P;
            curr_P = tmp_P;
            tmp_P = tmp_pointer;   
        }
        b = b >> 1; 
        if(!b){
            break;
        }
        mmm(cpy_A, cpy_A, tmp_A, n, n, n);
        tmp_pointer = cpy_A;
        cpy_A = tmp_A;
        tmp_A = tmp_pointer;
    }

    copy_matrix(curr_P, P, n, n);

    free(curr_P);
    free(tmp_P);
    free(cpy_A);
    free(tmp_A);
    return;
}
/**
 * @brief multiply a scalar to a matrix and then add it to another matrix
 * @param alpha the scalar we want to multiply the matrix with
 * @param A First input matrix that is multiplied with alpha
 * @param B second input matrix to add to alpha * A
 * @param C output matrix
 * @param m number of rows
 * @param n number of columns
*/
void mm_add(double alpha, const double *A, const double *B, double *C, int m, int n){
    
    __m256d alph = _mm256_set1_pd(alpha);
    
    __m256d a_0;
    __m256d b_0;
    __m256d c_0;
    
    for(int i = 0 ; i < m*n; i+=4 * 1){
        
         a_0 =_mm256_load_pd(&A[i + (0 * 4)]);
         b_0 = _mm256_load_pd(&B[i + (0 * 4)]);
         c_0 = _mm256_fmadd_pd(alph, a_0, b_0);
         _mm256_store_pd(&C[i + (0 * 4)], c_0);
        

    }
}

/**
 * @brief multiply a matrix by a scalar
 * @param alpha the scalar value
 * @param A input matrix
 * @param C output matrix
 * @param m number of rows
 * @param n number of columns
*/
void scalar_matrix_mult(double alpha, const double *A, double *C, int m, int n){
    
    __m256d alph = _mm256_set1_pd(alpha);
    
    __m256d temp_0;
    
    for (int i = 0 ; i < m*n ; i+= 4*1){
        
        temp_0 = _mm256_load_pd(&A[i + (0* 4)]);
        temp_0 = _mm256_mul_pd(alph, temp_0);
        _mm256_store_pd(&C[i + (0* 4)], temp_0);
        
    }
}

/**
 * @brief calculate the absolute value of a matrix
 * @param A input matrix
 * @param B output matrix
 * @param m number of rows
 * @param n number of columns
*/
void mat_abs(const double *A, double *B, int m, int n){
    
    __m256d sign_mask = _mm256_set1_pd(-0.0); // Set the sign bit to 1
    
    __m256d a_0;
    
    for(int i = 0 ; i < m*n ; i+= 4 * 1){
        
        a_0 = _mm256_load_pd(&A[i + (0*4)]);
        a_0 = _mm256_andnot_pd(sign_mask, a_0);
        _mm256_store_pd(&B[i + (0 * 4)], a_0);
        
    }
}

/**
 * @brief checks if the matrix A is lower triangular
 * 
 * @param A The input matrix A with dimensions m x n
 * @param m number of rows of A
 * @param n number of columns of A 
 * @return 1 if A is lower triangular
 * @return 0 if A is not lower triangular
 */
int is_lower_triangular(const double *A, int n){
for(int i = 1; i < n; i++){
        for(int j = 0; j < i; j++){
            
            if(A[i * n + j] != 0.0) return 0;
        }
    }
    return 1;
}

/**
 * @brief checks if the matrix A is upper triangular
 * 
 * @param A The input matrix A with dimensions m x n
 * @param m number of rows of A
 * @param n number of columns of A 
 * @return 1 if A is upper triangular
 * @return 0 if A is not upper triangular
 */
int is_upper_triangular(const double *A, int n){
    for(int i = 0; i < n; i++){
        for(int j = i + 1; j < n; j++){
            
            if(A[i * n + j] != 0.0) return 0;
        }
    }
    return 1;
}

/**
 * @brief checks if the matrix A is either upper or lower triangular
 * 
 * @param A The input matrix A with dimensions m x n
 * @param m number of rows of A
 * @param n number of columns of A 
 * @return 1 if A is upper triangular
 * @return 2 if A is lower triangular
 * @return 0 if A is neither upper nor lower triangular
 */
int is_triangular(const double *A, int n){
   // FLOP_COUNT_INC(0, "is_triangular");
   
   
   if(is_upper_triangular(A, n)){
    return 1;
   }else if(is_lower_triangular(A, n)){
    return 2;
   }else{
    return 0;
   }
}

//TODO: vectorize if necessary
int is_sparse(const double * A, int m, int n){
    int count = 0;
    for(int i=0; i<m*n; i++){
        
        if(A[i] != 0){
            count++;
        }
    }
    
    return (count < m * sqrt(n));
}


void transpose(const double *A, double *B, int n){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            B[j * n + i] = A[i * n + j];
        }
    }
}

/**
 * @brief find the highest row sum, assumes matrix has only positive values
 * @param A input matrix
 * @param m number of rows
 * @param n number of columns
 * @return the double containing the highest row sum
*/
double infinity_norm(const double* A, int m, int n){
    
    
    double max = 0.0;
    
    __m256d max_val_0 = _mm256_set1_pd(0.0);
    
    for(int i = 0; i < m; i+=4 * 1){
        
        __m256d sum_0 = _mm256_set1_pd(0.0);
        
        for(int j = 0; j < n; j++){
            
            __m256d ld_0 = _mm256_load_pd(&A[j * m + i + 4 * 0]);
            
            
            sum_0 = _mm256_add_pd(sum_0, ld_0);
            
        }
        
        max_val_0 = _mm256_max_pd(max_val_0, sum_0);
        
    }
    for(int i = 0; i < 4; i++){
        
        max = max_val_0[i] > max ? max_val_0[i] : max;
        
    }
    return max;
}



/**
 * @brief sum of the of the columns of matrix A
 * @param A input matrix
 * @param n column and row size
 * @param out output vector
*/

void mat_col_sum(const double* A, int n, double *out){
    
    
    double res_0 = 0.0;
    
    for(int i = 0; i < n; i+=1){ 
        
        __m256d acc_0 = _mm256_set1_pd(0.0); 
        
        for(int j = 0; j < n; j+=4){
            
            __m256d ld_0 = _mm256_load_pd(&A[(i + 0) * n + j]);
            
            
            acc_0 = _mm256_add_pd(acc_0, ld_0);
            
        }
        
        res_0 = acc_0[0] + acc_0[1] + acc_0[2] + acc_0[3];
        
        
        out[i+ 0] = res_0;
        
        
    }
}

void fill_diagonal_matrix(double* A, double diag, int n){
    // FLOP_COUNT_INC(0, "fill_diagonal_matrix");
    __m256d ZERO = _mm256_set1_pd(0.0);
    for(int i = 0; i < n*n; i+=4 * 1){
        
        _mm256_store_pd(&A[i + 4 * 0], ZERO);
        
    }

    for(int i = 0; i < n; i++){
        A[i * n + i] = diag;
    }
}




void forward_substitution_LU(double * A, double *y, double * b, int n){
 // ikj
    int k;
    int j;
   
   __m256i vindex = _mm256_set_epi64x(0, n, 2*n, 3*n);
    for(int i=0; i<n; i++){
        for(k = 0; k<n-3; k+=4){
            
            __m256d sum_vec = _mm256_setzero_pd();
            __m256d sum_vec1 = _mm256_setzero_pd();
            __m256d sum_vec2 = _mm256_setzero_pd();
            __m256d sum_vec3 = _mm256_setzero_pd();
            
            for(j=0; j<i-3; j+=4){

                __m256d A_vec = _mm256_set1_pd(A[j*n+i]);
                __m256d A_vec1 = _mm256_set1_pd(A[(j+1)*n+i]);
                __m256d A_vec2 = _mm256_set1_pd(A[(j+2)*n+i]);
                __m256d A_vec3 = _mm256_set1_pd(A[(j+3)*n+i]);

                __m256d y_vec = _mm256_i64gather_pd(&y[k*n+j], vindex, 8);
                __m256d y_vec1 = _mm256_i64gather_pd(&y[k*n+j +1], vindex, 8);
                __m256d y_vec2 = _mm256_i64gather_pd(&y[k*n+j +2], vindex, 8);
                __m256d y_vec3 = _mm256_i64gather_pd(&y[k*n+j +3], vindex, 8);

                sum_vec = _mm256_fmadd_pd(A_vec, y_vec, sum_vec);
                sum_vec1 = _mm256_fmadd_pd(A_vec1, y_vec1, sum_vec1);
                sum_vec2 = _mm256_fmadd_pd(A_vec2, y_vec2, sum_vec2);
                sum_vec3 = _mm256_fmadd_pd(A_vec3, y_vec3, sum_vec3);

                
                
            }
            //CLEANUP LOOP FOR J
            for(; j<i; j++){
                __m256d A_vec = _mm256_set1_pd(A[j*n+i]);
                __m256d y_vec = _mm256_i64gather_pd(&y[k*n+j], vindex, 8);
                sum_vec = _mm256_fmadd_pd(A_vec, y_vec, sum_vec);
                
                
            }
            
            __m256d b_vec = _mm256_i64gather_pd(&b[k*n+i], vindex, 8);

            __m256d tmp_final_sum = _mm256_add_pd(sum_vec3, sum_vec2);
            __m256d tmp_final_sum1 = _mm256_add_pd(sum_vec1, sum_vec);
            __m256d final_sum = _mm256_add_pd(tmp_final_sum1, tmp_final_sum);


            __m256d res_vec = _mm256_sub_pd(b_vec, final_sum);

            

            y[k*n+i] = res_vec[3];
            y[(k+1)*n+i] = res_vec[2];
            y[(k+2)*n+i] = res_vec[1];
            y[(k+3)*n+i] = res_vec[0];
            
        }

        //CLEANUP LOOP FOR K
        for(; k<n; k++){
            double sum = 0;
            for(j=0; j<i-3; j+=4){
                sum += A[j*n+i] * y[k*n + j];
                sum += A[(j+1)*n+i] * y[k*n + j +1];
                sum += A[(j+2)*n+i] * y[k*n + j +2];
                sum += A[(j+3)*n+i] * y[k*n + j +3];
                
            }
            for(; j<i; j++){
                sum += A[j*n+i] * y[k*n + j];
                
            }
            y[k*n+i] = b[k*n+i] - sum;
            
        }
    }

}

void backward_substitution(double * L, double *x, double * b, int n){
 // ikj
    int j;
    int k;
    int i;
    __m256i vindex = _mm256_set_epi64x(0, n, 2*n, 3*n);

    for(i = n-1; i>=0; i--){
        double rezi = 1.0/L[i*n+i];
        

        __m256d rezi_vec = _mm256_set1_pd(rezi);
        for(k = 0; k<n-3; k+=4){
            __m256d sum_vec = _mm256_setzero_pd();
            __m256d sum_vec1 = _mm256_setzero_pd();
            __m256d sum_vec2 = _mm256_setzero_pd();
            __m256d sum_vec3 = _mm256_setzero_pd();
            
            for(j = i+1; j<n-3; j+=4){
                __m256d L_vec = _mm256_set1_pd(L[j*n+i]);
                __m256d L_vec1 = _mm256_set1_pd(L[(j+1)*n+i]);
                __m256d L_vec2 = _mm256_set1_pd(L[(j+2)*n+i]);
                __m256d L_vec3 = _mm256_set1_pd(L[(j+3)*n+i]);

                __m256d x_vec = _mm256_i64gather_pd(&x[k*n+j], vindex, 8);
                __m256d x_vec1 = _mm256_i64gather_pd(&x[k*n+j +1], vindex, 8);
                __m256d x_vec2 = _mm256_i64gather_pd(&x[k*n+j +2], vindex, 8);
                __m256d x_vec3 = _mm256_i64gather_pd(&x[k*n+j +3], vindex, 8);

                sum_vec = _mm256_fmadd_pd(L_vec, x_vec, sum_vec);
                sum_vec1 = _mm256_fmadd_pd(L_vec1, x_vec1, sum_vec1);
                sum_vec2 = _mm256_fmadd_pd(L_vec2, x_vec2, sum_vec2);
                sum_vec3 = _mm256_fmadd_pd(L_vec3, x_vec3, sum_vec3);

                
            }

            for(; j<n; j++){      
                __m256d L_vec = _mm256_set1_pd(L[j*n+i]);
                __m256d x_vec = _mm256_i64gather_pd(&x[k*n+j], vindex, 8);
                sum_vec = _mm256_fmadd_pd(L_vec, x_vec, sum_vec); 
                
            }

            __m256d b_vec = _mm256_i64gather_pd(&b[k*n+i], vindex, 8);

            __m256d tmp_final_sum = _mm256_add_pd(sum_vec3, sum_vec2);
            __m256d tmp_final_sum1 = _mm256_add_pd(sum_vec1, sum_vec);
            __m256d final_sum = _mm256_add_pd(tmp_final_sum1, tmp_final_sum);

            __m256d tmp_res_vec = _mm256_sub_pd(b_vec, final_sum);
            __m256d res_vec = _mm256_mul_pd(tmp_res_vec, rezi_vec);

            

            x[k*n+i] = res_vec[3];
            x[(k+1)*n+i] = res_vec[2];
            x[(k+2)*n+i] = res_vec[1];
            x[(k+3)*n+i] = res_vec[0];
                
        }

        for(; k<n; k++){
            double sum = 0;

            for(j = i+1; j<n-3; j+=4){
                sum += L[j*n+i] * x[k*n+j];
                sum += L[(j+1)*n+i] * x[k*n+j +1];
                sum += L[(j+2)*n+i] * x[k*n+j +2];
                sum += L[(j+3)*n+i] * x[k*n+j +3];

                
            }

            for(; j<n; j++){
                sum += L[j*n+i] * x[k*n+j];
                
            }
            x[k*n + i] = b[k*n+i] - sum;
            x[k*n + i] *= rezi;   
            
        }
    }

}

void forward_substitution(double * A, double *y, double * b, int n){ // with div here
 // ikj

     int k;
    int j;
   
   __m256i vindex = _mm256_set_epi64x(0, n, 2*n, 3*n);
    for(int i=0; i<n; i++){
        double rezi = 1.0/A[i*n+i];
        __m256d rezi_vec = _mm256_set1_pd(rezi);

        
        
        for(k = 0; k<n-3; k+=4){
            __m256d sum_vec = _mm256_setzero_pd();
            __m256d sum_vec1 = _mm256_setzero_pd();
            __m256d sum_vec2 = _mm256_setzero_pd();
            __m256d sum_vec3 = _mm256_setzero_pd();
            
            for(j=0; j<i-3; j+=4){

                __m256d A_vec = _mm256_set1_pd(A[j*n+i]);
                __m256d A_vec1 = _mm256_set1_pd(A[(j+1)*n+i]);
                __m256d A_vec2 = _mm256_set1_pd(A[(j+2)*n+i]);
                __m256d A_vec3 = _mm256_set1_pd(A[(j+3)*n+i]);

                __m256d y_vec = _mm256_i64gather_pd(&y[k*n+j], vindex, 8);
                __m256d y_vec1 = _mm256_i64gather_pd(&y[k*n+j +1], vindex, 8);
                __m256d y_vec2 = _mm256_i64gather_pd(&y[k*n+j +2], vindex, 8);
                __m256d y_vec3 = _mm256_i64gather_pd(&y[k*n+j +3], vindex, 8);

                sum_vec = _mm256_fmadd_pd(A_vec, y_vec, sum_vec);
                sum_vec1 = _mm256_fmadd_pd(A_vec1, y_vec1, sum_vec1);
                sum_vec2 = _mm256_fmadd_pd(A_vec2, y_vec2, sum_vec2);
                sum_vec3 = _mm256_fmadd_pd(A_vec3, y_vec3, sum_vec3);

                
                
            }
            //CLEANUP LOOP FOR J
            for(; j<i; j++){
                __m256d A_vec = _mm256_set1_pd(A[j*n+i]);
                __m256d y_vec = _mm256_i64gather_pd(&y[k*n+j], vindex, 8);
                sum_vec = _mm256_fmadd_pd(A_vec, y_vec, sum_vec);

                
                
            }
            
            __m256d b_vec = _mm256_i64gather_pd(&b[k*n+i], vindex, 8);

            __m256d tmp_final_sum = _mm256_add_pd(sum_vec3, sum_vec2);
            __m256d tmp_final_sum1 = _mm256_add_pd(sum_vec1, sum_vec);
            __m256d final_sum = _mm256_add_pd(tmp_final_sum1, tmp_final_sum);


            __m256d tmp_res_vec = _mm256_sub_pd(b_vec, final_sum);
            __m256d res_vec = _mm256_mul_pd(tmp_res_vec, rezi_vec);

            


            y[k*n+i] = res_vec[3];
            y[(k+1)*n+i] = res_vec[2];
            y[(k+2)*n+i] = res_vec[1];
            y[(k+3)*n+i] = res_vec[0];
            
        }

        //CLEANUP LOOP FOR K
        for(; k<n; k++){
            double sum = 0;
            for(j=0; j<i-3; j+=4){
                sum += A[j*n+i] * y[k*n + j];
                sum += A[(j+1)*n+i] * y[k*n + j +1];
                sum += A[(j+2)*n+i] * y[k*n + j +2];
                sum += A[(j+3)*n+i] * y[k*n + j +3];

                
            }
            for(; j<i; j++){
                sum += A[j*n+i] * y[k*n + j];
                
            }
            y[k*n+i] = b[k*n+i] - sum;
            y[k*n+i] *= rezi; 

            
        }
    }


}

void LU_decomposition(double *LU, double *P, int n ){


    double curr_piv;
    int index_piv;
    double tmp;

    int *tmp_P = (int*) aligned_alloc(32, n*sizeof(int));
    for(int i=0; i<n; i++){
        tmp_P[i] = i;
    }

    for(int k=0; k<n; k++){
        // find pivot
        curr_piv = fabs(LU[k*n+k]);

        

        index_piv = k;

        for(int i=k+1; i<n; i++){ 
            double abs = fabs(LU[i*n+k]);

            

            if( abs > curr_piv ){
                curr_piv = abs;
                index_piv = i;
            }
            
        }

        if(index_piv != k){
        //swap rows to get pivot-row on top
            for(int x=0; x<n; x++){
                tmp = LU[x*n + k];
                LU[x*n +k] = LU[x*n+index_piv];
                LU[x*n+index_piv] = tmp;
            }
            //update permutation matrix
            tmp = tmp_P[k];
            tmp_P[k] = tmp_P[index_piv];
            tmp_P[index_piv] = tmp;
        }

        
        double rezi = 1.0 / LU[k*n+k];
        __m256d rezi_vec = _mm256_set1_pd(rezi);

        

       int i;
        for(i=1; i<(n-k)-3; i+=4){
            __m256d vec1 = _mm256_loadu_pd(&LU[k*n+ k +i]);
            __m256d res_vec = _mm256_mul_pd(vec1, rezi_vec);
            _mm256_storeu_pd(&LU[k*n+k+i], res_vec);

            
        }

        //CLEANUP LOOP FOR I
        for(; i<(n-k); i++){
            LU[k*n+k+i] = LU[k*n+ k +i] *rezi; 

            
        }


        for(int i=0; i<(n-k-1); i++){
            int j;
            for(j=0; j<(n-k-1)-3; j+=4){
                __m256d curr = _mm256_loadu_pd(&LU[(k+i+1)*n+ (k+j+1)]);
                __m256d a21_vec = _mm256_loadu_pd(&LU[k*n+ k +j +1]);
                __m256d a12_vec = _mm256_broadcast_sd(&LU[(k+i+1)*n+k]);

                __m256d prod = _mm256_mul_pd(a12_vec, a21_vec);
                __m256d res = _mm256_sub_pd(curr, prod);

                
                
                _mm256_storeu_pd(&LU[(k+i+1)*n+ (k+j+1)], res);           
            }
            //CLEANUP LOOP FOR J
            for(    ; j<(n-k-1); j++){
                LU[(k+i+1)*n + (k+j+1)] -= LU[(k+i+1)*n+k] * LU[k*n+k+1+j];
                
            }
        }

    }

    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            P[i*n+j] = 0;
        }
    }

    for(int i=0; i<n; i++){
        P[tmp_P[i]*n+i] = 1.0;
    }


    free(tmp_P);


}


/*----- onenorm functions template ----- */

/* ----- global constants ----- */


void mmm_nby2(const double* A, double* B, double* C, int n){
    
    
    for(int j = 0; j < n; j+=4){
        __m256d c_0 = _mm256_set1_pd(0.0);
        __m256d c_1 = _mm256_set1_pd(0.0);
        for(int k = 0; k < n; k+=4){
            __m256d a_0 = _mm256_load_pd(&A[n * k + j]);
            __m256d a_1 = _mm256_load_pd(&A[n * (k+1) + j]);
            __m256d a_2 = _mm256_load_pd(&A[n * (k+2) + j]);
            __m256d a_3 = _mm256_load_pd(&A[n * (k+3) + j]);

            __m256d b_00 = _mm256_set1_pd(B[k]);
            __m256d b_10 = _mm256_set1_pd(B[k + 1]);
            __m256d b_20 = _mm256_set1_pd(B[k + 2]);
            __m256d b_30 = _mm256_set1_pd(B[k + 3]);

            __m256d b_01 = _mm256_set1_pd(B[n + k ]);
            __m256d b_11 = _mm256_set1_pd(B[n + k + 1]);
            __m256d b_21 = _mm256_set1_pd(B[n + k + 2]);
            __m256d b_31 = _mm256_set1_pd(B[n + k + 3]);

            c_0 = _mm256_fmadd_pd(a_0, b_00, c_0);
            c_0 = _mm256_fmadd_pd(a_1, b_10, c_0);
            c_0 = _mm256_fmadd_pd(a_2, b_20, c_0);
            c_0 = _mm256_fmadd_pd(a_3, b_30, c_0);

            c_1 = _mm256_fmadd_pd(a_0, b_01, c_1);
            c_1 = _mm256_fmadd_pd(a_1, b_11, c_1);
            c_1 = _mm256_fmadd_pd(a_2, b_21, c_1);
            c_1 = _mm256_fmadd_pd(a_3, b_31, c_1);
        }
        _mm256_store_pd(&C[j], c_0);
        _mm256_store_pd(&C[n + j], c_1);
    }
}

/* ----- helper functions ----- */

/**
 * @brief checks if all columns in A are parallel to any column of B
 * 
 * @param A the input matrix A 
 * @param B the input matrix B 
 * @param m number of rows of A and B
 * @param n number of columns of A and B
 * @return int 1, if the condition in the description is met, 0 otherwise.
 */
///TODO: optimize for 2 column matrix
int check_all_columns_parallel(const double* A, const double* B, int m, int n){
    for(int i = 0; i < n; i++){
        int flag = 0;
        for(int j = 0; j < n; j+=2){
            __m256d acc_0 = _mm256_castsi256_pd(_mm256_set1_epi64x(0xFFFFFFFFFFFFFFFF)); 
            __m256d acc_1 = _mm256_castsi256_pd(_mm256_set1_epi64x(0xFFFFFFFFFFFFFFFF)); 
            
            for(int k = 0; k < m; k+=4){ //TODO: unroll more maybe
                __m256d a = _mm256_load_pd(&A[m+k]);
                __m256d b_0 = _mm256_load_pd(&B[(j + 0)*m+k]);
                __m256d b_1 = _mm256_load_pd(&B[(j + 1)*m+k]);
                
                __m256d cmp_0 = _mm256_cmp_pd(a, b_0, _CMP_EQ_OQ);
                __m256d cmp_1 = _mm256_cmp_pd(a, b_1, _CMP_EQ_OQ);
                
                acc_0 = _mm256_and_pd(acc_0, cmp_0);
                acc_1 = _mm256_and_pd(acc_1, cmp_1);

                
                
            }
            flag = flag || (acc_0[0] && acc_0[1] && acc_0[2] && acc_0[3]);
            flag = flag || (acc_1[0] && acc_1[1] && acc_1[2] && acc_1[3]);
            
        }
        if(!flag){
            return 0;
        }
    }
    
    return 1;
}

/**
 * @brief column in A needs resampling if it is parallel to any of its previous rows
 * or any row in B
 * 
 * @param k the index of the current column
 * @param v the current row
 * @param A the matrix A in which the current column resides
 * @param B the matrix B
 * @param m number of rows of A and B
 * @param n number of columns of A and B
 * @return int 1 if the the column needs resampling, 0 otherwise 
 */

int column_needs_resampling(int k, const double* v, const double* A, const double* B, int m, int n){
    
    for(int i = 0; i < k; i++){
        __m256d acc = _mm256_castsi256_pd(_mm256_set1_epi64x(0xFFFFFFFFFFFFFFFF)); 
        for(int j = 0; j < m; j += 4 * 1){
            
            __m256d v_0 = _mm256_load_pd(&v[j + 4 * 0]); 
            

            
            __m256d ld_0 = _mm256_load_pd(&A[i * m + j + 4 * 0]); 
            
            
            
            __m256d cmp_0 = _mm256_cmp_pd(v_0, ld_0, _CMP_EQ_OQ);
            
            

            
            acc = _mm256_and_pd(acc, cmp_0);
            
            
        }
        if(acc[0] && acc[1] && acc[2] && acc[3]){
            return 1;
        }
    }
    
    for(int i = 0; i < n; i++){
        __m256d acc = _mm256_castsi256_pd(_mm256_set1_epi64x(0xFFFFFFFFFFFFFFFF)); 
        for(int j = 0; j < m; j += 4 * 1){
            
            __m256d v_0 = _mm256_load_pd(&v[j + 4 * 0]);
            

            
            __m256d ld_0 = _mm256_load_pd(&B[i * m + j + 4 * 0]); 
            
            
            
            __m256d cmp_0 = _mm256_cmp_pd(v_0, ld_0, _CMP_EQ_OQ);
            
            

            
            acc = _mm256_and_pd(acc, cmp_0);
            
            
        }
        if(acc[0] && acc[1] && acc[2] && acc[3]){
            return 1;
        }
    }
    return 0;
}

/**
 * @brief Ensures that no column of A is parallel to another column of A
 * or to a column of B by replacing columns of A by rand{−1, 1}.
 * 
 * @param A Input matrix A, with all entries in {-1, 1} 
 * @param B Input matrix B, with all entries in {-1, 1} 
 * @param m number of rows of A and B 
 * @param n number of columns of A and B
 */
void resample_columns(double *A, double *B, int m, int n){
    for(int i = 0; i < n; i++){
        while(column_needs_resampling(i, &A[i * m], A, B, m, n)){
            for(int j = 0; j < m; j+=4*1){ 
                
                double rnd_0_0 = (double)(((rand() % 2) * 2) - 1);
                double rnd_1_0 = (double)(((rand() % 2) * 2) - 1);
                double rnd_2_0 = (double)(((rand() % 2) * 2) - 1);
                double rnd_3_0 = (double)(((rand() % 2) * 2) - 1);
                
                
                __m256d rand_0 = _mm256_set_pd(rnd_0_0,rnd_1_0,rnd_2_0,rnd_3_0);
                
                
                _mm256_store_pd(&A[i * m + j + 4*0], rand_0);
                
            }
        }
    }
}



/**
 * @brief checks if an index is already in the index history
 * 
 * @param idx the index to check
 * @param hist the index history 
 * @param hist_len the length of the index history
 * @return int 1 if the index is in the history, 0 otherwise
 */
int idx_in_hist(int idx, int *hist, int hist_len){
    if(hist_len == 0) return 0;
    int flag = 0;
    for(int i = 0; i < hist_len; i++){
        flag = flag || (idx == hist[i]);
    }
    return flag;
}




/* ----- one norm functions ----- */

double onenorm_best(const double* A, int m, int n, int* max_idx){
    double max = 0.0;
    __m256d ABS_MASK = _mm256_set1_pd(-0.0);
    double res_0 = 0.0;
    double res_1 = 0.0;
    
    for(int i = 0; i < n; i+=2){ 
        __m256d acc_0 = _mm256_set1_pd(0.0); 
        __m256d acc_1 = _mm256_set1_pd(0.0); 
        for(int j = 0; j < m; j+=4){
            __m256d ld_0 = _mm256_load_pd(&A[(i + 0) * m + j]);
            __m256d ld_1 = _mm256_load_pd(&A[(i + 1) * m + j]);
            
            ld_0 = _mm256_andnot_pd(ABS_MASK, ld_0);
            ld_1 = _mm256_andnot_pd(ABS_MASK, ld_1);
            
            acc_0 = _mm256_add_pd(acc_0, ld_0);
            acc_1 = _mm256_add_pd(acc_1, ld_1);
            
        }
        res_0 = acc_0[0] + acc_0[1] + acc_0[2] + acc_0[3];
        res_1 = acc_1[0] + acc_1[1] + acc_1[2] + acc_1[3];
        if(res_0 > max){
            max = res_0;
            *max_idx = i + 0;
        }
        if(res_1 > max){
            max = res_1;
            *max_idx = i + 1;
        }
        
        
        
        
    }
    return max;
}
//same as above, just without the best idx
double onenorm(const double* A, int m, int n){
    double max = 0.0;
    __m256d ABS_MASK = _mm256_set1_pd(-0.0);
    
    
    
    double res_0 = 0.0;
    

    for(int i = 0; i < n; i+=1){ 
        
        __m256d acc_0 = _mm256_set1_pd(0.0); 
        
        for(int j = 0; j < m; j+=4){
            
            __m256d ld_0 = _mm256_load_pd(&A[(i + 0) * m + j]);
            
            
            
            ld_0 = _mm256_andnot_pd(ABS_MASK, ld_0);
            
            
            
            acc_0 = _mm256_add_pd(acc_0, ld_0);
            
            
            
        

        }
        
        res_0 = acc_0[0] + acc_0[1] + acc_0[2] + acc_0[3];
        
        
        

        
        if(res_0 > max){
            max = res_0;
        }
        
        
    }
    return max;
}

//use this for matrices with only positive entries
double onenorm_abs_mat(const double* A, int m, int n){
    double max = 0.0;
    
    
    double res_0 = 0.0;
    

    for(int i = 0; i < n; i+=1){ 
        
        __m256d acc_0 = _mm256_set1_pd(0.0); 
        
        for(int j = 0; j < m; j+=4){
            
            __m256d ld_0 = _mm256_load_pd(&A[(i + 0) * m + j]);
            
            
            
            acc_0 = _mm256_add_pd(acc_0, ld_0);
            
            
        }
        
        res_0 = acc_0[0] + acc_0[1] + acc_0[2] + acc_0[3];
        
        

        
        if(res_0 > max){
            max = res_0;
        }
        
        
    }
    return max;
}

//TODO, maybe a special variable for these unrollings 
double normest(const double* A, int n){
    int t = 2;
   
    int itmax = 5;

    int k = 1;
    int best_j = 0;
    int ind_best = 0;
    int hist_len = 0;

    double est = 0.0;
    double est_old = 0.0;
    
    double max_h = 0.0;
    double x_elem = 1.0 / (double)n;
    double m_x_elem = -1.0 * x_elem;
    

    int fst_in_idx;
    int fst_out_idx, snd_out_idx, out_ctr;;
    double fst_in, snd_in;
    double fst_out, snd_out;

    __m256d ZERO = _mm256_set1_pd(0.0);
    __m256d ONE = _mm256_set1_pd(1.0);
    __m256d MONE = _mm256_set1_pd(-1.0);
    __m256d ABS_MASK = _mm256_set1_pd(-0.0);
    __m256d X_ELEM = _mm256_set1_pd(x_elem);
    
    int* ind_hist = (int*)aligned_alloc(32, t * itmax * sizeof(int));
    int* ind = (int*)aligned_alloc(32, n * sizeof(int));
    int* ind_in = (int*)aligned_alloc(32, n * sizeof(int));
    
    double* S = (double*)aligned_alloc(32, n * t * sizeof(double));
    double* S_old = (double*)aligned_alloc(32, n * t * sizeof(double));
    double* X = (double*)aligned_alloc(32, n * t * sizeof(double));
    double* Y = (double*)aligned_alloc(32, n * t * sizeof(double));
    double* Z = (double*)aligned_alloc(32, n * t * sizeof(double));

    double* h = (double*)aligned_alloc(32, n  * sizeof(double));
    
    
    double* AT = (double*)aligned_alloc(32, n * n * sizeof(double));
    transpose(A, AT, n);

    //t = 2, top has to be different
    for(int i = 0; i < t; i++){
        for(int j = 0; j < 4; j++){
            if(i == 0 || j > i){
                X[i * n + j] = x_elem;
            } else {
                X[i * n + j] = m_x_elem;
            }
        }
        _mm256_store_pd(&S[i*n], ZERO);
        for(int j = 4; j < n; j+=4 * 1){
            
            _mm256_store_pd(&X[i * n + j + 4 * 0], X_ELEM);
            _mm256_store_pd(&S[i * n + j + 4 * 0], ZERO);
            
        }

    }
    

    srand(time(0));
    while(1){
        //Y = A * X
        if(k == 1){
            
            mmm_nby2(A, X, Y, n);
            
        }else{
            for(int i = 0; i < n; i+=4){
                __m256d ld_0 = _mm256_load_pd(&A[ind[0] * n + i]);
                __m256d ld_1 = _mm256_load_pd(&A[ind[1] * n + i]);
                _mm256_store_pd(&Y[i], ld_0);
                _mm256_store_pd(&Y[n + i], ld_1);
            }
        }
        est = onenorm_best(Y,n,t, &best_j); 
        
        if(est > est_old || k == 2){
            if(k >= 2){
                ind_best = ind[best_j];
            }
        }

        //(1)
        if(k >= 2 && est <= est_old){
            est = est_old;
            break;
        }
        est_old = est;
        double *S_temp = S_old;
        S_old = S;
        S = S_temp;


        if(k > itmax){
            break;
        }

        //S = sign(Y)
          
        for(int i = 0; i < n * t; i+=4*1){ //this works for any possible t
            //S[i] = Y[i] >= 0.0 ? 1.0 : -1.0;
            
            __m256d ld_0 = _mm256_load_pd(&Y[i + 4 * 0]);
            

            
            __m256d cmp_0 = _mm256_cmp_pd(ld_0, ZERO, _CMP_GE_OQ);
            

            
            __m256d blend_0 =  _mm256_blendv_pd(MONE, ONE, cmp_0);
            

            
            _mm256_store_pd(&S[i + 4*0], blend_0);
            
        }

        if(check_all_columns_parallel(S, S_old, n, t)){
            break;
        }

        /* Ensure that no column of S is parallel to another column of S
        or to a column of S_old by replacing columns of S by rand{−1, 1}. */
        resample_columns(S, S_old, n, t);

        //(3)
        
        mmm_nby2(AT, S, Z, n);
        

        //only unroll once for n <= 4!!!!
        
        max_h = 0.0;
        for(int i = 0; i < n; i+=4 * 1){ // unroll here
            
            __m256d acc_0 = _mm256_set1_pd(0.0);
            
            for(int j = 0; j < t; j++){
                
                __m256d ld_0 = _mm256_load_pd(&Z[j * n + i + 4 * 0]);
                
                
                ld_0 = _mm256_andnot_pd(ABS_MASK, ld_0);
                
                
                acc_0 = _mm256_max_pd(acc_0, ld_0);
                
            }
            
            for(int k = 0; k < 4; k ++){
                //max_h = (acc_0[k] > max_h) ? acc_0[k] : max_h;
                
                if(acc_0[k] > max_h){
                    max_h = acc_0[k];
                }
            }
            
            
            _mm256_store_pd(&h[i + 4 * 0], acc_0);
            
        }

        //(4)
        if(k >= 2 && max_h == h[ind_best]){
            break;
        }

        fst_in_idx = -1;
        fst_out_idx = -1;
        snd_out_idx = -1;
        out_ctr = 0;

        fst_in = 0.0;
        snd_in = 0.0;
        fst_out = 0.0;
        snd_out = 0.0;

        
        for(int i = 0; i < n; i++){
            if(h[i] > snd_in || h[i] > snd_out){
                if(idx_in_hist(i, ind_hist, hist_len)){
                    
                    if(h[i] >= fst_in){
                        snd_in = fst_in;
                        fst_in = h[i];
                        fst_in_idx = i;
                    }else if(h[i] > snd_in){
                        
                        snd_in = h[i];
                    }
                }else{
                    
                    if(h[i] >= fst_out){
                        snd_out = fst_out;
                        fst_out = h[i];
                        snd_out_idx = fst_out_idx;
                        fst_out_idx = i;
                        out_ctr++;
                    }else if(h[i] > snd_out){
                        
                        snd_out = h[i];
                        snd_out_idx = i;
                        out_ctr++;
                    }
                }
            }
        }

        if(snd_in > fst_out){
            break;
        }

        if(out_ctr >= 2){
            ind[0] = fst_out_idx;
            ind[1] = snd_out_idx;
            ind_hist[hist_len++] = ind[0];
            ind_hist[hist_len++] = ind[1];
        }else if(out_ctr == 1){
            ind[0] = fst_out_idx;
            ind_hist[hist_len++] = ind[0];
            ind[1] = fst_in_idx;
        }
        
        k++;
    }

    free(ind_hist);
    free(ind);
    free(ind_in);
    free(S);
    free(S_old);
    free(X);
    free(Y);
    free(Z);
    free(h);
    
    free(AT);

    return est;
}
/* ---- eval functions template ---- */

/**
 * @brief eval 3.4 for m=3
 */
void eval3_4_m3(const double* A, const double* A_2, int n, double *P_3, double *Q_3)
{   
    
    double *U = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *V = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *Temp = (double*) aligned_alloc(32, n*n*sizeof(double));

    __m256d pade_3 = _mm256_set1_pd(pade_coefs[3]);
    __m256d pade_2 = _mm256_set1_pd(pade_coefs[2]);
    __m256d zero_v = _mm256_setzero_pd();

    // compute u and v separately
    for(int i = 0; i < n; i++){
        int j;
        for(j = 0; j < i-3; j+=4){  
            __m256d a = _mm256_load_pd(&A_2[i*n + j]);

            __m256d u_tmp = _mm256_fmadd_pd(pade_3, a, zero_v);
            __m256d v_tmp = _mm256_fmadd_pd(pade_2, a, zero_v);

            _mm256_store_pd(&U[i*n + j], u_tmp);
            _mm256_store_pd(&V[i*n + j], v_tmp);

        }

        // diagonal elements have to be initialized by some pade coefficient!
        __m256d a_diag = _mm256_load_pd(&A_2[i*n + j]);
        __m256d u_init_diag = _mm256_set_pd((i==j+3? pade_coefs[1] : 0.0), (i==j+2? pade_coefs[1] : 0.0), (i==j+1? pade_coefs[1] : 0.0), (i==j? pade_coefs[1] : 0.0));
        __m256d v_init_diag = _mm256_set_pd((i==j+3? pade_coefs[0] : 0.0), (i==j+2? pade_coefs[0] : 0.0), (i==j+1? pade_coefs[0] : 0.0), (i==j? pade_coefs[0] : 0.0));

        __m256d u_tmp_diag = _mm256_fmadd_pd(pade_3, a_diag, u_init_diag);
        __m256d v_tmp_diag = _mm256_fmadd_pd(pade_2, a_diag, v_init_diag);
            
        _mm256_store_pd(&U[i*n + j], u_tmp_diag);
        _mm256_store_pd(&V[i*n + j], v_tmp_diag);
        j+=4;

        for(; j < n-3; j+=4){ 
            __m256d a = _mm256_load_pd(&A_2[i*n + j]);

            __m256d u_tmp = _mm256_fmadd_pd(pade_3, a, zero_v);
            __m256d v_tmp = _mm256_fmadd_pd(pade_2, a, zero_v);
            
            _mm256_store_pd(&U[i*n + j], u_tmp);
            _mm256_store_pd(&V[i*n + j], v_tmp);
        }
    }

    // one matrix mult to get uneven powers of A
    mmm(A, U, Temp, n, n, n);

    for(int i=0; i<n*n; i+=4*1){
        
        __m256d u_tmp_0 = _mm256_load_pd(&Temp[i + 4*0]);
        __m256d v_tmp_0 = _mm256_load_pd(&V[i + 4*0]);
        

        
        __m256d p_tmp_0 = _mm256_add_pd(u_tmp_0, v_tmp_0);
        __m256d q_tmp_0 = _mm256_sub_pd(v_tmp_0, u_tmp_0);
        

        
        _mm256_store_pd(&P_3[i + 4*0], p_tmp_0);
        _mm256_store_pd(&Q_3[i + 4*0], q_tmp_0);
        
    }

    free(U);
    free(V);
    free(Temp);
}

/**
 * @brief eval 3.4 for m=5
 */
void eval3_4_m5(const double* A, const double* A_2, const double* A_4, int n, double *P_5, double *Q_5)
{   
    
    double *U = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *V = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *Temp = (double*) aligned_alloc(32, n*n*sizeof(double));

    __m256d pade_5 = _mm256_set1_pd(pade_coefs[5]);
    __m256d pade_4 = _mm256_set1_pd(pade_coefs[4]);
    __m256d pade_3 = _mm256_set1_pd(pade_coefs[3]);
    __m256d pade_2 = _mm256_set1_pd(pade_coefs[2]);
    __m256d zero_v = _mm256_setzero_pd();

    // compute u and v separately
    for(int i = 0; i < n; i++){
        int j;
        for(j = 0; j < i-3; j+=4){
            __m256d a2 = _mm256_load_pd(&A_2[i*n + j]);
            __m256d a4 = _mm256_load_pd(&A_4[i*n + j]);

            __m256d u_tmp = _mm256_fmadd_pd(pade_3, a2, zero_v);
            __m256d v_tmp = _mm256_fmadd_pd(pade_2, a2, zero_v);
            __m256d u     = _mm256_fmadd_pd(pade_5, a4, u_tmp);
            __m256d v     = _mm256_fmadd_pd(pade_4, a4, v_tmp);

            _mm256_store_pd(&U[i*n + j], u);
            _mm256_store_pd(&V[i*n + j], v);
        }
        __m256d a2 = _mm256_load_pd(&A_2[i*n + j]);
        __m256d a4 = _mm256_load_pd(&A_4[i*n + j]);
        
        __m256d u_init = _mm256_set_pd((i==j+3? pade_coefs[1] : 0.0), (i==j+2? pade_coefs[1] : 0.0), (i==j+1? pade_coefs[1] : 0.0), (i==j? pade_coefs[1] : 0.0));
        __m256d v_init = _mm256_set_pd((i==j+3? pade_coefs[0] : 0.0), (i==j+2? pade_coefs[0] : 0.0), (i==j+1? pade_coefs[0] : 0.0), (i==j? pade_coefs[0] : 0.0));

        __m256d u_tmp = _mm256_fmadd_pd(pade_3, a2, u_init);
        __m256d v_tmp = _mm256_fmadd_pd(pade_2, a2, v_init);
        __m256d u     = _mm256_fmadd_pd(pade_5, a4, u_tmp);
        __m256d v     = _mm256_fmadd_pd(pade_4, a4, v_tmp);

        _mm256_store_pd(&U[i*n + j], u);
        _mm256_store_pd(&V[i*n + j], v);

        j+=4;
        
        for(; j < n-3; j+=4){
            __m256d a2 = _mm256_load_pd(&A_2[i*n + j]);
            __m256d a4 = _mm256_load_pd(&A_4[i*n + j]);

            __m256d u_tmp = _mm256_fmadd_pd(pade_3, a2, zero_v);
            __m256d v_tmp = _mm256_fmadd_pd(pade_2, a2, zero_v);
            __m256d u     = _mm256_fmadd_pd(pade_5, a4, u_tmp);
            __m256d v     = _mm256_fmadd_pd(pade_4, a4, v_tmp);

            _mm256_store_pd(&U[i*n + j], u);
            _mm256_store_pd(&V[i*n + j], v);
        }
    }

    // one matrix mult to get uneven powers of A
    mmm(A, U, Temp, n, n, n);

    for(int i=0; i<n*n; i+=4*1){
        
        __m256d u_tmp_0 = _mm256_load_pd(&Temp[i + 4*0]);
        __m256d v_tmp_0 = _mm256_load_pd(&V[i + 4*0]);
        

        
        __m256d p_tmp_0 = _mm256_add_pd(u_tmp_0, v_tmp_0);
        __m256d q_tmp_0 = _mm256_sub_pd(v_tmp_0, u_tmp_0);
        

        
        _mm256_store_pd(&P_5[i + 4*0], p_tmp_0);
        _mm256_store_pd(&Q_5[i + 4*0], q_tmp_0);
        
    }

    free(U);
    free(V);
    free(Temp);
}

/**
 * @brief eval 3.4 for m=7
 */
void eval3_4_m7(const double* A, const double* A_2, const double* A_4, const double* A_6, int n, double *P_7, double *Q_7)
{   
    
    double *U = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *V = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *Temp = (double*) aligned_alloc(32, n*n*sizeof(double));

    __m256d pade_7 = _mm256_set1_pd(pade_coefs[7]);
    __m256d pade_6 = _mm256_set1_pd(pade_coefs[6]);
    __m256d pade_5 = _mm256_set1_pd(pade_coefs[5]);
    __m256d pade_4 = _mm256_set1_pd(pade_coefs[4]);
    __m256d pade_3 = _mm256_set1_pd(pade_coefs[3]);
    __m256d pade_2 = _mm256_set1_pd(pade_coefs[2]);

    __m256d zero_v = _mm256_setzero_pd();

    // compute u and v separately
    for(int i = 0; i < n; i++){
        int j;
        for(j = 0; j < i-3; j+=4){
            __m256d a2 = _mm256_load_pd(&A_2[i*n + j]);
            __m256d a4 = _mm256_load_pd(&A_4[i*n + j]);
            __m256d a6 = _mm256_load_pd(&A_6[i*n + j]);

            __m256d u_tmp1 = _mm256_fmadd_pd(pade_3, a2, zero_v);
            __m256d v_tmp1 = _mm256_fmadd_pd(pade_2, a2, zero_v);
            __m256d u_tmp2 = _mm256_fmadd_pd(pade_5, a4, u_tmp1);
            __m256d v_tmp2 = _mm256_fmadd_pd(pade_4, a4, v_tmp1);
            __m256d u      = _mm256_fmadd_pd(pade_7, a6, u_tmp2);
            __m256d v      = _mm256_fmadd_pd(pade_6, a6, v_tmp2);

            _mm256_store_pd(&U[i*n + j], u);
            _mm256_store_pd(&V[i*n + j], v);
        }

        __m256d a2 = _mm256_load_pd(&A_2[i*n + j]);
        __m256d a4 = _mm256_load_pd(&A_4[i*n + j]);
        __m256d a6 = _mm256_load_pd(&A_6[i*n + j]);

        __m256d u_init = _mm256_set_pd((i==j+3? pade_coefs[1] : 0.0), (i==j+2? pade_coefs[1] : 0.0), (i==j+1? pade_coefs[1] : 0.0), (i==j? pade_coefs[1] : 0.0));
        __m256d v_init = _mm256_set_pd((i==j+3? pade_coefs[0] : 0.0), (i==j+2? pade_coefs[0] : 0.0), (i==j+1? pade_coefs[0] : 0.0), (i==j? pade_coefs[0] : 0.0));

        __m256d u_tmp1 = _mm256_fmadd_pd(pade_3, a2, u_init);
        __m256d v_tmp1 = _mm256_fmadd_pd(pade_2, a2, v_init);
        __m256d u_tmp2 = _mm256_fmadd_pd(pade_5, a4, u_tmp1);
        __m256d v_tmp2 = _mm256_fmadd_pd(pade_4, a4, v_tmp1);
        __m256d u      = _mm256_fmadd_pd(pade_7, a6, u_tmp2);
        __m256d v      = _mm256_fmadd_pd(pade_6, a6, v_tmp2);

        _mm256_store_pd(&U[i*n + j], u);
        _mm256_store_pd(&V[i*n + j], v);

        j+=4;
        for(; j < n-3; j+=4){
            __m256d a2 = _mm256_load_pd(&A_2[i*n + j]);
            __m256d a4 = _mm256_load_pd(&A_4[i*n + j]);
            __m256d a6 = _mm256_load_pd(&A_6[i*n + j]);

            __m256d u_tmp1 = _mm256_fmadd_pd(pade_3, a2, zero_v);
            __m256d v_tmp1 = _mm256_fmadd_pd(pade_2, a2, zero_v);
            __m256d u_tmp2 = _mm256_fmadd_pd(pade_5, a4, u_tmp1);
            __m256d v_tmp2 = _mm256_fmadd_pd(pade_4, a4, v_tmp1);
            __m256d u      = _mm256_fmadd_pd(pade_7, a6, u_tmp2);
            __m256d v      = _mm256_fmadd_pd(pade_6, a6, v_tmp2);

            _mm256_store_pd(&U[i*n + j], u);
            _mm256_store_pd(&V[i*n + j], v);
        }
    }

    // one matrix mult to get uneven powers of A
    mmm(A, U, Temp, n, n, n);

    for(int i=0; i<n*n; i+=4*1){
        
        __m256d u_tmp_0 = _mm256_load_pd(&Temp[i + 4*0]);
        __m256d v_tmp_0 = _mm256_load_pd(&V[i + 4*0]);
        

        
        __m256d p_tmp_0 = _mm256_add_pd(u_tmp_0, v_tmp_0);
        __m256d q_tmp_0 = _mm256_sub_pd(v_tmp_0, u_tmp_0);
        

        
        _mm256_store_pd(&P_7[i + 4*0], p_tmp_0);
        _mm256_store_pd(&Q_7[i + 4*0], q_tmp_0);
        
    }

    free(U);
    free(V);
    free(Temp);
}

/**
 * @brief eval 3.4 for m=9
 */
void eval3_4_m9(const double* A, const double* A_2, const double* A_4, const double* A_6, const double* A_8, int n, double *P_9, double *Q_9)
{   
    
    double *U = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *V = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *Temp = (double*) aligned_alloc(32, n*n*sizeof(double));

    __m256d pade_9 = _mm256_set1_pd(pade_coefs[9]);
    __m256d pade_8 = _mm256_set1_pd(pade_coefs[8]);
    __m256d pade_7 = _mm256_set1_pd(pade_coefs[7]);
    __m256d pade_6 = _mm256_set1_pd(pade_coefs[6]);
    __m256d pade_5 = _mm256_set1_pd(pade_coefs[5]);
    __m256d pade_4 = _mm256_set1_pd(pade_coefs[4]);
    __m256d pade_3 = _mm256_set1_pd(pade_coefs[3]);
    __m256d pade_2 = _mm256_set1_pd(pade_coefs[2]);
    
    __m256d zero_v = _mm256_setzero_pd();

    // compute u and v separately
    for(int i = 0; i < n; i++){
        int j;
        for(j = 0; j < i-3; j+=4){
            __m256d a2 = _mm256_load_pd(&A_2[i*n + j]);
            __m256d a4 = _mm256_load_pd(&A_4[i*n + j]);
            __m256d a6 = _mm256_load_pd(&A_6[i*n + j]);
            __m256d a8 = _mm256_load_pd(&A_8[i*n + j]);

            __m256d u_tmp1 = _mm256_fmadd_pd(pade_3, a2, zero_v);
            __m256d v_tmp1 = _mm256_fmadd_pd(pade_2, a2, zero_v);
            __m256d u_tmp2 = _mm256_fmadd_pd(pade_5, a4, u_tmp1);
            __m256d v_tmp2 = _mm256_fmadd_pd(pade_4, a4, v_tmp1);
            __m256d u_tmp3 = _mm256_fmadd_pd(pade_7, a6, u_tmp2);
            __m256d v_tmp3 = _mm256_fmadd_pd(pade_6, a6, v_tmp2);
            __m256d u      = _mm256_fmadd_pd(pade_9, a8, u_tmp3);
            __m256d v      = _mm256_fmadd_pd(pade_8, a8, v_tmp3);

            _mm256_store_pd(&U[i*n + j], u);
            _mm256_store_pd(&V[i*n + j], v);
        }

        __m256d a2 = _mm256_load_pd(&A_2[i*n + j]);
        __m256d a4 = _mm256_load_pd(&A_4[i*n + j]);
        __m256d a6 = _mm256_load_pd(&A_6[i*n + j]);
        __m256d a8 = _mm256_load_pd(&A_8[i*n + j]);
        
        __m256d u_init = _mm256_set_pd((i==j+3? pade_coefs[1] : 0.0), (i==j+2? pade_coefs[1] : 0.0), (i==j+1? pade_coefs[1] : 0.0), (i==j? pade_coefs[1] : 0.0));
        __m256d v_init = _mm256_set_pd((i==j+3? pade_coefs[0] : 0.0), (i==j+2? pade_coefs[0] : 0.0), (i==j+1? pade_coefs[0] : 0.0), (i==j? pade_coefs[0] : 0.0));

        __m256d u_tmp1 = _mm256_fmadd_pd(pade_3, a2, u_init);
        __m256d v_tmp1 = _mm256_fmadd_pd(pade_2, a2, v_init);
        __m256d u_tmp2 = _mm256_fmadd_pd(pade_5, a4, u_tmp1);
        __m256d v_tmp2 = _mm256_fmadd_pd(pade_4, a4, v_tmp1);
        __m256d u_tmp3 = _mm256_fmadd_pd(pade_7, a6, u_tmp2);
        __m256d v_tmp3 = _mm256_fmadd_pd(pade_6, a6, v_tmp2);
        __m256d u      = _mm256_fmadd_pd(pade_9, a8, u_tmp3);
        __m256d v      = _mm256_fmadd_pd(pade_8, a8, v_tmp3);


        _mm256_store_pd(&U[i*n + j], u);
        _mm256_store_pd(&V[i*n + j], v);

        j+=4;

        for(; j < n-3; j+=4){
            __m256d a2 = _mm256_load_pd(&A_2[i*n + j]);
            __m256d a4 = _mm256_load_pd(&A_4[i*n + j]);
            __m256d a6 = _mm256_load_pd(&A_6[i*n + j]);
            __m256d a8 = _mm256_load_pd(&A_8[i*n + j]);

            __m256d u_tmp1 = _mm256_fmadd_pd(pade_3, a2, zero_v);
            __m256d v_tmp1 = _mm256_fmadd_pd(pade_2, a2, zero_v);
            __m256d u_tmp2 = _mm256_fmadd_pd(pade_5, a4, u_tmp1);
            __m256d v_tmp2 = _mm256_fmadd_pd(pade_4, a4, v_tmp1);
            __m256d u_tmp3 = _mm256_fmadd_pd(pade_7, a6, u_tmp2);
            __m256d v_tmp3 = _mm256_fmadd_pd(pade_6, a6, v_tmp2);
            __m256d u      = _mm256_fmadd_pd(pade_9, a8, u_tmp3);
            __m256d v      = _mm256_fmadd_pd(pade_8, a8, v_tmp3);

            _mm256_store_pd(&U[i*n + j], u);
            _mm256_store_pd(&V[i*n + j], v);
        }
    }

    // one matrix mult to get uneven powers of A
    mmm(A, U, Temp, n, n, n);

    for(int i=0; i<n*n; i+=4*1){
        
        __m256d u_tmp_0 = _mm256_load_pd(&Temp[i + 4*0]);
        __m256d v_tmp_0 = _mm256_load_pd(&V[i + 4*0]);
        

        
        __m256d p_tmp_0 = _mm256_add_pd(u_tmp_0, v_tmp_0);
        __m256d q_tmp_0 = _mm256_sub_pd(v_tmp_0, u_tmp_0);
        

        
        _mm256_store_pd(&P_9[i + 4*0], p_tmp_0);
        _mm256_store_pd(&Q_9[i + 4*0], q_tmp_0);
        
    }

    free(U);
    free(V);
    free(Temp);
}


/**
 * @brief Evaluates equation 3.5
 * 
 * @param A The input matrix A with dimension n x n
 * @param A_2 to A_8: precomputed powers of A
 * @param n The number of rows and columns of A, A_2, A_4 and A_6
 * @param P_13 The output matrix p_13(A) with dimension n x n
 * @param Q_13 The output matrix q_13(A) with dimension n x n
 */
void eval3_5(const double *A, double* A_2, double* A_4, double* A_6, int n, double *P_13, double *Q_13){
    
    double *U_tmp = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *V_tmp = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *U = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *V = (double*) aligned_alloc(32, n*n*sizeof(double));

    __m256d b_13 = _mm256_set1_pd(pade_coefs[13]);
    __m256d b_12 = _mm256_set1_pd(pade_coefs[12]);
    __m256d b_11 = _mm256_set1_pd(pade_coefs[11]);
    __m256d b_10 = _mm256_set1_pd(pade_coefs[10]);
    __m256d b_9 = _mm256_set1_pd(pade_coefs[9]);
    __m256d b_8 = _mm256_set1_pd(pade_coefs[8]);

    // computing u_13
    for(int i = 0; i < n*n; i+=4*1){
        
        __m256d a6_i_0 = _mm256_load_pd(&A_6[i+4*0]);
        __m256d a4_i_0 = _mm256_load_pd(&A_4[i+4*0]);
        __m256d a2_i_0 = _mm256_load_pd(&A_2[i+4*0]);
        

        
        __m256d u_tmp0_0 = _mm256_mul_pd(b_13, a6_i_0);
        __m256d v_tmp0_0 = _mm256_mul_pd(b_12, a6_i_0);

        __m256d u_tmp1_0 = _mm256_fmadd_pd(b_11, a4_i_0, u_tmp0_0);
        __m256d v_tmp1_0 = _mm256_fmadd_pd(b_10, a4_i_0, v_tmp0_0);

        __m256d u_tmp2_0 = _mm256_fmadd_pd(b_9, a2_i_0, u_tmp1_0);
        __m256d v_tmp2_0 = _mm256_fmadd_pd(b_8, a2_i_0, v_tmp1_0);
        

        
        _mm256_store_pd(&U_tmp[i+4*0], u_tmp2_0);
        _mm256_store_pd(&V_tmp[i+4*0], v_tmp2_0);
        
    }

    mmm(A_6, U_tmp, U, n, n, n);
    mmm(A_6, V_tmp, V, n, n, n);

    __m256d b_7 = _mm256_set1_pd(pade_coefs[7]);
    __m256d b_6 = _mm256_set1_pd(pade_coefs[6]);
    __m256d b_5 = _mm256_set1_pd(pade_coefs[5]);
    __m256d b_4 = _mm256_set1_pd(pade_coefs[4]);
    __m256d b_3 = _mm256_set1_pd(pade_coefs[3]);
    __m256d b_2 = _mm256_set1_pd(pade_coefs[2]); 

    // TODO: maybe we can combine this with the first loop
    for(int i = 0; i < n*n; i+=4*1){
        
        __m256d a6_i_0 = _mm256_load_pd(&A_6[i+4*0]);
        __m256d a4_i_0 = _mm256_load_pd(&A_4[i+4*0]);
        __m256d a2_i_0 = _mm256_load_pd(&A_2[i+4*0]);
        __m256d u_tmp0_0 = _mm256_load_pd(&U[i+4*0]);
        __m256d v_tmp0_0 = _mm256_load_pd(&V[i+4*0]);
        

        
        __m256d u_tmp1_0 = _mm256_fmadd_pd(b_7, a6_i_0, u_tmp0_0);
        __m256d v_tmp1_0 = _mm256_fmadd_pd(b_6, a6_i_0, v_tmp0_0);

        __m256d u_tmp2_0 = _mm256_fmadd_pd(b_5, a4_i_0, u_tmp1_0);
        __m256d v_tmp2_0 = _mm256_fmadd_pd(b_4, a4_i_0, v_tmp1_0);

        __m256d u_tmp3_0 = _mm256_fmadd_pd(b_3, a2_i_0, u_tmp2_0);
        __m256d v_tmp3_0 = _mm256_fmadd_pd(b_2, a2_i_0, v_tmp2_0);
        

        
        _mm256_store_pd(&U[i+4*0], u_tmp3_0);
        _mm256_store_pd(&V[i+4*0], v_tmp3_0);
        
    }

    for(int i = 0; i < n; i++){
            U[i*n+i] += pade_coefs[1];
            V[i*n+i] += pade_coefs[0];
    }

    mmm(A, U, U_tmp, n, n, n); 

    for(int i=0; i < n*n; i+=4*1){
        
        __m256d u_tmp_0 = _mm256_load_pd(&U_tmp[i+4*0]);
        __m256d v_tmp_0 = _mm256_load_pd(&V[i+4*0]);
        

        
        __m256d p_tmp_0 = _mm256_add_pd(u_tmp_0, v_tmp_0);
        __m256d q_tmp_0 = _mm256_sub_pd(v_tmp_0, u_tmp_0);
        

        
        _mm256_store_pd(&P_13[i+4*0], p_tmp_0);
        _mm256_store_pd(&Q_13[i+4*0], q_tmp_0);
        
    }

    free(U_tmp);
    free(V_tmp);
    free(U);
    free(V);
}

/**
 * @brief Evaluates equation 3.6 -> solves the linear system Q_m * R_m = P_m
 * 
 * @param P_m The input matrix p_m(A) of dimension n x n
 * @param Q_m The input matrix q_m(A) of dimension n x n
 * @param n The size of the matrices
 * @param R_m The output matrix r_m(A) of dimension n x n
 */
void eval3_6(double *P_m, double *Q_m, int n, double *R_m, int triangular_indicator){
    if(triangular_indicator == 1){
        backward_substitution(Q_m, R_m, P_m, n);
        return;
    }else if(triangular_indicator == 2){
        forward_substitution(Q_m, R_m, P_m, n);
        return;
    }else{
    
    double *P = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *Y = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *P_permuted = (double*) aligned_alloc(32, n*n*sizeof(double));

    

    LU_decomposition(Q_m, P, n);
    mmm(P, P_m, P_permuted, n, n, n);

    forward_substitution_LU(Q_m, Y , P_permuted, n);
    backward_substitution(Q_m, R_m, Y, n);

    free(Y);
    free(P);
    free(P_permuted);

    }
    
}
/* ---- matrix exponential  template ----- */

int ell(const double* A, int n, int m){ 
    

    /*-- ell --*/
    double* A_temp_abs = (double*) aligned_alloc(32, n*n*sizeof(double));
    double* A_temp_abs_tmp = (double*) aligned_alloc(32, n*n*sizeof(double));
    double* A_temp_abs_pow = (double*) aligned_alloc(32, n*n*sizeof(double));
    double* A_temp_abs_pow_tmp = (double*) aligned_alloc(32, n*n*sizeof(double));
    double* tmp_pointer;
    mat_abs(A, A_temp_abs,n,n);
    
    copy_matrix(A_temp_abs, A_temp_abs_pow,n,n); //always 1 as lsb
    mmm(A_temp_abs, A_temp_abs, A_temp_abs_tmp,n,n,n);
    tmp_pointer = A_temp_abs;
    A_temp_abs = A_temp_abs_tmp;
    A_temp_abs_tmp = tmp_pointer;
    int b = 2*m+1;
    b = b >> 1;
    while(b){
        if(b&1){
            mmm(A_temp_abs_pow, A_temp_abs, A_temp_abs_pow_tmp, n, n, n);
            tmp_pointer = A_temp_abs_pow;
            A_temp_abs_pow = A_temp_abs_pow_tmp;
            A_temp_abs_pow_tmp = tmp_pointer;
        }
        b = b >> 1;
        if(!b){
            break;
        }
        mmm(A_temp_abs, A_temp_abs, A_temp_abs_tmp,n,n,n);
        tmp_pointer = A_temp_abs;
        A_temp_abs = A_temp_abs_tmp;
        A_temp_abs_tmp = tmp_pointer;
    }
    
    
    double abs_norm = normest(A_temp_abs_pow, n);

    free(A_temp_abs);
    free(A_temp_abs_tmp);
    free(A_temp_abs_pow);
    free(A_temp_abs_pow_tmp);

    if(abs_norm <= 0.0){
        return 0;
    }
    
    double alpha = (coeffs[m] * abs_norm) / onenorm(A,n,n);
    return fmax((int)ceil(log2(alpha / (*unit_roundoff))/(2*m)), 0);
}

/* main function */
void mat_exp(const double *A, int n, double *E){
    // FLOP_COUNT_INC are added within the branches
    if(n % 4 != 0){
       printf("please use matrix with n being a multiple of 4!");
       return;
    }
    
    double *P_m = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *Q_m = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *R_m = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *Temp = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *A_abs = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *P_m_abs = (double*) aligned_alloc(32, n*n*sizeof(double));
    double * Q_m_abs = (double*) aligned_alloc(32, n*n*sizeof(double));
    
    //double* A_abs_pow_m = (double*) aligned_alloc(32, n*n*sizeof(double));
    //double* A_abs_pow_m_temp = (double*) aligned_alloc(32, n*n*sizeof(double));
    //double* abs_pow_swap;

    mat_abs(A, A_abs, n, n); 
    int triangular_indicator = is_triangular(A, n);
    

    double * A_2 = (double*) aligned_alloc(32, n*n*sizeof(double)); 
    double * A_abs_2 = (double*) aligned_alloc(32, n*n*sizeof(double)); 
    mmm(A, A, A_2, n, n, n); // A^2
    mmm(A_abs, A_abs, A_abs_2, n, n, n); // |A|^2
    
    double * A_4 = (double*) aligned_alloc(32, n*n*sizeof(double));
    double * A_abs_4 = (double*) aligned_alloc(32, n*n*sizeof(double));
    mmm(A_2, A_2, A_4, n, n, n); // A^4
    mmm(A_abs_2, A_abs_2, A_abs_4, n, n, n); // |A|^4

    double * A_6 = (double*) aligned_alloc(32, n*n*sizeof(double)); 
    double * A_abs_6 = (double*) aligned_alloc(32, n*n*sizeof(double)); 
    mmm(A_2, A_4, A_6, n, n, n); // A^6
    mmm(A_abs_2, A_abs_4, A_abs_6, n, n, n); // |A|^6

    double * A_8 = (double*) aligned_alloc(32, n*n*sizeof(double)); 
    // will be computed later

    do{
        // ========================= p = 3 =========================
        
        double d_6 = pow(normest(A_6, n), 0.16666666666666667); // onenormest(A_2, 3)
        double eta_1 = fmax(pow(normest(A_4, n), 0.25), d_6); // onenormest(A_2, 2)

        //mmm(A_abs, A_abs_6, A_abs_pow_m, n,n,n); //|A|^(2*3+1) = |A|^7
        if(eta_1 <= theta[3] && ell(A, n, 3) == 0){
            
            eval3_4_m3(A, A_2, n, P_m, Q_m);
            eval3_4_m3(A_abs, A_abs_2, n, P_m_abs, Q_m_abs);
            mat_col_sum(P_m_abs, n, Temp);

            double divider = fmin(onenorm(P_m, n, n), onenorm(Q_m, n, n));
            if(infinity_norm(Temp, n, 1)/divider <= theta3_exp_10){
                if(DEBUG) printf("returned m = 3\n");
                eval3_6(P_m, Q_m, n, E, triangular_indicator);
                break;
            }
        }

        // ======================== p = 5 =========================
        
        
        double d_4 = pow(onenorm(A_4, n, n), 0.25);
        double eta_2 = fmax(d_4, d_6);

        /*abs_pow_swap = A_abs_pow_m;
        A_abs_pow_m = A_abs_pow_m_temp;
        A_abs_pow_m_temp = abs_pow_swap;

        mmm(A_abs_4, A_abs_pow_m_temp, A_abs_pow_m, n,n,n); //|A|^11*/
        if(eta_2 <= theta[5] && ell(A, n, 5) == 0){
            

            eval3_4_m5(A, A_2, A_4, n, P_m, Q_m);
            eval3_4_m5(A_abs, A_abs_2, A_abs_4, n, P_m_abs, Q_m_abs);
            mat_col_sum(P_m_abs, n, Temp);

            double divider = fmin(onenorm(P_m, n, n), onenorm(Q_m, n, n));
            if(infinity_norm(Temp, n, 1)/divider <= theta5_exp_10){
                eval3_6(P_m, Q_m, n, E, triangular_indicator);
                if(DEBUG) printf("returned m = 5\n");
                break;
            }
        }

        // ======================== p = 7, 9 ========================

        mmm(A_4, A_4, A_8, n, n, n); // A_4^2

        
        d_6 = pow(onenorm(A_6, n, n), 0.16666666666666667);
        double d_8 = pow(normest(A_8, n), 0.125); //onenormest(A_4, 2)
        double eta_3 = fmax(d_6, d_8);
        
        int found = 0;
        for(int m = 7; m <= 9; m+=2){
            /*abs_pow_swap = A_abs_pow_m;
            A_abs_pow_m = A_abs_pow_m_temp;
            A_abs_pow_m_temp = abs_pow_swap;

            mmm(A_abs_4, A_abs_pow_m_temp, A_abs_pow_m, n,n,n); //|A|^15, |A|^19*/
            if(eta_3 <= theta[m] && ell(A, n, m) == 0){
                
                if(m == 7){
                    eval3_4_m7(A, A_2, A_4, A_6, n, P_m, Q_m);
                    eval3_4_m7(A_abs, A_abs_2, A_abs_4, A_abs_6, n, P_m_abs, Q_m_abs);
                }else{
                    double * A_abs_8 = (double*) aligned_alloc(32, n*n*sizeof(double));
                    mmm(A_abs_2, A_abs_6, A_abs_8, n, n, n); // |A|^8
                    eval3_4_m9(A, A_2, A_4, A_6, A_8, n, P_m, Q_m);
                    eval3_4_m9(A_abs, A_abs_2, A_abs_4, A_abs_6, A_abs_8, n, P_m_abs, Q_m_abs);
                    free(A_abs_8);
                }
                mat_col_sum(P_m_abs, n, Temp);

                double divider = fmin(onenorm(P_m, n, n), onenorm(Q_m, n, n));
                if(infinity_norm(Temp, n, 1)/divider <= (m==7? theta7_exp_10: theta9_exp_10)){
                    eval3_6(P_m, Q_m, n, E, triangular_indicator);
                    if(DEBUG) printf("returned m = %d\n", m);
                    found = 1;
                    break;
                }
            }
        }
        if(found) break;

        // ========================= p = 13 =========================
        
        double * A_10 = (double*) aligned_alloc(32, n*n*sizeof(double)); 
        mmm(A_4, A_6, A_10, n,n,n); // A_4 * A_6

        double eta_4 = fmax(d_8, pow(normest(A_10, n), 0.1)); // onenormest(A_4, A_6)
        free(A_10);
        double eta_5 = fmin(eta_3, eta_4);

        int s = (int) fmax(ceil(log2(eta_5 * theta13_inv)), 0.0);
        double* A_temp = (double*) aligned_alloc(32, n*n*sizeof(double));
        scalar_matrix_mult(pow(2.0, -s), A, A_temp, n, n); // 2^-s * A
        
        s = s + ell(A_temp, n, 13);
        
        scalar_matrix_mult(pow(2.0, -s), A, A_temp, n, n);
        scalar_matrix_mult(pow(2.0, -2.0*s), A_2, A_2, n, n);
        scalar_matrix_mult(pow(2.0, -4.0*s), A_4, A_4, n, n);
        scalar_matrix_mult(pow(2.0, -6.0*s), A_6, A_6, n, n);

        eval3_5(A_temp, A_2, A_4, A_6, n, P_m, Q_m);
        eval3_5(A_abs, A_abs_2, A_abs_4, A_abs_6, n, P_m_abs, Q_m_abs);
        mat_col_sum(P_m_abs, n, Temp);
        double divider = fmin(onenorm(P_m, n, n), onenorm(Q_m, n, n));
        int s_max = (int)ceil(log2(onenorm(A,n,n) * theta13_inv));
        if(infinity_norm(Temp, n, 1)/divider <= theta13_exp_10){
            if(DEBUG) printf("case scaled\n");
            eval3_6(P_m, Q_m, n, R_m, triangular_indicator);
        }else{
            
            if(DEBUG) printf("case scaled again\n");
            int s1 = s_max - s;
            s = s_max;
            scalar_matrix_mult(pow(2.0, -s1), A_temp, A_temp, n, n);
            scalar_matrix_mult(pow(2.0, -2.0*s1), A_2, A_2, n, n);
            scalar_matrix_mult(pow(2.0, -4.0*s1), A_4, A_4, n, n);
            scalar_matrix_mult(pow(2.0, -6.0*s1), A_6, A_6, n, n);
            eval3_5(A_temp, A_2, A_4, A_6, n, P_m, Q_m); 
            eval3_6(P_m, Q_m, n, R_m, triangular_indicator);
        }

        // ==================== squaring phase ======================
        if(triangular_indicator){
            if(DEBUG) printf("triangular\n");
            

            // E = r_m(2^(-s) * A)
            copy_matrix(R_m, E, n, n);


            //Replace diag(E) by e^(2^-s * diag(A))
            for(int j=0; j<n; j++){
                E[j*n + j] = exp(pow(2, -s) * A[j*n + j]);
            }

            for(int i = s-1; i>=0; i--){
                // E = E^2
                mmm(E, E, R_m, n, n, n);
                copy_matrix(R_m, E, n, n);

                // Replace diag(E) by e^(2^-i * diag(A))
                for(int j=0; j<n; j++){
                    E[j*n + j] = exp(pow(2, -i) * A[j*n + j]);
                }

                // Replace first superdiagonal of E by explicit formula ----> WHY IS THIS NOT USED..?!?!?!?!??!
                /*
                for(int j=0; j<n-1; j++){
                    double a1 = pow(2, -i) * A[j*n + j];
                    double a2 = pow(2, -i) * A[(j+1)*n + j + 1];
                    double t12 = pow(2, -i) * A[j*n + j + 1];
                    double t21 = pow(2, -i) * A[(j+1)*n + j -1];

                    E[j*n + j + 1] = t12 * exp( (a1+a2)/2 ) * sinch( (a2-a1)/2 );
                    E[j*n + j + 1] = t12 * (exp(a2) - exp(a1)) / (a2-a1); // Alternative formula to prevent overflow
        

                    E[(j+1)*n + j -1] = t21 * exp( (a1+a2)/2 ) * sinch( (a2-a1)/2 );
                    E[(j+1)*n + j -1] = t21 * (exp(a2) - exp(a1)) / (a2-a1);// Alternative formula to prevent overflow
                }*/

                mmm(E, E, R_m, n, n, n);
            }
        
        }else{
            if(DEBUG) printf("non triangular\n");
            if(DEBUG) assert(s >= 0);
            double* tmp_pointer;
            long b = (long)pow(2,s);
            while(1){
                if(b & 1){
                    copy_matrix(R_m, E, n, n);
                    break;
                }
                b = b >> 1;
                mmm(R_m, R_m, A_temp, n, n, n);
                tmp_pointer = R_m;
                R_m = A_temp;
                A_temp = tmp_pointer;

            }
        }

        free(A_temp);
    }while(0);
    

    if(DEBUG) printf("cleanup\n");
    
    free(Temp);
    free(A_2);
    free(A_4);
    free(A_6);
    free(A_abs_2);
    free(A_abs_4);
    free(A_abs_6);
    free(A_8);
    free(P_m);
    free(Q_m);
    free(R_m);
    free(A_abs);
    free(Q_m_abs);
    free(P_m_abs);
    //free(A_abs_pow_m);
    //free(A_abs_pow_m_temp);
    return;
    
}
//benchmark_template

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <filesystem>
#include <fmt/core.h>
#include <immintrin.h>
#include <string>
#include <vector>



#include "../benchmark/tsc_x86.h"
#include "../benchmark/utils.h"
#define CYCLES_REQUIRED 1e8
#define REP 50



double mmm_benchmark(const double* A, const double *B, double *C, int m, int n, int t){
    double cycles = 0.;
    long num_runs = 100;
    double multiplier = 1;
    myInt64 start, end;
    //list<double> cyclesList;
   
    do {
        num_runs = num_runs * multiplier;
        start = start_tsc();
        for (long i = 0; i < num_runs; i++) {
            mmm(A, B, C, m, n, t);       
        }
        end = stop_tsc(start);

        cycles = (double)end;
        multiplier = (CYCLES_REQUIRED) / (cycles);
        
    } while (multiplier > 2);
    double total_cycles = 0;
    for (size_t j = 0; j < REP; j++) {

        start = start_tsc();
        for (long i = 0; i < num_runs; ++i) {
            mmm(A, B, C, m, n, t);        
        }
        end = stop_tsc(start);

        cycles = ((double)end) / num_runs;
        total_cycles += cycles;

       // cyclesList.push_back(cycles);
    }
    total_cycles /= REP;

    cycles = total_cycles;
  
    return  cycles;
}

// Base operations function benchmark

// MM add
double scalar_matrix_mult_benchmark(double alpha, const double *A, double *C, int m, int n){
     double cycles = 0.;
    long num_runs = 100;
    double multiplier = 1;
    myInt64 start, end;
    //list<double> cyclesList;
   
    // Warm up phase
     do {
        num_runs = num_runs * multiplier;
        start = start_tsc();
        for (long i = 0; i < num_runs; i++) {
            scalar_matrix_mult(alpha, A, C, m, n);    
        }
        end = stop_tsc(start);

        cycles = (double)end;
        multiplier = (CYCLES_REQUIRED) / (cycles);
        
    } while (multiplier > 2);
    double total_cycles = 0;
    for (size_t j = 0; j < REP; j++) {

        start = start_tsc();
        for (long i = 0; i < num_runs; ++i) {
            scalar_matrix_mult(alpha, A, C, m, n);           
        }
        end = stop_tsc(start);

        cycles = ((double)end) / num_runs;
        total_cycles += cycles;

       // cyclesList.push_back(cycles);
    }
    total_cycles /= REP;

    cycles = total_cycles;

    return  cycles;
}

// One Norm

double onenorm_benchmark(const double* A, int m, int n){
     double cycles = 0.;
    long num_runs = 100;
    double multiplier = 1;
    myInt64 start, end;
    //list<double> cyclesList;

    // Warm up phase
     do {
        num_runs = num_runs * multiplier;
        start = start_tsc();
        double max_val;
        for (long i = 0; i < num_runs; i++) {
            max_val = onenorm(A,  m, n);       
        }
        end = stop_tsc(start);

        cycles = (double)end;
        multiplier = (CYCLES_REQUIRED) / (cycles);
        
    } while (multiplier > 2);
    double total_cycles = 0;
    double max ;
    for (size_t j = 0; j < REP; j++) {

        start = start_tsc();
        for (long i = 0; i < num_runs; ++i) {
            max = onenorm( A, m, n);          
        }
        end = stop_tsc(start);

        cycles = ((double)end) / num_runs;
        total_cycles += cycles;

       // cyclesList.push_back(cycles);
    }
    total_cycles /= REP;

    cycles = total_cycles;

    return  cycles;

}

// normest
double onenormest_benchmark(const double *A, int n){
    double cycles = 0.;
    long num_runs = 100;
    double multiplier = 1;
    myInt64 start, end;
    //list<double> cyclesList;
 
    // Warm up phase
     do {
        num_runs = num_runs * multiplier;
        start = start_tsc();
        double max_val;
        for (long i = 0; i < num_runs; i++) {
            max_val = normest(A,  n);       
        }
        end = stop_tsc(start);

        cycles = (double)end;
        multiplier = (CYCLES_REQUIRED) / (cycles);
        
    } while (multiplier > 2);
    double total_cycles = 0;
    double max ;
    for (size_t j = 0; j < REP; j++) {

        start = start_tsc();
        for (long i = 0; i < num_runs; ++i) {
            max = normest(A,   n);       
        }
        end = stop_tsc(start);

        cycles = ((double)end) / num_runs;
        total_cycles += cycles;

       // cyclesList.push_back(cycles);
    }
    total_cycles /= REP;

    cycles = total_cycles;
    
    return  cycles; 

}

// Eval 3.4

double eval3_4_m3_benchmark(const double* A, const double* A_2, int n, double *P_3, double *Q_3){
    double cycles = 0.;
    long num_runs = 100;
    double multiplier = 1;
    myInt64 start, end;
    
    // Warm up phase
     do {
        num_runs = num_runs * multiplier;
        start = start_tsc();
        double max_val;
        for (long i = 0; i < num_runs; i++) {
            eval3_4_m3(A, A_2, n, P_3, Q_3);      
        }
        end = stop_tsc(start);

        cycles = (double)end;
        multiplier = (CYCLES_REQUIRED) / (cycles);
        
    } while (multiplier > 2);
    double total_cycles = 0;
    double max ;
    for (size_t j = 0; j < REP; j++) {

        start = start_tsc();
        for (long i = 0; i < num_runs; ++i) {
            eval3_4_m3(A, A_2, n, P_3, Q_3);   
        }
        end = stop_tsc(start);

        cycles = ((double)end) / num_runs;
        total_cycles += cycles;

    }
    total_cycles /= REP;

    cycles = total_cycles;
   
    return  cycles;     
}


double eval3_4_m5_benchmark(const double* A, const double* A_2, const double* A_4, int n, double *P_5, double *Q_5){
    double cycles = 0.;
    long num_runs = 100;
    double multiplier = 1;
    myInt64 start, end;
    
    // Warm up phase
     do {
        num_runs = num_runs * multiplier;
        start = start_tsc();
        double max_val;
        for (long i = 0; i < num_runs; i++) {
            eval3_4_m5(A, A_2, A_4, n, P_5, Q_5);      
        }
        end = stop_tsc(start);

        cycles = (double)end;
        multiplier = (CYCLES_REQUIRED) / (cycles);
        
    } while (multiplier > 2);
    double total_cycles = 0;
    double max ;
    for (size_t j = 0; j < REP; j++) {

        start = start_tsc();
        for (long i = 0; i < num_runs; ++i) {
            eval3_4_m5(A, A_2, A_4, n, P_5, Q_5);   
        }
        end = stop_tsc(start);

        cycles = ((double)end) / num_runs;
        total_cycles += cycles;

    }
    total_cycles /= REP;

    cycles = total_cycles;
    
    return  cycles;     
}

double eval3_4_m7_benchmark(const double* A, const double* A_2, const double* A_4, const double* A_6, int n, double *P_7, double *Q_7){
    double cycles = 0.;
    long num_runs = 100;
    double multiplier = 1;
    myInt64 start, end;
    
    // Warm up phase
     do {
        num_runs = num_runs * multiplier;
        start = start_tsc();
        double max_val;
        for (long i = 0; i < num_runs; i++) {
            eval3_4_m7(A, A_2, A_4, A_6, n, P_7, Q_7);      
        }
        end = stop_tsc(start);

        cycles = (double)end;
        multiplier = (CYCLES_REQUIRED) / (cycles);
        
    } while (multiplier > 2);
    double total_cycles = 0;
    double max ;
    for (size_t j = 0; j < REP; j++) {

        start = start_tsc();
        for (long i = 0; i < num_runs; ++i) {
            eval3_4_m7(A, A_2, A_4, A_6, n, P_7, Q_7);  
        }
        end = stop_tsc(start);

        cycles = ((double)end) / num_runs;
        total_cycles += cycles;

    }
    total_cycles /= REP;

    cycles = total_cycles;
    
    return  cycles;     
}

double eval3_4_m9_benchmark(const double* A, const double* A_2, const double* A_4, const double* A_6, const double* A_8, int n, double *P_9, double *Q_9){
    double cycles = 0.;
    long num_runs = 100;
    double multiplier = 1;
    myInt64 start, end;
    
    // Warm up phase
     do {
        num_runs = num_runs * multiplier;
        start = start_tsc();
        double max_val;
        for (long i = 0; i < num_runs; i++) {
            eval3_4_m9(A, A_2, A_4, A_6, A_8, n, P_9, Q_9);      
        }
        end = stop_tsc(start);

        cycles = (double)end;
        multiplier = (CYCLES_REQUIRED) / (cycles);
        
    } while (multiplier > 2);
    double total_cycles = 0;
    double max ;
    for (size_t j = 0; j < REP; j++) {

        start = start_tsc();
        for (long i = 0; i < num_runs; ++i) {
            eval3_4_m9(A, A_2, A_4, A_6, A_8, n, P_9, Q_9);  
        }
        end = stop_tsc(start);

        cycles = ((double)end) / num_runs;
        total_cycles += cycles;

    }
    total_cycles /= REP;

    cycles = total_cycles;
 
    return  cycles;     
}



// Eval 3_5

double eval3_5_benchmark(const double* A, double *A_2, double* A_4, double* A_6, int n, double *P_13, double *Q_13){
    double cycles = 0.;
    long num_runs = 100;
    double multiplier = 1;
    myInt64 start, end;
    
    // Warm up phase
     do {
        num_runs = num_runs * multiplier;
        start = start_tsc();
        double max_val;
        for (long i = 0; i < num_runs; i++) {
            eval3_5(A,  A_2, A_4, A_6, n, P_13, Q_13);       
        }
        end = stop_tsc(start);

        cycles = (double)end;
        multiplier = (CYCLES_REQUIRED) / (cycles);
        
    } while (multiplier > 2);
    double total_cycles = 0;
    double max ;
    for (size_t j = 0; j < REP; j++) {

        start = start_tsc();
        for (long i = 0; i < num_runs; ++i) {
            eval3_5(A,  A_2, A_4, A_6, n, P_13, Q_13);      
        }
        end = stop_tsc(start);

        cycles = ((double)end) / num_runs;
        total_cycles += cycles;

    }
    total_cycles /= REP;

    cycles = total_cycles;
    
    return cycles;     
}

// Eval 3_6
double eval3_6_benchmark(double * P_m, double *Q_m, int n, double *R_m, int triangular_indicator){
    double cycles = 0.;
    long num_runs = 100;
    double multiplier = 1;
    myInt64 start, end;
    
    // Warm up phase
     do {
        num_runs = num_runs * multiplier;
        start = start_tsc();
        double max_val;
        for (long i = 0; i < num_runs; i++) {
            eval3_6(P_m,  Q_m, n, R_m , triangular_indicator);       
        }
        end = stop_tsc(start);

        cycles = (double)end;
        multiplier = (CYCLES_REQUIRED) / (cycles);
        
    } while (multiplier > 2);
    double total_cycles = 0;
    double max ;
    for (size_t j = 0; j < REP; j++) {

        start = start_tsc();
        for (long i = 0; i < num_runs; ++i) {
            eval3_6(P_m,  Q_m, n, R_m , triangular_indicator);  
        }
        end = stop_tsc(start);

        cycles = ((double)end) / num_runs;
        total_cycles += cycles;

    }
    total_cycles /= REP;

    cycles = total_cycles;
   
    return  cycles;  
}

int readmatrix(double *A, int n, const char* path){
    FILE *fptr = fopen(path, "r");
    if(fptr==NULL){
        printf("file not found\n");
        return 0;
    }
    char buf[50];
    fscanf(fptr, "%s", buf);
    int m = atoi(buf);
    fscanf(fptr, "%s", buf);
    int k = atoi(buf);
    if(m!=k){
        printf("non square matrix!\n");
        return 0;
    }
    if(m != n){
        printf("wrong size\n");
        return 0;
    }

    for(int i = 0; i < n; i++){
        for(int j = 0; j<n; j++){
            fscanf(fptr, "%s", buf);
            A[j * n + i] = atof(buf);
        }
    }
    return 1;
}



int main (int argc, char* argv[]) {
    printf("Function,#cycles,n,floats/cycle\n");

    int lower = 0;
    int upper = 4;
    int numpaths = 7;

    if(argc < 5){
        printf("mmm mm_add onenorm eval3_4 eval3_5\n");
        return 0;
    }
    int mmm_on = atoi(argv[1]);
    int scm_on = atoi(argv[2]);
    int onenorm_on = atoi(argv[3]);
    int eval3_4_on = atoi(argv[4]);
    int eval3_5_on = atoi(argv[5]);
    
    int sizes[] = {32,64,128,256,512,768,1024};
    const char **paths = (const char**)malloc(10 * sizeof(const char*));
    paths[0] = "data/scaled_twice_dense/dense_0032.txt";
    paths[1] = "data/scaled_twice_dense/dense_0064.txt";
    paths[2] = "data/scaled_twice_dense/dense_0128.txt";
    paths[3] = "data/scaled_twice_dense/dense_0256.txt";
    paths[4] = "data/scaled_twice_dense/dense_0512.txt";
    paths[5] = "data/scaled_twice_dense/dense_0768.txt";
    paths[6] = "data/scaled_twice_dense/dense_1024.txt";

    double** mats = (double**)malloc(10* sizeof(double*));

    
    for(int i = 0; i < numpaths; i++){
        mats[i] = (double *) aligned_alloc(32, sizes[i]*sizes[i]*sizeof(double));
        if(!readmatrix(mats[i],sizes[i],paths[i])){
            return 0;
        }
        
    }
    
    for(int i = lower; i < upper; i++){
        long n = sizes[i];
        //mmm benchmark
        double *A = (double *) aligned_alloc(32, n*n*sizeof(double));
        double *B = (double *) aligned_alloc(32, n*n*sizeof(double));
        double* C = (double *) aligned_alloc(32, n*n*sizeof(double));

        double *A_2 = (double *) aligned_alloc(32, n*n*sizeof(double));
        double *A_4 = (double *) aligned_alloc(32, n*n*sizeof(double));
        double *A_6 = (double *) aligned_alloc(32, n*n*sizeof(double));
        double *A_8 = (double *) aligned_alloc(32, n*n*sizeof(double));
        double *P_m = (double *) aligned_alloc(32, n*n*sizeof(double));
        double *Q_m = (double *) aligned_alloc(32, n*n*sizeof(double));
        if(mmm_on || scm_on){
            memcpy(A, mats[i], n*n*sizeof(double));
            memcpy(B, mats[i], n*n*sizeof(double));
        }

        double res = 0;
        double flops = 0;
        if(mmm_on){   
            res = mmm_benchmark(A,B,C,n,n,n);
            flops = 2*n*n*n;
            printf("mmm, %.4f, %4d, %.4f\n", res, n, flops/res);
        }

       
        //mm_add benchmark
        if(scm_on){
            res = scalar_matrix_mult_benchmark(2.5,A,C,n,n);
            flops = n*n;
            printf("scalar_mm, %.4f, %4d, %.4f\n", res, n, flops/res);  
        }
       
        //onenorm benchmark
        if(onenorm_on){
            res = onenorm_benchmark(A,n,n);
            flops = 2*n*n;
            printf("onenorm, %.4f, %4d, %.4f\n", res, n, flops/res);
        }

        if(eval3_4_on || eval3_5_on){
            memcpy(A, mats[i], n*n*sizeof(double));
            memcpy(A_2, mats[i], n*n*sizeof(double));
            memcpy(A_4, mats[i], n*n*sizeof(double));
            memcpy(A_6, mats[i], n*n*sizeof(double));
            memcpy(A_8, mats[i], n*n*sizeof(double));
        }        


        if(eval3_4_on){
            //flops = 6*n*n + 2*n*n*n;
            //res = eval3_4_m3_benchmark(A, A_2, n, P_m, Q_m);
            //printf("eval3_4_m3, %.4f, %4d, %.4f\n", res, n, flops/res);
//
            //flops = 10*n*n + 2*n*n*n;
            //res = eval3_4_m5_benchmark(A, A_2, A_4, n, P_m, Q_m);
            //printf("eval3_4_m5, %.4f, %4d, %.4f\n", res, n, flops/res);
            //
            //flops = 14*n*n + 2*n*n*n;
            //res = eval3_4_m7_benchmark(A, A_2, A_4, A_6, n, P_m, Q_m);
            //printf("eval3_4_m7, %.4f, %4d, %.4f\n", res, n, flops/res);        
           
            flops = 18*n*n + 2*n*n*n;
            res = eval3_4_m9_benchmark(A, A_2, A_4, A_6, A_8, n, P_m, Q_m);
            printf("eval3_4, %.4f, %4d, %.4f\n", res, n, flops/res);
            
        }

        if(eval3_5_on){
            flops = 26*n*n + 6*n*n*n;
            res = eval3_5_benchmark(A, A_2, A_4, A_6, n, P_m, Q_m);
            printf("eval3_5, %.4f, %4d, %.4f\n", res, n, flops/res);        
        }

        free(A);
        free(B);
        free(C);
        free(A_2);
        free(A_4);
        free(A_6);
        free(A_8);
        free(P_m);
        free(Q_m);
    }

    return 1;

}