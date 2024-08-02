#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>
#include <assert.h>

#include <cblas.h>


#define DEBUG 0


#define USEDGEMM 1



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



const double alpha = 1.0;
const double beta = 0.0;
const CBLAS_LAYOUT layout = CblasColMajor;



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

    const double alpha = 1.0;
    const double beta = 0.0;
    CBLAS_LAYOUT layout = CblasColMajor;
    CBLAS_TRANSPOSE tA = CblasNoTrans;
    CBLAS_TRANSPOSE tB = CblasNoTrans;
    cblas_dgemm(layout, tA, tB, rowsA, colsB, common ,alpha ,A, rowsA ,B ,common ,beta, C, rowsA);

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



void forward_substitution(double * A, double *y, double * b, int n){
 // ikj
    int k;
    int j;
    for(int i=0; i<n; i++){
        for(k=0; k<n-3; k+=4){
            __m256d sum0 = _mm256_setzero_pd();
            __m256d sum1 = _mm256_setzero_pd();
            __m256d sum2 = _mm256_setzero_pd();
            __m256d sum3 = _mm256_setzero_pd();
            __m256d sum_cleanup = _mm256_setzero_pd();

            //__m256d res_vec= _mm256_loadu_pd( &b[i*n+k] );
            __m256d res_vec= _mm256_load_pd( &b[i*n+k] );
        
            __m256d denominator_vec = _mm256_broadcast_sd(&A[i*n+i]);
            __m256d one_vec = _mm256_set1_pd(1.0);
            
            __m256d rezi_vec = _mm256_div_pd(one_vec, denominator_vec);

            for(j=0; j<i-3; j+=4){
                //__m256d y_vec0 = _mm256_loadu_pd( &y[j*n+k] );
                //__m256d y_vec1 = _mm256_loadu_pd( &y[(j+1)*n+k] );
                //__m256d y_vec2 = _mm256_loadu_pd( &y[(j+2)*n+k] );
                //__m256d y_vec3 = _mm256_loadu_pd( &y[(j+3)*n+k] );

                __m256d y_vec0 = _mm256_load_pd( &y[j*n+k] );
                __m256d y_vec1 = _mm256_load_pd( &y[(j+1)*n+k] );
                __m256d y_vec2 = _mm256_load_pd( &y[(j+2)*n+k] );
                __m256d y_vec3 = _mm256_load_pd( &y[(j+3)*n+k] );

                __m256d A_vec0 = _mm256_broadcast_sd(&A[i*n+j]);
                __m256d A_vec1 = _mm256_broadcast_sd(&A[i*n+j +1]);
                __m256d A_vec2 = _mm256_broadcast_sd(&A[i*n+j +2]);
                __m256d A_vec3 = _mm256_broadcast_sd(&A[i*n+j +3]);

                sum0 = _mm256_fmadd_pd(A_vec0, y_vec0, sum0);
                sum1 = _mm256_fmadd_pd(A_vec1, y_vec1, sum1);
                sum2 = _mm256_fmadd_pd(A_vec2, y_vec2, sum2);
                sum3 = _mm256_fmadd_pd(A_vec3, y_vec3, sum3);
                
            }

            // CLEANUP FOR J LOOP
            for(; j<i; j++){
                //__m256d y_vec_cleanup= _mm256_loadu_pd( &y[j*n+k] );
                __m256d y_vec_cleanup= _mm256_load_pd( &y[j*n+k] );
                __m256d A_vec_cleanup = _mm256_broadcast_sd(&A[i*n+j]);
                sum_cleanup = _mm256_fmadd_pd(A_vec_cleanup, y_vec_cleanup, sum_cleanup);
                

            }

            // add sums together
            __m256d tmp_sum0 = _mm256_add_pd(sum0, sum1);
            __m256d tmp_sum1 = _mm256_add_pd(sum2, sum3);
            __m256d final_sum = _mm256_add_pd(tmp_sum0, tmp_sum1);

            //subtract sums from result
            res_vec = _mm256_sub_pd(res_vec, sum_cleanup);
            res_vec = _mm256_sub_pd(res_vec, final_sum);

            //multiply by rezipocal vector instead of dividing
            res_vec = _mm256_mul_pd(res_vec, rezi_vec);
            
            //store result
            //_mm256_storeu_pd(&y[i*n+k], res_vec);
            _mm256_store_pd(&y[i*n+k], res_vec);
        }

        // CLEANUP LOOP for k
        for(; k<n; k++){
            double sum = 0;
            y[i*n + k] = b[i*n + k];
            double rezi = 1.0 / A[i*n+i];
            
            for(int j=0; j<i; j++){
                sum += A[i*n+j] * y[j*n + k];
                
            }
            y[i*n+k] -= sum;
            y[i*n+k] *=  rezi;
            
        }
    }

}

void backward_substitution(double * U, double *x, double * b, int n){
 // ikj
    int j;
    int k;
    for(int i=n-1; i>=0; i--){
        for(k=0; k<n-3; k+=4){
            __m256d sum0 = _mm256_setzero_pd();
            __m256d sum1 = _mm256_setzero_pd();
            __m256d sum2 = _mm256_setzero_pd();
            __m256d sum3 = _mm256_setzero_pd();
            __m256d sum_cleanup = _mm256_setzero_pd();

            //__m256d res_vec= _mm256_loadu_pd( &b[i*n+k] );
            __m256d res_vec= _mm256_load_pd( &b[i*n+k] );

            __m256d denominator_vec = _mm256_broadcast_sd(&U[i*n+i]);
            __m256d one_vec = _mm256_set1_pd(1.0);
            __m256d rezi_vec = _mm256_div_pd(one_vec, denominator_vec);
            

            for(j=i+1; j<n-3; j+=4){
                //__m256d x_vec0 = _mm256_loadu_pd( &x[j*n+k] );
                //__m256d x_vec1 = _mm256_loadu_pd( &x[(j+1)*n+k] );
                //__m256d x_vec2 = _mm256_loadu_pd( &x[(j+2)*n+k] );
                //__m256d x_vec3 = _mm256_loadu_pd( &x[(j+3)*n+k] );

                __m256d x_vec0 = _mm256_load_pd( &x[j*n+k] );
                __m256d x_vec1 = _mm256_load_pd( &x[(j+1)*n+k] );
                __m256d x_vec2 = _mm256_load_pd( &x[(j+2)*n+k] );
                __m256d x_vec3 = _mm256_load_pd( &x[(j+3)*n+k] );

                __m256d U_vec0 = _mm256_broadcast_sd(&U[i*n+j]);
                __m256d U_vec1 = _mm256_broadcast_sd(&U[i*n+j +1]);
                __m256d U_vec2 = _mm256_broadcast_sd(&U[i*n+j +2]);
                __m256d U_vec3 = _mm256_broadcast_sd(&U[i*n+j +3]);

                sum0 = _mm256_fmadd_pd(U_vec0, x_vec0, sum0);
                sum1 = _mm256_fmadd_pd(U_vec1, x_vec1, sum1);
                sum2 = _mm256_fmadd_pd(U_vec2, x_vec2, sum2);
                sum3 = _mm256_fmadd_pd(U_vec3, x_vec3, sum3);
                
            }
            //CLEANUP LOOP FOR J
            for(; j<n; j++){
            //__m256d x_vec_cleanup= _mm256_loadu_pd( &x[j*n+k] );
            __m256d x_vec_cleanup= _mm256_load_pd( &x[j*n+k] );
            __m256d U_vec_cleanup = _mm256_broadcast_sd(&U[i*n+j]);

            sum_cleanup = _mm256_fmadd_pd(U_vec_cleanup, x_vec_cleanup, sum_cleanup);
            

            }

            __m256d tmp_sum0 = _mm256_add_pd(sum0, sum1);
            __m256d tmp_sum1 = _mm256_add_pd(sum2, sum3);
            __m256d final_sum = _mm256_add_pd(tmp_sum0, tmp_sum1);

            res_vec = _mm256_sub_pd(res_vec, sum_cleanup);
            res_vec = _mm256_sub_pd(res_vec, final_sum);

            res_vec = _mm256_mul_pd(res_vec, rezi_vec);
            

            //_mm256_storeu_pd(&x[i*n+k], res_vec);
            _mm256_store_pd(&x[i*n+k], res_vec);
        }
        
        //CLEANUP LOOP FOR K
        for(; k<n; k++){
            double sum = 0;
                for(j=i+1; j<n; j++){
                    sum += U[i*n+j] * x[j*n+k];
                    
                }
                x[i*n + k] = b[i*n+k] - sum;
                x[i*n + k] /= U[i*n + i];
                
            }
        }

}

void backward_substitution_LU(double * U, double *x, double * b, int n){
 // ikj

    int j;
    int k;
    for(int i=n-1; i>=0; i--){
        for(k=0; k<n-3; k+=4){
            __m256d sum0 = _mm256_setzero_pd();
            __m256d sum1 = _mm256_setzero_pd();
            __m256d sum2 = _mm256_setzero_pd();
            __m256d sum3 = _mm256_setzero_pd();
            __m256d sum_cleanup = _mm256_setzero_pd();

            //__m256d res_vec= _mm256_loadu_pd( &b[i*n+k] );
            __m256d res_vec= _mm256_load_pd( &b[i*n+k] );

            //__m256d denominator_vec = _mm256_broadcast_sd(&U[i*n+i]);
            //__m256d one_vec = _mm256_set1_pd(1.0);
            //__m256d rezi_vec = _mm256_div_pd(one_vec, denominator_vec);

            for(j=i+1; j<n-3; j+=4){
                //__m256d x_vec0 = _mm256_loadu_pd( &x[j*n+k] );
                //__m256d x_vec1 = _mm256_loadu_pd( &x[(j+1)*n+k] );
                //__m256d x_vec2 = _mm256_loadu_pd( &x[(j+2)*n+k] );
                //__m256d x_vec3 = _mm256_loadu_pd( &x[(j+3)*n+k] );

                __m256d x_vec0 = _mm256_load_pd( &x[j*n+k] );
                __m256d x_vec1 = _mm256_load_pd( &x[(j+1)*n+k] );
                __m256d x_vec2 = _mm256_load_pd( &x[(j+2)*n+k] );
                __m256d x_vec3 = _mm256_load_pd( &x[(j+3)*n+k] );

                __m256d U_vec0 = _mm256_broadcast_sd(&U[i*n+j]);
                __m256d U_vec1 = _mm256_broadcast_sd(&U[i*n+j +1]);
                __m256d U_vec2 = _mm256_broadcast_sd(&U[i*n+j +2]);
                __m256d U_vec3 = _mm256_broadcast_sd(&U[i*n+j +3]);

                sum0 = _mm256_fmadd_pd(U_vec0, x_vec0, sum0);
                sum1 = _mm256_fmadd_pd(U_vec1, x_vec1, sum1);
                sum2 = _mm256_fmadd_pd(U_vec2, x_vec2, sum2);
                sum3 = _mm256_fmadd_pd(U_vec3, x_vec3, sum3);
                
            }
            //CLEANUP LOOP FOR J
            for(; j<n; j++){
           // __m256d x_vec_cleanup= _mm256_loadu_pd( &x[j*n+k] );
            __m256d x_vec_cleanup= _mm256_load_pd( &x[j*n+k] );

            __m256d U_vec_cleanup = _mm256_broadcast_sd(&U[i*n+j]);

            sum_cleanup = _mm256_fmadd_pd(U_vec_cleanup, x_vec_cleanup, sum_cleanup);
            

            }

            __m256d tmp_sum0 = _mm256_add_pd(sum0, sum1);
            __m256d tmp_sum1 = _mm256_add_pd(sum2, sum3);
            __m256d final_sum = _mm256_add_pd(tmp_sum0, tmp_sum1);

            res_vec = _mm256_sub_pd(res_vec, sum_cleanup);
            res_vec = _mm256_sub_pd(res_vec, final_sum);

            

            //res_vec = _mm256_mul_pd(res_vec, rezi_vec);

            //_mm256_storeu_pd(&x[i*n+k], res_vec);
            _mm256_store_pd(&x[i*n+k], res_vec);
        }
        
        //CLEANUP LOOP FOR K
        for(; k<n; k++){
            double sum = 0;
                for(j=i+1; j<n; j++){
                    sum += U[i*n+j] * x[j*n+k];
                    ;
                }
                x[i*n + k] = b[i*n+k] - sum;
                
                //x[i*n + k] /= U[i*n + i];
            }
        }


}

void LU_decomposition(double *LU, double *P, int n ){

 

    double curr_piv;
    int index_piv;
    double tmp;

    //copy_matrix(org_A, LU, n, n);

   
    int *tmp_P = (int*) aligned_alloc(32, n*sizeof(int));
    for(int i=0; i<n; i++){
        tmp_P[i] = i;
    }


    for(int k=0; k<n; k++){
        // find pivot
        //FLOP_COUNT_INC(1, "LU_vec");
        curr_piv = fabs(LU[k*n+k]);
       
        index_piv = k;
        //FLOP_COUNT_INC(2*(n-k), "LU_solve2");

        for(int i=k+1; i<n; i++){ 
            double abs = fabs(LU[k*n+i]);
            if( abs > curr_piv ){
                curr_piv = abs;
                index_piv = i;
            }
            
        }

        if(index_piv != k){
        //swap rows to get pivot-row on top
        //printf("\n\n swapped rows %d and %d \n\n", index_piv, k);
            for(int x=0; x<n; x++){
                tmp = LU[k*n + x];
                LU[k*n +x] = LU[index_piv*n+x];
                LU[index_piv*n+x] = tmp;
                
            }
            
            tmp = tmp_P[k];
            tmp_P[k] = tmp_P[index_piv];
            tmp_P[index_piv] = tmp;

        }

        /*if(LU[k*n+k] == 0){
            printf("non invertible, try again \n");
        }*/

        double rezi = 1.0 / LU[k*n+k];
        

        __m256d rezi_vec = _mm256_set1_pd(rezi);


       int i;
        for(i=1; i<(n-k)-3; i+=4){
            __m256d vec1 = _mm256_loadu_pd(&LU[k*n+ k +i]);
            //__m256d vec1 = _mm256_load_pd(&LU[k*n+ k +i]);

            __m256d res_vec = _mm256_mul_pd(vec1, rezi_vec);
            
            _mm256_storeu_pd(&LU[k*n+k+i], res_vec);
            //_mm256_store_pd(&LU[k*n+k+i], res_vec);

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
                //__m256d curr = _mm256_load_pd(&LU[(k+i+1)*n+ (k+j+1)]);
                //__m256d a21_vec = _mm256_load_pd(&LU[k*n+ k +j +1]);
                __m256d a12_vec = _mm256_broadcast_sd(&LU[(k+i+1)*n+k]);

                __m256d prod = _mm256_mul_pd(a12_vec, a21_vec);

                __m256d res = _mm256_sub_pd(curr, prod);
                
                _mm256_storeu_pd(&LU[(k+i+1)*n+ (k+j+1)], res);          
                //_mm256_store_pd(&LU[(k+i+1)*n+ (k+j+1)], res);            
            }
            //CLEANUP LOOP FOR J
            for(    ; j<(n-k-1); j++){
                LU[(k+i+1)*n + (k+j+1)] -= LU[(k+i+1)*n+k] * LU[k*n+k+1+j];//outer_prod[(i) * (n-k-1) + (j)];
                
            }
        }

    }

    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            P[i*n+j] = 0;
        }
    }

    for(int i=0; i<n; i++){
        P[i*n + tmp_P[i]] = 1.0;
        //P[tmp_P[i]*n+i] = 1.0;
    }


    free(tmp_P);

}
/*----- onenorm functions template ----- */

/* ----- global constants ----- */



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
            
            
            cblas_dgemm(layout, CblasNoTrans, CblasNoTrans, n, t, n ,alpha ,A, n ,X ,n ,beta, Y, n);
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
        
        
        cblas_dgemm(layout, CblasTrans, CblasNoTrans, n, t, n, alpha, A, n, S, n, beta, Z, n);

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
    

    return est;
}
/* ---- eval functions template ---- */

/**
 * @brief Evaluates equation 3.4
 * 
 * @param A The input matrix A with dimension n x n
 * @param n The number of rows and columns of A
 * @param m The padé approximant
 * @param P_m The output matrix p_m(A) with dimension n x n
 * @param Q_m The output matrix q_m(A) with dimension n x n
 */
void eval3_4(const double* A, const double* A_2, const double* A_4, const double* A_6, int n, const int m, double *P_m, double *Q_m)
{
    // FLOP_COUNT_INC(0, "eval3_4"); Note: all flops are already accounted for in the called methods
    double *U = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *V = (double*) aligned_alloc(32, n*n*sizeof(double));

    fill_diagonal_matrix(V, pade_coefs[0], n);
    fill_diagonal_matrix(U, pade_coefs[1], n); 

    mm_add(pade_coefs[2], A_2, V, V, n, n);
    mm_add(pade_coefs[3], A_2, U, U, n, n);

    if(m >= 5){
        mm_add(pade_coefs[4], A_4, V, V, n, n);
        mm_add(pade_coefs[5], A_4, U, U, n, n);
    }
    if(m >= 7){
        mm_add(pade_coefs[6], A_6, V, V, n, n);
        mm_add(pade_coefs[7], A_6, U, U, n, n);
    }
    if(m >= 9){
        double *A_8 = (double*) aligned_alloc(32, n*n*sizeof(double));
        mmm(A_4, A_4, A_8, n, n, n);
        mm_add(pade_coefs[8], A_8, V, V, n, n);
        mm_add(pade_coefs[9], A_8, U, U, n, n);
        free(A_8);
    }

    mmm(A, U, P_m, n, n, n);
    scalar_matrix_mult(-1.0, P_m, Q_m, n, n); // (-A)*U == -(A*U)

    mm_add(1.0, P_m, V, P_m, n, n);
    mm_add(1.0, Q_m, V, Q_m, n, n);

    free(U);
    free(V);
}

/**
 * @brief Evaluates equation 3.5
 * 
 * @param A The input matrix A with dimension n x n
 * @param A_2 precomputed matrix A^2
 * @param A_4 precomputed matrix A^4
 * @param A_6 precomputed matrix A^6
 * @param n The number of rows and columns of A, A_2, A_4 and A_6
 * @param P_13 The output matrix p_13(A) with dimension n x n
 * @param Q_13 The output matrix q_13(A) with dimension n x n
 */
void eval3_5(const double *A, double* A_2, double* A_4, double* A_6, int n, double *P_13, double *Q_13){
    // FLOP_COUNT_INC(0, "eval3_5"); Note: all flops are already accounted for in the called methods

    double *A_tmp = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *A_tmp2 = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *I_tmp = (double*) aligned_alloc(32, n*n*sizeof(double));
    fill_diagonal_matrix(I_tmp, 1.0, n); 

    // computing u_13
    scalar_matrix_mult(pade_coefs[13], A_6, A_tmp, n, n); // b_13 * A_6
    mm_add(pade_coefs[11], A_4, A_tmp, A_tmp, n, n); // b_11 * A_4
    mm_add(pade_coefs[9], A_2, A_tmp, A_tmp, n, n); // b_9 * A_2

    mmm(A_6, A_tmp, A_tmp2, n, n, n);

    mm_add(pade_coefs[7], A_6, A_tmp2, A_tmp, n, n); // b_7 * A_6
    mm_add(pade_coefs[5], A_4, A_tmp, A_tmp, n, n); // b_5 * A_4
    mm_add(pade_coefs[3], A_2, A_tmp, A_tmp, n, n); // b_3 * A_2
    mm_add(pade_coefs[1], I_tmp, A_tmp, A_tmp, n, n); // b_1 * I

    mmm(A, A_tmp, P_13, n, n, n);
    scalar_matrix_mult(-1.0, P_13, Q_13, n, n);
    // now P_13 = u_13 and Q_13 = -u_13


    // computing v_13
    scalar_matrix_mult(pade_coefs[12], A_6, A_tmp, n, n); // b_12 * A_6
    mm_add(pade_coefs[10], A_4, A_tmp, A_tmp, n, n); // b_10 * A_4
    mm_add(pade_coefs[8], A_2, A_tmp, A_tmp, n, n); // b_8 * A_2

    mmm(A_6, A_tmp, A_tmp2, n, n, n);

    mm_add(pade_coefs[6], A_6, A_tmp2, A_tmp, n, n); // b_6 * A_6
    mm_add(pade_coefs[4], A_4, A_tmp, A_tmp, n, n); // b_4 * A_4
    mm_add(pade_coefs[2], A_2, A_tmp, A_tmp, n, n); // b_2 * A_2
    mm_add(pade_coefs[0], I_tmp, A_tmp, A_tmp, n, n); // b_0 * I
    // now A_tmp = v_13

    mm_add(1, P_13, A_tmp, P_13, n, n);
    mm_add(1, Q_13, A_tmp, Q_13, n, n);
    free(A_tmp);
    free(A_tmp2);
    free(I_tmp);
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

    forward_substitution(Q_m, Y , P_permuted, n);
    backward_substitution_LU(Q_m, R_m, Y, n);

    free(Y);
    free(P);
    free(P_permuted);

    }
    
}
/* ---- matrix exponential  template ----- */

int ell(const double* A_pow, double normA, int n, int m){ 
    
    double abs_norm = normest(A_pow, n);
    if(abs_norm <= 0.0){
        return 0;
    }
    double alpha = (coeffs[m] * abs_norm) / normA;
    return fmax((int)ceil(log2(alpha/ (*unit_roundoff))/((double)(2*m))), 0);
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
    
    double* A_abs_pow_m = (double*) aligned_alloc(32, n*n*sizeof(double));
    double* A_abs_pow_m_temp = (double*) aligned_alloc(32, n*n*sizeof(double));
    double* abs_pow_swap;

    mat_abs(A, A_abs, n, n); 
    double normA = onenorm_abs_mat(A_abs,n,n);
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
        
        double d_6 = pow(normest(A_6, n), 1.0/6.0); // onenormest(A_2, 3)
        double eta_1 = fmax(pow(normest(A_4, n), 1.0/4.0), d_6); // onenormest(A_2, 2)

        mmm(A_abs, A_abs_6, A_abs_pow_m, n,n,n); //|A|^(2*3+1) = |A|^7
        if(eta_1 <= theta[3] && ell(A_abs_pow_m, normA, n, 3) == 0){
            
            eval3_4(A, A_2, A_4, A_6, n, 3, P_m, Q_m);
            eval3_4(A_abs, A_abs_2, A_abs_4, A_abs_6, n, 3, P_m_abs, Q_m_abs);
            mat_col_sum(P_m_abs, n, Temp);

            double divider = fmin(onenorm(P_m, n, n), onenorm(Q_m, n, n));
            if(infinity_norm(Temp, n, 1)/divider <= 10*exp(theta[3])){
                if(DEBUG) printf("returned m = 3\n");
                eval3_6(P_m, Q_m, n, E, triangular_indicator);
                break;
            }
        }

        // ======================== p = 5 =========================
        
        
        double d_4 = pow(onenorm(A_4, n, n), 1.0/4.0);
        double eta_2 = fmax(d_4, d_6);

        abs_pow_swap = A_abs_pow_m;
        A_abs_pow_m = A_abs_pow_m_temp;
        A_abs_pow_m_temp = abs_pow_swap;

        mmm(A_abs_4, A_abs_pow_m_temp, A_abs_pow_m, n,n,n); //|A|^11
        if(eta_2 <= theta[5] && ell(A_abs_pow_m, normA, n, 5) == 0){
            

            eval3_4(A, A_2, A_4, A_6, n, 5, P_m, Q_m);
            eval3_4(A_abs, A_abs_2, A_abs_4, A_abs_6, n, 5, P_m_abs, Q_m_abs);
            mat_col_sum(P_m_abs, n, Temp);

            double divider = fmin(onenorm(P_m, n, n), onenorm(Q_m, n, n));
            if(infinity_norm(Temp, n, 1)/divider <= 10*exp(theta[5])){
                eval3_6(P_m, Q_m, n, E, triangular_indicator);
                if(DEBUG) printf("returned m = 5\n");
                break;
            }
        }

        // ======================== p = 7, 9 ========================

        mmm(A_4, A_4, A_8, n, n, n); // A_4^2

        
        d_6 = pow(onenorm(A_6, n, n), 1.0/6.0);
        double d_8 = pow(normest(A_8, n), 1.0/8.0); //onenormest(A_4, 2)
        double eta_3 = fmax(d_6, d_8);
        
        int found = 0;
        for(int m = 7; m <= 9; m+=2){
            abs_pow_swap = A_abs_pow_m;
            A_abs_pow_m = A_abs_pow_m_temp;
            A_abs_pow_m_temp = abs_pow_swap;

            mmm(A_abs_4, A_abs_pow_m_temp, A_abs_pow_m, n,n,n); //|A|^15, |A|^19
            if(eta_3 <= theta[m] && ell(A_abs_pow_m, normA, n, m) == 0){
                
                eval3_4(A, A_2, A_4, A_6, n, m, P_m, Q_m);
                eval3_4(A_abs, A_abs_2, A_abs_4, A_abs_6, n, m, P_m_abs, Q_m_abs);
                mat_col_sum(P_m_abs, n, Temp);

                double divider = fmin(onenorm(P_m, n, n), onenorm(Q_m, n, n));
                if(infinity_norm(Temp, n, 1)/divider <= 10*exp(theta[m])){
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

        int s = (int) fmax(ceil(log2(eta_5/theta[13])), 0.0);
        double* A_temp = (double*) aligned_alloc(32, n*n*sizeof(double));
        scalar_matrix_mult(pow(2.0, -s), A, A_temp, n, n); // 2^-s * A
        
        /*-- ell --*/
        double* A_temp_abs = (double*) aligned_alloc(32, n*n*sizeof(double));
        double* A_temp_abs_pow = (double*) aligned_alloc(32, n*n*sizeof(double));
        mat_abs(A_temp, A_temp_abs,n,n);
        matpow_by_squaring(A_temp_abs, n, 27, A_temp_abs_pow); 
        
        s = s + ell(A_temp_abs_pow, onenorm_abs_mat(A_temp_abs,n,n), n, 13);
        
        free(A_temp_abs);
        free(A_temp_abs_pow);
        /*-- ell done -- */

        scalar_matrix_mult(pow(2.0, -s), A, A_temp, n, n);
        scalar_matrix_mult(pow(2.0, -2.0*s), A_2, A_2, n, n);
        scalar_matrix_mult(pow(2.0, -4.0*s), A_4, A_4, n, n);
        scalar_matrix_mult(pow(2.0, -6.0*s), A_6, A_6, n, n);

        eval3_5(A_temp, A_2, A_4, A_6, n, P_m, Q_m);
        eval3_5(A_abs, A_abs_2, A_abs_4, A_abs_6, n, P_m_abs, Q_m_abs);
        mat_col_sum(P_m_abs, n, Temp);
        double divider = fmin(onenorm(P_m, n, n), onenorm(Q_m, n, n));
        int s_max = (int)ceil(log2(onenorm(A,n,n) / theta[13]));
        if(infinity_norm(Temp, n, 1)/divider <= (10+s_max)*exp(theta[13])){
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
            //FLOP_COUNT_INC(2*n + 2*n*s, "mat_exp triangular");

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
            if(DEBUG) assert(s > 0);
            matpow_by_squaring(R_m, n, (long) pow(2, s), E); // TODO: FLOP_COUNT_INC: 1 flop?
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
    free(A_abs_pow_m);
    free(A_abs_pow_m_temp);
    return;
    
}
