#include "matrix_operations.h"
// For vectorization
#include <immintrin.h>


#define USEDGEMM 0

// reference: https://www.ibm.com/docs/en/zos/2.4.0?topic=uatlasal-examples-compiling-linking-running-simple-matrix-multiplication-atlas-program
//TODO USEDGEMM as compile flag
void mmm(const double *A, const double *B, double *C, int n, int m, int t){
    FLOP_COUNT_INC(2*n*m*t, "mmm");
    if(USEDGEMM){
        const double alpha = 1.0;
        const double beta = 0.0;
        CBLAS_LAYOUT layout = CblasRowMajor;
        CBLAS_TRANSPOSE transA = CblasNoTrans;
        CBLAS_TRANSPOSE transB = CblasNoTrans;
        cblas_dgemm(layout, transA, transB, m, t, n, alpha, A, n, B, t, beta, C, t);
    }else{
        for(int i = 0; i < m; i++){
            for(int j = 0; j < t; j++){
                C[i * t + j] = 0.0;
                for(int k = 0; k < n; k++){
                    C[i * t + j] += A[i * n + k] * B[k * t + j];
                }
            }
        }
    }
}

void copy_matrix(const double* A, double* B, int m, int n){
    // FLOP_COUNT_INC(0, "copy_matrix");
    memcpy(B, A, m*n*sizeof(double));
}


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

    double *curr_P = (double*) malloc(n*n*sizeof(double));
    double *tmp_P = (double*) malloc(n*n*sizeof(double));
    double *cpy_A = (double*) malloc(n*n*sizeof(double));
    double *tmp_A = (double*) malloc(n*n*sizeof(double));
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

void mm_add(double alpha, const double *A, const double *B, double *C, int m, int n){
    if(alpha == 0 || alpha == 1.0){
        FLOP_COUNT_INC(n*m, "mm_add");
    }else{
        FLOP_COUNT_INC(2*n*m, "mm_add");
    }
    
    for(int i = 0; i < m*n; i++){
        C[i] = alpha * A[i] + B[i];
    }
}

void scalar_matrix_mult(double alpha, const double *A, double *C, int m, int n){
    FLOP_COUNT_INC(n*m, "scalar_matrix_mult");
    for(int i = 0; i < m*n; i++){
        C[i] = alpha * A[i];
    }
}

void mat_abs(const double* A, double* B, int m, int n){
    FLOP_COUNT_INC(n*m, "mat_abs");
    for(int i = 0; i < m*n; i++){
        B[i] = fabs(A[i]);
    }
}


int is_lower_triangular(const double *A, int m, int n){
    for(int i = 0; i < m; i++){
        for(int j = i + 1; j < n; j++){
            FLOP_COUNT_INC(1, "is_lower_triangular");
            if(A[i * n + j] != 0.0) return 0;
        }
    }
    return 1;
}

int is_upper_triangular(const double *A, int m, int n){
    for(int i = 1; i < m; i++){
        for(int j = 0; j < i; j++){
            FLOP_COUNT_INC(1, "is_upper_triangular");
            if(A[i * n + j] != 0.0) return 0;
        }
    }
    return 1;
}

int is_sparse(const double * A, int m, int n){
    int count = 0;
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            if(A[i*n+j] != 0){
                count++;
            }
        }
    }

    
    return (count < m * sqrt(n));
}

int is_triangular(const double *A, int m, int n){
    // FLOP_COUNT_INC(0, "is_triangular");
   if(is_upper_triangular(A, m, n)){
    return 1;
   }else if(is_lower_triangular(A, m, n)){
    return 2;
   }else{
    return 0;
   }
}

void transpose(const double *A, double *B, int m, int n){
    // FLOP_COUNT_INC(0, "transpose");
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            B[j * m + i] = A[i * n + j];
        }
    }
}

double dot_product(const double *v, const double *w, int n){
    FLOP_COUNT_INC(2*n, "dot_product");
    double dp = 0.0;
    for(int i = 0; i < n; i++){
        dp += v[i] * w[i];
    }
    return dp;
}


void printmatrix(const double* M, int m, int n){
    // FLOP_COUNT_INC(0, "printmatrix");
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            printf("%f, ", M[i*n + j]);
        }
        printf("\n");
    }
}


double infinity_norm(const double* A, int m, int n){
    FLOP_COUNT_INC(2*n*m+m, "infinity_norm");
    double max_row_sum = 0.0;

    for(int i = 0; i < m; i++){
        double curr_row_sum = 0.0;
        for(int j = 0; j < n; j++){
            curr_row_sum += fabs(A[i * n + j]);
        }
        if(curr_row_sum > max_row_sum){
            max_row_sum = curr_row_sum;
        }
    }
    return max_row_sum;
}

void fill_diagonal_matrix(double* A, double diag, int n){
    // FLOP_COUNT_INC(0, "fill_diagonal_matrix");
    for(int i = 0; i < n*n; i++){
        A[i] = 0.0;
    }
    for(int i = 0; i < n; i++){
        A[i * n + i] = diag;
    }
}

void mat_col_sum(const double* A, int n, double* out){
    FLOP_COUNT_INC(n*n, "mat_col_sum");
    for(int j = 0; j < n; j++){
        double curr_sum = 0.0;
        for(int i = 0; i < n; i++){
            curr_sum += A[i * n + j];
        }
        out[j] = curr_sum;
    }
}


void forward_substitution_matrix(double * A, double *y, double * b, int n){
    for(int col=0; col<n; col++){
        for(int i=0; i<n; i++){
            y[i*n + col] = b[i*n + col] / A[i*n+i];
            FLOP_COUNT_INC(1,"forward_substitution_matrix");
            for(int j=0; j<i; j++){
                FLOP_COUNT_INC(2, "forward_substitution_matrix");
                y[i*n+col] -= (A[i*n+j] * y[j*n + col]) / A[i*n+i];
                FLOP_COUNT_INC(3,"forward_substitution_matrix");
            }
            
            
        }
    }
}

void forward_substitution_LU(double * A, double *y, double * b, int n){
    for(int col=0; col<n; col++){
        for(int i=0; i<n; i++){
            y[i*n + col] = b[i*n + col];// / A[i*n+i];
            for(int j=0; j<i; j++){
                y[i*n+col] -=  (A[i*n+j] * y[j*n + col]);// / A[i*n+i];
                FLOP_COUNT_INC(2,"forward_substitution_matrix");
            }
            //y[i*n+col] = y[i*n + col] ;/// A[i*n+i];
        }
    }
}
/*
void forward_substitution_matrix_vectorized(double * L, double *y, double * b, int n){
   for(int col=0; col<n; col+=4){

    for(int i=0; i<n; i++){

        __m256d res = _mm256_loadu_pd( &b[i*n+col] );
        __m256d divisor = _mm256_set1_pd( L[i*n+i] );

        for(int j=0; j<i; j++){
             __m256d L_vec = _mm256_set1_pd( L[i*n+j] );
             __m256d y_vec = _mm256_loadu_pd( &y[j*n+col] );
             __m256d prod = _mm256_mul_pd(L_vec, y_vec);
             res = _mm256_sub_pd(res, prod);


        }
        
            res = _mm256_div_pd(res, divisor);
            _mm256_storeu_pd(&y[i*n+col], res);

    }

}
}*/

void backward_substitution_matrix(double * U, double *x, double * y, int n){
    for(int col=0; col<n; col++){
        for(int i=n-1; i>=0; i--){
            FLOP_COUNT_INC(1, "backward_substitution_matrix");
            x[i*n + col] = y[i*n + col] / U[i*n+i];
            for(int j=i+1; j<n; j++){
                FLOP_COUNT_INC(3, "backward_substitution_matrix");
                x[i*n + col] -= (U[i*n+j] * x[j*n+col]) / U[i*n+i];
            }
            //x[i*n + col] = x[i*n + col] / U[i*n + i];
        }

    }
}

/*
void backward_substitution_matrix_vectorized(double * U, double *x, double * y, int n){
    for(int col=0; col<n; col+=4){

        for(int i=n-1; i>=0; i--){

            __m256d res = _mm256_loadu_pd( &y[i*n+col] );
            __m256d divisor = _mm256_set1_pd( U[i*n+i] );

            for(int j=i+1; j<n; j++){
                __m256d U_vec = _mm256_set1_pd( U[i*n+j] );
                __m256d x_vec = _mm256_loadu_pd( &x[j*n+col] );
                __m256d prod = _mm256_mul_pd(U_vec, x_vec);
                res = _mm256_sub_pd(res, prod);
            }

            res = _mm256_div_pd(res, divisor);
            _mm256_storeu_pd(&x[i*n+col], res);

        }

    }
}
*/

/*
void part_piv_lu(double * org_A, double *L, double *U, double *P, int n){
    // FLOP_COUNT_INC: has been added directly at the operations for simplicity
    double curr_piv;
    int index_piv;
    double tmp;

    double *A = (double*) malloc(n*n*sizeof(double));

    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            A[i*n+j] = org_A[i*n+j];
            if(i==j){
                L[i*n+j] = 1;
                U[i*n+j] = 1;
            }else{
                L[i*n+j] = 0;
                U[i*n+j] = 0;
            }
        }
    }

    int *tmp_P = (int*) malloc(n*sizeof(int));
    for(int i=0; i<n; i++){
        tmp_P[i] = i;
    }


    for(int k=0; k<n; k++){
        // find pivot
        curr_piv = A[k*n+k];
        index_piv = k;
        
        for(int i=k+1; i<n; i++){ // or just j????  
            FLOP_COUNT_INC(3, "part_piv_lu");
            if( fabs(A[i*n+k]) > fabs(curr_piv) ){
                curr_piv = A[i*n+k];
                index_piv = i;
            }
        }

        if(index_piv != k){
        //swap rows to get pivot-row on top
            for(int x=k; x<n; x++){
                tmp = A[k*n + x];
                A[k*n +x] = A[index_piv*n+x];
                A[index_piv*n+x] = tmp;
            }

            
            for(int y=0; y<k; y++){
                tmp = L[k*n+y];
                L[k*n+y] = L[index_piv*n+y];
                L[index_piv*n+y] = tmp;
            }

            tmp = tmp_P[k];
            tmp_P[k] = tmp_P[index_piv];
            tmp_P[index_piv] = tmp;

        }

    
        for(int j=k; j<n; j++){
            U[k*n+j] = A[k*n+j];

            FLOP_COUNT_INC(2*k, "part_piv_lu");
            for(int s=0; s<k; s++){
                U[k*n+j] -= L[k*n+s] * U[s*n+j]; 
            }
        }

    
        for(int i=k+1; i<n; i++){
            L[i*n+k] = A[i*n+k];// / U[k*n+k];
            FLOP_COUNT_INC(2*k, "part_piv_lu");
            for(int s=0; s<k; s++){
                L[i*n+k] -= (L[i*n+s]*U[s*n+k]);// / U[k*n+k];
            }
            FLOP_COUNT_INC(1, "part_piv_lu");
            L[i*n+k] /= U[k*n+k];
        }

    }

    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            P[i*n+j] = 0;
        }
    }

    for(int i=0; i<n; i++){
        P[i*n + tmp_P[i]] = 1.0;
    }
    
    free(tmp_P);
    free(A);
}
*/

void LU_decomposition(double *A, double *P, int n){

    double curr_piv;
    int index_piv;
    double tmp;

    //copy_matrix(org_A, LU, n, n);

   
    int *tmp_P = (int*) malloc( n*sizeof(int));
    for(int i=0; i<n; i++){
        tmp_P[i] = i;
    }


    for(int k=0; k<n; k++){
        // find pivot
        curr_piv = fabs(A[k*n+k]);
        FLOP_COUNT_INC(1, "LU_decomposition");
        index_piv = k;

        for(int i=k+1; i<n; i++){ 
            double abs = fabs(A[k*n+i]);
            FLOP_COUNT_INC(2, "LU_decomposition");
            if( abs > curr_piv ){
                curr_piv = abs;
                index_piv = i;
            }
           
        }

        if(index_piv != k){
        //swap rows to get pivot-row on top
            for(int x=0; x<n; x++){
                tmp = A[k*n + x];
                A[k*n +x] = A[index_piv*n+x];
                A[index_piv*n+x] = tmp;
                
            }
            
            tmp = tmp_P[k];
            tmp_P[k] = tmp_P[index_piv];
            tmp_P[index_piv] = tmp;

        }


        for(int i=1; i<(n-k); i++){
            A[(k+i)*n+k] = A[(k+i)*n+ k] / A[k*n+k]; 
            FLOP_COUNT_INC(1, "LU_decomposition");
        }


        for(int i=0; i<(n-k-1); i++){
            for( int j=0   ; j<(n-k-1); j++){
                A[(k+j+1)*n + (k+i+1)] -= A[(k+j+1)*n+k] * A[k*n+k+1+i];//outer_prod[(i) * (n-k-1) + (j)];
                FLOP_COUNT_INC(2, "LU_decomposition");
                
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

void LU_solve1(double * Q_m, double * R_m, double *P_m, int n){
    double *L = (double*) malloc(n*n*sizeof(double));
    double *U = (double*) malloc(n*n*sizeof(double));
    double *P = (double*) malloc(n*n*sizeof(double));
    double *Y = (double*) malloc(n*n*sizeof(double));
    double *A = (double*) malloc(n*n*sizeof(double));
    int *tmp_P = (int*) malloc(n*sizeof(int));
    double *permuted_P_m = (double*) malloc(n*n*sizeof(double));

    double curr_piv;
    int index_piv;
    double tmp;


    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            A[i*n+j] = Q_m[i*n+j];
            if(i==j){
                L[i*n+j] = 1;
                U[i*n+j] = 1;
            }else{
                L[i*n+j] = 0;
                U[i*n+j] = 0;
            }
        }
    }

   
    for(int i=0; i<n; i++){
        tmp_P[i] = i;
    }


    for(int k=0; k<n; k++){
        // find pivot
        curr_piv = A[k*n+k];
        index_piv = k;

        for(int i=k+1; i<n; i++){ // or just j????  
            FLOP_COUNT_INC(3, "LU_solve1");
            if( fabs(A[i*n+k]) > fabs(curr_piv) ){
                curr_piv = A[i*n+k];
                index_piv = i;
            }
        }

        if(index_piv != k){
        //swap rows to get pivot-row on top
            for(int x=k; x<n; x++){
                tmp = A[k*n + x];
                A[k*n +x] = A[index_piv*n+x];
                A[index_piv*n+x] = tmp;
            }

            
            for(int y=0; y<k; y++){
                tmp = L[k*n+y];
                L[k*n+y] = L[index_piv*n+y];
                L[index_piv*n+y] = tmp;
            }

            tmp = tmp_P[k];
            tmp_P[k] = tmp_P[index_piv];
            tmp_P[index_piv] = tmp;

        }

    
        for(int j=k; j<n; j++){
            U[k*n+j] = A[k*n+j];

            FLOP_COUNT_INC(2*k, "LU_solve1");
            for(int s=0; s<k; s++){
                U[k*n+j] -= L[k*n+s] * U[s*n+j]; 
            }
        }

    
        for(int i=k+1; i<n; i++){
            L[i*n+k] = A[i*n+k];// / U[k*n+k];
            FLOP_COUNT_INC(2*k, "LU_solve1");
            for(int s=0; s<k; s++){
                L[i*n+k] -= (L[i*n+s]*U[s*n+k]);// / U[k*n+k];
            }
            FLOP_COUNT_INC(1, "LU_solve1");
            L[i*n+k] /= U[k*n+k];
        }

    }

    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            P[i*n+j] = 0;
        }
    }

    for(int i=0; i<n; i++){
        P[i*n + tmp_P[i]] = 1.0;
    }

    //printf("L org: \n");
    //printmatrix(L, n, n);
    //printf("U org: \n");
    //printmatrix(U, n, n);
    
    
    mmm(P, P_m, permuted_P_m, n, n, n);


    for(int col=0; col<n; col++){
        for(int i=0; i<n; i++){
            FLOP_COUNT_INC(1, "LU_solve1");
            Y[i*n + col] = permuted_P_m[i*n + col] / L[i*n+i];
            for(int j=0; j<i; j++){
                FLOP_COUNT_INC(3, "LU_solve1");
                Y[i*n+col] -= (L[i*n+j] * Y[j*n + col]) / L[i*n+i];
            }
        }
    }


    for(int col=0; col<n; col++){
        for(int i=n-1; i>=0; i--){
            FLOP_COUNT_INC(1, "LU_solve1");
            R_m[i*n + col] = Y[i*n + col] / U[i*n+i];
            for(int j=i+1; j<n; j++){
                FLOP_COUNT_INC(3, "LU_solve1");
                R_m[i*n + col] -= (U[i*n+j] * R_m[j*n+col]) / U[i*n+i];
            }
        }
    }


    free(tmp_P);
    free(A);
    free(L);
    free(U);
    free(P);
    free(Y);
    free(permuted_P_m);
}
/*
void LU_solve2(double * org_A, double * R_m, double *P_m, int n){
   // FLOP_COUNT_INC: has been added directly at the operations for simplicity
    double curr_piv;
    int index_piv;
    double tmp;

    double *P = (double*) malloc(n*n*sizeof(double));
    double *LU = (double*) malloc(n*n*sizeof(double));

    copy_matrix(org_A, LU, n, n);

    int *tmp_P = (int*) malloc(n*sizeof(int));
    for(int i=0; i<n; i++){
        tmp_P[i] = i;
    }


    for(int k=0; k<n; k++){
        // find pivot
        curr_piv = LU[k*n+k];
        index_piv = k;
       
        for(int i=k+1; i<n; i++){
            FLOP_COUNT_INC(3, "LU_solve2");
            if( fabs(LU[i*n+k]) > fabs(curr_piv) ){
                curr_piv = LU[i*n+k];
                index_piv = i;
            }
        }

        if(index_piv != k){
        //swap rows to get pivot-row on top
            for(int x=0; x<n; x++){
                tmp = LU[k*n + x];
                LU[k*n +x] = LU[index_piv*n+x];
                LU[index_piv*n+x] = tmp;
            }


            tmp = tmp_P[k];
            tmp_P[k] = tmp_P[index_piv];
            tmp_P[index_piv] = tmp;

        }

    
        for(int j=k; j<n; j++){
            for(int s=0; s<k; s++){
                FLOP_COUNT_INC(2, "LU_solve2");
                LU[k*n+j] -= LU[k*n+s] * LU[s*n+j];
            }
        }

    
        for(int i=k+1; i<n; i++){
            for(int s=0; s<k; s++){
                FLOP_COUNT_INC(2, "LU_solve2");
                LU[i*n+k] -= LU[i*n+s] * LU[s*n+k];
            }
            FLOP_COUNT_INC(1, "LU_solve2");
            LU[i*n+k] /= LU[k*n+k];
        }

    }

    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            P[i*n+j] = 0;
        }
    }

    for(int i=0; i<n; i++){
        P[i*n + tmp_P[i]] = 1.0;
    }


    double *Y = (double*) malloc(n*n*sizeof(double));
    double *permuted_P_m = (double*) malloc(n*n*sizeof(double));

    //Permute P_m
    //mmm(P_m, P, permuted_P_m, n, n, n);
    mmm(P, P_m, permuted_P_m, n, n, n);

    //Forwards substituition
    

    for(int col=0; col<n; col++){
        for(int i=0; i<n; i++){
            Y[i*n + col] = permuted_P_m[i*n + col];
            for(int j=0; j<i; j++){
                FLOP_COUNT_INC(2, "LU_solve2");
                Y[i*n+col] -=  LU[i*n+j] * Y[j*n + col];
            }
        }
    }

    //Backward substitution
    FLOP_COUNT_INC(n*n + n*(n-1)*n/2 * 3, "LU_solve2");

    for(int col=0; col<n; col++){
        for(int i=n-1; i>=0; i--){
            FLOP_COUNT_INC(1, "LU_solve2");
            R_m[i*n + col] = Y[i*n + col] / LU[i*n+i];
            for(int j=i+1; j<n; j++){
                R_m[i*n + col] -= (LU[i*n+j] * R_m[j*n+col]) / LU[i*n+i];
            }
        }

    }

    free(Y);
    free(permuted_P_m);
    free(P);
    free(LU);
    free(tmp_P);
}


void LU_solve2_vec1(double * org_A, double * R_m, double *P_m, int n){
    // FLOP_COUNT_INC: has been added directly at the operations for simplicity
    double curr_piv;
    int index_piv;
    double tmp;
    
    double* P = (double*)aligned_alloc(32, n*n*sizeof(double));
    double* LU = (double*)aligned_alloc(32, n*n*sizeof(double));
    double* Y = (double*)aligned_alloc(32, n*n*sizeof(double));
    double* permuted_P_m = (double*)aligned_alloc(32, n*n*sizeof(double));
    int *tmp_P = (int*) aligned_alloc(32, n*sizeof(int));


    copy_matrix(org_A, LU, n, n);

    for(int i=0; i<n; i++){
        tmp_P[i] = i;
    }


    for(int k=0; k<n; k++){
        // find pivot
        curr_piv = LU[k*n+k];
        index_piv = k;

        for(int i=k+1; i<n; i++){ 
            if( fabs(LU[i*n+k]) > fabs(curr_piv) ){
                curr_piv = LU[i*n+k];
                index_piv = i;
            }
        }



        if(index_piv != k){
        //swap rows to get pivot-row on top
            int x;
            for(x=0; x<n-3; x+=4){
                __m256d tmp_vec = _mm256_loadu_pd( &LU[k*n+x] );
                __m256d piv_vec = _mm256_loadu_pd( &LU[index_piv*n+x] );

                _mm256_storeu_pd(&LU[k*n+x], piv_vec);
                _mm256_storeu_pd(&LU[index_piv*n+x], tmp_vec);
            }

            // CLEANUP LOOP
            for(; x<n; x++){
                tmp = LU[k*n + x];
                LU[k*n +x] = LU[index_piv*n+x];
                LU[index_piv*n+x] = tmp;
            }

            tmp = tmp_P[k];
            tmp_P[k] = tmp_P[index_piv];
            tmp_P[index_piv] = tmp;

        }

    
        int j;
       for(j=k; j<n-3; j+=4){

            __m256d LU_vec = _mm256_loadu_pd( &LU[k*n+j] );

            for(int s=0; s<k; s++){
                __m256d vec1 = _mm256_set1_pd( LU[k*n+s] );
                __m256d vec2 = _mm256_loadu_pd( &LU[s*n+j] );
                __m256d prod = _mm256_mul_pd(vec1, vec2);
                LU_vec = _mm256_sub_pd(LU_vec, prod);
            }
            _mm256_storeu_pd(&LU[k*n+j], LU_vec);
        }

        // CLEANUP LOOP
        for(; j<n; j++){
            for(int s=0; s<k; s++){
                LU[k*n+j] -= LU[k*n+s] * LU[s*n+j];
            }
        }

    
        // Is this loop vectorizable..?
        for(int i=k+1; i<n; i++){
            for(int s=0; s<k; s++){
                LU[i*n+k] -= LU[i*n+s] * LU[s*n+k];
            }
            LU[i*n+k] /= LU[k*n+k];
        }

       
    }

    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            P[i*n+j] = 0;
        }
    }

    for(int i=0; i<n; i++){
        P[i*n + tmp_P[i]] = 1.0;
    }


 
    
    mmm(P, P_m, permuted_P_m, n, n, n);

    //Forwards substituition
    FLOP_COUNT_INC((n-2)*(n-1), "LU_solve2");

    //swapped those two loops from k-i-j to i-k-j
    for(int i=0; i<n; i++){
        int k;
        for(k=0; k<n-3; k+=4){
            __m256d res = _mm256_loadu_pd( &permuted_P_m[i*n+k] );
            for(int j=0; j<i; j++){                
                __m256d L_vec = _mm256_set1_pd( LU[i*n+j] );
                __m256d Y_vec = _mm256_loadu_pd( &Y[j*n+k] );
                __m256d prod = _mm256_mul_pd(L_vec, Y_vec);
                res = _mm256_sub_pd(res, prod);
            }
                _mm256_storeu_pd(&Y[i*n+k], res);

        }

        // CLEANUP LOOP
        for(; k<n; k++){
            Y[i*n + k] = permuted_P_m[i*n + k];
            for(int j=0; j<i; j++){
                Y[i*n+k] -=  LU[i*n+j] * Y[j*n + k];
            }
        }
        
    }

    //Backward substitution
    FLOP_COUNT_INC(n*n + n*(n-1)*n/2 * 3, "LU_solve2");

    //swapped for loops from k-i-j to i-k-j
        for(int i=n-1; i>=0; i--){
            int k;
            for(k=0; k<n-3; k+=4){
                
                __m256d Y_vec1 = _mm256_loadu_pd( &Y[i*n+k] );
                __m256d divisor = _mm256_set1_pd( LU[i*n+i] );
                __m256d res = _mm256_div_pd(Y_vec1, divisor);

                for(int j=i+1; j<n; j++){
                    __m256d U_vec = _mm256_set1_pd( LU[i*n+j] );
                    __m256d X_vec = _mm256_loadu_pd( &R_m[j*n+k] );
                    __m256d prod = _mm256_mul_pd(U_vec, X_vec);
                    __m256d frac = _mm256_div_pd(prod, divisor);
                    res = _mm256_sub_pd(res, frac);

                }

            _mm256_storeu_pd(&R_m[i*n+k], res);

            }

        // CLEANUP LOOP
            for(; k<n; k++){
                R_m[i*n + k] = Y[i*n + k] / LU[i*n+i];
                for(int j=i+1; j<n; j++){
                    R_m[i*n + k] -= (LU[i*n+j] * R_m[j*n+k]) / LU[i*n+i];
                }
            }
    }

    free(Y);
    free(permuted_P_m);
    free(P);
    free(LU);
    free(tmp_P);
}




void Blocked_LU_solve(double * Q_m, double *R_m, double *P_m, int n){

    int block_size = 1;
	int num_blocks = n / block_size;

    double *LU = (double*) malloc(n*n*sizeof(double));
    double *Y = (double*) malloc(n*n*sizeof(double));

    copy_matrix(Q_m, LU, n, n);


	for(int i=0; i<num_blocks; i++) {
		for(int i2 = i*block_size; i2 < (i+1)*block_size-1; i2++){
			for(int j2 = i2+1; j2 < (i+1)*block_size; j2++){
				LU[j2*n + i2] /= LU[i2*n + i2];
				for(int k2 = i2+1; k2 < (i+1)*block_size; k2++){
						LU[j2*n + k2] -= LU[j2*n + i2] * LU[i2*n + k2];
				}
			}
		}

	    for(int j=i+1; j<num_blocks; j++) {
			for(int i3 = i*block_size; i3 < (i+1)*block_size -1; i3++){
				for(int j3 = i3+1; j3 < block_size; j3++){
					for(int k3 = j*block_size; k3 < (j+1)*block_size; k3++){
						LU[j3*n + k3] -=  LU[j3*n + i3] * LU[i3*n + k3];
					}
				}
			}
	    }
 
	    for(int j=i+1; j<num_blocks; j++) {
	        for(int i4 = i*block_size; i4 < (i+1)*block_size; i4++){
				for(int j4 = j*block_size; j4 < (j+1)*block_size; j4++){
					LU[j4*n + i4] /= LU[i4*n + i4];
					for(int k4 = i4 + 1; k4 < (i+1)*block_size; k4++){
							LU[j4*n + k4] -= LU[j4*n + i4] * LU[i4*n + k4];
					}
				}

			}

          	for(int k=i+1; k<num_blocks; k++) {
				for(int i5 = i*block_size; i5 < (i+1)*block_size; i5++){
					for(int j5 = j*block_size; j5 < (j+1)*block_size; j5++){
						for(int k5 = k*block_size; k5 < (k+1)*block_size; k5++){
								LU[j5*n + k5] -= LU[j5*n + i5] * LU[i5*n + k5];
						}
					}
				}
	        }
	    }
	}


   //Forwards substituition
    for(int col=0; col<n; col++){
        for(int i=0; i<n; i++){
            Y[i*n + col] = P_m[i*n + col];
            for(int j=0; j<i; j++){
                Y[i*n+col] -=  LU[i*n+j] * Y[j*n + col];
            }
        }
    }

    //Backward substitution
    for(int col=0; col<n; col++){
        for(int i=n-1; i>=0; i--){
            R_m[i*n + col] = Y[i*n + col] / LU[i*n+i];
            for(int j=i+1; j<n; j++){
                R_m[i*n + col] -= (LU[i*n+j] * R_m[j*n+col]) / LU[i*n+i];
            }
        }

    }

    free(LU);
    free(Y);
}


void backward_substitution_ColMaj_ikj(double * A, double *y, double * b, int n){
    for(int i=0; i<n; i++){
        for(int k=0; k<n; k++){
            double sum = 0;
            for(int j=0; j<i; j++){
                sum += A[i*n+j] * y[j*n + k];
            }
            y[i*n + k] = b[i*n + k] - sum;
            y[i*n+k] /= A[i*n+i];
         }
     }
}

void backward_substitution_ColMaj_kij(double * A, double *y, double * b, int n){
    for(int k=0; k<n; k++){
        for(int i=0; i<n; i++){
            double sum = 0;
            for(int j=0; j<i; j++){
                sum += A[i*n+j] * y[j*n + k];
                }
            y[i*n + k] = b[i*n + k] - sum;
            y[i*n+k] /= A[i*n+i];
        }
    }
}

void backward_substitution_ColMaj_vec_ikj(double * A, double *y, double * b, int n){
    int k;
    int j;

    for(int i=0; i<n; i++){
        for(k=0; k<n-3; k+=4){
            __m256d sum0 = _mm256_setzero_pd();
            __m256d sum1 = _mm256_setzero_pd();
            __m256d sum2 = _mm256_setzero_pd();
            __m256d sum3 = _mm256_setzero_pd();
            __m256d sum_cleanup = _mm256_setzero_pd();

            __m256d res_vec= _mm256_loadu_pd( &b[i*n+k] );
        
            __m256d denominator_vec = _mm256_broadcast_sd(&A[i*n+i]);
            __m256d one_vec = _mm256_set1_pd(1.0);
            __m256d rezi_vec = _mm256_div_pd(one_vec, denominator_vec);

            for(j=0; j<i-3; j+=4){
                __m256d y_vec0 = _mm256_loadu_pd( &y[j*n+k] );
                __m256d y_vec1 = _mm256_loadu_pd( &y[(j+1)*n+k] );
                __m256d y_vec2 = _mm256_loadu_pd( &y[(j+2)*n+k] );
                __m256d y_vec3 = _mm256_loadu_pd( &y[(j+3)*n+k] );

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
                __m256d y_vec_cleanup= _mm256_loadu_pd( &y[j*n+k] );
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
            _mm256_storeu_pd(&y[i*n+k], res_vec);
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

void backward_substitution_ColMaj_vec_kij(double * A, double *y, double * b, int n){

    int k;
    int j;
    for(k=0; k<n-3; k+=4){
        for(int i=0; i<n; i++){
            __m256d sum0 = _mm256_setzero_pd();
            __m256d sum1 = _mm256_setzero_pd();
            __m256d sum2 = _mm256_setzero_pd();
            __m256d sum3 = _mm256_setzero_pd();
            __m256d sum_cleanup = _mm256_setzero_pd();

            __m256d res_vec= _mm256_loadu_pd( &b[i*n+k] );
           
            __m256d denominator_vec = _mm256_broadcast_sd(&A[i*n+i]);
            __m256d one_vec = _mm256_set1_pd(1.0);
            __m256d rezi_vec = _mm256_div_pd(one_vec, denominator_vec);

            for(j=0; j<i-3; j+=4){
                __m256d y_vec0 = _mm256_loadu_pd( &y[j*n+k] );
                __m256d y_vec1 = _mm256_loadu_pd( &y[(j+1)*n+k] );
                __m256d y_vec2 = _mm256_loadu_pd( &y[(j+2)*n+k] );
                __m256d y_vec3 = _mm256_loadu_pd( &y[(j+3)*n+k] );

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
                __m256d y_vec_cleanup= _mm256_loadu_pd( &y[j*n+k] );
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
            _mm256_storeu_pd(&y[i*n+k], res_vec);
        }
    }
    // CLEANUP LOOP for k
    for(; k<n; k++){
        for(int i=0; i<n; i++){
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




void forward_substitution_ColMaj_ikj(double * U, double *x, double * b, int n){
    for(int i=n-1; i>=0; i--){
        for(int k=0; k<n; k++){
            double sum = 0;
            for(int j=i+1; j<n; j++){
                sum += U[i*n+j] * x[j*n+k];
            }
            x[i*n + k] = b[i*n+k] - sum;
            x[i*n + k] /= U[i*n + i];
        }
    } 
}

void forward_substitution_ColMaj_kij(double * U, double *x, double * b, int n){
    for(int k=0; k<n; k++){
        for(int i=n-1; i>=0; i--){
            double sum = 0;
            for(int j=i+1; j<n; j++){
                sum += U[i*n+j] * x[j*n+k];
            }
            x[i*n + k] = b[i*n+k] - sum;
            x[i*n + k] /= U[i*n + i];
        }
    } 
}


void forward_substitution_ColMaj_vec_ikj(double * U, double *x, double * b, int n){
    int j;
    int k;
    for(int i=n-1; i>=0; i--){
        for(k=0; k<n-3; k+=4){
            __m256d sum0 = _mm256_setzero_pd();
            __m256d sum1 = _mm256_setzero_pd();
            __m256d sum2 = _mm256_setzero_pd();
            __m256d sum3 = _mm256_setzero_pd();
            __m256d sum_cleanup = _mm256_setzero_pd();

            __m256d res_vec= _mm256_loadu_pd( &b[i*n+k] );

            __m256d denominator_vec = _mm256_broadcast_sd(&U[i*n+i]);
            __m256d one_vec = _mm256_set1_pd(1.0);
            __m256d rezi_vec = _mm256_div_pd(one_vec, denominator_vec);

            for(j=i+1; j<n-3; j+=4){
                __m256d x_vec0 = _mm256_loadu_pd( &x[j*n+k] );
                __m256d x_vec1 = _mm256_loadu_pd( &x[(j+1)*n+k] );
                __m256d x_vec2 = _mm256_loadu_pd( &x[(j+2)*n+k] );
                __m256d x_vec3 = _mm256_loadu_pd( &x[(j+3)*n+k] );

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
            __m256d x_vec_cleanup= _mm256_loadu_pd( &x[j*n+k] );
            __m256d U_vec_cleanup = _mm256_broadcast_sd(&U[i*n+j]);

            sum_cleanup = _mm256_fmadd_pd(U_vec_cleanup, x_vec_cleanup, sum_cleanup);

            }

            __m256d tmp_sum0 = _mm256_add_pd(sum0, sum1);
            __m256d tmp_sum1 = _mm256_add_pd(sum2, sum3);
            __m256d final_sum = _mm256_add_pd(tmp_sum0, tmp_sum1);

            res_vec = _mm256_sub_pd(res_vec, sum_cleanup);
            res_vec = _mm256_sub_pd(res_vec, final_sum);

            res_vec = _mm256_mul_pd(res_vec, rezi_vec);

            _mm256_storeu_pd(&x[i*n+k], res_vec);
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


void forward_substitution_ColMaj_vec_kij(double * U, double *x, double * b, int n){
    int j;
    int k;
    
    for(k=0; k<n-3; k+=4){
        for(int i=n-1; i>=0; i--){
            __m256d sum0 = _mm256_setzero_pd();
            __m256d sum1 = _mm256_setzero_pd();
            __m256d sum2 = _mm256_setzero_pd();
            __m256d sum3 = _mm256_setzero_pd();
            __m256d sum_cleanup = _mm256_setzero_pd();

            __m256d res_vec= _mm256_loadu_pd( &b[i*n+k] );


            __m256d denominator_vec = _mm256_broadcast_sd(&U[i*n+i]);
            __m256d one_vec = _mm256_set1_pd(1.0);
            __m256d rezi_vec = _mm256_div_pd(one_vec, denominator_vec);

            for(j=i+1; j<n-3; j+=4){
                __m256d x_vec0 = _mm256_loadu_pd( &x[j*n+k] );
                __m256d x_vec1 = _mm256_loadu_pd( &x[(j+1)*n+k] );
                __m256d x_vec2 = _mm256_loadu_pd( &x[(j+2)*n+k] );
                __m256d x_vec3 = _mm256_loadu_pd( &x[(j+3)*n+k] );

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
            __m256d x_vec_cleanup= _mm256_loadu_pd( &x[j*n+k] );
            __m256d U_vec_cleanup = _mm256_broadcast_sd(&U[i*n+j]);

            sum_cleanup = _mm256_fmadd_pd(U_vec_cleanup, x_vec_cleanup, sum_cleanup);

            }

            __m256d tmp_sum0 = _mm256_add_pd(sum0, sum1);
            __m256d tmp_sum1 = _mm256_add_pd(sum2, sum3);
            __m256d final_sum = _mm256_add_pd(tmp_sum0, tmp_sum1);

            res_vec = _mm256_sub_pd(res_vec, sum_cleanup);
            res_vec = _mm256_sub_pd(res_vec, final_sum);

            res_vec = _mm256_mul_pd(res_vec, rezi_vec);

            _mm256_storeu_pd(&x[i*n+k], res_vec);
        }
        
    } 

        //CLEANUP LOOP FOR K
        for(; k<n; k++){
            for(int i=n-1; i>=0; i--){
                double sum = 0;
                for(j=i+1; j<n; j++){
                    sum += U[i*n+j] * x[j*n+k];
                }
                x[i*n + k] = b[i*n+k] - sum;
                x[i*n + k] /= U[i*n + i];
            }
        }
}

//THIS NEEDS TO BE OPTIMIZED
void forward_substitution_ColMaj_LU(double * U, double *x, double * b, int n){
    for(int k=0; k<n; k++){
        for(int i=n-1; i>=0; i--){
            double sum = 0;
            for(int j=i+1; j<n; j++){
                sum += U[i*n+j] * x[j*n+k];
            }
            x[i*n + k] = b[i*n+k] - sum;
        }
    } 
}
*/
/*

void LU_ColMaj_PivRows_BLAS_vec(double * org_A, double *LU, double *P, int n ){//,double *dest){
   // FLOP_COUNT_INC: has been added directly at the operations for simplicity
    double curr_piv;
    int index_piv;
    double tmp;

    copy_matrix(org_A, LU, n, n);

   
    int *tmp_P = (int*) malloc(n*sizeof(int));
    for(int i=0; i<n; i++){
        tmp_P[i] = i;
    }


    for(int k=0; k<n; k++){
        // find pivot
        curr_piv = fabs(LU[k*n+k]);
        index_piv = k;
        //FLOP_COUNT_INC(2*(n-k), "LU_solve2");

        for(int i=k+1; i<n; i++){ 
            if( fabs(LU[k*n+i]) > curr_piv ){
                curr_piv = fabs(LU[k*n+i]);
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
            //printf("new: \n");
            //printmatrix(dest, n, n);


            tmp = tmp_P[k];
            tmp_P[k] = tmp_P[index_piv];
            tmp_P[index_piv] = tmp;

        }

        if(LU[k*n+k] == 0){
            printf("non invertible, try again \n");
        }


        double *a12 = (double*) malloc((n-k -1)*sizeof(double));
        double *outer_prod = (double*) malloc((n-k-1)*(n-k-1)*sizeof(double));

        double rezi = 1.0 / LU[k*n+k];

        __m256d rezi_vec = _mm256_set1_pd(rezi);


        // this was commented out 
        double *a21 = (double*) malloc((n-k -1)*sizeof(double));
        for(int i=1; i<(n-k); i++){
            double tmp_xxx = LU[k*n+ k +i] *rezi;  

            a21[i-1] = tmp_xxx;

            LU[k*n+k+i] = tmp_xxx;
            a12[i-1] = LU[(k+i)*n+k];
        }
        mmm(a12, a21, outer_prod, 1, (n-k-1), (n-k-1));
        

       int i;
        for(i=1; i<(n-k)-3; i+=4){

            __m256d vec1 = _mm256_loadu_pd(&LU[k*n+ k +i]);
            __m256d res_vec = _mm256_mul_pd(vec1, rezi_vec);
            _mm256_storeu_pd(&LU[k*n+k+i], res_vec);

            //__m256d vec2 = _mm256_i64gather_pd(&LU[(k+i)*n+k], index_vec, 8);
            //_mm256_storeu_pd(&a12[i-1], vec2);

            a12[i-1] = LU[(k+i)*n+k];
            a12[i] = LU[(k+i+1)*n+k];
            a12[i+1] = LU[(k+i+2)*n+k];
            a12[i+2] = LU[(k+i+3)*n+k];
        }
        //CLEANUP LOOP FOR I
        for(; i<(n-k); i++){
            LU[k*n+k+i] = LU[k*n+ k +i] *rezi;  
            a12[i-1] = LU[(k+i)*n+k];
        }

        mmm(a12, &LU[k*n+k+1], outer_prod, 1, (n-k-1), (n-k-1));
        

        for(int i=1; i<(n-k); i++){
            int j;
            
            for(j=1; j<(n-k)-3; j+=4){
                __m256d curr = _mm256_loadu_pd(&LU[(k+i)*n+ (k+j)]);
                __m256d to_sub = _mm256_loadu_pd(&outer_prod[(i-1) * (n-k-1) + (j-1)]);

                __m256d res = _mm256_sub_pd(curr, to_sub);
                _mm256_storeu_pd(&LU[(k+i)*n+ (k+j)], res);
            }
            //CLEANUP LOOP
            for(; j<(n-k); j++){
                LU[(k+i)*n + (k+j)] -= outer_prod[(i-1) * (n-k-1) + (j-1)];
            }
        }

    //free(a21);
    free(a12);
    free(outer_prod);
    
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
*/

/*
void LU_ColMaj_PivRows_vec(double *LU, double *P, int n ){//(double * org_A, double *LU, double *P, int n ){//,double *dest){
   // FLOP_COUNT_INC: has been added directly at the operations for simplicity
    double curr_piv;
    int index_piv;
    double tmp;

    //copy_matrix(org_A, LU, n, n);

   
    int *tmp_P = (int*) malloc(n*sizeof(int));
    for(int i=0; i<n; i++){
        tmp_P[i] = i;
    }


    for(int k=0; k<n; k++){
        // find pivot
        curr_piv = fabs(LU[k*n+k]);
        index_piv = k;
        //FLOP_COUNT_INC(2*(n-k), "LU_solve2");

        for(int i=k+1; i<n; i++){ 
            if( fabs(LU[k*n+i]) > curr_piv ){
                curr_piv = fabs(LU[k*n+i]);
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

        if(LU[k*n+k] == 0){
            printf("non invertible, try again \n");
        }


        double *a12 = (double*) malloc((n-k -1)*sizeof(double));
        double *outer_prod = (double*) malloc((n-k-1)*(n-k-1)*sizeof(double));

        double rezi = 1.0 / LU[k*n+k];

        __m256d rezi_vec = _mm256_set1_pd(rezi);


       int i;
        for(i=1; i<(n-k)-3; i+=4){

            __m256d vec1 = _mm256_loadu_pd(&LU[k*n+ k +i]);
            __m256d res_vec = _mm256_mul_pd(vec1, rezi_vec);
            _mm256_storeu_pd(&LU[k*n+k+i], res_vec);

            a12[i-1] = LU[(k+i)*n+k];
            a12[i] = LU[(k+i+1)*n+k];
            a12[i+1] = LU[(k+i+2)*n+k];
            a12[i+2] = LU[(k+i+3)*n+k];
        }
        //CLEANUP LOOP FOR I
        for(; i<(n-k); i++){
            LU[k*n+k+i] = LU[k*n+ k +i] *rezi;  
            a12[i-1] = LU[(k+i)*n+k];
        }


        for(int i=0; i<(n-k-1); i++){
            int j;
            for(j=0; j<(n-k-1)-3; j+=4){
                outer_prod[i*(n-k-1) + j] = a12[i] * LU[k*n+k+1+j];
                outer_prod[i*(n-k-1) + j +1] = a12[i] * LU[k*n+k+1+j +1];
                outer_prod[i*(n-k-1) + j +2] = a12[i] * LU[k*n+k+1+j +2];
                outer_prod[i*(n-k-1) + j +3] = a12[i] * LU[k*n+k+1+j +3];
            }
            //CLEANUP LOOP FOR J
            for(; j<(n-k-1); j++){
                outer_prod[i*(n-k-1) + j] = a12[i] * LU[k*n+k+1+j];
            }
        }

        mmm(a12, &LU[k*n+k+1], outer_prod, 1, (n-k-1), (n-k-1));
        

        for(int i=1; i<(n-k); i++){
            int j;
            
            for(j=1; j<(n-k)-3; j+=4){
                __m256d curr = _mm256_loadu_pd(&LU[(k+i)*n+ (k+j)]);
                __m256d to_sub = _mm256_loadu_pd(&outer_prod[(i-1) * (n-k-1) + (j-1)]);

                __m256d res = _mm256_sub_pd(curr, to_sub);
                _mm256_storeu_pd(&LU[(k+i)*n+ (k+j)], res);
            }
            //CLEANUP LOOP
            for(; j<(n-k); j++){
                LU[(k+i)*n + (k+j)] -= outer_prod[(i-1) * (n-k-1) + (j-1)];
            }
        }

    //free(a21);
    free(a12);
    free(outer_prod);
    
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
*/





