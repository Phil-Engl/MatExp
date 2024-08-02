

void Eigen_LU_decomposition(const double * inp_mat, const double * RHS_in,  int size, double * out_mat){
    Eigen::Map<const Eigen::MatrixXd> A(inp_mat, size, size);
    Eigen::Map<const Eigen::MatrixXd> RHS(RHS_in, size, size);
    Eigen::Map<Eigen::MatrixXd> LU(out_mat, size, size);
    //LU = A.partialPivLu().solve(RHS);
    LU = A.fullPivLu().solve(RHS);  
}

void Eigen_MatPow(const double * inp_mat, long power, int n, double * out_mat){
    Eigen::Map<const Eigen::MatrixXd> A(inp_mat, n, n);
    //Eigen::Map<const Eigen::MatrixXd> RHS(RHS_in, size, size);
    Eigen::Map<Eigen::MatrixXd> PowMat(out_mat, n, n);
    //LU = A.partialPivLu().solve(RHS);
    //LU = A.fullPivLu().solve(RHS); 
    PowMat = A.pow(power);
}

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
            y[i*n + col] = b[i*n + col];
            for(int j=0; j<i; j++){
                FLOP_COUNT_INC(2, "forward_substitution_matrix");
                y[i*n+col] = y[i*n + col] - A[i*n+j] * y[j*n + col];
            }
            FLOP_COUNT_INC(1,"forward_substitution_matrix");
            y[i*n+col] = y[i*n + col] / A[i*n+i];
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
                x[i*n + col] = x[i * n+col] - (U[i*n+j] * x[j*n+col]) / U[i*n+i];
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