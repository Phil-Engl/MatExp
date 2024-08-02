
/**
 * @brief ell function from the paper
 * 
 * @param A the input matrix n x n
 * @param n The dimensions of the input matrix
 * @param m ¯\ (ツ) /¯
 * @return int value of the ell function
 */
int ell(const double* A, int n, int m){ 
    FLOP_COUNT_INC(7, "ell");

    double *A_abs = (double*) malloc(n*n*sizeof(double));
    double *A_tmp = (double*) malloc(n*n*sizeof(double));

    mat_abs(A,A_abs,n,n);
    matpow_by_squaring(A_abs,n,2*m+1,A_tmp);
    double abs_norm = normest(A_tmp, n);
    //overflow check
    free(A_abs);
    free(A_tmp);
    if(abs_norm <= 0.0){
        return 0;
    }
    
    double alpha = (coeffs[m] * abs_norm) / onenorm(A,n,n);
    return fmax((int)ceil(log2(alpha / (*unit_roundoff))/(2*m)), 0);
}

/* main function */
void mat_exp(const double *A, int n, double *E){
    // FLOP_COUNT_INC are added within the branches
    if(n == 1){
        FLOP_COUNT_INC(1, "mat_exp n=1");
        E[0] = exp(A[0]);
        if(DEBUG) printf("returned trivial\n");
        return;
    }

    if(DEBUG) printf("mat_exp start\n");

    int flag = 0;
    int triangular_indicator = is_triangular(A, n, n);
    
    double *P_m = (double*) malloc(n*n*sizeof(double));
    double *Q_m = (double*) malloc(n*n*sizeof(double));
    double *R_m = (double*) malloc(n*n*sizeof(double));
    double *Temp = (double*) malloc(n*n*sizeof(double));
    double *A_abs = (double*) malloc(n*n*sizeof(double));
    double *P_m_abs = (double*) malloc(n*n*sizeof(double));
    double * Q_m_abs = (double*) malloc(n*n*sizeof(double));

    mat_abs(A, A_abs, n, n); 


    double * A_2 = (double*) malloc(n*n*sizeof(double)); 
    mmm(A, A, A_2, n, n, n); // A^2
    
    double * A_4 = (double*) malloc(n*n*sizeof(double));
    mmm(A_2, A_2, A_4, n, n, n); // A^4

    double * A_6 = (double*) malloc(n*n*sizeof(double)); 
    mmm(A_2, A_4, A_6, n, n, n); // A^6

    double * A_8 = (double*) malloc(n*n*sizeof(double)); 
    // will be computed later



    do{
        // ========================= p = 3 =========================
        FLOP_COUNT_INC(6, "mat_exp p=3");

        double d_6 = pow(normest(A_6, n), 1.0/6.0); // onenormest(A_2, 3)
        double eta_1 = fmax(pow(normest(A_4, n), 1.0/4.0), d_6); // onenormest(A_2, 2)

        if(eta_1 <= theta[3] && ell(A, n, 3) == 0){
            if(DEBUG) printf("Case m = 3\n");
            FLOP_COUNT_INC(5, "mat_exp p=3");

            eval3_4(A, A_2, A_4, A_6, n, 3, P_m, Q_m);
            eval3_4_abs(A_abs, n, 3, P_m_abs, Q_m_abs);
            mat_col_sum(P_m_abs, n, Temp);

            double divider = fmin(onenorm(P_m, n, n), onenorm(Q_m, n, n));
            if(infinity_norm(Temp, n, 1)/divider <= 10*exp(theta[3])){
                if(DEBUG) printf("returned m = 3\n");
                eval3_6(P_m, Q_m, n, E, triangular_indicator);
                break;
            }
        }

        // ======================== p = 5 =========================
        

        FLOP_COUNT_INC(4, "mat_exp p=5");
        double d_4 = pow(onenorm(A_4, n, n), 1.0/4.0);
        double eta_2 = fmax(d_4, d_6);

        if(eta_2 <= theta[5] && ell(A, n, 5) == 0){
            if(DEBUG) printf("Case m = 5\n");
            FLOP_COUNT_INC(5, "mat_exp p=5");

            eval3_4(A, A_2, A_4, A_6, n, 5, P_m, Q_m);
            eval3_4_abs(A_abs, n, 5, P_m_abs, Q_m_abs);
            mat_col_sum(P_m_abs, n, Temp);

            double divider = fmin(onenorm(P_m, n, n), onenorm(Q_m, n, n));
            if(infinity_norm(Temp, n, 1)/divider <= 10*exp(theta[5])){
                if(DEBUG) printf("returned m = 5\n");
                eval3_6(P_m, Q_m, n, E, triangular_indicator);
                break;
            }
        }

        // ======================== p = 7, 9 ========================

        mmm(A_4, A_4, A_8, n, n, n); // A_4^2

        FLOP_COUNT_INC(7, "mat_exp p=7,9");
        d_6 = pow(onenorm(A_6, n, n), 1.0/6.0);
        double d_8 = pow(normest(A_8, n), 1.0/8.0); //onenormest(A_4, 2)
        double eta_3 = fmax(d_6, d_8);
        
        for(int m = 7; m <= 9; m+=2){
            if(eta_3 <= theta[m] && ell(A, n, m) == 0){
                if(DEBUG) printf("Case m = %d\n", m);
                FLOP_COUNT_INC(5, "mat_exp p=7,9");

                eval3_4(A, A_2, A_4, A_6, n, m, P_m, Q_m);
                eval3_4_abs(A_abs, n, m, P_m_abs, Q_m_abs);
                mat_col_sum(P_m_abs, n, Temp);

                double divider = fmin(onenorm(P_m, n, n), onenorm(Q_m, n, n));
                if(infinity_norm(Temp, n, 1)/divider <= 10*exp(theta[m])){
                    if(DEBUG) printf("returned m = %d\n", m);
                    eval3_6(P_m, Q_m, n, E, triangular_indicator);
                    flag = 1;
                    break;
                }
            }
        }
        if(flag){
            break;
        }

        // ========================= p = 13 =========================

        FLOP_COUNT_INC(20, "mat_exp p=13");
        if(DEBUG) printf("Case m = 13\n");
        double * A_10 = (double*) malloc(n*n*sizeof(double)); 
        mmm(A_4, A_6, A_10, n, n, n); // A_4 * A_6

        double eta_4 = fmax(d_8, pow(normest(A_10, n), 0.1)); // onenormest(A_4, A_6)
        free(A_10);
        double eta_5 = fmin(eta_3, eta_4);

        int s = (int)fmax(ceil(log2(eta_5/theta[13])), 0.0);
        double * A_temp = (double*) malloc(n*n*sizeof(double));
        scalar_matrix_mult(pow(2.0, -s), A, A_temp, n, n); // 2^-s * A
        s = s + ell(A_temp, n, 13);

        scalar_matrix_mult(pow(2.0, -s), A, A_temp, n, n);
        scalar_matrix_mult(pow(2.0, -2.0*s), A_2, A_2, n, n);
        scalar_matrix_mult(pow(2.0, -4.0*s), A_4, A_4, n, n);
        scalar_matrix_mult(pow(2.0, -6.0*s), A_6, A_6, n, n);

        eval3_5(A_temp, A_2, A_4, A_6, n, P_m, Q_m);
        eval3_5_abs(A_abs, n, P_m_abs, Q_m_abs);
        mat_col_sum(P_m_abs, n, Temp);
        double divider = fmin(onenorm(P_m, n, n), onenorm(Q_m, n, n));
        int s_max = (int)ceil(log2(onenorm(A,n,n) / theta[13]));
        if(infinity_norm(Temp, n, 1)/divider <= (10+s_max)*exp(theta[13])){
            if(DEBUG) printf("case scaled\n");
            eval3_6(P_m, Q_m, n, R_m, triangular_indicator);
        }else{
            FLOP_COUNT_INC(4, "mat_exp scaled again");
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
            if(DEBUG) printf("triangular squaring\n");
            FLOP_COUNT_INC(2*n + 2*n*s, "mat_exp triangular");

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
            if(DEBUG) printf("non triangular squaring\n");
            if(DEBUG) assert(s >= 0);
            int use_eigen = 0; // set to 0 use out implementation to calculate matrix power, 1 to use eigen

            if(use_eigen){
                get_MatPow_from_eigen(R_m, (long) pow(2,s), n, E);
            }else{
                matpow_by_squaring(R_m, n, (long) pow(2, s), E); // TODO: FLOP_COUNT_INC: 1 flop?
            }

        }

        free(A_temp);
    }while(0);

    GET_FLOP_COUNT(); // Prints the total flop count

    if(DEBUG) printf("cleanup\n");
    
    free(Temp);
    free(A_2);
    free(A_4);
    free(A_6);
    free(A_8);
    free(P_m);
    free(Q_m);
    free(R_m);
    free(A_abs);
    free(Q_m_abs);
    free(P_m_abs);
    return;
    
}