/**
 * @brief Evaluates equation 3.4
 * 
 * @param A The input matrix A with dimension n x n
 * @param n The number of rows and columns of A
 * @param m The padé approximant
 * @param P_m The output matrix p_m(A) with dimension n x n
 * @param Q_m The output matrix q_m(A) with dimension n x n
 */
void eval3_4(const double* A, const double* A_2, const double* A_4, const double* A_6, const double* A_8, int n, const int m, double *P_m, double *Q_m)
{       
    if(m > 9){
        printf("eval3_4 is called with invalid parameter. Got m=%d, expected: 0<=m<=9\n", m);
        return;
    }

    double *U = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *V = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *Temp = (double*) aligned_alloc(32, n*n*sizeof(double));

    const double *A_pow_ptr[] = {A_2, A_2, A_2, A_4, A_4, A_6, A_6, A_8, A_8};

    // compute u and v separately
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j+=4*{{ k }}){
            {% for l in range(k) %}
            int idx_{{ l }} = i*n + j + 4*{{ l }};
            {% endfor %}
             // first term is b*A_0 = b*I (identity matrix)

            {% for l in range(k) %}
            __m256d u_tmp_{{l}} = _mm256_set_pd((i==j+3+ 4*{{ l }}? pade_coefs[1] : 0.0), (i==j+2+ 4*{{ l }}? pade_coefs[1] : 0.0), (i==j+1+ 4*{{ l }}? pade_coefs[1] : 0.0), (i==j+ 4*{{ l }}? pade_coefs[1] : 0.0));
            __m256d v_tmp_{{l}} = _mm256_set_pd((i==j+3+ 4*{{ l }}? pade_coefs[0] : 0.0), (i==j+2+ 4*{{ l }}? pade_coefs[0] : 0.0), (i==j+1+ 4*{{ l }}? pade_coefs[0] : 0.0), (i==j+ 4*{{ l }}? pade_coefs[0] : 0.0));
            {% endfor %}
            
            for(int l = 2; l < m; l+=2){ 
                {% for l in range(k) %}
                __m256d a_ij_{{l}} = _mm256_load_pd(&A_pow_ptr[l][idx_{{ l }}]);
                {% endfor %}

                __m256d b_u = _mm256_set1_pd(pade_coefs[l+1]);
                __m256d b_v = _mm256_set1_pd(pade_coefs[l]);

                {% for l in range(k) %}
                u_tmp_{{l}} = _mm256_fmadd_pd(b_u, a_ij_{{l}}, u_tmp_{{l}});
                v_tmp_{{l}} = _mm256_fmadd_pd(b_v, a_ij_{{l}}, v_tmp_{{l}});
                {% endfor %}
            }

            {% for l in range(k) %}
            _mm256_store_pd(&U[idx_{{ l }}], u_tmp_{{l}});
            _mm256_store_pd(&V[idx_{{ l }}], v_tmp_{{l}});
            {% endfor %}
        }
    }

    // one matrix mult to get uneven powers of A
    mmm(A, U, Temp, n, n, n);

    for(int i=0; i<n*n; i+=4*{{k}}){
        {% for l in range(k) %}
        __m256d u_tmp_{{l}} = _mm256_load_pd(&Temp[i + 4*{{ l }}]);
        __m256d v_tmp_{{l}} = _mm256_load_pd(&V[i + 4*{{ l }}]);
        {% endfor %}

        {% for l in range(k) %}
        __m256d p_tmp_{{l}} = _mm256_add_pd(u_tmp_{{l}}, v_tmp_{{l}});
        __m256d q_tmp_{{l}} = _mm256_sub_pd(v_tmp_{{l}}, u_tmp_{{l}});
        {% endfor %}

        {% for l in range(k) %}
        _mm256_store_pd(&P_m[i + 4*{{ l }}], p_tmp_{{l}});
        _mm256_store_pd(&Q_m[i + 4*{{ l }}], q_tmp_{{l}});
        {% endfor %}
    }

    free(U);
    free(V);
    free(Temp);
}

/**
 * @brief Evaluates equation 3.4. Special case for A where powers are not precomputed
 * 
 * @param A_abs The input matrix A with dimension n x n
 * @param n The number of rows and columns of A
 * @param m The padé approximant
 * @param P_m_abs The output matrix p_m(A) with dimension n x n
 * @param Q_m_abs The output matrix q_m(A) with dimension n x n
 */
void eval3_4_abs(const double *A_abs, int n, int m, double *P_m_abs, double *Q_m_abs){
    double *A_abs_2 = (double*) malloc(n*n*sizeof(double));
    double *A_abs_4 = (double*) malloc(n*n*sizeof(double));
    double *A_abs_6 = (double*) malloc(n*n*sizeof(double));
    double *A_abs_8 = (double*) malloc(n*n*sizeof(double));

    mmm(A_abs, A_abs, A_abs_2, n, n, n);
    mmm(A_abs_2, A_abs_2, A_abs_4, n, n, n);
    mmm(A_abs_2, A_abs_4, A_abs_6, n, n, n);
    mmm(A_abs_4, A_abs_4, A_abs_8, n, n, n);

    eval3_4(A_abs, A_abs_2, A_abs_4, A_abs_6, A_abs_8, n, m, P_m_abs, Q_m_abs);

    free(A_abs_2);
    free(A_abs_4);
    free(A_abs_6);
    free(A_abs_8);
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
    for(int i = 0; i < n*n; i+=4*{{k}}){
        {% for l in range(k) %}
        __m256d a6_i_{{l}} = _mm256_load_pd(&A_6[i+4*{{l}}]);
        __m256d a4_i_{{l}} = _mm256_load_pd(&A_4[i+4*{{l}}]);
        __m256d a2_i_{{l}} = _mm256_load_pd(&A_2[i+4*{{l}}]);
        {% endfor %}

        {% for l in range(k) %}
        __m256d u_tmp0_{{l}} = _mm256_mul_pd(b_13, a6_i_{{l}});
        __m256d v_tmp0_{{l}} = _mm256_mul_pd(b_12, a6_i_{{l}});

        __m256d u_tmp1_{{l}} = _mm256_fmadd_pd(b_11, a4_i_{{l}}, u_tmp0_{{l}});
        __m256d v_tmp1_{{l}} = _mm256_fmadd_pd(b_10, a4_i_{{l}}, v_tmp0_{{l}});

        __m256d u_tmp2_{{l}} = _mm256_fmadd_pd(b_9, a2_i_{{l}}, u_tmp1_{{l}});
        __m256d v_tmp2_{{l}} = _mm256_fmadd_pd(b_8, a2_i_{{l}}, v_tmp1_{{l}});
        {% endfor %}

        {% for l in range(k) %}
        _mm256_store_pd(&U_tmp[i+4*{{l}}], u_tmp2_{{l}});
        _mm256_store_pd(&V_tmp[i+4*{{l}}], v_tmp2_{{l}});
        {% endfor %}
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
    for(int i = 0; i < n*n; i+=4*{{k}}){
        {% for l in range(k) %}
        __m256d a6_i_{{l}} = _mm256_load_pd(&A_6[i+4*{{l}}]);
        __m256d a4_i_{{l}} = _mm256_load_pd(&A_4[i+4*{{l}}]);
        __m256d a2_i_{{l}} = _mm256_load_pd(&A_2[i+4*{{l}}]);
        __m256d u_tmp0_{{l}} = _mm256_load_pd(&U[i+4*{{l}}]);
        __m256d v_tmp0_{{l}} = _mm256_load_pd(&V[i+4*{{l}}]);
        {% endfor %}

        {% for l in range(k) %}
        __m256d u_tmp1_{{l}} = _mm256_fmadd_pd(b_7, a6_i_{{l}}, u_tmp0_{{l}});
        __m256d v_tmp1_{{l}} = _mm256_fmadd_pd(b_6, a6_i_{{l}}, v_tmp0_{{l}});

        __m256d u_tmp2_{{l}} = _mm256_fmadd_pd(b_5, a4_i_{{l}}, u_tmp1_{{l}});
        __m256d v_tmp2_{{l}} = _mm256_fmadd_pd(b_4, a4_i_{{l}}, v_tmp1_{{l}});

        __m256d u_tmp3_{{l}} = _mm256_fmadd_pd(b_3, a2_i_{{l}}, u_tmp2_{{l}});
        __m256d v_tmp3_{{l}} = _mm256_fmadd_pd(b_2, a2_i_{{l}}, v_tmp2_{{l}});
        {% endfor %}

        {% for l in range(k) %}
        _mm256_store_pd(&U[i+4*{{l}}], u_tmp3_{{l}});
        _mm256_store_pd(&V[i+4*{{l}}], v_tmp3_{{l}});
        {% endfor %}
    }

    for(int i = 0; i < n; i++){
            U[i*n+i] += pade_coefs[1];
            V[i*n+i] += pade_coefs[0];
    }

    mmm(A, U, U_tmp, n, n, n); 

    for(int i=0; i < n*n; i+=4*{{k}}){
        {% for l in range(k) %}
        __m256d u_tmp_{{l}} = _mm256_load_pd(&U_tmp[i+4*{{l}}]);
        __m256d v_tmp_{{l}} = _mm256_load_pd(&V[i+4*{{l}}]);
        {% endfor %}

        {% for l in range(k) %}
        __m256d p_tmp_{{l}} = _mm256_add_pd(u_tmp_{{l}}, v_tmp_{{l}});
        __m256d q_tmp_{{l}} = _mm256_sub_pd(v_tmp_{{l}}, u_tmp_{{l}});
        {% endfor %}

        {% for l in range(k) %}
        _mm256_store_pd(&P_13[i+4*{{l}}], p_tmp_{{l}});
        _mm256_store_pd(&Q_13[i+4*{{l}}], q_tmp_{{l}});
        {% endfor %}
    }

    free(U_tmp);
    free(V_tmp);
    free(U);
    free(V);
}

/**
 * @brief Evaluates equation 3.5. Special case for A where powers are not precomputed
 * 
 * @param A_abs The input matrix A with dimension n x n
 * @param n The number of rows and columns of A
 * @param P_13_abs The output matrix p_13(A) with dimension n x n
 * @param Q_13_abs The output matrix q_13(A) with dimension n x n
 */
void eval3_5_abs(const double *A_abs, int n, double *P_13_abs, double *Q_13_abs){
    double *A_abs_2 = (double*) malloc(n*n*sizeof(double));
    double *A_abs_4 = (double*) malloc(n*n*sizeof(double));
    double *A_abs_6 = (double*) malloc(n*n*sizeof(double));

    mmm(A_abs, A_abs, A_abs_2, n, n, n);
    mmm(A_abs_2, A_abs_2, A_abs_4, n, n, n);
    mmm(A_abs_2, A_abs_4, A_abs_6, n, n, n);

    eval3_5(A_abs, A_abs_2, A_abs_4, A_abs_6, n, P_13_abs, Q_13_abs);

    free(A_abs_2);
    free(A_abs_4);
    free(A_abs_6);
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
    // FLOP_COUNT_INC(0, "eval3_6"); Note: all flops are already accounted for in the called methods
   
    if(triangular_indicator == 1){
        if(DEBUG)printf("Case upper triangular\n");
        backward_substitution_matrix(Q_m, R_m, P_m, n);
        return;
    }else if(triangular_indicator == 2){
        if(DEBUG)printf("Case lower triangular\n");
        forward_substitution_matrix(Q_m, R_m, P_m, n);
        return;
    }else{
        if(DEBUG)printf("Case LU\n");
        LU_solve1(Q_m, R_m, P_m, n);
    }
}
