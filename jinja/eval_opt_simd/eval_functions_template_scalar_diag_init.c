
/**
 * @brief eval 3.4 for m=3
 */
void eval3_4_m3(const double* A, const double* A_2, int n, double *P_3, double *Q_3)
{   
    FLOP_COUNT_INC(6*n*n, "eval3_4_m3");
    double *U = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *V = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *Temp = (double*) aligned_alloc(32, n*n*sizeof(double));

    __m256d pade_3 = _mm256_set1_pd(pade_coefs[3]);
    __m256d pade_2 = _mm256_set1_pd(pade_coefs[2]);

    __m256d zero_v = _mm256_setzero_pd();

    // compute u and v separately
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j+=4*{{l}}){
            {% for i in range(l) %}
            int idx_{{i}} = i*n + j + 4*{{i}};
            __m256d a_{{i}} = _mm256_load_pd(&A_2[idx_{{i}}]);
            {% endfor %}

            {% for i in range(l) %}
            __m256d u_tmp_{{i}} = _mm256_fmadd_pd(pade_3, a_{{i}}, zero_v);
            __m256d v_tmp_{{i}} = _mm256_fmadd_pd(pade_2, a_{{i}}, zero_v);
            {% endfor %}

            {% for i in range(l) %}
            _mm256_store_pd(&U[idx_{{i}}], u_tmp_{{i}});
            _mm256_store_pd(&V[idx_{{i}}], v_tmp_{{i}});
            {% endfor %}
        }
    }

    for(int i = 0; i < n; i+={{l}}){
        {% for i in range(l) %}
        U[(i+{{i}})*n+i+{{i}}] += pade_coefs[1];
        V[(i+{{i}})*n+i+{{i}}] += pade_coefs[0];
        {% endfor %}
    }

    // one matrix mult to get uneven powers of A
    mmm(A, U, Temp, n, n, n);

    for(int i=0; i<n*n; i+=4*{{l}}){
        {% for i in range(l) %}
        __m256d u_tmp_{{i}} = _mm256_load_pd(&Temp[i + 4*{{i}}]);
        __m256d v_tmp_{{i}} = _mm256_load_pd(&V[i + 4*{{i}}]);
        {% endfor %}

        {% for i in range(l) %}
        __m256d p_tmp_{{i}} = _mm256_add_pd(u_tmp_{{i}}, v_tmp_{{i}});
        __m256d q_tmp_{{i}} = _mm256_sub_pd(v_tmp_{{i}}, u_tmp_{{i}});
        {% endfor %}

        {% for i in range(l) %}
        _mm256_store_pd(&P_3[i + 4*{{i}}], p_tmp_{{i}});
        _mm256_store_pd(&Q_3[i + 4*{{i}}], q_tmp_{{i}});
        {% endfor %}
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
    FLOP_COUNT_INC(10*n*n, "eval3_4_m5");
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
        for(int j = 0; j < n; j+=4*{{l}}){
            {% for i in range(l) %}
            int idx_{{i}} = i*n + j + 4*{{i}};
            __m256d a2_{{i}} = _mm256_load_pd(&A_2[idx_{{i}}]);
            __m256d a4_{{i}} = _mm256_load_pd(&A_4[idx_{{i}}]);
            {% endfor %}

            {% for i in range(l) %}
            __m256d u_tmp_{{i}} = _mm256_fmadd_pd(pade_3, a2_{{i}}, zero_v);
            __m256d v_tmp_{{i}} = _mm256_fmadd_pd(pade_2, a2_{{i}}, zero_v);
            __m256d u_{{i}}     = _mm256_fmadd_pd(pade_5, a4_{{i}}, u_tmp_{{i}});
            __m256d v_{{i}}     = _mm256_fmadd_pd(pade_4, a4_{{i}}, v_tmp_{{i}});
            {% endfor %}

            {% for i in range(l) %}
            _mm256_store_pd(&U[idx_{{i}}], u_{{i}});
            _mm256_store_pd(&V[idx_{{i}}], v_{{i}});
            {% endfor %}
        }
    }

    for(int i = 0; i < n; i+={{l}}){
        {% for i in range(l) %}
        U[(i+{{i}})*n+i+{{i}}] += pade_coefs[1];
        V[(i+{{i}})*n+i+{{i}}] += pade_coefs[0];
        {% endfor %}
    }

    // one matrix mult to get uneven powers of A
    mmm(A, U, Temp, n, n, n);

    for(int i=0; i<n*n; i+=4*{{l}}){
        {% for i in range(l) %}
        __m256d u_tmp_{{i}} = _mm256_load_pd(&Temp[i + 4*{{i}}]);
        __m256d v_tmp_{{i}} = _mm256_load_pd(&V[i + 4*{{i}}]);
        {% endfor %}

        {% for i in range(l) %}
        __m256d p_tmp_{{i}} = _mm256_add_pd(u_tmp_{{i}}, v_tmp_{{i}});
        __m256d q_tmp_{{i}} = _mm256_sub_pd(v_tmp_{{i}}, u_tmp_{{i}});
        {% endfor %}

        {% for i in range(l) %}
        _mm256_store_pd(&P_5[i + 4*{{i}}], p_tmp_{{i}});
        _mm256_store_pd(&Q_5[i + 4*{{i}}], q_tmp_{{i}});
        {% endfor %}
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
    FLOP_COUNT_INC(14*n*n, "eval3_4_m7");
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
        for(int j = 0; j < n; j+=4*{{l}}){
            {% for i in range(l) %}
            int idx_{{i}} = i*n + j + 4*{{i}};
            __m256d a2_{{i}} = _mm256_load_pd(&A_2[idx_{{i}}]);
            __m256d a4_{{i}} = _mm256_load_pd(&A_4[idx_{{i}}]);
            __m256d a6_{{i}} = _mm256_load_pd(&A_6[idx_{{i}}]);
            {% endfor %}

            {% for i in range(l) %}
            __m256d u_tmp1_{{i}} = _mm256_fmadd_pd(pade_3, a2_{{i}}, zero_v);
            __m256d v_tmp1_{{i}} = _mm256_fmadd_pd(pade_2, a2_{{i}}, zero_v);
            __m256d u_tmp2_{{i}} = _mm256_fmadd_pd(pade_5, a4_{{i}}, u_tmp1_{{i}});
            __m256d v_tmp2_{{i}} = _mm256_fmadd_pd(pade_4, a4_{{i}}, v_tmp1_{{i}});
            __m256d u_{{i}}      = _mm256_fmadd_pd(pade_7, a6_{{i}}, u_tmp2_{{i}});
            __m256d v_{{i}}      = _mm256_fmadd_pd(pade_6, a6_{{i}}, v_tmp2_{{i}});
            {% endfor %}

            {% for i in range(l) %}
            _mm256_store_pd(&U[idx_{{i}}], u_{{i}});
            _mm256_store_pd(&V[idx_{{i}}], v_{{i}});
            {% endfor %}
        }
    }

    for(int i = 0; i < n; i+={{l}}){
        {% for i in range(l) %}
        U[(i+{{i}})*n+i+{{i}}] += pade_coefs[1];
        V[(i+{{i}})*n+i+{{i}}] += pade_coefs[0];
        {% endfor %}
    }

    // one matrix mult to get uneven powers of A
    mmm(A, U, Temp, n, n, n);

    for(int i=0; i<n*n; i+=4*{{l}}){
        {% for i in range(l) %}
        __m256d u_tmp_{{i}} = _mm256_load_pd(&Temp[i + 4*{{i}}]);
        __m256d v_tmp_{{i}} = _mm256_load_pd(&V[i + 4*{{i}}]);
        {% endfor %}

        {% for i in range(l) %}
        __m256d p_tmp_{{i}} = _mm256_add_pd(u_tmp_{{i}}, v_tmp_{{i}});
        __m256d q_tmp_{{i}} = _mm256_sub_pd(v_tmp_{{i}}, u_tmp_{{i}});
        {% endfor %}

        {% for i in range(l) %}
        _mm256_store_pd(&P_7[i + 4*{{i}}], p_tmp_{{i}});
        _mm256_store_pd(&Q_7[i + 4*{{i}}], q_tmp_{{i}});
        {% endfor %}
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
    FLOP_COUNT_INC(18*n*n, "eval3_4_m9");
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
        for(int j = 0; j < n; j+=4*{{l}}){
            {% for i in range(l) %}
            int idx_{{i}} = i*n + j + 4*{{i}};
            __m256d a2_{{i}} = _mm256_load_pd(&A_2[idx_{{i}}]);
            __m256d a4_{{i}} = _mm256_load_pd(&A_4[idx_{{i}}]);
            __m256d a6_{{i}} = _mm256_load_pd(&A_6[idx_{{i}}]);
            __m256d a8_{{i}} = _mm256_load_pd(&A_8[idx_{{i}}]);
            {% endfor %}

            {% for i in range(l) %}
            __m256d u_tmp1_{{i}} = _mm256_fmadd_pd(pade_3, a2_{{i}}, zero_v);
            __m256d v_tmp1_{{i}} = _mm256_fmadd_pd(pade_2, a2_{{i}}, zero_v);
            __m256d u_tmp2_{{i}} = _mm256_fmadd_pd(pade_5, a4_{{i}}, u_tmp1_{{i}});
            __m256d v_tmp2_{{i}} = _mm256_fmadd_pd(pade_4, a4_{{i}}, v_tmp1_{{i}});
            __m256d u_tmp3_{{i}} = _mm256_fmadd_pd(pade_7, a6_{{i}}, u_tmp2_{{i}});
            __m256d v_tmp3_{{i}} = _mm256_fmadd_pd(pade_6, a6_{{i}}, v_tmp2_{{i}});
            __m256d u_{{i}}      = _mm256_fmadd_pd(pade_9, a8_{{i}}, u_tmp3_{{i}});
            __m256d v_{{i}}      = _mm256_fmadd_pd(pade_8, a8_{{i}}, v_tmp3_{{i}});
            {% endfor %}

            {% for i in range(l) %}
            _mm256_store_pd(&U[idx_{{i}}], u_{{i}});
            _mm256_store_pd(&V[idx_{{i}}], v_{{i}});
            {% endfor %}
        }
    }

    for(int i = 0; i < n; i+={{l}}){
        {% for i in range(l) %}
        U[(i+{{i}})*n+i+{{i}}] += pade_coefs[1];
        V[(i+{{i}})*n+i+{{i}}] += pade_coefs[0];
        {% endfor %}
    }

    // one matrix mult to get uneven powers of A
    mmm(A, U, Temp, n, n, n);

    for(int i=0; i<n*n; i+=4*{{l}}){
        {% for i in range(l) %}
        __m256d u_tmp_{{i}} = _mm256_load_pd(&Temp[i + 4*{{i}}]);
        __m256d v_tmp_{{i}} = _mm256_load_pd(&V[i + 4*{{i}}]);
        {% endfor %}

        {% for i in range(l) %}
        __m256d p_tmp_{{i}} = _mm256_add_pd(u_tmp_{{i}}, v_tmp_{{i}});
        __m256d q_tmp_{{i}} = _mm256_sub_pd(v_tmp_{{i}}, u_tmp_{{i}});
        {% endfor %}

        {% for i in range(l) %}
        _mm256_store_pd(&P_9[i + 4*{{i}}], p_tmp_{{i}});
        _mm256_store_pd(&Q_9[i + 4*{{i}}], q_tmp_{{i}});
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
    double *A_abs_2 = (double*) aligned_alloc(32, n*n*sizeof(double));


    mmm(A_abs, A_abs, A_abs_2, n, n, n);

    if(m == 3){
        eval3_4_m3(A_abs, A_abs_2, n, P_m_abs, Q_m_abs);
        free(A_abs_2);
        return;
    }

    double *A_abs_4 = (double*) aligned_alloc(32, n*n*sizeof(double));
    mmm(A_abs_2, A_abs_2, A_abs_4, n, n, n);

    if(m == 5){
        eval3_4_m5(A_abs, A_abs_2, A_abs_4, n, P_m_abs, Q_m_abs);
        free(A_abs_2);
        free(A_abs_4);
        return;
    }

    double *A_abs_6 = (double*) aligned_alloc(32, n*n*sizeof(double));
    mmm(A_abs_2, A_abs_4, A_abs_6, n, n, n);

    if(m == 7){
        eval3_4_m7(A_abs, A_abs_2, A_abs_4, A_abs_6, n, P_m_abs, Q_m_abs);
        free(A_abs_2);
        free(A_abs_4);
        free(A_abs_6);
        return;
    }

    double *A_abs_8 = (double*) aligned_alloc(32, n*n*sizeof(double));
    mmm(A_abs_4, A_abs_4, A_abs_8, n, n, n);
    eval3_4_m9(A_abs, A_abs_2, A_abs_4, A_abs_6, A_abs_8, n, P_m_abs, Q_m_abs);

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
    FLOP_COUNT_INC(26*n*n, "eval3_5");
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
    for(int i = 0; i < n*n; i+=4*{{l}}){
        {% for i in range(l) %}
        __m256d a6_i_{{i}} = _mm256_load_pd(&A_6[i+4*{{i}}]);
        __m256d a4_i_{{i}} = _mm256_load_pd(&A_4[i+4*{{i}}]);
        __m256d a2_i_{{i}} = _mm256_load_pd(&A_2[i+4*{{i}}]);
        {% endfor %}

        {% for i in range(l) %}
        __m256d u_tmp0_{{i}} = _mm256_mul_pd(b_13, a6_i_{{i}});
        __m256d v_tmp0_{{i}} = _mm256_mul_pd(b_12, a6_i_{{i}});

        __m256d u_tmp1_{{i}} = _mm256_fmadd_pd(b_11, a4_i_{{i}}, u_tmp0_{{i}});
        __m256d v_tmp1_{{i}} = _mm256_fmadd_pd(b_10, a4_i_{{i}}, v_tmp0_{{i}});

        __m256d u_tmp2_{{i}} = _mm256_fmadd_pd(b_9, a2_i_{{i}}, u_tmp1_{{i}});
        __m256d v_tmp2_{{i}} = _mm256_fmadd_pd(b_8, a2_i_{{i}}, v_tmp1_{{i}});
        {% endfor %}

        {% for i in range(l) %}
        _mm256_store_pd(&U_tmp[i+4*{{i}}], u_tmp2_{{i}});
        _mm256_store_pd(&V_tmp[i+4*{{i}}], v_tmp2_{{i}});
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
    for(int i = 0; i < n*n; i+=4*{{l}}){
        {% for i in range(l) %}
        __m256d a6_i_{{i}} = _mm256_load_pd(&A_6[i+4*{{i}}]);
        __m256d a4_i_{{i}} = _mm256_load_pd(&A_4[i+4*{{i}}]);
        __m256d a2_i_{{i}} = _mm256_load_pd(&A_2[i+4*{{i}}]);
        __m256d u_tmp0_{{i}} = _mm256_load_pd(&U[i+4*{{i}}]);
        __m256d v_tmp0_{{i}} = _mm256_load_pd(&V[i+4*{{i}}]);
        {% endfor %}

        {% for i in range(l) %}
        __m256d u_tmp1_{{i}} = _mm256_fmadd_pd(b_7, a6_i_{{i}}, u_tmp0_{{i}});
        __m256d v_tmp1_{{i}} = _mm256_fmadd_pd(b_6, a6_i_{{i}}, v_tmp0_{{i}});

        __m256d u_tmp2_{{i}} = _mm256_fmadd_pd(b_5, a4_i_{{i}}, u_tmp1_{{i}});
        __m256d v_tmp2_{{i}} = _mm256_fmadd_pd(b_4, a4_i_{{i}}, v_tmp1_{{i}});

        __m256d u_tmp3_{{i}} = _mm256_fmadd_pd(b_3, a2_i_{{i}}, u_tmp2_{{i}});
        __m256d v_tmp3_{{i}} = _mm256_fmadd_pd(b_2, a2_i_{{i}}, v_tmp2_{{i}});
        {% endfor %}

        {% for i in range(l) %}
        _mm256_store_pd(&U[i+4*{{i}}], u_tmp3_{{i}});
        _mm256_store_pd(&V[i+4*{{i}}], v_tmp3_{{i}});
        {% endfor %}
    }

    for(int i = 0; i < n; i++){
            U[i*n+i] += pade_coefs[1];
            V[i*n+i] += pade_coefs[0];
    }

    mmm(A, U, U_tmp, n, n, n); 

    for(int i=0; i < n*n; i+=4*{{l}}){
        {% for i in range(l) %}
        __m256d u_tmp_{{i}} = _mm256_load_pd(&U_tmp[i+4*{{i}}]);
        __m256d v_tmp_{{i}} = _mm256_load_pd(&V[i+4*{{i}}]);
        {% endfor %}

        {% for i in range(l) %}
        __m256d p_tmp_{{i}} = _mm256_add_pd(u_tmp_{{i}}, v_tmp_{{i}});
        __m256d q_tmp_{{i}} = _mm256_sub_pd(v_tmp_{{i}}, u_tmp_{{i}});
        {% endfor %}

        {% for i in range(l) %}
        _mm256_store_pd(&P_13[i+4*{{i}}], p_tmp_{{i}});
        _mm256_store_pd(&Q_13[i+4*{{i}}], q_tmp_{{i}});
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
    double *A_abs_2 = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *A_abs_4 = (double*) aligned_alloc(32, n*n*sizeof(double));
    double *A_abs_6 = (double*) aligned_alloc(32, n*n*sizeof(double));

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
