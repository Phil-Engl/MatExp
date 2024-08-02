
/* ---- eval functions template ---- */

/**
 * @brief eval 3.4 for m=3
 */
void eval3_4_m3(const double* A, const double* A_2, int n, double *P_3, double *Q_3)
{   
    {% if flop_count %}
    FLOP_COUNT_INC(6*n*n, "eval3_4_m3");
    {%- endif %}
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
    {% if flop_count %}   
    FLOP_COUNT_INC(10*n*n, "eval3_4_m5");
    {%- endif %}
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
    {% if flop_count %}   
    FLOP_COUNT_INC(14*n*n, "eval3_4_m7");
    {%- endif %}
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
    {% if flop_count %}   
    FLOP_COUNT_INC(18*n*n, "eval3_4_m9");
    {%- endif %}
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
 * @brief Evaluates equation 3.5
 * 
 * @param A The input matrix A with dimension n x n
 * @param A_2 to A_8: precomputed powers of A
 * @param n The number of rows and columns of A, A_2, A_4 and A_6
 * @param P_13 The output matrix p_13(A) with dimension n x n
 * @param Q_13 The output matrix q_13(A) with dimension n x n
 */
void eval3_5(const double *A, double* A_2, double* A_4, double* A_6, int n, double *P_13, double *Q_13){
    {% if flop_count %}
    FLOP_COUNT_INC(26*n*n, "eval3_5");
    {%- endif %}
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
