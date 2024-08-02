/*----- onenorm functions template ----- */

/* ----- global constants ----- */

{% if not dgemm %}
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
            {% if flop_count %}
            FLOP_COUNT_INC(2*4*8, "onenorm_nby2");
            {%- endif %}
        }
        _mm256_store_pd(&C[j], c_0);
        _mm256_store_pd(&C[n + j], c_1);
    }
}
{%- endif %}

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

                {% if flop_count %}
                FLOP_COUNT_INC(4*4, "onenorm_nby2");
                {%- endif %}
                
            }
            flag = flag || (acc_0[0] && acc_0[1] && acc_0[2] && acc_0[3]);
            flag = flag || (acc_1[0] && acc_1[1] && acc_1[2] && acc_1[3]);
            {% if flop_count %}
            FLOP_COUNT_INC(6, "onenorm_nby2");
            {%- endif %}
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
        for(int j = 0; j < m; j += 4 * {{ r }}){
            {% for i in range(r) %}
            __m256d v_{{ i }} = _mm256_load_pd(&v[j + 4 * {{ i }}]); 
            {% endfor %}

            {% for i in range(r) %}
            __m256d ld_{{ i }} = _mm256_load_pd(&A[i * m + j + 4 * {{ i }}]); 
            {% endfor %}
            
            {% for i in range(r) %}
            __m256d cmp_{{ i }} = _mm256_cmp_pd(v_{{ i }}, ld_{{ i }}, _CMP_EQ_OQ);
            {% if flop_count %}
            FLOP_COUNT_INC(4, "column_needs_resampling");
            {%- endif %}
            {% endfor %}

            {% for i in range(r) %}
            acc = _mm256_and_pd(acc, cmp_{{ i }});
            {% if flop_count %}
            FLOP_COUNT_INC(4, "column_needs_resampling");
            {%- endif %}
            {% endfor %}
        }
        if(acc[0] && acc[1] && acc[2] && acc[3]){
            return 1;
        }
    }
    
    for(int i = 0; i < n; i++){
        __m256d acc = _mm256_castsi256_pd(_mm256_set1_epi64x(0xFFFFFFFFFFFFFFFF)); 
        for(int j = 0; j < m; j += 4 * {{ r }}){
            {% for i in range(r) %}
            __m256d v_{{ i }} = _mm256_load_pd(&v[j + 4 * {{ i }}]);
            {% endfor %}

            {% for i in range(r) %}
            __m256d ld_{{ i }} = _mm256_load_pd(&B[i * m + j + 4 * {{ i }}]); 
            {% endfor %}
            
            {% for i in range(r) %}
            __m256d cmp_{{ i }} = _mm256_cmp_pd(v_{{ i }}, ld_{{ i }}, _CMP_EQ_OQ);
            {% if flop_count %}
            FLOP_COUNT_INC(4, "column_needs_resampling");
            {%- endif %}
            {% endfor %}

            {% for i in range(r) %}
            acc = _mm256_and_pd(acc, cmp_{{ i }});
            {% if flop_count %}
            FLOP_COUNT_INC(4, "column_needs_resampling");
            {%- endif %}
            {% endfor %}
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
//no ild here, just disgusting code 
void resample_columns(double *A, double *B, int m, int n){
    for(int i = 0; i < n; i++){
        while(column_needs_resampling(i, &A[i * m], A, B, m, n)){
            for(int j = 0; j < m; j+=4*{{ r }}){ 
                {% for i in range(r) %}
                double rnd_0_{{ i }} = (double)(((rand() % 2) * 2) - 1);
                double rnd_1_{{ i }} = (double)(((rand() % 2) * 2) - 1);
                double rnd_2_{{ i }} = (double)(((rand() % 2) * 2) - 1);
                double rnd_3_{{ i }} = (double)(((rand() % 2) * 2) - 1);
                {% endfor %}
                {% for i in range(r) %}
                __m256d rand_{{ i }} = _mm256_set_pd(rnd_0_{{ i }},rnd_1_{{ i }},rnd_2_{{ i }},rnd_3_{{ i }});
                {% endfor %}
                {% for i in range(r) %}
                _mm256_store_pd(&A[i * m + j + 4*{{ i }}], rand_{{ i }});
                {% endfor %}
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
        for(int j = 0; j < m; j+=4 * {{ r }}){
            {% for i in range((r/2) | int) %}
            __m256d ld_0_{{ 2*i }} = _mm256_load_pd(&A[i * m + j + 4 * {{ 2*i }}]);
            __m256d ld_0_{{ 2*i+1 }} = _mm256_load_pd(&A[i * m + j + 4 * {{ 2*i + 1}}]);
            __m256d ld_1_{{ 2*i }} = _mm256_load_pd(&A[(i + 1) * m + j + 4 * {{ 2*i }}]);
            __m256d ld_1_{{ 2*i+1 }} = _mm256_load_pd(&A[(i + 1) * m + j + 4 * {{ 2*i+1 }}]);
            {% endfor %}
            
            {% for i in range((r/2) | int) %}
            ld_0_{{ 2*i }} = _mm256_andnot_pd(ABS_MASK, ld_0_{{2*i}});
            ld_0_{{ 2*i + 1}} = _mm256_andnot_pd(ABS_MASK, ld_0_{{2*i + 1}});
            ld_1_{{ 2*i }} = _mm256_andnot_pd(ABS_MASK, ld_1_{{ 2*i }});
            ld_1_{{ 2*i+1 }} = _mm256_andnot_pd(ABS_MASK, ld_1_{{ 2*i+1 }});
            {% endfor %}

            {% for i in range((r/2) | int) %}
            __m256d sum_0_{{ i }} = _mm256_add_pd(ld_0_{{ 2*i }}, ld_0_{{ 2*i + 1}});
            __m256d sum_1_{{ i }} = _mm256_add_pd(ld_1_{{ 2*i }}, ld_1_{{ 2*i + 1}});
            {% endfor %}

            {% for i in range((r/2) | int) %}
            acc_0 = _mm256_add_pd(acc_0, sum_0_{{ i }});
            acc_1 = _mm256_add_pd(acc_1, sum_1_{{ i }});
            {% endfor %}

            {% if flop_count %}
            FLOP_COUNT_INC(2*2*4+2*4, "onenorm_best");
            {%- endif %}
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
        {% if flop_count %}
        FLOP_COUNT_INC(8, "onenorm_nby2");
        {%- endif %}
        
        
        
    }
    return max;
}
//same as above, just without the best idx, can be executed with minimal ild with big enough unrolling
double onenorm(const double* A, int m, int n){
    double max = 0.0;
    __m256d ABS_MASK = _mm256_set1_pd(-0.0);
    
    
    {% for i in range(c) %}
    double res_{{ i }} = 0.0;
    {% endfor %}

    for(int i = 0; i < n; i+={{ c }}){ 
        {% for i in range(c) %}
        __m256d acc_{{ i }} = _mm256_set1_pd(0.0); 
        {% endfor %}
        for(int j = 0; j < m; j+=4){
            {% for i in range(c) %}
            __m256d ld_{{ i }} = _mm256_load_pd(&A[(i + {{ i }}) * m + j]);
            {% endfor %}
            
            {% for i in range(c) %}
            ld_{{ i }} = _mm256_andnot_pd(ABS_MASK, ld_{{ i }});
            {% endfor %}
            
            {% for i in range(c) %}
            acc_{{ i }} = _mm256_add_pd(acc_{{ i }}, ld_{{ i }});
            {% endfor %}
            
            {% if flop_count %}
            FLOP_COUNT_INC({{c}}*4*3, "onenorm");
            {%- endif %}
        

        }
        {% for i in range(c) %}
        res_{{ i }} = acc_{{ i }}[0] + acc_{{ i }}[1] + acc_{{ i }}[2] + acc_{{ i }}[3];
        
        {% if flop_count %}
        FLOP_COUNT_INC(3, "onenorm");
        {%- endif %}
        
        {% endfor %}

        {% for i in range(c) %}
        if(res_{{ i }} > max){
            max = res_{{ i }};
        }
        {% endfor %}
        
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
    {% if flop_count %}
    FLOP_COUNT_INC(3, "normest");
    {%- endif %}

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
    
    {% if not dgemm %}
    double* AT = (double*)aligned_alloc(32, n * n * sizeof(double));
    transpose(A, AT, n);
    {%- endif %}

    
    for(int i = 0; i < t; i++){
        for(int j = 0; j < n; j++){
            if(i == 0 || j > i){
                X[i * n + j] = x_elem;
            } else {
                X[i * n + j] = m_x_elem;
            }
             S[i*n+j] = 0;
        }
       
    }
    

    srand(time(0));
    while(1){
        //Y = A * X
        if(k == 1){
            {% if not dgemm %}
            mmm_nby2(A, X, Y, n); 
            {%- endif %}
            {% if dgemm %}
            cblas_dgemm(layout, CblasNoTrans, CblasNoTrans, n, t, n ,alpha ,A, n ,X ,n ,beta, Y, n);
            {%- endif %}
        }else{
            for(int i = 0; i < n; i+=4){
                __m256d ld_0 = _mm256_load_pd(&A[ind[0] * n + i]);
                __m256d ld_1 = _mm256_load_pd(&A[ind[1] * n + i]);
                _mm256_store_pd(&Y[i], ld_0);
                _mm256_store_pd(&Y[n + i], ld_1);
            }
        }
        est = onenorm_best(Y,n,t, &best_j); 
        {% if flop_count %}
        FLOP_COUNT_INC(1, "normest");
        {%- endif %}
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
        {% if flop_count %}
        FLOP_COUNT_INC(n*t, "normest");
        {%- endif %}  
        for(int i = 0; i < n * t; i+=4*{{ r * 2}}){ //e.g. if 4 r*2 = 8, all at  once
            //S[i] = Y[i] >= 0.0 ? 1.0 : -1.0;
            {% for i in range(r) %}
            __m256d ld_{{ i }} = _mm256_load_pd(&Y[i + 4 * {{ i }}]);
            {% endfor %}

            {% for i in range(r) %}
            __m256d cmp_{{ i }} = _mm256_cmp_pd(ld_{{ i }}, ZERO, _CMP_GE_OQ);
            {% endfor %}

            {% for i in range(r) %}
            __m256d blend_{{ i }} =  _mm256_blendv_pd(MONE, ONE, cmp_{{ i }});
            {% endfor %}

            {% for i in range(r) %}
            _mm256_store_pd(&S[i + 4*{{ i }}], blend_{{ i }});
            {% endfor %}
        }

        if(check_all_columns_parallel(S, S_old, n, t)){
            break;
        }

        /* Ensure that no column of S is parallel to another column of S
        or to a column of S_old by replacing columns of S by rand{−1, 1}. */
        resample_columns(S, S_old, n, t);

        //(3)
        {% if not dgemm %}
        mmm_nby2(AT, S, Z, n); 
        {%- endif %}
        {% if dgemm %}
        cblas_dgemm(layout, CblasTrans, CblasNoTrans, n, t, n, alpha, A, n, S, n, beta, Z, n);
        {%- endif %}

        //only unroll once for n <= 4!!!!
        {% if flop_count %}
        FLOP_COUNT_INC(n*t*3, "normest");
        {%- endif %}
        
        for(int i = 0; i < n; i+=4 * {{ r }}){ // unroll here
            //unrolled inner loop completely, t = 2
            {% for i in range(r) %}
            __m256d ld_{{ i }}_0 = _mm256_load_pd(&Z[i + 4 * {{ i }}]);
            __m256d ld_{{ i }}_1 = _mm256_load_pd(&Z[n + i + 4 * {{ i }}]);
            {% endfor %}
            {% for i in range(r) %}
            ld_{{ i }}_0 = _mm256_andnot_pd(ABS_MASK, ld_{{ i }}_0);
            ld_{{ i }}_1 = _mm256_andnot_pd(ABS_MASK, ld_{{ i }}_1);
            {% endfor %}
            {% for i in range(r) %}
            __m256d acc_{{ i }} = _mm256_max_pd(ld_{{ i }}_0, ld_{{ i }}_1);
            {% endfor %}
            {% for i in range(r) %}
            _mm256_store_pd(&h[i + 4 * {{ i }}], acc_{{ i }});
            {% endfor %}

        }

        max_h = 0.0;
        for(int i = 0; i < n; i++){
            if(h[i] > max_h){
                max_h = h[i];
            }
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
            {% if flop_count %}
            FLOP_COUNT_INC(2*n, "normest");
            {%- endif %}
            if(h[i] > snd_in || h[i] > snd_out){
                if(idx_in_hist(i, ind_hist, hist_len)){
                    {% if flop_count %}
                    FLOP_COUNT_INC(1, "normest");
                    {%- endif %}
                    if(h[i] >= fst_in){
                        snd_in = fst_in;
                        fst_in = h[i];
                        fst_in_idx = i;
                    }else if(h[i] > snd_in){
                        {% if flop_count %}
                        FLOP_COUNT_INC(1, "normest");
                        {%- endif %}
                        snd_in = h[i];
                    }
                }else{
                    {% if flop_count %}
                    FLOP_COUNT_INC(1, "normest");
                    {%- endif %}
                    if(h[i] >= fst_out){
                        snd_out = fst_out;
                        fst_out = h[i];
                        snd_out_idx = fst_out_idx;
                        fst_out_idx = i;
                        out_ctr++;
                    }else if(h[i] > snd_out){
                        {% if flop_count %}
                        FLOP_COUNT_INC(1, "normest");
                        {%- endif %}
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
    {% if not dgemm %}
    free(AT);
    {%- endif %}

    return est;
}