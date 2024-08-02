/*----- onenorm functions template ----- */

/* ----- global constants ----- */

struct h_tmp
{
    double val;
    int idx;
};

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

int check_all_columns_parallel(const double* A, const double* B, int m, int n){
    for(int i = 0; i < n; i++){
        int flag = 0;
        for(int j = 0; j < n; j+={{ c }}){
            {% for i in range(c) %}
            __m256d acc_{{ i }} = _mm256_castsi256_pd(_mm256_set1_epi64x(0xFFFFFFFFFFFFFFFF)); 
            {% endfor %}
            for(int k = 0; k < m; k+=4){
                __m256d a = _mm256_load_pd(&A[i*m+k]);
                {% for i in range(c) %}
                __m256d b_{{ i }} = _mm256_load_pd(&B[(j + {{ i }})*m+k]);
                {% endfor %}
                {% for i in range(c) %}
                __m256d cmp_{{ i }} = _mm256_cmp_pd(a, b_{{ i }}, _CMP_EQ_OQ);
                {% endfor %}
                {% for i in range(c) %}
                acc_{{ i }} = _mm256_and_pd(acc_{{ i }}, cmp_{{ i }});
                {% endfor %}
            }
            {% for i in range(c) %}
            flag = flag || (acc_{{ i }}[0] && acc_{{ i }}[1] && acc_{{ i }}[2] && acc_{{ i }}[3]);
            {% endfor %}
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
            {% endfor %}

            {% for i in range(r) %}
            acc = _mm256_and_pd(acc, cmp_{{ i }});
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
            {% endfor %}

            {% for i in range(r) %}
            acc = _mm256_and_pd(acc, cmp_{{ i }});
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
 * @brief compare function for qsort
 * 
 * @param a struct h_tmp* a
 * @param b struct h_tmp* b
 * @return int -1 if (*a).val > (*b).val, -1 if (*a).val < (*b).val, 0 otherwise.
 */
int cmp_h(const void *a, const void *b){
    struct h_tmp* a1 = (struct h_tmp*)a;
    struct h_tmp* b1 = (struct h_tmp*)b;
    if((*a1).val > (*b1).val){
        return -1;
    } else if ((*a1).val < (*b1).val){
        return 1;
    } else {
        return 0;
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
    int flag = 0;
    for(int i = 0; i < hist_len; i++){
        flag = flag || (idx == hist[i]);
    }
    return flag;
}




/* ----- one norm functions ----- */
//TODO: this one might be better if vectorized in the other direction, especially if t  = 1
double onenorm_best(const double* A, int m, int n, int* max_idx){
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
        }
        {% for i in range(c) %}
        res_{{ i }} = acc_{{ i }}[0] + acc_{{ i }}[1] + acc_{{ i }}[2] + acc_{{ i }}[3];
        {% endfor %}

        {% for i in range(c) %}
        if(res_{{ i }} > max){
            max = res_{{ i }};
            *max_idx = i + {{ i }};
        }
        {% endfor %}
        
    }
    return max;
}

//same as above, just without the best idx
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
        }
        {% for i in range(c) %}
        res_{{ i }} = acc_{{ i }}[0] + acc_{{ i }}[1] + acc_{{ i }}[2] + acc_{{ i }}[3];
        {% endfor %}

        {% for i in range(c) %}
        if(res_{{ i }} > max){
            max = res_{{ i }};
        }
        {% endfor %}
        
    }
    return max;
}

//use this for matrices with only positive entries
double onenorm_abs_mat(const double* A, int m, int n){
    double max = 0.0;
    
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
            acc_{{ i }} = _mm256_add_pd(acc_{{ i }}, ld_{{ i }});
            {% endfor %}
        }
        {% for i in range(c) %}
        res_{{ i }} = acc_{{ i }}[0] + acc_{{ i }}[1] + acc_{{ i }}[2] + acc_{{ i }}[3];
        {% endfor %}

        {% for i in range(c) %}
        if(res_{{ i }} > max){
            max = res_{{ i }};
        }
        {% endfor %}
        
    }
    return max;
}

double normest(const double* A, int n){
    int t = 2;
   
    int itmax = 5;

    int k = 1;
    int best_j = 0;
    int ind_best = 0;
    int hist_len = 0;
    int new_length = t;

    double est = 0.0;
    double est_old = 0.0;
    
    double max_h = 0.0;
    int max_h_ind = 0;
    double x_elem = 1.0 / (double)n;
    double m_x_elem = -1.0 * x_elem;

    __m256d ZERO = _mm256_set1_pd(0.0);
    __m256d ONE = _mm256_set1_pd(1.0);
    __m256d MONE = _mm256_set1_pd(-1.0);
    __m256d ABS_MASK = _mm256_set1_pd(-0.0);
    
    int* ind_hist = (int*)aligned_alloc(32, t * itmax * sizeof(int));
    int* ind = (int*)aligned_alloc(32, n * sizeof(int));
    int* ind_in = (int*)aligned_alloc(32, n * sizeof(int));
    
    //double* AT = (double*)malloc(n * n * sizeof(double)); //TODO use precomputed if not dgemm, use transpA if dgemm
    double* S = (double*)aligned_alloc(32, n * t * sizeof(double));
    double* S_old = (double*)aligned_alloc(32, n * t * sizeof(double));
    double* X = (double*)aligned_alloc(32, n * t * sizeof(double));
    double* Y = (double*)aligned_alloc(32, n * t * sizeof(double));
    double* Z = (double*)aligned_alloc(32, n * t * sizeof(double));

    double* h = (double*)aligned_alloc(32, n  * sizeof(double));
    struct h_tmp* h_str = (struct h_tmp*)aligned_alloc(32, n * sizeof(struct h_tmp));
    
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
            S[i*n+j] = 0.0;
        }
    }
    

    srand(time(0));
    while(1){
        //Y = A * X
        {% if not dgemm %}
        mmm(A, X, Y, n, n, t); 
        {%- endif %}
        {% if dgemm %}
        cblas_dgemm(layout, CblasNoTrans, CblasNoTrans, n, t, n ,alpha ,A, n ,X ,n ,beta, Y, n);
        {%- endif %}
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
                   
        for(int i = 0; i < n * t; i+=4*{{ r }}){ //this works for any possible t
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
        mmm(AT, S, Z, n, n, t); //TODO mmm with transpA!
        {%- endif %}
        {% if dgemm %}
        cblas_dgemm(layout, CblasTrans, CblasNoTrans, n, t, n, alpha, A, n, S, n, beta, Z, n);
        {%- endif %}

        //only unroll once for n <= 4!!!!
        max_h = 0.0;
        max_h_ind = 0;
        for(int i = 0; i < n; i+=4 * {{ r }}){ // unroll here
            {% for i in range(r) %}
            __m256d acc_{{ i }} = _mm256_set1_pd(0.0);
            {% endfor %}
            for(int j = 0; j < t; j++){
                {% for i in range(r) %}
                __m256d ld_{{ i }} = _mm256_load_pd(&Z[j * n + i + 4 * {{ i }}]);
                {% endfor %}
                {% for i in range(r) %}
                ld_{{ i }} = _mm256_andnot_pd(ABS_MASK, ld_{{ i }});
                {% endfor %}
                {% for i in range(r) %}
                acc_{{ i }} = _mm256_max_pd(acc_{{ i }}, ld_{{ i }});
                {% endfor %}
            }
            {% for i in range(r) %}
            for(int k = 0; k < 4; k ++){
                //max_h = (acc_{{ i }}[k] > max_h) ? acc_{{ i }}[k] : max_h;
                if(acc_{{ i }}[k] > max_h){
                    max_h = acc_{{ i }}[k];
                    max_h_ind = i + 4 * {{ i }} + k;
                }


            }
            {% endfor %}
            {% for i in range(r) %}
            _mm256_store_pd(&h[i + 4 * {{ i }}], acc_{{ i }});
            {% endfor %}
        }

        //(4)
        if(k >= 2 && max_h == h[ind_best]){
            break;
        }

        // from here on, the code is not vectorizable anymore
        // the only possible speedup is to use t = 1 always which would make everything
        // very simple. On the other hand, the code is still quite fast because we're only working on
        // vectors of length max n. 
        for(int i = 0; i < n; i++){
            h_str[i].val = h[i];
            h_str[i].idx = i;
        }

        
        //Sort h so that h[0] ≥ · · · ≥ h[n-1] and re-order ind correspondingly.
        qsort(h_str,n, sizeof(struct h_tmp), cmp_h);
        
        //(5) 
        /*  
         *  If ind(0:t) is contained in ind_hist, break
         *  Replace ind(0:t) by the first t indices in ind(0:n) that are
         *  not in ind_hist. 
        */
        
            
        int flag = 1;
        int i_in_hist = 0;
        int outidx = 0;
        int inidx = 0;
        for(int i = 0; i < n; i++){ 
            i_in_hist = idx_in_hist(h_str[i].idx, ind_hist, hist_len);
            flag = flag && i_in_hist;
            if(i == t-1 && flag){
                break;
            }
            if(!i_in_hist){
                ind[outidx++] = h_str[i].idx;
            }else{
                ind_in[inidx++] = h_str[i].idx;
            }
        }
        
        if(flag){
            break;
        }
        
        new_length = outidx;
        for(int i = 0; i < n; i++){
            if(outidx == n){
                break;
            }
            ind[outidx++] = ind_in[i];
        }
         /*  
         *  create a new X (n x t) matrix with the unit vectors from 
         *  the best new indices 
         */
        for(int i = 0; i < n * t; i+=4 * {{ r }}){
            
            {% for i in range(r) %}
            _mm256_store_pd(&X[i + 4 * {{i}}], ZERO);
            {% endfor %}
        }

        for(int i = 0; i < t; i++){
            X[i * n + ind[i]] = 1.0;
        }

        
        int nhlen = hist_len + t;
        for(int i = 0; i < nhlen; i++){
            if(i == new_length){
                break;
            }
            ind_hist[hist_len++] = ind[i];
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
    free(h_str);
    {% if not dgemm %}
    free(AT);
    {%- endif %}

    return est;
}
