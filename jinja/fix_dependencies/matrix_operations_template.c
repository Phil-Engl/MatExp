
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
{% if flop_count %}
    FLOP_COUNT_INC(2*common*rowsA*colsB, "mmm");
{%- endif %}
{% if dgemm %}
    const double alpha = 1.0;
    const double beta = 0.0;
    CBLAS_LAYOUT layout = CblasColMajor;
    CBLAS_TRANSPOSE tA = CblasNoTrans;
    CBLAS_TRANSPOSE tB = CblasNoTrans;
    cblas_dgemm(layout, tA, tB, rowsA, colsB, common ,alpha ,A, rowsA ,B ,common ,beta, C, rowsA);

{%- endif %}
{% if not dgemm %}
    
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
    }else{ // fallback
        for(int i = 0; i < colsB; i++){
            for(int j = 0; j < rowsA; j++){
                C[i * rowsA + j] = 0.0;
                for(int k = 0;k < common; k++){
                    C[i * rowsA + j] += A[k * rowsA + j] * B[k + common * i];
                }
            }
        }
    }
{%- endif %}
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
    {% if flop_count %}
    FLOP_COUNT_INC(2*n*m, "mm_add");
    {%- endif %}
    __m256d alph = _mm256_set1_pd(alpha);
    {% for i in range(l) %}
    __m256d a_{{ i }};
    __m256d b_{{ i }};
    __m256d c_{{ i }};
    {% endfor %}
    for(int i = 0 ; i < m*n; i+=4 * {{ l }}){
        {% for i in range(l) %}
        a_{{ i }} =_mm256_load_pd(&A[i + ({{ i }} * 4)]);
        {% endfor %}
        
        {% for i in range(l) %}
        b_{{ i }} = _mm256_load_pd(&B[i + ({{ i }} * 4)]);
        {% endfor %}
        
        {% for i in range(l) %}
        c_{{ i }} = _mm256_fmadd_pd(alph, a_{{ i }}, b_{{ i }});
        {% endfor %}
        
        {% for i in range(l) %}
        _mm256_store_pd(&C[i + ({{ i }} * 4)], c_{{ i }});
        {% endfor %}
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
    {% if flop_count %}
    FLOP_COUNT_INC(n*m, "scalar_matrix_mult");
    {%- endif %}
    __m256d alph = _mm256_set1_pd(alpha);
    {% for i in range(l) %}
    __m256d temp_{{ i }};
    {% endfor %}
    for (int i = 0 ; i < m*n ; i+= 4*{{ l }}){
        {% for i in range(l) %}
        temp_{{ i }} = _mm256_load_pd(&A[i + ({{ i }}* 4)]);
        {% endfor %}
       
        {% for i in range(l) %}
        temp_{{ i }} = _mm256_mul_pd(alph, temp_{{ i }});
        {% endfor %}
        
        {% for i in range(l) %}
        _mm256_store_pd(&C[i + ({{ i }}* 4)], temp_{{ i }});
        {% endfor %}
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
    {% if flop_count %}
    FLOP_COUNT_INC(n*m, "mat_abs");
    {%- endif %}
    __m256d sign_mask = _mm256_set1_pd(-0.0); // Set the sign bit to 1
    {% for i in range(l) %}
    __m256d a_{{ i }};
    {% endfor %}
    for(int i = 0 ; i < m*n ; i+= 4 * {{ l }}){
        {% for i in range(l) %}
        a_{{ i }} = _mm256_load_pd(&A[i + ({{ i }}*4)]);
        {% endfor %}

        {% for i in range(l) %}
        a_{{ i }} = _mm256_andnot_pd(sign_mask, a_{{ i }});
        {% endfor %}
        
        {% for i in range(l) %}
        _mm256_store_pd(&B[i + ({{ i }} * 4)], a_{{ i }});
        {% endfor %}
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
            {% if flop_count %}
            FLOP_COUNT_INC(1, "is_lower_triangular");
            {%- endif %}
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
            {% if flop_count %}
            FLOP_COUNT_INC(1, "is_upper_triangular");
            {%- endif %}
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


{% if not dgemm %}
void transpose(const double *A, double *B, int n){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            B[j * n + i] = A[i * n + j];
        }
    }
}
{%- endif %}

/**
 * @brief find the highest row sum, assumes matrix has only positive values
 * @param A input matrix
 * @param m number of rows
 * @param n number of columns
 * @return the double containing the highest row sum
*/
//probably needs some sort of a different thingy for cache optimization
double infinity_norm(const double* A, int m, int n){
    {% if flop_count %}
    FLOP_COUNT_INC(2*n*m+m, "infinity_norm");
    {%- endif %}
    __m256d ABS_MASK = _mm256_set1_pd(-0.0);
    double max = 0.0;
    __m256d max_val = _mm256_set1_pd(0.0);
    for(int i = 0; i < m; i+=4 * {{ r }}){
        {% for i in range(r) %}
        __m256d sum_{{ i }} = _mm256_set1_pd(0.0);
        {% endfor %}
        for(int j = 0; j < n; j+={{ c }}){ //c must be multiple of 2!
            {% for i in range(r) %}
            {% for j in range((c/2) | int) %}
            __m256d ld_{{ i }}_{{ j*2 }} = _mm256_load_pd(&A[(j +{{ j*2 }}) * m + i + 4 * {{ i }}]);
            __m256d ld_{{ i }}_{{ j*2+1 }} = _mm256_load_pd(&A[(j +{{ j*2+1 }}) * m + i + 4 * {{ i }}]);
            {% endfor %}
            {% endfor %}

            {% for i in range(r) %}
            {% for j in range((c/2) | int) %}
            ld_{{ i }}_{{ j*2 }} = _mm256_andnot_pd(ABS_MASK, ld_{{ i }}_{{ j*2 }});
            ld_{{ i }}_{{ j*2+1 }} = _mm256_andnot_pd(ABS_MASK, ld_{{ i }}_{{ j*2+1 }});
            {% endfor %}
            {% endfor %}

            {% for i in range(r) %}
            {% for j in range((c/2) | int) %}
            __m256d sum_{{ i }}_{{ j }} = _mm256_add_pd(ld_{{ i }}_{{ j*2 }}, ld_{{ i }}_{{ j*2+1 }});
            {% endfor %}
            {% endfor %}
        
            {% for i in range(r) %}
            {% for j in range((c/2) | int ) %}
            sum_{{ i }} = _mm256_add_pd(sum_{{ i }}, sum_{{ i }}_{{ j }});
            {% endfor %}
            {% endfor %}
        }
        {% for i in range(r) %}
        max_val = _mm256_max_pd(max_val, sum_{{ i }});
        {% endfor %}
    }
    for(int i = 0; i < 4; i++){
        max = max_val[i] > max ? max_val[i] : max;
    }
    return max;
}



/**
 * @brief sum of the of the columns of matrix A
 * @param A input matrix
 * @param n column and row size
 * @param out output vector
*/
//no terrible inter loop dependencies here
void mat_col_sum(const double* A, int n, double *out){
    {% if flop_count %}
    FLOP_COUNT_INC(n*n, "mat_col_sum");
    {%- endif %}

    {% for i in range(c) %}
    double res_{{ i }} = 0.0;
    {% endfor %}
    for(int i = 0; i < n; i+={{ c }}){ 
        {% for i in range(c) %}
        __m256d acc_{{ i }} = _mm256_set1_pd(0.0); 
        {% endfor %}
        for(int j = 0; j < n; j+=4){
            {% for i in range(c) %}
            __m256d ld_{{ i }} = _mm256_load_pd(&A[(i + {{ i }}) * n + j]);
            {% endfor %}
            {% for i in range(c) %}
            acc_{{ i }} = _mm256_add_pd(acc_{{ i }}, ld_{{ i }});
            {% endfor %}
        }
        {% for i in range(c) %}
        res_{{ i }} = acc_{{ i }}[0] + acc_{{ i }}[1] + acc_{{ i }}[2] + acc_{{ i }}[3];
        {% endfor %}
        {% for i in range(c) %}
        out[i+ {{ i }}] = res_{{ i }};
        {% endfor %}
        
    }
}

void fill_diagonal_matrix(double* A, double diag, int n){
    // FLOP_COUNT_INC(0, "fill_diagonal_matrix");
    __m256d ZERO = _mm256_set1_pd(0.0);
    for(int i = 0; i < n*n; i+=4 * {{ l }}){
        {% for i in range(l) %}
        _mm256_store_pd(&A[i + 4 * {{ i }}], ZERO);
        {% endfor %}
    }

    for(int i = 0; i < n; i++){
        A[i * n + i] = diag;
    }
}




void forward_substitution_LU(double * A, double *y, double * b, int n){
{% if loop_order %} // ikj
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

                {% if flop_count %}
                    FLOP_COUNT_INC(4*4*2, "forward_substitution_LU");
                {%-endif%}
                
            }
            //CLEANUP LOOP FOR J
            for(; j<i; j++){
                __m256d A_vec = _mm256_set1_pd(A[j*n+i]);
                __m256d y_vec = _mm256_i64gather_pd(&y[k*n+j], vindex, 8);
                sum_vec = _mm256_fmadd_pd(A_vec, y_vec, sum_vec);
                {% if flop_count %}
                    FLOP_COUNT_INC(4*2, "forward_substitution_LU");
                {%-endif%}
                
            }
            
            __m256d b_vec = _mm256_i64gather_pd(&b[k*n+i], vindex, 8);

            __m256d tmp_final_sum = _mm256_add_pd(sum_vec3, sum_vec2);
            __m256d tmp_final_sum1 = _mm256_add_pd(sum_vec1, sum_vec);
            __m256d final_sum = _mm256_add_pd(tmp_final_sum1, tmp_final_sum);


            __m256d res_vec = _mm256_sub_pd(b_vec, final_sum);

            {% if flop_count %}
                FLOP_COUNT_INC(4*4, "forward_substitution_LU");
            {%-endif%}

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
                {% if flop_count %}
                    FLOP_COUNT_INC(4*2, "forward_substitution_LU");
                {%-endif%}
            }
            for(; j<i; j++){
                sum += A[j*n+i] * y[k*n + j];
                {% if flop_count %}
                    FLOP_COUNT_INC(2, "forward_substitution_LU");
                {%-endif%}
            }
            y[k*n+i] = b[k*n+i] - sum;
            {% if flop_count %}
                FLOP_COUNT_INC(1, "forward_substitution_LU");
            {%-endif%}
        }
    }   


{%- endif %}
{% if not loop_order %} // kij
    int k;
    int j;
   __m256i vindex = _mm256_set_epi64x(0, n, 2*n, 3*n);
    for(k = 0; k<n-3; k+=4){
        for(int i=0; i<n; i++){
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

                {% if flop_count %}
                    FLOP_COUNT_INC(4*4*2, "forward_substitution_LU");
                {%-endif%}
                
            }
            //CLEANUP LOOP FOR J
            for(; j<i; j++){
                __m256d A_vec = _mm256_set1_pd(A[j*n+i]);
                __m256d y_vec = _mm256_i64gather_pd(&y[k*n+j], vindex, 8);
                sum_vec = _mm256_fmadd_pd(A_vec, y_vec, sum_vec);
                {% if flop_count %}
                    FLOP_COUNT_INC(4*2, "forward_substitution_LU");
                {%-endif%}
                
            }
            
            __m256d b_vec = _mm256_i64gather_pd(&b[k*n+i], vindex, 8);

            __m256d tmp_final_sum = _mm256_add_pd(sum_vec3, sum_vec2);
            __m256d tmp_final_sum1 = _mm256_add_pd(sum_vec1, sum_vec);
            __m256d final_sum = _mm256_add_pd(tmp_final_sum1, tmp_final_sum);

            __m256d res_vec = _mm256_sub_pd(b_vec, final_sum);

            {% if flop_count %}
                FLOP_COUNT_INC(4*4, "forward_substitution_LU");
            {%-endif%}

            y[k*n+i] = res_vec[3];
            y[(k+1)*n+i] = res_vec[2];
            y[(k+2)*n+i] = res_vec[1];
            y[(k+3)*n+i] = res_vec[0];
            
        }
    }

        //CLEANUP LOOP FOR K
    for(; k<n; k++){
        for(int i=0; i<n; i++){
            double sum = 0;
            for(j=0; j<i-3; j+=4){
                sum += A[j*n+i] * y[k*n + j];
                sum += A[(j+1)*n+i] * y[k*n + j +1];
                sum += A[(j+2)*n+i] * y[k*n + j +2];
                sum += A[(j+3)*n+i] * y[k*n + j +3];
                {% if flop_count %}
                    FLOP_COUNT_INC(4*2, "forward_substitution_LU");
                {%-endif%}
            }
            for(; j<i; j++){
                sum += A[j*n+i] * y[k*n + j];
                {% if flop_count %}
                    FLOP_COUNT_INC(2, "forward_substitution_LU");
                {%-endif%}
            }
            y[k*n+i] = b[k*n+i] - sum;
            {% if flop_count %}
                FLOP_COUNT_INC(1, "forward_substitution_LU");
            {%-endif%}
        }
    }   

{%- endif %}
}

void backward_substitution(double * L, double *x, double * b, int n){
{% if loop_order %} // ikj
    int j;
    int k;
    int i;
    __m256i vindex = _mm256_set_epi64x(0, n, 2*n, 3*n);

    for(i = n-1; i>=0; i--){
        double rezi = 1.0/L[i*n+i];
        {% if flop_count %}
            FLOP_COUNT_INC(1, "backward_substitution");
        {%- endif %}

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

                {% if flop_count %}
                FLOP_COUNT_INC(4*4*2, "backward_substitution");
                {%- endif %}
            }

            for(; j<n; j++){      
                __m256d L_vec = _mm256_set1_pd(L[j*n+i]);
                __m256d x_vec = _mm256_i64gather_pd(&x[k*n+j], vindex, 8);
                sum_vec = _mm256_fmadd_pd(L_vec, x_vec, sum_vec); 
                {% if flop_count %}
                FLOP_COUNT_INC(4*2, "backward_substitution");
                {%- endif %}
            }

            __m256d b_vec = _mm256_i64gather_pd(&b[k*n+i], vindex, 8);

            __m256d tmp_final_sum = _mm256_add_pd(sum_vec3, sum_vec2);
            __m256d tmp_final_sum1 = _mm256_add_pd(sum_vec1, sum_vec);
            __m256d final_sum = _mm256_add_pd(tmp_final_sum1, tmp_final_sum);

            __m256d tmp_res_vec = _mm256_sub_pd(b_vec, final_sum);
            __m256d res_vec = _mm256_mul_pd(tmp_res_vec, rezi_vec);

            {% if flop_count %}
                FLOP_COUNT_INC(5*4, "backward_substitution");
            {%- endif %}

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

                {% if flop_count %}
                FLOP_COUNT_INC(4*2, "backward_substitution");
                {%- endif %}
            }

            for(; j<n; j++){
                sum += L[j*n+i] * x[k*n+j];
                {% if flop_count %}
                FLOP_COUNT_INC(2, "backward_substitution");
                {%- endif %}
            }
            x[k*n + i] = b[k*n+i] - sum;
            x[k*n + i] *= rezi;   
            {% if flop_count %}
                FLOP_COUNT_INC(2, "backward_substitution");
            {%- endif %}
        }
    }

{%- endif %}
{% if not loop_order %} // kij
      
    int j;
    int k;
    int i;
    __m256i vindex = _mm256_set_epi64x(0, n, 2*n, 3*n);

    for(k = 0; k<n-3; k+=4){
        for(i = n-1; i>=0; i--){
            double rezi = 1.0/L[i*n+i];
            __m256d rezi_vec = _mm256_set1_pd(rezi);
            {% if flop_count %}
                FLOP_COUNT_INC(1, "backward_substitution");
            {%- endif %}

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

                {% if flop_count %}
                FLOP_COUNT_INC(4*4*2, "backward_substitution");
                {%- endif %}
            }

            for(; j<n; j++){   
                __m256d L_vec = _mm256_set1_pd(L[j*n+i]);
                __m256d x_vec = _mm256_i64gather_pd(&x[k*n+j], vindex, 8);
                sum_vec = _mm256_fmadd_pd(L_vec, x_vec, sum_vec);
                {% if flop_count %}
                FLOP_COUNT_INC(4*2, "backward_substitution");
                {%- endif %}
            }

            __m256d b_vec = _mm256_i64gather_pd(&b[k*n+i], vindex, 8);

            __m256d tmp_final_sum = _mm256_add_pd(sum_vec3, sum_vec2);
            __m256d tmp_final_sum1 = _mm256_add_pd(sum_vec1, sum_vec);
            __m256d final_sum = _mm256_add_pd(tmp_final_sum1, tmp_final_sum);

            __m256d tmp_res_vec = _mm256_sub_pd(b_vec, final_sum);
            __m256d res_vec = _mm256_mul_pd(tmp_res_vec, rezi_vec);

            {% if flop_count %}
                FLOP_COUNT_INC(5*4, "backward_substitution");
            {%- endif %}

            x[k*n+i] = res_vec[3];
            x[(k+1)*n+i] = res_vec[2];
            x[(k+2)*n+i] = res_vec[1];
            x[(k+3)*n+i] = res_vec[0];
                
        }
    }

    for(; k<n; k++){
        for(i = n-1; i>=0; i--){
            double rezi = 1.0/L[i*n+i];
            //__m256d rezi_vec = _mm256_set1_pd(rezi);
            {% if flop_count %}
                FLOP_COUNT_INC(1, "backward_substitution");
            {%- endif %}
            double sum = 0;

            for(j = i+1; j<n-3; j+=4){
                sum += L[j*n+i] * x[k*n+j];
                sum += L[(j+1)*n+i] * x[k*n+j +1];
                sum += L[(j+2)*n+i] * x[k*n+j +2];
                sum += L[(j+3)*n+i] * x[k*n+j +3];

                {% if flop_count %}
                FLOP_COUNT_INC(4*2, "backward_substitution");
                {%- endif %}
            }

            for(; j<n; j++){
                sum += L[j*n+i] * x[k*n+j];
                {% if flop_count %}
                FLOP_COUNT_INC(2, "backward_substitution");
                {%- endif %}
            }

            x[k*n + i] = b[k*n+i] - sum;
            x[k*n + i] *= rezi;
            {% if flop_count %}
                FLOP_COUNT_INC(2, "backward_substitution");
            {%- endif %}
                
        }
    }
{%- endif %}
}

void forward_substitution(double * A, double *y, double * b, int n){ // with div here
{% if loop_order %} // ikj

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

                {% if flop_count %}
                FLOP_COUNT_INC(4*4*2, "forward_substitution");
                {%- endif %}
                
            }
            //CLEANUP LOOP FOR J
            for(; j<i; j++){
                __m256d A_vec = _mm256_set1_pd(A[j*n+i]);
                __m256d y_vec = _mm256_i64gather_pd(&y[k*n+j], vindex, 8);
                sum_vec = _mm256_fmadd_pd(A_vec, y_vec, sum_vec);

                {% if flop_count %}
                FLOP_COUNT_INC(4*2, "forward_substitution");
                {%- endif %}
                
            }
            
            __m256d b_vec = _mm256_i64gather_pd(&b[k*n+i], vindex, 8);

            __m256d tmp_final_sum = _mm256_add_pd(sum_vec3, sum_vec2);
            __m256d tmp_final_sum1 = _mm256_add_pd(sum_vec1, sum_vec);
            __m256d final_sum = _mm256_add_pd(tmp_final_sum1, tmp_final_sum);


            __m256d tmp_res_vec = _mm256_sub_pd(b_vec, final_sum);
            __m256d res_vec = _mm256_mul_pd(tmp_res_vec, rezi_vec);

            {% if flop_count %}
                FLOP_COUNT_INC(5*4, "forward_substitution");
            {%- endif %}


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

                {% if flop_count %}
                    FLOP_COUNT_INC(4*2, "forward_substitution");
                {%- endif %}
            }
            for(; j<i; j++){
                sum += A[j*n+i] * y[k*n + j];
                {% if flop_count %}
                FLOP_COUNT_INC(2, "forward_substitution");
                {%- endif %}
            }
            y[k*n+i] = b[k*n+i] - sum;
            y[k*n+i] *= rezi; 

            {% if flop_count %}
                FLOP_COUNT_INC(2, "forward_substitution");
            {%- endif %}
        }
    }   

{%- endif %}
{% if not loop_order %} // kij
     int k;
    int j;
   
   __m256i vindex = _mm256_set_epi64x(0, n, 2*n, 3*n);
    
    for(k = 0; k<n-3; k+=4){

        for(int i=0; i<n; i++){
            double rezi = 1.0/A[i*n+i];
            __m256d rezi_vec = _mm256_set1_pd(rezi);
            
            {% if flop_count %}
                FLOP_COUNT_INC(1, "forward_substitution");
            {%- endif %}
            

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

                {% if flop_count %}
                FLOP_COUNT_INC(4*4*2, "forward_substitution");
            {%- endif %}
                
            }
            //CLEANUP LOOP FOR J
            for(; j<i; j++){
                __m256d A_vec = _mm256_set1_pd(A[j*n+i]);
                __m256d y_vec = _mm256_i64gather_pd(&y[k*n+j], vindex, 8);
                sum_vec = _mm256_fmadd_pd(A_vec, y_vec, sum_vec);

                {% if flop_count %}
                FLOP_COUNT_INC(4*2, "forward_substitution");
                {%- endif %}
                
            }
            
            __m256d b_vec = _mm256_i64gather_pd(&b[k*n+i], vindex, 8);

            __m256d tmp_final_sum = _mm256_add_pd(sum_vec3, sum_vec2);
            __m256d tmp_final_sum1 = _mm256_add_pd(sum_vec1, sum_vec);
            __m256d final_sum = _mm256_add_pd(tmp_final_sum1, tmp_final_sum);


            __m256d tmp_res_vec = _mm256_sub_pd(b_vec, final_sum);
            __m256d res_vec = _mm256_mul_pd(tmp_res_vec, rezi_vec);

            {% if flop_count %}
                FLOP_COUNT_INC(5*4, "forward_substitution");
            {%- endif %}


            y[k*n+i] = res_vec[3];
            y[(k+1)*n+i] = res_vec[2];
            y[(k+2)*n+i] = res_vec[1];
            y[(k+3)*n+i] = res_vec[0];
            
        }
    }

        //CLEANUP LOOP FOR K
    for(; k<n; k++){
        for(int i=0; i<n; i++){
            double rezi = 1.0/A[i*n+i];
            //__m256d rezi_vec = _mm256_set1_pd(rezi);
            {% if flop_count %}
                FLOP_COUNT_INC(1, "forward_substitution");
            {%- endif %}
            double sum = 0;
            for(j=0; j<i-3; j+=4){
                sum += A[j*n+i] * y[k*n + j];
                sum += A[(j+1)*n+i] * y[k*n + j +1];
                sum += A[(j+2)*n+i] * y[k*n + j +2];
                sum += A[(j+3)*n+i] * y[k*n + j +3];

                {% if flop_count %}
                FLOP_COUNT_INC(4*2, "forward_substitution");
            {%- endif %}
            }
            for(; j<i; j++){
                sum += A[j*n+i] * y[k*n + j];
                {% if flop_count %}
                FLOP_COUNT_INC(2, "forward_substitution");
            {%- endif %}
            }
            y[k*n+i] = b[k*n+i] - sum;
            y[k*n+i] *= rezi; 
            {% if flop_count %}
                FLOP_COUNT_INC(2, "forward_substitution");
            {%- endif %}
        }
    }   


{%- endif %}

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

        {% if flop_count %}
            FLOP_COUNT_INC(1, "LU_decomposition");
        {%- endif %}

        index_piv = k;

        for(int i=k+1; i<n; i++){ 
            double abs = fabs(LU[i*n+k]);

            {% if flop_count %}
            FLOP_COUNT_INC(1, "LU_decomposition");
            {%- endif %}

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

        {% if flop_count %}
            FLOP_COUNT_INC(1, "LU_decomposition");
        {%- endif %}

       int i;
        for(i=1; i<(n-k)-3; i+=4){
            __m256d vec1 = _mm256_loadu_pd(&LU[k*n+ k +i]);
            __m256d res_vec = _mm256_mul_pd(vec1, rezi_vec);
            _mm256_storeu_pd(&LU[k*n+k+i], res_vec);

            {% if flop_count %}
            FLOP_COUNT_INC(4, "LU_decomposition");
        {%- endif %}
        }

        //CLEANUP LOOP FOR I
        for(; i<(n-k); i++){
            LU[k*n+k+i] = LU[k*n+ k +i] *rezi; 

            {% if flop_count %}
            FLOP_COUNT_INC(1, "LU_decomposition");
            {%- endif %}
        }


        for(int i=0; i<(n-k-1); i++){
            int j;
            for(j=0; j<(n-k-1)-3; j+=4){
                __m256d curr = _mm256_loadu_pd(&LU[(k+i+1)*n+ (k+j+1)]);
                __m256d a21_vec = _mm256_loadu_pd(&LU[k*n+ k +j +1]);
                __m256d a12_vec = _mm256_broadcast_sd(&LU[(k+i+1)*n+k]);

                __m256d prod = _mm256_mul_pd(a12_vec, a21_vec);
                __m256d res = _mm256_sub_pd(curr, prod);

                {% if flop_count %}
                    FLOP_COUNT_INC(8, "LU_decomposition");
                {%- endif %}
                
                _mm256_storeu_pd(&LU[(k+i+1)*n+ (k+j+1)], res);           
            }
            //CLEANUP LOOP FOR J
            for(    ; j<(n-k-1); j++){
                LU[(k+i+1)*n + (k+j+1)] -= LU[(k+i+1)*n+k] * LU[k*n+k+1+j];
                {% if flop_count %}
                    FLOP_COUNT_INC(2, "LU_decomposition");
                {%- endif %}
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



