#include "onenorm.h"
/**
 * @brief This is a C implementation of algorithm 2.4 from the
 * 2000 Higham and Tisseur paper.
 * It is made with the help of the scipy implementation
 * https://github.com/scipy/scipy/blob/main/scipy/sparse/linalg/_onenormest.py
 * 
 */

//TODO: rewrite for more efficient column major access 

/* ----- helper functions ----- */

/**
 * @brief checks if all columns in A are parallel to any column of B
 * 
 * @param A the input matrix A (transposed)
 * @param B the input matrix B (transposed)
 * @param m number of rows of A and B
 * @param n number of columns of A and B
 * @return int 1, if the condition in the description is met, 0 otherwise.
 */
int check_all_columns_parallel(const double *A, const double *B, int m, int n){
    for(int i = 0; i < n; i++){
        int flag = 0;
        for(int j = 0; j < n; j++){
            FLOP_COUNT_INC(1, "check_all_columns_parallel");
            flag = flag || (dot_product(&A[i * m], &B[j * m], m) == (double)m);
        }
        if(!flag){    
            return 0;
        }
    }
    return 1;
}

/**
 * @brief column in A needs resampling if it is parallel to any of its previous columns
 * or any column in B
 * 
 * @param k the index of the current row
 * @param v the current row
 * @param A the matrix A in which the current row resides
 * @param B the matrix B
 * @param m number of rows of A and B
 * @param n number of columns of A and B
 * @return int 1 if the the row needs resampling, 0 otherwise 
 */
int column_needs_resampling(int k, const double* v, const double* A, const double* B, int m, int n){
    //Assume all entries in the matrices are 1.0 or -1.0.
    //Comparison with double casted int would be dangerous otherwise
    for(int i = 0; i < k; i++){
        FLOP_COUNT_INC(1, "column_needs_resampling");
        if(dot_product(v, &A[i * n], n) == (double)n){
            return 1;
        }
    }
    for(int i = 0; i < m; i++){
        FLOP_COUNT_INC(1, "column_needs_resampling");
        if(dot_product(v, &B[i * n], n) == (double)n){
            return 1;
        }
    }
    return 0;
}

/**
 * @brief Ensures that no column of A is parallel to another column of A
 * or to a column of B by replacing columns of A by rand{−1, 1}.
 * 
 * @param A Input matrix A, with all entries in {-1, 1} (transposed)
 * @param B Input matrix B, with all entries in {-1, 1} (transposed)
 * @param m number of rows of A and B 
 * @param n number of columns of A and B
 */
void resample_columns(double *A, double *B, int m, int n){
    // TODO: FLOP_COUNT_INC(0, "resample_columns"); 
    for(int i = 0; i < n; i++){
        while(column_needs_resampling(i, &A[i * m], A, B, n, m)){
            for(int j = 0; j < m; j++){
                A[i * m + j] = (double)(((rand() % 2) * 2) - 1); 
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
    // FLOP_COUNT_INC(0, "idx_in_hist");
    int flag = 0;
    for(int i = 0; i < hist_len; i++){
        flag = flag || (idx == hist[i]);
    }
    return flag;
}


/* ----- one norm functions ----- */

double onenorm_best(const double* A, int m, int n, int* max_idx){
    FLOP_COUNT_INC(2*n*m, "onenorm_best");
    double max = 0.0;
    for(int j = 0; j < n; j++){
        double curr = 0.0;
        for(int i = 0; i < m; i++){
            curr += fabs(A[i * n + j]);
        }
        if(curr > max){
            max = curr;
            *max_idx = j;
        }
    }
    return max;
}

double onenorm(const double* A, int m, int n){
    // FLOP_COUNT_INC(0, "onenorm"); Note: all flops are already accounted for in onenorm_best
    int best = 0;
    return onenorm_best(A,m,n,&best);
}

double onenormest(const double* A, int n, int t, int itmax, int get_v, int get_w, double* v, double* w){
    // FLOP_COUNT_INC: has been added directly at the operations for simplicity
    if(n < 4 || t != 2 || itmax < 2){
        printf("Wrong param in onenormest!\n");
        return -1.0;
    }
    
    int k = 1;
    int best_j = 0;
    int ind_best = 0;
    int hist_len = 0;
    int new_length = t;

    double est = 0.0;
    double est_old = 0.0;
    
    double max_h = 0.0;
    FLOP_COUNT_INC(2, "onenormest");
    double x_elem = 1.0 / (double)n;
    double x_elem_n = -1.0 * x_elem;

    int fst_in_idx;
    int fst_out_idx, snd_out_idx, out_ctr;;
    double fst_in, snd_in;
    double fst_out, snd_out;
    
    int* ind_hist = (int*)malloc(n * itmax * sizeof(int));
    int* ind = (int*)malloc(n * sizeof(int));
    int* ind_in = (int*)malloc(n * sizeof(int));
    
    double* AT = (double*)malloc(n * n * sizeof(double));
    double* S = (double*)malloc(n * t * sizeof(double));
    double* S_old = (double*)malloc(n * t * sizeof(double));
    double *S_T = (double*)malloc(n * t * sizeof(double));
    double *S_old_T = (double*)malloc(n * t * sizeof(double));
    double* X = (double*)malloc(n * t * sizeof(double));
    double* Y = (double*)malloc(n * t * sizeof(double));
    double* Z = (double*)malloc(n * t * sizeof(double));
    
    double* h = (double*)malloc(n * sizeof(double));

    srand(time(0));

    //initialize matrices and vectors
    //X is similar to the scipy implementation, just more efficiently built
    for(int i = 0; i < n; i++){
        ind_hist[i] = -1;
        ind[i] = -1;
        for(int j = 0; j < t; j++){
            S[i * t + j] = 0.0;
            if(j == 0 || i >= j){
                X[i * t + j] = x_elem;
            } else {
                X[i * t + j] = x_elem_n;
            }
        }
    }
    transpose(A, AT, n, n);

    while(1){
        //Y = A * X
        if(k == 1){
            mmm(A, X, Y, n, n, t);
        }else{
            for(int i = 0; i < n; i++){
                Y[i * t] = A[i * n + ind[0]];
                Y[(i + 1) * t] = A[i * n + ind[1]];
            }
        }
        est = onenorm_best(Y,n,t, &best_j); 
        FLOP_COUNT_INC(1, "onenormest");
        if(est > est_old || k == 2){
            if(k >= 2){
                ind_best = ind[best_j];
            }
            if(get_w){
                for(int i = 0; i < n; i++){
                    w[i] = Y[i * t + best_j];
                }
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
        FLOP_COUNT_INC(n*t, "onenormest");            
        for(int i = 0; i < n * t; i++){
           S[i] = Y[i] >= 0.0 ? 1.0 : -1.0;        
        }
        
        //(2)
        //transpose the matrices for better access
        transpose(S, S_T, n, t);
        transpose(S_old, S_old_T, n, t);
        //If every column of S is parallel to a column of S_old, break 
        if(check_all_columns_parallel(S_T, S_old_T, n, t)){
            break;
        }

        /* Ensure that no column of S is parallel to another column of S
        or to a column of S_old by replacing columns of S by rand{−1, 1}. */
        resample_columns(S_T, S_old_T, n, t);
        transpose(S_T, S, t, n);

        //(3)
        mmm(AT, S, Z, n, n, t);

        max_h = 0.0;
        FLOP_COUNT_INC(2*n*t, "onenormest");
        for(int i = 0; i < n; i++){
            h[i]= 0.0;
            for(int j = 0; j < t; j++){
                double a = fabs(Z[i * t + j]);
                if(a > h[i]){
                    h[i] = a;
                    FLOP_COUNT_INC(1, "onenormest");
                    if(a > max_h){
                        max_h = a;
                    }
                }
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
            FLOP_COUNT_INC(2*n, "normest");
            if(h[i] > snd_in || h[i] > snd_out){
                if(idx_in_hist(i, ind_hist, hist_len)){
                    FLOP_COUNT_INC(1, "normest");
                    if(h[i] >= fst_in){
                        snd_in = fst_in;
                        fst_in = h[i];
                        fst_in_idx = i;
                    }else if(h[i] > snd_in){
                        FLOP_COUNT_INC(1, "normest");
                        snd_in = h[i];
                    }
                }else{
                    FLOP_COUNT_INC(1, "normest");
                    if(h[i] >= fst_out){
                        snd_out = fst_out;
                        fst_out = h[i];
                        snd_out_idx = fst_out_idx;
                        fst_out_idx = i;
                        out_ctr++;
                    }else if(h[i] > snd_out){
                        FLOP_COUNT_INC(1, "normest");
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
    //(6)
    //optionally create v, the unit vector of ind_best
    if(get_v){
        for(int i = 0; i < n; i++){
            v[i] = 0.0;
        }
        v[ind_best] = 1.0;
    }
    
    free(ind_hist);
    free(ind);
    free(ind_in);
    free(AT);
    free(S);
    free(S_old);
    free(S_T);
    free(S_old_T);
    free(X);
    free(Y);
    free(Z);
    free(h);

    return est;
}

double normest(const double* A, int n){
    // FLOP_COUNT_INC(0, "normest"); Note: all flops are already accounted for in onenormest
    double* dummyv = (double*)malloc(sizeof(double));
    double* dummyw = (double*)malloc(sizeof(double));
    int t = 1;
    if(n > 2){
        t = 2;
    }
    
    double res = onenormest(A,n,t,5,0,0,dummyv,dummyw);
   
    free(dummyv);
    free(dummyw);
    return res;
    
}