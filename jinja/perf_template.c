/* profiling template */

int main(int argc, char* argv[]){
    if(argc < 4){
        printf("please specify path to matrix and read mode (0 for row major, 1 for col major) and number of iterations\n");
        return -1;
    }
    char *path = argv[1];
    int col_major = atoi(argv[2]);
    int num_it = atoi(argv[3]);
    FILE *fptr = fopen(path, "r");
    if(fptr==NULL){
        printf("file not found\n");
        return -1;
    }
    char buf[50];
    fscanf(fptr, "%s", buf);
    int n = atoi(buf);
    fscanf(fptr, "%s", buf);
    int m = atoi(buf);
    if(m!=n){
        printf("non square matrix!\n");
        return -1;
    }

    double *A = (double *) aligned_alloc(32, n*n*sizeof(double));
    double *B = (double *) aligned_alloc(32, n*n*sizeof(double));

    if(!col_major){
        for(int i = 0; i < n; i++){
            for(int j = 0; j<n; j++){
                fscanf(fptr, "%s", buf);
                A[i * n + j] = atof(buf);
            }
        }
    }else{
        for(int i = 0; i < n; i++){
            for(int j = 0; j<n; j++){
                fscanf(fptr, "%s", buf);
                A[j * n + i] = atof(buf);
            }
        }
    }

    for(int i = 0; i < num_it; i++){
        mat_exp(A, n, B);
    }
    
    free(A);
    free(B);


}   