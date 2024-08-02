#include "LU_wrapper.h"


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