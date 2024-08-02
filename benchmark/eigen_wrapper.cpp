#include "eigen_wrapper.h"

void eigen_matrix_exp(const double * inp_mat, int size, double * out_mat){
    Eigen::Map<const Eigen::MatrixXd> A(inp_mat, size, size);
    Eigen::Map<Eigen::MatrixXd> expA(out_mat, size, size);
    expA= A.exp();  
}