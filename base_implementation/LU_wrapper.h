#ifndef EIGEN_WRAPPER_H
#define EIGEN_WRAPPER_H

// external call for c++ library
#ifndef __cplusplus
extern "C"{

    #endif

    // Include the Eigen library
    #include <Eigen/Dense>
    #include <Eigen/Eigenvalues>
    #include <Eigen/Core>
    #include <unsupported/Eigen/MatrixFunctions>
    #include <Eigen/LU>


    extern void Eigen_LU_decomposition(const double* inp_mat, const double * RHS, int size, double *out_mat);
    extern void Eigen_MatPow(const double * inp_mat, long power, int n, double * out_mat);



    #ifndef __cplusplus
}
#endif
#endif