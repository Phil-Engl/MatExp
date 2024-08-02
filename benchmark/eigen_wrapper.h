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

    // Now we declare the functions and data structures we need
   // typedef Eigen::Map<const Eigen::MatrixXd> ConstMatrixXdMap;
    typedef Eigen::Map<Eigen::MatrixXd> MatrixXdMap;
    extern void eigen_matrix_exp(const double* inp_mat, int size, double *out_mat);

    #ifndef __cplusplus
}
#endif
#endif