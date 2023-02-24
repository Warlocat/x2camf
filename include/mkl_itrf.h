#ifndef MKL_INTERFACE_H_
#define MKL_INTERFACE_H_

#include<Eigen/Dense>
#include<string>
#include<vector>
#include"mkl.h"
using namespace std;
using namespace Eigen;

/* Print matrix function defined by mkl examples */
void print_matrix_mkl(const string &desc, const int& m, const int& n, const double* a);

/* Matrix mulplication for double and complex */
void dgemm_itrf(const char transa, const char transb, const int m, const int n, const int k,
                const double a, const MatrixXd& A, const MatrixXd& B, const double c, MatrixXd& C);
void dgemm_itrf(const char transa, const char transb, const int m, const int n, const int k,
                const double a, double** A, double** B, const double c, double** C);
void dgemm_itrf(const char transa, const char transb, const int m, const int n, const int k,
                const double a, const vector<double>& A, const vector<double>& B, const double c, vector<double>& C);

/* Hermitian matrix eigenvalues solver for double and complex */
void eigh_d(const MatrixXd& inputM, const int& N, VectorXd& values, MatrixXd& vectors);
void eigh_d(const vector<double>& inputM, const int& N, vector<double>& values, vector<double>& vectors);

/* Solve linear equation */
void liearEqn_d(const MatrixXd& inputA, const VectorXd& inputB, const int& N, VectorXd& solution);
void liearEqn_d(const vector<double>& inputA, const vector<double>& inputB, const int& N, vector<double>& solution);

#endif
