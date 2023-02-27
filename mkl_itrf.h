#ifndef MKL_INTERFACE_H_
#define MKL_INTERFACE_H_

#include<string>
#include<vector>
#include"mkl.h"
using namespace std;

/* Print matrix function defined by mkl examples */
void print_matrix_mkl(const string &desc, const int& m, const int& n, const double* a);

/* Matrix mulplication for double and complex */
void dgemm_itrf(const char transa, const char transb, const int m, const int n, const int k,
                const double a, const vector<double>& A, const vector<double>& B, const double c, vector<double>& C);
void zgemm_itrf(const char transa_, const char transb_, const int m, const int n, const int k,
                const complex<double> a, const vector<complex<double>>& A, const vector<complex<double>>& B, 
                const complex<double> c, vector<complex<double>>& C);

/* Hermitian matrix eigenvalues solver for double and complex */
void eigh_d(const vector<double>& inputM, const int& N, vector<double>& values, vector<double>& vectors);

/* Matrix inversion */
vector<double> matInv_d(const vector<double>& inputM, const int& N);

/* Solve linear equation */
void liearEqn_d(const vector<double>& inputA, const vector<double>& inputB, const int& N, vector<double>& solution);

#endif
