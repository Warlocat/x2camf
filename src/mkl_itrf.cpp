#include<Eigen/Dense>
#include<string>
#include<iostream>
#include<fstream>
#include<vector>
#include"mkl_itrf.h"
#include"mkl.h"
using namespace std;
using namespace Eigen;

/* 
    Print matrix function modified from mkl examples 
*/
void print_matrix_mkl(const string &desc, const int& m, const int& n, const double* a) 
{
	int i, j;
	cout << "\n" << desc << "\n";
	for( i = 0; i < m; i++ ) {
		for( j = 0; j < n; j++ ) printf( " %20.6f", a[i*n+j] );
		printf( "\n" );
	}
}

/* 
    Hermitian matrix eigenvalues solver for double and complex 
*/
void eigh_d(const MatrixXd& inputM, const int& N, VectorXd& values, MatrixXd& vectors)
{
    int info;
    double w[N];
    double tmp[N*N];
    for(int ii = 0; ii < N; ii++)
    for(int jj = 0; jj < N; jj++)
    {
        tmp[ii*N+jj] = inputM(ii,jj);
    }
    /* Solve eigenproblem */
    info = LAPACKE_dsyev( LAPACK_ROW_MAJOR, 'V', 'U', N, tmp, N, w);
    /* Check for convergence */
    if( info > 0 ) 
    {
        printf( "The algorithm failed to compute eigenvalues.\n" );
        exit( 1 );
    }
    else
    {
        values.resize(N);
        vectors.resize(N,N);
        double norm[N];
        for(int ii = 0; ii < N; ii++)
        {
            values(ii) = w[ii];
            norm[ii] = 0.0;
            for(int jj = 0; jj < N; jj++)
            {
                norm[ii] += pow(tmp[jj*N+ii],2);
            }
            norm[ii] = sqrt(norm[ii]);
        }
        for(int ii = 0; ii < N; ii++)
        for(int jj = 0; jj < N; jj++)
            vectors(ii,jj)=tmp[ii*N+jj]/norm[jj];
    }
    return;
}
void eigh_d(const vector<double>& inputM, const int& N, vector<double>& values, vector<double>& vectors)
{
    int info;
    values.resize(N);
    vector<double> tmp(N*N);
    for(int ii = 0; ii < N; ii++)
    for(int jj = 0; jj < N; jj++)
    {
        tmp[ii*N+jj] = inputM[ii*N+jj];
    }
    /* Solve eigenproblem */
    info = LAPACKE_dsyev( LAPACK_ROW_MAJOR, 'V', 'U', N, tmp.data(), N, values.data());
    /* Check for convergence */
    if( info > 0 ) 
    {
        printf( "The algorithm failed to compute eigenvalues.\n" );
        exit( 1 );
    }
    else
    {
        vectors.resize(N*N);
        vector<double> norm(N);
        for(int ii = 0; ii < N; ii++)
        {
            norm[ii] = 0.0;
            for(int jj = 0; jj < N; jj++)
            {
                norm[ii] += pow(tmp[jj*N+ii],2);
            }
            norm[ii] = sqrt(norm[ii]);
        }
        for(int ii = 0; ii < N; ii++)
        for(int jj = 0; jj < N; jj++)
            vectors[ii*N+jj]=tmp[ii*N+jj]/norm[jj];
    }
    return;
}


/* 
    Matrix mulplication for double and complex 
*/
void dgemm_itrf(const char transa_, const char transb_, const int m, const int n, const int k,
                const double a, const MatrixXd& A, const MatrixXd& B, const double c, MatrixXd& C)
{
    
    vector<double> tmpA(m*k),tmpB(n*k),tmpC(m*n);
    for(int ii = 0; ii < m; ii++)
    for(int jj = 0; jj < k; jj++)
        tmpA[ii*k+jj]=A(ii,jj);
    for(int ii = 0; ii < k; ii++)
    for(int jj = 0; jj < n; jj++)
        tmpB[ii*n+jj]=B(ii,jj);
    dgemm_itrf(transa_, transb_, m, n, k, a, tmpA, tmpB, c, tmpC);
    C.resize(m,n);
    for(int ii = 0; ii < m; ii++)
    for(int jj = 0; jj < n; jj++)
        C(ii,jj)=tmpC[ii*n+jj];
    return;
}
void dgemm_itrf(const char transa_, const char transb_, const int m, const int n, const int k,
                const double a, double** A, double** B, const double c, double** C)
{
    vector<double> tmpA(m*k),tmpB(n*k),tmpC(m*n);
    for(int ii = 0; ii < m; ii++)
    for(int jj = 0; jj < k; jj++)
        tmpA[ii*k+jj]=A[ii][jj];
    for(int ii = 0; ii < k; ii++)
    for(int jj = 0; jj < n; jj++)
        tmpB[ii*n+jj]=B[ii][jj];
    dgemm_itrf(transa_, transb_, m, n, k, a, tmpA, tmpB, c, tmpC);
    C = new double*[m];
    for(int ii = 0; ii < m; ii++)
    {
        C[ii] = new double[n];
        for(int jj = 0; jj < n; jj++)
            C[ii][jj]=tmpC[ii*n+jj];
    }
    return;
}
void dgemm_itrf(const char transa_, const char transb_, const int m, const int n, const int k,
                const double a, const vector<double>& A, const vector<double>& B, const double c, vector<double>& C)
{
    CBLAS_TRANSPOSE transa, transb;
    switch(transa_)
    {
        case 'C': case 'c':
            transa=CblasConjTrans; break;
        case 'N': case 'n':
            transa=CblasNoTrans; break;
        case 'T': case 't':
            transa=CblasTrans; break;
        default:
            cout << "ERROR: Undefined char in CBLAS_TRANSPOSE" << endl;
            exit( 1 );
    }
    switch(transb_)
    {
        case 'C': case 'c':
            transb=CblasConjTrans; break;
        case 'N': case 'n':
            transb=CblasNoTrans; break;
        case 'T': case 't':
            transb=CblasTrans; break;
        default:
            cout << "ERROR: Undefined char in CBLAS_TRANSPOSE" << endl;
            exit( 1 );
    }
    C.resize(m*n);
    cblas_dgemm(CblasRowMajor, transa, transb, m, n, k, a, A.data(), k, B.data(), n, c, C.data(), n);

    return;
}

/* 
    Solve linear equation 
*/
void liearEqn_d(const MatrixXd& inputA, const VectorXd& inputB, const int& N, VectorXd& solution)
{
    int info, ipiv[N];
    vector<double> tmpA(N*N), tmpB(N);
    for(int ii = 0; ii < N; ii++)
    {
        tmpB[ii] = inputB(ii);
        for(int jj = 0; jj < N; jj++)
        {
            tmpA[ii*N+jj] = inputA(ii,jj);
        }
    }
    
    /* Solve the equations A*X = B */
    info = LAPACKE_dgesv( LAPACK_ROW_MAJOR, N, 1, tmpA.data(), N, ipiv, tmpB.data(), 1);
    /* Check for the exact singularity */
    if( info > 0 ) 
    {
            printf( "The diagonal element of the triangular factor of A,\n" );
            printf( "U(%i,%i) is zero, so that A is singular;\n", info, info );
            printf( "the solution could not be computed.\n" );
            exit( 1 );
    }
    else
    {
        solution.resize(N);
        for(int ii = 0; ii < N; ii++)
            solution(ii) = tmpB[ii];
    }
    return;
}
void liearEqn_d(const vector<double>& inputA, const vector<double>& inputB, const int& N, vector<double>& solution)
{
    int info, ipiv[N];
    vector<double> tmpA(N*N), tmpB(N);
    for(int ii = 0; ii < N; ii++)
    {
        tmpB[ii] = inputB[ii];
        for(int jj = 0; jj < N; jj++)
        {
            tmpA[ii*N+jj] = inputA[ii*N+jj];
        }
    }
    
    /* Solve the equations A*X = B */
    info = LAPACKE_dgesv( LAPACK_ROW_MAJOR, N, 1, tmpA.data(), N, ipiv, tmpB.data(), 1);
    /* Check for the exact singularity */
    if( info > 0 ) 
    {
            printf( "The diagonal element of the triangular factor of A,\n" );
            printf( "U(%i,%i) is zero, so that A is singular;\n", info, info );
            printf( "the solution could not be computed.\n" );
            exit( 1 );
    }
    else
    {
        solution.resize(N);
        for(int ii = 0; ii < N; ii++)
            solution[ii] = tmpB[ii];
    }
    return;
}
