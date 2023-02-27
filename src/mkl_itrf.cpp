#include<string>
#include<iostream>
#include<fstream>
#include<vector>
#include"general.h"
#include"mkl_itrf.h"
#include<mkl.h>
using namespace std;

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
                const double a, const vector<double>& A, const vector<double>& B, const double c, vector<double>& C)
{
    int lda, ldb;
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
    if(c == 0.0) C.resize(m*n);
    lda = transa==CblasNoTrans ? k : m;
    ldb = transb==CblasNoTrans ? n : k;
    cblas_dgemm(CblasRowMajor, transa, transb, m, n, k, a, A.data(), lda, B.data(), ldb, c, C.data(), n);

    return;
}
void zgemm_itrf(const char transa_, const char transb_, const int m, const int n, const int k,
                const complex<double> a, const vector<complex<double>>& A, const vector<complex<double>>& B, 
                const complex<double> c, vector<complex<double>>& C)
{
    int lda, ldb;
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
    if(c == zero_cp) C.resize(m*n);
    lda = transa==CblasNoTrans ? k : m;
    ldb = transb==CblasNoTrans ? n : k;
    cblas_zgemm(CblasRowMajor, transa, transb, m, n, k, &a, A.data(), lda, B.data(), ldb, &c, C.data(), n);

    return;
}
/*
    Matrix inversion
*/
vector<double> matInv_d(const vector<double>& inputM, const int& N)
{
    int info, ipiv[N];
    vector<double> tmpM = inputM;
    /* Solve the equations M^-1 */
    info = LAPACKE_dgetrf( LAPACK_ROW_MAJOR, N, N, tmpM.data(), N, ipiv);
    if( info > 0 ) 
    {
            printf( "The diagonal element of the triangular factor of M,\n" );
            printf( "U(%i,%i) is zero, so that A is singular;\n", info, info );
            printf( "the solution could not be computed.\n" );
            exit( 1 );
    }
    info = LAPACKE_dgetri( LAPACK_ROW_MAJOR, N, tmpM.data(), N, ipiv);
    /* Check for the exact singularity */
    if( info < 0 ) 
    {
            printf( "Parameter %i had an illegal value in LAPACKE_dgetri;\n", info);
            printf( "the solution could not be computed.\n" );
            exit( 1 );
    }
    
    return tmpM;
}

/* 
    Solve linear equation 
*/
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
