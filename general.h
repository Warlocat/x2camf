#ifndef GENERAL_H_
#define GENERAL_H_

#include<Eigen/Dense>
#include<complex>
#include<string>
using namespace std;
using namespace Eigen;

const double speedOfLight = 137.03599967994;

typedef Matrix<MatrixXd,-1,1> vMatrixXd;
typedef Matrix<VectorXd,-1,1> vVectorXd;
typedef Matrix<MatrixXd,-1,-1> mMatrixXd;

/*
    gtos in form of angular shell
*/
struct intShell
{
    VectorXd exp_a, norm;
    MatrixXd coeff;
    int l;
};
/*
    Irreducible rep |j, l, m_j>
*/
struct irrep_jm
{
    int l, two_j, two_mj, size;
};
/*
    Coulomb and exchange integral
*/
struct int2eJK
{
    mMatrixXd J, K;
};

/* factorial and double factorial functions */
double factorial(const int& n);
double double_factorial(const int& n);

/* functions used to evaluate wigner 3j symbols */
double wigner_3j(const int& l1, const int& l2, const int& l3, const int& m1, const int& m2, const int& m3);
double wigner_3j_zeroM(const int& l1, const int& l2, const int& l3);

/* evaluate "difference" between two MatrixXd */
double evaluateChange(const MatrixXd& M1, const MatrixXd& M2);
/* evaluate M^{-1/2} */
MatrixXd matrix_half_inverse(const MatrixXd& inputM);
/* evaluate M^{1/2} */
MatrixXd matrix_half(const MatrixXd& inputM);
/* solver for generalized eigen equation MC=SCE, s_h_i = S^{1/2} */
void eigensolverG(const MatrixXd& inputM, const MatrixXd& s_h_i, VectorXd& values, MatrixXd& vectors);


/* Static functions used in X2C */
namespace X2C
{
    MatrixXd get_X(const MatrixXd& S_, const MatrixXd& T_, const MatrixXd& W_, const MatrixXd& V_);
    MatrixXd get_X(const MatrixXd& coeff);
    MatrixXd get_R(const MatrixXd& S_, const MatrixXd& T_, const MatrixXd& X_);
    MatrixXd get_R(const MatrixXd& S_4c, const MatrixXd& X_);
    MatrixXd evaluate_h1e_x2c(const MatrixXd& S_, const MatrixXd& T_, const MatrixXd& W_, const MatrixXd& V_);
    MatrixXd evaluate_h1e_x2c(const MatrixXd& S_, const MatrixXd& T_, const MatrixXd& W_, const MatrixXd& V_, const MatrixXd& X_, const MatrixXd& R_);
}



#endif