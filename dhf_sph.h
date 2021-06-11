#ifndef DHF_SPH_H_
#define DHF_SPH_H_
#include<Eigen/Dense>
#include<complex>
#include<string>
#include"int_sph.h"
using namespace std;
using namespace Eigen;

class DHF_SPH
{
protected:
    int size_basis_spinor, Nirrep, occMax_irrep;
    Matrix<irrep_jm, Dynamic, 1> irrep_list;
    vMatrixXd overlap, kinetic, WWW, Vnuc;
    int2eJK h2eLLLL_JK, h2eSSLL_JK, h2eSSSS_JK, gauntLSLS_JK, gauntLSSL_JK;
    vMatrixXd density, fock_4c, h1e_4c, overlap_4c, overlap_half_i_4c, x2cXXX, x2cRRR;
    vVectorXd norm_s;
    vVectorXd occNumber;
    double d_density, nelec;
    bool converged = false, renormalizedSmall = false, with_gaunt = false;
    
    /* evaluate density martix */
    MatrixXd evaluateDensity_spinor(const MatrixXd& coeff_, const VectorXd& occNumber_, const bool& twoC = false);
    vMatrixXd evaluateDensity_spinor_irrep(const bool& twoC = false);
    /* evaluate Fock matrix */
    void evaluateFock(MatrixXd& fock, const bool& twoC, const vMatrixXd& den, const int& size, const int& Iirrep);
    MatrixXd evaluateErrorDIIS(const MatrixXd& fock_, const MatrixXd& overlap_, const MatrixXd& density_);
    MatrixXd evaluateErrorDIIS(const MatrixXd& den_old, const MatrixXd& den_new);
    /* solver for generalized eigen equation MC=SCE, s_h_i = S^{1/2} */
    void eigensolverG_irrep(const vMatrixXd& inputM, const vMatrixXd& s_h_i, vVectorXd& values, vMatrixXd& vectors);
    double evaluateChange_irrep(const vMatrixXd& M1, const vMatrixXd& M2);

public:
    int maxIter = 100, size_DIIS = 8;
    double convControl = 1e-9, ene_scf;
    vMatrixXd coeff;
    vVectorXd ene_orb;
    VectorXd ene_orb_total;

    DHF_SPH(INT_SPH& int_sph_, const string& filename, const bool& spinFree = false, const bool& sfx2c = false, const bool& with_gaunt_ = false);
    virtual ~DHF_SPH();
    virtual void runSCF(const bool& twoC = false);
    void renormalize_small();
    /* Read occupation numbers */
    void readOCC(const string& filename, const string& atomName);
    /* Evaluate amfi SOC integrals */
    vMatrixXd get_amfi_unc(INT_SPH& int_sph_, const bool& twoC = false, const string& Xmethod = "partialFock", const bool& amfi_with_gaunt = false);
    vMatrixXd get_amfi_unc(const int2eJK& h2eSSLL_SD, const int2eJK& h2eSSSS_SD, const vMatrixXd& density_, const string& Xmethod = "partialFock", const bool& amfi_with_gaunt = false);
    vMatrixXd get_amfi_unc_2c(const int2eJK& h2eSSLL_SD, const int2eJK& h2eSSSS_SD, const bool& amfi_with_gaunt = false);

    // Get private variables
    vMatrixXd get_fock_4c();
    vMatrixXd get_h1e_4c();
    vMatrixXd get_overlap_4c();
    void set_h1e_4c(const vMatrixXd& inputM);
};


#endif