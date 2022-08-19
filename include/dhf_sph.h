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
    
    VectorXi all2compact,compact2all;
    vMatrixXd overlap, kinetic, WWW, Vnuc;
    int2eJK h2eLLLL_JK, h2eSSLL_JK, h2eSSSS_JK, gauntLSLS_JK, gauntLSSL_JK;
    vMatrixXd density, fock_4c, h1e_4c, overlap_4c, overlap_half_i_4c, x2cXXX, x2cRRR;
    vVectorXd norm_s;
    vVectorXd occNumber, occNumberCore;
    double d_density, nelec;
    bool converged = false, renormalizedSmall = false, with_gaunt = false, with_gauge = false, X_calculated = false;
    
    /* evaluate density martix */
    MatrixXd evaluateDensity_spinor(const MatrixXd& coeff_, const VectorXd& occNumber_, const bool& twoC = false);
    vMatrixXd evaluateDensity_spinor_irrep(const bool& twoC = false);
    
    MatrixXd evaluateErrorDIIS(const MatrixXd& fock_, const MatrixXd& overlap_, const MatrixXd& density_);
    MatrixXd evaluateErrorDIIS(const MatrixXd& den_old, const MatrixXd& den_new);
    /* solver for generalized eigen equation MC=SCE, s_h_i = S^{1/2} */
    void eigensolverG_irrep(const vMatrixXd& inputM, const vMatrixXd& s_h_i, vVectorXd& values, vMatrixXd& vectors);
    double evaluateChange_irrep(const vMatrixXd& M1, const vMatrixXd& M2);

public:
Matrix<intShell, Dynamic, 1> shell_list;
    int size_basis_spinor, Nirrep, Nirrep_compact, occMax_irrep, occMax_irrep_compact;
    Matrix<irrep_jm, Dynamic, 1> irrep_list;
    int maxIter = 100, size_DIIS = 8;
    double convControl = 1e-8, ene_scf;
    vMatrixXd coeff;
    vVectorXd ene_orb;
    VectorXd ene_orb_total;

    DHF_SPH(INT_SPH& int_sph_, const string& filename, const bool& spinFree = false, const bool& twoC = false, const bool& with_gaunt_ = false, const bool& with_gauge_ = false, const bool& allInt = false, const bool& gaussian_nuc = false);
    virtual ~DHF_SPH();

    /* Renormalized small component s.t. the overlap is 1.0 */
    void renormalize_small();
    void renormalize_h2e(int2eJK& h2e, const string& intType);
    /* Symmetrize h2e in J - K form */
    void symmetrize_h2e(const bool& twoC = false);
    void symmetrize_JK(int2eJK& h2e, const int& Ncompact);
    void symmetrize_JK_gaunt(int2eJK& h2e, const int& Ncompact);
    /* Read occupation numbers */
    void setOCC(const string& filename, const string& atomName);

    /* evaluate Fock matrix */
    void evaluateFock(MatrixXd& fock, const bool& twoC, const vMatrixXd& den, const int& size, const int& Iirrep);

    /* x2c2e picture change */
    vMatrixXd x2c2ePCC(vMatrixXd* coeff2c = NULL);
    vMatrixXd h_x2c2e(vMatrixXd* coeff2c = NULL);
    vMatrixXd x2c2ePCC_density();
    vMatrixXd x2c2ePCC_test();
    void evaluateFock_2e(MatrixXd& fock, const bool& twoC, const vMatrixXd& den, const int& size, const int& Iirrep);
    void evaluateFock_J(MatrixXd& fock, const bool& twoC, const vMatrixXd& den, const int& size, const int& Iirrep);

    /* Get Coeff for basis set */
    vMatrixXd get_coeff_bs(const bool& twoC = true);

    /* Get private variables */
    vMatrixXd get_fock_4c();
    vMatrixXd get_fock_4c_2ePart();
    vMatrixXd get_h1e_4c();
    vMatrixXd get_overlap_4c();
    vMatrixXd get_density();
    vVectorXd get_occNumber();
    vMatrixXd get_X();
    vMatrixXd get_X_normalized();
    void set_h1e_4c(const vMatrixXd& inputM);

    /* SCF iterations */
    virtual void runSCF(const bool& twoC = false, const bool& renormSmall = true);
    virtual void runSCF(const bool& twoC, const bool& renormSmall, vMatrixXd* initialGuess);

    /* Evaluate amfi SOC integrals */
    virtual vMatrixXd get_amfi_unc(INT_SPH& int_sph_, const bool& twoC = false, const string& Xmethod = "partialFock", bool amfi_with_gaunt = false, bool amfi_with_gauge = false);
    virtual vMatrixXd get_amfi_unc(const int2eJK& h2eSSLL_SD, const int2eJK& h2eSSSS_SD, const int2eJK& gauntLSLS_SD, const int2eJK& gauntLSSL_SD, const vMatrixXd& density_, const string& Xmethod = "partialFock", const bool& amfi_with_gaunt = false);
    virtual vMatrixXd get_amfi_unc_2c(const int2eJK& h2eSSLL_SD, const int2eJK& h2eSSSS_SD, const bool& amfi_with_gaunt = false);

    virtual void basisGenerator(string basisName, string filename, const INT_SPH& intor, const INT_SPH& intorAll, const bool& sf = true, const string& tag = "DE4");
};


#endif
