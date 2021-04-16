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
private:
    int size_basis_spinor, Nirrep, occMax_irrep;
    Matrix<irrep_jm, Dynamic, 1> irrep_list;
    vMatrixXd overlap, kinetic, WWW, Vnuc;
    int2eJK h2eLLLL_JK, h2eSSLL_JK, h2eSSSS_JK;
    vMatrixXd density, fock_4c, h1e_4c, overlap_4c, overlap_half_i_4c, x2cXXX, x2cRRR;
    vVectorXd norm_s;
    MatrixXd coeff_contract;
    vVectorXd occNumber;
    double d_density, nelec;
    bool converged = false, renormalizedSmall = false;

    MatrixXd evaluateDensity_spinor(const MatrixXd& coeff_, const VectorXd& occNumber_, const bool& twoC = false);
    vMatrixXd evaluateDensity_spinor_irrep(const bool& twoC = false);
    MatrixXd evaluateErrorDIIS(const MatrixXd& fock_, const MatrixXd& overlap_, const MatrixXd& density_);
    /* solver for generalized eigen equation MC=SCE, s_h_i = S^{1/2} */
    void eigensolverG_irrep(const vMatrixXd& inputM, const vMatrixXd& s_h_i, vVectorXd& values, vMatrixXd& vectors);
    double evaluateChange_irrep(const vMatrixXd& M1, const vMatrixXd& M2);

public:
    int maxIter = 100, size_DIIS = 8;
    double convControl = 1e-9, ene_scf;
    vMatrixXd coeff;
    vVectorXd ene_orb;
    VectorXd ene_orb_total;

    DHF_SPH(INT_SPH& int_sph_, const string& filename, const bool& spinFree = false, const bool& sfx2c = false);
    ~DHF_SPH();
    void runSCF();
    void runSCF_2c();
    void renormalize_small();
    /* Read occupation numbers */
    void readOCC(const string& filename);
    /* Evaluate amfi SOC integrals */
    vMatrixXd get_amfi_unc(INT_SPH& int_sph_, const string& Xmethod = "partialFock");
    vMatrixXd get_amfi_unc(const int2eJK& h2eSSLL_SD, const int2eJK& h2eSSSS_SD, const string& Xmethod = "partialFock");
    vMatrixXd get_amfi_unc_2c(INT_SPH& int_sph_);
    vMatrixXd get_amfi_unc_2c(const int2eJK& h2eSSLL_SD, const int2eJK& h2eSSSS_SD);
    /* Put one-electron integrals in a single matrix and reorder them */
    static MatrixXd unite_irrep(const vMatrixXd& inputM, const Matrix<irrep_jm, Dynamic, 1>& irrep_list);
};


#endif