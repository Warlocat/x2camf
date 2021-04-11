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
    int size_basis_spinor, Nirrep, nelec, occMax_irrep;
    Matrix<irrep_jm, Dynamic, 1> irrep_list;
    vMatrixXd overlap, kinetic, WWW, Vnuc;
    int2eJK h2eLLLL_JK, h2eSSLL_JK, h2eSSSS_JK;
    vMatrixXd density, fock_4c, h1e_4c, overlap_4c, overlap_half_i_4c;
    vVectorXd norm_s;
    MatrixXd coeff_contract;
    vVectorXd occNumber;
    double d_density;
    bool converged;

    void readIntegrals(MatrixXd& h2e_, const string& filename);
    MatrixXd evaluateDensity_spinor(const MatrixXd& coeff_, const VectorXd& occNumber_);
    vMatrixXd evaluateDensity_spinor_irrep();
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

    DHF_SPH(INT_SPH& int_sph_, const string& filename);
    ~DHF_SPH();
    void runSCF();
    void renormalize_small();
    void readOCC(const string& filename);
};


#endif