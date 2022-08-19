#ifndef DHF_SPH_CA_H_
#define DHF_SPH_CA_H_
#include<Eigen/Dense>
#include<complex>
#include<string>
#include"dhf_sph.h"
using namespace std;
using namespace Eigen;

class DHF_SPH_CA: public DHF_SPH
{
private:
    int coreShell, openShell;
    double f_NM, NN, MM;
    /* in CAHF, density is density_c */
    vMatrixXd density_o;
    MatrixXd evaluateDensity_core(const MatrixXd& coeff_, const VectorXd& occNumber_, const bool& twoC);
    MatrixXd evaluateDensity_open(const MatrixXd& coeff_, const VectorXd& occNumber_, const bool& twoC);
    void evaluateDensity_ca_irrep(vMatrixXd& den_c, vMatrixXd& den_o, const vMatrixXd& coeff_, const bool& twoC);
    /* evaluate fock matrix */
    void evaluateFock(MatrixXd& fock_c, MatrixXd& fock_o, const bool& twoC, const vMatrixXd& den_c, const vMatrixXd den_o, const int& size, const int& Iirrep);
    void evaluateFock_core(MatrixXd& fock, const bool& twoC, const vMatrixXd& den_c, const vMatrixXd den_o, const int& size, const int& Iirrep);
    void evaluateFock_open(MatrixXd& fock, const bool& twoC, const vMatrixXd& den_c, const vMatrixXd den_o, const int& size, const int& Iirrep);

public:
    DHF_SPH_CA(INT_SPH& int_sph_, const string& filename, const bool& spinFree = false, const bool& twoC = false, const bool& with_gaunt_ = false, const bool& with_gauge_ = false, const bool& allInt = false, const bool& gaussian_nuc = false);
    virtual ~DHF_SPH_CA();
    virtual void runSCF(const bool& twoC = false, const bool& renormSmall = true) override;

    virtual vMatrixXd get_amfi_unc(INT_SPH& int_sph_, const bool& twoC = false, const string& Xmethod = "partialFock", bool amfi_with_gaunt = false, bool amfi_with_gauge = false) override;
    virtual vMatrixXd get_amfi_unc_2c(const int2eJK& h2eSSLL_SD, const int2eJK& h2eSSSS_SD, const bool& amfi_with_gaunt = false) override;

    virtual void basisGenerator(string basisName, string filename, const INT_SPH& intor, const INT_SPH& intorAll, const bool& sf = true, const string& tag = "-DE4") override;
};


#endif