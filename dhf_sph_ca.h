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
    void evaluateFock(MatrixXd& fock, const bool& twoC, const vMatrixXd& den_c, const vMatrixXd den_o, const int& size, const int& Iirrep);
    void evaluateFock_core(MatrixXd& fock, const bool& twoC, const vMatrixXd& den_c, const vMatrixXd den_o, const int& size, const int& Iirrep);
    void evaluateFock_open(MatrixXd& fock, const bool& twoC, const vMatrixXd& den_c, const vMatrixXd den_o, const int& size, const int& Iirrep);

public:
    DHF_SPH_CA(INT_SPH& int_sph_, const string& filename, const bool& spinFree = false, const bool& sfx2c = false);
    virtual ~DHF_SPH_CA();
    virtual void runSCF(const bool& twoC = false);
    virtual void runSCF_separate(const bool& twoC = false);

    vMatrixXd get_amfi_unc_ca(INT_SPH& int_sph_, const bool& twoC = false, const string& Xmethod = "partialFock");
    vMatrixXd get_amfi_unc_ca_2c(const int2eJK& h2eSSLL_SD, const int2eJK& h2eSSSS_SD);
};


#endif