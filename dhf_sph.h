#ifndef DHF_SPH_H_
#define DHF_SPH_H_
#include<complex>
#include<string>
#include"int_sph.h"
using namespace std;

class DHF_SPH
{
protected:
    
    vector<int> all2compact,compact2all;
    vVectorXd overlap, kinetic, WWW, Vnuc;
    int2eJK h2eLLLL_JK, h2eSSLL_JK, h2eSSSS_JK, gauntLSLS_JK, gauntLSSL_JK;
    vVectorXd x2cXXX, x2cRRR;
    vVectorXd overlap_4c, overlap_half_i_4c, fock_4c, h1e_4c, density;
    vVectorXd norm_s;
    vVectorXd occNumber;
    double d_density, nelec;
    bool converged = false, renormalizedSmall = false, with_gaunt = false, with_gauge = false, X_calculated = false;
    bool twoComponent = false;
    
    /* evaluate density martix */
    vector<double> evaluateDensity_spinor(const vector<double>& coeff_, const vector<double>& occNumber_, const int& size, const bool& twoC = false);
    vVectorXd evaluateDensity_spinor_irrep(const vector<int>& nbas, const bool& twoC = false);
    
    vector<double> evaluateErrorDIIS(const vector<double>& fock_, const vector<double>& overlap_, const vector<double>& density_, const int& N);
    vector<double> evaluateErrorDIIS(const vector<double>& den_old, const vector<double>& den_new);
    /* solver for generalized eigen equation MC=SCE, s_h_i = S^{1/2} */
    void eigensolverG_irrep(const vVectorXd& inputM, const vVectorXd& s_h_i, vVectorXd& values, vVectorXd& vectors);
    double evaluateChange_irrep(const vVectorXd& M1, const vVectorXd& M2);

public:
    vector<intShell> shell_list;
    int size_basis_spinor, Nirrep, Nirrep_compact, occMax_irrep, occMax_irrep_compact;
    vector<irrep_jm> irrep_list;
    int maxIter = 100, size_DIIS = 8, printLevel = 0;
    double convControl = 1e-8, ene_scf;
    vVectorXd ene_orb, coeff;

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
    void evaluateFock(vector<double>& fock, const bool& twoC, const vVectorXd& den, const int& size, const int& Iirrep);
    void evaluateFock_2e(vector<double>& fock, const bool& twoC, const vVectorXd& den, const int& size, const int& Iirrep);
    void evaluateFock_2e(vector<double>& fock, const bool& twoC, const vVectorXd& den, const int& size, const int& Iirrep,
                         const int2eJK& LLLL, const int2eJK& SSLL, const int2eJK& SSSS, const int2eJK& gLSLS, const int2eJK& gLSSL);
    void evaluateFock_SO(vector<double>& fock, const vVectorXd& den, const int& size, const int& Iirrep,
                         const int2eJK& SSLL, const int2eJK& SSSS, const int2eJK& gLSLS, const int2eJK& gLSSL);

    /* Get Coeff for basis set */
    vVectorXd get_coeff_bs(const bool& twoC = true);

    /* Get private variables */
    vVectorXd get_fock_4c();
    vVectorXd get_fock_4c_2ePart();
    vVectorXd get_h1e_4c();
    vVectorXd get_overlap_4c();
    vVectorXd get_density();
    vVectorXd get_occNumber();
    vVectorXd get_X();
    vVectorXd get_X_normalized();
    void set_h1e_4c(const vVectorXd& inputM);

    /* SCF iterations */
    virtual void runSCF(const bool& twoC = false, const bool& renormSmall = true);
    virtual void runSCF(const bool& twoC, const bool& renormSmall, vVectorXd* initialGuess);

    /* Evaluate amfi SOC integrals */
    vVectorXd get_amfi_unc(INT_SPH& int_sph_, const bool& twoC = false, const string& Xmethod = "partialFock", bool amfi_with_gaunt = false, bool amfi_with_gauge = false, bool amfi4c = false);
    vVectorXd get_amfi_unc(const int2eJK& h2eSSLL_SD, const int2eJK& h2eSSSS_SD, const int2eJK& gauntLSLS_SD, const int2eJK& gauntLSSL_SD, const vVectorXd& density_, const string& Xmethod = "partialFock", const bool& amfi_with_gaunt = false, bool amfi4c = false);
    vVectorXd get_amfi_unc_2c(const int2eJK& h2eSSLL_SD, const int2eJK& h2eSSSS_SD, const bool& amfi_with_gaunt = false, bool amfi4c = false);
    /* Return SO_4c before x2c transformation */
    vVectorXd get_amfi_unc_4c(INT_SPH& int_sph_, const bool& twoC = false, const string& Xmethod = "partialFock", bool amfi_with_gaunt = false, bool amfi_with_gauge = false);

    /* x2c2e picture change */
    virtual vVectorXd x2c2ePCC(vVectorXd* coeff2c = NULL);
    vVectorXd h_x2c2e(vVectorXd* coeff2c = NULL);
    
    /* Evaluate atomic radial density \rho(r) */
    double radialDensity(double rr);
    double radialDensity(double rr, const vVectorXd& den);
    double radialDensity_OCC(double rr, const vVectorXd& occ);
};


#endif
