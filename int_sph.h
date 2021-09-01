#ifndef INT_SPH_H_
#define INT_SPH_H_

#include<Eigen/Dense>
#include<complex>
#include<string>
#include"general.h"
using namespace std;
using namespace Eigen;

/*
    Class for spherical atom integrals in 2-spinor basis.

    Variables:
        
*/
class INT_SPH
{
protected:
    /* read basis set and normalization, used in construction functions*/
    void readBasis();
    void normalization();
    /* auxiliary functions used to evaluate 1e and 2e intergals */
    double auxiliary_1e(const int& l, const double& a) const;
    double auxiliary_2e_0_r(const int& l1, const int& l2, const double& a1, const double& a2) const;
    double auxiliary_2e_r_inf(const int& l1, const int& l2, const double& a1, const double& a2) const;
    /* evaluate radial part and angular part in 2e integrals */
    double int2e_get_radial(const int& l1, const double& a1, const int& l2, const double& a2, const int& l3, const double& a3, const int& l4, const double& a4, const int& LL) const;
    double int2e_get_angular(const int& l1, const int& two_m1, const int& s1, const int& l2, const int& two_m2, const int& s2, const int& l3, const int& two_m3, const int& s3, const int& l4, const int& two_m4, const int& s4, const int& LL) const;
    double int2e_get_angular_K(const int& l1, const int& two_m1, const int& s1, const int& l2, const int& two_m2, const int& s2, const int& LL) const;
    double int2e_get_angular_J(const int& l1, const int& two_m1, const int& s1, const int& l2, const int& two_m2, const int& s2, const int& LL) const;
    double int2e_get_angular_gaunt_LSLS(const int& l1, const int& two_m1, const int& s1, const int& l2, const int& two_m2, const int& s2, const int& l3, const int& two_m3, const int& s3, const int& l4, const int& two_m4, const int& s4, const int& LL) const;
    double int2e_get_angular_gaunt_LSSL(const int& l1, const int& two_m1, const int& s1, const int& l2, const int& two_m2, const int& s2, const int& l3, const int& two_m3, const int& s3, const int& l4, const int& two_m4, const int& s4, const int& LL) const;
    double int2e_get_threeSH(const int& l1, const int& m1, const int& l2, const int& m2, const int& l3, const int& m3, const double& threeJ) const;
    Vector3d int2e_get_angular_gaunt_ssp(const int& l1, const int& two_m1, const int& s1, const int& l2, const int& two_m2, const int& s2, const int& LL, const int& MM, const double& threeJ_p_12, const double& threeJ_m_12) const;
    Vector3d int2e_get_angular_gaunt_sps(const int& l1, const int& two_m1, const int& s1, const int& l2, const int& two_m2, const int& s2, const int& LL, const int& MM, const double& threeJ_p_12, const double& threeJ_m_12) const;

public:
    Matrix<intShell, Dynamic, 1> shell_list;
    Matrix<irrep_jm, Dynamic, 1> irrep_list;
    int atomNumber, Nirrep = 0;
    int size_gtoc, size_gtou, size_shell;
    string atomName, basisSet;
    int size_gtoc_spinor, size_gtou_spinor;

    INT_SPH(const string& atomName_, const string& basisSet_);
    ~INT_SPH();

    /* Evaluate one-electron integral */
    vMatrixXd get_h1e(const string& intType) const;
    /* Evaluate two-electron integral in J-K form */
    int2eJK get_h2e_JK(const string& intType, const int& occMaxL = -1) const;
    int2eJK get_h2e_JK_compact(const string& intType, const int& occMaxL = -1) const;
    void get_h2e_JK_direct(int2eJK& LLLL, int2eJK& SSLL, int2eJK& SSSS, const int& occMaxL = -1, const bool& spinFree = false);
    void get_h2eSD_JK_direct(int2eJK& SSLL, int2eJK& SSSS, const int& occMaxL = -1);
    int2eJK get_h2e_JK_gaunt(const string& intType, const int& occMaxL = -1) const;
    int2eJK get_h2e_JK_gaunt_compact(const string& intType, const int& occMaxL = -1) const;
    void get_h2e_JK_gaunt_direct(int2eJK& LSLS, int2eJK& LSSL, const int& occMaxL = -1, const bool& spinFree = false);
    void get_h2eSD_JK_gaunt_direct(int2eJK& LSLS, int2eJK& LSSL, const int& occMaxL = -1);
    
    /* get contraction coefficients for uncontracted calculations */
    MatrixXd get_coeff_contraction_spinor();
};

#endif