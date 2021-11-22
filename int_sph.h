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
    double get_radial_LLLL_J(const int& lp, const int& lq, const int& LL, const double& a1, const double& a2, const double& a3, const double& a4, const double& lk1, const double& lk2, const double& lk3, const double& lk4, double radial_list[3][3], const bool& spinFree = false) const;
    double get_radial_LLLL_K(const int& lp, const int& lq, const int& LL, const double& a1, const double& a2, const double& a3, const double& a4, const double& lk1, const double& lk2, const double& lk3, const double& lk4, double radial_list[3][3], const bool& spinFree = false) const;
    double get_radial_SSLL_J(const int& lp, const int& lq, const int& LL, const double& a1, const double& a2, const double& a3, const double& a4, const double& lk1, const double& lk2, const double& lk3, const double& lk4, double radial_list[3][3], const bool& spinFree = false) const;
    double get_radial_SSLL_K(const int& lp, const int& lq, const int& LL, const double& a1, const double& a2, const double& a3, const double& a4, const double& lk1, const double& lk2, const double& lk3, const double& lk4, double radial_list[3][3], const bool& spinFree = false) const;
    double get_radial_SSSS_J(const int& lp, const int& lq, const int& LL, const double& a1, const double& a2, const double& a3, const double& a4, const double& lk1, const double& lk2, const double& lk3, const double& lk4, double radial_list[3][3], const bool& spinFree = false) const;
    double get_radial_SSSS_K(const int& lp, const int& lq, const int& LL, const double& a1, const double& a2, const double& a3, const double& a4, const double& lk1, const double& lk2, const double& lk3, const double& lk4, double radial_list[3][3], const bool& spinFree = false) const;
    double int2e_get_radial_gauge(const int& l1, const double& a1, const int& l2, const double& a2, const int& l3, const double& a3, const int& l4, const double& a4, const int& LL, const int& v1, const int& v2) const;
    double int2e_get_angular(const int& l1, const int& two_m1, const int& s1, const int& l2, const int& two_m2, const int& s2, const int& l3, const int& two_m3, const int& s3, const int& l4, const int& two_m4, const int& s4, const int& LL) const;
    double int2e_get_angular_K(const int& l1, const int& two_m1, const int& s1, const int& l2, const int& two_m2, const int& s2, const int& LL) const;
    double int2e_get_angular_J(const int& l1, const int& two_m1, const int& s1, const int& l2, const int& two_m2, const int& s2, const int& LL) const;
    double int2e_get_angular_gaunt_LSLS(const int& l1, const int& two_m1, const int& s1, const int& l2, const int& two_m2, const int& s2, const int& l3, const int& two_m3, const int& s3, const int& l4, const int& two_m4, const int& s4, const int& LL) const;
    double int2e_get_angular_gaunt_LSSL(const int& l1, const int& two_m1, const int& s1, const int& l2, const int& two_m2, const int& s2, const int& l3, const int& two_m3, const int& s3, const int& l4, const int& two_m4, const int& s4, const int& LL) const;
    double int2e_get_angular_gaunt_LSLS_9j(const int& l1, const int& two_m1, const int& s1, const int& l2, const int& two_m2, const int& s2, const int& l3, const int& two_m3, const int& s3, const int& l4, const int& two_m4, const int& s4, const int& LL) const;
    double int2e_get_angular_gaunt_LSSL_9j(const int& l1, const int& two_m1, const int& s1, const int& l2, const int& two_m2, const int& s2, const int& l3, const int& two_m3, const int& s3, const int& l4, const int& two_m4, const int& s4, const int& LL) const;
    double int2e_get_angular_gauge_LSLS(const int& l1, const int& two_m1, const int& s1, const int& l2, const int& two_m2, const int& s2, const int& l3, const int& two_m3, const int& s3, const int& l4, const int& two_m4, const int& s4, const int& LL, const int& v1, const int& v2) const;
    double int2e_get_angular_gauge_LSSL(const int& l1, const int& two_m1, const int& s1, const int& l2, const int& two_m2, const int& s2, const int& l3, const int& two_m3, const int& s3, const int& l4, const int& two_m4, const int& s4, const int& LL, const int& v1, const int& v2) const;
    void int2e_get_angular_gauntSF_LSLS(const int& l1, const int& two_m1, const int& s1, const int& l2, const int& two_m2, const int& s2, const int& l3, const int& two_m3, const int& s3, const int& l4, const int& two_m4, const int& s4, const int& LL, double& lsls11, double& lsls12, double& lsls21, double& lsls22);
    void int2e_get_angular_gauntSF_LSSL(const int& l1, const int& two_m1, const int& s1, const int& l2, const int& two_m2, const int& s2, const int& l3, const int& two_m3, const int& s3, const int& l4, const int& two_m4, const int& s4, const int& LL, double& lssl11, double& lssl12, double& lssl21, double& lssl22);
    double int2e_get_angular_gauntSF_p1_LS(const int& l1, const int& two_m1, const int& s1, const int& l2, const int& two_m2, const int& s2, const int& LL, const int& MM) const;
    double int2e_get_angular_gauntSF_p2_LS(const int& l1, const int& two_m1, const int& s1, const int& l2, const int& two_m2, const int& s2, const int& LL, const int& MM) const;
    double int2e_get_angular_gauntSF_m1_LS(const int& l1, const int& two_m1, const int& s1, const int& l2, const int& two_m2, const int& s2, const int& LL, const int& MM) const;
    double int2e_get_angular_gauntSF_m2_LS(const int& l1, const int& two_m1, const int& s1, const int& l2, const int& two_m2, const int& s2, const int& LL, const int& MM) const;
    double int2e_get_angular_gauntSF_z1_LS(const int& l1, const int& two_m1, const int& s1, const int& l2, const int& two_m2, const int& s2, const int& LL, const int& MM) const;
    double int2e_get_angular_gauntSF_z2_LS(const int& l1, const int& two_m1, const int& s1, const int& l2, const int& two_m2, const int& s2, const int& LL, const int& MM) const;
    double int2e_get_angular_gauntSF_p1_SL(const int& l1, const int& two_m1, const int& s1, const int& l2, const int& two_m2, const int& s2, const int& LL, const int& MM) const;
    double int2e_get_angular_gauntSF_p2_SL(const int& l1, const int& two_m1, const int& s1, const int& l2, const int& two_m2, const int& s2, const int& LL, const int& MM) const;
    double int2e_get_angular_gauntSF_m1_SL(const int& l1, const int& two_m1, const int& s1, const int& l2, const int& two_m2, const int& s2, const int& LL, const int& MM) const;
    double int2e_get_angular_gauntSF_m2_SL(const int& l1, const int& two_m1, const int& s1, const int& l2, const int& two_m2, const int& s2, const int& LL, const int& MM) const;
    double int2e_get_angular_gauntSF_z1_SL(const int& l1, const int& two_m1, const int& s1, const int& l2, const int& two_m2, const int& s2, const int& LL, const int& MM) const;
    double int2e_get_angular_gauntSF_z2_SL(const int& l1, const int& two_m1, const int& s1, const int& l2, const int& two_m2, const int& s2, const int& LL, const int& MM) const;

    inline double int2e_get_threeSH(const int& l1, const int& m1, const int& l2, const int& m2, const int& l3, const int& m3, const double& threeJ) const;
    double int2e_get_angularX_RME(const int& two_j1, const int& l1, const int& two_j2, const int& l2, const int& LL, const int& vv, const double& threeJ) const;
    inline double factor_p1(const int& l, const int& m) const;
    inline double factor_p2(const int& l, const int& m) const;
    inline double factor_m1(const int& l, const int& m) const;
    inline double factor_m2(const int& l, const int& m) const;
    inline double factor_z1(const int& l, const int& m) const;
    inline double factor_z2(const int& l, const int& m) const;
    

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
    int2eJK get_h2e_JK_gauntSF_compact(const string& intType, const int& occMaxL = -1);
    void get_h2e_JK_gaunt_direct(int2eJK& LSLS, int2eJK& LSSL, const int& occMaxL = -1, const bool& spinFree = false);
    void get_h2eSD_JK_gaunt_direct(int2eJK& LSLS, int2eJK& LSSL, const int& occMaxL = -1);
    int2eJK get_h2e_JK_gauge_compact(const string& intType, const int& occMaxL = -1) const;
    void get_h2e_JK_gauge_direct(int2eJK& LSLS, int2eJK& LSSL, const int& occMaxL = -1, const bool& spinFree = false);
    
    /* get contraction coefficients for uncontracted calculations */
    MatrixXd get_coeff_contraction_spinor();
};

#endif