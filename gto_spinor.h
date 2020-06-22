#ifndef GTO_SPINOR_H_
#define GTO_SPINOR_H_

#include<Eigen/Dense>
#include<complex>
#include<string>
#include"gto.h"
using namespace std;
using namespace Eigen;


/*
    Class for single-atom (single-center) integrals in 2-spinor basis.
    Derived from class GTO.

    Variables:
        
*/
class GTO_SPINOR: public GTO
{
public:
    int size_gtoc_spinor, size_gtou_spinor;

    GTO_SPINOR(const string& atomName_, const string& basisSet_, const int& charge_ = 0, const int& spin_ = 1, const bool& uncontracted_ = false);
    ~GTO_SPINOR();

    /* return needed 1e and 2e integrals */
    MatrixXd get_h1e(const string& integralTYPE, const bool& uncontracted_ = false) const;
    MatrixXd get_h1e_spin_orbitals(const string& integralTYPE, const bool& uncontracted_ = false) const;
    MatrixXd get_h2e(const string& integralTYPE, const bool& uncontracted_ = false) const;

    /* evaluate radial part and angular part in 2e integrals */
    double int2e_get_radial_LLLL(const int& l1, const double& k1, const double& a1, const int& l2, const double& k2, const double& a2, const int& l3, const double& k3, const double& a3, const int& l4, const double& k4, const double& a4, const int& LL) const;
    double int2e_get_radial_SSLL(const int& l1, const double& k1, const double& a1, const int& l2, const double& k2, const double& a2, const int& l3, const double& k3, const double& a3, const int& l4, const double& k4, const double& a4, const int& LL) const;
    double int2e_get_radial_SSSS(const int& l1, const double& k1, const double& a1, const int& l2, const double& k2, const double& a2, const int& l3, const double& k3, const double& a3, const int& l4, const double& k4, const double& a4, const int& LL) const;

    double int2e_get_angular(const int& l1, const int& two_m1, const int& a1, const int& l2, const int& two_m2, const int& a2, const int& l3, const int& two_m3, const int& a3, const int& l4, const int& two_m4, const int& a4, const int& LL) const;
    
    /* write n_a, n_b, n_basis, and h2e for scf */
    void writeIntegrals_spinor(const MatrixXd& h2e, const string& filename);
};

#endif