#ifndef GTO_H_
#define GTO_H_

#include<Eigen/Dense>
#include<complex>
#include<string>
using namespace std;
using namespace Eigen;


/* factorial and double factorial functions */
double factorial(const int& n);
double double_factorial(const int& n);

/* functions used to evaluate wigner 3j symbols */
double wigner_3j(const int& l1, const int& l2, const int& l3, const int& m1, const int& m2, const int& m3);
double wigner_3j_zeroM(const int& l1, const int& l2, const int& l3);

/* function used to evaluate spherical harmonics transition matrix */
complex<double> U_SH_trans(const int& mu, const int& mm);

/*
    contracted gtos in form of angular shell
*/
class gto_contracted
{
public:
    VectorXd exp_a, norm;
    MatrixXd coeff;
    int l;
};


/*
    Class for single-atom (single-center) integrals

    Variables:
        atomNumber:         atomic number
        charge:             charge
        nelec:              number of electrons
        nelec_a:            number of alpha electrons
        nelec_b:            number of beta electrons
        spin:               spin state, 2S + 1
        size_gtoc:          number of contracted basis set
        size_shell:         number of angular shell

        atomName:           name of atom (all in capital letter)
        basisSet:           name of basis set (CFOUR basis set form)
    
        uncontracted:       for uncontracted calculation

        shell_list:         basis set information stored in 
                            the form of different angular shells
*/
class GTO
{
protected:
    Matrix<gto_contracted, Dynamic, 1> shell_list; 

public:
    int atomNumber, charge, nelec, nelec_a, nelec_b, spin;
    int size_gtoc, size_gtou, size_shell;
    string atomName, basisSet;
    bool uncontracted;

    /* Construction and destruction functions*/
    GTO(const string& atomName_, const string& basisSet_, const int& charge_ = 0, const int& spin_ = 1, const bool& uncontracted_ = false);
    ~GTO();
    /* read basis set and normalization, used in construction functions*/
    void readBasis();
    void normalization();

    /* return needed 1e and 2e integrals */
    MatrixXd get_h1e(const string& integralTYPE, const bool& uncontracted_ = false) const;
    MatrixXd get_h2e(const bool& uncontracted_ = false) const;

    /* auxiliary functions used to evaluate 1e and 2e intergals */
    inline double auxiliary_1e(const int& l, const double& a) const;
    inline double auxiliary_2e_0_r(const int& l1, const int& l2, const double& a1, const double& a2) const;
    inline double auxiliary_2e_r_inf(const int& l1, const int& l2, const double& a1, const double& a2) const;

    /* evaluate 1e and 2e integrals in single gto basis, not used in current version */
    double int1e_single_gto(const int& l1, const int& m1, const double& a1, const int& l2, const int& m2, const double& a2, const string& integralTYPE);
    double int2e_single_gto(const int& l1, const int& m1, const double& a1, const int& l2, const int& m2, const double& a2, const int& l3, const int& m3, const double& a3, const int& l4, const int& m4, const double& a4);

    /* evaluate radial part and angular part in 2e integrals */
    double int2e_get_radial(const int& l1, const double& a1, const int& l2, const double& a2, const int& l3, const double& a3, const int& l4, const double& a4, const int& LL) const;
    double int2e_get_angular(const int& l1, const int& m1, const int& l2, const int& m2, const int& l3, const int& m3, const int& l4, const int& m4, const int& LL) const;

    /* get contraction coefficients for uncontracted calculations */
    MatrixXd get_coeff_contraction() const;

    /* write n_a, n_b, n_basis, and h2e for scf */
    void writeIntegrals(const MatrixXd& h2e, const string& filename);
};

#endif