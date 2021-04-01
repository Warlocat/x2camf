#ifndef SCF_H_
#define SCF_H_

#include<Eigen/Dense>
#include<complex>
#include<string>
#include"gto.h"
#include"gto_spinor.h"
#include"x2c.h"
using namespace std;
using namespace Eigen;

extern "C" void eig_(double *A, double* B, int* N, int* N1);
void eig_cfour_symm(const MatrixXd inputM, VectorXd& values, MatrixXd& vectors);

class SCF
{
protected:
    int nelec_a, nelec_b, size_basis;
    MatrixXd overlap, overlap_half_i, h1e, h2e;

    void readIntegrals(const string& filename);
    static double evaluateChange(const MatrixXd& M1, const MatrixXd& M2);
    static MatrixXd evaluateErrorDIIS(const MatrixXd& fock_, const MatrixXd& overlap_, const MatrixXd& density_);
    static MatrixXd matrix_half_inverse(const MatrixXd& inputM);
    static MatrixXd matrix_half(const MatrixXd& inputM);
    static MatrixXd evaluateDensity(const MatrixXd& coeff_, const int& nocc, const bool& spherical = false);
    static void eigensolverG(const MatrixXd& inputM, const MatrixXd& s_h_i, VectorXd& values, MatrixXd& vectors);
    friend X2C;
public:
    bool converged = false;
    int maxIter = 200, size_DIIS = 8;
    double ene_scf, convControl = 1e-10;

    SCF(const GTO& gto_, const string& h2e_file, const string& relativistic = "off");
    SCF(const GTO& gto_, const MatrixXd& h2e_, const string& relativistic = "off");
    SCF();
    virtual ~SCF();

    virtual void runSCF() = 0;
};

class RHF: public SCF
{
private:
    MatrixXd density, fock;
    double d_density;
public:
    MatrixXd coeff;
    VectorXd ene_orb;

    RHF(const GTO& gto_, const string& h2e_file, const string& relativistic);
    RHF(const GTO& gto_, const MatrixXd& h2e_, const string& relativistic);
    virtual ~RHF();
    virtual void runSCF();
};


class UHF: public SCF
{
private:
    MatrixXd density_a, density_b, fock_a, fock_b;
    double d_density_a, d_density_b;
public:
    MatrixXd coeff_a, coeff_b;
    VectorXd ene_orb_a, ene_orb_b;

    UHF(const GTO& gto_, const string& h2e_file, const string& relativistic);
    UHF(const GTO& gto_, const MatrixXd& h2e_, const string& relativistic);
    virtual ~UHF();
    virtual void runSCF();
};

SCF* scf_init(const GTO& gto_, const string& h2e_file, const string& relativistic = "off");
SCF* scf_init(const GTO& gto_, const MatrixXd& h2e_, const string& relativistic = "off");



#endif
