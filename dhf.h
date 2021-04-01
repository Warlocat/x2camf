#ifndef DHF_H_
#define DHF_H_
#include<Eigen/Dense>
#include<complex>
#include<string>
#include"gto.h"
#include"gto_spinor.h"
#include"x2c.h"
#include"scf.h"
using namespace std;
using namespace Eigen;

class DHF: public SCF
{
private:
    /* In DHF, h1e is V and h2e is h2eLLLL */
    MatrixXd overlap_4c, overlap_half_i_4c, kinetic, WWW, h2eSSLL, h2eSSSS;
    MatrixXd density, fock_4c, h1e_4c, coeff_contract;
    VectorXd norm_s;
    double d_density;
    bool uncontracted = true;

    static void readIntegrals(MatrixXd& h2e_, const string& filename);
    static MatrixXd evaluateDensity_spinor(const MatrixXd& coeff_, const int& nocc, const bool& spherical = false);
public:
    MatrixXd coeff;
    VectorXd ene_orb;

    DHF(GTO_SPINOR& gto_, const bool& unc);
    DHF(const GTO_SPINOR& gto_, const string& h2e_file, const bool& unc);
    DHF(const GTO_SPINOR& gto_, const MatrixXd& h2eLLLL_, const MatrixXd& h2eSSLL_, const MatrixXd& h2eSSSS_, const bool& unc);
    DHF(const string& h1e_file, const string& h2e_file);
    virtual ~DHF();
    virtual void runSCF();
    MatrixXd get_amfi(const MatrixXd& h2eSSLL_SD, const MatrixXd& h2eSSSS_SD, const MatrixXd& coeff_con, const bool& spherical = false);
    static MatrixXd get_amfi(const MatrixXd& coeff_4c, const MatrixXd& h2eSSLL_SD, const MatrixXd& h2eSSSS_SD, const MatrixXd& h1e_4c_, const MatrixXd& overlap_4c_, const int& nocc, const MatrixXd& coeff_con, const bool& spherical = false);
    Matrix<MatrixXcd,3,1> get_amfi_Pauli(const MatrixXd& coeff_4c, const MatrixXd& h2eSSLL_SD, const MatrixXd& h2eSSSS_SD, const MatrixXd& h1e_4c_, const MatrixXd& overlap_4c_, const int& nocc, const MatrixXd& coeff_con, const bool& spherical);
    Matrix<MatrixXcd,3,1> get_amfi_Pauli(const MatrixXd& amfi_2c);
};


#endif