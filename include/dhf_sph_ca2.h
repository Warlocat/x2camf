#ifndef DHF_SPH_CA2_H_
#define DHF_SPH_CA2_H_
#include<Eigen/Dense>
#include<complex>
#include<string>
#include"dhf_sph.h"
using namespace std;
using namespace Eigen;

class DHF_SPH_CA2: public DHF_SPH
{
private:
    int openShell, NOpenShells;
    vector<double> NN_list, MM_list, f_list, a_list;
    vector<vVectorXd> occNumberShells;
    Matrix<vMatrixXd,-1,1> densityShells, coeffShells, fockShells;
    
    MatrixXd evaluateDensity_aoc(const MatrixXd& coeff_, const VectorXd& occNumber_, const bool& twoC);
    /* evaluate fock matrix */
    void evaluateFockShells(Matrix<vMatrixXd,-1,1>& fockShells, const bool& twoC, const Matrix<vMatrixXd,-1,1>& densities, const int& size, const int& Iirrep);
    double evaluateEnergy(const bool& twoC);

public:
    DHF_SPH_CA2(INT_SPH& int_sph_, const string& filename, const bool& spinFree = false, const bool& twoC = false, const bool& with_gaunt_ = false, const bool& with_gauge_ = false, const bool& allInt = false, const bool& gaussian_nuc = false);
    virtual ~DHF_SPH_CA2();
    virtual void runSCF(const bool& twoC = false, const bool& renormSmall = true) override;
};


#endif