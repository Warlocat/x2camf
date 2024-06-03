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
    int openShell, NOpenShells;
    vector<double> NN_list, MM_list, f_list;
    vector<vVectorXd> occNumberShells;
    /* in CAHF, density is density_c */
    Matrix<vMatrixXd,-1,1> densityShells;
    
    MatrixXd evaluateDensity_aoc(const MatrixXd& coeff_, const VectorXd& occNumber_, const bool& twoC);
    /* evaluate fock matrix */
    void evaluateFock(MatrixXd& fock, const bool& twoC, const Matrix<vMatrixXd,-1,1>& densities, const int& size, const int& Iirrep);
    double evaluateEnergy(const bool& twoC);

public:
    DHF_SPH_CA(INT_SPH& int_sph_, const string& filename, const int& printLevel,const bool& spinFree = false, const bool& twoC = false, const bool& with_gaunt_ = false, const bool& with_gauge_ = false, const bool& allInt = false, const bool& gaussian_nuc = false);
    virtual ~DHF_SPH_CA();
    void coreIonization(const vector<vector<int>> coreHoleInfo);
    virtual void runSCF(const bool& twoC = false, const bool& renormSmall = true) override;
};


#endif