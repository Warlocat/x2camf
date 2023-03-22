#ifndef DHF_SPH_CA_H_
#define DHF_SPH_CA_H_
#include<complex>
#include<string>
#include"dhf_sph.h"
using namespace std;

class DHF_SPH_CA: public DHF_SPH
{
private:
    int openShell, NOpenShells;
    vector<double> NN_list, MM_list, f_list;
    vector<vVectorXd> occNumberShells;
    /* in CAHF, density is density_c */
    vector<vVectorXd> densityShells;
    
    vector<double> evaluateDensity_aoc(const vector<double>& coeff_, const vector<double>& occNumber_, const int& size, const bool& twoC);
    /* evaluate fock matrix */
    void evaluateFock(vector<double>& fock, const bool& twoC, const vector<vVectorXd>& densities, const int& size, const int& Iirrep);
    void evaluateFock_2e(vector<double>& fock, const bool& twoC, const vector<vVectorXd>& densities, const int& size, const int& Iirrep);
    double evaluateEnergy(const bool& twoC);

public:
    DHF_SPH_CA(INT_SPH& int_sph_, const string& filename, const bool& spinFree = false, const bool& twoC = false, const bool& with_gaunt_ = false, const bool& with_gauge_ = false, const bool& allInt = false, const bool& gaussian_nuc = false);
    virtual ~DHF_SPH_CA();
    virtual void runSCF(const bool& twoC = false, const bool& renormSmall = true) override;
    virtual vVectorXd x2c2ePCC(vVectorXd* coeff2c = NULL) override;
};


#endif
