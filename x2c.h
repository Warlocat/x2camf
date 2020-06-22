#ifndef X2C_H_
#define X2C_H_

#include<Eigen/Dense>
#include<complex>
#include<string>
#include"gto.h"
using namespace std;
using namespace Eigen;

// const double speedOfLight = 13.7; // for debug
const double speedOfLight = 137.03599967994;

class X2C
{
protected:
    int size_basis;
    MatrixXd S, T, W, V, coeff_contraction;   
    
public:
    X2C(const GTO& gto_);
    ~X2C();
    static MatrixXd get_X(const MatrixXd& S_, const MatrixXd& T_, const MatrixXd& W_, const MatrixXd& V_);
    static MatrixXd get_R(const MatrixXd& S_, const MatrixXd& T_, const MatrixXd& X_);
    static MatrixXd evaluate_h1e_x2c(const MatrixXd& S_, const MatrixXd& T_, const MatrixXd& W_, const MatrixXd& V_, const MatrixXd coeff_contraction_);
};




#endif