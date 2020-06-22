#include<string>
#include<iostream>
#include<iomanip>
#include"scf.h"
#include"x2c.h"

using namespace std;
using namespace Eigen;

X2C::X2C(const GTO& gto_):
size_basis(gto_.size_gtou), S(gto_.get_h1e("overlap",true)), T(gto_.get_h1e("kinetic",true)), 
W(gto_.get_h1e("p.Vp",true)), V(gto_.get_h1e("nuc_attra",true)), coeff_contraction(gto_.get_coeff_contraction())
{
}

X2C::~X2C()
{
}

MatrixXd X2C::get_X(const MatrixXd& S_, const MatrixXd& T_, const MatrixXd& W_, const MatrixXd& V_)
{
    int size = S_.rows();
    MatrixXd h_4C(2*size, 2*size), overlap(2*size, 2*size), overlap_h_i(2*size, 2*size);
    MatrixXd coeff_tmp(2*size, 2*size), coeff_large(size,size), coeff_small(size,size);
    VectorXd ene_tmp(2*size);
    
    for(int ii = 0; ii < size; ii++)
    for(int jj = 0; jj < size; jj++)
    {
        h_4C(ii,jj) = V_(ii,jj);
        h_4C(ii,size+jj) = T_(ii,jj);
        h_4C(size+ii,jj) = T_(ii,jj);
        h_4C(size+ii,size+jj) = W_(ii,jj)/4.0/pow(speedOfLight,2) - T_(ii,jj);
        overlap(ii,jj) = S_(ii,jj);
        overlap(size+ii,size+jj) = T_(ii,jj)/2.0/pow(speedOfLight,2); 
        overlap(size+ii,jj) = 0.0;
        overlap(ii,size+jj) = 0.0;
    }
    
    overlap_h_i = SCF::matrix_half_inverse(overlap);
    SCF::eigensolverG(h_4C, overlap_h_i, ene_tmp, coeff_tmp);

    for(int ii = 0; ii < size; ii++)
    for(int jj = 0; jj < size; jj++)
    {
        coeff_large(ii,jj) = coeff_tmp(ii,size+jj);
        coeff_small(ii,jj) = coeff_tmp(ii+size,jj+size);
    }

    return coeff_small * coeff_large.inverse();
}

MatrixXd X2C::get_R(const MatrixXd& S_, const MatrixXd& T_, const MatrixXd& X_)
{
    int size = S_.rows();
    MatrixXd S_tilde(size,size), S_h_i(size,size), S_h(size,size);
    S_tilde = S_ + 0.5/pow(speedOfLight,2) * X_.transpose() * T_ * X_;
    S_h_i = SCF::matrix_half_inverse(S_);
    S_h = S_h_i.inverse();
    MatrixXd tmp = SCF::matrix_half_inverse(S_h_i * S_tilde * S_h_i);
    return S_h_i * tmp * S_h;
}


MatrixXd X2C::evaluate_h1e_x2c(const MatrixXd& S_, const MatrixXd& T_, const MatrixXd& W_, const MatrixXd& V_, const MatrixXd coeff_contraction_)
{
    MatrixXd X = get_X(S_, T_, W_, V_);
    MatrixXd R = get_R(S_, T_, X);
    MatrixXd h_eff = V_ + T_ * X + X.transpose() * T_ - X.transpose() * T_ * X + 0.25/pow(speedOfLight,2) * X.transpose() * W_ * X;
    MatrixXd h1e = R.transpose() *  h_eff * R;

    return coeff_contraction_.transpose() * h1e * coeff_contraction_;

    // int size = S_.rows();
    // MatrixXd h_4C(2*size, 2*size), overlap_4C(2*size, 2*size), overlap_4C_h_i(2*size, 2*size);
    // MatrixXd coeff_tmp(2*size, 2*size), coeff_large(size,size), coeff_small(size,size);
    // VectorXd ene_tmp(2*size),ene_elec(size);

    // for(int ii = 0; ii < 2*size; ii++)
    // for(int jj = 0; jj < 2*size; jj++)
    // {
    //     h_4C(ii,jj) = 0.0;
    //     overlap_4C(ii,jj) = 0.0;
    // }
    
    
    // for(int ii = 0; ii < size; ii++)
    // for(int jj = 0; jj < size; jj++)
    // {
    //     h_4C(ii,jj) = V_(ii,jj);
    //     h_4C(ii,size+jj) = T_(ii,jj);
    //     h_4C(size+ii,jj) = T_(ii,jj);
    //     h_4C(size+ii,size+jj) = W_(ii,jj)/4.0/pow(speedOfLight,2) - T_(ii,jj);
    //     overlap_4C(ii,jj) = S_(ii,jj);
    //     overlap_4C(size+ii,size+jj) = T_(ii,jj)/2.0/pow(speedOfLight,2); 
    // }
    
    // overlap_4C_h_i = SCF::matrix_half_inverse(overlap_4C);
    // SCF::eigensolverG(h_4C, overlap_4C_h_i, ene_tmp, coeff_tmp);

    // for(int ii = 0; ii < size; ii++)
    // {
    //     ene_elec(ii) = ene_tmp(ii+size);
    //     for(int jj = 0; jj < size; jj++)
    //     {
    //         coeff_large(ii,jj) = coeff_tmp(ii,size+jj);
    //         coeff_small(ii,jj) = coeff_tmp(ii+size,jj+size);
    //     }
    // }

    // MatrixXd RA_i = SCF::matrix_half_inverse(coeff_large.transpose()*S_*coeff_large);
    // MatrixXd tmp_mat = RA_i * coeff_large.transpose() * S_;
    // MatrixXd h1e(size,size);
    
    // for(int ii = 0; ii < size; ii++)
    // for(int jj = 0; jj < size; jj++)
    // {
    //     h1e(ii,jj) = 0.0;
    //     for(int kk = 0; kk < size; kk++)
    //         h1e(ii,jj) += tmp_mat.transpose()(ii,kk) * ene_elec(kk) * tmp_mat(kk,jj);
    // }

    // return coeff_contraction_.transpose() * h1e * coeff_contraction_;
}