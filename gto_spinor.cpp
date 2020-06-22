#include<Eigen/Dense>
#include<string>
#include<iostream>
#include<iomanip>
#include<fstream>
#include<cmath>
#include<complex>
#include<omp.h>
#include<gsl/gsl_sf_coupling.h>
#include"gto_spinor.h"
using namespace std;
using namespace Eigen;

GTO_SPINOR::GTO_SPINOR(const string& atomName_, const string& basisSet_, const int& charge_, const int& spin_, const bool& uncontracted_):
GTO(atomName_, basisSet_, charge_, spin_, uncontracted_)
{
    size_gtoc_spinor = 2 * size_gtoc;
    size_gtou_spinor = 2 * size_gtou;
}

GTO_SPINOR::~GTO_SPINOR()
{
}


/*
    Evaluate different one-electron integrals in 2-spinor basis
*/
MatrixXd GTO_SPINOR::get_h1e(const string& intType, const bool& uncontracted_) const
{
    MatrixXd int_1e;
    int int_tmp = 0;
    if(!uncontracted_)
    {
        int_1e.resize(size_gtoc_spinor, size_gtoc_spinor);
        int_1e = MatrixXd::Zero(size_gtoc_spinor,size_gtoc_spinor);
    }
    else
    {
        int_1e.resize(size_gtou_spinor, size_gtou_spinor);
        int_1e = MatrixXd::Zero(size_gtou_spinor,size_gtou_spinor);
    }

    for(int ishell = 0; ishell < size_shell; ishell++)
    {
        int ll = shell_list(ishell).l;
        int size_gtos = shell_list(ishell).coeff.rows();
        for(int twojj = abs(2*ll-1); twojj <= 2*ll+1; twojj = twojj + 2)
        {
            double kappa = (twojj + 1.0) * (ll - twojj/2.0);
            MatrixXd h1e_single_shell(size_gtos, size_gtos);
            for(int ii = 0; ii < size_gtos; ii++)
            for(int jj = 0; jj < size_gtos; jj++)
            {
                double a1 = shell_list(ishell).exp_a(ii), a2 = shell_list(ishell).exp_a(jj);
                if(intType == "s_p_nuc_s_p")
                {
                    h1e_single_shell(ii,jj) = 4*a1*a2 * auxiliary_1e(2*ll + 3, a1 + a2);
                    if(ll!=0)
                        h1e_single_shell(ii,jj) += pow(ll + kappa + 1.0, 2) * auxiliary_1e(2*ll-1, a1 + a2) - 2.0*(ll + kappa + 1.0)*(a1 + a2)*auxiliary_1e(2*ll + 1, a1 + a2);
                        h1e_single_shell(ii,jj) *= -atomNumber;
                }
                else if(intType == "s_p_s_p" )
                {
                    h1e_single_shell(ii,jj) = 4*a1*a2 * auxiliary_1e(2*ll + 4, a1 + a2);
                    if(ll!=0)
                        h1e_single_shell(ii,jj) += pow(ll + kappa + 1.0, 2) * auxiliary_1e(2*ll, a1 + a2) - 2.0*(ll + kappa + 1.0)*(a1 + a2)*auxiliary_1e(2*ll + 2, a1 + a2);
                }
                else if(intType == "overlap")  h1e_single_shell(ii,jj) = auxiliary_1e(2 + 2*ll, a1 + a2);
                else if(intType == "nuc_attra")  h1e_single_shell(ii,jj) = -atomNumber * auxiliary_1e(1 + 2*ll, a1 + a2);
                else if(intType == "kinetic")
                {
                    h1e_single_shell(ii,jj) = 4*a1*a2 * auxiliary_1e(2*ll + 4, a1 + a2);
                    if(ll!=0)
                        h1e_single_shell(ii,jj) += pow(ll + kappa + 1.0, 2) * auxiliary_1e(2*ll, a1 + a2) - 2.0*(ll + kappa + 1.0)*(a1 + a2)*auxiliary_1e(2*ll + 2, a1 + a2);
                    h1e_single_shell(ii,jj) /= 2.0;
                }
                else
                {
                    cout << "ERROR: get_h1e is called for undefined type of integrals!" << endl;
                    exit(99);
                }
                h1e_single_shell(ii,jj) = h1e_single_shell(ii,jj) / shell_list(ishell).norm(ii) / shell_list(ishell).norm(jj);
            }

            if(!uncontracted_)
            {
                int size_subshell = shell_list(ishell).coeff.cols();
                MatrixXd int_1e_shell(size_subshell,size_subshell);
                for(int ii = 0; ii < size_subshell; ii++)
                for(int jj = 0; jj < size_subshell; jj++)
                {
                    int_1e_shell(ii,jj) = 0.0;
                    for(int mm = 0; mm < size_gtos; mm++)
                    for(int nn = 0; nn < size_gtos; nn++)
                    {
                        int_1e_shell(ii,jj) += shell_list(ishell).coeff(mm, ii) * shell_list(ishell).coeff(nn, jj) * h1e_single_shell(mm,nn);
                    }
                }
                for(int ii = 0; ii < size_subshell; ii++)
                for(int jj = 0; jj < size_subshell; jj++)
                for(int kk = 0; kk < twojj+1; kk++)
                {
                    int_1e(int_tmp + kk + ii * (twojj+1), int_tmp + kk + jj * (twojj+1)) = int_1e_shell(ii,jj);
                }
                int_tmp += size_subshell * (twojj+1);
            }
            else
            {
                for(int ii = 0; ii < size_gtos; ii++)
                for(int jj = 0; jj < size_gtos; jj++)
                for(int kk = 0; kk < twojj+1; kk++)
                {
                    int_1e(int_tmp + kk + ii * (twojj+1), int_tmp + kk + jj * (twojj+1)) = h1e_single_shell(ii,jj);
                }
                int_tmp += size_gtos * (twojj+1);
            }
        }
    }

    return int_1e;
}


/*
    Evaluate different one-electron integrals in spin orbital basis,
    i.e. A_so = A_nr \otimes I_2
*/
MatrixXd GTO_SPINOR::get_h1e_spin_orbitals(const string& intType, const bool& uncontracted_) const
{
    MatrixXd int_1e_nr = GTO::get_h1e(intType, uncontracted_);
    int size_tmp = int_1e_nr.rows();
    MatrixXd int_1e_so(2*size_tmp, 2*size_tmp);

    for(int ii = 0; ii < size_tmp; ii++)
    for(int jj = 0; jj < size_tmp; jj++)
    {
        int_1e_so(2*ii,2*jj) = int_1e_nr(ii,jj);
        int_1e_so(2*ii,2*jj+1) = 0.0;
        int_1e_so(2*ii+1,2*jj) = 0.0;
        int_1e_so(2*ii+1,2*jj+1) = int_1e_nr(ii,jj);
    }
    
    return int_1e_so;
}

/*
    Evaluate different two-electron integrals (Coulomb) in 2-spinor basis
*/
MatrixXd GTO_SPINOR::get_h2e(const string& integralTYPE, const bool& uncontracted_) const
{
    int size_2e, size_2e_2;
    MatrixXd int_2e;
    if(!uncontracted_)
    {
        size_2e = size_gtoc_spinor;
    }
    else
    {
        size_2e = size_gtou_spinor;
    }
    size_2e_2 = size_2e * size_2e;
    int_2e.resize(size_2e_2,size_2e_2);
    int_2e = MatrixXd::Zero(size_2e_2,size_2e_2);
    
    VectorXd radial_tilde;
    int int_tmp_i, int_tmp_j, int_tmp_k, int_tmp_l;
    if(!uncontracted_)
    {
        int_tmp_i = 0;
        for(int ishell = 0; ishell < size_shell; ishell++)
        {
        int_tmp_j = 0;
        for(int jshell = 0; jshell < size_shell; jshell++)
        {
        int_tmp_k = 0;
        for(int kshell = 0; kshell < size_shell; kshell++)
        {
        int_tmp_l = 0;
        for(int lshell = 0; lshell < size_shell; lshell++)
        {
            int l_i = shell_list(ishell).l, l_j = shell_list(jshell).l, l_k = shell_list(kshell).l, l_l = shell_list(lshell).l, Lmax = min(l_i + l_j, l_k +l_l);
            if((l_i+l_j+l_k+l_l)%2) 
            {
                int_tmp_l += shell_list(lshell).coeff.cols() * (2*shell_list(lshell).l+1) * 2;
                continue;
            }
            int size_gtos_i = shell_list(ishell).coeff.rows(), size_gtos_j = shell_list(jshell).coeff.rows(), size_gtos_k = shell_list(kshell).coeff.rows(), size_gtos_l = shell_list(lshell).coeff.rows();
            int size_subshell_i = shell_list(ishell).coeff.cols(), size_subshell_j = shell_list(jshell).coeff.cols(), size_subshell_k = shell_list(kshell).coeff.cols(), size_subshell_l = shell_list(lshell).coeff.cols();

            radial_tilde.resize(Lmax+1);       
            
            int int_tmp2_i = 0;
            for(int twojj_i = abs(2*l_i-1); twojj_i <= 2*l_i+1; twojj_i = twojj_i + 2)
            {
            int int_tmp2_j = 0;
            for(int twojj_j = abs(2*l_j-1); twojj_j <= 2*l_j+1; twojj_j = twojj_j + 2)
            {
            int int_tmp2_k = 0;
            for(int twojj_k = abs(2*l_k-1); twojj_k <= 2*l_k+1; twojj_k = twojj_k + 2)
            {
            int int_tmp2_l = 0;
            for(int twojj_l = abs(2*l_l-1); twojj_l <= 2*l_l+1; twojj_l = twojj_l + 2)
            {
                int sym_ai = twojj_i - 2*l_i, sym_aj = twojj_j - 2*l_j, sym_ak = twojj_k - 2*l_k, sym_al = twojj_l - 2*l_l;
                double k_i = -(twojj_i+1.0)*sym_ai/2.0, k_j = -(twojj_j+1.0)*sym_aj/2.0, k_k = -(twojj_k+1.0)*sym_ak/2.0, k_l = -(twojj_l+1.0)*sym_al/2.0;
                VectorXd array_angular[twojj_i + 1][twojj_j + 1][twojj_k + 1][twojj_l + 1];

                for(int mi = 0; mi < twojj_i + 1; mi++)
                for(int mj = 0; mj < twojj_j + 1; mj++)
                for(int mk = 0; mk < twojj_k + 1; mk++)
                for(int ml = 0; ml < twojj_l + 1; ml++)
                {
                    array_angular[mi][mj][mk][ml].resize(Lmax+1);
                    array_angular[mi][mj][mk][ml] = VectorXd::Zero(Lmax+1);
                    for(int tmp = Lmax; tmp >= 0; tmp = tmp - 2)
                        array_angular[mi][mj][mk][ml](tmp) = int2e_get_angular(l_i, 2*mi-twojj_i, sym_ai, l_j, 2*mj-twojj_j, sym_aj, l_k, 2*mk-twojj_k, sym_ak, l_l, 2*ml-twojj_l, sym_al, tmp);
                }

                for(int ii = 0; ii < size_subshell_i; ii++)
                for(int jj = 0; jj < size_subshell_j; jj++)
                for(int kk = 0; kk < size_subshell_k; kk++)
                for(int ll = 0; ll < size_subshell_l; ll++)
                {
                    radial_tilde = VectorXd::Zero(Lmax+1);
                    for(int iii = 0; iii < size_gtos_i; iii++)
                    for(int jjj = 0; jjj < size_gtos_j; jjj++)
                    for(int kkk = 0; kkk < size_gtos_k; kkk++)
                    for(int lll = 0; lll < size_gtos_l; lll++)
                    {
                        double norm = shell_list(ishell).norm(iii) * shell_list(jshell).norm(jjj) * shell_list(kshell).norm(kkk) * shell_list(lshell).norm(lll);
                        /*
                            radial_tilde is the contracted radial part
                        */
                        for(int tmp = Lmax; tmp >= 0; tmp = tmp - 2)
                        {
                            if(integralTYPE == "LLLL")
                                radial_tilde(tmp) += shell_list(ishell).coeff(iii,ii) * shell_list(jshell).coeff(jjj,jj) * shell_list(kshell).coeff(kkk,kk) * shell_list(lshell).coeff(lll,ll) * int2e_get_radial_LLLL(l_i, k_i, shell_list(ishell).exp_a(iii), l_j, k_j, shell_list(jshell).exp_a(jjj), l_k, k_k, shell_list(kshell).exp_a(kkk), l_l, k_l, shell_list(lshell).exp_a(lll), tmp) / norm;
                            else if(integralTYPE == "SSLL")
                                radial_tilde(tmp) += shell_list(ishell).coeff(iii,ii) * shell_list(jshell).coeff(jjj,jj) * shell_list(kshell).coeff(kkk,kk) * shell_list(lshell).coeff(lll,ll) * int2e_get_radial_SSLL(l_i, k_i, shell_list(ishell).exp_a(iii), l_j, k_j, shell_list(jshell).exp_a(jjj), l_k, k_k, shell_list(kshell).exp_a(kkk), l_l, k_l, shell_list(lshell).exp_a(lll), tmp) / norm;
                            else if(integralTYPE == "LLSS")
                                radial_tilde(tmp) += shell_list(ishell).coeff(iii,ii) * shell_list(jshell).coeff(jjj,jj) * shell_list(kshell).coeff(kkk,kk) * shell_list(lshell).coeff(lll,ll) * int2e_get_radial_SSLL(l_k, k_k, shell_list(kshell).exp_a(kkk), l_l, k_l, shell_list(lshell).exp_a(lll), l_i, k_i, shell_list(ishell).exp_a(iii), l_j, k_j, shell_list(jshell).exp_a(jjj), tmp) / norm;
                            else if(integralTYPE == "SSSS")
                                radial_tilde(tmp) += shell_list(ishell).coeff(iii,ii) * shell_list(jshell).coeff(jjj,jj) * shell_list(kshell).coeff(kkk,kk) * shell_list(lshell).coeff(lll,ll) * int2e_get_radial_SSSS(l_i, k_i, shell_list(ishell).exp_a(iii), l_j, k_j, shell_list(jshell).exp_a(jjj), l_k, k_k, shell_list(kshell).exp_a(kkk), l_l, k_l, shell_list(lshell).exp_a(lll), tmp) / norm;
                            else
                            {
                                cout << "ERROR: integralTYPE of get_h2e must be one of these:\n";
                                cout << "       LLLL,  SSLL,  LLSS, SSSS\n";
                                exit(99);
                            }
                        }
                    }
                    
                    for(int mi = 0; mi < twojj_i + 1; mi++)
                    for(int mj = 0; mj < twojj_j + 1; mj++)
                    for(int mk = 0; mk < twojj_k + 1; mk++)
                    for(int ml = 0; ml < twojj_l + 1; ml++)
                    {
                        int ei = int_tmp_i + int_tmp2_i + mi + ii * (twojj_i+1), ej = int_tmp_j + int_tmp2_j + mj + jj * (twojj_j+1), ek = int_tmp_k + int_tmp2_k + mk + kk * (twojj_k+1), el = int_tmp_l + int_tmp2_l + ml + ll * (twojj_l+1);
                        int eij = ei*size_2e+ej, ekl = ek*size_2e+el;
                        int_2e(eij,ekl) = radial_tilde.transpose() * array_angular[mi][mj][mk][ml];
                    }
                }
                int_tmp2_l += shell_list(lshell).coeff.cols() * (twojj_l+1);
            }
                int_tmp2_k += shell_list(kshell).coeff.cols() * (twojj_k+1);
            }
                int_tmp2_j += shell_list(jshell).coeff.cols() * (twojj_j+1);
            }
                int_tmp2_i += shell_list(ishell).coeff.cols() * (twojj_i+1);
            }
            int_tmp_l += shell_list(lshell).coeff.cols() * (2*shell_list(lshell).l+1) * 2;
        }
            int_tmp_k += shell_list(kshell).coeff.cols() * (2*shell_list(kshell).l+1) * 2;
        }
            int_tmp_j += shell_list(jshell).coeff.cols() * (2*shell_list(jshell).l+1) * 2;
        }
            int_tmp_i += shell_list(ishell).coeff.cols() * (2*shell_list(ishell).l+1) * 2;
        }
    }
    else
    {
        int_tmp_i = 0;
        for(int ishell = 0; ishell < size_shell; ishell++)
        {
        int_tmp_j = 0;
        for(int jshell = 0; jshell < size_shell; jshell++)
        {
        int_tmp_k = 0;
        for(int kshell = 0; kshell < size_shell; kshell++)
        {
        int_tmp_l = 0;
        for(int lshell = 0; lshell < size_shell; lshell++)
        {
            int l_i = shell_list(ishell).l, l_j = shell_list(jshell).l, l_k = shell_list(kshell).l, l_l = shell_list(lshell).l, Lmax = min(l_i + l_j, l_k +l_l);
            if((l_i+l_j+l_k+l_l)%2) 
            {
                int_tmp_l += shell_list(lshell).coeff.rows() * (2*shell_list(lshell).l+1) * 2;
                continue;
            }
            int size_gtos_i = shell_list(ishell).coeff.rows(), size_gtos_j = shell_list(jshell).coeff.rows(), size_gtos_k = shell_list(kshell).coeff.rows(), size_gtos_l = shell_list(lshell).coeff.rows();
            
            radial_tilde.resize(Lmax+1);     
            
            int int_tmp2_i = 0;
            for(int twojj_i = abs(2*l_i-1); twojj_i <= 2*l_i+1; twojj_i = twojj_i + 2)
            {
            int int_tmp2_j = 0;
            for(int twojj_j = abs(2*l_j-1); twojj_j <= 2*l_j+1; twojj_j = twojj_j + 2)
            {
            int int_tmp2_k = 0;
            for(int twojj_k = abs(2*l_k-1); twojj_k <= 2*l_k+1; twojj_k = twojj_k + 2)
            {
            int int_tmp2_l = 0;
            for(int twojj_l = abs(2*l_l-1); twojj_l <= 2*l_l+1; twojj_l = twojj_l + 2)
            {
                int sym_ai = twojj_i - 2*l_i, sym_aj = twojj_j - 2*l_j, sym_ak = twojj_k - 2*l_k, sym_al = twojj_l - 2*l_l;
                double k_i = -(twojj_i+1.0)*sym_ai/2.0, k_j = -(twojj_j+1.0)*sym_aj/2.0, k_k = -(twojj_k+1.0)*sym_ak/2.0, k_l = -(twojj_l+1.0)*sym_al/2.0;
                VectorXd array_angular[twojj_i + 1][twojj_j + 1][twojj_k + 1][twojj_l + 1];

                for(int mi = 0; mi < twojj_i + 1; mi++)
                for(int mj = 0; mj < twojj_j + 1; mj++)
                for(int mk = 0; mk < twojj_k + 1; mk++)
                for(int ml = 0; ml < twojj_l + 1; ml++)
                {
                    array_angular[mi][mj][mk][ml].resize(Lmax+1);
                    array_angular[mi][mj][mk][ml] = VectorXd::Zero(Lmax+1);
                    for(int tmp = Lmax; tmp >= 0; tmp = tmp - 2)
                        array_angular[mi][mj][mk][ml](tmp) = int2e_get_angular(l_i, 2*mi-twojj_i, sym_ai, l_j, 2*mj-twojj_j, sym_aj, l_k, 2*mk-twojj_k, sym_ak, l_l, 2*ml-twojj_l, sym_al, tmp);
                }

                for(int ii = 0; ii < size_gtos_i; ii++)
                for(int jj = 0; jj < size_gtos_j; jj++)
                for(int kk = 0; kk < size_gtos_k; kk++)
                for(int ll = 0; ll < size_gtos_l; ll++)
                {
                    radial_tilde = VectorXd::Zero(Lmax+1);
                    double norm = shell_list(ishell).norm(ii) * shell_list(jshell).norm(jj) * shell_list(kshell).norm(kk) * shell_list(lshell).norm(ll);
                    /*
                        radial_tilde in uncontracted case is the radial tensor
                    */
                    for(int tmp = Lmax; tmp >= 0; tmp = tmp - 2)
                    {
                        if(integralTYPE == "LLLL")
                            radial_tilde(tmp) = int2e_get_radial_LLLL(l_i, k_i, shell_list(ishell).exp_a(ii), l_j, k_j, shell_list(jshell).exp_a(jj), l_k, k_k, shell_list(kshell).exp_a(kk), l_l, k_l, shell_list(lshell).exp_a(ll), tmp) / norm;
                        else if(integralTYPE == "SSLL")
                            radial_tilde(tmp) = int2e_get_radial_SSLL(l_i, k_i, shell_list(ishell).exp_a(ii), l_j, k_j, shell_list(jshell).exp_a(jj), l_k, k_k, shell_list(kshell).exp_a(kk), l_l, k_l, shell_list(lshell).exp_a(ll), tmp) / norm;
                        else if(integralTYPE == "LLSS")
                            radial_tilde(tmp) = int2e_get_radial_SSLL(l_k, k_k, shell_list(kshell).exp_a(kk), l_l, k_l, shell_list(lshell).exp_a(ll), l_i, k_i, shell_list(ishell).exp_a(ii), l_j, k_j, shell_list(jshell).exp_a(jj), tmp) / norm;
                        else if(integralTYPE == "SSSS")
                            radial_tilde(tmp) = int2e_get_radial_SSSS(l_i, k_i, shell_list(ishell).exp_a(ii), l_j, k_j, shell_list(jshell).exp_a(jj), l_k, k_k, shell_list(kshell).exp_a(kk), l_l, k_l, shell_list(lshell).exp_a(ll), tmp) / norm;
                        else
                        {
                            cout << "ERROR: integralTYPE of get_h2e must be one of these:\n";
                            cout << "       LLLL,  SSLL,  LLSS, SSSS\n";
                            exit(99);
                        }
                    }
                    
                    for(int mi = 0; mi < twojj_i + 1; mi++)
                    for(int mj = 0; mj < twojj_j + 1; mj++)
                    for(int mk = 0; mk < twojj_k + 1; mk++)
                    for(int ml = 0; ml < twojj_l + 1; ml++)
                    {
                        int ei = int_tmp_i + int_tmp2_i + mi + ii * (twojj_i+1), ej = int_tmp_j + int_tmp2_j + mj + jj * (twojj_j+1), ek = int_tmp_k + int_tmp2_k + mk + kk * (twojj_k+1), el = int_tmp_l + int_tmp2_l + ml + ll * (twojj_l+1);
                        int eij = ei*size_2e+ej, ekl = ek*size_2e+el;
                        int_2e(eij,ekl) = radial_tilde.transpose() * array_angular[mi][mj][mk][ml];
                    }
                }
                int_tmp2_l += shell_list(lshell).coeff.rows() * (twojj_l+1);
            }
                int_tmp2_k += shell_list(kshell).coeff.rows() * (twojj_k+1);
            }
                int_tmp2_j += shell_list(jshell).coeff.rows() * (twojj_j+1);
            }
                int_tmp2_i += shell_list(ishell).coeff.rows() * (twojj_i+1);
            }
            int_tmp_l += shell_list(lshell).coeff.rows() * (2*shell_list(lshell).l+1) * 2;
        }
            int_tmp_k += shell_list(kshell).coeff.rows() * (2*shell_list(kshell).l+1) * 2;
        }
            int_tmp_j += shell_list(jshell).coeff.rows() * (2*shell_list(jshell).l+1) * 2;
        }
            int_tmp_i += shell_list(ishell).coeff.rows() * (2*shell_list(ishell).l+1) * 2;
        }
    }


    return int_2e;
}


/* 
    evaluate radial part and angular part in 2e integrals 
*/
double GTO_SPINOR::int2e_get_radial_LLLL(const int& l1, const double& k1, const double& a1, const int& l2, const double& k2, const double& a2, const int& l3, const double& k3, const double& a3, const int& l4, const double& k4, const double& a4, const int& LL) const
{
    return GTO::int2e_get_radial(l1,a1,l2,a2,l3,a3,l4,a4,LL);
}
double GTO_SPINOR::int2e_get_radial_SSLL(const int& l1, const double& k1, const double& a1, const int& l2, const double& k2, const double& a2, const int& l3, const double& k3, const double& a3, const int& l4, const double& k4, const double& a4, const int& LL) const
{
    double lk1 = 1+l1+k1, lk2 = 1+l2+k2;
    double value = 4.0*a1*a2 * GTO::int2e_get_radial(l1+1,a1,l2+1,a2,l3,a3,l4,a4,LL);
    if(l1 != 0 && l2 != 0)
        value += lk1*lk2 * GTO::int2e_get_radial(l1-1,a1,l2-1,a2,l3,a3,l4,a4,LL)
                - (2.0*a1*lk2+2.0*a2*lk1) * GTO::int2e_get_radial(l1,a1,l2,a2,l3,a3,l4,a4,LL);
    else if(l1 != 0 || l2 != 0)
        value += - (2.0*a1*lk2+2.0*a2*lk1) * GTO::int2e_get_radial(l1,a1,l2,a2,l3,a3,l4,a4,LL);
    return value;
}
double GTO_SPINOR::int2e_get_radial_SSSS(const int& l1, const double& k1, const double& a1, const int& l2, const double& k2, const double& a2, const int& l3, const double& k3, const double& a3, const int& l4, const double& k4, const double& a4, const int& LL) const
{
    double lk1 = 1+l1+k1, lk2 = 1+l2+k2, lk3 = 1+l3+k3, lk4 = 1+l4+k4;
    double value = 4*a1*a2*4*a3*a4 * GTO::int2e_get_radial(l1+1,a1,l2+1,a2,l3+1,a3,l4+1,a4,LL);
    if(l1 != 0 && l2 != 0)
    {
        if(l3 != 0 && l4 != 0)
            value += lk1*lk2*lk3*lk4 * GTO::int2e_get_radial(l1-1,a1,l2-1,a2,l3-1,a3,l4-1,a4,LL)
                - (2*a1*lk2+2*a2*lk1)*lk3*lk4 * GTO::int2e_get_radial(l1,a1,l2,a2,l3-1,a3,l4-1,a4,LL)
                + 4*a1*a2*lk3*lk4 * GTO::int2e_get_radial(l1+1,a1,l2+1,a2,l3-1,a3,l4-1,a4,LL)
                - lk1*lk2*(2*a3*lk4+2*a4*lk3) * GTO::int2e_get_radial(l1-1,a1,l2-1,a2,l3,a3,l4,a4,LL)
                + (2*a1*lk2+2*a2*lk1)*(2*a3*lk4+2*a4*lk3) * GTO::int2e_get_radial(l1,a1,l2,a2,l3,a3,l4,a4,LL)
                - 4*a1*a2*(2*a3*lk4+2*a4*lk3) * GTO::int2e_get_radial(l1+1,a1,l2+1,a2,l3,a3,l4,a4,LL)
                + lk1*lk2*4*a3*a4 * GTO::int2e_get_radial(l1-1,a1,l2-1,a2,l3+1,a3,l4+1,a4,LL)
                - (2*a1*lk2+2*a2*lk1)*4*a3*a4 * GTO::int2e_get_radial(l1,a1,l2,a2,l3+1,a3,l4+1,a4,LL);
        else if(l3 != 0 || l4 != 0)
            value += - lk1*lk2*(2*a3*lk4+2*a4*lk3) * GTO::int2e_get_radial(l1-1,a1,l2-1,a2,l3,a3,l4,a4,LL)
                + (2*a1*lk2+2*a2*lk1)*(2*a3*lk4+2*a4*lk3) * GTO::int2e_get_radial(l1,a1,l2,a2,l3,a3,l4,a4,LL)
                - 4*a1*a2*(2*a3*lk4+2*a4*lk3) * GTO::int2e_get_radial(l1+1,a1,l2+1,a2,l3,a3,l4,a4,LL)
                + lk1*lk2*4*a3*a4 * GTO::int2e_get_radial(l1-1,a1,l2-1,a2,l3+1,a3,l4+1,a4,LL)
                - (2*a1*lk2+2*a2*lk1)*4*a3*a4 * GTO::int2e_get_radial(l1,a1,l2,a2,l3+1,a3,l4+1,a4,LL);
        else
            value += lk1*lk2*4*a3*a4 * GTO::int2e_get_radial(l1-1,a1,l2-1,a2,l3+1,a3,l4+1,a4,LL)
                - (2*a1*lk2+2*a2*lk1)*4*a3*a4 * GTO::int2e_get_radial(l1,a1,l2,a2,l3+1,a3,l4+1,a4,LL);
    }
    else if(l1 != 0 || l2 != 0)
    {
        if(l3 != 0 && l4 != 0)
            value += - (2*a1*lk2+2*a2*lk1)*lk3*lk4 * GTO::int2e_get_radial(l1,a1,l2,a2,l3-1,a3,l4-1,a4,LL)
                + 4*a1*a2*lk3*lk4 * GTO::int2e_get_radial(l1+1,a1,l2+1,a2,l3-1,a3,l4-1,a4,LL)
                + (2*a1*lk2+2*a2*lk1)*(2*a3*lk4+2*a4*lk3) * GTO::int2e_get_radial(l1,a1,l2,a2,l3,a3,l4,a4,LL)
                - 4*a1*a2*(2*a3*lk4+2*a4*lk3) * GTO::int2e_get_radial(l1+1,a1,l2+1,a2,l3,a3,l4,a4,LL)
                - (2*a1*lk2+2*a2*lk1)*4*a3*a4 * GTO::int2e_get_radial(l1,a1,l2,a2,l3+1,a3,l4+1,a4,LL);
        else if(l3 != 0 || l4 != 0)
            value += (2*a1*lk2+2*a2*lk1)*(2*a3*lk4+2*a4*lk3) * GTO::int2e_get_radial(l1,a1,l2,a2,l3,a3,l4,a4,LL)
                - 4*a1*a2*(2*a3*lk4+2*a4*lk3) * GTO::int2e_get_radial(l1+1,a1,l2+1,a2,l3,a3,l4,a4,LL)
                - (2*a1*lk2+2*a2*lk1)*4*a3*a4 * GTO::int2e_get_radial(l1,a1,l2,a2,l3+1,a3,l4+1,a4,LL);
        else
            value += - (2*a1*lk2+2*a2*lk1)*4*a3*a4 * GTO::int2e_get_radial(l1,a1,l2,a2,l3+1,a3,l4+1,a4,LL);
    }
    else
    {
        if(l3 != 0 && l4 != 0)
            value += 4*a1*a2*lk3*lk4 * GTO::int2e_get_radial(l1+1,a1,l2+1,a2,l3-1,a3,l4-1,a4,LL)
                - 4*a1*a2*(2*a3*lk4+2*a4*lk3) * GTO::int2e_get_radial(l1+1,a1,l2+1,a2,l3,a3,l4,a4,LL);
        else if(l3 != 0 || l4 != 0)
            value +=- 4*a1*a2*(2*a3*lk4+2*a4*lk3) * GTO::int2e_get_radial(l1+1,a1,l2+1,a2,l3,a3,l4,a4,LL);
        else
            value += 0.0;
    }
    
    
    return value;
}



double GTO_SPINOR::int2e_get_angular(const int& l1, const int& two_m1, const int& a1, const int& l2, const int& two_m2, const int& a2, const int& l3, const int& two_m3, const int& a3, const int& l4, const int& two_m4, const int& a4, const int& LL) const
{
    if((l1+l2+LL)%2 || (l3+l4+LL)%2) return 0.0;

    double angular = 0.0;
    for(int mm = -LL; mm <= LL; mm++)
    {
        angular += pow(-1, mm) 
            * (pow(-1,(two_m1-1)/2)*a1*a2*sqrt((l1+0.5+a1*two_m1/2.0)*(l2+0.5+a2*two_m2/2.0))*wigner_3j(l1,l2,LL,(1-two_m1)/2,(two_m2-1)/2,-mm)
            + pow(-1,(two_m1+1)/2)*sqrt((l1+0.5-a1*two_m1/2.0)*(l2+0.5-a2*two_m2/2.0))*wigner_3j(l1,l2,LL,(-1-two_m1)/2,(two_m2+1)/2,-mm)) 
            * (pow(-1,(two_m3-1)/2)*a3*a4*sqrt((l3+0.5+a3*two_m3/2.0)*(l4+0.5+a4*two_m4/2.0))*wigner_3j(l3,l4,LL,(1-two_m3)/2,(two_m4-1)/2,mm)
            + pow(-1,(two_m3+1)/2)*sqrt((l3+0.5-a3*two_m3/2.0)*(l4+0.5-a4*two_m4/2.0))*wigner_3j(l3,l4,LL,(-1-two_m3)/2,(two_m4+1)/2,mm));
    }

    return angular * wigner_3j_zeroM(l1, l2, LL) * wigner_3j_zeroM(l3, l4, LL);
}



/* 
    write overlap, h1e and h2e for scf 
*/
void GTO_SPINOR::writeIntegrals_spinor(const MatrixXd& h2e, const string& filename)
{
    int size = round(sqrt(h2e.cols()));
    ofstream ofs;
    ofs.open(filename);        
        for(int ii = 0; ii < size; ii++)
        for(int jj = 0; jj < size; jj++)
        for(int kk = 0; kk < size; kk++)
        for(int ll = 0; ll < size; ll++)
        {
            int ij = ii * size + jj, kl = kk * size + ll;
            if(abs(h2e(ij,kl)) > 1e-15)  ofs << setprecision(16) << h2e(ij,kl) << "\t" << ii+1 << "\t" << jj+1 << "\t" << kk+1 << "\t" << ll+1 << "\n";
        }
        
        ofs << 0.0 << "\t" << 0 << "\t" << 0 << "\t" << 0 << "\t" << 0 << "\n";
    ofs.close();
}