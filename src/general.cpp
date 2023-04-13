#include<Eigen/Dense>
#include<string>
#include<iostream>
#include<iomanip>
#include<fstream>
#include<cmath>
#include<complex>
#include<omp.h>
#include"general.h"
using namespace std;
using namespace Eigen;

/*
    Count and print CPU & wall time
*/
clock_t StartTimeCPU, EndTimeCPU;
std::chrono::_V2::system_clock::time_point StartTimeWall, EndTimeWall; 
void countTime(clock_t& timeCPU, std::chrono::_V2::system_clock::time_point& timeWall)
{
    timeCPU = clock();
    timeWall = chrono::high_resolution_clock::now();
}
void printTime(const string& processName)
{
    std::streamsize ss = std::cout.precision();
    cout << fixed << setprecision(2);
    cout << processName + " time (CPU/WALL): " << (EndTimeCPU - StartTimeCPU) / (double)CLOCKS_PER_SEC;
    cout << "/" << std::chrono::duration<double, std::milli>(EndTimeWall-StartTimeWall).count() / 1000.0 << " seconds." << endl;
    cout.unsetf(std::ios_base::floatfield);
    cout << setprecision(ss);
}


/*
    factorial and double_factorial
*/
double double_factorial(const int& n)
{
    switch (n)
    {
    case 0: return 1.0;
    case 1: return 1.0;
    case 2: return 2.0;
    case 3: return 3.0;
    case 4: return 8.0;
    case 5: return 15.0;
    case 6: return 48.0;
    case 7: return 105.0;
    case 8: return 384.0;
    case 9: return 945.0;
    case 10: return 3840.0;
    case 11: return 10395.0;
    case 12: return 46080.0;
    case 13: return 135135.0;
    case 14: return 645120.0;
    default:
        if(n < 0)
        {
            cout << "ERROR: double_factorial is called for a negative number!" << endl;
            cout << "n is " << n << endl;
            exit(99);
        }
        else return n * double_factorial(n - 2);
    }
}
double factorial(const int& n)
{
    switch (n)
    {
    case 0: return 1.0;
    case 1: return 1.0;
    case 2: return 2.0;
    case 3: return 6.0;
    case 4: return 24.0;
    case 5: return 120.0;
    case 6: return 720.0;
    case 7: return 5040.0;
    case 8: return 40320.0;
    case 9: return 362880.0;
    case 10: return 3628800.0;

    default:
        if(n < 0)
        {
            cout << "ERROR: factorial is called for a negative number!" << endl;
            cout << "n is " << n << endl;
            exit(99);
        }
        return n * factorial(n - 1);
    }
}


/* 
    transformation matrix for complex SH to solid SH
*/
complex<double> U_SH_trans(const int& mu, const int& mm)
{
    complex<double> result;
    if(abs(mu) != abs(mm)) result = 0.0;
    else if (mu == 0)
    {
        result = 1.0;
    }
    else if (mu > 0)
    {
        if(mu == mm) result = pow(-1.0, mu) / sqrt(2.0);
        else result = 1.0 / sqrt(2.0);
    }
    else
    {
        if(mu == mm) result = 1.0 / sqrt(2.0) * complex<double>(0.0,1.0);
        else result = -pow(-1.0, mu) / sqrt(2.0) * complex<double>(0.0,1.0);
    }
    
    return result;
}


double evaluateChange(const MatrixXd& M1, const MatrixXd& M2)
{
    int size = M1.rows();
    double tmp = 0.0;
    for(int ii = 0; ii < size; ii++)
    for(int jj = 0; jj < size; jj++)
    {
        if(abs(M1(ii,jj)-M2(ii,jj)) > tmp)
            tmp = abs(M1(ii,jj)-M2(ii,jj));
    }

    return tmp;
}

MatrixXd matrix_half_inverse(const MatrixXd& inputM)
{
    int size = inputM.rows();
    SelfAdjointEigenSolver<MatrixXd> solver(inputM);
    VectorXd eigenvalues = solver.eigenvalues();
    MatrixXd eigenvectors = solver.eigenvectors();
 
    for(int ii = 0; ii < size; ii++)
    {
        if(eigenvalues(ii) < 0)
        {
            cout << "ERROR: Matrix has negative eigenvalues!" << endl;
            exit(99);
        }
        else
        {
            eigenvalues(ii) = 1.0 / sqrt(eigenvalues(ii));
        }
    }

    MatrixXd tmp(size, size);
    for(int ii = 0; ii < size; ii++)
    for(int jj = 0; jj < size; jj++)
    {
        tmp(ii,jj) = 0.0;
        for(int kk = 0; kk < size; kk++)
            tmp(ii,jj) += eigenvectors(ii,kk) * eigenvalues(kk) * eigenvectors(jj,kk);
    }

    return tmp; 
}

MatrixXd matrix_half(const MatrixXd& inputM)
{
    int size = inputM.rows();
    SelfAdjointEigenSolver<MatrixXd> solver(inputM);
    VectorXd eigenvalues = solver.eigenvalues();
    MatrixXd eigenvectors = solver.eigenvectors();
    
    for(int ii = 0; ii < size; ii++)
    {
        if(eigenvalues(ii) < 0)
        {
            cout << "ERROR: Matrix has negative eigenvalues!" << endl;
            exit(99);
        }
        else
        {
            eigenvalues(ii) = sqrt(eigenvalues(ii));
        }
    }
    MatrixXd tmp(size, size);
    for(int ii = 0; ii < size; ii++)
    for(int jj = 0; jj < size; jj++)
    {
        tmp(ii,jj) = 0.0;
        for(int kk = 0; kk < size; kk++)
            tmp(ii,jj) += eigenvectors(ii,kk) * eigenvalues(kk) * eigenvectors(jj,kk);
    }

    return tmp;
}


void eigensolverG(const MatrixXd& inputM, const MatrixXd& s_h_i, VectorXd& values, MatrixXd& vectors)
{
    MatrixXd tmp = s_h_i * inputM * s_h_i;
    
    SelfAdjointEigenSolver<MatrixXd> solver(tmp);
    values = solver.eigenvalues();
    vectors = s_h_i * solver.eigenvectors();

    return;
}

string removeSpaces(const string& flags)
{
    string tmp_s = flags;
    for(int ii = 0; ii < tmp_s.size(); ii++)
    {
        if(tmp_s[ii] == ' ')
        {
            tmp_s.erase(tmp_s.begin()+ii);
            ii--;
        }
    }
    return tmp_s;
}

vector<string> stringSplit(const string& flags)
{
    string tmp_s = flags;
    vector<string> output;
    int pos = 0, pos2;
    while ((pos = tmp_s.find(' ')) != string::npos || (pos2 = tmp_s.find('\t')) != string::npos)
    {
        pos = tmp_s.find(' ');
        pos2 = tmp_s.find('\t');
        if(pos >= 0 && pos2 >= 0)
            pos = min(pos,pos2);
        else if(pos < 0 && pos2 >= 0)
            pos = pos2;
        if(pos != 0)
        {
            output.push_back(tmp_s.substr(0,pos));
            tmp_s.erase(0,pos);
        }
        else
            tmp_s.erase(0,1);
    }
    if(tmp_s.size()>=1)
    {
        output.push_back(tmp_s);
    }
    return output;
}
vector<string> stringSplit(const string& flags, const char delimiter)
{
    string tmp_s = flags;
    vector<string> output;
    int pos = 0;
    while ((pos = tmp_s.find(delimiter)) != string::npos)
    {
        // cout << pos << endl; exit(99);
        if(pos != 0)
        {
            output.push_back(tmp_s.substr(0,pos));
        }
        tmp_s.erase(0,1);
    }
    return output;
}


/*
    Functions used in X2C
*/
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
    
    overlap_h_i = matrix_half_inverse(overlap);
    eigensolverG(h_4C, overlap_h_i, ene_tmp, coeff_tmp);

    for(int ii = 0; ii < size; ii++)
    for(int jj = 0; jj < size; jj++)
    {
        coeff_large(ii,jj) = coeff_tmp(ii,size+jj);
        coeff_small(ii,jj) = coeff_tmp(ii+size,jj+size);
    }

    return coeff_small * coeff_large.inverse();
}

MatrixXd X2C::get_X(const MatrixXd& coeff)
{
    int size = coeff.cols()/2;
    MatrixXd coeff_large(size,size), coeff_small(size,size);
    for(int ii = 0; ii < size; ii++)
    for(int jj = 0; jj < size; jj++)
    {
        coeff_large(ii,jj) = coeff(ii,size+jj);
        coeff_small(ii,jj) = coeff(ii+size,jj+size);
    }

    return coeff_small * coeff_large.inverse();
}



MatrixXd X2C::get_R(const MatrixXd& S_, const MatrixXd& T_, const MatrixXd& X_)
{
    int size = S_.rows();
    MatrixXd S_tilde(size,size), S_h_i(size,size), S_h(size,size);
    S_tilde = S_ + 0.5/pow(speedOfLight,2) * X_.transpose() * T_ * X_;
    S_h_i = matrix_half_inverse(S_);
    S_h = S_h_i.inverse();
    MatrixXd tmp = matrix_half_inverse(S_h_i * S_tilde * S_h_i);
    return S_h_i * tmp * S_h;
}

MatrixXd X2C::get_R(const MatrixXd& S_4c, const MatrixXd& X_)
{
    int size = S_4c.cols()/2;
    MatrixXd S_tilde(size,size), S_h_i(size,size), S_h(size,size);
    S_tilde = S_4c.block(0,0,size,size) + X_.transpose() * S_4c.block(size,size,size,size) * X_;
    S_h_i = matrix_half_inverse(S_4c.block(0,0,size,size));
    S_h = S_h_i.inverse();
    MatrixXd tmp = matrix_half_inverse(S_h_i * S_tilde * S_h_i);
    return S_h_i * tmp * S_h;
}

MatrixXd X2C::evaluate_h1e_x2c(const MatrixXd& S_, const MatrixXd& T_, const MatrixXd& W_, const MatrixXd& V_)
{
    MatrixXd X = get_X(S_, T_, W_, V_);
    MatrixXd R = get_R(S_, T_, X);
    MatrixXd h_eff = V_ + T_ * X + X.transpose() * T_ - X.transpose() * T_ * X + 0.25/pow(speedOfLight,2) * X.transpose() * W_ * X;
    return R.transpose() *  h_eff * R;
}

MatrixXd X2C::evaluate_h1e_x2c(const MatrixXd& S_, const MatrixXd& T_, const MatrixXd& W_, const MatrixXd& V_, const MatrixXd& X_, const MatrixXd& R_)
{
    MatrixXd h_eff = V_ + T_ * X_ + X_.transpose() * T_ - X_.transpose() * T_ * X_ + 0.25/pow(speedOfLight,2) * X_.transpose() * W_ * X_;
    return R_.transpose() *  h_eff * R_;
}

MatrixXd X2C::transform_4c_2c(const MatrixXd& M_4c, const MatrixXd XXX, const MatrixXd& RRR)
{
    int size = M_4c.rows()/2;
    if(size*2 != M_4c.rows())
    {
        cout << "Incorrect input M_4c in transform_4c_2c with an odd size" << endl;
        exit(99);
    }
    MatrixXd tmp = M_4c.block(0,0,size,size) + M_4c.block(0,size,size,size) * XXX + XXX.adjoint() * M_4c.block(size,0,size,size) + XXX.adjoint() * M_4c.block(size,size,size,size) * XXX;
    return RRR.adjoint() * tmp * RRR;
}


/*
    Generate basis transformation matrix
    j-adapted spinor to complex spherical harmonics spinor
    complex spherical harmonics spinor to solid spherical harmonics spinor

    output order is aaa...bbb... for spin, and m_l = -l, -l+1, ..., +l for each l
*/
MatrixXd Rotate::jspinor2sph(const Matrix<irrep_jm, Dynamic, 1>& irrep_list)
{
    int size_spinor = 0, size_irrep = irrep_list.rows(), Lmax = irrep_list(size_irrep - 1).l;
    int Lsize[Lmax+1];
    for(int ir = 0; ir < size_irrep; ir++)
    {
        size_spinor += irrep_list(ir).size;
        Lsize[irrep_list(ir).l] = irrep_list(ir).size;
    }
    int size_nr = size_spinor/2, int_tmp = 0;
    Matrix<Matrix<VectorXi,-1,1>,-1,1>index_tmp(Lmax+1);
    for(int ll = 0; ll <= Lmax; ll++)
    {
        index_tmp(ll).resize(Lsize[ll]);
        for(int nn = 0; nn < Lsize[ll]; nn++)
        {
            index_tmp(ll)(nn).resize(2*ll+1);
            for(int mm = 0; mm < 2*ll+1; mm++)
            {
                index_tmp(ll)(nn)(mm) = int_tmp;
                int_tmp++;
            }
        }   
    }
    
    /*
        jspinor_i = \sum_j U_{ji} sph_j
        O^{jspinor} = U^\dagger O^{sph} U
    */
    MatrixXd output(size_spinor,size_spinor);
    output = MatrixXd::Zero(size_spinor,size_spinor);
    int_tmp = 0;
    for(int ll = 0; ll <= Lmax; ll++)
    for(int nn = 0; nn < Lsize[ll]; nn++)
    {
        if(ll != 0)
        {
            int twojj = 2*ll-1;
            for(int two_mj = -twojj; two_mj <= twojj; two_mj += 2)
            {
                output(index_tmp(ll)(nn)((two_mj-1)/2+ll),int_tmp) = -sqrt((ll+0.5-two_mj/2.0)/(2*ll+1.0));
                output(size_nr + index_tmp(ll)(nn)((two_mj+1)/2+ll),int_tmp) = sqrt((ll+0.5+two_mj/2.0)/(2*ll+1.0));
                int_tmp++;
            }
        }
        int twojj = 2*ll+1;
        for(int two_mj = -twojj; two_mj <= twojj; two_mj += 2)
        {
            if((two_mj-1)/2 >= -ll)
                output(index_tmp(ll)(nn)((two_mj-1)/2+ll),int_tmp) = sqrt((ll+0.5+two_mj/2.0)/(2*ll+1.0));
            if((two_mj+1)/2 <= ll)
                output(size_nr + index_tmp(ll)(nn)((two_mj+1)/2+ll),int_tmp) = sqrt((ll+0.5-two_mj/2.0)/(2*ll+1.0));
            int_tmp++;
        }
    }

    /* 
        return M = U^\dagger 
        O^{sph} = U O^{jspinor} U^\dagger = M^\dagger O^{jspinor} M
    */
    return output.adjoint();
}

MatrixXcd Rotate::sph2solid(const Matrix<irrep_jm, Dynamic, 1>& irrep_list)
{
    int size_spinor = 0, size_irrep = irrep_list.rows(), Lmax = irrep_list(size_irrep - 1).l;
    int Lsize[Lmax+1];
    for(int ir = 0; ir < size_irrep; ir++)
    {
        size_spinor += irrep_list(ir).size;
        Lsize[irrep_list(ir).l] = irrep_list(ir).size;
    }
    int size_nr = size_spinor/2;

    /*
        real_i = \sum_j U_{ji} complex_j
        O^{real} = U^\dagger O^{complex} U
    */
    MatrixXcd U_SH(2*Lmax+1,2*Lmax+1);
    for(int ii = 0; ii < 2*Lmax+1; ii++)
    for(int jj = 0; jj < 2*Lmax+1; jj++)
        U_SH(ii,jj) = U_SH_trans(jj - Lmax,ii - Lmax);

    int int_tmp = 0;
    MatrixXcd output(size_spinor,size_spinor);
    output = MatrixXcd::Zero(size_spinor,size_spinor);
    for(int ll = 0; ll <= Lmax; ll++)
    for(int ii = 0; ii < Lsize[ll]; ii++)
    {
        for(int mm = 0; mm < 2*ll+1; mm++)
        for(int nn = 0; nn < 2*ll+1; nn++)
        {
            output(int_tmp+mm,int_tmp+nn) = U_SH(mm-ll+Lmax,nn-ll+Lmax);
            output(size_nr+int_tmp+mm,size_nr+int_tmp+nn) = U_SH(mm-ll+Lmax,nn-ll+Lmax);
        }
        int_tmp += 2*ll+1;
    }

    return output;
}


/*
    For CFOUR interface
*/
MatrixXd Rotate::reorder_m_cfour(const int& LL)
{
    MatrixXd tmp = MatrixXd::Zero(2*LL+1,2*LL+1);
    switch (LL)
    {
    case 0:
        tmp(0,0) = 1.0;
        break;
    case 1:
        tmp(2,0) = 1.0;
        tmp(0,1) = 1.0;
        tmp(1,2) = 1.0;
        break;
    case 2:
        tmp(2,0) = 1.0;
        tmp(0,1) = 1.0;
        tmp(3,2) = 1.0;
        tmp(4,3) = 1.0;
        tmp(1,4) = 1.0;
        break;
    case 3:
        tmp(4,0) = 1.0;
        tmp(2,1) = 1.0;
        tmp(3,2) = 1.0;
        tmp(6,3) = 1.0;
        tmp(1,4) = 1.0;
        tmp(0,5) = 1.0;
        tmp(5,6) = 1.0;
        break;
    case 4:
        tmp(4,0) = 1.0;
        tmp(2,1) = 1.0;
        tmp(5,2) = 1.0;
        tmp(8,3) = 1.0;
        tmp(1,4) = 1.0;
        tmp(6,5) = 1.0;
        tmp(0,6) = 1.0;
        tmp(7,7) = 1.0;
        tmp(3,8) = 1.0;
        break;
    case 5:
        tmp(6,0) = 1.0;
        tmp(4,1) = 1.0;
        tmp(7,2) = 1.0;
        tmp(8,3) = 1.0;
        tmp(1,4) = 1.0;
        tmp(0,5) = 1.0;
        tmp(9,6) = 1.0;
        tmp(2,7) = 1.0;
        tmp(5,8) = 1.0;
        tmp(10,9) = 1.0;
        tmp(3,10) = 1.0;
        break;
    case 6:
        tmp(12,0) = 1.0;
        tmp(4,1) = 1.0;
        tmp(11,2) = 1.0;
        tmp(10,3) = 1.0;
        tmp(1,4) = 1.0;
        tmp(8,5) = 1.0;
        tmp(0,6) = 1.0;
        tmp(9,7) = 1.0;
        tmp(2,8) = 1.0;
        tmp(6,9) = 1.0;
        tmp(3,10) = 1.0;
        tmp(5,11) = 1.0;
        tmp(7,12) = 1.0;
        break;
    default:
        cout << "ERROR: L is too large to be supported!" << endl;
        exit(99);
        break;
    }

    return tmp;
}
MatrixXd Rotate::reorder_m_cfour_new(const int& LL)
{
    /*
        Transform -l, -l+1, ..., l-1, l 
        to        l, -l, l-1, -l+1, ..., 0
    */
    MatrixXd tmp = MatrixXd::Zero(2*LL+1,2*LL+1);
    for(int ii = 0 ; ii < 2*LL+1; ii++)
    {
        int index_tmp = ii%2? ii/2 : 2*LL-ii/2;
        tmp(index_tmp,ii) = 1.0;
    }

    return tmp;
}
MatrixXcd Rotate::jspinor2cfour_interface(const Matrix<irrep_jm, Dynamic, 1>& irrep_list, MatrixXd (*reorder_m)(const int&))
{
    int Lmax = irrep_list(irrep_list.rows()-1).l;
    vMatrixXd Lmatrices(Lmax+1);
    for(int ll = 0; ll <= Lmax; ll++)
        Lmatrices(ll) = reorder_m(ll);
    int size_spinor = 0, Lsize[Lmax+1];
    for(int ir = 0; ir < irrep_list.rows(); ir++)
    {
        Lsize[irrep_list(ir).l] = irrep_list(ir).size;
        size_spinor += irrep_list(ir).size;
    }
    int size_nr = size_spinor/2, int_tmp = 0;
    MatrixXd tmp = MatrixXd::Zero(size_spinor,size_spinor);
    for(int ll = 0; ll <= Lmax; ll++)
    for(int ii = 0; ii < Lsize[ll]; ii++)
    {
        for(int mm = 0; mm < 2*ll+1; mm++)
        for(int nn = 0; nn < 2*ll+1; nn++)
        {
            tmp(int_tmp + mm, int_tmp + nn) = Lmatrices(ll)(mm,nn);
            tmp(size_nr+int_tmp + mm, size_nr+int_tmp + nn) = Lmatrices(ll)(mm,nn);
        }
        int_tmp += 2*ll+1;
    }

    return jspinor2sph(irrep_list) * sph2solid(irrep_list) * tmp;
}
MatrixXcd Rotate::jspinor2cfour_interface_old(const Matrix<irrep_jm, Dynamic, 1>& irrep_list)
{
    return jspinor2cfour_interface(irrep_list, Rotate::reorder_m_cfour);
}
MatrixXcd Rotate::jspinor2cfour_interface_new(const Matrix<irrep_jm, Dynamic, 1>& irrep_list)
{
    return jspinor2cfour_interface(irrep_list, Rotate::reorder_m_cfour_new);
}


bool CG::triangle_fails(const int two_j1, const int two_j2, const int two_j3)
{
    return ( (( two_j1 + two_j2 + two_j3 ) % 2 != 0 ) || 
             ( two_j1 + two_j2 < two_j3 ) || 
             ( abs( two_j1 - two_j2 ) > two_j3 ) );
}
double CG::sqrt_delta(const int two_j1, const int two_j2, const int two_j3)
{
    return ( CG::sqrt_fact[ ( two_j1 + two_j2 - two_j3 ) / 2 ]
           * CG::sqrt_fact[ ( two_j1 - two_j2 + two_j3 ) / 2 ]
           * CG::sqrt_fact[ (-two_j1 + two_j2 + two_j3 ) / 2 ]
           / CG::sqrt_fact[ ( two_j1 + two_j2 + two_j3 ) / 2 + 1 ] );
}
/*
    Wigner 3j coefficients with l1,l2,l3 are integers
*/
double CG::wigner_3j_int(const int& l1, const int& l2, const int& l3, const int& m1, const int& m2, const int& m3)
{
    // return CG::wigner_3j(2*l1,2*l2,2*l3,2*m1,2*m2,2*m3);

    if(l3 > l1 + l2 || l3 < abs(l1 - l2) || m1 + m2 + m3 != 0 || abs(m1) > abs(l1) || abs(m2) > abs(l2) || abs(m3) > abs(l3))
    {
        return 0.0;
    }
    else if(m1 == 0 && m2 == 0 && m3 == 0)
    {
        return wigner_3j_zeroM(l1,l2,l3);
    }
    else
    {
        vector<int> L={l1,l2,l3}, M={m1,m2,m3};
        int tmp, Lmax = max(l1,max(l2,l3));
        for(int ii = 0; ii <= 1; ii++)
        {
            if(L[ii] == Lmax)
            {
                tmp = L[ii];
                L[ii] = L[2];
                L[2] = tmp;
                tmp = M[ii];
                M[ii] = M[2];
                M[2] = tmp;
                break;
            }
        }

        if(L[2] == L[0] + L[1])
        {
            return pow(-1, L[0] - L[1] - M[2]) * sqrt_fact[2*L[0]] * sqrt_fact[2*L[1]] / sqrt_fact[2*L[2] + 1] * sqrt_fact[L[2] - M[2]] * sqrt_fact[L[2] + M[2]] / sqrt_fact[L[0]+M[0]] / sqrt_fact[L[0]-M[0]] / sqrt_fact[L[1]+M[1]] / sqrt_fact[L[1]-M[1]];
        }
        else
        {
            return CG::wigner_3j(2*L[0],2*L[1],2*L[2],2*M[0],2*M[1],2*M[2]);
        }
        
    }
}
/*
    Wigner 3j coefficients with m1 = m2 = m3 = 0
*/
double CG::wigner_3j_zeroM(const int& l1, const int& l2, const int& l3)
{
    int J = l1+l2+l3, g = J/2;
    if(J%2 || l3 > l1 + l2 || l3 < abs(l1 - l2))
    {
        return 0.0;
    }
    else
    {
        return pow(-1,g) * sqrt_fact[J - 2*l1] * sqrt_fact[J - 2*l2] * sqrt_fact[J - 2*l3] / sqrt_fact[J + 1] 
                * factorial(g) / factorial(g-l1) / factorial(g-l2) / factorial(g-l3);
    }
}
/*
    General Wigner nj coefficients
*/
double CG::wigner_3j(const int& tj1, const int& tj2, const int& tj3, const int& tm1, const int& tm2, const int& tm3)
{
    /* Formula (C.21) in Albert Messiah - Quantum Mechanics */
    assert(tj1 >= 0 && tj2 >= 0 && tj3 >= 0);
    if ( triangle_fails(tj1, tj2, tj3) || (tm1 + tm2 + tm3 != 0) ||
         (tj1 + tm1) % 2 != 0 || (tj2 + tm2 ) % 2 != 0 || (tj3 + tm3) % 2 != 0 ||
         abs(tm1) > tj1 || abs(tm2) > tj2 || abs(tm3) > tj3)    return 0.0;

    int tmp_p1 = (tj3 - tj2 + tm1) / 2;
    int tmp_p2 = (tj3 - tj1 - tm2) / 2;
    int tmp_m1 = (tj1 + tj2 - tj3) / 2;
    int tmp_m2 = (tj1 - tm1) / 2;
    int tmp_m3 = (tj2 + tm2) / 2;
    int max_p = max(0,max(-tmp_p1,-tmp_p2));
    int min_m = min(tmp_m1,min(tmp_m2,tmp_m3));
    if(min_m < max_p) return 0.0;
    
    double result = 0.0, tmp_d;
    for(int tt = max_p; tt <= min_m; tt++)
    {
        tmp_d = CG::sqrt_fact[tt]*CG::sqrt_fact[tmp_p1+tt]*CG::sqrt_fact[tmp_p2+tt]*CG::sqrt_fact[tmp_m1-tt]*CG::sqrt_fact[tmp_m2-tt]*CG::sqrt_fact[tmp_m3-tt];
        tmp_d = pow(-1,tt)/tmp_d/tmp_d;
        // tmp_d = pow(-1,tt)/factorial(tt)/factorial(tmp_p1+tt)/factorial(tmp_p2+tt)/factorial(tmp_m1-tt)/factorial(tmp_m2-tt)/factorial(tmp_m3-tt);
        result += tmp_d;
    }

    return result * pow(-1,(tj1-tj2-tm3)/2) * CG::sqrt_delta(tj1,tj2,tj3) 
    * CG::sqrt_fact[(tj1+tm1)/2] * CG::sqrt_fact[(tj1-tm1)/2] * CG::sqrt_fact[(tj2+tm2)/2]
    * CG::sqrt_fact[(tj2-tm2)/2] * CG::sqrt_fact[(tj3+tm3)/2] * CG::sqrt_fact[(tj3-tm3)/2];
}
double CG::wigner_6j(const int& tj1, const int& tj2, const int& tj3, const int& tj4, const int& tj5, const int& tj6)
{
    /* Formula (C.36) in Albert Messiah - Quantum Mechanics */
    assert(tj1 >= 0 && tj2 >= 0 && tj3 >= 0 && tj4 >= 0 && tj5 >= 0 && tj6 >= 0);
    if( CG::triangle_fails(tj1, tj2, tj3) ||
        CG::triangle_fails(tj4, tj5, tj3) ||
        CG::triangle_fails(tj4, tj2, tj6) ||
        CG::triangle_fails(tj1, tj5, tj6) )  return 0.0; 
    
    int tmp_p1 = (- tj1 - tj2 - tj3) / 2;
    int tmp_p2 = (- tj1 - tj5 - tj6) / 2;
    int tmp_p3 = (- tj4 - tj2 - tj6) / 2;
    int tmp_p4 = (- tj4 - tj5 - tj3) / 2;
    int tmp_m1 = (tj1 + tj2 + tj4 + tj5) / 2;
    int tmp_m2 = (tj1 + tj3 + tj4 + tj6) / 2;
    int tmp_m3 = (tj2 + tj3 + tj5 + tj6) / 2;
    int max_p = max(-tmp_p1,max(-tmp_p2,max(-tmp_p3,-tmp_p4)));
    int min_m = min(tmp_m1,min(tmp_m2,tmp_m3));
    if(min_m < max_p) return 0.0;

    double result = 0.0, tmp_d;
    for(int tt = max_p; tt <= min_m; tt++)
    {
        tmp_d = CG::sqrt_fact[tmp_p1+tt]*CG::sqrt_fact[tmp_p2+tt]*CG::sqrt_fact[tmp_p3+tt]*CG::sqrt_fact[tmp_p4+tt]*CG::sqrt_fact[tmp_m1-tt]*CG::sqrt_fact[tmp_m2-tt]*CG::sqrt_fact[tmp_m3-tt];
        tmp_d = pow(-1,tt)*factorial(tt+1)/tmp_d/tmp_d;
        result += tmp_d;
    }

    return result * CG::sqrt_delta(tj1,tj2,tj3) * CG::sqrt_delta(tj1,tj5,tj6) 
                  * CG::sqrt_delta(tj4,tj2,tj6) * CG::sqrt_delta(tj4,tj5,tj3);
}
double CG::wigner_9j(const int& tj1, const int& tj2, const int& tj3, const int& tj4, const int& tj5, const int& tj6, const int& tj7, const int& tj8, const int& tj9)
{
    /* Formula (C.41) in Albert Messiah - Quantum Mechanics */
    assert(tj1 >= 0 && tj2 >= 0 && tj3 >= 0 && tj4 >= 0 && tj5 >= 0 && tj6 >= 0 && tj7 >= 0 && tj8 >= 0 && tj9 >= 0);
    if( CG::triangle_fails(tj1, tj2, tj3) ||
        CG::triangle_fails(tj4, tj5, tj6) ||
        CG::triangle_fails(tj7, tj8, tj9) ||
        CG::triangle_fails(tj1, tj4, tj7) ||
        CG::triangle_fails(tj2, tj5, tj8) ||
        CG::triangle_fails(tj3, tj6, tj9) )  return 0.0;

    int min_tg = max(abs(tj1-tj9),max(abs(tj2-tj6),abs(tj4-tj8)));
    int max_tg = min(tj1+tj9,min(tj2+tj6,tj4+tj8));
    int sign = pow(-1,min_tg);

    double result = 0.0, tmp_d;
    for(int tg = min_tg; tg <= max_tg; tg += 2)
    {
        tmp_d = CG::wigner_6j(tj1,tj2,tj3,tj6,tj9,tg)*CG::wigner_6j(tj4,tj5,tj6,tj2,tg,tj8)*CG::wigner_6j(tj7,tj8,tj9,tg,tj1,tj4);
        result += (tg+1) * tmp_d;
    }

    return result * sign;
}