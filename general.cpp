#include<string>
#include<iostream>
#include<iomanip>
#include<fstream>
#include<cmath>
#include<complex>
#include<omp.h>
#include"mkl_itrf.h"
#include"general.h"
using namespace std;

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


double evaluateChange(const vector<double>& M1, const vector<double>& M2)
{
    double tmp = 0.0;
    for(int ii = 0; ii < M1.size(); ii++)
    {
        if(abs(M1[ii]-M2[ii]) > tmp)
            tmp = abs(M1[ii]-M2[ii]);
    }

    return tmp;
}

vector<double> matrix_half_inverse(const vector<double>& inputM, const int& N)
{
    vector<double> eigenvalues(N), eigenvectors(N*N);
    eigh_d(inputM,N,eigenvalues,eigenvectors);
 
    for(int ii = 0; ii < N; ii++)
    {
        if(eigenvalues[ii] < 0)
        {
            cout << "ERROR: Matrix has negative eigenvalues: " << eigenvalues[ii] << endl;
            exit(99);
        }
        else
        {
            eigenvalues[ii] = 1.0 / sqrt(eigenvalues[ii]);
        }
    }

    vector<double> tmp(N*N);
    for(int ii = 0; ii < N; ii++)
    for(int jj = 0; jj < N; jj++)
    {
        tmp[ii*N+jj] = 0.0;
        for(int kk = 0; kk < N; kk++)
            tmp[ii*N+jj] += eigenvectors[ii*N+kk] * eigenvalues[kk] * eigenvectors[jj*N+kk];
    }

    return tmp; 
}

vector<double> matrix_half(const vector<double>& inputM, const int& N)
{
    vector<double> eigenvalues(N), eigenvectors(N*N);
    eigh_d(inputM,N,eigenvalues,eigenvectors);
 
    for(int ii = 0; ii < N; ii++)
    {
        if(eigenvalues[ii] < 0)
        {
            cout << "ERROR: Matrix has negative eigenvalues!" << endl;
            exit(99);
        }
        else
        {
            eigenvalues[ii] = sqrt(eigenvalues[ii]);
        }
    }

    vector<double> tmp(N*N);
    for(int ii = 0; ii < N; ii++)
    for(int jj = 0; jj < N; jj++)
    {
        tmp[ii*N+jj] = 0.0;
        for(int kk = 0; kk < N; kk++)
            tmp[ii*N+jj] += eigenvectors[ii*N+kk] * eigenvalues[kk] * eigenvectors[jj*N+kk];
    }

    return tmp; 
}


void eigensolverG(const vector<double>& inputM, const vector<double>& s_h_i, vector<double>& values, vector<double>& vectors, const int& N)
{
    vector<double> tmp1,tmp;
    dgemm_itrf('n','n',N,N,N,1.0,s_h_i,inputM,0.0,tmp1);
    dgemm_itrf('n','n',N,N,N,1.0,tmp1,s_h_i,0.0,tmp);
    eigh_d(tmp,N,values,tmp1);
    dgemm_itrf('n','n',N,N,N,1.0,s_h_i,tmp1,0.0,vectors);

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
vector<double> X2C::get_X(const vector<double>& S_, const vector<double>& T_, const vector<double>& W_, const vector<double>& V_, const int& size)
{
    int size2 = size*2;
    vector<double> h_4C(size2*size2), overlap(size2*size2), overlap_h_i(size2*size2);
    vector<double> coeff_tmp(size2*size2), CL(size*size), CS(size*size);
    vector<double> ene_tmp(size2);
    
    for(int ii = 0; ii < size; ii++)
    for(int jj = 0; jj < size; jj++)
    {
        h_4C[ii*size2+jj] = V_[ii*size+jj];
        h_4C[ii*size2+size+jj] = T_[ii*size+jj];
        h_4C[(size+ii)*size2+jj] = T_[ii*size+jj];
        h_4C[(size+ii)*size2+size+jj] = W_[ii*size+jj]/4.0/pow(speedOfLight,2) - T_[ii*size+jj];
        overlap[ii*size2+jj] = S_[ii*size+jj];
        overlap[(size+ii)*size2+size+jj] = T_[ii*size+jj]/2.0/pow(speedOfLight,2); 
        overlap[(size+ii)*size2+jj] = 0.0;
        overlap[ii*size2+size+jj] = 0.0;
    }
    
    overlap_h_i = matrix_half_inverse(overlap, size2);
    eigensolverG(h_4C, overlap_h_i, ene_tmp, coeff_tmp, size2);

    return get_X(coeff_tmp, size);
}

vector<double> X2C::get_X(const vector<double>& coeff, const int& size)
{
    int size2 = size*2;
    vector<double> CL(size*size), CS(size*size);
    for(int ii = 0; ii < size; ii++)
    for(int jj = 0; jj < size; jj++)
    {
        CL[ii*size+jj] = coeff[ii*size2+size+jj];
        CS[ii*size+jj] = coeff[(ii+size)*size2+jj+size];
    }
    vector<double> tmp, CLinv = matInv_d(CL, size);
    dgemm_itrf('n','n',size,size,size,1.0,CS,CLinv,0.0,tmp);

    return tmp;
}

vector<double> X2C::get_R(const vector<double>& S_, const vector<double>& T_, const vector<double>& X_, const int& size)
{
    int size2 = size*2;
    vector<double> S4c(size2*size2);
    for(int ii = 0; ii < size; ii++)
    for(int jj = 0; jj < size; jj++)
    {
        S4c[ii*size2+jj] = S_[ii*size+jj];
        S4c[(size+ii)*size2+size+jj] = T_[ii*size+jj]/2.0/pow(speedOfLight,2); 
        S4c[(size+ii)*size2+jj] = 0.0;
        S4c[ii*size2+size+jj] = 0.0;
    }
    return get_R(S4c, X_, size);
}

vector<double> X2C::get_R(const vector<double>& S_4c, const vector<double>& X_, const int& size)
{
    int size2 = size*2;
    vector<double> S_tilde(size,size), S_h_i(size,size), S_h(size,size);
    vector<double> Sll = matBlock(S_4c,size2,0,0,size,size), Sss = matBlock(S_4c,size2,size,size,size,size);
    vector<double> tmp1, tmp2;
    S_tilde = Sll;
    dgemm_itrf('t','n',size,size,size,1.0,X_,Sss,0.0,tmp1);
    dgemm_itrf('n','n',size,size,size,1.0,tmp1,X_,1.0,S_tilde);
    S_h_i = matrix_half_inverse(Sll,size);
    dgemm_itrf('n','n',size,size,size,1.0,S_h_i,S_tilde,0.0,tmp1);
    dgemm_itrf('n','n',size,size,size,1.0,tmp1,S_h_i,0.0,tmp2);
    tmp1 = matrix_half_inverse(tmp2, size);
    S_h = matInv_d(S_h_i,size);
    dgemm_itrf('n','n',size,size,size,1.0,S_h_i,tmp1,0.0,tmp2);
    dgemm_itrf('n','n',size,size,size,1.0,tmp2,S_h,0.0,tmp1);
    return tmp1;
}

vector<double> X2C::evaluate_h1e_x2c(const vector<double>& S_, const vector<double>& T_, const vector<double>& W_, const vector<double>& V_, const int& size)
{
    vector<double> X = get_X(S_, T_, W_, V_, size);
    vector<double> R = get_R(S_, T_, X, size);
    return evaluate_h1e_x2c(S_, T_, W_, V_, X, R, size);
}

vector<double> X2C::evaluate_h1e_x2c(const vector<double>& S_, const vector<double>& T_, const vector<double>& W_, const vector<double>& V_, const vector<double>& X_, const vector<double>& R_, const int& size)
{
    vector<double> L_NESC = V_, tmp1, tmp2;
    dgemm_itrf('n','n',size,size,size,1.0,T_,X_,1.0,L_NESC);
    dgemm_itrf('t','n',size,size,size,1.0,X_,T_,1.0,L_NESC);
    tmp1 = 0.25/pow(speedOfLight,2) * W_ - T_;
    dgemm_itrf('t','n',size,size,size,1.0,X_,tmp1,0.0,tmp2);
    dgemm_itrf('n','n',size,size,size,1.0,tmp2,X_,1.0,L_NESC);
    dgemm_itrf('t','n',size,size,size,1.0,R_,L_NESC,0.0,tmp1);
    dgemm_itrf('n','n',size,size,size,1.0,tmp1,R_,0.0,tmp2);
    return tmp2;
}

vector<double> X2C::transform_4c_2c(const vector<double>& M_4c_, const vector<double> X_, const vector<double>& R_, const int& size)
{
    vector<double> L_NESC, tmp1, tmp2;
    vector<double> Dll = matBlock(M_4c_, size*2, 0, 0, size, size), Dls = matBlock(M_4c_, size*2, 0, size, size, size);
    vector<double> Dsl = matBlock(M_4c_, size*2, size, 0, size, size), Dss = matBlock(M_4c_, size*2, size, size, size, size);
    L_NESC = Dll;
    dgemm_itrf('n','n',size,size,size,1.0,Dls,X_,1.0,L_NESC);
    dgemm_itrf('t','n',size,size,size,1.0,X_,Dsl,1.0,L_NESC);
    dgemm_itrf('t','n',size,size,size,1.0,X_,Dss,0.0,tmp2);
    dgemm_itrf('n','n',size,size,size,1.0,tmp2,X_,1.0,L_NESC);
    dgemm_itrf('t','n',size,size,size,1.0,R_,L_NESC,0.0,tmp1);
    dgemm_itrf('n','n',size,size,size,1.0,tmp1,R_,0.0,tmp2);
    return tmp2;
}

vVectorXd X2C::pauliDecompose(const vectorcd& M, const int& size)
{
    vVectorXd M_pauli(4);
    for(int ii = 0; ii < 4; ii++)
        M_pauli[ii].resize(size/2*size/2);
    for(int ii = 0; ii < size/2; ii++)
    for(int jj = 0; jj < size/2; jj++)
    {
        M_pauli[0][ii*size/2 + jj] = M[ii*size+jj].real(); // scalar
        M_pauli[1][ii*size/2 + jj] = M[ii*size+jj+size/2].imag(); // x
        M_pauli[2][ii*size/2 + jj] = M[ii*size+jj+size/2].real(); // y
        M_pauli[3][ii*size/2 + jj] = M[ii*size+jj].imag(); // z
    }
    return M_pauli;
}

/*
    Generate basis transformation matrix
    j-adapted spinor to complex spherical harmonics spinor
    complex spherical harmonics spinor to solid spherical harmonics spinor

    output order is aaa...bbb... for spin, and m_l = -l, -l+1, ..., +l for each l
*/
vector<double> Rotate::jspinor2sph(const vector<irrep_jm>& irrep_list)
{
    int size_spinor = 0, size_irrep = irrep_list.size(), Lmax = irrep_list[size_irrep - 1].l;
    int Lsize[Lmax+1];
    for(int ir = 0; ir < size_irrep; ir++)
    {
        size_spinor += irrep_list[ir].size;
        Lsize[irrep_list[ir].l] = irrep_list[ir].size;
    }
    int size_nr = size_spinor/2, int_tmp = 0;
    vector<vector<vector<int>>> index_tmp(Lmax+1);
    for(int ll = 0; ll <= Lmax; ll++)
    {
        index_tmp[ll].resize(Lsize[ll]);
        for(int nn = 0; nn < Lsize[ll]; nn++)
        {
            index_tmp[ll][nn].resize(2*ll+1);
            for(int mm = 0; mm < 2*ll+1; mm++)
            {
                index_tmp[ll][nn][mm] = int_tmp;
                int_tmp++;
            }
        }   
    }
    
    /*
        jspinor_i = \sum_j U_{ji} sph_j
        O^{jspinor} = U^\dagger O^{sph} U
    */
    vector<double> tmp(size_spinor*size_spinor, 0.0), output(size_spinor*size_spinor);
    int_tmp = 0;
    for(int ll = 0; ll <= Lmax; ll++)
    for(int nn = 0; nn < Lsize[ll]; nn++)
    {
        if(ll != 0)
        {
            int twojj = 2*ll-1;
            for(int two_mj = -twojj; two_mj <= twojj; two_mj += 2)
            {
                tmp[(index_tmp[ll][nn][(two_mj-1)/2+ll])*size_spinor+int_tmp] = -sqrt((ll+0.5-two_mj/2.0)/(2*ll+1.0));
                tmp[(size_nr + index_tmp[ll][nn][(two_mj+1)/2+ll])*size_spinor+int_tmp] = sqrt((ll+0.5+two_mj/2.0)/(2*ll+1.0));
                int_tmp++;
            }
        }
        int twojj = 2*ll+1;
        for(int two_mj = -twojj; two_mj <= twojj; two_mj += 2)
        {
            if((two_mj-1)/2 >= -ll)
                tmp[(index_tmp[ll][nn][(two_mj-1)/2+ll])*size_spinor+int_tmp] = sqrt((ll+0.5+two_mj/2.0)/(2*ll+1.0));
            if((two_mj+1)/2 <= ll)
                tmp[(size_nr + index_tmp[ll][nn][(two_mj+1)/2+ll])*size_spinor+int_tmp] = sqrt((ll+0.5-two_mj/2.0)/(2*ll+1.0));
            int_tmp++;
        }
    }

    /* 
        return M = U^\dagger 
        O^{sph} = U O^{jspinor} U^\dagger = M^\dagger O^{jspinor} M
    */
    for(int ii = 0; ii < size_spinor; ii++)
    for(int jj = 0; jj < size_spinor; jj++)
        output[ii*size_spinor+jj] = tmp[jj*size_spinor+ii];
    return output;
}

vectorcd Rotate::sph2solid(const vector<irrep_jm>& irrep_list)
{
    int size_spinor = 0, size_irrep = irrep_list.size(), Lmax = irrep_list[size_irrep - 1].l;
    vector<int> Lsize(Lmax+1);
    for(int ir = 0; ir < size_irrep; ir++)
    {
        size_spinor += irrep_list[ir].size;
        Lsize[irrep_list[ir].l] = irrep_list[ir].size;
    }
    int size_nr = size_spinor/2;

    /*
        real_i = \sum_j U_{ji} complex_j
        O^{real} = U^\dagger O^{complex} U
    */
    vectorcd U_SH((2*Lmax+1)*(2*Lmax+1));
    for(int ii = 0; ii < 2*Lmax+1; ii++)
    for(int jj = 0; jj < 2*Lmax+1; jj++)
        U_SH[ii*(2*Lmax+1)+jj] = U_SH_trans(jj - Lmax,ii - Lmax);

    int int_tmp = 0;
    vectorcd output(size_spinor*size_spinor, zero_cp);
    for(int ll = 0; ll <= Lmax; ll++)
    for(int ii = 0; ii < Lsize[ll]; ii++)
    {
        for(int mm = 0; mm < 2*ll+1; mm++)
        for(int nn = 0; nn < 2*ll+1; nn++)
        {
            output[(int_tmp+mm)*size_spinor+int_tmp+nn] = U_SH[(mm-ll+Lmax)*(2*Lmax+1)+nn-ll+Lmax];
            output[(size_nr+int_tmp+mm)*size_spinor+size_nr+int_tmp+nn] = U_SH[(mm-ll+Lmax)*(2*Lmax+1)+nn-ll+Lmax];
        }
        int_tmp += 2*ll+1;
    }

    return output;
}


/*
    For CFOUR interface
*/
vector<double> Rotate::reorder_m_cfour(const int& LL)
{
    vector<double> tmp((2*LL+1)*(2*LL+1));
    switch (LL)
    {
    case 0:
        tmp[0*(2*LL+1)+0] = 1.0;
        break;
    case 1:
        tmp[2*(2*LL+1)+0] = 1.0;
        tmp[0*(2*LL+1)+1] = 1.0;
        tmp[1*(2*LL+1)+2] = 1.0;
        break;
    case 2:
        tmp[2*(2*LL+1)+0] = 1.0;
        tmp[0*(2*LL+1)+1] = 1.0;
        tmp[3*(2*LL+1)+2] = 1.0;
        tmp[4*(2*LL+1)+3] = 1.0;
        tmp[1*(2*LL+1)+4] = 1.0;
        break;
    case 3:
        tmp[4*(2*LL+1)+0] = 1.0;
        tmp[2*(2*LL+1)+1] = 1.0;
        tmp[3*(2*LL+1)+2] = 1.0;
        tmp[6*(2*LL+1)+3] = 1.0;
        tmp[1*(2*LL+1)+4] = 1.0;
        tmp[0*(2*LL+1)+5] = 1.0;
        tmp[5*(2*LL+1)+6] = 1.0;
        break;
    case 4:
        tmp[4*(2*LL+1)+0] = 1.0;
        tmp[2*(2*LL+1)+1] = 1.0;
        tmp[5*(2*LL+1)+2] = 1.0;
        tmp[8*(2*LL+1)+3] = 1.0;
        tmp[1*(2*LL+1)+4] = 1.0;
        tmp[6*(2*LL+1)+5] = 1.0;
        tmp[0*(2*LL+1)+6] = 1.0;
        tmp[7*(2*LL+1)+7] = 1.0;
        tmp[3*(2*LL+1)+8] = 1.0;
        break;
    case 5:
        tmp[6*(2*LL+1)+0] = 1.0;
        tmp[4*(2*LL+1)+1] = 1.0;
        tmp[7*(2*LL+1)+2] = 1.0;
        tmp[8*(2*LL+1)+3] = 1.0;
        tmp[1*(2*LL+1)+4] = 1.0;
        tmp[0*(2*LL+1)+5] = 1.0;
        tmp[9*(2*LL+1)+6] = 1.0;
        tmp[2*(2*LL+1)+7] = 1.0;
        tmp[5*(2*LL+1)+8] = 1.0;
        tmp[10*(2*LL+1)+9] = 1.0;
        tmp[3*(2*LL+1)+10] = 1.0;
        break;
    case 6:
        tmp[12*(2*LL+1)+0] = 1.0;
        tmp[4*(2*LL+1)+1] = 1.0;
        tmp[11*(2*LL+1)+2] = 1.0;
        tmp[10*(2*LL+1)+3] = 1.0;
        tmp[1*(2*LL+1)+4] = 1.0;
        tmp[8*(2*LL+1)+5] = 1.0;
        tmp[0*(2*LL+1)+6] = 1.0;
        tmp[9*(2*LL+1)+7] = 1.0;
        tmp[2*(2*LL+1)+8] = 1.0;
        tmp[6*(2*LL+1)+9] = 1.0;
        tmp[3*(2*LL+1)+10] = 1.0;
        tmp[5*(2*LL+1)+11] = 1.0;
        tmp[7*(2*LL+1)+12] = 1.0;
        break;
    default:
        cout << "ERROR: L is too large to be supported!" << endl;
        exit(99);
        break;
    }

    return tmp;
}
vector<double> Rotate::reorder_m_cfour_new(const int& LL)
{
    /*
        Transform -l, -l+1, ..., l-1, l 
        to        l, -l, l-1, -l+1, ..., 0
    */
    vector<double> tmp((2*LL+1)*(2*LL+1), 0.0);
    for(int ii = 0 ; ii < 2*LL+1; ii++)
    {
        int index_tmp = ii%2 ? ii/2 : 2*LL-ii/2;
        tmp[index_tmp*(2*LL+1)+ii] = 1.0;
    }

    return tmp;
}
vectorcd Rotate::jspinor2cfour_interface(const vector<irrep_jm>& irrep_list, vector<double> (*reorder_m)(const int&))
{
    int Lmax = irrep_list[irrep_list.size()-1].l;
    vVectorXd Lmatrices(Lmax+1);
    for(int ll = 0; ll <= Lmax; ll++)
        Lmatrices[ll] = reorder_m(ll);
    int size_spinor = 0, Lsize[Lmax+1];
    for(int ir = 0; ir < irrep_list.size(); ir++)
    {
        Lsize[irrep_list[ir].l] = irrep_list[ir].size;
        size_spinor += irrep_list[ir].size;
    }
    int size_nr = size_spinor/2, int_tmp = 0;
    vectorcd tmp(size_spinor*size_spinor, zero_cp), tmp1 = real2complex(jspinor2sph(irrep_list)), tmp2;
    for(int ll = 0; ll <= Lmax; ll++)
    for(int ii = 0; ii < Lsize[ll]; ii++)
    {
        for(int mm = 0; mm < 2*ll+1; mm++)
        for(int nn = 0; nn < 2*ll+1; nn++)
        {
            tmp[(int_tmp + mm) *size_spinor+ int_tmp + nn] = complex<double>(Lmatrices[ll][mm*(2*ll+1)+nn], 0.0);
            tmp[(size_nr+int_tmp + mm) *size_spinor+  size_nr+int_tmp + nn] = complex<double>(Lmatrices[ll][mm*(2*ll+1)+nn], 0.0);
        }
        int_tmp += 2*ll+1;
    }

    zgemm_itrf('n','n',size_spinor,size_spinor,size_spinor,one_cp,tmp1,sph2solid(irrep_list),zero_cp,tmp2);
    zgemm_itrf('n','n',size_spinor,size_spinor,size_spinor,one_cp,tmp2,tmp,zero_cp,tmp1);

    return tmp1;
}
vectorcd Rotate::jspinor2cfour_interface_old(const vector<irrep_jm>& irrep_list)
{
    return jspinor2cfour_interface(irrep_list, Rotate::reorder_m_cfour);
}
vectorcd Rotate::jspinor2cfour_interface_new(const vector<irrep_jm>& irrep_list)
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
