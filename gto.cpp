#include<Eigen/Dense>
#include<string>
#include<iostream>
#include<iomanip>
#include<fstream>
#include<cmath>
#include<complex>
#include<omp.h>
#include<gsl/gsl_sf_coupling.h>
#include"gto.h"
using namespace std;
using namespace Eigen;

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
    Wigner 3j coefficients with J = l1 + l2 + l3 is even
*/
double wigner_3j(const int& l1, const int& l2, const int& l3, const int& m1, const int& m2, const int& m3)
{
    // return gsl_sf_coupling_3j(2*l1,2*l2,2*l3,2*m1,2*m2,2*m3);

    if(l3 > l1 + l2 || l3 < abs(l1 - l2) || m1 + m2 + m3 != 0 || abs(m1) > abs(l1) || abs(m2) > abs(l2) || abs(m3) > abs(l3))
    {
        return 0.0;
    }
    else
    {
        Vector3i L(l1,l2,l3), M(m1,m2,m3);
        int tmp, Lmax = L.maxCoeff();
        for(int ii = 0; ii <= 1; ii++)
        {
            if(L(ii) == Lmax)
            {
                tmp = L(ii);
                L(ii) = L(2);
                L(2) = tmp;
                tmp = M(ii);
                M(ii) = M(2);
                M(2) = tmp;
                break;
            }
        }

        if(L(2) == L(0) + L(1))
        {
            return pow(-1, L(0) - L(1) - M(2)) * sqrt(factorial(2*L(0)) * factorial(2*L(1)) / factorial(2*L(2) + 1) * factorial(L(2) - M(2)) * factorial(L(2) + M(2)) / factorial(L(0)+M(0)) / factorial(L(0)-M(0)) / factorial(L(1)+M(1)) / factorial(L(1)-M(1)));
        }
        else
        {
            return gsl_sf_coupling_3j(2*L(0),2*L(1),2*L(2),2*M(0),2*M(1),2*M(2));
        }
        
    }
}
/*
    Wigner 3j coefficients with m1 = m2 = m3 = 0
*/
double wigner_3j_zeroM(const int& l1, const int& l2, const int& l3)
{
    int J = l1+l2+l3, g = J/2;
    if(J%2 || l3 > l1 + l2 || l3 < abs(l1 - l2))
    {
        return 0.0;
    }
    else
    {
        return pow(-1,g) * sqrt(factorial(J - 2*l1) * factorial(J - 2*l2) * factorial(J - 2*l3) / factorial(J + 1)) 
                * factorial(g) / factorial(g-l1) / factorial(g-l2) / factorial(g-l3);
    }
}


/* 
    function used to evaluate spherical harmonics transition matrix 
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


/*
    construction function
*/
GTO::GTO(const string& atomName_, const string& basisSet_, const int& charge_, const int& spin_, const bool& uncontracted_):
atomName(atomName_), basisSet(basisSet_), charge(charge_), spin(spin_), uncontracted(uncontracted_)
{
    if(atomName == "H") atomNumber = 1;
    else if(atomName == "HE") atomNumber = 2;
    else if(atomName == "LI") atomNumber = 3;
    else if(atomName == "BE") atomNumber = 4;
    else if(atomName == "B") atomNumber = 5;
    else if(atomName == "C") atomNumber = 6;
    else if(atomName == "N") atomNumber = 7;
    else if(atomName == "O") atomNumber = 8;
    else if(atomName == "F") atomNumber = 9;
    else if(atomName == "NE") atomNumber = 10;
    else if(atomName == "NA") atomNumber = 11;
    else if(atomName == "MG") atomNumber = 12;
    else if(atomName == "AL") atomNumber = 13;
    else if(atomName == "SI") atomNumber = 14;
    else if(atomName == "P") atomNumber = 15;
    else if(atomName == "S") atomNumber = 16;
    else if(atomName == "CL") atomNumber = 17;
    else if(atomName == "AR") atomNumber = 18;
    else if(atomName == "K") atomNumber = 19;
    else if(atomName == "CA") atomNumber = 20;
    else if(atomName == "SC") atomNumber = 21;
    else if(atomName == "TI") atomNumber = 22;
    else if(atomName == "V") atomNumber = 23;
    else if(atomName == "CR") atomNumber = 24;
    else if(atomName == "MN") atomNumber = 25;
    else if(atomName == "FE") atomNumber = 26;
    else if(atomName == "CO") atomNumber = 27;
    else if(atomName == "NI") atomNumber = 28;
    else if(atomName == "CU") atomNumber = 29;
    else if(atomName == "ZN") atomNumber = 30;
    else if(atomName == "GA") atomNumber = 31;
    else if(atomName == "GE") atomNumber = 32;
    else if(atomName == "AS") atomNumber = 33;
    else if(atomName == "SE") atomNumber = 34;
    else if(atomName == "BR") atomNumber = 35;
    else if(atomName == "KR") atomNumber = 36;
    else if(atomName == "RB") atomNumber = 37;
    else if(atomName == "SR") atomNumber = 38;
    else if(atomName == "Y") atomNumber = 39;
    else if(atomName == "ZR") atomNumber = 40;
    else if(atomName == "NB") atomNumber = 41;
    else if(atomName == "MO") atomNumber = 42;
    else if(atomName == "TC") atomNumber = 43;
    else if(atomName == "RU") atomNumber = 44;
    else if(atomName == "RH") atomNumber = 45;
    else if(atomName == "PD") atomNumber = 46;
    else if(atomName == "AG") atomNumber = 47;
    else if(atomName == "CD") atomNumber = 48;
    else if(atomName == "IN") atomNumber = 49;
    else if(atomName == "SN") atomNumber = 50;
    else if(atomName == "SB") atomNumber = 51;
    else if(atomName == "TE") atomNumber = 52;
    else if(atomName == "I") atomNumber = 53;
    else if(atomName == "XE") atomNumber = 54;
    else if(atomName == "CS") atomNumber = 55;
    else if(atomName == "BA") atomNumber = 56;
    else if(atomName == "LA") atomNumber = 57;
    else if(atomName == "CE") atomNumber = 58;
    else if(atomName == "PR") atomNumber = 59;
    else if(atomName == "ND") atomNumber = 60;
    else if(atomName == "PM") atomNumber = 61;
    else if(atomName == "SM") atomNumber = 62;
    else if(atomName == "EU") atomNumber = 63;
    else if(atomName == "GD") atomNumber = 64;
    else if(atomName == "TB") atomNumber = 65;
    else if(atomName == "DY") atomNumber = 66;
    else if(atomName == "HO") atomNumber = 67;
    else if(atomName == "ER") atomNumber = 68;
    else if(atomName == "TM") atomNumber = 69;
    else if(atomName == "YB") atomNumber = 70;
    else if(atomName == "LU") atomNumber = 71;
    else if(atomName == "HF") atomNumber = 72;
    else if(atomName == "TA") atomNumber = 73;
    else if(atomName == "W") atomNumber = 74;
    else if(atomName == "RE") atomNumber = 75;
    else if(atomName == "OS") atomNumber = 76;
    else if(atomName == "IR") atomNumber = 77;
    else if(atomName == "PT") atomNumber = 78;
    else if(atomName == "AU") atomNumber = 79;
    else if(atomName == "HG") atomNumber = 80;
    else if(atomName == "TL") atomNumber = 81;
    else if(atomName == "PB") atomNumber = 82;
    else if(atomName == "BI") atomNumber = 83;
    else if(atomName == "PO") atomNumber = 84;
    else if(atomName == "AT") atomNumber = 85;
    else if(atomName == "RN") atomNumber = 86;
    else if(atomName == "FR") atomNumber = 87;
    else if(atomName == "RA") atomNumber = 88;
    else if(atomName == "AC") atomNumber = 89;
    else if(atomName == "TH") atomNumber = 90;
    else if(atomName == "PA") atomNumber = 91;
    else if(atomName == "U") atomNumber = 92;
    else if(atomName == "NP") atomNumber = 93;
    else if(atomName == "PU") atomNumber = 94;
    else if(atomName == "AM") atomNumber = 95;
    else if(atomName == "CM") atomNumber = 96;
    else if(atomName == "BK") atomNumber = 97;
    else if(atomName == "CF") atomNumber = 98;
    else if(atomName == "ES") atomNumber = 99;
    else if(atomName == "FM") atomNumber = 100;
    else if(atomName == "MD") atomNumber = 101;
    else if(atomName == "NO") atomNumber = 102;
    else if(atomName == "LR") atomNumber = 103; 
    else 
    {
        cout << "ERROR: Atom name is not supported." << endl;
        exit(99);
    }

    nelec = atomNumber - charge;
    if((nelec%2 && (spin == 1 || spin == 3)) || (nelec%2 - 1) && spin == 2)
    {
        cout << "ERROR: Spin state is NOT consistent with charge!" << endl;
        exit(99);
    }
    else
    {
        switch (spin)
        {
        case 1:
        case 2:
            nelec_b = nelec / 2;
            break;
        case 3:
            nelec_b = nelec / 2 - 1;
            break;
        default:
            cout << "ERROR: Spin state " << spin << " is NOT supported!" << endl;
            exit(99);
            break;
        }
        nelec_a = nelec - nelec_b;
    }
    

    readBasis();
    normalization();
}

GTO::~GTO()
{
}

/*
    Read basis file in CFOUR format
*/
void GTO::readBasis()
{
    string target = atomName + ":" + basisSet + " ", flags;

    ifstream ifs;
    int int_tmp;
    
    ifs.open("GENBAS");
        while (!ifs.eof())
        {
            getline(ifs,flags);
            flags.resize(target.size());
            if(flags == target) 
            {
                getline(ifs,flags);
                break;
            }
        }
        if(ifs.eof())
        {
            cout << "ERROR: can not find target basis (" + target + ") in the basis set file (GENBAS)\n";
            exit(99);
        }
        else
        {
            ifs >> size_shell;
            MatrixXi orbitalInfo(3,size_shell);
            shell_list.resize(size_shell);

            for(int ii = 0; ii < 3; ii++)
            for(int jj = 0; jj < size_shell; jj++)
            {
                ifs >> orbitalInfo(ii,jj);
            }
            size_gtoc = 0;
            size_gtou = 0;
            for(int ishell = 0; ishell < size_shell; ishell++)
            {
                size_gtou += (2 * orbitalInfo(0,ishell) + 1) * orbitalInfo(2,ishell);
                size_gtoc += (2 * orbitalInfo(0,ishell) + 1) * orbitalInfo(1,ishell);
                shell_list(ishell).l = orbitalInfo(0,ishell);
                shell_list(ishell).coeff.resize(orbitalInfo(2,ishell),orbitalInfo(1,ishell));
                shell_list(ishell).exp_a.resize(orbitalInfo(2,ishell));
                shell_list(ishell).norm.resize(orbitalInfo(2,ishell));
                for(int ii = 0; ii < orbitalInfo(2,ishell); ii++)   
                {    
                    ifs >> shell_list(ishell).exp_a(ii);
                    shell_list(ishell).norm(ii) = sqrt(auxiliary_1e(2*shell_list(ishell).l + 2, 2 * shell_list(ishell).exp_a(ii)));
                }
                for(int ii = 0; ii < orbitalInfo(2,ishell); ii++)
                for(int jj = 0; jj < orbitalInfo(1,ishell); jj++)
                {
                    ifs >> shell_list(ishell).coeff(ii,jj);
                    // shell_list(ishell).coeff(ii,jj) = shell_list(ishell).coeff(ii,jj) / sqrt(auxiliary_1e(2*shell_list(ishell).l + 2, 2 * shell_list(ishell).exp_a(ii)));
                }
            }

        }       
    ifs.close();
}


/*
    Normalization
*/
void GTO::normalization()
{
    for(int ishell = 0; ishell < size_shell; ishell++)
    {
        int size_gtos = shell_list(ishell).coeff.rows();
        MatrixXd norm_single_shell(size_gtos, size_gtos);
        for(int ii = 0; ii < size_gtos; ii++)
        for(int jj = 0; jj < size_gtos; jj++)
        {
            norm_single_shell(ii,jj) = auxiliary_1e(2+2*shell_list(ishell).l, shell_list(ishell).exp_a(ii)+shell_list(ishell).exp_a(jj)) / shell_list(ishell).norm(ii) / shell_list(ishell).norm(jj);
        }
        for(int subshell = 0; subshell < shell_list(ishell).coeff.cols(); subshell++)
        {
            double tmp = 0.0;
            for(int ii = 0; ii < size_gtos; ii++)
            for(int jj = 0; jj < size_gtos; jj++)
            {
                tmp += shell_list(ishell).coeff(ii,subshell) * shell_list(ishell).coeff(jj,subshell) * norm_single_shell(ii,jj);
            }
            for(int ii = 0; ii < size_gtos; ii++)
                shell_list(ishell).coeff(ii,subshell) = shell_list(ishell).coeff(ii,subshell) / sqrt(tmp);
        }
    }
    return;
}


/*
    Evaluate different one-electron integrals 
*/
MatrixXd GTO::get_h1e(const string& intType, const bool& uncontracted_) const
{
    MatrixXd int_1e;
    int int_tmp = 0;
    if(!uncontracted_)
    {
        int_1e.resize(size_gtoc, size_gtoc);
        for(int ii = 0; ii < size_gtoc; ii++)
        for(int jj = 0; jj < size_gtoc; jj++)
            int_1e(ii,jj) = 0.0;
    }
    else
    {
        int_1e.resize(size_gtou, size_gtou);
        for(int ii = 0; ii < size_gtou; ii++)
        for(int jj = 0; jj < size_gtou; jj++)
            int_1e(ii,jj) = 0.0;
    }
    
    for(int ishell = 0; ishell < size_shell; ishell++)
    {
        int ll = shell_list(ishell).l;
        int size_gtos = shell_list(ishell).coeff.rows();
        MatrixXd h1e_single_shell(size_gtos, size_gtos);
        for(int ii = 0; ii < size_gtos; ii++)
        for(int jj = 0; jj < size_gtos; jj++)
        {
            double a1 = shell_list(ishell).exp_a(ii), a2 = shell_list(ishell).exp_a(jj);
        
            if(intType == "overlap")  h1e_single_shell(ii,jj) = auxiliary_1e(2 + 2*ll, a1 + a2);
            else if(intType == "nuc_attra")  h1e_single_shell(ii,jj) = -atomNumber * auxiliary_1e(1 + 2*ll, a1 + a2);
            else if(intType == "kinetic")  h1e_single_shell(ii,jj) = a2 * (2*ll + 3) * auxiliary_1e(2 + 2*ll, a1 + a2) - 2 * a2 * a2 * auxiliary_1e(4 + 2*ll, a1 + a2);
            else if(intType == "h1e")  h1e_single_shell(ii,jj) = a2 * (2*ll + 3) * auxiliary_1e(2 + 2*ll, a1 + a2) - 2 * a2 * a2 * auxiliary_1e(4 + 2*ll, a1 + a2) + -atomNumber * auxiliary_1e(1 + 2*ll, a1 + a2);
            else if(intType == "p.Vp")  h1e_single_shell(ii,jj) = ((2*ll + 1) * ll * auxiliary_1e(2*ll-1, a1 + a2) - 2*ll*(a1 + a2)*auxiliary_1e(2*ll + 1, a1 + a2) + 4*a1*a2 * auxiliary_1e(2*ll + 3, a1 + a2)) * -atomNumber;
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
            Matrix<MatrixXd, -1, 1> int_1e_shell(size_shell);
            int_1e_shell(ishell).resize(size_subshell,size_subshell);
            for(int ii = 0; ii < size_subshell; ii++)
            for(int jj = 0; jj < size_subshell; jj++)
            {
                int_1e_shell(ishell)(ii,jj) = 0.0;
                for(int mm = 0; mm < size_gtos; mm++)
                for(int nn = 0; nn < size_gtos; nn++)
                {
                    int_1e_shell(ishell)(ii,jj) += shell_list(ishell).coeff(mm, ii) * shell_list(ishell).coeff(nn, jj) * h1e_single_shell(mm,nn);
                }
            }

            for(int ii = 0; ii < size_subshell; ii++)
            for(int jj = 0; jj < size_subshell; jj++)
            for(int kk = 0; kk < 2*ll+1; kk++)
            {
                int_1e(int_tmp + kk + ii * (2*ll+1), int_tmp + kk + jj * (2*ll+1)) = int_1e_shell(ishell)(ii,jj);
            }
            int_tmp += size_subshell * (2*ll+1);
        }
        else
        {
            for(int ii = 0; ii < size_gtos; ii++)
            for(int jj = 0; jj < size_gtos; jj++)
            for(int kk = 0; kk < 2*ll+1; kk++)
            {
                int_1e(int_tmp + kk + ii * (2*ll+1), int_tmp + kk + jj * (2*ll+1)) = h1e_single_shell(ii,jj);
            }
            int_tmp += size_gtos * (2*ll+1);
        }
        
    }
    
    return int_1e;
}



/*
    Evaluate different two-electron integrals 
*/
MatrixXd GTO::get_h2e(const bool& uncontracted_) const
{
    MatrixXd int_2e;
    if(!uncontracted_)
    {
        int_2e.resize(size_gtoc*(size_gtoc+1)/2, size_gtoc*(size_gtoc+1)/2);
        int_2e = MatrixXd::Zero(size_gtoc*(size_gtoc+1)/2, size_gtoc*(size_gtoc+1)/2);
    }
    else
    {
        int_2e.resize(size_gtou*(size_gtou+1)/2, size_gtou*(size_gtou+1)/2);
        int_2e = MatrixXd::Zero(size_gtou*(size_gtou+1)/2, size_gtou*(size_gtou+1)/2);
    }
    
    VectorXd radial_tilde;
    if(!uncontracted_)
    {
        int int_tmp_i = 0;
        for(int ishell = 0; ishell < size_shell; ishell++)
        {
        int int_tmp_j = 0;
        for(int jshell = 0; jshell <= ishell; jshell++)
        {
        int int_tmp_k = 0;
        for(int kshell = 0; kshell < size_shell; kshell++)
        {
        int int_tmp_l = 0; 
        for(int lshell = 0; lshell <= kshell; lshell++)
        {
            int l_i = shell_list(ishell).l, l_j = shell_list(jshell).l, l_k = shell_list(kshell).l, l_l = shell_list(lshell).l, Lmax = min(l_i + l_j, l_k +l_l);
            if((l_i+l_j+l_k+l_l)%2) 
            {
                int_tmp_l += shell_list(lshell).coeff.cols() * (2*shell_list(lshell).l+1);
                continue;
            }
            int size_gtos_i = shell_list(ishell).coeff.rows(), size_gtos_j = shell_list(jshell).coeff.rows(), size_gtos_k = shell_list(kshell).coeff.rows(), size_gtos_l = shell_list(lshell).coeff.rows();
            int size_subshell_i = shell_list(ishell).coeff.cols(), size_subshell_j = shell_list(jshell).coeff.cols(), size_subshell_k = shell_list(kshell).coeff.cols(), size_subshell_l = shell_list(lshell).coeff.cols();

            radial_tilde.resize(Lmax+1);
            VectorXd array_angular[2*l_i + 1][2*l_j + 1][2*l_k + 1][2*l_l + 1];
            for(int mi = 0; mi < 2*l_i + 1; mi++)
            for(int mj = 0; mj < 2*l_j + 1; mj++)
            for(int mk = 0; mk < 2*l_k + 1; mk++)
            for(int ml = 0; ml < 2*l_l + 1; ml++)
            {
                array_angular[mi][mj][mk][ml].resize(Lmax+1);
                array_angular[mi][mj][mk][ml] = VectorXd::Zero(Lmax+1);
                for(int tmp = Lmax; tmp >= 0; tmp = tmp - 2)
                    array_angular[mi][mj][mk][ml](tmp) = int2e_get_angular(l_i, mi - l_i, l_j, mj - l_j, l_k, mk - l_k, l_l, ml - l_l, tmp);
            }

            for(int ii = 0; ii < size_subshell_i; ii++)
            for(int jj = 0; jj < size_subshell_j; jj++)
            for(int kk = 0; kk < size_subshell_k; kk++)
            for(int ll = 0; ll < size_subshell_l; ll++)
            {
                for(int iii = 0; iii < Lmax+1; iii++)
                {
                    radial_tilde(iii) = 0.0;
                }

                for(int iii = 0; iii < size_gtos_i; iii++)
                for(int jjj = 0; jjj < size_gtos_j; jjj++)
                for(int kkk = 0; kkk < size_gtos_k; kkk++)
                for(int lll = 0; lll < size_gtos_l; lll++)
                {
                    double norm = shell_list(ishell).norm(iii) * shell_list(jshell).norm(jjj) * shell_list(kshell).norm(kkk) * shell_list(lshell).norm(lll);
                    for(int tmp = Lmax; tmp >= 0; tmp = tmp - 2)
                        radial_tilde(tmp) += shell_list(ishell).coeff(iii,ii) * shell_list(jshell).coeff(jjj,jj) * shell_list(kshell).coeff(kkk,kk) * shell_list(lshell).coeff(lll,ll) * int2e_get_radial(l_i, shell_list(ishell).exp_a(iii), l_j, shell_list(jshell).exp_a(jjj), l_k, shell_list(kshell).exp_a(kkk), l_l, shell_list(lshell).exp_a(lll), tmp) / norm;
                }

                for(int mi = 0; mi < 2*l_i + 1; mi++)
                for(int mj = 0; mj < 2*l_j + 1; mj++)
                for(int mk = 0; mk < 2*l_k + 1; mk++)
                for(int ml = 0; ml < 2*l_l + 1; ml++)
                {
                    int ei = int_tmp_i + mi + ii * (2*l_i + 1), ej = int_tmp_j + mj + jj * (2*l_j + 1), ek = int_tmp_k + mk + kk * (2*l_k + 1), el = int_tmp_l + ml + ll * (2*l_l + 1);
                    if(ei < ej || ek < el) continue;
                    int eij = ei*(ei+1)/2+ej, ekl = ek*(ek+1)/2+el;
                    int_2e(eij,ekl) = radial_tilde.transpose() * array_angular[mi][mj][mk][ml];
                }    
            }
            int_tmp_l += shell_list(lshell).coeff.cols() * (2*shell_list(lshell).l+1);
        }
            int_tmp_k += shell_list(kshell).coeff.cols() * (2*shell_list(kshell).l+1);
        }
            int_tmp_j += shell_list(jshell).coeff.cols() * (2*shell_list(jshell).l+1);
        }
            int_tmp_i += shell_list(ishell).coeff.cols() * (2*shell_list(ishell).l+1);
        }
    }
    else
    {
        int int_tmp_i = 0;
        for(int ishell = 0; ishell < size_shell; ishell++)
        {
        int int_tmp_j = 0;
        for(int jshell = 0; jshell <= ishell; jshell++)
        {
        int int_tmp_k = 0;
        for(int kshell = 0; kshell < size_shell; kshell++)
        {
        int int_tmp_l = 0; 
        for(int lshell = 0; lshell <= kshell; lshell++)
        {
            int l_i = shell_list(ishell).l, l_j = shell_list(jshell).l, l_k = shell_list(kshell).l, l_l = shell_list(lshell).l, Lmax = min(l_i + l_j, l_k +l_l);
            if((l_i+l_j+l_k+l_l)%2) 
            {
                int_tmp_l += shell_list(lshell).coeff.cols() * (2*shell_list(lshell).l+1);
                continue;
            }
            int size_gtos_i = shell_list(ishell).coeff.rows(), size_gtos_j = shell_list(jshell).coeff.rows(), size_gtos_k = shell_list(kshell).coeff.rows(), size_gtos_l = shell_list(lshell).coeff.rows();

            radial_tilde.resize(Lmax+1);
            VectorXd array_angular[2*l_i + 1][2*l_j + 1][2*l_k + 1][2*l_l + 1];
            for(int mi = 0; mi < 2*l_i + 1; mi++)
            for(int mj = 0; mj < 2*l_j + 1; mj++)
            for(int mk = 0; mk < 2*l_k + 1; mk++)
            for(int ml = 0; ml < 2*l_l + 1; ml++)
            {
                array_angular[mi][mj][mk][ml].resize(Lmax+1);
                array_angular[mi][mj][mk][ml] = VectorXd::Zero(Lmax+1);
                for(int tmp = Lmax; tmp >= 0; tmp = tmp - 2)
                    array_angular[mi][mj][mk][ml](tmp) = int2e_get_angular(l_i, mi - l_i, l_j, mj - l_j, l_k, mk - l_k, l_l, ml - l_l, tmp);
            }
            for(int ii = 0; ii < size_gtos_i; ii++)
            for(int jj = 0; jj < size_gtos_j; jj++)
            for(int kk = 0; kk < size_gtos_k; kk++)
            for(int ll = 0; ll < size_gtos_l; ll++)
            {
                double norm = shell_list(ishell).norm(ii) * shell_list(jshell).norm(jj) * shell_list(kshell).norm(kk) * shell_list(lshell).norm(ll);
                
                /*
                    radial_tilde in uncontracted case is the radial tensor
                */
                for(int iii = 0; iii < Lmax+1; iii++)
                {
                    radial_tilde(iii) = 0.0;
                }
                for(int tmp = Lmax; tmp >= 0; tmp = tmp - 2)
                        radial_tilde(tmp) = int2e_get_radial(l_i, shell_list(ishell).exp_a(ii), l_j, shell_list(jshell).exp_a(jj), l_k, shell_list(kshell).exp_a(kk), l_l, shell_list(lshell).exp_a(ll), tmp) / norm;

                for(int mi = 0; mi < 2*l_i + 1; mi++)
                for(int mj = 0; mj < 2*l_j + 1; mj++)
                for(int mk = 0; mk < 2*l_k + 1; mk++)
                for(int ml = 0; ml < 2*l_l + 1; ml++)
                {
                    int ei = int_tmp_i + mi + ii * (2*l_i + 1), ej = int_tmp_j + mj + jj * (2*l_j + 1), ek = int_tmp_k + mk + kk * (2*l_k + 1), el = int_tmp_l + ml + ll * (2*l_l + 1);
                    if(ei < ej || ek < el) continue;                  
                    int eij = ei*(ei+1)/2+ej, ekl = ek*(ek+1)/2+el;
                    int_2e(eij,ekl) = radial_tilde.transpose() * array_angular[mi][mj][mk][ml];
                }    
            }
            int_tmp_l += shell_list(lshell).coeff.rows() * (2*shell_list(lshell).l+1);
        }
            int_tmp_k += shell_list(kshell).coeff.rows() * (2*shell_list(kshell).l+1);
        }
            int_tmp_j += shell_list(jshell).coeff.rows() * (2*shell_list(jshell).l+1);
        }
            int_tmp_i += shell_list(ishell).coeff.rows() * (2*shell_list(ishell).l+1);
        }
    }
    
    return int_2e;
}



/*
    auxiliary_1e is to evaluate \int_0^inf x^l exp(-ax^2) dx
*/
inline double GTO::auxiliary_1e(const int& l, const double& a) const
{
    int n = l / 2;
    if(n*2 == l)    return double_factorial(2*n-1)/pow(a,n)/pow(2.0,n+1)*sqrt(M_PI/a);
    else    return factorial(n)/2.0/pow(a,n+1);
}

/*
    auxiliary_2e_0_r is to evaluate \int_0^inf \int_0^r2 r1^l1 r2^l2 exp(-a1 * r1^2) exp(-a2 * r2^2) dr1dr2
*/
inline double GTO::auxiliary_2e_0_r(const int& l1, const int& l2, const double& a1, const double& a2) const
{
    int n1 = l1 / 2;
    if(n1 * 2 == l1)
    {
        cout << "ERROR: When auxiliary_2e_0r is called, l1 must be set to an odd number!" << endl;
        exit(99);
    }
    else
    {
        double tmp = 0.5 / pow(a1, n1+1) * auxiliary_1e(l2, a2);
        for(int kk = 0; kk <= n1; kk++)
        {
            tmp -= 0.5 / factorial(kk) / pow(a1, n1 - kk + 1) * auxiliary_1e(l2 + 2*kk, a1 + a2);
        }
        return tmp * factorial(n1);
    }
    
}

/*
    auxiliary_2e_r_inf is to evaluate \int_0^inf \int_r2^inf r1^l1 r2^l2 exp(-a1 * r1^2) exp(-a2 * r2^2) dr1dr2
*/
inline double GTO::auxiliary_2e_r_inf(const int& l1, const int& l2, const double& a1, const double& a2) const
{
    int n1 = l1 / 2;
    if(n1 * 2 == l1)
    {
        cout << "ERROR: When auxiliary_2e_0r is called, l1 must be set to an odd number!" << endl;
        exit(99);
    }
    else
    {
        double tmp = 0.0;
        for(int kk = 0; kk <= n1; kk++)
        {
            tmp += 0.5 / factorial(kk) / pow(a1, n1 - kk + 1) * auxiliary_1e(l2 + 2*kk, a1 + a2);
        }
        return tmp * factorial(n1);
    }
    
}

/* 
    evaluate 1e and 2e integrals in single gto basis, not used in current version 
*/
double GTO::int1e_single_gto(const int& l1, const int& m1, const double& a1, const int& l2, const int& m2, const double& a2, const string& integralTYPE)
{
    if(l1 != l2 || m1 != m2) return 0.0;
    else if(integralTYPE == "overlap")  return auxiliary_1e(2 + 2*l1, a1 + a2);
    else if(integralTYPE == "nuc_attra")  return -atomNumber * auxiliary_1e(1 + 2*l1, a1 + a2);
    else if(integralTYPE == "kinetic")  return a2 * (2*l1 + 3) * auxiliary_1e(2 + 2*l1, a1 + a2) - 2 * a2 * a2 * auxiliary_1e(4 + 2*l1, a1 + a2);
    else if(integralTYPE == "h1e")  return a2 * (2*l1 + 3) * auxiliary_1e(2 + 2*l1, a1 + a2) - 2 * a2 * a2 * auxiliary_1e(4 + 2*l1, a1 + a2) + -atomNumber * auxiliary_1e(1 + 2*l1, a1 + a2);
    else if(integralTYPE == "p.Vp")  return ((2*l1 + 1) * l1 * auxiliary_1e(2*l1-1, a1 + a2) - 2*l1*(a1+a2)*auxiliary_1e(2*l1 + 1, a1 + a2) + 4*a1*a2 * auxiliary_1e(2*l1 + 3, a1 + a2)) * -atomNumber;
    else
    {
        cout << "ERROR: get_h1e is called for undefined type of integrals!" << endl;
        exit(99);
    }
}

double GTO::int2e_single_gto(const int& l1, const int& m1, const double& a1, const int& l2, const int& m2, const double& a2, const int& l3, const int& m3, const double& a3, const int& l4, const int& m4, const double& a4)
{
    if((l1+l2+l3+l4) % 2 || (m1+m2+m3+m4) % 2 || m1 * m2 * m3 * m4 < 0) return 0.0;
    else
    {
        double result = 0.0, radial, angular;
        
        for(int ll = min(l1 + l2, l3 + l4); ll >= 0; ll = ll - 2)
        {
            if((l1 + l2 + 2 + ll) % 2)
            {
                radial = auxiliary_2e_0_r(l1 + l2 + 2 + ll, l3 + l4 + 1 - ll, a1 + a2, a3 + a4)
                        + auxiliary_2e_0_r(l3 + l4 + 2 + ll, l1 + l2 + 1 - ll, a3 + a4, a1 + a2);
            }
            else
            {
                radial = auxiliary_2e_r_inf(l3 + l4 + 1 - ll, l1 + l2 + 2 + ll, a3 + a4, a1 + a2)
                        + auxiliary_2e_r_inf(l1 + l2 + 1 - ll, l3 + l4 + 2 + ll, a1 + a2, a3 + a4);
            }

            angular = 0.0;
            for(int mm = -ll; mm <= ll; mm++)
            {
                double tmp = 0.0;
                for(int m_i = -abs(m1); m_i <= abs(m1); m_i+=2*abs(m1))
                {
                    for(int m_j = -abs(m2); m_j <= abs(m2); m_j+=2*abs(m2))
                    {
                        for(int m_k = -abs(m3); m_k <= abs(m3); m_k+=2*abs(m3))
                        {
                            for(int m_l = -abs(m4); m_l <= abs(m4); m_l+=2*abs(m4))
                            {
                                if(m_i + m_j - mm != 0 || m_k + m_l + mm != 0)
                                {
                                    tmp += 0.0;
                                }
                                else
                                {
                                    tmp += real(U_SH_trans(m1, m_i) * U_SH_trans(m2, m_j) * U_SH_trans(m3, m_k) * U_SH_trans(m4, m_l))
                                            * wigner_3j(l1, l2, ll, m_i, m_j, -mm) * wigner_3j(l3, l4, ll, m_k, m_l, mm);
                                }
                                if(m_l == 0) break;
                            }
                            if(m_k == 0) break;
                        }
                        if(m_j == 0) break;
                    }
                    if(m_i == 0) break;
                }
                angular += tmp * pow(-1, mm) * sqrt((2.0 * l1 + 1.0)*(2.0 * l2 + 1.0)*(2.0 * l3 + 1.0)*(2.0 * l4 + 1.0))
                            * wigner_3j_zeroM(l1, l2, ll) * wigner_3j_zeroM(l3, l4, ll);
            }

            
            result += radial * angular;
        }
        return result;
    }    
}



/* 
    evaluate radial part and angular part in 2e integrals 
*/
double GTO::int2e_get_radial(const int& l1, const double& a1, const int& l2, const double& a2, const int& l3, const double& a3, const int& l4, const double& a4, const int& LL) const
{
    if((l1+l2+l3+l4) % 2) return 0.0;
    double radial = 0.0;
    if((l1 + l2 + 2 + LL) % 2)
    {
        radial = auxiliary_2e_0_r(l1 + l2 + 2 + LL, l3 + l4 + 1 - LL, a1 + a2, a3 + a4)
                + auxiliary_2e_0_r(l3 + l4 + 2 + LL, l1 + l2 + 1 - LL, a3 + a4, a1 + a2);
    }
    else
    {
        radial = auxiliary_2e_r_inf(l3 + l4 + 1 - LL, l1 + l2 + 2 + LL, a3 + a4, a1 + a2)
                + auxiliary_2e_r_inf(l1 + l2 + 1 - LL, l3 + l4 + 2 + LL, a1 + a2, a3 + a4);
    }

    return radial;
}

double GTO::int2e_get_angular(const int& l1, const int& m1, const int& l2, const int& m2, const int& l3, const int& m3, const int& l4, const int& m4, const int& LL) const
{
    if((l1+l2+LL)%2 || (l3+l4+LL)%2) return 0.0;

    double angular = 0.0;
    for(int mm = -LL; mm <= LL; mm++)
    {
        double tmp = 0.0;


        for(int m_i = -abs(m1); m_i <= abs(m1); m_i+=2*abs(m1))
        {
            for(int m_k = -abs(m3); m_k <= abs(m3); m_k+=2*abs(m3))
            {
                int m_j = mm - m_i, m_l = -mm - m_k;
                if(abs(m1) != abs(m_i) || abs(m2) != abs(m_j) || abs(m3) != abs(m_k) || abs(m4) != abs(m_l))
                {
                    tmp += 0.0;
                }
                else
                {
                    tmp += real(U_SH_trans(m1, m_i) * U_SH_trans(m2, m_j) * U_SH_trans(m3, m_k) * U_SH_trans(m4, m_l)) * wigner_3j(l1, l2, LL, m_i, m_j, -mm) * wigner_3j(l3, l4, LL, m_k, m_l, mm);
                }        
                if(m_k == 0) break;
            }
            if(m_i == 0) break;
        }

        angular += tmp * pow(-1, mm);
    }

    return angular * sqrt((2.0 * l1 + 1.0)*(2.0 * l2 + 1.0)*(2.0 * l3 + 1.0)*(2.0 * l4 + 1.0)) * wigner_3j_zeroM(l1, l2, LL) * wigner_3j_zeroM(l3, l4, LL);
}


/* 
    get contraction coefficients for uncontracted calculations 
*/
MatrixXd GTO::get_coeff_contraction() const
{
    MatrixXd coeff(size_gtou, size_gtoc);
    for(int ii = 0; ii < size_gtou; ii++)
    for(int jj = 0; jj < size_gtoc; jj++)
        coeff(ii,jj) = 0.0;

    int int_tmp1 = 0, int_tmp2 = 0;
    for(int ishell = 0; ishell < size_shell; ishell++)
    {
        int ll = shell_list(ishell).l;
        int size_contracted = shell_list(ishell).coeff.cols();
        
        for(int ii = 0; ii < shell_list(ishell).coeff.cols(); ii++)    
        {   
            for(int mm = 0; mm < 2*ll+1; mm++)
            {
                for(int jj = 0; jj < shell_list(ishell).coeff.rows(); jj++)
                {    
                    coeff(int_tmp2 + jj*(2*ll+1) + mm, int_tmp1) = shell_list(ishell).coeff(jj,ii);
                }
                int_tmp1 ++;
            }
        }   
        int_tmp2 += shell_list(ishell).coeff.rows() * (2*ll+1);
    }

    return coeff;
}




/* 
    write overlap, h1e and h2e for scf 
*/
void GTO::writeIntegrals(const MatrixXd& h2e, const string& filename)
{
    int size = round((-1.0+sqrt(1+8*h2e.rows()))/2.0);
    
    ofstream ofs;
    ofs.open(filename);        
        for(int ii = 0; ii < size; ii++)
        for(int jj = 0; jj <= ii; jj++)
        for(int kk = 0; kk < size; kk++)
        for(int ll = 0; ll <= kk; ll++)
        {
            int ij = ii * (ii + 1) / 2 + jj, kl = kk * (kk + 1) / 2 + ll;
            if(abs(h2e(ij,kl)) > 1e-15)  ofs << setprecision(16) << h2e(ij,kl) << "\t" << ii+1 << "\t" << jj+1 << "\t" << kk+1 << "\t" << ll+1 << "\n";
        }
        
        ofs << 0.0 << "\t" << 0 << "\t" << 0 << "\t" << 0 << "\t" << 0 << "\n";
    ofs.close();
}
