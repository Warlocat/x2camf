#include<Eigen/Dense>
#include<string>
#include<iostream>
#include<iomanip>
#include<fstream>
#include<cmath>
#include<complex>
#include<omp.h>
#include<gsl/gsl_sf_coupling.h>
#include"int_sph.h"
using namespace std;
using namespace Eigen;

INT_SPH::INT_SPH(const string& atomName_, const string& basisSet_):
atomName(atomName_), basisSet(basisSet_)
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
    
    readBasis();
    normalization();
    size_gtoc_spinor = 2 * size_gtoc;
    size_gtou_spinor = 2 * size_gtou;
    cout << endl << "Total number of uncontracted spinor: " << size_gtou_spinor << endl;
    cout << "Total number of irreducible representation: " << Nirrep << endl << endl;
}

INT_SPH::~INT_SPH()
{
}


/*
    Read basis file in CFOUR format
*/
void INT_SPH::readBasis()
{
    string target = basisSet, flags;
    
    ifstream ifs;
    int int_tmp;
    
    ifs.open("GENBAS");
    if(!ifs)
    {
        cout << "ERROR opening file GENBAS." << endl;
        exit(99);
    }
        while (!ifs.eof())
        {
            getline(ifs,flags);
            if(flags.size() == target.size() && flags == target)
            {
                getline(ifs,flags);
                break;
            }
            else if(flags.size() > target.size())
            {
                flags.resize(target.size() + 1);
                if(flags == target + " ") 
                {
                    getline(ifs,flags);
                    break;
                }
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

            Nirrep = 0;
            for(int ii = 0; ii < size_shell; ii++)
            {
                Nirrep += 2*(2*orbitalInfo(0,ii)+1);
            }
            irrep_list.resize(Nirrep);
            int tmp_i = 0;
            for(int ii = 0; ii < size_shell; ii++)
            {
                int two_jj = 2*orbitalInfo(0,ii)+1;
                if(orbitalInfo(0,ii) != 0)
                {
                    int two_jj = 2*orbitalInfo(0,ii)-1;
                    for(int two_mj = -two_jj; two_mj <= two_jj; two_mj += 2)
                    {
                        irrep_list(tmp_i).l = orbitalInfo(0,ii);
                        irrep_list(tmp_i).size = orbitalInfo(2,ii);
                        irrep_list(tmp_i).two_j = two_jj;
                        irrep_list(tmp_i).two_mj = two_mj;
                        tmp_i++;
                    }
                }
                for(int two_mj = -two_jj; two_mj <= two_jj; two_mj += 2)
                {
                    irrep_list(tmp_i).l = orbitalInfo(0,ii);
                    irrep_list(tmp_i).size = orbitalInfo(2,ii);
                    irrep_list(tmp_i).two_j = two_jj;
                    irrep_list(tmp_i).two_mj = two_mj;
                    tmp_i++;
                }             
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
void INT_SPH::normalization()
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
    auxiliary_1e is to evaluate \int_0^inf x^l exp(-ax^2) dx
*/
inline double INT_SPH::auxiliary_1e(const int& l, const double& a) const
{
    int n = l / 2;
    if(l == 0)  return 0.5*sqrt(M_PI/a);
    else if(n*2 == l)    return double_factorial(2*n-1)/pow(a,n)/pow(2.0,n+1)*sqrt(M_PI/a);
    else    return factorial(n)/2.0/pow(a,n+1);
}

/*
    auxiliary_2e_0_r is to evaluate \int_0^inf \int_0^r2 r1^l1 r2^l2 exp(-a1 * r1^2) exp(-a2 * r2^2) dr1dr2
*/
inline double INT_SPH::auxiliary_2e_0_r(const int& l1, const int& l2, const double& a1, const double& a2) const
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
inline double INT_SPH::auxiliary_2e_r_inf(const int& l1, const int& l2, const double& a1, const double& a2) const
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
    evaluate radial part and angular part in 2e integrals 
*/
double INT_SPH::int2e_get_radial(const int& l1, const double& a1, const int& l2, const double& a2, const int& l3, const double& a3, const int& l4, const double& a4, const int& LL) const
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
double INT_SPH::int2e_get_angular(const int& l1, const int& two_m1, const int& s1, const int& l2, const int& two_m2, const int& s2, const int& l3, const int& two_m3, const int& s3, const int& l4, const int& two_m4, const int& s4, const int& LL) const
{
    if((l1+l2+LL)%2 || (l3+l4+LL)%2) return 0.0;

    double angular = 0.0;
    for(int mm = -LL; mm <= LL; mm++)
    {
        if(two_m2 - two_m1 - 2*mm != 0 || two_m4 - two_m3 + 2*mm != 0) continue;
        else
        {
            angular += pow(-1, mm) 
            * (pow(-1,(two_m1-1)/2)*s1*s2*sqrt((l1+0.5+s1*two_m1/2.0)*(l2+0.5+s2*two_m2/2.0))*wigner_3j(l1,l2,LL,(1-two_m1)/2,(two_m2-1)/2,-mm)
            + pow(-1,(two_m1+1)/2)*sqrt((l1+0.5-s1*two_m1/2.0)*(l2+0.5-s2*two_m2/2.0))*wigner_3j(l1,l2,LL,(-1-two_m1)/2,(two_m2+1)/2,-mm)) 
            * (pow(-1,(two_m3-1)/2)*s3*s4*sqrt((l3+0.5+s3*two_m3/2.0)*(l4+0.5+s4*two_m4/2.0))*wigner_3j(l3,l4,LL,(1-two_m3)/2,(two_m4-1)/2,mm)
            + pow(-1,(two_m3+1)/2)*sqrt((l3+0.5-s3*two_m3/2.0)*(l4+0.5-s4*two_m4/2.0))*wigner_3j(l3,l4,LL,(-1-two_m3)/2,(two_m4+1)/2,mm));
        }
    }

    return angular * wigner_3j_zeroM(l1, l2, LL) * wigner_3j_zeroM(l3, l4, LL);
}
double INT_SPH::int2e_get_angular_J(const int& l1, const int& two_m1, const int& s1, const int& l2, const int& two_m2, const int& s2, const int& LL) const
{
    return int2e_get_angular(l1,two_m1,s1,l1,two_m1,s1,l2,two_m2,s2,l2,two_m2,s2,LL);
}
double INT_SPH::int2e_get_angular_K(const int& l1, const int& two_m1, const int& s1, const int& l2, const int& two_m2, const int& s2, const int& LL) const
{
    return int2e_get_angular(l1,two_m1,s1,l2,two_m2,s2,l2,two_m2,s2,l1,two_m1,s1,LL);
}


/*
    Evaluate different one-electron integral in 2-spinor basis
*/
vMatrixXd INT_SPH::get_h1e(const string& intType) const
{
    vMatrixXd int_1e(Nirrep);
    int int_tmp = 0;
    for(int irrep = 0; irrep < Nirrep; irrep++)
    {
        int_1e(irrep).resize(irrep_list(irrep).size, irrep_list(irrep).size);
        int_1e(irrep) = MatrixXd::Zero(irrep_list(irrep).size,irrep_list(irrep).size);
    }
    for(int ishell = 0; ishell < size_shell; ishell++)
    {
        int ll = shell_list(ishell).l;
        int size_gtos = shell_list(ishell).coeff.rows();
        vMatrixXd h1e_single_shell;
        if(ll == 0) h1e_single_shell.resize(1);
        else    h1e_single_shell.resize(2);
        for(int ii = 0; ii < h1e_single_shell.rows(); ii++)
            h1e_single_shell(ii).resize(size_gtos, size_gtos);
        
        for(int ii = 0; ii < size_gtos; ii++)
        for(int jj = 0; jj < size_gtos; jj++)
        {
            double a1 = shell_list(ishell).exp_a(ii), a2 = shell_list(ishell).exp_a(jj);
            VectorXd auxiliary_1e_list(6);
            for(int mm = 0; mm <= 4; mm++)
                auxiliary_1e_list(mm) = auxiliary_1e(2*ll + mm, a1 + a2);
            if(ll != 0)
            {
                auxiliary_1e_list(5) = auxiliary_1e(2*ll - 1, a1 + a2);
            }
            else
            {
                auxiliary_1e_list(5) = 0.0;
            }
            for(int twojj = abs(2*ll-1); twojj <= 2*ll+1; twojj = twojj + 2)
            {
                double kappa = (twojj + 1.0) * (ll - twojj/2.0);
                int index_tmp = 1 - (2*ll+1 - twojj)/2;
                if(ll == 0) index_tmp = 0;
                
                if(intType == "s_p_nuc_s_p")
                {
                    h1e_single_shell(index_tmp)(ii,jj) = 4*a1*a2 * auxiliary_1e_list(3);
                    if(ll!=0)
                        h1e_single_shell(index_tmp)(ii,jj) += pow(ll + kappa + 1.0, 2) * auxiliary_1e_list(5) - 2.0*(ll + kappa + 1.0)*(a1 + a2)*auxiliary_1e_list(1);
                    h1e_single_shell(index_tmp)(ii,jj) *= -atomNumber;
                }
                else if(intType == "s_p_nuc_s_p_sf")
                {
                    h1e_single_shell(index_tmp)(ii,jj) = 4*a1*a2 * auxiliary_1e_list(3);
                    if(ll!=0)
                        h1e_single_shell(index_tmp)(ii,jj) += (2*ll*ll + ll) * auxiliary_1e_list(5) - 2.0*ll*(a1 + a2)*auxiliary_1e_list(1);
                    h1e_single_shell(index_tmp)(ii,jj) *= -atomNumber;
                }
                else if(intType == "s_p_nuc_s_p_sd")
                {
                    h1e_single_shell(index_tmp)(ii,jj) = 0.0;
                    if(ll!=0)
                        h1e_single_shell(index_tmp)(ii,jj) += (kappa + 1.0) * auxiliary_1e_list(5);
                    h1e_single_shell(index_tmp)(ii,jj) *= -atomNumber;
                }
                else if(intType == "s_p_s_p" )
                {
                    h1e_single_shell(index_tmp)(ii,jj) = 4*a1*a2 * auxiliary_1e_list(4);
                    if(ll!=0)
                        h1e_single_shell(index_tmp)(ii,jj) += pow(ll + kappa + 1.0, 2) * auxiliary_1e_list(0) - 2.0*(ll + kappa + 1.0)*(a1 + a2)*auxiliary_1e_list(2);
                }
                else if(intType == "overlap")  h1e_single_shell(index_tmp)(ii,jj) = auxiliary_1e_list(2);
                else if(intType == "nuc_attra")  h1e_single_shell(index_tmp)(ii,jj) = -atomNumber * auxiliary_1e_list(1);
                else if(intType == "kinetic")
                {
                    h1e_single_shell(index_tmp)(ii,jj) = 4*a1*a2 * auxiliary_1e_list(4);
                    if(ll!=0)
                        h1e_single_shell(index_tmp)(ii,jj) += pow(ll + kappa + 1.0, 2) * auxiliary_1e_list(0) - 2.0*(ll + kappa + 1.0)*(a1 + a2)*auxiliary_1e_list(2);
                    h1e_single_shell(index_tmp)(ii,jj) /= 2.0;
                }
                else
                {
                    cout << "ERROR: get_h1e is called for undefined type of integrals!" << endl;
                    exit(99);
                }
                h1e_single_shell(index_tmp)(ii,jj) = h1e_single_shell(index_tmp)(ii,jj) / shell_list(ishell).norm(ii) / shell_list(ishell).norm(jj);
            }
        }
        for(int ii = 0; ii < irrep_list(int_tmp).two_j + 1; ii++)
            int_1e(int_tmp + ii) = h1e_single_shell(0);
        int_tmp += irrep_list(int_tmp).two_j + 1;
        if(ll != 0)
        {
            for(int ii = 0; ii < irrep_list(int_tmp).two_j + 1; ii++)
                int_1e(int_tmp + ii) = h1e_single_shell(1);
            int_tmp += irrep_list(int_tmp).two_j + 1;
        }
    }

    return int_1e;
}

/*
    Evaluate different two-electron Coulomb and Exchange integral in 2-spinor basis
*/
int2eJK INT_SPH::get_h2e_JK(const string& intType, const int& occMaxL) const
{
    double time_a = 0.0, time_r = 0.0, time_c = 0.0;
    int occMaxShell = 0;
    if(occMaxL == -1)    occMaxShell = size_shell;
    else
    {
        for(int ii = 0; ii < size_shell; ii++)
        {
            if(shell_list(ii).l <= occMaxL)
                occMaxShell++;
            else
                break;
        }
    }
    
    int2eJK int_2e_JK;
    int_2e_JK.J.resize(Nirrep, Nirrep);
    int_2e_JK.K.resize(Nirrep, Nirrep);
    
    int int_tmp1_p = 0;
    for(int pshell = 0; pshell < occMaxShell; pshell++)
    {
    int l_p = shell_list(pshell).l, int_tmp1_q = 0;
    for(int qshell = 0; qshell < occMaxShell; qshell++)
    {
        int l_q = shell_list(qshell).l, l_max = max(l_p,l_q), LmaxJ = min(l_p+l_p, l_q+l_q), LmaxK = l_p+l_q;
        int size_gtos_p = shell_list(pshell).coeff.rows(), size_gtos_q = shell_list(qshell).coeff.rows();
        double array_radial_J[LmaxJ+1][size_gtos_p*size_gtos_p][size_gtos_q*size_gtos_q];
        double array_radial_K[LmaxK+1][size_gtos_p*size_gtos_q][size_gtos_q*size_gtos_p];
        int size_tmp_p = (l_p == 0) ? 1 : 2, size_tmp_q = (l_q == 0) ? 1 : 2;
        
        MatrixXd array_angular_J[LmaxJ+1][size_tmp_p][size_tmp_q], array_angular_K[LmaxK+1][size_tmp_p][size_tmp_q];

        StartTime = clock();
        #pragma omp parallel  for
        for(int twojj_p = abs(2*l_p-1); twojj_p <= 2*l_p+1; twojj_p = twojj_p + 2)
        for(int twojj_q = abs(2*l_q-1); twojj_q <= 2*l_q+1; twojj_q = twojj_q + 2)
        {
            int sym_ap = twojj_p - 2*l_p, sym_aq = twojj_q - 2*l_q;
            int index_tmp_p = (l_p > 0) ? 1 - (2*l_p+1 - twojj_p)/2 : 0;
            int index_tmp_q = (l_q > 0) ? 1 - (2*l_q+1 - twojj_q)/2 : 0;

            for(int tmp = LmaxJ; tmp >= 0; tmp -= 2)
            {
                array_angular_J[tmp][index_tmp_p][index_tmp_q].resize(twojj_p + 1,twojj_q + 1);
                for(int mp = 0; mp < twojj_p + 1; mp++)
                for(int mq = 0; mq < twojj_q + 1; mq++)
                {
                    array_angular_J[tmp][index_tmp_p][index_tmp_q](mp,mq) = int2e_get_angular_J(l_p, 2*mp-twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, tmp);
                }
            }
            for(int tmp = LmaxK; tmp >= 0; tmp -= 2)
            {
                array_angular_K[tmp][index_tmp_p][index_tmp_q].resize(twojj_p + 1,twojj_q + 1);
                for(int mp = 0; mp < twojj_p + 1; mp++)
                for(int mq = 0; mq < twojj_q + 1; mq++)
                {
                    array_angular_K[tmp][index_tmp_p][index_tmp_q](mp,mq) = int2e_get_angular_K(l_p, 2*mp-twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, tmp);
                }
            }
        }
        EndTime = clock();
        time_a += (EndTime - StartTime)/(double)CLOCKS_PER_SEC;


        StartTime = clock();
        #pragma omp parallel  for
        for(int tt = 0; tt < size_gtos_p*size_gtos_p*size_gtos_q*size_gtos_q; tt++)
        {
            int e1J = tt/(size_gtos_q*size_gtos_q);
            int e2J = tt - e1J*(size_gtos_q*size_gtos_q);
            int ii = e1J/size_gtos_p, jj = e1J - ii*size_gtos_p;
            int kk = e2J/size_gtos_q, ll = e2J - kk*size_gtos_q;
            int e1K = ii*size_gtos_q+ll, e2K = kk*size_gtos_p+jj;
            MatrixXd radial_2e_list_J[LmaxJ+1], radial_2e_list_K[LmaxK+1];
            double a_i_J = shell_list(pshell).exp_a(ii), a_j_J = shell_list(pshell).exp_a(jj), a_k_J = shell_list(qshell).exp_a(kk), a_l_J = shell_list(qshell).exp_a(ll);
            double a_i_K = shell_list(pshell).exp_a(ii), a_j_K = shell_list(qshell).exp_a(ll), a_k_K = shell_list(qshell).exp_a(kk), a_l_K = shell_list(pshell).exp_a(jj);
        
            if(intType.substr(0,4) == "LLLL")
            {
                for(int LL = LmaxJ; LL >= 0; LL -= 2)
                {
                    radial_2e_list_J[LL].resize(1,1);
                    radial_2e_list_J[LL](0,0) = int2e_get_radial(l_p,a_i_J,l_p,a_j_J,l_q,a_k_J,l_q,a_l_J,LL);
                }
                for(int LL = LmaxK; LL >= 0; LL -= 2)
                {
                    radial_2e_list_K[LL].resize(1,1);
                    radial_2e_list_K[LL](0,0) = int2e_get_radial(l_p,a_i_K,l_q,a_j_K,l_q,a_k_K,l_p,a_l_K,LL);
                }
            }
            else if(intType.substr(0,4) == "SSLL")
            {
                for(int LL = LmaxJ; LL >= 0; LL -= 2)
                {
                    radial_2e_list_J[LL].resize(3,1);
                    radial_2e_list_J[LL](0,0) = int2e_get_radial(l_p,a_i_J,l_p,a_j_J,l_q,a_k_J,l_q,a_l_J,LL);
                    radial_2e_list_J[LL](1,0) = int2e_get_radial(l_p+1,a_i_J,l_p+1,a_j_J,l_q,a_k_J,l_q,a_l_J,LL);
                    if(l_p != 0)
                        radial_2e_list_J[LL](2,0) = int2e_get_radial(l_p-1,a_i_J,l_p-1,a_j_J,l_q,a_k_J,l_q,a_l_J,LL);
                }
                for(int LL = LmaxK; LL >= 0; LL -= 2)
                {
                    radial_2e_list_K[LL].resize(3,1);
                    radial_2e_list_K[LL](0,0) = int2e_get_radial(l_p,a_i_K,l_q,a_j_K,l_q,a_k_K,l_p,a_l_K,LL);
                    radial_2e_list_K[LL](1,0) = int2e_get_radial(l_p+1,a_i_K,l_q+1,a_j_K,l_q,a_k_K,l_p,a_l_K,LL);
                    if(l_p != 0 && l_q != 0)
                        radial_2e_list_K[LL](2,0) = int2e_get_radial(l_p-1,a_i_K,l_q-1,a_j_K,l_q,a_k_K,l_p,a_l_K,LL);
                }
            }
            else if(intType.substr(0,4) == "SSSS")
            {
                for(int LL = LmaxJ; LL >= 0; LL -= 2)
                {
                    radial_2e_list_J[LL].resize(3,3);
                    radial_2e_list_J[LL](0,0) = int2e_get_radial(l_p,a_i_J,l_p,a_j_J,l_q,a_k_J,l_q,a_l_J,LL);
                    radial_2e_list_J[LL](1,0) = int2e_get_radial(l_p+1,a_i_J,l_p+1,a_j_J,l_q,a_k_J,l_q,a_l_J,LL);
                    radial_2e_list_J[LL](0,1) = int2e_get_radial(l_p,a_i_J,l_p,a_j_J,l_q+1,a_k_J,l_q+1,a_l_J,LL);
                    radial_2e_list_J[LL](1,1) = int2e_get_radial(l_p+1,a_i_J,l_p+1,a_j_J,l_q+1,a_k_J,l_q+1,a_l_J,LL);
                    if(l_p != 0)
                    {
                        radial_2e_list_J[LL](2,0) = int2e_get_radial(l_p-1,a_i_J,l_p-1,a_j_J,l_q,a_k_J,l_q,a_l_J,LL);
                        radial_2e_list_J[LL](2,1) = int2e_get_radial(l_p-1,a_i_J,l_p-1,a_j_J,l_q+1,a_k_J,l_q+1,a_l_J,LL);
                    }
                    if(l_q != 0)
                    {
                        radial_2e_list_J[LL](0,2) = int2e_get_radial(l_p,a_i_J,l_p,a_j_J,l_q-1,a_k_J,l_q-1,a_l_J,LL);
                        radial_2e_list_J[LL](1,2) = int2e_get_radial(l_p+1,a_i_J,l_p+1,a_j_J,l_q-1,a_k_J,l_q-1,a_l_J,LL);
                    }
                    if(l_p!=0 && l_q!=0)
                        radial_2e_list_J[LL](2,2) = int2e_get_radial(l_p-1,a_i_J,l_p-1,a_j_J,l_q-1,a_k_J,l_q-1,a_l_J,LL);
                }
                for(int LL = LmaxK; LL >= 0; LL -= 2)
                {
                    radial_2e_list_K[LL].resize(3,3);
                    radial_2e_list_K[LL](0,0) = int2e_get_radial(l_p,a_i_K,l_q,a_j_K,l_q,a_k_K,l_p,a_l_K,LL);
                    radial_2e_list_K[LL](1,0) = int2e_get_radial(l_p+1,a_i_K,l_q+1,a_j_K,l_q,a_k_K,l_p,a_l_K,LL);
                    radial_2e_list_K[LL](0,1) = int2e_get_radial(l_p,a_i_K,l_q,a_j_K,l_q+1,a_k_K,l_p+1,a_l_K,LL);
                    radial_2e_list_K[LL](1,1) = int2e_get_radial(l_p+1,a_i_K,l_q+1,a_j_K,l_q+1,a_k_K,l_p+1,a_l_K,LL);
                    if(l_p != 0 && l_q != 0)
                    {
                        radial_2e_list_K[LL](2,0) = int2e_get_radial(l_p-1,a_i_K,l_q-1,a_j_K,l_q,a_k_K,l_p,a_l_K,LL);
                        radial_2e_list_K[LL](2,1) = int2e_get_radial(l_p-1,a_i_K,l_q-1,a_j_K,l_q+1,a_k_K,l_p+1,a_l_K,LL);
                        radial_2e_list_K[LL](0,2) = int2e_get_radial(l_p,a_i_K,l_q,a_j_K,l_q-1,a_k_K,l_p-1,a_l_K,LL);
                        radial_2e_list_K[LL](1,2) = int2e_get_radial(l_p+1,a_i_K,l_q+1,a_j_K,l_q-1,a_k_K,l_p-1,a_l_K,LL);
                        radial_2e_list_K[LL](2,2) = int2e_get_radial(l_p-1,a_i_K,l_q-1,a_j_K,l_q-1,a_k_K,l_p-1,a_l_K,LL);
                    }
                }
            }

            for(int twojj_p = abs(2*l_p-1); twojj_p <= 2*l_p+1; twojj_p = twojj_p + 2)
            for(int twojj_q = abs(2*l_q-1); twojj_q <= 2*l_q+1; twojj_q = twojj_q + 2)
            {
                int sym_ap = twojj_p - 2*l_p, sym_aq = twojj_q - 2*l_q;
                double k_p = -(twojj_p+1.0)*sym_ap/2.0, k_q = -(twojj_q+1.0)*sym_aq/2.0;
                double norm_J = shell_list(pshell).norm(ii) * shell_list(pshell).norm(jj) * shell_list(qshell).norm(kk) * shell_list(qshell).norm(ll), norm_K = shell_list(pshell).norm(ii) * shell_list(qshell).norm(ll) * shell_list(qshell).norm(kk) * shell_list(pshell).norm(jj);
                double lk1 = 1+l_p+k_p, lk2 = 1+l_p+k_p, lk3 = 1+l_q+k_q, lk4 = 1+l_q+k_q, a1 = shell_list(pshell).exp_a(ii), a2 = shell_list(pshell).exp_a(jj), a3 = shell_list(qshell).exp_a(kk), a4 = shell_list(qshell).exp_a(ll);

                for(int tmp = LmaxJ; tmp >= 0; tmp -= 2)
                {
                    if(intType == "LLLL")
                    {
                        array_radial_J[tmp][e1J][e2J] = radial_2e_list_J[tmp](0,0) / norm_J;
                    }
                    else if(intType == "SSLL")
                    {
                        array_radial_J[tmp][e1J][e2J] = 4.0*a1*a2 * radial_2e_list_J[tmp](1,0);
                        if(l_p != 0)
                            array_radial_J[tmp][e1J][e2J] += lk1*lk2 * radial_2e_list_J[tmp](2,0) - (2.0*a1*lk2+2.0*a2*lk1) * radial_2e_list_J[tmp](0,0);
                        array_radial_J[tmp][e1J][e2J] /= norm_J * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "SSSS")
                    {
                        array_radial_J[tmp][e1J][e2J] = 4*a1*a2*4*a3*a4 * radial_2e_list_J[tmp](1,1);
                        if(l_p != 0)
                        {
                            if(l_q != 0)
                                array_radial_J[tmp][e1J][e2J] += lk1*lk2*lk3*lk4 * radial_2e_list_J[tmp](2,2) - (2*a1*lk2+2*a2*lk1)*lk3*lk4 * radial_2e_list_J[tmp](0,2) + 4*a1*a2*lk3*lk4 * radial_2e_list_J[tmp](1,2) - lk1*lk2*(2*a3*lk4+2*a4*lk3) * radial_2e_list_J[tmp](2,0) + (2*a1*lk2+2*a2*lk1)*(2*a3*lk4+2*a4*lk3) * radial_2e_list_J[tmp](0,0) - 4*a1*a2*(2*a3*lk4+2*a4*lk3) * radial_2e_list_J[tmp](1,0) + lk1*lk2*4*a3*a4 * radial_2e_list_J[tmp](2,1) - (2*a1*lk2+2*a2*lk1)*4*a3*a4 * radial_2e_list_J[tmp](0,1);
                            else
                                array_radial_J[tmp][e1J][e2J] += lk1*lk2*4*a3*a4 * radial_2e_list_J[tmp](2,1) - (2*a1*lk2+2*a2*lk1)*4*a3*a4 * radial_2e_list_J[tmp](0,1);
                        }
                        else
                        {
                            if(l_q != 0)
                                array_radial_J[tmp][e1J][e2J] += 4*a1*a2*lk3*lk4 * radial_2e_list_J[tmp](1,2) - 4*a1*a2*(2*a3*lk4+2*a4*lk3) * radial_2e_list_J[tmp](1,0);
                        }
                        array_radial_J[tmp][e1J][e2J] /= norm_J * 16.0 * pow(speedOfLight,4);
                    }
                    else if(intType == "SSLL_SF")
                    {
                        array_radial_J[tmp][e1J][e2J] = 4.0*a1*a2 * radial_2e_list_J[tmp](1,0);
                        if(l_p != 0)
                            array_radial_J[tmp][e1J][e2J] += (l_p*l_p + l_p*(l_p+1)/2 + l_p*(l_p+1)/2 - tmp*(tmp+1)/2) * radial_2e_list_J[tmp](2,0) - (2.0*a1*l_p+2.0*a2*l_p) * radial_2e_list_J[tmp](0,0);
                        array_radial_J[tmp][e1J][e2J] /= norm_J * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "SSSS_SF")
                    {
                        double l12 = l_p*l_p + l_p*(l_p+1)/2 + l_p*(l_p+1)/2 - tmp*(tmp+1)/2, l34 = l_q*l_q + l_q*(l_q+1)/2 + l_q*(l_q+1)/2 - tmp*(tmp+1)/2;
                        array_radial_J[tmp][e1J][e2J] = 4*a1*a2*4*a3*a4 * radial_2e_list_J[tmp](1,1);
                        if(l_p != 0)
                        {
                            if(l_q != 0)
                                array_radial_J[tmp][e1J][e2J] += l12*l34 * radial_2e_list_J[tmp](2,2) - (2*a1*l_p+2*a2*l_p)*l34 * radial_2e_list_J[tmp](0,2) + 4*a1*a2*l34 * radial_2e_list_J[tmp](1,2) - l12*(2*a3*l_q+2*a4*l_q) * radial_2e_list_J[tmp](2,0) + (2*a1*l_p+2*a2*l_p)*(2*a3*l_q+2*a4*l_q) * radial_2e_list_J[tmp](0,0) - 4*a1*a2*(2*a3*l_q+2*a4*l_q) * radial_2e_list_J[tmp](1,0) + l12*4*a3*a4 * radial_2e_list_J[tmp](2,1) - (2*a1*l_p+2*a2*l_p)*4*a3*a4 * radial_2e_list_J[tmp](0,1);
                            else
                                array_radial_J[tmp][e1J][e2J] += l12*4*a3*a4 * radial_2e_list_J[tmp](2,1) - (2*a1*l_p+2*a2*l_p)*4*a3*a4 * radial_2e_list_J[tmp](0,1);
                        }
                        else
                        {
                            if(l_q != 0)
                                array_radial_J[tmp][e1J][e2J] += 4*a1*a2*l34 * radial_2e_list_J[tmp](1,2) - 4*a1*a2*(2*a3*l_q+2*a4*l_q) * radial_2e_list_J[tmp](1,0);
                        }
                        array_radial_J[tmp][e1J][e2J] /= norm_J * 16.0 * pow(speedOfLight,4);
                    }
                    else if(intType == "SSLL_SD")
                    {
                        array_radial_J[tmp][e1J][e2J] = 0.0;
                        if(l_p != 0)
                            array_radial_J[tmp][e1J][e2J] += (lk1*lk2 - (l_p*l_p + l_p*(l_p+1)/2 + l_p*(l_p+1)/2 - tmp*(tmp+1)/2)) * radial_2e_list_J[tmp](2,0) - (2.0*a1*lk2+2.0*a2*lk1 - 2.0*a1*l_p-2.0*a2*l_p) * radial_2e_list_J[tmp](0,0);
                        array_radial_J[tmp][e1J][e2J] /= norm_J * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "SSSS_SD")
                    {
                        double l12 = l_p*l_p + l_p*(l_p+1)/2 + l_p*(l_p+1)/2 - tmp*(tmp+1)/2, l34 = l_q*l_q + l_q*(l_q+1)/2 + l_q*(l_q+1)/2 - tmp*(tmp+1)/2;
                        array_radial_J[tmp][e1J][e2J] = 0.0;
                        if(l_p != 0)
                        {
                            if(l_q != 0)
                                array_radial_J[tmp][e1J][e2J] += (lk1*lk2*lk3*lk4 - l12*l34) * radial_2e_list_J[tmp](2,2) - ((2*a1*lk2+2*a2*lk1)*lk3*lk4 - (2*a1*l_p+2*a2*l_p)*l34) * radial_2e_list_J[tmp](0,2) + (4*a1*a2*lk3*lk4 - 4*a1*a2*l34) * radial_2e_list_J[tmp](1,2) - (lk1*lk2*(2*a3*lk4+2*a4*lk3) - l12*(2*a3*l_q+2*a4*l_q)) * radial_2e_list_J[tmp](2,0) + ((2*a1*lk2+2*a2*lk1)*(2*a3*lk4+2*a4*lk3) - (2*a1*l_p+2*a2*l_p)*(2*a3*l_q+2*a4*l_q)) * radial_2e_list_J[tmp](0,0) - (4*a1*a2*(2*a3*lk4+2*a4*lk3) - 4*a1*a2*(2*a3*l_q+2*a4*l_q)) * radial_2e_list_J[tmp](1,0) + (lk1*lk2*4*a3*a4 - l12*4*a3*a4) * radial_2e_list_J[tmp](2,1) - ((2*a1*lk2+2*a2*lk1)*4*a3*a4 - (2*a1*l_p+2*a2*l_p)*4*a3*a4) * radial_2e_list_J[tmp](0,1);
                            else
                                array_radial_J[tmp][e1J][e2J] += (lk1*lk2*4*a3*a4 - l12*4*a3*a4) * radial_2e_list_J[tmp](2,1) - ((2*a1*lk2+2*a2*lk1)*4*a3*a4 - (2*a1*l_p+2*a2*l_p)*4*a3*a4) * radial_2e_list_J[tmp](0,1);
                        }
                        else
                        {
                            if(l_q != 0)
                                array_radial_J[tmp][e1J][e2J] += (4*a1*a2*lk3*lk4 - 4*a1*a2*l34) * radial_2e_list_J[tmp](1,2) - (4*a1*a2*(2*a3*lk4+2*a4*lk3) - 4*a1*a2*(2*a3*l_q+2*a4*l_q)) * radial_2e_list_J[tmp](1,0);
                        }
                        array_radial_J[tmp][e1J][e2J] /= norm_J * 16.0 * pow(speedOfLight,4);
                    }
                    else
                    {
                        cout << "ERROR: Unknown integralTYPE in get_h2e:\n";
                        exit(99);
                    }
                }
                lk2 = 1+l_q+k_q; lk4 = 1+l_p+k_p; 
                a2 = shell_list(qshell).exp_a(ll); a4 = shell_list(pshell).exp_a(jj);
                for(int tmp = LmaxK; tmp >= 0; tmp -= 2)
                {
                    if(intType == "LLLL")
                    {
                        array_radial_K[tmp][e1K][e2K] = radial_2e_list_K[tmp](0,0) / norm_K;
                    }
                    else if(intType == "SSLL")
                    {
                        array_radial_K[tmp][e1K][e2K] = 4.0*a1*a2 * radial_2e_list_K[tmp](1,0);
                        if(l_p != 0 && l_q != 0)
                            array_radial_K[tmp][e1K][e2K] += lk1*lk2 * radial_2e_list_K[tmp](2,0) - (2.0*a1*lk2+2.0*a2*lk1) * radial_2e_list_K[tmp](0,0);
                        else if(l_p != 0 || l_q != 0)
                            array_radial_K[tmp][e1K][e2K] += - (2.0*a1*lk2+2.0*a2*lk1) * radial_2e_list_K[tmp](0,0);
                        array_radial_K[tmp][e1K][e2K] /= norm_K * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "SSSS")
                    {
                        array_radial_K[tmp][e1K][e2K] = 4*a1*a2*4*a3*a4 * radial_2e_list_K[tmp](1,1);
                        if(l_p != 0 && l_q != 0)
                        {
                            array_radial_K[tmp][e1K][e2K] += lk1*lk2*lk3*lk4 * radial_2e_list_K[tmp](2,2) - (2*a1*lk2+2*a2*lk1)*lk3*lk4 * radial_2e_list_K[tmp](0,2) + 4*a1*a2*lk3*lk4 * radial_2e_list_K[tmp](1,2) - lk1*lk2*(2*a3*lk4+2*a4*lk3) * radial_2e_list_K[tmp](2,0) + (2*a1*lk2+2*a2*lk1)*(2*a3*lk4+2*a4*lk3) * radial_2e_list_K[tmp](0,0) - 4*a1*a2*(2*a3*lk4+2*a4*lk3) * radial_2e_list_K[tmp](1,0) + lk1*lk2*4*a3*a4 * radial_2e_list_K[tmp](2,1) - (2*a1*lk2+2*a2*lk1)*4*a3*a4 * radial_2e_list_K[tmp](0,1);
                        }
                        else if(l_p != 0 || l_q != 0)
                        {
                            array_radial_K[tmp][e1K][e2K] += (2*a1*lk2+2*a2*lk1)*(2*a3*lk4+2*a4*lk3) * radial_2e_list_K[tmp](0,0) - 4*a1*a2*(2*a3*lk4+2*a4*lk3) * radial_2e_list_K[tmp](1,0) - (2*a1*lk2+2*a2*lk1)*4*a3*a4 * radial_2e_list_K[tmp](0,1);
                        }
                        array_radial_K[tmp][e1K][e2K] /= norm_K * 16.0 * pow(speedOfLight,4);
                    }
                    else if(intType == "SSLL_SF")
                    {
                        array_radial_K[tmp][e1K][e2K] = 4.0*a1*a2 * radial_2e_list_K[tmp](1,0);
                        if(l_p != 0 && l_q != 0)
                            array_radial_K[tmp][e1K][e2K] += (l_p*l_q + l_p*(l_p+1)/2 + l_q*(l_q+1)/2 - tmp*(tmp+1)/2) * radial_2e_list_K[tmp](2,0) - (2.0*a1*l_q+2.0*a2*l_p)* radial_2e_list_K[tmp](0,0);
                        else if(l_p != 0 || l_q != 0)
                            array_radial_K[tmp][e1K][e2K] += - (2.0*a1*l_q+2.0*a2*l_p) * radial_2e_list_K[tmp](0,0);
                        array_radial_K[tmp][e1K][e2K] /= norm_K * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "SSSS_SF")
                    {
                        double l12 = l_p*l_q + l_p*(l_p+1)/2 + l_q*(l_q+1)/2 - tmp*(tmp+1)/2, l34 = l_q*l_p + l_q*(l_q+1)/2 + l_p*(l_p+1)/2 - tmp*(tmp+1)/2;
                        array_radial_K[tmp][e1K][e2K] = 4*a1*a2*4*a3*a4 * radial_2e_list_K[tmp](1,1);
                        if(l_p != 0 && l_q != 0)
                        {
                            array_radial_K[tmp][e1K][e2K] += l12*l34 * radial_2e_list_K[tmp](2,2) - (2*a1*l_q+2*a2*l_p)*l34 * radial_2e_list_K[tmp](0,2) + 4*a1*a2*l34 * radial_2e_list_K[tmp](1,2) - l12*(2*a3*l_p+2*a4*l_q) * radial_2e_list_K[tmp](2,0) + (2*a1*l_q+2*a2*l_p)*(2*a3*l_p+2*a4*l_q) * radial_2e_list_K[tmp](0,0) - 4*a1*a2*(2*a3*l_p+2*a4*l_q) * radial_2e_list_K[tmp](1,0) + l12*4*a3*a4 * radial_2e_list_K[tmp](2,1) - (2*a1*l_q+2*a2*l_p)*4*a3*a4 * radial_2e_list_K[tmp](0,1);
                        }
                        else if(l_p != 0 || l_q != 0)
                        {
                            array_radial_K[tmp][e1K][e2K] += (2*a1*l_q+2*a2*l_p)*(2*a3*l_p+2*a4*l_q) * radial_2e_list_K[tmp](0,0) - 4*a1*a2*(2*a3*l_p+2*a4*l_q) * radial_2e_list_K[tmp](1,0) - (2*a1*l_q+2*a2*l_p)*4*a3*a4 * radial_2e_list_K[tmp](0,1);
                        }
                        array_radial_K[tmp][e1K][e2K] /= norm_K * 16.0 * pow(speedOfLight,4);
                    }
                    else if(intType == "SSLL_SD")
                    {
                        array_radial_K[tmp][e1K][e2K] = 0.0;
                        if(l_p != 0 && l_q != 0)
                            array_radial_K[tmp][e1K][e2K] += (lk1*lk2-(l_p*l_q + l_p*(l_p+1)/2 + l_q*(l_q+1)/2 - tmp*(tmp+1)/2)) * radial_2e_list_K[tmp](2,0) - (2.0*a1*lk2+2.0*a2*lk1 - 2.0*a1*l_q - 2.0*a2*l_p)* radial_2e_list_K[tmp](0,0);
                        else if(l_p != 0 || l_q != 0)
                            array_radial_K[tmp][e1K][e2K] += - (2.0*a1*lk2+2.0*a2*lk1 - 2.0*a1*l_q-2.0*a2*l_p) * radial_2e_list_K[tmp](0,0);
                        array_radial_K[tmp][e1K][e2K] /= norm_K * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "SSSS_SD")
                    {
                        double l12 = l_p*l_q + l_p*(l_p+1)/2 + l_q*(l_q+1)/2 - tmp*(tmp+1)/2, l34 = l_q*l_p + l_q*(l_q+1)/2 + l_p*(l_p+1)/2 - tmp*(tmp+1)/2;
                        array_radial_K[tmp][e1K][e2K] = 0.0;
                        if(l_p != 0 && l_q != 0)
                        {
                            array_radial_K[tmp][e1K][e2K] += (lk1*lk2*lk3*lk4 - l12*l34) * radial_2e_list_K[tmp](2,2) - ((2*a1*lk2+2*a2*lk1)*lk3*lk4 - (2*a1*l_q+2*a2*l_p)*l34) * radial_2e_list_K[tmp](0,2) + (4*a1*a2*lk3*lk4 - 4*a1*a2*l34) * radial_2e_list_K[tmp](1,2) - (lk1*lk2*(2*a3*lk4+2*a4*lk3) - l12*(2*a3*l_p+2*a4*l_q)) * radial_2e_list_K[tmp](2,0) + ((2*a1*lk2+2*a2*lk1)*(2*a3*lk4+2*a4*lk3) - (2*a1*l_q+2*a2*l_p)*(2*a3*l_p+2*a4*l_q)) * radial_2e_list_K[tmp](0,0) - (4*a1*a2*(2*a3*lk4+2*a4*lk3) -  4*a1*a2*(2*a3*l_p+2*a4*l_q)) * radial_2e_list_K[tmp](1,0) + (lk1*lk2*4*a3*a4 - l12*4*a3*a4) * radial_2e_list_K[tmp](2,1) - ((2*a1*lk2+2*a2*lk1)*4*a3*a4 - (2*a1*l_q+2*a2*l_p)*4*a3*a4) * radial_2e_list_K[tmp](0,1);
                        }
                        else if(l_p != 0 || l_q != 0)
                        {
                            array_radial_K[tmp][e1K][e2K] += ((2*a1*lk2+2*a2*lk1)*(2*a3*lk4+2*a4*lk3) - (2*a1*l_q+2*a2*l_p)*(2*a3*l_p+2*a4*l_q)) * radial_2e_list_K[tmp](0,0) - (4*a1*a2*(2*a3*lk4+2*a4*lk3) - 4*a1*a2*(2*a3*l_p+2*a4*l_q)) * radial_2e_list_K[tmp](1,0) - ((2*a1*lk2+2*a2*lk1)*4*a3*a4 - (2*a1*l_q+2*a2*l_p)*4*a3*a4) * radial_2e_list_K[tmp](0,1);
                        }
                        array_radial_K[tmp][e1K][e2K] /= norm_K * 16.0 * pow(speedOfLight,4);
                    }
                    else
                    {
                        cout << "ERROR: Unknown integralTYPE in get_h2e:\n";
                        exit(99);
                    }
                }
            }
        }
        EndTime = clock();
        time_r += (EndTime - StartTime)/(double)CLOCKS_PER_SEC;

        StartTime = clock();
        int l_p_cycle = (l_p == 0) ? 1 : 2, l_q_cycle = (l_q == 0) ? 1 : 2;
        for(int int_tmp2_p = 0; int_tmp2_p < l_p_cycle; int_tmp2_p++)
        for(int int_tmp2_q = 0; int_tmp2_q < l_q_cycle; int_tmp2_q++)
        {
            int add_p = int_tmp2_p*(irrep_list(int_tmp1_p).two_j+1), add_q = int_tmp2_q*(irrep_list(int_tmp1_q).two_j+1);
            for(int mp = 0; mp < irrep_list(int_tmp1_p+add_p).two_j + 1; mp++)
            for(int mq = 0; mq < irrep_list(int_tmp1_q+add_q).two_j + 1; mq++)
            {
                int_2e_JK.J(int_tmp1_p+add_p + mp, int_tmp1_q + add_q + mq).resize(size_gtos_p*size_gtos_p,size_gtos_q*size_gtos_q);
                int_2e_JK.K(int_tmp1_p+add_p + mp, int_tmp1_q+add_q + mq).resize(size_gtos_p*size_gtos_q,size_gtos_q*size_gtos_p);
                #pragma omp parallel  for
                for(int tt = 0; tt < size_gtos_p*size_gtos_p*size_gtos_q*size_gtos_q; tt++)
                {
                    int e1J = tt/(size_gtos_q*size_gtos_q);
                    int e2J = tt - e1J*(size_gtos_q*size_gtos_q);
                    int e1K = tt/(size_gtos_p*size_gtos_q);
                    int e2K = tt - e1K*(size_gtos_p*size_gtos_q);
                    int_2e_JK.J(int_tmp1_p+add_p + mp, int_tmp1_q+add_q + mq)(e1J,e2J) = 0.0;
                    int_2e_JK.K(int_tmp1_p+add_p + mp, int_tmp1_q+add_q + mq)(e1K,e2K) = 0.0;
                    for(int tmp = LmaxJ; tmp >= 0; tmp = tmp - 2)
                        int_2e_JK.J(int_tmp1_p+add_p + mp, int_tmp1_q+add_q + mq)(e1J,e2J) += array_radial_J[tmp][e1J][e2J] * array_angular_J[tmp][int_tmp2_p][int_tmp2_q](mp,mq);
                    for(int tmp = LmaxK; tmp >= 0; tmp = tmp - 2)
                        int_2e_JK.K(int_tmp1_p+add_p + mp, int_tmp1_q+add_q + mq)(e1K,e2K) += array_radial_K[tmp][e1K][e2K] * array_angular_K[tmp][int_tmp2_p][int_tmp2_q](mp,mq);
                }
            }
        }
        EndTime = clock();
        time_c += (EndTime - StartTime)/(double)CLOCKS_PER_SEC;
        int_tmp1_q += 4*l_q+2;
    }
    int_tmp1_p += 4*l_p+2;
    }

    cout << time_a << "\t" << time_r << "\t" << time_c <<endl;
    return int_2e_JK;
}
int2eJK INT_SPH::get_h2e_JK_compact(const string& intType, const int& occMaxL) const
{
    double time_a = 0.0, time_r = 0.0, time_c = 0.0;
    int occMaxShell = 0, Nirrep_compact = 0;
    if(occMaxL == -1)    occMaxShell = size_shell;
    else
    {
        for(int ii = 0; ii < size_shell; ii++)
        {
            if(shell_list(ii).l <= occMaxL)
                occMaxShell++;
            else
                break;
        }
    }
    for(int ii = 0; ii < occMaxShell; ii++)
    {
        if(shell_list(ii).l == 0) Nirrep_compact += 1;
        else Nirrep_compact += 2;
    }
    
    int2eJK int_2e_JK;
    int_2e_JK.J.resize(Nirrep_compact, Nirrep_compact);
    int_2e_JK.K.resize(Nirrep_compact, Nirrep_compact);
    
    int int_tmp1_p = 0;
    for(int pshell = 0; pshell < occMaxShell; pshell++)
    {
    int l_p = shell_list(pshell).l, int_tmp1_q = 0;
    for(int qshell = 0; qshell < occMaxShell; qshell++)
    {
        int l_q = shell_list(qshell).l, l_max = max(l_p,l_q), LmaxJ = min(l_p+l_p, l_q+l_q), LmaxK = l_p+l_q;
        int size_gtos_p = shell_list(pshell).coeff.rows(), size_gtos_q = shell_list(qshell).coeff.rows();
        int size_tmp_p = (l_p == 0) ? 1 : 2, size_tmp_q = (l_q == 0) ? 1 : 2;
        double array_radial_J[LmaxJ+1][size_gtos_p*size_gtos_p][size_gtos_q*size_gtos_q][size_tmp_p][size_tmp_q];
        double array_radial_K[LmaxK+1][size_gtos_p*size_gtos_q][size_gtos_q*size_gtos_p][size_tmp_p][size_tmp_q];
        double array_angular_J[LmaxJ+1][size_tmp_p][size_tmp_q], array_angular_K[LmaxK+1][size_tmp_p][size_tmp_q];

        StartTime = clock();
        #pragma omp parallel  for
        for(int twojj_p = abs(2*l_p-1); twojj_p <= 2*l_p+1; twojj_p = twojj_p + 2)
        for(int twojj_q = abs(2*l_q-1); twojj_q <= 2*l_q+1; twojj_q = twojj_q + 2)
        {
            int sym_ap = twojj_p - 2*l_p, sym_aq = twojj_q - 2*l_q;
            int index_tmp_p = (l_p > 0) ? 1 - (2*l_p+1 - twojj_p)/2 : 0;
            int index_tmp_q = (l_q > 0) ? 1 - (2*l_q+1 - twojj_q)/2 : 0;

            for(int tmp = LmaxJ; tmp >= 0; tmp -= 2)
            {
                double tmp_d = 0.0;
                for(int mq = 0; mq < twojj_q + 1; mq++)
                {
                    tmp_d += int2e_get_angular_J(l_p, twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, tmp);
                }
                tmp_d /= (twojj_q + 1);
                array_angular_J[tmp][index_tmp_p][index_tmp_q] = tmp_d;
            }
            for(int tmp = LmaxK; tmp >= 0; tmp -= 2)
            {
                double tmp_d = 0.0;
                for(int mq = 0; mq < twojj_q + 1; mq++)
                {
                    tmp_d += int2e_get_angular_K(l_p, twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, tmp);
                }
                tmp_d /= (twojj_q + 1);
                array_angular_K[tmp][index_tmp_p][index_tmp_q] = tmp_d;
            }
        }
        EndTime = clock();
        time_a += (EndTime - StartTime)/(double)CLOCKS_PER_SEC;


        StartTime = clock();
        #pragma omp parallel  for
        for(int tt = 0; tt < size_gtos_p*size_gtos_p*size_gtos_q*size_gtos_q; tt++)
        {
            int e1J = tt/(size_gtos_q*size_gtos_q);
            int e2J = tt - e1J*(size_gtos_q*size_gtos_q);
            int ii = e1J/size_gtos_p, jj = e1J - ii*size_gtos_p;
            int kk = e2J/size_gtos_q, ll = e2J - kk*size_gtos_q;
            int e1K = ii*size_gtos_q+ll, e2K = kk*size_gtos_p+jj;
            MatrixXd radial_2e_list_J[LmaxJ+1], radial_2e_list_K[LmaxK+1];
            double a_i_J = shell_list(pshell).exp_a(ii), a_j_J = shell_list(pshell).exp_a(jj), a_k_J = shell_list(qshell).exp_a(kk), a_l_J = shell_list(qshell).exp_a(ll);
            double a_i_K = shell_list(pshell).exp_a(ii), a_j_K = shell_list(qshell).exp_a(ll), a_k_K = shell_list(qshell).exp_a(kk), a_l_K = shell_list(pshell).exp_a(jj);
        
            if(intType.substr(0,4) == "LLLL")
            {
                for(int LL = LmaxJ; LL >= 0; LL -= 2)
                {
                    radial_2e_list_J[LL].resize(1,1);
                    radial_2e_list_J[LL](0,0) = int2e_get_radial(l_p,a_i_J,l_p,a_j_J,l_q,a_k_J,l_q,a_l_J,LL);
                }
                for(int LL = LmaxK; LL >= 0; LL -= 2)
                {
                    radial_2e_list_K[LL].resize(1,1);
                    radial_2e_list_K[LL](0,0) = int2e_get_radial(l_p,a_i_K,l_q,a_j_K,l_q,a_k_K,l_p,a_l_K,LL);
                }
            }
            else if(intType.substr(0,4) == "SSLL")
            {
                for(int LL = LmaxJ; LL >= 0; LL -= 2)
                {
                    radial_2e_list_J[LL].resize(3,1);
                    radial_2e_list_J[LL](0,0) = int2e_get_radial(l_p,a_i_J,l_p,a_j_J,l_q,a_k_J,l_q,a_l_J,LL);
                    radial_2e_list_J[LL](1,0) = int2e_get_radial(l_p+1,a_i_J,l_p+1,a_j_J,l_q,a_k_J,l_q,a_l_J,LL);
                    if(l_p != 0)
                        radial_2e_list_J[LL](2,0) = int2e_get_radial(l_p-1,a_i_J,l_p-1,a_j_J,l_q,a_k_J,l_q,a_l_J,LL);
                }
                for(int LL = LmaxK; LL >= 0; LL -= 2)
                {
                    radial_2e_list_K[LL].resize(3,1);
                    radial_2e_list_K[LL](0,0) = int2e_get_radial(l_p,a_i_K,l_q,a_j_K,l_q,a_k_K,l_p,a_l_K,LL);
                    radial_2e_list_K[LL](1,0) = int2e_get_radial(l_p+1,a_i_K,l_q+1,a_j_K,l_q,a_k_K,l_p,a_l_K,LL);
                    if(l_p != 0 && l_q != 0)
                        radial_2e_list_K[LL](2,0) = int2e_get_radial(l_p-1,a_i_K,l_q-1,a_j_K,l_q,a_k_K,l_p,a_l_K,LL);
                }
            }
            else if(intType.substr(0,4) == "SSSS")
            {
                for(int LL = LmaxJ; LL >= 0; LL -= 2)
                {
                    radial_2e_list_J[LL].resize(3,3);
                    radial_2e_list_J[LL](0,0) = int2e_get_radial(l_p,a_i_J,l_p,a_j_J,l_q,a_k_J,l_q,a_l_J,LL);
                    radial_2e_list_J[LL](1,0) = int2e_get_radial(l_p+1,a_i_J,l_p+1,a_j_J,l_q,a_k_J,l_q,a_l_J,LL);
                    radial_2e_list_J[LL](0,1) = int2e_get_radial(l_p,a_i_J,l_p,a_j_J,l_q+1,a_k_J,l_q+1,a_l_J,LL);
                    radial_2e_list_J[LL](1,1) = int2e_get_radial(l_p+1,a_i_J,l_p+1,a_j_J,l_q+1,a_k_J,l_q+1,a_l_J,LL);
                    if(l_p != 0)
                    {
                        radial_2e_list_J[LL](2,0) = int2e_get_radial(l_p-1,a_i_J,l_p-1,a_j_J,l_q,a_k_J,l_q,a_l_J,LL);
                        radial_2e_list_J[LL](2,1) = int2e_get_radial(l_p-1,a_i_J,l_p-1,a_j_J,l_q+1,a_k_J,l_q+1,a_l_J,LL);
                    }
                    if(l_q != 0)
                    {
                        radial_2e_list_J[LL](0,2) = int2e_get_radial(l_p,a_i_J,l_p,a_j_J,l_q-1,a_k_J,l_q-1,a_l_J,LL);
                        radial_2e_list_J[LL](1,2) = int2e_get_radial(l_p+1,a_i_J,l_p+1,a_j_J,l_q-1,a_k_J,l_q-1,a_l_J,LL);
                    }
                    if(l_p!=0 && l_q!=0)
                        radial_2e_list_J[LL](2,2) = int2e_get_radial(l_p-1,a_i_J,l_p-1,a_j_J,l_q-1,a_k_J,l_q-1,a_l_J,LL);
                }
                for(int LL = LmaxK; LL >= 0; LL -= 2)
                {
                    radial_2e_list_K[LL].resize(3,3);
                    radial_2e_list_K[LL](0,0) = int2e_get_radial(l_p,a_i_K,l_q,a_j_K,l_q,a_k_K,l_p,a_l_K,LL);
                    radial_2e_list_K[LL](1,0) = int2e_get_radial(l_p+1,a_i_K,l_q+1,a_j_K,l_q,a_k_K,l_p,a_l_K,LL);
                    radial_2e_list_K[LL](0,1) = int2e_get_radial(l_p,a_i_K,l_q,a_j_K,l_q+1,a_k_K,l_p+1,a_l_K,LL);
                    radial_2e_list_K[LL](1,1) = int2e_get_radial(l_p+1,a_i_K,l_q+1,a_j_K,l_q+1,a_k_K,l_p+1,a_l_K,LL);
                    if(l_p != 0 && l_q != 0)
                    {
                        radial_2e_list_K[LL](2,0) = int2e_get_radial(l_p-1,a_i_K,l_q-1,a_j_K,l_q,a_k_K,l_p,a_l_K,LL);
                        radial_2e_list_K[LL](2,1) = int2e_get_radial(l_p-1,a_i_K,l_q-1,a_j_K,l_q+1,a_k_K,l_p+1,a_l_K,LL);
                        radial_2e_list_K[LL](0,2) = int2e_get_radial(l_p,a_i_K,l_q,a_j_K,l_q-1,a_k_K,l_p-1,a_l_K,LL);
                        radial_2e_list_K[LL](1,2) = int2e_get_radial(l_p+1,a_i_K,l_q+1,a_j_K,l_q-1,a_k_K,l_p-1,a_l_K,LL);
                        radial_2e_list_K[LL](2,2) = int2e_get_radial(l_p-1,a_i_K,l_q-1,a_j_K,l_q-1,a_k_K,l_p-1,a_l_K,LL);
                    }
                }
            }

            for(int twojj_p = abs(2*l_p-1); twojj_p <= 2*l_p+1; twojj_p = twojj_p + 2)
            for(int twojj_q = abs(2*l_q-1); twojj_q <= 2*l_q+1; twojj_q = twojj_q + 2)
            {
                int index_tmp_p = (l_p > 0) ? 1 - (2*l_p+1 - twojj_p)/2 : 0;
                int index_tmp_q = (l_q > 0) ? 1 - (2*l_q+1 - twojj_q)/2 : 0;
                int sym_ap = twojj_p - 2*l_p, sym_aq = twojj_q - 2*l_q;
                double k_p = -(twojj_p+1.0)*sym_ap/2.0, k_q = -(twojj_q+1.0)*sym_aq/2.0;
                double norm_J = shell_list(pshell).norm(ii) * shell_list(pshell).norm(jj) * shell_list(qshell).norm(kk) * shell_list(qshell).norm(ll), norm_K = shell_list(pshell).norm(ii) * shell_list(qshell).norm(ll) * shell_list(qshell).norm(kk) * shell_list(pshell).norm(jj);
                double lk1 = 1+l_p+k_p, lk2 = 1+l_p+k_p, lk3 = 1+l_q+k_q, lk4 = 1+l_q+k_q, a1 = shell_list(pshell).exp_a(ii), a2 = shell_list(pshell).exp_a(jj), a3 = shell_list(qshell).exp_a(kk), a4 = shell_list(qshell).exp_a(ll);

                for(int tmp = LmaxJ; tmp >= 0; tmp -= 2)
                {
                    if(intType == "LLLL")
                    {
                        array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] = radial_2e_list_J[tmp](0,0) / norm_J;
                    }
                    else if(intType == "SSLL")
                    {
                        array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] = 4.0*a1*a2 * radial_2e_list_J[tmp](1,0);
                        if(l_p != 0)
                            array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] += lk1*lk2 * radial_2e_list_J[tmp](2,0) - (2.0*a1*lk2+2.0*a2*lk1) * radial_2e_list_J[tmp](0,0);
                        array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] /= norm_J * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "SSSS")
                    {
                        array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] = 4*a1*a2*4*a3*a4 * radial_2e_list_J[tmp](1,1);
                        if(l_p != 0)
                        {
                            if(l_q != 0)
                                array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] += lk1*lk2*lk3*lk4 * radial_2e_list_J[tmp](2,2) - (2*a1*lk2+2*a2*lk1)*lk3*lk4 * radial_2e_list_J[tmp](0,2) + 4*a1*a2*lk3*lk4 * radial_2e_list_J[tmp](1,2) - lk1*lk2*(2*a3*lk4+2*a4*lk3) * radial_2e_list_J[tmp](2,0) + (2*a1*lk2+2*a2*lk1)*(2*a3*lk4+2*a4*lk3) * radial_2e_list_J[tmp](0,0) - 4*a1*a2*(2*a3*lk4+2*a4*lk3) * radial_2e_list_J[tmp](1,0) + lk1*lk2*4*a3*a4 * radial_2e_list_J[tmp](2,1) - (2*a1*lk2+2*a2*lk1)*4*a3*a4 * radial_2e_list_J[tmp](0,1);
                            else
                                array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] += lk1*lk2*4*a3*a4 * radial_2e_list_J[tmp](2,1) - (2*a1*lk2+2*a2*lk1)*4*a3*a4 * radial_2e_list_J[tmp](0,1);
                        }
                        else
                        {
                            if(l_q != 0)
                                array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] += 4*a1*a2*lk3*lk4 * radial_2e_list_J[tmp](1,2) - 4*a1*a2*(2*a3*lk4+2*a4*lk3) * radial_2e_list_J[tmp](1,0);
                        }
                        array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] /= norm_J * 16.0 * pow(speedOfLight,4);
                    }
                    else if(intType == "SSLL_SF")
                    {
                        array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] = 4.0*a1*a2 * radial_2e_list_J[tmp](1,0);
                        if(l_p != 0)
                            array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] += (l_p*l_p + l_p*(l_p+1)/2 + l_p*(l_p+1)/2 - tmp*(tmp+1)/2) * radial_2e_list_J[tmp](2,0) - (2.0*a1*l_p+2.0*a2*l_p) * radial_2e_list_J[tmp](0,0);
                        array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] /= norm_J * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "SSSS_SF")
                    {
                        double l12 = l_p*l_p + l_p*(l_p+1)/2 + l_p*(l_p+1)/2 - tmp*(tmp+1)/2, l34 = l_q*l_q + l_q*(l_q+1)/2 + l_q*(l_q+1)/2 - tmp*(tmp+1)/2;
                        array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] = 4*a1*a2*4*a3*a4 * radial_2e_list_J[tmp](1,1);
                        if(l_p != 0)
                        {
                            if(l_q != 0)
                                array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] += l12*l34 * radial_2e_list_J[tmp](2,2) - (2*a1*l_p+2*a2*l_p)*l34 * radial_2e_list_J[tmp](0,2) + 4*a1*a2*l34 * radial_2e_list_J[tmp](1,2) - l12*(2*a3*l_q+2*a4*l_q) * radial_2e_list_J[tmp](2,0) + (2*a1*l_p+2*a2*l_p)*(2*a3*l_q+2*a4*l_q) * radial_2e_list_J[tmp](0,0) - 4*a1*a2*(2*a3*l_q+2*a4*l_q) * radial_2e_list_J[tmp](1,0) + l12*4*a3*a4 * radial_2e_list_J[tmp](2,1) - (2*a1*l_p+2*a2*l_p)*4*a3*a4 * radial_2e_list_J[tmp](0,1);
                            else
                                array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] += l12*4*a3*a4 * radial_2e_list_J[tmp](2,1) - (2*a1*l_p+2*a2*l_p)*4*a3*a4 * radial_2e_list_J[tmp](0,1);
                        }
                        else
                        {
                            if(l_q != 0)
                                array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] += 4*a1*a2*l34 * radial_2e_list_J[tmp](1,2) - 4*a1*a2*(2*a3*l_q+2*a4*l_q) * radial_2e_list_J[tmp](1,0);
                        }
                        array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] /= norm_J * 16.0 * pow(speedOfLight,4);
                    }
                    else if(intType == "SSLL_SD")
                    {
                        array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] = 0.0;
                        if(l_p != 0)
                            array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] += (lk1*lk2 - (l_p*l_p + l_p*(l_p+1)/2 + l_p*(l_p+1)/2 - tmp*(tmp+1)/2)) * radial_2e_list_J[tmp](2,0) - (2.0*a1*lk2+2.0*a2*lk1 - 2.0*a1*l_p-2.0*a2*l_p) * radial_2e_list_J[tmp](0,0);
                        array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] /= norm_J * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "SSSS_SD")
                    {
                        double l12 = l_p*l_p + l_p*(l_p+1)/2 + l_p*(l_p+1)/2 - tmp*(tmp+1)/2, l34 = l_q*l_q + l_q*(l_q+1)/2 + l_q*(l_q+1)/2 - tmp*(tmp+1)/2;
                        array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] = 0.0;
                        if(l_p != 0)
                        {
                            if(l_q != 0)
                                array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] += (lk1*lk2*lk3*lk4 - l12*l34) * radial_2e_list_J[tmp](2,2) - ((2*a1*lk2+2*a2*lk1)*lk3*lk4 - (2*a1*l_p+2*a2*l_p)*l34) * radial_2e_list_J[tmp](0,2) + (4*a1*a2*lk3*lk4 - 4*a1*a2*l34) * radial_2e_list_J[tmp](1,2) - (lk1*lk2*(2*a3*lk4+2*a4*lk3) - l12*(2*a3*l_q+2*a4*l_q)) * radial_2e_list_J[tmp](2,0) + ((2*a1*lk2+2*a2*lk1)*(2*a3*lk4+2*a4*lk3) - (2*a1*l_p+2*a2*l_p)*(2*a3*l_q+2*a4*l_q)) * radial_2e_list_J[tmp](0,0) - (4*a1*a2*(2*a3*lk4+2*a4*lk3) - 4*a1*a2*(2*a3*l_q+2*a4*l_q)) * radial_2e_list_J[tmp](1,0) + (lk1*lk2*4*a3*a4 - l12*4*a3*a4) * radial_2e_list_J[tmp](2,1) - ((2*a1*lk2+2*a2*lk1)*4*a3*a4 - (2*a1*l_p+2*a2*l_p)*4*a3*a4) * radial_2e_list_J[tmp](0,1);
                            else
                                array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] += (lk1*lk2*4*a3*a4 - l12*4*a3*a4) * radial_2e_list_J[tmp](2,1) - ((2*a1*lk2+2*a2*lk1)*4*a3*a4 - (2*a1*l_p+2*a2*l_p)*4*a3*a4) * radial_2e_list_J[tmp](0,1);
                        }
                        else
                        {
                            if(l_q != 0)
                                array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] += (4*a1*a2*lk3*lk4 - 4*a1*a2*l34) * radial_2e_list_J[tmp](1,2) - (4*a1*a2*(2*a3*lk4+2*a4*lk3) - 4*a1*a2*(2*a3*l_q+2*a4*l_q)) * radial_2e_list_J[tmp](1,0);
                        }
                        array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] /= norm_J * 16.0 * pow(speedOfLight,4);
                    }
                    else
                    {
                        cout << "ERROR: Unknown integralTYPE in get_h2e:\n";
                        exit(99);
                    }
                }
                lk2 = 1+l_q+k_q; lk4 = 1+l_p+k_p; 
                a2 = shell_list(qshell).exp_a(ll); a4 = shell_list(pshell).exp_a(jj);
                for(int tmp = LmaxK; tmp >= 0; tmp -= 2)
                {
                    if(intType == "LLLL")
                    {
                        array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] = radial_2e_list_K[tmp](0,0) / norm_K;
                    }
                    else if(intType == "SSLL")
                    {
                        array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] = 4.0*a1*a2 * radial_2e_list_K[tmp](1,0);
                        if(l_p != 0 && l_q != 0)
                            array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] += lk1*lk2 * radial_2e_list_K[tmp](2,0) - (2.0*a1*lk2+2.0*a2*lk1) * radial_2e_list_K[tmp](0,0);
                        else if(l_p != 0 || l_q != 0)
                            array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] += - (2.0*a1*lk2+2.0*a2*lk1) * radial_2e_list_K[tmp](0,0);
                        array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] /= norm_K * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "SSSS")
                    {
                        array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] = 4*a1*a2*4*a3*a4 * radial_2e_list_K[tmp](1,1);
                        if(l_p != 0 && l_q != 0)
                        {
                            array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] += lk1*lk2*lk3*lk4 * radial_2e_list_K[tmp](2,2) - (2*a1*lk2+2*a2*lk1)*lk3*lk4 * radial_2e_list_K[tmp](0,2) + 4*a1*a2*lk3*lk4 * radial_2e_list_K[tmp](1,2) - lk1*lk2*(2*a3*lk4+2*a4*lk3) * radial_2e_list_K[tmp](2,0) + (2*a1*lk2+2*a2*lk1)*(2*a3*lk4+2*a4*lk3) * radial_2e_list_K[tmp](0,0) - 4*a1*a2*(2*a3*lk4+2*a4*lk3) * radial_2e_list_K[tmp](1,0) + lk1*lk2*4*a3*a4 * radial_2e_list_K[tmp](2,1) - (2*a1*lk2+2*a2*lk1)*4*a3*a4 * radial_2e_list_K[tmp](0,1);
                        }
                        else if(l_p != 0 || l_q != 0)
                        {
                            array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] += (2*a1*lk2+2*a2*lk1)*(2*a3*lk4+2*a4*lk3) * radial_2e_list_K[tmp](0,0) - 4*a1*a2*(2*a3*lk4+2*a4*lk3) * radial_2e_list_K[tmp](1,0) - (2*a1*lk2+2*a2*lk1)*4*a3*a4 * radial_2e_list_K[tmp](0,1);
                        }
                        array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] /= norm_K * 16.0 * pow(speedOfLight,4);
                    }
                    else if(intType == "SSLL_SF")
                    {
                        array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] = 4.0*a1*a2 * radial_2e_list_K[tmp](1,0);
                        if(l_p != 0 && l_q != 0)
                            array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] += (l_p*l_q + l_p*(l_p+1)/2 + l_q*(l_q+1)/2 - tmp*(tmp+1)/2) * radial_2e_list_K[tmp](2,0) - (2.0*a1*l_q+2.0*a2*l_p)* radial_2e_list_K[tmp](0,0);
                        else if(l_p != 0 || l_q != 0)
                            array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] += - (2.0*a1*l_q+2.0*a2*l_p) * radial_2e_list_K[tmp](0,0);
                        array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] /= norm_K * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "SSSS_SF")
                    {
                        double l12 = l_p*l_q + l_p*(l_p+1)/2 + l_q*(l_q+1)/2 - tmp*(tmp+1)/2, l34 = l_q*l_p + l_q*(l_q+1)/2 + l_p*(l_p+1)/2 - tmp*(tmp+1)/2;
                        array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] = 4*a1*a2*4*a3*a4 * radial_2e_list_K[tmp](1,1);
                        if(l_p != 0 && l_q != 0)
                        {
                            array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] += l12*l34 * radial_2e_list_K[tmp](2,2) - (2*a1*l_q+2*a2*l_p)*l34 * radial_2e_list_K[tmp](0,2) + 4*a1*a2*l34 * radial_2e_list_K[tmp](1,2) - l12*(2*a3*l_p+2*a4*l_q) * radial_2e_list_K[tmp](2,0) + (2*a1*l_q+2*a2*l_p)*(2*a3*l_p+2*a4*l_q) * radial_2e_list_K[tmp](0,0) - 4*a1*a2*(2*a3*l_p+2*a4*l_q) * radial_2e_list_K[tmp](1,0) + l12*4*a3*a4 * radial_2e_list_K[tmp](2,1) - (2*a1*l_q+2*a2*l_p)*4*a3*a4 * radial_2e_list_K[tmp](0,1);
                        }
                        else if(l_p != 0 || l_q != 0)
                        {
                            array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] += (2*a1*l_q+2*a2*l_p)*(2*a3*l_p+2*a4*l_q) * radial_2e_list_K[tmp](0,0) - 4*a1*a2*(2*a3*l_p+2*a4*l_q) * radial_2e_list_K[tmp](1,0) - (2*a1*l_q+2*a2*l_p)*4*a3*a4 * radial_2e_list_K[tmp](0,1);
                        }
                        array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] /= norm_K * 16.0 * pow(speedOfLight,4);
                    }
                    else if(intType == "SSLL_SD")
                    {
                        array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] = 0.0;
                        if(l_p != 0 && l_q != 0)
                            array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] += (lk1*lk2-(l_p*l_q + l_p*(l_p+1)/2 + l_q*(l_q+1)/2 - tmp*(tmp+1)/2)) * radial_2e_list_K[tmp](2,0) - (2.0*a1*lk2+2.0*a2*lk1 - 2.0*a1*l_q - 2.0*a2*l_p)* radial_2e_list_K[tmp](0,0);
                        else if(l_p != 0 || l_q != 0)
                            array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] += - (2.0*a1*lk2+2.0*a2*lk1 - 2.0*a1*l_q-2.0*a2*l_p) * radial_2e_list_K[tmp](0,0);
                        array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] /= norm_K * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "SSSS_SD")
                    {
                        double l12 = l_p*l_q + l_p*(l_p+1)/2 + l_q*(l_q+1)/2 - tmp*(tmp+1)/2, l34 = l_q*l_p + l_q*(l_q+1)/2 + l_p*(l_p+1)/2 - tmp*(tmp+1)/2;
                        array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] = 0.0;
                        if(l_p != 0 && l_q != 0)
                        {
                            array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] += (lk1*lk2*lk3*lk4 - l12*l34) * radial_2e_list_K[tmp](2,2) - ((2*a1*lk2+2*a2*lk1)*lk3*lk4 - (2*a1*l_q+2*a2*l_p)*l34) * radial_2e_list_K[tmp](0,2) + (4*a1*a2*lk3*lk4 - 4*a1*a2*l34) * radial_2e_list_K[tmp](1,2) - (lk1*lk2*(2*a3*lk4+2*a4*lk3) - l12*(2*a3*l_p+2*a4*l_q)) * radial_2e_list_K[tmp](2,0) + ((2*a1*lk2+2*a2*lk1)*(2*a3*lk4+2*a4*lk3) - (2*a1*l_q+2*a2*l_p)*(2*a3*l_p+2*a4*l_q)) * radial_2e_list_K[tmp](0,0) - (4*a1*a2*(2*a3*lk4+2*a4*lk3) -  4*a1*a2*(2*a3*l_p+2*a4*l_q)) * radial_2e_list_K[tmp](1,0) + (lk1*lk2*4*a3*a4 - l12*4*a3*a4) * radial_2e_list_K[tmp](2,1) - ((2*a1*lk2+2*a2*lk1)*4*a3*a4 - (2*a1*l_q+2*a2*l_p)*4*a3*a4) * radial_2e_list_K[tmp](0,1);
                        }
                        else if(l_p != 0 || l_q != 0)
                        {
                            array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] += ((2*a1*lk2+2*a2*lk1)*(2*a3*lk4+2*a4*lk3) - (2*a1*l_q+2*a2*l_p)*(2*a3*l_p+2*a4*l_q)) * radial_2e_list_K[tmp](0,0) - (4*a1*a2*(2*a3*lk4+2*a4*lk3) - 4*a1*a2*(2*a3*l_p+2*a4*l_q)) * radial_2e_list_K[tmp](1,0) - ((2*a1*lk2+2*a2*lk1)*4*a3*a4 - (2*a1*l_q+2*a2*l_p)*4*a3*a4) * radial_2e_list_K[tmp](0,1);
                        }
                        array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] /= norm_K * 16.0 * pow(speedOfLight,4);
                    }
                    else
                    {
                        cout << "ERROR: Unknown integralTYPE in get_h2e:\n";
                        exit(99);
                    }
                }
            }
        }
        EndTime = clock();
        time_r += (EndTime - StartTime)/(double)CLOCKS_PER_SEC;

        StartTime = clock();
        int l_p_cycle = (l_p == 0) ? 1 : 2, l_q_cycle = (l_q == 0) ? 1 : 2;
        for(int int_tmp2_p = 0; int_tmp2_p < l_p_cycle; int_tmp2_p++)
        for(int int_tmp2_q = 0; int_tmp2_q < l_q_cycle; int_tmp2_q++)
        {
            int add_p = int_tmp2_p*(irrep_list(int_tmp1_p).two_j+1), add_q = int_tmp2_q*(irrep_list(int_tmp1_q).two_j+1);
            int_2e_JK.J(int_tmp1_p+int_tmp2_p, int_tmp1_q+int_tmp2_q).resize(size_gtos_p*size_gtos_p,size_gtos_q*size_gtos_q);
            int_2e_JK.K(int_tmp1_p+int_tmp2_p, int_tmp1_q+int_tmp2_q).resize(size_gtos_p*size_gtos_q,size_gtos_q*size_gtos_p);
            #pragma omp parallel  for
            for(int tt = 0; tt < size_gtos_p*size_gtos_p*size_gtos_q*size_gtos_q; tt++)
            {
                int e1J = tt/(size_gtos_q*size_gtos_q);
                int e2J = tt - e1J*(size_gtos_q*size_gtos_q);
                int e1K = tt/(size_gtos_p*size_gtos_q);
                int e2K = tt - e1K*(size_gtos_p*size_gtos_q);
                int_2e_JK.J(int_tmp1_p+int_tmp2_p, int_tmp1_q+int_tmp2_q)(e1J,e2J) = 0.0;
                int_2e_JK.K(int_tmp1_p+int_tmp2_p, int_tmp1_q+int_tmp2_q)(e1K,e2K) = 0.0;
                for(int tmp = LmaxJ; tmp >= 0; tmp = tmp - 2)
                    int_2e_JK.J(int_tmp1_p+int_tmp2_p, int_tmp1_q+int_tmp2_q)(e1J,e2J) += array_radial_J[tmp][e1J][e2J][int_tmp2_p][int_tmp2_q] * array_angular_J[tmp][int_tmp2_p][int_tmp2_q];
                for(int tmp = LmaxK; tmp >= 0; tmp = tmp - 2)
                    int_2e_JK.K(int_tmp1_p+int_tmp2_p, int_tmp1_q+int_tmp2_q)(e1K,e2K) += array_radial_K[tmp][e1K][e2K][int_tmp2_p][int_tmp2_q] * array_angular_K[tmp][int_tmp2_p][int_tmp2_q];
            }
        }
        EndTime = clock();
        time_c += (EndTime - StartTime)/(double)CLOCKS_PER_SEC;
        int_tmp1_q += (l_q == 0) ? 1 : 2;
    }
    int_tmp1_p += (l_p == 0) ? 1 : 2;
    }

    cout << time_a << "\t" << time_r << "\t" << time_c <<endl;
    return int_2e_JK;
}

/*
    Evaluate all compact 2e integral together for DHF calculations
*/
void INT_SPH::get_h2e_JK_direct(int2eJK& LLLL, int2eJK& SSLL, int2eJK& SSSS, const int& occMaxL, const bool& spinFree)
{
    double time_a = 0.0, time_r = 0.0, time_c = 0.0;
    int occMaxShell = 0, Nirrep_compact = 0;
    if(occMaxL == -1)    occMaxShell = size_shell;
    else
    {
        for(int ii = 0; ii < size_shell; ii++)
        {
            if(shell_list(ii).l <= occMaxL)
                occMaxShell++;
            else
                break;
        }
    }
    for(int ii = 0; ii < occMaxShell; ii++)
    {
        if(shell_list(ii).l == 0) Nirrep_compact += 1;
        else Nirrep_compact += 2;
    }

    LLLL.J.resize(Nirrep_compact, Nirrep_compact);
    LLLL.K.resize(Nirrep_compact, Nirrep_compact);
    SSLL.J.resize(Nirrep_compact, Nirrep_compact);
    SSLL.K.resize(Nirrep_compact, Nirrep_compact);
    SSSS.J.resize(Nirrep_compact, Nirrep_compact);
    SSSS.K.resize(Nirrep_compact, Nirrep_compact);

    int int_tmp1_p = 0;
    for(int pshell = 0; pshell < occMaxShell; pshell++)
    {
    int l_p = shell_list(pshell).l, int_tmp1_q = 0;
    for(int qshell = 0; qshell < occMaxShell; qshell++)
    {
        int l_q = shell_list(qshell).l, l_max = max(l_p,l_q), LmaxJ = min(l_p+l_p, l_q+l_q), LmaxK = l_p+l_q;
        int size_gtos_p = shell_list(pshell).coeff.rows(), size_gtos_q = shell_list(qshell).coeff.rows();
        int size_tmp_p = (l_p == 0) ? 1 : 2, size_tmp_q = (l_q == 0) ? 1 : 2; 
        double array_radial_J_LLLL[LmaxJ+1][size_gtos_p*size_gtos_p][size_gtos_q*size_gtos_q][size_tmp_p][size_tmp_q];
        double array_radial_K_LLLL[LmaxK+1][size_gtos_p*size_gtos_q][size_gtos_q*size_gtos_p][size_tmp_p][size_tmp_q];
        double array_radial_J_SSLL[LmaxJ+1][size_gtos_p*size_gtos_p][size_gtos_q*size_gtos_q][size_tmp_p][size_tmp_q];
        double array_radial_K_SSLL[LmaxK+1][size_gtos_p*size_gtos_q][size_gtos_q*size_gtos_p][size_tmp_p][size_tmp_q];
        double array_radial_J_SSSS[LmaxJ+1][size_gtos_p*size_gtos_p][size_gtos_q*size_gtos_q][size_tmp_p][size_tmp_q];
        double array_radial_K_SSSS[LmaxK+1][size_gtos_p*size_gtos_q][size_gtos_q*size_gtos_p][size_tmp_p][size_tmp_q];
        double array_angular_J[LmaxJ+1][size_tmp_p][size_tmp_q], array_angular_K[LmaxK+1][size_tmp_p][size_tmp_q];

        StartTime = clock();
        #pragma omp parallel  for
        for(int twojj_p = abs(2*l_p-1); twojj_p <= 2*l_p+1; twojj_p = twojj_p + 2)
        for(int twojj_q = abs(2*l_q-1); twojj_q <= 2*l_q+1; twojj_q = twojj_q + 2)
        {
            int sym_ap = twojj_p - 2*l_p, sym_aq = twojj_q - 2*l_q;
            int index_tmp_p = (l_p > 0) ? (1 - (2*l_p+1 - twojj_p)/2) : 0;
            int index_tmp_q = (l_q > 0) ? (1 - (2*l_q+1 - twojj_q)/2) : 0;

            for(int tmp = LmaxJ; tmp >= 0; tmp -= 2)
            {
                double tmp_d = 0.0;
                for(int mq = 0; mq < twojj_q + 1; mq++)
                {
                    tmp_d += int2e_get_angular_J(l_p, twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, tmp);
                }
                tmp_d /= (twojj_q + 1);
                array_angular_J[tmp][index_tmp_p][index_tmp_q] = tmp_d;
            }
            
            for(int tmp = LmaxK; tmp >= 0; tmp -= 2)
            {
                double tmp_d = 0.0;
                for(int mq = 0; mq < twojj_q + 1; mq++)
                {
                    tmp_d += int2e_get_angular_K(l_p, twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, tmp);
                }
                tmp_d /= (twojj_q + 1);
                array_angular_K[tmp][index_tmp_p][index_tmp_q] = tmp_d;
            }
        }
        EndTime = clock();
        time_a += (EndTime - StartTime)/(double)CLOCKS_PER_SEC;

        StartTime = clock();
        #pragma omp parallel  for
        for(int tt = 0; tt < size_gtos_p*size_gtos_p*size_gtos_q*size_gtos_q; tt++)
        {
            int e1J = tt/(size_gtos_q*size_gtos_q);
            int e2J = tt - e1J*(size_gtos_q*size_gtos_q);
            int ii = e1J/size_gtos_p, jj = e1J - ii*size_gtos_p;
            int kk = e2J/size_gtos_q, ll = e2J - kk*size_gtos_q;
            int e1K = ii*size_gtos_q+ll, e2K = kk*size_gtos_p+jj;
            MatrixXd radial_2e_list_J[LmaxJ+1], radial_2e_list_K[LmaxK+1];
            double a_i_J = shell_list(pshell).exp_a(ii), a_j_J = shell_list(pshell).exp_a(jj), a_k_J = shell_list(qshell).exp_a(kk), a_l_J = shell_list(qshell).exp_a(ll);
            double a_i_K = shell_list(pshell).exp_a(ii), a_j_K = shell_list(qshell).exp_a(ll), a_k_K = shell_list(qshell).exp_a(kk), a_l_K = shell_list(pshell).exp_a(jj);

            for(int LL = LmaxJ; LL >= 0; LL -= 2)
            {
                radial_2e_list_J[LL].resize(3,3);
                radial_2e_list_J[LL](0,0) = int2e_get_radial(l_p,a_i_J,l_p,a_j_J,l_q,a_k_J,l_q,a_l_J,LL);
                radial_2e_list_J[LL](1,0) = int2e_get_radial(l_p+1,a_i_J,l_p+1,a_j_J,l_q,a_k_J,l_q,a_l_J,LL);
                radial_2e_list_J[LL](0,1) = int2e_get_radial(l_p,a_i_J,l_p,a_j_J,l_q+1,a_k_J,l_q+1,a_l_J,LL);
                radial_2e_list_J[LL](1,1) = int2e_get_radial(l_p+1,a_i_J,l_p+1,a_j_J,l_q+1,a_k_J,l_q+1,a_l_J,LL);
                if(l_p != 0)
                {
                    radial_2e_list_J[LL](2,0) = int2e_get_radial(l_p-1,a_i_J,l_p-1,a_j_J,l_q,a_k_J,l_q,a_l_J,LL);
                    radial_2e_list_J[LL](2,1) = int2e_get_radial(l_p-1,a_i_J,l_p-1,a_j_J,l_q+1,a_k_J,l_q+1,a_l_J,LL);
                }
                if(l_q != 0)
                {
                    radial_2e_list_J[LL](0,2) = int2e_get_radial(l_p,a_i_J,l_p,a_j_J,l_q-1,a_k_J,l_q-1,a_l_J,LL);
                    radial_2e_list_J[LL](1,2) = int2e_get_radial(l_p+1,a_i_J,l_p+1,a_j_J,l_q-1,a_k_J,l_q-1,a_l_J,LL);
                }
                if(l_p!=0 && l_q!=0)
                    radial_2e_list_J[LL](2,2) = int2e_get_radial(l_p-1,a_i_J,l_p-1,a_j_J,l_q-1,a_k_J,l_q-1,a_l_J,LL);
            }
            for(int LL = LmaxK; LL >= 0; LL -= 2)
            {
                radial_2e_list_K[LL].resize(3,3);
                radial_2e_list_K[LL](0,0) = int2e_get_radial(l_p,a_i_K,l_q,a_j_K,l_q,a_k_K,l_p,a_l_K,LL);
                radial_2e_list_K[LL](1,0) = int2e_get_radial(l_p+1,a_i_K,l_q+1,a_j_K,l_q,a_k_K,l_p,a_l_K,LL);
                radial_2e_list_K[LL](0,1) = int2e_get_radial(l_p,a_i_K,l_q,a_j_K,l_q+1,a_k_K,l_p+1,a_l_K,LL);
                radial_2e_list_K[LL](1,1) = int2e_get_radial(l_p+1,a_i_K,l_q+1,a_j_K,l_q+1,a_k_K,l_p+1,a_l_K,LL);
                if(l_p != 0 && l_q != 0)
                {
                    radial_2e_list_K[LL](2,0) = int2e_get_radial(l_p-1,a_i_K,l_q-1,a_j_K,l_q,a_k_K,l_p,a_l_K,LL);
                    radial_2e_list_K[LL](2,1) = int2e_get_radial(l_p-1,a_i_K,l_q-1,a_j_K,l_q+1,a_k_K,l_p+1,a_l_K,LL);
                    radial_2e_list_K[LL](0,2) = int2e_get_radial(l_p,a_i_K,l_q,a_j_K,l_q-1,a_k_K,l_p-1,a_l_K,LL);
                    radial_2e_list_K[LL](1,2) = int2e_get_radial(l_p+1,a_i_K,l_q+1,a_j_K,l_q-1,a_k_K,l_p-1,a_l_K,LL);
                    radial_2e_list_K[LL](2,2) = int2e_get_radial(l_p-1,a_i_K,l_q-1,a_j_K,l_q-1,a_k_K,l_p-1,a_l_K,LL);
                }
            }

            for(int twojj_p = abs(2*l_p-1); twojj_p <= 2*l_p+1; twojj_p = twojj_p + 2)
            for(int twojj_q = abs(2*l_q-1); twojj_q <= 2*l_q+1; twojj_q = twojj_q + 2)
            {
                int index_tmp_p = (l_p > 0) ? (1 - (2*l_p+1 - twojj_p)/2) : 0;
                int index_tmp_q = (l_q > 0) ? (1 - (2*l_q+1 - twojj_q)/2) : 0;
                int sym_ap = twojj_p - 2*l_p, sym_aq = twojj_q - 2*l_q;
                double k_p = -(twojj_p+1.0)*sym_ap/2.0, k_q = -(twojj_q+1.0)*sym_aq/2.0;
                double norm_J = shell_list(pshell).norm(ii) * shell_list(pshell).norm(jj) * shell_list(qshell).norm(kk) * shell_list(qshell).norm(ll), norm_K = shell_list(pshell).norm(ii) * shell_list(qshell).norm(ll) * shell_list(qshell).norm(kk) * shell_list(pshell).norm(jj);
                double lk1 = 1+l_p+k_p, lk2 = 1+l_p+k_p, lk3 = 1+l_q+k_q, lk4 = 1+l_q+k_q, a1 = shell_list(pshell).exp_a(ii), a2 = shell_list(pshell).exp_a(jj), a3 = shell_list(qshell).exp_a(kk), a4 = shell_list(qshell).exp_a(ll);

                for(int tmp = LmaxJ; tmp >= 0; tmp -= 2)
                {
                    array_radial_J_LLLL[tmp][e1J][e2J][index_tmp_p][index_tmp_q] = radial_2e_list_J[tmp](0,0) / norm_J;
                    array_radial_J_SSLL[tmp][e1J][e2J][index_tmp_p][index_tmp_q] = 4.0*a1*a2 * radial_2e_list_J[tmp](1,0);
                    array_radial_J_SSSS[tmp][e1J][e2J][index_tmp_p][index_tmp_q] = 4*a1*a2*4*a3*a4 * radial_2e_list_J[tmp](1,1);
                    if(spinFree)
                    {
                        double l12 = l_p*l_p + l_p*(l_p+1)/2 + l_p*(l_p+1)/2 - tmp*(tmp+1)/2, l34 = l_q*l_q + l_q*(l_q+1)/2 + l_q*(l_q+1)/2 - tmp*(tmp+1)/2;
                        if(l_p != 0)
                        {
                            array_radial_J_SSLL[tmp][e1J][e2J][index_tmp_p][index_tmp_q] += l12 * radial_2e_list_J[tmp](2,0) - (2.0*a1*l_p+2.0*a2*l_p) * radial_2e_list_J[tmp](0,0);
                            if(l_q != 0)
                                array_radial_J_SSSS[tmp][e1J][e2J][index_tmp_p][index_tmp_q] += l12*l34 * radial_2e_list_J[tmp](2,2) - (2*a1*l_p+2*a2*l_p)*l34 * radial_2e_list_J[tmp](0,2) + 4*a1*a2*l34 * radial_2e_list_J[tmp](1,2) - l12*(2*a3*l_q+2*a4*l_q) * radial_2e_list_J[tmp](2,0) + (2*a1*l_p+2*a2*l_p)*(2*a3*l_q+2*a4*l_q) * radial_2e_list_J[tmp](0,0) - 4*a1*a2*(2*a3*l_q+2*a4*l_q) * radial_2e_list_J[tmp](1,0) + l12*4*a3*a4 * radial_2e_list_J[tmp](2,1) - (2*a1*l_p+2*a2*l_p)*4*a3*a4 * radial_2e_list_J[tmp](0,1);
                            else
                                array_radial_J_SSSS[tmp][e1J][e2J][index_tmp_p][index_tmp_q] += l12*4*a3*a4 * radial_2e_list_J[tmp](2,1) - (2*a1*l_p+2*a2*l_p)*4*a3*a4 * radial_2e_list_J[tmp](0,1);
                        }
                        else
                        {
                            if(l_q != 0)
                                array_radial_J_SSSS[tmp][e1J][e2J][index_tmp_p][index_tmp_q] += 4*a1*a2*l34 * radial_2e_list_J[tmp](1,2) - 4*a1*a2*(2*a3*l_q+2*a4*l_q) * radial_2e_list_J[tmp](1,0);
                        }
                    }
                    else
                    {    
                        if(l_p != 0)
                        {
                            array_radial_J_SSLL[tmp][e1J][e2J][index_tmp_p][index_tmp_q] += lk1*lk2 * radial_2e_list_J[tmp](2,0) - (2.0*a1*lk2+2.0*a2*lk1) * radial_2e_list_J[tmp](0,0);
                            if(l_q != 0)
                                array_radial_J_SSSS[tmp][e1J][e2J][index_tmp_p][index_tmp_q] += lk1*lk2*lk3*lk4 * radial_2e_list_J[tmp](2,2) - (2*a1*lk2+2*a2*lk1)*lk3*lk4 * radial_2e_list_J[tmp](0,2) + 4*a1*a2*lk3*lk4 * radial_2e_list_J[tmp](1,2) - lk1*lk2*(2*a3*lk4+2*a4*lk3) * radial_2e_list_J[tmp](2,0) + (2*a1*lk2+2*a2*lk1)*(2*a3*lk4+2*a4*lk3) * radial_2e_list_J[tmp](0,0) - 4*a1*a2*(2*a3*lk4+2*a4*lk3) * radial_2e_list_J[tmp](1,0) + lk1*lk2*4*a3*a4 * radial_2e_list_J[tmp](2,1) - (2*a1*lk2+2*a2*lk1)*4*a3*a4 * radial_2e_list_J[tmp](0,1);
                            else
                                array_radial_J_SSSS[tmp][e1J][e2J][index_tmp_p][index_tmp_q] += lk1*lk2*4*a3*a4 * radial_2e_list_J[tmp](2,1) - (2*a1*lk2+2*a2*lk1)*4*a3*a4 * radial_2e_list_J[tmp](0,1);
                        }
                        else
                        {
                            if(l_q != 0)
                                array_radial_J_SSSS[tmp][e1J][e2J][index_tmp_p][index_tmp_q] += 4*a1*a2*lk3*lk4 * radial_2e_list_J[tmp](1,2) - 4*a1*a2*(2*a3*lk4+2*a4*lk3) * radial_2e_list_J[tmp](1,0);
                        }
                    }
                    array_radial_J_SSLL[tmp][e1J][e2J][index_tmp_p][index_tmp_q] /= norm_J*4.0*pow(speedOfLight,2);
                    array_radial_J_SSSS[tmp][e1J][e2J][index_tmp_p][index_tmp_q] /= norm_J*16.0*pow(speedOfLight,4);
                }
                lk2 = 1+l_q+k_q; lk4 = 1+l_p+k_p; 
                a2 = shell_list(qshell).exp_a(ll); a4 = shell_list(pshell).exp_a(jj);
                for(int tmp = LmaxK; tmp >= 0; tmp -= 2)
                {
                    array_radial_K_LLLL[tmp][e1K][e2K][index_tmp_p][index_tmp_q] = radial_2e_list_K[tmp](0,0) / norm_K;
                    array_radial_K_SSLL[tmp][e1K][e2K][index_tmp_p][index_tmp_q] = 4.0*a1*a2 * radial_2e_list_K[tmp](1,0);
                    array_radial_K_SSSS[tmp][e1K][e2K][index_tmp_p][index_tmp_q] = 4*a1*a2*4*a3*a4 * radial_2e_list_K[tmp](1,1);
                    if(spinFree)
                    {
                        double l12 = l_p*l_q + l_p*(l_p+1)/2 + l_q*(l_q+1)/2 - tmp*(tmp+1)/2, l34 = l_q*l_p + l_q*(l_q+1)/2 + l_p*(l_p+1)/2 - tmp*(tmp+1)/2;
                        if(l_p != 0 && l_q != 0)
                        {
                            array_radial_K_SSLL[tmp][e1K][e2K][index_tmp_p][index_tmp_q] += l12 * radial_2e_list_K[tmp](2,0) - (2.0*a1*l_q+2.0*a2*l_p) * radial_2e_list_K[tmp](0,0);
                            array_radial_K_SSSS[tmp][e1K][e2K][index_tmp_p][index_tmp_q] += l12*l34 * radial_2e_list_K[tmp](2,2) - (2*a1*l_q+2*a2*l_p)*l34 * radial_2e_list_K[tmp](0,2) + 4*a1*a2*l34 * radial_2e_list_K[tmp](1,2) - l12*(2*a3*l_p+2*a4*l_q) * radial_2e_list_K[tmp](2,0) + (2*a1*l_q+2*a2*l_p)*(2*a3*l_p+2*a4*l_q) * radial_2e_list_K[tmp](0,0) - 4*a1*a2*(2*a3*l_p+2*a4*l_q) * radial_2e_list_K[tmp](1,0) + l12*4*a3*a4 * radial_2e_list_K[tmp](2,1) - (2*a1*l_q+2*a2*l_p)*4*a3*a4 * radial_2e_list_K[tmp](0,1);
                        }
                        else if(l_p != 0 || l_q != 0)
                        {
                            array_radial_K_SSLL[tmp][e1K][e2K][index_tmp_p][index_tmp_q] += - (2.0*a1*l_q+2.0*a2*l_p) * radial_2e_list_K[tmp](0,0);
                            array_radial_K_SSSS[tmp][e1K][e2K][index_tmp_p][index_tmp_q] += (2*a1*l_q+2*a2*l_p)*(2*a3*l_p+2*a4*l_q) * radial_2e_list_K[tmp](0,0) - 4*a1*a2*(2*a3*l_p+2*a4*l_q) * radial_2e_list_K[tmp](1,0) - (2*a1*l_q+2*a2*l_p)*4*a3*a4 * radial_2e_list_K[tmp](0,1);
                        }
                    }
                    else
                    {
                        if(l_p != 0 && l_q != 0)
                        {
                            array_radial_K_SSLL[tmp][e1K][e2K][index_tmp_p][index_tmp_q] += lk1*lk2 * radial_2e_list_K[tmp](2,0) - (2.0*a1*lk2+2.0*a2*lk1) * radial_2e_list_K[tmp](0,0);
                            array_radial_K_SSSS[tmp][e1K][e2K][index_tmp_p][index_tmp_q] += lk1*lk2*lk3*lk4 * radial_2e_list_K[tmp](2,2) - (2*a1*lk2+2*a2*lk1)*lk3*lk4 * radial_2e_list_K[tmp](0,2) + 4*a1*a2*lk3*lk4 * radial_2e_list_K[tmp](1,2) - lk1*lk2*(2*a3*lk4+2*a4*lk3) * radial_2e_list_K[tmp](2,0) + (2*a1*lk2+2*a2*lk1)*(2*a3*lk4+2*a4*lk3) * radial_2e_list_K[tmp](0,0) - 4*a1*a2*(2*a3*lk4+2*a4*lk3) * radial_2e_list_K[tmp](1,0) + lk1*lk2*4*a3*a4 * radial_2e_list_K[tmp](2,1) - (2*a1*lk2+2*a2*lk1)*4*a3*a4 * radial_2e_list_K[tmp](0,1);
                        }
                        else if(l_p != 0 || l_q != 0)
                        {
                            array_radial_K_SSLL[tmp][e1K][e2K][index_tmp_p][index_tmp_q] += - (2.0*a1*lk2+2.0*a2*lk1) * radial_2e_list_K[tmp](0,0);
                            array_radial_K_SSSS[tmp][e1K][e2K][index_tmp_p][index_tmp_q] += (2*a1*lk2+2*a2*lk1)*(2*a3*lk4+2*a4*lk3) * radial_2e_list_K[tmp](0,0) - 4*a1*a2*(2*a3*lk4+2*a4*lk3) * radial_2e_list_K[tmp](1,0) - (2*a1*lk2+2*a2*lk1)*4*a3*a4 * radial_2e_list_K[tmp](0,1);
                        }
                    }
                    array_radial_K_SSLL[tmp][e1K][e2K][index_tmp_p][index_tmp_q] /= norm_K*4.0*pow(speedOfLight,2);
                    array_radial_K_SSSS[tmp][e1K][e2K][index_tmp_p][index_tmp_q] /= norm_K*16.0*pow(speedOfLight,4);
                }
            }
        }
        EndTime = clock();
        time_r += (EndTime - StartTime)/(double)CLOCKS_PER_SEC;

        StartTime = clock();
        int l_p_cycle = (l_p == 0) ? 1 : 2, l_q_cycle = (l_q == 0) ? 1 : 2;
        for(int int_tmp2_p = 0; int_tmp2_p < l_p_cycle; int_tmp2_p++)
        for(int int_tmp2_q = 0; int_tmp2_q < l_q_cycle; int_tmp2_q++)
        {
            LLLL.J(int_tmp1_p+int_tmp2_p, int_tmp1_q+int_tmp2_q).resize(size_gtos_p*size_gtos_p,size_gtos_q*size_gtos_q);
            LLLL.K(int_tmp1_p+int_tmp2_p, int_tmp1_q+int_tmp2_q).resize(size_gtos_p*size_gtos_q,size_gtos_q*size_gtos_p);
            SSLL.J(int_tmp1_p+int_tmp2_p, int_tmp1_q+int_tmp2_q).resize(size_gtos_p*size_gtos_p,size_gtos_q*size_gtos_q);
            SSLL.K(int_tmp1_p+int_tmp2_p, int_tmp1_q+int_tmp2_q).resize(size_gtos_p*size_gtos_q,size_gtos_q*size_gtos_p);
            SSSS.J(int_tmp1_p+int_tmp2_p, int_tmp1_q+int_tmp2_q).resize(size_gtos_p*size_gtos_p,size_gtos_q*size_gtos_q);
            SSSS.K(int_tmp1_p+int_tmp2_p, int_tmp1_q+int_tmp2_q).resize(size_gtos_p*size_gtos_q,size_gtos_q*size_gtos_p);
            #pragma omp parallel  for
            for(int tt = 0; tt < size_gtos_p*size_gtos_p*size_gtos_q*size_gtos_q; tt++)
            {
                int e1J = tt/(size_gtos_q*size_gtos_q);
                int e2J = tt - e1J*(size_gtos_q*size_gtos_q);
                int e1K = tt/(size_gtos_p*size_gtos_q);
                int e2K = tt - e1K*(size_gtos_p*size_gtos_q);
                LLLL.J(int_tmp1_p+int_tmp2_p, int_tmp1_q+int_tmp2_q)(e1J,e2J) = 0.0;
                LLLL.K(int_tmp1_p+int_tmp2_p, int_tmp1_q+int_tmp2_q)(e1K,e2K) = 0.0;
                SSLL.J(int_tmp1_p+int_tmp2_p, int_tmp1_q+int_tmp2_q)(e1J,e2J) = 0.0;
                SSLL.K(int_tmp1_p+int_tmp2_p, int_tmp1_q+int_tmp2_q)(e1K,e2K) = 0.0;
                SSSS.J(int_tmp1_p+int_tmp2_p, int_tmp1_q+int_tmp2_q)(e1J,e2J) = 0.0;
                SSSS.K(int_tmp1_p+int_tmp2_p, int_tmp1_q+int_tmp2_q)(e1K,e2K) = 0.0;
                for(int tmp = LmaxJ; tmp >= 0; tmp = tmp - 2)
                {
                    LLLL.J(int_tmp1_p+int_tmp2_p, int_tmp1_q+int_tmp2_q)(e1J,e2J) += array_radial_J_LLLL[tmp][e1J][e2J][int_tmp2_p][int_tmp2_q] * array_angular_J[tmp][int_tmp2_p][int_tmp2_q];
                    SSLL.J(int_tmp1_p+int_tmp2_p, int_tmp1_q+int_tmp2_q)(e1J,e2J) += array_radial_J_SSLL[tmp][e1J][e2J][int_tmp2_p][int_tmp2_q] * array_angular_J[tmp][int_tmp2_p][int_tmp2_q];
                    SSSS.J(int_tmp1_p+int_tmp2_p, int_tmp1_q+int_tmp2_q)(e1J,e2J) += array_radial_J_SSSS[tmp][e1J][e2J][int_tmp2_p][int_tmp2_q] * array_angular_J[tmp][int_tmp2_p][int_tmp2_q];
                }
                for(int tmp = LmaxK; tmp >= 0; tmp = tmp - 2)
                {
                    LLLL.K(int_tmp1_p+int_tmp2_p, int_tmp1_q+int_tmp2_q)(e1K,e2K) += array_radial_K_LLLL[tmp][e1K][e2K][int_tmp2_p][int_tmp2_q] * array_angular_K[tmp][int_tmp2_p][int_tmp2_q];
                    SSLL.K(int_tmp1_p+int_tmp2_p, int_tmp1_q+int_tmp2_q)(e1K,e2K) += array_radial_K_SSLL[tmp][e1K][e2K][int_tmp2_p][int_tmp2_q] * array_angular_K[tmp][int_tmp2_p][int_tmp2_q];
                    SSSS.K(int_tmp1_p+int_tmp2_p, int_tmp1_q+int_tmp2_q)(e1K,e2K) += array_radial_K_SSSS[tmp][e1K][e2K][int_tmp2_p][int_tmp2_q] * array_angular_K[tmp][int_tmp2_p][int_tmp2_q];
                }
            }
        }
        EndTime = clock();
        time_c += (EndTime - StartTime)/(double)CLOCKS_PER_SEC;
        int_tmp1_q += (l_q == 0) ? 1 : 2;
    }
    int_tmp1_p += (l_p == 0) ? 1 : 2;
    }

    cout << time_a << "\t" << time_r << "\t" << time_c <<endl;
    return ;
}

void INT_SPH::get_h2eSD_JK_direct(int2eJK& SSLL, int2eJK& SSSS, const int& occMaxL)
{
    SSLL = get_h2e_JK("SSLL_SD", occMaxL);
    SSSS = get_h2e_JK("SSSS_SD", occMaxL);
    // int occMaxShell = 0;
    // if(occMaxL == -1)    occMaxShell = size_shell;
    // else
    // {
    //     for(int ii = 0; ii < size_shell; ii++)
    //     {
    //         if(shell_list(ii).l <= occMaxL)
    //             occMaxShell++;
    //         else
    //             break;
    //     }
    // }

    // LLLL.J.resize(Nirrep, Nirrep);
    // LLLL.K.resize(Nirrep, Nirrep);
    // SSLL.J.resize(Nirrep, Nirrep);
    // SSLL.K.resize(Nirrep, Nirrep);
    // SSSS.J.resize(Nirrep, Nirrep);
    // SSSS.K.resize(Nirrep, Nirrep);

    // for(int ii = 0; ii < Nirrep; ii++)
    // for(int jj = 0; jj < Nirrep; jj++)
    // {
    //     LLLL.J(ii,jj).resize(irrep_list(ii).size*irrep_list(ii).size, irrep_list(jj).size*irrep_list(jj).size);
    //     LLLL.K(ii,jj).resize(irrep_list(ii).size*irrep_list(ii).size, irrep_list(jj).size*irrep_list(jj).size);
    //     SSLL.J(ii,jj).resize(irrep_list(ii).size*irrep_list(ii).size, irrep_list(jj).size*irrep_list(jj).size);
    //     SSLL.K(ii,jj).resize(irrep_list(ii).size*irrep_list(ii).size, irrep_list(jj).size*irrep_list(jj).size);
    //     SSSS.J(ii,jj).resize(irrep_list(ii).size*irrep_list(ii).size, irrep_list(jj).size*irrep_list(jj).size);
    //     SSSS.K(ii,jj).resize(irrep_list(ii).size*irrep_list(ii).size, irrep_list(jj).size*irrep_list(jj).size);
    // }

    // int int_tmp1_p = 0;
    // for(int pshell = 0; pshell < occMaxShell; pshell++)
    // {
    // int l_p = shell_list(pshell).l, int_tmp1_q = 0;
    // for(int qshell = 0; qshell < occMaxShell; qshell++)
    // {
    //     int l_q = shell_list(qshell).l, l_max = max(l_p,l_q), LmaxJ = min(l_p+l_p, l_q+l_q), LmaxK = l_p+l_q;
    //     int size_gtos_p = shell_list(pshell).coeff.rows(), size_gtos_q = shell_list(qshell).coeff.rows();
    //     MatrixXd radial_2e_list_J[LmaxJ+1], radial_2e_list_K[LmaxK+1];
    //     double array_radial_J_LLLL[LmaxJ+1][size_gtos_p][size_gtos_p][size_gtos_q][size_gtos_q];
    //     double array_radial_K_LLLL[LmaxK+1][size_gtos_p][size_gtos_q][size_gtos_q][size_gtos_p];
    //     double array_radial_J_SSLL[LmaxJ+1][size_gtos_p][size_gtos_p][size_gtos_q][size_gtos_q];
    //     double array_radial_K_SSLL[LmaxK+1][size_gtos_p][size_gtos_q][size_gtos_q][size_gtos_p];
    //     double array_radial_J_SSSS[LmaxJ+1][size_gtos_p][size_gtos_p][size_gtos_q][size_gtos_q];
    //     double array_radial_K_SSSS[LmaxK+1][size_gtos_p][size_gtos_q][size_gtos_q][size_gtos_p];

    //     Matrix<mMatrixXd,-1,-1> h2eJ_LLLL, h2eK_LLLL, h2eJ_SSLL, h2eK_SSLL, h2eJ_SSSS, h2eK_SSSS;
    //     int size_tmp_p = 0, size_tmp_q = 0;
    //     if(l_p == 0)
    //         size_tmp_p = 1;
    //     else
    //         size_tmp_p = 2;
    //     if(l_q == 0)
    //         size_tmp_q = 1;
    //     else
    //         size_tmp_q = 2;
    //     h2eJ_LLLL.resize(size_tmp_p,size_tmp_q);
    //     h2eK_LLLL.resize(size_tmp_p,size_tmp_q);
    //     h2eJ_SSLL.resize(size_tmp_p,size_tmp_q);
    //     h2eK_SSLL.resize(size_tmp_p,size_tmp_q);
    //     h2eJ_SSSS.resize(size_tmp_p,size_tmp_q);
    //     h2eK_SSSS.resize(size_tmp_p,size_tmp_q);
 
    //     MatrixXd array_angular_J[LmaxJ+1][size_tmp_p][size_tmp_q], array_angular_K[LmaxK+1][size_tmp_p][size_tmp_q];

    //     for(int twojj_p = abs(2*l_p-1); twojj_p <= 2*l_p+1; twojj_p = twojj_p + 2)
    //     for(int twojj_q = abs(2*l_q-1); twojj_q <= 2*l_q+1; twojj_q = twojj_q + 2)
    //     {
    //         int sym_ap = twojj_p - 2*l_p, sym_aq = twojj_q - 2*l_q;
    //         int index_tmp_p = 1 - (2*l_p+1 - twojj_p)/2;
    //         if(l_p == 0) index_tmp_p = 0;
    //         int index_tmp_q = 1 - (2*l_q+1 - twojj_q)/2;
    //         if(l_q == 0) index_tmp_q = 0;

    //         h2eJ_LLLL(index_tmp_p,index_tmp_q).resize(twojj_p+1,twojj_q+1);
    //         h2eK_LLLL(index_tmp_p,index_tmp_q).resize(twojj_p+1,twojj_q+1);
    //         h2eJ_SSLL(index_tmp_p,index_tmp_q).resize(twojj_p+1,twojj_q+1);
    //         h2eK_SSLL(index_tmp_p,index_tmp_q).resize(twojj_p+1,twojj_q+1);
    //         h2eJ_SSSS(index_tmp_p,index_tmp_q).resize(twojj_p+1,twojj_q+1);
    //         h2eK_SSSS(index_tmp_p,index_tmp_q).resize(twojj_p+1,twojj_q+1);
    //         for(int mp = 0; mp < twojj_p + 1; mp++)
    //         for(int mq = 0; mq < twojj_q + 1; mq++)
    //         {
    //             h2eJ_LLLL(index_tmp_p,index_tmp_q)(mp,mq).resize(size_gtos_p*size_gtos_p,size_gtos_q*size_gtos_q);
    //             h2eK_LLLL(index_tmp_p,index_tmp_q)(mp,mq).resize(size_gtos_p*size_gtos_q,size_gtos_q*size_gtos_p);
    //             h2eJ_SSLL(index_tmp_p,index_tmp_q)(mp,mq).resize(size_gtos_p*size_gtos_p,size_gtos_q*size_gtos_q);
    //             h2eK_SSLL(index_tmp_p,index_tmp_q)(mp,mq).resize(size_gtos_p*size_gtos_q,size_gtos_q*size_gtos_p);
    //             h2eJ_SSSS(index_tmp_p,index_tmp_q)(mp,mq).resize(size_gtos_p*size_gtos_p,size_gtos_q*size_gtos_q);
    //             h2eK_SSSS(index_tmp_p,index_tmp_q)(mp,mq).resize(size_gtos_p*size_gtos_q,size_gtos_q*size_gtos_p);
    //         }

    //         for(int tmp = LmaxJ; tmp >= 0; tmp -= 2)
    //         {
    //             array_angular_J[tmp][index_tmp_p][index_tmp_q].resize(twojj_p + 1,twojj_q + 1);
    //             for(int mp = 0; mp < twojj_p + 1; mp++)
    //             for(int mq = 0; mq < twojj_q + 1; mq++)
    //                 array_angular_J[tmp][index_tmp_p][index_tmp_q](mp,mq) = int2e_get_angular(l_p, 2*mp-twojj_p, sym_ap, l_p, 2*mp-twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, tmp);
    //         }
    //         for(int tmp = LmaxK; tmp >= 0; tmp -= 2)
    //         {
    //             array_angular_K[tmp][index_tmp_p][index_tmp_q].resize(twojj_p + 1,twojj_q + 1);
    //             for(int mp = 0; mp < twojj_p + 1; mp++)
    //             for(int mq = 0; mq < twojj_q + 1; mq++)
    //                 array_angular_K[tmp][index_tmp_p][index_tmp_q](mp,mq) = int2e_get_angular(l_p, 2*mp-twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, l_p, 2*mp-twojj_p, sym_ap, tmp);
    //         }
    //     }

    //     for(int ii = 0; ii < size_gtos_p; ii++)
    //     for(int jj = 0; jj < size_gtos_p; jj++)
    //     for(int kk = 0; kk < size_gtos_q; kk++)
    //     for(int ll = 0; ll < size_gtos_q; ll++)
    //     {
    //         double a_i_J = shell_list(pshell).exp_a(ii), a_j_J = shell_list(pshell).exp_a(jj), a_k_J = shell_list(qshell).exp_a(kk), a_l_J = shell_list(qshell).exp_a(ll);
    //         double a_i_K = shell_list(pshell).exp_a(ii), a_j_K = shell_list(qshell).exp_a(ll), a_k_K = shell_list(qshell).exp_a(kk), a_l_K = shell_list(pshell).exp_a(jj);

    //         for(int LL = LmaxJ; LL >= 0; LL -= 2)
    //         {
    //             radial_2e_list_J[LL].resize(3,3);
    //             radial_2e_list_J[LL](0,0) = int2e_get_radial(l_p,a_i_J,l_p,a_j_J,l_q,a_k_J,l_q,a_l_J,LL);
    //             radial_2e_list_J[LL](1,0) = int2e_get_radial(l_p+1,a_i_J,l_p+1,a_j_J,l_q,a_k_J,l_q,a_l_J,LL);
    //             radial_2e_list_J[LL](0,1) = int2e_get_radial(l_p,a_i_J,l_p,a_j_J,l_q+1,a_k_J,l_q+1,a_l_J,LL);
    //             radial_2e_list_J[LL](1,1) = int2e_get_radial(l_p+1,a_i_J,l_p+1,a_j_J,l_q+1,a_k_J,l_q+1,a_l_J,LL);
    //             if(l_p != 0)
    //             {
    //                 radial_2e_list_J[LL](2,0) = int2e_get_radial(l_p-1,a_i_J,l_p-1,a_j_J,l_q,a_k_J,l_q,a_l_J,LL);
    //                 radial_2e_list_J[LL](2,1) = int2e_get_radial(l_p-1,a_i_J,l_p-1,a_j_J,l_q+1,a_k_J,l_q+1,a_l_J,LL);
    //             }
    //             if(l_q != 0)
    //             {
    //                 radial_2e_list_J[LL](0,2) = int2e_get_radial(l_p,a_i_J,l_p,a_j_J,l_q-1,a_k_J,l_q-1,a_l_J,LL);
    //                 radial_2e_list_J[LL](1,2) = int2e_get_radial(l_p+1,a_i_J,l_p+1,a_j_J,l_q-1,a_k_J,l_q-1,a_l_J,LL);
    //             }
    //             if(l_p!=0 && l_q!=0)
    //                 radial_2e_list_J[LL](2,2) = int2e_get_radial(l_p-1,a_i_J,l_p-1,a_j_J,l_q-1,a_k_J,l_q-1,a_l_J,LL);
    //         }
    //         for(int LL = LmaxK; LL >= 0; LL -= 2)
    //         {
    //             radial_2e_list_K[LL].resize(3,3);
    //             radial_2e_list_K[LL](0,0) = int2e_get_radial(l_p,a_i_K,l_q,a_j_K,l_q,a_k_K,l_p,a_l_K,LL);
    //             radial_2e_list_K[LL](1,0) = int2e_get_radial(l_p+1,a_i_K,l_q+1,a_j_K,l_q,a_k_K,l_p,a_l_K,LL);
    //             radial_2e_list_K[LL](0,1) = int2e_get_radial(l_p,a_i_K,l_q,a_j_K,l_q+1,a_k_K,l_p+1,a_l_K,LL);
    //             radial_2e_list_K[LL](1,1) = int2e_get_radial(l_p+1,a_i_K,l_q+1,a_j_K,l_q+1,a_k_K,l_p+1,a_l_K,LL);
    //             if(l_p != 0 && l_q != 0)
    //             {
    //                 radial_2e_list_K[LL](2,0) = int2e_get_radial(l_p-1,a_i_K,l_q-1,a_j_K,l_q,a_k_K,l_p,a_l_K,LL);
    //                 radial_2e_list_K[LL](2,1) = int2e_get_radial(l_p-1,a_i_K,l_q-1,a_j_K,l_q+1,a_k_K,l_p+1,a_l_K,LL);
    //                 radial_2e_list_K[LL](0,2) = int2e_get_radial(l_p,a_i_K,l_q,a_j_K,l_q-1,a_k_K,l_p-1,a_l_K,LL);
    //                 radial_2e_list_K[LL](1,2) = int2e_get_radial(l_p+1,a_i_K,l_q+1,a_j_K,l_q-1,a_k_K,l_p-1,a_l_K,LL);
    //                 radial_2e_list_K[LL](2,2) = int2e_get_radial(l_p-1,a_i_K,l_q-1,a_j_K,l_q-1,a_k_K,l_p-1,a_l_K,LL);
    //             }
    //         }

    //         for(int twojj_p = abs(2*l_p-1); twojj_p <= 2*l_p+1; twojj_p = twojj_p + 2)
    //         for(int twojj_q = abs(2*l_q-1); twojj_q <= 2*l_q+1; twojj_q = twojj_q + 2)
    //         {
    //             int sym_ap = twojj_p - 2*l_p, sym_aq = twojj_q - 2*l_q;
    //             double k_p = -(twojj_p+1.0)*sym_ap/2.0, k_q = -(twojj_q+1.0)*sym_aq/2.0;
    //             double norm_J = shell_list(pshell).norm(ii) * shell_list(pshell).norm(jj) * shell_list(qshell).norm(kk) * shell_list(qshell).norm(ll), norm_K = shell_list(pshell).norm(ii) * shell_list(qshell).norm(ll) * shell_list(qshell).norm(kk) * shell_list(pshell).norm(jj);
    //             double lk1 = 1+l_p+k_p, lk2 = 1+l_p+k_p, lk3 = 1+l_q+k_q, lk4 = 1+l_q+k_q, a1 = shell_list(pshell).exp_a(ii), a2 = shell_list(pshell).exp_a(jj), a3 = shell_list(qshell).exp_a(kk), a4 = shell_list(qshell).exp_a(ll);

    //             for(int tmp = LmaxJ; tmp >= 0; tmp -= 2)
    //             {
    //                 array_radial_J_LLLL[tmp][ii][jj][kk][ll] = radial_2e_list_J[tmp](0,0) / norm_J;
    //                 array_radial_J_SSLL[tmp][ii][jj][kk][ll] = 4.0*a1*a2 * radial_2e_list_J[tmp](1,0);
    //                 array_radial_J_SSSS[tmp][ii][jj][kk][ll] = 4*a1*a2*4*a3*a4 * radial_2e_list_J[tmp](1,1);
    //                 if(l_p != 0)
    //                 {
    //                     array_radial_J_SSLL[tmp][ii][jj][kk][ll] += lk1*lk2 * radial_2e_list_J[tmp](2,0) - (2.0*a1*lk2+2.0*a2*lk1) * radial_2e_list_J[tmp](0,0);
    //                     if(l_q != 0)
    //                         array_radial_J_SSSS[tmp][ii][jj][kk][ll] += lk1*lk2*lk3*lk4 * radial_2e_list_J[tmp](2,2) - (2*a1*lk2+2*a2*lk1)*lk3*lk4 * radial_2e_list_J[tmp](0,2) + 4*a1*a2*lk3*lk4 * radial_2e_list_J[tmp](1,2) - lk1*lk2*(2*a3*lk4+2*a4*lk3) * radial_2e_list_J[tmp](2,0) + (2*a1*lk2+2*a2*lk1)*(2*a3*lk4+2*a4*lk3) * radial_2e_list_J[tmp](0,0) - 4*a1*a2*(2*a3*lk4+2*a4*lk3) * radial_2e_list_J[tmp](1,0) + lk1*lk2*4*a3*a4 * radial_2e_list_J[tmp](2,1) - (2*a1*lk2+2*a2*lk1)*4*a3*a4 * radial_2e_list_J[tmp](0,1);
    //                     else
    //                         array_radial_J_SSSS[tmp][ii][jj][kk][ll] += lk1*lk2*4*a3*a4 * radial_2e_list_J[tmp](2,1) - (2*a1*lk2+2*a2*lk1)*4*a3*a4 * radial_2e_list_J[tmp](0,1);
    //                 }
    //                 else
    //                 {
    //                     if(l_q != 0)
    //                         array_radial_J_SSSS[tmp][ii][jj][kk][ll] += 4*a1*a2*lk3*lk4 * radial_2e_list_J[tmp](1,2) - 4*a1*a2*(2*a3*lk4+2*a4*lk3) * radial_2e_list_J[tmp](1,0);
    //                 }
    //                 array_radial_J_SSLL[tmp][ii][jj][kk][ll] /= norm_J*4.0*pow(speedOfLight,2);
    //                 array_radial_J_SSSS[tmp][ii][jj][kk][ll] /= norm_J*16.0*pow(speedOfLight,4);
    //             }
    //             lk2 = 1+l_q+k_q; lk4 = 1+l_p+k_p; 
    //             a2 = shell_list(qshell).exp_a(ll); a4 = shell_list(pshell).exp_a(jj);
    //             for(int tmp = LmaxK; tmp >= 0; tmp -= 2)
    //             {
    //                 array_radial_K_LLLL[tmp][ii][ll][kk][jj] = radial_2e_list_K[tmp](0,0) / norm_K;
    //                 array_radial_K_SSLL[tmp][ii][ll][kk][jj] = 4.0*a1*a2 * radial_2e_list_K[tmp](1,0);
    //                 array_radial_K_SSSS[tmp][ii][ll][kk][jj] = 4*a1*a2*4*a3*a4 * radial_2e_list_K[tmp](1,1);
    //                 if(l_p != 0 && l_q != 0)
    //                 {
    //                     array_radial_K_SSLL[tmp][ii][ll][kk][jj] += lk1*lk2 * radial_2e_list_K[tmp](2,0) - (2.0*a1*lk2+2.0*a2*lk1) * radial_2e_list_K[tmp](0,0);
    //                     array_radial_K_SSSS[tmp][ii][ll][kk][jj] += lk1*lk2*lk3*lk4 * radial_2e_list_K[tmp](2,2) - (2*a1*lk2+2*a2*lk1)*lk3*lk4 * radial_2e_list_K[tmp](0,2) + 4*a1*a2*lk3*lk4 * radial_2e_list_K[tmp](1,2) - lk1*lk2*(2*a3*lk4+2*a4*lk3) * radial_2e_list_K[tmp](2,0) + (2*a1*lk2+2*a2*lk1)*(2*a3*lk4+2*a4*lk3) * radial_2e_list_K[tmp](0,0) - 4*a1*a2*(2*a3*lk4+2*a4*lk3) * radial_2e_list_K[tmp](1,0) + lk1*lk2*4*a3*a4 * radial_2e_list_K[tmp](2,1) - (2*a1*lk2+2*a2*lk1)*4*a3*a4 * radial_2e_list_K[tmp](0,1);
    //                 }
    //                 else if(l_p != 0 || l_q != 0)
    //                 {
    //                     array_radial_K_SSLL[tmp][ii][ll][kk][jj] += - (2.0*a1*lk2+2.0*a2*lk1) * radial_2e_list_K[tmp](0,0);
    //                     array_radial_K_SSSS[tmp][ii][ll][kk][jj] += (2*a1*lk2+2*a2*lk1)*(2*a3*lk4+2*a4*lk3) * radial_2e_list_K[tmp](0,0) - 4*a1*a2*(2*a3*lk4+2*a4*lk3) * radial_2e_list_K[tmp](1,0) - (2*a1*lk2+2*a2*lk1)*4*a3*a4 * radial_2e_list_K[tmp](0,1);
    //                 }
    //                 array_radial_K_SSLL[tmp][ii][ll][kk][jj] /= norm_K*4.0*pow(speedOfLight,2);
    //                 array_radial_K_SSSS[tmp][ii][ll][kk][jj] /= norm_K*16.0*pow(speedOfLight,4);
    //             }

    //             int index_tmp_p = 1 - (2*l_p+1 - twojj_p)/2;
    //             if(l_p == 0) index_tmp_p = 0;
    //             int index_tmp_q = 1 - (2*l_q+1 - twojj_q)/2;
    //             if(l_q == 0) index_tmp_q = 0;
    //             for(int mp = 0; mp < twojj_p + 1; mp++)
    //             for(int mq = 0; mq < twojj_q + 1; mq++)
    //             {
    //                 int e1J = ii*size_gtos_p+jj, e2J = kk*size_gtos_q+ll;
    //                 int e1K = ii*size_gtos_q+ll, e2K = kk*size_gtos_p+jj;
    //                 h2eJ_LLLL(index_tmp_p,index_tmp_q)(mp,mq)(e1J,e2J) = 0.0;
    //                 h2eK_LLLL(index_tmp_p,index_tmp_q)(mp,mq)(e1K,e2K) = 0.0;
    //                 h2eJ_SSLL(index_tmp_p,index_tmp_q)(mp,mq)(e1J,e2J) = 0.0;
    //                 h2eK_SSLL(index_tmp_p,index_tmp_q)(mp,mq)(e1K,e2K) = 0.0;
    //                 h2eJ_SSSS(index_tmp_p,index_tmp_q)(mp,mq)(e1J,e2J) = 0.0;
    //                 h2eK_SSSS(index_tmp_p,index_tmp_q)(mp,mq)(e1K,e2K) = 0.0;
    //                 for(int tmp = LmaxJ; tmp >= 0; tmp = tmp - 2)
    //                 {
    //                     h2eJ_LLLL(index_tmp_p,index_tmp_q)(mp,mq)(e1J,e2J) += array_radial_J_LLLL[tmp][ii][jj][kk][ll] * array_angular_J[tmp][index_tmp_p][index_tmp_q](mp,mq);
    //                     h2eJ_SSLL(index_tmp_p,index_tmp_q)(mp,mq)(e1J,e2J) += array_radial_J_SSLL[tmp][ii][jj][kk][ll] * array_angular_J[tmp][index_tmp_p][index_tmp_q](mp,mq);
    //                     h2eJ_SSSS(index_tmp_p,index_tmp_q)(mp,mq)(e1J,e2J) += array_radial_J_SSSS[tmp][ii][jj][kk][ll] * array_angular_J[tmp][index_tmp_p][index_tmp_q](mp,mq);
    //                 }
    //                 for(int tmp = LmaxK; tmp >= 0; tmp = tmp - 2)
    //                 {
    //                     h2eK_LLLL(index_tmp_p,index_tmp_q)(mp,mq)(e1K,e2K) += array_radial_K_LLLL[tmp][ii][ll][kk][jj] * array_angular_K[tmp][index_tmp_p][index_tmp_q](mp,mq);
    //                     h2eK_SSLL(index_tmp_p,index_tmp_q)(mp,mq)(e1K,e2K) += array_radial_K_SSLL[tmp][ii][ll][kk][jj] * array_angular_K[tmp][index_tmp_p][index_tmp_q](mp,mq);
    //                     h2eK_SSSS(index_tmp_p,index_tmp_q)(mp,mq)(e1K,e2K) += array_radial_K_SSSS[tmp][ii][ll][kk][jj] * array_angular_K[tmp][index_tmp_p][index_tmp_q](mp,mq);
    //                 }
    //             }
    //         }
    //     }
    //     int int_tmp2_p = 0, int_tmp2_q = 0;
    //     for(int ii = 0; ii < irrep_list(int_tmp1_p+int_tmp2_p).two_j + 1; ii++)
    //     for(int jj = 0; jj < irrep_list(int_tmp1_q+int_tmp2_q).two_j + 1; jj++)
    //     {
    //         LLLL.J(int_tmp1_p+int_tmp2_p + ii, int_tmp1_q+int_tmp2_q + jj) = h2eJ_LLLL(0,0)(ii,jj);
    //         LLLL.K(int_tmp1_p+int_tmp2_p + ii, int_tmp1_q+int_tmp2_q + jj) = h2eK_LLLL(0,0)(ii,jj);
    //         SSLL.J(int_tmp1_p+int_tmp2_p + ii, int_tmp1_q+int_tmp2_q + jj) = h2eJ_SSLL(0,0)(ii,jj);
    //         SSLL.K(int_tmp1_p+int_tmp2_p + ii, int_tmp1_q+int_tmp2_q + jj) = h2eK_SSLL(0,0)(ii,jj);
    //         SSSS.J(int_tmp1_p+int_tmp2_p + ii, int_tmp1_q+int_tmp2_q + jj) = h2eJ_SSSS(0,0)(ii,jj);
    //         SSSS.K(int_tmp1_p+int_tmp2_p + ii, int_tmp1_q+int_tmp2_q + jj) = h2eK_SSSS(0,0)(ii,jj);
    //     }
    //     if(l_p != 0 && l_q == 0)
    //     {
    //         int_tmp2_p += irrep_list(int_tmp1_p+int_tmp2_p).two_j + 1;
    //         for(int ii = 0; ii < irrep_list(int_tmp1_p+int_tmp2_p).two_j + 1; ii++)
    //         for(int jj = 0; jj < irrep_list(int_tmp1_q+int_tmp2_q).two_j + 1; jj++)
    //         {
    //             LLLL.J(int_tmp1_p+int_tmp2_p + ii, int_tmp1_q+int_tmp2_q + jj) = h2eJ_LLLL(1,0)(ii,jj);
    //             LLLL.K(int_tmp1_p+int_tmp2_p + ii, int_tmp1_q+int_tmp2_q + jj) = h2eK_LLLL(1,0)(ii,jj);
    //             SSLL.J(int_tmp1_p+int_tmp2_p + ii, int_tmp1_q+int_tmp2_q + jj) = h2eJ_SSLL(1,0)(ii,jj);
    //             SSLL.K(int_tmp1_p+int_tmp2_p + ii, int_tmp1_q+int_tmp2_q + jj) = h2eK_SSLL(1,0)(ii,jj);
    //             SSSS.J(int_tmp1_p+int_tmp2_p + ii, int_tmp1_q+int_tmp2_q + jj) = h2eJ_SSSS(1,0)(ii,jj);
    //             SSSS.K(int_tmp1_p+int_tmp2_p + ii, int_tmp1_q+int_tmp2_q + jj) = h2eK_SSSS(1,0)(ii,jj);
    //         }
    //     }
    //     else if(l_q != 0 && l_p == 0)
    //     {
    //         int_tmp2_q += irrep_list(int_tmp1_q+int_tmp2_q).two_j + 1;
    //         for(int ii = 0; ii < irrep_list(int_tmp1_p+int_tmp2_p).two_j + 1; ii++)
    //         for(int jj = 0; jj < irrep_list(int_tmp1_q+int_tmp2_q).two_j + 1; jj++)
    //         {
    //             LLLL.J(int_tmp1_p+int_tmp2_p + ii, int_tmp1_q+int_tmp2_q + jj) = h2eJ_LLLL(0,1)(ii,jj);
    //             LLLL.K(int_tmp1_p+int_tmp2_p + ii, int_tmp1_q+int_tmp2_q + jj) = h2eK_LLLL(0,1)(ii,jj);
    //             SSLL.J(int_tmp1_p+int_tmp2_p + ii, int_tmp1_q+int_tmp2_q + jj) = h2eJ_SSLL(0,1)(ii,jj);
    //             SSLL.K(int_tmp1_p+int_tmp2_p + ii, int_tmp1_q+int_tmp2_q + jj) = h2eK_SSLL(0,1)(ii,jj);
    //             SSSS.J(int_tmp1_p+int_tmp2_p + ii, int_tmp1_q+int_tmp2_q + jj) = h2eJ_SSSS(0,1)(ii,jj);
    //             SSSS.K(int_tmp1_p+int_tmp2_p + ii, int_tmp1_q+int_tmp2_q + jj) = h2eK_SSSS(0,1)(ii,jj);
    //         }
            
    //     }
    //     else if(l_p != 0 && l_q != 0)
    //     {
    //         int int_tmp3_p = irrep_list(int_tmp1_p+int_tmp2_p).two_j + 1;
    //         int int_tmp3_q = irrep_list(int_tmp1_q+int_tmp2_q).two_j + 1;
    //         int_tmp2_p += int_tmp3_p;
    //         for(int ii = 0; ii < irrep_list(int_tmp1_p+int_tmp2_p).two_j + 1; ii++)
    //         for(int jj = 0; jj < irrep_list(int_tmp1_q+int_tmp2_q).two_j + 1; jj++)
    //         {
    //             LLLL.J(int_tmp1_p+int_tmp2_p + ii, int_tmp1_q+int_tmp2_q + jj) = h2eJ_LLLL(1,0)(ii,jj);
    //             LLLL.K(int_tmp1_p+int_tmp2_p + ii, int_tmp1_q+int_tmp2_q + jj) = h2eK_LLLL(1,0)(ii,jj);
    //             SSLL.J(int_tmp1_p+int_tmp2_p + ii, int_tmp1_q+int_tmp2_q + jj) = h2eJ_SSLL(1,0)(ii,jj);
    //             SSLL.K(int_tmp1_p+int_tmp2_p + ii, int_tmp1_q+int_tmp2_q + jj) = h2eK_SSLL(1,0)(ii,jj);
    //             SSSS.J(int_tmp1_p+int_tmp2_p + ii, int_tmp1_q+int_tmp2_q + jj) = h2eJ_SSSS(1,0)(ii,jj);
    //             SSSS.K(int_tmp1_p+int_tmp2_p + ii, int_tmp1_q+int_tmp2_q + jj) = h2eK_SSSS(1,0)(ii,jj);
    //         }
    //         int_tmp2_p -= int_tmp3_p;
    //         int_tmp2_q += int_tmp3_q;
    //         for(int ii = 0; ii < irrep_list(int_tmp1_p+int_tmp2_p).two_j + 1; ii++)
    //         for(int jj = 0; jj < irrep_list(int_tmp1_q+int_tmp2_q).two_j + 1; jj++)
    //         {
    //             LLLL.J(int_tmp1_p+int_tmp2_p + ii, int_tmp1_q+int_tmp2_q + jj) = h2eJ_LLLL(0,1)(ii,jj);
    //             LLLL.K(int_tmp1_p+int_tmp2_p + ii, int_tmp1_q+int_tmp2_q + jj) = h2eK_LLLL(0,1)(ii,jj);
    //             SSLL.J(int_tmp1_p+int_tmp2_p + ii, int_tmp1_q+int_tmp2_q + jj) = h2eJ_SSLL(0,1)(ii,jj);
    //             SSLL.K(int_tmp1_p+int_tmp2_p + ii, int_tmp1_q+int_tmp2_q + jj) = h2eK_SSLL(0,1)(ii,jj);
    //             SSSS.J(int_tmp1_p+int_tmp2_p + ii, int_tmp1_q+int_tmp2_q + jj) = h2eJ_SSSS(0,1)(ii,jj);
    //             SSSS.K(int_tmp1_p+int_tmp2_p + ii, int_tmp1_q+int_tmp2_q + jj) = h2eK_SSSS(0,1)(ii,jj);
    //         }
    //         int_tmp2_p += int_tmp3_p;
    //         for(int ii = 0; ii < irrep_list(int_tmp1_p+int_tmp2_p).two_j + 1; ii++)
    //         for(int jj = 0; jj < irrep_list(int_tmp1_q+int_tmp2_q).two_j + 1; jj++)
    //         {
    //             LLLL.J(int_tmp1_p+int_tmp2_p + ii, int_tmp1_q+int_tmp2_q + jj) = h2eJ_LLLL(1,1)(ii,jj);
    //             LLLL.K(int_tmp1_p+int_tmp2_p + ii, int_tmp1_q+int_tmp2_q + jj) = h2eK_LLLL(1,1)(ii,jj);
    //             SSLL.J(int_tmp1_p+int_tmp2_p + ii, int_tmp1_q+int_tmp2_q + jj) = h2eJ_SSLL(1,1)(ii,jj);
    //             SSLL.K(int_tmp1_p+int_tmp2_p + ii, int_tmp1_q+int_tmp2_q + jj) = h2eK_SSLL(1,1)(ii,jj);
    //             SSSS.J(int_tmp1_p+int_tmp2_p + ii, int_tmp1_q+int_tmp2_q + jj) = h2eJ_SSSS(1,1)(ii,jj);
    //             SSSS.K(int_tmp1_p+int_tmp2_p + ii, int_tmp1_q+int_tmp2_q + jj) = h2eK_SSSS(1,1)(ii,jj);
    //         }
    //     }
    //     int_tmp1_q += 4*l_q+2;
    // }
    // int_tmp1_p += 4*l_p+2;
    // }

    // return ;
}

/* 
    get contraction coefficients for uncontracted calculations 
*/
MatrixXd INT_SPH::get_coeff_contraction_spinor()
{
    MatrixXd coeff(size_gtou_spinor, size_gtoc_spinor);
    coeff = MatrixXd::Zero(size_gtou_spinor, size_gtoc_spinor);

    int int_tmp1 = 0, int_tmp2 = 0, int_tmp3 = 0;
    for(int ishell = 0; ishell < size_shell; ishell++)
    {
        int ll = shell_list(ishell).l;
        int size_con = shell_list(ishell).coeff.cols(), size_unc = shell_list(ishell).coeff.rows();
        int_tmp3 = 0;
        for(int twojj = abs(2*ll-1); twojj <= 2*ll+1; twojj = twojj + 2)
        {
            for(int ii = 0; ii < size_con; ii++)    
            {   
                for(int mm = 0; mm < twojj+1; mm++)
                {
                    for(int jj = 0; jj < size_unc; jj++)
                    {    
                        coeff(int_tmp2 + int_tmp3 + jj*(twojj+1) + mm, int_tmp1) = shell_list(ishell).coeff(jj,ii);
                    }
                    int_tmp1 += 1;
                }
            }
            int_tmp3 += size_unc * (twojj + 1);   
        }        
        int_tmp2 += size_unc * (2*ll+1) * 2;
    }

    return coeff;
}


