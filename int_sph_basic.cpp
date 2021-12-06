#include<Eigen/Dense>
#include<string>
#include<iostream>
#include<iomanip>
#include<fstream>
#include<cmath>
#include<complex>
#include<omp.h>
#include<vector>
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
double INT_SPH::auxiliary_1e(const int& l, const double& a) const
{
    int n = l / 2;
    if(l < 0)
    {
        if(l == -2) return 0.0;
        cout << "ERROR: l = " << l << " for auxiliary 1e integral." << endl;
        exit(99);
    }
    else if(l == 0)  return 0.5*sqrt(M_PI/a);
    else if(n*2 == l)    return double_factorial(2*n-1)/pow(a,n)/pow(2.0,n+1)*sqrt(M_PI/a);
    else    return factorial(n)/2.0/pow(a,n+1);
}

/*
    auxiliary_2e_0_r is to evaluate \int_0^inf \int_0^r2 r1^l1 r2^l2 exp(-a1 * r1^2) exp(-a2 * r2^2) dr1dr2
*/
double INT_SPH::auxiliary_2e_0_r(const int& l1, const int& l2, const double& a1, const double& a2) const
{
    int n1 = l1 / 2;
    if(l1 < 0 || l2 < 0)
    {
        return 0.0;
    }
    if(n1 * 2 == l1)
    {
        cout << "ERROR: When auxiliary_2e_0_r is called, l1 must be set to an odd number!" << endl;
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
double INT_SPH::auxiliary_2e_r_inf(const int& l1, const int& l2, const double& a1, const double& a2) const
{
    if(l1 < 0 || l2 < 0)
    {
        return 0.0;
    }
    int n1 = l1 / 2;
    if(n1 * 2 == l1)
    {
        cout << "ERROR: When auxiliary_2e_r_inf is called, l1 must be set to an odd number!" << endl;
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

    int two_j1 = 2*l1 + s1;
    int two_j2 = 2*l2 + s2;
    int two_j3 = 2*l3 + s3;
    int two_j4 = 2*l4 + s4;
    double angular = 0.0;
    for(int mm = -LL; mm <= LL; mm++)
    {
        if(two_m2 - two_m1 - 2*mm != 0 || two_m4 - two_m3 + 2*mm != 0) continue;
        else
        {
            angular += pow(-1, mm) * gsl_sf_coupling_3j(two_j1,2*LL,two_j2,-two_m1,-2*mm,two_m2)
                                   * gsl_sf_coupling_3j(two_j3,2*LL,two_j4,-two_m3, 2*mm,two_m4);
        }
    }

    return pow(-1.0,two_j1+two_j3-(two_m1+two_m3)/2-1) * angular
            * sqrt((two_j1+1.0)*(two_j2+1.0)*(two_j3+1.0)*(two_j4+1.0))
            * gsl_sf_coupling_3j(two_j1,2*LL,two_j2,1,0,-1)
            * gsl_sf_coupling_3j(two_j3,2*LL,two_j4,1,0,-1);


    // double angular = 0.0;
    // for(int mm = -LL; mm <= LL; mm++)
    // {
    //     if(two_m2 - two_m1 - 2*mm != 0 || two_m4 - two_m3 + 2*mm != 0) continue;
    //     else
    //     {
    //         angular += pow(-1, mm) 
    //         * (pow(-1,(two_m1-1)/2)*s1*s2*sqrt((l1+0.5+s1*two_m1/2.0)*(l2+0.5+s2*two_m2/2.0))*wigner_3j(l1,l2,LL,(1-two_m1)/2,(two_m2-1)/2,-mm)
    //         + pow(-1,(two_m1+1)/2)*sqrt((l1+0.5-s1*two_m1/2.0)*(l2+0.5-s2*two_m2/2.0))*wigner_3j(l1,l2,LL,(-1-two_m1)/2,(two_m2+1)/2,-mm)) 
    //         * (pow(-1,(two_m3-1)/2)*s3*s4*sqrt((l3+0.5+s3*two_m3/2.0)*(l4+0.5+s4*two_m4/2.0))*wigner_3j(l3,l4,LL,(1-two_m3)/2,(two_m4-1)/2,mm)
    //         + pow(-1,(two_m3+1)/2)*sqrt((l3+0.5-s3*two_m3/2.0)*(l4+0.5-s4*two_m4/2.0))*wigner_3j(l3,l4,LL,(-1-two_m3)/2,(two_m4+1)/2,mm));
    //     }
    // }

    // return angular * wigner_3j_zeroM(l1, l2, LL) * wigner_3j_zeroM(l3, l4, LL);
}
double INT_SPH::int2e_get_angular_J(const int& l1, const int& two_m1, const int& s1, const int& l2, const int& two_m2, const int& s2, const int& LL) const
{
    return int2e_get_angular(l1,two_m1,s1,l1,two_m1,s1,l2,two_m2,s2,l2,two_m2,s2,LL);
}
double INT_SPH::int2e_get_angular_K(const int& l1, const int& two_m1, const int& s1, const int& l2, const int& two_m2, const int& s2, const int& LL) const
{
    return int2e_get_angular(l1,two_m1,s1,l2,two_m2,s2,l2,two_m2,s2,l1,two_m1,s1,LL);
}

double INT_SPH::get_radial_LLLL_J(const int& lp, const int& lq, const int& LL, const double& a1, const double& a2, const double& a3, const double& a4, const double& lk1, const double& lk2, const double& lk3, const double& lk4, double radial_list[3][3], const bool& spinFree) const
{
    return radial_list[0][0];
}
double INT_SPH::get_radial_LLLL_K(const int& lp, const int& lq, const int& LL, const double& a1, const double& a2, const double& a3, const double& a4, const double& lk1, const double& lk2, const double& lk3, const double& lk4, double radial_list[3][3], const bool& spinFree) const
{
    return radial_list[0][0];
}
double INT_SPH::get_radial_SSLL_J(const int& lp, const int& lq, const int& LL, const double& a1, const double& a2, const double& a3, const double& a4, const double& lk1, const double& lk2, const double& lk3, const double& lk4, double radial_list[3][3], const bool& spinFree) const
{
    double tmp = 4.0*a1*a2 * radial_list[1][0];
    if(spinFree)
    {
        double l12 = lp*lp + lp*(lp+1)/2 + lp*(lp+1)/2 - LL*(LL+1)/2, l34 = lq*lq + lq*(lq+1)/2 + lq*(lq+1)/2 - LL*(LL+1)/2;
        if(lp != 0)
            tmp += l12 * radial_list[2][0] - (2.0*a1*lp+2.0*a2*lp) * radial_list[0][0];
    }
    else
    {
        if(lp != 0)
            tmp += lk1*lk2 * radial_list[2][0] - (2.0*a1*lk2+2.0*a2*lk1) * radial_list[0][0];
    }
    
    return tmp;
}
double INT_SPH::get_radial_SSLL_K(const int& lp, const int& lq, const int& LL, const double& a1, const double& a2, const double& a3, const double& a4, const double& lk1, const double& lk2, const double& lk3, const double& lk4, double radial_list[3][3], const bool& spinFree) const
{
    double tmp = 4.0*a1*a2 * radial_list[1][0];
    if(spinFree)
    {
        double l12 = lp*lq + lp*(lp+1)/2 + lq*(lq+1)/2 - LL*(LL+1)/2, l34 = lq*lp + lq*(lq+1)/2 + lp*(lp+1)/2 - LL*(LL+1)/2;
        if(lp != 0 && lq != 0)
            tmp += l12 * radial_list[2][0] - (2.0*a1*lq+2.0*a2*lp) * radial_list[0][0];
        else if(lp != 0 || lq != 0)
            tmp += - (2.0*a1*lq+2.0*a2*lp) * radial_list[0][0];
    }
    else
    {
        if(lp != 0 && lq != 0)
            tmp += lk1*lk2 * radial_list[2][0] - (2.0*a1*lk2+2.0*a2*lk1) * radial_list[0][0];
        else if(lp != 0 || lq != 0)
            tmp += - (2.0*a1*lk2+2.0*a2*lk1) * radial_list[0][0];
    }
    
    return tmp;
}
double INT_SPH::get_radial_SSSS_J(const int& lp, const int& lq, const int& LL, const double& a1, const double& a2, const double& a3, const double& a4, const double& lk1, const double& lk2, const double& lk3, const double& lk4, double radial_list[3][3], const bool& spinFree) const
{
    double tmp = 4*a1*a2*4*a3*a4 * radial_list[1][1];
    if(spinFree)
    {
        double l12 = lp*lp + lp*(lp+1)/2 + lp*(lp+1)/2 - LL*(LL+1)/2, l34 = lq*lq + lq*(lq+1)/2 + lq*(lq+1)/2 - LL*(LL+1)/2;
        if(lp != 0)
        {
            if(lq != 0)
                tmp += l12*l34 * radial_list[2][2] - (2*a1*lp+2*a2*lp)*l34 * radial_list[0][2] + 4*a1*a2*l34 * radial_list[1][2] - l12*(2*a3*lq+2*a4*lq) * radial_list[2][0] + (2*a1*lp+2*a2*lp)*(2*a3*lq+2*a4*lq) * radial_list[0][0] - 4*a1*a2*(2*a3*lq+2*a4*lq) * radial_list[1][0] + l12*4*a3*a4 * radial_list[2][1] - (2*a1*lp+2*a2*lp)*4*a3*a4 * radial_list[0][1];
            else
                tmp += l12*4*a3*a4 * radial_list[2][1] - (2*a1*lp+2*a2*lp)*4*a3*a4 * radial_list[0][1];
        }
        else
        {
            if(lq != 0)
                tmp += 4*a1*a2*l34 * radial_list[1][2] - 4*a1*a2*(2*a3*lq+2*a4*lq) * radial_list[1][0];
        }
    }
    else
    {
        if(lp != 0)
        {
            if(lq != 0)
                tmp += lk1*lk2*lk3*lk4 * radial_list[2][2] - (2*a1*lk2+2*a2*lk1)*lk3*lk4 * radial_list[0][2] + 4*a1*a2*lk3*lk4 * radial_list[1][2] - lk1*lk2*(2*a3*lk4+2*a4*lk3) * radial_list[2][0] + (2*a1*lk2+2*a2*lk1)*(2*a3*lk4+2*a4*lk3) * radial_list[0][0] - 4*a1*a2*(2*a3*lk4+2*a4*lk3) * radial_list[1][0] + lk1*lk2*4*a3*a4 * radial_list[2][1] - (2*a1*lk2+2*a2*lk1)*4*a3*a4 * radial_list[0][1];
            else
                tmp += lk1*lk2*4*a3*a4 * radial_list[2][1] - (2*a1*lk2+2*a2*lk1)*4*a3*a4 * radial_list[0][1];
        }
        else
        {
            if(lq != 0)
                tmp += 4*a1*a2*lk3*lk4 * radial_list[1][2] - 4*a1*a2*(2*a3*lk4+2*a4*lk3) * radial_list[1][0];
        }
    }
    
    return tmp;
}
double INT_SPH::get_radial_SSSS_K(const int& lp, const int& lq, const int& LL, const double& a1, const double& a2, const double& a3, const double& a4, const double& lk1, const double& lk2, const double& lk3, const double& lk4, double radial_list[3][3], const bool& spinFree) const
{
    double tmp = 4*a1*a2*4*a3*a4 * radial_list[1][1];
    if(spinFree)
    {
        double l12 = lp*lq + lp*(lp+1)/2 + lq*(lq+1)/2 - LL*(LL+1)/2, l34 = lq*lp + lq*(lq+1)/2 + lp*(lp+1)/2 - LL*(LL+1)/2;
        if(lp != 0 && lq != 0)
            tmp += l12*l34 * radial_list[2][2] - (2*a1*lq+2*a2*lp)*l34 * radial_list[0][2] + 4*a1*a2*l34 * radial_list[1][2] - l12*(2*a3*lp+2*a4*lq) * radial_list[2][0] + (2*a1*lq+2*a2*lp)*(2*a3*lp+2*a4*lq) * radial_list[0][0] - 4*a1*a2*(2*a3*lp+2*a4*lq) * radial_list[1][0] + l12*4*a3*a4 * radial_list[2][1] - (2*a1*lq+2*a2*lp)*4*a3*a4 * radial_list[0][1];
        else if(lp != 0 || lq != 0)
            tmp += (2*a1*lq+2*a2*lp)*(2*a3*lp+2*a4*lq) * radial_list[0][0] - 4*a1*a2*(2*a3*lp+2*a4*lq) * radial_list[1][0] - (2*a1*lq+2*a2*lp)*4*a3*a4 * radial_list[0][1];
    }
    else
    {
        if(lp != 0 && lq != 0)
            tmp += lk1*lk2*lk3*lk4 * radial_list[2][2] - (2*a1*lk2+2*a2*lk1)*lk3*lk4 * radial_list[0][2] + 4*a1*a2*lk3*lk4 * radial_list[1][2] - lk1*lk2*(2*a3*lk4+2*a4*lk3) * radial_list[2][0] + (2*a1*lk2+2*a2*lk1)*(2*a3*lk4+2*a4*lk3) * radial_list[0][0] - 4*a1*a2*(2*a3*lk4+2*a4*lk3) * radial_list[1][0] + lk1*lk2*4*a3*a4 * radial_list[2][1] - (2*a1*lk2+2*a2*lk1)*4*a3*a4 * radial_list[0][1];
        else if(lp != 0 || lq != 0)
            tmp += (2*a1*lk2+2*a2*lk1)*(2*a3*lk4+2*a4*lk3) * radial_list[0][0] - 4*a1*a2*(2*a3*lk4+2*a4*lk3) * radial_list[1][0] - (2*a1*lk2+2*a2*lk1)*4*a3*a4 * radial_list[0][1];
    }
    
    return tmp;
}

double INT_SPH::get_radial_LSLS_J(const int& lp, const int& lq, const int& LL, const double& a1, const double& a2, const double& a3, const double& a4, const double& lk1, const double& lk2, const double& lk3, const double& lk4, double radial_list[4], const bool& spinFree) const
{
    double tmp = 4.0*a2*a4 * radial_list[0];
    if(spinFree)
    {

    }
    else
    {
        if(lp != 0 && lq != 0)
            tmp += lk2*lk4 * radial_list[3] - 2.0*a4*lk2 * radial_list[1] - 2.0*a2*lk4 * radial_list[2];
        else if(lp != 0 && lq == 0)
            tmp -= 2.0*a4*lk2 * radial_list[1];
        else if(lp == 0 && lq != 0)
            tmp -= 2.0*a2*lk4 * radial_list[2];
    }
    
    return tmp;
}
double INT_SPH::get_radial_LSLS_K(const int& lp, const int& lq, const int& LL, const double& a1, const double& a2, const double& a3, const double& a4, const double& lk1, const double& lk2, const double& lk3, const double& lk4, double radial_list[4], const bool& spinFree) const
{
    double tmp = 4.0*a2*a4 * radial_list[0];
    if(spinFree)
    {

    }
    else
    {
        if(lp != 0 && lq != 0)
            tmp += lk2*lk4 * radial_list[3] - 2.0*a4*lk2 * radial_list[1] - 2.0*a2*lk4 * radial_list[2];
        else if(lp == 0 && lq != 0)
            tmp -= 2.0*a4*lk2 * radial_list[1];
        else if(lp != 0 && lq == 0)
            tmp -= 2.0*a2*lk4 * radial_list[2];
    }
    
    return tmp;
}
double INT_SPH::get_radial_LSSL_J(const int& lp, const int& lq, const int& LL, const double& a1, const double& a2, const double& a3, const double& a4, const double& lk1, const double& lk2, const double& lk3, const double& lk4, double radial_list[4], const bool& spinFree) const
{
    double tmp = 4.0*a2*a3 * radial_list[0];
    if(spinFree)
    {

    }
    else
    {
        if(lp != 0 && lq != 0)
            tmp += lk2*lk3 *radial_list[3] - 2.0*a3*lk2 * radial_list[1] - 2.0*a2*lk3 * radial_list[2];
        else if(lp != 0 && lq == 0)
            tmp -= 2.0*a3*lk2 * radial_list[1];
        else if(lp == 0 && lq != 0)
            tmp -= 2.0*a2*lk3 * radial_list[2];
    }
    
    return tmp;
}
double INT_SPH::get_radial_LSSL_K(const int& lp, const int& lq, const int& LL, const double& a1, const double& a2, const double& a3, const double& a4, const double& lk1, const double& lk2, const double& lk3, const double& lk4, double radial_list[4], const bool& spinFree) const
{
    double tmp = 4.0*a2*a3 * radial_list[0];
    if(spinFree)
    {

    }
    else
    {
        if(lq != 0)
            tmp += lk2*lk3 * radial_list[3] - 2.0*a3*lk2 * radial_list[1] - 2.0*a2*lk3 * radial_list[2];
    }
    
    return tmp;
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


double INT_SPH::int2e_get_threeSH(const int& l1, const int& m1, const int& l2, const int& m2, const int& l3, const int& m3, const double& threeJ) const
{
    // return pow(-1,m1)*threeJ*wigner_3j(l1,l2,l3,-m1,m2,m3);
    return pow(-1,m1)*threeJ*wigner_3j(l1,l2,l3,-m1,m2,m3)*sqrt((2.0*l1+1.0)*(2.0*l3+1.0));
}
double INT_SPH::int2e_get_angularX_RME(const int& two_j1, const int& l1, const int& two_j2, const int& l2, const int& LL, const int& vv, const double& threeJ) const
{
    return sqrt(6.0 * (two_j1+1.0)*(two_j2+1.0)*(2*LL+1.0) * (2*l1+1.0)*(2*l2+1.0)) * threeJ
           * gsl_sf_coupling_9j(2*l1,2*l2,2*vv,1,1,2,two_j1,two_j2,2*LL) * pow(-1,l1);
}

int2eJK INT_SPH::compact_h2e(const int2eJK& h2eFull, const Matrix<irrep_jm, Dynamic, 1>& irrepList, const int& occMaxL) const
{
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
    int_2e_JK.J = new double***[Nirrep_compact];
    int_2e_JK.K = new double***[Nirrep_compact];
    for(int ii = 0; ii < Nirrep_compact; ii++)
    {
        int_2e_JK.J[ii] = new double**[Nirrep_compact];
        int_2e_JK.K[ii] = new double**[Nirrep_compact];
    }
    int int_tmp1_p = 0, int_tmp1_pp = 0;
    for(int pshell = 0; pshell < occMaxShell; pshell++)
    {
    int l_p = shell_list(pshell).l, int_tmp1_q = 0, int_tmp1_qq = 0;
    for(int qshell = 0; qshell < occMaxShell; qshell++)
    {
        int l_q = shell_list(qshell).l;
        int l_p_cycle = (l_p == 0) ? 1 : 2, l_q_cycle = (l_q == 0) ? 1 : 2;
        int size_gtos_p = shell_list(pshell).coeff.rows(), size_gtos_q = shell_list(qshell).coeff.rows();
        int size_tmp_p = (l_p == 0) ? 1 : 2, size_tmp_q = (l_q == 0) ? 1 : 2;
        
        for(int twojj_p = abs(2*l_p-1); twojj_p <= 2*l_p+1; twojj_p = twojj_p + 2)
        for(int twojj_q = abs(2*l_q-1); twojj_q <= 2*l_q+1; twojj_q = twojj_q + 2)
        {
            int sym_ap = twojj_p - 2*l_p, sym_aq = twojj_q - 2*l_q;
            int int_tmp2_p = (twojj_p - abs(2*l_p-1)) / 2, int_tmp2_q = (twojj_q - abs(2*l_q-1))/2;
            int_2e_JK.J[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q] = new double*[size_gtos_p*size_gtos_p];
            for(int iii = 0; iii < size_gtos_p*size_gtos_p; iii++)
                int_2e_JK.J[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][iii] = new double[size_gtos_q*size_gtos_q];
            int_2e_JK.K[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q] = new double*[size_gtos_p*size_gtos_q];
            for(int iii = 0; iii < size_gtos_p*size_gtos_q; iii++)
                int_2e_JK.K[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][iii] = new double[size_gtos_p*size_gtos_q];

            // Radial 
            #pragma omp parallel  for
            for(int tt = 0; tt < size_gtos_p*size_gtos_p*size_gtos_q*size_gtos_q; tt++)
            {
                double radial_J_mm, radial_K_mm, radial_J_mp, radial_K_mp, radial_J_pm, radial_K_pm, radial_J_pp, radial_K_pp;
                int e1J = tt/(size_gtos_q*size_gtos_q);
                int e2J = tt - e1J*(size_gtos_q*size_gtos_q);
                int e1K = tt/(size_gtos_p*size_gtos_q);
                int e2K = tt - e1K*(size_gtos_p*size_gtos_q);

                int_2e_JK.J[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1J][e2J] = 0.0;
                int_2e_JK.K[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1K][e2K] = 0.0;
                int add_p = int_tmp2_p*(irrep_list(int_tmp1_pp).two_j+1), add_q = int_tmp2_q*(irrep_list(int_tmp1_qq).two_j+1);
                for(int mp = 0; mp < irrep_list(int_tmp1_pp+add_p).two_j + 1; mp++)
                for(int mq = 0; mq < irrep_list(int_tmp1_qq+add_q).two_j + 1; mq++)
                {
                    int_2e_JK.J[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1J][e2J] += h2eFull.J[int_tmp1_pp+add_p + mp][int_tmp1_qq+add_q + mq][e1J][e2J];
                    int_2e_JK.K[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1K][e2K] += h2eFull.K[int_tmp1_pp+add_p + mp][int_tmp1_qq+add_q + mq][e1K][e2K];
                }
                int_2e_JK.J[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1J][e2J] /= (irrep_list(int_tmp1_qq+add_q).two_j + 1.0)*(irrep_list(int_tmp1_pp+add_p).two_j + 1.0);
                int_2e_JK.K[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1K][e2K] /= (irrep_list(int_tmp1_qq+add_q).two_j + 1.0)*(irrep_list(int_tmp1_pp+add_p).two_j + 1.0);
            }
        }
        int_tmp1_q += (l_q == 0) ? 1 : 2;
        int_tmp1_qq += 4*l_q+2;
    }
    int_tmp1_p += (l_p == 0) ? 1 : 2;
    int_tmp1_pp += 4*l_p+2;
    }
    return int_2e_JK;
}