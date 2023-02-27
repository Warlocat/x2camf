#include<string>
#include<iostream>
#include<iomanip>
#include<fstream>
#include<cmath>
#include<complex>
#include<omp.h>
#include"int_sph.h"
using namespace std;

double get_N_coeff(const int& vv, const int& aa, const int& ll, const int& mm)
{
    if(abs(mm+aa) > ll+vv) return 0.0;
    double tmp;
    //  return CG::wigner_3j(2*ll,2,2*ll+2*vv,2*mm,2*aa,-2*mm-2*aa)*CG::wigner_3j(2*ll,2,2*ll+2*vv,0,0,0)*pow(-1,mm+aa)*(2.0*(ll+vv)+1.0);
    // return CG::wigner_3j(2*ll,2,2*ll+2*vv,2*mm,2*aa,-2*mm-2*aa)*CG::wigner_3j(2*ll,2,2*ll+2*vv,0,0,0)*pow(-1,mm)*(2.0*(ll+vv)+1.0);

    
    switch (vv)
    {
    case 1:
        switch (aa)
        {
        case -1:
            tmp = sqrt((ll-mm+1.0)*(ll-mm+2.0))/(2.0*ll+1)/sqrt(2.0);
            break;
        case 0:
            tmp = sqrt((ll+mm+1.0)*(ll-mm+1.0))/(2.0*ll+1);
            break;
        case 1:
            tmp = -sqrt((ll+mm+1.0)*(ll+mm+2.0))/(2.0*ll+1)/sqrt(2.0);
            break;
        }
        break;
    case -1:
        switch (aa)
        {
        case -1:
            tmp = -sqrt((ll+mm)*(ll+mm-1.0))/(2.0*ll+1)/sqrt(2.0);
            break;
        case 0:
            tmp = sqrt((ll+mm)*(ll-mm))/(2.0*ll+1);
            break;
        case 1:
            tmp = sqrt((ll-mm)*(ll-mm-1.0))/(2.0*ll+1)/sqrt(2.0);
            break;
        }
        break;
    default:
        cout << "ERROR: Input aa or vv is invalid for N_coeff" << endl;
        exit(99);
    }
    return tmp;
}

/* 
    evaluate radial part and angular part in 2e integrals 
*/
double INT_SPH::int2e_get_radial_gauge(const int& l1, const double& a1, const int& l2, const double& a2, const int& l3, const double& a3, const int& l4, const double& a4, const int& LL, const int& v1, const int& v2) const
{
    if(LL+v1<0||LL+v2<0||l1+l2+2<LL||l3+l4+2<LL) return 0.0;
    double radial = 0.0, fac;
    if(v1 == v2)
    {
        int vv = LL + v1;
        fac = -2.0*(2.0*LL+1.0)/(2.0*vv+1.0);
        radial = int2e_get_radial(l1,a1,l2,a2,l3,a3,l4,a4,vv);
    }
    else if(v1 == -1 && v2 == +1)
    {
        fac = -(2.0*LL+1.0);
        if((l1 + l2 + 3 + LL) % 2)
        {
            radial = auxiliary_2e_0_r(l1 + l2 + 3 + LL, l3 + l4 - LL, a1 + a2, a3 + a4)
                   - auxiliary_2e_0_r(l1 + l2 + 1 + LL, l3 + l4 + 2 - LL, a1 + a2, a3 + a4);
        }
        else
        {
            radial = auxiliary_2e_r_inf(l3 + l4 - LL, l1 + l2 + 3 + LL, a3 + a4, a1 + a2)
                   - auxiliary_2e_r_inf(l3 + l4 + 2 - LL, l1 + l2 + 1 + LL, a3 + a4, a1 + a2);
        }
    }
    else if(v1 == +1 && v2 == -1)
    {
        fac = -(2.0*LL+1.0);
        if((l3 + l4 + 3 + LL) % 2)
        {
            radial = auxiliary_2e_0_r(l3 + l4 + 3 + LL, l1 + l2 - LL, a3 + a4, a1 + a2)
                   - auxiliary_2e_0_r(l3 + l4 + 1 + LL, l1 + l2 + 2 - LL, a3 + a4, a1 + a2);
        }
        else
        {
            radial = auxiliary_2e_r_inf(l1 + l2 - LL, l3 + l4 + 3 + LL, a1 + a2, a3 + a4)
                   - auxiliary_2e_r_inf(l1 + l2 + 2 - LL, l3 + l4 + 1 + LL, a1 + a2, a3 + a4);
        }
    }
    else
    {
        cout << "WARNING: gauge radial input v1 and v2 are out of domain." << endl;
    }
    
    return radial*fac;
}


double INT_SPH::int2e_get_angular_gauge_LSLS(const int& l1, const int& two_m1, const int& s1, const int& l2, const int& two_m2, const int& s2, const int& l3, const int& two_m3, const int& s3, const int& l4, const int& two_m4, const int& s4, const int& ll, const int& v1, const int& v2) const
{
    if(ll+v1<0 || ll+v2<0) return 0.0;
    double angular = 0.0;
    int l2p = l2+s2, l4p = l4+s4;
    int two_j1 = 2*l1+s1, two_j2 = 2*l2+s2, two_j3 = 2*l3+s3, two_j4 = 2*l4+s4;
    double threeJ1 = CG::wigner_3j(2*l1,2*ll+2*v1,2*l2p,0,0,0), threeJ2 = CG::wigner_3j(2*l3,2*ll+2*v2,2*l4p,0,0,0), tmp;
    tmp = 0.0;
    double rme1 = int2e_get_angularX_RME(two_j1,l1,two_j2,l2p,ll,ll+v1,threeJ1);
    double rme2 = int2e_get_angularX_RME(two_j3,l3,two_j4,l4p,ll,ll+v2,threeJ2);
    for(int MMM = -ll; MMM <= ll; MMM++)
    {
        tmp += pow(-1,MMM) * CG::wigner_3j(two_j1,2*ll,two_j2,-two_m1,-2*MMM,two_m2)
             * CG::wigner_3j(two_j3,2*ll,two_j4,-two_m3,2*MMM,two_m4);
    }
    angular += tmp*rme1*rme2;
    
    return angular*pow(-1,(two_j1+two_j3-two_m1-two_m3)/2)*(2.0*(ll+v1)+1.0)*(2.0*(ll+v2)+1.0)/(2.0*ll+1)*CG::wigner_3j(2,2*ll+2*v1,2*ll,0,0,0)*CG::wigner_3j(2,2*ll+2*v2,2*ll,0,0,0);
    
    // double threeJ1 = CG::wigner_3j(2*l1,2*ll+2*v1,2*l2p,0,0,0);
    // double threeJ2 = CG::wigner_3j(2*l3,2*ll+2*v2,2*l4p,0,0,0);
    // int Lmin = max(abs(ll+v1-1),0), Lmax = ll+v1+1, Lpmin = max(abs(ll+v2-1),0), Lpmax = ll+v2+1;
    // for(int MM = -ll; MM <= ll; MM++)
    // {
    //     double tmpa, tmpb, tmpL = 0.0, tmpLp = 0.0;
    //     for(int LL = Lmin; LL <= Lmax; LL++)
    //     {
    //         tmpa = 0.0;
    //         for(int aa = -1; aa <= 1; aa++)
    //             tmpa += get_N_coeff(v1,aa,ll,-MM)*CG::wigner_3j(2,2*ll+2*v1,2*LL,-2*aa,-2*MM+2*aa,2*MM);
    //         tmpL += tmpa*sqrt(2.0*LL+1)*CG::wigner_3j(two_j1,2*LL,two_j2,-two_m1,-2*MM,two_m2)*int2e_get_angularX_RME(two_j1,l1,two_j2,l2p,LL,ll+v1,threeJ1);
    //     }
    //     for(int LP = Lpmin; LP<=Lpmax; LP++)
    //     {
    //         tmpb = 0.0;
    //         for(int bb = -1; bb <= 1; bb++)
    //             tmpb += get_N_coeff(v2,bb,ll,MM)*CG::wigner_3j(2,2*ll+2*v2,2*LP,-2*bb,2*MM+2*bb,-2*MM);
    //         tmpLp += tmpb*sqrt(2.0*LP+1)*CG::wigner_3j(two_j3,2*LP,two_j4,-two_m3,2*MM,two_m4)*int2e_get_angularX_RME(two_j3,l3,two_j4,l4p,LP,ll+v2,threeJ2);
    //     }
    //     angular += pow(-1,MM)*tmpL*tmpLp;
    // }
    
    // return angular*pow(-1,(two_j1+two_j3-two_m1-two_m3)/2);
}
double INT_SPH::int2e_get_angular_gauge_LSSL(const int& l1, const int& two_m1, const int& s1, const int& l2, const int& two_m2, const int& s2, const int& l3, const int& two_m3, const int& s3, const int& l4, const int& two_m4, const int& s4, const int& ll, const int& v1, const int& v2) const
{
    if(ll+v1<0 || ll+v2<0) return 0.0;
    double angular = 0.0;
    int l2p = l2+s2, l3p = l3+s3;
    int two_j1 = 2*l1+s1, two_j2 = 2*l2+s2, two_j3 = 2*l3+s3, two_j4 = 2*l4+s4;
    double threeJ1 = CG::wigner_3j(2*l1,2*ll+2*v1,2*l2p,0,0,0), threeJ2 = CG::wigner_3j(2*l3p,2*ll+2*v2,2*l4,0,0,0), tmp;
    tmp = 0.0;
    double rme1 = int2e_get_angularX_RME(two_j1,l1,two_j2,l2p,ll,ll+v1,threeJ1);
    double rme2 = int2e_get_angularX_RME(two_j3,l3p,two_j4,l4,ll,ll+v2,threeJ2);
    for(int MMM = -ll; MMM <= ll; MMM++)
    {
        tmp += pow(-1,MMM) * CG::wigner_3j(two_j1,2*ll,two_j2,-two_m1,-2*MMM,two_m2)
             * CG::wigner_3j(two_j3,2*ll,two_j4,-two_m3,2*MMM,two_m4);
    }
    angular += tmp*rme1*rme2;
    
    return angular*pow(-1,(two_j1+two_j3-two_m1-two_m3)/2)*(2.0*(ll+v1)+1.0)*(2.0*(ll+v2)+1.0)/(2.0*ll+1)*CG::wigner_3j(2,2*ll+2*v1,2*ll,0,0,0)*CG::wigner_3j(2,2*ll+2*v2,2*ll,0,0,0);

    // double threeJ1 = CG::wigner_3j(2*l1,2*ll+2*v1,2*l2p,0,0,0);
    // double threeJ2 = CG::wigner_3j(2*l3p,2*ll+2*v2,2*l4,0,0,0);
    // int Lmin = max(abs(ll+v1-1),0), Lmax = ll+v1+1, Lpmin = max(abs(ll+v2-1),0), Lpmax = ll+v2+1;
    // for(int MM = -ll; MM <= ll; MM++)
    // {
    //     double tmpa, tmpb, tmpL = 0.0, tmpLp = 0.0;
    //     for(int LL = Lmin; LL <= Lmax; LL++)
    //     {
    //         tmpa = 0.0;
    //         for(int aa = -1; aa <= 1; aa++)
    //             tmpa += get_N_coeff(v1,aa,ll,-MM)*CG::wigner_3j(2,2*ll+2*v1,2*LL,-2*aa,-2*MM+2*aa,2*MM);
    //         tmpL += tmpa*sqrt(2.0*LL+1)*CG::wigner_3j(two_j1,2*LL,two_j2,-two_m1,-2*MM,two_m2)*int2e_get_angularX_RME(two_j1,l1,two_j2,l2p,LL,ll+v1,threeJ1);
    //     }
    //     for(int LP = Lpmin; LP<=Lpmax; LP++)
    //     {
    //         tmpb = 0.0;
    //         for(int bb = -1; bb <= 1; bb++)
    //             tmpb += get_N_coeff(v2,bb,ll,MM)*CG::wigner_3j(2,2*ll+2*v2,2*LP,-2*bb,2*MM+2*bb,-2*MM);
    //         tmpLp += tmpb*sqrt(2.0*LP+1)*CG::wigner_3j(two_j3,2*LP,two_j4,-two_m3,2*MM,two_m4)*int2e_get_angularX_RME(two_j3,l3p,two_j4,l4,LP,ll+v2,threeJ2);
    //     }
    //     angular += pow(-1,MM)*tmpL*tmpLp;
    // }
    
    // return angular*pow(-1,(two_j1+two_j3-two_m1-two_m3)/2);
}


int2eJK INT_SPH::get_h2e_JK_gauge(const string& intType, const int& occMaxL) const
{
    double time_a = 0.0, time_r = 0.0, time_c = 0.0;
    int occMaxShell = 0;
    if(occMaxL == -1)    occMaxShell = size_shell;
    else
    {
        for(int ii = 0; ii < size_shell; ii++)
        {
            if(shell_list[ii].l <= occMaxL)
                occMaxShell++;
            else
                break;
        }
    }
    
    int2eJK int_2e_JK;
    int_2e_JK.J = new double***[Nirrep];
    int_2e_JK.K = new double***[Nirrep];
    for(int ii = 0; ii < Nirrep; ii++)
    {
        int_2e_JK.J[ii] = new double**[Nirrep];
        int_2e_JK.K[ii] = new double**[Nirrep];
    }
    
    int int_tmp1_p = 0;
    for(int pshell = 0; pshell < occMaxShell; pshell++)
    {
    int l_p = shell_list[pshell].l, int_tmp1_q = 0;
    for(int qshell = 0; qshell < occMaxShell; qshell++)
    {
        int l_q = shell_list[qshell].l;
        int LmaxJ[4], LminJ[4], LmaxK[4], LminK[4];
        LmaxK[0] = l_p+l_q+2; LminJ[0] = 1; LminK[0] = 1;
        LmaxK[1] = l_p+l_q  ; LminJ[1] = 1; LminK[1] = 1;
        LmaxK[2] = l_p+l_q  ; LminJ[2] = 1; LminK[2] = 1;
        LmaxK[3] = l_p+l_q  ; LminJ[3] = 0; LminK[3] = 0;
        LmaxJ[0] = 1;
        LmaxJ[1] = 1;
        LmaxJ[2] = 1;
        LmaxJ[3] = 0;
        int size_gtos_p = shell_list[pshell].nunc, size_gtos_q = shell_list[qshell].nunc;
        int size_tmp_p = (l_p == 0) ? 1 : 2, size_tmp_q = (l_q == 0) ? 1 : 2;
        vector<double> array_angular_Jmm[LmaxJ[0]-LminJ[0]+1][size_tmp_p][size_tmp_q], array_angular_Kmm[LmaxK[0]-LminK[0]+1][size_tmp_p][size_tmp_q];
        vector<double> array_angular_Jmp[LmaxJ[1]-LminJ[1]+1][size_tmp_p][size_tmp_q], array_angular_Kmp[LmaxK[1]-LminK[1]+1][size_tmp_p][size_tmp_q];
        vector<double> array_angular_Jpm[LmaxJ[2]-LminJ[2]+1][size_tmp_p][size_tmp_q], array_angular_Kpm[LmaxK[2]-LminK[2]+1][size_tmp_p][size_tmp_q];
        vector<double> array_angular_Jpp[LmaxJ[3]-LminJ[3]+1][size_tmp_p][size_tmp_q], array_angular_Kpp[LmaxK[3]-LminK[3]+1][size_tmp_p][size_tmp_q];
        double radial_2e_list_Jmm[LmaxJ[0]-LminJ[0]+1][size_gtos_p*size_gtos_p*size_gtos_q*size_gtos_q][4];
        double radial_2e_list_Kmm[LmaxK[0]-LminK[0]+1][size_gtos_p*size_gtos_p*size_gtos_q*size_gtos_q][4];
        double radial_2e_list_Jmp[LmaxJ[1]-LminJ[1]+1][size_gtos_p*size_gtos_p*size_gtos_q*size_gtos_q][4];
        double radial_2e_list_Kmp[LmaxK[1]-LminK[1]+1][size_gtos_p*size_gtos_p*size_gtos_q*size_gtos_q][4];
        double radial_2e_list_Jpm[LmaxJ[2]-LminJ[2]+1][size_gtos_p*size_gtos_p*size_gtos_q*size_gtos_q][4];
        double radial_2e_list_Kpm[LmaxK[2]-LminK[2]+1][size_gtos_p*size_gtos_p*size_gtos_q*size_gtos_q][4];
        double radial_2e_list_Jpp[LmaxJ[3]-LminJ[3]+1][size_gtos_p*size_gtos_p*size_gtos_q*size_gtos_q][4];
        double radial_2e_list_Kpp[LmaxK[3]-LminK[3]+1][size_gtos_p*size_gtos_p*size_gtos_q*size_gtos_q][4];
        double array_radial_Jmm[LmaxJ[0]-LminJ[0]+1][size_gtos_p*size_gtos_p][size_gtos_q*size_gtos_q][size_tmp_p][size_tmp_q];
        double array_radial_Kmm[LmaxK[0]-LminK[0]+1][size_gtos_p*size_gtos_q][size_gtos_q*size_gtos_p][size_tmp_p][size_tmp_q];
        double array_radial_Jmp[LmaxJ[1]-LminJ[1]+1][size_gtos_p*size_gtos_p][size_gtos_q*size_gtos_q][size_tmp_p][size_tmp_q];
        double array_radial_Kmp[LmaxK[1]-LminK[1]+1][size_gtos_p*size_gtos_q][size_gtos_q*size_gtos_p][size_tmp_p][size_tmp_q];
        double array_radial_Jpm[LmaxJ[2]-LminJ[2]+1][size_gtos_p*size_gtos_p][size_gtos_q*size_gtos_q][size_tmp_p][size_tmp_q];
        double array_radial_Kpm[LmaxK[2]-LminK[2]+1][size_gtos_p*size_gtos_q][size_gtos_q*size_gtos_p][size_tmp_p][size_tmp_q];
        double array_radial_Jpp[LmaxJ[3]-LminJ[3]+1][size_gtos_p*size_gtos_p][size_gtos_q*size_gtos_q][size_tmp_p][size_tmp_q];
        double array_radial_Kpp[LmaxK[3]-LminK[3]+1][size_gtos_p*size_gtos_q][size_gtos_q*size_gtos_p][size_tmp_p][size_tmp_q];

        countTime(StartTimeCPU,StartTimeWall);
        #pragma omp parallel  for
        for(int twojj_p = abs(2*l_p-1); twojj_p <= 2*l_p+1; twojj_p = twojj_p + 2)
        for(int twojj_q = abs(2*l_q-1); twojj_q <= 2*l_q+1; twojj_q = twojj_q + 2)
        {
            int sym_ap = twojj_p - 2*l_p, sym_aq = twojj_q - 2*l_q;
            int int_tmp2_p = (twojj_p - abs(2*l_p-1)) / 2, int_tmp2_q = (twojj_q - abs(2*l_q-1))/2;

            // Angular
            for(int LL = LmaxJ[0]-LminJ[0]; LL >= 0; LL-=2)
            {
                array_angular_Jmm[LL][int_tmp2_p][int_tmp2_q].resize((twojj_p+1)*(twojj_q+1));
                for(int mp = 0; mp < twojj_p + 1; mp++)
                for(int mq = 0; mq < twojj_q + 1; mq++)
                {
                    if(intType.substr(0,4) == "LSLS")
                        array_angular_Jmm[LL][int_tmp2_p][int_tmp2_q][mp*(twojj_q + 1)+mq] = int2e_get_angular_gauge_LSLS(l_p, 2*mp-twojj_p, sym_ap, l_p, 2*mp-twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, LL+LminJ[0], -1, -1);
                    else if(intType.substr(0,4) == "LSSL")
                        array_angular_Jmm[LL][int_tmp2_p][int_tmp2_q][mp*(twojj_q + 1)+mq] = int2e_get_angular_gauge_LSSL(l_p, 2*mp-twojj_p, sym_ap, l_p, 2*mp-twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, LL+LminJ[0], -1, -1);
                    else
                    {
                        cout << "ERROR: Unkonwn intType in get_h2e_JK_gauge." << endl;
                        exit(99);
                    }
                }
            }
            for(int LL = LmaxK[0]-LminK[0]; LL >= 0; LL-=2)
            {
                array_angular_Kmm[LL][int_tmp2_p][int_tmp2_q].resize((twojj_p+1)*(twojj_q+1));
                for(int mp = 0; mp < twojj_p + 1; mp++)
                for(int mq = 0; mq < twojj_q + 1; mq++)
                {
                    if(intType.substr(0,4) == "LSLS")
                        array_angular_Kmm[LL][int_tmp2_p][int_tmp2_q][mp*(twojj_q + 1)+mq] = int2e_get_angular_gauge_LSLS(l_p, 2*mp-twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, l_p, 2*mp-twojj_p, sym_ap, LL+LminK[0], -1, -1);
                    else if(intType.substr(0,4) == "LSSL")
                        array_angular_Kmm[LL][int_tmp2_p][int_tmp2_q][mp*(twojj_q + 1)+mq] = int2e_get_angular_gauge_LSSL(l_p, 2*mp-twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, l_p, 2*mp-twojj_p, sym_ap, LL+LminK[0], -1, -1);
                    else
                    {
                        cout << "ERROR: Unkonwn intType in get_h2e_JK_gauge." << endl;
                        exit(99);
                    }
                }
            }
            for(int LL = LmaxJ[1]-LminJ[1]; LL >= 0; LL-=2)
            {
                array_angular_Jmp[LL][int_tmp2_p][int_tmp2_q].resize((twojj_p+1)*(twojj_q+1));
                for(int mp = 0; mp < twojj_p + 1; mp++)
                for(int mq = 0; mq < twojj_q + 1; mq++)
                {
                    if(intType.substr(0,4) == "LSLS")
                        array_angular_Jmp[LL][int_tmp2_p][int_tmp2_q][mp*(twojj_q + 1)+mq] = int2e_get_angular_gauge_LSLS(l_p, 2*mp-twojj_p, sym_ap, l_p, 2*mp-twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, LL+LminJ[1], -1, 1);
                    else if(intType.substr(0,4) == "LSSL")
                        array_angular_Jmp[LL][int_tmp2_p][int_tmp2_q][mp*(twojj_q + 1)+mq] = int2e_get_angular_gauge_LSSL(l_p, 2*mp-twojj_p, sym_ap, l_p, 2*mp-twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, LL+LminJ[1], -1, 1);
                    else
                    {
                        cout << "ERROR: Unkonwn intType in get_h2e_JK_gauge." << endl;
                        exit(99);
                    }
                }
            }
            for(int LL = LmaxK[1]-LminK[1]; LL >= 0; LL-=2)
            {
                array_angular_Kmp[LL][int_tmp2_p][int_tmp2_q].resize((twojj_p+1)*(twojj_q+1));
                for(int mp = 0; mp < twojj_p + 1; mp++)
                for(int mq = 0; mq < twojj_q + 1; mq++)
                {
                    if(intType.substr(0,4) == "LSLS")
                        array_angular_Kmp[LL][int_tmp2_p][int_tmp2_q][mp*(twojj_q + 1)+mq] = int2e_get_angular_gauge_LSLS(l_p, 2*mp-twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, l_p, 2*mp-twojj_p, sym_ap, LL+LminK[1], -1, 1);
                    else if(intType.substr(0,4) == "LSSL")
                        array_angular_Kmp[LL][int_tmp2_p][int_tmp2_q][mp*(twojj_q + 1)+mq] = int2e_get_angular_gauge_LSSL(l_p, 2*mp-twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, l_p, 2*mp-twojj_p, sym_ap, LL+LminK[1], -1, 1);
                    else
                    {
                        cout << "ERROR: Unkonwn intType in get_h2e_JK_gauge." << endl;
                        exit(99);
                    }
                }
            }
            for(int LL = LmaxJ[2]-LminJ[2]; LL >= 0; LL-=2)
            {
                array_angular_Jpm[LL][int_tmp2_p][int_tmp2_q].resize((twojj_p+1)*(twojj_q+1));
                for(int mp = 0; mp < twojj_p + 1; mp++)
                for(int mq = 0; mq < twojj_q + 1; mq++)
                {
                    if(intType.substr(0,4) == "LSLS")
                        array_angular_Jpm[LL][int_tmp2_p][int_tmp2_q][mp*(twojj_q + 1)+mq] = int2e_get_angular_gauge_LSLS(l_p, 2*mp-twojj_p, sym_ap, l_p, 2*mp-twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, LL+LminJ[2], 1, -1);
                    else if(intType.substr(0,4) == "LSSL")
                        array_angular_Jpm[LL][int_tmp2_p][int_tmp2_q][mp*(twojj_q + 1)+mq] = int2e_get_angular_gauge_LSSL(l_p, 2*mp-twojj_p, sym_ap, l_p, 2*mp-twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, LL+LminJ[2], 1, -1);
                    else
                    {
                        cout << "ERROR: Unkonwn intType in get_h2e_JK_gauge." << endl;
                        exit(99);
                    }
                }
            }
            for(int LL = LmaxK[2]-LminK[2]; LL >= 0; LL-=2)
            {
                array_angular_Kpm[LL][int_tmp2_p][int_tmp2_q].resize((twojj_p+1)*(twojj_q+1));
                for(int mp = 0; mp < twojj_p + 1; mp++)
                for(int mq = 0; mq < twojj_q + 1; mq++)
                {
                    if(intType.substr(0,4) == "LSLS")
                        array_angular_Kpm[LL][int_tmp2_p][int_tmp2_q][mp*(twojj_q + 1)+mq] = int2e_get_angular_gauge_LSLS(l_p, 2*mp-twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, l_p, 2*mp-twojj_p, sym_ap, LL+LminK[2], 1, -1);
                    else if(intType.substr(0,4) == "LSSL")
                        array_angular_Kpm[LL][int_tmp2_p][int_tmp2_q][mp*(twojj_q + 1)+mq] = int2e_get_angular_gauge_LSSL(l_p, 2*mp-twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, l_p, 2*mp-twojj_p, sym_ap, LL+LminK[2], 1, -1);
                    else
                    {
                        cout << "ERROR: Unkonwn intType in get_h2e_JK_gauge." << endl;
                        exit(99);
                    }
                }
            }
            for(int LL = LmaxJ[3]-LminJ[3]; LL >= 0; LL-=2)
            {
                array_angular_Jpp[LL][int_tmp2_p][int_tmp2_q].resize((twojj_p+1)*(twojj_q+1));
                for(int mp = 0; mp < twojj_p + 1; mp++)
                for(int mq = 0; mq < twojj_q + 1; mq++)
                {
                    if(intType.substr(0,4) == "LSLS")
                        array_angular_Jpp[LL][int_tmp2_p][int_tmp2_q][mp*(twojj_q + 1)+mq] = int2e_get_angular_gauge_LSLS(l_p, 2*mp-twojj_p, sym_ap, l_p, 2*mp-twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, LL+LminJ[3], 1, 1);
                    else if(intType.substr(0,4) == "LSSL")
                        array_angular_Jpp[LL][int_tmp2_p][int_tmp2_q][mp*(twojj_q + 1)+mq] = int2e_get_angular_gauge_LSSL(l_p, 2*mp-twojj_p, sym_ap, l_p, 2*mp-twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, LL+LminJ[3], 1, 1);
                    else
                    {
                        cout << "ERROR: Unkonwn intType in get_h2e_JK_gauge." << endl;
                        exit(99);
                    }
                }
            }
            for(int LL = LmaxK[3]-LminK[3]; LL >= 0; LL-=2)
            {
                array_angular_Kpp[LL][int_tmp2_p][int_tmp2_q].resize((twojj_p+1)*(twojj_q+1));
                for(int mp = 0; mp < twojj_p + 1; mp++)
                for(int mq = 0; mq < twojj_q + 1; mq++)
                {
                    if(intType.substr(0,4) == "LSLS")
                        array_angular_Kpp[LL][int_tmp2_p][int_tmp2_q][mp*(twojj_q + 1)+mq] = int2e_get_angular_gauge_LSLS(l_p, 2*mp-twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, l_p, 2*mp-twojj_p, sym_ap, LL+LminK[3], 1, 1);
                    else if(intType.substr(0,4) == "LSSL")
                        array_angular_Kpp[LL][int_tmp2_p][int_tmp2_q][mp*(twojj_q + 1)+mq] = int2e_get_angular_gauge_LSSL(l_p, 2*mp-twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, l_p, 2*mp-twojj_p, sym_ap, LL+LminK[3], 1, 1);
                    else
                    {
                        cout << "ERROR: Unkonwn intType in get_h2e_JK_gauge." << endl;
                        exit(99);
                    }
                }
            }
        }
        countTime(EndTimeCPU,EndTimeWall);
        time_a += (EndTimeCPU - StartTimeCPU)/(double)CLOCKS_PER_SEC;

        countTime(StartTimeCPU,StartTimeWall);
        #pragma omp parallel  for
        for(int tt = 0; tt < size_gtos_p*size_gtos_p*size_gtos_q*size_gtos_q; tt++)
        {
            int e1J = tt/(size_gtos_q*size_gtos_q);
            int e2J = tt - e1J*(size_gtos_q*size_gtos_q);
            int ii = e1J/size_gtos_p, jj = e1J - ii*size_gtos_p;
            int kk = e2J/size_gtos_q, ll = e2J - kk*size_gtos_q;
            int e1K = ii*size_gtos_q+ll, e2K = kk*size_gtos_p+jj;
            double a_i_J = shell_list[pshell].exp_a[ii], a_j_J = shell_list[pshell].exp_a[jj], a_k_J = shell_list[qshell].exp_a[kk], a_l_J = shell_list[qshell].exp_a[ll];
            double a_i_K = shell_list[pshell].exp_a[ii], a_j_K = shell_list[qshell].exp_a[ll], a_k_K = shell_list[qshell].exp_a[kk], a_l_K = shell_list[pshell].exp_a[jj];
        
            if(intType.substr(0,4) == "LSLS")
            {
                for(int LL = LmaxJ[0]-LminJ[0]; LL >= 0; LL-=2)
                {
                    radial_2e_list_Jmm[LL][tt][0] = int2e_get_radial_gauge(l_p,a_i_J,l_p+1,a_j_J,l_q,a_k_J,l_q+1,a_l_J,LL+LminJ[0],-1,-1);
                    if(l_p != 0)
                        radial_2e_list_Jmm[LL][tt][1] = int2e_get_radial_gauge(l_p,a_i_J,l_p-1,a_j_J,l_q,a_k_J,l_q+1,a_l_J,LL+LminJ[0],-1,-1);
                    if(l_q != 0)
                        radial_2e_list_Jmm[LL][tt][2] = int2e_get_radial_gauge(l_p,a_i_J,l_p+1,a_j_J,l_q,a_k_J,l_q-1,a_l_J,LL+LminJ[0],-1,-1);
                    if(l_p != 0 && l_q != 0)
                        radial_2e_list_Jmm[LL][tt][3] = int2e_get_radial_gauge(l_p,a_i_J,l_p-1,a_j_J,l_q,a_k_J,l_q-1,a_l_J,LL+LminJ[0],-1,-1);
                }
                for(int LL = LmaxK[0]-LminK[0]; LL >= 0; LL-=2)
                {
                    radial_2e_list_Kmm[LL][tt][0] = int2e_get_radial_gauge(l_p,a_i_K,l_q+1,a_j_K,l_q,a_k_K,l_p+1,a_l_K,LL+LminK[0],-1,-1);
                    if(l_q != 0)
                        radial_2e_list_Kmm[LL][tt][1] = int2e_get_radial_gauge(l_p,a_i_K,l_q-1,a_j_K,l_q,a_k_K,l_p+1,a_l_K,LL+LminK[0],-1,-1);
                    if(l_p != 0)
                        radial_2e_list_Kmm[LL][tt][2] = int2e_get_radial_gauge(l_p,a_i_K,l_q+1,a_j_K,l_q,a_k_K,l_p-1,a_l_K,LL+LminK[0],-1,-1);
                    if(l_p != 0 && l_q != 0)
                        radial_2e_list_Kmm[LL][tt][3] = int2e_get_radial_gauge(l_p,a_i_K,l_q-1,a_j_K,l_q,a_k_K,l_p-1,a_l_K,LL+LminK[0],-1,-1);
                }
                for(int LL = LmaxJ[1]-LminJ[1]; LL >= 0; LL-=2)
                {
                    radial_2e_list_Jmp[LL][tt][0] = int2e_get_radial_gauge(l_p,a_i_J,l_p+1,a_j_J,l_q,a_k_J,l_q+1,a_l_J,LL+LminJ[1],-1,1);
                    if(l_p != 0)
                        radial_2e_list_Jmp[LL][tt][1] = int2e_get_radial_gauge(l_p,a_i_J,l_p-1,a_j_J,l_q,a_k_J,l_q+1,a_l_J,LL+LminJ[1],-1,1);
                    if(l_q != 0)
                        radial_2e_list_Jmp[LL][tt][2] = int2e_get_radial_gauge(l_p,a_i_J,l_p+1,a_j_J,l_q,a_k_J,l_q-1,a_l_J,LL+LminJ[1],-1,1);
                    if(l_p != 0 && l_q != 0)
                        radial_2e_list_Jmp[LL][tt][3] = int2e_get_radial_gauge(l_p,a_i_J,l_p-1,a_j_J,l_q,a_k_J,l_q-1,a_l_J,LL+LminJ[1],-1,1);
                }
                for(int LL = LmaxK[1]-LminK[1]; LL >= 0; LL-=2)
                {
                    radial_2e_list_Kmp[LL][tt][0] = int2e_get_radial_gauge(l_p,a_i_K,l_q+1,a_j_K,l_q,a_k_K,l_p+1,a_l_K,LL+LminK[1],-1,1);
                    if(l_q != 0)
                        radial_2e_list_Kmp[LL][tt][1] = int2e_get_radial_gauge(l_p,a_i_K,l_q-1,a_j_K,l_q,a_k_K,l_p+1,a_l_K,LL+LminK[1],-1,1);
                    if(l_p != 0)
                        radial_2e_list_Kmp[LL][tt][2] = int2e_get_radial_gauge(l_p,a_i_K,l_q+1,a_j_K,l_q,a_k_K,l_p-1,a_l_K,LL+LminK[1],-1,1);
                    if(l_p != 0 && l_q != 0)
                        radial_2e_list_Kmp[LL][tt][3] = int2e_get_radial_gauge(l_p,a_i_K,l_q-1,a_j_K,l_q,a_k_K,l_p-1,a_l_K,LL+LminK[1],-1,1);
                }
                for(int LL = LmaxJ[2]-LminJ[2]; LL >= 0; LL-=2)
                {
                    radial_2e_list_Jpm[LL][tt][0] = int2e_get_radial_gauge(l_p,a_i_J,l_p+1,a_j_J,l_q,a_k_J,l_q+1,a_l_J,LL+LminJ[2],1,-1);
                    if(l_p != 0)
                        radial_2e_list_Jpm[LL][tt][1] = int2e_get_radial_gauge(l_p,a_i_J,l_p-1,a_j_J,l_q,a_k_J,l_q+1,a_l_J,LL+LminJ[2],1,-1);
                    if(l_q != 0)
                        radial_2e_list_Jpm[LL][tt][2] = int2e_get_radial_gauge(l_p,a_i_J,l_p+1,a_j_J,l_q,a_k_J,l_q-1,a_l_J,LL+LminJ[2],1,-1);
                    if(l_p != 0 && l_q != 0)
                        radial_2e_list_Jpm[LL][tt][3] = int2e_get_radial_gauge(l_p,a_i_J,l_p-1,a_j_J,l_q,a_k_J,l_q-1,a_l_J,LL+LminJ[2],1,-1);
                }
                for(int LL = LmaxK[2]-LminK[2]; LL >= 0; LL-=2)
                {
                    radial_2e_list_Kpm[LL][tt][0] = int2e_get_radial_gauge(l_p,a_i_K,l_q+1,a_j_K,l_q,a_k_K,l_p+1,a_l_K,LL+LminK[2],1,-1);
                    if(l_q != 0)
                        radial_2e_list_Kpm[LL][tt][1] = int2e_get_radial_gauge(l_p,a_i_K,l_q-1,a_j_K,l_q,a_k_K,l_p+1,a_l_K,LL+LminK[2],1,-1);
                    if(l_p != 0)
                        radial_2e_list_Kpm[LL][tt][2] = int2e_get_radial_gauge(l_p,a_i_K,l_q+1,a_j_K,l_q,a_k_K,l_p-1,a_l_K,LL+LminK[2],1,-1);
                    if(l_p != 0 && l_q != 0)
                        radial_2e_list_Kpm[LL][tt][3] = int2e_get_radial_gauge(l_p,a_i_K,l_q-1,a_j_K,l_q,a_k_K,l_p-1,a_l_K,LL+LminK[2],1,-1);
                }
                for(int LL = LmaxJ[3]-LminJ[3]; LL >= 0; LL-=2)
                {
                    radial_2e_list_Jpp[LL][tt][0] = int2e_get_radial_gauge(l_p,a_i_J,l_p+1,a_j_J,l_q,a_k_J,l_q+1,a_l_J,LL+LminJ[3],1,1);
                    if(l_p != 0)
                        radial_2e_list_Jpp[LL][tt][1] = int2e_get_radial_gauge(l_p,a_i_J,l_p-1,a_j_J,l_q,a_k_J,l_q+1,a_l_J,LL+LminJ[3],1,1);
                    if(l_q != 0)
                        radial_2e_list_Jpp[LL][tt][2] = int2e_get_radial_gauge(l_p,a_i_J,l_p+1,a_j_J,l_q,a_k_J,l_q-1,a_l_J,LL+LminJ[3],1,1);
                    if(l_p != 0 && l_q != 0)
                        radial_2e_list_Jpp[LL][tt][3] = int2e_get_radial_gauge(l_p,a_i_J,l_p-1,a_j_J,l_q,a_k_J,l_q-1,a_l_J,LL+LminJ[3],1,1);
                }
                for(int LL = LmaxK[3]-LminK[3]; LL >= 0; LL-=2)
                {
                    radial_2e_list_Kpp[LL][tt][0] = int2e_get_radial_gauge(l_p,a_i_K,l_q+1,a_j_K,l_q,a_k_K,l_p+1,a_l_K,LL+LminK[3],1,1);
                    if(l_q != 0)
                        radial_2e_list_Kpp[LL][tt][1] = int2e_get_radial_gauge(l_p,a_i_K,l_q-1,a_j_K,l_q,a_k_K,l_p+1,a_l_K,LL+LminK[3],1,1);
                    if(l_p != 0)
                        radial_2e_list_Kpp[LL][tt][2] = int2e_get_radial_gauge(l_p,a_i_K,l_q+1,a_j_K,l_q,a_k_K,l_p-1,a_l_K,LL+LminK[3],1,1);
                    if(l_p != 0 && l_q != 0)
                        radial_2e_list_Kpp[LL][tt][3] = int2e_get_radial_gauge(l_p,a_i_K,l_q-1,a_j_K,l_q,a_k_K,l_p-1,a_l_K,LL+LminK[3],1,1);
                }
            }
            else if(intType.substr(0,4) == "LSSL")
            {
                for(int LL = LmaxJ[0]-LminJ[0]; LL >= 0; LL-=2)
                {
                    radial_2e_list_Jmm[LL][tt][0] = int2e_get_radial_gauge(l_p,a_i_J,l_p+1,a_j_J,l_q+1,a_k_J,l_q,a_l_J,LL+LminJ[0],-1,-1);
                    if(l_p != 0)
                        radial_2e_list_Jmm[LL][tt][1] = int2e_get_radial_gauge(l_p,a_i_J,l_p-1,a_j_J,l_q+1,a_k_J,l_q,a_l_J,LL+LminJ[0],-1,-1);
                    if(l_q != 0)
                        radial_2e_list_Jmm[LL][tt][2] = int2e_get_radial_gauge(l_p,a_i_J,l_p+1,a_j_J,l_q-1,a_k_J,l_q,a_l_J,LL+LminJ[0],-1,-1);
                    if(l_p != 0 && l_q != 0)
                        radial_2e_list_Jmm[LL][tt][3] = int2e_get_radial_gauge(l_p,a_i_J,l_p-1,a_j_J,l_q-1,a_k_J,l_q,a_l_J,LL+LminJ[0],-1,-1);
                }
                for(int LL = LmaxK[0]-LminK[0]; LL >= 0; LL-=2)
                {
                    radial_2e_list_Kmm[LL][tt][0] = int2e_get_radial_gauge(l_p,a_i_K,l_q+1,a_j_K,l_q+1,a_k_K,l_p,a_l_K,LL+LminK[0],-1,-1);
                    if(l_q != 0)
                    {
                        radial_2e_list_Kmm[LL][tt][1] = int2e_get_radial_gauge(l_p,a_i_K,l_q-1,a_j_K,l_q+1,a_k_K,l_p,a_l_K,LL+LminK[0],-1,-1);
                        radial_2e_list_Kmm[LL][tt][2] = int2e_get_radial_gauge(l_p,a_i_K,l_q+1,a_j_K,l_q-1,a_k_K,l_p,a_l_K,LL+LminK[0],-1,-1);
                        radial_2e_list_Kmm[LL][tt][3] = int2e_get_radial_gauge(l_p,a_i_K,l_q-1,a_j_K,l_q-1,a_k_K,l_p,a_l_K,LL+LminK[0],-1,-1);
                    }
                }
                for(int LL = LmaxJ[1]-LminJ[1]; LL >= 0; LL-=2)
                {
                    radial_2e_list_Jmp[LL][tt][0] = int2e_get_radial_gauge(l_p,a_i_J,l_p+1,a_j_J,l_q+1,a_k_J,l_q,a_l_J,LL+LminJ[1],-1,1);
                    if(l_p != 0)
                        radial_2e_list_Jmp[LL][tt][1] = int2e_get_radial_gauge(l_p,a_i_J,l_p-1,a_j_J,l_q+1,a_k_J,l_q,a_l_J,LL+LminJ[1],-1,1);
                    if(l_q != 0)
                        radial_2e_list_Jmp[LL][tt][2] = int2e_get_radial_gauge(l_p,a_i_J,l_p+1,a_j_J,l_q-1,a_k_J,l_q,a_l_J,LL+LminJ[1],-1,1);
                    if(l_p != 0 && l_q != 0)
                        radial_2e_list_Jmp[LL][tt][3] = int2e_get_radial_gauge(l_p,a_i_J,l_p-1,a_j_J,l_q-1,a_k_J,l_q,a_l_J,LL+LminJ[1],-1,1);
                }
                for(int LL = LmaxK[1]-LminK[1]; LL >= 0; LL-=2)
                {
                    radial_2e_list_Kmp[LL][tt][0] = int2e_get_radial_gauge(l_p,a_i_K,l_q+1,a_j_K,l_q+1,a_k_K,l_p,a_l_K,LL+LminK[1],-1,1);
                    if(l_q != 0)
                    {
                        radial_2e_list_Kmp[LL][tt][1] = int2e_get_radial_gauge(l_p,a_i_K,l_q-1,a_j_K,l_q+1,a_k_K,l_p,a_l_K,LL+LminK[1],-1,1);
                        radial_2e_list_Kmp[LL][tt][2] = int2e_get_radial_gauge(l_p,a_i_K,l_q+1,a_j_K,l_q-1,a_k_K,l_p,a_l_K,LL+LminK[1],-1,1);
                        radial_2e_list_Kmp[LL][tt][3] = int2e_get_radial_gauge(l_p,a_i_K,l_q-1,a_j_K,l_q-1,a_k_K,l_p,a_l_K,LL+LminK[1],-1,1);
                    }
                }
                for(int LL = LmaxJ[2]-LminJ[2]; LL >= 0; LL-=2)
                {
                    radial_2e_list_Jpm[LL][tt][0] = int2e_get_radial_gauge(l_p,a_i_J,l_p+1,a_j_J,l_q+1,a_k_J,l_q,a_l_J,LL+LminJ[2],1,-1);
                    if(l_p != 0)
                        radial_2e_list_Jpm[LL][tt][1] = int2e_get_radial_gauge(l_p,a_i_J,l_p-1,a_j_J,l_q+1,a_k_J,l_q,a_l_J,LL+LminJ[2],1,-1);
                    if(l_q != 0)
                        radial_2e_list_Jpm[LL][tt][2] = int2e_get_radial_gauge(l_p,a_i_J,l_p+1,a_j_J,l_q-1,a_k_J,l_q,a_l_J,LL+LminJ[2],1,-1);
                    if(l_p != 0 && l_q != 0)
                        radial_2e_list_Jpm[LL][tt][3] = int2e_get_radial_gauge(l_p,a_i_J,l_p-1,a_j_J,l_q-1,a_k_J,l_q,a_l_J,LL+LminJ[2],1,-1);
                }
                for(int LL = LmaxK[2]-LminK[2]; LL >= 0; LL-=2)
                {
                    radial_2e_list_Kpm[LL][tt][0] = int2e_get_radial_gauge(l_p,a_i_K,l_q+1,a_j_K,l_q+1,a_k_K,l_p,a_l_K,LL+LminK[2],1,-1);
                    if(l_q != 0)
                    {
                        radial_2e_list_Kpm[LL][tt][1] = int2e_get_radial_gauge(l_p,a_i_K,l_q-1,a_j_K,l_q+1,a_k_K,l_p,a_l_K,LL+LminK[2],1,-1);
                        radial_2e_list_Kpm[LL][tt][2] = int2e_get_radial_gauge(l_p,a_i_K,l_q+1,a_j_K,l_q-1,a_k_K,l_p,a_l_K,LL+LminK[2],1,-1);
                        radial_2e_list_Kpm[LL][tt][3] = int2e_get_radial_gauge(l_p,a_i_K,l_q-1,a_j_K,l_q-1,a_k_K,l_p,a_l_K,LL+LminK[2],1,-1);
                    }
                }
                for(int LL = LmaxJ[3]-LminJ[3]; LL >= 0; LL-=2)
                {
                    radial_2e_list_Jpp[LL][tt][0] = int2e_get_radial_gauge(l_p,a_i_J,l_p+1,a_j_J,l_q+1,a_k_J,l_q,a_l_J,LL+LminJ[3],1,1);
                    if(l_p != 0)
                        radial_2e_list_Jpp[LL][tt][1] = int2e_get_radial_gauge(l_p,a_i_J,l_p-1,a_j_J,l_q+1,a_k_J,l_q,a_l_J,LL+LminJ[3],1,1);
                    if(l_q != 0)
                        radial_2e_list_Jpp[LL][tt][2] = int2e_get_radial_gauge(l_p,a_i_J,l_p+1,a_j_J,l_q-1,a_k_J,l_q,a_l_J,LL+LminJ[3],1,1);
                    if(l_p != 0 && l_q != 0)
                        radial_2e_list_Jpp[LL][tt][3] = int2e_get_radial_gauge(l_p,a_i_J,l_p-1,a_j_J,l_q-1,a_k_J,l_q,a_l_J,LL+LminJ[3],1,1);
                }
                for(int LL = LmaxK[3]-LminK[3]; LL >= 0; LL-=2)
                {
                    radial_2e_list_Kpp[LL][tt][0] = int2e_get_radial_gauge(l_p,a_i_K,l_q+1,a_j_K,l_q+1,a_k_K,l_p,a_l_K,LL+LminK[3],1,1);
                    if(l_q != 0)
                    {
                        radial_2e_list_Kpp[LL][tt][1] = int2e_get_radial_gauge(l_p,a_i_K,l_q-1,a_j_K,l_q+1,a_k_K,l_p,a_l_K,LL+LminK[3],1,1);
                        radial_2e_list_Kpp[LL][tt][2] = int2e_get_radial_gauge(l_p,a_i_K,l_q+1,a_j_K,l_q-1,a_k_K,l_p,a_l_K,LL+LminK[3],1,1);
                        radial_2e_list_Kpp[LL][tt][3] = int2e_get_radial_gauge(l_p,a_i_K,l_q-1,a_j_K,l_q-1,a_k_K,l_p,a_l_K,LL+LminK[3],1,1);
                    }
                }
            }
            else
            {
                cout << "ERROR: Unkonwn intType in get_h2e_JK_gauge." << endl;
                exit(99);
            }
                
            for(int twojj_p = abs(2*l_p-1); twojj_p <= 2*l_p+1; twojj_p = twojj_p + 2)
            for(int twojj_q = abs(2*l_q-1); twojj_q <= 2*l_q+1; twojj_q = twojj_q + 2)
            {
                int index_tmp_p = (l_p > 0) ? 1 - (2*l_p+1 - twojj_p)/2 : 0;
                int index_tmp_q = (l_q > 0) ? 1 - (2*l_q+1 - twojj_q)/2 : 0;
                int sym_ap = twojj_p - 2*l_p, sym_aq = twojj_q - 2*l_q;
                double k_p = -(twojj_p+1.0)*sym_ap/2.0, k_q = -(twojj_q+1.0)*sym_aq/2.0;
                double norm_J = shell_list[pshell].norm[ii] * shell_list[pshell].norm[jj] * shell_list[qshell].norm[kk] * shell_list[qshell].norm[ll], norm_K = shell_list[pshell].norm[ii] * shell_list[qshell].norm[ll] * shell_list[qshell].norm[kk] * shell_list[pshell].norm[jj];
                double lk1 = 1+l_p+k_p, lk2 = 1+l_p+k_p, lk3 = 1+l_q+k_q, lk4 = 1+l_q+k_q, a1 = shell_list[pshell].exp_a[ii], a2 = shell_list[pshell].exp_a[jj], a3 = shell_list[qshell].exp_a[kk], a4 = shell_list[qshell].exp_a[ll];

                for(int LL = LmaxJ[0]-LminJ[0]; LL >= 0; LL-=2)
                {
                    if(intType == "LSLS")
                    {
                        array_radial_Jmm[LL][e1J][e2J][index_tmp_p][index_tmp_q] = get_radial_LSLS_J(l_p,l_q,LL+LminJ[0],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Jmm[LL][tt],false);
                        array_radial_Jmm[LL][e1J][e2J][index_tmp_p][index_tmp_q] /= -2.0 * norm_J * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "LSSL")
                    {
                        array_radial_Jmm[LL][e1J][e2J][index_tmp_p][index_tmp_q] = get_radial_LSSL_J(l_p,l_q,LL+LminJ[0],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Jmm[LL][tt],false);
                        array_radial_Jmm[LL][e1J][e2J][index_tmp_p][index_tmp_q] /= 2.0 * norm_J * 4.0 * pow(speedOfLight,2);
                    }
                }
                for(int LL = LmaxJ[1]-LminJ[1]; LL >= 0; LL-=2)
                {
                    if(intType == "LSLS")
                    {
                        array_radial_Jmp[LL][e1J][e2J][index_tmp_p][index_tmp_q] = get_radial_LSLS_J(l_p,l_q,LL+LminJ[1],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Jmp[LL][tt],false);
                        array_radial_Jmp[LL][e1J][e2J][index_tmp_p][index_tmp_q] /= -2.0 * norm_J * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "LSSL")
                    {
                        array_radial_Jmp[LL][e1J][e2J][index_tmp_p][index_tmp_q] = get_radial_LSSL_J(l_p,l_q,LL+LminJ[1],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Jmp[LL][tt],false);
                        array_radial_Jmp[LL][e1J][e2J][index_tmp_p][index_tmp_q] /= 2.0 * norm_J * 4.0 * pow(speedOfLight,2);
                    }
                }
                for(int LL = LmaxJ[2]-LminJ[2]; LL >= 0; LL-=2)
                {
                    if(intType == "LSLS")
                    {
                        array_radial_Jpm[LL][e1J][e2J][index_tmp_p][index_tmp_q] = get_radial_LSLS_J(l_p,l_q,LL+LminJ[2],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Jpm[LL][tt],false);
                        array_radial_Jpm[LL][e1J][e2J][index_tmp_p][index_tmp_q] /= -2.0 * norm_J * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "LSSL")
                    {
                        array_radial_Jpm[LL][e1J][e2J][index_tmp_p][index_tmp_q] = get_radial_LSSL_J(l_p,l_q,LL+LminJ[2],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Jpm[LL][tt],false);
                        array_radial_Jpm[LL][e1J][e2J][index_tmp_p][index_tmp_q] /= 2.0 * norm_J * 4.0 * pow(speedOfLight,2);
                    }
                }
                for(int LL = LmaxJ[3]-LminJ[3]; LL >= 0; LL-=2)
                {
                    if(intType == "LSLS")
                    {
                        array_radial_Jpp[LL][e1J][e2J][index_tmp_p][index_tmp_q] = get_radial_LSLS_J(l_p,l_q,LL+LminJ[3],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Jpp[LL][tt],false);
                        array_radial_Jpp[LL][e1J][e2J][index_tmp_p][index_tmp_q] /= -2.0 * norm_J * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "LSSL")
                    {
                        array_radial_Jpp[LL][e1J][e2J][index_tmp_p][index_tmp_q] = get_radial_LSSL_J(l_p,l_q,LL+LminJ[3],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Jpp[LL][tt],false);
                        array_radial_Jpp[LL][e1J][e2J][index_tmp_p][index_tmp_q] /= 2.0 * norm_J * 4.0 * pow(speedOfLight,2);
                    }
                }
                lk2 = 1+l_q+k_q; lk4 = 1+l_p+k_p; 
                a2 = shell_list[qshell].exp_a[ll]; a4 = shell_list[pshell].exp_a[jj];
                for(int LL = LmaxK[0]-LminK[0]; LL >= 0; LL-=2)
                {
                    if(intType == "LSLS")
                    {
                        array_radial_Kmm[LL][e1K][e2K][index_tmp_p][index_tmp_q] = get_radial_LSLS_K(l_p,l_q,LL+LminK[0],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Kmm[LL][tt],false);
                        array_radial_Kmm[LL][e1K][e2K][index_tmp_p][index_tmp_q] /= -2.0 * norm_K * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "LSSL")
                    {
                        array_radial_Kmm[LL][e1K][e2K][index_tmp_p][index_tmp_q] = get_radial_LSSL_K(l_p,l_q,LL+LminK[0],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Kmm[LL][tt],false);
                        array_radial_Kmm[LL][e1K][e2K][index_tmp_p][index_tmp_q] /= 2.0 * norm_K * 4.0 * pow(speedOfLight,2);
                    }
                }
                for(int LL = LmaxK[1]-LminK[1]; LL >= 0; LL-=2)
                {
                    if(intType == "LSLS")
                    {
                        array_radial_Kmp[LL][e1K][e2K][index_tmp_p][index_tmp_q] = get_radial_LSLS_K(l_p,l_q,LL+LminK[1],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Kmp[LL][tt],false);
                        array_radial_Kmp[LL][e1K][e2K][index_tmp_p][index_tmp_q] /= -2.0 * norm_K * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "LSSL")
                    {
                        array_radial_Kmp[LL][e1K][e2K][index_tmp_p][index_tmp_q] = get_radial_LSSL_K(l_p,l_q,LL+LminK[1],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Kmp[LL][tt],false);
                        array_radial_Kmp[LL][e1K][e2K][index_tmp_p][index_tmp_q] /= 2.0 * norm_K * 4.0 * pow(speedOfLight,2);
                    }
                }
                for(int LL = LmaxK[2]-LminK[2]; LL >= 0; LL-=2)
                {
                    if(intType == "LSLS")
                    {
                        array_radial_Kpm[LL][e1K][e2K][index_tmp_p][index_tmp_q] = get_radial_LSLS_K(l_p,l_q,LL+LminK[2],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Kpm[LL][tt],false);
                        array_radial_Kpm[LL][e1K][e2K][index_tmp_p][index_tmp_q] /= -2.0 * norm_K * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "LSSL")
                    {
                        array_radial_Kpm[LL][e1K][e2K][index_tmp_p][index_tmp_q] = get_radial_LSSL_K(l_p,l_q,LL+LminK[2],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Kpm[LL][tt],false);
                        array_radial_Kpm[LL][e1K][e2K][index_tmp_p][index_tmp_q] /= 2.0 * norm_K * 4.0 * pow(speedOfLight,2);
                    }
                }
                for(int LL = LmaxK[3]-LminK[3]; LL >= 0; LL-=2)
                {
                    if(intType == "LSLS")
                    {
                        array_radial_Kpp[LL][e1K][e2K][index_tmp_p][index_tmp_q] = get_radial_LSLS_K(l_p,l_q,LL+LminK[3],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Kpp[LL][tt],false);
                        array_radial_Kpp[LL][e1K][e2K][index_tmp_p][index_tmp_q] /= -2.0 * norm_K * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "LSSL")
                    {
                        array_radial_Kpp[LL][e1K][e2K][index_tmp_p][index_tmp_q] = get_radial_LSSL_K(l_p,l_q,LL+LminK[3],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Kpp[LL][tt],false);
                        array_radial_Kpp[LL][e1K][e2K][index_tmp_p][index_tmp_q] /= 2.0 * norm_K * 4.0 * pow(speedOfLight,2);
                    }
                }
            }
        }
        countTime(EndTimeCPU,EndTimeWall);
        time_r += (EndTimeCPU - StartTimeCPU)/(double)CLOCKS_PER_SEC;

        countTime(StartTimeCPU,StartTimeWall);
        int l_p_cycle = (l_p == 0) ? 1 : 2, l_q_cycle = (l_q == 0) ? 1 : 2;
        for(int int_tmp2_p = 0; int_tmp2_p < l_p_cycle; int_tmp2_p++)
        for(int int_tmp2_q = 0; int_tmp2_q < l_q_cycle; int_tmp2_q++)
        {
            int add_p = int_tmp2_p*(irrep_list[int_tmp1_p].two_j+1), add_q = int_tmp2_q*(irrep_list[int_tmp1_q].two_j+1);
            for(int mp = 0; mp < irrep_list[int_tmp1_p+add_p].two_j + 1; mp++)
            for(int mq = 0; mq < irrep_list[int_tmp1_q+add_q].two_j + 1; mq++)
            {
                int_2e_JK.J[int_tmp1_p+add_p + mp][int_tmp1_q + add_q + mq] = new double*[size_gtos_p*size_gtos_p];
                for(int iii = 0; iii < size_gtos_p*size_gtos_p; iii++)
                    int_2e_JK.J[int_tmp1_p+add_p + mp][int_tmp1_q + add_q + mq][iii] = new double[size_gtos_q*size_gtos_q];
                int_2e_JK.K[int_tmp1_p+add_p + mp][int_tmp1_q + add_q + mq] = new double*[size_gtos_p*size_gtos_q];
                for(int iii = 0; iii < size_gtos_p*size_gtos_q; iii++)
                    int_2e_JK.K[int_tmp1_p+add_p + mp][int_tmp1_q + add_q + mq][iii] = new double[size_gtos_q*size_gtos_p];
                #pragma omp parallel  for
                for(int tt = 0; tt < size_gtos_p*size_gtos_p*size_gtos_q*size_gtos_q; tt++)
                {
                    int e1J = tt/(size_gtos_q*size_gtos_q);
                    int e2J = tt - e1J*(size_gtos_q*size_gtos_q);
                    int e1K = tt/(size_gtos_p*size_gtos_q);
                    int e2K = tt - e1K*(size_gtos_p*size_gtos_q);
                    int_2e_JK.J[int_tmp1_p+add_p + mp][int_tmp1_q+add_q + mq][e1J][e2J] = 0.0;
                    int_2e_JK.K[int_tmp1_p+add_p + mp][int_tmp1_q+add_q + mq][e1K][e2K] = 0.0;
                    for(int tmp = LmaxJ[0]-LminJ[0]; tmp >= 0; tmp = tmp - 2)
                        int_2e_JK.J[int_tmp1_p+add_p + mp][int_tmp1_q+add_q + mq][e1J][e2J] += array_radial_Jmm[tmp][e1J][e2J][int_tmp2_p][int_tmp2_q] * array_angular_Jmm[tmp][int_tmp2_p][int_tmp2_q][mp*(irrep_list[int_tmp1_q+add_q].two_j + 1)+mq];
                    for(int tmp = LmaxJ[1]-LminJ[1]; tmp >= 0; tmp = tmp - 2)
                        int_2e_JK.J[int_tmp1_p+add_p + mp][int_tmp1_q+add_q + mq][e1J][e2J] += array_radial_Jmp[tmp][e1J][e2J][int_tmp2_p][int_tmp2_q] * array_angular_Jmp[tmp][int_tmp2_p][int_tmp2_q][mp*(irrep_list[int_tmp1_q+add_q].two_j + 1)+mq];
                    for(int tmp = LmaxJ[2]-LminJ[2]; tmp >= 0; tmp = tmp - 2)
                        int_2e_JK.J[int_tmp1_p+add_p + mp][int_tmp1_q+add_q + mq][e1J][e2J] += array_radial_Jpm[tmp][e1J][e2J][int_tmp2_p][int_tmp2_q] * array_angular_Jpm[tmp][int_tmp2_p][int_tmp2_q][mp*(irrep_list[int_tmp1_q+add_q].two_j + 1)+mq];
                    for(int tmp = LmaxJ[3]-LminJ[3]; tmp >= 0; tmp = tmp - 2)
                        int_2e_JK.J[int_tmp1_p+add_p + mp][int_tmp1_q+add_q + mq][e1J][e2J] += array_radial_Jpp[tmp][e1J][e2J][int_tmp2_p][int_tmp2_q] * array_angular_Jpp[tmp][int_tmp2_p][int_tmp2_q][mp*(irrep_list[int_tmp1_q+add_q].two_j + 1)+mq];
                    
                    for(int tmp = LmaxK[0]-LminK[0]; tmp >= 0; tmp = tmp - 2)
                        int_2e_JK.K[int_tmp1_p+add_p + mp][int_tmp1_q+add_q + mq][e1K][e2K] += array_radial_Kmm[tmp][e1K][e2K][int_tmp2_p][int_tmp2_q] * array_angular_Kmm[tmp][int_tmp2_p][int_tmp2_q][mp*(irrep_list[int_tmp1_q+add_q].two_j + 1)+mq];
                    for(int tmp = LmaxK[1]-LminK[1]; tmp >= 0; tmp = tmp - 2)
                        int_2e_JK.K[int_tmp1_p+add_p + mp][int_tmp1_q+add_q + mq][e1K][e2K] += array_radial_Kmp[tmp][e1K][e2K][int_tmp2_p][int_tmp2_q] * array_angular_Kmp[tmp][int_tmp2_p][int_tmp2_q][mp*(irrep_list[int_tmp1_q+add_q].two_j + 1)+mq];
                    for(int tmp = LmaxK[2]-LminK[2]; tmp >= 0; tmp = tmp - 2)
                        int_2e_JK.K[int_tmp1_p+add_p + mp][int_tmp1_q+add_q + mq][e1K][e2K] += array_radial_Kpm[tmp][e1K][e2K][int_tmp2_p][int_tmp2_q] * array_angular_Kpm[tmp][int_tmp2_p][int_tmp2_q][mp*(irrep_list[int_tmp1_q+add_q].two_j + 1)+mq];
                    for(int tmp = LmaxK[3]-LminK[3]; tmp >= 0; tmp = tmp - 2)
                        int_2e_JK.K[int_tmp1_p+add_p + mp][int_tmp1_q+add_q + mq][e1K][e2K] += array_radial_Kpp[tmp][e1K][e2K][int_tmp2_p][int_tmp2_q] * array_angular_Kpp[tmp][int_tmp2_p][int_tmp2_q][mp*(irrep_list[int_tmp1_q+add_q].two_j + 1)+mq];
                }
            }
        }
        countTime(EndTimeCPU,EndTimeWall);
        time_c += (EndTimeCPU - StartTimeCPU)/(double)CLOCKS_PER_SEC;
        int_tmp1_q += 4*l_q+2;
    }
    int_tmp1_p += 4*l_p+2;
    }

    cout << time_a << "\t" << time_r << "\t" << time_c <<endl;
    return int_2e_JK;
}

int2eJK INT_SPH::get_h2e_JK_gauge_compact(const string& intType, const int& occMaxL) const
{
    int occMaxShell = 0, Nirrep_compact = 0;
    if(occMaxL == -1)    occMaxShell = size_shell;
    else
    {
        for(int ii = 0; ii < size_shell; ii++)
        {
            if(shell_list[ii].l <= occMaxL)
                occMaxShell++;
            else
                break;
        }
    }
    for(int ii = 0; ii < occMaxShell; ii++)
    {
        if(shell_list[ii].l == 0) Nirrep_compact += 1;
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
    
    int int_tmp1_p = 0;
    for(int pshell = 0; pshell < occMaxShell; pshell++)
    {
    int l_p = shell_list[pshell].l, int_tmp1_q = 0;
    for(int qshell = 0; qshell < occMaxShell; qshell++)
    {
        int l_q = shell_list[qshell].l;
        int LmaxJ[4], LminJ[4], LmaxK[4], LminK[4];
        LmaxK[0] = l_p+l_q+2; LminJ[0] = 1; LminK[0] = 1;
        LmaxK[1] = l_p+l_q  ; LminJ[1] = 1; LminK[1] = 1;
        LmaxK[2] = l_p+l_q  ; LminJ[2] = 1; LminK[2] = 1;
        LmaxK[3] = l_p+l_q  ; LminJ[3] = 0; LminK[3] = 0;
        LmaxJ[0] = 1;
        LmaxJ[1] = 1;
        LmaxJ[2] = 1;
        LmaxJ[3] = 0;
        double radial;
        int size_gtos_p = shell_list[pshell].nunc, size_gtos_q = shell_list[qshell].nunc;
        int size_tmp_p = (l_p == 0) ? 1 : 2, size_tmp_q = (l_q == 0) ? 1 : 2;
        double array_angular_Jmm[LmaxJ[0]-LminJ[0]+1][size_tmp_p][size_tmp_q], array_angular_Kmm[LmaxK[0]-LminK[0]+1][size_tmp_p][size_tmp_q];
        double array_angular_Jmp[LmaxJ[1]-LminJ[1]+1][size_tmp_p][size_tmp_q], array_angular_Kmp[LmaxK[1]-LminK[1]+1][size_tmp_p][size_tmp_q];
        double array_angular_Jpm[LmaxJ[2]-LminJ[2]+1][size_tmp_p][size_tmp_q], array_angular_Kpm[LmaxK[2]-LminK[2]+1][size_tmp_p][size_tmp_q];
        double array_angular_Jpp[LmaxJ[3]-LminJ[3]+1][size_tmp_p][size_tmp_q], array_angular_Kpp[LmaxK[3]-LminK[3]+1][size_tmp_p][size_tmp_q];
        double radial_2e_list_Jmm[LmaxJ[0]-LminJ[0]+1][size_gtos_p*size_gtos_p*size_gtos_q*size_gtos_q][4];
        double radial_2e_list_Kmm[LmaxK[0]-LminK[0]+1][size_gtos_p*size_gtos_p*size_gtos_q*size_gtos_q][4];
        double radial_2e_list_Jmp[LmaxJ[1]-LminJ[1]+1][size_gtos_p*size_gtos_p*size_gtos_q*size_gtos_q][4];
        double radial_2e_list_Kmp[LmaxK[1]-LminK[1]+1][size_gtos_p*size_gtos_p*size_gtos_q*size_gtos_q][4];
        double radial_2e_list_Jpm[LmaxJ[2]-LminJ[2]+1][size_gtos_p*size_gtos_p*size_gtos_q*size_gtos_q][4];
        double radial_2e_list_Kpm[LmaxK[2]-LminK[2]+1][size_gtos_p*size_gtos_p*size_gtos_q*size_gtos_q][4];
        double radial_2e_list_Jpp[LmaxJ[3]-LminJ[3]+1][size_gtos_p*size_gtos_p*size_gtos_q*size_gtos_q][4];
        double radial_2e_list_Kpp[LmaxK[3]-LminK[3]+1][size_gtos_p*size_gtos_p*size_gtos_q*size_gtos_q][4];

        #pragma omp parallel  for
        for(int tt = 0; tt < size_gtos_p*size_gtos_p*size_gtos_q*size_gtos_q; tt++)
        {
            int e1J = tt/(size_gtos_q*size_gtos_q);
            int e2J = tt - e1J*(size_gtos_q*size_gtos_q);
            int ii = e1J/size_gtos_p, jj = e1J - ii*size_gtos_p;
            int kk = e2J/size_gtos_q, ll = e2J - kk*size_gtos_q;
            int e1K = ii*size_gtos_q+ll, e2K = kk*size_gtos_p+jj;
            double a_i_J = shell_list[pshell].exp_a[ii], a_j_J = shell_list[pshell].exp_a[jj], a_k_J = shell_list[qshell].exp_a[kk], a_l_J = shell_list[qshell].exp_a[ll];
            double a_i_K = shell_list[pshell].exp_a[ii], a_j_K = shell_list[qshell].exp_a[ll], a_k_K = shell_list[qshell].exp_a[kk], a_l_K = shell_list[pshell].exp_a[jj];
        
            if(intType.substr(0,4) == "LSLS")
            {
                for(int LL = LmaxJ[0]-LminJ[0]; LL >= 0; LL-=2)
                {
                    radial_2e_list_Jmm[LL][tt][0] = int2e_get_radial_gauge(l_p,a_i_J,l_p+1,a_j_J,l_q,a_k_J,l_q+1,a_l_J,LL+LminJ[0],-1,-1);
                    if(l_p != 0)
                        radial_2e_list_Jmm[LL][tt][1] = int2e_get_radial_gauge(l_p,a_i_J,l_p-1,a_j_J,l_q,a_k_J,l_q+1,a_l_J,LL+LminJ[0],-1,-1);
                    if(l_q != 0)
                        radial_2e_list_Jmm[LL][tt][2] = int2e_get_radial_gauge(l_p,a_i_J,l_p+1,a_j_J,l_q,a_k_J,l_q-1,a_l_J,LL+LminJ[0],-1,-1);
                    if(l_p != 0 && l_q != 0)
                        radial_2e_list_Jmm[LL][tt][3] = int2e_get_radial_gauge(l_p,a_i_J,l_p-1,a_j_J,l_q,a_k_J,l_q-1,a_l_J,LL+LminJ[0],-1,-1);
                }
                for(int LL = LmaxK[0]-LminK[0]; LL >= 0; LL-=2)
                {
                    radial_2e_list_Kmm[LL][tt][0] = int2e_get_radial_gauge(l_p,a_i_K,l_q+1,a_j_K,l_q,a_k_K,l_p+1,a_l_K,LL+LminK[0],-1,-1);
                    if(l_q != 0)
                        radial_2e_list_Kmm[LL][tt][1] = int2e_get_radial_gauge(l_p,a_i_K,l_q-1,a_j_K,l_q,a_k_K,l_p+1,a_l_K,LL+LminK[0],-1,-1);
                    if(l_p != 0)
                        radial_2e_list_Kmm[LL][tt][2] = int2e_get_radial_gauge(l_p,a_i_K,l_q+1,a_j_K,l_q,a_k_K,l_p-1,a_l_K,LL+LminK[0],-1,-1);
                    if(l_p != 0 && l_q != 0)
                        radial_2e_list_Kmm[LL][tt][3] = int2e_get_radial_gauge(l_p,a_i_K,l_q-1,a_j_K,l_q,a_k_K,l_p-1,a_l_K,LL+LminK[0],-1,-1);
                }
                for(int LL = LmaxJ[1]-LminJ[1]; LL >= 0; LL-=2)
                {
                    radial_2e_list_Jmp[LL][tt][0] = int2e_get_radial_gauge(l_p,a_i_J,l_p+1,a_j_J,l_q,a_k_J,l_q+1,a_l_J,LL+LminJ[1],-1,1);
                    if(l_p != 0)
                        radial_2e_list_Jmp[LL][tt][1] = int2e_get_radial_gauge(l_p,a_i_J,l_p-1,a_j_J,l_q,a_k_J,l_q+1,a_l_J,LL+LminJ[1],-1,1);
                    if(l_q != 0)
                        radial_2e_list_Jmp[LL][tt][2] = int2e_get_radial_gauge(l_p,a_i_J,l_p+1,a_j_J,l_q,a_k_J,l_q-1,a_l_J,LL+LminJ[1],-1,1);
                    if(l_p != 0 && l_q != 0)
                        radial_2e_list_Jmp[LL][tt][3] = int2e_get_radial_gauge(l_p,a_i_J,l_p-1,a_j_J,l_q,a_k_J,l_q-1,a_l_J,LL+LminJ[1],-1,1);
                }
                for(int LL = LmaxK[1]-LminK[1]; LL >= 0; LL-=2)
                {
                    radial_2e_list_Kmp[LL][tt][0] = int2e_get_radial_gauge(l_p,a_i_K,l_q+1,a_j_K,l_q,a_k_K,l_p+1,a_l_K,LL+LminK[1],-1,1);
                    if(l_q != 0)
                        radial_2e_list_Kmp[LL][tt][1] = int2e_get_radial_gauge(l_p,a_i_K,l_q-1,a_j_K,l_q,a_k_K,l_p+1,a_l_K,LL+LminK[1],-1,1);
                    if(l_p != 0)
                        radial_2e_list_Kmp[LL][tt][2] = int2e_get_radial_gauge(l_p,a_i_K,l_q+1,a_j_K,l_q,a_k_K,l_p-1,a_l_K,LL+LminK[1],-1,1);
                    if(l_p != 0 && l_q != 0)
                        radial_2e_list_Kmp[LL][tt][3] = int2e_get_radial_gauge(l_p,a_i_K,l_q-1,a_j_K,l_q,a_k_K,l_p-1,a_l_K,LL+LminK[1],-1,1);
                }
                for(int LL = LmaxJ[2]-LminJ[2]; LL >= 0; LL-=2)
                {
                    radial_2e_list_Jpm[LL][tt][0] = int2e_get_radial_gauge(l_p,a_i_J,l_p+1,a_j_J,l_q,a_k_J,l_q+1,a_l_J,LL+LminJ[2],1,-1);
                    if(l_p != 0)
                        radial_2e_list_Jpm[LL][tt][1] = int2e_get_radial_gauge(l_p,a_i_J,l_p-1,a_j_J,l_q,a_k_J,l_q+1,a_l_J,LL+LminJ[2],1,-1);
                    if(l_q != 0)
                        radial_2e_list_Jpm[LL][tt][2] = int2e_get_radial_gauge(l_p,a_i_J,l_p+1,a_j_J,l_q,a_k_J,l_q-1,a_l_J,LL+LminJ[2],1,-1);
                    if(l_p != 0 && l_q != 0)
                        radial_2e_list_Jpm[LL][tt][3] = int2e_get_radial_gauge(l_p,a_i_J,l_p-1,a_j_J,l_q,a_k_J,l_q-1,a_l_J,LL+LminJ[2],1,-1);
                }
                for(int LL = LmaxK[2]-LminK[2]; LL >= 0; LL-=2)
                {
                    radial_2e_list_Kpm[LL][tt][0] = int2e_get_radial_gauge(l_p,a_i_K,l_q+1,a_j_K,l_q,a_k_K,l_p+1,a_l_K,LL+LminK[2],1,-1);
                    if(l_q != 0)
                        radial_2e_list_Kpm[LL][tt][1] = int2e_get_radial_gauge(l_p,a_i_K,l_q-1,a_j_K,l_q,a_k_K,l_p+1,a_l_K,LL+LminK[2],1,-1);
                    if(l_p != 0)
                        radial_2e_list_Kpm[LL][tt][2] = int2e_get_radial_gauge(l_p,a_i_K,l_q+1,a_j_K,l_q,a_k_K,l_p-1,a_l_K,LL+LminK[2],1,-1);
                    if(l_p != 0 && l_q != 0)
                        radial_2e_list_Kpm[LL][tt][3] = int2e_get_radial_gauge(l_p,a_i_K,l_q-1,a_j_K,l_q,a_k_K,l_p-1,a_l_K,LL+LminK[2],1,-1);
                }
                for(int LL = LmaxJ[3]-LminJ[3]; LL >= 0; LL-=2)
                {
                    radial_2e_list_Jpp[LL][tt][0] = int2e_get_radial_gauge(l_p,a_i_J,l_p+1,a_j_J,l_q,a_k_J,l_q+1,a_l_J,LL+LminJ[3],1,1);
                    if(l_p != 0)
                        radial_2e_list_Jpp[LL][tt][1] = int2e_get_radial_gauge(l_p,a_i_J,l_p-1,a_j_J,l_q,a_k_J,l_q+1,a_l_J,LL+LminJ[3],1,1);
                    if(l_q != 0)
                        radial_2e_list_Jpp[LL][tt][2] = int2e_get_radial_gauge(l_p,a_i_J,l_p+1,a_j_J,l_q,a_k_J,l_q-1,a_l_J,LL+LminJ[3],1,1);
                    if(l_p != 0 && l_q != 0)
                        radial_2e_list_Jpp[LL][tt][3] = int2e_get_radial_gauge(l_p,a_i_J,l_p-1,a_j_J,l_q,a_k_J,l_q-1,a_l_J,LL+LminJ[3],1,1);
                }
                for(int LL = LmaxK[3]-LminK[3]; LL >= 0; LL-=2)
                {
                    radial_2e_list_Kpp[LL][tt][0] = int2e_get_radial_gauge(l_p,a_i_K,l_q+1,a_j_K,l_q,a_k_K,l_p+1,a_l_K,LL+LminK[3],1,1);
                    if(l_q != 0)
                        radial_2e_list_Kpp[LL][tt][1] = int2e_get_radial_gauge(l_p,a_i_K,l_q-1,a_j_K,l_q,a_k_K,l_p+1,a_l_K,LL+LminK[3],1,1);
                    if(l_p != 0)
                        radial_2e_list_Kpp[LL][tt][2] = int2e_get_radial_gauge(l_p,a_i_K,l_q+1,a_j_K,l_q,a_k_K,l_p-1,a_l_K,LL+LminK[3],1,1);
                    if(l_p != 0 && l_q != 0)
                        radial_2e_list_Kpp[LL][tt][3] = int2e_get_radial_gauge(l_p,a_i_K,l_q-1,a_j_K,l_q,a_k_K,l_p-1,a_l_K,LL+LminK[3],1,1);
                }
            }
            else if(intType.substr(0,4) == "LSSL")
            {
                for(int LL = LmaxJ[0]-LminJ[0]; LL >= 0; LL-=2)
                {
                    radial_2e_list_Jmm[LL][tt][0] = int2e_get_radial_gauge(l_p,a_i_J,l_p+1,a_j_J,l_q+1,a_k_J,l_q,a_l_J,LL+LminJ[0],-1,-1);
                    if(l_p != 0)
                        radial_2e_list_Jmm[LL][tt][1] = int2e_get_radial_gauge(l_p,a_i_J,l_p-1,a_j_J,l_q+1,a_k_J,l_q,a_l_J,LL+LminJ[0],-1,-1);
                    if(l_q != 0)
                        radial_2e_list_Jmm[LL][tt][2] = int2e_get_radial_gauge(l_p,a_i_J,l_p+1,a_j_J,l_q-1,a_k_J,l_q,a_l_J,LL+LminJ[0],-1,-1);
                    if(l_p != 0 && l_q != 0)
                        radial_2e_list_Jmm[LL][tt][3] = int2e_get_radial_gauge(l_p,a_i_J,l_p-1,a_j_J,l_q-1,a_k_J,l_q,a_l_J,LL+LminJ[0],-1,-1);
                }
                for(int LL = LmaxK[0]-LminK[0]; LL >= 0; LL-=2)
                {
                    radial_2e_list_Kmm[LL][tt][0] = int2e_get_radial_gauge(l_p,a_i_K,l_q+1,a_j_K,l_q+1,a_k_K,l_p,a_l_K,LL+LminK[0],-1,-1);
                    if(l_q != 0)
                    {
                        radial_2e_list_Kmm[LL][tt][1] = int2e_get_radial_gauge(l_p,a_i_K,l_q-1,a_j_K,l_q+1,a_k_K,l_p,a_l_K,LL+LminK[0],-1,-1);
                        radial_2e_list_Kmm[LL][tt][2] = int2e_get_radial_gauge(l_p,a_i_K,l_q+1,a_j_K,l_q-1,a_k_K,l_p,a_l_K,LL+LminK[0],-1,-1);
                        radial_2e_list_Kmm[LL][tt][3] = int2e_get_radial_gauge(l_p,a_i_K,l_q-1,a_j_K,l_q-1,a_k_K,l_p,a_l_K,LL+LminK[0],-1,-1);
                    }
                }
                for(int LL = LmaxJ[1]-LminJ[1]; LL >= 0; LL-=2)
                {
                    radial_2e_list_Jmp[LL][tt][0] = int2e_get_radial_gauge(l_p,a_i_J,l_p+1,a_j_J,l_q+1,a_k_J,l_q,a_l_J,LL+LminJ[1],-1,1);
                    if(l_p != 0)
                        radial_2e_list_Jmp[LL][tt][1] = int2e_get_radial_gauge(l_p,a_i_J,l_p-1,a_j_J,l_q+1,a_k_J,l_q,a_l_J,LL+LminJ[1],-1,1);
                    if(l_q != 0)
                        radial_2e_list_Jmp[LL][tt][2] = int2e_get_radial_gauge(l_p,a_i_J,l_p+1,a_j_J,l_q-1,a_k_J,l_q,a_l_J,LL+LminJ[1],-1,1);
                    if(l_p != 0 && l_q != 0)
                        radial_2e_list_Jmp[LL][tt][3] = int2e_get_radial_gauge(l_p,a_i_J,l_p-1,a_j_J,l_q-1,a_k_J,l_q,a_l_J,LL+LminJ[1],-1,1);
                }
                for(int LL = LmaxK[1]-LminK[1]; LL >= 0; LL-=2)
                {
                    radial_2e_list_Kmp[LL][tt][0] = int2e_get_radial_gauge(l_p,a_i_K,l_q+1,a_j_K,l_q+1,a_k_K,l_p,a_l_K,LL+LminK[1],-1,1);
                    if(l_q != 0)
                    {
                        radial_2e_list_Kmp[LL][tt][1] = int2e_get_radial_gauge(l_p,a_i_K,l_q-1,a_j_K,l_q+1,a_k_K,l_p,a_l_K,LL+LminK[1],-1,1);
                        radial_2e_list_Kmp[LL][tt][2] = int2e_get_radial_gauge(l_p,a_i_K,l_q+1,a_j_K,l_q-1,a_k_K,l_p,a_l_K,LL+LminK[1],-1,1);
                        radial_2e_list_Kmp[LL][tt][3] = int2e_get_radial_gauge(l_p,a_i_K,l_q-1,a_j_K,l_q-1,a_k_K,l_p,a_l_K,LL+LminK[1],-1,1);
                    }
                }
                for(int LL = LmaxJ[2]-LminJ[2]; LL >= 0; LL-=2)
                {
                    radial_2e_list_Jpm[LL][tt][0] = int2e_get_radial_gauge(l_p,a_i_J,l_p+1,a_j_J,l_q+1,a_k_J,l_q,a_l_J,LL+LminJ[2],1,-1);
                    if(l_p != 0)
                        radial_2e_list_Jpm[LL][tt][1] = int2e_get_radial_gauge(l_p,a_i_J,l_p-1,a_j_J,l_q+1,a_k_J,l_q,a_l_J,LL+LminJ[2],1,-1);
                    if(l_q != 0)
                        radial_2e_list_Jpm[LL][tt][2] = int2e_get_radial_gauge(l_p,a_i_J,l_p+1,a_j_J,l_q-1,a_k_J,l_q,a_l_J,LL+LminJ[2],1,-1);
                    if(l_p != 0 && l_q != 0)
                        radial_2e_list_Jpm[LL][tt][3] = int2e_get_radial_gauge(l_p,a_i_J,l_p-1,a_j_J,l_q-1,a_k_J,l_q,a_l_J,LL+LminJ[2],1,-1);
                }
                for(int LL = LmaxK[2]-LminK[2]; LL >= 0; LL-=2)
                {
                    radial_2e_list_Kpm[LL][tt][0] = int2e_get_radial_gauge(l_p,a_i_K,l_q+1,a_j_K,l_q+1,a_k_K,l_p,a_l_K,LL+LminK[2],1,-1);
                    if(l_q != 0)
                    {
                        radial_2e_list_Kpm[LL][tt][1] = int2e_get_radial_gauge(l_p,a_i_K,l_q-1,a_j_K,l_q+1,a_k_K,l_p,a_l_K,LL+LminK[2],1,-1);
                        radial_2e_list_Kpm[LL][tt][2] = int2e_get_radial_gauge(l_p,a_i_K,l_q+1,a_j_K,l_q-1,a_k_K,l_p,a_l_K,LL+LminK[2],1,-1);
                        radial_2e_list_Kpm[LL][tt][3] = int2e_get_radial_gauge(l_p,a_i_K,l_q-1,a_j_K,l_q-1,a_k_K,l_p,a_l_K,LL+LminK[2],1,-1);
                    }
                }
                for(int LL = LmaxJ[3]-LminJ[3]; LL >= 0; LL-=2)
                {
                    radial_2e_list_Jpp[LL][tt][0] = int2e_get_radial_gauge(l_p,a_i_J,l_p+1,a_j_J,l_q+1,a_k_J,l_q,a_l_J,LL+LminJ[3],1,1);
                    if(l_p != 0)
                        radial_2e_list_Jpp[LL][tt][1] = int2e_get_radial_gauge(l_p,a_i_J,l_p-1,a_j_J,l_q+1,a_k_J,l_q,a_l_J,LL+LminJ[3],1,1);
                    if(l_q != 0)
                        radial_2e_list_Jpp[LL][tt][2] = int2e_get_radial_gauge(l_p,a_i_J,l_p+1,a_j_J,l_q-1,a_k_J,l_q,a_l_J,LL+LminJ[3],1,1);
                    if(l_p != 0 && l_q != 0)
                        radial_2e_list_Jpp[LL][tt][3] = int2e_get_radial_gauge(l_p,a_i_J,l_p-1,a_j_J,l_q-1,a_k_J,l_q,a_l_J,LL+LminJ[3],1,1);
                }
                for(int LL = LmaxK[3]-LminK[3]; LL >= 0; LL-=2)
                {
                    radial_2e_list_Kpp[LL][tt][0] = int2e_get_radial_gauge(l_p,a_i_K,l_q+1,a_j_K,l_q+1,a_k_K,l_p,a_l_K,LL+LminK[3],1,1);
                    if(l_q != 0)
                    {
                        radial_2e_list_Kpp[LL][tt][1] = int2e_get_radial_gauge(l_p,a_i_K,l_q-1,a_j_K,l_q+1,a_k_K,l_p,a_l_K,LL+LminK[3],1,1);
                        radial_2e_list_Kpp[LL][tt][2] = int2e_get_radial_gauge(l_p,a_i_K,l_q+1,a_j_K,l_q-1,a_k_K,l_p,a_l_K,LL+LminK[3],1,1);
                        radial_2e_list_Kpp[LL][tt][3] = int2e_get_radial_gauge(l_p,a_i_K,l_q-1,a_j_K,l_q-1,a_k_K,l_p,a_l_K,LL+LminK[3],1,1);
                    }
                }
            }
            else
            {
                cout << "ERROR: Unkonwn intType in get_h2e_JK_gauge." << endl;
                exit(99);
            }
        }

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
            
            // Angular
            for(int LL = LmaxJ[0]-LminJ[0]; LL >= 0; LL-=2)
            {
                double tmp = 0.0;
                for(int mq = 0; mq < twojj_q + 1; mq++)
                {
                    if(intType.substr(0,4) == "LSLS")
                        tmp += int2e_get_angular_gauge_LSLS(l_p, -twojj_p, sym_ap, l_p, -twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, LL+LminJ[0], -1, -1);
                    else if(intType.substr(0,4) == "LSSL")
                        tmp += int2e_get_angular_gauge_LSSL(l_p, -twojj_p, sym_ap, l_p, -twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, LL+LminJ[0], -1, -1);
                    else
                    {
                        cout << "ERROR: Unkonwn intType in get_h2e_JK_gauge." << endl;
                        exit(99);
                    }
                }
                array_angular_Jmm[LL][int_tmp2_p][int_tmp2_q] = tmp / (twojj_q+1.0);
            }
            for(int LL = LmaxK[0]-LminK[0]; LL >= 0; LL-=2)
            {
                double tmp = 0.0;
                for(int mq = 0; mq < twojj_q + 1; mq++)
                {
                    if(intType.substr(0,4) == "LSLS")
                        tmp += int2e_get_angular_gauge_LSLS(l_p, -twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, l_p, -twojj_p, sym_ap, LL+LminK[0], -1, -1);
                    else if(intType.substr(0,4) == "LSSL")
                        tmp += int2e_get_angular_gauge_LSSL(l_p, -twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, l_p, -twojj_p, sym_ap, LL+LminK[0], -1, -1);
                    else
                    {
                        cout << "ERROR: Unkonwn intType in get_h2e_JK_gauge." << endl;
                        exit(99);
                    }
                }
                array_angular_Kmm[LL][int_tmp2_p][int_tmp2_q] = tmp /(twojj_q+1.0);
            }
            for(int LL = LmaxJ[1]-LminJ[1]; LL >= 0; LL-=2)
            {
                double tmp = 0.0;
                for(int mq = 0; mq < twojj_q + 1; mq++)
                {
                    if(intType.substr(0,4) == "LSLS")
                        tmp += int2e_get_angular_gauge_LSLS(l_p, -twojj_p, sym_ap, l_p, -twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, LL+LminJ[1], -1, 1);
                    else if(intType.substr(0,4) == "LSSL")
                        tmp += int2e_get_angular_gauge_LSSL(l_p, -twojj_p, sym_ap, l_p, -twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, LL+LminJ[1], -1, 1);
                    else
                    {
                        cout << "ERROR: Unkonwn intType in get_h2e_JK_gauge." << endl;
                        exit(99);
                    }
                }
                array_angular_Jmp[LL][int_tmp2_p][int_tmp2_q] = tmp/(twojj_q+1.0);
            }
            for(int LL = LmaxK[1]-LminK[1]; LL >= 0; LL-=2)
            {
                double tmp = 0.0;
                for(int mq = 0; mq < twojj_q + 1; mq++)
                {
                    if(intType.substr(0,4) == "LSLS")
                        tmp += int2e_get_angular_gauge_LSLS(l_p, -twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, l_p, -twojj_p, sym_ap, LL+LminK[1], -1, 1);
                    else if(intType.substr(0,4) == "LSSL")
                        tmp += int2e_get_angular_gauge_LSSL(l_p, -twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, l_p, -twojj_p, sym_ap, LL+LminK[1], -1, 1);
                    else
                    {
                        cout << "ERROR: Unkonwn intType in get_h2e_JK_gauge." << endl;
                        exit(99);
                    }
                }
                array_angular_Kmp[LL][int_tmp2_p][int_tmp2_q] = tmp/(twojj_q+1.0);
            }
            for(int LL = LmaxJ[2]-LminJ[2]; LL >= 0; LL-=2)
            {
                double tmp = 0.0;
                for(int mq = 0; mq < twojj_q + 1; mq++)
                {
                    if(intType.substr(0,4) == "LSLS")
                        tmp += int2e_get_angular_gauge_LSLS(l_p, -twojj_p, sym_ap, l_p, -twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, LL+LminJ[2], 1, -1);
                    else if(intType.substr(0,4) == "LSSL")
                        tmp += int2e_get_angular_gauge_LSSL(l_p, -twojj_p, sym_ap, l_p, -twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, LL+LminJ[2], 1, -1);
                    else
                    {
                        cout << "ERROR: Unkonwn intType in get_h2e_JK_gauge." << endl;
                        exit(99);
                    }
                }
                array_angular_Jpm[LL][int_tmp2_p][int_tmp2_q] = tmp/(twojj_q+1.0);
            }
            for(int LL = LmaxK[2]-LminK[2]; LL >= 0; LL-=2)
            {
                double tmp = 0.0;
                for(int mq = 0; mq < twojj_q + 1; mq++)
                {
                    if(intType.substr(0,4) == "LSLS")
                        tmp += int2e_get_angular_gauge_LSLS(l_p, -twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, l_p, -twojj_p, sym_ap, LL+LminK[2], 1, -1);
                    else if(intType.substr(0,4) == "LSSL")
                        tmp += int2e_get_angular_gauge_LSSL(l_p, -twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, l_p, -twojj_p, sym_ap, LL+LminK[2], 1, -1);
                    else
                    {
                        cout << "ERROR: Unkonwn intType in get_h2e_JK_gauge." << endl;
                        exit(99);
                    }
                }
                array_angular_Kpm[LL][int_tmp2_p][int_tmp2_q] = tmp/(twojj_q+1.0);
            }
            for(int LL = LmaxJ[3]-LminJ[3]; LL >= 0; LL-=2)
            {
                double tmp = 0.0;
                for(int mq = 0; mq < twojj_q + 1; mq++)
                {
                    if(intType.substr(0,4) == "LSLS")
                        tmp += int2e_get_angular_gauge_LSLS(l_p, -twojj_p, sym_ap, l_p, -twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, LL+LminJ[3], 1, 1);
                    else if(intType.substr(0,4) == "LSSL")
                        tmp += int2e_get_angular_gauge_LSSL(l_p, -twojj_p, sym_ap, l_p, -twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, LL+LminJ[3], 1, 1);
                    else
                    {
                        cout << "ERROR: Unkonwn intType in get_h2e_JK_gauge." << endl;
                        exit(99);
                    }
                }
                array_angular_Jpp[LL][int_tmp2_p][int_tmp2_q] = tmp/(twojj_q+1.0);
            }
            for(int LL = LmaxK[3]-LminK[3]; LL >= 0; LL-=2)
            {
                double tmp = 0.0;
                for(int mq = 0; mq < twojj_q + 1; mq++)
                {
                    if(intType.substr(0,4) == "LSLS")
                        tmp += int2e_get_angular_gauge_LSLS(l_p, -twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, l_p, -twojj_p, sym_ap, LL+LminK[3], 1, 1);
                    else if(intType.substr(0,4) == "LSSL")
                        tmp += int2e_get_angular_gauge_LSSL(l_p, -twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, l_p, -twojj_p, sym_ap, LL+LminK[3], 1, 1);
                    else
                    {
                        cout << "ERROR: Unkonwn intType in get_h2e_JK_gauge." << endl;
                        exit(99);
                    }
                }
                array_angular_Kpp[LL][int_tmp2_p][int_tmp2_q] = tmp/(twojj_q+1.0);
            }
        
            // Radial
            double k_p = -(twojj_p+1.0)*sym_ap/2.0, k_q = -(twojj_q+1.0)*sym_aq/2.0;
            #pragma omp parallel  for
            for(int tt = 0; tt < size_gtos_p*size_gtos_p*size_gtos_q*size_gtos_q; tt++)
            {
                int e1J = tt/(size_gtos_q*size_gtos_q);
                int e2J = tt - e1J*(size_gtos_q*size_gtos_q);
                int ii = e1J/size_gtos_p, jj = e1J - ii*size_gtos_p;
                int kk = e2J/size_gtos_q, ll = e2J - kk*size_gtos_q;
                int e1K = ii*size_gtos_q+ll, e2K = kk*size_gtos_p+jj;
                double norm_J = shell_list[pshell].norm[ii] * shell_list[pshell].norm[jj] * shell_list[qshell].norm[kk] * shell_list[qshell].norm[ll], norm_K = shell_list[pshell].norm[ii] * shell_list[qshell].norm[ll] * shell_list[qshell].norm[kk] * shell_list[pshell].norm[jj];
                double lk1 = 1+l_p+k_p, lk2 = 1+l_p+k_p, lk3 = 1+l_q+k_q, lk4 = 1+l_q+k_q, a1 = shell_list[pshell].exp_a[ii], a2 = shell_list[pshell].exp_a[jj], a3 = shell_list[qshell].exp_a[kk], a4 = shell_list[qshell].exp_a[ll];

                int_2e_JK.J[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1J][e2J] = 0.0;
                int_2e_JK.K[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1K][e2K] = 0.0;
                
                for(int LL = LmaxJ[0]-LminJ[0]; LL >= 0; LL-=2)
                {
                    if(intType == "LSLS")
                    {
                        radial = get_radial_LSLS_J(l_p,l_q,LL+LminJ[0],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Jmm[LL][tt],false);
                        radial /= -2.0 * norm_J * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "LSSL")
                    {
                        radial = get_radial_LSSL_J(l_p,l_q,LL+LminJ[0],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Jmm[LL][tt],false);
                        radial /= 2.0 * norm_J * 4.0 * pow(speedOfLight,2);
                    }
                    int_2e_JK.J[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1J][e2J] += radial * array_angular_Jmm[LL][int_tmp2_p][int_tmp2_q];
                }
                for(int LL = LmaxJ[1]-LminJ[1]; LL >= 0; LL-=2)
                {
                    if(intType == "LSLS")
                    {
                        radial = get_radial_LSLS_J(l_p,l_q,LL+LminJ[1],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Jmp[LL][tt],false);
                        radial /= -2.0 * norm_J * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "LSSL")
                    {
                        radial = get_radial_LSSL_J(l_p,l_q,LL+LminJ[1],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Jmp[LL][tt],false);
                        radial /= 2.0 * norm_J * 4.0 * pow(speedOfLight,2);
                    }
                    int_2e_JK.J[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1J][e2J] += radial * array_angular_Jmp[LL][int_tmp2_p][int_tmp2_q];
                }
                for(int LL = LmaxJ[2]-LminJ[2]; LL >= 0; LL-=2)
                {
                    if(intType == "LSLS")
                    {
                        radial = get_radial_LSLS_J(l_p,l_q,LL+LminJ[2],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Jpm[LL][tt],false);
                        radial /= -2.0 * norm_J * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "LSSL")
                    {
                        radial = get_radial_LSSL_J(l_p,l_q,LL+LminJ[2],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Jpm[LL][tt],false);
                        radial /= 2.0 * norm_J * 4.0 * pow(speedOfLight,2);
                    }
                    int_2e_JK.J[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1J][e2J] += radial * array_angular_Jpm[LL][int_tmp2_p][int_tmp2_q];
                }
                for(int LL = LmaxJ[3]-LminJ[3]; LL >= 0; LL-=2)
                {
                    if(intType == "LSLS")
                    {
                        radial = get_radial_LSLS_J(l_p,l_q,LL+LminJ[3],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Jpp[LL][tt],false);
                        radial /= -2.0 * norm_J * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "LSSL")
                    {
                        radial = get_radial_LSSL_J(l_p,l_q,LL+LminJ[3],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Jpp[LL][tt],false);
                        radial /= 2.0 * norm_J * 4.0 * pow(speedOfLight,2);
                    }
                    int_2e_JK.J[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1J][e2J] += radial * array_angular_Jpp[LL][int_tmp2_p][int_tmp2_q];
                }
                lk2 = 1+l_q+k_q; lk4 = 1+l_p+k_p; 
                a2 = shell_list[qshell].exp_a[ll]; a4 = shell_list[pshell].exp_a[jj];
                for(int LL = LmaxK[0]-LminK[0]; LL >= 0; LL-=2)
                {
                    if(intType == "LSLS")
                    {
                        radial = get_radial_LSLS_K(l_p,l_q,LL+LminK[0],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Kmm[LL][tt],false);
                        radial /= -2.0 * norm_K * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "LSSL")
                    {
                        radial = get_radial_LSSL_K(l_p,l_q,LL+LminK[0],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Kmm[LL][tt],false);
                        radial /= 2.0 * norm_K * 4.0 * pow(speedOfLight,2);
                    }
                    int_2e_JK.K[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1K][e2K] += radial * array_angular_Kmm[LL][int_tmp2_p][int_tmp2_q];
                }
                for(int LL = LmaxK[1]-LminK[1]; LL >= 0; LL-=2)
                {
                    if(intType == "LSLS")
                    {
                        radial = get_radial_LSLS_K(l_p,l_q,LL+LminK[1],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Kmp[LL][tt],false);
                        radial /= -2.0 * norm_K * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "LSSL")
                    {
                        radial = get_radial_LSSL_K(l_p,l_q,LL+LminK[1],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Kmp[LL][tt],false);
                        radial /= 2.0 * norm_K * 4.0 * pow(speedOfLight,2);
                    }
                    int_2e_JK.K[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1K][e2K] += radial * array_angular_Kmp[LL][int_tmp2_p][int_tmp2_q];
                }
                for(int LL = LmaxK[2]-LminK[2]; LL >= 0; LL-=2)
                {
                    if(intType == "LSLS")
                    {
                        radial = get_radial_LSLS_K(l_p,l_q,LL+LminK[2],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Kpm[LL][tt],false);
                        radial /= -2.0 * norm_K * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "LSSL")
                    {
                        radial = get_radial_LSSL_K(l_p,l_q,LL+LminK[2],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Kpm[LL][tt],false);
                        radial /= 2.0 * norm_K * 4.0 * pow(speedOfLight,2);
                    }
                    int_2e_JK.K[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1K][e2K] += radial * array_angular_Kpm[LL][int_tmp2_p][int_tmp2_q];
                }
                for(int LL = LmaxK[3]-LminK[3]; LL >= 0; LL-=2)
                {
                    if(intType == "LSLS")
                    {
                        radial = get_radial_LSLS_K(l_p,l_q,LL+LminK[3],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Kpp[LL][tt],false);
                        radial /= -2.0 * norm_K * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "LSSL")
                    {
                        radial = get_radial_LSSL_K(l_p,l_q,LL+LminK[3],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Kpp[LL][tt],false);
                        radial /= 2.0 * norm_K * 4.0 * pow(speedOfLight,2);
                    }
                    int_2e_JK.K[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1K][e2K] += radial * array_angular_Kpp[LL][int_tmp2_p][int_tmp2_q];
                }                    
            }
        }
        int_tmp1_q += (l_q == 0) ? 1 : 2;
    }
    int_tmp1_p += (l_p == 0) ? 1 : 2;
    }

    return int_2e_JK;
}


void INT_SPH::get_h2e_JK_gauge_direct(int2eJK& LSLS, int2eJK& LSSL, const int& occMaxL, const bool& spinFree)
{
    LSLS = get_h2e_JK_gauge_compact("LSLS",occMaxL);
    LSSL = get_h2e_JK_gauge_compact("LSSL",occMaxL);
    
    return;
}