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

double get_N_coeff(const int& vv, const int& aa, const int& ll, const int& mm)
{
    if(abs(mm+aa) > ll+vv) return 0.0;
    double tmp;
    //  return gsl_sf_coupling_3j(2*ll,2,2*ll+2*vv,2*mm,2*aa,-2*mm-2*aa)*gsl_sf_coupling_3j(2*ll,2,2*ll+2*vv,0,0,0)*pow(-1,mm+aa)*(2.0*(ll+vv)+1.0);
    // return gsl_sf_coupling_3j(2*ll,2,2*ll+2*vv,2*mm,2*aa,-2*mm-2*aa)*gsl_sf_coupling_3j(2*ll,2,2*ll+2*vv,0,0,0)*pow(-1,mm)*(2.0*(ll+vv)+1.0);

    
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
    double threeJ1 = gsl_sf_coupling_3j(2*l1,2*ll+2*v1,2*l2p,0,0,0), threeJ2 = gsl_sf_coupling_3j(2*l3,2*ll+2*v2,2*l4p,0,0,0), tmp;
    tmp = 0.0;
    double rme1 = int2e_get_angularX_RME(two_j1,l1,two_j2,l2p,ll,ll+v1,threeJ1);
    double rme2 = int2e_get_angularX_RME(two_j3,l3,two_j4,l4p,ll,ll+v2,threeJ2);
    for(int MMM = -ll; MMM <= ll; MMM++)
    {
        tmp += pow(-1,MMM) * gsl_sf_coupling_3j(two_j1,2*ll,two_j2,-two_m1,-2*MMM,two_m2)
             * gsl_sf_coupling_3j(two_j3,2*ll,two_j4,-two_m3,2*MMM,two_m4);
    }
    angular += tmp*rme1*rme2;
    
    return angular*pow(-1,(two_j1+two_j3-two_m1-two_m3)/2)*(2.0*(ll+v1)+1.0)*(2.0*(ll+v2)+1.0)/(2.0*ll+1)*gsl_sf_coupling_3j(2,2*ll+2*v1,2*ll,0,0,0)*gsl_sf_coupling_3j(2,2*ll+2*v2,2*ll,0,0,0);
    
    // double threeJ1 = gsl_sf_coupling_3j(2*l1,2*ll+2*v1,2*l2p,0,0,0);
    // double threeJ2 = gsl_sf_coupling_3j(2*l3,2*ll+2*v2,2*l4p,0,0,0);
    // int Lmin = max(abs(ll+v1-1),0), Lmax = ll+v1+1, Lpmin = max(abs(ll+v2-1),0), Lpmax = ll+v2+1;
    // for(int MM = -ll; MM <= ll; MM++)
    // {
    //     double tmpa, tmpb, tmpL = 0.0, tmpLp = 0.0;
    //     for(int LL = Lmin; LL <= Lmax; LL++)
    //     {
    //         tmpa = 0.0;
    //         for(int aa = -1; aa <= 1; aa++)
    //             tmpa += get_N_coeff(v1,aa,ll,-MM)*gsl_sf_coupling_3j(2,2*ll+2*v1,2*LL,-2*aa,-2*MM+2*aa,2*MM);
    //         tmpL += tmpa*sqrt(2.0*LL+1)*gsl_sf_coupling_3j(two_j1,2*LL,two_j2,-two_m1,-2*MM,two_m2)*int2e_get_angularX_RME(two_j1,l1,two_j2,l2p,LL,ll+v1,threeJ1);
    //     }
    //     for(int LP = Lpmin; LP<=Lpmax; LP++)
    //     {
    //         tmpb = 0.0;
    //         for(int bb = -1; bb <= 1; bb++)
    //             tmpb += get_N_coeff(v2,bb,ll,MM)*gsl_sf_coupling_3j(2,2*ll+2*v2,2*LP,-2*bb,2*MM+2*bb,-2*MM);
    //         tmpLp += tmpb*sqrt(2.0*LP+1)*gsl_sf_coupling_3j(two_j3,2*LP,two_j4,-two_m3,2*MM,two_m4)*int2e_get_angularX_RME(two_j3,l3,two_j4,l4p,LP,ll+v2,threeJ2);
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
    double threeJ1 = gsl_sf_coupling_3j(2*l1,2*ll+2*v1,2*l2p,0,0,0), threeJ2 = gsl_sf_coupling_3j(2*l3p,2*ll+2*v2,2*l4,0,0,0), tmp;
    tmp = 0.0;
    double rme1 = int2e_get_angularX_RME(two_j1,l1,two_j2,l2p,ll,ll+v1,threeJ1);
    double rme2 = int2e_get_angularX_RME(two_j3,l3p,two_j4,l4,ll,ll+v2,threeJ2);
    for(int MMM = -ll; MMM <= ll; MMM++)
    {
        tmp += pow(-1,MMM) * gsl_sf_coupling_3j(two_j1,2*ll,two_j2,-two_m1,-2*MMM,two_m2)
             * gsl_sf_coupling_3j(two_j3,2*ll,two_j4,-two_m3,2*MMM,two_m4);
    }
    angular += tmp*rme1*rme2;
    
    return angular*pow(-1,(two_j1+two_j3-two_m1-two_m3)/2)*(2.0*(ll+v1)+1.0)*(2.0*(ll+v2)+1.0)/(2.0*ll+1)*gsl_sf_coupling_3j(2,2*ll+2*v1,2*ll,0,0,0)*gsl_sf_coupling_3j(2,2*ll+2*v2,2*ll,0,0,0);

    // double threeJ1 = gsl_sf_coupling_3j(2*l1,2*ll+2*v1,2*l2p,0,0,0);
    // double threeJ2 = gsl_sf_coupling_3j(2*l3p,2*ll+2*v2,2*l4,0,0,0);
    // int Lmin = max(abs(ll+v1-1),0), Lmax = ll+v1+1, Lpmin = max(abs(ll+v2-1),0), Lpmax = ll+v2+1;
    // for(int MM = -ll; MM <= ll; MM++)
    // {
    //     double tmpa, tmpb, tmpL = 0.0, tmpLp = 0.0;
    //     for(int LL = Lmin; LL <= Lmax; LL++)
    //     {
    //         tmpa = 0.0;
    //         for(int aa = -1; aa <= 1; aa++)
    //             tmpa += get_N_coeff(v1,aa,ll,-MM)*gsl_sf_coupling_3j(2,2*ll+2*v1,2*LL,-2*aa,-2*MM+2*aa,2*MM);
    //         tmpL += tmpa*sqrt(2.0*LL+1)*gsl_sf_coupling_3j(two_j1,2*LL,two_j2,-two_m1,-2*MM,two_m2)*int2e_get_angularX_RME(two_j1,l1,two_j2,l2p,LL,ll+v1,threeJ1);
    //     }
    //     for(int LP = Lpmin; LP<=Lpmax; LP++)
    //     {
    //         tmpb = 0.0;
    //         for(int bb = -1; bb <= 1; bb++)
    //             tmpb += get_N_coeff(v2,bb,ll,MM)*gsl_sf_coupling_3j(2,2*ll+2*v2,2*LP,-2*bb,2*MM+2*bb,-2*MM);
    //         tmpLp += tmpb*sqrt(2.0*LP+1)*gsl_sf_coupling_3j(two_j3,2*LP,two_j4,-two_m3,2*MM,two_m4)*int2e_get_angularX_RME(two_j3,l3p,two_j4,l4,LP,ll+v2,threeJ2);
    //     }
    //     angular += pow(-1,MM)*tmpL*tmpLp;
    // }
    
    // return angular*pow(-1,(two_j1+two_j3-two_m1-two_m3)/2);
}


int2eJK INT_SPH::get_h2e_JK_gauge_compact(const string& intType, const int& occMaxL) const
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
    
    int int_tmp1_p = 0;
    for(int pshell = 0; pshell < occMaxShell; pshell++)
    {
    int l_p = shell_list(pshell).l, int_tmp1_q = 0;
    for(int qshell = 0; qshell < occMaxShell; qshell++)
    {
        int l_q = shell_list(qshell).l, l_max = max(l_p,l_q);
        int LmaxJ[4], LminJ[4], LmaxK[4], LminK[4];
        LmaxJ[0] = min(l_p+l_p, l_q+l_q)+2; LmaxK[0] = l_p+l_q+2; LminJ[0] = 1; LminK[0] = 1;
        LmaxJ[1] = min(l_p+l_p+2, l_q+l_q); LmaxK[1] = l_p+l_q  ; LminJ[1] = 1; LminK[1] = 1;
        LmaxJ[2] = min(l_p+l_p, l_q+l_q+2); LmaxK[2] = l_p+l_q  ; LminJ[2] = 1; LminK[2] = 1;
        LmaxJ[3] = min(l_p+l_p, l_q+l_q)  ; LmaxK[3] = l_p+l_q  ; LminJ[3] = 0; LminK[3] = 0;
        // LmaxJ[0] = min(l_p+l_p+1, l_q+l_q+1); LmaxK[0] = l_p+l_q+1; LminJ[0] = 1; LminK[0] = 1;
        // LmaxJ[1] = min(l_p+l_p+1, l_q+l_q-1); LmaxK[1] = l_p+l_q+1; LminJ[1] = 1; LminK[1] = 1;
        // LmaxJ[2] = min(l_p+l_p-1, l_q+l_q+1); LmaxK[2] = l_p+l_q+1; LminJ[2] = 1; LminK[2] = 1;
        // LmaxJ[3] = min(l_p+l_p+1, l_q+l_q+1); LmaxK[3] = l_p+l_q-1; LminJ[3] = 0; LminK[3] = 0;
        // for(int ii = 0; ii < 4; ii++)
        // {
        //     if(LmaxJ[ii] < 0) LmaxJ[ii]=0;
        //     if(LmaxK[ii] < 0) LmaxK[ii]=0;
        //     if(LminJ[ii] < 0) LminJ[ii]=0;
        //     if(LminK[ii] < 0) LminK[ii]=0;
        // }

        int size_gtos_p = shell_list(pshell).coeff.rows(), size_gtos_q = shell_list(qshell).coeff.rows();
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
            double a_i_J = shell_list(pshell).exp_a(ii), a_j_J = shell_list(pshell).exp_a(jj), a_k_J = shell_list(qshell).exp_a(kk), a_l_J = shell_list(qshell).exp_a(ll);
            double a_i_K = shell_list(pshell).exp_a(ii), a_j_K = shell_list(qshell).exp_a(ll), a_k_K = shell_list(qshell).exp_a(kk), a_l_K = shell_list(pshell).exp_a(jj);
        
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
                double tmp_d = 0.0;
                for(int mq = 0; mq < twojj_q + 1; mq++)
                {
                    if(intType.substr(0,4) == "LSLS")
                        tmp_d += int2e_get_angular_gauge_LSLS(l_p, twojj_p, sym_ap, l_p, twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, LL+LminJ[0], -1, -1);
                    else if(intType.substr(0,4) == "LSSL")
                        tmp_d += int2e_get_angular_gauge_LSSL(l_p, twojj_p, sym_ap, l_p, twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, LL+LminJ[0], -1, -1);
                    else
                    {
                        cout << "ERROR: Unkonwn intType in get_h2e_JK_gauge." << endl;
                        exit(99);
                    }
                }
                tmp_d /= (twojj_q + 1);
                array_angular_Jmm[LL][int_tmp2_p][int_tmp2_q] = tmp_d;
            }
            for(int LL = LmaxK[0]-LminK[0]; LL >= 0; LL-=2)
            {
                double tmp_d = 0.0;
                for(int mq = 0; mq < twojj_q + 1; mq++)
                {
                    if(intType.substr(0,4) == "LSLS")
                        tmp_d += int2e_get_angular_gauge_LSLS(l_p, twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, l_p, twojj_p, sym_ap, LL+LminK[0], -1, -1);
                    else if(intType.substr(0,4) == "LSSL")
                        tmp_d += int2e_get_angular_gauge_LSSL(l_p, twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, l_p, twojj_p, sym_ap, LL+LminK[0], -1, -1);
                    else
                    {
                        cout << "ERROR: Unkonwn intType in get_h2e_JK_gauge." << endl;
                        exit(99);
                    }
                }
                tmp_d /= (twojj_q + 1);
                array_angular_Kmm[LL][int_tmp2_p][int_tmp2_q] = tmp_d;
            }
            for(int LL = LmaxJ[1]-LminJ[1]; LL >= 0; LL-=2)
            {
                double tmp_d = 0.0;
                for(int mq = 0; mq < twojj_q + 1; mq++)
                {
                    if(intType.substr(0,4) == "LSLS")
                        tmp_d += int2e_get_angular_gauge_LSLS(l_p, twojj_p, sym_ap, l_p, twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, LL+LminJ[1], -1, +1);
                    else if(intType.substr(0,4) == "LSSL")
                        tmp_d += int2e_get_angular_gauge_LSSL(l_p, twojj_p, sym_ap, l_p, twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, LL+LminJ[1], -1, +1);
                    else
                    {
                        cout << "ERROR: Unkonwn intType in get_h2e_JK_gauge." << endl;
                        exit(99);
                    }
                }
                tmp_d /= (twojj_q + 1);
                array_angular_Jmp[LL][int_tmp2_p][int_tmp2_q] = tmp_d;
            }
            for(int LL = LmaxK[1]-LminK[1]; LL >= 0; LL-=2)
            {
                double tmp_d = 0.0;
                for(int mq = 0; mq < twojj_q + 1; mq++)
                {
                    if(intType.substr(0,4) == "LSLS")
                        tmp_d += int2e_get_angular_gauge_LSLS(l_p, twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, l_p, twojj_p, sym_ap, LL+LminK[1], -1, 1);
                    else if(intType.substr(0,4) == "LSSL")
                        tmp_d += int2e_get_angular_gauge_LSSL(l_p, twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, l_p, twojj_p, sym_ap, LL+LminK[1], -1, 1);
                    else
                    {
                        cout << "ERROR: Unkonwn intType in get_h2e_JK_gauge." << endl;
                        exit(99);
                    }
                }
                tmp_d /= (twojj_q + 1);
                array_angular_Kmp[LL][int_tmp2_p][int_tmp2_q] = tmp_d;
            }
            for(int LL = LmaxJ[2]-LminJ[2]; LL >= 0; LL-=2)
            {
                double tmp_d = 0.0;
                for(int mq = 0; mq < twojj_q + 1; mq++)
                {
                    if(intType.substr(0,4) == "LSLS")
                        tmp_d += int2e_get_angular_gauge_LSLS(l_p, twojj_p, sym_ap, l_p, twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, LL+LminJ[2], 1, -1);
                    else if(intType.substr(0,4) == "LSSL")
                        tmp_d += int2e_get_angular_gauge_LSSL(l_p, twojj_p, sym_ap, l_p, twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, LL+LminJ[2], 1, -1);
                    else
                    {
                        cout << "ERROR: Unkonwn intType in get_h2e_JK_gauge." << endl;
                        exit(99);
                    }
                }
                tmp_d /= (twojj_q + 1);
                array_angular_Jpm[LL][int_tmp2_p][int_tmp2_q] = tmp_d;
            }
            for(int LL = LmaxK[2]-LminK[2]; LL >= 0; LL-=2)
            {
                double tmp_d = 0.0;
                for(int mq = 0; mq < twojj_q + 1; mq++)
                {
                    if(intType.substr(0,4) == "LSLS")
                        tmp_d += int2e_get_angular_gauge_LSLS(l_p, twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, l_p, twojj_p, sym_ap, LL+LminK[2], 1, -1);
                    else if(intType.substr(0,4) == "LSSL")
                        tmp_d += int2e_get_angular_gauge_LSSL(l_p, twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, l_p, twojj_p, sym_ap, LL+LminK[2], 1, -1);
                    else
                    {
                        cout << "ERROR: Unkonwn intType in get_h2e_JK_gauge." << endl;
                        exit(99);
                    }
                }
                tmp_d /= (twojj_q + 1);
                array_angular_Kpm[LL][int_tmp2_p][int_tmp2_q] = tmp_d;
            }
            for(int LL = LmaxJ[3]-LminJ[3]; LL >= 0; LL-=2)
            {
                double tmp_d = 0.0;
                for(int mq = 0; mq < twojj_q + 1; mq++)
                {
                    if(intType.substr(0,4) == "LSLS")
                        tmp_d += int2e_get_angular_gauge_LSLS(l_p, twojj_p, sym_ap, l_p, twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, LL+LminJ[3], 1, 1);
                    else if(intType.substr(0,4) == "LSSL")
                        tmp_d += int2e_get_angular_gauge_LSSL(l_p, twojj_p, sym_ap, l_p, twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, LL+LminJ[3], 1, 1);
                    else
                    {
                        cout << "ERROR: Unkonwn intType in get_h2e_JK_gauge." << endl;
                        exit(99);
                    }
                }
                tmp_d /= (twojj_q + 1);
                array_angular_Jpp[LL][int_tmp2_p][int_tmp2_q] = tmp_d;
            }
            for(int LL = LmaxK[3]-LminK[3]; LL >= 0; LL-=2)
            {
                double tmp_d = 0.0;
                for(int mq = 0; mq < twojj_q + 1; mq++)
                {
                    if(intType.substr(0,4) == "LSLS")
                        tmp_d += int2e_get_angular_gauge_LSLS(l_p, twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, l_p, twojj_p, sym_ap, LL+LminK[3], 1, 1);
                    else if(intType.substr(0,4) == "LSSL")
                        tmp_d += int2e_get_angular_gauge_LSSL(l_p, twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, l_p, twojj_p, sym_ap, LL+LminK[3], 1, 1);
                    else
                    {
                        cout << "ERROR: Unkonwn intType in get_h2e_JK_gauge." << endl;
                        exit(99);
                    }
                }
                tmp_d /= (twojj_q + 1);
                array_angular_Kpp[LL][int_tmp2_p][int_tmp2_q] = tmp_d;
            }

            // Radial 
            #pragma omp parallel  for
            for(int tt = 0; tt < size_gtos_p*size_gtos_p*size_gtos_q*size_gtos_q; tt++)
            {
                double radial_J_mm, radial_K_mm, radial_J_mp, radial_K_mp, radial_J_pm, radial_K_pm, radial_J_pp, radial_K_pp;
                int e1J = tt/(size_gtos_q*size_gtos_q);
                int e2J = tt - e1J*(size_gtos_q*size_gtos_q);
                int e1K = tt/(size_gtos_p*size_gtos_q);
                int e2K = tt - e1K*(size_gtos_p*size_gtos_q);
                int ii = e1J/size_gtos_p, jj = e1J - ii*size_gtos_p;
                int kk = e2J/size_gtos_q, ll = e2J - kk*size_gtos_q;
                double k_p = -(twojj_p+1.0)*sym_ap/2.0, k_q = -(twojj_q+1.0)*sym_aq/2.0;
                double norm_J = shell_list(pshell).norm(ii) * shell_list(pshell).norm(jj) * shell_list(qshell).norm(kk) * shell_list(qshell).norm(ll), norm_K = shell_list(pshell).norm(ii) * shell_list(qshell).norm(ll) * shell_list(qshell).norm(kk) * shell_list(pshell).norm(jj);
                double lk1 = 1+l_p+k_p, lk2 = 1+l_p+k_p, lk3 = 1+l_q+k_q, lk4 = 1+l_q+k_q, a1 = shell_list(pshell).exp_a(ii), a2 = shell_list(pshell).exp_a(jj), a3 = shell_list(qshell).exp_a(kk), a4 = shell_list(qshell).exp_a(ll);

                int_2e_JK.J[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1J][e2J] = 0.0;
                int_2e_JK.K[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1K][e2K] = 0.0;

                for(int LL = LmaxJ[0]-LminJ[0]; LL >= 0; LL-=2)
                {
                    if(intType == "LSLS")
                    {
                        radial_J_mm = get_radial_LSLS_J(l_p,l_q,LL+LminJ[0],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Jmm[LL][tt],false);
                        radial_J_mm /= -2.0 * norm_J * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "LSSL")
                    {
                        radial_J_mm = get_radial_LSSL_J(l_p,l_q,LL+LminJ[0],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Jmm[LL][tt],false);
                        radial_J_mm /= 2.0 * norm_J * 4.0 * pow(speedOfLight,2);
                    }
                    int_2e_JK.J[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1J][e2J] += radial_J_mm * array_angular_Jmm[LL][int_tmp2_p][int_tmp2_q];
                }
                for(int LL = LmaxJ[1]-LminJ[1]; LL >= 0; LL-=2)
                {
                    if(intType == "LSLS")
                    {
                        radial_J_mp = get_radial_LSLS_J(l_p,l_q,LL+LminJ[1],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Jmp[LL][tt],false);
                        radial_J_mp /= -2.0 * norm_J * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "LSSL")
                    {
                        radial_J_mp = get_radial_LSSL_J(l_p,l_q,LL+LminJ[1],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Jmp[LL][tt],false);
                        radial_J_mp /= 2.0 * norm_J * 4.0 * pow(speedOfLight,2);
                    }
                    int_2e_JK.J[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1J][e2J] += radial_J_mp * array_angular_Jmp[LL][int_tmp2_p][int_tmp2_q];
                }
                for(int LL = LmaxJ[2]-LminJ[2]; LL >= 0; LL-=2)
                {
                    if(intType == "LSLS")
                    {
                        radial_J_pm = get_radial_LSLS_J(l_p,l_q,LL+LminJ[2],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Jpm[LL][tt],false);
                        radial_J_pm /= -2.0 * norm_J * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "LSSL")
                    {
                        radial_J_pm = get_radial_LSSL_J(l_p,l_q,LL+LminJ[2],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Jpm[LL][tt],false);
                        radial_J_pm /= 2.0 * norm_J * 4.0 * pow(speedOfLight,2);
                    }
                    int_2e_JK.J[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1J][e2J] += radial_J_pm * array_angular_Jpm[LL][int_tmp2_p][int_tmp2_q];
                }
                for(int LL = LmaxJ[3]-LminJ[3]; LL >= 0; LL-=2)
                {
                    if(intType == "LSLS")
                    {
                        radial_J_pp = get_radial_LSLS_J(l_p,l_q,LL+LminJ[3],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Jpp[LL][tt],false);
                        radial_J_pp /= -2.0 * norm_J * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "LSSL")
                    {
                        radial_J_pp = get_radial_LSSL_J(l_p,l_q,LL+LminJ[3],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Jpp[LL][tt],false);
                        radial_J_pp /= 2.0 * norm_J * 4.0 * pow(speedOfLight,2);
                    }
                    int_2e_JK.J[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1J][e2J] += radial_J_pp * array_angular_Jpp[LL][int_tmp2_p][int_tmp2_q];
                }
                lk2 = 1+l_q+k_q; lk4 = 1+l_p+k_p; 
                a2 = shell_list(qshell).exp_a(ll); a4 = shell_list(pshell).exp_a(jj);
                for(int LL = LmaxK[0]-LminK[0]; LL >= 0; LL-=2)
                {
                    if(intType == "LSLS")
                    {
                        radial_K_mm = get_radial_LSLS_K(l_p,l_q,LL+LminK[0],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Kmm[LL][tt],false);
                        radial_K_mm /= -2.0 * norm_K * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "LSSL")
                    {
                        radial_K_mm = get_radial_LSSL_K(l_p,l_q,LL+LminK[0],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Kmm[LL][tt],false);
                        radial_K_mm /= 2.0 * norm_K * 4.0 * pow(speedOfLight,2);
                    }
                    int_2e_JK.K[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1K][e2K] += radial_K_mm * array_angular_Kmm[LL][int_tmp2_p][int_tmp2_q];
                }
                for(int LL = LmaxK[1]-LminK[1]; LL >= 0; LL-=2)
                {
                    if(intType == "LSLS")
                    {
                        radial_K_mp = get_radial_LSLS_K(l_p,l_q,LL+LminK[1],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Kmp[LL][tt],false);
                        radial_K_mp /= -2.0 * norm_K * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "LSSL")
                    {
                        radial_K_mp = get_radial_LSSL_K(l_p,l_q,LL+LminK[1],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Kmp[LL][tt],false);
                        radial_K_mp /= 2.0 * norm_K * 4.0 * pow(speedOfLight,2);
                    }
                    int_2e_JK.K[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1K][e2K] += radial_K_mp * array_angular_Kmp[LL][int_tmp2_p][int_tmp2_q];
                }
                for(int LL = LmaxK[2]-LminK[2]; LL >= 0; LL-=2)
                {
                    if(intType == "LSLS")
                    {
                        radial_K_pm = get_radial_LSLS_K(l_p,l_q,LL+LminK[2],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Kpm[LL][tt],false);
                        radial_K_pm /= -2.0 * norm_K * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "LSSL")
                    {
                        radial_K_pm = get_radial_LSSL_K(l_p,l_q,LL+LminK[2],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Kpm[LL][tt],false);
                        radial_K_pm /= 2.0 * norm_K * 4.0 * pow(speedOfLight,2);
                    }
                    int_2e_JK.K[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1K][e2K] += radial_K_pm * array_angular_Kpm[LL][int_tmp2_p][int_tmp2_q];
                }
                for(int LL = LmaxK[3]-LminK[3]; LL >= 0; LL-=2)
                {
                    if(intType == "LSLS")
                    {
                        radial_K_pp = get_radial_LSLS_K(l_p,l_q,LL+LminK[3],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Kpp[LL][tt],false);
                        radial_K_pp /= -2.0 * norm_K * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "LSSL")
                    {
                        radial_K_pp = get_radial_LSSL_K(l_p,l_q,LL+LminK[3],a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_Kpp[LL][tt],false);
                        radial_K_pp /= 2.0 * norm_K * 4.0 * pow(speedOfLight,2);
                    }
                    int_2e_JK.K[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1K][e2K] += radial_K_pp * array_angular_Kpp[LL][int_tmp2_p][int_tmp2_q];
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