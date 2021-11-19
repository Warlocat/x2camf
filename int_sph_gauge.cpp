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

/* 
    evaluate radial part and angular part in 2e integrals 
*/
double INT_SPH::int2e_get_radial_gauge(const int& l1, const double& a1, const int& l2, const double& a2, const int& l3, const double& a3, const int& l4, const double& a4, const int& LL, const int& v1, const int& v2) const
{
    double radial = 0.0;
    if(v1 == v2) radial = 2.0/(2.0*v1+1.0)*int2e_get_radial(l1,a1,l2,a2,l3,a3,l4,a4,LL);
    else if(v2-v1 == 2)
    {
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
    else if(v1-v2 == 2)
    {
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
        cout << "ERROR: Input v1,v2,L are inconsistent." << endl;
        exit(99);
    }

    return -radial*(2.0*LL+1.0);
}


double INT_SPH::int2e_get_angular_gauge_LSLS(const int& l1, const int& two_m1, const int& s1, const int& l2, const int& two_m2, const int& s2, const int& l3, const int& two_m3, const int& s3, const int& l4, const int& two_m4, const int& s4, const int& LL, const int& v1, const int& v2) const
{
    int l2p = l2+s2, l4p = l4+s4;
    int two_j1 = 2*l1+s1, two_j2 = 2*l2+s2, two_j3 = 2*l3+s3, two_j4 = 2*l4+s4;
    double threeJ1 = wigner_3j_zeroM(l2p,v1,l1), threeJ2 = wigner_3j_zeroM(l3,v2,l4p), tmp = 0.0;
    double rme1 = int2e_get_angularX_RME(two_j2,l2p,two_j1,l1,LL,v1,threeJ1);
    double rme2 = int2e_get_angularX_RME(two_j3,l3,two_j4,l4p,LL,v2,threeJ2);
    for(int MMM = -LL; MMM <= LL; MMM++)
    {
        tmp += gsl_sf_coupling_3j(two_j2,2*LL,two_j1,-two_m2,2*MMM,two_m1)
             * gsl_sf_coupling_3j(two_j3,2*LL,two_j4,-two_m3,2*MMM,two_m4);
    }
    
    return tmp*rme1*rme2*pow(-1,(two_j2+two_j3-two_m2-two_m3)/2);
}
double INT_SPH::int2e_get_angular_gauge_LSSL(const int& l1, const int& two_m1, const int& s1, const int& l2, const int& two_m2, const int& s2, const int& l3, const int& two_m3, const int& s3, const int& l4, const int& two_m4, const int& s4, const int& LL, const int& v1, const int& v2) const
{
    int l2p = l2+s2, l3p = l3+s3;
    int two_j1 = 2*l1+s1, two_j2 = 2*l2+s2, two_j3 = 2*l3+s3, two_j4 = 2*l4+s4, vv = LL;
    double threeJ1 = wigner_3j_zeroM(l2p,v1,l1), threeJ2 = wigner_3j_zeroM(l3p,v2,l4), tmp = 0.0;
    double rme1 = int2e_get_angularX_RME(two_j2,l2p,two_j1,l1,LL,v1,threeJ1);
    double rme2 = int2e_get_angularX_RME(two_j3,l3p,two_j4,l4,LL,v2,threeJ2);
    for(int MMM = -LL; MMM <= LL; MMM++)
    {
        tmp += gsl_sf_coupling_3j(two_j2,2*LL,two_j1,-two_m2,2*MMM,two_m1)
             * gsl_sf_coupling_3j(two_j3,2*LL,two_j4,-two_m3,2*MMM,two_m4);
    }
    
    return tmp*rme1*rme2*pow(-1,(two_j2+two_j3-two_m2-two_m3)/2);
}


int2eJK INT_SPH::get_h2e_JK_gauge_compact(const string& intType, const int& occMaxL) const
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
        int l_q = shell_list(qshell).l, l_max = max(l_p,l_q), LmaxJ = min(l_p+l_p, l_q+l_q)+1, LmaxK = l_p+l_q+1;
        int size_gtos_p = shell_list(pshell).coeff.rows(), size_gtos_q = shell_list(qshell).coeff.rows();
        int size_tmp_p = (l_p == 0) ? 1 : 2, size_tmp_q = (l_q == 0) ? 1 : 2;
        double array_radial_J[LmaxJ+1][2][2][size_gtos_p*size_gtos_p][size_gtos_q*size_gtos_q][size_tmp_p][size_tmp_q];
        double array_radial_K[LmaxK+1][2][2][size_gtos_p*size_gtos_q][size_gtos_q*size_gtos_p][size_tmp_p][size_tmp_q];
        double array_angular_J[LmaxJ+1][2][2][size_tmp_p][size_tmp_q], array_angular_K[LmaxK+1][2][2][size_tmp_p][size_tmp_q];

        StartTime = clock();
        #pragma omp parallel  for
        for(int twojj_p = abs(2*l_p-1); twojj_p <= 2*l_p+1; twojj_p = twojj_p + 2)
        for(int twojj_q = abs(2*l_q-1); twojj_q <= 2*l_q+1; twojj_q = twojj_q + 2)
        {
            int sym_ap = twojj_p - 2*l_p, sym_aq = twojj_q - 2*l_q;
            int index_tmp_p = (l_p > 0) ? 1 - (2*l_p+1 - twojj_p)/2 : 0;
            int index_tmp_q = (l_q > 0) ? 1 - (2*l_q+1 - twojj_q)/2 : 0;

            for(int tmp = LmaxJ; tmp >= 0; tmp--)
            {
                double tmp_d[2][2];
                for(int dd = 0; dd < 2; dd++)
                for(int ee = 0; ee < 2; ee++)
                {
                    tmp_d[dd][ee] = 0.0;
                }
                for(int mq = 0; mq < twojj_q + 1; mq++)
                {
                    if(intType.substr(0,4) == "LSLS")
                    {
                        tmp_d[0][0] += int2e_get_angular_gauge_LSLS(l_p, twojj_p, sym_ap, l_p, twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, tmp, tmp+1, tmp+1);
                        if(tmp >= 1)
                        {
                            tmp_d[0][1] += int2e_get_angular_gauge_LSLS(l_p, twojj_p, sym_ap, l_p, twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, tmp, tmp+1, tmp-1);
                            tmp_d[1][0] += int2e_get_angular_gauge_LSLS(l_p, twojj_p, sym_ap, l_p, twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, tmp, tmp-1, tmp+1);
                            tmp_d[1][1] += int2e_get_angular_gauge_LSLS(l_p, twojj_p, sym_ap, l_p, twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, tmp, tmp-1, tmp-1);
                        }
                    }
                    else if(intType.substr(0,4) == "LSSL")
                    {
                        tmp_d[0][0] += int2e_get_angular_gauge_LSSL(l_p, twojj_p, sym_ap, l_p, twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, tmp, tmp+1, tmp+1);
                        if(tmp >= 1)
                        {
                            tmp_d[0][1] += int2e_get_angular_gauge_LSSL(l_p, twojj_p, sym_ap, l_p, twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, tmp, tmp+1, tmp-1);
                            tmp_d[1][0] += int2e_get_angular_gauge_LSSL(l_p, twojj_p, sym_ap, l_p, twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, tmp, tmp-1, tmp+1);
                            tmp_d[1][1] += int2e_get_angular_gauge_LSSL(l_p, twojj_p, sym_ap, l_p, twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, tmp, tmp-1, tmp-1);
                        }
                    }
                    else
                    {
                        cout << "ERROR: Unkonwn intType in get_h2e_JK_gaunt." << endl;
                        exit(99);
                    }
                }
                for(int dd = 0; dd < 2; dd++)
                for(int ee = 0; ee < 2; ee++)
                {
                    tmp_d[dd][ee] /= (twojj_q + 1);
                    array_angular_J[tmp][dd][ee][index_tmp_p][index_tmp_q] = tmp_d[dd][ee];
                }
            }
            for(int tmp = LmaxK; tmp >= 0; tmp--)
            {
                double tmp_d[2][2];
                for(int dd = 0; dd < 2; dd++)
                for(int ee = 0; ee < 2; ee++)
                {
                    tmp_d[dd][ee] = 0.0;
                }
                for(int mq = 0; mq < twojj_q + 1; mq++)
                {
                    if(intType.substr(0,4) == "LSLS")
                    {
                        tmp_d[0][0] += int2e_get_angular_gauge_LSLS(l_p, twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, l_p, twojj_p, sym_ap, tmp, tmp+1, tmp+1);
                        if(tmp >= 1)
                        {
                            tmp_d[0][1] += int2e_get_angular_gauge_LSLS(l_p, twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, l_p, twojj_p, sym_ap, tmp, tmp+1, tmp-1);
                            tmp_d[1][0] += int2e_get_angular_gauge_LSLS(l_p, twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, l_p, twojj_p, sym_ap, tmp, tmp-1, tmp+1);
                            tmp_d[1][1] += int2e_get_angular_gauge_LSLS(l_p, twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, l_p, twojj_p, sym_ap, tmp, tmp-1, tmp-1);
                        }
                    }
                    else if(intType.substr(0,4) == "LSSL")
                    {
                        tmp_d[0][0] += int2e_get_angular_gauge_LSSL(l_p, twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, l_p, twojj_p, sym_ap, tmp, tmp+1, tmp+1);
                        if(tmp >= 1)
                        {
                            tmp_d[0][1] += int2e_get_angular_gauge_LSSL(l_p, twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, l_p, twojj_p, sym_ap, tmp, tmp+1, tmp-1);
                            tmp_d[1][0] += int2e_get_angular_gauge_LSSL(l_p, twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, l_p, twojj_p, sym_ap, tmp, tmp-1, tmp+1);
                            tmp_d[1][1] += int2e_get_angular_gauge_LSSL(l_p, twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, l_p, twojj_p, sym_ap, tmp, tmp-1, tmp-1);
                        }
                    }
                    else
                    {
                        cout << "ERROR: Unkonwn intType in get_h2e_JK_gaunt." << endl;
                        exit(99);
                    }
                }
                for(int dd = 0; dd < 2; dd++)
                for(int ee = 0; ee < 2; ee++)
                {
                    tmp_d[dd][ee] /= (twojj_q + 1);
                    array_angular_K[tmp][dd][ee][index_tmp_p][index_tmp_q] = tmp_d[dd][ee];
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
            MatrixXd radial_2e_list_J[LmaxJ+1][2][2], radial_2e_list_K[LmaxK+1][2][2];
            double a_i_J = shell_list(pshell).exp_a(ii), a_j_J = shell_list(pshell).exp_a(jj), a_k_J = shell_list(qshell).exp_a(kk), a_l_J = shell_list(qshell).exp_a(ll);
            double a_i_K = shell_list(pshell).exp_a(ii), a_j_K = shell_list(qshell).exp_a(ll), a_k_K = shell_list(qshell).exp_a(kk), a_l_K = shell_list(pshell).exp_a(jj);
        
            if(intType.substr(0,4) == "LSLS")
            {
                for(int LL = LmaxJ; LL >= 0; LL--)
                for(int dd = 0; dd < 2; dd++)
                for(int ee = 0; ee < 2; ee++)
                {
                    int v1 = LL-dd*2+1, v2 = LL-ee*2+1;
                    radial_2e_list_J[LL][dd][ee].resize(4,1);
                    radial_2e_list_J[LL][dd][ee](0,0) = int2e_get_radial_gauge(l_p,a_i_J,l_p+1,a_j_J,l_q,a_k_J,l_q+1,a_l_J,LL,v1,v2);
                    if(l_p != 0)
                        radial_2e_list_J[LL][dd][ee](1,0) = int2e_get_radial_gauge(l_p,a_i_J,l_p-1,a_j_J,l_q,a_k_J,l_q+1,a_l_J,LL,v1,v2);
                    if(l_q != 0)
                        radial_2e_list_J[LL][dd][ee](2,0) = int2e_get_radial_gauge(l_p,a_i_J,l_p+1,a_j_J,l_q,a_k_J,l_q-1,a_l_J,LL,v1,v2);
                    if(l_p != 0 && l_q != 0)
                        radial_2e_list_J[LL][dd][ee](3,0) = int2e_get_radial_gauge(l_p,a_i_J,l_p-1,a_j_J,l_q,a_k_J,l_q-1,a_l_J,LL,v1,v2);
                }
                for(int LL = LmaxK; LL >= 0; LL--)
                for(int dd = 0; dd < 2; dd++)
                for(int ee = 0; ee < 2; ee++)
                {
                    int v1 = LL-dd*2+1, v2 = LL-ee*2+1;
                    radial_2e_list_K[LL][dd][ee].resize(4,1);
                    radial_2e_list_K[LL][dd][ee](0,0) =  int2e_get_radial_gauge(l_p,a_i_K,l_q+1,a_j_K,l_q,a_k_K,l_p+1,a_l_K,LL,v1,v2);
                    if(l_q != 0)
                        radial_2e_list_K[LL][dd][ee](1,0) =  int2e_get_radial_gauge(l_p,a_i_K,l_q-1,a_j_K,l_q,a_k_K,l_p+1,a_l_K,LL,v1,v2);
                    if(l_p != 0)
                        radial_2e_list_K[LL][dd][ee](2,0) =  int2e_get_radial_gauge(l_p,a_i_K,l_q+1,a_j_K,l_q,a_k_K,l_p-1,a_l_K,LL,v1,v2);
                    if(l_p != 0 && l_q != 0)
                        radial_2e_list_K[LL][dd][ee](3,0) =  int2e_get_radial_gauge(l_p,a_i_K,l_q-1,a_j_K,l_q,a_k_K,l_p-1,a_l_K,LL,v1,v2);
                }
            }
            else if(intType.substr(0,4) == "LSSL")
            {
                for(int LL = LmaxJ; LL >= 0; LL--)
                for(int dd = 0; dd < 2; dd++)
                for(int ee = 0; ee < 2; ee++)
                {
                    int v1 = LL-dd*2+1, v2 = LL-ee*2+1;
                    radial_2e_list_J[LL][dd][ee].resize(4,1);
                    radial_2e_list_J[LL][dd][ee](0,0) =  int2e_get_radial_gauge(l_p,a_i_J,l_p+1,a_j_J,l_q+1,a_k_J,l_q,a_l_J,LL,v1,v2);
                    if(l_p != 0)
                        radial_2e_list_J[LL][dd][ee](1,0) =  int2e_get_radial_gauge(l_p,a_i_J,l_p-1,a_j_J,l_q+1,a_k_J,l_q,a_l_J,LL,v1,v2);
                    if(l_q != 0)
                        radial_2e_list_J[LL][dd][ee](2,0) =  int2e_get_radial_gauge(l_p,a_i_J,l_p+1,a_j_J,l_q-1,a_k_J,l_q,a_l_J,LL,v1,v2);
                    if(l_p != 0 && l_q != 0)
                        radial_2e_list_J[LL][dd][ee](3,0) =  int2e_get_radial_gauge(l_p,a_i_J,l_p-1,a_j_J,l_q-1,a_k_J,l_q,a_l_J,LL,v1,v2);
                }
                for(int LL = LmaxK; LL >= 0; LL--)
                for(int dd = 0; dd < 2; dd++)
                for(int ee = 0; ee < 2; ee++)
                {
                    int v1 = LL-dd*2+1, v2 = LL-ee*2+1;
                    radial_2e_list_K[LL][dd][ee].resize(4,1);
                    radial_2e_list_K[LL][dd][ee](0,0) =  int2e_get_radial_gauge(l_p,a_i_K,l_q+1,a_j_K,l_q+1,a_k_K,l_p,a_l_K,LL,v1,v2);
                    if(l_q != 0)
                    {
                        radial_2e_list_K[LL][dd][ee](1,0) =  int2e_get_radial_gauge(l_p,a_i_K,l_q-1,a_j_K,l_q+1,a_k_K,l_p,a_l_K,LL,v1,v2);
                        radial_2e_list_K[LL][dd][ee](2,0) =  int2e_get_radial_gauge(l_p,a_i_K,l_q+1,a_j_K,l_q-1,a_k_K,l_p,a_l_K,LL,v1,v2);
                        radial_2e_list_K[LL][dd][ee](3,0) =  int2e_get_radial_gauge(l_p,a_i_K,l_q-1,a_j_K,l_q-1,a_k_K,l_p,a_l_K,LL,v1,v2);
                    }
                }
            }
            else
            {
                cout << "ERROR: Unkonwn intType in get_h2e_JK_gaunt." << endl;
                exit(99);
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

                for(int tmp = LmaxJ; tmp >= 0; tmp--)
                for(int dd = 0; dd < 2; dd++)
                for(int ee = 0; ee < 2; ee++)
                {
                    if(intType == "LSLS")
                    {
                        array_radial_J[tmp][dd][ee][e1J][e2J][index_tmp_p][index_tmp_q] = 4.0*a2*a4 * radial_2e_list_J[tmp][dd][ee](0,0);
                        if(l_p != 0 && l_q != 0)
                            array_radial_J[tmp][dd][ee][e1J][e2J][index_tmp_p][index_tmp_q] += lk2*lk4 * radial_2e_list_J[tmp][dd][ee](3,0)
                                    - 2.0*a4*lk2 * radial_2e_list_J[tmp][dd][ee](1,0) - 2.0*a2*lk4 * radial_2e_list_J[tmp][dd][ee](2,0);
                        else if(l_p != 0 && l_q == 0)
                            array_radial_J[tmp][dd][ee][e1J][e2J][index_tmp_p][index_tmp_q] -= 2.0*a4*lk2 * radial_2e_list_J[tmp][dd][ee](1,0);
                        else if(l_p == 0 && l_q != 0)
                            array_radial_J[tmp][dd][ee][e1J][e2J][index_tmp_p][index_tmp_q] -= 2.0*a2*lk4 * radial_2e_list_J[tmp][dd][ee](2,0);
                        array_radial_J[tmp][dd][ee][e1J][e2J][index_tmp_p][index_tmp_q] /= -1.0 * norm_J * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "LSSL")
                    {
                        array_radial_J[tmp][dd][ee][e1J][e2J][index_tmp_p][index_tmp_q] = 4.0*a2*a3 * radial_2e_list_J[tmp][dd][ee](0,0);
                        if(l_p != 0 && l_q != 0)
                            array_radial_J[tmp][dd][ee][e1J][e2J][index_tmp_p][index_tmp_q] += lk2*lk3 * radial_2e_list_J[tmp][dd][ee](3,0)
                                    - 2.0*a3*lk2 * radial_2e_list_J[tmp][dd][ee](1,0) - 2.0*a2*lk3 * radial_2e_list_J[tmp][dd][ee](2,0);
                        else if(l_p != 0 && l_q == 0)
                            array_radial_J[tmp][dd][ee][e1J][e2J][index_tmp_p][index_tmp_q] -= 2.0*a3*lk2 * radial_2e_list_J[tmp][dd][ee](1,0);
                        else if(l_p == 0 && l_q != 0)
                            array_radial_J[tmp][dd][ee][e1J][e2J][index_tmp_p][index_tmp_q] -= 2.0*a2*lk3 * radial_2e_list_J[tmp][dd][ee](2,0);
                        array_radial_J[tmp][dd][ee][e1J][e2J][index_tmp_p][index_tmp_q] /= norm_J * 4.0 * pow(speedOfLight,2);
                    }
                    else
                    {
                        cout << "ERROR: Unkonwn intType in get_h2e_JK_gaunt." << endl;
                        exit(99);
                    }
                }
                lk2 = 1+l_q+k_q; lk4 = 1+l_p+k_p; 
                a2 = shell_list(qshell).exp_a(ll); a4 = shell_list(pshell).exp_a(jj);
                for(int tmp = LmaxK; tmp >= 0; tmp--)
                for(int dd = 0; dd < 2; dd++)
                for(int ee = 0; ee < 2; ee++)
                {
                    if(intType == "LSLS")
                    {
                        array_radial_K[tmp][dd][ee][e1K][e2K][index_tmp_p][index_tmp_q] = 4.0*a2*a4 * radial_2e_list_K[tmp][dd][ee](0,0);
                        if(l_p != 0 && l_q != 0)
                            array_radial_K[tmp][dd][ee][e1K][e2K][index_tmp_p][index_tmp_q] += lk2*lk4 * radial_2e_list_K[tmp][dd][ee](3,0) 
                                    - 2.0*a4*lk2 * radial_2e_list_K[tmp][dd][ee](1,0) - 2.0*a2*lk4 * radial_2e_list_K[tmp][dd][ee](2,0);
                        else if(l_p == 0 && l_q != 0)
                            array_radial_K[tmp][dd][ee][e1K][e2K][index_tmp_p][index_tmp_q] -= 2.0*a4*lk2 * radial_2e_list_K[tmp][dd][ee](1,0);
                        else if(l_p != 0 && l_q == 0)
                            array_radial_K[tmp][dd][ee][e1K][e2K][index_tmp_p][index_tmp_q] -= 2.0*a2*lk4 * radial_2e_list_K[tmp][dd][ee](2,0);
                        array_radial_K[tmp][dd][ee][e1K][e2K][index_tmp_p][index_tmp_q] /= -1.0 * norm_K * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "LSSL")
                    {
                        array_radial_K[tmp][dd][ee][e1K][e2K][index_tmp_p][index_tmp_q] = 4.0*a2*a3 * radial_2e_list_K[tmp][dd][ee](0,0);
                        if(l_q != 0)
                            array_radial_K[tmp][dd][ee][e1K][e2K][index_tmp_p][index_tmp_q] += lk2*lk3 * radial_2e_list_K[tmp][dd][ee](3,0) 
                                    - 2.0*a3*lk2 * radial_2e_list_K[tmp][dd][ee](1,0) - 2.0*a2*lk3 * radial_2e_list_K[tmp][dd][ee](2,0);
                        array_radial_K[tmp][dd][ee][e1K][e2K][index_tmp_p][index_tmp_q] /= norm_K * 4.0 * pow(speedOfLight,2);
                    }
                    else
                    {
                        cout << "ERROR: Unkonwn intType in get_h2e_JK_gaunt." << endl;
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
            int_2e_JK.J[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q] = new double*[size_gtos_p*size_gtos_p];
            for(int iii = 0; iii < size_gtos_p*size_gtos_p; iii++)
                int_2e_JK.J[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][iii] = new double[size_gtos_q*size_gtos_q];
            int_2e_JK.K[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q] = new double*[size_gtos_p*size_gtos_q];
            for(int iii = 0; iii < size_gtos_p*size_gtos_q; iii++)
                int_2e_JK.K[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][iii] = new double[size_gtos_p*size_gtos_q];
            #pragma omp parallel  for
            for(int tt = 0; tt < size_gtos_p*size_gtos_p*size_gtos_q*size_gtos_q; tt++)
            {
                int e1J = tt/(size_gtos_q*size_gtos_q);
                int e2J = tt - e1J*(size_gtos_q*size_gtos_q);
                int e1K = tt/(size_gtos_p*size_gtos_q);
                int e2K = tt - e1K*(size_gtos_p*size_gtos_q);
                int_2e_JK.J[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1J][e2J] = 0.0;
                int_2e_JK.K[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1K][e2K] = 0.0;
                for(int tmp = LmaxJ; tmp >= 0; tmp--)
                for(int dd = 0; dd < 2; dd++)
                for(int ee = 0; ee < 2; ee++)
                {
                    int v1 = tmp+2*dd-1, v2 = tmp+2*ee-1;
                    if(dd+ee==0 || tmp >=1)
                        int_2e_JK.J[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1J][e2J] += array_radial_J[tmp][dd][ee][e1J][e2J][int_tmp2_p][int_tmp2_q] * array_angular_J[tmp][dd][ee][int_tmp2_p][int_tmp2_q]*0.5*(2.0*v1+1.0)*(2.0*v2+1.0)*wigner_3j_zeroM(1,v1,tmp)*wigner_3j_zeroM(1,v2,tmp);
                }
                for(int tmp = LmaxK; tmp >= 0; tmp--)
                for(int dd = 0; dd < 2; dd++)
                for(int ee = 0; ee < 2; ee++)
                {
                    int v1 = tmp+2*dd-1, v2 = tmp+2*ee-1;
                    if(dd+ee==0 || tmp >=1)
                        int_2e_JK.K[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1K][e2K] += array_radial_K[tmp][dd][ee][e1K][e2K][int_tmp2_p][int_tmp2_q] * array_angular_K[tmp][dd][ee][int_tmp2_p][int_tmp2_q]*0.5*(2.0*v1+1.0)*(2.0*v2+1.0)*wigner_3j_zeroM(1,v1,tmp)*wigner_3j_zeroM(1,v2,tmp);
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
    return int_2e_JK;
}


void INT_SPH::get_h2e_JK_gauge_direct(int2eJK& LSLS, int2eJK& LSSL, const int& occMaxL, const bool& spinFree)
{
    LSLS = get_h2e_JK_gauge_compact("LSLS",occMaxL);
    LSSL = get_h2e_JK_gauge_compact("LSSL",occMaxL);
    
    return;
}