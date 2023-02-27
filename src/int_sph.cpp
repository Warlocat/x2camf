#include<string>
#include<iostream>
#include<iomanip>
#include<fstream>
#include<cmath>
#include<complex>
#include<omp.h>
#include"int_sph.h"
using namespace std;

/*
    Evaluate different one-electron integral in 2-spinor basis
*/
vVectorXd INT_SPH::get_h1e(const string& intType) const
{
    vVectorXd int_1e(Nirrep);
    // isotope mass for finite nuclear calculation
    vector<double> mass_tmp = {1,4,7,9,11,12,14,16,19,20,23,24,27,28,31,32,35,40,39,40,45,48,51,52,55,56,59,58,63,
                64,69,74,75,80,79,84,85,88,89,90,93,98,98,102,103,106,107,114,115,120,121,130,127,
                132,133,138,139,140,141,144,145,152,153,158,159,162,162,168,169,174,175,180,181,184,
                187,192,193,195,197,202,205,208,209,209,210,222,223,226,227,232,231,238,237,244,243,
                247,247,251,252,257,258,259,266,267,268,269,270,277,278,281,282,285,286,289,290,293,294,294};
    int int_tmp = 0;
    for(int irrep = 0; irrep < Nirrep; irrep++)
    {
        int_1e[irrep].resize(irrep_list[irrep].size*irrep_list[irrep].size, 0.0);
    }
    for(int ishell = 0; ishell < size_shell; ishell++)
    {
        int ll = shell_list[ishell].l;
        int size_gtos = shell_list[ishell].nunc;
        vVectorXd h1e_single_shell;
        if(ll == 0) h1e_single_shell.resize(1);
        else    h1e_single_shell.resize(2);
        for(int ii = 0; ii < h1e_single_shell.size(); ii++)
            h1e_single_shell[ii].resize(size_gtos*size_gtos);
        
        for(int ii = 0; ii < size_gtos; ii++)
        for(int jj = 0; jj < size_gtos; jj++)
        {
            double a1 = shell_list[ishell].exp_a[ii], a2 = shell_list[ishell].exp_a[jj];
            vector<double> auxiliary_1e_list(6);
            for(int mm = 0; mm <= 4; mm++)
                auxiliary_1e_list[mm] = auxiliary_1e(2*ll + mm, a1 + a2);
            if(ll != 0)
            {
                auxiliary_1e_list[5] = auxiliary_1e(2*ll - 1, a1 + a2);
            }
            else
            {
                auxiliary_1e_list[5] = 0.0;
            }
            for(int twojj = abs(2*ll-1); twojj <= 2*ll+1; twojj = twojj + 2)
            {
                double kappa = (twojj + 1.0) * (ll - twojj/2.0);
                int index_tmp = 1 - (2*ll+1 - twojj)/2;
                if(ll == 0) index_tmp = 0;
                
                if(intType == "s_p_nuc_s_p")
                {
                    h1e_single_shell[index_tmp][ii*size_gtos+jj] = 4*a1*a2 * auxiliary_1e_list[3];
                    if(ll!=0)
                        h1e_single_shell[index_tmp][ii*size_gtos+jj] += pow(ll + kappa + 1.0, 2) * auxiliary_1e_list[5] - 2.0*(ll + kappa + 1.0)*(a1 + a2)*auxiliary_1e_list[1];
                    h1e_single_shell[index_tmp][ii*size_gtos+jj] *= -atomNumber;
                }
                else if(intType == "s_p_nuc_s_p_sf")
                {
                    h1e_single_shell[index_tmp][ii*size_gtos+jj] = 4*a1*a2 * auxiliary_1e_list[3];
                    if(ll!=0)
                        h1e_single_shell[index_tmp][ii*size_gtos+jj] += (2*ll*ll + ll) * auxiliary_1e_list[5] - 2.0*ll*(a1 + a2)*auxiliary_1e_list[1];
                    h1e_single_shell[index_tmp][ii*size_gtos+jj] *= -atomNumber;
                }
                else if(intType == "s_p_nuc_s_p_sd")
                {
                    h1e_single_shell[index_tmp][ii*size_gtos+jj] = 0.0;
                    if(ll!=0)
                        h1e_single_shell[index_tmp][ii*size_gtos+jj] += (kappa + 1.0) * auxiliary_1e_list[5];
                    h1e_single_shell[index_tmp][ii*size_gtos+jj] *= -atomNumber;
                }
                else if(intType == "s_p_s_p" )
                {
                    h1e_single_shell[index_tmp][ii*size_gtos+jj] = 4*a1*a2 * auxiliary_1e_list[4];
                    if(ll!=0)
                        h1e_single_shell[index_tmp][ii*size_gtos+jj] += pow(ll + kappa + 1.0, 2) * auxiliary_1e_list[0] - 2.0*(ll + kappa + 1.0)*(a1 + a2)*auxiliary_1e_list[2];
                }
                else if(intType == "overlap")  h1e_single_shell[index_tmp][ii*size_gtos+jj] = auxiliary_1e_list[2];
                else if(intType == "nuc_attra")  h1e_single_shell[index_tmp][ii*size_gtos+jj] = -atomNumber * auxiliary_1e_list[1];
                else if(intType == "nucGau_attra")
                {
                    double a_13 = pow(mass_tmp[atomNumber-1],1.0/3.0);
                    double rnuc = (0.836*a_13+0.570)/52917.7249, xi = 3.0/2.0/rnuc/rnuc;
                    double norm = -atomNumber*pow(xi/M_PI,1.5);
                    h1e_single_shell[index_tmp][ii*size_gtos+jj] = norm*int2e_get_radial(0,0.0,0,xi,ll,a1,ll,a2,0)*4.0*M_PI;
                }
                else if(intType == "s_p_nucGau_s_p")
                {
                    double a_13 = pow(mass_tmp[atomNumber-1],1.0/3.0);
                    double rnuc = (0.836*a_13+0.570)/52917.7249, xi = 3.0/2.0/rnuc/rnuc;
                    double norm = -atomNumber*pow(xi/M_PI,1.5);
                    double tmp = 4.0*a1*a2*int2e_get_radial(0,0.0,0,xi,ll+1,a1,ll+1,a2,0);
                    if(ll != 0)
                        tmp += (1.0+ll+kappa)*(1.0+ll+kappa)*int2e_get_radial(0,0.0,0,xi,ll-1,a1,ll-1,a2,0) - 2.0*(a1+a2)*(1+ll+kappa)*int2e_get_radial(0,0.0,0,xi,ll+1,a1,ll-1,a2,0);
                    h1e_single_shell[index_tmp][ii*size_gtos+jj] = norm*tmp*4.0*M_PI;
                }
                else if(intType == "s_p_nucGau_s_p_sf")
                {
                    double a_13 = pow(mass_tmp[atomNumber-1],1.0/3.0);
                    double rnuc = (0.836*a_13+0.570)/52917.7249, xi = 3.0/2.0/rnuc/rnuc;
                    double norm = -atomNumber*pow(xi/M_PI,1.5);
                    double tmp = 4.0*a1*a2*int2e_get_radial(0,0.0,0,xi,ll+1,a1,ll+1,a2,0);
                    if(ll != 0)
                        tmp += (2.0*ll*ll + ll)*int2e_get_radial(0,0.0,0,xi,ll-1,a1,ll-1,a2,0) - 2.0*(a1+a2)*(ll)*int2e_get_radial(0,0.0,0,xi,ll+1,a1,ll-1,a2,0);
                    h1e_single_shell[index_tmp][ii*size_gtos+jj] = norm*tmp*4.0*M_PI;
                }
                else if(intType == "kinetic")
                {
                    h1e_single_shell[index_tmp][ii*size_gtos+jj] = 4*a1*a2 * auxiliary_1e_list[4];
                    if(ll!=0)
                        h1e_single_shell[index_tmp][ii*size_gtos+jj] += pow(ll + kappa + 1.0, 2) * auxiliary_1e_list[0] - 2.0*(ll + kappa + 1.0)*(a1 + a2)*auxiliary_1e_list[2];
                    h1e_single_shell[index_tmp][ii*size_gtos+jj] /= 2.0;
                }
                else
                {
                    cout << "ERROR: get_h1e is called for undefined type of integrals!" << endl;
                    exit(99);
                }
                h1e_single_shell[index_tmp][ii*size_gtos+jj] = h1e_single_shell[index_tmp][ii*size_gtos+jj] / shell_list[ishell].norm[ii] / shell_list[ishell].norm[jj];
            }
        }
        for(int ii = 0; ii < irrep_list[int_tmp].two_j + 1; ii++)
            int_1e[int_tmp + ii] = h1e_single_shell[0];
        int_tmp += irrep_list[int_tmp].two_j + 1;
        if(ll != 0)
        {
            for(int ii = 0; ii < irrep_list[int_tmp].two_j + 1; ii++)
                int_1e[int_tmp + ii] = h1e_single_shell[1];
            int_tmp += irrep_list[int_tmp].two_j + 1;
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
        int l_q = shell_list[qshell].l, l_max = max(l_p,l_q), LmaxJ = 0, LmaxK = l_p+l_q;
        int size_gtos_p = shell_list[pshell].nunc, size_gtos_q = shell_list[qshell].nunc;
        int size_tmp_p = (l_p == 0) ? 1 : 2, size_tmp_q = (l_q == 0) ? 1 : 2;
        double array_radial_J[LmaxJ+1][size_gtos_p*size_gtos_p][size_gtos_q*size_gtos_q][size_tmp_p][size_tmp_q];
        double array_radial_K[LmaxK+1][size_gtos_p*size_gtos_q][size_gtos_q*size_gtos_p][size_tmp_p][size_tmp_q];
        vector<double> array_angular_J[LmaxJ+1][size_tmp_p][size_tmp_q], array_angular_K[LmaxK+1][size_tmp_p][size_tmp_q];

        countTime(StartTimeCPU,StartTimeWall);
        #pragma omp parallel  for
        for(int twojj_p = abs(2*l_p-1); twojj_p <= 2*l_p+1; twojj_p = twojj_p + 2)
        for(int twojj_q = abs(2*l_q-1); twojj_q <= 2*l_q+1; twojj_q = twojj_q + 2)
        {
            int sym_ap = twojj_p - 2*l_p, sym_aq = twojj_q - 2*l_q;
            int index_tmp_p = (l_p > 0) ? 1 - (2*l_p+1 - twojj_p)/2 : 0;
            int index_tmp_q = (l_q > 0) ? 1 - (2*l_q+1 - twojj_q)/2 : 0;

            for(int tmp = LmaxJ; tmp >= 0; tmp -= 2)
            {
                array_angular_J[tmp][index_tmp_p][index_tmp_q].resize((twojj_p + 1)*(twojj_q + 1));
                for(int mp = 0; mp < twojj_p + 1; mp++)
                for(int mq = 0; mq < twojj_q + 1; mq++)
                {
                    array_angular_J[tmp][index_tmp_p][index_tmp_q][mp*(twojj_q + 1)+mq] = int2e_get_angular_J(l_p, 2*mp-twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, tmp);
                }
            }
            for(int tmp = LmaxK; tmp >= 0; tmp -= 2)
            {
                array_angular_K[tmp][index_tmp_p][index_tmp_q].resize((twojj_p + 1)*(twojj_q + 1));
                for(int mp = 0; mp < twojj_p + 1; mp++)
                for(int mq = 0; mq < twojj_q + 1; mq++)
                {
                    array_angular_K[tmp][index_tmp_p][index_tmp_q][mp*(twojj_q + 1)+mq] = int2e_get_angular_K(l_p, 2*mp-twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, tmp);
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
            int Nvec;
            vector<double> radial_2e_list_J[LmaxJ+1], radial_2e_list_K[LmaxK+1];
            double a_i_J = shell_list[pshell].exp_a[ii], a_j_J = shell_list[pshell].exp_a[jj], a_k_J = shell_list[qshell].exp_a[kk], a_l_J = shell_list[qshell].exp_a[ll];
            double a_i_K = shell_list[pshell].exp_a[ii], a_j_K = shell_list[qshell].exp_a[ll], a_k_K = shell_list[qshell].exp_a[kk], a_l_K = shell_list[pshell].exp_a[jj];
        
            if(intType.substr(0,4) == "LLLL")
            {
                Nvec = 1;
                for(int LL = LmaxJ; LL >= 0; LL -= 2)
                {
                    radial_2e_list_J[LL].resize(1);
                    radial_2e_list_J[LL][0] = int2e_get_radial(l_p,a_i_J,l_p,a_j_J,l_q,a_k_J,l_q,a_l_J,LL);
                }
                for(int LL = LmaxK; LL >= 0; LL -= 2)
                {
                    radial_2e_list_K[LL].resize(1);
                    radial_2e_list_K[LL][0] = int2e_get_radial(l_p,a_i_K,l_q,a_j_K,l_q,a_k_K,l_p,a_l_K,LL);
                }
            }
            else if(intType.substr(0,4) == "SSLL")
            {
                Nvec = 1;
                for(int LL = LmaxJ; LL >= 0; LL -= 2)
                {
                    radial_2e_list_J[LL].resize(3);
                    radial_2e_list_J[LL][0] = int2e_get_radial(l_p,a_i_J,l_p,a_j_J,l_q,a_k_J,l_q,a_l_J,LL);
                    radial_2e_list_J[LL][1] = int2e_get_radial(l_p+1,a_i_J,l_p+1,a_j_J,l_q,a_k_J,l_q,a_l_J,LL);
                    if(l_p != 0)
                        radial_2e_list_J[LL][2] = int2e_get_radial(l_p-1,a_i_J,l_p-1,a_j_J,l_q,a_k_J,l_q,a_l_J,LL);
                }
                for(int LL = LmaxK; LL >= 0; LL -= 2)
                {
                    radial_2e_list_K[LL].resize(3);
                    radial_2e_list_K[LL][0] = int2e_get_radial(l_p,a_i_K,l_q,a_j_K,l_q,a_k_K,l_p,a_l_K,LL);
                    radial_2e_list_K[LL][1] = int2e_get_radial(l_p+1,a_i_K,l_q+1,a_j_K,l_q,a_k_K,l_p,a_l_K,LL);
                    if(l_p != 0 && l_q != 0)
                        radial_2e_list_K[LL][2] = int2e_get_radial(l_p-1,a_i_K,l_q-1,a_j_K,l_q,a_k_K,l_p,a_l_K,LL);
                }
            }
            else if(intType.substr(0,4) == "SSSS")
            {
                Nvec = 3;
                for(int LL = LmaxJ; LL >= 0; LL -= 2)
                {
                    radial_2e_list_J[LL].resize(3*3);
                    radial_2e_list_J[LL][0*3+0] = int2e_get_radial(l_p,a_i_J,l_p,a_j_J,l_q,a_k_J,l_q,a_l_J,LL);
                    radial_2e_list_J[LL][1*3+0] = int2e_get_radial(l_p+1,a_i_J,l_p+1,a_j_J,l_q,a_k_J,l_q,a_l_J,LL);
                    radial_2e_list_J[LL][0*3+1] = int2e_get_radial(l_p,a_i_J,l_p,a_j_J,l_q+1,a_k_J,l_q+1,a_l_J,LL);
                    radial_2e_list_J[LL][1*3+1] = int2e_get_radial(l_p+1,a_i_J,l_p+1,a_j_J,l_q+1,a_k_J,l_q+1,a_l_J,LL);
                    if(l_p != 0)
                    {
                        radial_2e_list_J[LL][2*3+0] = int2e_get_radial(l_p-1,a_i_J,l_p-1,a_j_J,l_q,a_k_J,l_q,a_l_J,LL);
                        radial_2e_list_J[LL][2*3+1] = int2e_get_radial(l_p-1,a_i_J,l_p-1,a_j_J,l_q+1,a_k_J,l_q+1,a_l_J,LL);
                    }
                    if(l_q != 0)
                    {
                        radial_2e_list_J[LL][0*3+2] = int2e_get_radial(l_p,a_i_J,l_p,a_j_J,l_q-1,a_k_J,l_q-1,a_l_J,LL);
                        radial_2e_list_J[LL][1*3+2] = int2e_get_radial(l_p+1,a_i_J,l_p+1,a_j_J,l_q-1,a_k_J,l_q-1,a_l_J,LL);
                    }
                    if(l_p!=0 && l_q!=0)
                        radial_2e_list_J[LL][2*3+2] = int2e_get_radial(l_p-1,a_i_J,l_p-1,a_j_J,l_q-1,a_k_J,l_q-1,a_l_J,LL);
                }
                for(int LL = LmaxK; LL >= 0; LL -= 2)
                {
                    radial_2e_list_K[LL].resize(3,3);
                    radial_2e_list_K[LL][0*3+0] = int2e_get_radial(l_p,a_i_K,l_q,a_j_K,l_q,a_k_K,l_p,a_l_K,LL);
                    radial_2e_list_K[LL][1*3+0] = int2e_get_radial(l_p+1,a_i_K,l_q+1,a_j_K,l_q,a_k_K,l_p,a_l_K,LL);
                    radial_2e_list_K[LL][0*3+1] = int2e_get_radial(l_p,a_i_K,l_q,a_j_K,l_q+1,a_k_K,l_p+1,a_l_K,LL);
                    radial_2e_list_K[LL][1*3+1] = int2e_get_radial(l_p+1,a_i_K,l_q+1,a_j_K,l_q+1,a_k_K,l_p+1,a_l_K,LL);
                    if(l_p != 0 && l_q != 0)
                    {
                        radial_2e_list_K[LL][2*3+0] = int2e_get_radial(l_p-1,a_i_K,l_q-1,a_j_K,l_q,a_k_K,l_p,a_l_K,LL);
                        radial_2e_list_K[LL][2*3+1] = int2e_get_radial(l_p-1,a_i_K,l_q-1,a_j_K,l_q+1,a_k_K,l_p+1,a_l_K,LL);
                        radial_2e_list_K[LL][0*3+2] = int2e_get_radial(l_p,a_i_K,l_q,a_j_K,l_q-1,a_k_K,l_p-1,a_l_K,LL);
                        radial_2e_list_K[LL][1*3+2] = int2e_get_radial(l_p+1,a_i_K,l_q+1,a_j_K,l_q-1,a_k_K,l_p-1,a_l_K,LL);
                        radial_2e_list_K[LL][2*3+2] = int2e_get_radial(l_p-1,a_i_K,l_q-1,a_j_K,l_q-1,a_k_K,l_p-1,a_l_K,LL);
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
                double norm_J = shell_list[pshell].norm[ii] * shell_list[pshell].norm[jj] * shell_list[qshell].norm[kk] * shell_list[qshell].norm[ll], norm_K = shell_list[pshell].norm[ii] * shell_list[qshell].norm[ll] * shell_list[qshell].norm[kk] * shell_list[pshell].norm[jj];
                double lk1 = 1+l_p+k_p, lk2 = 1+l_p+k_p, lk3 = 1+l_q+k_q, lk4 = 1+l_q+k_q, a1 = shell_list[pshell].exp_a[ii], a2 = shell_list[pshell].exp_a[jj], a3 = shell_list[qshell].exp_a[kk], a4 = shell_list[qshell].exp_a[ll];

                for(int tmp = LmaxJ; tmp >= 0; tmp -= 2)
                {
                    if(intType == "LLLL")
                    {
                        array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] = radial_2e_list_J[tmp][0*Nvec+0] / norm_J;
                    }
                    else if(intType == "SSLL")
                    {
                        array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] = 4.0*a1*a2 * radial_2e_list_J[tmp][1*Nvec+0];
                        if(l_p != 0)
                            array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] += lk1*lk2 * radial_2e_list_J[tmp][2*Nvec+0] - (2.0*a1*lk2+2.0*a2*lk1) * radial_2e_list_J[tmp][0*Nvec+0];
                        array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] /= norm_J * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "SSSS")
                    {
                        array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] = 4*a1*a2*4*a3*a4 * radial_2e_list_J[tmp][1*Nvec+1];
                        if(l_p != 0)
                        {
                            if(l_q != 0)
                                array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] += lk1*lk2*lk3*lk4 * radial_2e_list_J[tmp][2*Nvec+2] - (2*a1*lk2+2*a2*lk1)*lk3*lk4 * radial_2e_list_J[tmp][0*Nvec+2] + 4*a1*a2*lk3*lk4 * radial_2e_list_J[tmp][1*Nvec+2] - lk1*lk2*(2*a3*lk4+2*a4*lk3) * radial_2e_list_J[tmp][2*Nvec+0] + (2*a1*lk2+2*a2*lk1)*(2*a3*lk4+2*a4*lk3) * radial_2e_list_J[tmp][0*Nvec+0] - 4*a1*a2*(2*a3*lk4+2*a4*lk3) * radial_2e_list_J[tmp][1*Nvec+0] + lk1*lk2*4*a3*a4 * radial_2e_list_J[tmp][2*Nvec+1] - (2*a1*lk2+2*a2*lk1)*4*a3*a4 * radial_2e_list_J[tmp][0*Nvec+1];
                            else
                                array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] += lk1*lk2*4*a3*a4 * radial_2e_list_J[tmp][2*Nvec+1] - (2*a1*lk2+2*a2*lk1)*4*a3*a4 * radial_2e_list_J[tmp][0*Nvec+1];
                        }
                        else
                        {
                            if(l_q != 0)
                                array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] += 4*a1*a2*lk3*lk4 * radial_2e_list_J[tmp][1*Nvec+2] - 4*a1*a2*(2*a3*lk4+2*a4*lk3) * radial_2e_list_J[tmp][1*Nvec+0];
                        }
                        array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] /= norm_J * 16.0 * pow(speedOfLight,4);
                    }
                    else if(intType == "SSLL_SF")
                    {
                        array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] = 4.0*a1*a2 * radial_2e_list_J[tmp][1*Nvec+0];
                        if(l_p != 0)
                            array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] += (l_p*l_p + l_p*(l_p+1)/2 + l_p*(l_p+1)/2 - tmp*(tmp+1)/2) * radial_2e_list_J[tmp][2*Nvec+0] - (2.0*a1*l_p+2.0*a2*l_p) * radial_2e_list_J[tmp][0*Nvec+0];
                        array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] /= norm_J * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "SSSS_SF")
                    {
                        double l12 = l_p*l_p + l_p*(l_p+1)/2 + l_p*(l_p+1)/2 - tmp*(tmp+1)/2, l34 = l_q*l_q + l_q*(l_q+1)/2 + l_q*(l_q+1)/2 - tmp*(tmp+1)/2;
                        array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] = 4*a1*a2*4*a3*a4 * radial_2e_list_J[tmp][1*Nvec+1];
                        if(l_p != 0)
                        {
                            if(l_q != 0)
                                array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] += l12*l34 * radial_2e_list_J[tmp][2*Nvec+2] - (2*a1*l_p+2*a2*l_p)*l34 * radial_2e_list_J[tmp][0*Nvec+2] + 4*a1*a2*l34 * radial_2e_list_J[tmp][1*Nvec+2] - l12*(2*a3*l_q+2*a4*l_q) * radial_2e_list_J[tmp][2*Nvec+0] + (2*a1*l_p+2*a2*l_p)*(2*a3*l_q+2*a4*l_q) * radial_2e_list_J[tmp][0*Nvec+0] - 4*a1*a2*(2*a3*l_q+2*a4*l_q) * radial_2e_list_J[tmp][1*Nvec+0] + l12*4*a3*a4 * radial_2e_list_J[tmp][2*Nvec+1] - (2*a1*l_p+2*a2*l_p)*4*a3*a4 * radial_2e_list_J[tmp][0*Nvec+1];
                            else
                                array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] += l12*4*a3*a4 * radial_2e_list_J[tmp][2*Nvec+1] - (2*a1*l_p+2*a2*l_p)*4*a3*a4 * radial_2e_list_J[tmp][0*Nvec+1];
                        }
                        else
                        {
                            if(l_q != 0)
                                array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] += 4*a1*a2*l34 * radial_2e_list_J[tmp][1*Nvec+2] - 4*a1*a2*(2*a3*l_q+2*a4*l_q) * radial_2e_list_J[tmp][1*Nvec+0];
                        }
                        array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] /= norm_J * 16.0 * pow(speedOfLight,4);
                    }
                    else if(intType == "SSLL_SD")
                    {
                        array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] = 0.0;
                        if(l_p != 0)
                            array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] += (lk1*lk2 - (l_p*l_p + l_p*(l_p+1)/2 + l_p*(l_p+1)/2 - tmp*(tmp+1)/2)) * radial_2e_list_J[tmp][2*Nvec+0] - (2.0*a1*lk2+2.0*a2*lk1 - 2.0*a1*l_p-2.0*a2*l_p) * radial_2e_list_J[tmp][0*Nvec+0];
                        array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] /= norm_J * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "SSSS_SD")
                    {
                        double l12 = l_p*l_p + l_p*(l_p+1)/2 + l_p*(l_p+1)/2 - tmp*(tmp+1)/2, l34 = l_q*l_q + l_q*(l_q+1)/2 + l_q*(l_q+1)/2 - tmp*(tmp+1)/2;
                        array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] = 0.0;
                        if(l_p != 0)
                        {
                            if(l_q != 0)
                                array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] += (lk1*lk2*lk3*lk4 - l12*l34) * radial_2e_list_J[tmp][2*Nvec+2] - ((2*a1*lk2+2*a2*lk1)*lk3*lk4 - (2*a1*l_p+2*a2*l_p)*l34) * radial_2e_list_J[tmp][0*Nvec+2] + (4*a1*a2*lk3*lk4 - 4*a1*a2*l34) * radial_2e_list_J[tmp][1*Nvec+2] - (lk1*lk2*(2*a3*lk4+2*a4*lk3) - l12*(2*a3*l_q+2*a4*l_q)) * radial_2e_list_J[tmp][2*Nvec+0] + ((2*a1*lk2+2*a2*lk1)*(2*a3*lk4+2*a4*lk3) - (2*a1*l_p+2*a2*l_p)*(2*a3*l_q+2*a4*l_q)) * radial_2e_list_J[tmp][0*Nvec+0] - (4*a1*a2*(2*a3*lk4+2*a4*lk3) - 4*a1*a2*(2*a3*l_q+2*a4*l_q)) * radial_2e_list_J[tmp][1*Nvec+0] + (lk1*lk2*4*a3*a4 - l12*4*a3*a4) * radial_2e_list_J[tmp][2*Nvec+1] - ((2*a1*lk2+2*a2*lk1)*4*a3*a4 - (2*a1*l_p+2*a2*l_p)*4*a3*a4) * radial_2e_list_J[tmp][0*Nvec+1];
                            else
                                array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] += (lk1*lk2*4*a3*a4 - l12*4*a3*a4) * radial_2e_list_J[tmp][2*Nvec+1] - ((2*a1*lk2+2*a2*lk1)*4*a3*a4 - (2*a1*l_p+2*a2*l_p)*4*a3*a4) * radial_2e_list_J[tmp][0*Nvec+1];
                        }
                        else
                        {
                            if(l_q != 0)
                                array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] += (4*a1*a2*lk3*lk4 - 4*a1*a2*l34) * radial_2e_list_J[tmp][1*Nvec+2] - (4*a1*a2*(2*a3*lk4+2*a4*lk3) - 4*a1*a2*(2*a3*l_q+2*a4*l_q)) * radial_2e_list_J[tmp][1*Nvec+0];
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
                a2 = shell_list[qshell].exp_a[ll]; a4 = shell_list[pshell].exp_a[jj];
                for(int tmp = LmaxK; tmp >= 0; tmp -= 2)
                {
                    if(intType == "LLLL")
                    {
                        array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] = radial_2e_list_K[tmp][0*Nvec+0] / norm_K;
                    }
                    else if(intType == "SSLL")
                    {
                        array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] = 4.0*a1*a2 * radial_2e_list_K[tmp][1*Nvec+0];
                        if(l_p != 0 && l_q != 0)
                            array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] += lk1*lk2 * radial_2e_list_K[tmp][2*Nvec+0] - (2.0*a1*lk2+2.0*a2*lk1) * radial_2e_list_K[tmp][0*Nvec+0];
                        else if(l_p != 0 || l_q != 0)
                            array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] += - (2.0*a1*lk2+2.0*a2*lk1) * radial_2e_list_K[tmp][0*Nvec+0];
                        array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] /= norm_K * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "SSSS")
                    {
                        array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] = 4*a1*a2*4*a3*a4 * radial_2e_list_K[tmp][1*Nvec+1];
                        if(l_p != 0 && l_q != 0)
                        {
                            array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] += lk1*lk2*lk3*lk4 * radial_2e_list_K[tmp][2*Nvec+2] - (2*a1*lk2+2*a2*lk1)*lk3*lk4 * radial_2e_list_K[tmp][0*Nvec+2] + 4*a1*a2*lk3*lk4 * radial_2e_list_K[tmp][1*Nvec+2] - lk1*lk2*(2*a3*lk4+2*a4*lk3) * radial_2e_list_K[tmp][2*Nvec+0] + (2*a1*lk2+2*a2*lk1)*(2*a3*lk4+2*a4*lk3) * radial_2e_list_K[tmp][0*Nvec+0] - 4*a1*a2*(2*a3*lk4+2*a4*lk3) * radial_2e_list_K[tmp][1*Nvec+0] + lk1*lk2*4*a3*a4 * radial_2e_list_K[tmp][2*Nvec+1] - (2*a1*lk2+2*a2*lk1)*4*a3*a4 * radial_2e_list_K[tmp][0*Nvec+1];
                        }
                        else if(l_p != 0 || l_q != 0)
                        {
                            array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] += (2*a1*lk2+2*a2*lk1)*(2*a3*lk4+2*a4*lk3) * radial_2e_list_K[tmp][0*Nvec+0] - 4*a1*a2*(2*a3*lk4+2*a4*lk3) * radial_2e_list_K[tmp][1*Nvec+0] - (2*a1*lk2+2*a2*lk1)*4*a3*a4 * radial_2e_list_K[tmp][0*Nvec+1];
                        }
                        array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] /= norm_K * 16.0 * pow(speedOfLight,4);
                    }
                    else if(intType == "SSLL_SF")
                    {
                        array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] = 4.0*a1*a2 * radial_2e_list_K[tmp][1*Nvec+0];
                        if(l_p != 0 && l_q != 0)
                            array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] += (l_p*l_q + l_p*(l_p+1)/2 + l_q*(l_q+1)/2 - tmp*(tmp+1)/2) * radial_2e_list_K[tmp][2*Nvec+0] - (2.0*a1*l_q+2.0*a2*l_p)* radial_2e_list_K[tmp][0*Nvec+0];
                        else if(l_p != 0 || l_q != 0)
                            array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] += - (2.0*a1*l_q+2.0*a2*l_p) * radial_2e_list_K[tmp][0*Nvec+0];
                        array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] /= norm_K * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "SSSS_SF")
                    {
                        double l12 = l_p*l_q + l_p*(l_p+1)/2 + l_q*(l_q+1)/2 - tmp*(tmp+1)/2, l34 = l_q*l_p + l_q*(l_q+1)/2 + l_p*(l_p+1)/2 - tmp*(tmp+1)/2;
                        array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] = 4*a1*a2*4*a3*a4 * radial_2e_list_K[tmp][1*Nvec+1];
                        if(l_p != 0 && l_q != 0)
                        {
                            array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] += l12*l34 * radial_2e_list_K[tmp][2*Nvec+2] - (2*a1*l_q+2*a2*l_p)*l34 * radial_2e_list_K[tmp][0*Nvec+2] + 4*a1*a2*l34 * radial_2e_list_K[tmp][1*Nvec+2] - l12*(2*a3*l_p+2*a4*l_q) * radial_2e_list_K[tmp][2*Nvec+0] + (2*a1*l_q+2*a2*l_p)*(2*a3*l_p+2*a4*l_q) * radial_2e_list_K[tmp][0*Nvec+0] - 4*a1*a2*(2*a3*l_p+2*a4*l_q) * radial_2e_list_K[tmp][1*Nvec+0] + l12*4*a3*a4 * radial_2e_list_K[tmp][2*Nvec+1] - (2*a1*l_q+2*a2*l_p)*4*a3*a4 * radial_2e_list_K[tmp][0*Nvec+1];
                        }
                        else if(l_p != 0 || l_q != 0)
                        {
                            array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] += (2*a1*l_q+2*a2*l_p)*(2*a3*l_p+2*a4*l_q) * radial_2e_list_K[tmp][0*Nvec+0] - 4*a1*a2*(2*a3*l_p+2*a4*l_q) * radial_2e_list_K[tmp][1*Nvec+0] - (2*a1*l_q+2*a2*l_p)*4*a3*a4 * radial_2e_list_K[tmp][0*Nvec+1];
                        }
                        array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] /= norm_K * 16.0 * pow(speedOfLight,4);
                    }
                    else if(intType == "SSLL_SD")
                    {
                        array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] = 0.0;
                        if(l_p != 0 && l_q != 0)
                            array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] += (lk1*lk2-(l_p*l_q + l_p*(l_p+1)/2 + l_q*(l_q+1)/2 - tmp*(tmp+1)/2)) * radial_2e_list_K[tmp][2*Nvec+0] - (2.0*a1*lk2+2.0*a2*lk1 - 2.0*a1*l_q - 2.0*a2*l_p)* radial_2e_list_K[tmp][0*Nvec+0];
                        else if(l_p != 0 || l_q != 0)
                            array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] += - (2.0*a1*lk2+2.0*a2*lk1 - 2.0*a1*l_q-2.0*a2*l_p) * radial_2e_list_K[tmp][0*Nvec+0];
                        array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] /= norm_K * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "SSSS_SD")
                    {
                        double l12 = l_p*l_q + l_p*(l_p+1)/2 + l_q*(l_q+1)/2 - tmp*(tmp+1)/2, l34 = l_q*l_p + l_q*(l_q+1)/2 + l_p*(l_p+1)/2 - tmp*(tmp+1)/2;
                        array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] = 0.0;
                        if(l_p != 0 && l_q != 0)
                        {
                            array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] += (lk1*lk2*lk3*lk4 - l12*l34) * radial_2e_list_K[tmp][2*Nvec+2] - ((2*a1*lk2+2*a2*lk1)*lk3*lk4 - (2*a1*l_q+2*a2*l_p)*l34) * radial_2e_list_K[tmp][0*Nvec+2] + (4*a1*a2*lk3*lk4 - 4*a1*a2*l34) * radial_2e_list_K[tmp][1*Nvec+2] - (lk1*lk2*(2*a3*lk4+2*a4*lk3) - l12*(2*a3*l_p+2*a4*l_q)) * radial_2e_list_K[tmp][2*Nvec+0] + ((2*a1*lk2+2*a2*lk1)*(2*a3*lk4+2*a4*lk3) - (2*a1*l_q+2*a2*l_p)*(2*a3*l_p+2*a4*l_q)) * radial_2e_list_K[tmp][0*Nvec+0] - (4*a1*a2*(2*a3*lk4+2*a4*lk3) -  4*a1*a2*(2*a3*l_p+2*a4*l_q)) * radial_2e_list_K[tmp][1*Nvec+0] + (lk1*lk2*4*a3*a4 - l12*4*a3*a4) * radial_2e_list_K[tmp][2*Nvec+1] - ((2*a1*lk2+2*a2*lk1)*4*a3*a4 - (2*a1*l_q+2*a2*l_p)*4*a3*a4) * radial_2e_list_K[tmp][0*Nvec+1];
                        }
                        else if(l_p != 0 || l_q != 0)
                        {
                            array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] += ((2*a1*lk2+2*a2*lk1)*(2*a3*lk4+2*a4*lk3) - (2*a1*l_q+2*a2*l_p)*(2*a3*l_p+2*a4*l_q)) * radial_2e_list_K[tmp][0*Nvec+0] - (4*a1*a2*(2*a3*lk4+2*a4*lk3) - 4*a1*a2*(2*a3*l_p+2*a4*l_q)) * radial_2e_list_K[tmp][1*Nvec+0] - ((2*a1*lk2+2*a2*lk1)*4*a3*a4 - (2*a1*l_q+2*a2*l_p)*4*a3*a4) * radial_2e_list_K[tmp][0*Nvec+1];
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
                    for(int tmp = LmaxJ; tmp >= 0; tmp = tmp - 2)
                        int_2e_JK.J[int_tmp1_p+add_p + mp][int_tmp1_q+add_q + mq][e1J][e2J] += array_radial_J[tmp][e1J][e2J][int_tmp2_p][int_tmp2_q] * array_angular_J[tmp][int_tmp2_p][int_tmp2_q][mp*(irrep_list[int_tmp1_q+add_q].two_j + 1)+mq];
                    for(int tmp = LmaxK; tmp >= 0; tmp = tmp - 2)
                        int_2e_JK.K[int_tmp1_p+add_p + mp][int_tmp1_q+add_q + mq][e1K][e2K] += array_radial_K[tmp][e1K][e2K][int_tmp2_p][int_tmp2_q] * array_angular_K[tmp][int_tmp2_p][int_tmp2_q][mp*(irrep_list[int_tmp1_q+add_q].two_j + 1)+mq];
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
int2eJK INT_SPH::get_h2e_JK_compact(const string& intType, const int& occMaxL) const
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
        int l_q = shell_list[qshell].l, l_max = max(l_p,l_q), LmaxJ = 0, LmaxK = l_p+l_q;
        int size_gtos_p = shell_list[pshell].nunc, size_gtos_q = shell_list[qshell].nunc;
        int size_tmp_p = (l_p == 0) ? 1 : 2, size_tmp_q = (l_q == 0) ? 1 : 2;
        double array_angular_J[LmaxJ+1][size_tmp_p][size_tmp_q], array_angular_K[LmaxK+1][size_tmp_p][size_tmp_q];
        double radial_2e_list_J[size_gtos_p*size_gtos_p*size_gtos_q*size_gtos_q][LmaxJ+1][3][3];
        double radial_2e_list_K[size_gtos_p*size_gtos_p*size_gtos_q*size_gtos_q][LmaxK+1][3][3];

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
        
            for(int LL = LmaxJ; LL >= 0; LL -= 2)
            {
                radial_2e_list_J[tt][LL][0][0] = int2e_get_radial(l_p,a_i_J,l_p,a_j_J,l_q,a_k_J,l_q,a_l_J,LL);
            }
            for(int LL = LmaxK; LL >= 0; LL -= 2)
            {
                radial_2e_list_K[tt][LL][0][0] = int2e_get_radial(l_p,a_i_K,l_q,a_j_K,l_q,a_k_K,l_p,a_l_K,LL);
            }
            if(intType.substr(0,2) == "SS")
            {
                for(int LL = LmaxJ; LL >= 0; LL -= 2)
                {
                    radial_2e_list_J[tt][LL][1][0] = int2e_get_radial(l_p+1,a_i_J,l_p+1,a_j_J,l_q,a_k_J,l_q,a_l_J,LL);
                    if(l_p != 0)
                        radial_2e_list_J[tt][LL][2][0] = int2e_get_radial(l_p-1,a_i_J,l_p-1,a_j_J,l_q,a_k_J,l_q,a_l_J,LL);
                }
                for(int LL = LmaxK; LL >= 0; LL -= 2)
                {
                    radial_2e_list_K[tt][LL][1][0] = int2e_get_radial(l_p+1,a_i_K,l_q+1,a_j_K,l_q,a_k_K,l_p,a_l_K,LL);
                    if(l_p != 0 && l_q != 0)
                        radial_2e_list_K[tt][LL][2][0] = int2e_get_radial(l_p-1,a_i_K,l_q-1,a_j_K,l_q,a_k_K,l_p,a_l_K,LL);
                }
            }
            if(intType.substr(0,4) == "SSSS")
            {
                for(int LL = LmaxJ; LL >= 0; LL -= 2)
                {
                    radial_2e_list_J[tt][LL][0][1] = int2e_get_radial(l_p,a_i_J,l_p,a_j_J,l_q+1,a_k_J,l_q+1,a_l_J,LL);
                    radial_2e_list_J[tt][LL][1][1] = int2e_get_radial(l_p+1,a_i_J,l_p+1,a_j_J,l_q+1,a_k_J,l_q+1,a_l_J,LL);
                    if(l_p != 0)
                    {
                        radial_2e_list_J[tt][LL][2][1] = int2e_get_radial(l_p-1,a_i_J,l_p-1,a_j_J,l_q+1,a_k_J,l_q+1,a_l_J,LL);
                    }
                    if(l_q != 0)
                    {
                        radial_2e_list_J[tt][LL][0][2] = int2e_get_radial(l_p,a_i_J,l_p,a_j_J,l_q-1,a_k_J,l_q-1,a_l_J,LL);
                        radial_2e_list_J[tt][LL][1][2] = int2e_get_radial(l_p+1,a_i_J,l_p+1,a_j_J,l_q-1,a_k_J,l_q-1,a_l_J,LL);
                    }
                    if(l_p!=0 && l_q!=0)
                        radial_2e_list_J[tt][LL][2][2] = int2e_get_radial(l_p-1,a_i_J,l_p-1,a_j_J,l_q-1,a_k_J,l_q-1,a_l_J,LL);
                }
                for(int LL = LmaxK; LL >= 0; LL -= 2)
                {
                    radial_2e_list_K[tt][LL][0][1] = int2e_get_radial(l_p,a_i_K,l_q,a_j_K,l_q+1,a_k_K,l_p+1,a_l_K,LL);
                    radial_2e_list_K[tt][LL][1][1] = int2e_get_radial(l_p+1,a_i_K,l_q+1,a_j_K,l_q+1,a_k_K,l_p+1,a_l_K,LL);
                    if(l_p != 0 && l_q != 0)
                    {
                        radial_2e_list_K[tt][LL][2][1] = int2e_get_radial(l_p-1,a_i_K,l_q-1,a_j_K,l_q+1,a_k_K,l_p+1,a_l_K,LL);
                        radial_2e_list_K[tt][LL][0][2] = int2e_get_radial(l_p,a_i_K,l_q,a_j_K,l_q-1,a_k_K,l_p-1,a_l_K,LL);
                        radial_2e_list_K[tt][LL][1][2] = int2e_get_radial(l_p+1,a_i_K,l_q+1,a_j_K,l_q-1,a_k_K,l_p-1,a_l_K,LL);
                        radial_2e_list_K[tt][LL][2][2] = int2e_get_radial(l_p-1,a_i_K,l_q-1,a_j_K,l_q-1,a_k_K,l_p-1,a_l_K,LL);
                    }
                }
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
            for(int tmp = LmaxJ; tmp >= 0; tmp -= 2)
            {
                double tmp_d = 0.0;
                for(int mq = 0; mq < twojj_q + 1; mq++)
                {
                    tmp_d += int2e_get_angular_J(l_p, twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, tmp);
                }
                tmp_d /= (twojj_q + 1);
                array_angular_J[tmp][int_tmp2_p][int_tmp2_q] = tmp_d;
            }
            for(int tmp = LmaxK; tmp >= 0; tmp -= 2)
            {
                double tmp_d = 0.0;
                for(int mq = 0; mq < twojj_q + 1; mq++)
                {
                    tmp_d += int2e_get_angular_K(l_p, twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, tmp);
                }
                tmp_d /= (twojj_q + 1);
                array_angular_K[tmp][int_tmp2_p][int_tmp2_q] = tmp_d;
            }
            // Radial
            double k_p = -(twojj_p+1.0)*sym_ap/2.0, k_q = -(twojj_q+1.0)*sym_aq/2.0;
            #pragma omp parallel  for
            for(int tt = 0; tt < size_gtos_p*size_gtos_p*size_gtos_q*size_gtos_q; tt++)
            {
                double radial_J, radial_K;
                int e1J = tt/(size_gtos_q*size_gtos_q);
                int e2J = tt - e1J*(size_gtos_q*size_gtos_q);
                int ii = e1J/size_gtos_p, jj = e1J - ii*size_gtos_p;
                int kk = e2J/size_gtos_q, ll = e2J - kk*size_gtos_q;
                int e1K = ii*size_gtos_q+ll, e2K = kk*size_gtos_p+jj;
                double norm_J = shell_list[pshell].norm[ii] * shell_list[pshell].norm[jj] * shell_list[qshell].norm[kk] * shell_list[qshell].norm[ll], norm_K = shell_list[pshell].norm[ii] * shell_list[qshell].norm[ll] * shell_list[qshell].norm[kk] * shell_list[pshell].norm[jj];
                double lk1 = 1+l_p+k_p, lk2 = 1+l_p+k_p, lk3 = 1+l_q+k_q, lk4 = 1+l_q+k_q, a1 = shell_list[pshell].exp_a[ii], a2 = shell_list[pshell].exp_a[jj], a3 = shell_list[qshell].exp_a[kk], a4 = shell_list[qshell].exp_a[ll];
                int_2e_JK.J[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1J][e2J] = 0.0;
                int_2e_JK.K[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1K][e2K] = 0.0;

                for(int tmp = LmaxJ; tmp >= 0; tmp -= 2)
                {
                    if(intType == "LLLL")
                    {
                        radial_J = get_radial_LLLL_J(l_p,l_q,tmp,a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_J[tt][tmp],false) / norm_J;
                    }
                    else if(intType == "SSLL")
                    {
                        radial_J = get_radial_SSLL_J(l_p,l_q,tmp,a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_J[tt][tmp],false) / norm_J / 4.0 / pow(speedOfLight,2);
                    }
                    else if(intType == "SSSS")
                    {
                        radial_J = get_radial_SSSS_J(l_p,l_q,tmp,a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_J[tt][tmp],false) / norm_J / 16.0 / pow(speedOfLight,4);
                    }
                    else if(intType == "SSLL_SF")
                    {
                        radial_J = get_radial_SSLL_J(l_p,l_q,tmp,a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_J[tt][tmp],true) / norm_J / 4.0 / pow(speedOfLight,2);
                    }
                    else if(intType == "SSSS_SF")
                    {
                        radial_J = get_radial_SSSS_J(l_p,l_q,tmp,a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_J[tt][tmp],true) / norm_J / 16.0 / pow(speedOfLight,4);
                    }
                    else if(intType == "SSLL_SD")
                    {
                        radial_J = (get_radial_SSLL_J(l_p,l_q,tmp,a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_J[tt][tmp],false)-get_radial_SSLL_J(l_p,l_q,tmp,a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_J[tt][tmp],true)) / norm_J / 4.0 / pow(speedOfLight,2);
                    }
                    else if(intType == "SSSS_SD")
                    {
                        radial_J = (get_radial_SSSS_J(l_p,l_q,tmp,a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_J[tt][tmp],false)-get_radial_SSSS_J(l_p,l_q,tmp,a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_J[tt][tmp],true))/ norm_J / 16.0 / pow(speedOfLight,4);
                    }
                    else
                    {
                        cout << "ERROR: Unknown integralTYPE in get_h2e:\n";
                        exit(99);
                    }
                    int_2e_JK.J[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1J][e2J] += radial_J * array_angular_J[tmp][int_tmp2_p][int_tmp2_q];
                }
                lk2 = 1+l_q+k_q; lk4 = 1+l_p+k_p; 
                a2 = shell_list[qshell].exp_a[ll]; a4 = shell_list[pshell].exp_a[jj];
                for(int tmp = LmaxK; tmp >= 0; tmp -= 2)
                {
                    if(intType == "LLLL")
                    {
                        radial_K = get_radial_LLLL_K(l_p,l_q,tmp,a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_K[tt][tmp],false) / norm_K;
                    }
                    else if(intType == "SSLL")
                    {
                        radial_K = get_radial_SSLL_K(l_p,l_q,tmp,a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_K[tt][tmp],false) / norm_K / 4.0 / pow(speedOfLight,2);
                    }
                    else if(intType == "SSSS")
                    {
                        radial_K = get_radial_SSSS_K(l_p,l_q,tmp,a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_K[tt][tmp],false) / norm_K / 16.0 / pow(speedOfLight,4);
                    }
                    else if(intType == "SSLL_SF")
                    {
                        radial_K = get_radial_SSLL_K(l_p,l_q,tmp,a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_K[tt][tmp],true) / norm_K / 4.0 / pow(speedOfLight,2);
                    }
                    else if(intType == "SSSS_SF")
                    {
                        radial_K = get_radial_SSSS_K(l_p,l_q,tmp,a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_K[tt][tmp],true) / norm_K / 16.0 / pow(speedOfLight,4);
                    }
                    else if(intType == "SSLL_SD")
                    {
                        radial_K = (get_radial_SSLL_K(l_p,l_q,tmp,a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_K[tt][tmp],false)-get_radial_SSLL_K(l_p,l_q,tmp,a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_K[tt][tmp],true)) / norm_K / 4.0 / pow(speedOfLight,2);
                    }
                    else if(intType == "SSSS_SD")
                    {
                        radial_K = (get_radial_SSSS_K(l_p,l_q,tmp,a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_K[tt][tmp],false)-get_radial_SSSS_K(l_p,l_q,tmp,a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_K[tt][tmp],true))/ norm_K / 16.0 / pow(speedOfLight,4);
                    }
                    else
                    {
                        cout << "ERROR: Unknown integralTYPE in get_h2e:\n";
                        exit(99);
                    }
                    int_2e_JK.K[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1K][e2K] += radial_K * array_angular_K[tmp][int_tmp2_p][int_tmp2_q];
                }
            }
        }
        int_tmp1_q += (l_q == 0) ? 1 : 2;
    }
    int_tmp1_p += (l_p == 0) ? 1 : 2;
    }

    return int_2e_JK;
}

/*
    Evaluate all compact 2e integral together for DHF calculations
*/
void INT_SPH::get_h2e_JK_direct(int2eJK& LLLL, int2eJK& SSLL, int2eJK& SSSS, const int& occMaxL, const bool& spinFree)
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

    LLLL.J = new double***[Nirrep_compact];
    LLLL.K = new double***[Nirrep_compact];
    SSLL.J = new double***[Nirrep_compact];
    SSLL.K = new double***[Nirrep_compact];
    SSSS.J = new double***[Nirrep_compact];
    SSSS.K = new double***[Nirrep_compact];
    for(int ii = 0; ii < Nirrep_compact; ii++)
    {
        LLLL.J[ii] = new double**[Nirrep_compact];
        LLLL.K[ii] = new double**[Nirrep_compact];
        SSLL.J[ii] = new double**[Nirrep_compact];
        SSLL.K[ii] = new double**[Nirrep_compact];
        SSSS.J[ii] = new double**[Nirrep_compact];
        SSSS.K[ii] = new double**[Nirrep_compact];
    }

    int int_tmp1_p = 0;
    for(int pshell = 0; pshell < occMaxShell; pshell++)
    {
    int l_p = shell_list[pshell].l, int_tmp1_q = 0;
    for(int qshell = 0; qshell < occMaxShell; qshell++)
    {
        // LmaxJ = 0 is correct for J (and not K) due to symmetry.
        // Same for Gaunt and gauge term. A very limited acceleration.
        int l_q = shell_list[qshell].l, l_max = max(l_p,l_q), LmaxJ = 0, LmaxK = l_p+l_q;
        int size_gtos_p = shell_list[pshell].nunc, size_gtos_q = shell_list[qshell].nunc;
        int size_tmp_p = (l_p == 0) ? 1 : 2, size_tmp_q = (l_q == 0) ? 1 : 2; 
        double array_angular_J[LmaxJ+1][size_tmp_p][size_tmp_q], array_angular_K[LmaxK+1][size_tmp_p][size_tmp_q];
        double radial_2e_list_J[size_gtos_p*size_gtos_p*size_gtos_q*size_gtos_q][LmaxJ+1][3][3];
        double radial_2e_list_K[size_gtos_p*size_gtos_p*size_gtos_q*size_gtos_q][LmaxK+1][3][3];

        #pragma omp parallel  for
        for(int tt = 0; tt < size_gtos_p*size_gtos_p*size_gtos_q*size_gtos_q; tt++)
        {
            int e1J = tt/(size_gtos_q*size_gtos_q);
            int e2J = tt - e1J*(size_gtos_q*size_gtos_q);
            int ii = e1J/size_gtos_p, jj = e1J - ii*size_gtos_p;
            int kk = e2J/size_gtos_q, ll = e2J - kk*size_gtos_q;
            double a_i_J = shell_list[pshell].exp_a[ii], a_j_J = shell_list[pshell].exp_a[jj], a_k_J = shell_list[qshell].exp_a[kk], a_l_J = shell_list[qshell].exp_a[ll];
            double a_i_K = shell_list[pshell].exp_a[ii], a_j_K = shell_list[qshell].exp_a[ll], a_k_K = shell_list[qshell].exp_a[kk], a_l_K = shell_list[pshell].exp_a[jj];

            for(int LL = LmaxJ; LL >= 0; LL -= 2)
            {
                radial_2e_list_J[tt][LL][0][0] = int2e_get_radial(l_p,a_i_J,l_p,a_j_J,l_q,a_k_J,l_q,a_l_J,LL);
                radial_2e_list_J[tt][LL][1][0] = int2e_get_radial(l_p+1,a_i_J,l_p+1,a_j_J,l_q,a_k_J,l_q,a_l_J,LL);
                radial_2e_list_J[tt][LL][0][1] = int2e_get_radial(l_p,a_i_J,l_p,a_j_J,l_q+1,a_k_J,l_q+1,a_l_J,LL);
                radial_2e_list_J[tt][LL][1][1] = int2e_get_radial(l_p+1,a_i_J,l_p+1,a_j_J,l_q+1,a_k_J,l_q+1,a_l_J,LL);
                if(l_p != 0)
                {
                    radial_2e_list_J[tt][LL][2][0] = int2e_get_radial(l_p-1,a_i_J,l_p-1,a_j_J,l_q,a_k_J,l_q,a_l_J,LL);
                    radial_2e_list_J[tt][LL][2][1] = int2e_get_radial(l_p-1,a_i_J,l_p-1,a_j_J,l_q+1,a_k_J,l_q+1,a_l_J,LL);
                }
                if(l_q != 0)
                {
                    radial_2e_list_J[tt][LL][0][2] = int2e_get_radial(l_p,a_i_J,l_p,a_j_J,l_q-1,a_k_J,l_q-1,a_l_J,LL);
                    radial_2e_list_J[tt][LL][1][2] = int2e_get_radial(l_p+1,a_i_J,l_p+1,a_j_J,l_q-1,a_k_J,l_q-1,a_l_J,LL);
                }
                if(l_p!=0 && l_q!=0)
                    radial_2e_list_J[tt][LL][2][2] = int2e_get_radial(l_p-1,a_i_J,l_p-1,a_j_J,l_q-1,a_k_J,l_q-1,a_l_J,LL);
            }
            for(int LL = LmaxK; LL >= 0; LL -= 2)
            {
                radial_2e_list_K[tt][LL][0][0] = int2e_get_radial(l_p,a_i_K,l_q,a_j_K,l_q,a_k_K,l_p,a_l_K,LL);
                radial_2e_list_K[tt][LL][1][0] = int2e_get_radial(l_p+1,a_i_K,l_q+1,a_j_K,l_q,a_k_K,l_p,a_l_K,LL);
                radial_2e_list_K[tt][LL][0][1] = int2e_get_radial(l_p,a_i_K,l_q,a_j_K,l_q+1,a_k_K,l_p+1,a_l_K,LL);
                radial_2e_list_K[tt][LL][1][1] = int2e_get_radial(l_p+1,a_i_K,l_q+1,a_j_K,l_q+1,a_k_K,l_p+1,a_l_K,LL);
                if(l_p != 0 && l_q != 0)
                {
                    radial_2e_list_K[tt][LL][2][0] = int2e_get_radial(l_p-1,a_i_K,l_q-1,a_j_K,l_q,a_k_K,l_p,a_l_K,LL);
                    radial_2e_list_K[tt][LL][2][1] = int2e_get_radial(l_p-1,a_i_K,l_q-1,a_j_K,l_q+1,a_k_K,l_p+1,a_l_K,LL);
                    radial_2e_list_K[tt][LL][0][2] = int2e_get_radial(l_p,a_i_K,l_q,a_j_K,l_q-1,a_k_K,l_p-1,a_l_K,LL);
                    radial_2e_list_K[tt][LL][1][2] = int2e_get_radial(l_p+1,a_i_K,l_q+1,a_j_K,l_q-1,a_k_K,l_p-1,a_l_K,LL);
                    radial_2e_list_K[tt][LL][2][2] = int2e_get_radial(l_p-1,a_i_K,l_q-1,a_j_K,l_q-1,a_k_K,l_p-1,a_l_K,LL);
                }
            }
        }

        for(int twojj_p = abs(2*l_p-1); twojj_p <= 2*l_p+1; twojj_p = twojj_p + 2)
        for(int twojj_q = abs(2*l_q-1); twojj_q <= 2*l_q+1; twojj_q = twojj_q + 2)
        {
            int sym_ap = twojj_p - 2*l_p, sym_aq = twojj_q - 2*l_q;
            int int_tmp2_p = (twojj_p - abs(2*l_p-1)) / 2, int_tmp2_q = (twojj_q - abs(2*l_q-1))/2;

            LLLL.J[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q] = new double*[size_gtos_p*size_gtos_p];
            LLLL.K[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q] = new double*[size_gtos_p*size_gtos_q];
            SSLL.J[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q] = new double*[size_gtos_p*size_gtos_p];
            SSLL.K[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q] = new double*[size_gtos_p*size_gtos_q];
            SSSS.J[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q] = new double*[size_gtos_p*size_gtos_p];
            SSSS.K[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q] = new double*[size_gtos_p*size_gtos_q];
            for(int iii = 0; iii < size_gtos_p*size_gtos_p; iii++)
            {
                LLLL.J[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][iii] = new double[size_gtos_q*size_gtos_q];
                SSLL.J[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][iii] = new double[size_gtos_q*size_gtos_q];
                SSSS.J[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][iii] = new double[size_gtos_q*size_gtos_q];
            }
            for(int iii = 0; iii < size_gtos_p*size_gtos_q; iii++)
            {
                LLLL.K[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][iii] = new double[size_gtos_q*size_gtos_p];
                SSLL.K[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][iii] = new double[size_gtos_q*size_gtos_p];
                SSSS.K[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][iii] = new double[size_gtos_q*size_gtos_p];
            }

            // Angular
            for(int tmp = LmaxJ; tmp >= 0; tmp -= 2)
            {
                double tmp_d = 0.0;
                for(int mq = 0; mq < twojj_q + 1; mq++)
                {
                    tmp_d += int2e_get_angular_J(l_p, twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, tmp);
                }
                tmp_d /= (twojj_q + 1);
                array_angular_J[tmp][int_tmp2_p][int_tmp2_q] = tmp_d;
            }    
            for(int tmp = LmaxK; tmp >= 0; tmp -= 2)
            {
                double tmp_d = 0.0;
                for(int mq = 0; mq < twojj_q + 1; mq++)
                {
                    tmp_d += int2e_get_angular_K(l_p, twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, tmp);
                }
                tmp_d /= (twojj_q + 1);
                array_angular_K[tmp][int_tmp2_p][int_tmp2_q] = tmp_d;
            }

            // Radial
            double k_p = -(twojj_p+1.0)*sym_ap/2.0, k_q = -(twojj_q+1.0)*sym_aq/2.0;
            #pragma omp parallel  for
            for(int tt = 0; tt < size_gtos_p*size_gtos_p*size_gtos_q*size_gtos_q; tt++)
            {
                double radial_J_LLLL, radial_K_LLLL, radial_J_SSLL, radial_K_SSLL, radial_J_SSSS, radial_K_SSSS;
                int e1J = tt/(size_gtos_q*size_gtos_q);
                int e2J = tt - e1J*(size_gtos_q*size_gtos_q);
                int ii = e1J/size_gtos_p, jj = e1J - ii*size_gtos_p;
                int kk = e2J/size_gtos_q, ll = e2J - kk*size_gtos_q;
                int e1K = ii*size_gtos_q+ll, e2K = kk*size_gtos_p+jj;
                double norm_J = shell_list[pshell].norm[ii] * shell_list[pshell].norm[jj] * shell_list[qshell].norm[kk] * shell_list[qshell].norm[ll], norm_K = shell_list[pshell].norm[ii] * shell_list[qshell].norm[ll] * shell_list[qshell].norm[kk] * shell_list[pshell].norm[jj];
                double lk1 = 1+l_p+k_p, lk2 = 1+l_p+k_p, lk3 = 1+l_q+k_q, lk4 = 1+l_q+k_q, a1 = shell_list[pshell].exp_a[ii], a2 = shell_list[pshell].exp_a[jj], a3 = shell_list[qshell].exp_a[kk], a4 = shell_list[qshell].exp_a[ll];
                LLLL.J[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1J][e2J] = 0.0;
                LLLL.K[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1K][e2K] = 0.0;
                SSLL.J[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1J][e2J] = 0.0;
                SSLL.K[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1K][e2K] = 0.0;
                SSSS.J[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1J][e2J] = 0.0;
                SSSS.K[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1K][e2K] = 0.0;

                for(int tmp = LmaxJ; tmp >= 0; tmp -= 2)
                {
                    radial_J_LLLL = get_radial_LLLL_J(l_p,l_q,tmp,a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_J[tt][tmp],spinFree) / norm_J;
                    radial_J_SSLL = get_radial_SSLL_J(l_p,l_q,tmp,a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_J[tt][tmp],spinFree)/norm_J/4.0/pow(speedOfLight,2);
                    radial_J_SSSS = get_radial_SSSS_J(l_p,l_q,tmp,a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_J[tt][tmp],spinFree)/norm_J/16.0/pow(speedOfLight,4);
                    LLLL.J[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1J][e2J] += radial_J_LLLL * array_angular_J[tmp][int_tmp2_p][int_tmp2_q];
                    SSLL.J[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1J][e2J] += radial_J_SSLL * array_angular_J[tmp][int_tmp2_p][int_tmp2_q];
                    SSSS.J[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1J][e2J] += radial_J_SSSS * array_angular_J[tmp][int_tmp2_p][int_tmp2_q];
                }
                lk2 = 1+l_q+k_q; lk4 = 1+l_p+k_p; 
                a2 = shell_list[qshell].exp_a[ll]; a4 = shell_list[pshell].exp_a[jj];
                for(int tmp = LmaxK; tmp >= 0; tmp -= 2)
                {
                    radial_K_LLLL = get_radial_LLLL_K(l_p,l_q,tmp,a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_K[tt][tmp],spinFree) / norm_K;
                    radial_K_SSLL = get_radial_SSLL_K(l_p,l_q,tmp,a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_K[tt][tmp],spinFree)/norm_K/4.0/pow(speedOfLight,2);
                    radial_K_SSSS = get_radial_SSSS_K(l_p,l_q,tmp,a1,a2,a3,a4,lk1,lk2,lk3,lk4,radial_2e_list_K[tt][tmp],spinFree)/norm_K/16.0/pow(speedOfLight,4);
                    LLLL.K[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1K][e2K] += radial_K_LLLL * array_angular_K[tmp][int_tmp2_p][int_tmp2_q];
                    SSLL.K[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1K][e2K] += radial_K_SSLL * array_angular_K[tmp][int_tmp2_p][int_tmp2_q];
                    SSSS.K[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1K][e2K] += radial_K_SSSS * array_angular_K[tmp][int_tmp2_p][int_tmp2_q];
                }
            }
        }
        int_tmp1_q += (l_q == 0) ? 1 : 2;
    }
    int_tmp1_p += (l_p == 0) ? 1 : 2;
    }

    return ;
}

void INT_SPH::get_h2eSD_JK_direct(int2eJK& SSLL, int2eJK& SSSS, const int& occMaxL)
{
    SSLL = get_h2e_JK_compact("SSLL_SD", occMaxL);
    SSSS = get_h2e_JK_compact("SSSS_SD", occMaxL);
}
