#include"general.h"
#include"dhf_sph.h"
#include"dhf_sph_ca.h"
#include<iomanip>
#include<iostream>
using namespace std;
using namespace Eigen;

vMatrixXd DHF_SPH::x2c2ePCC(vMatrixXd* coeff2c)
{
    cout << "Running DHF_SPH::x2c2ePCC" << endl;
    if(!converged)
    {
        cout << "SCF did not converge. x2c2ePCC cannot be used!" << endl;
        exit(99);
    }
    vMatrixXd fock_pcc(occMax_irrep), fock_4c_2e(occMax_irrep), fock_x2c2e(occMax_irrep), fock_x2c2e_2e(occMax_irrep), JK_x2c2c(occMax_irrep), coeff_2c(occMax_irrep), density_2c(occMax_irrep), density_pcc(occMax_irrep), h1e_x2c2e(occMax_irrep), h1e_x2c1e(occMax_irrep), densityCore_4c(occMax_irrep), densityCore_2c(occMax_irrep), fock_pcc_mo(occMax_irrep);
    vMatrixXd XXX(occMax_irrep), RRR(occMax_irrep), XXX_1e(occMax_irrep), RRR_1e(occMax_irrep);
    vMatrixXd overlap_2c(occMax_irrep), overlap_h_i_2c(occMax_irrep);


    for(int ir = 0; ir < occMax_irrep; ir++)
    {
        overlap_2c(ir) = overlap_4c(ir).block(0,0,overlap_4c(ir).rows()/2,overlap_4c(ir).cols()/2);
        overlap_h_i_2c(ir) = matrix_half_inverse(overlap_2c(ir));
        VectorXd ene_mo_tmp;
        XXX(ir) = X2C::get_X(coeff(ir));
        RRR(ir) = X2C::get_R(overlap_4c(ir),XXX(ir));
        
        h1e_x2c2e(ir) = X2C::transform_4c_2c(h1e_4c(ir), XXX(ir), RRR(ir));
        fock_x2c2e(ir) = X2C::transform_4c_2c(fock_4c(ir), XXX(ir), RRR(ir));
        if(coeff2c == NULL)
            eigensolverG(fock_x2c2e(ir),overlap_h_i_2c(ir),ene_mo_tmp,coeff_2c(ir));
        else
            coeff_2c(ir) = (*coeff2c)(ir);

        density_2c(ir) = evaluateDensity_spinor(coeff_2c(ir),occNumber(ir),true);

        // X2C1E
        XXX_1e(ir) = X2C::get_X(overlap(ir),kinetic(ir),WWW(ir),Vnuc(ir));
        RRR_1e(ir) = X2C::get_R(overlap(ir),kinetic(ir),XXX_1e(ir));
        h1e_x2c1e(ir) = X2C::evaluate_h1e_x2c(overlap(ir),kinetic(ir),WWW(ir),Vnuc(ir),XXX_1e(ir),RRR_1e(ir));
    }    

    for(int ir = 0; ir < occMax_irrep; ir++)
    {
        int size_nr = density_2c(ir).rows();
        evaluateFock_2e(fock_4c_2e(ir),false,density,irrep_list(ir).size,ir);
        evaluateFock_2e(JK_x2c2c(ir),true,density_2c,irrep_list(ir).size,ir);
        fock_x2c2e_2e(ir) = X2C::transform_4c_2c(fock_4c_2e(ir), XXX(ir), RRR(ir));
    }
    for(int ir = 0; ir < occMax_irrep; ir++)
    {  
        fock_pcc(ir) = fock_x2c2e_2e(ir) - JK_x2c2c(ir) + h1e_x2c2e(ir) - h1e_x2c1e(ir);
    }

    x2cXXX = XXX;
    x2cRRR = RRR;
    X_calculated = true;

    return fock_pcc;
}

vMatrixXd DHF_SPH::h_x2c2e(vMatrixXd* coeff2c)
{
    if(!converged)
    {
        cout << "SCF did not converge. x2c2ePCC cannot be used!" << endl;
        exit(99);
    }
    vMatrixXd fock_pcc(occMax_irrep), fock_4c_2e(occMax_irrep), fock_x2c2e(occMax_irrep), fock_x2c2e_2e(occMax_irrep), JK_x2c2c(occMax_irrep), coeff_2c(occMax_irrep), density_2c(occMax_irrep), density_pcc(occMax_irrep), h1e_x2c2e(occMax_irrep), h1e_x2c1e(occMax_irrep), densityCore_4c(occMax_irrep), densityCore_2c(occMax_irrep), fock_pcc_mo(occMax_irrep);
    vMatrixXd XXX(occMax_irrep), RRR(occMax_irrep), XXX_1e(occMax_irrep), RRR_1e(occMax_irrep);
    vMatrixXd overlap_2c(occMax_irrep), overlap_h_i_2c(occMax_irrep);

    for(int ir = 0; ir < occMax_irrep; ir++)
    {
        overlap_2c(ir) = overlap_4c(ir).block(0,0,overlap_4c(ir).rows()/2,overlap_4c(ir).cols()/2);
        overlap_h_i_2c(ir) = matrix_half_inverse(overlap_2c(ir));
        VectorXd ene_mo_tmp;
        XXX(ir) = X2C::get_X(coeff(ir));
        RRR(ir) = X2C::get_R(overlap_4c(ir),XXX(ir));
        
        h1e_x2c2e(ir) = X2C::transform_4c_2c(h1e_4c(ir), XXX(ir), RRR(ir));
        fock_x2c2e(ir) = X2C::transform_4c_2c(fock_4c(ir), XXX(ir), RRR(ir));
        if(coeff2c == NULL)
            eigensolverG(fock_x2c2e(ir),overlap_h_i_2c(ir),ene_mo_tmp,coeff_2c(ir));
        else
            coeff_2c(ir) = (*coeff2c)(ir);

        density_2c(ir) = evaluateDensity_spinor(coeff_2c(ir),occNumber(ir),true);

        // X2C1E
        XXX_1e(ir) = X2C::get_X(overlap(ir),kinetic(ir),WWW(ir),Vnuc(ir));
        RRR_1e(ir) = X2C::get_R(overlap(ir),kinetic(ir),XXX_1e(ir));
        h1e_x2c1e(ir) = X2C::evaluate_h1e_x2c(overlap(ir),kinetic(ir),WWW(ir),Vnuc(ir),XXX_1e(ir),RRR_1e(ir));
    }    

    for(int ir = 0; ir < occMax_irrep; ir++)
    {
        int size_nr = density_2c(ir).rows();
        evaluateFock_2e(fock_4c_2e(ir),false,density,irrep_list(ir).size,ir);
        evaluateFock_2e(JK_x2c2c(ir),true,density_2c,irrep_list(ir).size,ir);
        fock_x2c2e_2e(ir) = X2C::transform_4c_2c(fock_4c_2e(ir), XXX(ir), RRR(ir));
    }
    for(int ir = 0; ir < occMax_irrep; ir++)
    {  
        fock_pcc(ir) = h1e_x2c2e(ir);
    }

    x2cXXX = XXX;
    x2cRRR = RRR;
    X_calculated = true;

    return fock_pcc;
}

/* 
    evaluate Fock matrix (only 2e/Coulomb)
*/
void DHF_SPH::evaluateFock_2e(MatrixXd& fock, const bool& twoC, const vMatrixXd& den, const int& size, const int& Iirrep)
{
    int ir = all2compact(Iirrep);
    if(!twoC)
    {
        fock.resize(size*2,size*2);
        #pragma omp parallel  for
        for(int mm = 0; mm < size; mm++)
        for(int nn = 0; nn <= mm; nn++)
        {
            fock(mm,nn) = 0.0;
            fock(mm+size,nn) = 0.0;
            if(mm != nn) fock(nn+size,mm) = 0.0;
            fock(mm+size,nn+size) = 0.0;
            for(int jr = 0; jr < occMax_irrep_compact; jr++)
            {
                int Jirrep = compact2all(jr);
                double twojP1 = irrep_list(Jirrep).two_j+1;
                int size_tmp2 = irrep_list(Jirrep).size;
                for(int ss = 0; ss < size_tmp2; ss++)
                for(int rr = 0; rr < size_tmp2; rr++)
                {
                    int emn = mm*size+nn, esr = ss*size_tmp2+rr, emr = mm*size_tmp2+rr, esn = ss*size+nn;
                    fock(mm,nn) += twojP1*den(Jirrep)(ss,rr) * h2eLLLL_JK.J[ir][jr][emn][esr] + twojP1*den(Jirrep)(size_tmp2+ss,size_tmp2+rr) * h2eSSLL_JK.J[jr][ir][esr][emn];
                    fock(mm+size,nn) -= twojP1*den(Jirrep)(ss,size_tmp2+rr) * h2eSSLL_JK.K[ir][jr][emr][esn];
                    if(mm != nn) 
                    {
                        int enr = nn*size_tmp2+rr, esm = ss*size+mm;
                        fock(nn+size,mm) -= twojP1*den(Jirrep)(ss,size_tmp2+rr) * h2eSSLL_JK.K[ir][jr][enr][esm];
                    }
                    fock(mm+size,nn+size) += twojP1*den(Jirrep)(size_tmp2+ss,size_tmp2+rr) * h2eSSSS_JK.J[ir][jr][emn][esr] + twojP1*den(Jirrep)(ss,rr) * h2eSSLL_JK.J[ir][jr][emn][esr];
                    if(with_gaunt)
                    {
                        int enm = nn*size+mm, ers = rr*size_tmp2+ss, erm = rr*size+mm, ens = nn*size_tmp2+ss;
                        fock(mm,nn) -= twojP1*den(Jirrep)(size_tmp2+ss,size_tmp2+rr) * gauntLSSL_JK.K[ir][jr][emr][esn];
                        fock(mm+size,nn+size) -= twojP1*den(Jirrep)(ss,rr) * gauntLSSL_JK.K[jr][ir][esn][emr];
                        fock(mm+size,nn) += twojP1*den(Jirrep)(size_tmp2+ss,rr)*gauntLSLS_JK.J[ir][jr][enm][ers] + twojP1*den(Jirrep)(ss,size_tmp2+rr) * gauntLSSL_JK.J[jr][ir][esr][emn];
                        if(mm != nn) 
                        {
                            int ern = rr*size+nn, ems = mm*size_tmp2+ss;
                            fock(nn+size,mm) += twojP1*den(Jirrep)(size_tmp2+ss,rr)*gauntLSLS_JK.J[ir][jr][emn][ers] + twojP1*den(Jirrep)(ss,size_tmp2+rr) * gauntLSSL_JK.J[jr][ir][esr][enm];
                        }
                    }
                }
            }
            fock(nn,mm) = fock(mm,nn);
            fock(nn+size,mm+size) = fock(mm+size,nn+size);
            fock(nn,mm+size) = fock(mm+size,nn);
            fock(mm,nn+size) = fock(nn+size,mm);
        }
    }
    else
    {
        fock.resize(size,size);
        #pragma omp parallel  for
        for(int mm = 0; mm < size; mm++)
        for(int nn = 0; nn <= mm; nn++)
        {
            fock(mm,nn) = 0.0;
            for(int jr = 0; jr < occMax_irrep_compact; jr++)
            {
                int Jirrep = compact2all(jr);
                double twojP1 = irrep_list(Jirrep).two_j+1;
                int size_tmp2 = irrep_list(Jirrep).size;
                for(int ss = 0; ss < size_tmp2; ss++)
                for(int rr = 0; rr < size_tmp2; rr++)
                {
                    int emn = mm*size+nn, esr = ss*size_tmp2+rr, emr = mm*size_tmp2+rr, esn = ss*size+nn;
                    fock(mm,nn) += twojP1*den(Jirrep)(ss,rr) * h2eLLLL_JK.J[ir][jr][emn][esr];
                }
            }
            fock(nn,mm) = fock(mm,nn);
        }
    }
}
void DHF_SPH::evaluateFock_J(MatrixXd& fock, const bool& twoC, const vMatrixXd& den, const int& size, const int& Iirrep)
{
    cout << "evaluateFock_J is closed now" << endl;
    exit(99);
    // if(!twoC)
    // {
    //     fock.resize(size*2,size*2);
    //     #pragma omp parallel  for
    //     for(int mm = 0; mm < size; mm++)
    //     for(int nn = 0; nn <= mm; nn++)
    //     {
    //         fock(mm,nn) = 0.0;
    //         fock(mm+size,nn) = 0.0;
    //         if(mm != nn) fock(nn+size,mm) = 0.0;
    //         fock(mm+size,nn+size) = 0.0;
    //         for(int jr = 0; jr < occMax_irrep; jr++)
    //         {
    //             int size_tmp2 = irrep_list(jr).size;
    //             for(int ss = 0; ss < size_tmp2; ss++)
    //             for(int rr = 0; rr < size_tmp2; rr++)
    //             {
    //                 int emn = mm*size+nn, esr = ss*size_tmp2+rr, emr = mm*size_tmp2+rr, esn = ss*size+nn;
    //                 fock(mm,nn) += den(jr)(ss,rr) * (h2eLLLL_JK.J(Iirrep,jr)(emn,esr)) + den(jr)(size_tmp2+ss,size_tmp2+rr) * h2eSSLL_JK.J(jr,Iirrep)(esr,emn);
    //                 fock(mm+size,nn+size) += den(jr)(size_tmp2+ss,size_tmp2+rr) * (h2eSSSS_JK.J(Iirrep,jr)(emn,esr)) + den(jr)(ss,rr) * h2eSSLL_JK.J(Iirrep,jr)(emn,esr);
    //                 if(with_gaunt)
    //                 {
    //                     int enm = nn*size+mm, ers = rr*size_tmp2+ss, erm = rr*size+mm, ens = nn*size_tmp2+ss;
    //                     fock(mm+size,nn) += den(jr)(size_tmp2+ss,rr)*(gauntLSLS_JK.J(Iirrep,jr)(enm,ers)) + den(jr)(ss,size_tmp2+rr) * gauntLSSL_JK.J(jr,Iirrep)(esr,emn);
    //                     if(mm != nn) 
    //                     {
    //                         int ern = rr*size+nn, ems = mm*size_tmp2+ss;
    //                         fock(nn+size,mm) += den(jr)(size_tmp2+ss,rr)*(gauntLSLS_JK.J(Iirrep,jr)(emn,ers)) + den(jr)(ss,size_tmp2+rr) * gauntLSSL_JK.J(jr,Iirrep)(esr,enm);
    //                     }
    //                 }
    //             }
    //         }
    //         fock(nn,mm) = fock(mm,nn);
    //         fock(nn+size,mm+size) = fock(mm+size,nn+size);
    //         fock(nn,mm+size) = fock(mm+size,nn);
    //         fock(mm,nn+size) = fock(nn+size,mm);
    //     }
    // }
    // else
    // {
    //     fock.resize(size,size);
    //     #pragma omp parallel  for
    //     for(int mm = 0; mm < size; mm++)
    //     for(int nn = 0; nn <= mm; nn++)
    //     {
    //         fock(mm,nn) = 0.0;
    //         for(int jr = 0; jr < occMax_irrep; jr++)
    //         {
    //             int size_tmp2 = irrep_list(jr).size;
    //             for(int ss = 0; ss < size_tmp2; ss++)
    //             for(int rr = 0; rr < size_tmp2; rr++)
    //             {
    //                 int emn = mm*size+nn, esr = ss*size_tmp2+rr, emr = mm*size_tmp2+rr, esn = ss*size+nn;
    //                 fock(mm,nn) += den(jr)(ss,rr) * (h2eLLLL_JK.J(Iirrep,jr)(emn,esr));
    //             }
    //         }
    //         fock(nn,mm) = fock(mm,nn);
    //     }
    // }
}

void DHF_SPH_CA::evaluateFock_2e(MatrixXd& fock_c, const bool& twoC, const Matrix<vMatrixXd,-1,1>& densities, const int& size, const int& Iirrep)
{
    int ir = all2compact(Iirrep);
    vMatrixXd R(NOpenShells+2);
    vMatrixXd Q(NOpenShells+1);
    for(int ii = 0; ii < NOpenShells+2; ii++)
    {
        R(ii) = densities(ii)(Iirrep).transpose();
        if(ii < NOpenShells+1)
        {
            if(twoC)  Q(ii) = MatrixXd::Zero(size,size);
            else      Q(ii) = MatrixXd::Zero(2*size,2*size);
        }
            
    }     
    if(twoC)
    {
        #pragma omp parallel  for
        for(int mm = 0; mm < size; mm++)
        for(int nn = 0; nn <= mm; nn++)
        {
            for(int jr = 0; jr < occMax_irrep_compact; jr++)
            {
                int Jirrep = compact2all(jr);
                double twojP1 = irrep_list(Jirrep).two_j+1;
                ;
                int size_tmp2 = irrep_list(Jirrep).size;
                for(int aa = 0; aa < size_tmp2; aa++)
                for(int bb = 0; bb < size_tmp2; bb++)
                {
                    int emn = mm*size+nn, eab = aa*size_tmp2+bb, emb = mm*size_tmp2+bb, ean = aa*size+nn;
                    for(int ii = 0; ii < NOpenShells+1; ii++)
                    {
                        Q(ii)(mm,nn) += twojP1*densities(ii)(Jirrep)(aa,bb) * h2eLLLL_JK.J[ir][jr][emn][eab];
                    }
                }
            }
            for(int ii = 0; ii < NOpenShells+1; ii++)
            {
                Q(ii)(nn,mm) = Q(ii)(mm,nn);
            }
        }
    }
    else
    {
        #pragma omp parallel  for
        for(int mm = 0; mm < size; mm++)
        for(int nn = 0; nn <= mm; nn++)
        {
            MatrixXd den_tmp;            
            for(int jr = 0; jr < occMax_irrep_compact; jr++)
            {
                int Jirrep = compact2all(jr);
                double twojP1 = irrep_list(Jirrep).two_j+1;
                int size_tmp2 = irrep_list(Jirrep).size;
                for(int ss = 0; ss < size_tmp2; ss++)
                for(int rr = 0; rr < size_tmp2; rr++)
                {
                    int emn = mm*size+nn, esr = ss*size_tmp2+rr, emr = mm*size_tmp2+rr, esn = ss*size+nn;
                    for(int ii = 0; ii < NOpenShells+1; ii++)
                    {
                        den_tmp = densities(ii)(Jirrep);
                        Q(ii)(mm,nn) += twojP1*den_tmp(ss,rr) * h2eLLLL_JK.J[ir][jr][emn][esr] + twojP1*den_tmp(size_tmp2+ss,size_tmp2+rr) * h2eSSLL_JK.J[jr][ir][esr][emn];
                        Q(ii)(mm+size,nn) -= twojP1*den_tmp(ss,size_tmp2+rr) * h2eSSLL_JK.K[ir][jr][emr][esn];
                        Q(ii)(mm+size,nn+size) += twojP1*den_tmp(size_tmp2+ss,size_tmp2+rr) * h2eSSSS_JK.J[ir][jr][emn][esr] + twojP1*den_tmp(ss,rr) * h2eSSLL_JK.J[ir][jr][emn][esr];
                        if(mm != nn) 
                        {
                            int enr = nn*size_tmp2+rr, esm = ss*size+mm;
                            Q(ii)(nn+size,mm) -= twojP1*den_tmp(ss,size_tmp2+rr) * h2eSSLL_JK.K[ir][jr][enr][esm];
                        }
                        if(with_gaunt)
                        {
                            int enm = nn*size+mm, ers = rr*size_tmp2+ss, erm = rr*size+mm, ens = nn*size_tmp2+ss;

                            Q(ii)(mm,nn) -= twojP1*den_tmp(size_tmp2+ss,size_tmp2+rr) * gauntLSSL_JK.K[ir][jr][emr][esn];
                            Q(ii)(mm+size,nn) += twojP1*den_tmp(ss+size_tmp2,rr)*gauntLSLS_JK.J[ir][jr][enm][ers] + twojP1*den_tmp(ss,size_tmp2+rr) * gauntLSSL_JK.J[jr][ir][esr][emn];
                            Q(ii)(mm+size,nn+size) -= twojP1*den_tmp(ss,rr) * gauntLSSL_JK.K[jr][ir][esn][emr];
                            if(mm != nn)
                            {
                                int ern = rr*size+nn, ems = mm*size_tmp2+ss;
                                Q(ii)(nn+size,mm) += twojP1*den_tmp(size_tmp2+ss,rr)*gauntLSLS_JK.J[ir][jr][emn][ers] + twojP1*den_tmp(ss,size_tmp2+rr) * gauntLSSL_JK.J[jr][ir][esr][enm];
                            }
                        }
                    }               
                }
            }
            for(int ii = 0; ii < NOpenShells+1; ii++)
            {
                Q(ii)(nn,mm) = Q(ii)(mm,nn);
                Q(ii)(mm,nn+size) = Q(ii)(nn+size,mm);
                Q(ii)(nn,mm+size) = Q(ii)(mm+size,nn);
                Q(ii)(size+nn,size+mm) = Q(ii)(size+mm,size+nn);
            }
        }
    }

    if(twoC)  fock_c = MatrixXd::Zero(size,size);
    else      fock_c = MatrixXd::Zero(2*size,2*size);
    for(int ii = 0; ii < NOpenShells+1; ii++)
    {
        if(ii != 0)
            Q(ii) = Q(ii)*f_list[ii-1];
        fock_c += Q(ii);
    }

    MatrixXd S;
    if(!twoC) S = overlap_4c(Iirrep);
    else S = overlap(Iirrep);

    MatrixXd LM;
    if(twoC)  LM = MatrixXd::Zero(size,size);
    else      LM = MatrixXd::Zero(2*size,2*size);

    for(int ii = 1; ii < NOpenShells+1; ii++)
    {
        double f_u = f_list[ii-1];
        double a_u = MM_list[ii-1]*(NN_list[ii-1]-1.0)/NN_list[ii-1]/(MM_list[ii-1]-1.0);
        double alpha_u = (1-a_u)/(1-f_u);
        LM += S*R(ii)*Q(ii)*(alpha_u*f_u*R(0)+(a_u-1.0)*(R(ii)+R(NOpenShells+1)))*S;
        
        for(int jj = 1; jj < NOpenShells+1; jj++)
        {
            if(ii != jj)
            {
                double a_v = MM_list[jj-1]*(NN_list[jj-1]-1.0)/NN_list[jj-1]/(MM_list[jj-1]-1.0);
                double f_v = f_list[jj-1];
                if(abs(f_u-f_v) > 1e-4)
                {
                    // LM += S*R[ii]*( (a_u-1.0)/(f_u-f_v)*Q(ii) + (a_v-1.0)/(f_v-f_u)*Q(jj) ) *R(jj)*S;
                    LM += S*R[ii]*( (a_u-1.0)/(f_u-f_v)*f_u*Q(ii) + (a_v-1.0)/(f_v-f_u)*f_v*Q(jj) ) *R(jj)*S;
                }
                else
                {
                    LM += S*R(ii)*(-fock_c + (a_u-1.0)*f_u*Q(ii) - (a_v-1.0)*f_v*Q(jj))*R(jj)*S;
                }
            }
        }
    }
    fock_c += LM + LM.adjoint();
}

vMatrixXd DHF_SPH_CA::x2c2ePCC(vMatrixXd* coeff2c)
{
    cout << "Running DHF_SPH_CA::x2c2ePCC" << endl;
    if(!converged)
    {
        cout << "SCF did not converge. x2c2ePCC cannot be used!" << endl;
        exit(99);
    }
    vMatrixXd fock_pcc(occMax_irrep), fock_4c_2e(occMax_irrep), fock_x2c2e(occMax_irrep), fock_x2c2e_2e(occMax_irrep), JK_x2c2c(occMax_irrep), coeff_2c(occMax_irrep), density_pcc(occMax_irrep), h1e_x2c2e(occMax_irrep), h1e_x2c1e(occMax_irrep), densityCore_4c(occMax_irrep), densityCore_2c(occMax_irrep), fock_pcc_mo(occMax_irrep);
    vMatrixXd XXX(occMax_irrep), RRR(occMax_irrep), XXX_1e(occMax_irrep), RRR_1e(occMax_irrep);
    vMatrixXd overlap_2c(occMax_irrep), overlap_h_i_2c(occMax_irrep);
    
    Matrix<vMatrixXd,-1,1> densityShells_2c(NOpenShells+2);
    for(int ii = 0; ii < NOpenShells+2; ii++)
        densityShells_2c(ii).resize(occMax_irrep);

    for(int ir = 0; ir < occMax_irrep; ir++)
    {
        overlap_2c(ir) = overlap_4c(ir).block(0,0,overlap_4c(ir).rows()/2,overlap_4c(ir).cols()/2);
        overlap_h_i_2c(ir) = matrix_half_inverse(overlap_2c(ir));
        VectorXd ene_mo_tmp;
        XXX(ir) = X2C::get_X(coeff(ir));
        RRR(ir) = X2C::get_R(overlap_4c(ir),XXX(ir));
        
        h1e_x2c2e(ir) = X2C::transform_4c_2c(h1e_4c(ir), XXX(ir), RRR(ir));
        fock_x2c2e(ir) = X2C::transform_4c_2c(fock_4c(ir), XXX(ir), RRR(ir));
        if(coeff2c == NULL)
            eigensolverG(fock_x2c2e(ir),overlap_h_i_2c(ir),ene_mo_tmp,coeff_2c(ir));
        else
            coeff_2c(ir) = (*coeff2c)(ir);

        for (int ii = 0; ii < NOpenShells+2; ii++)
        {
            densityShells_2c(ii)(ir) = evaluateDensity_aoc(coeff_2c(ir),occNumberShells[ii](ir),true);
        }
        // X2C1E
        XXX_1e(ir) = X2C::get_X(overlap(ir),kinetic(ir),WWW(ir),Vnuc(ir));
        RRR_1e(ir) = X2C::get_R(overlap(ir),kinetic(ir),XXX_1e(ir));
        h1e_x2c1e(ir) = X2C::evaluate_h1e_x2c(overlap(ir),kinetic(ir),WWW(ir),Vnuc(ir),XXX_1e(ir),RRR_1e(ir));
    }    

    for(int ir = 0; ir < occMax_irrep; ir++)
    {
        evaluateFock_2e(fock_4c_2e(ir),false,densityShells,irrep_list(ir).size,ir);
        evaluateFock_2e(JK_x2c2c(ir),true,densityShells_2c,irrep_list(ir).size,ir);
        fock_x2c2e_2e(ir) = X2C::transform_4c_2c(fock_4c_2e(ir), XXX(ir), RRR(ir));
    }
    for(int ir = 0; ir < occMax_irrep; ir++)
    {  
        fock_pcc(ir) = fock_x2c2e_2e(ir) - JK_x2c2c(ir) + h1e_x2c2e(ir) - h1e_x2c1e(ir);
    }

    x2cXXX = XXX;
    x2cRRR = RRR;
    X_calculated = true;

    return fock_pcc;
}
