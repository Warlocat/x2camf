#include"general.h"
#include"dhf_sph.h"
#include<iostream>
using namespace std;
using namespace Eigen;

vMatrixXd DHF_SPH::x2c2ePCC()
{
    if(!converged)
    {
        cout << "SCF did not converge. x2c2ePCC cannot be used!" << endl;
        exit(99);
    }
    vMatrixXd fock_pcc(occMax_irrep), fock_4c_2e(occMax_irrep), fock_x2c2e(occMax_irrep), fock_x2c2e_2e(occMax_irrep), JK_x2c2c(occMax_irrep), coeff_2c(occMax_irrep), density_2c(occMax_irrep), h1e_x2c2e(occMax_irrep), h1e_x2c1e(occMax_irrep);
    vMatrixXd overlap_2c(occMax_irrep), overlap_h_i_2c(occMax_irrep);
    for(int ir = 0; ir < occMax_irrep; ir++)
    {
        overlap_2c(ir) = overlap_4c(ir).block(0,0,overlap_4c(ir).rows()/2,overlap_4c(ir).cols()/2);
        overlap_h_i_2c(ir) = matrix_half_inverse(overlap_2c(ir));
        VectorXd ene_mo_tmp;
        MatrixXd XXX = X2C::get_X(coeff(ir));
        MatrixXd RRR = X2C::get_R(overlap_4c(ir),XXX);
        fock_4c_2e(ir) = fock_4c(ir) - h1e_4c(ir);
        h1e_x2c2e(ir) = X2C::transform_4c_2c(h1e_4c(ir), XXX, RRR);
        fock_x2c2e(ir) = X2C::transform_4c_2c(fock_4c(ir), XXX, RRR);
        fock_x2c2e_2e(ir) = X2C::transform_4c_2c(fock_4c_2e(ir), XXX, RRR);
        eigensolverG(fock_x2c2e(ir),overlap_h_i_2c(ir),ene_mo_tmp,coeff_2c(ir));
        density_2c(ir) = evaluateDensity_spinor(coeff_2c(ir),occNumber(ir),true);

        // X2C1E
        MatrixXd XXX_1e = X2C::get_X(overlap(ir),kinetic(ir),WWW(ir),Vnuc(ir));
        MatrixXd RRR_1e = X2C::get_R(overlap(ir),kinetic(ir),XXX_1e);
        h1e_x2c1e(ir) = X2C::evaluate_h1e_x2c(overlap(ir),kinetic(ir),WWW(ir),Vnuc(ir),XXX_1e,RRR_1e);
    }
    for(int ir = 0; ir < occMax_irrep; ir++)
    {
        evaluateFock_2e(JK_x2c2c(ir),true,density_2c,irrep_list(ir).size,ir);
        fock_pcc(ir) = fock_x2c2e_2e(ir) - JK_x2c2c(ir) + h1e_x2c2e(ir) - h1e_x2c1e(ir);
    }
    return fock_pcc;
}


/* 
    evaluate Fock matrix (only 2e/Coulomb)
*/
void DHF_SPH::evaluateFock_2e(MatrixXd& fock, const bool& twoC, const vMatrixXd& den, const int& size, const int& Iirrep)
{
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
            for(int jr = 0; jr < occMax_irrep; jr++)
            {
                int size_tmp2 = irrep_list(jr).size;
                for(int ss = 0; ss < size_tmp2; ss++)
                for(int rr = 0; rr < size_tmp2; rr++)
                {
                    int emn = mm*size+nn, esr = ss*size_tmp2+rr, emr = mm*size_tmp2+rr, esn = ss*size+nn;
                    fock(mm,nn) += den(jr)(ss,rr) * (h2eLLLL_JK.J(Iirrep,jr)(emn,esr) - h2eLLLL_JK.K(Iirrep,jr)(emr,esn)) + den(jr)(size_tmp2+ss,size_tmp2+rr) * h2eSSLL_JK.J(jr,Iirrep)(esr,emn);
                    fock(mm+size,nn) -= den(jr)(ss,size_tmp2+rr) * h2eSSLL_JK.K(Iirrep,jr)(emr,esn);
                    if(mm != nn) 
                    {
                        int enr = nn*size_tmp2+rr, esm = ss*size+mm;
                        fock(nn+size,mm) -= den(jr)(ss,size_tmp2+rr) * h2eSSLL_JK.K(Iirrep,jr)(enr,esm);
                    }
                    fock(mm+size,nn+size) += den(jr)(size_tmp2+ss,size_tmp2+rr) * (h2eSSSS_JK.J(Iirrep,jr)(emn,esr) - h2eSSSS_JK.K(Iirrep,jr)(emr,esn)) + den(jr)(ss,rr) * h2eSSLL_JK.J(Iirrep,jr)(emn,esr);
                    if(with_gaunt)
                    {
                        int enm = nn*size+mm, ers = rr*size_tmp2+ss, erm = rr*size+mm, ens = nn*size_tmp2+ss;
                        fock(mm,nn) -= den(jr)(size_tmp2+ss,size_tmp2+rr) * gauntLSSL_JK.K(Iirrep,jr)(emr,esn);
                        fock(mm+size,nn+size) -= den(jr)(ss,rr) * gauntLSSL_JK.K(jr,Iirrep)(esn,emr);
                        fock(mm+size,nn) += den(jr)(size_tmp2+ss,rr)*(gauntLSLS_JK.J(Iirrep,jr)(enm,ers) - gauntLSLS_JK.K(jr,Iirrep)(erm,ens)) + den(jr)(ss,size_tmp2+rr) * gauntLSSL_JK.J(jr,Iirrep)(esr,emn);
                        if(mm != nn) 
                        {
                            int ern = rr*size+nn, ems = mm*size_tmp2+ss;
                            fock(nn+size,mm) += den(jr)(size_tmp2+ss,rr)*(gauntLSLS_JK.J(Iirrep,jr)(emn,ers) - gauntLSLS_JK.K(jr,Iirrep)(ern,ems)) + den(jr)(ss,size_tmp2+rr) * gauntLSSL_JK.J(jr,Iirrep)(esr,enm);
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
            for(int jr = 0; jr < occMax_irrep; jr++)
            {
                int size_tmp2 = irrep_list(jr).size;
                for(int ss = 0; ss < size_tmp2; ss++)
                for(int rr = 0; rr < size_tmp2; rr++)
                {
                    int emn = mm*size+nn, esr = ss*size_tmp2+rr, emr = mm*size_tmp2+rr, esn = ss*size+nn;
                    fock(mm,nn) += den(jr)(ss,rr) * (h2eLLLL_JK.J(Iirrep,jr)(emn,esr) - h2eLLLL_JK.K(Iirrep,jr)(emr,esn));
                }
            }
            fock(nn,mm) = fock(mm,nn);
        }
    }
}
void DHF_SPH::evaluateFock_J(MatrixXd& fock, const bool& twoC, const vMatrixXd& den, const int& size, const int& Iirrep)
{
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
            for(int jr = 0; jr < occMax_irrep; jr++)
            {
                int size_tmp2 = irrep_list(jr).size;
                for(int ss = 0; ss < size_tmp2; ss++)
                for(int rr = 0; rr < size_tmp2; rr++)
                {
                    int emn = mm*size+nn, esr = ss*size_tmp2+rr, emr = mm*size_tmp2+rr, esn = ss*size+nn;
                    fock(mm,nn) += den(jr)(ss,rr) * (h2eLLLL_JK.J(Iirrep,jr)(emn,esr)) + den(jr)(size_tmp2+ss,size_tmp2+rr) * h2eSSLL_JK.J(jr,Iirrep)(esr,emn);
                    fock(mm+size,nn+size) += den(jr)(size_tmp2+ss,size_tmp2+rr) * (h2eSSSS_JK.J(Iirrep,jr)(emn,esr)) + den(jr)(ss,rr) * h2eSSLL_JK.J(Iirrep,jr)(emn,esr);
                    if(with_gaunt)
                    {
                        int enm = nn*size+mm, ers = rr*size_tmp2+ss, erm = rr*size+mm, ens = nn*size_tmp2+ss;
                        fock(mm+size,nn) += den(jr)(size_tmp2+ss,rr)*(gauntLSLS_JK.J(Iirrep,jr)(enm,ers)) + den(jr)(ss,size_tmp2+rr) * gauntLSSL_JK.J(jr,Iirrep)(esr,emn);
                        if(mm != nn) 
                        {
                            int ern = rr*size+nn, ems = mm*size_tmp2+ss;
                            fock(nn+size,mm) += den(jr)(size_tmp2+ss,rr)*(gauntLSLS_JK.J(Iirrep,jr)(emn,ers)) + den(jr)(ss,size_tmp2+rr) * gauntLSSL_JK.J(jr,Iirrep)(esr,enm);
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
            for(int jr = 0; jr < occMax_irrep; jr++)
            {
                int size_tmp2 = irrep_list(jr).size;
                for(int ss = 0; ss < size_tmp2; ss++)
                for(int rr = 0; rr < size_tmp2; rr++)
                {
                    int emn = mm*size+nn, esr = ss*size_tmp2+rr, emr = mm*size_tmp2+rr, esn = ss*size+nn;
                    fock(mm,nn) += den(jr)(ss,rr) * (h2eLLLL_JK.J(Iirrep,jr)(emn,esr));
                }
            }
            fock(nn,mm) = fock(mm,nn);
        }
    }
}