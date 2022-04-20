#include"general.h"
#include"dhf_sph.h"
#include<iomanip>
#include<iostream>
using namespace std;
using namespace Eigen;

vMatrixXd DHF_SPH::x2c2ePCC(vMatrixXd* coeff2c)
{
    if(!converged)
    {
        cout << "SCF did not converge. x2c2ePCC cannot be used!" << endl;
        exit(99);
    }
    vMatrixXd fock_pcc(occMax_irrep), fock_4c_2e(occMax_irrep), fock_x2c2e(occMax_irrep), fock_x2c2e_2e(occMax_irrep), JK_x2c2c(occMax_irrep), coeff_2c(occMax_irrep), density_2c(occMax_irrep), density_pcc(occMax_irrep), h1e_x2c2e(occMax_irrep), h1e_x2c1e(occMax_irrep), densityCore_4c(occMax_irrep), densityCore_2c(occMax_irrep), fock_pcc_mo(occMax_irrep);
    vMatrixXd XXX(occMax_irrep), RRR(occMax_irrep), XXX_1e(occMax_irrep), RRR_1e(occMax_irrep);
    vMatrixXd overlap_2c(occMax_irrep), overlap_h_i_2c(occMax_irrep);
    // for(int ir = 0; ir < occNumberCore.rows(); ir++)
    // {
    //     occNumberCore(ir) = VectorXd::Zero(occNumberCore(ir).rows());
    //     if(ir == 0 || ir == 1)
    //     {
    //         occNumberCore(ir)(0) = 1.0;
    //     }
    //     cout << irrep_list(ir).l << "\t" << irrep_list(ir).two_j/2.0 << "\t" << irrep_list(ir).two_mj/2.0 << "\t" << occNumberCore(ir).transpose() << endl;
    // }


    for(int ir = 0; ir < occMax_irrep; ir++)
    {
        overlap_2c(ir) = overlap_4c(ir).block(0,0,overlap_4c(ir).rows()/2,overlap_4c(ir).cols()/2);
        overlap_h_i_2c(ir) = matrix_half_inverse(overlap_2c(ir));
        VectorXd ene_mo_tmp;
        XXX(ir) = X2C::get_X(coeff(ir));
        RRR(ir) = X2C::get_R(overlap_4c(ir),XXX(ir));
        // densityCore_4c(ir) = evaluateDensity_spinor(coeff(ir), occNumberCore(ir), false);
        
        h1e_x2c2e(ir) = X2C::transform_4c_2c(h1e_4c(ir), XXX(ir), RRR(ir));
        fock_x2c2e(ir) = X2C::transform_4c_2c(fock_4c(ir), XXX(ir), RRR(ir));
        if(coeff2c == NULL)
            eigensolverG(fock_x2c2e(ir),overlap_h_i_2c(ir),ene_mo_tmp,coeff_2c(ir));
        else
            coeff_2c(ir) = (*coeff2c)(ir);

        // densityCore_2c(ir) = evaluateDensity_spinor(coeff_2c(ir),occNumberCore(ir),true);
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
    // for(int ir = 0; ir < occNumberCore.rows(); ir++)
    // {
    //     occNumberCore(ir) = VectorXd::Zero(occNumberCore(ir).rows());
    //     if(ir == 0 || ir == 1)
    //     {
    //         occNumberCore(ir)(0) = 1.0;
    //     }
    //     cout << irrep_list(ir).l << "\t" << irrep_list(ir).two_j/2.0 << "\t" << irrep_list(ir).two_mj/2.0 << "\t" << occNumberCore(ir).transpose() << endl;
    // }


    for(int ir = 0; ir < occMax_irrep; ir++)
    {
        overlap_2c(ir) = overlap_4c(ir).block(0,0,overlap_4c(ir).rows()/2,overlap_4c(ir).cols()/2);
        overlap_h_i_2c(ir) = matrix_half_inverse(overlap_2c(ir));
        VectorXd ene_mo_tmp;
        XXX(ir) = X2C::get_X(coeff(ir));
        RRR(ir) = X2C::get_R(overlap_4c(ir),XXX(ir));
        // densityCore_4c(ir) = evaluateDensity_spinor(coeff(ir), occNumberCore(ir), false);
        
        h1e_x2c2e(ir) = X2C::transform_4c_2c(h1e_4c(ir), XXX(ir), RRR(ir));
        fock_x2c2e(ir) = X2C::transform_4c_2c(fock_4c(ir), XXX(ir), RRR(ir));
        if(coeff2c == NULL)
            eigensolverG(fock_x2c2e(ir),overlap_h_i_2c(ir),ene_mo_tmp,coeff_2c(ir));
        else
            coeff_2c(ir) = (*coeff2c)(ir);

        // densityCore_2c(ir) = evaluateDensity_spinor(coeff_2c(ir),occNumberCore(ir),true);
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

vMatrixXd DHF_SPH::x2c2ePCC_density()
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
        densityCore_4c(ir) = evaluateDensity_spinor(coeff(ir), occNumberCore(ir), false);
        
        // fock_4c_2e(ir) = fock_4c(ir) - h1e_4c(ir);
        h1e_x2c2e(ir) = X2C::transform_4c_2c(h1e_4c(ir), XXX(ir), RRR(ir));
        fock_x2c2e(ir) = X2C::transform_4c_2c(fock_4c(ir), XXX(ir), RRR(ir));
        eigensolverG(fock_x2c2e(ir),overlap_h_i_2c(ir),ene_mo_tmp,coeff_2c(ir));
        densityCore_2c(ir) = evaluateDensity_spinor(coeff_2c(ir),occNumberCore(ir),true);
        density_2c(ir) = evaluateDensity_spinor(coeff_2c(ir),occNumber(ir),true);

        // X2C1E
        XXX_1e(ir) = X2C::get_X(overlap(ir),kinetic(ir),WWW(ir),Vnuc(ir));
        RRR_1e(ir) = X2C::get_R(overlap(ir),kinetic(ir),XXX_1e(ir));
        h1e_x2c1e(ir) = X2C::evaluate_h1e_x2c(overlap(ir),kinetic(ir),WWW(ir),Vnuc(ir),XXX_1e(ir),RRR_1e(ir));
    }
    for(int ir = 0; ir < occMax_irrep; ir++)
    {
        int size_nr = density_2c(ir).rows();
        // evaluateFock_J(fock_4c_2e(ir),false,density,irrep_list(ir).size,ir);
        // evaluateFock_J(JK_x2c2c(ir),true,density_2c,irrep_list(ir).size,ir);
        evaluateFock_2e(fock_4c_2e(ir),false,density,irrep_list(ir).size,ir);
        evaluateFock_2e(JK_x2c2c(ir),true,density_2c,irrep_list(ir).size,ir);
        //evaluateFock_2e(fock_4c_2e(ir),false,densityCore_4c,irrep_list(ir).size,ir);
        //evaluateFock_2e(JK_x2c2c(ir),true,densityCore_2c,irrep_list(ir).size,ir);
        fock_x2c2e_2e(ir) = X2C::transform_4c_2c(fock_4c_2e(ir), XXX(ir), RRR(ir));

        density_pcc(ir).resize(size_nr,size_nr);
        for(int ii = 0; ii < size_nr; ii++)
        for(int jj = 0; jj < size_nr; jj++)
        {
            density_pcc(ir)(ii,jj) = density(ir)(ii,jj);
        }
    }
    for(int ir = 0; ir < occMax_irrep; ir++)
    {   
        fock_pcc(ir) = fock_x2c2e_2e(ir) - JK_x2c2c(ir) + h1e_x2c2e(ir) - h1e_x2c1e(ir);
        // fock_pcc(ir) = fock_4c_2e(ir).block(0,0,fock_4c_2e(ir).rows()/2,fock_4c_2e(ir).cols()/2) - JK_x2c2c(ir);
    }

    // for(int ir = 0; ir < occMax_irrep; ir ++)
    // {
    //     MatrixXd fock_pcc_mo;
    //     fock_pcc_mo = coeff_2c(ir).adjoint() * fock_pcc(ir) * coeff_2c(ir);
    //     cout << fock_pcc_mo << endl<< endl;
    //     // for(int ii = 0; ii < fock_pcc_mo.rows(); ii++)
    //     // {
    //     //     // if(ir == 0|| ir ==1)
    //     //     if(occNumberCore(ir).rows() > 0 && occNumberCore(ir)(0) > 1e-4)
    //     //     {
    //     //         if(abs(occNumber(ir)(ii)) < 1e-4)
    //     //         // if(ir == 0|| ir ==1)
    //     //         {
    //     //             for(int jj = ii+3; jj < fock_pcc_mo.rows(); jj++)
    //     //             for(int kk = 0; kk < fock_pcc_mo.rows(); kk++)
    //     //             {
    //     //                 fock_pcc_mo(kk,jj) = 0.0;
    //     //                 fock_pcc_mo(jj,kk) = 0.0;
    //     //             }
    //     //             break;
    //     //         }
    //     //     }
    //     //     else
    //     //     {
    //     //         for(int jj = 0; jj < fock_pcc_mo.rows(); jj++)
    //     //         for(int kk = 0; kk < fock_pcc_mo.rows(); kk++)
    //     //         {
    //     //             fock_pcc_mo(kk,jj) = 0.0;
    //     //             fock_pcc_mo(jj,kk) = 0.0;
    //     //         }
    //     //         break;
    //     //     }
    //     // }
    //     for(int ii = 0; ii < fock_pcc_mo.rows(); ii++)
    //     {
    //         if(abs(fock_pcc_mo(ii,ii)) < 1e-0)
    //         {
    //             for( int jj = 0; jj < fock_pcc_mo.rows(); jj++)
    //             {
    //                 fock_pcc_mo(ii,jj) = 0.0;
    //                 fock_pcc_mo(jj,ii) = 0.0;
    //             }
    //         }
    //     }
    //     cout << fock_pcc_mo << endl<< endl;
    //     fock_pcc(ir) = coeff_2c(ir).adjoint().inverse() * fock_pcc_mo * coeff_2c(ir).inverse();
    // }

    int size_solve = 0;
    for(int ir = 0; ir < occMax_irrep; ir ++)
    {   
        size_solve += irrep_list(ir).size*(irrep_list(ir).size+1)/2;
    }
    VectorXd solve_X(size_solve),solve_B(size_solve);
    MatrixXd solve_A(size_solve,size_solve);
    
    int tmp_int_i = 0;
    for(int ir = 0; ir < occMax_irrep; ir ++)
    {
        int size_i = irrep_list(ir).size;
        for(int ii = 0; ii < size_i; ii++)
        for(int jj = 0; jj <= ii; jj++)
        {
            solve_B(tmp_int_i + ii*(ii+1)/2+jj) = fock_pcc(ir)(ii,jj);
            int tmp_int_j = 0;
            for(int jr = 0; jr < occMax_irrep; jr++)
            {
                int size_j = irrep_list(jr).size;
                for(int kk = 0; kk < size_j; kk++)
                for(int ll = 0; ll <= kk; ll++)
                {
                    int eij=ii*size_i+jj, ekl=kk*size_j+ll, eil=ii*size_j+ll, ekj=kk*size_i+jj;
                    if(kk == ll)
                        // solve_A(tmp_int_i+ii*(ii+1)/2+jj, tmp_int_j+kk*(kk+1)/2+ll) = h2eLLLL_JK.J[ir][jr](eij,ekl);
                        solve_A(tmp_int_i+ii*(ii+1)/2+jj, tmp_int_j+kk*(kk+1)/2+ll) = h2eLLLL_JK.J[ir][jr][eij][ekl];
                    else
                    {
                        int elk=ll*size_j+kk, eik=ii*size_j+kk, elj=ll*size_i+jj;
                        // solve_A(tmp_int_i+ii*(ii+1)/2+jj, tmp_int_j+kk*(kk+1)/2+ll) = h2eLLLL_JK.J[ir][jr](eij,ekl) + h2eLLLL_JK.J[ir][jr](eij,elk);
                        solve_A(tmp_int_i+ii*(ii+1)/2+jj, tmp_int_j+kk*(kk+1)/2+ll) = h2eLLLL_JK.J[ir][jr][eij][ekl] + h2eLLLL_JK.J[ir][jr][eij][elk];
                    }
                }
                tmp_int_j += size_j*(size_j+1)/2;
            }
        }
        tmp_int_i += size_i*(size_i+1)/2;
    }
    // cout << solve_A.inverse()*solve_A << endl;
    // solve_X = solve_A.inverse()*solve_B;
    solve_X = solve_A.householderQr().solve(solve_B);
    double relative_error = (solve_A*solve_X - solve_B).norm() / solve_B.norm(); // norm() is L2 norm
    if(relative_error > 1e-10 || isnanf(relative_error))
    {
        cout << "The relative error in solving Density:\n" << relative_error << endl;
    }

    tmp_int_i = 0;
    for(int ir = 0; ir < occMax_irrep; ir ++)
    {
        for(int ii = 0; ii < irrep_list(ir).size; ii++)
        for(int jj = 0; jj <= ii; jj++)
        {
            density_pcc(ir)(ii,jj) = solve_X(tmp_int_i + ii*(ii+1)/2+jj);
            density_pcc(ir)(jj,ii) = solve_X(tmp_int_i + ii*(ii+1)/2+jj);
        }
        tmp_int_i += irrep_list(ir).size*(irrep_list(ir).size+1)/2;
    }
    

    return density_pcc;
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
