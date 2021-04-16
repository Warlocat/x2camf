#include"dhf_sph.h"
#include<iostream>
#include<omp.h>
#include<vector>
#include<ctime>
#include<iostream>
#include<iomanip>
#include<fstream>

using namespace std;
using namespace Eigen;

clock_t StartTime, EndTime; 

DHF_SPH::DHF_SPH(INT_SPH& int_sph_, const string& filename, const bool& spinFree, const bool& sfx2c):
irrep_list(int_sph_.irrep_list)
{
    cout << "Initializing Dirac-HF for " << int_sph_.atomName << " atom." << endl;
    Nirrep = int_sph_.irrep_list.rows();
    size_basis_spinor = int_sph_.size_gtou_spinor;

    occNumber.resize(Nirrep);
    occMax_irrep = 0;
    readOCC(filename);
    nelec = 0.0;
    cout << "Occupation number vector:" << endl;
    cout << "l\t2j\t2mj\tOcc" << endl;
    for(int ii = 0; ii < Nirrep; ii++)
    {
        cout << irrep_list(ii).l << "\t" << irrep_list(ii).two_j << "\t" << irrep_list(ii).two_mj << "\t" << occNumber(ii).transpose() << endl;
        for(int jj = 0; jj < occNumber(ii).rows(); jj++)
            nelec += occNumber(ii)(jj);
    }
    cout << "Highest occupied irrep: " << occMax_irrep << endl;
    cout << "Total number of electrons: " << nelec << endl << endl;

    StartTime = clock();
    overlap = int_sph_.get_h1e("overlap");
    kinetic = int_sph_.get_h1e("kinetic");
    Vnuc = int_sph_.get_h1e("nuc_attra");
    if(spinFree || sfx2c)
        WWW = int_sph_.get_h1e("s_p_nuc_s_p_sf");
    else
        WWW = int_sph_.get_h1e("s_p_nuc_s_p");
    EndTime = clock();
    cout << "1e-integral finished in " << (EndTime - StartTime) / (double)CLOCKS_PER_SEC << " seconds." << endl;

    StartTime = clock();
    if(sfx2c)
        h2eLLLL_JK = int_sph_.get_h2e_JK("LLLL",irrep_list(occMax_irrep-1).l);
    else
        int_sph_.get_h2e_JK_direct(h2eLLLL_JK,h2eSSLL_JK,h2eSSSS_JK,irrep_list(occMax_irrep-1).l, spinFree);
    EndTime = clock();
    cout << "2e-integral finished in " << (EndTime - StartTime) / (double)CLOCKS_PER_SEC << " seconds." << endl << endl; 
        
    fock_4c.resize(occMax_irrep);
    h1e_4c.resize(occMax_irrep);
    overlap_4c.resize(occMax_irrep);
    overlap_half_i_4c.resize(occMax_irrep);
    density.resize(occMax_irrep);
    coeff.resize(occMax_irrep);
    ene_orb.resize(occMax_irrep);
    x2cXXX.resize(Nirrep);
    x2cRRR.resize(Nirrep);
    if(!sfx2c)
    {
        /*
            overlap_4c = [[S, 0], [0, T/2c^2]]
            h1e_4c = [[V, T], [T, W/4c^2 - T]]
        */
        for(int ii = 0; ii < occMax_irrep; ii++)
        {
            int size_tmp = irrep_list(ii).size;
            fock_4c(ii).resize(size_tmp*2,size_tmp*2);
            h1e_4c(ii).resize(size_tmp*2,size_tmp*2);
            overlap_4c(ii).resize(size_tmp*2,size_tmp*2);
            for(int mm = 0; mm < size_tmp; mm++)
            for(int nn = 0; nn < size_tmp; nn++)
            {
                overlap_4c(ii)(mm,nn) = overlap(ii)(mm,nn);
                overlap_4c(ii)(size_tmp+mm,nn) = 0.0;
                overlap_4c(ii)(mm,size_tmp+nn) = 0.0;
                overlap_4c(ii)(size_tmp+mm,size_tmp+nn) = kinetic(ii)(mm,nn) / 2.0 / speedOfLight / speedOfLight;
                h1e_4c(ii)(mm,nn) = Vnuc(ii)(mm,nn);
                h1e_4c(ii)(size_tmp+mm,nn) = kinetic(ii)(mm,nn);
                h1e_4c(ii)(mm,size_tmp+nn) = kinetic(ii)(mm,nn);
                h1e_4c(ii)(size_tmp+mm,size_tmp+nn) = WWW(ii)(mm,nn)/4.0/speedOfLight/speedOfLight - kinetic(ii)(mm,nn);
            }
            overlap_half_i_4c(ii) = matrix_half_inverse(overlap_4c(ii));
        }   
    }
    else
    {
        /* 
            In SFX2C-1e, h1e_4c, fock_4c, overlap_4c, and overlap_half_i_4c are the corresponding 2-c matrices. 
            spin free 4-c 1-e Hamiltonian is diagonalized to calculate X and R
            
            h1e_4c = [[V, T], [T, W_sf/4c^2 - T]]
        */
        for(int ir = 0; ir < occMax_irrep; ir++)
        {
            fock_4c(ir).resize(irrep_list(ir).size,irrep_list(ir).size);
            x2cXXX(ir) = X2C::get_X(overlap(ir),kinetic(ir),WWW(ir),Vnuc(ir));
            x2cRRR(ir) = X2C::get_R(overlap(ir),kinetic(ir),x2cXXX(ir));
            h1e_4c(ir) = X2C::evaluate_h1e_x2c(overlap(ir),kinetic(ir),WWW(ir),Vnuc(ir),x2cXXX(ir),x2cRRR(ir));
            overlap_4c(ir) = overlap(ir);
            overlap_half_i_4c(ir) = matrix_half_inverse(overlap_4c(ir));
        }   
    }
}


DHF_SPH::~DHF_SPH()
{
}


/*
    The generalized eigen solver
*/
void DHF_SPH::eigensolverG_irrep(const vMatrixXd& inputM, const vMatrixXd& s_h_i, vVectorXd& values, vMatrixXd& vectors)
{
    for(int ii = 0; ii < occMax_irrep; ii++)
        eigensolverG(inputM(ii),s_h_i(ii),values(ii),vectors(ii));
    return;
}

/*
    Evaluate the difference between two vMatrixXd
*/
double DHF_SPH::evaluateChange_irrep(const vMatrixXd& M1, const vMatrixXd& M2)
{
    VectorXd vecd_tmp(occMax_irrep);
    for(int ii = 0; ii < occMax_irrep; ii++)
    {
        vecd_tmp(ii) = evaluateChange(M1(ii),M2(ii));
    }
    return vecd_tmp.maxCoeff();
}

/*
    Evaluate error matrix in DIIS
*/
MatrixXd DHF_SPH::evaluateErrorDIIS(const MatrixXd& fock_, const MatrixXd& overlap_, const MatrixXd& density_)
{
    MatrixXd tmp = fock_*density_*overlap_ - overlap_*density_*fock_;
    int size = fock_.rows();
    MatrixXd err(size*size,1);
    for(int ii = 0; ii < size; ii++)
    for(int jj = 0; jj < size; jj++)
    {
        err(ii*size+jj,0) = tmp(ii,jj);
    }
    return err;
}

/*
    SCF procedure for 4-c and 2-c calculation
*/
void DHF_SPH::runSCF()
{
    vector<MatrixXd> error4DIIS[occMax_irrep], fock4DIIS[occMax_irrep];
    StartTime = clock();
    cout << endl;
    cout << "Start Dirac Hartree-Fock iterations..." << endl;
    cout << endl;
    vMatrixXd newDen;
    eigensolverG_irrep(h1e_4c, overlap_half_i_4c, ene_orb, coeff);
    density = evaluateDensity_spinor_irrep();

    for(int iter = 1; iter <= maxIter; iter++)
    {
        if(iter <= 2)
        {
            for(int ir = 0; ir < occMax_irrep; ir++)    
            {
                int size_tmp = irrep_list(ir).size;
                #pragma omp parallel  for
                for(int mm = 0; mm < size_tmp; mm++)
                for(int nn = 0; nn <= mm; nn++)
                {
                    fock_4c(ir)(mm,nn) = h1e_4c(ir)(mm,nn);
                    fock_4c(ir)(mm+size_tmp,nn) = h1e_4c(ir)(mm+size_tmp,nn);
                    if(mm != nn) fock_4c(ir)(nn+size_tmp,mm) = h1e_4c(ir)(nn+size_tmp,mm);
                    fock_4c(ir)(mm+size_tmp,nn+size_tmp) = h1e_4c(ir)(mm+size_tmp,nn+size_tmp);
                    for(int jr = 0; jr < occMax_irrep; jr++)
                    {
                        int size_tmp2 = irrep_list(jr).size;
                        for(int ss = 0; ss < size_tmp2; ss++)
                        for(int rr = 0; rr < size_tmp2; rr++)
                        {
                            int emn = mm*size_tmp+nn, esr = ss*size_tmp2+rr, emr = mm*size_tmp2+rr, esn = ss*size_tmp+nn;
                            fock_4c(ir)(mm,nn) += density(jr)(ss,rr) * (h2eLLLL_JK.J(ir,jr)(emn,esr) - h2eLLLL_JK.K(ir,jr)(emr,esn)) + density(jr)(size_tmp2+ss,size_tmp2+rr) * h2eSSLL_JK.J(jr,ir)(esr,emn);
                            fock_4c(ir)(mm+size_tmp,nn) -= density(jr)(ss,size_tmp2+rr) * h2eSSLL_JK.K(ir,jr)(emr,esn);
                            if(mm != nn) 
                            {
                                int enr = nn*size_tmp2+rr, esm = ss*size_tmp+mm;
                                fock_4c(ir)(nn+size_tmp,mm) -= density(jr)(ss,size_tmp2+rr) * h2eSSLL_JK.K(ir,jr)(enr,esm);
                            }
                            fock_4c(ir)(mm+size_tmp,nn+size_tmp) += density(jr)(size_tmp2+ss,size_tmp2+rr) * (h2eSSSS_JK.J(ir,jr)(emn,esr) - h2eSSSS_JK.K(ir,jr)(emr,esn)) + density(jr)(ss,rr) * h2eSSLL_JK.J(ir,jr)(emn,esr);
                        }
                    }

                    fock_4c(ir)(nn,mm) = fock_4c(ir)(mm,nn);
                    fock_4c(ir)(nn+size_tmp,mm+size_tmp) = fock_4c(ir)(mm+size_tmp,nn+size_tmp);
                    fock_4c(ir)(nn,mm+size_tmp) = fock_4c(ir)(mm+size_tmp,nn);
                    fock_4c(ir)(mm,nn+size_tmp) = fock_4c(ir)(nn+size_tmp,mm);
                }
            }
        }
        else
        {
            int tmp_size = fock4DIIS[0].size();
            MatrixXd B4DIIS(tmp_size+1,tmp_size+1);
            VectorXd vec_b(tmp_size+1);    
            for(int ii = 0; ii < tmp_size; ii++)
            {    
                for(int jj = 0; jj <= ii; jj++)
                {
                    B4DIIS(ii,jj) = 0.0;
                    for(int ir = 0; ir < occMax_irrep; ir++)
                        B4DIIS(ii,jj) += (error4DIIS[ir][ii].adjoint()*error4DIIS[ir][jj])(0,0);
                    B4DIIS(jj,ii) = B4DIIS(ii,jj);
                }
                B4DIIS(tmp_size, ii) = -1.0;
                B4DIIS(ii, tmp_size) = -1.0;
                vec_b(ii) = 0.0;
            }
            B4DIIS(tmp_size, tmp_size) = 0.0;
            vec_b(tmp_size) = -1.0;
            VectorXd C = B4DIIS.partialPivLu().solve(vec_b);
            for(int ir = 0; ir < occMax_irrep; ir++)
            {
                fock_4c(ir) = MatrixXd::Zero(fock_4c(ir).rows(),fock_4c(ir).cols());
                for(int ii = 0; ii < tmp_size; ii++)
                {
                    fock_4c(ir) += C(ii) * fock4DIIS[ir][ii];
                }
            }
        }
        eigensolverG_irrep(fock_4c, overlap_half_i_4c, ene_orb, coeff);
        newDen = evaluateDensity_spinor_irrep();
        d_density = evaluateChange_irrep(density, newDen);
        
        cout << "Iter #" << iter << " maximum density difference: " << d_density << endl;
        
        density = newDen;
        if(d_density < convControl) 
        {
            converged = true;
            cout << endl << "DHF converges after " << iter << " iterations." << endl << endl;

            cout << "\tOrbital\t\tEnergy(in hartree)\n";
            cout << "\t*******\t\t******************\n";
            for(int ir = 0; ir < occMax_irrep; ir++)
            for(int ii = 1; ii <= irrep_list(ir).size; ii++)
            {
                cout << "\t" << ii << "\t\t" << setprecision(15) << ene_orb(ir)(irrep_list(ir).size + ii - 1) << endl;
            }

            ene_scf = 0.0;
            for(int ir = 0; ir < occMax_irrep; ir++)
            {
                int size_tmp = irrep_list(ir).size;
                for(int ii = 0; ii < size_tmp * 2; ii++)
                for(int jj = 0; jj < size_tmp * 2; jj++)
                {
                    ene_scf += 0.5 * density(ir)(ii,jj) * (h1e_4c(ir)(jj,ii) + fock_4c(ir)(jj,ii));
                }
            }
            cout << "Final DHF energy is " << setprecision(15) << ene_scf << " hartree." << endl;
            break;            
        }
        for(int ir = 0; ir < occMax_irrep; ir++)    
        {
            int size_tmp = irrep_list(ir).size;
            #pragma omp parallel  for
            for(int mm = 0; mm < size_tmp; mm++)
            for(int nn = 0; nn <= mm; nn++)
            {
                fock_4c(ir)(mm,nn) = h1e_4c(ir)(mm,nn);
                fock_4c(ir)(mm+size_tmp,nn) = h1e_4c(ir)(mm+size_tmp,nn);
                if(mm != nn) fock_4c(ir)(nn+size_tmp,mm) = h1e_4c(ir)(nn+size_tmp,mm);
                fock_4c(ir)(mm+size_tmp,nn+size_tmp) = h1e_4c(ir)(mm+size_tmp,nn+size_tmp);
                for(int jr = 0; jr < occMax_irrep; jr++)
                {
                    int size_tmp2 = irrep_list(jr).size;
                    for(int ss = 0; ss < size_tmp2; ss++)
                    for(int rr = 0; rr < size_tmp2; rr++)
                    {
                        int emn = mm*size_tmp+nn, esr = ss*size_tmp2+rr, emr = mm*size_tmp2+rr, esn = ss*size_tmp+nn;
                        fock_4c(ir)(mm,nn) += density(jr)(ss,rr) * (h2eLLLL_JK.J(ir,jr)(emn,esr) - h2eLLLL_JK.K(ir,jr)(emr,esn)) + density(jr)(size_tmp2+ss,size_tmp2+rr) * h2eSSLL_JK.J(jr,ir)(esr,emn);
                        fock_4c(ir)(mm+size_tmp,nn) -= density(jr)(ss,size_tmp2+rr) * h2eSSLL_JK.K(ir,jr)(emr,esn);
                        if(mm != nn) 
                        {
                            int enr = nn*size_tmp2+rr, esm = ss*size_tmp+mm;
                            fock_4c(ir)(nn+size_tmp,mm) -= density(jr)(ss,size_tmp2+rr) * h2eSSLL_JK.K(ir,jr)(enr,esm);
                        }
                        fock_4c(ir)(mm+size_tmp,nn+size_tmp) += density(jr)(size_tmp2+ss,size_tmp2+rr) * (h2eSSSS_JK.J(ir,jr)(emn,esr) - h2eSSSS_JK.K(ir,jr)(emr,esn)) + density(jr)(ss,rr) * h2eSSLL_JK.J(ir,jr)(emn,esr);
                    }
                }

                fock_4c(ir)(nn,mm) = fock_4c(ir)(mm,nn);
                fock_4c(ir)(nn+size_tmp,mm+size_tmp) = fock_4c(ir)(mm+size_tmp,nn+size_tmp);
                fock_4c(ir)(nn,mm+size_tmp) = fock_4c(ir)(mm+size_tmp,nn);
                fock_4c(ir)(mm,nn+size_tmp) = fock_4c(ir)(nn+size_tmp,mm);
            }
            if(error4DIIS[ir].size() >= size_DIIS)
            {
                error4DIIS[ir].erase(error4DIIS[ir].begin());
                error4DIIS[ir].push_back(evaluateErrorDIIS(fock_4c(ir),overlap_4c(ir),density(ir)));
                fock4DIIS[ir].erase(fock4DIIS[ir].begin());
                fock4DIIS[ir].push_back(fock_4c(ir));
            }
            else
            {
                error4DIIS[ir].push_back(evaluateErrorDIIS(fock_4c(ir),overlap_4c(ir),density(ir)));
                fock4DIIS[ir].push_back(fock_4c(ir));
            }
        }
    }
    EndTime = clock();
    cout << "DHF iterations finished in " << (EndTime - StartTime) / (double)CLOCKS_PER_SEC << " seconds." << endl << endl;
}

void DHF_SPH::runSCF_2c()
{
    vector<MatrixXd> error4DIIS[occMax_irrep], fock4DIIS[occMax_irrep];
    StartTime = clock();
    cout << endl;
    cout << "Start 2-c Hartree-Fock iterations..." << endl;
    cout << endl;
    vMatrixXd newDen;
    eigensolverG_irrep(h1e_4c, overlap_half_i_4c, ene_orb, coeff);
    density = evaluateDensity_spinor_irrep(true);

    for(int iter = 1; iter <= maxIter; iter++)
    {
        if(iter <= 2)
        {
            for(int ir = 0; ir < occMax_irrep; ir++)    
            {
                int size_tmp = irrep_list(ir).size;
                #pragma omp parallel  for
                for(int mm = 0; mm < size_tmp; mm++)
                for(int nn = 0; nn <= mm; nn++)
                {
                    fock_4c(ir)(mm,nn) = h1e_4c(ir)(mm,nn);
                    for(int jr = 0; jr < occMax_irrep; jr++)
                    {
                        int size_tmp2 = irrep_list(jr).size;
                        for(int ss = 0; ss < size_tmp2; ss++)
                        for(int rr = 0; rr < size_tmp2; rr++)
                        {
                            int emn = mm*size_tmp+nn, esr = ss*size_tmp2+rr, emr = mm*size_tmp2+rr, esn = ss*size_tmp+nn;
                            fock_4c(ir)(mm,nn) += density(jr)(ss,rr) * (h2eLLLL_JK.J(ir,jr)(emn,esr) - h2eLLLL_JK.K(ir,jr)(emr,esn));
                        }
                    }
                    fock_4c(ir)(nn,mm) = fock_4c(ir)(mm,nn);
                }
            }
        }
        else
        {
            int tmp_size = fock4DIIS[0].size();
            MatrixXd B4DIIS(tmp_size+1,tmp_size+1);
            VectorXd vec_b(tmp_size+1);    
            for(int ii = 0; ii < tmp_size; ii++)
            {    
                for(int jj = 0; jj <= ii; jj++)
                {
                    B4DIIS(ii,jj) = 0.0;
                    for(int ir = 0; ir < occMax_irrep; ir++)
                        B4DIIS(ii,jj) += (error4DIIS[ir][ii].adjoint()*error4DIIS[ir][jj])(0,0);
                    B4DIIS(jj,ii) = B4DIIS(ii,jj);
                }
                B4DIIS(tmp_size, ii) = -1.0;
                B4DIIS(ii, tmp_size) = -1.0;
                vec_b(ii) = 0.0;
            }
            B4DIIS(tmp_size, tmp_size) = 0.0;
            vec_b(tmp_size) = -1.0;
            VectorXd C = B4DIIS.partialPivLu().solve(vec_b);
            for(int ir = 0; ir < occMax_irrep; ir++)
            {
                fock_4c(ir) = MatrixXd::Zero(fock_4c(ir).rows(),fock_4c(ir).cols());
                for(int ii = 0; ii < tmp_size; ii++)
                {
                    fock_4c(ir) += C(ii) * fock4DIIS[ir][ii];
                }
            }
        }
        eigensolverG_irrep(fock_4c, overlap_half_i_4c, ene_orb, coeff);
        newDen = evaluateDensity_spinor_irrep(true);
        d_density = evaluateChange_irrep(density, newDen);
        
        cout << "Iter #" << iter << " maximum density difference: " << d_density << endl;
        
        density = newDen;
        if(d_density < convControl) 
        {
            converged = true;
            cout << endl << "DHF converges after " << iter << " iterations." << endl << endl;

            cout << "\tOrbital\t\tEnergy(in hartree)\n";
            cout << "\t*******\t\t******************\n";
            for(int ir = 0; ir < occMax_irrep; ir++)
            for(int ii = 1; ii <= irrep_list(ir).size; ii++)
            {
                cout << "\t" << ii << "\t\t" << setprecision(15) << ene_orb(ir)(ii - 1) << endl;
            }

            ene_scf = 0.0;
            for(int ir = 0; ir < occMax_irrep; ir++)
            {
                int size_tmp = irrep_list(ir).size;
                for(int ii = 0; ii < size_tmp; ii++)
                for(int jj = 0; jj < size_tmp; jj++)
                {
                    ene_scf += 0.5 * density(ir)(ii,jj) * (h1e_4c(ir)(jj,ii) + fock_4c(ir)(jj,ii));
                }
            }
            cout << "Final sfx2c1e-HF energy is " << setprecision(15) << ene_scf << " hartree." << endl;
            break;            
        }
        for(int ir = 0; ir < occMax_irrep; ir++)    
        {
            int size_tmp = irrep_list(ir).size;
            #pragma omp parallel  for
            for(int mm = 0; mm < size_tmp; mm++)
            for(int nn = 0; nn <= mm; nn++)
            {
                fock_4c(ir)(mm,nn) = h1e_4c(ir)(mm,nn);
                for(int jr = 0; jr < occMax_irrep; jr++)
                {
                    int size_tmp2 = irrep_list(jr).size;
                    for(int ss = 0; ss < size_tmp2; ss++)
                    for(int rr = 0; rr < size_tmp2; rr++)
                    {
                        int emn = mm*size_tmp+nn, esr = ss*size_tmp2+rr, emr = mm*size_tmp2+rr, esn = ss*size_tmp+nn;
                        fock_4c(ir)(mm,nn) += density(jr)(ss,rr) * (h2eLLLL_JK.J(ir,jr)(emn,esr) - h2eLLLL_JK.K(ir,jr)(emr,esn));
                    }
                }
                fock_4c(ir)(nn,mm) = fock_4c(ir)(mm,nn);
            }
            if(error4DIIS[ir].size() >= size_DIIS)
            {
                error4DIIS[ir].erase(error4DIIS[ir].begin());
                error4DIIS[ir].push_back(evaluateErrorDIIS(fock_4c(ir),overlap_4c(ir),density(ir)));
                fock4DIIS[ir].erase(fock4DIIS[ir].begin());
                fock4DIIS[ir].push_back(fock_4c(ir));
            }
            else
            {
                error4DIIS[ir].push_back(evaluateErrorDIIS(fock_4c(ir),overlap_4c(ir),density(ir)));
                fock4DIIS[ir].push_back(fock_4c(ir));
            }
        }
    }
    EndTime = clock();
    cout << "DHF iterations finished in " << (EndTime - StartTime) / (double)CLOCKS_PER_SEC << " seconds." << endl << endl;
}

/*
    Renormalize small component to enhance the stability
*/
void DHF_SPH::renormalize_small()
{
    norm_s.resize(occMax_irrep);
    for(int ii = 0; ii < occMax_irrep; ii++)
    {
        norm_s(ii).resize(irrep_list(ii).size);
        for(int jj = 0; jj < irrep_list(ii).size; jj++)
        {
            norm_s(ii)(jj) = sqrt(kinetic(ii)(jj,jj) / 2.0 / speedOfLight / speedOfLight);
        }
    }
    for(int ii = 0; ii < occMax_irrep; ii++)
    {
        int size_tmp = irrep_list(ii).size;
        for(int mm = 0; mm < size_tmp; mm++)
        for(int nn = 0; nn < size_tmp; nn++)
        {
            overlap_4c(ii)(size_tmp+mm,size_tmp+nn) /= norm_s(ii)(mm) * norm_s(ii)(nn);
            h1e_4c(ii)(size_tmp+mm,nn) /= norm_s(ii)(mm);
            h1e_4c(ii)(mm,size_tmp+nn) /= norm_s(ii)(nn);
            h1e_4c(ii)(size_tmp+mm,size_tmp+nn) /= norm_s(ii)(mm) * norm_s(ii)(nn);
        }
        overlap_half_i_4c(ii) = matrix_half_inverse(overlap_4c(ii));
    }
    for(int ir = 0; ir < occMax_irrep; ir++)
    for(int jr = 0; jr < occMax_irrep; jr++)
    {
        int sizei = irrep_list(ir).size, sizej = irrep_list(jr).size;
        for(int ii = 0; ii < sizei*sizei; ii++)
        for(int jj = 0; jj < sizej*sizej; jj++)
        {
            int a = ii / sizei, b = ii - a * sizei, c = jj / sizej, d = jj - c * sizej;
            h2eSSLL_JK.J(ir,jr)(ii,jj) /= norm_s(ir)(a) * norm_s(ir)(b);
            h2eSSSS_JK.J(ir,jr)(ii,jj) /= norm_s(ir)(a) * norm_s(ir)(b) * norm_s(jr)(c) * norm_s(jr)(d);
        }
        for(int ii = 0; ii < sizei*sizej; ii++)
        for(int jj = 0; jj < sizej*sizei; jj++)
        {
            int a = ii / sizej, b = ii - a * sizej, c = jj / sizei, d = jj - c * sizei;
            h2eSSLL_JK.K(ir,jr)(ii,jj) /= norm_s(ir)(a) * norm_s(jr)(b);
            h2eSSSS_JK.K(ir,jr)(ii,jj) /= norm_s(ir)(a) * norm_s(jr)(b) * norm_s(jr)(c) * norm_s(ir)(d);
        }
    }

    renormalizedSmall = true;
}


/*
    Evaluate density matrix
*/
MatrixXd DHF_SPH::evaluateDensity_spinor(const MatrixXd& coeff_, const VectorXd& occNumber_, const bool& twoC)
{
    if(!twoC)
    {
        int size = coeff_.cols()/2;
        MatrixXd den(2*size,2*size);
        den = MatrixXd::Zero(2*size,2*size);        
        for(int aa = 0; aa < size; aa++)
        for(int bb = 0; bb < size; bb++)
        {
            for(int ii = 0; ii < occNumber_.rows(); ii++)
            {
                den(aa,bb) += occNumber_(ii) * coeff_(aa,ii+size) * coeff_(bb,ii+size);
                den(size+aa,bb) += occNumber_(ii) * coeff_(size+aa,ii+size) * coeff_(bb,ii+size);
                den(aa,size+bb) += occNumber_(ii) * coeff_(aa,ii+size) * coeff_(size+bb,ii+size);
                den(size+aa,size+bb) += occNumber_(ii) * coeff_(size+aa,ii+size) * coeff_(size+bb,ii+size);
            }
        }
        return den;
    }
    else
    {
        int size = coeff_.cols();
        MatrixXd den(size,size);
        den = MatrixXd::Zero(size,size);        
        for(int aa = 0; aa < size; aa++)
        for(int bb = 0; bb < size; bb++)
        for(int ii = 0; ii < occNumber_.rows(); ii++)
            den(aa,bb) += occNumber_(ii) * coeff_(aa,ii) * coeff_(bb,ii);
        
        return den;
    }
}

vMatrixXd DHF_SPH::evaluateDensity_spinor_irrep(const bool& twoC)
{
    vMatrixXd den(occMax_irrep);
    for(int ir = 0; ir < occMax_irrep; ir++)
    {
        den(ir) = evaluateDensity_spinor(coeff(ir),occNumber(ir), twoC);
    }

    return den;
}


/* 
    Read occupation numbers 
*/
void DHF_SPH::readOCC(const string& filename)
{
    string flags;
    VectorXi veci_tmp;
    int int_tmp;
    ifstream ifs;
    ifs.open(filename);
        while(!ifs.eof())
        {
            ifs >> flags;
            if(flags == "%occAMFI")
            {
                ifs >> int_tmp;
                veci_tmp.resize(int_tmp);
                int int_tmp2 = 0;
                for(int ii = 0; ii < int_tmp; ii++)
                {
                    ifs >> veci_tmp(ii);
                    int int_tmp3 = veci_tmp(ii) / (irrep_list(int_tmp2).two_j + 1);
                    double d_tmp = (double)(veci_tmp(ii) - int_tmp3*(irrep_list(int_tmp2).two_j + 1)) / (double)(irrep_list(int_tmp2).two_j + 1);
                    for(int jj = 0; jj < irrep_list(int_tmp2).two_j + 1; jj++)
                    {
                        occNumber(int_tmp2+jj).resize(irrep_list(int_tmp2+jj).size);
                        occNumber(int_tmp2+jj) = VectorXd::Zero(irrep_list(int_tmp2+jj).size);
                        for(int kk = 0; kk < int_tmp3; kk++)
                            occNumber(int_tmp2+jj)(kk) = 1.0;
                        if(occNumber(int_tmp2+jj).rows() > int_tmp3)
                            occNumber(int_tmp2+jj)(int_tmp3) = d_tmp;
                    }
                    occMax_irrep += irrep_list(int_tmp2).two_j+1;
                    int_tmp2 += irrep_list(int_tmp2).two_j+1;
                }
                break;
            }
        }
        if(ifs.eof())
        {
            cout << "ERROR: did NOT find %occAMFI in " << filename << endl;
            exit(99);
        }
    ifs.close();

    return;
}


/* 
    Evaluate amfi SOC integrals in j-adapted spinor basis
    Xmethod defines the algorithm to calculate X matrix in x2c transformation
                        Occupied shells     Virtual shells
        h1e:            h1e                 h1e
        partialFock:    fock                h1e
        fullFock:       Fock                Fock
    When necessary, the program will recalculate two-electron integrals.
*/
vMatrixXd DHF_SPH::get_amfi_unc(INT_SPH& int_sph_, const string& Xmethod)
{
    int2eJK SSLL_SD, SSSS_SD;
    int_sph_.get_h2eSD_JK_direct(SSLL_SD, SSSS_SD);
    if(occMax_irrep < Nirrep && Xmethod == "fullFock")
    {
        cout << "fullFock is used in amfi function with incomplete h2e." << endl;
        cout << "Recalculate h2e..." << endl;
        StartTime = clock();
        int_sph_.get_h2e_JK_direct(h2eLLLL_JK,h2eSSLL_JK,h2eSSSS_JK);
        EndTime = clock();
        cout << "Complete 2e-integral finished in " << (EndTime - StartTime) / (double)CLOCKS_PER_SEC << " seconds." << endl << endl; 
    }
    return get_amfi_unc(SSLL_SD, SSSS_SD, Xmethod);
}

vMatrixXd DHF_SPH::get_amfi_unc(const int2eJK& h2eSSLL_SD, const int2eJK& h2eSSSS_SD, const string& Xmethod)
{
    if(!converged)
    {
        cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
        cout << "!!  WARNING: Dirac HF did NOT converge  !!" << endl;
        cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
    }
    if(renormalizedSmall)
    {
        cout << "ERROR: AMFI integrals cannot be calculated with renormalizedSmall." << endl;
        exit(99);
    }
    vMatrixXd amfi_unc(Nirrep), h1e_4c_full(Nirrep), overlap_4c_full(Nirrep);
    /*
        Construct h1e_4c_full and overlap_4c_full 
    */
    for(int ir = 0; ir < occMax_irrep; ir++)
    {
        h1e_4c_full(ir) = h1e_4c(ir);
        overlap_4c_full(ir) = overlap_4c(ir);
    }   
    for(int ir = occMax_irrep; ir < Nirrep; ir++)
    {
        int size_tmp = irrep_list(ir).size;
        h1e_4c_full(ir).resize(size_tmp*2,size_tmp*2);
        overlap_4c_full(ir).resize(size_tmp*2,size_tmp*2);
        for(int mm = 0; mm < size_tmp; mm++)
        for(int nn = 0; nn < size_tmp; nn++)
        {
            overlap_4c_full(ir)(mm,nn) = overlap(ir)(mm,nn);
            overlap_4c_full(ir)(size_tmp+mm,nn) = 0.0;
            overlap_4c_full(ir)(mm,size_tmp+nn) = 0.0;
            overlap_4c_full(ir)(size_tmp+mm,size_tmp+nn) = kinetic(ir)(mm,nn) / 2.0 / speedOfLight / speedOfLight;
            h1e_4c_full(ir)(mm,nn) = Vnuc(ir)(mm,nn);
            h1e_4c_full(ir)(size_tmp+mm,nn) = kinetic(ir)(mm,nn);
            h1e_4c_full(ir)(mm,size_tmp+nn) = kinetic(ir)(mm,nn);
            h1e_4c_full(ir)(size_tmp+mm,size_tmp+nn) = WWW(ir)(mm,nn)/4.0/speedOfLight/speedOfLight - kinetic(ir)(mm,nn);
        }
    }
    
    for(int ir = 0; ir < Nirrep; ir++)
    {
        int size_tmp = irrep_list(ir).size;
        MatrixXd SO_4c(2*size_tmp,2*size_tmp);
        /* 
            Evaluate SO integrals in 4c basis
            The structure is the same as 2e Coulomb integrals in fock matrix 
        */
        for(int mm = 0; mm < size_tmp; mm++)
        for(int nn = 0; nn <= mm; nn++)
        {
            SO_4c(mm,nn) = 0.0;
            SO_4c(mm+size_tmp,nn) = 0.0;
            if(mm != nn) SO_4c(nn+size_tmp,mm) = 0.0;
            SO_4c(mm+size_tmp,nn+size_tmp) = 0.0;
            for(int jr = 0; jr < occMax_irrep; jr++)
            {
                int size_tmp2 = irrep_list(jr).size;
                for(int ss = 0; ss < size_tmp2; ss++)
                for(int rr = 0; rr < size_tmp2; rr++)
                {
                    int emn = mm*size_tmp+nn, esr = ss*size_tmp2+rr, emr = mm*size_tmp2+rr, esn = ss*size_tmp+nn;
                    SO_4c(mm,nn) += density(jr)(size_tmp2+ss,size_tmp2+rr) * h2eSSLL_SD.J(jr,ir)(esr,emn);
                    SO_4c(mm+size_tmp,nn) -= density(jr)(ss,size_tmp2+rr) * h2eSSLL_SD.K(ir,jr)(emr,esn);
                    if(mm != nn) 
                    {
                        int enr = nn*size_tmp2+rr, esm = ss*size_tmp+mm;
                        SO_4c(nn+size_tmp,mm) -= density(jr)(ss,size_tmp2+rr) * h2eSSLL_SD.K(ir,jr)(enr,esm);
                    }
                    SO_4c(mm+size_tmp,nn+size_tmp) += density(jr)(size_tmp2+ss,size_tmp2+rr) * (h2eSSSS_SD.J(ir,jr)(emn,esr) - h2eSSSS_SD.K(ir,jr)(emr,esn)) + density(jr)(ss,rr) * h2eSSLL_SD.J(ir,jr)(emn,esr);
                }
            }
            SO_4c(nn,mm) = SO_4c(mm,nn);
            SO_4c(nn+size_tmp,mm+size_tmp) = SO_4c(mm+size_tmp,nn+size_tmp);
            SO_4c(nn,mm+size_tmp) = SO_4c(mm+size_tmp,nn);
            SO_4c(mm,nn+size_tmp) = SO_4c(nn+size_tmp,mm);
        }
        /* 
            Evaluate X with various options
        */
        if(Xmethod == "h1e")
        {
            x2cXXX(ir) = X2C::get_X(overlap(ir),kinetic(ir),WWW(ir),Vnuc(ir));
        }
        else
        {
            if(ir < occMax_irrep)
            {
                x2cXXX(ir) = X2C::get_X(coeff(ir));
            }
            else
            {
                if(Xmethod == "partialFock")
                    x2cXXX(ir) = X2C::get_X(overlap(ir),kinetic(ir),WWW(ir),Vnuc(ir));
                else if(Xmethod == "fullFock")
                {
                    MatrixXd fock_tmp(2*size_tmp,2*size_tmp), overlap_half_i_4c_tmp = matrix_half_inverse(overlap_4c_full(ir));
                    for(int mm = 0; mm < size_tmp; mm++)
                    for(int nn = 0; nn <= mm; nn++)
                    {
                        fock_tmp(mm,nn) = h1e_4c_full(ir)(mm,nn);
                        fock_tmp(mm+size_tmp,nn) = h1e_4c_full(ir)(mm+size_tmp,nn);
                        if(mm != nn) fock_tmp(nn+size_tmp,mm) = h1e_4c_full(ir)(nn+size_tmp,mm);
                        fock_tmp(mm+size_tmp,nn+size_tmp) = h1e_4c_full(ir)(mm+size_tmp,nn+size_tmp);
                        for(int jr = 0; jr < occMax_irrep; jr++)
                        {
                            int size_tmp2 = irrep_list(jr).size;
                            for(int ss = 0; ss < size_tmp2; ss++)
                            for(int rr = 0; rr < size_tmp2; rr++)
                            {
                                int emn = mm*size_tmp+nn, esr = ss*size_tmp2+rr, emr = mm*size_tmp2+rr, esn = ss*size_tmp+nn;
                                fock_tmp(mm,nn) += density(jr)(ss,rr) * (h2eLLLL_JK.J(ir,jr)(emn,esr) - h2eLLLL_JK.K(ir,jr)(emr,esn)) + density(jr)(size_tmp2+ss,size_tmp2+rr) * h2eSSLL_JK.J(jr,ir)(esr,emn);
                                fock_tmp(mm+size_tmp,nn) -= density(jr)(ss,size_tmp2+rr) * h2eSSLL_JK.K(ir,jr)(emr,esn);
                                if(mm != nn) 
                                {
                                    int enr = nn*size_tmp2+rr, esm = ss*size_tmp+mm;
                                    fock_tmp(nn+size_tmp,mm) -= density(jr)(ss,size_tmp2+rr) * h2eSSLL_JK.K(ir,jr)(enr,esm);
                                }
                                fock_tmp(mm+size_tmp,nn+size_tmp) += density(jr)(size_tmp2+ss,size_tmp2+rr) * (h2eSSSS_JK.J(ir,jr)(emn,esr) - h2eSSSS_JK.K(ir,jr)(emr,esn)) + density(jr)(ss,rr) * h2eSSLL_JK.J(ir,jr)(emn,esr);
                            }
                        }
                        fock_tmp(nn,mm) = fock_tmp(mm,nn);
                        fock_tmp(nn+size_tmp,mm+size_tmp) = fock_tmp(mm+size_tmp,nn+size_tmp);
                        fock_tmp(nn,mm+size_tmp) = fock_tmp(mm+size_tmp,nn);
                        fock_tmp(mm,nn+size_tmp) = fock_tmp(nn+size_tmp,mm);
                    }
                    MatrixXd coeff_tmp;
                    VectorXd ene_orb_tmp;
                    eigensolverG(fock_tmp,overlap_half_i_4c_tmp,ene_orb_tmp,coeff_tmp);
                    x2cXXX(ir) = X2C::get_X(coeff_tmp);
                }
                else
                {
                    cout << "ERROR: unknown Xmethod in get_amfi" << endl;
                    exit(99);
                }
            }
        }
        /* 
            Evaluate R and amfi
        */
        x2cRRR(ir) = X2C::get_R(overlap_4c_full(ir),x2cXXX(ir));
        amfi_unc(ir) = SO_4c.block(0,0,size_tmp,size_tmp) + SO_4c.block(0,size_tmp,size_tmp,size_tmp) * x2cXXX(ir) + x2cXXX(ir).transpose() * SO_4c.block(size_tmp,0,size_tmp,size_tmp) + x2cXXX(ir).transpose() * SO_4c.block(size_tmp,size_tmp,size_tmp,size_tmp) * x2cXXX(ir);
        amfi_unc(ir) = x2cRRR(ir).transpose() * amfi_unc(ir) * x2cRRR(ir);
    }

    return amfi_unc;
}

/* 
    Evaluate amfi SOC integrals in j-adapted spinor basis for two-component calculation
    X will always be calculated based on spin-free h1e.
*/
vMatrixXd DHF_SPH::get_amfi_unc_2c(INT_SPH& int_sph_)
{
    int2eJK SSLL_SD, SSSS_SD;
    int_sph_.get_h2eSD_JK_direct(SSLL_SD, SSSS_SD);

    return get_amfi_unc_2c(SSLL_SD, SSSS_SD);
}

vMatrixXd DHF_SPH::get_amfi_unc_2c(const int2eJK& h2eSSLL_SD, const int2eJK& h2eSSSS_SD)
{
    if(!converged)
    {
        cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
        cout << "!!  WARNING: 2-c HF did NOT converge  !!" << endl;
        cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
    }

    vMatrixXd amfi_unc(Nirrep), h1e_2c_full(Nirrep), overlap_2c_full(Nirrep);
    /*
        Construct h1e_2c_full and overlap_2c_full 
    */
    for(int ir = 0; ir < occMax_irrep; ir++)
    {
        h1e_2c_full(ir) = h1e_4c(ir);
        overlap_2c_full(ir) = overlap_4c(ir);
    }   
    for(int ir = occMax_irrep; ir < Nirrep; ir++)
    {
        x2cXXX(ir) = X2C::get_X(overlap(ir),kinetic(ir),WWW(ir),Vnuc(ir));
        x2cRRR(ir) = X2C::get_R(overlap(ir),kinetic(ir),x2cXXX(ir));
        h1e_2c_full(ir) = X2C::evaluate_h1e_x2c(overlap(ir),kinetic(ir),WWW(ir),Vnuc(ir),x2cXXX(ir),x2cRRR(ir));
        overlap_2c_full(ir) = overlap(ir);
    }
    /*
        Calculate 4-c density using approximate PES C_L and C_S
        C_L = R C_{2c}
        C_S = X C_L
        This is used in L. Cheng, et al, J. Chem. Phys. 141, 164107 (2014)
    */
    vMatrixXd coeff_tmp(occMax_irrep), density_tmp(occMax_irrep), coeff_L_tmp(occMax_irrep), coeff_S_tmp(occMax_irrep);
    for(int ir = 0; ir < occMax_irrep; ir++)
    {
        int size_tmp = irrep_list(ir).size;
        coeff_L_tmp(ir) = x2cRRR(ir) * coeff(ir);
        coeff_S_tmp(ir) = x2cXXX(ir) * coeff_L_tmp(ir);
        coeff_tmp(ir).resize(2*size_tmp,2*size_tmp);
        coeff_tmp(ir) = MatrixXd::Zero(2*size_tmp,2*size_tmp);
        for(int ii = 0; ii < size_tmp; ii++)
        for(int jj = 0; jj < size_tmp; jj++)
        {
            coeff_tmp(ir)(ii,size_tmp+jj) = coeff_L_tmp(ir)(ii,jj);
            coeff_tmp(ir)(size_tmp+ii,size_tmp+jj) = coeff_S_tmp(ir)(ii,jj);
        }
        density_tmp(ir) = evaluateDensity_spinor(coeff_tmp(ir),occNumber(ir),false);
    }

    for(int ir = 0; ir < Nirrep; ir++)
    {
        int size_tmp = irrep_list(ir).size;
        MatrixXd SO_4c(2*size_tmp,2*size_tmp);
        /* 
            Evaluate SO integrals in 4c basis
            The structure is the same as 2e Coulomb integrals in fock matrix 
        */
        for(int mm = 0; mm < size_tmp; mm++)
        for(int nn = 0; nn <= mm; nn++)
        {
            SO_4c(mm,nn) = 0.0;
            SO_4c(mm+size_tmp,nn) = 0.0;
            if(mm != nn) SO_4c(nn+size_tmp,mm) = 0.0;
            SO_4c(mm+size_tmp,nn+size_tmp) = 0.0;
            for(int jr = 0; jr < occMax_irrep; jr++)
            {
                int size_tmp2 = irrep_list(jr).size;
                for(int ss = 0; ss < size_tmp2; ss++)
                for(int rr = 0; rr < size_tmp2; rr++)
                {
                    int emn = mm*size_tmp+nn, esr = ss*size_tmp2+rr, emr = mm*size_tmp2+rr, esn = ss*size_tmp+nn;
                    SO_4c(mm,nn) += density_tmp(jr)(size_tmp2+ss,size_tmp2+rr) * h2eSSLL_SD.J(jr,ir)(esr,emn);
                    SO_4c(mm+size_tmp,nn) -= density_tmp(jr)(ss,size_tmp2+rr) * h2eSSLL_SD.K(ir,jr)(emr,esn);
                    if(mm != nn) 
                    {
                        int enr = nn*size_tmp2+rr, esm = ss*size_tmp+mm;
                        SO_4c(nn+size_tmp,mm) -= density_tmp(jr)(ss,size_tmp2+rr) * h2eSSLL_SD.K(ir,jr)(enr,esm);
                    }
                    SO_4c(mm+size_tmp,nn+size_tmp) += density_tmp(jr)(size_tmp2+ss,size_tmp2+rr) * (h2eSSSS_SD.J(ir,jr)(emn,esr) - h2eSSSS_SD.K(ir,jr)(emr,esn)) + density_tmp(jr)(ss,rr) * h2eSSLL_SD.J(ir,jr)(emn,esr);
                }
            }
            SO_4c(nn,mm) = SO_4c(mm,nn);
            SO_4c(nn+size_tmp,mm+size_tmp) = SO_4c(mm+size_tmp,nn+size_tmp);
            SO_4c(nn,mm+size_tmp) = SO_4c(mm+size_tmp,nn);
            SO_4c(mm,nn+size_tmp) = SO_4c(nn+size_tmp,mm);
        }
        
        amfi_unc(ir) = SO_4c.block(0,0,size_tmp,size_tmp) + SO_4c.block(0,size_tmp,size_tmp,size_tmp) * x2cXXX(ir) + x2cXXX(ir).transpose() * SO_4c.block(size_tmp,0,size_tmp,size_tmp) + x2cXXX(ir).transpose() * SO_4c.block(size_tmp,size_tmp,size_tmp,size_tmp) * x2cXXX(ir);
        amfi_unc(ir) = x2cRRR(ir).transpose() * amfi_unc(ir) * x2cRRR(ir);
    }

    return amfi_unc;
}


/*
    Put one-electron integrals in a single matrix and reorder them.
    The new ordering is to put the single uncontracted spinors with same l together.
*/
MatrixXd DHF_SPH::unite_irrep(const vMatrixXd& inputM, const Matrix<irrep_jm, Dynamic, 1>& irrep_list)
{
    int size_spinor = 0, size_irrep = irrep_list.rows(), Lmax = irrep_list(size_irrep - 1).l;
    if(inputM.rows() != size_irrep)
    {
        cout << "ERROR: the size of inputM is not equal to Nirrep." << endl;
        exit(99);
    }
    for(int ir = 0; ir < size_irrep; ir++)
    {
        size_spinor += irrep_list(ir).size;
    }
    MatrixXd outputM(size_spinor,size_spinor);
    outputM = MatrixXd::Zero(size_spinor,size_spinor);
    int i_output = 0;
    for(int ir = 0; ir < size_irrep; ir += 4*irrep_list(ir).l+2)
    {
        for(int ii = 0; ii < irrep_list(ir).size; ii++)
        for(int jj = 0; jj < irrep_list(ir).size; jj++)
        for(int mi = 0; mi < 4*irrep_list(ir).l+2; mi++)
        {
            outputM(i_output + ii*(4*irrep_list(ir).l+2) + mi, i_output + jj*(4*irrep_list(ir).l+2) + mi) = inputM(ir+mi)(ii,jj);
        }
        i_output += (4*irrep_list(ir).l+2) * irrep_list(ir).size;
    }

    return outputM;
}


