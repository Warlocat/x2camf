#include"dhf_sph_ca.h"
#include<iostream>
#include<omp.h>
#include<vector>
#include<ctime>
#include<iostream>
#include<iomanip>
#include<fstream>

using namespace std;
using namespace Eigen;

vVectorXd occNumberDebug;


DHF_SPH_CA::DHF_SPH_CA(INT_SPH& int_sph_, const string& filename, const bool& spinFree, const bool& sfx2c, const bool& with_gaunt_):
DHF_SPH(int_sph_,filename,spinFree,sfx2c,with_gaunt_)
{
    openShell = -1;
    for(int ir = 0; ir < occMax_irrep; ir++)
    {
        for(int ii = 0; ii < occNumber(ir).rows(); ii++)
        {
            // if 1 > occNumber(ir)(ii) > 0
            if(abs(occNumber(ir)(ii)) > 1e-4 && abs(occNumber(ir)(ii)) < (1.0-1e-4))
            {
                f_NM = occNumber(ir)(ii);
                openShell = ir;
            }
        }
    }

    if(openShell == -1)
    {
        NN = 0;
        f_NM = 0.0;
        MM = 0.0;
    }
    else 
    {
        MM = irrep_list(openShell).l*4+2;
        NN = f_NM * MM;
    }

    occNumberDebug.resize(occMax_irrep);
    for(int ir = 0; ir < occMax_irrep; ir++)
    {
        occNumberDebug(ir).resize(irrep_list(ir).size);
        occNumberDebug(ir) = VectorXd::Zero(irrep_list(ir).size);
        for(int ii = 0; ii < occNumber(ir).rows(); ii++)
        {
            if(abs(occNumber(ir)(ii) - 1.0) < 1e-5)
                occNumberDebug(ir)(ii) = 0.0;
            else if(abs(occNumber(ir)(ii)) > 1e-4 && abs(occNumber(ir)(ii)) < (1.0-1e-4))
            {
                occNumber(ir)(ii) = 0.0;
                occNumberDebug(ir)(ii) = 1.0;
            }
        }
    }
    for(int ir = 0; ir < occMax_irrep; ir++)
    {
        cout << "1: " << occNumber(ir).transpose() << endl;
        cout << "2: " << occNumberDebug(ir).transpose() << endl;
    }
    cout << "Configuration-averaged HF initialization." << endl;
    cout << "f = N/M = " << f_NM << ", N = " << NN << ", M = " << MM << endl;
}


DHF_SPH_CA::~DHF_SPH_CA()
{
}



/*
    Evaluate density matrix
*/
MatrixXd DHF_SPH_CA::evaluateDensity_core(const MatrixXd& coeff_, const VectorXd& occNumber_, const bool& twoC)
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
                if(abs(occNumber_(ii) - 1.0) < 1e-5)
                {    
                    den(aa,bb) += coeff_(aa,ii+size) * coeff_(bb,ii+size);
                    den(size+aa,bb) += coeff_(size+aa,ii+size) * coeff_(bb,ii+size);
                    den(aa,size+bb) += coeff_(aa,ii+size) * coeff_(size+bb,ii+size);
                    den(size+aa,size+bb) += coeff_(size+aa,ii+size) * coeff_(size+bb,ii+size);
                }
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
        {
            if(abs(occNumber_(ii) - 1.0) < 1e-5)
                den(aa,bb) += coeff_(aa,ii) * coeff_(bb,ii);
        }
        return den;
    }
}

MatrixXd DHF_SPH_CA::evaluateDensity_open(const MatrixXd& coeff_, const VectorXd& occNumber_, const bool& twoC)
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
                // if 1 > occNumber(ir)(ii) > 0
                if(abs(occNumber_(ii)) > 1e-4 && abs(occNumber_(ii)) < (1.0-1e-4))
                {    
                    den(aa,bb) += coeff_(aa,ii+size) * coeff_(bb,ii+size);
                    den(size+aa,bb) += coeff_(size+aa,ii+size) * coeff_(bb,ii+size);
                    den(aa,size+bb) += coeff_(aa,ii+size) * coeff_(size+bb,ii+size);
                    den(size+aa,size+bb) += coeff_(size+aa,ii+size) * coeff_(size+bb,ii+size);
                }
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
        {
            // if 1 > occNumber(ir)(ii) > 0
            if(abs(occNumber_(ii)) > 1e-4 && abs(occNumber_(ii)) < (1.0-1e-4))
            {
                den(aa,bb) += coeff_(aa,ii) * coeff_(bb,ii);
            }
        }
        return den;
    }
}

void DHF_SPH_CA::evaluateDensity_ca_irrep(vMatrixXd& den_c, vMatrixXd& den_o, const vMatrixXd& coeff_, const bool& twoC)
{
    den_c.resize(occMax_irrep);
    den_o.resize(occMax_irrep);
    for(int ir = 0; ir < occMax_irrep; ir++)
    {
        den_c(ir) = evaluateDensity_core(coeff_(ir),occNumber(ir), twoC);
        den_o(ir) = evaluateDensity_core(coeff_(ir),occNumberDebug(ir), twoC);
    }
}


/*
    SCF procedure for 4-c and 2-c calculation
*/
void DHF_SPH_CA::runSCF(const bool& twoC)
{
size_DIIS = 1;
    vector<MatrixXd> error4DIIS[occMax_irrep], fock4DIIS[occMax_irrep];
    StartTime = clock();
    cout << endl;
    if(twoC) cout << "Start CA-SFX2C-1e Hartree-Fock iterations..." << endl;
    else cout << "Start CA-Dirac Hartree-Fock iterations..." << endl;
    cout << endl;

    vMatrixXd newDen_c(occMax_irrep), newDen_o(occMax_irrep), oldDen_t(occMax_irrep), newDen_t(occMax_irrep);
    eigensolverG_irrep(h1e_4c, overlap_half_i_4c, ene_orb, coeff);
    evaluateDensity_ca_irrep(density, density_o, coeff, twoC);

    for(int iter = 1; iter <= maxIter; iter++)
    {
        if(iter <= 2)
        {
            for(int ir = 0; ir < occMax_irrep; ir++)    
            {
                int size_tmp = irrep_list(ir).size;
                evaluateFock(fock_4c(ir),twoC,density,density_o,size_tmp,ir);
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
        evaluateDensity_ca_irrep(newDen_c, newDen_o, coeff, twoC);

        d_density = max(evaluateChange_irrep(density, newDen_c),evaluateChange_irrep(density_o,newDen_o));
        
        cout << "Iter #" << iter << " maximum density difference: " << d_density << endl;
        
        for(int ir = 0; ir < occMax_irrep; ir++)
        {
            density(ir) = newDen_c(ir);
            density_o(ir) = newDen_o(ir);
            // density(ir) = 0.7*density(ir) + 0.3*newDen_c(ir);
            // density_o(ir) = 0.7*density_o(ir) + 0.3*newDen_o(ir);
        }
        

        if(d_density < convControl) 
        {
            converged = true;
            cout << endl << "CA-SCF converges after " << iter << " iterations." << endl << endl;

            cout << "\tOrbital\t\tEnergy(in hartree)\n";
            cout << "\t*******\t\t******************\n";
            for(int ir = 0; ir < occMax_irrep; ir++)
            for(int ii = 1; ii <= irrep_list(ir).size; ii++)
            {
                if(twoC) cout << "\t" << ii << "\t\t" << setprecision(15) << ene_orb(ir)(ii - 1) << endl;
                else cout << "\t" << ii << "\t\t" << setprecision(15) << ene_orb(ir)(irrep_list(ir).size + ii - 1) << endl;
            }

            ene_scf = 0.0;
            for(int ir = 0; ir < occMax_irrep; ir++)
            {
                int size_tmp = irrep_list(ir).size;
                if(twoC)
                {
                    for(int ii = 0; ii < size_tmp; ii++)
                    for(int jj = 0; jj < size_tmp; jj++)
                    {
                        ene_scf += (density(ir)(ii,jj) + f_NM*density_o(ir)(ii,jj)) * h1e_4c(ir)(jj,ii);
                        for(int jr = 0; jr < occMax_irrep; jr++)
                        for(int kk = 0; kk < irrep_list(jr).size; kk++)
                        for(int ll = 0; ll < irrep_list(jr).size; ll++)
                        {
                            int eij = ii*size_tmp+jj, ekl = kk*irrep_list(jr).size+ll, eil = ii*irrep_list(jr).size+ll, ekj = kk*size_tmp+jj;
                            ene_scf += (0.5*density(ir)(ii,jj)*density(jr)(kk,ll) + f_NM*density(ir)(ii,jj)*density_o(jr)(kk,ll) + 0.5*f_NM*(NN-1)/(MM-1)*density_o(ir)(ii,jj)*density_o(jr)(kk,ll)) * (h2eLLLL_JK.J(ir,jr)(eij,ekl) - h2eLLLL_JK.K(ir,jr)(eil,ekj));
                        }
                    }
                }
                else
                {
                    for(int ii = 0; ii < size_tmp * 2; ii++)
                    for(int jj = 0; jj < size_tmp * 2; jj++)
                    {
                        ene_scf += 0.5 * density(ir)(ii,jj) * (h1e_4c(ir)(jj,ii) + fock_4c(ir)(jj,ii));
                    }
                }
            }
            if(twoC) cout << "Final CA-SFX2C-1e HF energy is " << setprecision(15) << ene_scf << " hartree." << endl;
            else cout << "Final CA-DHF energy is " << setprecision(15) << ene_scf << " hartree." << endl;
            break;            
        }
        for(int ir = 0; ir < occMax_irrep; ir++)    
        {
            int size_tmp = irrep_list(ir).size;
            evaluateFock(fock_4c(ir),twoC,density,density_o,size_tmp,ir);
            eigensolverG(fock_4c(ir), overlap_half_i_4c(ir), ene_orb(ir), coeff(ir));
            newDen_c(ir) = evaluateDensity_core(coeff(ir),occNumber(ir), twoC);
            // newDen_o(ir) = evaluateDensity_open(coeff(ir),occNumber(ir), twoC);
            newDen_o(ir) = evaluateDensity_core(coeff(ir),occNumberDebug(ir),twoC);
            if(error4DIIS[ir].size() >= size_DIIS)
            {
                error4DIIS[ir].erase(error4DIIS[ir].begin());
                // error4DIIS[ir].push_back(evaluateErrorDIIS(fock_4c(ir),overlap_4c(ir),density(ir) + f_NM * density_o(ir)));
                error4DIIS[ir].push_back(newDen_c(ir)+f_NM*newDen_o(ir)-density(ir)-f_NM*density_o(ir));
                fock4DIIS[ir].erase(fock4DIIS[ir].begin());
                fock4DIIS[ir].push_back(fock_4c(ir));
            }
            else
            {
                // error4DIIS[ir].push_back(evaluateErrorDIIS(fock_4c(ir),overlap_4c(ir),density(ir) + f_NM * density_o(ir)));
                error4DIIS[ir].push_back(newDen_c(ir)+f_NM*newDen_o(ir)-density(ir)-f_NM*density_o(ir));
                fock4DIIS[ir].push_back(fock_4c(ir));
            }
        }
    }
    EndTime = clock();
    cout << "DHF iterations finished in " << (EndTime - StartTime) / (double)CLOCKS_PER_SEC << " seconds." << endl << endl;
}

void DHF_SPH_CA::runSCF_separate(const bool& twoC)
{
    vector<MatrixXd> error4DIIS_c[occMax_irrep], fock4DIIS_c[occMax_irrep], error4DIIS_o[occMax_irrep], fock4DIIS_o[occMax_irrep];
    vMatrixXd coeff_c(occMax_irrep), coeff_o(occMax_irrep), fock_c(occMax_irrep), fock_o(occMax_irrep);
    StartTime = clock();
    cout << endl;
    if(twoC) cout << "Start CA-SFX2C-1e Hartree-Fock iterations..." << endl;
    else cout << "Start CA-Dirac Hartree-Fock iterations..." << endl;
    cout << endl;

    vMatrixXd newDen_c(occMax_irrep), newDen_o(occMax_irrep), oldDen_t(occMax_irrep), newDen_t(occMax_irrep);
    eigensolverG_irrep(h1e_4c, overlap_half_i_4c, ene_orb, coeff_c);
    coeff_o = coeff_c;
    density.resize(occMax_irrep);
    density_o.resize(occMax_irrep);
    for(int ir = 0; ir < occMax_irrep; ir++)
    {
        density(ir) = evaluateDensity_core(coeff_c(ir),occNumber(ir),twoC);
        density_o(ir) = evaluateDensity_core(coeff_o(ir),occNumberDebug(ir),twoC);
    }

    for(int iter = 1; iter <= maxIter; iter++)
    {
        if(iter <= 2)
        {
            for(int ir = 0; ir < occMax_irrep; ir++)    
            {
                int size_tmp = irrep_list(ir).size;
                evaluateFock_core(fock_c(ir),twoC,density,density_o,size_tmp,ir);
                evaluateFock_open(fock_o(ir),twoC,density,density_o,size_tmp,ir);
            }
        }
        else
        {
            int tmp_size = fock4DIIS_c[0].size();
            MatrixXd B4DIIS(tmp_size+1,tmp_size+1);
            VectorXd vec_b(tmp_size+1);    
            for(int ii = 0; ii < tmp_size; ii++)
            {    
                for(int jj = 0; jj <= ii; jj++)
                {
                    B4DIIS(ii,jj) = 0.0;
                    for(int ir = 0; ir < occMax_irrep; ir++)
                        B4DIIS(ii,jj) += (error4DIIS_c[ir][ii].adjoint()*error4DIIS_c[ir][jj])(0,0);
                    for(int ir = 0; ir < occMax_irrep; ir++)
                        B4DIIS(ii,jj) += (error4DIIS_o[ir][ii].adjoint()*error4DIIS_o[ir][jj])(0,0);
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
                fock_c(ir) = MatrixXd::Zero(fock_c(ir).rows(),fock_c(ir).cols());
                fock_o(ir) = MatrixXd::Zero(fock_o(ir).rows(),fock_o(ir).cols());
                for(int ii = 0; ii < tmp_size; ii++)
                {
                    fock_c(ir) += C(ii) * fock4DIIS_c[ir][ii];
                    fock_o(ir) += C(ii) * fock4DIIS_o[ir][ii];
                }
            }
        }
        eigensolverG_irrep(fock_c, overlap_half_i_4c, ene_orb, coeff_c);
        eigensolverG_irrep(fock_o, overlap_half_i_4c, ene_orb, coeff_o);

        for(int ir = 0; ir < occMax_irrep; ir++)
        {
            newDen_c(ir) = evaluateDensity_core(coeff_c(ir),occNumber(ir),twoC);
            newDen_o(ir) = evaluateDensity_core(coeff_o(ir),occNumberDebug(ir),twoC);
        }
        d_density = max(evaluateChange_irrep(density, newDen_c),evaluateChange_irrep(density_o,newDen_o));              
        cout << "Iter #" << iter << " maximum density difference: " << d_density << endl;     
        for(int ir = 0; ir < occMax_irrep; ir++)
        {
            density(ir) = newDen_c(ir);
            density_o(ir) = newDen_o(ir);
        }

        if(d_density < convControl) 
        {
            converged = true;
            cout << endl << "CA-SCF converges after " << iter << " iterations." << endl << endl;

            cout << "\tOrbital\t\tEnergy(in hartree)\n";
            cout << "\t*******\t\t******************\n";
            for(int ir = 0; ir < occMax_irrep; ir++)
            for(int ii = 1; ii <= irrep_list(ir).size; ii++)
            {
                if(twoC) cout << "\t" << ii << "\t\t" << setprecision(15) << ene_orb(ir)(ii - 1) << endl;
                else cout << "\t" << ii << "\t\t" << setprecision(15) << ene_orb(ir)(irrep_list(ir).size + ii - 1) << endl;
            }
            
            coeff.resize(occMax_irrep);
            for(int ir = 0; ir < occMax_irrep; ir++)
            {
                coeff(ir) = coeff_c(ir);
                for(int ii = 0; ii < irrep_list(ir).size; ii++)
                {
                    if(abs(occNumberDebug(ir)(ii) - 1.0) < 1e-5)
                    {
                        for(int jj = 0; jj < coeff(ir).rows(); jj++)
                            coeff(ir)(jj,ii) = coeff_o(ir)(jj,ii);
                    }
                }
            }
            

            ene_scf = 0.0;
            for(int ir = 0; ir < occMax_irrep; ir++)
            {
                int size_tmp = irrep_list(ir).size;
                if(twoC)
                {
                    for(int ii = 0; ii < size_tmp; ii++)
                    for(int jj = 0; jj < size_tmp; jj++)
                    {
                        ene_scf += (density(ir)(ii,jj) + f_NM*density_o(ir)(ii,jj)) * h1e_4c(ir)(jj,ii);
                        for(int jr = 0; jr < occMax_irrep; jr++)
                        for(int kk = 0; kk < irrep_list(jr).size; kk++)
                        for(int ll = 0; ll < irrep_list(jr).size; ll++)
                        {
                            int eij = ii*size_tmp+jj, ekl = kk*irrep_list(jr).size+ll, eil = ii*irrep_list(jr).size+ll, ekj = kk*size_tmp+jj;
                            ene_scf += (0.5*density(ir)(ii,jj)*density(jr)(kk,ll) + f_NM*density(ir)(ii,jj)*density_o(jr)(kk,ll) + 0.5*f_NM*(NN-1)/(MM-1)*density_o(ir)(ii,jj)*density_o(jr)(kk,ll)) * (h2eLLLL_JK.J(ir,jr)(eij,ekl) - h2eLLLL_JK.K(ir,jr)(eil,ekj));
                        }
                    }
                }
                else
                {
                    for(int ii = 0; ii < size_tmp; ii++)
                    for(int jj = 0; jj < size_tmp; jj++)
                    {
                        ene_scf += (density(ir)(ii,jj) + f_NM*density_o(ir)(ii,jj)) * h1e_4c(ir)(jj,ii);
                        ene_scf += (density(ir)(ii+size_tmp,jj) + f_NM*density_o(ir)(ii+size_tmp,jj)) * h1e_4c(ir)(jj,ii+size_tmp);
                        ene_scf += (density(ir)(ii,jj+size_tmp) + f_NM*density_o(ir)(ii,jj+size_tmp)) * h1e_4c(ir)(jj+size_tmp,ii);
                        ene_scf += (density(ir)(ii+size_tmp,jj+size_tmp) + f_NM*density_o(ir)(ii+size_tmp,jj+size_tmp)) * h1e_4c(ir)(jj+size_tmp,ii+size_tmp);
                        for(int jr = 0; jr < occMax_irrep; jr++)
                        for(int kk = 0; kk < irrep_list(jr).size; kk++)
                        for(int ll = 0; ll < irrep_list(jr).size; ll++)
                        {
                            int size_tmp2 = irrep_list(jr).size;
                            int eij = ii*size_tmp+jj, ekl = kk*irrep_list(jr).size+ll, eil = ii*irrep_list(jr).size+ll, ekj = kk*size_tmp+jj;
                            //LLLL
                            ene_scf += (0.5*density(ir)(ii,jj)*density(jr)(kk,ll) + f_NM*density(ir)(ii,jj)*density_o(jr)(kk,ll) + 0.5*f_NM*(NN-1)/(MM-1)*density_o(ir)(ii,jj)*density_o(jr)(kk,ll)) * (h2eLLLL_JK.J(ir,jr)(eij,ekl) - h2eLLLL_JK.K(ir,jr)(eil,ekj));
                            //SSSS
                            ene_scf += (0.5*density(ir)(ii+size_tmp,jj+size_tmp)*density(jr)(kk+size_tmp2,ll+size_tmp2) + f_NM*density(ir)(ii+size_tmp,jj+size_tmp)*density_o(jr)(kk+size_tmp2,ll+size_tmp2) + 0.5*f_NM*(NN-1)/(MM-1)*density_o(ir)(ii+size_tmp,jj+size_tmp)*density_o(jr)(kk+size_tmp2,ll+size_tmp2)) * (h2eSSSS_JK.J(ir,jr)(eij,ekl) - h2eSSSS_JK.K(ir,jr)(eil,ekj));
                            //LLSS
                            ene_scf += (0.5*density(ir)(ii,jj)*density(jr)(kk+size_tmp2,ll+size_tmp2) + f_NM*density(ir)(ii,jj)*density_o(jr)(kk+size_tmp2,ll+size_tmp2) + 0.5*f_NM*(NN-1)/(MM-1)*density_o(ir)(ii,jj)*density_o(jr)(kk+size_tmp2,ll+size_tmp2)) * h2eSSLL_JK.J(jr,ir)(ekl,eij);
                            ene_scf -= (0.5*density(ir)(ii,jj+size_tmp)*density(jr)(kk+size_tmp2,ll) + f_NM*density(ir)(ii,jj+size_tmp)*density_o(jr)(kk+size_tmp2,ll) + 0.5*f_NM*(NN-1)/(MM-1)*density_o(ir)(ii,jj+size_tmp)*density_o(jr)(kk+size_tmp2,ll)) * h2eSSLL_JK.K(jr,ir)(ekj,eil);
                            //SSLL
                            ene_scf += (0.5*density(ir)(ii+size_tmp,jj+size_tmp)*density(jr)(kk,ll) + f_NM*density(ir)(ii+size_tmp,jj+size_tmp)*density_o(jr)(kk,ll) + 0.5*f_NM*(NN-1)/(MM-1)*density_o(ir)(ii+size_tmp,jj+size_tmp)*density_o(jr)(kk,ll)) * h2eSSLL_JK.J(ir,jr)(eij,ekl);
                            ene_scf -= (0.5*density(ir)(ii+size_tmp,jj)*density(jr)(kk,ll+size_tmp2) + f_NM*density(ir)(ii+size_tmp,jj)*density_o(jr)(kk,ll+size_tmp2) + 0.5*f_NM*(NN-1)/(MM-1)*density_o(ir)(ii+size_tmp,jj)*density_o(jr)(kk,ll+size_tmp2)) * h2eSSLL_JK.K(ir,jr)(eil,ekj);
                        }
                    }
                }
            }
            if(twoC) cout << "Final CA-SFX2C-1e HF energy is " << setprecision(15) << ene_scf << " hartree." << endl;
            else cout << "Final CA-DHF energy is " << setprecision(15) << ene_scf << " hartree." << endl;
            break;            
        }
        for(int ir = 0; ir < occMax_irrep; ir++)    
        {
            int size_tmp = irrep_list(ir).size;
            evaluateFock_core(fock_c(ir),twoC,density,density_o,size_tmp,ir);
            evaluateFock_open(fock_o(ir),twoC,density,density_o,size_tmp,ir);

            eigensolverG(fock_c(ir), overlap_half_i_4c(ir), ene_orb(ir), coeff_c(ir));
            newDen_c(ir) = evaluateDensity_core(coeff_c(ir),occNumber(ir),twoC);
            error4DIIS_c[ir].push_back(evaluateErrorDIIS(density(ir),newDen_c(ir)));
            fock4DIIS_c[ir].push_back(fock_c(ir));

            eigensolverG(fock_o(ir), overlap_half_i_4c(ir), ene_orb(ir), coeff_o(ir));
            newDen_o(ir) = evaluateDensity_core(coeff_o(ir),occNumberDebug(ir),twoC);
            error4DIIS_o[ir].push_back(evaluateErrorDIIS(density_o(ir),newDen_o(ir)));
            fock4DIIS_o[ir].push_back(fock_o(ir));
    
            if(error4DIIS_c[ir].size() > size_DIIS)
            {
                error4DIIS_c[ir].erase(error4DIIS_c[ir].begin());
                fock4DIIS_c[ir].erase(fock4DIIS_c[ir].begin());
                error4DIIS_o[ir].erase(error4DIIS_o[ir].begin());
                fock4DIIS_o[ir].erase(fock4DIIS_o[ir].begin());
            }            
        }
    }
    EndTime = clock();
    cout << "DHF iterations finished in " << (EndTime - StartTime) / (double)CLOCKS_PER_SEC << " seconds." << endl << endl;
}


/* 
    evaluate Fock matrix 
*/
void DHF_SPH_CA::evaluateFock(MatrixXd& fock, const bool& twoC, const vMatrixXd& den_c, const vMatrixXd den_o, const int& size, const int& Iirrep)
{        
    if(!twoC)
    {
    }
    else
    {
        // fock.resize(size,size);
        // MatrixXd QQ(size,size);
        // #pragma omp parallel  for
        // for(int mm = 0; mm < size; mm++)
        // for(int nn = 0; nn <= mm; nn++)
        // {
        //     fock(mm,nn) = h1e_4c(Iirrep)(mm,nn);
        //     // fock(mm,nn) = kinetic(Iirrep)(mm,nn) + Vnuc(Iirrep)(mm,nn);
        //     QQ(mm,nn) = 0.0;
        //     for(int jr = 0; jr < occMax_irrep; jr++)
        //     {
        //         int size_tmp2 = irrep_list(jr).size;
        //         for(int aa = 0; aa < size_tmp2; aa++)
        //         for(int bb = 0; bb < size_tmp2; bb++)
        //         {
        //             int emn = mm*size+nn, eab = aa*size_tmp2+bb, emb = mm*size_tmp2+bb, ean = aa*size+nn;
        //             // fock(mm,nn) += (den_c(jr)(aa,bb) + (f_NM-1.0/(MM-1.0))*den_o(jr)(aa,bb)) * (h2eLLLL_JK.J(Iirrep,jr)(emn,eab) - h2eLLLL_JK.K(Iirrep,jr)(emb,ean));
        //             fock(mm,nn) += (den_c(jr)(aa,bb) + f_NM*den_o(jr)(aa,bb)) * (h2eLLLL_JK.J(Iirrep,jr)(emn,eab) - h2eLLLL_JK.K(Iirrep,jr)(emb,ean));
        //             QQ(mm,nn) += den_o(jr)(aa,bb) * (h2eLLLL_JK.J(Iirrep,jr)(emn,eab) - h2eLLLL_JK.K(Iirrep,jr)(emb,ean));
        //         }
        //     }
        //     QQ(nn,mm) = QQ(mm,nn);
        // }
        // #pragma omp parallel  for
        // for(int mm = 0; mm < size; mm++)
        // for(int nn = 0; nn <= mm; nn++)
        // {
        //     for(int ss = 0; ss < size; ss++)
        //     for(int rr = 0; rr < size; rr++)
        //     {
        //         fock(mm,nn) += f_NM/(MM-1.0) * den_o(Iirrep)(rr,ss) * (overlap(Iirrep)(mm,ss)*QQ(rr,nn) + QQ(mm,ss)*overlap(Iirrep)(rr,nn));
        //     }
        // }
        // for(int mm = 0; mm < size; mm++)
        // for(int nn = 0; nn <= mm; nn++)
        // {
        //     for(int ss = 0; ss < size; ss++)
        //     for(int rr = 0; rr < size; rr++)
        //     {
        //         fock(mm,nn) += 1.0/(MM-1.0) * den_c(Iirrep)(rr,ss) * (overlap(Iirrep)(mm,ss)*QQ(rr,nn) + QQ(mm,ss)*overlap(Iirrep)(rr,nn));
        //     }
        //     fock(mm,nn) -= 1.0/(MM-1.0) * QQ(mm,nn);
        //     fock(nn,mm) = fock(mm,nn);
        // }

        fock.resize(size,size);
        MatrixXd QQ(size,size);
        #pragma omp parallel  for
        for(int mm = 0; mm < size; mm++)
        for(int nn = 0; nn <= mm; nn++)
        {
            fock(mm,nn) = h1e_4c(Iirrep)(mm,nn);
            // fock(mm,nn) = kinetic(Iirrep)(mm,nn) + Vnuc(Iirrep)(mm,nn);
            QQ(mm,nn) = 0.0;
            for(int jr = 0; jr < occMax_irrep; jr++)
            {
                int size_tmp2 = irrep_list(jr).size;
                for(int aa = 0; aa < size_tmp2; aa++)
                for(int bb = 0; bb < size_tmp2; bb++)
                {
                    int emn = mm*size+nn, eab = aa*size_tmp2+bb, emb = mm*size_tmp2+bb, ean = aa*size+nn;
                    fock(mm,nn) += (den_c(jr)(aa,bb) + (NN-1.0)/(MM-1.0)*den_o(jr)(aa,bb)) * (h2eLLLL_JK.J(Iirrep,jr)(emn,eab) - h2eLLLL_JK.K(Iirrep,jr)(emb,ean));
                    QQ(mm,nn) += den_o(jr)(aa,bb) * (h2eLLLL_JK.J(Iirrep,jr)(emn,eab) - h2eLLLL_JK.K(Iirrep,jr)(emb,ean));
                }
            }
            QQ(nn,mm) = QQ(mm,nn);
        }
        #pragma omp parallel  for
        for(int mm = 0; mm < size; mm++)
        for(int nn = 0; nn <= mm; nn++)
        {
            for(int ss = 0; ss < size; ss++)
            for(int rr = 0; rr < size; rr++)
            {
                fock(mm,nn) += 1.0/(MM-1.0) * den_c(Iirrep)(rr,ss) * (overlap(Iirrep)(mm,ss)*QQ(rr,nn) + QQ(mm,ss)*overlap(Iirrep)(rr,nn));
            }
            fock(nn,mm) = fock(mm,nn);
        }
        MatrixXd fock2(size,size);
        for(int mm = 0; mm < size; mm++)
        for(int nn = 0; nn <= mm; nn++)
        {
            for(int ss = 0; ss < size; ss++)
            for(int rr = 0; rr < size; rr++)
            {
                fock2(mm,nn) += f_NM/(MM-1.0) * den_o(Iirrep)(rr,ss) * (overlap(Iirrep)(mm,ss)*QQ(rr,nn) + QQ(mm,ss)*overlap(Iirrep)(rr,nn));
            }
            fock2(mm,nn) -= f_NM/(MM-1.0) * QQ(mm,nn);
            fock2(nn,mm) = fock2(mm,nn);
        }
        cout << overlap_half_i_4c(Iirrep)*(fock*overlap(Iirrep).inverse()*fock2 - fock2*overlap(Iirrep).inverse()*fock)*overlap_half_i_4c(Iirrep) << endl << endl;
    }

    return;
}


void DHF_SPH_CA::evaluateFock_core(MatrixXd& fock, const bool& twoC, const vMatrixXd& den_c, const vMatrixXd den_o, const int& size, const int& Iirrep)
{
    if(twoC)
    {
        fock.resize(size,size);
        MatrixXd QQ(size,size);
        #pragma omp parallel  for
        for(int mm = 0; mm < size; mm++)
        for(int nn = 0; nn <= mm; nn++)
        {
            fock(mm,nn) = h1e_4c(Iirrep)(mm,nn);
            // fock(mm,nn) = kinetic(Iirrep)(mm,nn) + Vnuc(Iirrep)(mm,nn);
            QQ(mm,nn) = 0.0;
            for(int jr = 0; jr < occMax_irrep; jr++)
            {
                int size_tmp2 = irrep_list(jr).size;
                for(int aa = 0; aa < size_tmp2; aa++)
                for(int bb = 0; bb < size_tmp2; bb++)
                {
                    int emn = mm*size+nn, eab = aa*size_tmp2+bb, emb = mm*size_tmp2+bb, ean = aa*size+nn;
                    fock(mm,nn) += (den_c(jr)(aa,bb) + f_NM*den_o(jr)(aa,bb)) * (h2eLLLL_JK.J(Iirrep,jr)(emn,eab) - h2eLLLL_JK.K(Iirrep,jr)(emb,ean));
                    QQ(mm,nn) += den_o(jr)(aa,bb) * (h2eLLLL_JK.J(Iirrep,jr)(emn,eab) - h2eLLLL_JK.K(Iirrep,jr)(emb,ean));
                }
            }
            QQ(nn,mm) = QQ(mm,nn);
        }
        #pragma omp parallel  for
        for(int mm = 0; mm < size; mm++)
        for(int nn = 0; nn <= mm; nn++)
        {
            for(int ss = 0; ss < size; ss++)
            for(int rr = 0; rr < size; rr++)
            {
                fock(mm,nn) += f_NM/(MM-1.0) * den_o(Iirrep)(rr,ss) * (overlap(Iirrep)(mm,ss)*QQ(rr,nn) + QQ(mm,ss)*overlap(Iirrep)(rr,nn));
            }
            fock(nn,mm) = fock(mm,nn);
        }
    }
    else
    {
        MatrixXd QQ = MatrixXd::Zero(size*2,size*2);
        fock.resize(size*2,size*2);
        #pragma omp parallel  for
        for(int mm = 0; mm < size; mm++)
        for(int nn = 0; nn <= mm; nn++)
        {
            fock(mm,nn) = h1e_4c(Iirrep)(mm,nn);
            fock(mm+size,nn) = h1e_4c(Iirrep)(mm+size,nn);
            if(mm != nn) fock(nn+size,mm) = h1e_4c(Iirrep)(nn+size,mm);
            fock(mm+size,nn+size) = h1e_4c(Iirrep)(mm+size,nn+size);
            for(int jr = 0; jr < occMax_irrep; jr++)
            {
                int size_tmp2 = irrep_list(jr).size;
                for(int ss = 0; ss < size_tmp2; ss++)
                for(int rr = 0; rr < size_tmp2; rr++)
                {
                    int emn = mm*size+nn, esr = ss*size_tmp2+rr, emr = mm*size_tmp2+rr, esn = ss*size+nn;
                    fock(mm,nn) += (den_c(jr)(ss,rr)+f_NM*den_o(jr)(ss,rr)) * (h2eLLLL_JK.J(Iirrep,jr)(emn,esr) - h2eLLLL_JK.K(Iirrep,jr)(emr,esn)) + (den_c(jr)(size_tmp2+ss,size_tmp2+rr)+f_NM*den_o(jr)(size_tmp2+ss,size_tmp2+rr)) * h2eSSLL_JK.J(jr,Iirrep)(esr,emn);
                    fock(mm+size,nn) -= (den_c(jr)(ss,size_tmp2+rr)+f_NM*den_o(jr)(ss,size_tmp2+rr)) * h2eSSLL_JK.K(Iirrep,jr)(emr,esn);
                    if(mm != nn) 
                    {
                        int enr = nn*size_tmp2+rr, esm = ss*size+mm;
                        fock(nn+size,mm) -= (den_c(jr)(ss,size_tmp2+rr)+f_NM*den_o(jr)(ss,size_tmp2+rr)) * h2eSSLL_JK.K(Iirrep,jr)(enr,esm);
                    }
                    fock(mm+size,nn+size) += (den_c(jr)(size_tmp2+ss,size_tmp2+rr)+f_NM*den_o(jr)(size_tmp2+ss,size_tmp2+rr)) * (h2eSSSS_JK.J(Iirrep,jr)(emn,esr) - h2eSSSS_JK.K(Iirrep,jr)(emr,esn)) + (den_c(jr)(ss,rr)+f_NM*den_o(jr)(ss,rr)) * h2eSSLL_JK.J(Iirrep,jr)(emn,esr);

                    QQ(mm,nn) += den_o(jr)(ss,rr) * (h2eLLLL_JK.J(Iirrep,jr)(emn,esr) - h2eLLLL_JK.K(Iirrep,jr)(emr,esn)) + den_o(jr)(size_tmp2+ss,size_tmp2+rr) * h2eSSLL_JK.J(jr,Iirrep)(esr,emn);
                    QQ(mm+size,nn) -= den_o(jr)(ss,size_tmp2+rr) * h2eSSLL_JK.K(Iirrep,jr)(emr,esn);
                    if(mm != nn) 
                    {
                        int enr = nn*size_tmp2+rr, esm = ss*size+mm;
                        QQ(nn+size,mm) -= den_o(jr)(ss,size_tmp2+rr) * h2eSSLL_JK.K(Iirrep,jr)(enr,esm);
                    }
                    QQ(mm+size,nn+size) += den_o(jr)(size_tmp2+ss,size_tmp2+rr) * (h2eSSSS_JK.J(Iirrep,jr)(emn,esr) - h2eSSSS_JK.K(Iirrep,jr)(emr,esn)) + den_o(jr)(ss,rr) * h2eSSLL_JK.J(Iirrep,jr)(emn,esr);
                }
            }
            QQ(nn,mm) = QQ(mm,nn);
            QQ(nn+size,mm+size) = QQ(mm+size,nn+size);
            QQ(nn,mm+size) = QQ(mm+size,nn);
            QQ(mm,nn+size) = QQ(nn+size,mm);
        }
        #pragma omp parallel  for
        for(int mm = 0; mm < 2*size; mm++)
        for(int nn = 0; nn <= mm; nn++)
        {
            for(int ss = 0; ss < 2*size; ss++)
            for(int rr = 0; rr < 2*size; rr++)
            {
                fock(mm,nn) += f_NM/(MM-1.0) * den_o(Iirrep)(rr,ss) * (overlap_4c(Iirrep)(mm,ss)*QQ(rr,nn) + QQ(mm,ss)*overlap_4c(Iirrep)(rr,nn));
            }
            fock(nn,mm) = fock(mm,nn);
        }
    }

    return;
}

void DHF_SPH_CA::evaluateFock_open(MatrixXd& fock, const bool& twoC, const vMatrixXd& den_c, const vMatrixXd den_o, const int& size, const int& Iirrep)
{
    if(twoC)
    {
        fock.resize(size,size);
        MatrixXd QQ(size,size);
        #pragma omp parallel  for
        for(int mm = 0; mm < size; mm++)
        for(int nn = 0; nn <= mm; nn++)
        {
            fock(mm,nn) = h1e_4c(Iirrep)(mm,nn);
            // fock(mm,nn) = kinetic(Iirrep)(mm,nn) + Vnuc(Iirrep)(mm,nn);
            QQ(mm,nn) = 0.0;
            for(int jr = 0; jr < occMax_irrep; jr++)
            {
                int size_tmp2 = irrep_list(jr).size;
                for(int aa = 0; aa < size_tmp2; aa++)
                for(int bb = 0; bb < size_tmp2; bb++)
                {
                    int emn = mm*size+nn, eab = aa*size_tmp2+bb, emb = mm*size_tmp2+bb, ean = aa*size+nn;
                    fock(mm,nn) += (den_c(jr)(aa,bb) + (NN-1.0)/(MM-1.0)*den_o(jr)(aa,bb)) * (h2eLLLL_JK.J(Iirrep,jr)(emn,eab) - h2eLLLL_JK.K(Iirrep,jr)(emb,ean));
                    QQ(mm,nn) += den_o(jr)(aa,bb) * (h2eLLLL_JK.J(Iirrep,jr)(emn,eab) - h2eLLLL_JK.K(Iirrep,jr)(emb,ean));
                }
            }
            QQ(nn,mm) = QQ(mm,nn);
        }
        #pragma omp parallel  for
        for(int mm = 0; mm < size; mm++)
        for(int nn = 0; nn <= mm; nn++)
        {
            for(int ss = 0; ss < size; ss++)
            for(int rr = 0; rr < size; rr++)
            {
                fock(mm,nn) += 1.0/(MM-1.0) * den_c(Iirrep)(rr,ss) * (overlap(Iirrep)(mm,ss)*QQ(rr,nn) + QQ(mm,ss)*overlap(Iirrep)(rr,nn));
            }
            fock(nn,mm) = fock(mm,nn);
        }
    }
    else
    {
        MatrixXd QQ = MatrixXd::Zero(size*2,size*2);
        fock.resize(size*2,size*2);
        #pragma omp parallel  for
        for(int mm = 0; mm < size; mm++)
        for(int nn = 0; nn <= mm; nn++)
        {
            fock(mm,nn) = h1e_4c(Iirrep)(mm,nn);
            fock(mm+size,nn) = h1e_4c(Iirrep)(mm+size,nn);
            if(mm != nn) fock(nn+size,mm) = h1e_4c(Iirrep)(nn+size,mm);
            fock(mm+size,nn+size) = h1e_4c(Iirrep)(mm+size,nn+size);
            for(int jr = 0; jr < occMax_irrep; jr++)
            {
                int size_tmp2 = irrep_list(jr).size;
                for(int ss = 0; ss < size_tmp2; ss++)
                for(int rr = 0; rr < size_tmp2; rr++)
                {
                    int emn = mm*size+nn, esr = ss*size_tmp2+rr, emr = mm*size_tmp2+rr, esn = ss*size+nn;
                    fock(mm,nn) += (den_c(jr)(ss,rr)+(NN-1.0)/(MM-1.0)*den_o(jr)(ss,rr)) * (h2eLLLL_JK.J(Iirrep,jr)(emn,esr) - h2eLLLL_JK.K(Iirrep,jr)(emr,esn)) + (den_c(jr)(size_tmp2+ss,size_tmp2+rr)+(NN-1.0)/(MM-1.0)*den_o(jr)(size_tmp2+ss,size_tmp2+rr)) * h2eSSLL_JK.J(jr,Iirrep)(esr,emn);
                    fock(mm+size,nn) -= (den_c(jr)(ss,size_tmp2+rr)+(NN-1.0)/(MM-1.0)*den_o(jr)(ss,size_tmp2+rr)) * h2eSSLL_JK.K(Iirrep,jr)(emr,esn);
                    if(mm != nn) 
                    {
                        int enr = nn*size_tmp2+rr, esm = ss*size+mm;
                        fock(nn+size,mm) -= (den_c(jr)(ss,size_tmp2+rr)+(NN-1.0)/(MM-1.0)*den_o(jr)(ss,size_tmp2+rr)) * h2eSSLL_JK.K(Iirrep,jr)(enr,esm);
                    }
                    fock(mm+size,nn+size) += (den_c(jr)(size_tmp2+ss,size_tmp2+rr)+(NN-1.0)/(MM-1.0)*den_o(jr)(size_tmp2+ss,size_tmp2+rr)) * (h2eSSSS_JK.J(Iirrep,jr)(emn,esr) - h2eSSSS_JK.K(Iirrep,jr)(emr,esn)) + (den_c(jr)(ss,rr)+(NN-1.0)/(MM-1.0)*den_o(jr)(ss,rr)) * h2eSSLL_JK.J(Iirrep,jr)(emn,esr);

                    QQ(mm,nn) += den_o(jr)(ss,rr) * (h2eLLLL_JK.J(Iirrep,jr)(emn,esr) - h2eLLLL_JK.K(Iirrep,jr)(emr,esn)) + den_o(jr)(size_tmp2+ss,size_tmp2+rr) * h2eSSLL_JK.J(jr,Iirrep)(esr,emn);
                    QQ(mm+size,nn) -= den_o(jr)(ss,size_tmp2+rr)* h2eSSLL_JK.K(Iirrep,jr)(emr,esn);
                    if(mm != nn) 
                    {
                        int enr = nn*size_tmp2+rr, esm = ss*size+mm;
                        QQ(nn+size,mm) -= den_o(jr)(ss,size_tmp2+rr) * h2eSSLL_JK.K(Iirrep,jr)(enr,esm);
                    }
                    QQ(mm+size,nn+size) += den_o(jr)(size_tmp2+ss,size_tmp2+rr) * (h2eSSSS_JK.J(Iirrep,jr)(emn,esr) - h2eSSSS_JK.K(Iirrep,jr)(emr,esn)) + den_o(jr)(ss,rr) * h2eSSLL_JK.J(Iirrep,jr)(emn,esr);
                }
            }
            QQ(nn,mm) = QQ(mm,nn);
            QQ(nn+size,mm+size) = QQ(mm+size,nn+size);
            QQ(nn,mm+size) = QQ(mm+size,nn);
            QQ(mm,nn+size) = QQ(nn+size,mm);
        }
        #pragma omp parallel  for
        for(int mm = 0; mm < 2*size; mm++)
        for(int nn = 0; nn <= mm; nn++)
        {
            for(int ss = 0; ss < 2*size; ss++)
            for(int rr = 0; rr < 2*size; rr++)
            {
                fock(mm,nn) += 1.0/(MM-1.0) * den_c(Iirrep)(rr,ss) * (overlap_4c(Iirrep)(mm,ss)*QQ(rr,nn) + QQ(mm,ss)*overlap_4c(Iirrep)(rr,nn));
            }
            fock(nn,mm) = fock(mm,nn);
        }
    }

    return;
}


vMatrixXd DHF_SPH_CA::get_amfi_unc_ca(INT_SPH& int_sph_, const bool& twoC, const string& Xmethod)
{
    int2eJK SSLL_SD, SSSS_SD;
    int_sph_.get_h2eSD_JK_direct(SSLL_SD, SSSS_SD);
    if(twoC)
    {
        return get_amfi_unc_ca_2c(SSLL_SD, SSSS_SD);
    }
    else 
    {
        if(occMax_irrep < Nirrep && Xmethod == "fullFock")
        {
            cout << "fullFock is used in amfi function with incomplete h2e." << endl;
            cout << "Recalculate h2e..." << endl;
            StartTime = clock();
            int_sph_.get_h2e_JK_direct(h2eLLLL_JK,h2eSSLL_JK,h2eSSSS_JK);
            EndTime = clock();
            cout << "Complete 2e-integral finished in " << (EndTime - StartTime) / (double)CLOCKS_PER_SEC << " seconds." << endl << endl; 
        }
        vMatrixXd density_t(occMax_irrep);
        for(int ir = 0;ir < occMax_irrep; ir++)
        {
            density_t(ir) = density(ir) + f_NM*density_o(ir);
        }
        return get_amfi_unc(SSLL_SD, SSSS_SD, density_t, Xmethod);
    }
}


vMatrixXd DHF_SPH_CA::get_amfi_unc_ca_2c(const int2eJK& h2eSSLL_SD, const int2eJK& h2eSSSS_SD)
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
        density_tmp(ir) = evaluateDensity_core(coeff_tmp(ir),occNumber(ir),false) + f_NM*evaluateDensity_core(coeff_tmp(ir),occNumberDebug(ir),false);
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

