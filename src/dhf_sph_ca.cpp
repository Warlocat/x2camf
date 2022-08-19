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

vVectorXd occNumberOpen;


DHF_SPH_CA::DHF_SPH_CA(INT_SPH& int_sph_, const string& filename, const bool& spinFree, const bool& twoC, const bool& with_gaunt_, const bool& with_gauge_, const bool& allInt, const bool& gaussian_nuc):
DHF_SPH(int_sph_,filename,spinFree,twoC,with_gaunt_,with_gauge_,allInt,gaussian_nuc)
{
    openShell = -1;
    for(int ir = 0; ir < occMax_irrep; ir++)
    {
        for(int ii = 0; ii < occNumber(ir).rows(); ii++)
        {
            // if 1 > occNumber(ir)(ii) > 0
            if(abs(occNumber(ir)(ii)) > 1e-4 && abs(occNumber(ir)(ii)) < 1.0)
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

    occNumberOpen.resize(occMax_irrep);
    for(int ir = 0; ir < occMax_irrep; ir++)
    {
        occNumberOpen(ir).resize(irrep_list(ir).size);
        occNumberOpen(ir) = VectorXd::Zero(irrep_list(ir).size);
        for(int ii = 0; ii < occNumber(ir).rows(); ii++)
        {
            if(abs(occNumber(ir)(ii) - 1.0) < 1e-5)
                occNumberOpen(ir)(ii) = 0.0;
            else if(abs(occNumber(ir)(ii)) > 1e-4 && abs(occNumber(ir)(ii)) < (1.0-1e-4))
            {
                occNumber(ir)(ii) = 0.0;
                occNumberOpen(ir)(ii) = 1.0;
            }
        }
    }
    for(int ir = 0; ir < occMax_irrep; ir++)
    {
        cout << "1: " << occNumber(ir).transpose() << endl;
        cout << "2: " << occNumberOpen(ir).transpose() << endl;
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
    for(int ir = 0; ir < occMax_irrep; ir+=irrep_list(ir).two_j+1)
    {
        den_c(ir) = evaluateDensity_core(coeff_(ir),occNumber(ir), twoC);
        den_o(ir) = evaluateDensity_core(coeff_(ir),occNumberOpen(ir), twoC);
    }
}


/*
    SCF procedure for 4-c and 2-c calculation
*/
void DHF_SPH_CA::runSCF(const bool& twoC, const bool& renormSmall)
{
    if(renormSmall)
    {
        renormalize_small();
    }
    vector<MatrixXd> error4DIIS_c[occMax_irrep], fock4DIIS_c[occMax_irrep], error4DIIS_o[occMax_irrep], fock4DIIS_o[occMax_irrep];
    vMatrixXd coeff_c(occMax_irrep), coeff_o(occMax_irrep), fock_c(occMax_irrep), fock_o(occMax_irrep);
    StartTime = clock();
    cout << endl;
    if(twoC) cout << "Start CA-X2C-1e Hartree-Fock iterations..." << endl;
    else cout << "Start CA-Dirac Hartree-Fock iterations..." << endl;
    cout << endl;

    vMatrixXd newDen_c(occMax_irrep), newDen_o(occMax_irrep), oldDen_t(occMax_irrep), newDen_t(occMax_irrep);
    eigensolverG_irrep(h1e_4c, overlap_half_i_4c, ene_orb, coeff_c);
    coeff_o = coeff_c;
    density.resize(occMax_irrep);
    density_o.resize(occMax_irrep);
    for(int ir = 0; ir < occMax_irrep; ir+=irrep_list(ir).two_j+1)
    {
        density(ir) = evaluateDensity_core(coeff_c(ir),occNumber(ir),twoC);
        density_o(ir) = evaluateDensity_core(coeff_o(ir),occNumberOpen(ir),twoC);
        // for(int jj = 1; jj < irrep_list(ir).two_j+1; jj++)
        // {
        //     density_o(ir+jj) = density_o(ir);
        //     density(ir+jj) = density(ir);
        // }
    }

    for(int iter = 1; iter <= maxIter; iter++)
    {
        if(iter <= 2)
        {
            for(int ir = 0; ir < occMax_irrep; ir += irrep_list(ir).two_j+1)    
            {
                int size_tmp = irrep_list(ir).size;
                evaluateFock(fock_c(ir),fock_o(ir),twoC,density,density_o,size_tmp,ir);
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
                    for(int ir = 0; ir < occMax_irrep; ir += irrep_list(ir).two_j+1)
                        B4DIIS(ii,jj) += (error4DIIS_c[ir][ii].adjoint()*error4DIIS_c[ir][jj])(0,0);
                    for(int ir = 0; ir < occMax_irrep; ir += irrep_list(ir).two_j+1)
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
            for(int ir = 0; ir < occMax_irrep; ir += irrep_list(ir).two_j+1)
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

        for(int ir = 0; ir < occMax_irrep; ir += irrep_list(ir).two_j+1)
        {
            newDen_c(ir) = evaluateDensity_core(coeff_c(ir),occNumber(ir),twoC);
            newDen_o(ir) = evaluateDensity_core(coeff_o(ir),occNumberOpen(ir),twoC);
            // for(int jj = 1; jj < irrep_list(ir).two_j+1; jj++)
            // {
            //     newDen_c(ir+jj) = newDen_c(ir);
            //     newDen_o(ir+jj) = newDen_o(ir);
            // }
        }
        d_density = max(evaluateChange_irrep(density, newDen_c),evaluateChange_irrep(density_o,newDen_o));              
        cout << "Iter #" << iter << " maximum density difference: " << d_density << endl;     
        density = newDen_c;
        density_o = newDen_o;

        if(d_density < convControl) 
        {
            converged = true;
            cout << endl << "CA-SCF converges after " << iter << " iterations." << endl;
            cout << endl << "WARNING: CA-SCF orbital energies are fake!!!" << endl << endl;

            cout << "\tOrbital\t\tEnergy(in hartree)\n";
            cout << "\t*******\t\t******************\n";
            for(int ir = 0; ir < occMax_irrep; ir += irrep_list(ir).two_j+1)
            for(int ii = 1; ii <= irrep_list(ir).size; ii++)
            {
                if(twoC) cout << "\t" << ii << "\t\t" << setprecision(15) << ene_orb(ir)(ii - 1) << endl;
                else cout << "\t" << ii << "\t\t" << setprecision(15) << ene_orb(ir)(irrep_list(ir).size + ii - 1) << endl;
            }
            
            coeff.resize(occMax_irrep);
            if(twoC)
            {
                for(int ir = 0; ir < occMax_irrep; ir += irrep_list(ir).two_j+1)
                {
                    coeff(ir) = coeff_c(ir);
                    for(int ii = 0; ii < irrep_list(ir).size; ii++)
                    {
                        if(abs(occNumberOpen(ir)(ii) - 1.0) < 1e-5)
                        {
                            for(int jj = 0; jj < coeff(ir).rows(); jj++)
                                coeff(ir)(jj,ii) = coeff_o(ir)(jj,ii);
                        }
                    }
                }
            }
            else
            {
                for(int ir = 0; ir < occMax_irrep; ir += irrep_list(ir).two_j+1)
                {
                    coeff(ir) = coeff_c(ir);
                    for(int ii = 0; ii < irrep_list(ir).size; ii++)
                    {
                        if(abs(occNumberOpen(ir)(ii) - 1.0) < 1e-5)
                        {
                            for(int jj = 0; jj < coeff(ir).rows(); jj++)
                                coeff(ir)(jj,ii+coeff(ir).rows()/2) = coeff_o(ir)(jj,ii+coeff(ir).rows()/2);
                        }
                    }
                }
            }

            ene_scf = 0.0;
            for(int ir = 0; ir < occMax_irrep_compact; ir++)
            {
                int Iirrep = compact2all(ir);
                double tmp_d = 0.0;
                int size_tmp = irrep_list(Iirrep).size;
                if(twoC)
                {
                    for(int ii = 0; ii < size_tmp; ii++)
                    for(int jj = 0; jj < size_tmp; jj++)
                    {
                        tmp_d += (density(Iirrep)(ii,jj) + f_NM*density_o(Iirrep)(ii,jj)) * h1e_4c(Iirrep)(jj,ii);
                        for(int jr = 0; jr < occMax_irrep_compact; jr++)
                        {
                            int Jirrep = compact2all(jr);
                            double twojP1 = irrep_list(Jirrep).two_j+1.0;
                            for(int kk = 0; kk < irrep_list(Jirrep).size; kk++)
                            for(int ll = 0; ll < irrep_list(Jirrep).size; ll++)
                            {
                                int eij = ii*size_tmp+jj, ekl = kk*irrep_list(Jirrep).size+ll, eil = ii*irrep_list(Jirrep).size+ll, ekj = kk*size_tmp+jj;
                                tmp_d += twojP1*(0.5*density(Iirrep)(ii,jj)*density(Jirrep)(kk,ll) + f_NM*density(Iirrep)(ii,jj)*density_o(Jirrep)(kk,ll) + 0.5*f_NM*(NN-1)/(MM-1)*density_o(Iirrep)(ii,jj)*density_o(Jirrep)(kk,ll)) * h2eLLLL_JK.J[ir][jr][eij][ekl];
                            }
                        }
                    }
                }
                else
                {
                    for(int ii = 0; ii < size_tmp; ii++)
                    for(int jj = 0; jj < size_tmp; jj++)
                    {
                        tmp_d += (density(Iirrep)(ii,jj) + f_NM*density_o(Iirrep)(ii,jj)) * h1e_4c(Iirrep)(jj,ii);
                        tmp_d += (density(Iirrep)(ii+size_tmp,jj) + f_NM*density_o(Iirrep)(ii+size_tmp,jj)) * h1e_4c(Iirrep)(jj,ii+size_tmp);
                        tmp_d += (density(Iirrep)(ii,jj+size_tmp) + f_NM*density_o(Iirrep)(ii,jj+size_tmp)) * h1e_4c(Iirrep)(jj+size_tmp,ii);
                        tmp_d += (density(Iirrep)(ii+size_tmp,jj+size_tmp) + f_NM*density_o(Iirrep)(ii+size_tmp,jj+size_tmp)) * h1e_4c(Iirrep)(jj+size_tmp,ii+size_tmp);
                        for(int jr = 0; jr < occMax_irrep_compact; jr++)
                        {
                            int Jirrep = compact2all(jr);
                            double twojP1 = irrep_list(Jirrep).two_j+1.0;
                            int size_tmp2 = irrep_list(Jirrep).size;
                            for(int kk = 0; kk < irrep_list(Jirrep).size; kk++)
                            for(int ll = 0; ll < irrep_list(Jirrep).size; ll++)
                            {
                                int eij = ii*size_tmp+jj, ekl = kk*irrep_list(Jirrep).size+ll, eil = ii*irrep_list(Jirrep).size+ll, ekj = kk*size_tmp+jj;
                                //LLLL
                                tmp_d += twojP1*(0.5*density(Iirrep)(ii,jj)*density(Jirrep)(kk,ll) + f_NM*density(Iirrep)(ii,jj)*density_o(Jirrep)(kk,ll) + 0.5*f_NM*(NN-1)/(MM-1)*density_o(Iirrep)(ii,jj)*density_o(Jirrep)(kk,ll)) * h2eLLLL_JK.J[ir][jr][eij][ekl];
                                //SSSS
                                tmp_d += twojP1*(0.5*density(Iirrep)(ii+size_tmp,jj+size_tmp)*density(Jirrep)(kk+size_tmp2,ll+size_tmp2) + f_NM*density(Iirrep)(ii+size_tmp,jj+size_tmp)*density_o(Jirrep)(kk+size_tmp2,ll+size_tmp2) + 0.5*f_NM*(NN-1)/(MM-1)*density_o(Iirrep)(ii+size_tmp,jj+size_tmp)*density_o(Jirrep)(kk+size_tmp2,ll+size_tmp2)) * h2eSSSS_JK.J[ir][jr][eij][ekl];
                                //LLSS
                                tmp_d += twojP1*(0.5*density(Iirrep)(ii,jj)*density(Jirrep)(kk+size_tmp2,ll+size_tmp2) + f_NM*density(Iirrep)(ii,jj)*density_o(Jirrep)(kk+size_tmp2,ll+size_tmp2) + 0.5*f_NM*(NN-1)/(MM-1)*density_o(Iirrep)(ii,jj)*density_o(Jirrep)(kk+size_tmp2,ll+size_tmp2)) * h2eSSLL_JK.J[jr][ir][ekl][eij];
                                tmp_d -= twojP1*(0.5*density(Iirrep)(ii,jj+size_tmp)*density(Jirrep)(kk+size_tmp2,ll) + f_NM*density(Iirrep)(ii,jj+size_tmp)*density_o(Jirrep)(kk+size_tmp2,ll) + 0.5*f_NM*(NN-1)/(MM-1)*density_o(Iirrep)(ii,jj+size_tmp)*density_o(Jirrep)(kk+size_tmp2,ll)) * h2eSSLL_JK.K[jr][ir][ekj][eil];
                                //SSLL
                                tmp_d += twojP1*(0.5*density(Iirrep)(ii+size_tmp,jj+size_tmp)*density(Jirrep)(kk,ll) + f_NM*density(Iirrep)(ii+size_tmp,jj+size_tmp)*density_o(Jirrep)(kk,ll) + 0.5*f_NM*(NN-1)/(MM-1)*density_o(Iirrep)(ii+size_tmp,jj+size_tmp)*density_o(Jirrep)(kk,ll)) * h2eSSLL_JK.J[ir][jr][eij][ekl];
                                tmp_d -= twojP1*(0.5*density(Iirrep)(ii+size_tmp,jj)*density(Jirrep)(kk,ll+size_tmp2) + f_NM*density(Iirrep)(ii+size_tmp,jj)*density_o(Jirrep)(kk,ll+size_tmp2) + 0.5*f_NM*(NN-1)/(MM-1)*density_o(Iirrep)(ii+size_tmp,jj)*density_o(Jirrep)(kk,ll+size_tmp2)) * h2eSSLL_JK.K[ir][jr][eil][ekj];
                                if(with_gaunt)
                                {
                                    int eji = jj*size_tmp+ii, elk = ll*size_tmp2+kk, eli = ll*size_tmp+ii, ejk = jj*size_tmp2+kk;
                                    //LSLS
                                    tmp_d += twojP1*(0.5*density(Iirrep)(ii,jj+size_tmp)*density(Jirrep)(kk,ll+size_tmp2) + f_NM*density(Iirrep)(ii,jj+size_tmp)*density_o(Jirrep)(kk,ll+size_tmp2) + 0.5*f_NM*(NN-1)/(MM-1)*density_o(Iirrep)(ii,jj+size_tmp)*density_o(Jirrep)(kk,ll+size_tmp2)) * gauntLSLS_JK.J[ir][jr][eij][ekl];
                                    //SLSL
                                    tmp_d += twojP1*(0.5*density(Iirrep)(ii+size_tmp,jj)*density(Jirrep)(kk+size_tmp2,ll) + f_NM*density(Iirrep)(ii+size_tmp,jj)*density_o(Jirrep)(kk+size_tmp2,ll) + 0.5*f_NM*(NN-1)/(MM-1)*density_o(Iirrep)(ii+size_tmp,jj)*density_o(Jirrep)(kk+size_tmp2,ll)) * gauntLSLS_JK.J[ir][jr][eji][elk];
                                    //LSSL
                                    tmp_d += twojP1*(0.5*density(Iirrep)(ii,jj+size_tmp)*density(Jirrep)(kk+size_tmp2,ll) + f_NM*density(Iirrep)(ii,jj+size_tmp)*density_o(Jirrep)(kk+size_tmp2,ll) + 0.5*f_NM*(NN-1)/(MM-1)*density_o(Iirrep)(ii,jj+size_tmp)*density_o(Jirrep)(kk+size_tmp2,ll)) * gauntLSSL_JK.J[ir][jr][eij][ekl];
                                    tmp_d -= twojP1*(0.5*density(Iirrep)(ii,jj)*density(Jirrep)(kk+size_tmp2,ll+size_tmp2) + f_NM*density(Iirrep)(ii,jj)*density_o(Jirrep)(kk+size_tmp2,ll+size_tmp2) + 0.5*f_NM*(NN-1)/(MM-1)*density_o(Iirrep)(ii,jj)*density_o(Jirrep)(kk+size_tmp2,ll+size_tmp2)) * gauntLSSL_JK.K[ir][jr][eil][ekj];
                                    //SLLS
                                    tmp_d += twojP1*(0.5*density(Iirrep)(ii+size_tmp,jj)*density(Jirrep)(kk,ll+size_tmp2) + f_NM*density(Iirrep)(ii+size_tmp,jj)*density_o(Jirrep)(kk,ll+size_tmp2) + 0.5*f_NM*(NN-1)/(MM-1)*density_o(Iirrep)(ii+size_tmp,jj)*density_o(Jirrep)(kk,ll+size_tmp2)) * gauntLSSL_JK.J[ir][jr][eji][elk];
                                    tmp_d -= twojP1*(0.5*density(Iirrep)(ii+size_tmp,jj+size_tmp)*density(Jirrep)(kk,ll) + f_NM*density(Iirrep)(ii+size_tmp,jj+size_tmp)*density_o(Jirrep)(kk,ll) + 0.5*f_NM*(NN-1)/(MM-1)*density_o(Iirrep)(ii+size_tmp,jj+size_tmp)*density_o(Jirrep)(kk,ll)) * gauntLSSL_JK.K[jr][ir][eli][ejk];
                                }
                            }
                        }
                    }
                }
                ene_scf += tmp_d * (irrep_list(Iirrep).two_j+1);
            }
            if(twoC) cout << "Final CA-X2C-1e HF energy is " << setprecision(15) << ene_scf << " hartree." << endl;
            else cout << "Final CA-DHF energy is " << setprecision(15) << ene_scf << " hartree." << endl;
            break;            
        }
        for(int ir = 0; ir < occMax_irrep; ir += irrep_list(ir).two_j+1)    
        {
            int size_tmp = irrep_list(ir).size;
            evaluateFock(fock_c(ir),fock_o(ir),twoC,density,density_o,size_tmp,ir);

            eigensolverG(fock_c(ir), overlap_half_i_4c(ir), ene_orb(ir), coeff_c(ir));
            newDen_c(ir) = evaluateDensity_core(coeff_c(ir),occNumber(ir),twoC);
            error4DIIS_c[ir].push_back(evaluateErrorDIIS(density(ir),newDen_c(ir)));
            fock4DIIS_c[ir].push_back(fock_c(ir));

            eigensolverG(fock_o(ir), overlap_half_i_4c(ir), ene_orb(ir), coeff_o(ir));
            newDen_o(ir) = evaluateDensity_core(coeff_o(ir),occNumberOpen(ir),twoC);
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

    
    for(int ir = 0; ir < occMax_irrep; ir += irrep_list(ir).two_j+1)
    {
        for(int jj = 1; jj < irrep_list(ir).two_j+1; jj++)
        {
            // fock_c(ir+jj) = fock_c(ir);
            // fock_o(ir+jj) = fock_o(ir);
            ene_orb(ir+jj) = ene_orb(ir);
            coeff(ir+jj) = coeff(ir);
            density(ir+jj) = density(ir);
            density_o(ir+jj) = density_o(ir);
        }
    }
    cout << "DHF iterations finished in " << (EndTime - StartTime) / (double)CLOCKS_PER_SEC << " seconds." << endl << endl;
}


/* 
    evaluate Fock matrix 
*/
void DHF_SPH_CA::evaluateFock(MatrixXd& fock_c, MatrixXd& fock_o, const bool& twoC, const vMatrixXd& den_c, const vMatrixXd den_o, const int& size, const int& Iirrep)
{
    int ir = all2compact(Iirrep);
    if(twoC)
    {
        fock_c.resize(size,size);
        fock_o.resize(size,size);
        MatrixXd QQ(size,size);
        #pragma omp parallel  for
        for(int mm = 0; mm < size; mm++)
        for(int nn = 0; nn <= mm; nn++)
        {
            fock_c(mm,nn) = h1e_4c(Iirrep)(mm,nn);
            fock_o(mm,nn) = h1e_4c(Iirrep)(mm,nn);
            QQ(mm,nn) = 0.0;
            for(int jr = 0; jr < occMax_irrep_compact; jr++)
            {
                int Jirrep = compact2all(jr);
                double twojP1 = irrep_list(Jirrep).two_j+1;
                MatrixXd den_tc = den_c(Jirrep) + f_NM*den_o(Jirrep);
                MatrixXd den_to = den_c(Jirrep) + (NN-1.0)/(MM-1.0)*den_o(Jirrep);
                int size_tmp2 = irrep_list(Jirrep).size;
                for(int aa = 0; aa < size_tmp2; aa++)
                for(int bb = 0; bb < size_tmp2; bb++)
                {
                    int emn = mm*size+nn, eab = aa*size_tmp2+bb, emb = mm*size_tmp2+bb, ean = aa*size+nn;
                    fock_c(mm,nn) += twojP1*den_tc(aa,bb) * h2eLLLL_JK.J[ir][jr][emn][eab];
                    fock_o(mm,nn) += twojP1*den_to(aa,bb) * h2eLLLL_JK.J[ir][jr][emn][eab];
                    QQ(mm,nn) += twojP1*den_o(Jirrep)(aa,bb) * h2eLLLL_JK.J[ir][jr][emn][eab];
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
                fock_c(mm,nn) += f_NM/(MM-1.0) * den_o(Iirrep)(rr,ss) * (overlap(Iirrep)(mm,ss)*QQ(rr,nn) + QQ(mm,ss)*overlap(Iirrep)(rr,nn));
                fock_o(mm,nn) += 1.0/(MM-1.0) * den_c(Iirrep)(rr,ss) * (overlap(Iirrep)(mm,ss)*QQ(rr,nn) + QQ(mm,ss)*overlap(Iirrep)(rr,nn));
            }
            fock_c(nn,mm) = fock_c(mm,nn);
            fock_o(nn,mm) = fock_o(mm,nn);
        }
    }
    else
    {
        MatrixXd QQ = MatrixXd::Zero(size*2,size*2);
        fock_c.resize(size*2,size*2);
        fock_o.resize(size*2,size*2);
        #pragma omp parallel  for
        for(int mm = 0; mm < size; mm++)
        for(int nn = 0; nn <= mm; nn++)
        {
            fock_c(mm,nn) = h1e_4c(Iirrep)(mm,nn);
            fock_c(mm+size,nn) = h1e_4c(Iirrep)(mm+size,nn);
            fock_c(mm+size,nn+size) = h1e_4c(Iirrep)(mm+size,nn+size);
            fock_o(mm,nn) = h1e_4c(Iirrep)(mm,nn);
            fock_o(mm+size,nn) = h1e_4c(Iirrep)(mm+size,nn);
            fock_o(mm+size,nn+size) = h1e_4c(Iirrep)(mm+size,nn+size);
            if(mm != nn) 
            {
                fock_c(nn+size,mm) = h1e_4c(Iirrep)(nn+size,mm);
                fock_o(nn+size,mm) = h1e_4c(Iirrep)(nn+size,mm);
            }
            
            for(int jr = 0; jr < occMax_irrep_compact; jr++)
            {
                int Jirrep = compact2all(jr);
                double twojP1 = irrep_list(Jirrep).two_j+1;
                int size_tmp2 = irrep_list(Jirrep).size;
                MatrixXd den_tc = den_c(Jirrep) + f_NM*den_o(Jirrep);
                MatrixXd den_to = den_c(Jirrep) + (NN-1.0)/(MM-1.0)*den_o(Jirrep);
                for(int ss = 0; ss < size_tmp2; ss++)
                for(int rr = 0; rr < size_tmp2; rr++)
                {
                    int emn = mm*size+nn, esr = ss*size_tmp2+rr, emr = mm*size_tmp2+rr, esn = ss*size+nn;
                    fock_c(mm,nn) += twojP1*den_tc(ss,rr) * h2eLLLL_JK.J[ir][jr][emn][esr] + twojP1*den_tc(size_tmp2+ss,size_tmp2+rr) * h2eSSLL_JK.J[jr][ir][esr][emn];
                    fock_c(mm+size,nn) -= twojP1*den_tc(ss,size_tmp2+rr) * h2eSSLL_JK.K[ir][jr][emr][esn];
                    fock_c(mm+size,nn+size) += twojP1*den_tc(size_tmp2+ss,size_tmp2+rr) * h2eSSSS_JK.J[ir][jr][emn][esr] + twojP1*den_tc(ss,rr) * h2eSSLL_JK.J[ir][jr][emn][esr];

                    fock_o(mm,nn) += twojP1*den_to(ss,rr) * h2eLLLL_JK.J[ir][jr][emn][esr] + twojP1*den_to(size_tmp2+ss,size_tmp2+rr) * h2eSSLL_JK.J[jr][ir][esr][emn];
                    fock_o(mm+size,nn) -= twojP1*den_to(ss,size_tmp2+rr) * h2eSSLL_JK.K[ir][jr][emr][esn];
                    fock_o(mm+size,nn+size) += twojP1*den_to(size_tmp2+ss,size_tmp2+rr) * h2eSSSS_JK.J[ir][jr][emn][esr] + twojP1*den_to(ss,rr) * h2eSSLL_JK.J[ir][jr][emn][esr];

                    QQ(mm,nn) += twojP1*den_o(Jirrep)(ss,rr) * h2eLLLL_JK.J[ir][jr][emn][esr] + twojP1*den_o(Jirrep)(size_tmp2+ss,size_tmp2+rr) * h2eSSLL_JK.J[jr][ir][esr][emn];
                    QQ(mm+size,nn) -= twojP1*den_o(Jirrep)(ss,size_tmp2+rr) * h2eSSLL_JK.K[ir][jr][emr][esn];
                    QQ(mm+size,nn+size) += twojP1*den_o(Jirrep)(size_tmp2+ss,size_tmp2+rr) * h2eSSSS_JK.J[ir][jr][emn][esr] + twojP1*den_o(Jirrep)(ss,rr) * h2eSSLL_JK.J[ir][jr][emn][esr];
                    
                    if(mm != nn) 
                    {
                        int enr = nn*size_tmp2+rr, esm = ss*size+mm;
                        fock_c(nn+size,mm) -= twojP1*den_tc(ss,size_tmp2+rr) * h2eSSLL_JK.K[ir][jr][enr][esm];
                        fock_o(nn+size,mm) -= twojP1*den_to(ss,size_tmp2+rr) * h2eSSLL_JK.K[ir][jr][enr][esm];
                        QQ(nn+size,mm) -= twojP1*den_o(Jirrep)(ss,size_tmp2+rr) * h2eSSLL_JK.K[ir][jr][enr][esm];
                    }

                    if(with_gaunt)
                    {
                        int enm = nn*size+mm, ers = rr*size_tmp2+ss, erm = rr*size+mm, ens = nn*size_tmp2+ss;

                        fock_c(mm,nn) -= twojP1*den_tc(size_tmp2+ss,size_tmp2+rr) * gauntLSSL_JK.K[ir][jr][emr][esn];
                        fock_c(mm+size,nn) += twojP1*den_tc(ss+size_tmp2,rr)*gauntLSLS_JK.J[ir][jr][enm][ers] + twojP1*den_tc(ss,size_tmp2+rr) * gauntLSSL_JK.J[jr][ir][esr][emn];
                        fock_c(mm+size,nn+size) -= twojP1*den_tc(ss,rr) * gauntLSSL_JK.K[jr][ir][esn][emr];

                        fock_o(mm,nn) -= twojP1*den_to(size_tmp2+ss,size_tmp2+rr) * gauntLSSL_JK.K[ir][jr][emr][esn];       
                        fock_o(mm+size,nn) += twojP1*den_to(ss+size_tmp2,rr)*gauntLSLS_JK.J[ir][jr][enm][ers] + twojP1*den_to(ss,size_tmp2+rr) * gauntLSSL_JK.J[jr][ir][esr][emn];              
                        fock_o(mm+size,nn+size) -= twojP1*den_to(ss,rr) * gauntLSSL_JK.K[jr][ir][esn][emr];

                        QQ(mm,nn) -= twojP1*den_o(Jirrep)(size_tmp2+ss,size_tmp2+rr) * gauntLSSL_JK.K[ir][jr][emr][esn];
                        QQ(mm+size,nn) += twojP1*den_o(Jirrep)(size_tmp2+ss,rr)*gauntLSLS_JK.J[ir][jr][enm][ers] + twojP1*den_o(Jirrep)(ss,size_tmp2+rr) * gauntLSSL_JK.J[jr][ir][esr][emn];                        
                        QQ(mm+size,nn+size) -= twojP1*den_o(Jirrep)(ss,rr) * gauntLSSL_JK.K[jr][ir][esn][emr];

                        if(mm != nn)
                        {
                            int ern = rr*size+nn, ems = mm*size_tmp2+ss;
                            fock_c(nn+size,mm) += twojP1*den_tc(size_tmp2+ss,rr)*gauntLSLS_JK.J[ir][jr][emn][ers] + twojP1*den_tc(ss,size_tmp2+rr) * gauntLSSL_JK.J[jr][ir][esr][enm];
                            fock_o(nn+size,mm) += twojP1*den_to(size_tmp2+ss,rr)*gauntLSLS_JK.J[ir][jr][emn][ers] + twojP1*den_to(ss,size_tmp2+rr) * gauntLSSL_JK.J[jr][ir][esr][enm];
                            QQ(nn+size,mm) += twojP1*den_o(Jirrep)(size_tmp2+ss,rr)*gauntLSLS_JK.J[ir][jr][emn][ers] + twojP1*den_o(Jirrep)(ss,size_tmp2+rr) * gauntLSSL_JK.J[jr][ir][esr][enm];
                        }
                    }
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
                fock_c(mm,nn) += f_NM/(MM-1.0) * den_o(Iirrep)(rr,ss) * (overlap_4c(Iirrep)(mm,ss)*QQ(rr,nn) + QQ(mm,ss)*overlap_4c(Iirrep)(rr,nn));
                fock_o(mm,nn) += 1.0/(MM-1.0) * den_c(Iirrep)(rr,ss) * (overlap_4c(Iirrep)(mm,ss)*QQ(rr,nn) + QQ(mm,ss)*overlap_4c(Iirrep)(rr,nn));
            }
            fock_c(nn,mm) = fock_c(mm,nn);
            fock_o(nn,mm) = fock_o(mm,nn);
        }
    }
}


vMatrixXd DHF_SPH_CA::get_amfi_unc(INT_SPH& int_sph_, const bool& twoC, const string& Xmethod, bool amfi_with_gaunt, bool amfi_with_gauge)
{
    cout << "Running DHF_SPH_CA::get_amfi_unc" << endl;
    if(with_gaunt && !amfi_with_gaunt)
    {
        cout << endl << "ATTENTION! Since gaunt terms are included in SCF, they are automatically calculated in amfi integrals." << endl << endl;
        amfi_with_gaunt = true;
        if(with_gauge && !amfi_with_gauge)
            amfi_with_gauge = true;
    }
    if(!with_gaunt && amfi_with_gaunt || twoC)
    {
        StartTime = clock();
        int_sph_.get_h2e_JK_gaunt_direct(gauntLSLS_JK,gauntLSSL_JK);
        EndTime = clock();
        cout << "2e-integral-Gaunt finished in " << (EndTime - StartTime) / (double)CLOCKS_PER_SEC << " seconds." << endl << endl; 
        if(amfi_with_gauge)
        {
            int2eJK tmp1, tmp2, tmp3, tmp4;
            int_sph_.get_h2e_JK_gauge_direct(tmp1,tmp2);
            for(int ir = 0; ir < Nirrep_compact; ir++)
            for(int jr = 0; jr < Nirrep_compact; jr++)
            {
                int size_i = irrep_list(compact2all(ir)).size, size_j = irrep_list(compact2all(jr)).size;
                for(int mm = 0; mm < size_i; mm++)
                for(int nn = 0; nn < size_i; nn++)
                for(int ss = 0; ss < size_j; ss++)
                for(int rr = 0; rr < size_j; rr++)
                {
                    int emn = mm*size_i+nn, esr = ss*size_j+rr, emr = mm*size_j+rr, esn = ss*size_i+nn;
                    gauntLSLS_JK.J[ir][jr][emn][esr] -= tmp1.J[ir][jr][emn][esr];
                    gauntLSLS_JK.K[ir][jr][emr][esn] -= tmp1.K[ir][jr][emr][esn];
                    gauntLSSL_JK.J[ir][jr][emn][esr] -= tmp2.J[ir][jr][emn][esr];
                    gauntLSSL_JK.K[ir][jr][emr][esn] -= tmp2.K[ir][jr][emr][esn];
                }
            }
            EndTime = clock();
            cout << "2e-integral-gauge finished in " << (EndTime - StartTime) / (double)CLOCKS_PER_SEC << " seconds." << endl << endl;  
        }
        symmetrize_JK_gaunt(gauntLSLS_JK,Nirrep_compact);
        if(renormalizedSmall)
        {
            renormalize_h2e(gauntLSLS_JK,"LSLS");
            renormalize_h2e(gauntLSSL_JK,"LSSL");
        }
        
    }
    if(amfi_with_gauge && !amfi_with_gaunt)
    {
        cout << "ERROR: When gauge term is included, the Gaunt term must be included." << endl;
        exit(99);
    }
    int2eJK gauntLSLS_SD, gauntLSSL_SD;
    if(amfi_with_gaunt)
    {
        /* Enable SD gaunt */ 
        // int_sph_.get_h2e_JK_gaunt_direct(gauntLSLS_SD,gauntLSSL_SD,-1,true);
        // if(renormalizedSmall)
        // {
        //     renormalize_h2e(gauntLSLS_SD,"LSLS");
        //     renormalize_h2e(gauntLSSL_SD,"LSSL");
        // }
        // symmetrize_JK_gaunt(gauntLSLS_SD,Nirrep_compact);
        // for(int ir = 0; ir < Nirrep_compact; ir++)
        // for(int jr = 0; jr < Nirrep_compact; jr++)
        // {
        //     int size_i = irrep_list(compact2all(ir)).size, size_j = irrep_list(compact2all(jr)).size;
        //     #pragma omp parallel  for
        //     for(int nn = 0; nn < size_i*size_i*size_j*size_j; nn++)
        //     {
        //         int kk = nn / (size_j * size_j), ll = nn - kk*size_j*size_j;
        //         gauntLSLS_SD.J[ir][jr][kk][ll] = gauntLSLS_JK.J[ir][jr][kk][ll] - gauntLSLS_SD.J[ir][jr][kk][ll];
        //         gauntLSSL_SD.J[ir][jr][kk][ll] = gauntLSSL_JK.J[ir][jr][kk][ll] - gauntLSSL_SD.J[ir][jr][kk][ll];
        //         kk = nn / (size_i * size_j), ll = nn - kk*size_i*size_j;
        //         gauntLSLS_SD.K[ir][jr][kk][ll] = gauntLSLS_JK.K[ir][jr][kk][ll] - gauntLSLS_SD.K[ir][jr][kk][ll];
        //         gauntLSSL_SD.K[ir][jr][kk][ll] = gauntLSSL_JK.K[ir][jr][kk][ll] - gauntLSSL_SD.K[ir][jr][kk][ll];
        //     }
        // }
        gauntLSLS_SD = gauntLSLS_JK;
        gauntLSSL_SD = gauntLSSL_JK;
    }
    int2eJK SSLL_SD, SSSS_SD;
    int_sph_.get_h2eSD_JK_direct(SSLL_SD, SSSS_SD);
    symmetrize_JK(SSSS_SD,Nirrep_compact);
    if(renormalizedSmall)
    {
        renormalize_h2e(SSLL_SD,"SSLL");
        renormalize_h2e(SSSS_SD,"SSSS");
    }
    if(twoC)
    {
        return get_amfi_unc_2c(SSLL_SD, SSSS_SD, amfi_with_gaunt);
    }
    else 
    {
        if(occMax_irrep < Nirrep && Xmethod == "fullFock")
        {
            cout << "fullFock is used in amfi function with incomplete h2e." << endl;
            cout << "Recalculate h2e and gaunt2e..." << endl;
            StartTime = clock();
            int_sph_.get_h2e_JK_direct(h2eLLLL_JK,h2eSSLL_JK,h2eSSSS_JK);
            symmetrize_JK(h2eLLLL_JK,Nirrep_compact);
            symmetrize_JK(h2eSSSS_JK,Nirrep_compact);
            if(renormalizedSmall)
            {
                renormalize_h2e(h2eSSLL_JK,"SSLL");
                renormalize_h2e(h2eSSSS_JK,"SSSS");
            }
            EndTime = clock();
            cout << "Complete 2e-integral finished in " << (EndTime - StartTime) / (double)CLOCKS_PER_SEC << " seconds." << endl << endl; 
        }
        vMatrixXd density_t(occMax_irrep);
        for(int ir = 0;ir < occMax_irrep; ir++)
        {
            density_t(ir) = density(ir) + f_NM*density_o(ir);
        }
        return DHF_SPH::get_amfi_unc(SSLL_SD, SSSS_SD, gauntLSLS_SD, gauntLSSL_SD, density_t, Xmethod, amfi_with_gaunt);
    }
}


vMatrixXd DHF_SPH_CA::get_amfi_unc_2c(const int2eJK& h2eSSLL_SD, const int2eJK& h2eSSSS_SD, const bool& amfi_with_gaunt)
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
        density_tmp(ir) = evaluateDensity_core(coeff_tmp(ir),occNumber(ir),false) + f_NM*evaluateDensity_core(coeff_tmp(ir),occNumberOpen(ir),false);
    }

    for(int ir = 0; ir < Nirrep; ir++)
    {
        int ir_c = all2compact(ir);
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
                int jr_c = all2compact(jr);
                int size_tmp2 = irrep_list(jr).size;
                for(int ss = 0; ss < size_tmp2; ss++)
                for(int rr = 0; rr < size_tmp2; rr++)
                {
                    int emn = mm*size_tmp+nn, esr = ss*size_tmp2+rr, emr = mm*size_tmp2+rr, esn = ss*size_tmp+nn;
                    SO_4c(mm,nn) += density_tmp(jr)(size_tmp2+ss,size_tmp2+rr) * h2eSSLL_SD.J[jr_c][ir_c][esr][emn];
                    SO_4c(mm+size_tmp,nn) -= density_tmp(jr)(ss,size_tmp2+rr) * h2eSSLL_SD.K[ir_c][jr_c][emr][esn];
                    if(mm != nn) 
                    {
                        int enr = nn*size_tmp2+rr, esm = ss*size_tmp+mm;
                        SO_4c(nn+size_tmp,mm) -= density_tmp(jr)(ss,size_tmp2+rr) * h2eSSLL_SD.K[ir_c][jr_c][enr][esm];
                    }
                    SO_4c(mm+size_tmp,nn+size_tmp) += density_tmp(jr)(size_tmp2+ss,size_tmp2+rr) * h2eSSSS_SD.J[ir_c][jr_c][emn][esr] + density_tmp(jr)(ss,rr) * h2eSSLL_SD.J[ir_c][jr_c][emn][esr];
                    if(amfi_with_gaunt)
                    {
                        int enm = nn*size_tmp+mm, ers = rr*size_tmp2+ss, erm = rr*size_tmp+mm, ens = nn*size_tmp2+ss;
                        SO_4c(mm,nn) -= density_tmp(jr)(size_tmp2+ss,size_tmp2+rr) * gauntLSSL_JK.K[ir_c][jr_c][emr][esn];
                        SO_4c(mm+size_tmp,nn+size_tmp) -= density_tmp(jr)(ss,rr) * gauntLSSL_JK.K[jr_c][ir_c][esn][emr];
                        SO_4c(mm+size_tmp,nn) += density_tmp(jr)(size_tmp2+ss,rr)*gauntLSLS_JK.J[ir_c][jr_c][enm][ers] + density_tmp(jr)(ss,size_tmp2+rr) * gauntLSSL_JK.J[jr_c][ir_c][esr][emn];
                        if(mm != nn) 
                        {
                            int ern = rr*size_tmp+nn, ems = mm*size_tmp2+ss;
                            SO_4c(nn+size_tmp,mm) += density_tmp(jr)(size_tmp2+ss,rr)*gauntLSLS_JK.J[ir_c][jr_c][emn][ers] + density_tmp(jr)(ss,size_tmp2+rr) * gauntLSSL_JK.J[jr_c][ir_c][esr][enm];
                        }
                    }
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

    X_calculated = true;
    return amfi_unc;
}



/*
    basisGenerator to generate relativistic (j-adapted) contracted basis sets.
    WARNING: This method was implemented for CFOUR format of basis sets only!
*/
void DHF_SPH_CA::basisGenerator(string basisName, string filename, const INT_SPH& intor, const INT_SPH& intorAll, const bool& sf, const string& tag)
{
    cout << "Running DHF_SPH_CA::basisGenerator" << endl;
    Matrix<VectorXi,-1,1> basisInfo, basisSmall, basisAll;
    basisSmall.resize(intor.shell_list.rows());
    vMatrixXd resortedCoeffInput(basisSmall.rows());
    for(int ll = 0; ll < basisSmall.rows(); ll++)
    {
        basisSmall(ll).resize(3);
        basisSmall(ll)(0) = ll;
        basisSmall(ll)(1) = intor.shell_list(ll).coeff.cols();
        basisSmall(ll)(2) = intor.shell_list(ll).coeff.rows();
        /*  
            Reorganize coeff in intor
            make sure all the contracted coefficients are put to the left 
        */
        vector<int> vec_i;
        for(int ii = 0; ii < intor.shell_list(ll).coeff.cols(); ii++)
        {
            if(abs(intor.shell_list(ll).coeff(0,ii)) >= 1e-12)
                vec_i.push_back(ii);
        }
        for(int ii = 0; ii < intor.shell_list(ll).coeff.cols(); ii++)
        {
            if(abs(intor.shell_list(ll).coeff(0,ii)) < 1e-12)
                vec_i.push_back(ii);
        }
        MatrixXd tmp;
        tmp = MatrixXd::Zero(intor.shell_list(ll).coeff.rows(),intor.shell_list(ll).coeff.cols());
        for(int ii = 0; ii < intor.shell_list(ll).coeff.cols(); ii++)
        for(int jj = 0; jj < intor.shell_list(ll).coeff.rows(); jj++)
        {
            tmp(jj,ii) = intor.shell_list(ll).coeff(jj,vec_i[ii]);
        }
        resortedCoeffInput(ll) = tmp;
    }
    basisAll.resize(intorAll.shell_list.rows());
    for(int ll = 0; ll < basisAll.rows(); ll++)
    {
        basisAll(ll).resize(3);
        basisAll(ll)(0) = ll;
        basisAll(ll)(1) = intorAll.shell_list(ll).coeff.cols();
        basisAll(ll)(2) = intorAll.shell_list(ll).coeff.rows();
    }

    // Count how many l-shells containing electrons to resize basisInfo
    int occL = 0;
    for(int ir = 0; ir < irrep_list.rows(); ir += 4*irrep_list(ir).l+2)
    {
        if(occNumberCore(ir).rows() == 0) break;
        occL++;
    }
    basisInfo.resize(occL);
    
    // For each l-shell, count how many orbitals are fully or partially occupied
    // and resize basisInfo(l)
    occL = 0;
    for(int ir = 0; ir < irrep_list.rows(); ir += 4*irrep_list(ir).l+2)
    {
        int occN = 0;
        if(occNumberCore(ir).rows() == 0) break;
        for(int ii = 0; ii < occNumberCore(ir).rows(); ii++)
        {
            if(abs(occNumberCore(ir)(ii)-1) < 1e-4 || abs(occNumberOpen(ir)(ii)-1) < 1e-4)
                occN++;
        }
        basisInfo(occL).resize(occN);
        occL++;
    }

    /* 
        Construct the final contraction coefficients
        
        The contraction coefficients were firsly obtained using HF with intor.
        They were combined with other basis functions to form the coeff_final, including
            other decontracted functions in intor, stored in resortedCoeffInput
            extra diffuse or core-correlating functions in intorAll
        
        The linearly dependent core-correlating functions are replaced by the basis function
        with the closet alpha.

        For j-adapted basis, the number of contracted basis sets doubles for l >= 1.
    */
    vMatrixXd coeff_final(intorAll.shell_list.rows());
    Matrix<vector<double>,-1,1> exp_a_final(intorAll.shell_list.rows());
    for(int ir = 0; ir < intorAll.irrep_list.rows(); ir += 4*intorAll.irrep_list(ir).l+2)
    {
        // nLD is the number of linearly dependent basis functions in this shell whose alpha value
        // is close to that of another basis function with a threshold of 1.25 
        int ll = intorAll.irrep_list(ir).l, nLD = 0;
        // nPVXZ is the number of primitive Gaussian functions in original basis set
        // and is likely to be smaller than ll, which comes from the basisAll
        int nPVXZ = (ll < intor.shell_list.rows() ? intor.shell_list(ll).exp_a.rows() : 0);
        int nAll = intorAll.shell_list(ll).exp_a.rows();
        vector<int> n_closest;
        for(int ii = 0; ii < nPVXZ; ii++)
            exp_a_final(ll).push_back(intor.shell_list(ll).exp_a(ii));

        // Here we assume that the first nPVXZ basis functions are the same in intor and intorAll
        for(int ii = nPVXZ; ii < nAll; ii++)
        {
            // tmp_i'th function is the linearly dependent core-correlating function
            int tmp_i;
            double closest = 100000000000.0;
            double alpha = intorAll.shell_list(ll).exp_a(ii);
            bool LD = false;
            for(int jj = 0; jj < nPVXZ; jj++)
            {
                double tmp = max(alpha/intor.shell_list(ll).exp_a(jj),intor.shell_list(ll).exp_a(jj)/alpha);
                if(tmp < 1.25)
                    LD = true;
                if(tmp < closest)
                {
                    closest = tmp;
                    tmp_i = jj;
                }
            }
            // Add the basis function only if it is not liearly dependent with others
            if(!LD)
            {
                exp_a_final(ll).push_back(alpha);
            } 
            else
            {
                nLD++;
                n_closest.push_back(tmp_i);
            } 
        }
        
        if(ll < occL)
        {
            if(sf || ll == 0)
            {
                // (exp_a_final(ll).size() - nPVXZ) is the number of extra linearly independent Gaussian functions
                coeff_final(ll) = MatrixXd::Zero(exp_a_final(ll).size(), basisSmall(ll)(1) + nLD + exp_a_final(ll).size() - nPVXZ);
                for(int ii = 0; ii < coeff(ir).rows(); ii++)
                {
                    for(int jj = 0; jj < basisInfo(ll).rows(); jj++)
                        coeff_final(ll)(ii,jj) = coeff(ir)(ii,jj);
                    for(int jj = basisInfo(ll).rows(); jj < basisSmall(ll)(1); jj++)
                        coeff_final(ll)(ii,jj) = resortedCoeffInput(ll)(ii,jj);
                }
                for(int jj = 0; jj < nLD; jj++)
                    coeff_final(ll)(n_closest[jj],basisSmall(ll)(1)+jj) = 1.0;
                for(int ii = 0; ii < exp_a_final(ll).size() - nPVXZ; ii++)
                    coeff_final(ll)(nPVXZ+ii,basisSmall(ll)(1)+nLD+ii) = 1.0;
            }
            else
            {
                // (exp_a_final(ll).size() - nPVXZ) is the number of extra linearly independent Gaussian functions
                coeff_final(ll) = MatrixXd::Zero(exp_a_final(ll).size(), basisSmall(ll)(1) + basisInfo(ll).rows() + nLD + exp_a_final(ll).size() - nPVXZ);
                for(int ii = 0; ii < coeff(ir).rows(); ii++)
                {
                    for(int jj = 0; jj < basisInfo(ll).rows(); jj++)
                        coeff_final(ll)(ii,jj) = coeff(ir)(ii,jj);
                    for(int jj = 0; jj < basisInfo(ll).rows(); jj++)
                        coeff_final(ll)(ii,jj+basisInfo(ll).rows()) = coeff(ir + irrep_list(ir).two_j + 1)(ii,jj);
                    for(int jj = 2*basisInfo(ll).rows(); jj < basisSmall(ll)(1) + basisInfo(ll).rows(); jj++)
                        coeff_final(ll)(ii,jj) = resortedCoeffInput(ll)(ii,jj - basisInfo(ll).rows());
                }
                for(int jj = 0; jj < nLD; jj++)
                    coeff_final(ll)(n_closest[jj],basisSmall(ll)(1)+basisInfo(ll).rows()+jj) = 1.0;
                for(int jj = 0; jj < exp_a_final(ll).size() - nPVXZ; jj++)
                    coeff_final(ll)(nPVXZ+jj,basisSmall(ll)(1)+basisInfo(ll).rows()+nLD+jj) = 1.0;
            }
            
        }
        else
        {
            coeff_final(ll) = MatrixXd::Identity(exp_a_final(ll).size(),exp_a_final(ll).size());
        }
    }
    
    int pos = basisName.find("-X2C");
    if(pos != string::npos)
        basisName.erase(pos,4);
    pos = basisName.find("-DK3");
    if(pos != string::npos)
        basisName.erase(pos,4);
    pos = basisName.find("_X2C");
    if(pos != string::npos)
        basisName.erase(pos,4);
    pos = basisName.find("_DK3");
    if(pos != string::npos)
        basisName.erase(pos,4);
    ofstream ofs;
    ofs.open(filename,std::ofstream::app);
        ofs << basisName + tag << endl;
        ofs << "obtained from atomic calculation" << endl;
        ofs << endl;
        ofs << intorAll.shell_list.rows() << endl;
        for(int jj = 0; jj < intorAll.shell_list.rows(); jj++)
        {
            ofs << "    " << jj;
        }
        ofs << endl;
        for(int jj = 0; jj < intorAll.shell_list.rows(); jj++)
        {
            ofs << "    " << coeff_final(jj).cols();
        }
        ofs << endl;
        for(int jj = 0; jj < intorAll.shell_list.rows(); jj++)
        {
            ofs << "    " << coeff_final(jj).rows();
        }
        ofs << endl;
        ofs << fixed << setprecision(8);
        for(int ll = 0; ll < intorAll.shell_list.rows(); ll++)
        {
            for(int ii = 0; ii < exp_a_final(ll).size(); ii++)
            {
                if((ii+1) %5 == 1)  ofs << endl;
                ofs << "    " << exp_a_final(ll)[ii];
            }
            ofs << endl;
            ofs << endl;
            ofs << coeff_final(ll) << endl;
        }
        ofs << endl << endl;
    ofs.close();

    return;
}