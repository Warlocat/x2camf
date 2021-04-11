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

DHF_SPH::DHF_SPH(INT_SPH& int_sph_, const string& filename):
irrep_list(int_sph_.irrep_list)
{
    cout << "Initializing Dirac-HF for " << int_sph_.atomName << endl;
    Nirrep = int_sph_.irrep_list.rows();
    size_basis_spinor = int_sph_.size_gtou_spinor;

    occNumber.resize(Nirrep);
    occMax_irrep = 0;
    readOCC(filename);
    nelec = 0;
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
    int_sph_.get_h1e_direct(overlap,kinetic,Vnuc,WWW);
    EndTime = clock();
    cout << "1e-integral finished in " << (EndTime - StartTime) / (double)CLOCKS_PER_SEC << " seconds." << endl;

    StartTime = clock();
    int_sph_.get_h2e_JK_direct(h2eLLLL_JK,h2eSSLL_JK,h2eSSSS_JK,irrep_list(occMax_irrep-1).l);
    EndTime = clock();
    cout << "2e-integral finished in " << (EndTime - StartTime) / (double)CLOCKS_PER_SEC << " seconds." << endl << endl; 
    /*
        overlap_4c = [[S, 0], [0, T/2c^2]]
        h1e_4c = [[V, T], [T, W/4c^2 - T]]
    */
    fock_4c.resize(occMax_irrep);
    h1e_4c.resize(occMax_irrep);
    overlap_4c.resize(occMax_irrep);
    overlap_half_i_4c.resize(occMax_irrep);
    density.resize(occMax_irrep);
    coeff.resize(occMax_irrep);
    ene_orb.resize(occMax_irrep);
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


DHF_SPH::~DHF_SPH()
{
}

void DHF_SPH::eigensolverG_irrep(const vMatrixXd& inputM, const vMatrixXd& s_h_i, vVectorXd& values, vMatrixXd& vectors)
{
    for(int ii = 0; ii < occMax_irrep; ii++)
        eigensolverG(inputM(ii),s_h_i(ii),values(ii),vectors(ii));
    return;
}
double DHF_SPH::evaluateChange_irrep(const vMatrixXd& M1, const vMatrixXd& M2)
{
    VectorXd vecd_tmp(occMax_irrep);
    for(int ii = 0; ii < occMax_irrep; ii++)
    {
        vecd_tmp(ii) = evaluateChange(M1(ii),M2(ii));
    }
    return vecd_tmp.maxCoeff();
}
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
            // for(int ir = 0; ir < occMax_irrep; ir++)
            // {
            //     int tmp_size = fock4DIIS[ir].size();
            //     MatrixXd B4DIIS(tmp_size+1,tmp_size+1);
            //     VectorXd vec_b(tmp_size+1);
            //     for(int ii = 0; ii < tmp_size; ii++)
            //     {    
            //         for(int jj = 0; jj <= ii; jj++)
            //         {
            //             B4DIIS(ii,jj) = (error4DIIS[ir][ii].adjoint()*error4DIIS[ir][jj])(0,0);
            //             B4DIIS(jj,ii) = B4DIIS(ii,jj);
            //         }
            //         B4DIIS(tmp_size, ii) = -1.0;
            //         B4DIIS(ii, tmp_size) = -1.0;
            //         vec_b(ii) = 0.0;
            //     }
            //     B4DIIS(tmp_size, tmp_size) = 0.0;
            //     vec_b(tmp_size) = -1.0;
            //     VectorXd C = B4DIIS.partialPivLu().solve(vec_b);
            //     fock_4c(ir) = MatrixXd::Zero(fock_4c(ir).rows(),fock_4c(ir).cols());
            //     for(int ii = 0; ii < tmp_size; ii++)
            //     {
            //         fock_4c(ir) += C(ii) * fock4DIIS[ir][ii];
            //     }
            // }
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
    
}

MatrixXd DHF_SPH::evaluateDensity_spinor(const MatrixXd& coeff_, const VectorXd& occNumber_)
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

vMatrixXd DHF_SPH::evaluateDensity_spinor_irrep()
{
    vMatrixXd den(occMax_irrep);
    for(int ir = 0; ir < occMax_irrep; ir++)
    {
        den(ir) = evaluateDensity_spinor(coeff(ir),occNumber(ir));
    }

    return den;
}


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