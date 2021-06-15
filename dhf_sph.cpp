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


DHF_SPH::DHF_SPH(INT_SPH& int_sph_, const string& filename, const bool& spinFree, const bool& sfx2c, const bool& with_gaunt_):
irrep_list(int_sph_.irrep_list), with_gaunt(with_gaunt_)
{
    cout << "Initializing Dirac-HF for " << int_sph_.atomName << " atom." << endl;
    Nirrep = int_sph_.irrep_list.rows();
    size_basis_spinor = int_sph_.size_gtou_spinor;

    occNumber.resize(Nirrep);
    occMax_irrep = 0;
    readOCC(filename, int_sph_.atomName);
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

    // Calculate the maximum memory cost for 2e integrals
    double numberOfDouble = 0;
    for(int ir = 0; ir < irrep_list.size(); ir++)
    for(int jr = 0; jr < irrep_list.size(); jr++)
    {
        numberOfDouble += 2.0*irrep_list(ir).size*irrep_list(ir).size*irrep_list(jr).size*irrep_list(jr).size;
    }
    numberOfDouble *= 5.0;
    if(with_gaunt)  numberOfDouble = numberOfDouble/5.0*7.0;
    cout << "Maximum memory cost (2e part) in SCF and amfi calculation: " << numberOfDouble*sizeof(double)/pow(1024.0,3) << " GB." << endl;

    StartTime = clock();
    overlap = int_sph_.get_h1e("overlap");
    kinetic = int_sph_.get_h1e("kinetic");
    Vnuc = int_sph_.get_h1e("nuc_attra");
    if(spinFree)
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
    cout << "2e-integral finished in " << (EndTime - StartTime) / (double)CLOCKS_PER_SEC << " seconds." << endl; 
    
    if(with_gaunt && !sfx2c)
    {
        StartTime = clock();
        //Always calculate all Gaunt integrals for amfi integrals
        int_sph_.get_h2e_JK_gaunt_direct(gauntLSLS_JK,gauntLSSL_JK);
        EndTime = clock();
        cout << "2e-integral-Gaunt finished in " << (EndTime - StartTime) / (double)CLOCKS_PER_SEC << " seconds." << endl << endl; 
    }
        
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
MatrixXd DHF_SPH::evaluateErrorDIIS(const MatrixXd& den_old, const MatrixXd& den_new)
{
    MatrixXd tmp = den_old - den_new;
    int size = den_old.rows();
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
void DHF_SPH::runSCF(const bool& twoC, vMatrixXd* initialGuess)
{
    vector<MatrixXd> error4DIIS[occMax_irrep], fock4DIIS[occMax_irrep];
    StartTime = clock();
    cout << endl;
    if(twoC) cout << "Start SFX2C-1e Hartree-Fock iterations..." << endl;
    else cout << "Start Dirac Hartree-Fock iterations..." << endl;
    cout << endl;
    vMatrixXd newDen;
    if(initialGuess == NULL)
    {
        eigensolverG_irrep(h1e_4c, overlap_half_i_4c, ene_orb, coeff);
        density = evaluateDensity_spinor_irrep(twoC);
    }
    else
    {
        density = *initialGuess;
    }

    for(int iter = 1; iter <= maxIter; iter++)
    {
        if(iter <= 2)
        {
            for(int ir = 0; ir < occMax_irrep; ir++)    
            {
                int size_tmp = irrep_list(ir).size;
                evaluateFock(fock_4c(ir),twoC,density,size_tmp,ir);
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
        newDen = evaluateDensity_spinor_irrep(twoC);
        d_density = evaluateChange_irrep(density, newDen);
        
        cout << "Iter #" << iter << " maximum density difference: " << d_density << endl;
        
        density = newDen;
        if(d_density < convControl) 
        {
            converged = true;
            cout << endl << "SCF converges after " << iter << " iterations." << endl << endl;

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
                        ene_scf += 0.5 * density(ir)(ii,jj) * (h1e_4c(ir)(jj,ii) + fock_4c(ir)(jj,ii));
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
            if(twoC) cout << "Final SFX2C-1e HF energy is " << setprecision(15) << ene_scf << " hartree." << endl;
            else cout << "Final DHF energy is " << setprecision(15) << ene_scf << " hartree." << endl;
            break;            
        }
        for(int ir = 0; ir < occMax_irrep; ir++)    
        {
            int size_tmp = irrep_list(ir).size;
            evaluateFock(fock_4c(ir),twoC,density,size_tmp,ir);
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
    evaluate Fock matrix 
*/
void DHF_SPH::evaluateFock(MatrixXd& fock, const bool& twoC, const vMatrixXd& den, const int& size, const int& Iirrep)
{
    if(!twoC)
    {
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
            fock(mm,nn) = h1e_4c(Iirrep)(mm,nn);
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

/* 
    Read occupation numbers 
*/
void DHF_SPH::readOCC(const string& filename, const string& atomName)
{
    string flags;
    VectorXd vecd_tmp = VectorXd::Zero(10);
    int int_tmp, int_tmp2, int_tmp3;
    ifstream ifs;
    ifs.open(filename);
    if(!ifs)
    {
        cout << "ERROR opening file " << filename << endl;
        exit(99);
    }
        while(!ifs.eof())
        {
            ifs >> flags;
            if(flags == "%occAMFI_" + atomName)
            {
                cout << "Found input occupation number for " << atomName << endl;
                ifs >> int_tmp;
                for(int ii = 0; ii < int_tmp; ii++)
                {
                    ifs >> vecd_tmp(ii);
                }
                break;
            }
        }
        if(ifs.eof())
        {
            cout << "Did NOT find %occAMFI in " << filename << endl;
            cout << "Using default occupation number for " << atomName << endl;
            if(atomName == "H") {int_tmp = 1; vecd_tmp(0) = 1.0;}
            else if(atomName == "HE") {int_tmp = 1; vecd_tmp(0) = 2.0;}
            else if(atomName == "LI") {int_tmp = 1; vecd_tmp(0) = 3.0;}
            else if(atomName == "BE") {int_tmp = 1; vecd_tmp(0) = 4.0;}
            else if(atomName == "B") {int_tmp = 3; vecd_tmp(0) = 4.0; vecd_tmp(1) = 1.0/3.0; vecd_tmp(2) = 2.0/3.0;}
            else if(atomName == "C") {int_tmp = 3; vecd_tmp(0) = 4.0; vecd_tmp(1) = 2.0/3.0; vecd_tmp(2) = 4.0/3.0;}
            else if(atomName == "N") {int_tmp = 3; vecd_tmp(0) = 4.0; vecd_tmp(1) = 1.0; vecd_tmp(2) = 2.0;}
            else if(atomName == "O") {int_tmp = 3; vecd_tmp(0) = 4.0; vecd_tmp(1) = 4.0/3.0; vecd_tmp(2) = 8.0/3.0;}
            else if(atomName == "F") {int_tmp = 3; vecd_tmp(0) = 4.0; vecd_tmp(1) = 5.0/3.0; vecd_tmp(2) = 10.0/3.0;}
            else if(atomName == "NE") {int_tmp = 3; vecd_tmp(0) = 4.0; vecd_tmp(1) = 2.0; vecd_tmp(2) = 4.0;}
            else if(atomName == "NA") {int_tmp = 3; vecd_tmp(0) = 5.0; vecd_tmp(1) = 2.0; vecd_tmp(2) = 4.0;}
            else if(atomName == "MG") {int_tmp = 3; vecd_tmp(0) = 6.0; vecd_tmp(1) = 2.0; vecd_tmp(2) = 4.0;}
            else if(atomName == "AL") {int_tmp = 3; vecd_tmp(0) = 6.0; vecd_tmp(1) = 7.0/3.0; vecd_tmp(2) = 14.0/3.0;}
            else if(atomName == "SI") {int_tmp = 3; vecd_tmp(0) = 6.0; vecd_tmp(1) = 8.0/3.0; vecd_tmp(2) = 16.0/3.0;}
            else if(atomName == "P") {int_tmp = 3; vecd_tmp(0) = 6.0; vecd_tmp(1) = 3.0; vecd_tmp(2) = 6.0;}
            else if(atomName == "S") {int_tmp = 3; vecd_tmp(0) = 6.0; vecd_tmp(1) = 10.0/3.0; vecd_tmp(2) = 20.0/3.0;}
            else if(atomName == "CL") {int_tmp = 3; vecd_tmp(0) = 6.0; vecd_tmp(1) = 11.0/3.0; vecd_tmp(2) = 22.0/3.0;}
            else if(atomName == "AR") {int_tmp = 3; vecd_tmp(0) = 6.0; vecd_tmp(1) = 4.0; vecd_tmp(2) = 8.0;}
            else if(atomName == "K") {int_tmp = 3; vecd_tmp(0) = 7.0; vecd_tmp(1) = 4.0; vecd_tmp(2) = 8.0;}
            else if(atomName == "CA") {int_tmp = 3; vecd_tmp(0) = 8.0; vecd_tmp(1) = 4.0; vecd_tmp(2) = 8.0;}
            else if(atomName == "SC") {int_tmp = 5; vecd_tmp(0) = 8.0; vecd_tmp(1) = 4.0; vecd_tmp(2) = 8.0; vecd_tmp(3) = 1.0/2.5; vecd_tmp(4) = 1.5/2.5;}
            else if(atomName == "TI") {int_tmp = 5; vecd_tmp(0) = 8.0; vecd_tmp(1) = 4.0; vecd_tmp(2) = 8.0; vecd_tmp(3) = 2.0/2.5; vecd_tmp(4) = 3.0/2.5;}
            else if(atomName == "V") {int_tmp = 5; vecd_tmp(0) = 8.0; vecd_tmp(1) = 4.0; vecd_tmp(2) = 8.0; vecd_tmp(3) = 3.0/2.5; vecd_tmp(4) = 4.5/2.5;}
            else if(atomName == "CR") {int_tmp = 5; vecd_tmp(0) = 7.0; vecd_tmp(1) = 4.0; vecd_tmp(2) = 8.0; vecd_tmp(3) = 2.0; vecd_tmp(4) = 3.0;}
            else if(atomName == "MN") {int_tmp = 5; vecd_tmp(0) = 8.0; vecd_tmp(1) = 4.0; vecd_tmp(2) = 8.0; vecd_tmp(3) = 2.0; vecd_tmp(4) = 3.0;}
            else if(atomName == "FE") {int_tmp = 5; vecd_tmp(0) = 8.0; vecd_tmp(1) = 4.0; vecd_tmp(2) = 8.0; vecd_tmp(3) = 6.0/2.5; vecd_tmp(4) = 9.0/2.5;}
            else if(atomName == "CO") {int_tmp = 5; vecd_tmp(0) = 8.0; vecd_tmp(1) = 4.0; vecd_tmp(2) = 8.0; vecd_tmp(3) = 7.0/2.5; vecd_tmp(4) = 10.5/2.5;}
            else if(atomName == "NI") {int_tmp = 5; vecd_tmp(0) = 8.0; vecd_tmp(1) = 4.0; vecd_tmp(2) = 8.0; vecd_tmp(3) = 8.0/2.5; vecd_tmp(4) = 12.0/2.5;}
            else if(atomName == "CU") {int_tmp = 5; vecd_tmp(0) = 7.0; vecd_tmp(1) = 4.0; vecd_tmp(2) = 8.0; vecd_tmp(3) = 4.0; vecd_tmp(4) = 6.0;}
            else if(atomName == "ZN") {int_tmp = 5; vecd_tmp(0) = 8.0; vecd_tmp(1) = 4.0; vecd_tmp(2) = 8.0; vecd_tmp(3) = 4.0; vecd_tmp(4) = 6.0;}
            else if(atomName == "GA") {int_tmp = 5; vecd_tmp(0) = 8.0; vecd_tmp(1) = 13.0/3.0; vecd_tmp(2) = 26.0/3.0; vecd_tmp(3) = 4.0; vecd_tmp(4) = 6.0;}
            else if(atomName == "GE") {int_tmp = 5; vecd_tmp(0) = 8.0; vecd_tmp(1) = 14.0/3.0; vecd_tmp(2) = 28.0/3.0; vecd_tmp(3) = 4.0; vecd_tmp(4) = 6.0;}
            else if(atomName == "AS") {int_tmp = 5; vecd_tmp(0) = 8.0; vecd_tmp(1) = 5.0; vecd_tmp(2) = 10.0; vecd_tmp(3) = 4.0; vecd_tmp(4) = 6.0;}
            else if(atomName == "SR") {int_tmp = 5; vecd_tmp(0) = 8.0; vecd_tmp(1) = 16.0/3.0; vecd_tmp(2) = 32.0/3.0; vecd_tmp(3) = 4.0; vecd_tmp(4) = 6.0;}
            else if(atomName == "BR") {int_tmp = 5; vecd_tmp(0) = 8.0; vecd_tmp(1) = 17.0/3.0; vecd_tmp(2) = 34.0/3.0; vecd_tmp(3) = 4.0; vecd_tmp(4) = 6.0;}
            else if(atomName == "KR") {int_tmp = 5; vecd_tmp(0) = 8.0; vecd_tmp(1) = 6.0; vecd_tmp(2) = 12.0; vecd_tmp(3) = 4.0; vecd_tmp(4) = 6.0;}
            else if(atomName == "RB") {int_tmp = 5; vecd_tmp(0) = 9.0; vecd_tmp(1) = 6.0; vecd_tmp(2) = 12.0; vecd_tmp(3) = 4.0; vecd_tmp(4) = 6.0;}
            else if(atomName == "SR") {int_tmp = 5; vecd_tmp(0) = 10.0; vecd_tmp(1) = 6.0; vecd_tmp(2) = 12.0; vecd_tmp(3) = 4.0; vecd_tmp(4) = 6.0;}
            else if(atomName == "Y") {int_tmp = 5; vecd_tmp(0) = 10.0; vecd_tmp(1) = 6.0; vecd_tmp(2) = 12.0; vecd_tmp(3) = 11.0/2.5; vecd_tmp(4) = 16.5/2.5;}
            else if(atomName == "ZR") {int_tmp = 5; vecd_tmp(0) = 10.0; vecd_tmp(1) = 6.0; vecd_tmp(2) = 12.0; vecd_tmp(3) = 12.0/2.5; vecd_tmp(4) = 18.0/2.5;}
            else if(atomName == "NB") {int_tmp = 5; vecd_tmp(0) = 9.0; vecd_tmp(1) = 6.0; vecd_tmp(2) = 12.0; vecd_tmp(3) = 14.0/2.5; vecd_tmp(4) = 21.0/2.5;}
            else if(atomName == "MO") {int_tmp = 5; vecd_tmp(0) = 9.0; vecd_tmp(1) = 6.0; vecd_tmp(2) = 12.0; vecd_tmp(3) = 6.0; vecd_tmp(4) = 9.0;}
            else if(atomName == "TC") {int_tmp = 5; vecd_tmp(0) = 10.0; vecd_tmp(1) = 6.0; vecd_tmp(2) = 12.0; vecd_tmp(3) = 6.0; vecd_tmp(4) = 9.0;}
            else if(atomName == "RU") {int_tmp = 5; vecd_tmp(0) = 9.0; vecd_tmp(1) = 6.0; vecd_tmp(2) = 12.0; vecd_tmp(3) = 17.0/2.5; vecd_tmp(4) = 25.5/2.5;}
            else if(atomName == "RH") {int_tmp = 5; vecd_tmp(0) = 9.0; vecd_tmp(1) = 6.0; vecd_tmp(2) = 12.0; vecd_tmp(3) = 18.0/2.5; vecd_tmp(4) = 27.0/2.5;}
            else if(atomName == "PD") {int_tmp = 5; vecd_tmp(0) = 8.0; vecd_tmp(1) = 6.0; vecd_tmp(2) = 12.0; vecd_tmp(3) = 8.0; vecd_tmp(4) = 12.0;}
            else if(atomName == "AG") {int_tmp = 5; vecd_tmp(0) = 9.0; vecd_tmp(1) = 6.0; vecd_tmp(2) = 12.0; vecd_tmp(3) = 8.0; vecd_tmp(4) = 12.0;}
            else if(atomName == "CD") {int_tmp = 5; vecd_tmp(0) = 10.0; vecd_tmp(1) = 6.0; vecd_tmp(2) = 12.0; vecd_tmp(3) = 8.0; vecd_tmp(4) = 12.0;}
            else if(atomName == "IN") {int_tmp = 5; vecd_tmp(0) = 10.0; vecd_tmp(1) = 19.0/3.0; vecd_tmp(2) = 38.0/3.0; vecd_tmp(3) = 8.0; vecd_tmp(4) = 12.0;}
            else if(atomName == "SN") {int_tmp = 5; vecd_tmp(0) = 10.0; vecd_tmp(1) = 20.0/3.0; vecd_tmp(2) = 40.0/3.0; vecd_tmp(3) = 8.0; vecd_tmp(4) = 12.0;}
            else if(atomName == "SB") {int_tmp = 5; vecd_tmp(0) = 10.0; vecd_tmp(1) = 21.0/3.0; vecd_tmp(2) = 42.0/3.0; vecd_tmp(3) = 8.0; vecd_tmp(4) = 12.0;}
            else if(atomName == "TE") {int_tmp = 5; vecd_tmp(0) = 10.0; vecd_tmp(1) = 22.0/3.0; vecd_tmp(2) = 44.0/3.0; vecd_tmp(3) = 8.0; vecd_tmp(4) = 12.0;}
            else if(atomName == "I") {int_tmp = 5; vecd_tmp(0) = 10.0; vecd_tmp(1) = 23.0/3.0; vecd_tmp(2) = 46.0/3.0; vecd_tmp(3) = 8.0; vecd_tmp(4) = 12.0;}
            else if(atomName == "XE") {int_tmp = 5; vecd_tmp(0) = 10.0; vecd_tmp(1) = 8.0; vecd_tmp(2) = 16.0; vecd_tmp(3) = 8.0; vecd_tmp(4) = 12.0;}
            else if(atomName == "CS") {int_tmp = 5; vecd_tmp(0) = 11.0; vecd_tmp(1) = 8.0; vecd_tmp(2) = 16.0; vecd_tmp(3) = 8.0; vecd_tmp(4) = 12.0;}
            else if(atomName == "BA") {int_tmp = 5; vecd_tmp(0) = 12.0; vecd_tmp(1) = 8.0; vecd_tmp(2) = 16.0; vecd_tmp(3) = 8.0; vecd_tmp(4) = 12.0;}
            else if(atomName == "LA") {int_tmp = 5; vecd_tmp(0) = 12.0; vecd_tmp(1) = 8.0; vecd_tmp(2) = 16.0; vecd_tmp(3) = 21.0/2.5; vecd_tmp(4) = 31.5/2.5;}
            else if(atomName == "CE") {int_tmp = 7; vecd_tmp(0) = 12.0; vecd_tmp(1) = 8.0; vecd_tmp(2) = 16.0; vecd_tmp(3) = 21.0/2.5; vecd_tmp(4) = 31.5/2.5; vecd_tmp(5) = 3.0/7.0; vecd_tmp(6) = 4.0/7.0;}
            else if(atomName == "PR") {int_tmp = 7; vecd_tmp(0) = 12.0; vecd_tmp(1) = 8.0; vecd_tmp(2) = 16.0; vecd_tmp(3) = 8.0; vecd_tmp(4) = 12.0; vecd_tmp(5) = 9.0/7.0; vecd_tmp(6) = 12.0/7.0;}
            else if(atomName == "ND") {int_tmp = 7; vecd_tmp(0) = 12.0; vecd_tmp(1) = 8.0; vecd_tmp(2) = 16.0; vecd_tmp(3) = 8.0; vecd_tmp(4) = 12.0; vecd_tmp(5) = 12.0/7.0; vecd_tmp(6) = 16.0/7.0;}
            else if(atomName == "PM") {int_tmp = 7; vecd_tmp(0) = 12.0; vecd_tmp(1) = 8.0; vecd_tmp(2) = 16.0; vecd_tmp(3) = 8.0; vecd_tmp(4) = 12.0; vecd_tmp(5) = 15.0/7.0; vecd_tmp(6) = 20.0/7.0;}
            else if(atomName == "SM") {int_tmp = 7; vecd_tmp(0) = 12.0; vecd_tmp(1) = 8.0; vecd_tmp(2) = 16.0; vecd_tmp(3) = 8.0; vecd_tmp(4) = 12.0; vecd_tmp(5) = 18.0/7.0; vecd_tmp(6) = 24.0/7.0;}
            else if(atomName == "EU") {int_tmp = 7; vecd_tmp(0) = 12.0; vecd_tmp(1) = 8.0; vecd_tmp(2) = 16.0; vecd_tmp(3) = 8.0; vecd_tmp(4) = 12.0; vecd_tmp(5) = 3.0; vecd_tmp(6) = 4.0;}
            else if(atomName == "GD") {int_tmp = 7; vecd_tmp(0) = 12.0; vecd_tmp(1) = 8.0; vecd_tmp(2) = 16.0; vecd_tmp(3) = 21.0/2.5; vecd_tmp(4) = 31.5/2.5; vecd_tmp(5) = 21.0/7.0; vecd_tmp(6) = 28.0/7.0;}
            else if(atomName == "TB") {int_tmp = 7; vecd_tmp(0) = 12.0; vecd_tmp(1) = 8.0; vecd_tmp(2) = 16.0; vecd_tmp(3) = 8.0; vecd_tmp(4) = 12.0; vecd_tmp(5) = 27.0/7.0; vecd_tmp(6) = 36.0/7.0;}
            else if(atomName == "DY") {int_tmp = 7; vecd_tmp(0) = 12.0; vecd_tmp(1) = 8.0; vecd_tmp(2) = 16.0; vecd_tmp(3) = 8.0; vecd_tmp(4) = 12.0; vecd_tmp(5) = 30.0/7.0; vecd_tmp(6) = 40.0/7.0;}
            else if(atomName == "HO") {int_tmp = 7; vecd_tmp(0) = 12.0; vecd_tmp(1) = 8.0; vecd_tmp(2) = 16.0; vecd_tmp(3) = 8.0; vecd_tmp(4) = 12.0; vecd_tmp(5) = 33.0/7.0; vecd_tmp(6) = 44.0/7.0;}
            else if(atomName == "ER") {int_tmp = 7; vecd_tmp(0) = 12.0; vecd_tmp(1) = 8.0; vecd_tmp(2) = 16.0; vecd_tmp(3) = 8.0; vecd_tmp(4) = 12.0; vecd_tmp(5) = 36.0/7.0; vecd_tmp(6) = 48.0/7.0;}
            else if(atomName == "TM") {int_tmp = 7; vecd_tmp(0) = 12.0; vecd_tmp(1) = 8.0; vecd_tmp(2) = 16.0; vecd_tmp(3) = 8.0; vecd_tmp(4) = 12.0; vecd_tmp(5) = 39.0/7.0; vecd_tmp(6) = 52.0/7.0;}
            else if(atomName == "YB") {int_tmp = 7; vecd_tmp(0) = 12.0; vecd_tmp(1) = 8.0; vecd_tmp(2) = 16.0; vecd_tmp(3) = 8.0; vecd_tmp(4) = 12.0; vecd_tmp(5) = 6.0; vecd_tmp(6) = 8.0;}
            else if(atomName == "LU") {int_tmp = 7; vecd_tmp(0) = 12.0; vecd_tmp(1) = 8.0; vecd_tmp(2) = 16.0; vecd_tmp(3) = 21.0/2.5; vecd_tmp(4) = 31.5/2.5; vecd_tmp(5) = 6.0; vecd_tmp(6) = 8.0;}
            else if(atomName == "HF") {int_tmp = 7; vecd_tmp(0) = 12.0; vecd_tmp(1) = 8.0; vecd_tmp(2) = 16.0; vecd_tmp(3) = 22.0/2.5; vecd_tmp(4) = 33.0/2.5; vecd_tmp(5) = 6.0; vecd_tmp(6) = 8.0;}
            else if(atomName == "TA") {int_tmp = 7; vecd_tmp(0) = 12.0; vecd_tmp(1) = 8.0; vecd_tmp(2) = 16.0; vecd_tmp(3) = 23.0/2.5; vecd_tmp(4) = 34.5/2.5; vecd_tmp(5) = 6.0; vecd_tmp(6) = 8.0;}
            else if(atomName == "W") {int_tmp = 7; vecd_tmp(0) = 12.0; vecd_tmp(1) = 8.0; vecd_tmp(2) = 16.0; vecd_tmp(3) = 24.0/2.5; vecd_tmp(4) = 36.0/2.5; vecd_tmp(5) = 6.0; vecd_tmp(6) = 8.0;}
            else if(atomName == "RE") {int_tmp = 7; vecd_tmp(0) = 12.0; vecd_tmp(1) = 8.0; vecd_tmp(2) = 16.0; vecd_tmp(3) = 10.0; vecd_tmp(4) = 15.0; vecd_tmp(5) = 6.0; vecd_tmp(6) = 8.0;}
            else if(atomName == "OS") {int_tmp = 7; vecd_tmp(0) = 12.0; vecd_tmp(1) = 8.0; vecd_tmp(2) = 16.0; vecd_tmp(3) = 26.0/2.5; vecd_tmp(4) = 39.0/2.5; vecd_tmp(5) = 6.0; vecd_tmp(6) = 8.0;}
            else if(atomName == "IR") {int_tmp = 7; vecd_tmp(0) = 12.0; vecd_tmp(1) = 8.0; vecd_tmp(2) = 16.0; vecd_tmp(3) = 27.0/2.5; vecd_tmp(4) = 40.5/2.5; vecd_tmp(5) = 6.0; vecd_tmp(6) = 8.0;}
            else if(atomName == "PT") {int_tmp = 7; vecd_tmp(0) = 11.0; vecd_tmp(1) = 8.0; vecd_tmp(2) = 16.0; vecd_tmp(3) = 29.0/2.5; vecd_tmp(4) = 43.5/2.5; vecd_tmp(5) = 6.0; vecd_tmp(6) = 8.0;}
            else if(atomName == "AU") {int_tmp = 7; vecd_tmp(0) = 11.0; vecd_tmp(1) = 8.0; vecd_tmp(2) = 16.0; vecd_tmp(3) = 12.0; vecd_tmp(4) = 18.0; vecd_tmp(5) = 6.0; vecd_tmp(6) = 8.0;}
            else if(atomName == "HG") {int_tmp = 7; vecd_tmp(0) = 12.0; vecd_tmp(1) = 8.0; vecd_tmp(2) = 16.0; vecd_tmp(3) = 12.0; vecd_tmp(4) = 18.0; vecd_tmp(5) = 6.0; vecd_tmp(6) = 8.0;}
            else if(atomName == "TL") {int_tmp = 7; vecd_tmp(0) = 12.0; vecd_tmp(1) = 25.0/3.0; vecd_tmp(2) = 50.0/3.0; vecd_tmp(3) = 12.0; vecd_tmp(4) = 18.0; vecd_tmp(5) = 6.0; vecd_tmp(6) = 8.0;}
            else if(atomName == "PB") {int_tmp = 7; vecd_tmp(0) = 12.0; vecd_tmp(1) = 26.0/3.0; vecd_tmp(2) = 52.0/3.0; vecd_tmp(3) = 12.0; vecd_tmp(4) = 18.0; vecd_tmp(5) = 6.0; vecd_tmp(6) = 8.0;}
            else if(atomName == "BI") {int_tmp = 7; vecd_tmp(0) = 12.0; vecd_tmp(1) = 9.0; vecd_tmp(2) = 18.0; vecd_tmp(3) = 12.0; vecd_tmp(4) = 18.0; vecd_tmp(5) = 6.0; vecd_tmp(6) = 8.0;}
            else if(atomName == "PO") {int_tmp = 7; vecd_tmp(0) = 12.0; vecd_tmp(1) = 28.0/3.0; vecd_tmp(2) = 56.0/3.0; vecd_tmp(3) = 12.0; vecd_tmp(4) = 18.0; vecd_tmp(5) = 6.0; vecd_tmp(6) = 8.0;}
            else if(atomName == "AT") {int_tmp = 7; vecd_tmp(0) = 12.0; vecd_tmp(1) = 29.0/3.0; vecd_tmp(2) = 58.0/3.0; vecd_tmp(3) = 12.0; vecd_tmp(4) = 18.0; vecd_tmp(5) = 6.0; vecd_tmp(6) = 8.0;}
            else if(atomName == "RN") {int_tmp = 7; vecd_tmp(0) = 12.0; vecd_tmp(1) = 10.0; vecd_tmp(2) = 20.0; vecd_tmp(3) = 12.0; vecd_tmp(4) = 18.0; vecd_tmp(5) = 6.0; vecd_tmp(6) = 8.0;}
            else if(atomName == "FR") {int_tmp = 7; vecd_tmp(0) = 13.0; vecd_tmp(1) = 10.0; vecd_tmp(2) = 20.0; vecd_tmp(3) = 12.0; vecd_tmp(4) = 18.0; vecd_tmp(5) = 6.0; vecd_tmp(6) = 8.0;}
            else if(atomName == "RA") {int_tmp = 7; vecd_tmp(0) = 14.0; vecd_tmp(1) = 10.0; vecd_tmp(2) = 20.0; vecd_tmp(3) = 12.0; vecd_tmp(4) = 18.0; vecd_tmp(5) = 6.0; vecd_tmp(6) = 8.0;}
            else if(atomName == "AC") {int_tmp = 7; vecd_tmp(0) = 14.0; vecd_tmp(1) = 10.0; vecd_tmp(2) = 20.0; vecd_tmp(3) = 31.0/2.5; vecd_tmp(4) = 46.5/2.5; vecd_tmp(5) = 6.0; vecd_tmp(6) = 8.0;}
            else if(atomName == "TH") {int_tmp = 7; vecd_tmp(0) = 14.0; vecd_tmp(1) = 10.0; vecd_tmp(2) = 20.0; vecd_tmp(3) = 32.0/2.5; vecd_tmp(4) = 48.0/2.5; vecd_tmp(5) = 6.0; vecd_tmp(6) = 8.0;}
            else if(atomName == "PA") {int_tmp = 7; vecd_tmp(0) = 14.0; vecd_tmp(1) = 10.0; vecd_tmp(2) = 20.0; vecd_tmp(3) = 31.0/2.5; vecd_tmp(4) = 46.5/2.5; vecd_tmp(5) = 48.0/7.0; vecd_tmp(6) = 64.0/7.0;}
            else if(atomName == "U") {int_tmp = 7; vecd_tmp(0) = 14.0; vecd_tmp(1) = 10.0; vecd_tmp(2) = 20.0; vecd_tmp(3) = 31.0/2.5; vecd_tmp(4) = 46.5/2.5; vecd_tmp(5) = 51.0/7.0; vecd_tmp(6) = 68.0/7.0;}
            else if(atomName == "NP") {int_tmp = 7; vecd_tmp(0) = 14.0; vecd_tmp(1) = 10.0; vecd_tmp(2) = 20.0; vecd_tmp(3) = 31.0/2.5; vecd_tmp(4) = 46.5/2.5; vecd_tmp(5) = 54.0/7.0; vecd_tmp(6) = 72.0/7.0;}
            else if(atomName == "PU") {int_tmp = 7; vecd_tmp(0) = 14.0; vecd_tmp(1) = 10.0; vecd_tmp(2) = 20.0; vecd_tmp(3) = 12.0; vecd_tmp(4) = 18.0; vecd_tmp(5) = 60.0/7.0; vecd_tmp(6) = 80.0/7.0;}
            else if(atomName == "AM") {int_tmp = 7; vecd_tmp(0) = 14.0; vecd_tmp(1) = 10.0; vecd_tmp(2) = 20.0; vecd_tmp(3) = 12.0; vecd_tmp(4) = 18.0; vecd_tmp(5) = 63.0/7.0; vecd_tmp(6) = 84.0/7.0;}
            else if(atomName == "CM") {int_tmp = 7; vecd_tmp(0) = 14.0; vecd_tmp(1) = 10.0; vecd_tmp(2) = 20.0; vecd_tmp(3) = 31.0/2.5; vecd_tmp(4) = 46.5/2.5; vecd_tmp(5) = 63.0/7.0; vecd_tmp(6) = 84.0/7.0;}
            else if(atomName == "BK") {int_tmp = 7; vecd_tmp(0) = 14.0; vecd_tmp(1) = 10.0; vecd_tmp(2) = 20.0; vecd_tmp(3) = 12.0; vecd_tmp(4) = 18.0; vecd_tmp(5) = 69.0/7.0; vecd_tmp(6) = 92.0/7.0;}
            else if(atomName == "CF") {int_tmp = 7; vecd_tmp(0) = 14.0; vecd_tmp(1) = 10.0; vecd_tmp(2) = 20.0; vecd_tmp(3) = 12.0; vecd_tmp(4) = 18.0; vecd_tmp(5) = 72.0/7.0; vecd_tmp(6) = 96.0/7.0;}
            else if(atomName == "ES") {int_tmp = 7; vecd_tmp(0) = 14.0; vecd_tmp(1) = 10.0; vecd_tmp(2) = 20.0; vecd_tmp(3) = 12.0; vecd_tmp(4) = 18.0; vecd_tmp(5) = 75.0/7.0; vecd_tmp(6) = 100.0/7.0;}
            else if(atomName == "FM") {int_tmp = 7; vecd_tmp(0) = 14.0; vecd_tmp(1) = 10.0; vecd_tmp(2) = 20.0; vecd_tmp(3) = 12.0; vecd_tmp(4) = 18.0; vecd_tmp(5) = 78.0/7.0; vecd_tmp(6) = 104.0/7.0;}
            else if(atomName == "MD") {int_tmp = 7; vecd_tmp(0) = 14.0; vecd_tmp(1) = 10.0; vecd_tmp(2) = 20.0; vecd_tmp(3) = 12.0; vecd_tmp(4) = 18.0; vecd_tmp(5) = 81.0/7.0; vecd_tmp(6) = 108.0/7.0;}
            else if(atomName == "NO") {int_tmp = 7; vecd_tmp(0) = 14.0; vecd_tmp(1) = 10.0; vecd_tmp(2) = 20.0; vecd_tmp(3) = 12.0; vecd_tmp(4) = 18.0; vecd_tmp(5) = 12.0; vecd_tmp(6) = 16.0;}
            else if(atomName == "LR") {int_tmp = 7; vecd_tmp(0) = 14.0; vecd_tmp(1) = 31.0/3.0; vecd_tmp(2) = 62.0/3.0; vecd_tmp(3) = 12.0; vecd_tmp(4) = 18.0; vecd_tmp(5) = 12.0; vecd_tmp(6) = 16.0;}
            else if(atomName == "RF") {int_tmp = 7; vecd_tmp(0) = 14.0; vecd_tmp(1) = 10.0; vecd_tmp(2) = 20.0; vecd_tmp(3) = 32.0/2.5; vecd_tmp(4) = 48.0/2.5; vecd_tmp(5) = 12.0; vecd_tmp(6) = 16.0;}
            else if(atomName == "DB") {int_tmp = 7; vecd_tmp(0) = 14.0; vecd_tmp(1) = 10.0; vecd_tmp(2) = 20.0; vecd_tmp(3) = 33.0/2.5; vecd_tmp(4) = 49.5/2.5; vecd_tmp(5) = 12.0; vecd_tmp(6) = 16.0;}
            else if(atomName == "SG") {int_tmp = 7; vecd_tmp(0) = 14.0; vecd_tmp(1) = 10.0; vecd_tmp(2) = 20.0; vecd_tmp(3) = 34.0/2.5; vecd_tmp(4) = 51.0/2.5; vecd_tmp(5) = 12.0; vecd_tmp(6) = 16.0;}
            else if(atomName == "BH") {int_tmp = 7; vecd_tmp(0) = 14.0; vecd_tmp(1) = 10.0; vecd_tmp(2) = 20.0; vecd_tmp(3) = 14.0; vecd_tmp(4) = 21.0; vecd_tmp(5) = 12.0; vecd_tmp(6) = 16.0;}
            else if(atomName == "HS") {int_tmp = 7; vecd_tmp(0) = 14.0; vecd_tmp(1) = 10.0; vecd_tmp(2) = 20.0; vecd_tmp(3) = 36.0/2.5; vecd_tmp(4) = 54.0/2.5; vecd_tmp(5) = 12.0; vecd_tmp(6) = 16.0;}
            else
            {
                cout << "ERROR: " << atomName << " does NOT have a default ON. Please input the occupation numbers by hand." << endl;
                exit(99);
            }            
        }
        int_tmp2 = 0;
        for(int ii = 0; ii < int_tmp; ii++)
        {
            int int_tmp3 = vecd_tmp(ii) / (irrep_list(int_tmp2).two_j + 1);
            double d_tmp = (double)(vecd_tmp(ii) - int_tmp3*(irrep_list(int_tmp2).two_j + 1)) / (double)(irrep_list(int_tmp2).two_j + 1);
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
    ifs.close();

    return;
}


/* 
    Evaluate amfi SOC integrals in j-adapted spinor basis
    Xmethod defines the algorithm to calculate X matrix in x2c transformation
                        Occupied shells     Virtual shells
        h1e:            h1e                 h1e
        partialFock:    fock                h1e             (default)
        fullFock:       Fock                Fock
    When necessary, the program will recalculate two-electron integrals.
*/
vMatrixXd DHF_SPH::get_amfi_unc(INT_SPH& int_sph_, const bool& twoC, const string& Xmethod, const bool& amfi_with_gaunt)
{
    bool amfi_with_gaunt_real = amfi_with_gaunt;
    if(with_gaunt && !amfi_with_gaunt)
    {
        cout << endl << "ATTENTION! Since gaunt terms are included in SCF, they are automatically calculated in amfi integrals." << endl << endl;
        amfi_with_gaunt_real = true;
    }
    else if (!with_gaunt && amfi_with_gaunt)
    {
        StartTime = clock();
        int_sph_.get_h2e_JK_gaunt_direct(gauntLSLS_JK,gauntLSSL_JK);
        EndTime = clock();
        cout << "2e-integral-Gaunt finished in " << (EndTime - StartTime) / (double)CLOCKS_PER_SEC << " seconds." << endl << endl; 
    }
    int2eJK SSLL_SD, SSSS_SD;
    int_sph_.get_h2eSD_JK_direct(SSLL_SD, SSSS_SD);
    if(twoC)
    {
        return get_amfi_unc_2c(SSLL_SD, SSSS_SD, amfi_with_gaunt_real);
    }
    else 
    {
        if(occMax_irrep < Nirrep && Xmethod == "fullFock")
        {
            cout << "fullFock is used in amfi function with incomplete h2e." << endl;
            cout << "Recalculate h2e and gaunt2e..." << endl;
            StartTime = clock();
            int_sph_.get_h2e_JK_direct(h2eLLLL_JK,h2eSSLL_JK,h2eSSSS_JK);
            EndTime = clock();
            cout << "Complete 2e-integral finished in " << (EndTime - StartTime) / (double)CLOCKS_PER_SEC << " seconds." << endl << endl; 
        }
        return get_amfi_unc(SSLL_SD, SSSS_SD, density, Xmethod, amfi_with_gaunt_real);
    }
}

vMatrixXd DHF_SPH::get_amfi_unc(const int2eJK& h2eSSLL_SD, const int2eJK& h2eSSSS_SD, const vMatrixXd& density_, const string& Xmethod, const bool& amfi_with_gaunt)
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
                    SO_4c(mm,nn) += density_(jr)(size_tmp2+ss,size_tmp2+rr) * h2eSSLL_SD.J(jr,ir)(esr,emn);
                    SO_4c(mm+size_tmp,nn) -= density_(jr)(ss,size_tmp2+rr) * h2eSSLL_SD.K(ir,jr)(emr,esn);
                    if(mm != nn) 
                    {
                        int enr = nn*size_tmp2+rr, esm = ss*size_tmp+mm;
                        SO_4c(nn+size_tmp,mm) -= density_(jr)(ss,size_tmp2+rr) * h2eSSLL_SD.K(ir,jr)(enr,esm);
                    }
                    SO_4c(mm+size_tmp,nn+size_tmp) += density_(jr)(size_tmp2+ss,size_tmp2+rr) * (h2eSSSS_SD.J(ir,jr)(emn,esr) - h2eSSSS_SD.K(ir,jr)(emr,esn)) + density_(jr)(ss,rr) * h2eSSLL_SD.J(ir,jr)(emn,esr);
                    if(amfi_with_gaunt)
                    {
                        int enm = nn*size_tmp+mm, ers = rr*size_tmp2+ss, erm = rr*size_tmp+mm, ens = nn*size_tmp2+ss;
                        SO_4c(mm,nn) -= density_(jr)(size_tmp2+ss,size_tmp2+rr) * gauntLSSL_JK.K(ir,jr)(emr,esn);
                        SO_4c(mm+size_tmp,nn+size_tmp) -= density_(jr)(ss,rr) * gauntLSSL_JK.K(jr,ir)(esn,emr);
                        SO_4c(mm+size_tmp,nn) += density_(jr)(size_tmp2+ss,rr)*(gauntLSLS_JK.J(ir,jr)(enm,ers) - gauntLSLS_JK.K(jr,ir)(erm,ens)) + density_(jr)(ss,size_tmp2+rr) * gauntLSSL_JK.J(jr,ir)(esr,emn);
                        if(mm != nn) 
                        {
                            int ern = rr*size_tmp+nn, ems = mm*size_tmp2+ss;
                            SO_4c(nn+size_tmp,mm) += density_(jr)(size_tmp2+ss,rr)*(gauntLSLS_JK.J(ir,jr)(emn,ers) - gauntLSLS_JK.K(jr,ir)(ern,ems)) + density_(jr)(ss,size_tmp2+rr) * gauntLSSL_JK.J(jr,ir)(esr,enm);
                        }
                    }
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
                                fock_tmp(mm,nn) += density_(jr)(ss,rr) * (h2eLLLL_JK.J(ir,jr)(emn,esr) - h2eLLLL_JK.K(ir,jr)(emr,esn)) + density_(jr)(size_tmp2+ss,size_tmp2+rr) * h2eSSLL_JK.J(jr,ir)(esr,emn);
                                fock_tmp(mm+size_tmp,nn) -= density_(jr)(ss,size_tmp2+rr) * h2eSSLL_JK.K(ir,jr)(emr,esn);
                                if(mm != nn) 
                                {
                                    int enr = nn*size_tmp2+rr, esm = ss*size_tmp+mm;
                                    fock_tmp(nn+size_tmp,mm) -= density_(jr)(ss,size_tmp2+rr) * h2eSSLL_JK.K(ir,jr)(enr,esm);
                                }
                                fock_tmp(mm+size_tmp,nn+size_tmp) += density_(jr)(size_tmp2+ss,size_tmp2+rr) * (h2eSSSS_JK.J(ir,jr)(emn,esr) - h2eSSSS_JK.K(ir,jr)(emr,esn)) + density_(jr)(ss,rr) * h2eSSLL_JK.J(ir,jr)(emn,esr);
                                if(with_gaunt)
                                {
                                    int enm = nn*size_tmp+mm, ers = rr*size_tmp2+ss, erm = rr*size_tmp+mm, ens = nn*size_tmp2+ss;
                                    fock_tmp(mm,nn) -= density_(jr)(size_tmp2+ss,size_tmp2+rr) * gauntLSSL_JK.K(ir,jr)(emr,esn);
                                    fock_tmp(mm+size_tmp,nn+size_tmp) -= density_(jr)(ss,rr) * gauntLSSL_JK.K(jr,ir)(esn,emr);
                                    fock_tmp(mm+size_tmp,nn) += density_(jr)(size_tmp2+ss,rr)*(gauntLSLS_JK.J(ir,jr)(enm,ers) - gauntLSLS_JK.K(jr,ir)(erm,ens)) + density_(jr)(ss,size_tmp2+rr) * gauntLSSL_JK.J(jr,ir)(esr,emn);
                                    if(mm != nn) 
                                    {
                                        int ern = rr*size_tmp+nn, ems = mm*size_tmp2+ss;
                                        fock_tmp(nn+size_tmp,mm) += density_(jr)(size_tmp2+ss,rr)*(gauntLSLS_JK.J(ir,jr)(emn,ers) - gauntLSLS_JK.K(jr,ir)(ern,ems)) + density_(jr)(ss,size_tmp2+rr) * gauntLSSL_JK.J(jr,ir)(esr,enm);
                                    }
                                }
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
vMatrixXd DHF_SPH::get_amfi_unc_2c(const int2eJK& h2eSSLL_SD, const int2eJK& h2eSSSS_SD, const bool& amfi_with_gaunt)
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
                    if(amfi_with_gaunt)
                    {
                        int enm = nn*size_tmp+mm, ers = rr*size_tmp2+ss, erm = rr*size_tmp+mm, ens = nn*size_tmp2+ss;
                        SO_4c(mm,nn) -= density_tmp(jr)(size_tmp2+ss,size_tmp2+rr) * gauntLSSL_JK.K(ir,jr)(emr,esn);
                        SO_4c(mm+size_tmp,nn+size_tmp) -= density_tmp(jr)(ss,rr) * gauntLSSL_JK.K(jr,ir)(esn,emr);
                        SO_4c(mm+size_tmp,nn) += density_tmp(jr)(size_tmp2+ss,rr)*(gauntLSLS_JK.J(ir,jr)(enm,ers) - gauntLSLS_JK.K(jr,ir)(erm,ens)) + density_tmp(jr)(ss,size_tmp2+rr) * gauntLSSL_JK.J(jr,ir)(esr,emn);
                        if(mm != nn) 
                        {
                            int ern = rr*size_tmp+nn, ems = mm*size_tmp2+ss;
                            SO_4c(nn+size_tmp,mm) += density_tmp(jr)(size_tmp2+ss,rr)*(gauntLSLS_JK.J(ir,jr)(emn,ers) - gauntLSLS_JK.K(jr,ir)(ern,ems)) + density_tmp(jr)(ss,size_tmp2+rr) * gauntLSSL_JK.J(jr,ir)(esr,enm);
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

    return amfi_unc;
}


/*
    Get private variable
*/
vMatrixXd DHF_SPH::get_fock_4c()
{
    return fock_4c;
}
vMatrixXd DHF_SPH::get_fock_4c_2ePart()
{
    vMatrixXd fock_2e(occMax_irrep);
    for(int ii = 0; ii < occMax_irrep; ii++)
        fock_2e(ii) = fock_4c(ii) - h1e_4c(ii);
    return fock_2e;
}
vMatrixXd DHF_SPH::get_h1e_4c()
{
    return h1e_4c;
}
vMatrixXd DHF_SPH::get_overlap_4c()
{
    return overlap_4c;
}
/*
    Set private variable
*/
void DHF_SPH::set_h1e_4c(const vMatrixXd& inputM)
{
    cout << "VERY DANGEROUS!! You changed h1e_4c!!" << endl;
    for(int ir = 0; ir < h1e_4c.rows(); ir++)
    {
        h1e_4c(ir) = inputM(ir);
    }
    return;
}

