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


DHF_SPH::DHF_SPH(INT_SPH& int_sph_, const string& filename, const bool& spinFree, const bool& twoC, const bool& with_gaunt_, const bool& with_gauge_, const bool& allInt, const bool& gaussian_nuc):
irrep_list(int_sph_.irrep_list), with_gaunt(with_gaunt_), with_gauge(with_gauge_), shell_list(int_sph_.shell_list)
{
    cout << "Initializing Dirac-HF for " << int_sph_.atomName << " atom." << endl;
    Nirrep = int_sph_.irrep_list.rows();
    size_basis_spinor = int_sph_.size_gtou_spinor;

    occNumber.resize(Nirrep);
    occNumberCore.resize(Nirrep);
    occMax_irrep = 0;
    setOCC(filename, int_sph_.atomName);

    if(allInt)
    {
        occMax_irrep = Nirrep;
    }
    occMax_irrep_compact = irrep_list(occMax_irrep-1).l*2+1;
    Nirrep_compact = irrep_list(Nirrep-1).l*2+1;
    compact2all.resize(Nirrep_compact);
    all2compact.resize(Nirrep);
    for(int ir = 0; ir < Nirrep ; ir++)
    {
        if(irrep_list(ir).two_j - 2*irrep_list(ir).l > 0)   all2compact(ir) = 2*irrep_list(ir).l;
        else all2compact(ir) = 2*irrep_list(ir).l - 1;
    }
    int tmp_i = 0;
    for(int ir = 0; ir < Nirrep ; ir+=irrep_list(ir).two_j+1)
    {
        compact2all(tmp_i) = ir;
        tmp_i++;
    }

    nelec = 0.0;
    cout << "Occupation number vector:" << endl;
    cout << "l\t2j\t2mj\tOcc" << endl;
    for(int ii = 0; ii < Nirrep; ii++)
    {
        cout << irrep_list(ii).l << "\t" << irrep_list(ii).two_j << "\t" << irrep_list(ii).two_mj << "\t" << occNumber(ii).transpose() << endl;
        for(int jj = 0; jj < occNumber(ii).rows(); jj++)
            nelec += occNumber(ii)(jj);
    }
    // cout << "Core occupation number vector:" << endl;
    // cout << "l\t2j\t2mj\tOcc" << endl;
    // for(int ii = 0; ii < Nirrep; ii++)
    // {
    //     cout << irrep_list(ii).l << "\t" << irrep_list(ii).two_j << "\t" << irrep_list(ii).two_mj << "\t" << occNumberCore(ii).transpose() << endl;
    // }
    cout << "Highest occupied irrep: " << occMax_irrep << endl;
    cout << "Total number of electrons: " << nelec << endl << endl;

    // Calculate the approximate maximum memory cost for SCF-amfi integrals
    double numberOfDouble = 0;
    for(int ir = 0; ir < irrep_list.size(); ir+=irrep_list(ir).two_j+1)
    for(int jr = 0; jr < irrep_list.size(); jr+=irrep_list(jr).two_j+1)
    {
        numberOfDouble += 2.0*irrep_list(ir).size*irrep_list(ir).size*irrep_list(jr).size*irrep_list(jr).size;
    }
    numberOfDouble *= 5.0;
    if(with_gaunt)  numberOfDouble = numberOfDouble/5.0*9.0;
    if(with_gauge)  numberOfDouble = numberOfDouble/9.0*12.0;
    cout << "Maximum memory cost (2e part) in SCF and amfi calculation: " << numberOfDouble*sizeof(double)/pow(1024.0,3) << " GB." << endl;

    StartTime = clock();
    overlap = int_sph_.get_h1e("overlap");
    kinetic = int_sph_.get_h1e("kinetic");
    if(gaussian_nuc)
    {
        cout << "Using Gaussian nuclear model!" << endl << endl;
        Vnuc = int_sph_.get_h1e("nucGau_attra");
        if(spinFree)
            WWW = int_sph_.get_h1e("s_p_nucGau_s_p_sf");
        else
            WWW = int_sph_.get_h1e("s_p_nucGau_s_p");
    }
    else
    {
        Vnuc = int_sph_.get_h1e("nuc_attra");
        if(spinFree)
            WWW = int_sph_.get_h1e("s_p_nuc_s_p_sf");
        else
            WWW = int_sph_.get_h1e("s_p_nuc_s_p");
    }
        
    EndTime = clock();
    cout << "1e-integral finished in " << (EndTime - StartTime) / (double)CLOCKS_PER_SEC << " seconds." << endl;

    StartTime = clock();
    if(twoC)
        h2eLLLL_JK = int_sph_.get_h2e_JK_compact("LLLL",irrep_list(occMax_irrep-1).l);
    else
        int_sph_.get_h2e_JK_direct(h2eLLLL_JK,h2eSSLL_JK,h2eSSSS_JK,irrep_list(occMax_irrep-1).l, spinFree);
    EndTime = clock();
    cout << "2e-integral finished in " << (EndTime - StartTime) / (double)CLOCKS_PER_SEC << " seconds." << endl; 

    if(with_gaunt && !twoC)
    {
        StartTime = clock();
        //Always calculate all Gaunt integrals for amfi integrals
        if(spinFree)
        {
            cout << "ATTENTION! Spin-free Gaunt integrals are used!" << endl;
            gauntLSLS_JK = int_sph_.get_h2e_JK_gauntSF_compact("LSLS");
            gauntLSSL_JK = int_sph_.get_h2e_JK_gauntSF_compact("LSSL");
        }
        else
            int_sph_.get_h2e_JK_gaunt_direct(gauntLSLS_JK,gauntLSSL_JK);
        EndTime = clock();
        cout << "2e-integral-Gaunt finished in " << (EndTime - StartTime) / (double)CLOCKS_PER_SEC << " seconds." << endl << endl; 
    }
    if(with_gauge && !twoC)
    {
        if(!with_gaunt)
        {
            cout << "ERROR: When gauge term is included, the Gaunt term must be included." << endl;
            exit(99);
        }
        int2eJK tmp1, tmp2, tmp3, tmp4;
        // tmp3 = int_sph_.get_h2e_JK_gauge("LSLS",-1);
        // tmp4 = int_sph_.get_h2e_JK_gauge("LSSL",-1);
        // tmp1 = int_sph_.compact_h2e(tmp3,irrep_list,-1);
        // tmp2 = int_sph_.compact_h2e(tmp4,irrep_list,-1);
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
    symmetrize_h2e(twoC);

    fock_4c.resize(occMax_irrep);
    h1e_4c.resize(occMax_irrep);
    overlap_4c.resize(occMax_irrep);
    overlap_half_i_4c.resize(occMax_irrep);
    density.resize(occMax_irrep);
    coeff.resize(occMax_irrep);
    ene_orb.resize(occMax_irrep);
    x2cXXX.resize(Nirrep);
    x2cRRR.resize(Nirrep);
    if(!twoC)
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
            In X2C-1e, h1e_4c, fock_4c, overlap_4c, and overlap_half_i_4c are the corresponding 2-c matrices. 
            spin free 4-c 1-e Hamiltonian is diagonalized to calculate X and R
            
            h1e_4c = [[V, T], [T, W_sf/4c^2 - T]]
        */
        for(int ir = 0; ir < occMax_irrep; ir++)
        {
            fock_4c(ir).resize(irrep_list(ir).size,irrep_list(ir).size);
            x2cXXX(ir) = X2C::get_X(overlap(ir),kinetic(ir),WWW(ir),Vnuc(ir));
            x2cRRR(ir) = X2C::get_R(overlap(ir),kinetic(ir),x2cXXX(ir));
            h1e_4c(ir) = X2C::evaluate_h1e_x2c(overlap(ir),kinetic(ir),WWW(ir),Vnuc(ir),x2cXXX(ir),x2cRRR(ir));
            // h1e_4c(ir) = kinetic(ir) + Vnuc(ir);
            overlap_4c(ir) = overlap(ir);
            overlap_half_i_4c(ir) = matrix_half_inverse(overlap_4c(ir));
        }   
    }
}


DHF_SPH::~DHF_SPH()
{
}

/*
    Symmetrize h2e with averaged |j,m>
*/
void DHF_SPH::symmetrize_h2e(const bool& twoC)
{
    if(twoC)
    {
        symmetrize_JK(h2eLLLL_JK, occMax_irrep_compact);
    }
    else
    {
        symmetrize_JK(h2eLLLL_JK, occMax_irrep_compact);
        symmetrize_JK(h2eSSSS_JK, occMax_irrep_compact);
        if(with_gaunt)
        {
            symmetrize_JK_gaunt(gauntLSLS_JK,Nirrep_compact);
        }
    }

    return;    
}
void DHF_SPH::symmetrize_JK(int2eJK& h2e, const int& Ncompact)
{
    for(int ir = 0; ir < Ncompact; ir++)
    for(int jr = 0; jr < Ncompact; jr++)
    {
        int size_i = irrep_list(compact2all(ir)).size, size_j = irrep_list(compact2all(jr)).size;
        double tmpJ[size_i*size_i][size_j*size_j];
        for(int mm = 0; mm < size_i; mm++)
        for(int nn = 0; nn < size_i; nn++)
        for(int ss = 0; ss < size_j; ss++)
        for(int rr = 0; rr < size_j; rr++)
        {
            int emn = mm*size_i+nn, esr = ss*size_j+rr, emr = mm*size_j+rr, esn = ss*size_i+nn;
            tmpJ[emn][esr] = h2e.J[ir][jr][emn][esr] - h2e.K[ir][jr][emr][esn];
        }
        for(int ii = 0; ii < size_i*size_i; ii++)
        for(int jj = 0; jj < size_j*size_j; jj++)
            h2e.J[ir][jr][ii][jj] = tmpJ[ii][jj];
    }
    return;
}
void DHF_SPH::symmetrize_JK_gaunt(int2eJK& h2e, const int& Ncompact)
{
    for(int ir = 0; ir < Ncompact; ir++)
    for(int jr = 0; jr < Ncompact; jr++)
    {
        int size_i = irrep_list(compact2all(ir)).size, size_j = irrep_list(compact2all(jr)).size;
        double tmpJ1[size_i*size_i][size_j*size_j];;
        for(int nn = 0; nn < size_i; nn++)
        for(int mm = 0; mm < size_i; mm++)
        for(int rr = 0; rr < size_j; rr++)
        for(int ss = 0; ss < size_j; ss++)
        {
            int enm = nn*size_i+mm, ers = rr*size_j+ss, erm = rr*size_i+mm, ens = nn*size_j+ss;
            tmpJ1[enm][ers] = h2e.J[ir][jr][enm][ers] - h2e.K[jr][ir][erm][ens];
        }
        for(int ii = 0; ii < size_i*size_i; ii++)
        for(int jj = 0; jj < size_j*size_j; jj++)
        {
            h2e.J[ir][jr][ii][jj] = tmpJ1[ii][jj];
        }
    }
    return;
}

/*
    The generalized eigen solver
*/
void DHF_SPH::eigensolverG_irrep(const vMatrixXd& inputM, const vMatrixXd& s_h_i, vVectorXd& values, vMatrixXd& vectors)
{
    for(int ir = 0; ir < occMax_irrep; ir += irrep_list(ir).two_j+1)
        eigensolverG(inputM(ir),s_h_i(ir),values(ir),vectors(ir));
    return;
}

/*
    Evaluate the difference between two vMatrixXd
*/
double DHF_SPH::evaluateChange_irrep(const vMatrixXd& M1, const vMatrixXd& M2)
{
    VectorXd vecd_tmp(occMax_irrep);
    vecd_tmp = VectorXd::Zero(occMax_irrep);
    for(int ir = 0; ir < occMax_irrep; ir += irrep_list(ir).two_j+1)
    {
        vecd_tmp(ir) = evaluateChange(M1(ir),M2(ir));
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
void DHF_SPH::runSCF(const bool& twoC, const bool& renormSmall)
{
    DHF_SPH::runSCF(twoC,renormSmall,NULL);
}
void DHF_SPH::runSCF(const bool& twoC, const bool& renormSmall, vMatrixXd* initialGuess)
{
    if(renormSmall)
    {
        renormalize_small();
    }
    vector<MatrixXd> error4DIIS[occMax_irrep], fock4DIIS[occMax_irrep];
    StartTime = clock();
    cout << endl;
    if(twoC) cout << "Start X2C-1e Hartree-Fock iterations..." << endl;
    else cout << "Start Dirac Hartree-Fock iterations..." << endl;
    cout << "with SCF convergence = " << convControl << endl;
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
            for(int ir = 0; ir < occMax_irrep; ir += irrep_list(ir).two_j+1) 
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
                    for(int ir = 0; ir < occMax_irrep; ir += irrep_list(ir).two_j+1)
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
            for(int ir = 0; ir < occMax_irrep; ir += irrep_list(ir).two_j+1)
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
        if(d_density < convControl || abs(nelec -1) < 1e-5) 
        {
            /* Special case for H atom */
            if(abs(nelec-1) < 1e-5)
            {
                eigensolverG_irrep(h1e_4c, overlap_half_i_4c, ene_orb, coeff);
                density = evaluateDensity_spinor_irrep(twoC);
            }
            converged = true;
            cout << endl << "SCF converges after " << iter << " iterations." << endl << endl;

            cout << "\tOrbital\t\tEnergy(in hartree)\n";
            cout << "\t*******\t\t******************\n";
            for(int ir = 0; ir < occMax_irrep; ir += irrep_list(ir).two_j+1)
            for(int ii = 1; ii <= irrep_list(ir).size; ii++)
            {
                if(twoC) cout << "\t" << ii << "\t\t" << setprecision(15) << ene_orb(ir)(ii - 1) << endl;
                else cout << "\t" << ii << "\t\t" << setprecision(15) << ene_orb(ir)(irrep_list(ir).size + ii - 1) << endl;
            }

            ene_scf = 0.0;
            for(int ir = 0; ir < occMax_irrep; ir += irrep_list(ir).two_j+1)
            {
                int size_tmp = irrep_list(ir).size;
                if(twoC)
                {
                    for(int ii = 0; ii < size_tmp; ii++)
                    for(int jj = 0; jj < size_tmp; jj++)
                    {
                        ene_scf += 0.5 * density(ir)(ii,jj) * (h1e_4c(ir)(jj,ii) + fock_4c(ir)(jj,ii)) * (irrep_list(ir).two_j+1.0);
                    }
                }
                else
                {
                    for(int ii = 0; ii < size_tmp * 2; ii++)
                    for(int jj = 0; jj < size_tmp * 2; jj++)
                    {
                        ene_scf += 0.5 * density(ir)(ii,jj) * (h1e_4c(ir)(jj,ii) + fock_4c(ir)(jj,ii)) * (irrep_list(ir).two_j+1.0);
                    }
                }
            }
            if(twoC) cout << "Final X2C-1e HF energy is " << setprecision(15) << ene_scf << " hartree." << endl;
            else cout << "Final DHF energy is " << setprecision(15) << ene_scf << " hartree." << endl;
            break;            
        }
        for(int ir = 0; ir < occMax_irrep; ir += irrep_list(ir).two_j+1)    
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

    
    for(int ir = 0; ir < occMax_irrep; ir += irrep_list(ir).two_j+1)
    {
        for(int jj = 1; jj < irrep_list(ir).two_j+1; jj++)
        {
            fock_4c(ir+jj) = fock_4c(ir);
            ene_orb(ir+jj) = ene_orb(ir);
            coeff(ir+jj) = coeff(ir);
            density(ir+jj) = density(ir);
        }
    }
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
    cout << "Renormalizing small component...." << endl;
    cout << "overlap_4c, h1e_4c, overlap_half_i_4c," << endl
            << "and all h2e will be renormalized." << endl << endl; 
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
    for(int ir = 0; ir < occMax_irrep_compact; ir++)
    for(int jr = 0; jr < occMax_irrep_compact; jr++)
    {
        int Iirrep = compact2all(ir), Jirrep = compact2all(jr);
        int sizei = irrep_list(Iirrep).size, sizej = irrep_list(Jirrep).size;
        for(int ii = 0; ii < sizei*sizei; ii++)
        for(int jj = 0; jj < sizej*sizej; jj++)
        {
            int a = ii / sizei, b = ii - a * sizei, c = jj / sizej, d = jj - c * sizej;
            h2eSSLL_JK.J[ir][jr][ii][jj] /= norm_s(Iirrep)(a) * norm_s(Iirrep)(b);
            h2eSSSS_JK.J[ir][jr][ii][jj] /= norm_s(Iirrep)(a) * norm_s(Iirrep)(b) * norm_s(Jirrep)(c) * norm_s(Jirrep)(d);
        }
        for(int ii = 0; ii < sizei*sizej; ii++)
        for(int jj = 0; jj < sizej*sizei; jj++)
        {
            int a = ii / sizej, b = ii - a * sizej, c = jj / sizei, d = jj - c * sizei;
            h2eSSLL_JK.K[ir][jr][ii][jj] /= norm_s(Iirrep)(a) * norm_s(Jirrep)(b);
            h2eSSSS_JK.K[ir][jr][ii][jj] /= norm_s(Iirrep)(a) * norm_s(Jirrep)(b) * norm_s(Jirrep)(c) * norm_s(Iirrep)(d);
        }
        if(with_gaunt)
        {
            for(int ii = 0; ii < sizei*sizei; ii++)
            for(int jj = 0; jj < sizej*sizej; jj++)
            {
                int a = ii / sizei, b = ii - a * sizei, c = jj / sizej, d = jj - c * sizej;
                gauntLSLS_JK.J[ir][jr][ii][jj] /= norm_s(Iirrep)(b) * norm_s(Jirrep)(d);
                gauntLSSL_JK.J[ir][jr][ii][jj] /= norm_s(Iirrep)(b) * norm_s(Jirrep)(c);
            }
            for(int ii = 0; ii < sizei*sizej; ii++)
            for(int jj = 0; jj < sizej*sizei; jj++)
            {
                int a = ii / sizej, b = ii - a * sizej, c = jj / sizei, d = jj - c * sizei;
                gauntLSLS_JK.K[ir][jr][ii][jj] /= norm_s(Jirrep)(b) * norm_s(Iirrep)(d);
                gauntLSSL_JK.K[ir][jr][ii][jj] /= norm_s(Jirrep)(b) * norm_s(Jirrep)(c);
            }
        }
    }

    renormalizedSmall = true;
}
void DHF_SPH::renormalize_h2e(int2eJK& h2eInput, const string& intType)
{
    for(int ir = 0; ir < occMax_irrep_compact; ir++)
    for(int jr = 0; jr < occMax_irrep_compact; jr++)
    {
        int Iirrep = compact2all(ir), Jirrep = compact2all(jr);
        int sizei = irrep_list(Iirrep).size, sizej = irrep_list(Jirrep).size;
        for(int ii = 0; ii < sizei*sizei; ii++)
        for(int jj = 0; jj < sizej*sizej; jj++)
        {
            int a = ii / sizei, b = ii - a * sizei, c = jj / sizej, d = jj - c * sizej;
            if(intType == "SSLL")
                h2eInput.J[ir][jr][ii][jj] /= norm_s(Iirrep)(a) * norm_s(Iirrep)(b);
            else if(intType == "SSSS")
                h2eInput.J[ir][jr][ii][jj] /= norm_s(Iirrep)(a) * norm_s(Iirrep)(b) * norm_s(Jirrep)(c) * norm_s(Jirrep)(d);
            else if(intType == "LSLS")
                h2eInput.J[ir][jr][ii][jj] /= norm_s(Iirrep)(b) * norm_s(Jirrep)(d);
            else if(intType == "LSSL")
                h2eInput.J[ir][jr][ii][jj] /= norm_s(Iirrep)(b) * norm_s(Jirrep)(c);
            else
            {
                cout << "ERROR: Unkown intType in renormalize_h2e" << endl;
                exit(99);
            }
        }
        for(int ii = 0; ii < sizei*sizej; ii++)
        for(int jj = 0; jj < sizej*sizei; jj++)
        {
            int a = ii / sizej, b = ii - a * sizej, c = jj / sizei, d = jj - c * sizei;
            if(intType == "SSLL")
                h2eInput.K[ir][jr][ii][jj] /= norm_s(Iirrep)(a) * norm_s(Jirrep)(b);
            else if(intType == "SSSS")
                h2eInput.K[ir][jr][ii][jj] /= norm_s(Iirrep)(a) * norm_s(Jirrep)(b) * norm_s(Jirrep)(c) * norm_s(Iirrep)(d);
            else if(intType == "LSLS")
                h2eInput.K[ir][jr][ii][jj] /= norm_s(Jirrep)(b) * norm_s(Iirrep)(d);
            else if(intType == "LSSL")
                h2eInput.K[ir][jr][ii][jj] /= norm_s(Jirrep)(b) * norm_s(Jirrep)(c);
            else
            {
                cout << "ERROR: Unkown intType in renormalize_h2e" << endl;
                exit(99);
            }
        }
    }
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
    for(int ir = 0; ir < occMax_irrep; ir += irrep_list(ir).two_j+1)
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
    int ir = all2compact(Iirrep);
    if(!twoC)
    {
        fock.resize(size*2,size*2);
        #pragma omp parallel  for
        for(int NN = 0; NN < size*(size+1)/2; NN++)
        {
            int tmp_i = int(sqrt(NN*2.0)), mm, nn;
            if(tmp_i *(tmp_i + 1) / 2 > NN)
            {
                mm = tmp_i - 1;
            }
            else
            {
                mm = tmp_i;
            }
            nn = NN - mm*(mm+1)/2;
            
            fock(mm,nn) = h1e_4c(Iirrep)(mm,nn);
            fock(mm+size,nn) = h1e_4c(Iirrep)(mm+size,nn);
            if(mm != nn) fock(nn+size,mm) = h1e_4c(Iirrep)(nn+size,mm);
            fock(mm+size,nn+size) = h1e_4c(Iirrep)(mm+size,nn+size);
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
        for(int NN = 0; NN < size*(size+1)/2; NN++)
        {
            int tmp_i = int(sqrt(NN*2.0)), mm, nn;
            if(tmp_i *(tmp_i + 1) / 2 > NN)
            {
                mm = tmp_i - 1;
            }
            else
            {
                mm = tmp_i;
            }
            nn = NN - mm*(mm+1)/2;
            fock(mm,nn) = h1e_4c(Iirrep)(mm,nn);
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

/* 
    Read occupation numbers 
*/
void DHF_SPH::setOCC(const string& filename, const string& atomName)
{
    string flags;
    VectorXd vecd_tmp = VectorXd::Zero(10);
    int int_tmp, int_tmp2, int_tmp3;
    ifstream ifs;
    ifs.open(filename);
    cout << ifs.eof() << endl;
    if (!ifs.fail())
    {
        while(!ifs.eof())
        {
            ifs >> flags;
            if(flags == "%occAMFI_" + atomName)
            {
                cout << "Found input occupation number for " << atomName <<     endl;
                ifs >> int_tmp;
                for(int ii = 0; ii < int_tmp; ii++)
                {
                    ifs >> vecd_tmp(ii);
                }
                break;
            }
        }
    }
    
    if(ifs.eof() or ifs.fail())
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
        else if(atomName == "SE") {int_tmp = 5; vecd_tmp(0) = 8.0; vecd_tmp(1) = 16.0/3.0; vecd_tmp(2) = 32.0/3.0; vecd_tmp(3) = 4.0; vecd_tmp(4) = 6.0;}
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

        else if(atomName == "OG") {int_tmp = 7; vecd_tmp(0) = 14.0; vecd_tmp(1) = 12.0; vecd_tmp(2) = 24.0; vecd_tmp(3) = 16.0; vecd_tmp(4) = 24.0; vecd_tmp(5) = 12.0; vecd_tmp(6) = 16.0;}
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
            occNumberCore(int_tmp2+jj).resize(irrep_list(int_tmp2+jj).size);
            occNumberCore(int_tmp2+jj) = VectorXd::Zero(irrep_list(int_tmp2+jj).size);
            for(int kk = 0; kk < int_tmp3; kk++)
            {    
                occNumber(int_tmp2+jj)(kk) = 1.0;
                occNumberCore(int_tmp2+jj)(kk) = 1.0;
                // if(kk == int_tmp3-1 && abs(d_tmp) < 1e-4)
                // {
                //    occNumberCore(int_tmp2+jj)(kk) = 0.0;
                // }
            }
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
vMatrixXd DHF_SPH::get_amfi_unc(INT_SPH& int_sph_, const bool& twoC, const string& Xmethod, bool amfi_with_gaunt, bool amfi_with_gauge)
{
    cout << "Running DHF_SPH::get_amfi_unc" << endl;
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
        return get_amfi_unc(SSLL_SD, SSSS_SD, gauntLSLS_SD, gauntLSSL_SD, density, Xmethod, amfi_with_gaunt);
    }
}

vMatrixXd DHF_SPH::get_amfi_unc(const int2eJK& h2eSSLL_SD, const int2eJK& h2eSSSS_SD, const int2eJK& gauntLSLS_SD, const int2eJK& gauntLSSL_SD, const vMatrixXd& density_, const string& Xmethod, const bool& amfi_with_gaunt)
{
    if(!converged)
    {
        cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
        cout << "!!  WARNING: Dirac HF did NOT converge  !!" << endl;
        cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
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
                    SO_4c(mm,nn) += density_(jr)(size_tmp2+ss,size_tmp2+rr) * h2eSSLL_SD.J[jr_c][ir_c][esr][emn];
                    SO_4c(mm+size_tmp,nn) -= density_(jr)(ss,size_tmp2+rr) * h2eSSLL_SD.K[ir_c][jr_c][emr][esn];
                    if(mm != nn) 
                    {
                        int enr = nn*size_tmp2+rr, esm = ss*size_tmp+mm;
                        SO_4c(nn+size_tmp,mm) -= density_(jr)(ss,size_tmp2+rr) * h2eSSLL_SD.K[ir_c][jr_c][enr][esm];
                    }
                    SO_4c(mm+size_tmp,nn+size_tmp) += density_(jr)(size_tmp2+ss,size_tmp2+rr) * h2eSSSS_SD.J[ir_c][jr_c][emn][esr] + density_(jr)(ss,rr) * h2eSSLL_SD.J[ir_c][jr_c][emn][esr];
                    if(amfi_with_gaunt)
                    {
                        int enm = nn*size_tmp+mm, ers = rr*size_tmp2+ss, erm = rr*size_tmp+mm, ens = nn*size_tmp2+ss;
                        SO_4c(mm,nn) -= density_(jr)(size_tmp2+ss,size_tmp2+rr) * gauntLSSL_SD.K[ir_c][jr_c][emr][esn];
                        SO_4c(mm+size_tmp,nn+size_tmp) -= density_(jr)(ss,rr) * gauntLSSL_SD.K[jr_c][ir_c][esn][emr];
                        SO_4c(mm+size_tmp,nn) += density_(jr)(size_tmp2+ss,rr)*gauntLSLS_SD.J[ir_c][jr_c][enm][ers] + density_(jr)(ss,size_tmp2+rr) * gauntLSSL_SD.J[jr_c][ir_c][esr][emn];
                        if(mm != nn) 
                        {
                            int ern = rr*size_tmp+nn, ems = mm*size_tmp2+ss;
                            SO_4c(nn+size_tmp,mm) += density_(jr)(size_tmp2+ss,rr)*gauntLSLS_SD.J[ir_c][jr_c][emn][ers] + density_(jr)(ss,size_tmp2+rr) * gauntLSSL_SD.J[jr_c][ir_c][esr][enm];
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
                            int jr_c = all2compact(jr);
                            int size_tmp2 = irrep_list(jr).size;
                            for(int ss = 0; ss < size_tmp2; ss++)
                            for(int rr = 0; rr < size_tmp2; rr++)
                            {
                                int emn = mm*size_tmp+nn, esr = ss*size_tmp2+rr, emr = mm*size_tmp2+rr, esn = ss*size_tmp+nn;
                                fock_tmp(mm,nn) += density_(jr)(ss,rr) * h2eLLLL_JK.J[ir_c][jr_c][emn][esr] + density_(jr)(size_tmp2+ss,size_tmp2+rr) * h2eSSLL_JK.J[jr_c][ir_c][esr][emn];
                                fock_tmp(mm+size_tmp,nn) -= density_(jr)(ss,size_tmp2+rr) * h2eSSLL_JK.K[ir_c][jr_c][emr][esn];
                                if(mm != nn) 
                                {
                                    int enr = nn*size_tmp2+rr, esm = ss*size_tmp+mm;
                                    fock_tmp(nn+size_tmp,mm) -= density_(jr)(ss,size_tmp2+rr) * h2eSSLL_JK.K[ir_c][jr_c][enr][esm];
                                }
                                fock_tmp(mm+size_tmp,nn+size_tmp) += density_(jr)(size_tmp2+ss,size_tmp2+rr) * h2eSSSS_JK.J[ir_c][jr_c][emn][esr] + density_(jr)(ss,rr) * h2eSSLL_JK.J[ir_c][jr_c][emn][esr];
                                if(with_gaunt)
                                {
                                    int enm = nn*size_tmp+mm, ers = rr*size_tmp2+ss, erm = rr*size_tmp+mm, ens = nn*size_tmp2+ss;
                                    fock_tmp(mm,nn) -= density_(jr)(size_tmp2+ss,size_tmp2+rr) * gauntLSSL_JK.K[ir_c][jr_c][emr][esn];
                                    fock_tmp(mm+size_tmp,nn+size_tmp) -= density_(jr)(ss,rr) * gauntLSSL_JK.K[jr_c][ir_c][esn][emr];
                                    fock_tmp(mm+size_tmp,nn) += density_(jr)(size_tmp2+ss,rr)*gauntLSLS_JK.J[ir_c][jr_c][enm][ers] + density_(jr)(ss,size_tmp2+rr) * gauntLSSL_JK.J[jr_c][ir_c][esr][emn];
                                    if(mm != nn) 
                                    {
                                        int ern = rr*size_tmp+nn, ems = mm*size_tmp2+ss;
                                        fock_tmp(nn+size_tmp,mm) += density_(jr)(size_tmp2+ss,rr)*gauntLSLS_JK.J[ir_c][jr_c][emn][ers] + density_(jr)(ss,size_tmp2+rr) * gauntLSSL_JK.J[jr_c][ir_c][esr][enm];
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

    X_calculated = true;
    return amfi_unc;
}

/* 
    Evaluate amfi SOC integrals in j-adapted spinor basis for two-component calculation
    X matrices are obtained from the corresponding (SF)X2C-1e procedure.
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
        Please see L. Cheng, et al, J. Chem. Phys. 141, 164107 (2014)
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
    Get coeff for basis set
    2c -> coeff
    4c -> x2c2e coeff
*/
vMatrixXd DHF_SPH::get_coeff_bs(const bool& twoC)
{
    if(!converged)
    {
        cout << "SCF did not converge. get_coeff_bs cannot be used!" << endl;
        exit(99);
    }

    if(twoC)
        return coeff;
    else
    {
        vMatrixXd overlap_2c(occMax_irrep), XXX(occMax_irrep), RRR(occMax_irrep), coeff_2c(occMax_irrep);
        for(int ir = 0; ir < occMax_irrep; ir++)
        {
            overlap_2c(ir) = overlap_4c(ir).block(0,0,overlap_4c(ir).rows()/2,overlap_4c(ir).cols()/2);
            VectorXd ene_mo_tmp;
            XXX(ir) = X2C::get_X(coeff(ir));
            RRR(ir) = X2C::get_R(overlap_4c(ir),XXX(ir));
            coeff_2c(ir) = RRR(ir).inverse()*coeff(ir).block(0,coeff(ir).rows()/2,coeff(ir).rows()/2,coeff(ir).rows()/2);
        }
        return coeff_2c;
    }
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
vMatrixXd DHF_SPH::get_density()
{
    return density;
}
vVectorXd DHF_SPH::get_occNumber()
{
    return occNumber;
}
vMatrixXd DHF_SPH::get_X()
{
    if(X_calculated)
        return x2cXXX;
    else
    {
        cout << "ERROR: get_X was called before X matrices calculated!" << endl;
        exit(99);
    }
}
vMatrixXd DHF_SPH::get_X_normalized()
{
    /*
        return X_tilde = CS_tilde CL_tilde^-1 = (T/2c2)^{1/2} XXX S^{-1/2}
        where C_tilde = S^{1/2} C  
    */   
    if(X_calculated)
    {
        vMatrixXd X_tilde(occMax_irrep);
        for(int ir = 0; ir < occMax_irrep; ir++)
        {
            int size = irrep_list(ir).size;
            MatrixXd tmp1(size,size),tmp2(size,size);
            for(int ii = 0; ii < size; ii++)
            for(int jj = 0; jj < size; jj++)
            {
                tmp1(ii,jj) = overlap_half_i_4c(ir)(ii,jj);
                tmp2(ii,jj) = overlap_half_i_4c(ir)(size+ii,size+jj);
            }
            tmp2 = tmp2.inverse();
            X_tilde(ir) = tmp2*x2cXXX(ir)*tmp1;
        }
        return X_tilde;
    }
    else
    {
        cout << "ERROR: get_X was called before X matrices calculated!" << endl;
        exit(99);
    }
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



/*
    basisGenerator to generate relativistic (j-adapted) contracted basis sets.
    WARNING: This method was implemented for CFOUR format of basis sets only!
*/
void DHF_SPH::basisGenerator(string basisName, string filename, const INT_SPH& intor, const INT_SPH& intorAll, const bool& sf, const string& tag)
{
    cout << "Running DHF_SPH::basisGenerator" << endl;
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
        if(occNumber(ir).rows() == 0) break;
        occL++;
    }
    basisInfo.resize(occL);
    
    // For each l-shell, count how many orbitals are fully or partially occupied
    // and resize basisInfo(l)
    occL = 0;
    for(int ir = 0; ir < irrep_list.rows(); ir += 4*irrep_list(ir).l+2)
    {
        int occN = 0;
        if(occNumber(ir).rows() == 0) break;
        for(int ii = 0; ii < occNumber(ir).rows(); ii++)
        {
            if(abs(occNumber(ir)(ii)) > 1e-4)
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