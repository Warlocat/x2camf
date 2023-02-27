#include"dhf_sph.h"
#include"mkl_itrf.h"
#include<iostream>
#include<algorithm>
#include<omp.h>
#include<vector>
#include<ctime>
#include<iostream>
#include<iomanip>
#include<fstream>

using namespace std;


DHF_SPH::DHF_SPH(INT_SPH& int_sph_, const string& filename, const bool& spinFree, const bool& twoC, const bool& with_gaunt_, const bool& with_gauge_, const bool& allInt, const bool& gaussian_nuc):
irrep_list(int_sph_.irrep_list), with_gaunt(with_gaunt_), with_gauge(with_gauge_), shell_list(int_sph_.shell_list), twoComponent(twoC)
{
    string method = "HF";
    if(twoC) method = "2c-" + method;
    else method = "4c-" + method;
    if(spinFree) method = "SF-" + method;
    if(with_gaunt_) method = method + " with Gaunt";
    if(with_gauge_) method = method + " with Gaunt";
    if(gaussian_nuc) method = method + " with Gaussian nuclear model";
    cout << "Initializing " << method << " for " << int_sph_.atomName << " atom." << endl;
    Nirrep = int_sph_.irrep_list.size();
    size_basis_spinor = int_sph_.size_gtou_spinor;

    occNumber.resize(Nirrep);
    occMax_irrep = 0;
    setOCC(filename, int_sph_.atomName);

    if(allInt)
    {
        occMax_irrep = Nirrep;
    }
    occMax_irrep_compact = irrep_list[occMax_irrep-1].l*2+1;
    Nirrep_compact = irrep_list[Nirrep-1].l*2+1;
    compact2all.resize(Nirrep_compact);
    all2compact.resize(Nirrep);
    for(int ir = 0; ir < Nirrep ; ir++)
    {
        if(irrep_list[ir].two_j - 2*irrep_list[ir].l > 0)   all2compact[ir] = 2*irrep_list[ir].l;
        else all2compact[ir] = 2*irrep_list[ir].l - 1;
    }
    int tmp_i = 0;
    for(int ir = 0; ir < Nirrep ; ir+=irrep_list[ir].two_j+1)
    {
        compact2all[tmp_i] = ir;
        tmp_i++;
    }

    nelec = 0.0;
    cout << "Occupation number vector:" << endl;
    cout << "l\t2j\t2mj\tOcc" << endl;
    for(int ii = 0; ii < Nirrep; ii++)
    {
        cout << irrep_list[ii].l << "\t" << irrep_list[ii].two_j << "\t" << irrep_list[ii].two_mj;
        for(int jj = 0; jj < occNumber[ii].size(); jj++)
            cout << "\t" << occNumber[ii][jj];
        cout << endl;
        for(int jj = 0; jj < occNumber[ii].size(); jj++)
            nelec += occNumber[ii][jj];
    }
    cout << "Highest occupied irrep: " << occMax_irrep << endl;
    cout << "Total number of electrons: " << nelec << endl << endl;

    // Calculate the approximate maximum memory cost for SCF-amfi integrals
    double numberOfDouble = 0;
    for(int ir = 0; ir < irrep_list.size(); ir+=irrep_list[ir].two_j+1)
    for(int jr = 0; jr < irrep_list.size(); jr+=irrep_list[jr].two_j+1)
    {
        numberOfDouble += 2.0*irrep_list[ir].size*irrep_list[ir].size*irrep_list[jr].size*irrep_list[jr].size;
    }
    numberOfDouble *= 5.0;
    if(with_gaunt)  numberOfDouble = numberOfDouble/5.0*9.0;
    if(with_gauge)  numberOfDouble = numberOfDouble/9.0*12.0;
    cout << "Maximum memory cost (2e part) in SCF and amfi calculation: " << numberOfDouble*sizeof(double)/pow(1024.0,3) << " GB." << endl;

    countTime(StartTimeCPU,StartTimeWall);
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
        
    countTime(EndTimeCPU,EndTimeWall);
    printTime("1e-integrals");

    countTime(StartTimeCPU,StartTimeWall);
    if(twoC)
        h2eLLLL_JK = int_sph_.get_h2e_JK_compact("LLLL",irrep_list[occMax_irrep-1].l);
    else
        int_sph_.get_h2e_JK_direct(h2eLLLL_JK,h2eSSLL_JK,h2eSSSS_JK,irrep_list[occMax_irrep-1].l, spinFree);
    countTime(EndTimeCPU,EndTimeWall);
    printTime("2e-Coulomb-integrals");

    if(with_gaunt && !twoC)
    {
        countTime(StartTimeCPU,StartTimeWall);
        //Always calculate all Gaunt integrals for amfi integrals
        if(spinFree)
        {
            cout << "ATTENTION! Spin-free Gaunt integrals are used!" << endl;
            gauntLSLS_JK = int_sph_.get_h2e_JK_gauntSF_compact("LSLS");
            gauntLSSL_JK = int_sph_.get_h2e_JK_gauntSF_compact("LSSL");
        }
        else
            int_sph_.get_h2e_JK_gaunt_direct(gauntLSLS_JK,gauntLSSL_JK);
        countTime(EndTimeCPU,EndTimeWall);
        printTime("2e-Gaunt-integrals");
    }
    if(with_gauge && !twoC)
    {
        if(!with_gaunt)
        {
            cout << "ERROR: When gauge term is included, the Gaunt term must be included." << endl;
            exit(99);
        }
        int2eJK tmp1, tmp2, tmp3, tmp4;
        int_sph_.get_h2e_JK_gauge_direct(tmp1,tmp2);
        for(int ir = 0; ir < Nirrep_compact; ir++)
        for(int jr = 0; jr < Nirrep_compact; jr++)
        {
            int size_i = irrep_list[compact2all[ir]].size, size_j = irrep_list[compact2all[jr]].size;
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
        countTime(EndTimeCPU,EndTimeWall);
        printTime("2e-gauge-integrals");
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
            int size_tmp = irrep_list[ii].size;
            fock_4c[ii].resize(size_tmp*2*size_tmp*2);
            h1e_4c[ii].resize(size_tmp*2*size_tmp*2);
            overlap_4c[ii].resize(size_tmp*2*size_tmp*2);
            for(int mm = 0; mm < size_tmp; mm++)
            for(int nn = 0; nn < size_tmp; nn++)
            {
                overlap_4c[ii][mm*size_tmp*2+nn] = overlap[ii][mm*size_tmp+nn];
                overlap_4c[ii][(size_tmp+mm)*size_tmp*2+nn] = 0.0;
                overlap_4c[ii][mm*size_tmp*2+size_tmp+nn] = 0.0;
                overlap_4c[ii][(size_tmp+mm)*size_tmp*2+size_tmp+nn] = kinetic[ii][mm*size_tmp+nn] / 2.0 / speedOfLight / speedOfLight;
                h1e_4c[ii][mm*size_tmp*2+nn] = Vnuc[ii][mm*size_tmp+nn];
                h1e_4c[ii][(size_tmp+mm)*size_tmp*2+nn] = kinetic[ii][mm*size_tmp+nn];
                h1e_4c[ii][mm*size_tmp*2+size_tmp+nn] = kinetic[ii][mm*size_tmp+nn];
                h1e_4c[ii][(size_tmp+mm)*size_tmp*2+size_tmp+nn] = WWW[ii][mm*size_tmp+nn]/4.0/speedOfLight/speedOfLight - kinetic[ii][mm*size_tmp+nn];
            }
            overlap_half_i_4c[ii] = matrix_half_inverse(overlap_4c[ii], size_tmp*2);
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
            int size_tmp = irrep_list[ir].size;
            fock_4c[ir].resize(size_tmp*size_tmp);
            x2cXXX[ir] = X2C::get_X(overlap[ir],kinetic[ir],WWW[ir],Vnuc[ir],size_tmp);
            x2cRRR[ir] = X2C::get_R(overlap[ir],kinetic[ir],x2cXXX[ir],size_tmp);
            h1e_4c[ir] = X2C::evaluate_h1e_x2c(overlap[ir],kinetic[ir],WWW[ir],Vnuc[ir],x2cXXX[ir],x2cRRR[ir],size_tmp);
            overlap_4c[ir] = overlap[ir];
            overlap_half_i_4c[ir] = matrix_half_inverse(overlap_4c[ir], size_tmp);
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
        int size_i = irrep_list[compact2all[ir]].size, size_j = irrep_list[compact2all[jr]].size;
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
        int size_i = irrep_list[compact2all[ir]].size, size_j = irrep_list[compact2all[jr]].size;
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
void DHF_SPH::eigensolverG_irrep(const vVectorXd& inputM, const vVectorXd& s_h_i, vVectorXd& values, vVectorXd& vectors)
{
    for(int ir = 0; ir < occMax_irrep; ir += irrep_list[ir].two_j+1)
    {
        int n = twoComponent ? irrep_list[ir].size : irrep_list[ir].size*2;
        eigensolverG(inputM[ir], s_h_i[ir], values[ir], vectors[ir], n);
    }
        
    return;
}

/*
    Evaluate the difference between two vVectorXd
*/
double DHF_SPH::evaluateChange_irrep(const vVectorXd& M1, const vVectorXd& M2)
{
    double diff = 0.0, tmp;
    for(int ir = 0; ir < occMax_irrep; ir += irrep_list[ir].two_j+1)
    {
        tmp = evaluateChange(M1[ir],M2[ir]);
        if(tmp > diff)  diff = tmp;
    }
    return diff;
}

/*
    Evaluate error matrix in DIIS
*/
vector<double> DHF_SPH::evaluateErrorDIIS(const vector<double>& fock_, const vector<double>& overlap_, const vector<double>& density_, const int& N)
{
    vector<double> tmp1, tmp2;
    dgemm_itrf('n','n',N,N,N,1.0,fock_,density_,0.0,tmp1);
    dgemm_itrf('n','n',N,N,N,1.0,tmp1,overlap_,0.0,tmp2);
    dgemm_itrf('n','n',N,N,N,1.0,overlap_,density_,0.0,tmp1);
    dgemm_itrf('n','n',N,N,N,-1.0,tmp1,fock_,1.0,tmp2);
    vector<double> err(N*N);
    for(int ii = 0; ii < N; ii++)
    for(int jj = 0; jj < N; jj++)
    {
        err[ii*N+jj] = tmp2[ii*N+jj];
    }
    return err;
}
vector<double> DHF_SPH::evaluateErrorDIIS(const vector<double>& den_old, const vector<double>& den_new)
{
    int size = den_old.size();
    vector<double> err(size);
    for(int ii = 0; ii < size; ii++)
    {
        err[ii] = den_old[ii] - den_new[ii];
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
void DHF_SPH::runSCF(const bool& twoC, const bool& renormSmall, vVectorXd* initialGuess)
{
    vector<int> nbas(occMax_irrep);
    for(int ir = 0; ir < occMax_irrep; ir++)
        nbas[ir] = twoC ? irrep_list[ir].size : irrep_list[ir].size*2;

    if(renormSmall && !twoC)
    {
        renormalize_small();
    }
    vVectorXd error4DIIS[occMax_irrep], fock4DIIS[occMax_irrep];
    countTime(StartTimeCPU,StartTimeWall);
    cout << endl;
    if(twoC) cout << "Start X2C-1e Hartree-Fock iterations..." << endl;
    else cout << "Start Dirac Hartree-Fock iterations..." << endl;
    cout << "with SCF convergence = " << convControl << endl;
    cout << endl;
    vVectorXd newDen;
    if(initialGuess == NULL)
    {
        eigensolverG_irrep(h1e_4c, overlap_half_i_4c, ene_orb, coeff);
        density = evaluateDensity_spinor_irrep(nbas, twoC);
    }
    else
    {
        density = *initialGuess;
    }

    for(int iter = 1; iter <= maxIter; iter++)
    {
        if(iter <= 2)
        {
            for(int ir = 0; ir < occMax_irrep; ir += irrep_list[ir].two_j+1) 
            {
                int size_tmp = irrep_list[ir].size;
                evaluateFock(fock_4c[ir],twoC,density,size_tmp,ir);
            }
        }
        else
        {
            int tmp_size = fock4DIIS[0].size() + 1;
            vector<double> B4DIIS(tmp_size*tmp_size);
            vector<double> vec_b(tmp_size), C;    
            for(int ii = 0; ii < tmp_size - 1; ii++)
            {    
                for(int jj = 0; jj <= ii; jj++)
                {
                    B4DIIS[ii*tmp_size+jj] = 0.0;
                    for(int ir = 0; ir < occMax_irrep; ir += irrep_list[ir].two_j+1)
                    for(int aa = 0; aa < error4DIIS[ir][ii].size(); aa++)
                        B4DIIS[ii*tmp_size+jj] += (error4DIIS[ir][ii][aa]*error4DIIS[ir][jj][aa]);
                    B4DIIS[jj*tmp_size+ii] = B4DIIS[ii*tmp_size+jj];
                }
                B4DIIS[(tmp_size-1)*tmp_size+ii] = -1.0;
                B4DIIS[ii*tmp_size+(tmp_size-1)] = -1.0;
                vec_b[ii] = 0.0;
            }
            B4DIIS[(tmp_size-1)*tmp_size+(tmp_size-1)] = 0.0;
            vec_b[tmp_size-1] = -1.0;
            liearEqn_d(B4DIIS, vec_b, tmp_size, C);
            for(int ir = 0; ir < occMax_irrep; ir += irrep_list[ir].two_j+1)
            {
                fill(fock_4c[ir].begin(), fock_4c[ir].end(), 0.0);
                for(int ii = 0; ii < tmp_size - 1; ii++)
                {
                    fock_4c[ir] = fock_4c[ir] + C[ii] * fock4DIIS[ir][ii];
                }
            }
        }

        eigensolverG_irrep(fock_4c, overlap_half_i_4c, ene_orb, coeff);
        newDen = evaluateDensity_spinor_irrep(nbas, twoC);
        d_density = evaluateChange_irrep(density, newDen);
        
        cout << "Iter #" << iter << " maximum density difference: " << d_density << endl;
        
        density = newDen;
        if(d_density < convControl || abs(nelec -1) < 1e-5) 
        {
            /* Special case for H atom */
            if(abs(nelec-1) < 1e-5)
            {
                cout << "Special treatment for fractional occupation of H atom." << endl;
                eigensolverG_irrep(h1e_4c, overlap_half_i_4c, ene_orb, coeff);
                density = evaluateDensity_spinor_irrep(nbas, twoC);
            }
            converged = true;
            cout << endl << "SCF converges after " << iter << " iterations." << endl << endl;

            cout << "\tOrbital\t\tEnergy(in hartree)\n";
            cout << "\t*******\t\t******************\n";
            for(int ir = 0; ir < occMax_irrep; ir += irrep_list[ir].two_j+1)
            for(int ii = 1; ii <= irrep_list[ir].size; ii++)
            {
                if(twoC) cout << "\t" << ii << "\t\t" << setprecision(15) << ene_orb[ir][ii - 1] << endl;
                else cout << "\t" << ii << "\t\t" << setprecision(15) << ene_orb[ir][irrep_list[ir].size + ii - 1] << endl;
            }

            ene_scf = 0.0;
            for(int ir = 0; ir < occMax_irrep; ir += irrep_list[ir].two_j+1)
            {
                for(int ii = 0; ii < nbas[ir]; ii++)
                for(int jj = 0; jj < nbas[ir]; jj++)
                {
                    ene_scf += 0.5 * density[ir][ii*nbas[ir]+jj] * (h1e_4c[ir][jj*nbas[ir]+ii] + 
                                     fock_4c[ir][jj*nbas[ir]+ii]) * (irrep_list[ir].two_j+1.0);
                }
            }
            if(twoC) cout << "Final X2C-1e HF energy is " << setprecision(15) << ene_scf << " hartree." << endl;
            else cout << "Final DHF energy is " << setprecision(15) << ene_scf << " hartree." << endl;
            break;            
        }
        for(int ir = 0; ir < occMax_irrep; ir += irrep_list[ir].two_j+1)    
        {
            int size_tmp = irrep_list[ir].size;
            int size2 = twoC ? size_tmp : size_tmp*2;
            evaluateFock(fock_4c[ir],twoC,density,size_tmp,ir);
            if(error4DIIS[ir].size() >= size_DIIS)
            {
                error4DIIS[ir].erase(error4DIIS[ir].begin());
                error4DIIS[ir].push_back(evaluateErrorDIIS(fock_4c[ir],overlap_4c[ir],density[ir],size2));
                fock4DIIS[ir].erase(fock4DIIS[ir].begin());
                fock4DIIS[ir].push_back(fock_4c[ir]);
            }
            else
            {
                error4DIIS[ir].push_back(evaluateErrorDIIS(fock_4c[ir],overlap_4c[ir],density[ir],size2));
                fock4DIIS[ir].push_back(fock_4c[ir]);
            }
        }
    }
    
    for(int ir = 0; ir < occMax_irrep; ir += irrep_list[ir].two_j+1)
    {
        for(int jj = 1; jj < irrep_list[ir].two_j+1; jj++)
        {
            fock_4c[ir+jj] = fock_4c[ir];
            ene_orb[ir+jj] = ene_orb[ir];
            coeff[ir+jj] = coeff[ir];
            density[ir+jj] = density[ir];
        }
    }

    countTime(EndTimeCPU,EndTimeWall);
    printTime("DHF iterations");
}

/*
    Renormalize small component to enhance the stability
*/
void DHF_SPH::renormalize_small()
{
    norm_s.resize(occMax_irrep);
    for(int ii = 0; ii < occMax_irrep; ii++)
    {
        norm_s[ii].resize(irrep_list[ii].size);
        for(int jj = 0; jj < irrep_list[ii].size; jj++)
        {
            norm_s[ii][jj] = sqrt(kinetic[ii][jj*irrep_list[ii].size+jj] / 2.0 / speedOfLight / speedOfLight);
        }
    }
    cout << "Renormalizing small component...." << endl;
    cout << "overlap_4c, h1e_4c, overlap_half_i_4c," << endl
            << "and all h2e will be renormalized." << endl << endl; 
    for(int ii = 0; ii < occMax_irrep; ii++)
    {
        int size_tmp = irrep_list[ii].size;
        for(int mm = 0; mm < size_tmp; mm++)
        for(int nn = 0; nn < size_tmp; nn++)
        {
            overlap_4c[ii][(size_tmp+mm)*size_tmp*2+size_tmp+nn] /= norm_s[ii][mm] * norm_s[ii][nn];
            h1e_4c[ii][(size_tmp+mm)*size_tmp*2+nn] /= norm_s[ii][mm];
            h1e_4c[ii][mm*size_tmp*2+size_tmp+nn] /= norm_s[ii][nn];
            h1e_4c[ii][(size_tmp+mm)*size_tmp*2+size_tmp+nn] /= norm_s[ii][mm] * norm_s[ii][nn];
        }
        overlap_half_i_4c[ii] = matrix_half_inverse(overlap_4c[ii], size_tmp*2);
    }
    for(int ir = 0; ir < occMax_irrep_compact; ir++)
    for(int jr = 0; jr < occMax_irrep_compact; jr++)
    {
        int Iirrep = compact2all[ir], Jirrep = compact2all[jr];
        int sizei = irrep_list[Iirrep].size, sizej = irrep_list[Jirrep].size;
        for(int ii = 0; ii < sizei*sizei; ii++)
        for(int jj = 0; jj < sizej*sizej; jj++)
        {
            int a = ii / sizei, b = ii - a * sizei, c = jj / sizej, d = jj - c * sizej;
            h2eSSLL_JK.J[ir][jr][ii][jj] /= norm_s[Iirrep][a] * norm_s[Iirrep][b];
            h2eSSSS_JK.J[ir][jr][ii][jj] /= norm_s[Iirrep][a] * norm_s[Iirrep][b] * norm_s[Jirrep][c] * norm_s[Jirrep][d];
        }
        for(int ii = 0; ii < sizei*sizej; ii++)
        for(int jj = 0; jj < sizej*sizei; jj++)
        {
            int a = ii / sizej, b = ii - a * sizej, c = jj / sizei, d = jj - c * sizei;
            h2eSSLL_JK.K[ir][jr][ii][jj] /= norm_s[Iirrep][a] * norm_s[Jirrep][b];
            h2eSSSS_JK.K[ir][jr][ii][jj] /= norm_s[Iirrep][a] * norm_s[Jirrep][b] * norm_s[Jirrep][c] * norm_s[Iirrep][d];
        }
        if(with_gaunt)
        {
            for(int ii = 0; ii < sizei*sizei; ii++)
            for(int jj = 0; jj < sizej*sizej; jj++)
            {
                int a = ii / sizei, b = ii - a * sizei, c = jj / sizej, d = jj - c * sizej;
                gauntLSLS_JK.J[ir][jr][ii][jj] /= norm_s[Iirrep][b] * norm_s[Jirrep][d];
                gauntLSSL_JK.J[ir][jr][ii][jj] /= norm_s[Iirrep][b] * norm_s[Jirrep][c];
            }
            for(int ii = 0; ii < sizei*sizej; ii++)
            for(int jj = 0; jj < sizej*sizei; jj++)
            {
                int a = ii / sizej, b = ii - a * sizej, c = jj / sizei, d = jj - c * sizei;
                gauntLSLS_JK.K[ir][jr][ii][jj] /= norm_s[Jirrep][b] * norm_s[Iirrep][d];
                gauntLSSL_JK.K[ir][jr][ii][jj] /= norm_s[Jirrep][b] * norm_s[Jirrep][c];
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
        int Iirrep = compact2all[ir], Jirrep = compact2all[jr];
        int sizei = irrep_list[Iirrep].size, sizej = irrep_list[Jirrep].size;
        for(int ii = 0; ii < sizei*sizei; ii++)
        for(int jj = 0; jj < sizej*sizej; jj++)
        {
            int a = ii / sizei, b = ii - a * sizei, c = jj / sizej, d = jj - c * sizej;
            if(intType == "SSLL")
                h2eInput.J[ir][jr][ii][jj] /= norm_s[Iirrep][a] * norm_s[Iirrep][b];
            else if(intType == "SSSS")
                h2eInput.J[ir][jr][ii][jj] /= norm_s[Iirrep][a] * norm_s[Iirrep][b] * norm_s[Jirrep][c] * norm_s[Jirrep][d];
            else if(intType == "LSLS")
                h2eInput.J[ir][jr][ii][jj] /= norm_s[Iirrep][b] * norm_s[Jirrep][d];
            else if(intType == "LSSL")
                h2eInput.J[ir][jr][ii][jj] /= norm_s[Iirrep][b] * norm_s[Jirrep][c];
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
                h2eInput.K[ir][jr][ii][jj] /= norm_s[Iirrep][a] * norm_s[Jirrep][b];
            else if(intType == "SSSS")
                h2eInput.K[ir][jr][ii][jj] /= norm_s[Iirrep][a] * norm_s[Jirrep][b] * norm_s[Jirrep][c] * norm_s[Iirrep][d];
            else if(intType == "LSLS")
                h2eInput.K[ir][jr][ii][jj] /= norm_s[Jirrep][b] * norm_s[Iirrep][d];
            else if(intType == "LSSL")
                h2eInput.K[ir][jr][ii][jj] /= norm_s[Jirrep][b] * norm_s[Jirrep][c];
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
vector<double> DHF_SPH::evaluateDensity_spinor(const vector<double>& coeff_, const vector<double>& occNumber_, const int& size, const bool& twoC)
{
    if(!twoC)
    {
        int size2 = size/2;
        vector<double> den(size*size, 0.0);
        for(int aa = 0; aa < size2; aa++)
        for(int bb = 0; bb < size2; bb++)
        {
            for(int ii = 0; ii < occNumber_.size(); ii++)
            {
                den[aa*size+bb] += occNumber_[ii] * coeff_[aa*size+ii+size2] * coeff_[bb*size+ii+size2];
                den[(size2+aa)*size+bb] += occNumber_[ii] * coeff_[(size2+aa)*size+ii+size2] * coeff_[bb*size+ii+size2];
                den[aa*size+size2+bb] += occNumber_[ii] * coeff_[aa*size+ii+size2] * coeff_[(size2+bb)*size+ii+size2];
                den[(size2+aa)*size+size2+bb] += occNumber_[ii] * coeff_[(size2+aa)*size+ii+size2] * coeff_[(size2+bb)*size+ii+size2];
            }
        }
        return den;
    }
    else
    {
        vector<double> den(size*size, 0.0);
        for(int aa = 0; aa < size; aa++)
        for(int bb = 0; bb < size; bb++)
        for(int ii = 0; ii < occNumber_.size(); ii++)
            den[aa*size+bb] += occNumber_[ii] * coeff_[aa*size+ii] * coeff_[bb*size+ii];
        
        return den;
    }
}

vVectorXd DHF_SPH::evaluateDensity_spinor_irrep(const vector<int>& nbas, const bool& twoC)
{
    vVectorXd den(occMax_irrep);
    for(int ir = 0; ir < occMax_irrep; ir += irrep_list[ir].two_j+1)
    {
        den[ir] = evaluateDensity_spinor(coeff[ir], occNumber[ir], nbas[ir], twoC);
    }

    return den;
}

/* 
    evaluate Fock matrix 
*/
void DHF_SPH::evaluateFock(vector<double>& fock, const bool& twoC, const vVectorXd& den, const int& size, const int& Iirrep)
{
    evaluateFock_2e(fock, twoC, den, size, Iirrep);
    for(int ii = 0; ii < fock.size(); ii++)
        fock[ii] += h1e_4c[Iirrep][ii];
}
void DHF_SPH::evaluateFock_2e(vector<double>& fock, const bool& twoC, const vVectorXd& den, const int& size, const int& Iirrep)
{
    evaluateFock_2e(fock, twoC, den, size, Iirrep, h2eLLLL_JK, h2eSSLL_JK, h2eSSSS_JK, gauntLSLS_JK, gauntLSSL_JK);
}
void DHF_SPH::evaluateFock_2e(vector<double>& fock, const bool& twoC, const vVectorXd& den, const int& size, const int& Iirrep,
                              const int2eJK& LLLL, const int2eJK& SSLL, const int2eJK& SSSS, const int2eJK& gLSLS, const int2eJK& gLSSL)
{
    int ir = all2compact[Iirrep];
    if(!twoC)
    {
        int size2 = size*2;
        fock.resize(size2*size2);
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
            
            fock[mm*size2+nn] = 0.0;
            fock[(mm+size)*size2+nn] = 0.0;
            if(mm != nn) fock[(nn+size)*size2+mm] = 0.0;
            fock[(mm+size)*size2+nn+size] = 0.0;
            for(int jr = 0; jr < occMax_irrep_compact; jr++)
            {
                int Jirrep = compact2all[jr];
                double twojP1 = irrep_list[Jirrep].two_j+1;
                int size_tmp2 = irrep_list[Jirrep].size;
                int size2j = 2 * size_tmp2;
                for(int ss = 0; ss < size_tmp2; ss++)
                for(int rr = 0; rr < size_tmp2; rr++)
                {
                    int emn = mm*size+nn, esr = ss*size_tmp2+rr, emr = mm*size_tmp2+rr, esn = ss*size+nn;
                    fock[mm*size2+nn] += twojP1*den[Jirrep][ss*size2j+rr] * LLLL.J[ir][jr][emn][esr] + twojP1*den[Jirrep][(size_tmp2+ss)*size2j+size_tmp2+rr] * SSLL.J[jr][ir][esr][emn];
                    fock[(mm+size)*size2+nn] -= twojP1*den[Jirrep][ss*size2j+size_tmp2+rr] * SSLL.K[ir][jr][emr][esn];
                    if(mm != nn) 
                    {
                        int enr = nn*size_tmp2+rr, esm = ss*size+mm;
                        fock[(nn+size)*size2+mm] -= twojP1*den[Jirrep][ss*size2j+size_tmp2+rr] * SSLL.K[ir][jr][enr][esm];
                    }
                    fock[(mm+size)*size2+nn+size] += twojP1*den[Jirrep][(size_tmp2+ss)*size2j+size_tmp2+rr] * SSSS.J[ir][jr][emn][esr] + twojP1*den[Jirrep][ss*size2j+rr] * SSLL.J[ir][jr][emn][esr];
                    if(with_gaunt)
                    {
                        int enm = nn*size+mm, ers = rr*size_tmp2+ss, erm = rr*size+mm, ens = nn*size_tmp2+ss;
                        fock[mm*size2+nn] -= twojP1*den[Jirrep][(size_tmp2+ss)*size2j+size_tmp2+rr] * gLSSL.K[ir][jr][emr][esn];
                        fock[(mm+size)*size2+nn+size] -= twojP1*den[Jirrep][ss*size2j+rr] * gLSSL.K[jr][ir][esn][emr];
                        fock[(mm+size)*size2+nn] += twojP1*den[Jirrep][(size_tmp2+ss)*size2j+rr]*gLSLS.J[ir][jr][enm][ers] + twojP1*den[Jirrep][ss*size2j+size_tmp2+rr] * gLSSL.J[jr][ir][esr][emn];
                        if(mm != nn) 
                        {
                            int ern = rr*size+nn, ems = mm*size_tmp2+ss;
                            fock[(nn+size)*size2+mm] += twojP1*den[Jirrep][(size_tmp2+ss)*size2j+rr]*gLSLS.J[ir][jr][emn][ers] + twojP1*den[Jirrep][ss*size2j+size_tmp2+rr] * gLSSL.J[jr][ir][esr][enm];
                        }
                    }
                }
            }
            fock[nn*size2+mm] = fock[mm*size2+nn];
            fock[(nn+size)*size2+mm+size] = fock[(mm+size)*size2+nn+size];
            fock[nn*size2+mm+size] = fock[(mm+size)*size2+nn];
            fock[mm*size2+nn+size] = fock[(nn+size)*size2+mm];
        }
    }
    else
    {
        fock.resize(size*size);
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
            fock[mm*size+nn] = 0.0;
            for(int jr = 0; jr < occMax_irrep_compact; jr++)
            {
                int Jirrep = compact2all[jr];
                double twojP1 = irrep_list[Jirrep].two_j+1;
                int size_tmp2 = irrep_list[Jirrep].size;
                for(int ss = 0; ss < size_tmp2; ss++)
                for(int rr = 0; rr < size_tmp2; rr++)
                {
                    int emn = mm*size+nn, esr = ss*size_tmp2+rr, emr = mm*size_tmp2+rr, esn = ss*size+nn;
                    fock[mm*size+nn] += twojP1*den[Jirrep][ss*size_tmp2+rr] * LLLL.J[ir][jr][emn][esr];
                }
            }
            fock[nn*size+mm] = fock[mm*size+nn];
        }
    }
}
void DHF_SPH::evaluateFock_SO(vector<double>& fock, const vVectorXd& den, const int& size, const int& Iirrep,
                              const int2eJK& SSLL, const int2eJK& SSSS, const int2eJK& gLSLS, const int2eJK& gLSSL)
{
    int ir = all2compact[Iirrep];
    int size2 = size*2;
    fock.resize(size2*size2);
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
        
        fock[mm*size2+nn] = 0.0;
        fock[(mm+size)*size2+nn] = 0.0;
        if(mm != nn) fock[(nn+size)*size2+mm] = 0.0;
        fock[(mm+size)*size2+nn+size] = 0.0;
        for(int jr = 0; jr < occMax_irrep_compact; jr++)
        {
            int Jirrep = compact2all[jr];
            double twojP1 = irrep_list[Jirrep].two_j+1;
            int size_tmp2 = irrep_list[Jirrep].size;
            int size2j = 2 * size_tmp2;
            for(int ss = 0; ss < size_tmp2; ss++)
            for(int rr = 0; rr < size_tmp2; rr++)
            {
                int emn = mm*size+nn, esr = ss*size_tmp2+rr, emr = mm*size_tmp2+rr, esn = ss*size+nn;
                fock[mm*size2+nn] += twojP1*den[Jirrep][(size_tmp2+ss)*size2j+size_tmp2+rr] * SSLL.J[jr][ir][esr][emn];
                fock[(mm+size)*size2+nn] -= twojP1*den[Jirrep][ss*size2j+size_tmp2+rr] * SSLL.K[ir][jr][emr][esn];
                if(mm != nn) 
                {
                    int enr = nn*size_tmp2+rr, esm = ss*size+mm;
                    fock[(nn+size)*size2+mm] -= twojP1*den[Jirrep][ss*size2j+size_tmp2+rr] * SSLL.K[ir][jr][enr][esm];
                }
                fock[(mm+size)*size2+nn+size] += twojP1*den[Jirrep][(size_tmp2+ss)*size2j+size_tmp2+rr] * SSSS.J[ir][jr][emn][esr] + twojP1*den[Jirrep][ss*size2j+rr] * SSLL.J[ir][jr][emn][esr];
                if(with_gaunt)
                {
                    int enm = nn*size+mm, ers = rr*size_tmp2+ss, erm = rr*size+mm, ens = nn*size_tmp2+ss;
                    fock[mm*size2+nn] -= twojP1*den[Jirrep][(size_tmp2+ss)*size2j+size_tmp2+rr] * gLSSL.K[ir][jr][emr][esn];
                    fock[(mm+size)*size2+nn+size] -= twojP1*den[Jirrep][ss*size2j+rr] * gLSSL.K[jr][ir][esn][emr];
                    fock[(mm+size)*size2+nn] += twojP1*den[Jirrep][(size_tmp2+ss)*size2j+rr]*gLSLS.J[ir][jr][enm][ers] + twojP1*den[Jirrep][ss*size2j+size_tmp2+rr] * gLSSL.J[jr][ir][esr][emn];
                    if(mm != nn) 
                    {
                        int ern = rr*size+nn, ems = mm*size_tmp2+ss;
                        fock[(nn+size)*size2+mm] += twojP1*den[Jirrep][(size_tmp2+ss)*size2j+rr]*gLSLS.J[ir][jr][emn][ers] + twojP1*den[Jirrep][ss*size2j+size_tmp2+rr] * gLSSL.J[jr][ir][esr][enm];
                    }
                }
            }
        }
        fock[nn*size2+mm] = fock[mm*size2+nn];
        fock[(nn+size)*size2+mm+size] = fock[(mm+size)*size2+nn+size];
        fock[nn*size2+mm+size] = fock[(mm+size)*size2+nn];
        fock[mm*size2+nn+size] = fock[(nn+size)*size2+mm];
    }
}

/* 
    Read occupation numbers 
*/
void DHF_SPH::setOCC(const string& filename, const string& atomName)
{
    string flags;
    vector<double> vecd_tmp(10, 0.0);
    int int_tmp, int_tmp2, int_tmp3;
    ifstream ifs;
    ifs.open(filename);
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
                    ifs >> vecd_tmp[ii];
                }
                break;
            }
        }
    }
    
    if(ifs.eof() or ifs.fail())
    {
        cout << "Did NOT find %occAMFI in " << filename << endl;
        cout << "Using default occupation number for " << atomName << endl;
        if(atomName == "H") {int_tmp = 1; vecd_tmp[0] = 1.0;}
        else if(atomName == "HE") {int_tmp = 1; vecd_tmp[0] = 2.0;}
        else if(atomName == "LI") {int_tmp = 1; vecd_tmp[0] = 3.0;}
        else if(atomName == "BE") {int_tmp = 1; vecd_tmp[0] = 4.0;}
        else if(atomName == "B") {int_tmp = 3; vecd_tmp[0] = 4.0; vecd_tmp[1] = 1.0/3.0; vecd_tmp[2] = 2.0/3.0;}
        else if(atomName == "C") {int_tmp = 3; vecd_tmp[0] = 4.0; vecd_tmp[1] = 2.0/3.0; vecd_tmp[2] = 4.0/3.0;}
        else if(atomName == "N") {int_tmp = 3; vecd_tmp[0] = 4.0; vecd_tmp[1] = 1.0; vecd_tmp[2] = 2.0;}
        else if(atomName == "O") {int_tmp = 3; vecd_tmp[0] = 4.0; vecd_tmp[1] = 4.0/3.0; vecd_tmp[2] = 8.0/3.0;}
        else if(atomName == "F") {int_tmp = 3; vecd_tmp[0] = 4.0; vecd_tmp[1] = 5.0/3.0; vecd_tmp[2] = 10.0/3.0;}
        else if(atomName == "NE") {int_tmp = 3; vecd_tmp[0] = 4.0; vecd_tmp[1] = 2.0; vecd_tmp[2] = 4.0;}
        else if(atomName == "NA") {int_tmp = 3; vecd_tmp[0] = 5.0; vecd_tmp[1] = 2.0; vecd_tmp[2] = 4.0;}
        else if(atomName == "MG") {int_tmp = 3; vecd_tmp[0] = 6.0; vecd_tmp[1] = 2.0; vecd_tmp[2] = 4.0;}
        else if(atomName == "AL") {int_tmp = 3; vecd_tmp[0] = 6.0; vecd_tmp[1] = 7.0/3.0; vecd_tmp[2] = 14.0/3.0;}
        else if(atomName == "SI") {int_tmp = 3; vecd_tmp[0] = 6.0; vecd_tmp[1] = 8.0/3.0; vecd_tmp[2] = 16.0/3.0;}
        else if(atomName == "P") {int_tmp = 3; vecd_tmp[0] = 6.0; vecd_tmp[1] = 3.0; vecd_tmp[2] = 6.0;}
        else if(atomName == "S") {int_tmp = 3; vecd_tmp[0] = 6.0; vecd_tmp[1] = 10.0/3.0; vecd_tmp[2] = 20.0/3.0;}
        else if(atomName == "CL") {int_tmp = 3; vecd_tmp[0] = 6.0; vecd_tmp[1] = 11.0/3.0; vecd_tmp[2] = 22.0/3.0;}
        else if(atomName == "AR") {int_tmp = 3; vecd_tmp[0] = 6.0; vecd_tmp[1] = 4.0; vecd_tmp[2] = 8.0;}
        else if(atomName == "K") {int_tmp = 3; vecd_tmp[0] = 7.0; vecd_tmp[1] = 4.0; vecd_tmp[2] = 8.0;}
        else if(atomName == "CA") {int_tmp = 3; vecd_tmp[0] = 8.0; vecd_tmp[1] = 4.0; vecd_tmp[2] = 8.0;}
        else if(atomName == "SC") {int_tmp = 5; vecd_tmp[0] = 8.0; vecd_tmp[1] = 4.0; vecd_tmp[2] = 8.0; vecd_tmp[3] = 1.0/2.5; vecd_tmp[4] = 1.5/2.5;}
        else if(atomName == "TI") {int_tmp = 5; vecd_tmp[0] = 8.0; vecd_tmp[1] = 4.0; vecd_tmp[2] = 8.0; vecd_tmp[3] = 2.0/2.5; vecd_tmp[4] = 3.0/2.5;}
        else if(atomName == "V") {int_tmp = 5; vecd_tmp[0] = 8.0; vecd_tmp[1] = 4.0; vecd_tmp[2] = 8.0; vecd_tmp[3] = 3.0/2.5; vecd_tmp[4] = 4.5/2.5;}
        else if(atomName == "CR") {int_tmp = 5; vecd_tmp[0] = 7.0; vecd_tmp[1] = 4.0; vecd_tmp[2] = 8.0; vecd_tmp[3] = 2.0; vecd_tmp[4] = 3.0;}
        else if(atomName == "MN") {int_tmp = 5; vecd_tmp[0] = 8.0; vecd_tmp[1] = 4.0; vecd_tmp[2] = 8.0; vecd_tmp[3] = 2.0; vecd_tmp[4] = 3.0;}
        else if(atomName == "FE") {int_tmp = 5; vecd_tmp[0] = 8.0; vecd_tmp[1] = 4.0; vecd_tmp[2] = 8.0; vecd_tmp[3] = 6.0/2.5; vecd_tmp[4] = 9.0/2.5;}
        else if(atomName == "CO") {int_tmp = 5; vecd_tmp[0] = 8.0; vecd_tmp[1] = 4.0; vecd_tmp[2] = 8.0; vecd_tmp[3] = 7.0/2.5; vecd_tmp[4] = 10.5/2.5;}
        else if(atomName == "NI") {int_tmp = 5; vecd_tmp[0] = 8.0; vecd_tmp[1] = 4.0; vecd_tmp[2] = 8.0; vecd_tmp[3] = 8.0/2.5; vecd_tmp[4] = 12.0/2.5;}
        else if(atomName == "CU") {int_tmp = 5; vecd_tmp[0] = 7.0; vecd_tmp[1] = 4.0; vecd_tmp[2] = 8.0; vecd_tmp[3] = 4.0; vecd_tmp[4] = 6.0;}
        else if(atomName == "ZN") {int_tmp = 5; vecd_tmp[0] = 8.0; vecd_tmp[1] = 4.0; vecd_tmp[2] = 8.0; vecd_tmp[3] = 4.0; vecd_tmp[4] = 6.0;}
        else if(atomName == "GA") {int_tmp = 5; vecd_tmp[0] = 8.0; vecd_tmp[1] = 13.0/3.0; vecd_tmp[2] = 26.0/3.0; vecd_tmp[3] = 4.0; vecd_tmp[4] = 6.0;}
        else if(atomName == "GE") {int_tmp = 5; vecd_tmp[0] = 8.0; vecd_tmp[1] = 14.0/3.0; vecd_tmp[2] = 28.0/3.0; vecd_tmp[3] = 4.0; vecd_tmp[4] = 6.0;}
        else if(atomName == "AS") {int_tmp = 5; vecd_tmp[0] = 8.0; vecd_tmp[1] = 5.0; vecd_tmp[2] = 10.0; vecd_tmp[3] = 4.0; vecd_tmp[4] = 6.0;}
        else if(atomName == "SE") {int_tmp = 5; vecd_tmp[0] = 8.0; vecd_tmp[1] = 16.0/3.0; vecd_tmp[2] = 32.0/3.0; vecd_tmp[3] = 4.0; vecd_tmp[4] = 6.0;}
        else if(atomName == "BR") {int_tmp = 5; vecd_tmp[0] = 8.0; vecd_tmp[1] = 17.0/3.0; vecd_tmp[2] = 34.0/3.0; vecd_tmp[3] = 4.0; vecd_tmp[4] = 6.0;}
        else if(atomName == "KR") {int_tmp = 5; vecd_tmp[0] = 8.0; vecd_tmp[1] = 6.0; vecd_tmp[2] = 12.0; vecd_tmp[3] = 4.0; vecd_tmp[4] = 6.0;}
        else if(atomName == "RB") {int_tmp = 5; vecd_tmp[0] = 9.0; vecd_tmp[1] = 6.0; vecd_tmp[2] = 12.0; vecd_tmp[3] = 4.0; vecd_tmp[4] = 6.0;}
        else if(atomName == "SR") {int_tmp = 5; vecd_tmp[0] = 10.0; vecd_tmp[1] = 6.0; vecd_tmp[2] = 12.0; vecd_tmp[3] = 4.0; vecd_tmp[4] = 6.0;}
        else if(atomName == "Y") {int_tmp = 5; vecd_tmp[0] = 10.0; vecd_tmp[1] = 6.0; vecd_tmp[2] = 12.0; vecd_tmp[3] = 11.0/2.5; vecd_tmp[4] = 16.5/2.5;}
        else if(atomName == "ZR") {int_tmp = 5; vecd_tmp[0] = 10.0; vecd_tmp[1] = 6.0; vecd_tmp[2] = 12.0; vecd_tmp[3] = 12.0/2.5; vecd_tmp[4] = 18.0/2.5;}
        else if(atomName == "NB") {int_tmp = 5; vecd_tmp[0] = 9.0; vecd_tmp[1] = 6.0; vecd_tmp[2] = 12.0; vecd_tmp[3] = 14.0/2.5; vecd_tmp[4] = 21.0/2.5;}
        else if(atomName == "MO") {int_tmp = 5; vecd_tmp[0] = 9.0; vecd_tmp[1] = 6.0; vecd_tmp[2] = 12.0; vecd_tmp[3] = 6.0; vecd_tmp[4] = 9.0;}
        else if(atomName == "TC") {int_tmp = 5; vecd_tmp[0] = 10.0; vecd_tmp[1] = 6.0; vecd_tmp[2] = 12.0; vecd_tmp[3] = 6.0; vecd_tmp[4] = 9.0;}
        else if(atomName == "RU") {int_tmp = 5; vecd_tmp[0] = 9.0; vecd_tmp[1] = 6.0; vecd_tmp[2] = 12.0; vecd_tmp[3] = 17.0/2.5; vecd_tmp[4] = 25.5/2.5;}
        else if(atomName == "RH") {int_tmp = 5; vecd_tmp[0] = 9.0; vecd_tmp[1] = 6.0; vecd_tmp[2] = 12.0; vecd_tmp[3] = 18.0/2.5; vecd_tmp[4] = 27.0/2.5;}
        else if(atomName == "PD") {int_tmp = 5; vecd_tmp[0] = 8.0; vecd_tmp[1] = 6.0; vecd_tmp[2] = 12.0; vecd_tmp[3] = 8.0; vecd_tmp[4] = 12.0;}
        else if(atomName == "AG") {int_tmp = 5; vecd_tmp[0] = 9.0; vecd_tmp[1] = 6.0; vecd_tmp[2] = 12.0; vecd_tmp[3] = 8.0; vecd_tmp[4] = 12.0;}
        else if(atomName == "CD") {int_tmp = 5; vecd_tmp[0] = 10.0; vecd_tmp[1] = 6.0; vecd_tmp[2] = 12.0; vecd_tmp[3] = 8.0; vecd_tmp[4] = 12.0;}
        else if(atomName == "IN") {int_tmp = 5; vecd_tmp[0] = 10.0; vecd_tmp[1] = 19.0/3.0; vecd_tmp[2] = 38.0/3.0; vecd_tmp[3] = 8.0; vecd_tmp[4] = 12.0;}
        else if(atomName == "SN") {int_tmp = 5; vecd_tmp[0] = 10.0; vecd_tmp[1] = 20.0/3.0; vecd_tmp[2] = 40.0/3.0; vecd_tmp[3] = 8.0; vecd_tmp[4] = 12.0;}
        else if(atomName == "SB") {int_tmp = 5; vecd_tmp[0] = 10.0; vecd_tmp[1] = 21.0/3.0; vecd_tmp[2] = 42.0/3.0; vecd_tmp[3] = 8.0; vecd_tmp[4] = 12.0;}
        else if(atomName == "TE") {int_tmp = 5; vecd_tmp[0] = 10.0; vecd_tmp[1] = 22.0/3.0; vecd_tmp[2] = 44.0/3.0; vecd_tmp[3] = 8.0; vecd_tmp[4] = 12.0;}
        else if(atomName == "I") {int_tmp = 5; vecd_tmp[0] = 10.0; vecd_tmp[1] = 23.0/3.0; vecd_tmp[2] = 46.0/3.0; vecd_tmp[3] = 8.0; vecd_tmp[4] = 12.0;}
        else if(atomName == "XE") {int_tmp = 5; vecd_tmp[0] = 10.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 8.0; vecd_tmp[4] = 12.0;}
        else if(atomName == "CS") {int_tmp = 5; vecd_tmp[0] = 11.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 8.0; vecd_tmp[4] = 12.0;}
        else if(atomName == "BA") {int_tmp = 5; vecd_tmp[0] = 12.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 8.0; vecd_tmp[4] = 12.0;}
        else if(atomName == "LA") {int_tmp = 5; vecd_tmp[0] = 12.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 21.0/2.5; vecd_tmp[4] = 31.5/2.5;}
        else if(atomName == "CE") {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 21.0/2.5; vecd_tmp[4] = 31.5/2.5; vecd_tmp[5] = 3.0/7.0; vecd_tmp[6] = 4.0/7.0;}
        else if(atomName == "PR") {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 8.0; vecd_tmp[4] = 12.0; vecd_tmp[5] = 9.0/7.0; vecd_tmp[6] = 12.0/7.0;}
        else if(atomName == "ND") {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 8.0; vecd_tmp[4] = 12.0; vecd_tmp[5] = 12.0/7.0; vecd_tmp[6] = 16.0/7.0;}
        else if(atomName == "PM") {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 8.0; vecd_tmp[4] = 12.0; vecd_tmp[5] = 15.0/7.0; vecd_tmp[6] = 20.0/7.0;}
        else if(atomName == "SM") {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 8.0; vecd_tmp[4] = 12.0; vecd_tmp[5] = 18.0/7.0; vecd_tmp[6] = 24.0/7.0;}
        else if(atomName == "EU") {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 8.0; vecd_tmp[4] = 12.0; vecd_tmp[5] = 3.0; vecd_tmp[6] = 4.0;}
        else if(atomName == "GD") {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 21.0/2.5; vecd_tmp[4] = 31.5/2.5; vecd_tmp[5] = 21.0/7.0; vecd_tmp[6] = 28.0/7.0;}
        else if(atomName == "TB") {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 8.0; vecd_tmp[4] = 12.0; vecd_tmp[5] = 27.0/7.0; vecd_tmp[6] = 36.0/7.0;}
        else if(atomName == "DY") {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 8.0; vecd_tmp[4] = 12.0; vecd_tmp[5] = 30.0/7.0; vecd_tmp[6] = 40.0/7.0;}
        else if(atomName == "HO") {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 8.0; vecd_tmp[4] = 12.0; vecd_tmp[5] = 33.0/7.0; vecd_tmp[6] = 44.0/7.0;}
        else if(atomName == "ER") {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 8.0; vecd_tmp[4] = 12.0; vecd_tmp[5] = 36.0/7.0; vecd_tmp[6] = 48.0/7.0;}
        else if(atomName == "TM") {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 8.0; vecd_tmp[4] = 12.0; vecd_tmp[5] = 39.0/7.0; vecd_tmp[6] = 52.0/7.0;}
        else if(atomName == "YB") {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 8.0; vecd_tmp[4] = 12.0; vecd_tmp[5] = 6.0; vecd_tmp[6] = 8.0;}
        else if(atomName == "LU") {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 21.0/2.5; vecd_tmp[4] = 31.5/2.5; vecd_tmp[5] = 6.0; vecd_tmp[6] = 8.0;}
        else if(atomName == "HF") {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 22.0/2.5; vecd_tmp[4] = 33.0/2.5; vecd_tmp[5] = 6.0; vecd_tmp[6] = 8.0;}
        else if(atomName == "TA") {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 23.0/2.5; vecd_tmp[4] = 34.5/2.5; vecd_tmp[5] = 6.0; vecd_tmp[6] = 8.0;}
        else if(atomName == "W") {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 24.0/2.5; vecd_tmp[4] = 36.0/2.5; vecd_tmp[5] = 6.0; vecd_tmp[6] = 8.0;}
        else if(atomName == "RE") {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 10.0; vecd_tmp[4] = 15.0; vecd_tmp[5] = 6.0; vecd_tmp[6] = 8.0;}
        else if(atomName == "OS") {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 26.0/2.5; vecd_tmp[4] = 39.0/2.5; vecd_tmp[5] = 6.0; vecd_tmp[6] = 8.0;}
        else if(atomName == "IR") {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 27.0/2.5; vecd_tmp[4] = 40.5/2.5; vecd_tmp[5] = 6.0; vecd_tmp[6] = 8.0;}
        else if(atomName == "PT") {int_tmp = 7; vecd_tmp[0] = 11.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 29.0/2.5; vecd_tmp[4] = 43.5/2.5; vecd_tmp[5] = 6.0; vecd_tmp[6] = 8.0;}
        else if(atomName == "AU") {int_tmp = 7; vecd_tmp[0] = 11.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 12.0; vecd_tmp[4] = 18.0; vecd_tmp[5] = 6.0; vecd_tmp[6] = 8.0;}
        else if(atomName == "HG") {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 8.0; vecd_tmp[2] = 16.0; vecd_tmp[3] = 12.0; vecd_tmp[4] = 18.0; vecd_tmp[5] = 6.0; vecd_tmp[6] = 8.0;}
        else if(atomName == "TL") {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 25.0/3.0; vecd_tmp[2] = 50.0/3.0; vecd_tmp[3] = 12.0; vecd_tmp[4] = 18.0; vecd_tmp[5] = 6.0; vecd_tmp[6] = 8.0;}
        else if(atomName == "PB") {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 26.0/3.0; vecd_tmp[2] = 52.0/3.0; vecd_tmp[3] = 12.0; vecd_tmp[4] = 18.0; vecd_tmp[5] = 6.0; vecd_tmp[6] = 8.0;}
        else if(atomName == "BI") {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 9.0; vecd_tmp[2] = 18.0; vecd_tmp[3] = 12.0; vecd_tmp[4] = 18.0; vecd_tmp[5] = 6.0; vecd_tmp[6] = 8.0;}
        else if(atomName == "PO") {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 28.0/3.0; vecd_tmp[2] = 56.0/3.0; vecd_tmp[3] = 12.0; vecd_tmp[4] = 18.0; vecd_tmp[5] = 6.0; vecd_tmp[6] = 8.0;}
        else if(atomName == "AT") {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 29.0/3.0; vecd_tmp[2] = 58.0/3.0; vecd_tmp[3] = 12.0; vecd_tmp[4] = 18.0; vecd_tmp[5] = 6.0; vecd_tmp[6] = 8.0;}
        else if(atomName == "RN") {int_tmp = 7; vecd_tmp[0] = 12.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 12.0; vecd_tmp[4] = 18.0; vecd_tmp[5] = 6.0; vecd_tmp[6] = 8.0;}
        else if(atomName == "FR") {int_tmp = 7; vecd_tmp[0] = 13.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 12.0; vecd_tmp[4] = 18.0; vecd_tmp[5] = 6.0; vecd_tmp[6] = 8.0;}
        else if(atomName == "RA") {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 12.0; vecd_tmp[4] = 18.0; vecd_tmp[5] = 6.0; vecd_tmp[6] = 8.0;}
        else if(atomName == "AC") {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 31.0/2.5; vecd_tmp[4] = 46.5/2.5; vecd_tmp[5] = 6.0; vecd_tmp[6] = 8.0;}
        else if(atomName == "TH") {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 32.0/2.5; vecd_tmp[4] = 48.0/2.5; vecd_tmp[5] = 6.0; vecd_tmp[6] = 8.0;}
        else if(atomName == "PA") {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 31.0/2.5; vecd_tmp[4] = 46.5/2.5; vecd_tmp[5] = 48.0/7.0; vecd_tmp[6] = 64.0/7.0;}
        else if(atomName == "U") {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 31.0/2.5; vecd_tmp[4] = 46.5/2.5; vecd_tmp[5] = 51.0/7.0; vecd_tmp[6] = 68.0/7.0;}
        else if(atomName == "NP") {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 31.0/2.5; vecd_tmp[4] = 46.5/2.5; vecd_tmp[5] = 54.0/7.0; vecd_tmp[6] = 72.0/7.0;}
        else if(atomName == "PU") {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 12.0; vecd_tmp[4] = 18.0; vecd_tmp[5] = 60.0/7.0; vecd_tmp[6] = 80.0/7.0;}
        else if(atomName == "AM") {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 12.0; vecd_tmp[4] = 18.0; vecd_tmp[5] = 63.0/7.0; vecd_tmp[6] = 84.0/7.0;}
        else if(atomName == "CM") {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 31.0/2.5; vecd_tmp[4] = 46.5/2.5; vecd_tmp[5] = 63.0/7.0; vecd_tmp[6] = 84.0/7.0;}
        else if(atomName == "BK") {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 12.0; vecd_tmp[4] = 18.0; vecd_tmp[5] = 69.0/7.0; vecd_tmp[6] = 92.0/7.0;}
        else if(atomName == "CF") {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 12.0; vecd_tmp[4] = 18.0; vecd_tmp[5] = 72.0/7.0; vecd_tmp[6] = 96.0/7.0;}
        else if(atomName == "ES") {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 12.0; vecd_tmp[4] = 18.0; vecd_tmp[5] = 75.0/7.0; vecd_tmp[6] = 100.0/7.0;}
        else if(atomName == "FM") {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 12.0; vecd_tmp[4] = 18.0; vecd_tmp[5] = 78.0/7.0; vecd_tmp[6] = 104.0/7.0;}
        else if(atomName == "MD") {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 12.0; vecd_tmp[4] = 18.0; vecd_tmp[5] = 81.0/7.0; vecd_tmp[6] = 108.0/7.0;}
        else if(atomName == "NO") {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 12.0; vecd_tmp[4] = 18.0; vecd_tmp[5] = 12.0; vecd_tmp[6] = 16.0;}
        else if(atomName == "LR") {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 31.0/3.0; vecd_tmp[2] = 62.0/3.0; vecd_tmp[3] = 12.0; vecd_tmp[4] = 18.0; vecd_tmp[5] = 12.0; vecd_tmp[6] = 16.0;}
        else if(atomName == "RF") {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 32.0/2.5; vecd_tmp[4] = 48.0/2.5; vecd_tmp[5] = 12.0; vecd_tmp[6] = 16.0;}
        else if(atomName == "DB") {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 33.0/2.5; vecd_tmp[4] = 49.5/2.5; vecd_tmp[5] = 12.0; vecd_tmp[6] = 16.0;}
        else if(atomName == "SG") {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 34.0/2.5; vecd_tmp[4] = 51.0/2.5; vecd_tmp[5] = 12.0; vecd_tmp[6] = 16.0;}
        else if(atomName == "BH") {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 14.0; vecd_tmp[4] = 21.0; vecd_tmp[5] = 12.0; vecd_tmp[6] = 16.0;}
        else if(atomName == "HS") {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 10.0; vecd_tmp[2] = 20.0; vecd_tmp[3] = 36.0/2.5; vecd_tmp[4] = 54.0/2.5; vecd_tmp[5] = 12.0; vecd_tmp[6] = 16.0;}

        else if(atomName == "OG") {int_tmp = 7; vecd_tmp[0] = 14.0; vecd_tmp[1] = 12.0; vecd_tmp[2] = 24.0; vecd_tmp[3] = 16.0; vecd_tmp[4] = 24.0; vecd_tmp[5] = 12.0; vecd_tmp[6] = 16.0;}
        else
        {
            cout << "ERROR: " << atomName << " does NOT have a default ON. Please input the occupation numbers by hand." << endl;
            exit(99);
        }            
    }
    int_tmp2 = 0;
    for(int ii = 0; ii < int_tmp; ii++)
    {
        int int_tmp3 = vecd_tmp[ii] / (irrep_list[int_tmp2].two_j + 1);
        double d_tmp = (double)(vecd_tmp[ii] - int_tmp3*(irrep_list[int_tmp2].two_j + 1)) / (double)(irrep_list[int_tmp2].two_j + 1);
        for(int jj = 0; jj < irrep_list[int_tmp2].two_j + 1; jj++)
        {
            occNumber[int_tmp2+jj].resize(irrep_list[int_tmp2+jj].size, 0.0);
            for(int kk = 0; kk < int_tmp3; kk++)
            {    
                occNumber[int_tmp2+jj][kk] = 1.0;
            }
            if(occNumber[int_tmp2+jj].size() > int_tmp3)
                occNumber[int_tmp2+jj][int_tmp3] = d_tmp;
        }
        occMax_irrep += irrep_list[int_tmp2].two_j+1;
        int_tmp2 += irrep_list[int_tmp2].two_j+1;
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
vVectorXd DHF_SPH::get_amfi_unc(INT_SPH& int_sph_, const bool& twoC, const string& Xmethod, bool amfi_with_gaunt, bool amfi_with_gauge, bool amfi4c)
{
    cout << "Running DHF_SPH::get_amfi_unc" << endl;
    if(with_gaunt && !amfi_with_gaunt)
    {
        cout << endl << "ATTENTION! Since gaunt terms are included in SCF, they are automatically calculated in amfi integrals." << endl << endl;
        amfi_with_gaunt = true;
        if(with_gauge && !amfi_with_gauge)
            amfi_with_gauge = true;
    }
    if((!with_gaunt && amfi_with_gaunt) || twoC)
    {
        countTime(StartTimeCPU,StartTimeWall);
        int_sph_.get_h2e_JK_gaunt_direct(gauntLSLS_JK,gauntLSSL_JK);
        countTime(EndTimeCPU,EndTimeWall);
        printTime("2e-Gaunt-integrals");
        if(amfi_with_gauge)
        {
            int2eJK tmp1, tmp2, tmp3, tmp4;
            int_sph_.get_h2e_JK_gauge_direct(tmp1,tmp2);
            for(int ir = 0; ir < Nirrep_compact; ir++)
            for(int jr = 0; jr < Nirrep_compact; jr++)
            {
                int size_i = irrep_list[compact2all[ir]].size, size_j = irrep_list[compact2all[jr]].size;
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
            countTime(EndTimeCPU,EndTimeWall);
            printTime("2e-gauge-integrals");
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
        //     int size_i = irrep_list(compact2all[ir]).size, size_j = irrep_list(compact2all[jr]).size;
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
        return get_amfi_unc_2c(SSLL_SD, SSSS_SD, amfi_with_gaunt, amfi4c);
    }
    else 
    {
        if(occMax_irrep < Nirrep && Xmethod == "fullFock")
        {
            cout << "fullFock is used in amfi function with incomplete h2e." << endl;
            cout << "Recalculate h2e and gaunt2e..." << endl;
            countTime(StartTimeCPU,StartTimeWall);
            int_sph_.get_h2e_JK_direct(h2eLLLL_JK,h2eSSLL_JK,h2eSSSS_JK);
            symmetrize_JK(h2eLLLL_JK,Nirrep_compact);
            symmetrize_JK(h2eSSSS_JK,Nirrep_compact);
            if(renormalizedSmall)
            {
                renormalize_h2e(h2eSSLL_JK,"SSLL");
                renormalize_h2e(h2eSSSS_JK,"SSSS");
            }
            countTime(EndTimeCPU,EndTimeWall);
            printTime("Extra 2e-integrals");
        }
        return get_amfi_unc(SSLL_SD, SSSS_SD, gauntLSLS_SD, gauntLSSL_SD, density, Xmethod, amfi_with_gaunt, amfi4c);
    }
}

vVectorXd DHF_SPH::get_amfi_unc(const int2eJK& h2eSSLL_SD, const int2eJK& h2eSSSS_SD, const int2eJK& gauntLSLS_SD, const int2eJK& gauntLSSL_SD, const vVectorXd& density_, const string& Xmethod, const bool& amfi_with_gaunt, bool amfi4c)
{
    if(!converged)
    {
        cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
        cout << "!!  WARNING: Dirac HF did NOT converge  !!" << endl;
        cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
    }
    vVectorXd overlap_4c_full(Nirrep), h1e_4c_full(Nirrep), SO_4c(Nirrep), amfi_unc(Nirrep);
    /*
        Construct h1e_4c_full and overlap_4c_full 
    */
    for(int ir = 0; ir < occMax_irrep; ir++)
    {
        h1e_4c_full[ir] = h1e_4c[ir];
        overlap_4c_full[ir] = overlap_4c[ir];
    }   
    for(int ir = occMax_irrep; ir < Nirrep; ir++)
    {
        int size_tmp = irrep_list[ir].size;
        h1e_4c_full[ir].resize(size_tmp*2*size_tmp*2);
        overlap_4c_full[ir].resize(size_tmp*2*size_tmp*2);
        for(int mm = 0; mm < size_tmp; mm++)
        for(int nn = 0; nn < size_tmp; nn++)
        {
            overlap_4c_full[ir][mm*size_tmp*2+nn] = overlap[ir][mm*size_tmp+nn];
            overlap_4c_full[ir][(size_tmp+mm)*size_tmp*2+nn] = 0.0;
            overlap_4c_full[ir][mm*size_tmp*2+size_tmp+nn] = 0.0;
            overlap_4c_full[ir][(size_tmp+mm)*size_tmp*2+size_tmp+nn] = kinetic[ir][mm*size_tmp+nn] / 2.0 / speedOfLight / speedOfLight;
            h1e_4c_full[ir][mm*size_tmp*2+nn] = Vnuc[ir][mm*size_tmp+nn];
            h1e_4c_full[ir][(size_tmp+mm)*size_tmp*2+nn] = kinetic[ir][mm*size_tmp+nn];
            h1e_4c_full[ir][mm*size_tmp*2+size_tmp+nn] = kinetic[ir][mm*size_tmp+nn];
            h1e_4c_full[ir][(size_tmp+mm)*size_tmp*2+size_tmp+nn] = WWW[ir][mm*size_tmp+nn]/4.0/speedOfLight/speedOfLight - kinetic[ir][mm*size_tmp+nn];
        }
    }
    
    for(int ir = 0; ir < Nirrep; ir++)
    {
        int ir_c = all2compact[ir];
        int size = irrep_list[ir].size;
        evaluateFock_SO(SO_4c[ir], density_, irrep_list[ir].size, ir, h2eSSLL_SD, h2eSSSS_SD, gauntLSLS_SD, gauntLSSL_SD);
        /* 
            Evaluate X with various options
        */
        if(Xmethod == "h1e")
        {
            x2cXXX[ir] = X2C::get_X(overlap[ir],kinetic[ir],WWW[ir],Vnuc[ir],irrep_list[ir].size);
        }
        else
        {
            if(ir < occMax_irrep)
            {
                x2cXXX[ir] = X2C::get_X(coeff[ir], irrep_list[ir].size);
            }
            else
            {
                if(Xmethod == "partialFock")
                    x2cXXX[ir] = X2C::get_X(overlap[ir],kinetic[ir],WWW[ir],Vnuc[ir],irrep_list[ir].size);
                else if(Xmethod == "fullFock")
                {
                    vector<double> fock_tmp;
                    evaluateFock_2e(fock_tmp, false, density_, size, ir);
                    for(int mm = 0; mm < fock_tmp.size(); mm++)
                        fock_tmp[mm] += h1e_4c_full[ir][mm];
                    vector<double> overlap_half_i_4c_tmp = matrix_half_inverse(overlap_4c_full[ir], 2*size);
                    vector<double> ene_orb_tmp, coeff_tmp;
                    vector<double> tmpV;
                    eigensolverG(fock_tmp,overlap_half_i_4c_tmp,ene_orb_tmp,coeff_tmp, 2*size);
                    x2cXXX[ir] = X2C::get_X(coeff_tmp, size);
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
        x2cRRR[ir] = X2C::get_R(overlap_4c_full[ir],x2cXXX[ir],irrep_list[ir].size);
        amfi_unc[ir] = X2C::transform_4c_2c(SO_4c[ir], x2cXXX[ir], x2cRRR[ir], size);
    }

    X_calculated = true;
    if(amfi4c)
        return SO_4c;
    else
        return amfi_unc;
}

/* 
    Evaluate amfi SOC integrals in j-adapted spinor basis for two-component calculation
    X matrices are obtained from the corresponding (SF)X2C-1e procedure.
*/
vVectorXd DHF_SPH::get_amfi_unc_2c(const int2eJK& h2eSSLL_SD, const int2eJK& h2eSSSS_SD, const bool& amfi_with_gaunt, bool amfi4c)
{
    if(!converged)
    {
        cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
        cout << "!!  WARNING: 2-c HF did NOT converge  !!" << endl;
        cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
    }

    vVectorXd overlap_2c_full(Nirrep), SO_4c(Nirrep), h1e_2c_full(Nirrep), amfi_unc(Nirrep);
    /*
        Construct h1e_2c_full and overlap_2c_full 
    */
    for(int ir = 0; ir < occMax_irrep; ir++)
    {
        h1e_2c_full[ir] = h1e_4c[ir];
        overlap_2c_full[ir] = overlap_4c[ir];
    }   
    for(int ir = occMax_irrep; ir < Nirrep; ir++)
    {
        x2cXXX[ir] = X2C::get_X(overlap[ir],kinetic[ir],WWW[ir],Vnuc[ir],irrep_list[ir].size);
        x2cRRR[ir] = X2C::get_R(overlap[ir],kinetic[ir],x2cXXX[ir],irrep_list[ir].size);
        h1e_2c_full[ir] = X2C::evaluate_h1e_x2c(overlap[ir],kinetic[ir],WWW[ir],Vnuc[ir],x2cXXX[ir],x2cRRR[ir],irrep_list[ir].size);
        int N = irrep_list[ir].size;
        overlap_2c_full[ir].resize(N*N);
        for(int ii = 0; ii < N; ii++)
        for(int jj = 0; jj < N; jj++)
            overlap_2c_full[ir][ii*N+jj] = overlap[ir][ii*N+jj];
    }
    /*
        Calculate 4-c density using approximate PES C_L and C_S
        C_L = R C_{2c}
        C_S = X C_L
        Please see L. Cheng, et al, J. Chem. Phys. 141, 164107 (2014)
    */
    vVectorXd coeff_L_tmp(occMax_irrep), coeff_S_tmp(occMax_irrep);
    vVectorXd coeff_tmp(occMax_irrep), density_tmp(occMax_irrep);
    for(int ir = 0; ir < occMax_irrep; ir++)
    {
        int size_tmp = irrep_list[ir].size;
        coeff_L_tmp[ir].resize(x2cRRR[ir].size());
        coeff_S_tmp[ir].resize(x2cRRR[ir].size());
        dgemm_itrf('n','n',size_tmp,size_tmp,size_tmp,1.0,x2cRRR[ir],coeff[ir],0.0,coeff_L_tmp[ir]);
        dgemm_itrf('n','n',size_tmp,size_tmp,size_tmp,1.0,x2cXXX[ir],coeff_L_tmp[ir],0.0,coeff_S_tmp[ir]);
        coeff_tmp[ir].resize(2*size_tmp*2*size_tmp);
        std::fill(coeff_tmp[ir].begin(), coeff_tmp[ir].end(), 0.0);
        for(int ii = 0; ii < size_tmp; ii++)
        for(int jj = 0; jj < size_tmp; jj++)
        {
            coeff_tmp[ir][ii*size_tmp*2+size_tmp+jj] = coeff_L_tmp[ir][ii*size_tmp+jj];
            coeff_tmp[ir][(size_tmp+ii)*size_tmp*2+size_tmp+jj] = coeff_S_tmp[ir][ii*size_tmp+jj];
        }
        density_tmp[ir] = evaluateDensity_spinor(coeff_tmp[ir],occNumber[ir],irrep_list[ir].size*2,false);
    }

    for(int ir = 0; ir < Nirrep; ir++)
    {
        int ir_c = all2compact[ir];
        int size = irrep_list[ir].size;
        evaluateFock_SO(SO_4c[ir], density_tmp, irrep_list[ir].size, ir, h2eSSLL_SD, h2eSSSS_SD, gauntLSLS_JK, gauntLSSL_JK);
        amfi_unc[ir] = X2C::transform_4c_2c(SO_4c[ir], x2cXXX[ir], x2cRRR[ir], size);
    }

    X_calculated = true;
    if(amfi4c)
        return SO_4c;
    else
        return amfi_unc;
}
/* 
    Return SO_4c before x2c transformation.
    This is for perturbative treatment of SOC inetegrals and testing purpose.
*/
vVectorXd DHF_SPH::get_amfi_unc_4c(INT_SPH& int_sph_, const bool& twoC, const string& Xmethod, bool amfi_with_gaunt, bool amfi_with_gauge)
{
    return DHF_SPH::get_amfi_unc(int_sph_, twoC,Xmethod, amfi_with_gaunt, amfi_with_gauge, true);
}

/*
    Get coeff for basis set
    2c -> coeff
    4c -> x2c2e coeff
*/
vVectorXd DHF_SPH::get_coeff_bs(const bool& twoC)
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
        vVectorXd XXX(occMax_irrep), RRR(occMax_irrep);
        vVectorXd coeff_2c(occMax_irrep);
        for(int ir = 0; ir < occMax_irrep; ir++)
        {
            int size = irrep_list[ir].size;
            XXX[ir] = X2C::get_X(coeff[ir], size);
            RRR[ir] = X2C::get_R(overlap_4c[ir],XXX[ir], size);
            vector<double> R_inv = matInv_d(RRR[ir], size);
            vector<double> CL = matBlock(coeff[ir], size*2, 0, size, size, size);
            dgemm_itrf('n','n',size,size,size,1.0,R_inv,CL,0.0,coeff_2c[ir]);
        }
        return coeff_2c;
    }
}


/*
    Get private variable
*/
vVectorXd DHF_SPH::get_fock_4c()
{
    return fock_4c;
}
vVectorXd DHF_SPH::get_fock_4c_2ePart()
{
    vVectorXd fock_2e(occMax_irrep);
    for(int ii = 0; ii < occMax_irrep; ii++)
    {
        fock_2e[ii].resize(fock_4c[ii].size());
        for(int jj = 0; jj < fock_4c[ii].size(); jj++)
            fock_2e[ii][jj] = fock_4c[ii][jj] - h1e_4c[ii][jj];
    }
        
    return fock_2e;
}
vVectorXd DHF_SPH::get_h1e_4c()
{
    return h1e_4c;
}
vVectorXd DHF_SPH::get_overlap_4c()
{
    return overlap_4c;
}
vVectorXd DHF_SPH::get_density()
{
    return density;
}
vVectorXd DHF_SPH::get_occNumber()
{
    return occNumber;
}
vVectorXd DHF_SPH::get_X()
{
    if(X_calculated)
        return x2cXXX;
    else
    {
        cout << "ERROR: get_X was called before X matrices calculated!" << endl;
        exit(99);
    }
}
vVectorXd DHF_SPH::get_X_normalized()
{
    /*
        return X_tilde = CS_tilde CL_tilde^-1 = (T/2c2)^{1/2} XXX S^{-1/2}
        where C_tilde = S^{1/2} C  
    */   
    if(X_calculated)
    {
        vVectorXd X_tilde(occMax_irrep);
        for(int ir = 0; ir < occMax_irrep; ir++)
        {
            int size = irrep_list[ir].size;
            vector<double> tmp1(size*size),tmp2(size*size),tmp3;
            for(int ii = 0; ii < size; ii++)
            for(int jj = 0; jj < size; jj++)
            {
                tmp1[ii*size+jj] = overlap_half_i_4c[ir][ii*size*2+jj];
                tmp2[ii*size+jj] = overlap_half_i_4c[ir][(size+ii)*size*2+size+jj];
            }
            tmp2 = matInv_d(tmp2, size);
            dgemm_itrf('n','n',size,size,size,1.0,tmp2,x2cXXX[ir],0.0,tmp3);
            dgemm_itrf('n','n',size,size,size,1.0,tmp3,tmp1,0.0,X_tilde[ir]);
            // X_tilde(ir) = tmp2*x2cXXX[ir]*tmp1;
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
void DHF_SPH::set_h1e_4c(const vVectorXd& inputM)
{
    cout << "VERY DANGEROUS!! You changed h1e_4c!!" << endl;
    for(int ir = 0; ir < inputM.size(); ir++)
    {
        h1e_4c[ir] = inputM[ir];
    }
    return;
}


double DHF_SPH::radialDensity(double rr, const vVectorXd& den)
{
    bool twoC = true;
    if(pow(irrep_list[0].size*2,2) == density[0].size())
    {
        twoC = false;
    }
    if(!converged)
    {
        cout << "WARNING: SCF not converged in radialDensity." << endl;
    }

    double rho = 0.0;
    for(int ir = 0; ir < Nirrep; ir+=irrep_list[ir].two_j+1)
    {
        int size = irrep_list[ir].size, two_j = irrep_list[ir].two_j, two_mj = irrep_list[ir].two_mj, ll = irrep_list[ir].l;
        int size2 = twoC ? size : size*2;
        double kappa = (two_j + 1.0) * (ll - two_j/2.0);
        double lk = ll + kappa + 1.0;
        for(int mm = 0; mm < size; mm++)
        for(int nn = 0; nn < size; nn++)
        {
            double norm_m = shell_list[ll].norm[mm], alpha_m = shell_list[ll].exp_a[mm];
            double norm_n = shell_list[ll].norm[nn], alpha_n = shell_list[ll].exp_a[nn];
            rho += den[ir][mm*size2+nn]/norm_m/norm_n*pow(rr,2*ll)*exp(-(alpha_m+alpha_n)*rr*rr)*(two_j+1);
            if(!twoC)
            {
                double tmp = 4.0*alpha_m*alpha_n*pow(rr,2*ll+2);
                if(ll>=1) 
                {
                    tmp -= 2.0*lk*(alpha_m+alpha_n)*pow(rr,2*ll);
                    tmp += lk*lk*pow(rr,2*ll-2);
                }
                rho += den[ir][(mm+size)*size2+nn+size]/norm_m/norm_n*tmp*exp(-(alpha_m+alpha_n)*rr*rr)/4.0/speedOfLight/speedOfLight*(two_j+1);
            }
        }
    }
    return rho;
}

double DHF_SPH::radialDensity_OCC(double rr, const vVectorXd& occ)
{
    bool twoC = true;
    if(pow(irrep_list[0].size*2,2) == density[0].size())
    {
        twoC = false;
    }
    
    vVectorXd den(density.size());
    for(int ir = 0; ir < occ.size(); ir++)
    {
        den[ir] = evaluateDensity_spinor(coeff[ir], occ[ir], twoC ? irrep_list[ir].size : irrep_list[ir].size*2, twoC);
    }
    return radialDensity(rr,den);
}

double DHF_SPH::radialDensity(double rr)
{
    return radialDensity(rr,density);
}