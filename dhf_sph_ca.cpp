#include"dhf_sph_ca.h"
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

DHF_SPH_CA::DHF_SPH_CA(INT_SPH& int_sph_, const string& filename, const bool& spinFree, const bool& twoC, const bool& with_gaunt_, const bool& with_gauge_, const bool& allInt, const bool& gaussian_nuc):
DHF_SPH(int_sph_,filename,spinFree,twoC,with_gaunt_,with_gauge_,allInt,gaussian_nuc)
{
    vector<int> openIrreps;
    for(int ir = 0; ir < occMax_irrep; ir+=4*irrep_list[ir].l+2)
    {
        for(int ii = 0; ii < occNumber[ir].size(); ii++)
        {
            // if 1 > occNumber(ir)(ii) > 0
            if(abs(occNumber[ir][ii]) > 1e-4 && occNumber[ir][ii] < 0.9999)
            {
                openIrreps.push_back(ir);
                break;
            }
        }
    }
    NOpenShells = openIrreps.size();
    occNumberShells.resize(NOpenShells+2);
    for(int ii = 0; ii < NOpenShells; ii++)
    {
        MM_list.push_back(irrep_list[openIrreps[ii]].l*4+2);  
    }
    occNumberShells[0] = occNumber;
    for(int ii = 1; ii < occNumberShells.size()-1; ii++)
        occNumberShells[ii].resize(occMax_irrep);
    occNumberShells[NOpenShells+1].resize(irrep_list.size());

    for(int ir = 0; ir < occMax_irrep; ir++)
    {
        for(int ii = 1; ii < occNumberShells.size()-1; ii++)
        {
            occNumberShells[ii][ir].resize(irrep_list[ir].size, 0.0);
        }
        occNumberShells[NOpenShells+1][ir].resize(irrep_list[ir].size, 1.0);
            
        for(int ii = 0; ii < occNumber[ir].size(); ii++)
        {
            if(abs(occNumber[ir][ii]) > 1e-4)
            {
                // if 1 > occNumber(ir)(ii) > 0
                if(occNumber[ir][ii] < 0.9999)
                {
                    if(ir == openIrreps[f_list.size()])
                    {
                        f_list.push_back(occNumber[ir][ii]);
                        NN_list.push_back(f_list[f_list.size()-1]*MM_list[f_list.size()-1]);
                    }
                    occNumberShells[0][ir][ii] = 0.0;
                    occNumberShells[f_list.size()][ir][ii] = 1.0;
                }
                occNumberShells[NOpenShells+1][ir][ii] = 0.0;
            }
        }
    }
    for(int ir = occMax_irrep; ir < irrep_list.size(); ir++)
    {
        for(int jj = 0; jj < irrep_list[ir].size; jj++)
            occNumberShells[NOpenShells+1][ir][jj] = 1.0;
    }

    cout << "Open shell occupations:" << endl;
    for(int ir = 0; ir < occMax_irrep; ir+=irrep_list[ir].l*4+2)
    {
        cout << "l = " << irrep_list[ir].l << endl;
        for(int ii = 0; ii < occNumberShells.size(); ii++)
        {
            cout << ii << ":"; 
            for(int jj = 0; jj < occNumberShells[ii][ir].size(); jj++)
                cout << "\t" << occNumberShells[ii][ir][jj];
            cout << endl;
        }
    }
    for(int ir = occMax_irrep; ir < irrep_list.size(); ir++)
    {
        cout << occNumberShells.size()-1 << ":";
        for(int jj = 0; jj < occNumberShells[occNumberShells.size()-1][ir].size(); jj++)
            cout << "\t" << occNumberShells[occNumberShells.size()-1][ir][jj];
        cout << endl;
    }
    cout << "Configuration-averaged HF initialization." << endl;
    cout << "Number of open shells: " << NOpenShells << endl;
    cout << "No.\tMM\tNN\tf=NN/MM" << endl;
    for(int ii = 0; ii < NOpenShells; ii++)
    {
        cout << ii+1 << "\t" << MM_list[ii] << "\t" << NN_list[ii] << "\t" << f_list[ii] << endl;
    }
}


DHF_SPH_CA::~DHF_SPH_CA()
{
}



/*
    Evaluate density matrix
*/
vector<double> DHF_SPH_CA::evaluateDensity_aoc(const vector<double>& coeff_, const vector<double>& occNumber_, const int& size, const bool& twoC)
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
                if(abs(occNumber_[ii] - 1.0) < 1e-5)
                {    
                    den[aa*size+bb] += coeff_[aa*size+ii+size2] * coeff_[bb*size+ii+size2];
                    den[(size2+aa)*size+bb] += coeff_[(size2+aa)*size+ii+size2] * coeff_[bb*size+ii+size2];
                    den[aa*size+size2+bb] += coeff_[aa*size+ii+size2] * coeff_[(size2+bb)*size+ii+size2];
                    den[(size2+aa)*size+size2+bb] += coeff_[(size2+aa)*size+ii+size2] * coeff_[(size2+bb)*size+ii+size2];
                }
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
        {
            if(abs(occNumber_[ii] - 1.0) < 1e-5)
                den[aa*size+bb] += coeff_[aa*size+ii] * coeff_[bb*size+ii];
        }
        return den;
    }
}

/*
    SCF procedure for 4-c and 2-c calculation
*/
void DHF_SPH_CA::runSCF(const bool& twoC, const bool& renormSmall)
{
    vector<int> nbas(occMax_irrep);
    for(int ir = 0; ir < occMax_irrep; ir++)
        nbas[ir] = twoC ? irrep_list[ir].size : irrep_list[ir].size*2;

    if(renormSmall && !twoC)
    {
        renormalize_small();
    }
    vVectorXd error4DIIS[occMax_irrep][NOpenShells+2];
    vVectorXd fock4DIIS[occMax_irrep];
    countTime(StartTimeCPU,StartTimeWall);
    cout << endl;
    if(twoC) cout << "Start CA-X2C-1e Hartree-Fock iterations..." << endl;
    else cout << "Start CA-Dirac Hartree-Fock iterations..." << endl;
    cout << endl;

    densityShells.resize(NOpenShells+2);
    vector<vVectorXd> newDensityShells(NOpenShells+2);
    eigensolverG_irrep(h1e_4c, overlap_half_i_4c, ene_orb, coeff);
    for(int ii = 0; ii < NOpenShells+1; ii++)
    {
        densityShells[ii].resize(occMax_irrep);
        newDensityShells[ii].resize(occMax_irrep);
    }
    densityShells[NOpenShells+1].resize(irrep_list.size());
    newDensityShells[NOpenShells+1].resize(irrep_list.size());

    for(int ir = 0; ir < occMax_irrep; ir+=irrep_list[ir].two_j+1)
    {
        for(int ii = 0; ii < NOpenShells+2; ii++)
            densityShells[ii][ir] = evaluateDensity_aoc(coeff[ir],occNumberShells[ii][ir],nbas[ir],twoC);
    }
    for(int ir = occMax_irrep; ir < irrep_list.size(); ir+=irrep_list[ir].two_j+1)
    {
        //WORNG
        densityShells[NOpenShells+1][ir] = evaluateDensity_aoc(coeff[ir],occNumberShells[NOpenShells+1][ir],nbas[ir],twoC);
    }

    for(int iter = 1; iter <= maxIter; iter++)
    {
        if(iter <= 2)
        {
            for(int ir = 0; ir < occMax_irrep; ir += irrep_list[ir].two_j+1)    
            {
                int size_tmp = irrep_list[ir].size;
                evaluateFock(fock_4c[ir],twoC,densityShells,size_tmp,ir);
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
                    for(int kk = 0; kk < NOpenShells+1; kk++)
                    for(int ll = 0; ll < error4DIIS[ir][kk][ii].size(); ll++)
                        B4DIIS[ii*tmp_size+jj] += (error4DIIS[ir][kk][ii][ll]*error4DIIS[ir][kk][jj][ll]);
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

        for(int ir = 0; ir < occMax_irrep; ir+=irrep_list[ir].two_j+1)
        {
            for(int ii = 0; ii < NOpenShells+2; ii++)
                newDensityShells[ii][ir] = evaluateDensity_aoc(coeff[ir],occNumberShells[ii][ir],nbas[ir],twoC);
        }
        for(int ir = occMax_irrep; ir < irrep_list.size(); ir+=irrep_list[ir].two_j+1)
        {
            //WORNG
            newDensityShells[NOpenShells+1][ir] = evaluateDensity_aoc(coeff[ir],occNumberShells[NOpenShells+1][ir],nbas[ir],twoC);
        }
        d_density = 0.0;
        for(int ii = 0; ii < NOpenShells+1; ii++)
            d_density += evaluateChange_irrep(densityShells[ii],newDensityShells[ii]); 
        cout << "Iter #" << iter << " maximum density difference: " << d_density << endl;     
        for(int ii = 0; ii < NOpenShells+2; ii++)
            densityShells[ii] = newDensityShells[ii];

        if(d_density < convControl) 
        {
            converged = true;
            cout << endl << "CA-SCF converges after " << iter << " iterations." << endl;
            cout << endl << "WARNING: CA-SCF orbital energies are fake!!!" << endl << endl;

            cout << "\tOrbital\t\tEnergy(in hartree)\n";
            cout << "\t*******\t\t******************\n";
            for(int ir = 0; ir < occMax_irrep; ir += irrep_list[ir].two_j+1)
            for(int ii = 1; ii <= irrep_list[ir].size; ii++)
            {
                if(twoC) cout << "\t" << ii << "\t\t" << setprecision(15) << ene_orb[ir][ii - 1] << endl;
                else cout << "\t" << ii << "\t\t" << setprecision(15) << ene_orb[ir][irrep_list[ir].size + ii - 1] << endl;
            }
            ene_scf = evaluateEnergy(twoC);
            if(twoC) cout << "Final CA-X2C-1e HF energy is " << setprecision(15) << ene_scf << " hartree." << endl;
            else cout << "Final CA-DHF energy is " << setprecision(15) << ene_scf << " hartree." << endl;
            break;            
        }
        for(int ir = 0; ir < occMax_irrep; ir += irrep_list[ir].two_j+1)    
        {
            int size_tmp = irrep_list[ir].size;
            evaluateFock(fock_4c[ir],twoC,densityShells,size_tmp,ir);
            eigensolverG(fock_4c[ir], overlap_half_i_4c[ir], ene_orb[ir], coeff[ir], nbas[ir]);
            for(int ii = 0; ii < NOpenShells+2; ii++)
            {
                newDensityShells[ii][ir] = evaluateDensity_aoc(coeff[ir],occNumberShells[ii][ir],nbas[ir],twoC);
                error4DIIS[ir][ii].push_back(evaluateErrorDIIS(densityShells[ii][ir],newDensityShells[ii][ir]));
            }
            fock4DIIS[ir].push_back(fock_4c[ir]);
    
            if(error4DIIS[ir][0].size() > size_DIIS)
            {
                for(int ii = 0; ii < NOpenShells+2; ii++)
                    error4DIIS[ir][ii].erase(error4DIIS[ir][ii].begin());
                fock4DIIS[ir].erase(fock4DIIS[ir].begin());
            }            
        }
    }

    density.resize(occMax_irrep);
    for(int ir = 0; ir < occMax_irrep; ir += irrep_list[ir].two_j+1)
    {
        density[ir] = densityShells[0][ir];
        for(int ii = 1; ii < NOpenShells+1; ii++)
        for(int jj = 0; jj < densityShells[ii][ir].size(); jj++)
            density[ir][jj] += f_list[ii-1]*densityShells[ii][ir][jj];
        for(int jj = 1; jj < irrep_list[ir].two_j+1; jj++)
        {
            // fock_4c is not inlcuded here becaues the Fock matrix of AOC-SCF is not well-defined.
            // In 2e-PCC, fock_4c will be recalculated using AOC density matrix and methods in DHF_SPH.
            ene_orb[ir+jj] = ene_orb[ir];
            coeff[ir+jj] = coeff[ir];
            density[ir+jj] = density[ir];
            for(int ii = 0; ii < NOpenShells+2; ii++)
                densityShells[ii][ir+jj] = densityShells[ii][ir];
        }
    }

    countTime(EndTimeCPU,EndTimeWall);
    printTime("DHF iterations");
}


/* 
    evaluate Fock matrix 
*/
void DHF_SPH_CA::evaluateFock(vector<double>& fock_c, const bool& twoC, const vector<vVectorXd>& densities, const int& size, const int& Iirrep)
{
    int ir = all2compact[Iirrep];
    int size2 = twoC ? size : 2*size;
    vVectorXd R(NOpenShells+2);
    vVectorXd Q(NOpenShells+1);
    for(int ii = 0; ii < NOpenShells+2; ii++)
    {
        if(ii < NOpenShells+1) Q[ii].resize(size2*size2, 0.0);
        R[ii].resize(size2*size2);
        for(int mm = 0; mm < size2; mm++)
        for(int nn = 0; nn < size2; nn++)
            R[ii][mm*size2+nn] = densities[ii][Iirrep][nn*size2+mm];   
    }     
    if(twoC)
    {
        #pragma omp parallel  for
        for(int mm = 0; mm < size; mm++)
        for(int nn = 0; nn <= mm; nn++)
        {
            for(int ii = 0; ii < NOpenShells+1; ii++)
            for(int jr = 0; jr < occMax_irrep_compact; jr++)
            {
                int Jirrep = compact2all[jr];
                vector<double> den_tmp = densities[ii][Jirrep];
                double twojP1 = irrep_list[Jirrep].two_j+1;
                int size_tmp2 = irrep_list[Jirrep].size;
                for(int aa = 0; aa < size_tmp2; aa++)
                for(int bb = 0; bb < size_tmp2; bb++)
                {
                    int emn = mm*size+nn, eab = aa*size_tmp2+bb, emb = mm*size_tmp2+bb, ean = aa*size+nn;
                    Q[ii][mm*size+nn] += twojP1*den_tmp[aa*size_tmp2+bb] * h2eLLLL_JK.J[ir][jr][emn][eab];
                }
                Q[ii][nn*size+mm] = Q[ii][mm*size+nn];
            }
        }
    }
    else
    {
        #pragma omp parallel  for
        for(int mm = 0; mm < size; mm++)
        for(int nn = 0; nn <= mm; nn++)
        {         
            for(int ii = 0; ii < NOpenShells+1; ii++)
            for(int jr = 0; jr < occMax_irrep_compact; jr++)
            {
                int Jirrep = compact2all[jr];
                vector<double> den_tmp = densities[ii][Jirrep];
                double twojP1 = irrep_list[Jirrep].two_j+1;
                int size_tmp2 = irrep_list[Jirrep].size;
                int size2j = size_tmp2 * 2;
                for(int ss = 0; ss < size_tmp2; ss++)
                for(int rr = 0; rr < size_tmp2; rr++)
                {
                    int emn = mm*size+nn, esr = ss*size_tmp2+rr, emr = mm*size_tmp2+rr, esn = ss*size+nn;
                    Q[ii][mm*size2+nn] += twojP1*den_tmp[ss*size2j+rr] * h2eLLLL_JK.J[ir][jr][emn][esr] + twojP1*den_tmp[(size_tmp2+ss)*size2j+size_tmp2+rr] * h2eSSLL_JK.J[jr][ir][esr][emn];
                    Q[ii][(mm+size)*size2+nn] -= twojP1*den_tmp[ss*size2j+size_tmp2+rr] * h2eSSLL_JK.K[ir][jr][emr][esn];
                    Q[ii][(mm+size)*size2+nn+size] += twojP1*den_tmp[(size_tmp2+ss)*size2j+size_tmp2+rr] * h2eSSSS_JK.J[ir][jr][emn][esr] + twojP1*den_tmp[ss*size2j+rr] * h2eSSLL_JK.J[ir][jr][emn][esr];
                    if(mm != nn) 
                    {
                        int enr = nn*size_tmp2+rr, esm = ss*size+mm;
                        Q[ii][(nn+size)*size2+mm] -= twojP1*den_tmp[ss*size2j+size_tmp2+rr] * h2eSSLL_JK.K[ir][jr][enr][esm];
                    }
                    if(with_gaunt)
                    {
                        int enm = nn*size+mm, ers = rr*size_tmp2+ss, erm = rr*size+mm, ens = nn*size_tmp2+ss;
                        Q[ii][mm*size2+nn] -= twojP1*den_tmp[(size_tmp2+ss)*size2j+size_tmp2+rr] * gauntLSSL_JK.K[ir][jr][emr][esn];
                        Q[ii][(mm+size)*size2+nn] += twojP1*den_tmp[(ss+size_tmp2)*size2j+rr]*gauntLSLS_JK.J[ir][jr][enm][ers] + twojP1*den_tmp[ss*size2j+size_tmp2+rr] * gauntLSSL_JK.J[jr][ir][esr][emn];
                        Q[ii][(mm+size)*size2+nn+size] -= twojP1*den_tmp[ss*size2j+rr] * gauntLSSL_JK.K[jr][ir][esn][emr];
                        if(mm != nn)
                        {
                            int ern = rr*size+nn, ems = mm*size_tmp2+ss;
                            Q[ii][(nn+size)*size2+mm] += twojP1*den_tmp[(size_tmp2+ss)*size2j+rr]*gauntLSLS_JK.J[ir][jr][emn][ers] + twojP1*den_tmp[ss*size2j+size_tmp2+rr] * gauntLSSL_JK.J[jr][ir][esr][enm];
                        }
                    }
                }    
                Q[ii][nn*size2+mm] = Q[ii][mm*size2+nn];
                Q[ii][mm*size2+nn+size] = Q[ii][(nn+size)*size2+mm];
                Q[ii][nn*size2+mm+size] = Q[ii][(mm+size)*size2+nn];
                Q[ii][(size+nn)*size2+size+mm] = Q[ii][(mm+size)*size2+nn+size];
            }
        }
    }

    fock_c = h1e_4c[Iirrep];
    for(int ii = 0; ii < NOpenShells+1; ii++)
    {
        if(ii != 0)
            Q[ii] = f_list[ii-1]*Q[ii];
        fock_c = fock_c + Q[ii];
    }
    vector<double> LM(size2*size2, 0.0);
    for(int ii = 1; ii < NOpenShells+1; ii++)
    {
        double f_u = f_list[ii-1];
        double a_u = MM_list[ii-1]*(NN_list[ii-1]-1.0)/NN_list[ii-1]/(MM_list[ii-1]-1.0);
        double alpha_u = (1-a_u)/(1-f_u);
        vector<double> tmp1, tmp2, tmp3;
        // LM += S*R(ii)*Q(ii)*(alpha_u*f_u*R(0)+(a_u-1.0)*(0.5*R(ii)+R(NOpenShells+1)))*S;
        dgemm_itrf('n','n',size2,size2,size2,1.0,overlap_4c[Iirrep],R[ii],0.0,tmp1);
        dgemm_itrf('n','n',size2,size2,size2,1.0,tmp1,Q[ii],0.0,tmp2);
        tmp3 = alpha_u*f_u*R[0]+(a_u-1.0)*(0.5*R[ii]+R[NOpenShells+1]);
        dgemm_itrf('n','n',size2,size2,size2,1.0,tmp2,tmp3,0.0,tmp1);
        dgemm_itrf('n','n',size2,size2,size2,1.0,tmp1,overlap_4c[Iirrep],1.0,LM);
    }
    fock_c = fock_c + LM + vectorTrans(LM, size2);
}
void DHF_SPH_CA::evaluateFock_2e(vector<double>& fock_c, const bool& twoC, const vector<vVectorXd>& densities, const int& size, const int& Iirrep)
{
    int ir = all2compact[Iirrep];
    int size2 = twoC ? size : 2*size;
    vVectorXd R(NOpenShells+2);
    vVectorXd Q(NOpenShells+1);
    for(int ii = 0; ii < NOpenShells+2; ii++)
    {
        if(ii < NOpenShells+1) Q[ii].resize(size2*size2, 0.0);
        R[ii].resize(size2*size2);
        for(int mm = 0; mm < size2; mm++)
        for(int nn = 0; nn < size2; nn++)
            R[ii][mm*size2+nn] = densities[ii][Iirrep][nn*size2+mm];   
    }     
    if(twoC)
    {
        #pragma omp parallel  for
        for(int mm = 0; mm < size; mm++)
        for(int nn = 0; nn <= mm; nn++)
        {
            for(int ii = 0; ii < NOpenShells+1; ii++)
            for(int jr = 0; jr < occMax_irrep_compact; jr++)
            {
                int Jirrep = compact2all[jr];
                vector<double> den_tmp = densities[ii][Jirrep];
                double twojP1 = irrep_list[Jirrep].two_j+1;
                int size_tmp2 = irrep_list[Jirrep].size;
                for(int aa = 0; aa < size_tmp2; aa++)
                for(int bb = 0; bb < size_tmp2; bb++)
                {
                    int emn = mm*size+nn, eab = aa*size_tmp2+bb, emb = mm*size_tmp2+bb, ean = aa*size+nn;
                    Q[ii][mm*size+nn] += twojP1*den_tmp[aa*size_tmp2+bb] * h2eLLLL_JK.J[ir][jr][emn][eab];
                }
                Q[ii][nn*size+mm] = Q[ii][mm*size+nn];
            }
        }
    }
    else
    {
        #pragma omp parallel  for
        for(int mm = 0; mm < size; mm++)
        for(int nn = 0; nn <= mm; nn++)
        {         
            for(int ii = 0; ii < NOpenShells+1; ii++)
            for(int jr = 0; jr < occMax_irrep_compact; jr++)
            {
                int Jirrep = compact2all[jr];
                vector<double> den_tmp = densities[ii][Jirrep];
                double twojP1 = irrep_list[Jirrep].two_j+1;
                int size_tmp2 = irrep_list[Jirrep].size;
                int size2j = size_tmp2 * 2;
                for(int ss = 0; ss < size_tmp2; ss++)
                for(int rr = 0; rr < size_tmp2; rr++)
                {
                    int emn = mm*size+nn, esr = ss*size_tmp2+rr, emr = mm*size_tmp2+rr, esn = ss*size+nn;
                    Q[ii][mm*size2+nn] += twojP1*den_tmp[ss*size2j+rr] * h2eLLLL_JK.J[ir][jr][emn][esr] + twojP1*den_tmp[(size_tmp2+ss)*size2j+size_tmp2+rr] * h2eSSLL_JK.J[jr][ir][esr][emn];
                    Q[ii][(mm+size)*size2+nn] -= twojP1*den_tmp[ss*size2j+size_tmp2+rr] * h2eSSLL_JK.K[ir][jr][emr][esn];
                    Q[ii][(mm+size)*size2+nn+size] += twojP1*den_tmp[(size_tmp2+ss)*size2j+size_tmp2+rr] * h2eSSSS_JK.J[ir][jr][emn][esr] + twojP1*den_tmp[ss*size2j+rr] * h2eSSLL_JK.J[ir][jr][emn][esr];
                    if(mm != nn) 
                    {
                        int enr = nn*size_tmp2+rr, esm = ss*size+mm;
                        Q[ii][(nn+size)*size2+mm] -= twojP1*den_tmp[ss*size2j+size_tmp2+rr] * h2eSSLL_JK.K[ir][jr][enr][esm];
                    }
                    if(with_gaunt)
                    {
                        int enm = nn*size+mm, ers = rr*size_tmp2+ss, erm = rr*size+mm, ens = nn*size_tmp2+ss;
                        Q[ii][mm*size2+nn] -= twojP1*den_tmp[(size_tmp2+ss)*size2j+size_tmp2+rr] * gauntLSSL_JK.K[ir][jr][emr][esn];
                        Q[ii][(mm+size)*size2+nn] += twojP1*den_tmp[(ss+size_tmp2)*size2j+rr]*gauntLSLS_JK.J[ir][jr][enm][ers] + twojP1*den_tmp[ss*size2j+size_tmp2+rr] * gauntLSSL_JK.J[jr][ir][esr][emn];
                        Q[ii][(mm+size)*size2+nn+size] -= twojP1*den_tmp[ss*size2j+rr] * gauntLSSL_JK.K[jr][ir][esn][emr];
                        if(mm != nn)
                        {
                            int ern = rr*size+nn, ems = mm*size_tmp2+ss;
                            Q[ii][(nn+size)*size2+mm] += twojP1*den_tmp[(size_tmp2+ss)*size2j+rr]*gauntLSLS_JK.J[ir][jr][emn][ers] + twojP1*den_tmp[ss*size2j+size_tmp2+rr] * gauntLSSL_JK.J[jr][ir][esr][enm];
                        }
                    }
                }    
                Q[ii][nn*size2+mm] = Q[ii][mm*size2+nn];
                Q[ii][mm*size2+nn+size] = Q[ii][(nn+size)*size2+mm];
                Q[ii][nn*size2+mm+size] = Q[ii][(mm+size)*size2+nn];
                Q[ii][(size+nn)*size2+size+mm] = Q[ii][(mm+size)*size2+nn+size];
            }
        }
    }

    fock_c.resize(size2*size2, 0.0);
    for(int ii = 0; ii < NOpenShells+1; ii++)
    {
        if(ii != 0)
            Q[ii] = f_list[ii-1]*Q[ii];
        fock_c = fock_c + Q[ii];
    }
    vector<double> LM(size2*size2, 0.0);
    for(int ii = 1; ii < NOpenShells+1; ii++)
    {
        double f_u = f_list[ii-1];
        double a_u = MM_list[ii-1]*(NN_list[ii-1]-1.0)/NN_list[ii-1]/(MM_list[ii-1]-1.0);
        double alpha_u = (1-a_u)/(1-f_u);
        vector<double> tmp1, tmp2, tmp3;
        // LM += S*R(ii)*Q(ii)*(alpha_u*f_u*R(0)+(a_u-1.0)*(0.5*R(ii)+R(NOpenShells+1)))*S;
        dgemm_itrf('n','n',size2,size2,size2,1.0,overlap_4c[Iirrep],R[ii],0.0,tmp1);
        dgemm_itrf('n','n',size2,size2,size2,1.0,tmp1,Q[ii],0.0,tmp2);
        tmp3 = alpha_u*f_u*R[0]+(a_u-1.0)*(0.5*R[ii]+R[NOpenShells+1]);
        dgemm_itrf('n','n',size2,size2,size2,1.0,tmp2,tmp3,0.0,tmp1);
        dgemm_itrf('n','n',size2,size2,size2,1.0,tmp1,overlap_4c[Iirrep],1.0,LM);
    }
    fock_c = fock_c + LM + vectorTrans(LM, size2);
}

double DHF_SPH_CA::evaluateEnergy(const bool& twoC)
{
    double ene = 0.0;
    for(int ir = 0; ir < occMax_irrep_compact; ir++)
    {
        int Iirrep = compact2all[ir];
        int size = irrep_list[Iirrep].size;
        int size2 = twoC ? size : size*2;
        vVectorXd Q(NOpenShells+1);
        for(int ii = 0; ii < NOpenShells+1; ii++)
        {
            Q[ii].resize(size2*size2, 0.0);
        }
        if(twoC)
        {
            #pragma omp parallel  for
            for(int mm = 0; mm < size; mm++)
            for(int nn = 0; nn <= mm; nn++)
            {
                for(int ii = 0; ii < NOpenShells+1; ii++)
                for(int jr = 0; jr < occMax_irrep_compact; jr++)
                {
                    int Jirrep = compact2all[jr];
                    vector<double> den_tmp = densityShells[ii][Jirrep];
                    double twojP1 = irrep_list[Jirrep].two_j+1;
                    int size_tmp2 = irrep_list[Jirrep].size;
                    for(int aa = 0; aa < size_tmp2; aa++)
                    for(int bb = 0; bb < size_tmp2; bb++)
                    {
                        int emn = mm*size+nn, eab = aa*size_tmp2+bb, emb = mm*size_tmp2+bb, ean = aa*size+nn;
                        Q[ii][mm*size2+nn] += twojP1*den_tmp[aa*size_tmp2+bb] * h2eLLLL_JK.J[ir][jr][emn][eab];
                    }
                    Q[ii][nn*size2+mm] = Q[ii][mm*size2+nn];
                }
            }
        }
        else
        {
            #pragma omp parallel  for
            for(int mm = 0; mm < size; mm++)
            for(int nn = 0; nn <= mm; nn++)
            {
                for(int jr = 0; jr < occMax_irrep_compact; jr++)
                for(int ii = 0; ii < NOpenShells+1; ii++)
                {
                    int Jirrep = compact2all[jr];
                    vector<double> den_tmp = densityShells[ii][Jirrep];
                    double twojP1 = irrep_list[Jirrep].two_j+1;
                    int size_tmp2 = irrep_list[Jirrep].size;
                    int size2j = size_tmp2 * 2;
                    for(int ss = 0; ss < size_tmp2; ss++)
                    for(int rr = 0; rr < size_tmp2; rr++)
                    {
                        int emn = mm*size+nn, esr = ss*size_tmp2+rr, emr = mm*size_tmp2+rr, esn = ss*size+nn;
                        Q[ii][mm*size2+nn] += twojP1*den_tmp[ss*size2j+rr] * h2eLLLL_JK.J[ir][jr][emn][esr] + twojP1*den_tmp[(size_tmp2+ss)*size2j+size_tmp2+rr] * h2eSSLL_JK.J[jr][ir][esr][emn];
                        Q[ii][(mm+size)*size2+nn] -= twojP1*den_tmp[ss*size2j+size_tmp2+rr] * h2eSSLL_JK.K[ir][jr][emr][esn];
                        Q[ii][(mm+size)*size2+nn+size] += twojP1*den_tmp[(size_tmp2+ss)*size2j+size_tmp2+rr] * h2eSSSS_JK.J[ir][jr][emn][esr] + twojP1*den_tmp[ss*size2j+rr] * h2eSSLL_JK.J[ir][jr][emn][esr];
                        if(mm != nn) 
                        {
                            int enr = nn*size_tmp2+rr, esm = ss*size+mm;
                            Q[ii][(nn+size)*size2+mm] -= twojP1*den_tmp[ss*size2j+size_tmp2+rr] * h2eSSLL_JK.K[ir][jr][enr][esm];
                        }
                        if(with_gaunt)
                        {
                            int enm = nn*size+mm, ers = rr*size_tmp2+ss, erm = rr*size+mm, ens = nn*size_tmp2+ss;
                            Q[ii][mm*size2+nn] -= twojP1*den_tmp[(size_tmp2+ss)*size2j+size_tmp2+rr] * gauntLSSL_JK.K[ir][jr][emr][esn];
                            Q[ii][(mm+size)*size2+nn] += twojP1*den_tmp[(ss+size_tmp2)*size2j+rr]*gauntLSLS_JK.J[ir][jr][enm][ers] + twojP1*den_tmp[ss*size2j+size_tmp2+rr] * gauntLSSL_JK.J[jr][ir][esr][emn];
                            Q[ii][(mm+size)*size2+nn+size] -= twojP1*den_tmp[ss*size2j+rr] * gauntLSSL_JK.K[jr][ir][esn][emr];
                            if(mm != nn)
                            {
                                int ern = rr*size+nn, ems = mm*size_tmp2+ss;
                                Q[ii][(nn+size)*size2+mm] += twojP1*den_tmp[(size_tmp2+ss)*size2j+rr]*gauntLSLS_JK.J[ir][jr][emn][ers] + twojP1*den_tmp[ss*size2j+size_tmp2+rr] * gauntLSSL_JK.J[jr][ir][esr][enm];
                            }
                        }             
                    }
                    Q[ii][nn*size2+mm] = Q[ii][mm*size2+nn];
                    Q[ii][mm*size2+nn+size] = Q[ii][(nn+size)*size2+mm];
                    Q[ii][nn*size2+mm+size] = Q[ii][(mm+size)*size2+nn];
                    Q[ii][(size+nn)*size2+size+mm] = Q[ii][(mm+size)*size2+nn+size];
                }
            }
        }

        for(int ii = 0; ii < NOpenShells+1; ii++)
        {
            double f_i;
            if(ii == 0) f_i = 1.0;
            else f_i = f_list[ii-1];
            vector<double> fock_e = h1e_4c[Iirrep] + 0.5*Q[0];
            for(int jj = 1; jj < NOpenShells+1; jj++)
            {
                double f_j;
                if(jj == ii) f_j = (NN_list[jj-1]-1.0)/(MM_list[jj-1]-1.0);
                else f_j = f_list[jj-1];
                fock_e = fock_e + 0.5*f_j*Q[jj];
            }
            double tmp = 0.0;
            for(int mm = 0; mm < size2; mm++)
            for(int nn = 0; nn < size2; nn++)
            {
                tmp += fock_e[mm*size2+nn]*densityShells[ii][Iirrep][mm*size2+nn];
            }
            ene += f_i * tmp * (irrep_list[Iirrep].two_j+1);
        }
    }

    return ene;
}

