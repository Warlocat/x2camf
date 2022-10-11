#include"dhf_sph_ca2.h"
#include<iostream>
#include<omp.h>
#include<vector>
#include<ctime>
#include<iostream>
#include<iomanip>
#include<fstream>

using namespace std;
using namespace Eigen;

DHF_SPH_CA2::DHF_SPH_CA2(INT_SPH& int_sph_, const string& filename, const bool& spinFree, const bool& twoC, const bool& with_gaunt_, const bool& with_gauge_, const bool& allInt, const bool& gaussian_nuc):
DHF_SPH(int_sph_,filename,spinFree,twoC,with_gaunt_,with_gauge_,allInt,gaussian_nuc)
{
    vector<int> openIrreps;
    for(int ir = 0; ir < occMax_irrep; ir+=4*irrep_list(ir).l+2)
    // for(int ir = 0; ir < occMax_irrep; ir+=irrep_list(ir).two_j+1)
    {
        for(int ii = 0; ii < occNumber(ir).rows(); ii++)
        {
            // if 1 > occNumber(ir)(ii) > 0
            if(abs(occNumber(ir)(ii)) > 1e-4 && occNumber(ir)(ii) < 0.9999)
            {
                openIrreps.push_back(ir);
                break;
            }
        }
    }
    NOpenShells = openIrreps.size();
    occNumberShells.resize(NOpenShells+2);
    fockShells.resize(NOpenShells+1);
    coeffShells.resize(NOpenShells+1);
    densityShells.resize(NOpenShells+1);
    for(int ii = 0; ii < NOpenShells; ii++)
    {
        MM_list.push_back(irrep_list(openIrreps[ii]).l*4+2);  
        // MM_list.push_back(irrep_list(openIrreps[ii]).two_j+1);   
    }
    occNumberShells[0] = occNumber;
    for(int ii = 1; ii < occNumberShells.size()-1; ii++)
        occNumberShells[ii].resize(occMax_irrep);
    occNumberShells[NOpenShells+1].resize(irrep_list.rows());

    for(int ir = 0; ir < occMax_irrep; ir++)
    {
        for(int ii = 1; ii < occNumberShells.size()-1; ii++)
        {
            occNumberShells[ii](ir) = VectorXd::Zero(irrep_list(ir).size);
        }
        occNumberShells[NOpenShells+1](ir) = VectorXd::Ones(irrep_list(ir).size);
            
        for(int ii = 0; ii < occNumber(ir).rows(); ii++)
        {
            if(abs(occNumber(ir)(ii)) > 1e-4)
            {
                // if 1 > occNumber(ir)(ii) > 0
                if(occNumber(ir)(ii) < 0.9999)
                {
                    if(ir == openIrreps[f_list.size()])
                    {
                        f_list.push_back(occNumber(ir)(ii));
                        NN_list.push_back(f_list[f_list.size()-1]*MM_list[f_list.size()-1]);
                        a_list.push_back((NN_list[f_list.size()-1] - 1) / (MM_list[f_list.size()-1] - 1));
                    }
                    // f_NM = occNumber(ir)(ii);
                    occNumberShells[0](ir)(ii) = 0.0;
                    occNumberShells[f_list.size()](ir)(ii) = 1.0;
                }
                occNumberShells[NOpenShells+1](ir)(ii) = 0.0;
            }
        }
    }
    for(int ir = occMax_irrep; ir < irrep_list.rows(); ir++)
    {
        occNumberShells[NOpenShells+1](ir) = VectorXd::Ones(irrep_list(ir).size);
    }

    cout << "Open shell occupations:" << endl;
    for(int ir = 0; ir < occMax_irrep; ir+=irrep_list(ir).l*4+2)
    {
        cout << "l = " << irrep_list(ir).l << endl;
        for(int ii = 0; ii < occNumberShells.size(); ii++)
            cout << ii << ": " << occNumberShells[ii](ir).transpose() << endl;
    }
    for(int ir = occMax_irrep; ir < irrep_list.rows(); ir++)
    {
        cout << occNumberShells.size()-1 << ": " << occNumberShells[occNumberShells.size()-1](ir).transpose() << endl;
    }
    cout << "Configuration-averaged HF initialization." << endl;
    cout << "Number of open shells: " << NOpenShells << endl;
    cout << "No.\tM\tN\tf=N/M\ta=(N-1)/(M-1)" << endl;
    for(int ii = 0; ii < NOpenShells; ii++)
    {
        cout << ii+1 << "\t" << MM_list[ii] << "\t" << NN_list[ii] << "\t" << f_list[ii] << "\t" << a_list[ii] << endl;
    }
}


DHF_SPH_CA2::~DHF_SPH_CA2()
{
}



/*
    Evaluate density matrix
*/
MatrixXd DHF_SPH_CA2::evaluateDensity_aoc(const MatrixXd& coeff_, const VectorXd& occNumber_, const bool& twoC)
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


/*
    SCF procedure for 4-c and 2-c calculation
*/
void DHF_SPH_CA2::runSCF(const bool& twoC, const bool& renormSmall)
{
    if(renormSmall)
    {
        renormalize_small();
    }
    Matrix<vector<MatrixXd>,-1,-1> error4DIIS(occMax_irrep,NOpenShells+1), fock4DIIS(occMax_irrep,NOpenShells+1);
    Matrix<vMatrixXd,-1,1> newDensityShells(NOpenShells+1);
    for(int ii = 0; ii < NOpenShells+1; ii++)
    {
        newDensityShells(ii).resize(occMax_irrep);
        fockShells(ii).resize(occMax_irrep);
        coeffShells(ii).resize(occMax_irrep);
        densityShells(ii).resize(occMax_irrep);
    }
    countTime(StartTimeCPU,StartTimeWall);
    cout << endl;
    if(twoC) cout << "Start CA-X2C-1e Hartree-Fock iterations..." << endl;
    else cout << "Start CA-Dirac Hartree-Fock iterations..." << endl;
    cout << endl;

    
    eigensolverG_irrep(h1e_4c, overlap_half_i_4c, ene_orb, coeffShells(0));
    for(int ii = 1; ii < NOpenShells+1; ii++)
    {
        coeffShells(ii) = coeffShells(0);
    }
    for(int ir = 0; ir < occMax_irrep; ir+=irrep_list(ir).two_j+1)
    {
        for(int ii = 0; ii < NOpenShells+1; ii++)
            densityShells(ii)(ir) = evaluateDensity_aoc(coeffShells(ii)(ir),occNumberShells[ii](ir),twoC);
    }

    for(int iter = 1; iter <= maxIter; iter++)
    {
        if(iter <= 2)
        {
            for(int ir = 0; ir < occMax_irrep; ir += irrep_list(ir).two_j+1)    
            {
                int size_tmp = irrep_list(ir).size;
                evaluateFockShells(fockShells,twoC,densityShells,size_tmp,ir);
            }
        }
        else
        {
            int tmp_size = fock4DIIS(0,0).size();
            MatrixXd B4DIIS(tmp_size+1,tmp_size+1);
            VectorXd vec_b(tmp_size+1);    
            for(int ii = 0; ii < tmp_size; ii++)
            {    
                for(int jj = 0; jj <= ii; jj++)
                {
                    B4DIIS(ii,jj) = 0.0;
                    for(int ir = 0; ir < occMax_irrep; ir += irrep_list(ir).two_j+1)
                    for(int kk = 0; kk < NOpenShells+1; kk++)
                        B4DIIS(ii,jj) += (error4DIIS(ir,kk)[ii].adjoint()*error4DIIS(ir,kk)[jj])(0,0);
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
                for(int kk = 0; kk < NOpenShells+1; kk++)
                    fockShells(kk)(ir) = MatrixXd::Zero(fockShells(kk)(ir).rows(),fockShells(kk)(ir).cols());
                for(int ii = 0; ii < tmp_size; ii++)
                for(int kk = 0; kk < NOpenShells+1; kk++)
                {
                    fockShells(kk)(ir) += C(ii) * fock4DIIS(ir,kk)[ii];
                }
            }
        }
        for(int ii = 0; ii < NOpenShells+1; ii++)
            eigensolverG_irrep(fockShells(ii), overlap_half_i_4c, ene_orb, coeffShells(ii));

        for(int ir = 0; ir < occMax_irrep; ir+=irrep_list(ir).two_j+1)
        {
            for(int ii = 0; ii < NOpenShells+1; ii++)
                newDensityShells(ii)(ir) = evaluateDensity_aoc(coeffShells(ii)(ir),occNumberShells[ii](ir),twoC);
        }
        d_density = 0.0;
        for(int ii = 0; ii < NOpenShells+1; ii++)
            d_density += evaluateChange_irrep(densityShells[ii],newDensityShells[ii]);          
        cout << "Iter #" << iter << " maximum density difference: " << d_density << endl;     
        for(int ii = 0; ii < NOpenShells+1; ii++)
            densityShells[ii] = newDensityShells[ii];

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
                    coeff(ir) = coeffShells(0)(ir);
                    for(int ii = 0; ii < irrep_list(ir).size; ii++)
                    for(int kk = 1; kk < NOpenShells+1; kk++)
                    {
                        if(abs(occNumberShells[kk](ir)(ii) - 1.0) < 1e-5)
                        {
                            for(int jj = 0; jj < coeffShells(kk)(ir).rows(); jj++)
                                coeff(ir)(jj,ii) = coeffShells(kk)(ir)(jj,ii);
                        }
                    }
                }
            }
            else
            {
                for(int ir = 0; ir < occMax_irrep; ir += irrep_list(ir).two_j+1)
                {
                    coeff(ir) = coeffShells(0)(ir);
                    for(int ii = 0; ii < irrep_list(ir).size; ii++)
                    for(int kk = 1; kk < NOpenShells+1; kk++)
                    {
                        if(abs(occNumberShells[kk](ir)(ii) - 1.0) < 1e-5)
                        {
                            for(int jj = 0; jj < coeffShells(kk)(ir).rows(); jj++)
                                coeff(ir)(jj,ii+coeff(ir).rows()/2) = coeffShells(kk)(ir)(jj,ii+coeff(ir).rows()/2);
                        }
                    }                        
                }
            }

            ene_scf = evaluateEnergy(twoC);
            
            if(twoC) cout << "Final CA-X2C-1e HF energy is " << setprecision(15) << ene_scf << " hartree." << endl;
            else cout << "Final CA-DHF energy is " << setprecision(15) << ene_scf << " hartree." << endl;
            break;            
        }
        for(int ir = 0; ir < occMax_irrep; ir += irrep_list(ir).two_j+1)    
        {
            int size_tmp = irrep_list(ir).size;
            evaluateFockShells(fockShells,twoC,densityShells,size_tmp,ir);
            for(int ii = 0; ii < NOpenShells+1; ii++)
            {
                eigensolverG(fockShells(ii)(ir), overlap_half_i_4c(ir), ene_orb(ir), coeffShells(ii)(ir));
                newDensityShells(ii)(ir) = evaluateDensity_aoc(coeffShells(ii)(ir),occNumberShells[ii](ir),twoC);
                error4DIIS(ir,ii).push_back(evaluateErrorDIIS(densityShells(ii)(ir),newDensityShells(ii)(ir)));
                fock4DIIS(ir,ii).push_back(fockShells(ii)(ir));
            }
    
            if(error4DIIS(ir,0).size() > size_DIIS)
            {
                for(int ii = 0; ii < NOpenShells+1; ii++)
                {
                    error4DIIS(ir,ii).erase(error4DIIS(ir,ii).begin());
                    fock4DIIS(ir,ii).erase(fock4DIIS(ir,ii).begin());
                }
            }            
        }
    }

    density.resize(occMax_irrep);
    for(int ir = 0; ir < occMax_irrep; ir += irrep_list(ir).two_j+1)
    {
        density(ir) = densityShells(0)(ir);
        for(int ii = 1; ii < NOpenShells+1; ii++)
            density(ir) += f_list[ii-1]*densityShells(ii)(ir);
        for(int jj = 1; jj < irrep_list(ir).two_j+1; jj++)
        {
            ene_orb(ir+jj) = ene_orb(ir);
            coeff(ir+jj) = coeff(ir);
            density(ir+jj) = density(ir);
            for(int ii = 0; ii < NOpenShells+1; ii++)
                densityShells(ii)(ir+jj) = densityShells(ii)(ir);
        }
    }    

    countTime(EndTimeCPU,EndTimeWall);
    printTime("DHF iterations");
}


/* 
    evaluate Fock matrix 
*/
void DHF_SPH_CA2::evaluateFockShells(Matrix<vMatrixXd,-1,1>& fockShells, const bool& twoC, const Matrix<vMatrixXd,-1,1>& densities, const int& size, const int& Iirrep)
{
    int ir = all2compact(Iirrep);
    MatrixXd S = overlap_4c(Iirrep);
    int fockSize = S.rows();
    vMatrixXd R(NOpenShells+1), Q(NOpenShells+1);
    for(int ii = 0; ii < NOpenShells+1; ii++)
    {
        R(ii) = (densities(ii)(Iirrep)).transpose();
        Q(ii) = MatrixXd::Zero(fockSize,fockSize);
    }
    if(twoC)
    {
        #pragma omp parallel  for
        for(int mm = 0; mm < size; mm++)
        for(int nn = 0; nn <= mm; nn++)
        {
            for(int jr = 0; jr < occMax_irrep_compact; jr++)
            {
                int Jirrep = compact2all(jr);
                double twojP1 = irrep_list(Jirrep).two_j+1;
                ;
                int size_tmp2 = irrep_list(Jirrep).size;
                for(int aa = 0; aa < size_tmp2; aa++)
                for(int bb = 0; bb < size_tmp2; bb++)
                {
                    int emn = mm*size+nn, eab = aa*size_tmp2+bb, emb = mm*size_tmp2+bb, ean = aa*size+nn;
                    for(int ii = 0; ii < NOpenShells+1; ii++)
                    {
                        Q(ii)(mm,nn) += twojP1*densities(ii)(Jirrep)(aa,bb) * h2eLLLL_JK.J[ir][jr][emn][eab];
                    }
                }
            }
            for(int ii = 0; ii < NOpenShells+1; ii++)
            {
                Q(ii)(nn,mm) = Q(ii)(mm,nn);
            }
        }
    }
    else
    {
        #pragma omp parallel  for
        for(int mm = 0; mm < size; mm++)
        for(int nn = 0; nn <= mm; nn++)
        {
            MatrixXd den_tmp;            
            for(int jr = 0; jr < occMax_irrep_compact; jr++)
            {
                int Jirrep = compact2all(jr);
                double twojP1 = irrep_list(Jirrep).two_j+1;
                int size_tmp2 = irrep_list(Jirrep).size;
                for(int ss = 0; ss < size_tmp2; ss++)
                for(int rr = 0; rr < size_tmp2; rr++)
                {
                    int emn = mm*size+nn, esr = ss*size_tmp2+rr, emr = mm*size_tmp2+rr, esn = ss*size+nn;
                    for(int ii = 0; ii < NOpenShells+1; ii++)
                    {
                        den_tmp = densities(ii)(Jirrep);
                        Q(ii)(mm,nn) += twojP1*den_tmp(ss,rr) * h2eLLLL_JK.J[ir][jr][emn][esr] + twojP1*den_tmp(size_tmp2+ss,size_tmp2+rr) * h2eSSLL_JK.J[jr][ir][esr][emn];
                        Q(ii)(mm+size,nn) -= twojP1*den_tmp(ss,size_tmp2+rr) * h2eSSLL_JK.K[ir][jr][emr][esn];
                        Q(ii)(mm+size,nn+size) += twojP1*den_tmp(size_tmp2+ss,size_tmp2+rr) * h2eSSSS_JK.J[ir][jr][emn][esr] + twojP1*den_tmp(ss,rr) * h2eSSLL_JK.J[ir][jr][emn][esr];
                        if(mm != nn) 
                        {
                            int enr = nn*size_tmp2+rr, esm = ss*size+mm;
                            Q(ii)(nn+size,mm) -= twojP1*den_tmp(ss,size_tmp2+rr) * h2eSSLL_JK.K[ir][jr][enr][esm];
                        }
                        if(with_gaunt)
                        {
                            int enm = nn*size+mm, ers = rr*size_tmp2+ss, erm = rr*size+mm, ens = nn*size_tmp2+ss;

                            Q(ii)(mm,nn) -= twojP1*den_tmp(size_tmp2+ss,size_tmp2+rr) * gauntLSSL_JK.K[ir][jr][emr][esn];
                            Q(ii)(mm+size,nn) += twojP1*den_tmp(ss+size_tmp2,rr)*gauntLSLS_JK.J[ir][jr][enm][ers] + twojP1*den_tmp(ss,size_tmp2+rr) * gauntLSSL_JK.J[jr][ir][esr][emn];
                            Q(ii)(mm+size,nn+size) -= twojP1*den_tmp(ss,rr) * gauntLSSL_JK.K[jr][ir][esn][emr];
                            if(mm != nn)
                            {
                                int ern = rr*size+nn, ems = mm*size_tmp2+ss;
                                Q(ii)(nn+size,mm) += twojP1*den_tmp(size_tmp2+ss,rr)*gauntLSLS_JK.J[ir][jr][emn][ers] + twojP1*den_tmp(ss,size_tmp2+rr) * gauntLSSL_JK.J[jr][ir][esr][enm];
                            }
                        }
                    }               
                }
            }
            for(int ii = 0; ii < NOpenShells+1; ii++)
            {
                Q(ii)(nn,mm) = Q(ii)(mm,nn);
                Q(ii)(mm,nn+size) = Q(ii)(nn+size,mm);
                Q(ii)(nn,mm+size) = Q(ii)(mm+size,nn);
                Q(ii)(size+nn,size+mm) = Q(ii)(size+mm,size+nn);
            }
        }
    }

    vMatrixXd fock_ir_shells(NOpenShells+1);
    for(int ii = 0; ii < NOpenShells+1; ii++)
    {
        fock_ir_shells(ii) = h1e_4c(Iirrep) + Q(0);
        for(int jj = 1; jj < NOpenShells+1; jj++)
        {
            if(ii == jj)
            {
                fock_ir_shells(ii) += a_list[jj-1] * Q(jj);
            }
            else
            {
                fock_ir_shells(ii) += f_list[jj-1] * Q(jj);
            }
        }
        if(ii == 0)
        {
            for(int jj = 1; jj < NOpenShells+1; jj++)
                fock_ir_shells(ii) += f_list[jj-1]*(f_list[jj-1]-a_list[jj-1])/(1.0-f_list[jj-1])*(S*R(jj)*Q(jj)+Q(jj)*R(jj)*S);
        }
        else
        {
            fock_ir_shells(ii) += (f_list[ii-1]-a_list[ii-1])/(1.0-f_list[ii-1])*(S*R(0)*Q(ii)+Q(ii)*R(0)*S);
        }
        fockShells(ii)(Iirrep) = fock_ir_shells(ii);
    }
}
double DHF_SPH_CA2::evaluateEnergy(const bool& twoC)
{
    double ene = 0.0;
    for(int ir = 0; ir < occMax_irrep_compact; ir++)
    {
        int Iirrep = compact2all(ir);
        int size = irrep_list(Iirrep).size;
        vMatrixXd Q(NOpenShells+1);
        for(int ii = 0; ii < NOpenShells+1; ii++)
        {
            if(twoC)  Q(ii) = MatrixXd::Zero(size,size);
            else      Q(ii) = MatrixXd::Zero(2*size,2*size);
        }     
        if(twoC)
        {
            #pragma omp parallel  for
            for(int mm = 0; mm < size; mm++)
            for(int nn = 0; nn <= mm; nn++)
            {
                for(int jr = 0; jr < occMax_irrep_compact; jr++)
                {
                    int Jirrep = compact2all(jr);
                    double twojP1 = irrep_list(Jirrep).two_j+1;
                    ;
                    int size_tmp2 = irrep_list(Jirrep).size;
                    for(int aa = 0; aa < size_tmp2; aa++)
                    for(int bb = 0; bb < size_tmp2; bb++)
                    {
                        int emn = mm*size+nn, eab = aa*size_tmp2+bb, emb = mm*size_tmp2+bb, ean = aa*size+nn;
                        for(int ii = 0; ii < NOpenShells+1; ii++)
                        {
                            Q(ii)(mm,nn) += twojP1*densityShells(ii)(Jirrep)(aa,bb) * h2eLLLL_JK.J[ir][jr][emn][eab];
                        }
                    }
                }
                for(int ii = 0; ii < NOpenShells+1; ii++)
                {
                    Q(ii)(nn,mm) = Q(ii)(mm,nn);
                }
            }
        }
        else
        {
            #pragma omp parallel  for
            for(int mm = 0; mm < size; mm++)
            for(int nn = 0; nn <= mm; nn++)
            {
                MatrixXd den_tmp;            
                for(int jr = 0; jr < occMax_irrep_compact; jr++)
                {
                    int Jirrep = compact2all(jr);
                    double twojP1 = irrep_list(Jirrep).two_j+1;
                    int size_tmp2 = irrep_list(Jirrep).size;
                    for(int ss = 0; ss < size_tmp2; ss++)
                    for(int rr = 0; rr < size_tmp2; rr++)
                    {
                        int emn = mm*size+nn, esr = ss*size_tmp2+rr, emr = mm*size_tmp2+rr, esn = ss*size+nn;
                        for(int ii = 0; ii < NOpenShells+1; ii++)
                        {
                            den_tmp = densityShells(ii)(Jirrep);
                            Q(ii)(mm,nn) += twojP1*den_tmp(ss,rr) * h2eLLLL_JK.J[ir][jr][emn][esr] + twojP1*den_tmp(size_tmp2+ss,size_tmp2  +rr) * h2eSSLL_JK.J[jr][ir][esr][emn];
                            Q(ii)(mm+size,nn) -= twojP1*den_tmp(ss,size_tmp2+rr) * h2eSSLL_JK.K[ir][jr][emr][esn];
                            Q(ii)(mm+size,nn+size) += twojP1*den_tmp(size_tmp2+ss,size_tmp2+rr) * h2eSSSS_JK.J[ir][jr][emn][esr] +  twojP1*den_tmp(ss,rr) * h2eSSLL_JK.J[ir][jr][emn][esr];
                            if(mm != nn) 
                            {
                                int enr = nn*size_tmp2+rr, esm = ss*size+mm;
                                Q(ii)(nn+size,mm) -= twojP1*den_tmp(ss,size_tmp2+rr) * h2eSSLL_JK.K[ir][jr][enr][esm];
                            }
                            if(with_gaunt)
                            {
                                int enm = nn*size+mm, ers = rr*size_tmp2+ss, erm = rr*size+mm, ens = nn*size_tmp2+ss;

                                Q(ii)(mm,nn) -= twojP1*den_tmp(size_tmp2+ss,size_tmp2+rr) * gauntLSSL_JK.K[ir][jr][emr][esn];
                                Q(ii)(mm+size,nn) += twojP1*den_tmp(ss+size_tmp2,rr)*gauntLSLS_JK.J[ir][jr][enm][ers] + twojP1*den_tmp(ss,  size_tmp2+rr) * gauntLSSL_JK.J[jr][ir][esr][emn];
                                Q(ii)(mm+size,nn+size) -= twojP1*den_tmp(ss,rr) * gauntLSSL_JK.K[jr][ir][esn][emr];
                                if(mm != nn)
                                {
                                    int ern = rr*size+nn, ems = mm*size_tmp2+ss;
                                    Q(ii)(nn+size,mm) += twojP1*den_tmp(size_tmp2+ss,rr)*gauntLSLS_JK.J[ir][jr][emn][ers] + twojP1*den_tmp  (ss,size_tmp2+rr) * gauntLSSL_JK.J[jr][ir][esr][enm];
                                }
                            }
                        }               
                    }
                }
                for(int ii = 0; ii < NOpenShells+1; ii++)
                {
                    Q(ii)(nn,mm) = Q(ii)(mm,nn);
                    Q(ii)(mm,nn+size) = Q(ii)(nn+size,mm);
                    Q(ii)(nn,mm+size) = Q(ii)(mm+size,nn);
                    Q(ii)(size+nn,size+mm) = Q(ii)(size+mm,size+nn);
                }
            }
        }

        for(int ii = 0; ii < NOpenShells+1; ii++)
        {
            double f_i;
            if(ii == 0) f_i = 1.0;
            else f_i = f_list[ii-1];
            MatrixXd fock_e = h1e_4c(Iirrep)+0.5*Q(0);
            for(int jj = 1; jj < NOpenShells+1; jj++)
            {
                double f_j;
                if(jj == ii) f_j = (NN_list[jj-1]-1.0)/(MM_list[jj-1]-1.0);
                else f_j = f_list[jj-1];
                fock_e += 0.5*f_j*Q(jj);
            }
            double tmp = 0.0;
            for(int mm = 0; mm < fock_e.rows(); mm++)
            for(int nn = 0; nn < fock_e.rows(); nn++)
            {
                tmp += fock_e(mm,nn)*densityShells(ii)(Iirrep)(mm,nn);
            }
            ene += f_i * tmp * (irrep_list(Iirrep).two_j+1);
        }
    }

    return ene;
}
