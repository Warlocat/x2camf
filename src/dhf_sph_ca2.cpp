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

    // to be deleted
    NN = NN_list[0];
    MM = MM_list[0];
    f_NM = f_list[0];
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
    StartTime = clock();
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
                // evaluateFock(fockShells(0)(ir),fockShells(1)(ir),twoC,densityShells(0),densityShells(1),size_tmp,ir);
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
            density.resize(occMax_irrep);
            density_o.resize(occMax_irrep);

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
            // evaluateFock(fockShells(0)(ir),fockShells(1)(ir),twoC,densityShells(0),densityShells(1),size_tmp,ir);
            // evaluateFock(fock_c(ir),fock_o(ir),twoC,density,density_o,size_tmp,ir);
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
    EndTime = clock();

    
    for(int ir = 0; ir < occMax_irrep; ir += irrep_list(ir).two_j+1)
    {
        for(int jj = 1; jj < irrep_list(ir).two_j+1; jj++)
        {
            // fock_c(ir+jj) = fock_c(ir);
            // fock_o(ir+jj) = fock_o(ir);
            ene_orb(ir+jj) = ene_orb(ir);
            coeff(ir+jj) = coeff(ir);
            density(ir+jj) = densityShells(0)(ir);
            density_o(ir+jj) = densityShells(1)(ir);
            for(int ii = 0; ii < NOpenShells+1; ii++)
                densityShells(ii)(ir+jj) = densityShells(ii)(ir);
        }
    }    
    cout << "DHF iterations finished in " << (EndTime - StartTime) / (double)CLOCKS_PER_SEC << " seconds." << endl << endl;
}


/* 
    evaluate Fock matrix 
*/
void DHF_SPH_CA2::evaluateFock(MatrixXd& fock_c, MatrixXd& fock_o, const bool& twoC, const vMatrixXd& den_c, const vMatrixXd& den_o, const int& size, const int& Iirrep)
{
    int ir = all2compact(Iirrep);
    if(twoC)
    {
        MatrixXd S = overlap_4c(Iirrep);
        MatrixXd Rc = (den_c(Iirrep)).transpose(),
                 Ro = (den_o(Iirrep)).transpose();
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
            fock_c(nn,mm) = fock_c(mm,nn);
            fock_o(nn,mm) = fock_o(mm,nn);
            QQ(nn,mm) = QQ(mm,nn);
        }
        fock_c = fock_c + f_NM/(MM-1.0)*(S*Ro*QQ+QQ*Ro*S);
        fock_o = fock_o + 1.0/(MM-1.0)*(S*Rc*QQ+QQ*Rc*S);
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
void DHF_SPH_CA2::evaluateFock_oneF(MatrixXd& fock_c, const bool& twoC, const vMatrixXd& den_c, const vMatrixXd& den_o, const vMatrixXd& den_u, const int& size, const int& Iirrep)
{
    int ir = all2compact(Iirrep);
    if(twoC)
    {
        // cout << (den_c(Iirrep) + den_o(Iirrep) + den_u(Iirrep)).transpose() * overlap_4c(Iirrep) << endl << endl;
        MatrixXd Rcu = (den_c(Iirrep)+den_u(Iirrep)).transpose(),
                 Rou = (den_o(Iirrep)+den_u(Iirrep)).transpose(),
                 Rco = (den_c(Iirrep)+den_o(Iirrep)).transpose();
        MatrixXd Hc(size,size), Ho(size,size);
        #pragma omp parallel  for
        for(int mm = 0; mm < size; mm++)
        for(int nn = 0; nn <= mm; nn++)
        {
            Hc(mm,nn) = h1e_4c(Iirrep)(mm,nn);
            Ho(mm,nn) = h1e_4c(Iirrep)(mm,nn);
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
                    Hc(mm,nn) += twojP1*den_tc(aa,bb) * h2eLLLL_JK.J[ir][jr][emn][eab];
                    Ho(mm,nn) += twojP1*den_to(aa,bb) * h2eLLLL_JK.J[ir][jr][emn][eab];
                }
            }
            Hc(nn,mm) = Hc(mm,nn);
            Ho(nn,mm) = Ho(mm,nn);
        }
        fock_c = 0.5*Rcu*Hc*Rcu + 0.5*Rou*Ho*Rou + 0.5/(1.0-f_NM)*Rco*(Hc-f_NM*Ho)*Rco;
    }
    else
    {
        cout << (den_c(Iirrep) + den_o(Iirrep) + den_u(Iirrep)).transpose() * overlap_4c(Iirrep) << endl << endl;
        MatrixXd Rcu = (den_c(Iirrep)+den_u(Iirrep)).transpose(),
                 Rou = (den_o(Iirrep)+den_u(Iirrep)).transpose(),
                 Rco = (den_c(Iirrep)+den_o(Iirrep)).transpose();
        MatrixXd Hc(2*size,2*size), Ho(2*size,2*size);
        #pragma omp parallel  for
        for(int mm = 0; mm < size; mm++)
        for(int nn = 0; nn <= mm; nn++)
        {
            Hc(mm,nn) = h1e_4c(Iirrep)(mm,nn);
            Hc(mm+size,nn) = h1e_4c(Iirrep)(mm+size,nn);
            Hc(mm+size,nn+size) = h1e_4c(Iirrep)(mm+size,nn+size);
            Ho(mm,nn) = h1e_4c(Iirrep)(mm,nn);
            Ho(mm+size,nn) = h1e_4c(Iirrep)(mm+size,nn);
            Ho(mm+size,nn+size) = h1e_4c(Iirrep)(mm+size,nn+size);
            if(mm != nn) 
            {
                Hc(nn+size,mm) = h1e_4c(Iirrep)(nn+size,mm);
                Ho(nn+size,mm) = h1e_4c(Iirrep)(nn+size,mm);
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
                    Hc(mm,nn) += twojP1*den_tc(ss,rr) * h2eLLLL_JK.J[ir][jr][emn][esr] + twojP1*den_tc(size_tmp2+ss,size_tmp2+rr) * h2eSSLL_JK.J[jr][ir][esr][emn];
                    Hc(mm+size,nn) -= twojP1*den_tc(ss,size_tmp2+rr) * h2eSSLL_JK.K[ir][jr][emr][esn];
                    Hc(mm+size,nn+size) += twojP1*den_tc(size_tmp2+ss,size_tmp2+rr) * h2eSSSS_JK.J[ir][jr][emn][esr] + twojP1*den_tc(ss,rr) * h2eSSLL_JK.J[ir][jr][emn][esr];

                    Ho(mm,nn) += twojP1*den_to(ss,rr) * h2eLLLL_JK.J[ir][jr][emn][esr] + twojP1*den_to(size_tmp2+ss,size_tmp2+rr) * h2eSSLL_JK.J[jr][ir][esr][emn];
                    Ho(mm+size,nn) -= twojP1*den_to(ss,size_tmp2+rr) * h2eSSLL_JK.K[ir][jr][emr][esn];
                    Ho(mm+size,nn+size) += twojP1*den_to(size_tmp2+ss,size_tmp2+rr) * h2eSSSS_JK.J[ir][jr][emn][esr] + twojP1*den_to(ss,rr) * h2eSSLL_JK.J[ir][jr][emn][esr];
                    
                    if(mm != nn) 
                    {
                        int enr = nn*size_tmp2+rr, esm = ss*size+mm;
                        Hc(nn+size,mm) -= twojP1*den_tc(ss,size_tmp2+rr) * h2eSSLL_JK.K[ir][jr][enr][esm];
                        Ho(nn+size,mm) -= twojP1*den_to(ss,size_tmp2+rr) * h2eSSLL_JK.K[ir][jr][enr][esm];
                    }

                    if(with_gaunt)
                    {
                        int enm = nn*size+mm, ers = rr*size_tmp2+ss, erm = rr*size+mm, ens = nn*size_tmp2+ss;

                        Hc(mm,nn) -= twojP1*den_tc(size_tmp2+ss,size_tmp2+rr) * gauntLSSL_JK.K[ir][jr][emr][esn];
                        Hc(mm+size,nn) += twojP1*den_tc(ss+size_tmp2,rr)*gauntLSLS_JK.J[ir][jr][enm][ers] + twojP1*den_tc(ss,size_tmp2+rr) * gauntLSSL_JK.J[jr][ir][esr][emn];
                        Hc(mm+size,nn+size) -= twojP1*den_tc(ss,rr) * gauntLSSL_JK.K[jr][ir][esn][emr];

                        Ho(mm,nn) -= twojP1*den_to(size_tmp2+ss,size_tmp2+rr) * gauntLSSL_JK.K[ir][jr][emr][esn];       
                        Ho(mm+size,nn) += twojP1*den_to(ss+size_tmp2,rr)*gauntLSLS_JK.J[ir][jr][enm][ers] + twojP1*den_to(ss,size_tmp2+rr) * gauntLSSL_JK.J[jr][ir][esr][emn];              
                        Ho(mm+size,nn+size) -= twojP1*den_to(ss,rr) * gauntLSSL_JK.K[jr][ir][esn][emr];

                        if(mm != nn)
                        {
                            int ern = rr*size+nn, ems = mm*size_tmp2+ss;
                            Hc(nn+size,mm) += twojP1*den_tc(size_tmp2+ss,rr)*gauntLSLS_JK.J[ir][jr][emn][ers] + twojP1*den_tc(ss,size_tmp2+rr) * gauntLSSL_JK.J[jr][ir][esr][enm];
                            Ho(nn+size,mm) += twojP1*den_to(size_tmp2+ss,rr)*gauntLSLS_JK.J[ir][jr][emn][ers] + twojP1*den_to(ss,size_tmp2+rr) * gauntLSSL_JK.J[jr][ir][esr][enm];
                        }
                    }
                }
            }
            Hc(nn,mm) = Hc(mm,nn);
            Hc(mm,nn+size) = Hc(nn+size,mm);
            Hc(nn,mm+size) = Hc(mm+size,nn);
            Hc(size+nn,size+mm) = Hc(size+mm,size+nn);
            Ho(nn,mm) = Ho(mm,nn);
            Ho(mm,nn+size) = Ho(nn+size,mm);
            Ho(nn,mm+size) = Ho(mm+size,nn);
            Ho(size+nn,size+mm) = Ho(size+mm,size+nn);
        }
        fock_c = 0.5*Rcu*Hc*Rcu + 0.5*Rou*Ho*Rou + 0.5/(1.0-f_NM)*Rco*(Hc-f_NM*Ho)*Rco;
    }
}
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

vMatrixXd DHF_SPH_CA2::get_amfi_unc(INT_SPH& int_sph_, const bool& twoC, const string& Xmethod, bool amfi_with_gaunt, bool amfi_with_gauge)
{
    cout << "Running DHF_SPH_CA2::get_amfi_unc" << endl;
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


vMatrixXd DHF_SPH_CA2::get_amfi_unc_2c(const int2eJK& h2eSSLL_SD, const int2eJK& h2eSSSS_SD, const bool& amfi_with_gaunt)
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
        density_tmp(ir) = evaluateDensity_aoc(coeff_tmp(ir),occNumberShells[0](ir),false) + f_NM*evaluateDensity_aoc(coeff_tmp(ir),occNumberShells[1](ir),false);
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
void DHF_SPH_CA2::basisGenerator(string basisName, string filename, const INT_SPH& intor, const INT_SPH& intorAll, const bool& sf, const string& tag)
{
    cout << "Running DHF_SPH_CA2::basisGenerator" << endl;
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
        if(occNumber(ir).rows() == 0 || abs(occNumber(ir)(0) < 1e-5)) break;
        occL++;
    }
    basisInfo.resize(occL);
    
    // For each l-shell, count how many orbitals are fully or partially occupied
    // and resize basisInfo(l)
    occL = 0;
    for(int ir = 0; ir < irrep_list.rows(); ir += 4*irrep_list(ir).l+2)
    {
        int occN = 0;
        if(occNumber(ir).rows() == 0 || abs(occNumber(ir)(0) < 1e-5)) break;
        for(int ii = 0; ii < occNumber(ir).rows(); ii++)
        {
            if(abs(occNumber(ir)(ii)-1) < 1e-4 || abs(occNumberShells[1](ir)(ii)-1) < 1e-4)
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