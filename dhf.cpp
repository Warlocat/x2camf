#include"dhf.h"
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

/*********************************************************/
/**********    Member functions of class DHF    **********/
/*********************************************************/

DHF::DHF(GTO_SPINOR& gto_, const bool& unc)
{
    nelec_a = gto_.nelec_a;
    nelec_b = gto_.nelec_b;
    if(unc)    size_basis = gto_.size_gtou_spinor;
    else    size_basis = gto_.size_gtoc_spinor;
    /* In DHF, h1e is V and h2e is h2eLLLL */
    StartTime = clock();
    gto_.get_h1e_fast(overlap,kinetic,h1e,WWW,unc);
    EndTime = clock();
    cout << "1e-integral finished in " << (EndTime - StartTime) / (double)CLOCKS_PER_SEC << " seconds." << endl;

    norm_s.resize(size_basis);
    norm_s = VectorXd::Zero(size_basis);

    /* Turn on or off the normalization conditions */
    for(int ii = 0; ii < size_basis; ii++)
    {
        // norm_s(ii) = sqrt(kinetic(ii,ii) / 2.0 / speedOfLight / speedOfLight);
        norm_s(ii) = 1.0;
    }
    
    StartTime = clock();
    gto_.get_h2e_fast(h2e,h2eSSLL,h2eSSSS,unc);
    EndTime = clock();
    cout << "1e-integral finished in " << (EndTime - StartTime) / (double)CLOCKS_PER_SEC << " seconds." << endl;

    for(int ii = 0; ii < size_basis*size_basis; ii++)
    for(int jj = 0; jj < size_basis*size_basis; jj++)
    {
        int a = ii / size_basis, b = ii - a * size_basis, c = jj / size_basis, d = jj - c * size_basis;
        h2eSSLL(ii,jj) = h2eSSLL(ii,jj) / 4.0 / pow(speedOfLight,2) / norm_s(a) / norm_s(b);
        h2eSSSS(ii,jj) = h2eSSSS(ii,jj) / 16.0 / pow(speedOfLight,4) / norm_s(a)/norm_s(b)/norm_s(c)/norm_s(d);
    } 

    /*
        overlap_4c = [[S, 0], [0, T/2c^2]]
        h1e_4c = [[V, T], [T, W/4c^2 - T]]
    */
    h1e_4c.resize(size_basis*2,size_basis*2);
    overlap_4c.resize(size_basis*2,size_basis*2);
    h1e_4c = MatrixXd::Zero(size_basis*2,size_basis*2);
    overlap_4c = MatrixXd::Zero(size_basis*2,size_basis*2);
    for(int ii = 0; ii < size_basis; ii++)
    for(int jj = 0; jj < size_basis; jj++)
    {
        overlap_4c(ii,jj) = overlap(ii,jj);
        overlap_4c(size_basis+ii, size_basis+jj) = kinetic(ii,jj) / 2.0 / speedOfLight / speedOfLight/ norm_s(ii)/norm_s(jj);
        h1e_4c(ii,jj) = h1e(ii,jj);
        h1e_4c(size_basis+ii,jj) = kinetic(ii,jj) / norm_s(ii);
        h1e_4c(ii,size_basis+jj) = kinetic(ii,jj) / norm_s(jj);
        h1e_4c(size_basis+ii,size_basis+jj) = (WWW(ii,jj)/4.0/speedOfLight/speedOfLight - kinetic(ii,jj))/ norm_s(ii)/norm_s(jj);
    }
    overlap_half_i_4c = matrix_half_inverse(overlap_4c);
}

DHF::DHF(const GTO_SPINOR& gto_, const string& h2e_file, const bool& unc)
{
    nelec_a = gto_.nelec_a;
    nelec_b = gto_.nelec_b;
    if(unc)    size_basis = gto_.size_gtou_spinor;
    else    size_basis = gto_.size_gtoc_spinor;
    uncontracted = unc;

    /* In DHF, h1e is V and h2e is h2eLLLL */
    overlap = gto_.get_h1e("overlap",unc);
    kinetic = gto_.get_h1e("s_p_s_p",unc) / 2.0;
    h1e = gto_.get_h1e("nuc_attra",unc);
    WWW = gto_.get_h1e("s_p_nuc_s_p",unc);

    /*
        overlap_4c = [[S, 0], [0, T/2c^2]]
        h1e_4c = [[V, T], [T, W/4c^2 - T]]
    */
    h1e_4c.resize(size_basis*2,size_basis*2);
    overlap_4c.resize(size_basis*2,size_basis*2);
    h1e_4c = MatrixXd::Zero(size_basis*2,size_basis*2);
    overlap_4c = MatrixXd::Zero(size_basis*2,size_basis*2);
    for(int ii = 0; ii < size_basis; ii++)
    for(int jj = 0; jj < size_basis; jj++)
    {
        overlap_4c(ii,jj) = overlap(ii,jj);
        overlap_4c(size_basis+ii, size_basis+jj) = kinetic(ii,jj) / 2.0 / speedOfLight / speedOfLight;
        h1e_4c(ii,jj) = h1e(ii,jj);
        h1e_4c(size_basis+ii,jj) = kinetic(ii,jj);
        h1e_4c(ii,size_basis+jj) = kinetic(ii,jj);
        h1e_4c(size_basis+ii,size_basis+jj) = WWW(ii,jj)/4.0/speedOfLight/speedOfLight - kinetic(ii,jj);
    }
    overlap_half_i_4c = matrix_half_inverse(overlap_4c);
    
    h2e.resize(size_basis*size_basis, size_basis*size_basis);
    h2e = MatrixXd::Zero(size_basis*size_basis, size_basis*size_basis);
    h2eSSLL.resize(size_basis*size_basis, size_basis*size_basis);
    h2eSSLL = MatrixXd::Zero(size_basis*size_basis, size_basis*size_basis);
    h2eSSSS.resize(size_basis*size_basis, size_basis*size_basis);
    h2eSSSS = MatrixXd::Zero(size_basis*size_basis, size_basis*size_basis); 

    readIntegrals(h2e, h2e_file+"LLLL");
    readIntegrals(h2eSSLL, h2e_file+"SSLL");
    readIntegrals(h2eSSSS, h2e_file+"SSSS");
    h2eSSLL = h2eSSLL / 4.0 / pow(speedOfLight,2);
    h2eSSSS = h2eSSSS / 16.0 / pow(speedOfLight,4);
}

DHF::DHF(const GTO_SPINOR& gto_, const MatrixXd& h2eLLLL_, const MatrixXd& h2eSSLL_, const MatrixXd& h2eSSSS_, const bool& unc)
{
    nelec_a = gto_.nelec_a;
    nelec_b = gto_.nelec_b;
    if(unc)    size_basis = gto_.size_gtou_spinor;
    else    size_basis = gto_.size_gtoc_spinor;
    /* In DHF, h1e is V and h2e is h2eLLLL */
    overlap = gto_.get_h1e("overlap",unc);
    kinetic = gto_.get_h1e("s_p_s_p",unc) / 2.0;
    h1e = gto_.get_h1e("nuc_attra",unc);
    WWW = gto_.get_h1e("s_p_nuc_s_p",unc);

    norm_s.resize(size_basis);
    norm_s = VectorXd::Zero(size_basis);

    /* Turn on or off the normalization conditions */
    for(int ii = 0; ii < size_basis; ii++)
    {
        // norm_s(ii) = sqrt(kinetic(ii,ii) / 2.0 / speedOfLight / speedOfLight);
        norm_s(ii) = 1.0;
    }
    
    h2e = h2eLLLL_;
    h2eSSLL.resize(size_basis*size_basis,size_basis*size_basis);
    h2eSSSS.resize(size_basis*size_basis,size_basis*size_basis);

    for(int ii = 0; ii < size_basis*size_basis; ii++)
    for(int jj = 0; jj < size_basis*size_basis; jj++)
    {
        int a = ii / size_basis, b = ii - a * size_basis, c = jj / size_basis, d = jj - c * size_basis;
        h2eSSLL(ii,jj) = h2eSSLL_(ii,jj) / 4.0 / pow(speedOfLight,2) / norm_s(a) / norm_s(b);
        h2eSSSS(ii,jj) = h2eSSSS_(ii,jj) / 16.0 / pow(speedOfLight,4) / norm_s(a)/norm_s(b)/norm_s(c)/norm_s(d);
    }
    

    /*
        overlap_4c = [[S, 0], [0, T/2c^2]]
        h1e_4c = [[V, T], [T, W/4c^2 - T]]
    */
    h1e_4c.resize(size_basis*2,size_basis*2);
    overlap_4c.resize(size_basis*2,size_basis*2);
    h1e_4c = MatrixXd::Zero(size_basis*2,size_basis*2);
    overlap_4c = MatrixXd::Zero(size_basis*2,size_basis*2);
    for(int ii = 0; ii < size_basis; ii++)
    for(int jj = 0; jj < size_basis; jj++)
    {
        overlap_4c(ii,jj) = overlap(ii,jj);
        overlap_4c(size_basis+ii, size_basis+jj) = kinetic(ii,jj) / 2.0 / speedOfLight / speedOfLight/ norm_s(ii)/norm_s(jj);
        h1e_4c(ii,jj) = h1e(ii,jj);
        h1e_4c(size_basis+ii,jj) = kinetic(ii,jj) / norm_s(ii);
        h1e_4c(ii,size_basis+jj) = kinetic(ii,jj) / norm_s(jj);
        h1e_4c(size_basis+ii,size_basis+jj) = (WWW(ii,jj)/4.0/speedOfLight/speedOfLight - kinetic(ii,jj))/ norm_s(ii)/norm_s(jj);
    }
    overlap_half_i_4c = matrix_half_inverse(overlap_4c);
}


DHF::DHF(const string& h1e_file, const string& h2e_file)
{
    ifstream ifs;
    ifs.open(h1e_file);
        ifs >> nelec_a >> nelec_b >> size_basis;
        overlap.resize(size_basis,size_basis);
        kinetic.resize(size_basis,size_basis);
        h1e.resize(size_basis,size_basis);
        WWW.resize(size_basis,size_basis);
        for(int ii = 0; ii < size_basis; ii++)
        for(int jj = 0; jj < size_basis; jj++)
            ifs >> overlap(ii,jj);
        for(int ii = 0; ii < size_basis; ii++)
        for(int jj = 0; jj < size_basis; jj++)
            ifs >> kinetic(ii,jj);
        for(int ii = 0; ii < size_basis; ii++)
        for(int jj = 0; jj < size_basis; jj++)
            ifs >> h1e(ii,jj);
        for(int ii = 0; ii < size_basis; ii++)
        for(int jj = 0; jj < size_basis; jj++)
            ifs >> WWW(ii,jj);
    ifs.close();

    /*
        overlap_4c = [[S, 0], [0, T/2c^2]]
        h1e_4c = [[V, T], [T, W/4c^2 - T]]
    */
    h1e_4c.resize(size_basis*2,size_basis*2);
    overlap_4c.resize(size_basis*2,size_basis*2);
    h1e_4c = MatrixXd::Zero(size_basis*2,size_basis*2);
    overlap_4c = MatrixXd::Zero(size_basis*2,size_basis*2);
    for(int ii = 0; ii < size_basis; ii++)
    for(int jj = 0; jj < size_basis; jj++)
    {
        overlap_4c(ii,jj) = overlap(ii,jj);
        overlap_4c(size_basis+ii, size_basis+jj) = kinetic(ii,jj) / 2.0 / speedOfLight / speedOfLight;
        h1e_4c(ii,jj) = h1e(ii,jj);
        h1e_4c(size_basis+ii,jj) = kinetic(ii,jj);
        h1e_4c(ii,size_basis+jj) = kinetic(ii,jj);
        h1e_4c(size_basis+ii,size_basis+jj) = WWW(ii,jj)/4.0/speedOfLight/speedOfLight - kinetic(ii,jj);
    }
    overlap_half_i_4c = matrix_half_inverse(overlap_4c);
    
    h2e.resize(size_basis*size_basis, size_basis*size_basis);
    h2e = MatrixXd::Zero(size_basis*size_basis, size_basis*size_basis);
    h2eSSLL.resize(size_basis*size_basis, size_basis*size_basis);
    h2eSSLL = MatrixXd::Zero(size_basis*size_basis, size_basis*size_basis);
    h2eSSSS.resize(size_basis*size_basis, size_basis*size_basis);
    h2eSSSS = MatrixXd::Zero(size_basis*size_basis, size_basis*size_basis); 

    readIntegrals(h2e, h2e_file+"LLLL");
    readIntegrals(h2eSSLL, h2e_file+"SSLL");
    readIntegrals(h2eSSSS, h2e_file+"SSSS");
    h2eSSLL = h2eSSLL / 4.0 / pow(speedOfLight,2);
    h2eSSSS = h2eSSSS / 16.0 / pow(speedOfLight,4);
}

DHF::~DHF()
{
}

void DHF::runSCF()
{
    vector<MatrixXd> error4DIIS, fock4DIIS;
    fock_4c.resize(2*size_basis,2*size_basis);
    cout << endl;
    cout << "Start Dirac Hartree-Fock iterations..." << endl;
    cout << "size of 4-c basis set: " << 2*size_basis << endl;
    cout << endl;
    MatrixXd newDen;
    eigensolverG(h1e_4c, overlap_half_i_4c, ene_orb, coeff);
    density = evaluateDensity_spinor(coeff, nelec_a+nelec_b);

    for(int iter = 1; iter <= maxIter; iter++)
    {
        if(iter <= 2)
        {
            #pragma omp parallel  for
            for(int mm = 0; mm < size_basis; mm++)
            for(int nn = 0; nn <= mm; nn++)
            {
                fock_4c(mm,nn) = h1e_4c(mm,nn);
                fock_4c(mm+size_basis,nn) = h1e_4c(mm+size_basis,nn);
                if(mm != nn) fock_4c(nn+size_basis,mm) = h1e_4c(nn+size_basis,mm);
                fock_4c(mm+size_basis,nn+size_basis) = h1e_4c(mm+size_basis,nn+size_basis);
                for(int ss = 0; ss < size_basis; ss++)
                for(int rr = 0; rr < size_basis; rr++)
                {
                    int emn = mm*size_basis+nn, esr = ss*size_basis+rr, emr = mm*size_basis+rr, esn = ss*size_basis+nn;
                    fock_4c(mm,nn) += density(ss,rr) * (h2e(emn,esr) - h2e(emr,esn)) 
                                    + density(size_basis+ss,size_basis+rr) * h2eSSLL(esr,emn);
                    fock_4c(mm+size_basis,nn) -= density(ss,size_basis+rr) * h2eSSLL(emr,esn);
                    if(mm != nn) 
                    {
                        int enr = nn*size_basis+rr, esm = ss*size_basis+mm;
                        fock_4c(nn+size_basis,mm) -= density(ss,size_basis+rr) * h2eSSLL(enr,esm);
                    }
                    fock_4c(mm+size_basis,nn+size_basis) += density(size_basis+ss,size_basis+rr) * (h2eSSSS(emn,esr) - h2eSSSS(emr,esn)) + density(ss,rr) * h2eSSLL(emn,esr);
                }
                fock_4c(nn,mm) = fock_4c(mm,nn);
                fock_4c(nn+size_basis,mm+size_basis) = fock_4c(mm+size_basis,nn+size_basis);
                fock_4c(nn,mm+size_basis) = fock_4c(mm+size_basis,nn);
                fock_4c(mm,nn+size_basis) = fock_4c(nn+size_basis,mm);
            }
        }
        else
        {
            int tmp_size = fock4DIIS.size();
            MatrixXd B4DIIS(tmp_size+1,tmp_size+1);
            VectorXd vec_b(tmp_size+1);
            for(int ii = 0; ii < tmp_size; ii++)
            {    
                for(int jj = 0; jj <= ii; jj++)
                {
                    B4DIIS(ii,jj) = (error4DIIS[ii].adjoint()*error4DIIS[jj])(0,0);
                    B4DIIS(jj,ii) = B4DIIS(ii,jj);
                }
                B4DIIS(tmp_size, ii) = -1.0;
                B4DIIS(ii, tmp_size) = -1.0;
                vec_b(ii) = 0.0;
            }
            B4DIIS(tmp_size, tmp_size) = 0.0;
            vec_b(tmp_size) = -1.0;
            VectorXd C = B4DIIS.partialPivLu().solve(vec_b);
            fock_4c = MatrixXd::Zero(2*size_basis,2*size_basis);
            for(int ii = 0; ii < tmp_size; ii++)
            {
                fock_4c += C(ii) * fock4DIIS[ii];
            }
        }
        eigensolverG(fock_4c, overlap_half_i_4c, ene_orb, coeff);
        newDen = evaluateDensity_spinor(coeff, nelec_a+nelec_b);
        d_density = evaluateChange(density, newDen);
        
        cout << "Iter #" << iter << " maximum density difference: " << d_density << endl;
        
        density = newDen;
        if(d_density < convControl) 
        {
            converged = true;
            cout << endl << "DHF converges after " << iter << " iterations." << endl << endl;

            cout << "\tOrbital\t\tEnergy(in hartree)\n";
            cout << "\t*******\t\t******************\n";
            for(int ii = 1; ii <= size_basis; ii++)
            {
                cout << "\t" << ii << "\t\t" << setprecision(15) << ene_orb(size_basis + ii - 1) << endl;
            }

            ene_scf = 0.0;
            MatrixXd coeff_tmp = coeff.block(0,size_basis,2*size_basis,nelec_a+nelec_b);
            density = coeff_tmp*coeff_tmp.transpose();
            for(int ii = 0; ii < size_basis * 2; ii++)
            for(int jj = 0; jj < size_basis * 2; jj++)
            {
                ene_scf += 0.5 * density(ii,jj) * (h1e_4c(jj,ii) + fock_4c(jj,ii));
            }
            cout << "Final DHF energy is " << setprecision(15) << ene_scf << " hartree." << endl;
            break;            
        }
        #pragma omp parallel  for
        for(int mm = 0; mm < size_basis; mm++)
        for(int nn = 0; nn <= mm; nn++)
        {
            fock_4c(mm,nn) = h1e_4c(mm,nn);
            fock_4c(mm+size_basis,nn) = h1e_4c(mm+size_basis,nn);
            if(mm != nn) fock_4c(nn+size_basis,mm) = h1e_4c(nn+size_basis,mm);
            fock_4c(mm+size_basis,nn+size_basis) = h1e_4c(mm+size_basis,nn+size_basis);
            for(int ss = 0; ss < size_basis; ss++)
            for(int rr = 0; rr < size_basis; rr++)
            {
                int emn = mm*size_basis+nn, esr = ss*size_basis+rr, emr = mm*size_basis+rr, esn = ss*size_basis+nn;
                fock_4c(mm,nn) += density(ss,rr) * (h2e(emn,esr) - h2e(emr,esn)) 
                                + density(size_basis+ss,size_basis+rr) * h2eSSLL(esr,emn);
                fock_4c(mm+size_basis,nn) -= density(ss,size_basis+rr) * h2eSSLL(emr,esn);
                if(mm != nn) 
                {
                    int enr = nn*size_basis+rr, esm = ss*size_basis+mm;
                    fock_4c(nn+size_basis,mm) -= density(ss,size_basis+rr) * h2eSSLL(enr,esm);
                }
                fock_4c(mm+size_basis,nn+size_basis) += density(size_basis+ss,size_basis+rr) * (h2eSSSS(emn,esr) - h2eSSSS(emr,esn)) + density(ss,rr) * h2eSSLL(emn,esr);
            }
            fock_4c(nn,mm) = fock_4c(mm,nn);
            fock_4c(nn+size_basis,mm+size_basis) = fock_4c(mm+size_basis,nn+size_basis);
            fock_4c(nn,mm+size_basis) = fock_4c(mm+size_basis,nn);
            fock_4c(mm,nn+size_basis) = fock_4c(nn+size_basis,mm);
        }
        if(error4DIIS.size() >= size_DIIS)
        {
            error4DIIS.erase(error4DIIS.begin());
            error4DIIS.push_back(evaluateErrorDIIS(fock_4c,overlap_4c,density));
            fock4DIIS.erase(fock4DIIS.begin());
            fock4DIIS.push_back(fock_4c);
        }
        else
        {
            error4DIIS.push_back(evaluateErrorDIIS(fock_4c,overlap_4c,density));
            fock4DIIS.push_back(fock_4c);
        }
    }
}

void DHF::readIntegrals(MatrixXd& h2e_, const string& filename)
{
    int size = round(sqrt(h2e_.cols()));
    ifstream ifs;
    double tmp;
    VectorXi indices(4);
    ifs.open(filename);
        while(true)
        {
            ifs >> tmp >> indices(0) >> indices(1) >> indices(2) >> indices(3);
            if(indices(0) == 0) break;
            else
            {
                int ij = (indices(0) - 1) * size + indices(1) - 1, kl = (indices(2) - 1) * size + indices(3) - 1;
                h2e_(ij,kl) = tmp;
            }
        }            
    ifs.close();
}


MatrixXd DHF::evaluateDensity_spinor(const MatrixXd& coeff_, const int& nocc, const bool& spherical)
{
    int size = coeff_.cols()/2;
    MatrixXd den(2*size,2*size);
    den = MatrixXd::Zero(2*size,2*size);
    if(!spherical)
    {
        for(int aa = 0; aa < size; aa++)
        for(int bb = 0; bb < size; bb++)
        {
            for(int ii = 0; ii < nocc; ii++)
            {
                den(aa,bb) += coeff_(aa,ii+size) * coeff_(bb,ii+size);
                den(size+aa,bb) += coeff_(size+aa,ii+size) * coeff_(bb,ii+size);
                den(aa,size+bb) += coeff_(aa,ii+size) * coeff_(size+bb,ii+size);
                den(size+aa,size+bb) += coeff_(size+aa,ii+size) * coeff_(size+bb,ii+size);
            }
        }
    }
    else
    {
        cout << "ERROR: Spherical occupation is NOT supported now!" << endl;
        exit(99);
    }    
    return den;
}


MatrixXd DHF::get_amfi(const MatrixXd& h2eSSLL_SD, const MatrixXd& h2eSSSS_SD, const MatrixXd& coeff_con, const bool& spherical)
{
    if(!uncontracted)
    {
        cout << "ERROR: get_amfi is called with contracted basis!" << endl;
        exit(99);
    }
    else if(!converged)
    {
        cout << "Warning: SCF did NOT converge when get_amfi is called!" << endl;
    }
    return get_amfi(coeff, h2eSSLL_SD, h2eSSSS_SD, h1e_4c, overlap_4c, nelec_a+nelec_b, coeff_con, spherical);
}

MatrixXd DHF::get_amfi(const MatrixXd& coeff_4c, const MatrixXd& h2eSSLL_SD, const MatrixXd& h2eSSSS_SD, const MatrixXd& h1e_4c_, const MatrixXd& overlap_4c_, const int& nocc, const MatrixXd& coeff_con, const bool& spherical)
{
    MatrixXd density = evaluateDensity_spinor(coeff_4c, nocc, spherical);
    int size = round(sqrt(h2eSSLL_SD.cols()));
    MatrixXd SO_LL(size,size), SO_LS(size,size), SO_SL(size,size), SO_SS(size,size);

    /* 
        Evaluate SO integrals in 4c basis
        The structure is the same as 2e Coulomb integrals in fock matrix 
    */
    for(int mm = 0; mm < size; mm++)
    for(int nn = 0; nn < size; nn++)
    {
        SO_LL(mm,nn) = SO_LS(mm,nn) = SO_SL(mm,nn) = SO_SS(mm,nn) = 0.0;
        for(int ss = 0; ss < size; ss++)
        for(int rr = 0; rr < size; rr++)
        {
            int emn = mm*size+nn, esr = ss*size+rr, emr = mm*size+rr, esn = ss*size+nn;
            SO_LL(mm,nn) += density(size+ss,size+rr) * h2eSSLL_SD(esr,emn)/ 4.0 / pow(speedOfLight,2);
            SO_LS(mm,nn) -= density(size+ss,rr) * h2eSSLL_SD(esn,emr)/ 4.0 / pow(speedOfLight,2);
            SO_SL(mm,nn) -= density(ss,size+rr) * h2eSSLL_SD(emr,esn)/ 4.0 / pow(speedOfLight,2);
            SO_SS(mm,nn) += density(size+ss,size+rr) * (h2eSSSS_SD(emn,esr) - h2eSSSS_SD(emr,esn))/ 16.0 / pow(speedOfLight,4) + density(ss,rr) * h2eSSLL_SD(emn,esr)/ 4.0 / pow(speedOfLight,2);
        }
    }

    /* Transform SO_4c to SO_2c_eff and then SO_2c using X2C techniques */
    GeneralizedSelfAdjointEigenSolver<MatrixXd> solver(h1e_4c_, overlap_4c_);
    MatrixXd XXX = X2C::get_X(solver.eigenvectors());
    MatrixXd RRR = X2C::get_R(overlap_4c_, XXX);
    MatrixXd SO_2c_eff = SO_LL + SO_LS * XXX + XXX.transpose() * SO_SL + XXX.transpose() * SO_SS * XXX;
    return coeff_con.transpose() * (RRR.transpose() * SO_2c_eff * RRR) * coeff_con;
}


Matrix<MatrixXcd,3,1> DHF::get_amfi_Pauli(const MatrixXd& coeff_4c, const MatrixXd& h2eSSLL_SD, const MatrixXd& h2eSSSS_SD, const MatrixXd& h1e_4c_, const MatrixXd& overlap_4c_, const int& nocc, const MatrixXd& coeff_con, const bool& spherical)
{
    MatrixXd amfi2c = get_amfi(coeff_4c, h2eSSLL_SD, h2eSSSS_SD, h1e_4c_, overlap_4c_, nocc, coeff_con, spherical);
    return get_amfi_Pauli(amfi2c);
}

Matrix<MatrixXcd,3,1> DHF::get_amfi_Pauli(const MatrixXd& amfi_2c)
{
    cout << "get_amfi_Pauli is not implemented yet!" << endl;
    exit(99);
}