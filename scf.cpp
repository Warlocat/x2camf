#include"scf.h"
#include"mkl_interface.h"
#include<omp.h>
#include<iostream>
#include<iomanip>
#include<fstream>


/*
    Eigen solver from cfour
*/

// void eig_cfour_symm(const MatrixXd inputM, VectorXd& values, MatrixXd& vectors)
// {
//     int n = inputM.rows(), n1 = 0;
//     values.resize(n);
//     vectors.resize(n,n);
//     values = VectorXd::Zero(n);
//     vectors = MatrixXd::Zero(n,n);
//     double A[n*n], B[n*n];
//     for(int ii = 0; ii < n; ii++)
//     for(int jj = 0; jj < n; jj++)
//     {   
//         A[ii*n+jj] = inputM(ii,jj);
//     }
//     eig_(A, B, &n, &n1);
//     for(int ii = 0; ii < n; ii++)
//     {
//         values(ii) = A[ii*n+ii];
//         for(int jj = 0; jj < n ;jj++)
//             vectors(jj,ii) = B[ii*n+jj];
//     }
//     return;
// }






SCF* scf_init(const GTO& gto_, const string& h2e_file, const string& relativistic)
{
    SCF* ptr_scf;
    if(gto_.nelec_a == gto_.nelec_b)
    {
        cout << "RHF program is used." << endl << endl;
        ptr_scf = new RHF(gto_, h2e_file, relativistic);
    }
    else
    {
        cout << "UHF program is used." << endl << endl;
        ptr_scf = new UHF(gto_, h2e_file, relativistic);
    }
    return ptr_scf;
}
SCF* scf_init(const GTO& gto_, const MatrixXd& h2e_, const string& relativistic)
{
    SCF* ptr_scf;
    if(gto_.nelec_a == gto_.nelec_b)
    {
        cout << "RHF program is used." << endl << endl;
        ptr_scf = new RHF(gto_, h2e_, relativistic);
    }
    else
    {
        cout << "UHF program is used." << endl << endl;
        ptr_scf = new UHF(gto_, h2e_, relativistic);
    }
    return ptr_scf;
}


/*********************************************************/
/**********    Member functions of class SCF    **********/
/*********************************************************/


SCF::SCF(const GTO& gto_, const string& h2e_file, const string& relativistic):
nelec_a(gto_.nelec_a), nelec_b(gto_.nelec_b), size_basis(gto_.size_gtoc)
{
    overlap = gto_.get_h1e("overlap");
    if(relativistic == "off")
        h1e = gto_.get_h1e("h1e");
    else if(relativistic == "sfx2c1e")
        h1e = X2C::evaluate_h1e_x2c(gto_.get_h1e("overlap", true), gto_.get_h1e("kinetic", true), gto_.get_h1e("p.Vp", true), gto_.get_h1e("nuc_attra", true), gto_.get_coeff_contraction());
    else
    {
        cout << "ERROR: UNSUPPORTED relativistic method used!" << endl;
        exit(99);
    }
    

    h2e.resize(size_basis * (size_basis + 1) / 2, size_basis * (size_basis + 1) / 2); 
    for(int ii = 0; ii < size_basis * (size_basis + 1) / 2; ii++)
    for(int jj = 0; jj < size_basis * (size_basis + 1) / 2; jj++)
        h2e = h2e * 0.0;

    readIntegrals(h2e_file);
    overlap_half_i = matrix_half_inverse(overlap);
}
SCF::SCF(const GTO& gto_, const MatrixXd& h2e_, const string& relativistic):
nelec_a(gto_.nelec_a), nelec_b(gto_.nelec_b), size_basis(gto_.size_gtoc), h2e(h2e_)
{
    overlap = gto_.get_h1e("overlap");
    if(relativistic == "off")
        h1e = gto_.get_h1e("h1e");
    else if(relativistic == "sfx2c1e")
        h1e = X2C::evaluate_h1e_x2c(gto_.get_h1e("overlap", true), gto_.get_h1e("kinetic", true), gto_.get_h1e("p.Vp", true), gto_.get_h1e("nuc_attra", true), gto_.get_coeff_contraction());
    else
    {
        cout << "ERROR: UNSUPPORTED relativistic method used!" << endl;
        exit(99);
    }
    overlap_half_i = matrix_half_inverse(overlap);
}

SCF::SCF()
{
    cout << "Special constructor of SCF class was used.\nIt should be only used in DHF\n";
}

SCF::~SCF()
{
}


void SCF::readIntegrals(const string& filename)
{
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
                int ij = (indices(0) - 1) * indices(0) / 2 + indices(1) - 1, kl = (indices(2) - 1) * indices(2) / 2 + indices(3) - 1;
                h2e(ij,kl) = tmp;
                h2e(kl,ij) = tmp;
            }
        }            
    ifs.close();
}

double SCF::evaluateChange(const MatrixXd& M1, const MatrixXd& M2)
{
    int size = M1.rows();
    double tmp = 0.0;
    for(int ii = 0; ii < size; ii++)
    for(int jj = 0; jj < size; jj++)
        tmp += pow((M1(ii,jj) - M2(ii,jj)),2);

    return sqrt(tmp);
}


MatrixXd SCF::matrix_half_inverse(const MatrixXd& inputM)
{
    int size = inputM.rows();
    SelfAdjointEigenSolver<MatrixXd> solver(inputM);
    VectorXd eigenvalues = solver.eigenvalues();
    MatrixXd eigenvectors = solver.eigenvectors();

    // int size = inputM.rows();
    // MatrixXd eigenvectors;
    // VectorXd eigenvalues;
    // eig_cfour_symm(inputM, eigenvalues, eigenvectors);
    
    for(int ii = 0; ii < size; ii++)
    {
        if(eigenvalues(ii) < 0)
        {
            cout << "ERROR: Matrix has negative eigenvalues!" << endl;
            exit(99);
        }
        else
        {
            eigenvalues(ii) = 1.0 / sqrt(eigenvalues(ii));
        }
    }
    MatrixXd tmp(size, size);
    for(int ii = 0; ii < size; ii++)
    for(int jj = 0; jj < size; jj++)
    {
        tmp(ii,jj) = 0.0;
        for(int kk = 0; kk < size; kk++)
            tmp(ii,jj) += eigenvectors(ii,kk) * eigenvalues(kk) * eigenvectors(jj,kk);
    }

    return tmp; 
}

MatrixXd SCF::matrix_half(const MatrixXd& inputM)
{
    int size = inputM.rows();
    SelfAdjointEigenSolver<MatrixXd> solver(inputM);
    VectorXd eigenvalues = solver.eigenvalues();
    MatrixXd eigenvectors = solver.eigenvectors();

    // int size = inputM.rows();
    // MatrixXd eigenvectors;
    // VectorXd eigenvalues;
    // eig_cfour_symm(inputM, eigenvalues, eigenvectors);
    
    for(int ii = 0; ii < size; ii++)
    {
        if(eigenvalues(ii) < 0)
        {
            cout << "ERROR: Matrix has negative eigenvalues!" << endl;
            exit(99);
        }
        else
        {
            eigenvalues(ii) = sqrt(eigenvalues(ii));
        }
    }
    MatrixXd tmp(size, size);
    for(int ii = 0; ii < size; ii++)
    for(int jj = 0; jj < size; jj++)
    {
        tmp(ii,jj) = 0.0;
        for(int kk = 0; kk < size; kk++)
            tmp(ii,jj) += eigenvectors(ii,kk) * eigenvalues(kk) * eigenvectors(jj,kk);
    }

    return tmp;
}

MatrixXd SCF::evaluateDensity(const MatrixXd& coeff_, const int& occ, const bool& spherical)
{
    int size = coeff_.rows();
    MatrixXd den(size, size);
    if(!spherical)
    {
        for(int aa = 0; aa < size; aa++)
        for(int bb = 0; bb < size; bb++)
        {
            den(aa,bb) = 0.0;
            for(int ii = 0; ii < occ; ii++)
                den(aa,bb) += coeff_(aa,ii) * coeff_(bb,ii);
        }
    }
    else
    {
        cout << "ERROR: Spherical occupation is NOT supported now!" << endl;
        exit(99);
    }
    
    
    return den;
}


void SCF::eigensolverG(const MatrixXd& inputM, const MatrixXd& s_h_i, VectorXd& values, MatrixXd& vectors)
{
    MatrixXd tmp = s_h_i * inputM * s_h_i;
    
    
    SelfAdjointEigenSolver<MatrixXd> solver(tmp);
    values = solver.eigenvalues();
    vectors = s_h_i * solver.eigenvectors();

    // eig_cfour_symm(tmp, values, vectors);
    // vectors = s_h_i * vectors;

    return;
}




/*********************************************************/
/**********    Member functions of class RHF    **********/
/*********************************************************/

RHF::RHF(const GTO& gto_, const string& h2e_file, const string& relativistic):
SCF(gto_, h2e_file, relativistic)
{
}
RHF::RHF(const GTO& gto_, const MatrixXd& h2e_, const string& relativistic):
SCF(gto_, h2e_, relativistic)
{
}

RHF::~RHF()
{
}

void RHF::runSCF()
{
    fock.resize(size_basis,size_basis);

    MatrixXd newDen;
    // GeneralizedSelfAdjointEigenSolver<MatrixXd> gsolver(h1e, overlap);
    // ene_orb = gsolver.eigenvalues();
    // coeff = gsolver.eigenvectors();
    eigensolverG(h1e, overlap_half_i, ene_orb, coeff);
    density = evaluateDensity(coeff, nelec_a);

    for(int iter = 1; iter <= maxIter; iter++)
    {
        #pragma omp parallel  for
        for(int mm = 0; mm < size_basis; mm++)
        for(int nn = 0; nn < size_basis; nn++)
        {
            fock(mm,nn) = h1e(mm,nn);
            for(int aa = 0; aa < size_basis; aa++)
            for(int bb = 0; bb < size_basis; bb++)
            {
                int ab, mn, an, mb;
                ab = max(aa,bb)*(max(aa,bb)+1)/2 + min(aa,bb);
                mn = max(mm,nn)*(max(mm,nn)+1)/2 + min(mm,nn);
                an = max(aa,nn)*(max(aa,nn)+1)/2 + min(aa,nn);
                mb = max(mm,bb)*(max(mm,bb)+1)/2 + min(mm,bb);
                fock(mm,nn) += density(aa,bb) * (2.0 * h2e(ab,mn) - h2e(an, mb));
            }
        }
        // GeneralizedSelfAdjointEigenSolver<MatrixXd> gsolver(fock, overlap);
        // ene_orb = gsolver.eigenvalues();
        // coeff = gsolver.eigenvectors();
        eigensolverG(fock, overlap_half_i, ene_orb, coeff);
        newDen = evaluateDensity(coeff, nelec_a);
        d_density = evaluateChange(density, newDen);
        cout << "Iter #" << iter << " maximum density difference: " << d_density << endl;
        
        density = newDen;
        if(d_density < convControl) 
        {
            converged = true;
            cout << endl << "RHF converges after " << iter << " iterations." << endl << endl;

            cout << "\tOrbital\t\tEnergy(in hartree)\n";
            cout << "\t*******\t\t******************\n";
            for(int ii = 1; ii <= size_basis; ii++)
            {
                cout << "\t" << ii << "\t\t" << setprecision(15) << ene_orb(ii - 1) << endl;
            }

            ene_scf = 0.0;
            for(int ii = 0; ii < size_basis; ii++)
            for(int jj = 0; jj < size_basis; jj++)
            {
                ene_scf += density(ii,jj) * (h1e(jj,ii) + fock(jj,ii));
            }
            cout << "Final RHF energy is " << setprecision(15) << ene_scf << " hartree." << endl;
            break;            
        }           
    }
}








/*********************************************************/
/**********    Member functions of class UHF    **********/
/*********************************************************/

UHF::UHF(const GTO& gto_, const string& h2e_file, const string& relativistic):
SCF(gto_, h2e_file, relativistic)
{
}
UHF::UHF(const GTO& gto_, const MatrixXd& h2e_, const string& relativistic):
SCF(gto_, h2e_, relativistic)
{
}

UHF::~UHF()
{
}

void UHF::runSCF()
{
    fock_a.resize(size_basis, size_basis);
    fock_b.resize(size_basis, size_basis);
    MatrixXd newDen_a, newDen_b, density_total;
    
    eigensolverG(h1e, overlap_half_i, ene_orb_a, coeff_a);
    density_a = evaluateDensity(coeff_a, nelec_a);
    density_b = evaluateDensity(coeff_a, nelec_b);
    density_total = density_a + density_b;

    for(int iter = 1; iter <= maxIter; iter++)
    {
        #pragma omp parallel  for
        for(int mm = 0; mm < size_basis; mm++)
        for(int nn = 0; nn < size_basis; nn++)
        {
            fock_a(mm,nn) = h1e(mm,nn);
            fock_b(mm,nn) = h1e(mm,nn);
            for(int aa = 0; aa < size_basis; aa++)
            for(int bb = 0; bb < size_basis; bb++)
            {
                int ab, mn, an, mb;
                ab = max(aa,bb)*(max(aa,bb)+1)/2 + min(aa,bb);
                mn = max(mm,nn)*(max(mm,nn)+1)/2 + min(mm,nn);
                an = max(aa,nn)*(max(aa,nn)+1)/2 + min(aa,nn);
                mb = max(mm,bb)*(max(mm,bb)+1)/2 + min(mm,bb);
                fock_a(mm,nn) += density_total(aa,bb) * h2e(ab,mn) - density_a(aa,bb) * h2e(an, mb);
                fock_b(mm,nn) += density_total(aa,bb) * h2e(ab,mn) - density_b(aa,bb) * h2e(an, mb);
            }
        }
        eigensolverG(fock_a, overlap_half_i, ene_orb_a, coeff_a);
        eigensolverG(fock_b, overlap_half_i, ene_orb_b, coeff_b);
        newDen_a = evaluateDensity(coeff_a, nelec_a);
        newDen_b = evaluateDensity(coeff_b, nelec_b);

        d_density_a = evaluateChange(density_a, newDen_a);
        d_density_b = evaluateChange(density_b, newDen_b);
        cout << "Iter #" << iter << " maximum density difference :\talpha " << d_density_a << "\t\tbeta " << d_density_b << endl;
            
        // density_a = 0.5 * (newDen_a + density_a);
        // density_b = 0.5 * (newDen_b + density_b);
        density_a = newDen_a;
        density_b = newDen_b;
        density_total = density_a + density_b;
        if(max(d_density_a,d_density_b) < convControl) 
        {
            converged = true;
            cout << endl << "UHF converges after " << iter << " iterations." << endl << endl;

            cout << "\tOrbital\t\tEnergy_a(in hartree)\t\tEnergy_b(in hartree)\n";
            cout << "\t*******\t\t********************\t\t********************\n";
            for(int ii = 1; ii <= size_basis; ii++)
            {
                cout << "\t" << ii << "\t\t" << setprecision(15) << ene_orb_a(ii - 1) << "\t\t" << ene_orb_b(ii - 1) << endl;
            }

            ene_scf = 0.0;
            for(int ii = 0; ii < size_basis; ii++)
            for(int jj = 0; jj < size_basis; jj++)
            {
                ene_scf += 0.5 * (density_total(ii,jj) * h1e(jj,ii) + density_a(ii,jj) * fock_a(jj,ii) + density_b(ii,jj) * fock_b(jj,ii));
            }
            cout << "Final UHF energy is " << setprecision(15) << ene_scf << " hartree." << endl;
            break;            
        }           
    }
}





/*********************************************************/
/**********    Member functions of class DHF    **********/
/*********************************************************/

DHF::DHF(const GTO_SPINOR& gto_, const string& h2e_file, const bool& unc)
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
    for(int ii = 0; ii < size_basis; ii++)
    {
        norm_s(ii) = sqrt(kinetic(ii,ii) / 2.0 / speedOfLight / speedOfLight);
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
    fock_4c.resize(2*size_basis,2*size_basis);

    MatrixXd newDen;
    eigensolverG(h1e_4c, overlap_half_i_4c, ene_orb, coeff);
    density = evaluateDensity_spinor(coeff, nelec_a+nelec_b);

    for(int iter = 1; iter <= maxIter; iter++)
    {
        fock_4c = h1e_4c;
        for(int mm = 0; mm < size_basis; mm++)
        for(int nn = 0; nn < size_basis; nn++)
        {
            for(int ss = 0; ss < size_basis; ss++)
            for(int rr = 0; rr < size_basis; rr++)
            {
                int emn = mm*size_basis+nn, esr = ss*size_basis+rr, emr = mm*size_basis+rr, esn = ss*size_basis+nn;
                fock_4c(mm,nn) += density(ss,rr) * (h2e(emn,esr) - h2e(emr,esn)) 
                                + density(size_basis+ss,size_basis+rr) * h2eSSLL(esr,emn);
                fock_4c(mm,nn+size_basis) -= density(size_basis+ss,rr) * h2eSSLL(esn,emr);
                fock_4c(mm+size_basis,nn) -= density(ss,size_basis+rr) * h2eSSLL(emr,esn);
                fock_4c(mm+size_basis,nn+size_basis) += density(size_basis+ss,size_basis+rr) * (h2eSSSS(emn,esr) - h2eSSSS(emr,esn)) + density(ss,rr) * h2eSSLL(emn,esr);
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
