#include<iostream>
#include<fstream>
#include<string>
#include<Eigen/Dense>
#include<iomanip>
#include<cmath>
#include<ctime>
#include<memory>
#include"int_sph.h"
#include"dhf_sph.h"
#include"general.h"
#include"dhf_sph_ca.h"
#include"dhf_sph_ca2.h"
using namespace Eigen;
using namespace std;

/* Global information */
int charge, spin;
string atomName, basisSet, flags, jobs;
VectorXd occ;
bool unc;

/* Read input file and set global variables */
void readInput(const string filename);

int main()
{
    bool twoC = true, spinFree = true;
    readInput("input");
    INT_SPH intor(atomName, basisSet);
    //                                             sf   2c   Gaunt gauge allint  gauNuc
    DHF_SPH *scfer_new = new DHF_SPH_CA(intor,"input",spinFree,twoC,false,false,true,false);
    DHF_SPH *scfer_old = new DHF_SPH_CA2(intor,"input",spinFree,twoC,false,false,true,false);
    scfer_old->convControl = 1e-9;
    scfer_old->runSCF(twoC,false);
    scfer_new->convControl = 1e-9;
    scfer_new->runSCF(twoC,false);

    vMatrixXd Dcoeff(scfer_new->coeff.rows());
    for(int ir = 0; ir < Dcoeff.rows(); ir+=scfer_new->irrep_list(ir).two_j+1)
    {
        Dcoeff(ir).resize(scfer_new->coeff(ir).rows(),scfer_new->coeff(ir).cols());
        for(int ii = 0; ii < scfer_new->coeff(ir).cols(); ii++)
        for(int jj = 0; jj < scfer_new->coeff(ir).rows(); jj++)
        {
            // Dcoeff(ir)(jj,ii) = scfer_new->coeff(ir)(jj,ii) +  scfer_old->coeff(ir)(jj,ii);
            Dcoeff(ir)(jj,ii) = scfer_new->coeff(ir)(jj,ii) + ((scfer_new->coeff(ir)(0,ii)*scfer_old->coeff(ir)(0,ii) > 0) ? -1.0 : 1.0 ) * scfer_old->coeff(ir)(jj,ii);
        }
        
        // cout << fixed << setprecision(8) << Dcoeff(ir) << endl << endl;
        cout << fixed << setprecision(8) << scfer_new->coeff(ir) << endl << endl;
        cout << fixed << setprecision(8) << scfer_old->coeff(ir) << endl << endl;
    }
    // auto h1e = scfer->get_h1e_4c(), amfso = scfer2->get_amfi_unc(intor,false);
    // for(int ir = 0; ir < h1e.rows(); ir++)
    // {
    //     h1e(ir) += amfso(ir);
    //     cout << amfso(ir) << endl << endl;
    // }
    // scfer->set_h1e_4c(h1e);

    // scfer->convControl = 1e-9;
    // scfer->runSCF(true,false);
    // scfer->basisGenerator(basisSet, "out", intor, intor, false, "SO");

    return 0;
}



void readInput(const string filename)
{
    ifstream ifs;
    ifs.open(filename);
    if(!ifs)
    {
        cout << "ERROR opening file " << filename << endl;
        exit(99);
    }
        ifs >> atomName >> flags;
        ifs >> basisSet >> flags;
        ifs >> jobs >> flags;
        cout << atomName << endl << basisSet << endl << jobs <<endl;
    ifs.close();
}

