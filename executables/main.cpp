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
    bool spinFree = false, twoC = false, Gaunt = true, gauge = true, allint = true, gauNuc = false, aoc = false, renormS = false;
    readInput("input");
    INT_SPH intor(atomName, basisSet);
    DHF_SPH* scfer;
    if(aoc)
    {
        scfer = new DHF_SPH_CA(intor,"input",spinFree,twoC,Gaunt,gauge,allint,gauNuc);
    }
    else
    {
        scfer = new DHF_SPH(intor,"input",spinFree,twoC,Gaunt,gauge,allint,gauNuc);
    }
    scfer->convControl = 1e-10;
    scfer->runSCF(twoC,renormS);

    /*
        amfi_all contains all the AMFSO integrals in the j-spinor basis
        The transformations to spherical harmonics and real spherical harmonics basis 
        can be found in namespace Rotate.
    */
    vMatrixXd amfi = scfer->get_amfi_unc(intor,twoC);
    MatrixXd amfi_all = Rotate::unite_irrep(amfi,intor.irrep_list);
    for(int ii = 0; ii < amfi_all.rows(); ii++)
    for(int jj = 0; jj < amfi_all.cols(); jj++)
        cout << ii << "\t" << jj << "\t" << amfi_all(ii,jj) << endl;

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

