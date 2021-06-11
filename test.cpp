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
    bool twoC = false;
    readInput("input");
    INT_SPH intor(atomName, basisSet);
    // auto gaunt = intor.get_h2e_JK_gaunt("LSLS");
    // int ir1 = 2, ir2 = 2;
    // int size1 = intor.irrep_list(ir1).size, size2 = intor.irrep_list(ir2).size;
    // for(int ii = 0; ii < size1; ii++)
    // for(int jj = 0; jj < size1; jj++)
    // for(int kk = 0; kk < size2; kk++)
    // for(int ll = 0; ll < size2; ll++)
    // {
    //     cout << ii << "\t" << jj << "\t" << kk << "\t" << ll << "\t" << gaunt.K(ir1,ir2)(ii*size1+jj, kk*size2+ll) * -4.0*pow(speedOfLight,2) << endl;
    // }
    // DHF_SPH dhf(intor,"ZMAT",false,false,true);
    // dhf.runSCF();

    DHF_SPH_CA dhf_ca(intor,"ZMAT",false,false,false);
    dhf_ca.runSCF();
    auto amfi = dhf_ca.get_amfi_unc(intor,false);
    for(int ii = 0; ii < amfi.rows(); ii++)
    {
        cout << amfi(ii) << endl;
    }
    return 0;
}



void readInput(const string filename)
{
    ifstream ifs;
    ifs.open(filename);
        ifs >> atomName >> flags;
        ifs >> basisSet >> flags;
        ifs >> jobs >> flags;
        cout << atomName << endl << basisSet << endl << jobs <<endl;
    ifs.close();
}

