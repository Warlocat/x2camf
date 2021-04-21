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
    readInput("input");
    INT_SPH intor(atomName, basisSet);
    DHF_SPH* dhf_test;
    if(jobs == "DHF")
        dhf_test = new DHF_SPH(intor,"input");
    else if(jobs == "SFDHF")
        dhf_test = new DHF_SPH(intor,"input",true);
    else if(jobs == "SFX2C1E")
        dhf_test = new DHF_SPH(intor,"input",true,true);
    else
    {
        cout << "Wrong Jobs!" << endl;
        exit(99);
    }
    vMatrixXd amfi;
    if(jobs == "SFX2C1E")
    {
        (*dhf_test).runSCF_2c();
        amfi = (*dhf_test).get_amfi_unc_2c(intor);
    }
    else
    {
        (*dhf_test).runSCF();
        amfi = (*dhf_test).get_amfi_unc(intor,"partialFock");
    }
    for(int ir = 0; ir < amfi.rows(); ir++)
        cout << amfi(ir) << endl;
    MatrixXd amfi2 = dhf_test->unite_irrep(amfi,intor.irrep_list);
    MatrixXd M1 = dhf_test->jspinor2sph(intor.irrep_list);
    MatrixXcd M2 = dhf_test->sph2solid(intor.irrep_list);
    amfi2 = M1.adjoint()*amfi2*M1;
    MatrixXcd amfi3 = M2.adjoint()*amfi2*M2;
    for(int ii = 0; ii < amfi3.rows(); ii++)
    for(int jj = 0; jj < amfi3.cols(); jj++)
        // if(abs(amfi2(ii,jj)) > 1e-8)
            cout << ii << "\t" << jj << "\t" << amfi3(ii,jj) << endl;

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

