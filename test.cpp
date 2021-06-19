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
    readInput("input");
    INT_SPH intor(atomName, basisSet);
    vMatrixXd overlap_2c = intor.get_h1e("overlap");
    vMatrixXd s_h_i_2c(overlap_2c.rows());
    for(int ii = 0; ii < overlap_2c.rows(); ii++)
    {
        s_h_i_2c(ii) = matrix_half_inverse(overlap_2c(ii));
    }

    DHF_SPH scf4c(intor,"ZMAT",true,false,false);
    scf4c.runSCF(false);
    vMatrixXd fock_pcc = scf4c.x2c2ePCC();

    DHF_SPH scf2c(intor,"ZMAT",true,true,false);
    vMatrixXd h1e_2c = scf2c.get_h1e_4c();

    for(int ir = 0; ir < h1e_2c.rows(); ir++)
    {
        h1e_2c(ir) = h1e_2c(ir) + fock_pcc(ir);
    }
    scf2c.set_h1e_4c(h1e_2c);
    scf2c.runSCF(true);

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

