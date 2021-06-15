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
    readInput("input");
    INT_SPH intor(atomName, basisSet);

    DHF_SPH scf4c(intor,"ZMAT",false,false,false);
    scf4c.runSCF(false);
    vMatrixXd overlap_4c = scf4c.get_overlap_4c();
    vMatrixXd fock2e_4c = scf4c.get_fock_4c();
    vMatrixXd coeff_4c = scf4c.coeff;
    // auto amfi = gaunt4c.get_amfi_unc(intor);
    
    DHF_SPH scf2c(intor,"ZMAT",true,true,false);
    scf2c.runSCF(true);
    auto h1e_scf2c = scf2c.get_h1e_4c();
    int size_irrep = h1e_scf2c.rows();
    auto fock2e_scf2c = scf2c.get_fock_4c();
    vMatrixXd fock_pc(size_irrep);
    vMatrixXd fock2e_2c_from_4c(size_irrep);
    vMatrixXd h1e_2c_pc(size_irrep);
    for(int ir = 0; ir < size_irrep; ir++)
    {
        MatrixXd XXX = X2C::get_X(coeff_4c(ir));
        MatrixXd RRR = X2C::get_R(overlap_4c(ir),XXX);
        fock2e_2c_from_4c(ir) = X2C::transform_4c_2c(fock2e_4c(ir), XXX, RRR);
        fock_pc(ir) = fock2e_2c_from_4c(ir) - fock2e_scf2c(ir);
        h1e_2c_pc(ir) = h1e_scf2c(ir) + fock_pc(ir);
    }

    DHF_SPH pc2c(intor,"ZMAT",true,true,false);
    pc2c.set_h1e_4c(h1e_2c_pc);
    pc2c.runSCF(true);
    
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

