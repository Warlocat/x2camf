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

    DHF_SPH sf4c(intor,"ZMAT",true,false,false);
    sf4c.runSCF();
    auto fock_sf4c = sf4c.get_fock_4c();
    auto overlap_4c = sf4c.get_overlap_4c();
    auto coeff_sf4c = sf4c.coeff;
    int size_irrep = fock_sf4c.rows();
    
    DHF_SPH sf2c(intor,"ZMAT",true,true,false);
    sf2c.runSCF(true);
    auto h1e_sf2c = sf2c.get_h1e_4c();
    auto fock_sf2c = sf2c.get_fock_4c();
    vMatrixXd fock_pc(size_irrep);
    vMatrixXd fock_2c_from_4c(size_irrep);
    vMatrixXd h1e_2c_pc(size_irrep);
    for(int ir = 0; ir < size_irrep; ir++)
    {
        MatrixXd XXX = X2C::get_X(coeff_sf4c(ir));
        MatrixXd RRR = X2C::get_R(overlap_4c(ir),XXX);
        fock_2c_from_4c(ir) = X2C::transform_4c_2c(fock_sf4c(ir), XXX, RRR);
        fock_pc(ir) = fock_2c_from_4c(ir) - fock_sf2c(ir);
        h1e_2c_pc(ir) = h1e_sf2c(ir) + fock_pc(ir);
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

