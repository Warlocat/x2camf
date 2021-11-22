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
    bool twoC = false, spinFree = false, withGaunt = true, withgauge = false, gaussian_nuc = false;

    DHF_SPH scf4c(intor,"input",spinFree,twoC,withGaunt,withgauge,true,gaussian_nuc);
    scf4c.runSCF(twoC,false);
    auto ene_4c = scf4c.ene_orb;
    auto amfi = scf4c.get_amfi_unc(intor,false);
    DHF_SPH scf2c(intor,"input",true,true,false,false,true,false);
    scf2c.runSCF(true,false);
    auto ene_sfx2c = scf2c.ene_orb;
    DHF_SPH scf2c_pc(intor,"input",true,true,false,false,true,false);
    auto h1e = scf2c.get_h1e_4c();
    for(int ir = 0; ir < h1e.rows(); ir++)
    {
        h1e(ir) = h1e(ir) + amfi(ir);
    }
    scf2c_pc.set_h1e_4c(h1e);
    scf2c_pc.runSCF(true,false);
    auto ene_pc = scf2c_pc.ene_orb;
    cout << ene_4c(0)(0+ene_4c(0).rows()/2) << endl;
    cout << ene_sfx2c(0)(0) << endl;
    cout << ene_pc(0)(0) << endl;

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

