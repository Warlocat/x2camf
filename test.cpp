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
    bool twoC = false, spinFree = false, withGaunt = true, withgauge = true, gaussian_nuc = false;

    // DHF_SPH scf4c(intor,"input",spinFree,twoC,withGaunt,withgauge,true,gaussian_nuc);
    // scf4c.convControl = 1e-10;
    // scf4c.runSCF(twoC,false);
    int2eJK gaunt,gauge,gauge_compact;
    gauge_compact = intor.get_h2e_JK_gauge_compact("LSLS",-1); 
    gaunt = intor.get_h2e_JK_gaunt("LSLS",-1);
    gauge = intor.get_h2e_JK_gauge("LSLS",-1);
     cout << gauge.J[0][0][0][0] << "\t" << gauge.J[0][1][0][0] << "\t" << gauge.J[1][0][0][0] << "\t" << gauge.J[1][1][0][0] << endl;
     cout << gauge.K[0][0][0][0] << "\t" << gauge.K[0][1][0][0] << "\t" << gauge.K[1][0][0][0] << "\t" << gauge.K[1][1][0][0] << endl;
     cout << gauge_compact.J[0][0][0][0]<<endl;
     cout << gauge_compact.K[0][0][0][0]<<endl;
     exit(99);
    int size = intor.irrep_list(0).size;
    cout << size << endl;
    for(int ii = 0; ii < size; ii++)
    for(int jj = 0; jj < size; jj++)
    for(int kk = 0; kk < size; kk++)
    for(int ll = 0; ll < size; ll++)
    {
        int e1 = ii*size+jj, e2 = kk*size+ll;
        cout << gaunt.J[0][0][e1][e2] << "\t" << gauge.J[0][0][e1][e2] << "\t" << gaunt.J[0][1][e1][e2] << "\t" << gauge.J[0][1][e1][e2] << endl;
    }
    

    // auto ene_4c = scf4c.ene_orb;
    // auto amfi = scf4c.get_amfi_unc(intor,false);
    // DHF_SPH scf2c(intor,"input",true,true,false,false,true,false);
    // scf2c.runSCF(true,false);
    // auto ene_sfx2c = scf2c.ene_orb;
    // DHF_SPH scf2c_pc(intor,"input",true,true,false,false,true,false);
    // auto h1e = scf2c.get_h1e_4c();
    // for(int ir = 0; ir < h1e.rows(); ir++)
    // {
    //     h1e(ir) = h1e(ir) + amfi(ir);
    // }
    // scf2c_pc.set_h1e_4c(h1e);
    // scf2c_pc.runSCF(true,false);
    // auto ene_pc = scf2c_pc.ene_orb;
    // cout << ene_4c(0)(0+ene_4c(0).rows()/2) << endl;
    // cout << ene_sfx2c(0)(0) << endl;
    // cout << ene_pc(0)(0) << endl;

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

