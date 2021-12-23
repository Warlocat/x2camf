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
    //                          sf   2c   Gaunt gauge allint  gauNuc
    DHF_SPH_CA sfx2c(intor,"input",true,true,false,false,true,false);
    // DHF_SPH sf4c(intor,"input",true,false,false,false,true,false);
    // DHF_SPH dc4c(intor,"input",false,false,false,false,true,false);
    // DHF_SPH dcg4c(intor,"input",false,false,true,false,true,false);
    // DHF_SPH dcb4c(intor,"input",false,false,true,true,true,false);
    sfx2c.convControl = 1e-10;
    //  sf4c.convControl = 1e-10;
    //  dc4c.convControl = 1e-10;
    // dcg4c.convControl = 1e-10;
    // dcb4c.convControl = 1e-10;
    sfx2c.runSCF(true,false);
    auto coeff = sfx2c.coeff;
    for(int ir = 0; ir < coeff.rows(); ir += 2+4*intor.irrep_list(ir).l)
    {
        cout << coeff(ir) << endl << endl;
    }
    //  sf4c.runSCF(false,false);
    //  dc4c.runSCF(false,false);
    // dcg4c.runSCF(false,false);
    // dcb4c.runSCF(false,false);
    // 
    // int size = dc4c.ene_orb(0).rows();
    // VectorXd ene_1s(5), de_1s(4);
    // ene_1s << sfx2c.ene_orb(0)(0),
    //            sf4c.ene_orb(0)(0+size/2),
    //            dc4c.ene_orb(0)(0+size/2),
    //           dcg4c.ene_orb(0)(0+size/2),
    //           dcb4c.ene_orb(0)(0+size/2);
    // for(int ii = 0; ii < 4; ii++)
    // {
    //     de_1s(ii) = (ene_1s(ii+1) - ene_1s(ii))*27.21138;
    // }

    // cout << ene_1s << endl << de_1s << endl;

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

