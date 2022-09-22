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
    readInput("input");
    INT_SPH intor(atomName, basisSet);
    //                                             sf   2c   Gaunt gauge allint  gauNuc
    DHF_SPH *scfer = new DHF_SPH_CA(intor,"input",false,true,false,false,true,false);
    DHF_SPH *scfer2 = new DHF_SPH_CA2(intor,"input",false,false,false,false,true,false);
    scfer2->convControl = 1e-9;
    scfer2->runSCF(false,false);
    auto h1e = scfer->get_h1e_4c(), amfso = scfer2->get_amfi_unc(intor,false);
    for(int ir = 0; ir < h1e.rows(); ir++)
    {
        h1e(ir) += amfso(ir);
        cout << amfso(ir) << endl << endl;
    }
    scfer->set_h1e_4c(h1e);

    scfer->convControl = 1e-9;
    scfer->runSCF(true,false);
    scfer->basisGenerator(basisSet, "out", intor, intor, false, "SO");

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

