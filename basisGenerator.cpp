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
string atomName, basisSetPVXZ, basisSetAll, flags, jobs;

/* Read input file and set global variables */
void readInput(const string filename);

int main()
{
    readInput("input");
    INT_SPH intor(atomName, basisSetPVXZ);
    INT_SPH intorAll(atomName, basisSetAll);
    DHF_SPH_CA sfx2c(intor,"input",true,true,false,false,true,false);
    sfx2c.convControl = 1e-9;
    sfx2c.runSCF(true,false);
    sfx2c.basisGenerator(basisSetAll, "GENBAS2", intor, intorAll);

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
        ifs >> basisSetPVXZ >> flags;
        ifs >> basisSetAll >> flags;
        atomName = removeSpaces(atomName);
        basisSetPVXZ = removeSpaces(basisSetPVXZ);
        basisSetAll = removeSpaces(basisSetAll);
        cout << atomName << endl << basisSetPVXZ << endl << jobs <<endl;
    ifs.close();
}


