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
bool unc;

/* Read input file and set global variables */
void readInput(const string filename);

int main()
{
    bool twoC = false, spinFree = true, finiteNuc = true;
    readInput("input");
    INT_SPH intor(atomName, basisSet);    
    DHF_SPH *scfer = new DHF_SPH(intor,"input",spinFree,twoC,false,false,true,finiteNuc);
    scfer->runSCF(twoC,true);

    double tmp = 0.0;
    for(int ii=0;ii<50000;ii++)
    {
        double dx = 0.0001;
        double rr = ii*dx;
        tmp += dx*scfer->radialDensity(rr)*rr*rr;
        // cout << dx*scfer->radialDensity(rr)*rr*rr << endl;
    }
    cout << tmp << endl;

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

