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
        amfi = (*dhf_test).get_amfi_unc(intor,"h1e");
    }
    for(int ir = 0; ir < amfi.rows(); ir++)
        cout << amfi(ir) << endl;

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

