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
    DHF_SPH_CA dhf_test(intor,"input",true,true);
    dhf_test.runSCF_separate(true);
    //vMatrixXd amfi = dhf_test.get_amfi_unc_ca(intor,true);
    DHF_SPH_CA dhf_test2(intor,"input",false,false);
    dhf_test2.runSCF_separate(false);
    //vMatrixXd amfi = dhf_test2.get_amfi_unc_ca(intor,false);
    //for(int ir = 0; ir < amfi.rows(); ir++)
    //    cout << amfi(ir) << endl;
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

