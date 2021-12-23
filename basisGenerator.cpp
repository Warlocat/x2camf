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

/* Read input file and set global variables */
void readInput(const string filename);
string removeSpaces(const string& flags);
void basisGenerator(const string& basisName, const DHF_SPH_CA& scf);

int main()
{
    readInput("input");
    INT_SPH intor(atomName, basisSet);
    DHF_SPH_CA sfx2c(intor,"input",true,true,false,false,true,false);
    sfx2c.convControl = 1e-11;
    sfx2c.runSCF(true,false);
    sfx2c.basisGenerator(basisSet, intor);

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
        atomName = removeSpaces(atomName);
        basisSet = removeSpaces(basisSet);
        cout << atomName << endl << basisSet << endl << jobs <<endl;
    ifs.close();
}

string removeSpaces(const string& flags)
{
    string tmp_s = flags;
    for(int ii = 0; ii < tmp_s.size(); ii++)
    {
        if(tmp_s[ii] == ' ')
        {
            tmp_s.erase(tmp_s.begin()+ii);
            ii--;
        }
    }
    return tmp_s;
}
