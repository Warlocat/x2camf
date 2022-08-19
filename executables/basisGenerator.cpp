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
string atomName, basisSetPVXZ, basisSetAll, flags, newTag, filenameOuput;
bool spinFree = true, aoc = true;

/* Read input file and set global variables */
void readInput(const string filename);

int main()
{
    readInput("input");
    INT_SPH intor(atomName, basisSetPVXZ);
    INT_SPH intorAll(atomName, basisSetAll);
    DHF_SPH *scfer, *scfer2;

    if(aoc)
    {
        scfer = new DHF_SPH_CA(intor,"input",spinFree,true,false,false,true,false);
    }
    else
    {
        scfer = new DHF_SPH(intor,"input",spinFree,true,false,false,true,false);
    }
    if(!spinFree)
    {
        if(aoc)
        {
            scfer2 = new DHF_SPH_CA(intor,"input",false,false,false,false,true,false);
        }
        else
        {
            scfer2 = new DHF_SPH(intor,"input",false,false,false,false,true,false);
        }
        scfer2->convControl = 1e-9;
        scfer2->runSCF(false,false);
        auto h1e = scfer->get_h1e_4c(), amfso = scfer2->get_amfi_unc(intor,false);
        for(int ir = 0; ir < h1e.rows(); ir++)
            h1e(ir) += amfso(ir);
        scfer->set_h1e_4c(h1e);
    }

    scfer->convControl = 1e-9;
    scfer->runSCF(true,false);
    scfer->basisGenerator(basisSetAll, filenameOuput, intor, intorAll, spinFree, newTag);


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
        ifs >> spinFree >> flags;
        ifs >> aoc >> flags;
        ifs >> newTag >> flags;
        ifs >> filenameOuput >> flags;
        atomName = removeSpaces(atomName);
        basisSetPVXZ = removeSpaces(basisSetPVXZ);
        basisSetAll = removeSpaces(basisSetAll);
        cout << atomName << endl << basisSetPVXZ << endl;
        cout << "spin free: " << spinFree << endl << "aoc: " << aoc << endl << "newTag: " << newTag << endl;
    ifs.close();
}


