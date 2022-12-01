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
bool spinFree = true, aoc = true, finiteN = false;
MatrixXi basisInfo;
Matrix<VectorXi,-1,1> deconInfo; 

/* Read input file and set global variables */
void readInput(const string filename);

int main()
{
    readInput("input");
    INT_SPH intor(atomName, basisSetPVXZ);
    // INT_SPH intorAll(atomName, basisSetAll);
    DHF_SPH *scfer, *scfer2;

    if(aoc)
    {
        scfer = new DHF_SPH_CA(intor,"input",spinFree,true,false,false,true,finiteN);
    }
    else
    {
        scfer = new DHF_SPH(intor,"input",spinFree,true,false,false,true,finiteN);
    }
    if(!spinFree)
    {
        if(aoc)
        {
            scfer2 = new DHF_SPH_CA(intor,"input",false,false,false,false,true,finiteN);
        }
        else
        {
            scfer2 = new DHF_SPH(intor,"input",false,false,false,false,true,finiteN);
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
    scfer->basisGenerator(basisSetAll, filenameOuput, intor, basisInfo, deconInfo, spinFree);
    // print mo coefficients for convenience
    cout << fixed << setprecision(8);
    for(int ir = 0; ir < scfer->irrep_list.rows(); ir += scfer->irrep_list(ir).two_j+1)
    {
        cout << scfer->coeff(ir) << endl << endl;
    }

    delete scfer;
    return 0;
}



void readInput(const string filename)
{
    ifstream ifs;
    int Ntmp;
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
        ifs >> finiteN >> flags;
        ifs >> newTag >> flags;
        ifs >> filenameOuput >> flags;
        atomName = removeSpaces(atomName);
        basisSetPVXZ = removeSpaces(basisSetPVXZ);
        basisSetAll = removeSpaces(basisSetAll);
        cout << atomName << endl << basisSetPVXZ << endl;
        cout << "spin free: " << spinFree << endl << "aoc: " << aoc << endl << "newTag: " << newTag << endl;

        ifs >> Ntmp;
        basisInfo.resize(Ntmp,2);
        deconInfo.resize(Ntmp);
        for(int ii = 0; ii < Ntmp; ii++)
        {
            ifs >> basisInfo(ii,0) >> basisInfo(ii,1);
            deconInfo(ii).resize(basisInfo(ii,1));
            for(int jj = 0; jj < basisInfo(ii,1); jj++)
                ifs >> deconInfo(ii)(jj);
        }
    ifs.close();
}


