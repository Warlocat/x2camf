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
    bool twoC = false, spinFree = false;
    readInput("input");
    INT_SPH intor(atomName, basisSet);

    DHF_SPH *scfer_aoc = new DHF_SPH_CA(intor,"input",spinFree,twoC,false,false,true,false);
    scfer_aoc->convControl = 1e-9;
    scfer_aoc->runSCF(twoC,false);
    // scfer_aoc->basisGenerator(basisSet,"out",intor,intor,spinFree);

    //                                             sf   2c   Gaunt gauge allint  gauNuc
    //dc4c
    // DHF_SPH *scfer_aoc = new DHF_SPH_CA(intor,"input",false,false,false,false,true,false);
    // scfer_aoc->convControl = 1e-9;
    // scfer_aoc->runSCF(false,false);
    // auto pcc_aoc = scfer_aoc->x2c2ePCC();

    // //x2c1e
    // DHF_SPH *scfer_2c = new DHF_SPH_CA(intor,"input",false,true,false,false,true,false);
    // auto h1e = scfer_2c->get_h1e_4c();
    // for(int ir = 0; ir < pcc_aoc.rows(); ir++)
    // {
    //     h1e(ir) += pcc_aoc(ir);
    // }
    // scfer_2c->set_h1e_4c(h1e);
    // scfer_2c->convControl = 1e-9;
    // scfer_2c->runSCF(true,false);

    // DHF_SPH *scfer_frac = new DHF_SPH(intor,"input",true,true,false,false,true,false);
    // scfer_frac->runSCF(true,false);
    // DHF_SPH *scfer_aoc = new DHF_SPH_CA(intor,"input",true,true,false,false,true,false);
    // scfer_aoc->runSCF(true,false);
    // DHF_SPH *scfer_old = new DHF_SPH_CA2(intor,"input",true,true,false,false,true,false);
    // scfer_old->runSCF(true,false);

    // // vMatrixXd Dcoeff(scfer_new->coeff.rows());
    // for(int ir = 0; ir < scfer_aoc->irrep_list.rows(); ir+=scfer_aoc->irrep_list(ir).two_j+1)
    // {        
    //     // cout << fixed << setprecision(8) << Dcoeff(ir) << endl << endl;
    //     cout << fixed << setprecision(8) << scfer_aoc->coeff(ir) << endl << endl;
    //     cout << fixed << setprecision(8) << scfer_old->coeff(ir) << endl << endl;
    // }
    // // auto h1e = scfer->get_h1e_4c(), amfso = scfer2->get_amfi_unc(intor,false);
    // for(int ir = 0; ir < h1e.rows(); ir++)
    // {
    //     h1e(ir) += amfso(ir);
    //     cout << amfso(ir) << endl << endl;
    // }
    // scfer->set_h1e_4c(h1e);

    // scfer->convControl = 1e-9;
    // scfer->runSCF(true,false);
    // scfer->basisGenerator(basisSet, "out", intor, intor, false, "SO");

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

