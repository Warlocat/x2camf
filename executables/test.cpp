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
    //                           sf   2c   Gaunt gauge allint  gauNuc
    // DHF_SPH_CA sfx2c(intor,"input",true,true,false,false,true,false);
    DHF_SPH_CA x2c1e(intor,"input",false,true,false,false,true,false);
    // DHF_SPH sf4c(intor,"input",true,false,false,false,true,false);
    DHF_SPH_CA dc4c(intor,"input",false,false,false,false,true,false);
    // DHF_SPH dcg4c(intor,"input",false,false,true,false,true,false);
    // DHF_SPH dcb4c(intor,"input",false,false,true,true,true,false);
    // sfx2c.convControl = 1e-10;
    //  sf4c.convControl = 1e-10;
    //  dc4c.convControl = 1e-10;
    // dcg4c.convControl = 1e-10;
    // dcb4c.convControl = 1e-10;
    
    //  sf4c.runSCF(false,false);
    dc4c.runSCF(false,false);
    // sfx2c.set_h1e_4c(dc4c.x2c2ePCC()); 
    auto h1e_sf = x2c1e.get_h1e_4c(), amfso = dc4c.get_amfi_unc(intor,false);
    for(int ir = 0; ir < h1e_sf.rows(); ir++)
    {
        h1e_sf(ir) += amfso(ir);
    }
    x2c1e.set_h1e_4c(h1e_sf); 
    x2c1e.runSCF(true,false);
    auto coeff = x2c1e.coeff;
    cout << fixed << setprecision(8);
    for(int ir = 0; ir < coeff.rows(); ir += intor.irrep_list(ir).two_j+1)
    {
        cout << coeff(ir) << endl << endl;
    }

    // vMatrixXd amfi = sfx2c.get_amfi_unc(intor,true);
    // MatrixXd amfi_all = Rotate::unite_irrep(amfi,intor.irrep_list);
    // MatrixXcd rotate = Rotate::jspinor2cfour_interface_old(intor.irrep_list);
    // MatrixXcd amfi_final = rotate.adjoint() * amfi_all * rotate;
    // int size = amfi_final.rows()/2;
    // MatrixXd X(size,size),Y(size,size),Z(size,size);
    // for(int ii = 0; ii < size; ii++)
    // for(int jj = 0; jj < size; jj++)
    // {
    //     Z(ii,jj) = amfi_final(ii,jj).imag();
    //     Y(ii,jj) = amfi_final(ii,size+jj).real();
    //     X(ii,jj) = amfi_final(ii,size+jj).imag();
    // }
    // cout << X << endl << endl << Y << endl << endl << Z << endl;

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

