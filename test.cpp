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
    //                          sf   2c   Gaunt gauge allint  gauNuc
    // DHF_SPH sfx2c(intor,"input",true,true,false,false,true,false);
    // DHF_SPH sf4c(intor,"input",true,false,false,false,true,false);
     DHF_SPH_CA dc4c(intor,"input",false,false,false,false,true,false);
    // DHF_SPH dcg4c(intor,"input",false,false,true,false,true,false);
    // DHF_SPH dcb4c(intor,"input",false,false,true,true,true,false);
    // sfx2c.convControl = 1e-10;
    //  sf4c.convControl = 1e-10;
      dc4c.convControl = 1e-10;
    // dcg4c.convControl = 1e-10;
    // dcb4c.convControl = 1e-10;
    // sfx2c.runSCF(true,false);
    
    //  sf4c.runSCF(false,false);
     dc4c.runSCF(false,false);
     //sfx2c.set_h1e_4c(dc4c.x2c2ePCC()); 
    //sfx2c.runSCF(true,false);
   // auto h2c2e = dc4c.h_x2c2e();
   // auto den = sfx2c.get_density();
   // auto fock = sfx2c.get_fock_4c();
   // double ene_scf = 0.0;
   // for(int ir = 0; ir < sfx2c.occMax_irrep; ir += sfx2c.irrep_list(ir).two_j+1)
   // {
   //     int size_tmp = sfx2c.irrep_list(ir).size;
   //         for(int ii = 0; ii < size_tmp; ii++)
   //         for(int jj = 0; jj < size_tmp; jj++)
   //         {
   //             // ene_scf += 0.5 * density(ir)(ii,jj) * (h1e_4c(ir)(jj,ii) + fock_4c(ir)(jj,ii)) * (irrep_list(ir).two_j+1.0);
   //             ene_scf += 0.5 * den(ir)(ii,jj) * (fock(ir)(jj,ii) + h2c2e(ir)(jj,ii)) * (sfx2c.irrep_list(ir).two_j+1.0);
   //         }
   // }
   // cout << ene_scf << endl;
   // // dcg4c.runSCF(false,false);
    // dcb4c.runSCF(false,false);
    // 
    // int size = dc4c.ene_orb(0).rows();
    // VectorXd ene_1s(5), de_1s(4);
    // ene_1s << sfx2c.ene_orb(0)(0),
    //            sf4c.ene_orb(0)(0+size/2),
    //            dc4c.ene_orb(0)(0+size/2),
    //           dcg4c.ene_orb(0)(0+size/2),
    //           dcb4c.ene_orb(0)(0+size/2);
    // for(int ii = 0; ii < 4; ii++)
    // {
    //     de_1s(ii) = (ene_1s(ii+1) - ene_1s(ii))*27.21138;
    // }

    // cout << ene_1s << endl << de_1s << endl;

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

