#include<iostream>
#include<fstream>
#include<string>
#include<Eigen/Dense>
#include<iomanip>
#include<cmath>
#include<ctime>
#include<memory>
#include"gto.h"
#include"scf.h"
#include"x2c.h"
#include"gto_spinor.h"
using namespace Eigen;
using namespace std;

/* Global information */
int charge, spin;
string atomName, basisSet, flags, jobs, rel;
bool unc;
double conv;

/* Read input file and set global variables */
void readInput(const string filename);

/* Write 1e integrals */
void writeOneE(const string filename, const MatrixXd& oneE);

int main()
{
    readInput("input");
    clock_t startTime, endTime;
       
    GTO_SPINOR gto_spinor_test(atomName, basisSet, charge, spin);
    cout << "size_c_spinor: " << gto_spinor_test.size_gtoc_spinor << endl;
    cout << "size_u_spinor: " << gto_spinor_test.size_gtou_spinor << endl;

    MatrixXd h2eLLLL = gto_spinor_test.get_h2e("LLLL", true);
    MatrixXd h2eSSLL = gto_spinor_test.get_h2e("SSLL", true);
    MatrixXd h2eSSSS = gto_spinor_test.get_h2e("SSSS", true);
    gto_spinor_test.writeIntegrals_spinor(h2eLLLL, "h2e_"+atomName+"_LLLL");
    gto_spinor_test.writeIntegrals_spinor(h2eSSLL, "h2e_"+atomName+"_SSLL");
    gto_spinor_test.writeIntegrals_spinor(h2eSSSS, "h2e_"+atomName+"_SSSS");
    DHF dhf_test(gto_spinor_test, "h2e_"+atomName+"_", true);
    dhf_test.runSCF();


    h2eLLLL.resize(0,0);
    h2eSSLL.resize(0,0);
    h2eSSSS.resize(0,0);
    MatrixXd h2eSSLL_SD = gto_spinor_test.get_h2e("SSLL_SD", true);
    MatrixXd h2eSSSS_SD = gto_spinor_test.get_h2e("SSSS_SD", true);
    MatrixXd amfi;
    if(unc)
    {
        amfi = dhf_test.get_amfi(h2eSSLL_SD, h2eSSSS_SD, MatrixXd::Identity(gto_spinor_test.size_gtou_spinor,gto_spinor_test.size_gtou_spinor));
    }
    else
    {
        amfi = dhf_test.get_amfi(h2eSSLL_SD, h2eSSSS_SD, gto_spinor_test.get_coeff_contraction_spinor());
    }
    for(int ii = 0; ii < amfi.rows(); ii++)
    for(int jj = 0; jj < amfi.cols(); jj++)
    {
        if(abs(amfi(ii,jj)) > 1e-8)
        {
            cout << ii << "\t" << jj << "\t" << amfi(ii,jj) << "\n";
        }
    }
    
    return 0;
}



void readInput(const string filename)
{
    ifstream ifs;
    ifs.open(filename);
        ifs >> atomName >> flags;
        ifs >> basisSet >> flags;
        ifs >> charge >> flags;
        ifs >> spin >> flags;
        ifs >> jobs >> flags;
        ifs >> rel >> flags;
        ifs >> unc >> flags;
        ifs >> conv >> flags;
        cout << atomName << endl << basisSet <<endl << charge<< endl << spin << endl << jobs << endl << rel << endl << unc << endl << conv << endl;
    ifs.close();

    return;
}

void writeOneE(const string filename, const MatrixXd& oneE)
{
    int size = oneE.cols();
    ofstream ofs;
    ofs.open(filename);
        ofs << setprecision(16);
        for(int ii = 0; ii < size; ii++)
        for(int jj = 0; jj < size; jj++)
        {
            if(abs(oneE(ii,jj)) > 1e-12) ofs << ii << "\t" << jj << "\t" << oneE(ii,jj) << "\n";
        }    
    ofs.close();

    return;
}