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
using namespace Eigen;
using namespace std;

/* Global information */
int charge, spin;
string atomName, basisSet, flags, jobs, rel;
bool unc;

/* Read input file and set global variables */
void readInput(const string filename);

int main()
{
    readInput("input");
    clock_t startTime, endTime;
       
    GTO gto_test(atomName, basisSet, charge, spin);

    int size_c = gto_test.size_gtoc, size_u = gto_test.size_gtou;

    cout << "size_c: " << size_c << endl;
    cout << "size_u: " << size_u << endl;

    startTime = clock();
    const MatrixXd h2e = gto_test.get_h2e(unc);
    endTime = clock();
    cout << "2e integrals finished in " << (endTime - startTime) / (double)CLOCKS_PER_SEC << " seconds." << endl;
// exit(99);
    // gto_test.writeIntegrals(h2e, "h2e_"+atomName+".txt");

    if(jobs == "SCF" && rel == "SFX2C1E")
    {
        cout << "SFX2C-1E procedure is used.\n";
        shared_ptr<SCF> ptr_scf(scf_init(gto_test, h2e, "sfx2c1e"));
        
        startTime = clock();
        ptr_scf->runSCF();
        endTime = clock();
        cout << "HF (SFX2C 1E) scf finished in " << (endTime - startTime) / (double)CLOCKS_PER_SEC << " seconds." << endl; 
    }
    else if(jobs == "SCF")
    {
        cout << "Non-relativistic calculation is used.\n";
        shared_ptr<SCF> ptr_scf(scf_init(gto_test, h2e, "off"));
        
        startTime = clock();
        ptr_scf->runSCF();
        endTime = clock();
        cout << "HF scf finished in " << (endTime - startTime) / (double)CLOCKS_PER_SEC << " seconds." << endl; 
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
        cout << atomName << endl << basisSet <<endl << charge << endl << spin << endl << jobs << endl << rel << endl << unc << endl;
    ifs.close();
}
