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

/* Read input file and set global variables */
void readInput(const string filename);

int main()
{
    readInput("input");
    clock_t startTime, endTime;
       
    GTO_SPINOR gto_spinor_test(atomName, basisSet, charge, spin);
    cout << "size_c_spinor: " << gto_spinor_test.size_gtoc_spinor << endl;
    cout << "size_u_spinor: " << gto_spinor_test.size_gtou_spinor << endl;

    // MatrixXd spnucsp = gto_spinor_test.get_h1e("s_p_nuc_s_p", unc);
    // for(int ii = 0; ii < spnucsp.rows(); ii++)
    // for(int jj = 0; jj < spnucsp.cols(); jj++)
        // cout << ii+1 << "\t" << jj+1 << "\t" << spnucsp(ii,jj) <<"\n";
    // exit(99);
    // MatrixXd w_sf = gto_spinor_test.get_h1e("p.Vp", unc), w_sd = gto_spinor_test.get_h1e("i_s_pV_x_p", unc);
    // cout << (w_sf + w_sd - spnucsp).maxCoeff() << endl << endl;

    MatrixXd h2eLLLL = gto_spinor_test.get_h2e("LLLL",unc);
    MatrixXd h2eSSLL = gto_spinor_test.get_h2e("SSLL",unc);
    MatrixXd h2eSSSS = gto_spinor_test.get_h2e("SSSS",unc);
    gto_spinor_test.writeIntegrals_spinor(h2eLLLL, "h2etestLLLL");    
    gto_spinor_test.writeIntegrals_spinor(h2eSSLL, "h2etestSSLL"); 
    gto_spinor_test.writeIntegrals_spinor(h2eSSSS, "h2etestSSSS");
    DHF dhf_test(gto_spinor_test, h2eLLLL, h2eSSLL, h2eSSSS, unc);
    
    // DHF dhf_test("h1epyscf", "h2epyscf");
    
    dhf_test.runSCF();

    // int size = round(sqrt(h2eLLLL.cols()));

    // for(int ii = 6; ii < 10; ii ++)
    // for(int jj = 6; jj < 10; jj ++)
    // for(int kk = 6; kk < 10; kk ++)
    // for(int ll = 6; ll < 10; ll ++)
    //     cout << setprecision(16) << h2eSSSS(ii*size+jj,kk*size+ll) << endl;

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
        cout << atomName << endl << basisSet <<endl << charge<< endl << spin << endl << jobs << endl << rel << endl << unc << endl;
    ifs.close();
}
