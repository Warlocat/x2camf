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
#include"element.h"
#include"dhf_sph_ca.h"
using namespace Eigen;
using namespace std;

/* Global information */
int n, l, twoj;
string atomName, basisSet, flags;
double dc_sfx2c, dcb_dc, qed = 0.0;

/* Read input file and set global variables */
void readInput(const string filename);

double QEDcorrection(const int& N, const int& L, const int& twoJ, const int& Z);

int main()
{
    readInput("input");
    INT_SPH intor(atomName, basisSet);
    //                           sf   2c   Gaunt gauge allint  gauNuc
    DHF_SPH sfx2c(intor,"input",4,true,true,false,false,true,false);
    DHF_SPH dc4c(intor,"input",4,false,false,false,false,true,false);
    DHF_SPH dcb4c(intor,"input",4,false,false,true,true,true,false);

    sfx2c.runSCF(true,false);
    dc4c.runSCF(false,false);
    dcb4c.runSCF(false,false);
    
    vVectorXd mo_ene_sfx2c = sfx2c.ene_orb, mo_ene_dc = dc4c.ene_orb, mo_ene_dcb = dcb4c.ene_orb;
    auto irrep_list = intor.irrep_list;
    for(int ir = 0; ir < irrep_list.rows(); ir++)
    {
        if(irrep_list(ir).l == l && irrep_list(ir).two_j == twoj)
        {
            int n_tmp = n - 1 - l, n2c = mo_ene_sfx2c(ir).rows();
            dc_sfx2c = mo_ene_dc(ir)(n2c + n_tmp) - mo_ene_sfx2c(ir)(n_tmp);
            dcb_dc = mo_ene_dcb(ir)(n2c + n_tmp) - mo_ene_dc(ir)(n2c + n_tmp);
            break;
        }
    }
    qed = QEDcorrection(n,l,twoj,elem_map.find(atomName)->second);

    string orbitalName = to_string(n) + orbL[l] + "_" + to_string(twoj) + "/2";

    cout << "The correction (in eV) to orbital energy of " << orbitalName << endl;
    cout << fixed << setprecision(4);
    cout << "DC - SFX2C1e:\t\t" << dc_sfx2c*au2ev << endl; 
    cout << "Breit term:\t\t" << dcb_dc*au2ev << endl;
    cout << "QED term:\t\t" << qed*au2ev << endl;
    cout << "In total:\t\t" << (dcb_dc + dc_sfx2c + qed)*au2ev << endl;

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
        ifs >> n >> l >> twoj;
        cout << atomName << endl << basisSet << endl << "quantum numbers:" << endl;
        cout << "n\tl\ttwoJ" << endl;
        cout << n << "\t" << l << "\t" << twoj << endl;
        if(l+1 > n || abs(twoj - 2*l) > 1)
        {
            cout << "The input N, L, and twoJ are not consistent!" << endl;
            exit(99);
        }
    ifs.close();
}

double QEDcorrection(const int& N, const int& L, const int& twoJ, const int& Z)
{
    cout << "Using fitted QED correction documented in " << endl;
    cout << "K. Kozioł and G. A. Aucar, J. Chem. Phys., 2018, 148, 134101" << endl << endl;
    if(N > 4 || L > 1)
    {
        cout << "The QED correction is not available for given N or L and set to 0." << endl;
        return 0.0;
    }
    if(Z<30 || Z>118)
    {
        cout << "Warning: The QED correction of given atomic number may not be accurate." << endl;
    }
    double aa, bb;
    double a_s[4] = {8.020e-7, 1.568e-8, 8.182e-11, 2.562e-12};
    double a_l_1[3] = {1.715e-17, 7.619e-20, 2.961e-21};
    double a_l_3[3] = {4.819e-11, 1.237e-12, 2.085e-14};
    double b_s[4] = {3.607, 4.082, 4.926, 5.398};
    double b_l_1[3] = {8.191, 9.104, 9.544};
    double b_l_3[3] = {4.955, 5.466, 6.079};
    if(L == 0)
    {
        aa = a_s[N-1];
        bb = b_s[N-1];
    }
    else if(L == 1)
    {
        if(twoJ == 1)
        {
            aa = a_l_1[N-2];
            bb = b_l_1[N-2];
        }
        else if(twoJ == 3)
        {
            aa = a_l_3[N-2];
            bb = b_l_3[N-2];
        }
    }
    return aa*pow(Z,bb);
}