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
using namespace Eigen;
using namespace std;

/* Global information */
int charge, spin;
string atomName, basisSet, flags, jobs, rel;
VectorXd occ;
bool unc;

/* Read input file and set global variables */
void readInput(const string filename);

int main()
{
    readInput("input");
    clock_t startTime, endTime;
       
    INT_SPH intor(atomName, basisSet);
    DHF_SPH dhf_test(intor,"input");
    dhf_test.runSCF();
    // auto irrepList = intor.irrep_list;
    // // auto h1e = intor.get_h1e("overlap");
    // // for(int ir = 0; ir < intor.Nirrep; ir++)
    // //     cout << h1e(ir) << endl;
    
    // VectorXi test1(10), test2(10);
    // test1 << 0,1,0,1,2,3,4,5,6,7;
    // test2 << 0,0,1,1,0,0,0,0,0,0;
    // auto h2eLLLL_JK = intor.get_h2e_JK("LLLL");
    // for(int ii = 0; ii < 10; ii++)
    // for(int jj = 0; jj < 10; jj++)
    // for(int kk = 0; kk < 10; kk++)
    // for(int ll = 0; ll < 10; ll++)
    // {
    //     int ri = test1(ii), rj = test1(jj), rk = test1(kk), rl = test1(ll);
    //     int ip = test2(ii), jp = test2(jj), kp = test2(kk), lp = test2(ll);
    //     if(ri == rl && rj == rk)
    //     {
    //         cout << h2eLLLL_JK.K(ri,rk)(ip*irrepList(rj).size+jp,kp*irrepList(rl).size+lp)<< "\t" << ii+1 << "\t" << jj+1 << "\t" << kk+1 << "\t" << ll+1 << "\t K" << endl;
    //     }
    //     else if(ri == rj && rk == rl)
    //     {
    //         cout << h2eLLLL_JK.J(ri,rk)(ip*irrepList(rj).size+jp,kp*irrepList(rl).size+lp) << "\t" << ii+1 << "\t" << jj+1 << "\t" << kk+1 << "\t" << ll+1 << "\t J" << endl;
    //     }
    // }
    

    return 0;
}



void readInput(const string filename)
{
    ifstream ifs;
    ifs.open(filename);
        ifs >> atomName >> flags;
        ifs >> basisSet >> flags;
        cout << atomName << endl << basisSet <<endl;
    ifs.close();
}

