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
bool unc;

/* Read input file and set global variables */
void readInput(const string filename);

int main()
{
    readInput("input");
    INT_SPH intor(atomName, basisSet);   

    bool twoC = true, spinFree = true, finiteNuc = true, gaunt = false, gauge = false;
    // DHF_SPH *scfer1 = new DHF_SPH(intor,"input",spinFree,twoC,gaunt,gauge,true,finiteNuc);
    // scfer1->runSCF(twoC);

    twoC = false; spinFree = false; finiteNuc = true; gaunt = false; gauge = false;
    DHF_SPH *scfer2 = new DHF_SPH(intor,"input",spinFree,twoC,gaunt,gauge,true,finiteNuc);
    scfer2->runSCF(twoC, false);

    // auto density = scfer2->get_density();
    // for(int ir = 0; ir < intor.irrep_list.rows(); ir++)
    // {
    //     int size = intor.irrep_list(ir).size, two_j = intor.irrep_list(ir).two_j, ll = intor.irrep_list(ir).l;
    //     double kappa = (two_j + 1.0) * (ll - two_j/2.0);
    //     double lk = ll + kappa + 1.0;
    //     for(int mm = 0; mm < size; mm++)
    //     for(int nn = 0; nn < size; nn++)
    //     {
    //         double norm_m = intor.shell_list(ll).norm(mm), alpha_m = intor.shell_list(ll).exp_a(mm);
    //         double norm_n = intor.shell_list(ll).norm(nn), alpha_n = intor.shell_list(ll).exp_a(nn);
    //         cout << density(ir)(mm,nn)/norm_m/norm_n << "\t" << alpha_m << "\t" << alpha_n << "\t" << ll;
    //         if(!twoC)
    //         {
    //             cout << "\t" << density(ir)(mm+size,nn+size)/norm_m/norm_n << "\t" << lk;
    //         }
    //         cout << endl;
    //     }
    // }

    double integral = 0.0, dx = 0.001;
    for(int ii = 0; ii < 30000; ii++)
    {
        double rr = ii*dx;
        integral += rr*rr*scfer2->radialDensity(rr)*dx;
    }
    cout << integral << endl;
    // double rr = 0.01;
    // vector<double> rho1, xx, rho2;// diff;
    // while (true)
    // {
    //     xx.push_back(rr);
    //     // rho1.push_back(scfer1->radialDensity(rr));
    //     rho2.push_back(scfer2->radialDensity(rr));
    //     // diff.push_back(rho2[rho2.size()-1] - rho1[rho1.size()-1]);
    //     if(abs(rho2[rho2.size()-1]) < 1e-4)
    //         break;
    //     else if(abs(rho2[rho2.size()-1]) < 1e-2)
    //         rr += 0.2;
    //     else
    //         rr += 0.1;
    // }
    // cout << "total number of points: " << xx.size() << endl;
    // // for(int ii = 0; ii < diff.size(); ii++)
    // //     cout << xx[ii] << "\t" << diff[ii] << endl;

    // // int size = scfer2->irrep_list(0).size;
    // // VectorXd aList = scfer2->shell_list(0).exp_a;
    // int size = 34;
    // VectorXd aList(size);
    // for(int ii = 0; ii < size; ii++)
    //     aList(ii) = pow(1.5,-2+ii);
    // MatrixXd A = MatrixXd::Zero(size+1,size+1);
    // VectorXd B = VectorXd::Zero(size+1);
    // MatrixXd basis(size, xx.size());
    // for(int ii = 0; ii < xx.size(); ii++)
    // for(int mm = 0; mm < size; mm++)
    //     basis(mm,ii) = pow(aList[mm]/M_PI, 1.5) * exp(-aList[mm]*xx[ii]*xx[ii]);

    // for(int mm = 0; mm < size; mm++)
    // {
    //     for(int ii = 0; ii < xx.size(); ii++)
    //     {
    //         B(mm) += rho2[ii]*basis(mm,ii);
    //         for(int nn = 0; nn < size; nn++)
    //             A(mm,nn) += basis(nn,ii)*basis(mm,ii);
    //     }
    //     A(mm,size) = -1.0;
    //     A(size,mm) = -1.0;
    // }
    // B(size) = -intor.atomNumber*4.0*M_PI;
    // VectorXd C = A.fullPivLu().solve(B);
    

    // for(int ii = 0; ii < xx.size(); ii++)
    // {
    //     double res = 0.0;
        
    //     for(int mm = 0; mm < C.rows()-1; mm++)
    //     {
    //         res += C(mm)*basis(mm,ii);
    //     }
    //     cout << xx[ii] << "\t" << rho2[ii] << "\t" << res << "\t" << rho2[ii]-res << endl;
    // }


    // double sum = 0.0;
    // for(int mm = 0; mm < C.rows()-1; mm++)
    // {
    //     sum += C(mm);
    // }
    // cout << "sum of coefficients vs Z: " << sum << "\t" << intor.atomNumber*4.0*M_PI << endl;
    // cout << "coeff" << endl << C << endl;

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

