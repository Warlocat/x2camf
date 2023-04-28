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
    DHF_SPH *scfer2 = new DHF_SPH(intor,"input",4,spinFree,twoC,gaunt,gauge,true,finiteNuc);
    scfer2->runSCF(twoC, false);

    // VectorXd aList = intor.shell_list(0).exp_a;
    vector<double> aList;
    for(double xx = 1e-4; xx < 1e6; xx = xx*1.8)
        aList.push_back(xx);
    auto density = scfer2->get_density();
    int size = aList.size();
    MatrixXd A = MatrixXd::Zero(size+1,size+1);
    VectorXd B = VectorXd::Zero(size+1), basisNorm(size);
    for(int mm = 0; mm < size; mm++)
        basisNorm(mm) = pow(aList[mm]/M_PI, 1.5);

    for(int mm = 0; mm < size; mm++)
    {
        for(int ir = 0; ir < intor.irrep_list.rows(); ir += intor.irrep_list(ir).two_j+1)
        {
            int size_ir = intor.irrep_list(ir).size, two_j = intor.irrep_list(ir).two_j, ll = intor.irrep_list(ir).l;
            double kappa = (two_j + 1.0) * (ll - two_j/2.0);
            double lk = ll + kappa + 1.0;
            for(int aa = 0; aa < size_ir; aa++)
            for(int bb = 0; bb < size_ir; bb++)
            {
                double norm_a = intor.shell_list(ll).norm(aa), alpha_a = intor.shell_list(ll).exp_a(aa);
                double norm_b = intor.shell_list(ll).norm(bb), alpha_b = intor.shell_list(ll).exp_a(bb);
                B(mm) += (two_j+1.0) * density(ir)(aa,bb)/norm_a/norm_b*basisNorm(mm)*intor.auxiliary_1e(2+2*ll, aList[mm]+alpha_a+alpha_b);
                if(!twoC)
                {
                    double tmp = 4.0*alpha_a*alpha_b*intor.auxiliary_1e(4+2*ll, aList[mm]+alpha_a+alpha_b);
                    if(ll>=1) 
                    {
                        tmp -= 2.0*lk*(alpha_a+alpha_b)*intor.auxiliary_1e(2+2*ll, aList[mm]+alpha_a+alpha_b);
                        tmp += lk*lk*intor.auxiliary_1e(2*ll, aList[mm]+alpha_a+alpha_b);
                    }
                    B(mm) += (two_j+1.0) * density(ir)(aa+size_ir,bb+size_ir)/norm_a/norm_b*basisNorm(mm)*tmp/4.0/speedOfLight/speedOfLight;
                }
            }
        }
        for(int nn = 0; nn < size; nn++)
            A(mm,nn) = basisNorm(mm)*basisNorm(nn)*intor.auxiliary_1e(2, aList[nn]+aList[mm])*4.0*M_PI;
        A(mm,size) = -1.0;
        A(size,mm) = -1.0;
    }
    B(size) = -intor.atomNumber;

    VectorXd C = A.fullPivLu().solve(B);
    double sum = 0.0;
    for(int mm = 0; mm < C.rows()-1; mm++)
    {
        sum += C(mm);
    }
    cout << "sum of coefficients vs Z: " << sum << "\t" << intor.atomNumber << endl;
    cout << "coeff" << endl << C << endl;

    cout << "density: " << endl;
    for(double rr = 0.0; rr < 10.0; rr += 0.01)
    {
        double rho1 = scfer2->radialDensity(rr);
        double rho2 = 0.0;
        for(int mm = 0; mm < size; mm++)
            rho2 += C(mm)*basisNorm(mm)*exp(-aList[mm]*rr*rr);
        cout << rr << "\t" << rho1/4.0/M_PI << "\t" << rho2 << "\t" << (rho2-rho1/4.0/M_PI)/rho1*4.0*M_PI*100 << "%" << endl;
    }

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

