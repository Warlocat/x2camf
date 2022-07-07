#include<iostream>
#include<fstream>
#include<string>
#include<Eigen/Dense>
#include<iomanip>
#include<cmath>
#include<ctime>
#include<memory>
#include<vector>
#include"int_sph.h"
#include"dhf_sph.h"
#include"dhf_sph_ca.h"
using namespace Eigen;
using namespace std;

void readPyInput(const string& filename, vector<string>& atoms, vector<string>& basis, vector<bool>& amfiMethod);

int main()
{
    vector<bool> amfiMethod;
    vector<string> atomList, basisList;
    vector<string> atomListUnique, basisListUnique;
    vector<int> indexList;
    
    readPyInput("amf_input", atomList, basisList, amfiMethod);

    // find unique calculations
    for(int ii = 0; ii < basisList.size(); ii++)
    {
        bool unique = true;
        for(int jj = 0; jj < basisListUnique.size(); jj++)
        {
            if(atomList[ii] == atomListUnique[jj] && basisList[ii] == basisListUnique[jj])
            {
                unique = false;
                indexList.push_back(jj);
            }
        }
        if(unique)
        {
            indexList.push_back(atomListUnique.size());
            atomListUnique.push_back(atomList[ii]);
            basisListUnique.push_back(basisList[ii]);
        }
        cout << "index " << ii << ": " << indexList[ii] << endl;
    }
    cout << "All unique calculations are:" << endl;
    for(int ii = 0; ii < atomListUnique.size(); ii++)
    {
        cout << atomListUnique[ii] << "\t" << basisListUnique[ii] << endl;
    }

    string method = "";
    if(amfiMethod[0])   method = method + "aoc-";
    method = method + "Dirac Hatree Fock\n";
    if(amfiMethod[1])   method = method + "with Gaunt\n";
    if(amfiMethod[2])   method = method + "with gauge\n";
    cout << "amfi Method input: " << method << endl;

    vector<MatrixXcd> amfiUnique;
    for(int ii = 0; ii < atomListUnique.size(); ii++)
    {
        INT_SPH intor(atomListUnique[ii],basisListUnique[ii]);
        auto irrep_list_main(intor.irrep_list);
        if(amfiMethod[0])
        {
            cout << "Average of configuration SCF amfi is suppressed for testing purpose." << endl;
            exit(99);
            DHF_SPH_CA scfer(intor, "amf_input", false, false, amfiMethod[1], amfiMethod[2]);
            scfer.runSCF(false,false);
            amfiUnique.push_back(Rotate::unite_irrep(scfer.get_amfi_unc(intor,false), irrep_list_main));
        }
        else
        {
            DHF_SPH scfer(intor, "amf_input", false, false, amfiMethod[1], amfiMethod[2]);
            scfer.runSCF(false,false);
            amfiUnique.push_back(Rotate::unite_irrep(scfer.get_amfi_unc(intor,false), intor.irrep_list));
        }
        // MatrixXcd tmp = Rotate::jspinor2sph(irrep_list_main) * Rotate::sph2solid(irrep_list_main);
        // amfiUnique[ii] = tmp.adjoint() * amfiUnique[ii] * tmp;
    }

    int sizeAll = 0, int_tmp = 0;
    for(int ii = 0; ii < atomList.size(); ii++)
    {
        sizeAll += amfiUnique[indexList[ii]].rows();
    }
    MatrixXcd amfiAll(sizeAll,sizeAll);
    amfiAll = MatrixXcd::Zero(sizeAll,sizeAll);
    for(int ii = 0; ii < atomList.size(); ii++)
    {
        int size_tmp = amfiUnique[indexList[ii]].rows();
        for(int mm = 0; mm < size_tmp; mm++)
        for(int nn = 0; nn < size_tmp; nn++)
        {
            amfiAll(int_tmp+mm,int_tmp+nn) = amfiUnique[indexList[ii]](mm,nn);
        }
        int_tmp += amfiUnique[indexList[ii]].rows();
    }

    ofstream ofs;
    ofs.open("amf_int");
    for(int mm = 0; mm < sizeAll; mm++)
    for(int nn = 0; nn < sizeAll; nn++)
    {
        ofs << setprecision(16) << amfiAll(mm,nn).real() << "+" << amfiAll(mm,nn).imag() << "j" << endl;
    }
    ofs.close();
     

    return 0;
}



void readPyInput(const string& filename, vector<string>& atoms, vector<string>& basis, vector<bool>& amfiMethod)
{
    int Nsize;
    bool readAtom = true, readBasis = true;
    ifstream ifs;
    string flags, flags2, basisSet;
    ifs.open(filename);
        if(!ifs)
        {
            cout << "ERROR opening file " << filename << endl;
            exit(99);
        }
        ifs >> Nsize;
        atoms.resize(Nsize);
        basis.resize(Nsize);
        for(int ii = 0; ii < Nsize; ii++)
        {
            ifs >> atoms[ii] >> basis[ii];
            cout << atoms[ii] << endl << basis[ii] << endl;
        }
        
        while (!ifs.eof())
        {
            getline(ifs,flags);
            removeSpaces(flags);
            if(flags == "%amfiMethod*")
            {   
                //aoc
                //gaunt
                //gauge
                for(int ii = 0 ; ii < 3; ii++)
                {
                    bool tmp;
                    ifs >> tmp;
                    amfiMethod.push_back(tmp);
                    cout << amfiMethod[ii] << endl;
                }
                break;
            }
        }
    ifs.close();

    return ;
}


