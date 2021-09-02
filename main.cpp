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

void readZMAT(const string& filename, vector<string>& atoms, vector<string>& basis, vector<bool>& amfiMethod);
string removeSpaces(const string& flags);
vector<string> splitSrting(const string& flags, const char& targetChar);

int main()
{
    vector<bool> amfiMethod;
    vector<string> atomList, basisList;
    vector<string> atomListUnique, basisListUnique;
    vector<int> indexList;
    
    readZMAT("ZMAT", atomList, basisList, amfiMethod);
    // special treatment of PCVXZ and PWCVXZ basis
    for(int ii = 0; ii < basisList.size(); ii++)
    {
        size_t found1 = basisList[ii].find("PCV"), found2 = basisList[ii].find("PWCV");
        if(found1 != string::npos || found2 != string::npos)
        {
            basisList[ii] = basisList[ii] + "-unc";
        }
    }
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
    if(amfiMethod[1])   method = method + "spin-free-";
    if(amfiMethod[2])   method = method + "x2c1e-";
    else method = method + "Dirac Hatree Fock";
    if(amfiMethod[3])   method = method + "with Gaunt";
    cout << "amfi Method input: " << method << endl;

    vector<MatrixXcd> amfiUnique;
    for(int ii = 0; ii < atomListUnique.size(); ii++)
    {
        INT_SPH intor(atomListUnique[ii],basisListUnique[ii]);
        if(amfiMethod[0])
        {
            DHF_SPH_CA scfer(intor, "ZMAT", amfiMethod[1], amfiMethod[2], amfiMethod[3]);
            scfer.runSCF(amfiMethod[2]);
            amfiUnique.push_back(Rotate::unite_irrep(scfer.get_amfi_unc_ca(intor,amfiMethod[2]), intor.irrep_list));
        }
        else
        {
            DHF_SPH scfer(intor, "ZMAT", amfiMethod[1], amfiMethod[2], amfiMethod[3]);
            scfer.runSCF(amfiMethod[2]);
            //amfiUnique.push_back(Rotate::unite_irrep(scfer.x2c2ePCC(),intor.irrep_list));
            amfiUnique.push_back(Rotate::unite_irrep(scfer.get_amfi_unc(intor,amfiMethod[2]), intor.irrep_list));
        }
        MatrixXcd tmp = Rotate::jspinor2cfour_interface_old(intor.irrep_list);
        amfiUnique[ii] = tmp.adjoint() * amfiUnique[ii] * tmp;
        amfiUnique[ii] = Rotate::separate2mCompact(amfiUnique[ii],intor.irrep_list);
    }

    int sizeAll = 0, int_tmp = 0;
    for(int ii = 0; ii < atomList.size(); ii++)
    {
        sizeAll += amfiUnique[indexList[ii]].rows();
    }
    int sizeAll2 = sizeAll*sizeAll, sizeHalf = sizeAll/2;
    F_INTERFACE::f_dcomplex amfiAll[sizeAll*sizeAll];
    for(int ii = 0; ii < sizeAll2; ii++)
    {
        amfiAll[ii].dr = 0.0;
        amfiAll[ii].di = 0.0;
    }
    for(int ii = 0; ii < atomList.size(); ii++)
    {
        int size_tmp_half = amfiUnique[indexList[ii]].rows()/2;
        for(int mm = 0; mm < size_tmp_half; mm++)
        for(int nn = 0; nn < size_tmp_half; nn++)
        {
            // transpose for Fortran interface
	        // separate alpha and beta
            amfiAll[(int_tmp+mm)*sizeAll + int_tmp+nn].dr = amfiUnique[indexList[ii]](nn,mm).real();
            amfiAll[(int_tmp+mm)*sizeAll + int_tmp+nn].di = amfiUnique[indexList[ii]](nn,mm).imag();
            amfiAll[(int_tmp+mm+sizeHalf)*sizeAll + int_tmp+nn].dr = amfiUnique[indexList[ii]](nn,size_tmp_half+mm).real();
            amfiAll[(int_tmp+mm+sizeHalf)*sizeAll + int_tmp+nn].di = amfiUnique[indexList[ii]](nn,size_tmp_half+mm).imag();
            amfiAll[(int_tmp+mm)*sizeAll + int_tmp+nn+sizeHalf].dr = amfiUnique[indexList[ii]](size_tmp_half+nn,mm).real();
            amfiAll[(int_tmp+mm)*sizeAll + int_tmp+nn+sizeHalf].di = amfiUnique[indexList[ii]](size_tmp_half+nn,mm).imag();
            amfiAll[(int_tmp+mm+sizeHalf)*sizeAll + int_tmp+nn+sizeHalf].dr = amfiUnique[indexList[ii]](size_tmp_half+nn,size_tmp_half+mm).real();
            amfiAll[(int_tmp+mm+sizeHalf)*sizeAll + int_tmp+nn+sizeHalf].di = amfiUnique[indexList[ii]](size_tmp_half+nn,size_tmp_half+mm).imag();
        }
        int_tmp += amfiUnique[indexList[ii]].rows()/2;
    }

    int sizeAllReal = 2*sizeAll2;
    for(int ii = 0; ii < sizeAll2; ii++)
        cout << amfiAll[ii].dr <<endl << amfiAll[ii].di<< endl;
    //F_INTERFACE::rfile_("X2CMFSOM_CFOUR",tmp,&sizeAllReal);
    //F_INTERFACE::prvecr_(tmp,&sizeAllReal);
    F_INTERFACE::wfile_("X2CMFSOM",(double*)amfiAll,&sizeAllReal);
    F_INTERFACE::prvecr_((double*)amfiAll,&sizeAllReal);

    

    return 0;
}



void readZMAT(const string& filename, vector<string>& atoms, vector<string>& basis, vector<bool>& amfiMethod)
{
    bool readAtom = true, readBasis = true;
    ifstream ifs;
    string flags, flags2, basisSet;
    ifs.open(filename);
    if(!ifs)
    {
        cout << "ERROR opening file " << filename << endl;
        exit(99);
    }
        getline(ifs,flags);
        while (!ifs.eof())
        {
            getline(ifs,flags);
            flags2 = removeSpaces(flags);
            if(flags2.size() != 0 && readAtom)
            {
                atoms.push_back(splitSrting(flags, ' ')[0]);
                if(atoms[atoms.size()-1] == "X")
                {
                    atoms.erase(atoms.end());
                }
                for(int ii = 0; ii < atoms[atoms.size()-1].size(); ii++)
                {
                    if(atoms[atoms.size()-1][ii] >= 97 && atoms[atoms.size()-1][ii] <= 122)
                        atoms[atoms.size()-1][ii] = atoms[atoms.size()-1][ii] - 32;
                }
            }
            else
            {
                readAtom = false;
            }
            if(flags.substr(0,7) == "*CFOUR(" || flags.substr(0,7) == "*ACES2(")    break;
        }
        while (!ifs.eof())
        {
            getline(ifs,flags);
            removeSpaces(flags);
            size_t found = flags.find("BASIS");
            if(found != string::npos)
            {
                string tmp_s = flags.substr(found+6,flags.size()-6);
                if(tmp_s != "SPECIAL")
                {
                    for(int ii = 0; ii < atoms.size(); ii++)
                    {
                        basis.push_back(atoms[ii]+":"+tmp_s);
                    }
                    readBasis = false;
                }
            }
            found = flags.find(")");
            if(found != string::npos)
            {
                break;
            }
        }
        while (!ifs.eof() && readBasis)
        {
            getline(ifs,flags);
	    size_t found = flags.find(":");
            if(found != string::npos)
	    {
                flags = removeSpaces(flags);
                basis.push_back(flags);
                for(int ii = 1; ii < atoms.size(); ii++)
                {
                    getline(ifs,flags);
                    flags = removeSpaces(flags);
                    basis.push_back(flags);
                }
		break;
	    }
        }
        while (!ifs.eof())
        {
            getline(ifs,flags);
            removeSpaces(flags);
            if(flags == "%amfiMethod*")
            {   
                //aoc
                //spin-free
                //two-component
                //gaunt
                for(int ii = 0 ; ii < 4; ii++)
                {
                    bool tmp;
                    ifs >> tmp;
                    amfiMethod.push_back(tmp);
                }
                break;
            }
        }
        if(amfiMethod.size() == 0)
        {
            cout << "%amfiMethod is not found in ZMAT and set to default frac-DHF-gaunt" << endl;
            amfiMethod.push_back(false); //aoc
            amfiMethod.push_back(false); //spin-free
            amfiMethod.push_back(false); //two-component
            amfiMethod.push_back(true); //with gaunt
        }
    ifs.close();

    return ;
}


string removeSpaces(const string& flags)
{
    string tmp_s = flags;
    for(int ii = 0; ii < tmp_s.size(); ii++)
    {
        if(tmp_s[ii] == ' ')
        {
            tmp_s.erase(tmp_s.begin()+ii);
            ii--;
        }
    }
    return tmp_s;
}

vector<string> splitSrting(const string& flags, const char& targetChar)
{
    string tmp_s = flags;
    vector<string> res;
    while (true)
    {
        size_t found = tmp_s.find(targetChar);
        if(found != string::npos)
        {
            res.push_back(tmp_s.substr(0,found));
            tmp_s = tmp_s.substr(found+1,tmp_s.size()-found-1);
        }
        else
        {
            res.push_back(tmp_s);
            break;
        }
    }
    return res;
}
