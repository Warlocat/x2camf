#include<iostream>
#include<fstream>
#include<string>
#include<iomanip>
#include<cmath>
#include<ctime>
#include<memory>
#include<vector>
#include<complex>
#include"int_sph.h"
#include"dhf_sph.h"
#include"dhf_sph_ca.h"
#include"mkl_itrf.h"
using namespace std;

void readZMAT(const string& filename, vector<string>& atoms, vector<string>& basis, vector<bool>& amfiMethod, double& SCFconv);

int main()
{
    double amfiSCFconv = 1e-8;
    vector<bool> amfiMethod;
    vector<string> atomList, basisList;
    vector<string> atomListUnique, basisListUnique;
    vector<int> indexList;
    bool PT = false, twoC = false;
    
    readZMAT("ZMAT", atomList, basisList, amfiMethod, amfiSCFconv);
    PT = amfiMethod[5];
    twoC = PT;
    // special treatment of PCVXZ and PWCVXZ basis
    // for(int ii = 0; ii < basisList.size(); ii++)
    // {
    //     size_t found1 = basisList[ii].find("PCV"), found2 = basisList[ii].find("PWCV"), found3 = basisList[ii].find("UNC");
    //     if((found1 != string::npos || found2 != string::npos) && found3 == string::npos)
    //     {
    //         basisList[ii] = basisList[ii] + "-unc";
    //     }
    // }
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

    if(amfiMethod[0])
    {
        cout << endl << endl;
        cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
        cout << "!!  WARNING: Average-of-configuration calculations MIGHT BE WRONG  !!" << endl;
        cout << "!!  for atoms with more than one partially occupied l-shell, e.g., !!" << endl;
        cout << "!!  uranium atom with both 5f and 6d partially occupied.           !!" << endl;
        cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
        cout << endl << endl;
    }

    vector<vectorcd> amfiUnique, XUnique, denUnique;
    for(int ii = 0; ii < atomListUnique.size(); ii++)
    {
        INT_SPH intor(atomListUnique[ii],basisListUnique[ii]);
        DHF_SPH *scfer;
        if(!PT)
        {
            /* Variational treatment */
            if(amfiMethod[0])
                /* Average of configuration */
                scfer = new DHF_SPH_CA(intor, "ZMAT", false, twoC, amfiMethod[1], amfiMethod[2],true, amfiMethod[4]);
            else
                /* Fractional occupation */
                scfer = new DHF_SPH(intor, "ZMAT", false, twoC, amfiMethod[1], amfiMethod[2],true, amfiMethod[4]);
            scfer->convControl = amfiSCFconv;
            scfer->runSCF(false);
        }
        else
        {
            /* Perturbative treatment */
            if(amfiMethod[0])
                /* Average of configuration */
                scfer = new DHF_SPH_CA(intor, "ZMAT", true, twoC, amfiMethod[1], amfiMethod[2],true, amfiMethod[4]);
            else
                /* Fractional occupation */
                scfer = new DHF_SPH(intor, "ZMAT", true, twoC, amfiMethod[1], amfiMethod[2],true, amfiMethod[4]);
            scfer->convControl = amfiSCFconv;
            scfer->runSCF(true,false);
        }
        
        if(amfiMethod[3])
            amfiUnique.push_back(real2complex(Rotate::unite_irrep(scfer->x2c2ePCC(),intor.irrep_list))); // for 2e-pcc test
        else
            amfiUnique.push_back(real2complex(Rotate::unite_irrep(scfer->get_amfi_unc(intor,twoC), intor.irrep_list)));

        XUnique.push_back(real2complex(Rotate::unite_irrep(scfer->get_X(), intor.irrep_list)));
        denUnique.push_back(real2complex(Rotate::unite_irrep(scfer->get_density(), intor.irrep_list)));
        
        // vectorcd tmp = Rotate::jspinor2cfour_interface_old(intor.irrep_list);
        vectorcd tmp = Rotate::jspinor2cfour_interface_new(intor.irrep_list);
        vectorcd tmp1;

        int NN = round(sqrt(tmp.size()));
        zgemm_itrf('c','n',NN,NN,NN,one_cp,tmp,amfiUnique[ii],zero_cp,tmp1);
        zgemm_itrf('n','n',NN,NN,NN,one_cp,tmp1,tmp,zero_cp,amfiUnique[ii]);
        amfiUnique[ii] = Rotate::separate2mCompact(amfiUnique[ii],intor.irrep_list);
        zgemm_itrf('c','n',NN,NN,NN,one_cp,tmp,XUnique[ii],zero_cp,tmp1);
        zgemm_itrf('n','n',NN,NN,NN,one_cp,tmp1,tmp,zero_cp,XUnique[ii]);
        XUnique[ii] = Rotate::separate2mCompact(XUnique[ii],intor.irrep_list);
        zgemm_itrf('c','n',NN,NN,NN,one_cp,tmp,denUnique[ii],zero_cp,tmp1);
        zgemm_itrf('n','n',NN,NN,NN,one_cp,tmp1,tmp,zero_cp,denUnique[ii]);
        denUnique[ii] = Rotate::separate2mCompact(denUnique[ii],intor.irrep_list);

        delete scfer;
    }

    cout << "Constructing amfso integrals...." << endl;
    int sizeAll = 0, int_tmp = 0;
    vector<int> sizeList(atomList.size());
    for(int ii = 0; ii < atomList.size(); ii++)
    {
        sizeList[ii] = round(sqrt(amfiUnique[indexList[ii]].size()));
        sizeAll += sizeList[ii];
    }
    int sizeAll2 = sizeAll*sizeAll, sizeHalf = sizeAll/2;
    complex<double> amfiAll[sizeAll*sizeAll], XAll[sizeAll*sizeAll];
    double drAll[sizeAll*sizeAll], diAll[sizeAll*sizeAll];
    for(int ii = 0; ii < sizeAll2; ii++)
    {
        amfiAll[ii] = zero_cp;
        XAll[ii] = zero_cp;
        drAll[ii] = 0.0;
        diAll[ii] = 0.0;
    }
    for(int ii = 0; ii < atomList.size(); ii++)
    {
        int size_tmp_half = sizeList[ii]/2;
        for(int mm = 0; mm < size_tmp_half; mm++)
        for(int nn = 0; nn < size_tmp_half; nn++)
        {
            // transpose for Fortran interface
            // separate alpha and beta
            amfiAll[(int_tmp+mm)*sizeAll + int_tmp+nn] = amfiUnique[indexList[ii]][nn*sizeList[ii]+mm];
            amfiAll[(int_tmp+mm+sizeHalf)*sizeAll + int_tmp+nn] = amfiUnique[indexList[ii]][nn*sizeList[ii]+size_tmp_half+mm];
            amfiAll[(int_tmp+mm)*sizeAll + int_tmp+nn+sizeHalf] = amfiUnique[indexList[ii]][(size_tmp_half+nn)*sizeList[ii]+mm];
            amfiAll[(int_tmp+mm+sizeHalf)*sizeAll + int_tmp+nn+sizeHalf] = amfiUnique[indexList[ii]][(size_tmp_half+nn)*sizeList[ii]+size_tmp_half+mm];

            XAll[(int_tmp+mm)*sizeAll + int_tmp+nn] = XUnique[indexList[ii]][nn*sizeList[ii]+mm];
            XAll[(int_tmp+mm+sizeHalf)*sizeAll + int_tmp+nn] = XUnique[indexList[ii]][nn*sizeList[ii]+size_tmp_half+mm];
            XAll[(int_tmp+mm)*sizeAll + int_tmp+nn+sizeHalf] = XUnique[indexList[ii]][(size_tmp_half+nn)*sizeList[ii]+mm];
            XAll[(int_tmp+mm+sizeHalf)*sizeAll + int_tmp+nn+sizeHalf] = XUnique[indexList[ii]][(size_tmp_half+nn)*sizeList[ii]+size_tmp_half+mm];

            drAll[(int_tmp+mm)*sizeAll + int_tmp+nn] = denUnique[indexList[ii]][nn*sizeList[ii]+mm].real();
            diAll[(int_tmp+mm)*sizeAll + int_tmp+nn] = denUnique[indexList[ii]][nn*sizeList[ii]+mm].imag();
            drAll[(int_tmp+mm+sizeHalf)*sizeAll + int_tmp+nn] = denUnique[indexList[ii]][nn*sizeList[ii]+size_tmp_half+mm].real();
            diAll[(int_tmp+mm+sizeHalf)*sizeAll + int_tmp+nn] = denUnique[indexList[ii]][nn*sizeList[ii]+size_tmp_half+mm].imag();
            drAll[(int_tmp+mm)*sizeAll + int_tmp+nn+sizeHalf] = denUnique[indexList[ii]][(size_tmp_half+nn)*sizeList[ii]+mm].real();
            diAll[(int_tmp+mm)*sizeAll + int_tmp+nn+sizeHalf] = denUnique[indexList[ii]][(size_tmp_half+nn)*sizeList[ii]+mm].imag();
            drAll[(int_tmp+mm+sizeHalf)*sizeAll + int_tmp+nn+sizeHalf] = denUnique[indexList[ii]][(size_tmp_half+nn)*sizeList[ii]+size_tmp_half+mm].real();
            diAll[(int_tmp+mm+sizeHalf)*sizeAll + int_tmp+nn+sizeHalf] = denUnique[indexList[ii]][(size_tmp_half+nn)*sizeList[ii]+size_tmp_half+mm].imag();
        }
        int_tmp += sizeList[ii]/2;
    }

    int sizeAllReal = 2*sizeAll2;
    cout << "Writing amfso integrals...." << endl;
    if(!PT)
    {
        writeMatrixBinary((double*)amfiAll,sizeAllReal,"X2CMFSOM");
        writeMatrixBinary((double*)XAll,sizeAllReal,"X2CATMXM");
        writeMatrixBinary((double*)drAll,sizeAll2,"X2CATMDR");
        writeMatrixBinary((double*)diAll,sizeAll2,"X2CATMDI");
    }
    else
    {
	cout << "ERROR: PT interface has not been implemented" << endl;
	exit(99);
    }
    
    cout << "xx2camf FINISHED!" << endl;
    return 0;
}



void readZMAT(const string& filename, vector<string>& atoms, vector<string>& basis, vector<bool>& amfiMethod, double& SCFconv)
{
    bool readAtom = true, readBasis = true, GauNuc = false, PT = false;
    ifstream ifs;
    string flags, flags2, basisSet;
    ifs.open(filename);
    if(!ifs)
    {
        cout << "ERROR opening file " << filename << endl;
        exit(99);
    }
        cout << "Reading ZMAT..." << endl;
        getline(ifs,flags);
        cout << flags << endl;
        while (!ifs.eof())
        {
            getline(ifs,flags);
            cout << flags << endl;
            flags2 = removeSpaces(flags);
            if(flags2.size() != 0 && readAtom)
            {
                atoms.push_back(stringSplit(flags)[0]);
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
            cout << flags << endl;
            removeSpaces(flags);
            size_t found = flags.find("BASIS=");
            if(found != string::npos)
            {
                string tmp_s = flags.substr(found+6,flags.size()-6);
                if(tmp_s.substr(0,7) != "SPECIAL")
                {
                    for(int ii = 0; ii < atoms.size(); ii++)
                    {
                        basis.push_back(atoms[ii]+":"+tmp_s);
                    }
                    readBasis = false;
                }
            }
            found = flags.find("NUC_MODEL=GAUSSIAN");
            if(found != string::npos)
            {
                GauNuc = true;
            }
            found = flags.find("X2CKEY=X2CAMFP");
            if(found != string::npos)
            {
                PT = true;
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
            cout << flags << endl;
            size_t found = flags.find(":");
            if(found != string::npos)
            {
                flags = removeSpaces(flags);
                basis.push_back(flags);
                for(int ii = 1; ii < atoms.size(); ii++)
                {
                    getline(ifs,flags);
                    cout << flags << endl;
                    flags = removeSpaces(flags);
                    basis.push_back(flags);
                }
                break;
            }
        }
        while (!ifs.eof())
        {
            getline(ifs,flags);
            cout << flags << endl;
            removeSpaces(flags);
            if(flags.substr(0,12) == "%amfiMethod*")
            {   
                //average-of-configuration
                //gaunt
                //gauge
                //use entire 2e-picture-change-correction
                //Gaussian finite nuclear model
                for(int ii = 0 ; ii < 4; ii++)
                {
                    bool tmp;
                    ifs >> tmp;
                    amfiMethod.push_back(tmp);
                }
                amfiMethod.push_back(GauNuc);
                amfiMethod.push_back(PT);
                break;
            }
            else if(flags.substr(0,8) == "%atmconv")
            {
                ifs >> SCFconv;
            }
        }
        if(amfiMethod.size() == 0)
        {
            cout << "%amfiMethod is not found in ZMAT and set to default frac-DHF" << endl;
            amfiMethod.push_back(false); //aoc
            amfiMethod.push_back(false); //with gaunt
            amfiMethod.push_back(false); //with gauge
            amfiMethod.push_back(false); //normal integral rather than entire 2e-pcc
            amfiMethod.push_back(false); //without Gaussian nuclear model 
            amfiMethod.push_back(false); //calculate amfso integrals using variational approach 
        }
    ifs.close();

    return ;
}
