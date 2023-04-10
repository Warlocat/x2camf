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
void x2camf(const vector<bool>& amfiMethod, const vector<string>& atomListUnique, const vector<string>& basisListUnique, 
            const vector<string>& atomList, const vector<int>&  indexList, const double& amfiSCFconv);
void x2camfp(const vector<bool>& amfiMethod, const vector<string>& atomListUnique, const vector<string>& basisListUnique, 
             const vector<string>& atomList, const vector<int>&  indexList, const double& amfiSCFconv);
vectorcd rotate4cfour(const vectorcd& inputM, const vector<irrep_jm>& irrep_list, bool newInterface = true);
void writeAMFI(const vector<vectorcd>& mUnique, const vector<string>& atomList, const vector<int>&  indexList, const string& filename);
void writeAMFI_scalar(const vVectorXd& mUnique, const vector<string>& atomList, const vector<int>&  indexList, const string& filename);


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
    if(PT)
    {
        cout << "*******************************" << endl;
        cout << "**  Perturbative treatment   **" << endl;
        cout << "*******************************" << endl;
        if(amfiMethod[3])
        {
            cout << "Perturbative treatment for 2ePCC has NOT been implemented." << endl;
            exit(99);
        }
        x2camfp(amfiMethod, atomListUnique, basisListUnique, atomList, indexList, amfiSCFconv);
    }
    else
    {
        cout << "*******************************" << endl;
        cout << "**   Variational treatment   **" << endl;
        cout << "*******************************" << endl;
        x2camf(amfiMethod, atomListUnique, basisListUnique, atomList, indexList, amfiSCFconv);
    }
    
    cout << "xx2camf FINISHED!" << endl;
    return 0;
}



void readZMAT(const string& filename, vector<string>& atoms, vector<string>& basis, vector<bool>& amfiMethod, double& SCFconv)
{
    bool readAtom = true, readBasis = true, GauNuc = false, PT = false, x2ckeyFound = false;
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
            if(flags.substr(0,7) == "*CFOUR(" || flags.substr(0,7) == "*ACES2(")
            {
                break;
            }
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
                    if(tmp_s.substr(tmp_s.size()-1,1) == ")")
                        tmp_s = tmp_s.substr(0,tmp_s.size()-1);
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
                x2ckeyFound = true;
            }
            found = flags.find("X2CKEY=SOATMX");
            if(found != string::npos)
            {
                PT = false;
                x2ckeyFound = true;
            }
            found = flags.find(")");
            if(found != string::npos)
            {
                if(!x2ckeyFound)
                {
                    cout << "WARNING!!! X2CKEY not recognized in the ZMAT." << endl;
                    cout << "Using default X2CKEY=SOATMX (variational) option." << endl;
                    PT = false;
                }
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
        if(basis.size() != atoms.size())
        {
            cout << "ERROR: Number of atoms and number of basis sets do not match." << endl;
            exit(99);
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
                    cout << tmp << endl;
                    amfiMethod.push_back(tmp);
                }
                cout << endl;
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
            amfiMethod.push_back(false); //fractional occupation
            amfiMethod.push_back(false); //without gaunt
            amfiMethod.push_back(false); //without gauge
            amfiMethod.push_back(false); //normal integral rather than entire 2e-pcc
            amfiMethod.push_back(GauNuc);
            amfiMethod.push_back(PT); 
        }
    ifs.close();

    return ;
}

void x2camf(const vector<bool>& amfiMethod, const vector<string>& atomListUnique, const vector<string>& basisListUnique, 
            const vector<string>& atomList, const vector<int>&  indexList, const double& amfiSCFconv)
{
    vector<vectorcd> amfiUnique, XUnique;//     denUnique;
    for(int ii = 0; ii < atomListUnique.size(); ii++)
    {
        INT_SPH intor(atomListUnique[ii],basisListUnique[ii]);
        DHF_SPH *scfer;
        /* Variational treatment */
        if(amfiMethod[0])
            /* Average of configuration */
            scfer = new DHF_SPH_CA(intor, "ZMAT", false, false, amfiMethod[1], amfiMethod[2], true, amfiMethod[4]);
        else
            /* Fractional occupation */
            scfer = new DHF_SPH(intor, "ZMAT", false, false, amfiMethod[1], amfiMethod[2], true, amfiMethod[4]);
        scfer->convControl = amfiSCFconv;
        scfer->runSCF(false);
        if(amfiMethod[3])
            amfiUnique.push_back(real2complex(Rotate::unite_irrep(scfer->x2c2ePCC(),intor.irrep_list))); // for 2e-pcc test
        else
            amfiUnique.push_back(real2complex(Rotate::unite_irrep(scfer->get_amfi_unc(intor,false), intor.irrep_list)));
        XUnique.push_back(real2complex(Rotate::unite_irrep(scfer->get_X(), intor.irrep_list)));
        delete scfer;

        amfiUnique[ii] = rotate4cfour(amfiUnique[ii], intor.irrep_list);
        XUnique[ii] = rotate4cfour(XUnique[ii], intor.irrep_list);
    }

    cout << "Writing amfso integrals...." << endl;
    writeAMFI(amfiUnique, atomList, indexList, "X2CMFSOM");
    writeAMFI(XUnique, atomList, indexList, "X2CATMXM");

    return;
}

void x2camfp(const vector<bool>& amfiMethod, const vector<string>& atomListUnique, const vector<string>& basisListUnique, 
             const vector<string>& atomList, const vector<int>&  indexList, const double& amfiSCFconv)
{
    vVectorXd lls, llx, lly, llz, lss, lsx, lsy, lsz, sls, slx, sly, slz, sss, ssx, ssy, ssz;
    for(int ii = 0; ii < atomListUnique.size(); ii++)
    {
        vectorcd amfi4c, soll, sosl, sols, soss;
        INT_SPH intor(atomListUnique[ii],basisListUnique[ii]);
        DHF_SPH *scfer;
        /* Perturbative treatment */
        if(amfiMethod[0])
            /* Average of configuration */
            scfer = new DHF_SPH_CA(intor, "ZMAT", true, true, amfiMethod[1], amfiMethod[2], true, amfiMethod[4]);
        else
            /* Fractional occupation */
            scfer = new DHF_SPH(intor, "ZMAT", true, true, amfiMethod[1], amfiMethod[2], true, amfiMethod[4]);
        scfer->convControl = amfiSCFconv;
        scfer->runSCF(true,false);
        amfi4c = real2complex(Rotate::unite_irrep_4c(scfer->get_amfi_unc(intor,true,"partialFock",amfiMethod[1],amfiMethod[2],true), intor.irrep_list));
        int size = round(sqrt(amfi4c.size()));
        soll = matBlock(amfi4c, size, 0, 0, size/2, size/2);
        sosl = matBlock(amfi4c, size, size/2, 0, size/2, size/2);
        sols = matBlock(amfi4c, size, 0, size/2, size/2, size/2);
        soss = matBlock(amfi4c, size, size/2, size/2, size/2, size/2);
        soll = rotate4cfour(soll, intor.irrep_list, false);
        sosl = rotate4cfour(sosl, intor.irrep_list, false);
        sols = rotate4cfour(sols, intor.irrep_list, false);
        soss = rotate4cfour(soss, intor.irrep_list, false);

        for(int mm = 0; mm < size/4; mm++)
        for(int nn = 0; nn < size/4; nn++)
        {
            /*
                amfi4c:
                La Sa Lb Sb
            */
            amfi4c[mm*size+nn] = soll[mm*size/2+nn]; // LaLa
            amfi4c[(mm+size/4)*size+nn] = sosl[mm*size/2+nn]; //SaLa
            amfi4c[(mm+size/2)*size+nn] = soll[(mm+size/4)*size/2+nn]; //LbLa
            amfi4c[(mm+size/2+size/4)*size+nn] = sosl[(mm+size/4)*size/2+nn]; //SbLa
            amfi4c[mm*size+nn+size/4] = sols[mm*size/2+nn]; //LaSa
            amfi4c[(mm+size/4)*size+nn+size/4] = soss[mm*size/2+nn]; //SaSa
            amfi4c[(mm+size/2)*size+nn+size/4] = sols[(mm+size/4)*size/2+nn]; //LbSa
            amfi4c[(mm+size/2+size/4)*size+nn+size/4] = soss[(mm+size/4)*size/2+nn]; //SbSa
            amfi4c[mm*size+nn+size/2] = soll[mm*size/2+nn+size/4]; //LaLb
            amfi4c[(mm+size/4)*size+nn+size/2] = sosl[mm*size/2+nn+size/4]; //SaLb
            amfi4c[(mm+size/2)*size+nn+size/2] = soll[(mm+size/4)*size/2+nn+size/4]; //LbLb
            amfi4c[(mm+size/2+size/4)*size+nn+size/2] = sosl[(mm+size/4)*size/2+nn+size/4]; //SbLb
            amfi4c[mm*size+nn+size/2+size/4] = sols[mm*size/2+nn+size/4]; //LaSb
            amfi4c[(mm+size/4)*size+nn+size/2+size/4] = soss[mm*size/2+nn+size/4]; //SaSb
            amfi4c[(mm+size/2)*size+nn+size/2+size/4] = sols[(mm+size/4)*size/2+nn+size/4]; //LbSb
            amfi4c[(mm+size/2+size/4)*size+nn+size/2+size/4] = soss[(mm+size/4)*size/2+nn+size/4]; //SbSb
        }
        vVectorXd sxyz = X2C::pauliDecompose(amfi4c, size);
        lls.push_back(matBlock(sxyz[0], size/2, 0, 0, size/4, size/4));
        llx.push_back(matBlock(sxyz[1], size/2, 0, 0, size/4, size/4));
        lly.push_back(matBlock(sxyz[2], size/2, 0, 0, size/4, size/4));
        llz.push_back(matBlock(sxyz[3], size/2, 0, 0, size/4, size/4));
        sls.push_back(matBlock(sxyz[0], size/2, size/4, 0, size/4, size/4));
        slx.push_back(matBlock(sxyz[1], size/2, size/4, 0, size/4, size/4));
        sly.push_back(matBlock(sxyz[2], size/2, size/4, 0, size/4, size/4));
        slz.push_back(matBlock(sxyz[3], size/2, size/4, 0, size/4, size/4));
        lss.push_back(matBlock(sxyz[0], size/2, 0, size/4, size/4, size/4));
        lsx.push_back(matBlock(sxyz[1], size/2, 0, size/4, size/4, size/4));
        lsy.push_back(matBlock(sxyz[2], size/2, 0, size/4, size/4, size/4));
        lsz.push_back(matBlock(sxyz[3], size/2, 0, size/4, size/4, size/4));
        sss.push_back(matBlock(sxyz[0], size/2, size/4, size/4, size/4, size/4));
        ssx.push_back(matBlock(sxyz[1], size/2, size/4, size/4, size/4, size/4));
        ssy.push_back(matBlock(sxyz[2], size/2, size/4, size/4, size/4, size/4));
        ssz.push_back(matBlock(sxyz[3], size/2, size/4, size/4, size/4, size/4));
    }

    cout << "Writing amfso integrals...." << endl;
    writeAMFI_scalar(lls, atomList, indexList, "SOC4CLLS");
    writeAMFI_scalar(llx, atomList, indexList, "SOC4CLLX");
    writeAMFI_scalar(lly, atomList, indexList, "SOC4CLLY");
    writeAMFI_scalar(llz, atomList, indexList, "SOC4CLLZ");
    writeAMFI_scalar(sls, atomList, indexList, "SOC4CSLS");
    writeAMFI_scalar(slx, atomList, indexList, "SOC4CSLX");
    writeAMFI_scalar(sly, atomList, indexList, "SOC4CSLY");
    writeAMFI_scalar(slz, atomList, indexList, "SOC4CSLZ");
    writeAMFI_scalar(lss, atomList, indexList, "SOC4CLSS");
    writeAMFI_scalar(lsx, atomList, indexList, "SOC4CLSX");
    writeAMFI_scalar(lsy, atomList, indexList, "SOC4CLSY");
    writeAMFI_scalar(lsz, atomList, indexList, "SOC4CLSZ");
    writeAMFI_scalar(sss, atomList, indexList, "SOC4CSSS");
    writeAMFI_scalar(ssx, atomList, indexList, "SOC4CSSX");
    writeAMFI_scalar(ssy, atomList, indexList, "SOC4CSSY");
    writeAMFI_scalar(ssz, atomList, indexList, "SOC4CSSZ");

    return;
}

vectorcd rotate4cfour(const vectorcd& inputM, const vector<irrep_jm>& irrep_list, bool newInterface)
{
    vectorcd tmp;
    if(newInterface)
        tmp = Rotate::jspinor2cfour_interface_new(irrep_list);
    else
        tmp = Rotate::jspinor2cfour_interface_old(irrep_list);
    vectorcd tmp1,tmp2;
    int NN = round(sqrt(tmp.size()));
    zgemm_itrf('c','n',NN,NN,NN,one_cp,tmp,inputM,zero_cp,tmp1);
    zgemm_itrf('n','n',NN,NN,NN,one_cp,tmp1,tmp,zero_cp,tmp2);
    return Rotate::separate2mCompact(tmp2,irrep_list);
}

void writeAMFI(const vector<vectorcd>& mUnique, const vector<string>& atomList, const vector<int>&  indexList, const string& filename)
{
    int sizeAll = 0, int_tmp = 0;
    vector<int> sizeList(atomList.size());
    for(int ii = 0; ii < atomList.size(); ii++)
    {
        sizeList[ii] = round(sqrt(mUnique[indexList[ii]].size()));
        sizeAll += sizeList[ii];
    }
    int sizeAll2 = sizeAll*sizeAll, sizeHalf = sizeAll/2;
    complex<double> mAll[sizeAll*sizeAll];
    for(int ii = 0; ii < sizeAll2; ii++)
    {
        mAll[ii] = zero_cp;
    }
    for(int ii = 0; ii < atomList.size(); ii++)
    {
        int size_tmp_half = sizeList[ii]/2;
        for(int mm = 0; mm < size_tmp_half; mm++)
        for(int nn = 0; nn < size_tmp_half; nn++)
        {
            // transpose for Fortran interface
            // separate alpha and beta
            mAll[(int_tmp+mm)*sizeAll + int_tmp+nn] = mUnique[indexList[ii]][nn*sizeList[ii]+mm];
            mAll[(int_tmp+mm+sizeHalf)*sizeAll + int_tmp+nn] = mUnique[indexList[ii]][nn*sizeList[ii]+size_tmp_half+mm];
            mAll[(int_tmp+mm)*sizeAll + int_tmp+nn+sizeHalf] = mUnique[indexList[ii]][(size_tmp_half+nn)*sizeList[ii]+mm];
            mAll[(int_tmp+mm+sizeHalf)*sizeAll + int_tmp+nn+sizeHalf] = mUnique[indexList[ii]][(size_tmp_half+nn)*sizeList[ii]+size_tmp_half+mm];
        }
        int_tmp += sizeList[ii]/2;
    }

    int sizeAllReal = 2*sizeAll2;
    writeMatrixBinary((double*)mAll, sizeAllReal, filename);
    return;
}
void writeAMFI_scalar(const vVectorXd& mUnique, const vector<string>& atomList, const vector<int>&  indexList, const string& filename)
{
    int sizeAll = 0, int_tmp = 0;
    vector<int> sizeList(atomList.size());
    for(int ii = 0; ii < atomList.size(); ii++)
    {
        sizeList[ii] = round(sqrt(mUnique[indexList[ii]].size()));
        sizeAll += sizeList[ii];
    }
    int sizeAll2 = sizeAll*sizeAll;
    double mAll[sizeAll*sizeAll];
    for(int ii = 0; ii < sizeAll2; ii++)
    {
        mAll[ii] = 0.0;
    }
    for(int ii = 0; ii < atomList.size(); ii++)
    {
        for(int mm = 0; mm < sizeList[ii]; mm++)
        for(int nn = 0; nn < sizeList[ii]; nn++)
        {
            // transpose for Fortran interface
            mAll[(int_tmp+mm)*sizeAll + int_tmp+nn] = mUnique[indexList[ii]][nn*sizeList[ii]+mm];
        }
        int_tmp += sizeList[ii];
    }

    writeMatrixBinary((double*)mAll, sizeAll2, filename);
    return;
}
