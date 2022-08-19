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
#include"finterface.h"
using namespace Eigen;
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
    string method = "";
    if(amfiMethod[0])   method = method + "aoc-HF Dirac-Coulomb";
    else method = method + "fractional-occupation Dirac-Coulomb";
    if(amfiMethod[1])   method = method + "-Gaunt";
    if(amfiMethod[2])   method = method + "-gauge\n";
    if(amfiMethod[4])   method = method + " with Gaussian nuclear model";
    if(amfiMethod[3])   method = "NOTHING: special for x2c1e calculation";
    cout << "amfi Method input: " << method << endl;

    if(amfiMethod[0])
    {
        cout << endl << endl;
        cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
        cout << "!!  WARNING: Average-of-configuration calculations are INCORRECT   !!" << endl;
        cout << "!!  for atoms with more than one partially occupied l-shell, e.g., !!" << endl;
        cout << "!!  uranium atom with both 5f and 6d partially occupied.           !!" << endl;
        cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
        cout << endl << endl;
    }

    vector<MatrixXcd> amfiUnique, XUnique;
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
        
        //amfiUnique.push_back(Rotate::unite_irrep(scfer.x2c2ePCC(),intor.irrep_list)); // for 2e-pcc test
        amfiUnique.push_back(Rotate::unite_irrep(scfer->get_amfi_unc(intor,twoC), intor.irrep_list));
        XUnique.push_back(Rotate::unite_irrep(scfer->get_X(), intor.irrep_list));
        
        MatrixXcd tmp = Rotate::jspinor2cfour_interface_old(intor.irrep_list);
        amfiUnique[ii] = tmp.adjoint() * amfiUnique[ii] * tmp;
        amfiUnique[ii] = Rotate::separate2mCompact(amfiUnique[ii],intor.irrep_list);
        XUnique[ii] = tmp.adjoint() * XUnique[ii] * tmp;
        XUnique[ii] = Rotate::separate2mCompact(XUnique[ii],intor.irrep_list);
    }

    cout << "Constructing amfso integrals...." << endl;
    int sizeAll = 0, int_tmp = 0;
    for(int ii = 0; ii < atomList.size(); ii++)
    {
        sizeAll += amfiUnique[indexList[ii]].rows();
    }
    int sizeAll2 = sizeAll*sizeAll, sizeHalf = sizeAll/2;
    F_INTERFACE::f_dcomplex amfiAll[sizeAll*sizeAll], XAll[sizeAll*sizeAll];
    for(int ii = 0; ii < sizeAll2; ii++)
    {
        amfiAll[ii].dr = 0.0;
        amfiAll[ii].di = 0.0;
        XAll[ii].dr = 0.0;
        XAll[ii].di = 0.0;
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

            XAll[(int_tmp+mm)*sizeAll + int_tmp+nn].dr = XUnique[indexList[ii]](nn,mm).real();
            XAll[(int_tmp+mm)*sizeAll + int_tmp+nn].di = XUnique[indexList[ii]](nn,mm).imag();
            XAll[(int_tmp+mm+sizeHalf)*sizeAll + int_tmp+nn].dr = XUnique[indexList[ii]](nn,size_tmp_half+mm).real();
            XAll[(int_tmp+mm+sizeHalf)*sizeAll + int_tmp+nn].di = XUnique[indexList[ii]](nn,size_tmp_half+mm).imag();
            XAll[(int_tmp+mm)*sizeAll + int_tmp+nn+sizeHalf].dr = XUnique[indexList[ii]](size_tmp_half+nn,mm).real();
            XAll[(int_tmp+mm)*sizeAll + int_tmp+nn+sizeHalf].di = XUnique[indexList[ii]](size_tmp_half+nn,mm).imag();
            XAll[(int_tmp+mm+sizeHalf)*sizeAll + int_tmp+nn+sizeHalf].dr = XUnique[indexList[ii]](size_tmp_half+nn,size_tmp_half+mm).real();
            XAll[(int_tmp+mm+sizeHalf)*sizeAll + int_tmp+nn+sizeHalf].di = XUnique[indexList[ii]](size_tmp_half+nn,size_tmp_half+mm).imag();
        }
        int_tmp += amfiUnique[indexList[ii]].rows()/2;
    }

    int sizeAllReal = 2*sizeAll2;
    if(amfiMethod[3])
    {
        for(int ii = 0; ii < sizeAll2; ii++)
        {
            amfiAll[ii].dr = 0.0;
            amfiAll[ii].di = 0.0;
        }
    }
    cout << "Writing amfso integrals...." << endl;
    if(!PT)
    {
        F_INTERFACE::wfile_("X2CMFSOM",(double*)amfiAll,&sizeAllReal);
        F_INTERFACE::wfile_("X2CATMXM",(double*)XAll,&sizeAllReal);
    }
    else
    {
        int sizePT = sizeAll/2*sizeAll/2;
        double amfiX[sizePT],amfiY[sizePT],amfiZ[sizePT];
        for(int ii = 0; ii < sizePT; ii++)
        {
            amfiX[ii] = 0.0;
            amfiY[ii] = 0.0;
            amfiZ[ii] = 0.0;
        }
        int_tmp = 0;
        for(int ii = 0; ii < atomList.size(); ii++)
        {
            int size_tmp_half = amfiUnique[indexList[ii]].rows()/2;
            for(int mm = 0; mm < size_tmp_half; mm++)
            for(int nn = 0; nn < size_tmp_half; nn++)
            {
                // separate X, Y, Z components
                amfiZ[(int_tmp+mm)*sizeAll/2 + int_tmp+nn] = amfiUnique[indexList[ii]](nn,mm).imag();
                amfiY[(int_tmp+mm)*sizeAll/2 + int_tmp+nn] = amfiUnique[indexList[ii]](nn,size_tmp_half+mm).real();
                amfiX[(int_tmp+mm)*sizeAll/2 + int_tmp+nn] = amfiUnique[indexList[ii]](nn,size_tmp_half+mm).imag();
            }
            int_tmp += amfiUnique[indexList[ii]].rows()/2;
        }
        F_INTERFACE::wfile_("XX2CSOCM",(double*)amfiX,&sizePT);
        F_INTERFACE::wfile_("YX2CSOCM",(double*)amfiY,&sizePT);
        F_INTERFACE::wfile_("ZX2CSOCM",(double*)amfiZ,&sizePT);
    }
    
    

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
                //set all integrals to zero
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
            amfiMethod.push_back(false); //normal integral
            amfiMethod.push_back(false); //without Gaussian nuclear model 
            amfiMethod.push_back(false); //calculate amfso integrals using variational approach 
        }
    ifs.close();

    return ;
}
