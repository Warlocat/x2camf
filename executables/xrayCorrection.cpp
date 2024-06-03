#include<iostream>
#include<fstream>
#include<string>
#include<Eigen/Dense>
#include<iomanip>
#include<cmath>
#include<cctype>
#include<ctime>
#include<memory>
#include"int_sph.h"
#include"dhf_sph.h"
#include"general.h"
#include"element.h"
#include"dhf_sph_ca.h"
#define toupperstr(x) transform(x.begin(), x.end(), x.begin(), ::toupper)
#define cleanScreen cout << "\033[2J\033[1;1H"
using namespace Eigen;
using namespace std;

/* Global information */
vector<vector<int>> coreHoleInfo;
string atomName, basisSet;
bool gaussian_nuc;
int methodVariant = -1, qedVariant = -1, nuc_model = -1;

void interactiveInput();

double QEDcorrection(const int& N, const int& L, const int& twoJ, const int& Z);
void correction();
void calculate(int methodVariant);
vector<string> parseInput(const string& inputString);
bool yesnoInput(const string& printInfo);
int numberInput(const string& printInfo, const int& max, const int& min = 1);
vector<string> findBasisSet(const string& atomName, const string& basisFile = "GENBAS");

int main()
{
    interactiveInput();
    gaussian_nuc = (nuc_model == 1);
    if (methodVariant == 0)
        correction();
    else
        calculate(methodVariant);
}

void calculate(int methodVariant)
{
    /*
        methodVariant
        1: spin-free X2C-1e
        2: X2C-1e
        3: spin-free Dirac-Coulomb-4c
        4: Dirac-Coulomb-4c
        5: Dirac-Coulomb-Gaunt-4c
        6: Dirac-Coulomb-Breit-4c
    */
    bool spinFree, twoC, with_gaunt, with_gauge, allInt = true;
    switch (methodVariant)
    {
    case 1:
        spinFree = true; twoC = true; with_gaunt = false; with_gauge = false;
        break;
    case 2:
        spinFree = false; twoC = true; with_gaunt = false; with_gauge = false;
        break;
    case 3:
        spinFree = true; twoC = false; with_gaunt = false; with_gauge = false;
        break;
    case 4:
        spinFree = false; twoC = false; with_gaunt = false; with_gauge = false;
        break;
    case 5:
        spinFree = false; twoC = false; with_gaunt = true; with_gauge = false;
        break;
    case 6:
        spinFree = false; twoC = false; with_gaunt = true; with_gauge = true;
        break;
    default:
        cout << "Invalid method variant!" << endl;
        exit(99);
        break;
    }

    double ene, ene_ip;
    INT_SPH intor(atomName, basisSet);
    DHF_SPH_CA neutral(intor,"input",4,spinFree,twoC,with_gaunt,with_gauge,true,gaussian_nuc);
    neutral.runSCF(twoC, false);
    ene = neutral.ene_scf;
    vVectorXd mo_ene_n = neutral.ene_orb;
    DHF_SPH_CA ionized(intor,"input",4,spinFree,twoC,with_gaunt,with_gauge,true,gaussian_nuc);
    ionized.coreIonization(coreHoleInfo);
    ionized.runSCF(twoC, false);
    ene_ip = ionized.ene_scf;

    cout << "Core ionization potential (in eV) to orbitals/spinors of " << endl;
    double ene_orb_ip = 0.0;
    for(int ihole = 0; ihole < coreHoleInfo.size(); ihole++)
    {
        int n = coreHoleInfo[ihole][0], l = coreHoleInfo[ihole][1], twoj = coreHoleInfo[ihole][2];
        for(int ir = 0; ir < intor.irrep_list.rows(); ir++)
        {
            if(intor.irrep_list(ir).l == l && intor.irrep_list(ir).two_j == twoj)
            {
                int n_tmp = n - 1 - l;
                if(twoC)
                    ene_orb_ip += mo_ene_n(ir)(n_tmp);
                else
                    ene_orb_ip += mo_ene_n(ir)(mo_ene_n(ir).rows()/2 + n_tmp);
                break;
            }
        }
        string orbitalName = to_string(n) + orbL[l] + "_" + to_string(twoj) + "/2";
        cout << orbitalName << endl;
    }

    cout << fixed << setprecision(2);
    cout << "using orbital energy/Koopman theorem: " << -ene_orb_ip*au2ev << endl;
    cout << "using Delta SCF: " << (ene_ip - ene)*au2ev << endl;
}

void correction()
{
    double sum_sfx2c = 0.0, sum_sfdc_sfx2c = 0.0, sum_dc_sfx2c = 0.0, sum_dcb_dc = 0.0, sum_qed = 0.0;
    double sfx2c_orb_ene, dc_sfx2c, dcb_dc, sfdc_sfx2c, qed = 0.0;
    INT_SPH intor(atomName, basisSet);
    //                           sf   2c   Gaunt gauge allint  gauNuc
    DHF_SPH sfx2c(intor,"input",4,true,true,false,false,true,gaussian_nuc);
    DHF_SPH sfdc4c(intor,"input",4,true,false,false,false,true,gaussian_nuc);
    DHF_SPH dc4c(intor,"input",4,false,false,false,false,true,gaussian_nuc);
    DHF_SPH dcb4c(intor,"input",4,false,false,true,true,true,gaussian_nuc);

    sfx2c.runSCF(true,false);
    sfdc4c.runSCF(false,false);
    dc4c.runSCF(false,false);
    dcb4c.runSCF(false,false);
    
    vVectorXd mo_ene_sfx2c = sfx2c.ene_orb, mo_ene_dc = dc4c.ene_orb, mo_ene_dcb = dcb4c.ene_orb, mo_ene_sfdc = sfdc4c.ene_orb;
    auto irrep_list = intor.irrep_list;
    for(int ihole = 0; ihole < coreHoleInfo.size(); ihole++)
    {
        int n = coreHoleInfo[ihole][0], l = coreHoleInfo[ihole][1], twoj = coreHoleInfo[ihole][2];
        for(int ir = 0; ir < irrep_list.rows(); ir++)
        {
            if(irrep_list(ir).l == l && irrep_list(ir).two_j == twoj)
            {
                int n_tmp = n - 1 - l, n2c = mo_ene_sfx2c(ir).rows();
                sfx2c_orb_ene = mo_ene_sfx2c(ir)(n_tmp);
                dc_sfx2c = mo_ene_dc(ir)(n2c + n_tmp) - mo_ene_sfx2c(ir)(n_tmp);
                sfdc_sfx2c = mo_ene_sfdc(ir)(n2c + n_tmp) - mo_ene_sfx2c(ir)(n_tmp);
                dcb_dc = mo_ene_dcb(ir)(n2c + n_tmp) - mo_ene_dc(ir)(n2c + n_tmp);
                break;
            }
        }
        sum_sfx2c += sfx2c_orb_ene;
        sum_sfdc_sfx2c += sfdc_sfx2c;
        sum_dc_sfx2c += dc_sfx2c;
        sum_dcb_dc += dcb_dc;
        if(qedVariant == 2)
        {
            qed = QEDcorrection(n,l,twoj,elem_map.find(atomName)->second);
            sum_qed += qed;
        }

        string orbitalName = to_string(n) + orbL[l] + "_" + to_string(twoj) + "/2";

        cout << "The correction (in eV) to ionization potential of " << orbitalName << endl;
        cout << fixed << setprecision(2);
        cout << "SFX2C-1e Koopman theorem:\t\t" << -sfx2c_orb_ene*au2ev << endl;
        cout << "SFDC - SFX2C1e (2e scalar picture change):\t\t" << -sfdc_sfx2c*au2ev << endl; 
        cout << "DC - SFX2C1e (2e picture change):\t\t" << -dc_sfx2c*au2ev << endl; 
        cout << "Breit term:\t\t" << -dcb_dc*au2ev << endl;
        if(qedVariant == 2)
        {
            cout << "QED term:\t\t" << -qed*au2ev << endl;
            cout << "2e picture change + Breit + QED:\t\t" << -(dcb_dc + dc_sfx2c + qed)*au2ev << endl;
        }
        else
        {
            cout << "2e picture change + Breit:\t\t" << -(dcb_dc + dc_sfx2c)*au2ev << endl;
        }
    }

    cout << "The total correction (in eV) to ionization potential of " << atomName << " is:" << endl;
    cout << fixed << setprecision(2);
    cout << "SFX2C-1e Koopman theorem:\t\t" << -sum_sfx2c*au2ev << endl;
    cout << "SFDC - SFX2C1e (2e scalar picture change):\t\t" << -sum_sfdc_sfx2c*au2ev << endl;
    cout << "DC - SFX2C1e (2e picture change):\t\t" << -sum_dc_sfx2c*au2ev << endl;
    cout << "Breit term:\t\t" << -sum_dcb_dc*au2ev << endl;
    if(qedVariant == 2)
    {
        cout << "QED term:\t\t" << -sum_qed*au2ev << endl;
        cout << "2e picture change + Breit + QED:\t\t" << -(sum_dcb_dc + sum_dc_sfx2c + sum_qed)*au2ev << endl;
    }
    else
    {
        cout << "2e picture change + Breit:\t\t" << -(sum_dcb_dc + sum_dc_sfx2c)*au2ev << endl;
    }
}


void interactiveInput()
{
    string flags;
    vector<string> parsed;
    cout << "**********************************************************************\n"
            "*   Atomic Core Ionization Potential (ACIP) program implemented by   *\n"
            "*   Chaoqun Zhang (bbzhchq@gmail.com).                               *\n"
            "*                                                                    *\n"
            "*   This program can be used to efficiently estimate atomic (core)   *\n"
            "*   ionization potential (IP) or calculate relativistic corrections  *\n"
            "*   to the IP. All the calculations running by this program assume   *\n"
            "*   spherical symmetry.                                              *\n"
            "**********************************************************************\n\n\n" << endl;
    cout << "You can terminate program by typing -1 or q at any time." << endl;
    cout << "Press enter to continue..." << endl;
    getline(cin, flags);
    if(flags == "-1" or flags == "q")
    {
        cout << "Exiting the program..." << endl;
        exit(-1);
    }
    cleanScreen;

    cout << "Please provide the following information:" << endl;
    cout << "Atom name (e.g. H, He, Li, ...): " << endl;
    getline(cin, flags);
    parsed = parseInput(flags);
    toupperstr(parsed[0]);
    atomName = parsed[0];
    cleanScreen;

    auto availBasis = findBasisSet(atomName);
    cout << "Basis set name (e.g. " + availBasis[0] + "...): " << endl;
    cout << "Input 0 to see all available basis sets." << endl;
    getline(cin, flags);
    parsed = parseInput(flags);
    if (parsed[0] == "0")
    {
        for(int ii = 0; ii < availBasis.size(); ii++)
        {
            cout << availBasis[ii] << endl;
        }
        cout << "Please input one basis set name from above (including " + atomName + ":)" << endl;
        getline(cin, flags);
        parsed = parseInput(flags);
    }
    toupperstr(parsed[0]);
    basisSet = parsed[0];
    bool found = false;
    while (true)
    {
        for(int ii = 0; ii < availBasis.size(); ii++)
        {
            if(availBasis[ii] == basisSet)
            {
                found = true;
                break;
            }
        }
        if(found)
            break;
        else
        {
            for(int ii = 0; ii < availBasis.size(); ii++)
            {
                cout << availBasis[ii] << endl;
            }
            cout << "Invalid basis set name!\n"
                    "Please input one basis set name from above (including " + atomName + ":)" << endl;
            getline(cin, flags);
            parsed = parseInput(flags);
            toupperstr(parsed[0]);
            basisSet = parsed[0];
        }
    }
    cleanScreen;

    nuc_model = numberInput("Nuclear model:\n"
                            "1. Point charge\n" 
                            "2. Gaussian Nuclear Model\n"
                            "3. Two-parameter Fermi Model (Woods-Saxon Potential)", 3);
    cleanScreen;

    qedVariant = numberInput("Method to include QED correction:\n"
                             "1. No QED\n"
                             "2. Fitted QED corrections\n"
                             "3. Variational Effective QED potential", 3);
    cleanScreen;
    

    methodVariant = numberInput("Method variant:\n"
                                "0. Relativistic corrections to core ionization potential\n"
                                "1. Spin-free X2C-1e Delta SCF\n"
                                "2. X2C-1e Delta SCF\n"
                                "3. Spin-free Dirac-Coulomb-4c Delta SCF\n"
                                "4. Dirac-Coulomb-4c Delta SCF\n"
                                "5. Dirac-Coulomb-Gaunt-4c Delta SCF\n"
                                "6. Dirac-Coulomb-Breit-4c Delta SCF\n\n"
                                "Attention:\n"
                                "Delta SCF (variant > 0) is not available for atoms which have open shell electrons\n"
                                "in the same irrep as the core hole. In this case, the program will automatically\n"
                                "terminate. For example, for uranium atom, the open shell electrons are in 5f and 6d\n"
                                "shells. The core hole in d and f shells thus can not be calculated using Delta SCF.\n",
    6, 0);
    cleanScreen;

    int nholes = numberInput("Number of core holes:", 5, 1);
    coreHoleInfo.resize(nholes);
    cout << "Core hole information:\n"
            "Please input information for each core hole in one line.\n"
            "The information should be in the following format (for each line):\n"
            "N L 2J\n"
            "where N is the principal quantum number, L is the azimuthal quantum number,\n"
            "2J is two times total angular quantum number. For example, if you want to\n"
            "calculate the L2-edge (2p_1/2) energy of some atom, you should input 2 1 1.\n" << endl;
    for(int ii = 0; ii < nholes; ii++)
    {
        while(true)
        {
            getline(cin, flags);
            parsed = parseInput(flags);
            if(parsed.size() == 3)
            {
                int n, l, tj;
                coreHoleInfo[ii].resize(3);
                try
                {
                    n = stoi(parsed[0]);
                    l = stoi(parsed[1]);
                    tj = stoi(parsed[2]);
                }
                catch(const invalid_argument& e)
                {
                    cout << "Invalid input! Please input N L 2J:" << endl;
                    continue;
                }
                coreHoleInfo[ii][0] = n;
                coreHoleInfo[ii][1] = l;
                coreHoleInfo[ii][2] = tj;
                break;
            }
            else
            {
                cout << "Invalid input! Please input N L 2J:" << endl;
                continue;
            }
        }
    }
    cleanScreen;

    if(nuc_model == 3)
    {
        cout << "Two-parameter Fermi Model is not available in this version." << endl;
        exit(99);
    }
    if(qedVariant == 3)
    {
        cout << "Variational Effective QED potential is not available in this version." << endl;
        exit(99);
    }
    else if(qedVariant == 2)
    {
        cout << "Using fitted QED correction documented in " << endl;
        cout << "K. Kozioł and G. A. Aucar, J. Chem. Phys., 2018, 148, 134101" << endl << endl;
    }
}



vector<string> parseInput(const string& inputString)
{
    vector<string> parsed = stringSplit(inputString);
    if(parsed.size() == 0)
    {
        cout << "Invalid empty input!" << endl;
        exit(99);
    }
    else if (parsed[0] == "-1" or parsed[0] == "q")
    {
        cout << "Exiting the program..." << endl;
        exit(-1);
    }
    return parsed;
}

vector<string> findBasisSet(const string& atomName, const string& basisFile)
{
    vector<string> availBasis;
    ifstream ifs;
    ifs.open(basisFile);
    cout << "Reading basis set file " + basisFile + "..." << endl;
    if(!ifs)
    {
        cout << "ERROR opening file " << basisFile << endl;
        exit(99);
    }
    string line;
    while (!ifs.eof())
    {
        getline(ifs, line);
        if(line.substr(0,atomName.size()+1) == atomName+":")
        {
            availBasis.push_back(removeSpaces(line));
        }
    }
    if(availBasis.size() == 0)
    {
        cout << "ERROR: can not find any basis sets for " + atomName + " in the basis set file (GENBAS)\n";
        exit(99);
    }
    ifs.close();

    return availBasis;
}

bool yesnoInput(const string& printInfo)
{
    string flags;
    cout << printInfo << endl;
    while(true)
    {
        cout << "Please input y or n:" << endl;
        getline(cin, flags);
        flags = removeSpaces(flags);
        if(flags == "y" or flags == "Y")
            return true;
        else if(flags == "n" or flags == "N")
            return false;
        else
            continue;
    }
}

int numberInput(const string& printInfo, const int& max, const int& min)
{
    string flags;
    cout << printInfo << endl;
    while(true)
    {
        cout << "Please input a number between " << min << " and " << max << ":" << endl;
        getline(cin, flags);
        flags = parseInput(flags)[0];
        int tmp_i;
        try
        {
            tmp_i = stoi(flags);
        }
        catch(const invalid_argument& e)
        {
            continue;
        }

        if(tmp_i <= max && tmp_i >= min)
            return tmp_i;
        else
            continue;
    }
}

double QEDcorrection(const int& N, const int& L, const int& twoJ, const int& Z)
{
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