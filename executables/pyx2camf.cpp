#include "dhf_sph.h"
#include "dhf_sph_ca.h"
#include "general.h"
#include "int_sph.h"
#include <Eigen/Core>
#include <Eigen/LU>
#include <bitset>
#include <iostream>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

using namespace Eigen;
using namespace std;

namespace py = pybind11;

// SOC integrals within X2CAMF scheme
Eigen::MatrixXd amfi(const int input_string, const int atom_number,
                        const int nshell, const int nbas, const int printLevel,
                        const Eigen::MatrixXi &shell,
                        const Eigen::MatrixXd &exp_a)
{
    // input_string is a internally coded string
    auto input_config = std::bitset<7>(input_string);
    bool Gaunt = input_config[0];
    bool gauge = input_config[1];
    bool gauNuc = input_config[2];
    bool aoc = input_config[3];
    bool pt = input_config[4];
    bool pcc = input_config[5];
    bool int4c = input_config[6];
    bool allint = true, renormS = false, spinFree, twoC; // internal parameters, don't change.
    if(pt)
    {
        spinFree = true; twoC = true;
        if(pcc)
        {
            cout << "ERROR: PCC is not implemented for PT scheme." << endl;
            exit(99);
        }
    }
    else
    {
        spinFree = false; twoC = false;
    }
    Eigen::VectorXi shell_vec(nbas);
    Eigen::VectorXd exp_a_vec(nbas);
    for (int i = 0; i < nbas; i++){
        shell_vec(i) = shell(i,0);
        exp_a_vec(i) = exp_a(i,0);
        if(printLevel >= 4) cout << shell(i,0) << " " << exp_a(i,0) << endl;
    }
    INT_SPH intor(atom_number, nshell, nbas, shell, exp_a);
    DHF_SPH *scfer;
    if (aoc)
    {
        cout << endl << endl;
        cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
        cout << "!!  WARNING: Average-of-configuration calculations MIGHT BE WRONG  !!" << endl;
        cout << "!!  for atoms with more than one partially occupied l-shell, e.g., !!" << endl;
        cout << "!!  uranium atom with both 5f and 6d partially occupied.           !!" << endl;
        cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
        cout << endl << endl;
        scfer = new DHF_SPH_CA(intor, "input", printLevel, spinFree, twoC, Gaunt, gauge, allint,
                               gauNuc);
    }
    else
    {
        scfer = new DHF_SPH(intor, "input", printLevel, spinFree, twoC, Gaunt, gauge, allint,
                            gauNuc);
    }
    scfer->convControl = 1e-9;
    scfer->runSCF(twoC, renormS);
    vMatrixXd amfi;
    if(!pcc) amfi = scfer->get_amfi_unc(intor, twoC, "partialFock", Gaunt, gauge, int4c);
    else amfi = scfer->x2c2ePCC(int4c);
    MatrixXd amfi_all;
    if(int4c) amfi_all = Rotate::unite_irrep_4c(amfi, intor.irrep_list);
    else amfi_all = Rotate::unite_irrep(amfi, intor.irrep_list);

    if(printLevel >= 4)
    {
        cout << "libx2camf.amfi finished normally." << endl;
    }

    delete scfer;
    return amfi_all;
}

PYBIND11_MODULE(libx2camf, m)
{
    m.doc() = "Python Interface to X2CAMF code."; // optional module docstring
    m.def("amfi", &amfi, "Compute the AMFI matrix");
}
