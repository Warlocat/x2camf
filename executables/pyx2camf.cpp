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

namespace py = pybind11;

Eigen::MatrixXd amfi(const int input_string, const int atom_number,
                     const int nshell, const int nbas,
                     const Eigen::MatrixXi &shell,
                     const Eigen::MatrixXd &exp_a)
{
    // input_string is a internally coded string
    auto input_config = std::bitset<6>(input_string);
    bool spinFree = input_config[0];
    bool twoC = input_config[1];
    bool Gaunt = input_config[2];
    bool gauge = input_config[3];
    bool gauNuc = input_config[4];
    bool aoc = input_config[5];
    bool allint = true, renormS = false; // internal parameters, don't change.
    INT_SPH intor(atom_number, nshell, nbas, shell, exp_a);
    DHF_SPH *scfer;
    if (aoc)
    {
        scfer = new DHF_SPH_CA(intor, "input", spinFree, twoC, Gaunt, gauge, allint,
                               gauNuc);
    }
    else
    {
        scfer = new DHF_SPH_CA(intor, "input", spinFree, twoC, Gaunt, gauge, allint,
                               gauNuc);
    }
    scfer->convControl = 1e-10;
    scfer->runSCF(twoC, renormS);
    vMatrixXd amfi = scfer->get_amfi_unc(intor, twoC);
    MatrixXd amfi_all = Rotate::unite_irrep(amfi, intor.irrep_list);
    return amfi_all;
}

PYBIND11_MODULE(example, m)
{
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("amfi", &amfi, "A function to test");
}
