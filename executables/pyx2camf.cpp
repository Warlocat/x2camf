#include "dhf_sph.h"
#include "dhf_sph_ca.h"
#include "general.h"
#include "int_sph.h"
#include <Eigen/Core>
#include <Eigen/LU>
#include <bitset>
#include <vector>
#include <iostream>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace Eigen;
using namespace std;

namespace py = pybind11;

// SOC integrals within X2CAMF scheme
vector<MatrixXd> amfi(const int input_string, const int atom_number,
                      const int nshell, const int nbas, const int printLevel,
                      const Eigen::MatrixXi &shell,
                      const Eigen::MatrixXd &exp_a)
{
    // input_string is a internally coded string
    auto input_config = std::bitset<8>(input_string);
    bool Gaunt = input_config[0];
    bool gauge = input_config[1];
    bool gauNuc = input_config[2];
    bool aoc = input_config[3];
    bool pt = input_config[4];
    bool pcc = input_config[5];
    bool int4c = input_config[6];
    bool sdGaunt = input_config[7];
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
    if(!pcc) amfi = scfer->get_amfi_unc(intor, twoC, "partialFock", Gaunt, gauge, int4c, sdGaunt);
    else amfi = scfer->x2c2ePCC(int4c);
    MatrixXd amfi_all;
    if(int4c) amfi_all = Rotate::unite_irrep_4c(amfi, intor.irrep_list);
    else amfi_all = Rotate::unite_irrep(amfi, intor.irrep_list);

    vector<MatrixXd> results;
    results.push_back(amfi_all);

    if(printLevel >= 4)
    {
        cout << "libx2camf.amfi finished normally." << endl;
    }

    delete scfer;
    return results;
}

vector<MatrixXd> atm_integrals(const int input_string, const int atom_number,
                               const int nshell, const int nbas, const int printLevel,
                               const Eigen::MatrixXi &shell,
                               const Eigen::MatrixXd &exp_a)
{
    // input_string is a internally coded string
    auto input_config = std::bitset<8>(input_string);
    bool Gaunt = input_config[0];
    bool gauge = input_config[1];
    bool gauNuc = input_config[2];
    bool aoc = input_config[3];
    bool sdGaunt = input_config[4];
    bool allint = true, renormS = false, spinFree = false, twoC = false; // internal parameters, don't change.
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
    
    auto so_2c = Rotate::unite_irrep(scfer->get_amfi_unc(intor, twoC, "partialFock", Gaunt, gauge, false, sdGaunt), intor.irrep_list);
    // The den_fw should be called after so_2c, since the X and R matrices are not available before that.
    vMatrixXd den_fw = scfer->get_density_fw();
    auto fock_2c = Rotate::unite_irrep(scfer->get_fock_fw(), intor.irrep_list);
    auto den_2c = Rotate::unite_irrep(den_fw, intor.irrep_list);
    //
    auto fock_2c_2e = Rotate::unite_irrep(scfer->get_fock_4c_2ePart(den_fw, true), intor.irrep_list);
    auto fock_2c_K = Rotate::unite_irrep(scfer->get_fock_4c_K(den_fw, true), intor.irrep_list);

    auto atm_X = Rotate::unite_irrep(scfer->get_X(), intor.irrep_list);
    auto atm_R = Rotate::unite_irrep(scfer->get_R(), intor.irrep_list);

    auto so_4c = Rotate::unite_irrep_4c(scfer->get_amfi_unc(intor, twoC, "partialFock", Gaunt, gauge, true, sdGaunt), intor.irrep_list);
    auto fock_4c = Rotate::unite_irrep_4c(scfer->get_fock_4c(), intor.irrep_list);
    auto den_4c = Rotate::unite_irrep_4c(scfer->get_density(), intor.irrep_list);
    auto h1e_4c = Rotate::unite_irrep_4c(scfer->get_h1e_4c(), intor.irrep_list);
    auto fock_4c_2e = Rotate::unite_irrep_4c(scfer->get_fock_4c_2ePart(scfer->get_density(), false), intor.irrep_list);
    auto fock_4c_K = Rotate::unite_irrep_4c(scfer->get_fock_4c_K(scfer->get_density(), false), intor.irrep_list);

    delete scfer;
    
    /*
        fock_4c: 4c Fock matrix (h1e_4c + fock_4c_2e)
        h1e_4c: 4c one-electron Hamiltonian
        fock_4c_2e: 4c effective two-electron Veff
        fock_4c_K: 4c exchange part of Coulomb and the entire Breit term
        so_4c: Spin-dependent Coulomb and the entire Breit term
        den_4c: 4c density matrix

        so_2c is the FW transformed so_4c
        fock_2c is the FW transformed fock_4c
        den_2c is the "FW transformed" den_4c (NOT the 2c density matrix from the SCF procedure)
        fock_2c_2e: 2c Veff obtained using den_2c (NOT the FW transformed fock_4c_2e)
        fock_2c_K: 2c exchange obtained using den_2c (NOT the FW transformed fock_4c_K)

        atm_X: X matrix from atomic fock_4c
        atm_R: R matrix from atm_X
    */
    
    vector<MatrixXd> results{atm_X, atm_R, h1e_4c, fock_4c, fock_2c, fock_4c_2e, fock_2c_2e, fock_4c_K, fock_2c_K, so_4c, so_2c, den_4c, den_2c};

    if(printLevel >= 4)
    {
        cout << "libx2camf.atm_integrals finished normally." << endl;
    }

    return results;
}

vector<MatrixXd> pcc_K(const int input_string, const int atom_number,
                       const int nshell, const int nbas, const int printLevel,
                       const Eigen::MatrixXi &shell,
                       const Eigen::MatrixXd &exp_a)
{
    // input_string is a internally coded string
    auto input_config = std::bitset<8>(input_string);
    bool Gaunt = input_config[0];
    bool gauge = input_config[1];
    bool gauNuc = input_config[2];
    bool aoc = input_config[3];
    bool pt = input_config[4];
    bool pcc = input_config[5];
    bool int4c = input_config[6];
    bool sdGaunt = input_config[7];
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
    vMatrixXd amfi = scfer->x2c2ePCC_K(int4c);
    MatrixXd amfi_all;
    if(int4c) amfi_all = Rotate::unite_irrep_4c(amfi, intor.irrep_list);
    else amfi_all = Rotate::unite_irrep(amfi, intor.irrep_list);

    vector<MatrixXd> results;
    results.push_back(amfi_all);

    if(printLevel >= 4)
    {
        cout << "libx2camf.pcc_k finished normally." << endl;
    }

    delete scfer;
    return results;
}



PYBIND11_MODULE(libx2camf, m)
{
    m.doc() = "Python Interface to X2CAMF code."; // optional module docstring
    m.def("amfi", &amfi, "Compute the AMFI matrix");
    m.def("atm_integrals", &atm_integrals, "Compute all the atomic integrals");
    m.def("pcc_K", &pcc_K, "Compute the exchange-only 2e-PCC matrix");
}
