# amfso
Spin-orbit coupling (SOC) integrals generator with atomic-mean-field (AMF) approximation 
for both perturbative and non-perturbative treatment of SOC. This program aims to make 
full use of spherical symmetry (take |J,mj> as an irrep) of atoms.


It supports:

    Various one-electron and two-electron (J,K) integrals in j-adapted spinor basis set for single atom;

    Spin-free exact-two-component in one-electron variant Hartree-Fock (SFX2C1E-HF) for single atom;

    Dirac-Coulomb Hartree-Fock (DCHF) and spin-free DCHF for single atom;

    Generate spin-orbit coupling integrals based on AMF approximation with density matrix from above methods.


To use, you must have:

    Eigen3 C++

    GSL library



