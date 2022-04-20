# AMFSO
An efficient implementation of atomic relativistic Hartree-Fock calculation to 
generator spin-orbit integrals based on atomic-mean-field approach.

AMFSO has been interfaced with CFOUR and PySCF.
 
AMFSO is a free software, you can redistribute it and/or modify it under
the terms of the GNU General Public License.

It supports:

    Various one-electron and two-electron (J,K) integrals in j-adapted spinor basis set for single atom.

    Four-component and two-component atomic Hartree-Fock calculation with two-electron Coulomb and Breit interaction.

    Generate spin-orbit and Breit integrals in spinor basis based on the X2CAMF approach, 
    see Liu, J.; Cheng, L. J. Chem. Phys. 2018, 148, 144108. 
    and Zhang, C.; Cheng, L. J. Phys. Chem. A. (in press)


To compile and use, you must have:

    Eigen3 C++


