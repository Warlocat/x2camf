# X2CAMF
An efficient implementation of atomic relativistic four-component Hartree-Fock calculation
with spherical symmetry and two-electron Coulomb and Breit interaction. The X2CAMF program 
can generate spin-orbit integrals within the exact two-component theory with atomic-mean-field 
integral (the X2CAMF scheme, which has the same name as the program).

The X2CAMF program is a free open-source software.

For a detailed description of X2CAMF scheme and the implementation of spherical symmetry, 
please see https://pubs.acs.org/doi/pdf/10.1021/acs.jpca.2c02181.

The X2CAMF program has been embarrassingly interfaced with CFOUR and PySCF. 
The interface to PySCF makes use of pybind.

To install the program, you must have Eigen3 C++ and CMake version >= 3.9. One can modify 
the cmake list to adjust what to install.

    git submodule update --init --recursive  (to enable git submodule pybind)
    mkdir build
    cd build
    cmake ..
    make

Known problems:

    The default stack size is not enough for larger systems. Use "ulimit -s unlimited" to solve this. 

 

