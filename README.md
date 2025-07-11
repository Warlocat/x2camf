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

To install the program, the quick way is to use pip.

    pip install git+https://github.com/warlocat/x2camf

Please let us konw if you have trouble pip installing. To manually install the library, you have to 
(the third line is to enable pybind11 and eigen)

    git clone https://github.com/Warlocat/x2camf.git
    cd x2camf
    git submodule update --init --recursive
    mkdir build
    cd build
    cmake ..
    make
    cp libx2camf.*.so ../x2camf

If you are not using pip, after compiling the library, you have to manually add the x2camf directory (NOT the x2camf/x2camf) to PYTHONPATH.

<br />
Known problems:

The default stack size might not be enough for larger systems due to the implementation. Use "ulimit -s unlimited" to solve this. 

 

