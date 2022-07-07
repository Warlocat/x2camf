#ifndef FINTERFACE
#define FINTERFACE

/*
    Fortran interface read and write binary file
*/
namespace F_INTERFACE
{
    struct f_dcomplex
    {
        double dr,di;
    };
    
    extern "C" void wfile_(char* file, double* CORE, int* LENGTH);
    extern "C" void rfile_(char* file, double* CORE, int* LENGTH);
    extern "C" void prvecr_(double* CORE, int* LENGTH);
} // namespace F_INTERFACE

#endif