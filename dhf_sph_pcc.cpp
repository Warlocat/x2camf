#include"general.h"
#include"dhf_sph.h"
#include"dhf_sph_ca.h"
#include<iomanip>
#include<iostream>
using namespace std;

vVectorXd DHF_SPH::x2c2ePCC(vVectorXd* coeff2c)
{
    cout << "Running DHF_SPH::x2c2ePCC" << endl;
    if(!converged)
    {
        cout << "SCF did not converge. x2c2ePCC cannot be used!" << endl;
        exit(99);
    }
    
    //Special case for H-like atoms.
    if(abs(nelec-1.0)<1e-5)
    {
        for(int ir = 0; ir < occMax_irrep; ir++)
        {
            fock_4c[ir] = h1e_4c[ir];
        }
    }

    vVectorXd fock_pcc(occMax_irrep);
    vVectorXd XXX(occMax_irrep), RRR(occMax_irrep), XXX_1e(occMax_irrep), RRR_1e(occMax_irrep);
    vVectorXd overlap_2c(occMax_irrep), fock_4c_2e(occMax_irrep), JK_x2c2c(occMax_irrep), coeff_2c(occMax_irrep), fock_x2c2e(occMax_irrep), overlap_h_i_2c(occMax_irrep), density_2c(occMax_irrep), h1e_x2c1e(occMax_irrep), h1e_x2c2e(occMax_irrep), fock_x2c2e_2e(occMax_irrep);

    for(int ir = 0; ir < occMax_irrep; ir++)
    {
        int size = irrep_list[ir].size, size2 = size*2;
        overlap_2c[ir] = matBlock(overlap_4c[ir], size2, 0, 0, size, size);
        overlap_h_i_2c[ir] = matrix_half_inverse(overlap_2c[ir], size);
        
        XXX[ir] = X2C::get_X(coeff[ir], size);
        RRR[ir] = X2C::get_R(overlap_4c[ir], XXX[ir], size);
        
        h1e_x2c2e[ir] = X2C::transform_4c_2c(h1e_4c[ir], XXX[ir], RRR[ir], size);
        
        if(coeff2c == NULL)
        {
            vector<double> ene_mo_tmp;
            fock_x2c2e[ir] = X2C::transform_4c_2c(fock_4c[ir], XXX[ir], RRR[ir], irrep_list[ir].size);
            eigensolverG(fock_x2c2e[ir], overlap_h_i_2c[ir], ene_mo_tmp, coeff_2c[ir], irrep_list[ir].size);
        }
        else
        {
            coeff_2c[ir] = (*coeff2c)[ir];
        }
        density_2c[ir] = evaluateDensity_spinor(coeff_2c[ir],occNumber[ir],irrep_list[ir].size,true);

        // X2C1E
        XXX_1e[ir] = X2C::get_X(overlap[ir],kinetic[ir],WWW[ir],Vnuc[ir],irrep_list[ir].size);
        RRR_1e[ir] = X2C::get_R(overlap[ir],kinetic[ir],XXX_1e[ir],irrep_list[ir].size);
        h1e_x2c1e[ir] = X2C::evaluate_h1e_x2c(overlap[ir],kinetic[ir],WWW[ir],Vnuc[ir],XXX_1e[ir],RRR_1e[ir],irrep_list[ir].size);
    }    

    for(int ir = 0; ir < occMax_irrep; ir++)
    {
        evaluateFock_2e(fock_4c_2e[ir],false,density,irrep_list[ir].size,ir);
        evaluateFock_2e(JK_x2c2c[ir],true,density_2c,irrep_list[ir].size,ir);
        fock_x2c2e_2e[ir] = X2C::transform_4c_2c(fock_4c_2e[ir], XXX[ir], RRR[ir], irrep_list[ir].size);
    }
    for(int ir = 0; ir < occMax_irrep; ir++)
    {
        fock_pcc[ir] = fock_x2c2e_2e[ir] - JK_x2c2c[ir] + h1e_x2c2e[ir] - h1e_x2c1e[ir];
    }

    x2cXXX = XXX;
    x2cRRR = RRR;
    X_calculated = true;

    return fock_pcc;
}

vVectorXd DHF_SPH::h_x2c2e(vVectorXd* coeff2c)
{
    if(!converged)
    {
        cout << "SCF did not converge. x2c2ePCC cannot be used!" << endl;
        exit(99);
    }
    vVectorXd XXX(occMax_irrep), RRR(occMax_irrep), XXX_1e(occMax_irrep), RRR_1e(occMax_irrep);
    vVectorXd overlap_2c(occMax_irrep), overlap_h_i_2c(occMax_irrep), coeff_2c(occMax_irrep), fock_4c_2e(occMax_irrep), JK_x2c2c(occMax_irrep), fock_x2c2e(occMax_irrep), density_2c(occMax_irrep), fock_x2c2e_2e(occMax_irrep), h1e_x2c2e(occMax_irrep);

    for(int ir = 0; ir < occMax_irrep; ir++)
    {
        int size = irrep_list[ir].size, size2 = size*2;
        overlap_2c[ir] = matBlock(overlap_4c[ir], size2, 0, 0, size, size);
        overlap_h_i_2c[ir] = matrix_half_inverse(overlap_2c[ir], size);
        vector<double> ene_mo_tmp;
        XXX[ir] = X2C::get_X(coeff[ir],size);
        RRR[ir] = X2C::get_R(overlap_4c[ir],XXX[ir],size);
        
        h1e_x2c2e[ir] = X2C::transform_4c_2c(h1e_4c[ir], XXX[ir], RRR[ir], irrep_list[ir].size);
        fock_x2c2e[ir] = X2C::transform_4c_2c(fock_4c[ir], XXX[ir], RRR[ir], irrep_list[ir].size);
        if(coeff2c == NULL)
            eigensolverG(fock_x2c2e[ir],overlap_h_i_2c[ir],ene_mo_tmp,coeff_2c[ir],irrep_list[ir].size);
        else
            coeff_2c[ir] = (*coeff2c)[ir];

        density_2c[ir] = evaluateDensity_spinor(coeff_2c[ir],occNumber[ir],irrep_list[ir].size,true);

        // X2C1E
        XXX_1e[ir] = X2C::get_X(overlap[ir],kinetic[ir],WWW[ir],Vnuc[ir],irrep_list[ir].size);
        RRR_1e[ir] = X2C::get_R(overlap[ir],kinetic[ir],XXX_1e[ir],irrep_list[ir].size);
    }    

    for(int ir = 0; ir < occMax_irrep; ir++)
    {
        evaluateFock_2e(fock_4c_2e[ir],false,density,irrep_list[ir].size,ir);
        evaluateFock_2e(JK_x2c2c[ir],true,density_2c,irrep_list[ir].size,ir);
        fock_x2c2e_2e[ir] = X2C::transform_4c_2c(fock_4c_2e[ir], XXX[ir], RRR[ir], irrep_list[ir].size);
    }

    x2cXXX = XXX;
    x2cRRR = RRR;
    X_calculated = true;

    return fock_x2c2e_2e;
}


