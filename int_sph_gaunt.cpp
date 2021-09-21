#include<Eigen/Dense>
#include<string>
#include<iostream>
#include<iomanip>
#include<fstream>
#include<cmath>
#include<complex>
#include<omp.h>
#include<gsl/gsl_sf_coupling.h>
#include"int_sph.h"
using namespace std;
using namespace Eigen;

/* 
    evaluate angular part in 2e gaunt integrals 
*/
inline double INT_SPH::int2e_get_threeSH(const int& l1, const int& m1, const int& l2, const int& m2, const int& l3, const int& m3, const double& threeJ) const
{
    return sqrt((2.0*l1+1.0)*(2.0*l2+1.0)*(2.0*l3+1.0)/4.0/M_PI)*threeJ*wigner_3j(l1,l2,l3,m1,m2,m3);
}



double INT_SPH::int2e_get_angular_gaunt_LSLS(const int& l1, const int& two_m1, const int& s1, const int& l2, const int& two_m2, const int& s2, const int& l3, const int& two_m3, const int& s3, const int& l4, const int& two_m4, const int& s4, const int& LL) const
{
    double angular = 0.0;
    // double threeJ_p_12 = wigner_3j_zeroM(l1,l2,LL+1), threeJ_m_12 = wigner_3j_zeroM(l1,l2,LL-1), threeJ_p_34 = wigner_3j_zeroM(l3,l4,LL+1), threeJ_m_34 = wigner_3j_zeroM(l3,l4,LL-1);
    // double y1a = s1*sqrt(l1+0.5+s1*two_m1/2.0), y1b = sqrt(l1+0.5-s1*two_m1/2.0), y2a = s2*sqrt(l2+0.5+s2*two_m2/2.0), y2b = sqrt(l2+0.5-s2*two_m2/2.0), y3a = s3*sqrt(l3+0.5+s3*two_m3/2.0), y3b = sqrt(l3+0.5-s3*two_m3/2.0), y4a = s4*sqrt(l4+0.5+s4*two_m4/2.0), y4b = sqrt(l4+0.5-s4*two_m4/2.0);
    // for(int MM = -LL; MM <= LL; MM++)
    // {
    //     Vector3d v12 = int2e_get_angular_gaunt_ssp(l1,two_m1,s1,l2,two_m2,s2,LL,-MM, threeJ_p_12, threeJ_m_12), v34 = int2e_get_angular_gaunt_ssp(l3,two_m3,s3,l4,two_m4,s4,LL,MM, threeJ_p_34, threeJ_m_34);
    //     angular += pow(-1, MM)*(v12(0)*v34(0) - v12(1)*v34(1) + v12(2)*v34(2));
    // }
    // return angular * 4.0*M_PI/(2.0*LL+1.0);

    int l2p = l2+s2, l4p = l4+s4;
    for(int MM = -LL; MM <= LL; MM++)
    {
        double tmp1 = s1*sqrt(l1+0.5+s1*two_m1/2.0)*sqrt(l2p+0.5+s2*two_m2/2.0)*wigner_3j(l1,LL,l2p,(-two_m1+1)/2,-MM,(two_m2+1)/2);
        double tmp2 = s2*sqrt(l1+0.5-s1*two_m1/2.0)*sqrt(l2p+0.5-s2*two_m2/2.0)*wigner_3j(l1,LL,l2p,(-two_m1-1)/2,-MM,(two_m2-1)/2);
        double tmp3 = s3*sqrt(l3+0.5+s3*two_m3/2.0)*sqrt(l4p+0.5+s4*two_m4/2.0)*wigner_3j(l3,LL,l4p,(-two_m3+1)/2,MM,(two_m4+1)/2);
        double tmp4 = s4*sqrt(l3+0.5-s3*two_m3/2.0)*sqrt(l4p+0.5-s4*two_m4/2.0)*wigner_3j(l3,LL,l4p,(-two_m3-1)/2,MM,(two_m4-1)/2);
        double tmp5 = s1*s2*sqrt(l1+0.5+s1*two_m1/2.0)*sqrt(l2p+0.5-s2*two_m2/2.0)*wigner_3j(l1,LL,l2p,(-two_m1+1)/2,-MM,(two_m2-1)/2);
        double tmp6 = sqrt(l1+0.5-s1*two_m1/2.0)*sqrt(l2p+0.5+s2*two_m2/2.0)*wigner_3j(l1,LL,l2p,(-two_m1-1)/2,-MM,(two_m2+1)/2);
        double tmp7 = s3*s4*sqrt(l3+0.5+s3*two_m3/2.0)*sqrt(l4p+0.5-s4*two_m4/2.0)*wigner_3j(l3,LL,l4p,(-two_m3+1)/2,MM,(two_m4-1)/2);
        double tmp8 = sqrt(l3+0.5-s3*two_m3/2.0)*sqrt(l4p+0.5+s4*two_m4/2.0)*wigner_3j(l3,LL,l4p,(-two_m3-1)/2,MM,(two_m4+1)/2);
        angular += pow(-1,MM)*(2.0*tmp1*tmp4 + 2.0*tmp2*tmp3 + (-tmp5+tmp6)*(-tmp7+tmp8) );
    }
    return angular * pow(-1,(two_m1+two_m3)/2+1) * wigner_3j_zeroM(l1,LL,l2p)*wigner_3j_zeroM(l3,LL,l4p);
}
double INT_SPH::int2e_get_angular_gaunt_LSSL(const int& l1, const int& two_m1, const int& s1, const int& l2, const int& two_m2, const int& s2, const int& l3, const int& two_m3, const int& s3, const int& l4, const int& two_m4, const int& s4, const int& LL) const
{
    double angular = 0.0;
    // double threeJ_p_12 = wigner_3j_zeroM(l1,l2,LL+1), threeJ_m_12 = wigner_3j_zeroM(l1,l2,LL-1), threeJ_p_34 = wigner_3j_zeroM(l3,l4,LL+1), threeJ_m_34 = wigner_3j_zeroM(l3,l4,LL-1);
    // double y1a = s1*sqrt(l1+0.5+s1*two_m1/2.0), y1b = sqrt(l1+0.5-s1*two_m1/2.0), y2a = s2*sqrt(l2+0.5+s2*two_m2/2.0), y2b = sqrt(l2+0.5-s2*two_m2/2.0), y3a = s3*sqrt(l3+0.5+s3*two_m3/2.0), y3b = sqrt(l3+0.5-s3*two_m3/2.0), y4a = s4*sqrt(l4+0.5+s4*two_m4/2.0), y4b = sqrt(l4+0.5-s4*two_m4/2.0);
    // for(int MM = -LL; MM <= LL; MM++)
    // {
    //     Vector3d v12 = int2e_get_angular_gaunt_ssp(l1,two_m1,s1,l2,two_m2,s2,LL,-MM, threeJ_p_12, threeJ_m_12), v34 = int2e_get_angular_gaunt_sps(l3,two_m3,s3,l4,two_m4,s4,LL,MM, threeJ_p_34, threeJ_m_34);
    //     angular += pow(-1, MM)*(v12(0)*v34(0) - v12(1)*v34(1) + v12(2)*v34(2));
    // }
    // return angular * 4.0*M_PI/(2.0*LL+1.0);

    int l2p = l2+s2, l3p = l3+s3;
    for(int MM = -LL; MM <= LL; MM++)
    {
        double tmp1 = s1*sqrt(l1+0.5+s1*two_m1/2.0)*sqrt(l2p+0.5+s2*two_m2/2.0)*wigner_3j(l1,LL,l2p,(-two_m1+1)/2,-MM,(two_m2+1)/2);
        double tmp2 = s2*sqrt(l1+0.5-s1*two_m1/2.0)*sqrt(l2p+0.5-s2*two_m2/2.0)*wigner_3j(l1,LL,l2p,(-two_m1-1)/2,-MM,(two_m2-1)/2);
        double tmp3 = s3*sqrt(l3p+0.5-s3*two_m3/2.0)*sqrt(l4+0.5-s4*two_m4/2.0)*wigner_3j(l3p,LL,l4,(-two_m3+1)/2,MM,(two_m4+1)/2);
        double tmp4 = s4*sqrt(l3p+0.5+s3*two_m3/2.0)*sqrt(l4+0.5+s4*two_m4/2.0)*wigner_3j(l3p,LL,l4,(-two_m3-1)/2,MM,(two_m4-1)/2);
        double tmp5 = s1*s2*sqrt(l1+0.5+s1*two_m1/2.0)*sqrt(l2p+0.5-s2*two_m2/2.0)*wigner_3j(l1,LL,l2p,(-two_m1+1)/2,-MM,(two_m2-1)/2);
        double tmp6 = sqrt(l1+0.5-s1*two_m1/2.0)*sqrt(l2p+0.5+s2*two_m2/2.0)*wigner_3j(l1,LL,l2p,(-two_m1-1)/2,-MM,(two_m2+1)/2);
        double tmp7 = s3*s4*sqrt(l3p+0.5-s3*two_m3/2.0)*sqrt(l4+0.5+s4*two_m4/2.0)*wigner_3j(l3p,LL,l4,(-two_m3+1)/2,MM,(two_m4-1)/2);
        double tmp8 = sqrt(l3p+0.5+s3*two_m3/2.0)*sqrt(l4+0.5-s4*two_m4/2.0)*wigner_3j(l3p,LL,l4,(-two_m3-1)/2,MM,(two_m4+1)/2);
        angular += pow(-1,MM)*(2.0*tmp1*tmp4 + 2.0*tmp2*tmp3 + (-tmp5+tmp6)*(tmp7-tmp8) );
    }
    return angular * pow(-1,(two_m1+two_m3)/2) * wigner_3j_zeroM(l1,LL,l2p)*wigner_3j_zeroM(l3p,LL,l4);
}
Vector3d INT_SPH::int2e_get_angular_gaunt_ssp(const int& l1, const int& two_m1, const int& s1, const int& l2, const int& two_m2, const int& s2, const int& LL, const int& MM, const double& threeJ_p_12, const double& threeJ_m_12) const
{
    // Y1a = s1*sqrt(l1+0.5+s1*two_m1/2.0) | l1, (two_m1-1)/2 >
    // Y1b = sqrt(l1+0.5-s1*two_m1/2.0) | l1, (two_m1+1)/2 >
    // Y2a = s2*sqrt(l2+0.5+s2*two_m2/2.0) | l2, (two_m2-1)/2 >
    // Y2b = sqrt(l2+0.5-s2*two_m2/2.0) | l1, (two_m2+1)/2 >
    // d = sqrt((LL-MM)*(LL-MM-1.0)/(2.0*LL+1.0)/(2.0*LL-1.0)) |LL-1,MM+1> + sqrt((LL+MM+1.0)*(LL+MM+2.0)/(2.0*LL+1.0)/(2.0*LL+3.0)) |LL+1,MM+1>
    // d'= -sqrt((LL+MM)*(LL+MM-1.0)/(2.0*LL+1.0)/(2.0*LL-1.0)) |LL-1,MM-1> + sqrt((LL-MM+1.0)*(LL-MM+2.0)/(2.0*LL+1.0)/(2.0*LL+3.0)) |LL+1,MM-1>
    // c = sqrt((LL+MM)*(LL-MM)/(2.0*LL+1.0)/(2.0*LL-1.0)) |LL-1,MM> + sqrt((LL+MM+1.0)*(LL-MM+1.0)/(2.0*LL+1.0)/(2.0*LL+3.0)) |LL+1,MM>
    Vector3d res = Vector3d::Zero();
    double Y1a = s1*sqrt(l1+0.5+s1*two_m1/2.0), Y1b = sqrt(l1+0.5-s1*two_m1/2.0), Y2a = s2*sqrt(l2+0.5+s2*two_m2/2.0), Y2b = sqrt(l2+0.5-s2*two_m2/2.0);
    double d1 = sqrt((LL-MM)*(LL-MM-1.0)/(2.0*LL+1.0)/(2.0*LL-1.0)), d2 = -sqrt((LL+MM+1.0)*(LL+MM+2.0)/(2.0*LL+1.0)/(2.0*LL+3.0)), ds1 = -sqrt((LL+MM)*(LL+MM-1.0)/(2.0*LL+1.0)/(2.0*LL-1.0)), ds2 = sqrt((LL-MM+1.0)*(LL-MM+2.0)/(2.0*LL+1.0)/(2.0*LL+3.0)), c1 = sqrt((LL+MM)*(LL-MM)/(2.0*LL+1.0)/(2.0*LL-1.0)), c2 = sqrt((LL+MM+1.0)*(LL-MM+1.0)/(2.0*LL+1.0)/(2.0*LL+3.0));
    double tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8;
    if(LL != 0)
    {
        tmp1 = Y1a*Y2a*(d1*int2e_get_threeSH(l1,-(two_m1-1)/2,LL-1,MM+1,l2,(two_m2-1)/2,threeJ_m_12) + d2*int2e_get_threeSH(l1,-(two_m1-1)/2,LL+1,MM+1,l2,(two_m2-1)/2,threeJ_p_12));
        tmp2 = Y1b*Y2a*(c1*int2e_get_threeSH(l1,-(two_m1+1)/2,LL-1,MM,l2,(two_m2-1)/2,threeJ_m_12) + c2*int2e_get_threeSH(l1,-(two_m1+1)/2,LL+1,MM,l2,(two_m2-1)/2,threeJ_p_12));
        tmp3 = Y1a*Y2b*(c1*int2e_get_threeSH(l1,-(two_m1-1)/2,LL-1,MM,l2,(two_m2+1)/2,threeJ_m_12) + c2*int2e_get_threeSH(l1,-(two_m1-1)/2,LL+1,MM,l2,(two_m2+1)/2,threeJ_p_12));
        tmp4 = Y1b*Y2b*(ds1*int2e_get_threeSH(l1,-(two_m1+1)/2,LL-1,MM-1,l2,(two_m2+1)/2,threeJ_m_12) + ds2*int2e_get_threeSH(l1,-(two_m1+1)/2,LL+1,MM-1,l2,(two_m2+1)/2,threeJ_p_12));
        tmp5 = Y1a*Y2a*(c1*int2e_get_threeSH(l1,-(two_m1-1)/2,LL-1,MM,l2,(two_m2-1)/2,threeJ_m_12) + c2*int2e_get_threeSH(l1,-(two_m1-1)/2,LL+1,MM,l2,(two_m2-1)/2,threeJ_p_12));
        tmp6 = Y1b*Y2a*(d1*int2e_get_threeSH(l1,-(two_m1+1)/2,LL-1,MM+1,l2,(two_m2-1)/2,threeJ_m_12) + d2*int2e_get_threeSH(l1,-(two_m1+1)/2,LL+1,MM+1,l2,(two_m2-1)/2,threeJ_p_12));
        tmp7 = Y1a*Y2b*(ds1*int2e_get_threeSH(l1,-(two_m1-1)/2,LL-1,MM-1,l2,(two_m2+1)/2,threeJ_m_12) + ds2*int2e_get_threeSH(l1,-(two_m1-1)/2,LL+1,MM-1,l2,(two_m2+1)/2,threeJ_p_12));
        tmp8 = Y1b*Y2b*(c1*int2e_get_threeSH(l1,-(two_m1+1)/2,LL-1,MM,l2,(two_m2+1)/2,threeJ_m_12) + c2*int2e_get_threeSH(l1,-(two_m1+1)/2,LL+1,MM,l2,(two_m2+1)/2,threeJ_p_12));
    }
    else
    {
        tmp1 = Y1a*Y2a*(d2*int2e_get_threeSH(l1,-(two_m1-1)/2,LL+1,MM+1,l2,(two_m2-1)/2,threeJ_p_12));
        tmp2 = Y1b*Y2a*(c2*int2e_get_threeSH(l1,-(two_m1+1)/2,LL+1,MM,l2,(two_m2-1)/2,threeJ_p_12));
        tmp3 = Y1a*Y2b*(c2*int2e_get_threeSH(l1,-(two_m1-1)/2,LL+1,MM,l2,(two_m2+1)/2,threeJ_p_12));
        tmp4 = Y1b*Y2b*(ds2*int2e_get_threeSH(l1,-(two_m1+1)/2,LL+1,MM-1,l2,(two_m2+1)/2,threeJ_p_12));
        tmp5 = Y1a*Y2a*(c2*int2e_get_threeSH(l1,-(two_m1-1)/2,LL+1,MM,l2,(two_m2-1)/2,threeJ_p_12));
        tmp6 = Y1b*Y2a*(d2*int2e_get_threeSH(l1,-(two_m1+1)/2,LL+1,MM+1,l2,(two_m2-1)/2,threeJ_p_12));
        tmp7 = Y1a*Y2b*(ds2*int2e_get_threeSH(l1,-(two_m1-1)/2,LL+1,MM-1,l2,(two_m2+1)/2,threeJ_p_12));
        tmp8 = Y1b*Y2b*(c2*int2e_get_threeSH(l1,-(two_m1+1)/2,LL+1,MM,l2,(two_m2+1)/2,threeJ_p_12));
    }

    res(0) = pow(-1,(two_m1-1)/2)/sqrt((1.0+2.0*l1)*(1.0+2.0*l2)) * (tmp1-tmp2-tmp3-tmp4);
    res(1) = pow(-1,(two_m1-1)/2)/sqrt((1.0+2.0*l1)*(1.0+2.0*l2)) * (-tmp1-tmp2+tmp3-tmp4);
    res(2) = pow(-1,(two_m1-1)/2)/sqrt((1.0+2.0*l1)*(1.0+2.0*l2)) * (tmp5+tmp6+tmp7-tmp8);
    return res;
}
Vector3d INT_SPH::int2e_get_angular_gaunt_sps(const int& l1, const int& two_m1, const int& s1, const int& l2, const int& two_m2, const int& s2, const int& LL, const int& MM, const double& threeJ_p_12, const double& threeJ_m_12) const
{
    // Y1a = s1*sqrt(l1+0.5+s1*two_m1/2.0) | l1, (two_m1-1)/2 >
    // Y1b = sqrt(l1+0.5-s1*two_m1/2.0) | l1, (two_m1+1)/2 >
    // Y2a = s2*sqrt(l2+0.5+s2*two_m2/2.0) | l2, (two_m2-1)/2 >
    // Y2b = sqrt(l2+0.5-s2*two_m2/2.0) | l1, (two_m2+1)/2 >
    // d = sqrt((LL-MM)*(LL-MM-1.0)/(2.0*LL+1.0)/(2.0*LL-1.0)) |LL-1,MM+1> + sqrt((LL+MM+1.0)*(LL+MM+2.0)/(2.0*LL+1.0)/(2.0*LL+3.0)) |LL+1,MM+1>
    // d'= -sqrt((LL+MM)*(LL+MM-1.0)/(2.0*LL+1.0)/(2.0*LL-1.0)) |LL-1,MM-1> + sqrt((LL-MM+1.0)*(LL-MM+2.0)/(2.0*LL+1.0)/(2.0*LL+3.0)) |LL+1,MM-1>
    // c = sqrt((LL+MM)*(LL-MM)/(2.0*LL+1.0)/(2.0*LL-1.0)) |LL-1,MM> + sqrt((LL+MM+1.0)*(LL-MM+1.0)/(2.0*LL+1.0)/(2.0*LL+3.0)) |LL+1,MM>
    Vector3d res = Vector3d::Zero();
    double Y1a = s1*sqrt(l1+0.5+s1*two_m1/2.0), Y1b = sqrt(l1+0.5-s1*two_m1/2.0), Y2a = s2*sqrt(l2+0.5+s2*two_m2/2.0), Y2b = sqrt(l2+0.5-s2*two_m2/2.0);
    double d1 = sqrt((LL-MM)*(LL-MM-1.0)/(2.0*LL+1.0)/(2.0*LL-1.0)), d2 = -sqrt((LL+MM+1.0)*(LL+MM+2.0)/(2.0*LL+1.0)/(2.0*LL+3.0)), ds1 = -sqrt((LL+MM)*(LL+MM-1.0)/(2.0*LL+1.0)/(2.0*LL-1.0)), ds2 = sqrt((LL-MM+1.0)*(LL-MM+2.0)/(2.0*LL+1.0)/(2.0*LL+3.0)), c1 = sqrt((LL+MM)*(LL-MM)/(2.0*LL+1.0)/(2.0*LL-1.0)), c2 = sqrt((LL+MM+1.0)*(LL-MM+1.0)/(2.0*LL+1.0)/(2.0*LL+3.0));
    double tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8;
    if(LL != 0)
    {
        tmp1 = Y1a*Y2a*(ds1*int2e_get_threeSH(l1,-(two_m1-1)/2,LL-1,MM-1,l2,(two_m2-1)/2,threeJ_m_12) + ds2*int2e_get_threeSH(l1,-(two_m1-1)/2,LL+1,MM-1,l2,(two_m2-1)/2,threeJ_p_12));
        tmp2 = Y1b*Y2a*(c1*int2e_get_threeSH(l1,-(two_m1+1)/2,LL-1,MM,l2,(two_m2-1)/2,threeJ_m_12) + c2*int2e_get_threeSH(l1,-(two_m1+1)/2,LL+1,MM,l2,(two_m2-1)/2,threeJ_p_12));
        tmp3 = Y1a*Y2b*(c1*int2e_get_threeSH(l1,-(two_m1-1)/2,LL-1,MM,l2,(two_m2+1)/2,threeJ_m_12) + c2*int2e_get_threeSH(l1,-(two_m1-1)/2,LL+1,MM,l2,(two_m2+1)/2,threeJ_p_12));
        tmp4 = Y1b*Y2b*(d1*int2e_get_threeSH(l1,-(two_m1+1)/2,LL-1,MM+1,l2,(two_m2+1)/2,threeJ_m_12) + d2*int2e_get_threeSH(l1,-(two_m1+1)/2,LL+1,MM+1,l2,(two_m2+1)/2,threeJ_p_12));
        tmp5 = Y1a*Y2a*(c1*int2e_get_threeSH(l1,-(two_m1-1)/2,LL-1,MM,l2,(two_m2-1)/2,threeJ_m_12) + c2*int2e_get_threeSH(l1,-(two_m1-1)/2,LL+1,MM,l2,(two_m2-1)/2,threeJ_p_12));
        tmp6 = Y1b*Y2a*(d1*int2e_get_threeSH(l1,-(two_m1+1)/2,LL-1,MM+1,l2,(two_m2-1)/2,threeJ_m_12) + d2*int2e_get_threeSH(l1,-(two_m1+1)/2,LL+1,MM+1,l2,(two_m2-1)/2,threeJ_p_12));
        tmp7 = Y1a*Y2b*(ds1*int2e_get_threeSH(l1,-(two_m1-1)/2,LL-1,MM-1,l2,(two_m2+1)/2,threeJ_m_12) + ds2*int2e_get_threeSH(l1,-(two_m1-1)/2,LL+1,MM-1,l2,(two_m2+1)/2,threeJ_p_12));
        tmp8 = Y1b*Y2b*(c1*int2e_get_threeSH(l1,-(two_m1+1)/2,LL-1,MM,l2,(two_m2+1)/2,threeJ_m_12) + c2*int2e_get_threeSH(l1,-(two_m1+1)/2,LL+1,MM,l2,(two_m2+1)/2,threeJ_p_12));
    }
    else
    {
        tmp1 = Y1a*Y2a*(ds2*int2e_get_threeSH(l1,-(two_m1-1)/2,LL+1,MM-1,l2,(two_m2-1)/2,threeJ_p_12));
        tmp2 = Y1b*Y2a*(c2*int2e_get_threeSH(l1,-(two_m1+1)/2,LL+1,MM,l2,(two_m2-1)/2,threeJ_p_12));
        tmp3 = Y1a*Y2b*(c2*int2e_get_threeSH(l1,-(two_m1-1)/2,LL+1,MM,l2,(two_m2+1)/2,threeJ_p_12));
        tmp4 = Y1b*Y2b*(d2*int2e_get_threeSH(l1,-(two_m1+1)/2,LL+1,MM+1,l2,(two_m2+1)/2,threeJ_p_12));
        tmp5 = Y1a*Y2a*(c2*int2e_get_threeSH(l1,-(two_m1-1)/2,LL+1,MM,l2,(two_m2-1)/2,threeJ_p_12));
        tmp6 = Y1b*Y2a*(d2*int2e_get_threeSH(l1,-(two_m1+1)/2,LL+1,MM+1,l2,(two_m2-1)/2,threeJ_p_12));
        tmp7 = Y1a*Y2b*(ds2*int2e_get_threeSH(l1,-(two_m1-1)/2,LL+1,MM-1,l2,(two_m2+1)/2,threeJ_p_12));
        tmp8 = Y1b*Y2b*(c2*int2e_get_threeSH(l1,-(two_m1+1)/2,LL+1,MM,l2,(two_m2+1)/2,threeJ_p_12));
    }

    res(0) = pow(-1,(two_m1-1)/2)/sqrt((1.0+2.0*l1)*(1.0+2.0*l2)) * (tmp1+tmp2+tmp3-tmp4);
    res(1) = pow(-1,(two_m1-1)/2)/sqrt((1.0+2.0*l1)*(1.0+2.0*l2)) * (tmp1+tmp2-tmp3+tmp4);
    res(2) = pow(-1,(two_m1-1)/2)/sqrt((1.0+2.0*l1)*(1.0+2.0*l2)) * (tmp5-tmp6-tmp7-tmp8);
    return res;
}


/*
    Evaluate different two-electron Coulomb and Exchange integral in 2-spinor basis
*/
int2eJK INT_SPH::get_h2e_JK_gaunt(const string& intType, const int& occMaxL) const
{
    double time_a = 0.0, time_r = 0.0, time_c = 0.0;
    int occMaxShell = 0;
    if(occMaxL == -1)    occMaxShell = size_shell;
    else
    {
        for(int ii = 0; ii < size_shell; ii++)
        {
            if(shell_list(ii).l <= occMaxL)
                occMaxShell++;
            else
                break;
        }
    }
    
    int2eJK int_2e_JK;
    int_2e_JK.J = new double***[Nirrep];
    int_2e_JK.K = new double***[Nirrep];
    for(int ii = 0; ii < Nirrep; ii++)
    {
        int_2e_JK.J[ii] = new double**[Nirrep];
        int_2e_JK.K[ii] = new double**[Nirrep];
    }
    
    int int_tmp1_p = 0;
    for(int pshell = 0; pshell < occMaxShell; pshell++)
    {
    int l_p = shell_list(pshell).l, int_tmp1_q = 0;
    for(int qshell = 0; qshell < occMaxShell; qshell++)
    {
        int l_q = shell_list(qshell).l, l_max = max(l_p,l_q), LmaxJ = min(l_p+l_p, l_q+l_q)+1, LmaxK = l_p+l_q+1;
        int size_gtos_p = shell_list(pshell).coeff.rows(), size_gtos_q = shell_list(qshell).coeff.rows();
        int size_tmp_p = (l_p == 0) ? 1 : 2, size_tmp_q = (l_q == 0) ? 1 : 2;
        double array_radial_J[LmaxJ+1][size_gtos_p*size_gtos_p][size_gtos_q*size_gtos_q][size_tmp_p][size_tmp_q];
        double array_radial_K[LmaxK+1][size_gtos_p*size_gtos_q][size_gtos_q*size_gtos_p][size_tmp_p][size_tmp_q];
        MatrixXd array_angular_J[LmaxJ+1][size_tmp_p][size_tmp_q], array_angular_K[LmaxK+1][size_tmp_p][size_tmp_q];

        StartTime = clock();
        #pragma omp parallel  for
        for(int twojj_p = abs(2*l_p-1); twojj_p <= 2*l_p+1; twojj_p = twojj_p + 2)
        for(int twojj_q = abs(2*l_q-1); twojj_q <= 2*l_q+1; twojj_q = twojj_q + 2)
        {
            int sym_ap = twojj_p - 2*l_p, sym_aq = twojj_q - 2*l_q;
            int index_tmp_p = (l_p > 0) ? 1 - (2*l_p+1 - twojj_p)/2 : 0;
            int index_tmp_q = (l_q > 0) ? 1 - (2*l_q+1 - twojj_q)/2 : 0;

            for(int tmp = LmaxJ; tmp >= 0; tmp -= 2)
            {
                array_angular_J[tmp][index_tmp_p][index_tmp_q].resize(twojj_p + 1,twojj_q + 1);
                for(int mp = 0; mp < twojj_p + 1; mp++)
                for(int mq = 0; mq < twojj_q + 1; mq++)
                {
                    if(intType.substr(0,4) == "LSLS")
                        array_angular_J[tmp][index_tmp_p][index_tmp_q](mp,mq) = int2e_get_angular_gaunt_LSLS(l_p, 2*mp-twojj_p, sym_ap, l_p, 2*mp-twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, tmp);
                    else if(intType.substr(0,4) == "LSSL")
                        array_angular_J[tmp][index_tmp_p][index_tmp_q](mp,mq) = int2e_get_angular_gaunt_LSSL(l_p, 2*mp-twojj_p, sym_ap, l_p, 2*mp-twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, tmp);
                    else
                    {
                        cout << "ERROR: Unkonwn intType in get_h2e_JK_gaunt." << endl;
                        exit(99);
                    }
                }
            }
            for(int tmp = LmaxK; tmp >= 0; tmp -= 2)
            {
                array_angular_K[tmp][index_tmp_p][index_tmp_q].resize(twojj_p + 1,twojj_q + 1);
                for(int mp = 0; mp < twojj_p + 1; mp++)
                for(int mq = 0; mq < twojj_q + 1; mq++)
                {
                    if(intType.substr(0,4) == "LSLS")
                        array_angular_K[tmp][index_tmp_p][index_tmp_q](mp,mq) = int2e_get_angular_gaunt_LSLS(l_p, 2*mp-twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, l_p, 2*mp-twojj_p, sym_ap, tmp);
                    else if(intType.substr(0,4) == "LSSL")
                        array_angular_K[tmp][index_tmp_p][index_tmp_q](mp,mq) = int2e_get_angular_gaunt_LSSL(l_p, 2*mp-twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, l_p, 2*mp-twojj_p, sym_ap, tmp);
                    else
                    {
                        cout << "ERROR: Unkonwn intType in get_h2e_JK_gaunt." << endl;
                        exit(99);
                    }
                }
            }
        }
        EndTime = clock();
        time_a += (EndTime - StartTime)/(double)CLOCKS_PER_SEC;

        StartTime = clock();
        #pragma omp parallel  for
        for(int tt = 0; tt < size_gtos_p*size_gtos_p*size_gtos_q*size_gtos_q; tt++)
        {
            int e1J = tt/(size_gtos_q*size_gtos_q);
            int e2J = tt - e1J*(size_gtos_q*size_gtos_q);
            int ii = e1J/size_gtos_p, jj = e1J - ii*size_gtos_p;
            int kk = e2J/size_gtos_q, ll = e2J - kk*size_gtos_q;
            int e1K = ii*size_gtos_q+ll, e2K = kk*size_gtos_p+jj;
            MatrixXd radial_2e_list_J[LmaxJ+1], radial_2e_list_K[LmaxK+1];
            double a_i_J = shell_list(pshell).exp_a(ii), a_j_J = shell_list(pshell).exp_a(jj), a_k_J = shell_list(qshell).exp_a(kk), a_l_J = shell_list(qshell).exp_a(ll);
            double a_i_K = shell_list(pshell).exp_a(ii), a_j_K = shell_list(qshell).exp_a(ll), a_k_K = shell_list(qshell).exp_a(kk), a_l_K = shell_list(pshell).exp_a(jj);
        
            if(intType.substr(0,4) == "LSLS")
            {
                for(int LL = LmaxJ; LL >= 0; LL -= 2)
                {
                    radial_2e_list_J[LL].resize(4,1);
                    radial_2e_list_J[LL](0,0) = int2e_get_radial(l_p,a_i_J,l_p+1,a_j_J,l_q,a_k_J,l_q+1,a_l_J,LL);
                    if(l_p != 0)
                        radial_2e_list_J[LL](1,0) = int2e_get_radial(l_p,a_i_J,l_p-1,a_j_J,l_q,a_k_J,l_q+1,a_l_J,LL);
                    if(l_q != 0)
                        radial_2e_list_J[LL](2,0) = int2e_get_radial(l_p,a_i_J,l_p+1,a_j_J,l_q,a_k_J,l_q-1,a_l_J,LL);
                    if(l_p != 0 && l_q != 0)
                        radial_2e_list_J[LL](3,0) = int2e_get_radial(l_p,a_i_J,l_p-1,a_j_J,l_q,a_k_J,l_q-1,a_l_J,LL);
                }
                for(int LL = LmaxK; LL >= 0; LL -= 2)
                {
                    radial_2e_list_K[LL].resize(4,1);
                    radial_2e_list_K[LL](0,0) = int2e_get_radial(l_p,a_i_K,l_q+1,a_j_K,l_q,a_k_K,l_p+1,a_l_K,LL);
                    if(l_q != 0)
                        radial_2e_list_K[LL](1,0) = int2e_get_radial(l_p,a_i_K,l_q-1,a_j_K,l_q,a_k_K,l_p+1,a_l_K,LL);
                    if(l_p != 0)
                        radial_2e_list_K[LL](2,0) = int2e_get_radial(l_p,a_i_K,l_q+1,a_j_K,l_q,a_k_K,l_p-1,a_l_K,LL);
                    if(l_p != 0 && l_q != 0)
                        radial_2e_list_K[LL](3,0) = int2e_get_radial(l_p,a_i_K,l_q-1,a_j_K,l_q,a_k_K,l_p-1,a_l_K,LL);
                }
            }
            else if(intType.substr(0,4) == "LSSL")
            {
                for(int LL = LmaxJ; LL >= 0; LL -= 2)
                {
                    radial_2e_list_J[LL].resize(4,1);
                    radial_2e_list_J[LL](0,0) = int2e_get_radial(l_p,a_i_J,l_p+1,a_j_J,l_q+1,a_k_J,l_q,a_l_J,LL);
                    if(l_p != 0)
                        radial_2e_list_J[LL](1,0) = int2e_get_radial(l_p,a_i_J,l_p-1,a_j_J,l_q+1,a_k_J,l_q,a_l_J,LL);
                    if(l_q != 0)
                        radial_2e_list_J[LL](2,0) = int2e_get_radial(l_p,a_i_J,l_p+1,a_j_J,l_q-1,a_k_J,l_q,a_l_J,LL);
                    if(l_p != 0 && l_q != 0)
                        radial_2e_list_J[LL](3,0) = int2e_get_radial(l_p,a_i_J,l_p-1,a_j_J,l_q-1,a_k_J,l_q,a_l_J,LL);
                }
                for(int LL = LmaxK; LL >= 0; LL -= 2)
                {
                    radial_2e_list_K[LL].resize(4,1);
                    radial_2e_list_K[LL](0,0) = int2e_get_radial(l_p,a_i_K,l_q+1,a_j_K,l_q+1,a_k_K,l_p,a_l_K,LL);
                    if(l_q != 0)
                    {
                        radial_2e_list_K[LL](1,0) = int2e_get_radial(l_p,a_i_K,l_q-1,a_j_K,l_q+1,a_k_K,l_p,a_l_K,LL);
                        radial_2e_list_K[LL](2,0) = int2e_get_radial(l_p,a_i_K,l_q+1,a_j_K,l_q-1,a_k_K,l_p,a_l_K,LL);
                        radial_2e_list_K[LL](3,0) = int2e_get_radial(l_p,a_i_K,l_q-1,a_j_K,l_q-1,a_k_K,l_p,a_l_K,LL);
                    }
                }
            }
            else
            {
                cout << "ERROR: Unkonwn intType in get_h2e_JK_gaunt." << endl;
                exit(99);
            }
                
            for(int twojj_p = abs(2*l_p-1); twojj_p <= 2*l_p+1; twojj_p = twojj_p + 2)
            for(int twojj_q = abs(2*l_q-1); twojj_q <= 2*l_q+1; twojj_q = twojj_q + 2)
            {
                int index_tmp_p = (l_p > 0) ? 1 - (2*l_p+1 - twojj_p)/2 : 0;
                int index_tmp_q = (l_q > 0) ? 1 - (2*l_q+1 - twojj_q)/2 : 0;
                int sym_ap = twojj_p - 2*l_p, sym_aq = twojj_q - 2*l_q;
                double k_p = -(twojj_p+1.0)*sym_ap/2.0, k_q = -(twojj_q+1.0)*sym_aq/2.0;
                double norm_J = shell_list(pshell).norm(ii) * shell_list(pshell).norm(jj) * shell_list(qshell).norm(kk) * shell_list(qshell).norm(ll), norm_K = shell_list(pshell).norm(ii) * shell_list(qshell).norm(ll) * shell_list(qshell).norm(kk) * shell_list(pshell).norm(jj);
                double lk1 = 1+l_p+k_p, lk2 = 1+l_p+k_p, lk3 = 1+l_q+k_q, lk4 = 1+l_q+k_q, a1 = shell_list(pshell).exp_a(ii), a2 = shell_list(pshell).exp_a(jj), a3 = shell_list(qshell).exp_a(kk), a4 = shell_list(qshell).exp_a(ll);

                for(int tmp = LmaxJ; tmp >= 0; tmp -= 2)
                {
                    if(intType == "LSLS")
                    {
                        array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] = 4.0*a2*a4 * radial_2e_list_J[tmp](0,0);
                        if(l_p != 0 && l_q != 0)
                            array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] += lk2*lk4 * radial_2e_list_J[tmp](3,0)
                                    - 2.0*a4*lk2 * radial_2e_list_J[tmp](1,0) - 2.0*a2*lk4 * radial_2e_list_J[tmp](2,0);
                        else if(l_p != 0 && l_q == 0)
                            array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] -= 2.0*a4*lk2 * radial_2e_list_J[tmp](1,0);
                        else if(l_p == 0 && l_q != 0)
                            array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] -= 2.0*a2*lk4 * radial_2e_list_J[tmp](2,0);
                        array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] /= norm_J * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "LSSL")
                    {
                        array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] = 4.0*a2*a3 * radial_2e_list_J[tmp](0,0);
                        if(l_p != 0 && l_q != 0)
                            array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] += lk2*lk3 * radial_2e_list_J[tmp](3,0)
                                    - 2.0*a3*lk2 * radial_2e_list_J[tmp](1,0) - 2.0*a2*lk3 * radial_2e_list_J[tmp](2,0);
                        else if(l_p != 0 && l_q == 0)
                            array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] -= 2.0*a3*lk2 * radial_2e_list_J[tmp](1,0);
                        else if(l_p == 0 && l_q != 0)
                            array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] -= 2.0*a2*lk3 * radial_2e_list_J[tmp](2,0);
                        array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] /= -1.0 * norm_J * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "LSLS_SF")
                    {
                        
                    }
                    else if(intType == "LSSL_SF")
                    {
                        
                    }
                    else if(intType == "LSLS_SD")
                    {
                        
                    }
                    else if(intType == "LSSL_SD")
                    {
                        
                    }
                    else
                    {
                        cout << "ERROR: Unkonwn intType in get_h2e_JK_gaunt." << endl;
                        exit(99);
                    }
                }
                lk2 = 1+l_q+k_q; lk4 = 1+l_p+k_p; 
                a2 = shell_list(qshell).exp_a(ll); a4 = shell_list(pshell).exp_a(jj);
                for(int tmp = LmaxK; tmp >= 0; tmp -= 2)
                {
                    if(intType == "LSLS")
                    {
                        array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] = 4.0*a2*a4 * radial_2e_list_K[tmp](0,0);
                        if(l_p != 0 && l_q != 0)
                            array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] += lk2*lk4 * radial_2e_list_K[tmp](3,0) 
                                    - 2.0*a4*lk2 * radial_2e_list_K[tmp](1,0) - 2.0*a2*lk4 * radial_2e_list_K[tmp](2,0);
                        else if(l_p == 0 && l_q != 0)
                            array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] -= 2.0*a4*lk2 * radial_2e_list_K[tmp](1,0);
                        else if(l_p != 0 && l_q == 0)
                            array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] -= 2.0*a2*lk4 * radial_2e_list_K[tmp](2,0);
                        array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] /= norm_K * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "LSSL")
                    {
                        array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] = 4.0*a2*a3 * radial_2e_list_K[tmp](0,0);
                        if(l_q != 0)
                            array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] += lk2*lk3 * radial_2e_list_K[tmp](3,0) 
                                    - 2.0*a3*lk2 * radial_2e_list_K[tmp](1,0) - 2.0*a2*lk3 * radial_2e_list_K[tmp](2,0);
                        array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] /= -1.0 * norm_K * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "LSLS_SF")
                    {
                        
                    }
                    else if(intType == "LSSL_SF")
                    {
                        
                    }
                    else if(intType == "LSLS_SD")
                    {
                        
                    }
                    else if(intType == "LSSL_SD")
                    {
                        
                    }
                    else
                    {
                        cout << "ERROR: Unkonwn intType in get_h2e_JK_gaunt." << endl;
                        exit(99);
                    }
                }
            }
        }
        EndTime = clock();
        time_r += (EndTime - StartTime)/(double)CLOCKS_PER_SEC;

        StartTime = clock();
        int l_p_cycle = (l_p == 0) ? 1 : 2, l_q_cycle = (l_q == 0) ? 1 : 2;
        for(int int_tmp2_p = 0; int_tmp2_p < l_p_cycle; int_tmp2_p++)
        for(int int_tmp2_q = 0; int_tmp2_q < l_q_cycle; int_tmp2_q++)
        {
            int add_p = int_tmp2_p*(irrep_list(int_tmp1_p).two_j+1), add_q = int_tmp2_q*(irrep_list(int_tmp1_q).two_j+1);
            for(int mp = 0; mp < irrep_list(int_tmp1_p+add_p).two_j + 1; mp++)
            for(int mq = 0; mq < irrep_list(int_tmp1_q+add_q).two_j + 1; mq++)
            {
                int_2e_JK.J[int_tmp1_p+add_p + mp][int_tmp1_q + add_q + mq] = new double*[size_gtos_p*size_gtos_p];
                for(int iii = 0; iii < size_gtos_p*size_gtos_p; iii++)
                    int_2e_JK.J[int_tmp1_p+add_p + mp][int_tmp1_q + add_q + mq][iii] = new double[size_gtos_q*size_gtos_q];
                int_2e_JK.K[int_tmp1_p+add_p + mp][int_tmp1_q + add_q + mq] = new double*[size_gtos_p*size_gtos_q];
                for(int iii = 0; iii < size_gtos_p*size_gtos_q; iii++)
                    int_2e_JK.K[int_tmp1_p+add_p + mp][int_tmp1_q + add_q + mq][iii] = new double[size_gtos_q*size_gtos_p];
                #pragma omp parallel  for
                for(int tt = 0; tt < size_gtos_p*size_gtos_p*size_gtos_q*size_gtos_q; tt++)
                {
                    int e1J = tt/(size_gtos_q*size_gtos_q);
                    int e2J = tt - e1J*(size_gtos_q*size_gtos_q);
                    int e1K = tt/(size_gtos_p*size_gtos_q);
                    int e2K = tt - e1K*(size_gtos_p*size_gtos_q);
                    int_2e_JK.J[int_tmp1_p+add_p + mp][int_tmp1_q+add_q + mq][e1J][e2J] = 0.0;
                    int_2e_JK.K[int_tmp1_p+add_p + mp][int_tmp1_q+add_q + mq][e1K][e2K] = 0.0;
                    for(int tmp = LmaxJ; tmp >= 0; tmp = tmp - 2)
                        int_2e_JK.J[int_tmp1_p+add_p + mp][int_tmp1_q+add_q + mq][e1J][e2J] += array_radial_J[tmp][e1J][e2J][int_tmp2_p][int_tmp2_q] * array_angular_J[tmp][int_tmp2_p][int_tmp2_q](mp,mq);
                    for(int tmp = LmaxK; tmp >= 0; tmp = tmp - 2)
                        int_2e_JK.K[int_tmp1_p+add_p + mp][int_tmp1_q+add_q + mq][e1K][e2K] += array_radial_K[tmp][e1K][e2K][int_tmp2_p][int_tmp2_q] * array_angular_K[tmp][int_tmp2_p][int_tmp2_q](mp,mq);
                }
            }
        }
        EndTime = clock();
        time_c += (EndTime - StartTime)/(double)CLOCKS_PER_SEC;
        int_tmp1_q += 4*l_q+2;
    }
    int_tmp1_p += 4*l_p+2;
    }

    cout << time_a << "\t" << time_r << "\t" << time_c <<endl;
    return int_2e_JK;
}
int2eJK INT_SPH::get_h2e_JK_gaunt_compact(const string& intType, const int& occMaxL) const
{
    double time_a = 0.0, time_r = 0.0, time_c = 0.0;
    int occMaxShell = 0, Nirrep_compact = 0;
    if(occMaxL == -1)    occMaxShell = size_shell;
    else
    {
        for(int ii = 0; ii < size_shell; ii++)
        {
            if(shell_list(ii).l <= occMaxL)
                occMaxShell++;
            else
                break;
        }
    }
    for(int ii = 0; ii < occMaxShell; ii++)
    {
        if(shell_list(ii).l == 0) Nirrep_compact += 1;
        else Nirrep_compact += 2;
    }
    
    int2eJK int_2e_JK;
    int_2e_JK.J = new double***[Nirrep_compact];
    int_2e_JK.K = new double***[Nirrep_compact];
    for(int ii = 0; ii < Nirrep_compact; ii++)
    {
        int_2e_JK.J[ii] = new double**[Nirrep_compact];
        int_2e_JK.K[ii] = new double**[Nirrep_compact];
    }
    
    int int_tmp1_p = 0;
    for(int pshell = 0; pshell < occMaxShell; pshell++)
    {
    int l_p = shell_list(pshell).l, int_tmp1_q = 0;
    for(int qshell = 0; qshell < occMaxShell; qshell++)
    {
        int l_q = shell_list(qshell).l, l_max = max(l_p,l_q), LmaxJ = min(l_p+l_p, l_q+l_q)+1, LmaxK = l_p+l_q+1;
        int size_gtos_p = shell_list(pshell).coeff.rows(), size_gtos_q = shell_list(qshell).coeff.rows();
        int size_tmp_p = (l_p == 0) ? 1 : 2, size_tmp_q = (l_q == 0) ? 1 : 2;
        double array_radial_J[LmaxJ+1][size_gtos_p*size_gtos_p][size_gtos_q*size_gtos_q][size_tmp_p][size_tmp_q];
        double array_radial_K[LmaxK+1][size_gtos_p*size_gtos_q][size_gtos_q*size_gtos_p][size_tmp_p][size_tmp_q];
        double array_angular_J[LmaxJ+1][size_tmp_p][size_tmp_q], array_angular_K[LmaxK+1][size_tmp_p][size_tmp_q];

        StartTime = clock();
        #pragma omp parallel  for
        for(int twojj_p = abs(2*l_p-1); twojj_p <= 2*l_p+1; twojj_p = twojj_p + 2)
        for(int twojj_q = abs(2*l_q-1); twojj_q <= 2*l_q+1; twojj_q = twojj_q + 2)
        {
            int sym_ap = twojj_p - 2*l_p, sym_aq = twojj_q - 2*l_q;
            int index_tmp_p = (l_p > 0) ? 1 - (2*l_p+1 - twojj_p)/2 : 0;
            int index_tmp_q = (l_q > 0) ? 1 - (2*l_q+1 - twojj_q)/2 : 0;

            for(int tmp = LmaxJ; tmp >= 0; tmp -= 2)
            {
                double tmp_d = 0.0;
                for(int mq = 0; mq < twojj_q + 1; mq++)
                {
                    if(intType.substr(0,4) == "LSLS")
                        tmp_d += int2e_get_angular_gaunt_LSLS(l_p, twojj_p, sym_ap, l_p, twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, tmp);
                    else if(intType.substr(0,4) == "LSSL")
                        tmp_d += int2e_get_angular_gaunt_LSSL(l_p, twojj_p, sym_ap, l_p, twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, tmp);
                    else
                    {
                        cout << "ERROR: Unkonwn intType in get_h2e_JK_gaunt." << endl;
                        exit(99);
                    }
                }
                tmp_d /= (twojj_q + 1);
                array_angular_J[tmp][index_tmp_p][index_tmp_q] = tmp_d;
            }
            for(int tmp = LmaxK; tmp >= 0; tmp -= 2)
            {
                double tmp_d = 0.0;
                for(int mq = 0; mq < twojj_q + 1; mq++)
                {
                    if(intType.substr(0,4) == "LSLS")
                        tmp_d += int2e_get_angular_gaunt_LSLS(l_p, twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, l_p, twojj_p, sym_ap, tmp);
                    else if(intType.substr(0,4) == "LSSL")
                        tmp_d += int2e_get_angular_gaunt_LSSL(l_p, twojj_p, sym_ap, l_q, 2*mq-twojj_q, sym_aq, l_q, 2*mq-twojj_q, sym_aq, l_p, twojj_p, sym_ap, tmp);
                    else
                    {
                        cout << "ERROR: Unkonwn intType in get_h2e_JK_gaunt." << endl;
                        exit(99);
                    }
                }
                tmp_d /= (twojj_q + 1);
                array_angular_K[tmp][index_tmp_p][index_tmp_q] = tmp_d;
            }
        }
        EndTime = clock();
        time_a += (EndTime - StartTime)/(double)CLOCKS_PER_SEC;

        StartTime = clock();
        #pragma omp parallel  for
        for(int tt = 0; tt < size_gtos_p*size_gtos_p*size_gtos_q*size_gtos_q; tt++)
        {
            int e1J = tt/(size_gtos_q*size_gtos_q);
            int e2J = tt - e1J*(size_gtos_q*size_gtos_q);
            int ii = e1J/size_gtos_p, jj = e1J - ii*size_gtos_p;
            int kk = e2J/size_gtos_q, ll = e2J - kk*size_gtos_q;
            int e1K = ii*size_gtos_q+ll, e2K = kk*size_gtos_p+jj;
            MatrixXd radial_2e_list_J[LmaxJ+1], radial_2e_list_K[LmaxK+1];
            double a_i_J = shell_list(pshell).exp_a(ii), a_j_J = shell_list(pshell).exp_a(jj), a_k_J = shell_list(qshell).exp_a(kk), a_l_J = shell_list(qshell).exp_a(ll);
            double a_i_K = shell_list(pshell).exp_a(ii), a_j_K = shell_list(qshell).exp_a(ll), a_k_K = shell_list(qshell).exp_a(kk), a_l_K = shell_list(pshell).exp_a(jj);
        
            if(intType.substr(0,4) == "LSLS")
            {
                for(int LL = LmaxJ; LL >= 0; LL -= 2)
                {
                    radial_2e_list_J[LL].resize(4,1);
                    radial_2e_list_J[LL](0,0) = int2e_get_radial(l_p,a_i_J,l_p+1,a_j_J,l_q,a_k_J,l_q+1,a_l_J,LL);
                    if(l_p != 0)
                        radial_2e_list_J[LL](1,0) = int2e_get_radial(l_p,a_i_J,l_p-1,a_j_J,l_q,a_k_J,l_q+1,a_l_J,LL);
                    if(l_q != 0)
                        radial_2e_list_J[LL](2,0) = int2e_get_radial(l_p,a_i_J,l_p+1,a_j_J,l_q,a_k_J,l_q-1,a_l_J,LL);
                    if(l_p != 0 && l_q != 0)
                        radial_2e_list_J[LL](3,0) = int2e_get_radial(l_p,a_i_J,l_p-1,a_j_J,l_q,a_k_J,l_q-1,a_l_J,LL);
                }
                for(int LL = LmaxK; LL >= 0; LL -= 2)
                {
                    radial_2e_list_K[LL].resize(4,1);
                    radial_2e_list_K[LL](0,0) = int2e_get_radial(l_p,a_i_K,l_q+1,a_j_K,l_q,a_k_K,l_p+1,a_l_K,LL);
                    if(l_q != 0)
                        radial_2e_list_K[LL](1,0) = int2e_get_radial(l_p,a_i_K,l_q-1,a_j_K,l_q,a_k_K,l_p+1,a_l_K,LL);
                    if(l_p != 0)
                        radial_2e_list_K[LL](2,0) = int2e_get_radial(l_p,a_i_K,l_q+1,a_j_K,l_q,a_k_K,l_p-1,a_l_K,LL);
                    if(l_p != 0 && l_q != 0)
                        radial_2e_list_K[LL](3,0) = int2e_get_radial(l_p,a_i_K,l_q-1,a_j_K,l_q,a_k_K,l_p-1,a_l_K,LL);
                }
            }
            else if(intType.substr(0,4) == "LSSL")
            {
                for(int LL = LmaxJ; LL >= 0; LL -= 2)
                {
                    radial_2e_list_J[LL].resize(4,1);
                    radial_2e_list_J[LL](0,0) = int2e_get_radial(l_p,a_i_J,l_p+1,a_j_J,l_q+1,a_k_J,l_q,a_l_J,LL);
                    if(l_p != 0)
                        radial_2e_list_J[LL](1,0) = int2e_get_radial(l_p,a_i_J,l_p-1,a_j_J,l_q+1,a_k_J,l_q,a_l_J,LL);
                    if(l_q != 0)
                        radial_2e_list_J[LL](2,0) = int2e_get_radial(l_p,a_i_J,l_p+1,a_j_J,l_q-1,a_k_J,l_q,a_l_J,LL);
                    if(l_p != 0 && l_q != 0)
                        radial_2e_list_J[LL](3,0) = int2e_get_radial(l_p,a_i_J,l_p-1,a_j_J,l_q-1,a_k_J,l_q,a_l_J,LL);
                }
                for(int LL = LmaxK; LL >= 0; LL -= 2)
                {
                    radial_2e_list_K[LL].resize(4,1);
                    radial_2e_list_K[LL](0,0) = int2e_get_radial(l_p,a_i_K,l_q+1,a_j_K,l_q+1,a_k_K,l_p,a_l_K,LL);
                    if(l_q != 0)
                    {
                        radial_2e_list_K[LL](1,0) = int2e_get_radial(l_p,a_i_K,l_q-1,a_j_K,l_q+1,a_k_K,l_p,a_l_K,LL);
                        radial_2e_list_K[LL](2,0) = int2e_get_radial(l_p,a_i_K,l_q+1,a_j_K,l_q-1,a_k_K,l_p,a_l_K,LL);
                        radial_2e_list_K[LL](3,0) = int2e_get_radial(l_p,a_i_K,l_q-1,a_j_K,l_q-1,a_k_K,l_p,a_l_K,LL);
                    }
                }
            }
            else
            {
                cout << "ERROR: Unkonwn intType in get_h2e_JK_gaunt." << endl;
                exit(99);
            }
                
            for(int twojj_p = abs(2*l_p-1); twojj_p <= 2*l_p+1; twojj_p = twojj_p + 2)
            for(int twojj_q = abs(2*l_q-1); twojj_q <= 2*l_q+1; twojj_q = twojj_q + 2)
            {
                int index_tmp_p = (l_p > 0) ? 1 - (2*l_p+1 - twojj_p)/2 : 0;
                int index_tmp_q = (l_q > 0) ? 1 - (2*l_q+1 - twojj_q)/2 : 0;
                int sym_ap = twojj_p - 2*l_p, sym_aq = twojj_q - 2*l_q;
                double k_p = -(twojj_p+1.0)*sym_ap/2.0, k_q = -(twojj_q+1.0)*sym_aq/2.0;
                double norm_J = shell_list(pshell).norm(ii) * shell_list(pshell).norm(jj) * shell_list(qshell).norm(kk) * shell_list(qshell).norm(ll), norm_K = shell_list(pshell).norm(ii) * shell_list(qshell).norm(ll) * shell_list(qshell).norm(kk) * shell_list(pshell).norm(jj);
                double lk1 = 1+l_p+k_p, lk2 = 1+l_p+k_p, lk3 = 1+l_q+k_q, lk4 = 1+l_q+k_q, a1 = shell_list(pshell).exp_a(ii), a2 = shell_list(pshell).exp_a(jj), a3 = shell_list(qshell).exp_a(kk), a4 = shell_list(qshell).exp_a(ll);

                for(int tmp = LmaxJ; tmp >= 0; tmp -= 2)
                {
                    if(intType == "LSLS")
                    {
                        array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] = 4.0*a2*a4 * radial_2e_list_J[tmp](0,0);
                        if(l_p != 0 && l_q != 0)
                            array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] += lk2*lk4 * radial_2e_list_J[tmp](3,0)
                                    - 2.0*a4*lk2 * radial_2e_list_J[tmp](1,0) - 2.0*a2*lk4 * radial_2e_list_J[tmp](2,0);
                        else if(l_p != 0 && l_q == 0)
                            array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] -= 2.0*a4*lk2 * radial_2e_list_J[tmp](1,0);
                        else if(l_p == 0 && l_q != 0)
                            array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] -= 2.0*a2*lk4 * radial_2e_list_J[tmp](2,0);
                        array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] /= norm_J * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "LSSL")
                    {
                        array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] = 4.0*a2*a3 * radial_2e_list_J[tmp](0,0);
                        if(l_p != 0 && l_q != 0)
                            array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] += lk2*lk3 * radial_2e_list_J[tmp](3,0)
                                    - 2.0*a3*lk2 * radial_2e_list_J[tmp](1,0) - 2.0*a2*lk3 * radial_2e_list_J[tmp](2,0);
                        else if(l_p != 0 && l_q == 0)
                            array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] -= 2.0*a3*lk2 * radial_2e_list_J[tmp](1,0);
                        else if(l_p == 0 && l_q != 0)
                            array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] -= 2.0*a2*lk3 * radial_2e_list_J[tmp](2,0);
                        array_radial_J[tmp][e1J][e2J][index_tmp_p][index_tmp_q] /= -1.0 * norm_J * 4.0 * pow(speedOfLight,2);
                    }
                    else
                    {
                        cout << "ERROR: Unkonwn intType in get_h2e_JK_gaunt." << endl;
                        exit(99);
                    }
                }
                lk2 = 1+l_q+k_q; lk4 = 1+l_p+k_p; 
                a2 = shell_list(qshell).exp_a(ll); a4 = shell_list(pshell).exp_a(jj);
                for(int tmp = LmaxK; tmp >= 0; tmp -= 2)
                {
                    if(intType == "LSLS")
                    {
                        array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] = 4.0*a2*a4 * radial_2e_list_K[tmp](0,0);
                        if(l_p != 0 && l_q != 0)
                            array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] += lk2*lk4 * radial_2e_list_K[tmp](3,0) 
                                    - 2.0*a4*lk2 * radial_2e_list_K[tmp](1,0) - 2.0*a2*lk4 * radial_2e_list_K[tmp](2,0);
                        else if(l_p == 0 && l_q != 0)
                            array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] -= 2.0*a4*lk2 * radial_2e_list_K[tmp](1,0);
                        else if(l_p != 0 && l_q == 0)
                            array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] -= 2.0*a2*lk4 * radial_2e_list_K[tmp](2,0);
                        array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] /= norm_K * 4.0 * pow(speedOfLight,2);
                    }
                    else if(intType == "LSSL")
                    {
                        array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] = 4.0*a2*a3 * radial_2e_list_K[tmp](0,0);
                        if(l_q != 0)
                            array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] += lk2*lk3 * radial_2e_list_K[tmp](3,0) 
                                    - 2.0*a3*lk2 * radial_2e_list_K[tmp](1,0) - 2.0*a2*lk3 * radial_2e_list_K[tmp](2,0);
                        array_radial_K[tmp][e1K][e2K][index_tmp_p][index_tmp_q] /= -1.0 * norm_K * 4.0 * pow(speedOfLight,2);
                    }
                    else
                    {
                        cout << "ERROR: Unkonwn intType in get_h2e_JK_gaunt." << endl;
                        exit(99);
                    }
                }
            }
        }
        EndTime = clock();
        time_r += (EndTime - StartTime)/(double)CLOCKS_PER_SEC;

        StartTime = clock();
        int l_p_cycle = (l_p == 0) ? 1 : 2, l_q_cycle = (l_q == 0) ? 1 : 2;
        for(int int_tmp2_p = 0; int_tmp2_p < l_p_cycle; int_tmp2_p++)
        for(int int_tmp2_q = 0; int_tmp2_q < l_q_cycle; int_tmp2_q++)
        {
            int_2e_JK.J[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q] = new double*[size_gtos_p*size_gtos_p];
            for(int iii = 0; iii < size_gtos_p*size_gtos_p; iii++)
                int_2e_JK.J[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][iii] = new double[size_gtos_q*size_gtos_q];
            int_2e_JK.K[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q] = new double*[size_gtos_p*size_gtos_q];
            for(int iii = 0; iii < size_gtos_p*size_gtos_q; iii++)
                int_2e_JK.K[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][iii] = new double[size_gtos_p*size_gtos_q];
            #pragma omp parallel  for
            for(int tt = 0; tt < size_gtos_p*size_gtos_p*size_gtos_q*size_gtos_q; tt++)
            {
                int e1J = tt/(size_gtos_q*size_gtos_q);
                int e2J = tt - e1J*(size_gtos_q*size_gtos_q);
                int e1K = tt/(size_gtos_p*size_gtos_q);
                int e2K = tt - e1K*(size_gtos_p*size_gtos_q);
                int_2e_JK.J[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1J][e2J] = 0.0;
                int_2e_JK.K[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1K][e2K] = 0.0;
                for(int tmp = LmaxJ; tmp >= 0; tmp = tmp - 2)
                    int_2e_JK.J[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1J][e2J] += array_radial_J[tmp][e1J][e2J][int_tmp2_p][int_tmp2_q] * array_angular_J[tmp][int_tmp2_p][int_tmp2_q];
                for(int tmp = LmaxK; tmp >= 0; tmp = tmp - 2)
                    int_2e_JK.K[int_tmp1_p+int_tmp2_p][int_tmp1_q+int_tmp2_q][e1K][e2K] += array_radial_K[tmp][e1K][e2K][int_tmp2_p][int_tmp2_q] * array_angular_K[tmp][int_tmp2_p][int_tmp2_q];
            }
        }
        EndTime = clock();
        time_c += (EndTime - StartTime)/(double)CLOCKS_PER_SEC;
        int_tmp1_q += (l_q == 0) ? 1 : 2;
    }
    int_tmp1_p += (l_p == 0) ? 1 : 2;
    }

    cout << time_a << "\t" << time_r << "\t" << time_c <<endl;
    return int_2e_JK;
}


void INT_SPH::get_h2e_JK_gaunt_direct(int2eJK& LSLS, int2eJK& LSSL, const int& occMaxL, const bool& spinFree)
{
    LSLS = get_h2e_JK_gaunt_compact("LSLS",occMaxL);
    LSSL = get_h2e_JK_gaunt_compact("LSSL",occMaxL);
}