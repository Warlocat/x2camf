#ifndef GENERAL_H_
#define GENERAL_H_

#include<Eigen/Dense>
#include<complex>
#include<string>
#include<ctime>
using namespace std;
using namespace Eigen;

static clock_t StartTime, EndTime; 

const double speedOfLight = 137.03599967994;

typedef Matrix<MatrixXd,-1,1> vMatrixXd;
typedef Matrix<VectorXd,-1,1> vVectorXd;
typedef Matrix<MatrixXd,-1,-1> mMatrixXd;


/*
    Read and write matrix
*/
template<typename T> void writeMatrixBinary(const Matrix<T,-1,-1>& inputM, const string& filename)
{
    ofstream ofs;
    ofs.open(filename, ios::binary);
        for(int ii = 0; ii < inputM.rows(); ii++)
        for(int jj = 0; jj < inputM.cols(); jj++)
        {
            ofs.write((char*) &inputM(ii,jj), sizeof(T));
        }
    ofs.close();
    return;
}
template<typename T> void readMatrixBinary(Matrix<T,-1,-1>& inputM, const string& filename)
{
    ifstream ifs;
    ifs.open(filename, ios::binary);
    if(!ifs)
    {
        cout << "ERROR opening file " << filename << endl;
        exit(99);
    }
        for(int ii = 0; ii < inputM.rows(); ii++)
        for(int jj = 0; jj < inputM.cols(); jj++)
        {
            ifs.read((char*) &inputM(ii,jj), sizeof(T));
        }
    ifs.close();
    return;
}
template<typename T> void writeMatrixBinary(T* inputM, const int& size, const string& filename)
{
    ofstream ofs;
    ofs.open(filename, ios::binary);
        for(int ii = 0; ii < size; ii++)
        {
            ofs.write((char*) &(inputM[ii]), sizeof(T));
        }
    ofs.close();
    return;
}
template<typename T> void readMatrixBinary(T* inputM, const int& size, const string& filename)
{
    ifstream ifs;
    inputM = new T[size];
    ifs.open(filename, ios::binary);
    if(!ifs)
    {
        cout << "ERROR opening file " << filename << endl;
        exit(99);
    }
        for(int ii = 0; ii < size; ii++)
        {
            ifs.read((char*) &(inputM[ii]), sizeof(T));
        }
    ifs.close();
    return;
}


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



/*
    gtos in form of angular shell
*/
struct intShell
{
    VectorXd exp_a, norm;
    MatrixXd coeff;
    int l;
};
/*
    Irreducible rep |j, l, m_j>
*/
struct irrep_jm
{
    int l, two_j, two_mj, size;
};
/*
    Coulomb and exchange integral
*/
struct int2eJK
{
    mMatrixXd J, K;
};

/* factorial and double factorial functions */
double factorial(const int& n);
double double_factorial(const int& n);

/* evaluate wigner 3j symbols */
double wigner_3j(const int& l1, const int& l2, const int& l3, const int& m1, const int& m2, const int& m3);
double wigner_3j_zeroM(const int& l1, const int& l2, const int& l3);

/* transformation matrix for complex SH to solid SH */
complex<double> U_SH_trans(const int& mu, const int& mm);

/* evaluate "difference" between two MatrixXd */
double evaluateChange(const MatrixXd& M1, const MatrixXd& M2);
/* evaluate M^{-1/2} */
MatrixXd matrix_half_inverse(const MatrixXd& inputM);
/* evaluate M^{1/2} */
MatrixXd matrix_half(const MatrixXd& inputM);
/* solver for generalized eigen equation MC=SCE, s_h_i = S^{1/2} */
void eigensolverG(const MatrixXd& inputM, const MatrixXd& s_h_i, VectorXd& values, MatrixXd& vectors);


/* Static functions used in X2C */
namespace X2C
{
    MatrixXd get_X(const MatrixXd& S_, const MatrixXd& T_, const MatrixXd& W_, const MatrixXd& V_);
    MatrixXd get_X(const MatrixXd& coeff);
    MatrixXd get_R(const MatrixXd& S_, const MatrixXd& T_, const MatrixXd& X_);
    MatrixXd get_R(const MatrixXd& S_4c, const MatrixXd& X_);
    MatrixXd evaluate_h1e_x2c(const MatrixXd& S_, const MatrixXd& T_, const MatrixXd& W_, const MatrixXd& V_);
    MatrixXd evaluate_h1e_x2c(const MatrixXd& S_, const MatrixXd& T_, const MatrixXd& W_, const MatrixXd& V_, const MatrixXd& X_, const MatrixXd& R_);
    MatrixXd transform_4c_2c(const MatrixXd& M_4c, const MatrixXd XXX, const MatrixXd& RRR);
}


/* Reoder and basis transformation */
namespace Rotate
{
    
    /* Generate basis transformation matrix */
    MatrixXd jspinor2sph(const Matrix<irrep_jm, Dynamic, 1>& irrep_list);
    MatrixXcd sph2solid(const Matrix<irrep_jm, Dynamic, 1>& irrep_list);
    /* For CFOUR interface */
    MatrixXd reorder_m_cfour(const int& LL);
    MatrixXcd jspinor2cfour_interface_old(const Matrix<irrep_jm, Dynamic, 1>& irrep_list);
    /*
        Put one-electron integrals in a single matrix and reorder them.
        The new ordering is to put the single uncontracted spinors together (separate)
    */
    template<typename T> Matrix<T,-1,-1> unite_irrep(const Matrix<Matrix<T,-1,-1>,-1,1>& inputM, const Matrix<irrep_jm, Dynamic, 1>& irrep_list)
    {
        int size_spinor = 0, size_irrep = irrep_list.rows(), Lmax = irrep_list(size_irrep - 1).l;
        if(inputM.rows() != size_irrep)
        {
            cout << "ERROR: the size of inputM is not equal to Nirrep." << endl;
            exit(99);
        }
        for(int ir = 0; ir < size_irrep; ir++)
        {
            size_spinor += irrep_list(ir).size;
        }
        Matrix<T,-1,-1> outputM = Matrix<T,-1,-1>::Zero(size_spinor,size_spinor);
        int i_output = 0;
        for(int ir = 0; ir < size_irrep; ir += 4*irrep_list(ir).l+2)
        {
            for(int ii = 0; ii < irrep_list(ir).size; ii++)
            for(int jj = 0; jj < irrep_list(ir).size; jj++)
            for(int mi = 0; mi < 4*irrep_list(ir).l+2; mi++)
            {
                outputM(i_output + ii*(4*irrep_list(ir).l+2) + mi, i_output + jj*(4*irrep_list(ir).l+2) + mi) = inputM(ir+mi)(ii,jj);
            }
            i_output += (4*irrep_list(ir).l+2) * irrep_list(ir).size;
        }

        return outputM;
    }
    /* 
        Transfer separate basis to m-compact basis 
        px,py,pz,px,py,pz to px,px,py,py,pz,pz
    */
    template<typename T> Matrix<T,-1,-1> separate2mCompact(const Matrix<T,-1,-1>& inputM, const Matrix<irrep_jm, Dynamic, 1>& irrep_list)
    {
        int size = inputM.rows(), size_nr = size/2, size_irrep = irrep_list.rows(), Lmax = irrep_list(size_irrep - 1).l;
        int Lsize[Lmax+1];
        for(int ir = 0; ir < size_irrep; ir++)
        {
            Lsize[irrep_list(ir).l] = irrep_list(ir).size;
        }
        Matrix<T,-1,-1> outputM = Matrix<T,-1,-1>::Zero(size,size);
        int int_tmp = 0;
        for(int ll = 0; ll <= Lmax; ll++)
        {
            for(int mm = 0; mm < 2*ll+1; mm++)
            for(int nn = 0; nn < 2*ll+1; nn++)
            for(int ii = 0; ii < Lsize[ll]; ii++)
            for(int jj = 0; jj < Lsize[ll]; jj++)
            {
                outputM(int_tmp+mm*Lsize[ll]+ii,int_tmp+nn*Lsize[ll]+jj) = inputM(int_tmp+ii*(2*ll+1)+mm,int_tmp+jj*(2*ll+1)+nn);
                outputM(size_nr+int_tmp+mm*Lsize[ll]+ii,int_tmp+nn*Lsize[ll]+jj) = inputM(size_nr+int_tmp+ii*(2*ll+1)+mm,int_tmp+jj*(2*ll+1)+nn);
                outputM(int_tmp+mm*Lsize[ll]+ii,size_nr+int_tmp+nn*Lsize[ll]+jj) = inputM(int_tmp+ii*(2*ll+1)+mm,size_nr+int_tmp+jj*(2*ll+1)+nn);
                outputM(size_nr+int_tmp+mm*Lsize[ll]+ii,size_nr+int_tmp+nn*Lsize[ll]+jj) = inputM(size_nr+int_tmp+ii*(2*ll+1)+mm,size_nr+int_tmp+jj*(2*ll+1)+nn);
            }
            int_tmp += (2*ll+1)*Lsize[ll];
        }
        return outputM;
    }
} // namespace Rotate



#endif