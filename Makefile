CPP = icpc
FF = ifort
CPPFLAG = -O3 -std=c++11 -fopenmp -mkl -DEIGEN_USE_MKL_ALL
LIBSFLAG = -I ${EIGEN} -I ${GSL} -L ${GSL}/lib -l gsl -l ifcore 
EIGEN = ~/apps/Eigen3
GSL = ~/apps/gsl-2.6
FILES = int_sph_basic.o int_sph.o int_sph_gaunt.o int_sph_gauge.o general.o dhf_sph.o dhf_sph_ca.o dhf_sph_pcc.o wfile.o rfile.o prvecr.o
MAIN = main.o ${FILES}
BASIS = basisGenerator.o ${FILES}
TEST = test.o ${FILES}
PYSCF = pyscf_interface.o ${FILES}

main.exe: ${MAIN}
	${CPP} ${CPPFLAG} ${LIBSFLAG} ${MAIN} -o main.exe

amf_pyscf.exe: ${PYSCF}
	${CPP} ${CPPFLAG} ${LIBSFLAG} ${PYSCF} -o amf_pyscf.exe

basisGenerator.exe: ${BASIS}
	${CPP} ${CPPFLAG} ${LIBSFLAG} ${BASIS} -o basisGenerator.exe

test.exe: ${TEST}
	${CPP} ${CPPFLAG} ${LIBSFLAG} ${TEST} -o test.exe

%.o: %.cpp
	$(CPP) $(CPPFLAG) -c $< -o $@ -I ${EIGEN} -I ${GSL}

%.o: %.f
	${FF} -c $< -o $@


clean:
	rm *.o *.exe
