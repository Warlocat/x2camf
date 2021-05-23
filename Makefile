CPP = icpc
FF = ifort
CPPFLAG = -O3 -std=c++11 -qopenmp -mkl -DEIGEN_USE_MKL_ALL
EIGEN = ~/apps/Eigen3
GSL = ~/apps/gsl-2.6
MAIN = main.o int_sph.o general.o dhf_sph.o dhf_sph_ca.o wfile.o rfile.o prvecr.o
TEST = test.o int_sph.o general.o dhf_sph.o dhf_sph_ca.o

main.exe: ${MAIN}
	${CPP} ${CPPFLAG} -I ${EIGEN} -I ${GSL} -L ${GSL}/.libs ${MAIN} -l gsl -l ifcore -o main.exe

test.exe: ${TEST}
	${CPP} ${CPPFLAG} -I ${EIGEN} -I ${GSL} -L ${GSL}/.libs ${TEST} -l gsl -o test.exe

%.o: %.cpp
	$(CPP) $(CPPFLAG) -c $< -o $@ -I ${EIGEN} -I ${GSL}

%.o: %.f
	${FF} -c $< -o $@


clean:
	rm *.o *.exe
