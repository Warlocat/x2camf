CPP = icpc
FF = ifort
CPPFLAG = -O3 -std=c++11 -qopenmp -mkl -DEIGEN_USE_MKL_ALL
LIBSFLAG = -I ${EIGEN} -I ${GSL} -L ${GSL}/.libs -l gsl -l ifcore 
EIGEN = ~/apps/Eigen3
GSL = ~/apps/gsl-2.6
FILES = int_sph.o int_sph_gaunt.o general.o dhf_sph.o dhf_sph_ca.o dhf_sph_pcc.o wfile.o rfile.o prvecr.o
MAIN = main.o ${FILES}
TEST = test.o ${FILES}

main.exe: ${MAIN}
	${CPP} ${CPPFLAG} ${LIBSFLAG} ${MAIN} -o main.exe

test.exe: ${TEST}
	${CPP} ${CPPFLAG} ${LIBSFLAG} ${TEST} -o test.exe

%.o: %.cpp
	$(CPP) $(CPPFLAG) -c $< -o $@ -I ${EIGEN} -I ${GSL}

%.o: %.f
	${FF} -c $< -o $@


clean:
	rm *.o *.exe
