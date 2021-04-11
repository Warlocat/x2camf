CPP = icpc
CPPFLAG = -O3 -std=c++11 -qopenmp -mkl -DEIGEN_USE_MKL_ALL
EIGEN = ~/apps/Eigen3
GSL = ~/apps/gsl-2.6
MAIN = main.o int_sph.o general.o dhf_sph.o

main.exe: ${MAIN}
	${CPP} ${CPPFLAG} -I ${EIGEN} -I ${GSL} -L ${GSL}/.libs ${MAIN} -l gsl -o main.exe

%.o: %.cpp
	$(CPP) $(CPPFLAG) -c $< -o $@ -I ${EIGEN}


clean:
	rm *.o *.exe
