CPP = icpc
CPPFLAG = -O3 -std=c++11 -qopenmp -mkl -DEIGEN_USE_MKL_ALL
EIGEN = ~/apps/Eigen3
GSL = ~/apps/gsl-2.6
MAIN = main.o gto.o scf.o x2c.o gto_spinor.o  
TEST = test.o gto.o scf.o x2c.o gto_spinor.o

main.exe: ${MAIN}
	${CPP} ${CPPFLAG} -I ${EIGEN} -I ${GSL} -L ${GSL}/.libs ${MAIN} -l gsl -o main.exe

test.exe: ${TEST}
	${CPP} ${CPPFLAG} -I ${EIGEN} -I ${GSL} -L ${GSL}/.libs ${TEST} -l gsl -o test.exe


%.o: %.cpp
	$(CPP) $(CPPFLAG) -c $< -o $@ -I ${EIGEN}


clean:
	rm *.o *.exe
