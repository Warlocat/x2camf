CPP = icpc
FF = ifort
CPPFLAG = -O3 -std=c++11 -fopenmp 
LIBSFLAG = -I ${EIGEN} -l ifcore 
EIGEN = ~/apps/Eigen3
FILES = int_sph_basic.o int_sph.o int_sph_gaunt.o int_sph_gauge.o general.o dhf_sph.o dhf_sph_ca.o dhf_sph_pcc.o wfile.o rfile.o prvecr.o gsl_functions.o
MAIN = main.o ${FILES}
CFOUR = cfour_interface.o ${FILES}
BASIS = basisGenerator.o ${FILES}
TEST = test.o ${FILES}
PYSCF = pyscf_interface.o ${FILES}

all: main.exe amf_cfour.exe amf_pyscf.exe basisGenerator.exe test.exe


main.exe: ${MAIN}
	${CPP} ${CPPFLAG} ${LIBSFLAG} ${MAIN} -o main.exe

amf_cfour.exe: ${CFOUR}
	${CPP} ${CPPFLAG} ${LIBSFLAG} ${CFOUR} -o amf_cfour.exe

amf_pyscf.exe: ${PYSCF}
	${CPP} ${CPPFLAG} ${LIBSFLAG} ${PYSCF} -o amf_pyscf.exe

basisGenerator.exe: ${BASIS}
	${CPP} ${CPPFLAG} ${LIBSFLAG} ${BASIS} -o basisGenerator.exe

test.exe: ${TEST}
	${CPP} ${CPPFLAG} ${LIBSFLAG} ${TEST} -o test.exe

%.o: %.cpp
	$(CPP) $(CPPFLAG) -c $< -o $@ -I ${EIGEN}

%.o: %.f
	${FF} -c $< -o $@


clean:
	rm *.o *.exe
