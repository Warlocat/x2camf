CPP = icpc
FF = ifort
CPPFLAG = -O3 -std=c++11 -fopenmp -qmkl -I include -D EIGEN_USE_MKL_ALL
LIBSFLAG = -l ifcore -D EIGEN_USE_MKL_ALL
FILES = src/int_sph_basic.o src/int_sph.o src/int_sph_gaunt.o src/int_sph_gauge.o src/general.o src/dhf_sph.o src/dhf_sph_ca.o src/dhf_sph_pcc.o src/finterface.o src/mkl_itrf.o
CFOUR = executables/cfour_interface.o ${FILES}

xx2camf: ${CFOUR}
	${CPP} ${CPPFLAG} ${LIBSFLAG} ${CFOUR} -o xx2camf
%.o: %.cpp
	$(CPP) $(CPPFLAG) -c $< -o $@ 
%.o: %.f90
	${FF} -c $< -o $@
clean:
	rm executables/*.o src/*.o xx2camf
