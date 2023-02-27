CPP = icpc
FF = ifort
CPPFLAG = -O3 -std=c++11 -fopenmp -qmkl -I include
LIBSFLAG = -l ifcore
FILES = int_sph_basic.o int_sph.o int_sph_gaunt.o int_sph_gauge.o general.o dhf_sph.o dhf_sph_ca.o dhf_sph_pcc.o finterface.o mkl_itrf.o
CFOUR = cfour_interface.o ${FILES}

xx2camf: ${CFOUR}
	${CPP} ${CPPFLAG} ${LIBSFLAG} ${CFOUR} -o xx2camf
%.o: %.cpp
	$(CPP) $(CPPFLAG) -c $< -o $@ 
%.o: %.f90
	${FF} -c $< -o $@
clean:
	rm *.o xx2camf
