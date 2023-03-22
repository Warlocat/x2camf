
topdir:=..
builddir:=..

include $(builddir)/make.config
CXXFLAGS+= -O3 -std=c++2a -fopenmp -I.  -DMKL_ILP64  -I"${MKLROOT}/include"
INTEL =  -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group
LIBSFLAG = -static-intel ${INTEL}
FILES = int_sph_basic.o int_sph.o int_sph_gaunt.o int_sph_gauge.o general.o dhf_sph.o dhf_sph_ca.o dhf_sph_pcc.o mkl_itrf.o
CFOUR = cfour_interface.o ${FILES}

$(bindir)/xx2camf: ${CFOUR}
	${CXX} ${CFOUR} ${CXXFLAGS} ${LIBSFLAG} -o $@
%.o: %.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS)  
clean:
	rm -f *.o $(bindir)/xx2camf

