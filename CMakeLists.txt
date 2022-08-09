cmake_minimum_required(VERSION 3.9)
project(x2camf)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP)
set(SHARED_SRC
  src/dhf_sph_ca.cpp
  src/dhf_sph_pcc.cpp
  src/dhf_sph.cpp
  src/general.cpp
  src/int_sph_basic.cpp
  src/int_sph_gauge.cpp
  src/int_sph_gaunt.cpp
  src/int_sph.cpp
)

set(CMAKE_CXX_FLAGS "-fPIC")

add_subdirectory(pybind11)
include_directories(include)
link_libraries(Eigen3::Eigen)
link_libraries(OpenMP::OpenMP_CXX)

add_library(shared_objects OBJECT ${SHARED_SRC})

pybind11_add_module(libx2camf $<TARGET_OBJECTS:shared_objects> executables/pyx2camf.cpp)
#pybind11_add_module(libx2camf ${SHARED_SRC} executables/pyx2camf.cpp)
add_executable(amf_pyscf $<TARGET_OBJECTS:shared_objects> executables/pyscf_interface.cpp)
add_executable(xray      $<TARGET_OBJECTS:shared_objects> executables/xrayCorrection.cpp)
# add_executable(main      $<TARGET_OBJECTS:shared_objects> executables/main.cpp)
#add_executable(test      $<TARGET_OBJECTS:shared_objects> executables/test.cpp)

enable_language (Fortran)
include(FortranCInterface)
FortranCInterface_VERIFY(CXX)
add_executable(xx2camf      $<TARGET_OBJECTS:shared_objects> src/finterface.f90 executables/cfour_interface.cpp)

#install(TARGETS amf_pyscf main test)
