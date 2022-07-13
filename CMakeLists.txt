cmake_minimum_required(VERSION 3.10)
project(x2camf)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
set(SHARED_SRC
  src/dhf_sph_ca.cpp
  src/dhf_sph_pcc.cpp
  src/dhf_sph.cpp
  src/general.cpp
  src/gsl_functions.cpp
  src/int_sph_basic.cpp
  src/int_sph_gauge.cpp
  src/int_sph_gaunt.cpp
  src/int_sph.cpp
)

add_subdirectory(pybind11)
include_directories(include)
link_libraries(Eigen3::Eigen)

add_library(shared_objects OBJECT ${SHARED_SRC})

#pybind11_add_module(libx2camf SHARED ${SHARED_SRC} executables/pyx2camf.cpp)
pybind11_add_module(libx2camf ${SHARED_SRC} executables/pyx2camf.cpp)
#add_executable(amf_pyscf $<TARGET_OBJECTS:shared_objects> executables/pyscf_interface.cpp)
#add_executable(main      $<TARGET_OBJECTS:shared_objects> executables/main.cpp)
#add_executable(test      $<TARGET_OBJECTS:shared_objects> executables/test.cpp)

#install(TARGETS amf_pyscf main test)
