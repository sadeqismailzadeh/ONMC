#cmake_minimum_required(VERSION 3.10)
cmake_minimum_required(VERSION 3.12)
#set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")
project(dipoles C CXX)

if (UNIX)
    message("UNIX")
 #   set(PLATFORM UNIX)
elseif(WIN32)
    message("Windows")
#    set(PLATFORM MINGW)
endif ()

set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_EXTENSIONS ON)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)

if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native") #yoooooohoooooooooo
    message("default  flags = ${CMAKE_CXX_FLAGS}")
    #        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 yoooooohoooooooooo")
    message("COMPILER SUPPORTS MARCH_NATIVE")
endif()
message("default debug flags = ${CMAKE_CXX_FLAGS_DEBUG}")
message("default release flags = ${CMAKE_CXX_FLAGS_RELEASE} ")
message("default relwithdebinfo flags = ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
#    message("default minsizerel flags = ${CMAKE_CXX_FLAGS_MINSIZEREL}")

set(CMAKE_CXX_FLAGS_RELWITHDEBINFO " ${CMAKE_CXX_FLAGS} -g -O3 -Wall")
set(CMAKE_CXX_FLAGS_DEBUG " ${CMAKE_CXX_FLAGS_DEBUG} -g  -Wall")
#    set(CMAKE_CXX_FLAGS_RELSANITIZER "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
set(CMAKE_CXX_FLAGS_RELSANITIZER "-g ")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS} -flto")

IF(CMAKE_BUILD_TYPE MATCHES Debug)
    message("Debug mode")
    add_compile_definitions(_GLIBCXX_DEBUG)
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize=undefined")  # google sanitizer
ENDIF(CMAKE_BUILD_TYPE MATCHES Debug)

IF(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
    message("RelWithDebInfo mode")
    add_compile_definitions(_GLIBCXX_DEBUG)
    #        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize=undefined")  # google sanitizer
ENDIF(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)

IF(CMAKE_BUILD_TYPE MATCHES RelSanitizer)
    message("RelSanitizer mode")
    #        add_compile_definitions(_GLIBCXX_DEBUG)
    #        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")  #  AddressSanitizer, a fast memory error detector.
    #        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=hwaddress")  # Hardware-assisted AddressSanitizer
#            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined")  # google sanitizer
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=leak")  # google sanitizer
    #        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")  # google sanitizer

    if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=memory -fPIE -pie -fno-omit-frame-pointer -g -O2")
#        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=memory")  # google sanitizer
    endif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
ENDIF(CMAKE_BUILD_TYPE MATCHES RelSanitizer)

IF(CMAKE_BUILD_TYPE MATCHES Release)
    message("Release mode")
ENDIF(CMAKE_BUILD_TYPE MATCHES Release)


#    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast") # it breaks eigen
#-Ofast Disregard strict standards compliance. -Ofast enables all -O3 optimizations.
#It also enables optimizations that are not valid for all standard-compliant programs.

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -stdlib=libc++")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -stdlib=libstdc++")
endif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")

IF(CMAKE_COMPILER_IS_GNUCC)

ENDIF(CMAKE_COMPILER_IS_GNUCC)


find_package(Eigen3  REQUIRED fatal)
if(Eigen3_FOUND)
#    message("Found Eigen!!!!!!!")
endif()
if (FFTW_Non_Standard_Location)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/../ext/FindFFTW")
    find_package(FFTW)
endif ()

if (IS_VCPKG)
    find_package(hdf5 CONFIG REQUIRED)
    find_package(HighFive CONFIG REQUIRED)
    find_package(Eigen3 CONFIG REQUIRED)
else ()
    add_subdirectory(lib/HighFive-2.6.1)
endif ()


#IF(CMAKE_BUILD_TYPE MATCHES Release)
#    add_compile_definitions(EIGEN_DONT_VECTORIZE )
#ENDIF(CMAKE_BUILD_TYPE MATCHES Release)



#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wa,-aslh >list.txt")


include_directories(cpp)
include_directories(cpp/include)
include_directories(lib/fftwpp-2.05)
#include_directories(/opt/AMD/aocl/aocl-linux-gcc-3.1.0/include)


#link_directories(/usr/lib/x86_64-linux-gnu/)
#find_package(Threads REQUIRED) # pthread
find_package(OpenMP REQUIRED)
#find_package(PkgConfig REQUIRED)
#pkg_search_module(FFTW REQUIRED fftw3 IMPORTED_TARGET)

#add_compile_definitions(EIGEN_USE_BLAS)
#add_compile_definitions(EIGEN_DONT_VECTORIZE)
#add_compile_definitions(NDEBUG)

#add_executable(src
#        main.cpp
#        cpp/test/Test_2D_Direct.cpp
#        cpp/include/prng.h
#        cpp/include/MC.h
#        cpp/test/Test_2D_Direct.cpp
#        cpp/include/utils.h cpp/include/utils.cpp)

file(GLOB SOURCES
        *.h
        *.cpp
        )

add_executable(test_2d
        cpp/test/Test_2D_Ewald.cpp
        cpp/2d/MC2d.h
        cpp/2d/Lattice2d.cpp
        cpp/2d/Lattice2d.h
        cpp/include/utils.h cpp/include/utils.cpp
#        ${SOURCES}
        )

add_executable(prodMC
        cpp/test/prodMC.cpp
        cpp/2d/MC2d.h
        cpp/2d/Lattice2d.cpp
        cpp/2d/Lattice2d.h
        cpp/include/utils.h
        cpp/include/utils.cpp
        cpp/include/prng.h
        cpp/include/prng.cpp
        cpp/include/WalkerAlias.h
        cpp/2d/MC2d.cpp
        cpp/2d/Simulation.h
        cpp/2d/Simulation.cpp
        lib/fftwpp-2.05/fftw++.h
        lib/fftwpp-2.05/fftw++.cc
        lib/fftwpp-2.05/ArrayFFT.h
#        ${SOURCES}
        cpp/2d/MCProperties.cpp cpp/2d/MCProperties.h
        cpp/2d/Param.cpp cpp/2d/Param.h
        cpp/2d/MC2dArchive.cpp
        cpp/2d/MC2dDebug.cpp cpp/include/WalkerAlias.cpp)


#message("${CMAKE_CXX_COMPILER_VERSION}")
if(CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    find_package(Boost  REQUIRED COMPONENTS filesystem )
    target_link_libraries(prodMC Boost::filesystem)
endif()



if (IS_VCPKG)
    target_link_libraries(prodMC  Eigen3::Eigen)
    target_link_libraries(prodMC  libdeps HighFive HighFive_libdeps HighFive_HighFive)
    target_link_libraries(prodMC  OpenMP::OpenMP_CXX)
else ()
    target_link_libraries(prodMC  Eigen3::Eigen)
    target_link_libraries(prodMC  HighFive)
    target_link_libraries(prodMC  OpenMP::OpenMP_CXX)
endif ()


#target_link_libraries(prodMC  Threads::Threads)
#target_link_libraries(testMc -static)


if(FFTW_FOUND AND FFTW_Non_Standard_Location)
    target_link_libraries(prodMC FFTW::Double FFTW::DoubleOpenMP)
else()
    target_link_libraries(prodMC fftw3 fftw3_omp)
endif()

#set(TARGET testMc)
#set(OUTPUT_DIRECTORY "~")
#set(SOURCE_FILENAME MC2d)
#set(ASSEMBLY_LISTING_FILE MC2dlist.s)
#add_custom_command(TARGET ${TARGET}
#        POST_BUILD
#        COMMAND make ARGS ${SOURCE_FILENAME}.s
#        COMMAND ${CMAKE_COMMAND} -E copy
#        "${CMAKE_BINARY_DIR}/CMakeFiles/${TARGET}.dir/${SOURCE_FILENAME}.cpp.s"
#        "${OUTPUT_PATH}/${ASSEMBLY_LISTING_FILE}"
#        WORKING_DIRECTORY ${CMAKE_BINARY_DIR})


message("final debug flags = ${CMAKE_CXX_FLAGS_DEBUG}")
message("final release flags = ${CMAKE_CXX_FLAGS_RELEASE}")
message("final relwithdebinfo flags = ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
message("final relsanitizer flags = ${CMAKE_CXX_FLAGS_RELSANITIZER}")