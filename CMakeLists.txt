cmake_minimum_required(VERSION 3.16)

project(libsmoothuv2 LANGUAGES CXX)

find_package (Git)

if (GIT_FOUND)
    execute_process (COMMAND ${GIT_EXECUTABLE} describe --tags --abbrev=0
        OUTPUT_VARIABLE ver
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
else ()
    message (STATUS "GIT not found")
endif ()

add_library(smoothuv2 SHARED
    src/SmoothUV2.cpp
    src/SmoothUV2_SSE2.cpp
    src/SmoothUV2_SSSE3.cpp
    src/SmoothUV2_SSE41.cpp
    src/SmoothUV2_AVX2.cpp
    src/SmoothUV2_AVX512.cpp
)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

message(STATUS "Build type - ${CMAKE_BUILD_TYPE}")

set_source_files_properties(src/SmoothUV2_SSE2.cpp PROPERTIES COMPILE_OPTIONS "-mfpmath=sse;-msse2")
set_source_files_properties(src/SmoothUV2_SSSE3.cpp PROPERTIES COMPILE_OPTIONS "-mfpmath=sse;-mssse3")
set_source_files_properties(src/SmoothUV2_SSE41.cpp PROPERTIES COMPILE_OPTIONS "-mfpmath=sse;-msse4.1")
set_source_files_properties(src/SmoothUV2_AVX2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
set_source_files_properties(src/SmoothUV2_AVX512.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512dq;-mavx512vl;-mfma")

target_include_directories(smoothuv2 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    /usr/local/include/avisynth
)

set_target_properties(smoothuv2 PROPERTIES OUTPUT_NAME "smoothuv2.${ver}")

target_compile_features(smoothuv2 PRIVATE cxx_std_17)

include(GNUInstallDirs)

INSTALL(TARGETS smoothuv2 LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/avisynth")
        
# uninstall target
if(NOT TARGET uninstall)
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)

  add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()
