cmake_minimum_required (VERSION 2.8)
project (tinyrenderer)

add_definitions(-D_USE_MATH_DEFINES)
include(CheckCXXCompilerFlag)

function(enable_cxx_compiler_flag_if_supported flag)
    string(FIND "${CMAKE_CXX_FLAGS}" "${flag}" flag_already_set)
    if(flag_already_set EQUAL -1)
        check_cxx_compiler_flag("${flag}" flag_supported)
        if(flag_supported)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}" PARENT_SCOPE)
        endif()
        unset(flag_supported CACHE)
    endif()
endfunction()

enable_cxx_compiler_flag_if_supported("-Wall")
enable_cxx_compiler_flag_if_supported("-Wextra")
enable_cxx_compiler_flag_if_supported("-pedantic")
enable_cxx_compiler_flag_if_supported("-std=c++14")
enable_cxx_compiler_flag_if_supported("-O3")
enable_cxx_compiler_flag_if_supported("-fopenmp")

file(GLOB SOURCES *.h *.cpp)

add_library(${PROJECT_NAME} ${SOURCES})

add_executable(hello_tinyrenderer examples/cpp/hello_tinyrenderer)
target_link_libraries(hello_tinyrenderer PUBLIC tinyrenderer)
target_include_directories(hello_tinyrenderer PUBLIC .)

subdirs(python)
