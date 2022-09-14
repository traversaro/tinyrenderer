cmake_minimum_required (VERSION 2.8)
project (tinyrenderer)

add_definitions(-D_USE_MATH_DEFINES)

file(GLOB SOURCES *.h *.cpp)

add_library(${PROJECT_NAME} ${SOURCES})

add_executable(hello_tinyrenderer examples/cpp/hello_tinyrenderer)
target_link_libraries(hello_tinyrenderer PUBLIC tinyrenderer)
target_include_directories(hello_tinyrenderer PUBLIC .)

subdirs(python)
