#!/usr/bin/python
#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


find_package(Python3 COMPONENTS NumPy Development.Module REQUIRED)
find_package(pybind11 REQUIRED)

IF(WIN32)
	SET(BUILD_SHARED_LIBS OFF CACHE BOOL "Shared Libs" FORCE)
ELSE(WIN32)
	SET(BUILD_SHARED_LIBS ON CACHE BOOL "Shared Libs" FORCE)
ENDIF(WIN32)

IF(APPLE)
	OPTION(BUILD_TINY_RENDERER_MAC_USE_PYTHON_FRAMEWORK "Set when you want to use the Python Framework on Mac" ON)
	IF(NOT BUILD_TINY_RENDERER_MAC_USE_PYTHON_FRAMEWORK)
		add_definitions(-DB3_NO_PYTHON_FRAMEWORK)
	ENDIF(NOT BUILD_TINY_RENDERER_MAC_USE_PYTHON_FRAMEWORK)
	OPTION(BUILD_TINY_RENDERER_SHOW_PY_VERSION "Set when you want to show the PY_MAJOR_VERSION and PY_MAJOR_VERSION using #pragme message." OFF)
	IF(BUILD_TINY_RENDERER_SHOW_PY_VERSION)
		add_definitions(-DB3_DUMP_PYTHON_VERSION)
	ENDIF()
ENDIF(APPLE)



##################################################################

SET(pytinyrenderer_SRCS
	pytinyrenderer.cc
	../geometry.cpp
	../geometry.h
	../our_gl.cpp
	../our_gl.h
	../model.cpp
	../model.h
	../tgaimage.cpp
	../tgaimage.h
	../tinyrenderer.cpp
	../tinyrenderer.h
)

ADD_LIBRARY(pytinyrenderer SHARED ${pytinyrenderer_SRCS} ${TDS_HDRS})
#TARGET_LINK_LIBRARIES(pytinyrenderer opengl_window)
target_include_directories(pytinyrenderer PRIVATE 
  .. 
)

target_link_libraries(pytinyrenderer PRIVATE
  Python3::Python
  Python3::NumPy
  pybind11::module)

IF (WIN32)
  TARGET_COMPILE_OPTIONS(pytinyrenderer PUBLIC -bigobj)
  MESSAGE("PYTHON_LIBRARIES")
  MESSAGE(${PYTHON_LIBRARIES})
  GET_FILENAME_COMPONENT(PYTHON_LIBRARY_DIR ${PYTHON_LIBRARIES} DIRECTORY)# CACHE)
  MESSAGE("PYTHON_LIBRARY_DIR ")
  MESSAGE(${PYTHON_LIBRARY_DIR})
  target_link_directories (pytinyrenderer PUBLIC "${PYTHON_LIBRARY_DIR}")
  #TARGET_LINK_LIBRARIES(pytinyrenderer ${PYTHON_LIBRARIES} Opengl32)
  
ELSEIF (APPLE)
  SET_TARGET_PROPERTIES(pytinyrenderer PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
ENDIF ()


