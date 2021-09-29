// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdio.h>

#include <string>

#include "model.h"
#include "our_gl.h"
#include "tinyrenderer.h"

std::string file_open_dialog(const std::string& path) {
  return std::string("opening: ") + path;
}

using namespace TinyRender;

namespace py = pybind11;

PYBIND11_MODULE(pytinyrenderer, m) {
  m.doc() = R"pbdoc(
        python bindings for tiny renderer 
        -----------------------

        .. currentmodule:: pytinyrenderer

        .. autosummary::
           :toctree: _generate

    )pbdoc";

  m.def("file_open_dialog", &file_open_dialog);

  py::class_<RenderBuffers>(m, "RenderBuffers")
      .def(py::init<int, int>())
      .def_readwrite("width", &RenderBuffers::m_width)
      .def_readwrite("height", &RenderBuffers::m_height)
      .def_readwrite("rgb", &RenderBuffers::rgb)
      .def_readwrite("depthbuffer", &RenderBuffers::zbuffer)
      .def_readwrite("segmentation_mask", &RenderBuffers::segmentation_mask);

  py::class_<TinySceneRenderer>(m, "TinySceneRenderer")
      .def(py::init<>())
      .def("create_mesh", &TinySceneRenderer::create_mesh)
      .def("create_cube", &TinySceneRenderer::create_cube)
      .def("create_capsule", &TinySceneRenderer::create_capsule)
      .def("create_object_instance", &TinySceneRenderer::create_object_instance)
      .def("set_object_position", &TinySceneRenderer::set_object_position)
      .def("set_object_orientation", &TinySceneRenderer::set_object_orientation)
      .def("set_object_local_scaling",
           &TinySceneRenderer::set_object_local_scaling)
      .def("set_object_color", &TinySceneRenderer::set_object_color)
      .def("set_object_double_sided",
           &TinySceneRenderer::set_object_double_sided)
      .def("set_object_segmentation_uid",
           &TinySceneRenderer::set_object_segmentation_uid)
      .def("get_object_segmentation_uid",
           &TinySceneRenderer::get_object_segmentation_uid)
      .def("get_camera_image", &TinySceneRenderer::get_camera_image_py)
      .def("delete_mesh",  &TinySceneRenderer::delete_mesh)
      .def("delete_instance",  &TinySceneRenderer::delete_instance)
      ;

  m.def("compute_projection_matrix",
        &TinySceneRenderer::compute_projection_matrix);
  m.def("compute_view_matrix", &TinySceneRenderer::compute_view_matrix);

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
