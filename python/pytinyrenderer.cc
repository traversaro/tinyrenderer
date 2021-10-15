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

  py::class_<TinyRenderCamera>(m, "TinyRenderCamera")
      .def(py::init<int, int, float, float, float, float,
                    const std::vector<float>&, const std::vector<float>&,
                    const std::vector<float>&>(),
           py::arg("viewWidth") = 640, py::arg("viewHeight") = 480,
           py::arg("near") = 0.01, py::arg("far") = 1000.0,
           py::arg("hfov") = 58.0, py::arg("vfov") = 45.0,
           py::arg("position") = std::vector<float>{1, 1, 1},
           py::arg("target") = std::vector<float>{0, 0, 0},
           py::arg("up") = std::vector<float>{0, 0, 1})
      .def(py::init<int, int, const std::vector<float>&,
                    const std::vector<float>&>(),
           py::arg("viewWidth"), py::arg("viewHeight"),
           py::arg("viewMatrix"),
           py::arg("projectionMatrix"))
      .def_readonly("view_width", &TinyRenderCamera::m_viewWidth)
      .def_readonly("view_height", &TinyRenderCamera::m_viewHeight);

  py::class_<TinyRenderLight>(m, "TinyRenderLight")
      .def(py::init<const std::vector<float>&, const std::vector<float>&,
          const std::vector<float>&, float, float, float, float, bool, float>(),
           py::arg("direction") = std::vector<float>{0.57735, 0.57735, 0.57735},
           py::arg("color") = std::vector<float>{1, 1, 1},
           py::arg("shadowmap_center") = std::vector<float>{0, 0, 0},
           py::arg("distance") = 10.0, py::arg("ambient") = 0.6,
           py::arg("diffuse") = 0.35, py::arg("specular") = 0.05,
           py::arg("has_shadow") = true, py::arg("shadow_coefficient")= 0.4);

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
  m.def("compute_projection_matrix2",
        &TinySceneRenderer::compute_projection_matrix2);
  m.def("compute_view_matrix", &TinySceneRenderer::compute_view_matrix);
  m.def("compute_view_matrix_from_yaw_pitch_roll", &TinySceneRenderer::compute_view_matrix_from_yaw_pitch_roll);



#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}
