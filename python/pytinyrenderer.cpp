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
#include <map>
#include <string>
#include "TinyShapeData.h"
#include "TinyRenderer.h"
#include "model.h"
#include "our_gl.h"



std::string file_open_dialog(const std::string& path)
{
  return std::string("opening: ")+path;
}

using namespace TinyRender;


std::vector<float> compute_view_matrix(const std::vector<float>& cameraPosition, const std::vector<float>& cameraTargetPosition, const std::vector<float>& cameraUp)
{
    std::vector<float> viewMatrix;
    viewMatrix.resize(16);

	Vec3f eye = Vec3f(cameraPosition[0], cameraPosition[1], cameraPosition[2]);
	Vec3f center = Vec3f(cameraTargetPosition[0], cameraTargetPosition[1], cameraTargetPosition[2]);
	Vec3f up = Vec3f(cameraUp[0], cameraUp[1], cameraUp[2]);
	Vec3f f = (center - eye).normalize();
	Vec3f u = up.normalize();
	Vec3f s = cross(f,u).normalize();
	u = cross(s,f);

	viewMatrix[0 * 4 + 0] = s.x;
	viewMatrix[1 * 4 + 0] = s.y;
	viewMatrix[2 * 4 + 0] = s.z;

	viewMatrix[0 * 4 + 1] = u.x;
	viewMatrix[1 * 4 + 1] = u.y;
	viewMatrix[2 * 4 + 1] = u.z;

	viewMatrix[0 * 4 + 2] = -f.x;
	viewMatrix[1 * 4 + 2] = -f.y;
	viewMatrix[2 * 4 + 2] = -f.z;

	viewMatrix[0 * 4 + 3] = 0.f;
	viewMatrix[1 * 4 + 3] = 0.f;
	viewMatrix[2 * 4 + 3] = 0.f;

	viewMatrix[3 * 4 + 0] = -TinyRender::dot(s,eye);
	viewMatrix[3 * 4 + 1] = -TinyRender::dot(u,eye);
	viewMatrix[3 * 4 + 2] = TinyRender::dot(f,eye);
	viewMatrix[3 * 4 + 3] = 1.f;
    return viewMatrix;
}
#if 0
void b3ComputeViewMatrixFromYawPitchRoll(const float cameraTargetPosition[3], float distance, float yaw, float pitch, float roll, int upAxis, float viewMatrix[16])
{
	b3Vector3 camUpVector;
	b3Vector3 camForward;
	b3Vector3 camPos;
	b3Vector3 camTargetPos = b3MakeVector3(cameraTargetPosition[0], cameraTargetPosition[1], cameraTargetPosition[2]);
	b3Vector3 eyePos = b3MakeVector3(0, 0, 0);

	b3Scalar yawRad = yaw * b3Scalar(0.01745329251994329547);      // rads per deg
	b3Scalar pitchRad = pitch * b3Scalar(0.01745329251994329547);  // rads per deg
	b3Scalar rollRad = roll * b3Scalar(0.01745329251994329547);      // rads per deg
	b3Quaternion eyeRot;

	int forwardAxis(-1);
	switch (upAxis)
	{
		case 1:
			forwardAxis = 2;
			camUpVector = b3MakeVector3(0, 1, 0);
			eyeRot.setEulerZYX(rollRad, yawRad, -pitchRad);
			break;
		case 2:
			forwardAxis = 1;
			camUpVector = b3MakeVector3(0, 0, 1);
			eyeRot.setEulerZYX(yawRad, rollRad, pitchRad);
			break;
		default:
			return;
	};

	eyePos[forwardAxis] = -distance;

	camForward = b3MakeVector3(eyePos[0], eyePos[1], eyePos[2]);
	if (camForward.length2() < B3_EPSILON)
	{
		camForward.setValue(1.f, 0.f, 0.f);
	}
	else
	{
		camForward.normalize();
	}

	eyePos = b3Matrix3x3(eyeRot) * eyePos;
	camUpVector = b3Matrix3x3(eyeRot) * camUpVector;

	camPos = eyePos;
	camPos += camTargetPos;

	float camPosf[4] = {camPos[0], camPos[1], camPos[2], 0};
	float camPosTargetf[4] = {camTargetPos[0], camTargetPos[1], camTargetPos[2], 0};
	float camUpf[4] = {camUpVector[0], camUpVector[1], camUpVector[2], 0};

	b3ComputeViewMatrixFromPositions(camPosf, camPosTargetf, camUpf, viewMatrix);
}
#endif

std::vector<float> compute_projection_matrix(float left, float right, float bottom, float top, float nearVal, float farVal)
{
    std::vector<float> projectionMatrix;
    projectionMatrix.resize(16);

	projectionMatrix[0 * 4 + 0] = (float(2) * nearVal) / (right - left);
	projectionMatrix[0 * 4 + 1] = float(0);
	projectionMatrix[0 * 4 + 2] = float(0);
	projectionMatrix[0 * 4 + 3] = float(0);

	projectionMatrix[1 * 4 + 0] = float(0);
	projectionMatrix[1 * 4 + 1] = (float(2) * nearVal) / (top - bottom);
	projectionMatrix[1 * 4 + 2] = float(0);
	projectionMatrix[1 * 4 + 3] = float(0);

	projectionMatrix[2 * 4 + 0] = (right + left) / (right - left);
	projectionMatrix[2 * 4 + 1] = (top + bottom) / (top - bottom);
	projectionMatrix[2 * 4 + 2] = -(farVal + nearVal) / (farVal - nearVal);
	projectionMatrix[2 * 4 + 3] = float(-1);

	projectionMatrix[3 * 4 + 0] = float(0);
	projectionMatrix[3 * 4 + 1] = float(0);
	projectionMatrix[3 * 4 + 2] = -(float(2) * farVal * nearVal) / (farVal - nearVal);
	projectionMatrix[3 * 4 + 3] = float(0);

    return projectionMatrix;
}



class TinySceneRenderer
{
    int m_guid;

    std::map<int, TinyRender::Model*> m_models;
    std::map<int, TinyRenderObjectInstance*> m_object_instances;

    
public:
    TinySceneRenderer() : m_guid(1) {

    }
    virtual ~TinySceneRenderer() {

    }

    int create_mesh(const std::vector<double>& vertices, const std::vector<double>& normals, 
                    const std::vector<double>& uvs, const std::vector<int>& indices,
                    const std::vector<unsigned char>& texture, int texture_width, 
                    int texture_height, float texture_scaling)
    {
        int uid = m_guid++;
        TinyRender::Model* model = new TinyRender::Model();
        
        if (texture.size() && texture.size() == texture_width*texture_height*3)
        {
            model->setDiffuseTextureFromData(&texture[0], texture_width, texture_height);
        }

        int numVertices = vertices.size()/3;
        int numTriangles = indices.size()/3;
        
        for (int i = 0; i < numVertices; i++) {
            model->addVertex(   vertices[i*3+0],
                                vertices[i*3+1],
                                vertices[i*3+2],
                                normals[i*3],
                                normals[i*3+1],
                                normals[i*3+2],
                                uvs[i*2+0]*texture_scaling,
                                uvs[i*2+1]*texture_scaling);
        }

        for (int i=0;i<numTriangles;i++)
        {
            model->addTriangle( indices[i*3+0],indices[i*3+0],indices[i*3+0],
                                indices[i*3+1],indices[i*3+1],indices[i*3+1],
                                indices[i*3+2],indices[i*3+2],indices[i*3+2]);
        }
        m_models[uid] = model;
        return uid;
    }

        
    int create_cube(const std::vector<double>& half_extents, const std::vector<unsigned char>& texture, int texture_width, int texture_height, float texture_scaling)
    {
        int uid = m_guid++;
        TinyRender::Model* model = new TinyRender::Model();
        
        if (texture.size() && texture.size() == texture_width*texture_height*3)
        {
            model->setDiffuseTextureFromData(&texture[0], texture_width, texture_height);
        }

        int strideInBytes = 9 * sizeof(float);
        int numVertices = sizeof(cube_vertices_textured) / strideInBytes;
        int numIndices = sizeof(cube_indices) / sizeof(int);

        float halfExtentsX = half_extents[0];
        float halfExtentsY = half_extents[1];
        float halfExtentsZ = half_extents[2];
        
        for (int i = 0; i < numVertices; i++) {
            model->addVertex(halfExtentsX * cube_vertices_textured[i * 9],
                halfExtentsY * cube_vertices_textured[i * 9 + 1],
                halfExtentsZ * cube_vertices_textured[i * 9 + 2],

                cube_vertices_textured[i * 9 + 4],
                cube_vertices_textured[i * 9 + 5],
                cube_vertices_textured[i * 9 + 6],
                cube_vertices_textured[i * 9 + 7] * texture_scaling,
                cube_vertices_textured[i * 9 + 8] * texture_scaling);
        }

        for (int i=0;i<numIndices;i+=3)
        {
            model->addTriangle(cube_indices[i],cube_indices[i],cube_indices[i],
                cube_indices[i+1],cube_indices[i+1],cube_indices[i+1],
                cube_indices[i+2],cube_indices[i+2],cube_indices[i+2]);
        }
        m_models[uid] = model;
        return uid;
    }

    int create_capsule(float radius, float half_height, int up_axis,const std::vector<unsigned char>& texture, int texture_width, int texture_height) {

        int uid = m_guid++;
        TinyRender::Model* model = new TinyRender::Model();

        if (texture.size() && texture.size() == texture_width*texture_height*3)
        {
            model->setDiffuseTextureFromData(&texture[0], texture_width, texture_height);
        }
        int red = 0;
        int green = 255;
        int blue = 0;  // 0;// 128;
        int strideInBytes = 9 * sizeof(float);
        int graphicsShapeIndex = -1;
    
        int numVertices =
            sizeof(textured_detailed_sphere_vertices) / strideInBytes;
        int numIndices = sizeof(textured_detailed_sphere_indices) / sizeof(int);

        //scale and transform
        std::vector<float> transformedVertices;
        {
            int numVertices = sizeof(textured_detailed_sphere_vertices) / strideInBytes;
            transformedVertices.resize(numVertices * 9);
            for(int i = 0; i < numVertices; i++)
            {
                float trVert[3] = {textured_detailed_sphere_vertices[i * 9 + 0]*2 *radius,
                    textured_detailed_sphere_vertices[i * 9 + 1]*2 *radius,
                    textured_detailed_sphere_vertices[i * 9 + 2]*2 *radius};

                if(trVert[up_axis] > 0)
                    trVert[up_axis] += half_height;
                else
                    trVert[up_axis] -= half_height;

                transformedVertices[i * 9 + 0] = trVert[0];
                transformedVertices[i * 9 + 1] = trVert[1];
                transformedVertices[i * 9 + 2] = trVert[2];
                transformedVertices[i * 9 + 3] = textured_detailed_sphere_vertices[i * 9 + 3];
                transformedVertices[i * 9 + 4] = textured_detailed_sphere_vertices[i * 9 + 4];
                transformedVertices[i * 9 + 5] = textured_detailed_sphere_vertices[i * 9 + 5];
                transformedVertices[i * 9 + 6] = textured_detailed_sphere_vertices[i * 9 + 6];
                transformedVertices[i * 9 + 7] = textured_detailed_sphere_vertices[i * 9 + 7];
                transformedVertices[i * 9 + 8] = textured_detailed_sphere_vertices[i * 9 + 8];
            }
        }

        for (int i = 0; i < numVertices; i++) {
                model->addVertex(
                    transformedVertices[i * 9],
                    transformedVertices[i * 9 + 1],
                    transformedVertices[i * 9 + 2],
                    transformedVertices[i * 9 + 4],
                    transformedVertices[i * 9 + 5],
                    transformedVertices[i * 9 + 6],
                    transformedVertices[i * 9 + 7],
                    transformedVertices[i * 9 + 8]);
        }

        for (int i=0;i<numIndices;i+=3)
        {
            model->addTriangle(textured_detailed_sphere_indices[i],textured_detailed_sphere_indices[i],textured_detailed_sphere_indices[i],
                textured_detailed_sphere_indices[i+1],textured_detailed_sphere_indices[i+1],textured_detailed_sphere_indices[i+1],
                textured_detailed_sphere_indices[i+2],textured_detailed_sphere_indices[i+2],textured_detailed_sphere_indices[i+2]);
        }
        m_models[uid] = model;
        return uid;
    }

    void set_object_position(int instance_uid, const std::vector<float>& position)
    {
        auto object_instance = m_object_instances[instance_uid];
        if (object_instance && position.size()==3)
        {
            object_instance->m_modelMatrix[0][3] = position[0];
            object_instance->m_modelMatrix[1][3] = position[1];
            object_instance->m_modelMatrix[2][3] = position[2];
        }
    }
    void set_object_orientation(int instance_uid, const std::vector<float>& orientation)
    {
        auto object_instance = m_object_instances[instance_uid];
        if (object_instance && orientation.size()==4)
        {
            float x = orientation[0];
            float y = orientation[1];
            float z = orientation[2];
            float w = orientation[3];

            float d = x*x+y*y+z*z+w*w;
		    assert(d!=0.);
		    float s = float (2.0) / d;

            float xs = x * s, ys = y * s, zs = z * s;
		    float wx = w * xs, wy = w * ys, wz = w * zs;
		    float xx = x * xs, xy = x * ys, xz = x * zs;
		    float yy = y * ys, yz = y * zs, zz = z * zs;
		    
            //todo: matrix may require transpose
            object_instance->m_modelMatrix[0][0] = float (1.0) - (yy + zz);
            object_instance->m_modelMatrix[0][1] = xy - wz;
            object_instance->m_modelMatrix[0][2] = xz + wy;

            object_instance->m_modelMatrix[1][0] = xy + wz;
            object_instance->m_modelMatrix[1][1] = float (1.0) - (xx + zz);
            object_instance->m_modelMatrix[1][2] = yz - wx;
			    
            object_instance->m_modelMatrix[2][0] = xz - wy;
            object_instance->m_modelMatrix[2][1] = yz + wx;
            object_instance->m_modelMatrix[2][2] = float (1.0) - (xx + yy);
        }
    }

    void set_object_color(int instance_uid, const std::vector<float>& color)
    {
        auto object_instance = m_object_instances[instance_uid];
        if (object_instance)
        {
            if (color.size()==4)
            {
                object_instance->m_model->setColorRGBA(&color[0]);
            }
        }
    }

    void set_object_segmentation_uid(int instance_uid, int object_segmentation_uid)
    {
        auto object_instance = m_object_instances[instance_uid];
        if (object_instance)
        {
            object_instance->m_object_segmentation_uid = object_segmentation_uid;
        }
    }

    int get_object_segmentation_uid(int instance_uid) const
    {
        if ( m_object_instances.find(instance_uid) == m_object_instances.end() ) {
            return -1;
        }
        const auto object_instance = m_object_instances.at(instance_uid);
        return object_instance->m_object_segmentation_uid;
    }


    void set_object_double_sided(int instance_uid, bool double_sided)
    {
        auto object_instance = m_object_instances[instance_uid];
        if (object_instance)
        {
            object_instance->m_doubleSided = double_sided;
        }
    }

     void set_object_local_scaling(int instance_uid, const std::vector<float>& local_scaling)
    {
        auto object_instance = m_object_instances[instance_uid];
        if (object_instance && local_scaling.size()==3)
        {
            object_instance->m_localScaling[0] = local_scaling[0];
            object_instance->m_localScaling[1] = local_scaling[1];
            object_instance->m_localScaling[2] = local_scaling[2];
        }
    }

    
    int create_object_instance(int model_uid)
    {
        TinyRender::Model* model = this->m_models[model_uid];
        if (model==0)
            return -1;
        
        TinyRenderObjectInstance* tinyObj = new TinyRenderObjectInstance();
        tinyObj->m_model = model;
        tinyObj->m_doubleSided = true;
        
        int uid = m_guid++;
        m_object_instances[uid] = tinyObj;
        return uid;
    }


    
    RenderBuffers get_camera_image(int width, int height, const std::vector<int>& objects, const std::vector<float>& viewMatrix, const std::vector<float>& projectionMatrix)
    {
        RenderBuffers buffers(width, height);

        //clear the color buffer
	    TGAColor clearColor;
	    clearColor.bgra[0] = 255;
	    clearColor.bgra[1] = 255;
	    clearColor.bgra[2] = 255;
	    clearColor.bgra[3] = 255;

        float nearPlane = projectionMatrix[14] / (projectionMatrix[10] - 1);
	    float farPlane = projectionMatrix[14] / (projectionMatrix[10] + 1);


        for (int x=0;x<width;x++)
        {
            for (int y=0;y<height;y++)
            {
                buffers.rgb[3*(x+y*width)+0]=clearColor[0];
                buffers.rgb[3*(x+y*width)+1]=clearColor[1];
                buffers.rgb[3*(x+y*width)+2]=clearColor[2];
                buffers.zbuffer[x + y * width] = -farPlane;
            }
        }

        for (int i=0;i<objects.size();i++)
        {
            int uid = objects[i];
            auto object_instance = m_object_instances[uid];
            if (object_instance)
            {
                
                for (int i=0;i<4;i++)
                {
                    TinyRender::Vec4f p;
                    TinyRender::Vec4f v;
                    for (int j=0;j<4;j++)
                    {
                        p[j] = projectionMatrix[i*4+j];
                        v[j] = viewMatrix[i*4+j];
                    }
                    object_instance->m_projectionMatrix.set_col(i,p);
                    object_instance->m_viewMatrix.set_col(i,v);
                }
                
                object_instance->m_lightDirWorld.x = 0.5773502;
                object_instance->m_lightDirWorld.y= 0.5773502;
                object_instance->m_lightDirWorld.z = 0.5773502;

                object_instance->m_lightDistance = 2;
                object_instance->m_viewportMatrix = TinyRender::viewport(0, 0, width, height);
                
        
                TinyRenderer::renderObject(width, height, *object_instance, buffers);
            }
        }

        return buffers;
    }

};

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

    m.def("compute_projection_matrix", &compute_projection_matrix);
    m.def("compute_view_matrix", &compute_view_matrix);
    

    py::class_<RenderBuffers>(m, "RenderBuffers")
    .def(py::init<int,int>())
    .def_readwrite("width", &RenderBuffers::m_width)
    .def_readwrite("height", &RenderBuffers::m_height)
    .def_readwrite("rgb", &RenderBuffers::rgb)
    .def_readwrite("depthbuffer", &RenderBuffers::zbuffer)
    .def_readwrite("segmentation_mask", &RenderBuffers::segmentation_mask)
    ;

    py::class_<TinySceneRenderer>(m, "TinySceneRenderer")
    .def(py::init<>())
    .def("create_mesh", &TinySceneRenderer::create_mesh)
    .def("create_cube", &TinySceneRenderer::create_cube)
    .def("create_capsule", &TinySceneRenderer::create_capsule)
    .def("create_object_instance", &TinySceneRenderer::create_object_instance)
    .def("set_object_position", &TinySceneRenderer::set_object_position)
    .def("set_object_orientation", &TinySceneRenderer::set_object_orientation)
    .def("set_object_local_scaling", &TinySceneRenderer::set_object_local_scaling)
    .def("set_object_color", &TinySceneRenderer::set_object_color)
    .def("set_object_double_sided", &TinySceneRenderer::set_object_double_sided)
    .def("set_object_segmentation_uid", &TinySceneRenderer::set_object_segmentation_uid)
    .def("get_object_segmentation_uid", &TinySceneRenderer::get_object_segmentation_uid)
    .def("get_camera_image", &TinySceneRenderer::get_camera_image)
      ;

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
 
}
