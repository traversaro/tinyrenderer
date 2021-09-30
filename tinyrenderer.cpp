#include "tinyrenderer.h"

#include <cmath>
#include <iostream>
#include <limits>

#include "geometry.h"
#include "model.h"
#include "our_gl.h"
#include "tgaimage.h"
#include "tinyshapedata.h"
#include <array>

using namespace TinyRender;

struct DepthShader : public IShader {
  Model* m_model;
  Matrix& m_modelMat;
  Matrix m_invModelMat;

  Matrix& m_projectionMat;
  Vec3f m_localScaling;
  Matrix& m_lightModelView;
  float m_lightDistance;

  mat<2, 3, float> varying_uv;   // triangle uv coordinates, written by the
                                 // vertex shader, read by the fragment shader
  mat<4, 3, float> varying_tri;  // triangle coordinates (clip coordinates),
                                 // written by VS, read by FS

  mat<3, 3, float> varying_nrm;  // normal per vertex to be interpolated by FS

  DepthShader(Model* model, Matrix& lightModelView, Matrix& projectionMat,
              Matrix& modelMat, Vec3f localScaling, float lightDistance)
      : m_model(model),
        m_modelMat(modelMat),
        m_projectionMat(projectionMat),
        m_localScaling(localScaling),
        m_lightModelView(lightModelView),
        m_lightDistance(lightDistance) {
    m_nearPlane = m_projectionMat.col(3)[2] / (m_projectionMat.col(2)[2] - 1);
    m_farPlane = m_projectionMat.col(3)[2] / (m_projectionMat.col(2)[2] + 1);

    m_invModelMat = m_modelMat.invert_transpose();
  }
  virtual Vec4f vertex(int iface, int nthvert) {
    Vec2f uv = m_model->uv(iface, nthvert);
    varying_uv.set_col(nthvert, uv);
    varying_nrm.set_col(
        nthvert, proj<3>(m_invModelMat *
                         embed<4>(m_model->normal(iface, nthvert), 0.f)));
    Vec3f unScaledVert = m_model->vert(iface, nthvert);
    Vec3f scaledVert = Vec3f(unScaledVert[0] * m_localScaling[0],
                             unScaledVert[1] * m_localScaling[1],
                             unScaledVert[2] * m_localScaling[2]);
    Vec4f gl_Vertex = m_projectionMat * m_lightModelView * embed<4>(scaledVert);
    varying_tri.set_col(nthvert, gl_Vertex);
    return gl_Vertex;
  }

  virtual bool fragment(Vec3f bar, TGAColor& color) {
    Vec4f p = varying_tri * bar;
    color = TGAColor(255, 255, 255) * (p[2] / m_lightDistance);
    return false;
  }
};

struct Shader : public IShader {
  Model* m_model;
  Vec3f m_light_dir_local;
  Vec3f m_light_color;
  Matrix& m_modelMat;
  Matrix m_invModelMat;
  Matrix& m_modelView1;
  Matrix& m_projectionMat;
  Vec3f m_localScaling;
  Matrix& m_lightModelView;
  Vec4f m_colorRGBA;
  Matrix& m_viewportMat;
  Matrix m_projectionModelViewMat;
  Matrix m_projectionLightViewMat;
  float m_ambient_coefficient;
  float m_diffuse_coefficient;
  float m_specular_coefficient;

  std::vector<float>* m_shadowBuffer;

  int m_width;
  int m_height;

  int m_index;

  mat<2, 3, float> varying_uv;   // triangle uv coordinates, written by the
                                 // vertex shader, read by the fragment shader
  mat<4, 3, float> varying_tri;  // triangle coordinates (clip coordinates),
                                 // written by VS, read by FS
  mat<4, 3, float> varying_tri_light_view;
  mat<3, 3, float> varying_nrm;  // normal per vertex to be interpolated by FS
  mat<4, 3, float> world_tri;  // model triangle coordinates in the world space
                               // used for backface culling, written by VS

  Shader(Model* model, Vec3f light_dir_local, Vec3f light_color,
         Matrix& modelView, Matrix& lightModelView, Matrix& projectionMat,
         Matrix& modelMat, Matrix& viewportMat, Vec3f localScaling,
         const Vec4f& colorRGBA, int width, int height,
         std::vector<float>* shadowBuffer, float ambient_coefficient = 0.6,
         float diffuse_coefficient = 0.35, float specular_coefficient = 0.05)
      : m_model(model),
        m_light_dir_local(light_dir_local),
        m_light_color(light_color),
        m_modelMat(modelMat),
        m_modelView1(modelView),
        m_projectionMat(projectionMat),
        m_localScaling(localScaling),
        m_lightModelView(lightModelView),
        m_colorRGBA(colorRGBA),
        m_viewportMat(viewportMat),
        m_ambient_coefficient(ambient_coefficient),
        m_diffuse_coefficient(diffuse_coefficient),
        m_specular_coefficient(specular_coefficient),

        m_shadowBuffer(shadowBuffer),
        m_width(width),
        m_height(height)

  {
    m_nearPlane = m_projectionMat.col(3)[2] / (m_projectionMat.col(2)[2] - 1);
    m_farPlane = m_projectionMat.col(3)[2] / (m_projectionMat.col(2)[2] + 1);
    // printf("near=%f, far=%f\n", m_nearPlane, m_farPlane);
    m_invModelMat = m_modelMat.invert_transpose();
    m_projectionModelViewMat = m_projectionMat * m_modelView1;
    m_projectionLightViewMat = m_projectionMat * m_lightModelView;
  }
  virtual Vec4f vertex(int iface, int nthvert) {
    Vec2f uv = m_model->uv(iface, nthvert);
    varying_uv.set_col(nthvert, uv);
    varying_nrm.set_col(
        nthvert, proj<3>(m_invModelMat *
                         embed<4>(m_model->normal(iface, nthvert), 0.f)));
    Vec3f unScaledVert = m_model->vert(iface, nthvert);
    Vec3f scaledVert = Vec3f(unScaledVert[0] * m_localScaling[0],
                             unScaledVert[1] * m_localScaling[1],
                             unScaledVert[2] * m_localScaling[2]);
    Vec4f gl_Vertex = m_projectionModelViewMat * embed<4>(scaledVert);
    varying_tri.set_col(nthvert, gl_Vertex);
    Vec4f world_Vertex = m_modelMat * embed<4>(scaledVert);
    world_tri.set_col(nthvert, world_Vertex);
    Vec4f gl_VertexLightView = m_projectionLightViewMat * embed<4>(scaledVert);
    varying_tri_light_view.set_col(nthvert, gl_VertexLightView);
    return gl_Vertex;
  }

  virtual bool fragment(Vec3f bar, TGAColor& color) {
    Vec4f p = m_viewportMat * (varying_tri_light_view * bar);
    float depth = p[2];
    p = p / p[3];

    float index_x = std::max(float(0.0), std::min(float(m_width - 1), p[0]));
    float index_y = std::max(float(0.0), std::min(float(m_height - 1), p[1]));
    int idx = int(index_x) +
              int(index_y) * m_width;  // index in the shadowbuffer array
    float shadow = 1.0;
    if (m_shadowBuffer && idx >= 0 && idx < m_shadowBuffer->size()) {
      shadow = 0.8 + 0.2 * (m_shadowBuffer->at(idx) <
                            -depth + 0.05);  // magic coeff to avoid z-fighting
    }
    Vec3f bn = (varying_nrm * bar).normalize();
    Vec2f uv = varying_uv * bar;

    Vec3f reflection_direction =
        (bn * (bn * m_light_dir_local * 2.f) - m_light_dir_local).normalize();
    float specular =
        std::pow(std::max(reflection_direction.z, 0.f), m_model->specular(uv));
    float diffuse = std::max(0.f, bn * m_light_dir_local);

    color = m_model->diffuse(uv);
    color[0] *= m_colorRGBA[0];
    color[1] *= m_colorRGBA[1];
    color[2] *= m_colorRGBA[2];
    color[3] *= m_colorRGBA[3];

    for (int i = 0; i < 3; ++i) {
      int orgColor = 0;
      float floatColor = (m_ambient_coefficient * color[i] +
                          shadow *
                              (m_diffuse_coefficient * diffuse +
                               m_specular_coefficient * specular) *
                              color[i] * m_light_color[i]);
      if (floatColor == floatColor) {
        orgColor = int(floatColor);
      }
      color[i] = std::min(orgColor, 255);
    }

    return false;
  }
};

TinyRenderObjectInstance::TinyRenderObjectInstance()
    : m_mesh_uid(-1), m_object_segmentation_uid(-1), m_doubleSided(false) {
  Vec3f eye(1, 1, 3);
  Vec3f center(0, 0, 0);
  Vec3f up(0, 0, 1);
  m_lightDirWorld = TinyRender::Vec3f(0, 0, 0);
  m_lightColor = TinyRender::Vec3f(1, 1, 1);
  m_localScaling = TinyRender::Vec3f(1, 1, 1);
  m_modelMatrix = Matrix::identity();
  m_lightAmbientCoeff = 0.6;
  m_lightDiffuseCoeff = 0.35;
  m_lightSpecularCoeff = 0.05;
}

TinyRenderObjectInstance::~TinyRenderObjectInstance() {}


static bool equals(const Vec4f& vA, const Vec4f& vB) { return false; }

static void clipEdge(const mat<4, 3, float>& triangleIn, int vertexIndexA,
                     int vertexIndexB, std::vector<Vec4f>& vertices) {
  Vec4f v0New = triangleIn.col(vertexIndexA);
  Vec4f v1New = triangleIn.col(vertexIndexB);

  bool v0Inside = v0New[3] > 0.f && v0New[2] > -v0New[3];
  bool v1Inside = v1New[3] > 0.f && v1New[2] > -v1New[3];

  if (v0Inside && v1Inside) {
  } else if (v0Inside || v1Inside) {
    float d0 = v0New[2] + v0New[3];
    float d1 = v1New[2] + v1New[3];
    float factor = 1.0 / (d1 - d0);
    Vec4f newVertex = (v0New * d1 - v1New * d0) * factor;
    if (v0Inside) {
      v1New = newVertex;
    } else {
      v0New = newVertex;
    }
  } else {
    return;
  }

  if (vertices.empty() || !(equals(vertices[vertices.size() - 1], v0New))) {
    vertices.push_back(v0New);
  }

  vertices.push_back(v1New);
}

static bool clipTriangleAgainstNearplane(
    const mat<4, 3, float>& triangleIn,
    std::vector<mat<4, 3, float> >& clippedTrianglesOut) {
  // discard triangle if all vertices are behind near-plane
  if (triangleIn[3][0] < 0 && triangleIn[3][1] < 0 && triangleIn[3][2] < 0) {
    return true;
  }

  // accept triangle if all vertices are in front of the near-plane
  if (triangleIn[3][0] >= 0 && triangleIn[3][1] >= 0 && triangleIn[3][2] >= 0) {
    clippedTrianglesOut.push_back(triangleIn);
    return false;
  }

  std::vector<Vec4f> vertices;
  vertices.reserve(5);

  clipEdge(triangleIn, 0, 1, vertices);
  clipEdge(triangleIn, 1, 2, vertices);
  clipEdge(triangleIn, 2, 0, vertices);

  if (vertices.size() < 3) return true;

  if (equals(vertices[0], vertices[vertices.size() - 1])) {
    vertices.pop_back();
  }

  // create a fan of triangles
  for (int i = 1; i < vertices.size() - 1; i++) {
    mat<4, 3, float> vtx;
    vtx.set_col(0, vertices[0]);
    vtx.set_col(1, vertices[i]);
    vtx.set_col(2, vertices[i + 1]);
    clippedTrianglesOut.push_back(vtx);
  }
  return true;
}

void TinySceneRenderer::renderObject(int width, int height,
                                     TinyRenderObjectInstance& object_instance,
                                     RenderBuffers& render_buffers) {
  Vec3f light_dir_local = Vec3f(object_instance.m_lightDirWorld[0],
                                object_instance.m_lightDirWorld[1],
                                object_instance.m_lightDirWorld[2]);
  Vec3f light_color =
      Vec3f(object_instance.m_lightColor[0], object_instance.m_lightColor[1],
            object_instance.m_lightColor[2]);
  float light_distance = object_instance.m_lightDistance;
  Model* model = m_models[object_instance.m_mesh_uid];
  if (0 == model) return;

  // discard invisible objects (zero alpha)
  if (model->getColorRGBA()[3] == 0) return;

  object_instance.m_viewportMatrix = viewport(0, 0, width, height);
  std::vector<float>* shadowBufferPtr = 0;  // object_instance.m_shadowBuffer;

  {
    // light target is set to be the origin, and the up direction is set to be
    // vertical up.
    Matrix lightViewMatrix = lookat(light_dir_local * light_distance,
                                    Vec3f(0.0, 0.0, 0.0), Vec3f(0.0, 0.0, 1.0));
    Matrix lightModelViewMatrix =
        lightViewMatrix * object_instance.m_modelMatrix;
    Matrix modelViewMatrix =
        object_instance.m_viewMatrix * object_instance.m_modelMatrix;
    Vec3f localScaling(object_instance.m_localScaling[0],
                       object_instance.m_localScaling[1],
                       object_instance.m_localScaling[2]);
    Matrix viewMatrixInv = object_instance.m_viewMatrix.invert();
    TinyRender::Vec3f P(viewMatrixInv[0][3], viewMatrixInv[1][3],
                        viewMatrixInv[2][3]);

    Shader shader(model, light_dir_local, light_color, modelViewMatrix,
                  lightModelViewMatrix, object_instance.m_projectionMatrix,
                  object_instance.m_modelMatrix,
                  object_instance.m_viewportMatrix, localScaling,
                  model->getColorRGBA(), width, height, shadowBufferPtr,
                  object_instance.m_lightAmbientCoeff,
                  object_instance.m_lightDiffuseCoeff,
                  object_instance.m_lightSpecularCoeff);

    {
      for (int i = 0; i < model->nfaces(); i++) {
        for (int j = 0; j < 3; j++) {
          shader.vertex(i, j);
        }

        if (!object_instance.m_doubleSided) {
          // backface culling
          TinyRender::Vec3f v0(shader.world_tri.col(0)[0],
                               shader.world_tri.col(0)[1],
                               shader.world_tri.col(0)[2]);
          TinyRender::Vec3f v1(shader.world_tri.col(1)[0],
                               shader.world_tri.col(1)[1],
                               shader.world_tri.col(1)[2]);
          TinyRender::Vec3f v2(shader.world_tri.col(2)[0],
                               shader.world_tri.col(2)[1],
                               shader.world_tri.col(2)[2]);
          TinyRender::Vec3f N = TinyRender::cross((v1 - v0), (v2 - v0));
          if (TinyRender::dot((v0 - P), (N)) >= 0) continue;
        }

        std::vector<mat<4, 3, float> > clippedTriangles;
        clippedTriangles.reserve(3);

        bool hasClipped =
            clipTriangleAgainstNearplane(shader.varying_tri, clippedTriangles);

        if (hasClipped) {
          for (int t = 0; t < clippedTriangles.size(); t++) {
            triangleClipped(clippedTriangles[t], shader.varying_tri, shader,
                            render_buffers, object_instance.m_viewportMatrix,
                            object_instance.m_object_segmentation_uid);
          }
        } else {
          triangle(shader.varying_tri, shader, render_buffers,
                   object_instance.m_viewportMatrix,
                   object_instance.m_object_segmentation_uid);
        }
      }
    }
  }
  }

std::vector<float> TinySceneRenderer::compute_view_matrix(
    const std::vector<float>& cameraPosition,
    const std::vector<float>& cameraTargetPosition,
    const std::vector<float>& cameraUp) {
  std::vector<float> viewMatrix;
  viewMatrix.resize(16);

  Vec3f eye = Vec3f(cameraPosition[0], cameraPosition[1], cameraPosition[2]);
  Vec3f center = Vec3f(cameraTargetPosition[0], cameraTargetPosition[1],
                       cameraTargetPosition[2]);
  Vec3f up = Vec3f(cameraUp[0], cameraUp[1], cameraUp[2]);
  Vec3f f = (center - eye).normalize();
  Vec3f u = up.normalize();
  Vec3f s = cross(f, u).normalize();
  u = cross(s, f);

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

  viewMatrix[3 * 4 + 0] = -TinyRender::dot(s, eye);
  viewMatrix[3 * 4 + 1] = -TinyRender::dot(u, eye);
  viewMatrix[3 * 4 + 2] = TinyRender::dot(f, eye);
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

std::vector<float> TinySceneRenderer::compute_projection_matrix(
    float left, float right, float bottom, float top, float nearVal,
    float farVal) {
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
  projectionMatrix[3 * 4 + 2] =
      -(float(2) * farVal * nearVal) / (farVal - nearVal);
  projectionMatrix[3 * 4 + 3] = float(0);

  return projectionMatrix;
}

TinySceneRenderer::TinySceneRenderer() : m_guid(1) {}

TinySceneRenderer::~TinySceneRenderer() 
{
    //free all memory
    {
        auto it = m_object_instances.begin();
        while (it != m_object_instances.end())
        {
            auto value = it->second;
            delete value;
            it++;
        }
        m_object_instances.clear();
    }
    {
        auto it = m_models.begin();
        while (it != m_models.end())
        {
            auto value = it->second;
            delete value;
            it++;
        }
        m_models.clear();
    }
}

int TinySceneRenderer::create_mesh(const std::vector<double>& vertices,
                                   const std::vector<double>& normals,
                                   const std::vector<double>& uvs,
                                   const std::vector<int>& indices,
                                   const std::vector<unsigned char>& texture,
                                   int texture_width, int texture_height,
                                   float texture_scaling) {
  int uid = m_guid++;
  TinyRender::Model* model = new TinyRender::Model();

  if (!texture.empty() && texture.size() == texture_width * texture_height * 3) {
    model->setDiffuseTextureFromData(&texture[0], texture_width,
                                     texture_height);
  }

  int numVertices = vertices.size() / 3;
  int numTriangles = indices.size() / 3;

  for (int i = 0; i < numVertices; i++) {
    model->addVertex(vertices[i * 3 + 0], vertices[i * 3 + 1],
                     vertices[i * 3 + 2], normals[i * 3], normals[i * 3 + 1],
                     normals[i * 3 + 2], uvs[i * 2 + 0] * texture_scaling,
                     uvs[i * 2 + 1] * texture_scaling);
  }

  for (int i = 0; i < numTriangles; i++) {
    model->addTriangle(
        indices[i * 3 + 0], indices[i * 3 + 0], indices[i * 3 + 0],
        indices[i * 3 + 1], indices[i * 3 + 1], indices[i * 3 + 1],
        indices[i * 3 + 2], indices[i * 3 + 2], indices[i * 3 + 2]);
  }
  m_models[uid] = model;
  return uid;
}

int TinySceneRenderer::create_cube(const std::vector<double>& half_extents,
                                   const std::vector<unsigned char>& texture,
                                   int texture_width, int texture_height,
                                   float texture_scaling) {
  int uid = m_guid++;
  TinyRender::Model* model = new TinyRender::Model();

  if (!texture.empty() && texture.size() == texture_width * texture_height * 3) {
    model->setDiffuseTextureFromData(&texture[0], texture_width,
                                     texture_height);
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

  for (int i = 0; i < numIndices; i += 3) {
    model->addTriangle(cube_indices[i], cube_indices[i], cube_indices[i],
                       cube_indices[i + 1], cube_indices[i + 1],
                       cube_indices[i + 1], cube_indices[i + 2],
                       cube_indices[i + 2], cube_indices[i + 2]);
  }
  m_models[uid] = model;
  return uid;
}

int TinySceneRenderer::create_capsule(float radius, float half_height,
                                      int up_axis,
                                      const std::vector<unsigned char>& texture,
                                      int texture_width, int texture_height) {
  int uid = m_guid++;
  TinyRender::Model* model = new TinyRender::Model();

  if (!texture.empty() && texture.size() == texture_width * texture_height * 3) {
    model->setDiffuseTextureFromData(&texture[0], texture_width,
                                     texture_height);
  }
  int red = 0;
  int green = 255;
  int blue = 0;  // 0;// 128;
  int strideInBytes = 9 * sizeof(float);
  int graphicsShapeIndex = -1;

  int numVertices = sizeof(textured_sphere_vertices) / strideInBytes;
  int numIndices = sizeof(textured_sphere_indices) / sizeof(int);

  std::array<std::array<int,3>,3> index_order = {std::array<int,3>{1,0,2},std::array<int,3>{0,1,2},std::array<int,3>{0,2,1}};


  std::array<int,3> shuffled = index_order[up_axis];
  

  // scale and transform
  std::vector<float> transformedVertices;
  {
    int numVertices = sizeof(textured_sphere_vertices) / strideInBytes;
    transformedVertices.resize(numVertices * 9);
    for (int i = 0; i < numVertices; i++) {
      float trVert[3] = {textured_sphere_vertices[i * 9 + shuffled[0]] * 2 * radius,
                         textured_sphere_vertices[i * 9 + shuffled[1]] * 2 * radius,
                         textured_sphere_vertices[i * 9 + shuffled[2]] * 2 * radius};

      if (trVert[up_axis] > 0)
        trVert[up_axis] += half_height;
      else
        trVert[up_axis] -= half_height;

      transformedVertices[i * 9 + 0] = trVert[0];
      transformedVertices[i * 9 + 1] = trVert[1];
      transformedVertices[i * 9 + 2] = trVert[2];
      transformedVertices[i * 9 + 3] = textured_sphere_vertices[i * 9 + 3];
      transformedVertices[i * 9 + 4] = textured_sphere_vertices[i * 9 + 4];
      transformedVertices[i * 9 + 5] = textured_sphere_vertices[i * 9 + 5];
      transformedVertices[i * 9 + 6] = textured_sphere_vertices[i * 9 + 6];
      transformedVertices[i * 9 + 7] = textured_sphere_vertices[i * 9 + 7];
      transformedVertices[i * 9 + 8] = textured_sphere_vertices[i * 9 + 8];
    }
  }

  for (int i = 0; i < numVertices; i++) {
    model->addVertex(
        transformedVertices[i * 9], transformedVertices[i * 9 + 1],
        transformedVertices[i * 9 + 2], transformedVertices[i * 9 + 4],
        transformedVertices[i * 9 + 5], transformedVertices[i * 9 + 6],
        transformedVertices[i * 9 + 7], transformedVertices[i * 9 + 8]);
  }

  for (int i = 0; i < numIndices; i += 3) {
    model->addTriangle(
        textured_sphere_indices[i], textured_sphere_indices[i],
        textured_sphere_indices[i], textured_sphere_indices[i + 1],
        textured_sphere_indices[i + 1], textured_sphere_indices[i + 1],
        textured_sphere_indices[i + 2], textured_sphere_indices[i + 2],
        textured_sphere_indices[i + 2]);
  }
  m_models[uid] = model;
  return uid;
}

void TinySceneRenderer::set_object_position(
    int instance_uid, const std::vector<float>& position) {
  auto object_instance = m_object_instances[instance_uid];
  if (object_instance && position.size() == 3) {
    object_instance->m_modelMatrix[0][3] = position[0];
    object_instance->m_modelMatrix[1][3] = position[1];
    object_instance->m_modelMatrix[2][3] = position[2];
  }
}
void TinySceneRenderer::set_object_orientation(
    int instance_uid, const std::vector<float>& orientation) {
  auto object_instance = m_object_instances[instance_uid];
  if (object_instance && orientation.size() == 4) {
    float x = orientation[0];
    float y = orientation[1];
    float z = orientation[2];
    float w = orientation[3];

    float d = x * x + y * y + z * z + w * w;
    assert(d != 0.);
    float s = float(2.0) / d;

    float xs = x * s, ys = y * s, zs = z * s;
    float wx = w * xs, wy = w * ys, wz = w * zs;
    float xx = x * xs, xy = x * ys, xz = x * zs;
    float yy = y * ys, yz = y * zs, zz = z * zs;

    // todo: matrix may require transpose
    object_instance->m_modelMatrix[0][0] = float(1.0) - (yy + zz);
    object_instance->m_modelMatrix[0][1] = xy - wz;
    object_instance->m_modelMatrix[0][2] = xz + wy;

    object_instance->m_modelMatrix[1][0] = xy + wz;
    object_instance->m_modelMatrix[1][1] = float(1.0) - (xx + zz);
    object_instance->m_modelMatrix[1][2] = yz - wx;

    object_instance->m_modelMatrix[2][0] = xz - wy;
    object_instance->m_modelMatrix[2][1] = yz + wx;
    object_instance->m_modelMatrix[2][2] = float(1.0) - (xx + yy);
  }
}

void TinySceneRenderer::set_object_color(int instance_uid,
                                         const std::vector<float>& color) {
  auto object_instance = m_object_instances[instance_uid];
  if (object_instance) {
    if (color.size() == 4) {
      Model* model = m_models[object_instance->m_mesh_uid];
      if (model)
      {
        model->setColorRGBA(&color[0]);
      }
    }
  }
}

void TinySceneRenderer::set_object_segmentation_uid(
    int instance_uid, int object_segmentation_uid) {
  auto object_instance = m_object_instances[instance_uid];
  if (object_instance) {
    object_instance->m_object_segmentation_uid = object_segmentation_uid;
  }
}

int TinySceneRenderer::get_object_segmentation_uid(int instance_uid) const {
  if (m_object_instances.find(instance_uid) == m_object_instances.end()) {
    return -1;
  }
  const auto object_instance = m_object_instances.at(instance_uid);
  return object_instance->m_object_segmentation_uid;
}

void TinySceneRenderer::set_object_double_sided(int instance_uid,
                                                bool double_sided) {
  auto object_instance = m_object_instances[instance_uid];
  if (object_instance) {
    object_instance->m_doubleSided = double_sided;
  }
}

void TinySceneRenderer::set_object_local_scaling(
    int instance_uid, const std::vector<float>& local_scaling) {
  auto object_instance = m_object_instances[instance_uid];
  if (object_instance && local_scaling.size() == 3) {
    object_instance->m_localScaling[0] = local_scaling[0];
    object_instance->m_localScaling[1] = local_scaling[1];
    object_instance->m_localScaling[2] = local_scaling[2];
  }
}

int TinySceneRenderer::create_object_instance(int mesh_uid) {
  TinyRender::Model* model = this->m_models[mesh_uid];
  if (model == 0) 
      return -1;

  TinyRenderObjectInstance* tinyObj = new TinyRenderObjectInstance();
  tinyObj->m_mesh_uid = mesh_uid;
  tinyObj->m_doubleSided = true;

  int uid = m_guid++;
  m_object_instances[uid] = tinyObj;
  return uid;
}

RenderBuffers TinySceneRenderer::get_camera_image_py(
    int width, int height, const std::vector<int>& objects,
    const std::vector<float>& viewMatrix,
    const std::vector<float>& projectionMatrix) {
  RenderBuffers buffers(width, height);
  get_camera_image(objects, viewMatrix, projectionMatrix,
                   buffers);
  return buffers;
}

void TinySceneRenderer::get_camera_image(
    const std::vector<int>& objects,
    const std::vector<float>& viewMatrix,
    const std::vector<float>& projectionMatrix, RenderBuffers& buffers) {

  int width = buffers.m_width;
  int height = buffers.m_height;
  // clear the color buffer
  TGAColor clearColor;
  clearColor.bgra[0] = 255;
  clearColor.bgra[1] = 255;
  clearColor.bgra[2] = 255;
  clearColor.bgra[3] = 255;

  float nearPlane = projectionMatrix[14] / (projectionMatrix[10] - 1);
  float farPlane = projectionMatrix[14] / (projectionMatrix[10] + 1);

  for (int x = 0; x < width; x++) {
    for (int y = 0; y < height; y++) {
      buffers.rgb[3 * (x + y * width) + 0] = clearColor[0];
      buffers.rgb[3 * (x + y * width) + 1] = clearColor[1];
      buffers.rgb[3 * (x + y * width) + 2] = clearColor[2];
      buffers.zbuffer[x + y * width] = -farPlane;
    }
  }

  for (int i = 0; i < objects.size(); i++) {
    int uid = objects[i];
    auto object_instance = m_object_instances[uid];
    if (object_instance) {
      for (int i = 0; i < 4; i++) {
        TinyRender::Vec4f p;
        TinyRender::Vec4f v;
        for (int j = 0; j < 4; j++) {
          p[j] = projectionMatrix[i * 4 + j];
          v[j] = viewMatrix[i * 4 + j];
        }
        object_instance->m_projectionMatrix.set_col(i, p);
        object_instance->m_viewMatrix.set_col(i, v);
      }

      object_instance->m_lightDirWorld.x = 0.5773502;
      object_instance->m_lightDirWorld.y = 0.5773502;
      object_instance->m_lightDirWorld.z = 0.5773502;

      object_instance->m_lightDistance = 2;
      object_instance->m_viewportMatrix =
          TinyRender::viewport(0, 0, width, height);

      renderObject(width, height, *object_instance, buffers);
    }
  }
}
void TinySceneRenderer::delete_mesh(int mesh_uid) {
    auto mesh_instance = m_models[mesh_uid];
    if (mesh_instance)
    {
        m_models.erase(mesh_uid);
        delete mesh_instance;
    }
}

void TinySceneRenderer::delete_instance(int instance_uid) {
  auto object_instance = m_object_instances[instance_uid];
  if (object_instance) {
      m_object_instances.erase(instance_uid);
      delete object_instance;
  }
}
