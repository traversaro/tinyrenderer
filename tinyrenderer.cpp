#include "tinyrenderer.h"

#include <float.h>

#include <array>
#include <cmath>
#include <iostream>
#include <limits>

#include "geometry.h"
#include "model.h"
#include "our_gl.h"
#include "tgaimage.h"
#include "tinyshapedata.h"

using namespace TinyRender;


struct DepthShader : public IShader {
  const Model* m_model;
  const Matrix& m_modelMat;
  Matrix m_invModelMat;

  const Matrix& m_projectionMat;
  Vec3f m_localScaling;
  const Matrix& m_lightModelView;
  float m_lightDistance;

  mat<2, 3, float> varying_uv;   // triangle uv coordinates, written by the
                                 // vertex shader, read by the fragment shader
  mat<4, 3, float> varying_tri;  // triangle coordinates (clip coordinates),
                                 // written by VS, read by FS

  mat<3, 3, float> varying_nrm;  // normal per vertex to be interpolated by FS

  DepthShader(const Model* model, const Matrix& lightModelView,
              const Matrix& projectionMat, const Matrix& modelMat,
              Vec3f localScaling, float lightDistance)
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
  const Matrix& m_modelMat;
  Matrix m_invModelMat;
  Matrix& m_modelView1;
  const Matrix& m_projectionMat;
  Vec3f m_localScaling;
  Matrix& m_lightModelView;
  Vec4f m_colorRGBA;
  const Matrix& m_viewportMat;
  Matrix m_projectionModelViewMat;
  Matrix m_projectionLightViewMat;
  float m_ambient_coefficient;
  float m_diffuse_coefficient;
  float m_specular_coefficient;

  const std::vector<float>* m_shadowBuffer;
  const std::vector<int>* m_shadowObjectUniqueIds;
  int m_objectUniqueId;
  int m_width;
  int m_height;
  float m_shadow_coefficient;

  mat<2, 3, float> varying_uv;   // triangle uv coordinates, written by the
                                 // vertex shader, read by the fragment shader
  mat<4, 3, float> varying_tri;  // triangle coordinates (clip coordinates),
                                 // written by VS, read by FS
  mat<4, 3, float> varying_tri_light_view;
  mat<3, 3, float> varying_nrm;  // normal per vertex to be interpolated by FS
  mat<4, 3, float> world_tri;  // model triangle coordinates in the world space
                               // used for backface culling, written by VS

  Shader(Model* model, Vec3f light_dir_local, Vec3f light_color,
         Matrix& modelView, Matrix& lightModelView, const Matrix& projectionMat,
         const Matrix& modelMat, const Matrix& viewportMat, Vec3f localScaling,
         const Vec4f& colorRGBA, int width, int height,
         const std::vector<float>* shadowBuffer,
         const std::vector<int>* shadowObjectUniqueIds, int objectUniqueId,
         float shadow_coefficient = 0.4, float ambient_coefficient = 0.6,
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
        m_shadowObjectUniqueIds(shadowObjectUniqueIds),
        m_objectUniqueId(objectUniqueId),
        m_width(width),
        m_height(height),
       m_shadow_coefficient(shadow_coefficient)
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
    if (m_shadowBuffer && m_shadowObjectUniqueIds && idx >= 0 &&
        idx < m_shadowBuffer->size() && idx < m_shadowObjectUniqueIds->size()) {
      float shadowVal = m_shadowBuffer->at(idx);
      int shadowObjecUniqueId = m_shadowObjectUniqueIds->at(idx);
      if (shadowObjecUniqueId != m_objectUniqueId &&
          (shadowVal > (-depth + 0.05)))  // magic coeff to avoid z-fighting
      {
        shadow = m_shadow_coefficient;  // darkness of shadow, smaller value is
                                        // darker
      }
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

TinyRenderCamera::TinyRenderCamera(int viewWidth, int viewHeight, float near,
                                   float far, float hfov, float vfov,
                                   const std::vector<float>& position,
                                   const std::vector<float>& target,
                                   const std::vector<float>& up)
    : m_viewWidth(viewWidth), m_viewHeight(viewHeight) {
  auto viewMatrix =
      TinySceneRenderer::compute_view_matrix(position, target, up);
  auto projectionMatrix =
      TinySceneRenderer::compute_projection_matrix(hfov, vfov, near, far);

  for (int i = 0; i < 4; i++) {
    TinyRender::Vec4f p;
    TinyRender::Vec4f v;
    for (int j = 0; j < 4; j++) {
      p[j] = projectionMatrix[i * 4 + j];
      v[j] = viewMatrix[i * 4 + j];
    }
    m_projectionMatrix.set_col(i, p);
    m_viewMatrix.set_col(i, v);
  }
  m_viewportMatrix = viewport(0, 0, viewWidth, viewHeight);
}

TinyRenderCamera::TinyRenderCamera(int viewWidth, int viewHeight,
                                   const std::vector<float>& viewMatrix,
                                   const std::vector<float>& projectionMatrix)
    : m_viewWidth(viewWidth), m_viewHeight(viewHeight) {
  for (int i = 0; i < 4; i++) {
    TinyRender::Vec4f p;
    TinyRender::Vec4f v;
    for (int j = 0; j < 4; j++) {
      p[j] = projectionMatrix[i * 4 + j];
      v[j] = viewMatrix[i * 4 + j];
    }
    m_projectionMatrix.set_col(i, p);
    m_viewMatrix.set_col(i, v);
  }
  m_viewportMatrix = viewport(0, 0, viewWidth, viewHeight);
}

TinyRenderCamera::~TinyRenderCamera() {}

TinyRenderLight::TinyRenderLight(const std::vector<float>& direction,
                                 const std::vector<float>& color,
                                 const std::vector<float>& shadowmap_center,
                                 float distance, float ambient, float diffuse,
                                 float specular, bool has_shadow,
                                 float shadow_coefficient)
    : m_distance(distance),
      m_ambientCoeff(ambient),
      m_diffuseCoeff(diffuse),
      m_specularCoeff(specular),
      m_has_shadow(has_shadow),
      m_shadow_coefficient(shadow_coefficient) {
  m_dirWorld = Vec3f(direction[0], direction[1], direction[2]);
  m_color = Vec3f(color[0], color[1], color[2]);
  m_shadowmap_center = Vec3f(0,0,0);
}

TinyRenderLight::~TinyRenderLight() {}

TinyRenderObjectInstance::TinyRenderObjectInstance()
    : m_mesh_uid(-1), m_object_segmentation_uid(-1), m_doubleSided(false) {
  m_localScaling = TinyRender::Vec3f(1, 1, 1);
  m_modelMatrix = Matrix::identity();
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

void TinySceneRenderer::renderObject(
    const TinyRenderLight& light, const TinyRenderCamera& camera,
    const TinyRenderObjectInstance& object_instance,
    RenderBuffers& render_buffers) {
  Model* model = m_models[object_instance.m_mesh_uid];
  if (0 == model) return;

  // discard invisible objects (zero alpha)
  if (model->getColorRGBA()[3] == 0) return;

  const std::vector<float>* shadowBufferPtr =
      (render_buffers.shadow_buffer.size()) ? &render_buffers.shadow_buffer : 0;
  const std::vector<int>* shadowBufferObjectUniqueIdPtr =
      (render_buffers.shadow_uids.size()) ? &render_buffers.shadow_uids : 0;

  {
    // light target is set to be the origin, and the up direction is set to be
    // vertical up.
    
    Matrix lightViewMatrix = lookat(light.m_shadowmap_center+light.m_dirWorld * light.m_distance,
                                    light.m_shadowmap_center, Vec3f(0.0, 0.0, 1.0));
    Matrix lightModelViewMatrix =
        lightViewMatrix * object_instance.m_modelMatrix;
    Matrix modelViewMatrix =
        camera.m_viewMatrix * object_instance.m_modelMatrix;
    Vec3f localScaling(object_instance.m_localScaling[0],
                       object_instance.m_localScaling[1],
                       object_instance.m_localScaling[2]);
    Matrix viewMatrixInv = camera.m_viewMatrix.invert();
    TinyRender::Vec3f P(viewMatrixInv[0][3], viewMatrixInv[1][3],
                        viewMatrixInv[2][3]);

    Shader shader(
        model, light.m_dirWorld, light.m_color, modelViewMatrix,
        lightModelViewMatrix, camera.m_projectionMatrix,
        object_instance.m_modelMatrix, camera.m_viewportMatrix, localScaling,
        model->getColorRGBA(), camera.m_viewWidth, camera.m_viewHeight,
        shadowBufferPtr, shadowBufferObjectUniqueIdPtr,
        object_instance.m_object_segmentation_uid, light.m_shadow_coefficient,
        light.m_ambientCoeff, light.m_diffuseCoeff, light.m_specularCoeff);

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
                            render_buffers, camera.m_viewportMatrix,
                            object_instance.m_object_segmentation_uid, false);
          }
        } else {
          triangle(shader.varying_tri, shader, render_buffers,
                   camera.m_viewportMatrix,
                   object_instance.m_object_segmentation_uid, false);
        }
      }
    }
  }
}

#define _SQRT12 float(0.7071067811865475244008443621048490)

template <class T>
void planeSpace(const T& n, T& p, T& q) {
  if (fabsf(n[2]) > _SQRT12) {
    // choose p in y-z plane
    float a = n[1] * n[1] + n[2] * n[2];
    float k = 1. / sqrtf(a);
    p[0] = 0;
    p[1] = -n[2] * k;
    p[2] = n[1] * k;
    // set q = n x p
    q[0] = a * k;
    q[1] = -n[0] * p[2];
    q[2] = n[0] * p[1];
  } else {
    // choose p in x-y plane
    float a = n[0] * n[0] + n[1] * n[1];
    float k = 1. / sqrtf(a);
    p[0] = -n[1] * k;
    p[1] = n[0] * k;
    p[2] = 0;
    // set q = n x p
    q[0] = -n[2] * p[1];
    q[1] = n[2] * p[0];
    q[2] = a * k;
  }
}

void TinySceneRenderer::renderObjectDepth(
    const TinyRenderLight& light, const TinyRenderCamera& camera,
    const TinyRenderObjectInstance& object_instance,
    RenderBuffers& render_buffers) {
  int width = render_buffers.m_width;
  int height = render_buffers.m_height;

  Vec3f light_dir_local =
      Vec3f(light.m_dirWorld[0], light.m_dirWorld[1], light.m_dirWorld[2]);
  float light_distance = light.m_distance;

  Model* model = m_models[object_instance.m_mesh_uid];
  if (0 == model) return;

  // discard invisible objects (zero alpha)
  if (model->getColorRGBA()[3] == 0) return;

  float* shadowBufferPtr = (render_buffers.shadow_buffer.size())
                               ? &render_buffers.shadow_buffer.at(0)
                               : 0;
  int* segmentationMaskBufferPtr = 0;

  TGAImage depthFrame(width, height, TGAImage::RGB);

  {
    // light target is set to be the origin, and the up direction is set to be
    // vertical up.
    
    Vec3f up(0.0, 0.0, 1.0);

    Matrix lightViewMatrix =
        lookat(light.m_shadowmap_center+light_dir_local * light_distance, light.m_shadowmap_center, up);
    Matrix lightModelViewMatrix =
        lightViewMatrix * object_instance.m_modelMatrix;
    Matrix lightViewProjectionMatrix = camera.m_projectionMatrix;
    Vec3f localScaling(object_instance.m_localScaling[0],
                       object_instance.m_localScaling[1],
                       object_instance.m_localScaling[2]);

    DepthShader shader(model, lightModelViewMatrix, lightViewProjectionMatrix,
                       object_instance.m_modelMatrix, localScaling,
                       light_distance);
    for (int i = 0; i < model->nfaces(); i++) {
      for (int j = 0; j < 3; j++) {
        shader.vertex(i, j);
      }

      mat<4, 3, float> stackTris[3];

      std::vector<mat<4, 3, float> > clippedTriangles;
      clippedTriangles.reserve(3);

      bool hasClipped =
          clipTriangleAgainstNearplane(shader.varying_tri, clippedTriangles);

      if (hasClipped) {
        for (int t = 0; t < clippedTriangles.size(); t++) {
          triangleClipped(clippedTriangles[t], shader.varying_tri, shader,
                          render_buffers, camera.m_viewportMatrix,
                          object_instance.m_object_segmentation_uid, true);
        }
      } else {
        triangle(shader.varying_tri, shader, render_buffers,
                 camera.m_viewportMatrix,
                 object_instance.m_object_segmentation_uid, true);
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

std::vector<float> TinySceneRenderer::compute_view_matrix_from_positions(
    const float cameraPosition[3], const float cameraTargetPosition[3],
    const float cameraUp[3]) {
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

  viewMatrix[3 * 4 + 0] = -dot(s, eye);
  viewMatrix[3 * 4 + 1] = -dot(u, eye);
  viewMatrix[3 * 4 + 2] = dot(f, eye);
  viewMatrix[3 * 4 + 3] = 1.f;
  return viewMatrix;
}

void TinySceneRenderer::setEulerZYX(const float& yawZ, const float& pitchY,
                                    const float& rollX, Vec4f& euler) {
  float halfYaw = float(yawZ) * float(0.5);
  float halfPitch = float(pitchY) * float(0.5);
  float halfRoll = float(rollX) * float(0.5);
  float cosYaw = cosf(halfYaw);
  float sinYaw = sinf(halfYaw);
  float cosPitch = cos(halfPitch);
  float sinPitch = sin(halfPitch);
  float cosRoll = cos(halfRoll);
  float sinRoll = sin(halfRoll);
  euler[0] = sinRoll * cosPitch * cosYaw - cosRoll * sinPitch * sinYaw;
  euler[1] = cosRoll * sinPitch * cosYaw + sinRoll * cosPitch * sinYaw;
  euler[2] = cosRoll * cosPitch * sinYaw - sinRoll * sinPitch * cosYaw;
  euler[3] = cosRoll * cosPitch * cosYaw + sinRoll * sinPitch * sinYaw;
  euler.normalize();
}

Vec4f TinySceneRenderer::quatMul3(const Vec4f& q, const Vec3f& w) {
  Vec4f mul;
  mul[0] = q[3] * w[0] + q[1] * w[2] - q[2] * w[1];
  mul[1] = q[3] * w[1] + q[2] * w[0] - q[0] * w[2];
  mul[2] = q[3] * w[2] + q[0] * w[1] - q[1] * w[0];
  mul[3] = -q[0] * w[0] - q[1] * w[1] - q[2] * w[2];
  return mul;
}

Vec4f TinySceneRenderer::quatMul4(const Vec4f& q, const Vec4f& w) {
  Vec4f mul;
  mul[0] = q[3] * w[0] + q[0] * w[3] + q[1] * w[2] - q[2] * w[1];
  mul[1] = q[3] * w[1] + q[1] * w[3] + q[2] * w[0] - q[0] * w[2];
  mul[2] = q[3] * w[2] + q[2] * w[3] + q[0] * w[1] - q[1] * w[0];
  mul[3] = q[3] * w[3] - q[0] * w[0] - q[1] * w[1] - q[2] * w[2];
  return mul;
}

Vec4f TinySceneRenderer::inverse(const Vec4f& rotation) {
  Vec4f inv;
  inv[0] = -rotation[0];
  inv[1] = -rotation[1];
  inv[2] = -rotation[2];
  inv[3] = rotation[3];
  return inv;
}

Vec3f TinySceneRenderer::quatRotate(const Vec4f& rotation, const Vec3f& v) {
  Vec4f q = quatMul3(rotation, v);
  Vec4f inv = inverse(rotation);
  q = quatMul4(q, inv);
  return Vec3f(q[0], q[1], q[2]);
}

std::vector<float> TinySceneRenderer::compute_view_matrix_from_yaw_pitch_roll(
    const float cameraTargetPosition[3], float distance, float yaw, float pitch,
    float roll, int upAxis) {
  Vec3f camUpVector;
  Vec3f camForward;
  Vec3f camPos;
  Vec3f camTargetPos = Vec3f(cameraTargetPosition[0], cameraTargetPosition[1],
                             cameraTargetPosition[2]);
  Vec3f eyePos = Vec3f(0, 0, 0);

  float yawRad = yaw * float(0.01745329251994329547);      // rads per deg
  float pitchRad = pitch * float(0.01745329251994329547);  // rads per deg
  float rollRad = roll * float(0.01745329251994329547);    // rads per deg
  Vec4f eyeRot;

  int forwardAxis(-1);
  switch (upAxis) {
    case 1:
      forwardAxis = 2;
      camUpVector = Vec3f(0, 1, 0);
      setEulerZYX(rollRad, yawRad, -pitchRad, eyeRot);
      break;
    case 2:
    default:
      forwardAxis = 1;
      camUpVector = Vec3f(0, 0, 1);
      setEulerZYX(yawRad, rollRad, pitchRad, eyeRot);
      break;
  };

  eyePos[forwardAxis] = -distance;

  camForward = Vec3f(eyePos[0], eyePos[1], eyePos[2]);
  float len2 = dot(camForward, camForward);
  if (len2 < FLT_EPSILON) {
    camForward = Vec3f(1.f, 0.f, 0.f);
  } else {
    camForward.normalize();
  }

  eyePos = quatRotate(eyeRot, eyePos);
  camUpVector = quatRotate(eyeRot, camUpVector);

  camPos = eyePos;
  camPos = camPos + camTargetPos;

  float camPosf[4] = {camPos[0], camPos[1], camPos[2], 0};
  float camPosTargetf[4] = {camTargetPos[0], camTargetPos[1], camTargetPos[2],
                            0};
  float camUpf[4] = {camUpVector[0], camUpVector[1], camUpVector[2], 0};

  return compute_view_matrix_from_positions(camPosf, camPosTargetf, camUpf);
}

std::vector<float> TinySceneRenderer::compute_projection_matrix2(
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

std::vector<float> TinySceneRenderer::compute_projection_matrix_fov(
    float fov, float aspect, float nearVal, float farVal) {
  std::vector<float> projectionMatrix;
  projectionMatrix.resize(16);
  float yScale = 1.0 / tan((M_PI / 180.0) * fov / 2);
  float xScale = yScale / aspect;

  projectionMatrix[0 * 4 + 0] = xScale;
  projectionMatrix[0 * 4 + 1] = float(0);
  projectionMatrix[0 * 4 + 2] = float(0);
  projectionMatrix[0 * 4 + 3] = float(0);

  projectionMatrix[1 * 4 + 0] = float(0);
  projectionMatrix[1 * 4 + 1] = yScale;
  projectionMatrix[1 * 4 + 2] = float(0);
  projectionMatrix[1 * 4 + 3] = float(0);

  projectionMatrix[2 * 4 + 0] = 0;
  projectionMatrix[2 * 4 + 1] = 0;
  projectionMatrix[2 * 4 + 2] = (nearVal + farVal) / (nearVal - farVal);
  projectionMatrix[2 * 4 + 3] = float(-1);

  projectionMatrix[3 * 4 + 0] = float(0);
  projectionMatrix[3 * 4 + 1] = float(0);
  projectionMatrix[3 * 4 + 2] =
      (float(2) * farVal * nearVal) / (nearVal - farVal);
  projectionMatrix[3 * 4 + 3] = float(0);
  return projectionMatrix;
}

std::vector<float> TinySceneRenderer::compute_projection_matrix(float hfov,
                                                                float vfov,
                                                                float near,
                                                                float far) {
  float left = -tan(M_PI * hfov / 360.0) * near;
  float right = -left;
  float bottom = -tan(M_PI * vfov / 360.0) * near;
  float top = -bottom;

  std::vector<float> projectionMatrix;
  projectionMatrix.resize(16);

  projectionMatrix[0 * 4 + 0] = (float(2) * near) / (right - left);
  projectionMatrix[0 * 4 + 1] = float(0);
  projectionMatrix[0 * 4 + 2] = float(0);
  projectionMatrix[0 * 4 + 3] = float(0);

  projectionMatrix[1 * 4 + 0] = float(0);
  projectionMatrix[1 * 4 + 1] = (float(2) * near) / (top - bottom);
  projectionMatrix[1 * 4 + 2] = float(0);
  projectionMatrix[1 * 4 + 3] = float(0);

  projectionMatrix[2 * 4 + 0] = (right + left) / (right - left);
  projectionMatrix[2 * 4 + 1] = (top + bottom) / (top - bottom);
  projectionMatrix[2 * 4 + 2] = -(far + near) / (far - near);
  projectionMatrix[2 * 4 + 3] = float(-1);

  projectionMatrix[3 * 4 + 0] = float(0);
  projectionMatrix[3 * 4 + 1] = float(0);
  projectionMatrix[3 * 4 + 2] = -(float(2) * far * near) / (far - near);
  projectionMatrix[3 * 4 + 3] = float(0);

  return projectionMatrix;
}

TinySceneRenderer::TinySceneRenderer() : m_guid(1) {}

TinySceneRenderer::~TinySceneRenderer() {
  // free all memory
  {
    auto it = m_object_instances.begin();
    while (it != m_object_instances.end()) {
      auto value = it->second;
      delete value;
      it++;
    }
    m_object_instances.clear();
  }
  {
    auto it = m_models.begin();
    while (it != m_models.end()) {
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

  if (!texture.empty() &&
      texture.size() == texture_width * texture_height * 3) {
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

  if (!texture.empty() &&
      texture.size() == texture_width * texture_height * 3) {
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

  if (!texture.empty() &&
      texture.size() == texture_width * texture_height * 3) {
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

  std::array<std::array<int, 3>, 3> index_order = {std::array<int, 3>{1, 0, 2},
                                                   std::array<int, 3>{0, 1, 2},
                                                   std::array<int, 3>{0, 2, 1}};

  std::array<int, 3> shuffled = index_order[up_axis];

  // scale and transform
  std::vector<float> transformedVertices;
  {
    int numVertices = sizeof(textured_sphere_vertices) / strideInBytes;
    transformedVertices.resize(numVertices * 9);
    for (int i = 0; i < numVertices; i++) {
      float trVert[3] = {
          textured_sphere_vertices[i * 9 + shuffled[0]] * radius,
          textured_sphere_vertices[i * 9 + shuffled[1]] * radius,
          textured_sphere_vertices[i * 9 + shuffled[2]] * radius};

      if (trVert[up_axis] > 0)
        trVert[up_axis] += half_height;
      else
        trVert[up_axis] -= half_height;

      transformedVertices[i * 9 + 0] = trVert[0];
      transformedVertices[i * 9 + 1] = trVert[1];
      transformedVertices[i * 9 + 2] = trVert[2];
      transformedVertices[i * 9 + 3] = textured_sphere_vertices[i * 9 + 3];
      transformedVertices[i * 9 + 4] = textured_sphere_vertices[i * 9 + 4+shuffled[0]];
      transformedVertices[i * 9 + 5] = textured_sphere_vertices[i * 9 + 4+shuffled[1]];
      transformedVertices[i * 9 + 6] = textured_sphere_vertices[i * 9 + 4+shuffled[2]];
      transformedVertices[i * 9 + 7] = textured_sphere_vertices[i * 9 + 7];
      transformedVertices[i * 9 + 8] = textured_sphere_vertices[i * 9 + 8];
    }
  }

  for (int i = 0; i < numVertices; i++) {
    model->addVertex(transformedVertices[i * 9], transformedVertices[i * 9 + 1],
        transformedVertices[i * 9 + 2], 
        transformedVertices[i * 9 + 4],        transformedVertices[i * 9 + 5], transformedVertices[i * 9 + 6],
        transformedVertices[i * 9 + 7], transformedVertices[i * 9 + 8]);
  }

  for (int i = 0; i < numIndices; i += 3) {
    model->addTriangle(
        textured_sphere_indices[i], textured_sphere_indices[i],textured_sphere_indices[i], 
        textured_sphere_indices[i + 1],textured_sphere_indices[i + 1], textured_sphere_indices[i + 1],
        textured_sphere_indices[i + 2], textured_sphere_indices[i + 2],textured_sphere_indices[i + 2]);
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
      if (model) {
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
  if (model == 0) return -1;

  TinyRenderObjectInstance* tinyObj = new TinyRenderObjectInstance();
  tinyObj->m_mesh_uid = mesh_uid;
  tinyObj->m_doubleSided = false;

  int uid = m_guid++;
  tinyObj->m_object_segmentation_uid = uid;
  m_object_instances[uid] = tinyObj;
  return uid;
}

RenderBuffers TinySceneRenderer::get_camera_image_py(
    const std::vector<int>& objects, const TinyRenderLight& light,
    const TinyRenderCamera& camera) {
  RenderBuffers buffers(camera.m_viewWidth, camera.m_viewHeight);
  get_camera_image(objects, light, camera, buffers);
  return buffers;
}

void TinySceneRenderer::get_camera_image(const std::vector<int>& objects,
                                         const TinyRenderLight& light,
                                         const TinyRenderCamera& camera,
                                         RenderBuffers& buffers) {
  int width = camera.m_viewWidth;
  int height = camera.m_viewHeight;
  buffers.resize(width, height);

  float farPlane =
      camera.m_projectionMatrix[3][2] / (camera.m_projectionMatrix[2][2] + 1);

  buffers.clear(farPlane);

  if (light.m_has_shadow) {
    for (int i = 0; i < objects.size(); i++) {
      int uid = objects[i];
      auto object_instance = m_object_instances[uid];
      if (object_instance) {
        renderObjectDepth(light, camera, *object_instance, buffers);
      }
    }
  }

  for (int i = 0; i < objects.size(); i++) {
    int uid = objects[i];
    auto object_instance = m_object_instances[uid];
    if (object_instance) {
      renderObject(light, camera, *object_instance, buffers);
    }
  }
}
void TinySceneRenderer::delete_mesh(int mesh_uid) {
  auto mesh_instance = m_models[mesh_uid];
  if (mesh_instance) {
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
