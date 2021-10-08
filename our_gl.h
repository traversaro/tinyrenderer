#ifndef __OUR_GL_H__
#define __OUR_GL_H__
#include "geometry.h"
#include "renderbuffers.h"
#include "tgaimage.h"

namespace TinyRender {
Matrix viewport(int x, int y, int w, int h);
Matrix projection(float coeff = 0.f);  // coeff = -1/c
Matrix lookat(Vec3f eye, Vec3f center, Vec3f up);

struct IShader {
  float m_nearPlane;
  float m_farPlane;
  IShader() : m_nearPlane(0.01), m_farPlane(1000.) {}
  virtual ~IShader();
  virtual Vec4f vertex(int iface, int nthvert) = 0;
  virtual bool fragment(Vec3f bar, TGAColor &color) = 0;
};

void triangle(mat<4, 3, float> &clipc, IShader &shader,
              RenderBuffers &render_buffers, const Matrix &viewPortMatrix,
              int objectUniqueId, bool create_shadow_map);

void triangleClipped(mat<4, 3, float> &clipc, mat<4, 3, float> &orgClipc,
                     IShader &shader, RenderBuffers &render_buffers,
                     const Matrix &viewPortMatrix, int objectUniqueId,
                     bool create_shadow_map);
}  // namespace TinyRender

#endif  //__OUR_GL_H__
