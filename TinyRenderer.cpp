#include "TinyRenderer.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

#include "geometry.h"
#include "model.h"
#include "our_gl.h"
#include "tgaimage.h"

using namespace TinyRender;

struct DepthShader : public IShader
{
	Model* m_model;
	Matrix& m_modelMat;
	Matrix m_invModelMat;

	Matrix& m_projectionMat;
	Vec3f m_localScaling;
	Matrix& m_lightModelView;
	float m_lightDistance;

	mat<2, 3, float> varying_uv;   // triangle uv coordinates, written by the vertex shader, read by the fragment shader
	mat<4, 3, float> varying_tri;  // triangle coordinates (clip coordinates), written by VS, read by FS

	mat<3, 3, float> varying_nrm;  // normal per vertex to be interpolated by FS

	DepthShader(Model* model, Matrix& lightModelView, Matrix& projectionMat, Matrix& modelMat, Vec3f localScaling, float lightDistance)
		: m_model(model),
		  m_modelMat(modelMat),
		  m_projectionMat(projectionMat),
		  m_localScaling(localScaling),
		  m_lightModelView(lightModelView),
		  m_lightDistance(lightDistance)
	{
		m_nearPlane = m_projectionMat.col(3)[2] / (m_projectionMat.col(2)[2] - 1);
		m_farPlane = m_projectionMat.col(3)[2] / (m_projectionMat.col(2)[2] + 1);

		m_invModelMat = m_modelMat.invert_transpose();
	}
	virtual Vec4f vertex(int iface, int nthvert)
	{
		Vec2f uv = m_model->uv(iface, nthvert);
		varying_uv.set_col(nthvert, uv);
		varying_nrm.set_col(nthvert, proj<3>(m_invModelMat * embed<4>(m_model->normal(iface, nthvert), 0.f)));
		Vec3f unScaledVert = m_model->vert(iface, nthvert);
		Vec3f scaledVert = Vec3f(unScaledVert[0] * m_localScaling[0],
								 unScaledVert[1] * m_localScaling[1],
								 unScaledVert[2] * m_localScaling[2]);
		Vec4f gl_Vertex = m_projectionMat * m_lightModelView * embed<4>(scaledVert);
		varying_tri.set_col(nthvert, gl_Vertex);
		return gl_Vertex;
	}

	virtual bool fragment(Vec3f bar, TGAColor& color)
	{
		Vec4f p = varying_tri * bar;
		color = TGAColor(255, 255, 255) * (p[2] / m_lightDistance);
		return false;
	}
};

struct Shader : public IShader
{
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

	mat<2, 3, float> varying_uv;   // triangle uv coordinates, written by the vertex shader, read by the fragment shader
	mat<4, 3, float> varying_tri;  // triangle coordinates (clip coordinates), written by VS, read by FS
	mat<4, 3, float> varying_tri_light_view;
	mat<3, 3, float> varying_nrm;  // normal per vertex to be interpolated by FS
	mat<4, 3, float> world_tri;    // model triangle coordinates in the world space used for backface culling, written by VS

	Shader(Model* model, Vec3f light_dir_local, Vec3f light_color, Matrix& modelView, Matrix& lightModelView, Matrix& projectionMat, Matrix& modelMat, Matrix& viewportMat, Vec3f localScaling, const Vec4f& colorRGBA, int width, int height, std::vector<float>* shadowBuffer, float ambient_coefficient = 0.6, float diffuse_coefficient = 0.35, float specular_coefficient = 0.05)
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
		//printf("near=%f, far=%f\n", m_nearPlane, m_farPlane);
		m_invModelMat = m_modelMat.invert_transpose();
		m_projectionModelViewMat = m_projectionMat * m_modelView1;
		m_projectionLightViewMat = m_projectionMat * m_lightModelView;
	}
	virtual Vec4f vertex(int iface, int nthvert)
	{
		
		Vec2f uv = m_model->uv(iface, nthvert);
		varying_uv.set_col(nthvert, uv);
		varying_nrm.set_col(nthvert, proj<3>(m_invModelMat * embed<4>(m_model->normal(iface, nthvert), 0.f)));
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

	virtual bool fragment(Vec3f bar, TGAColor& color)
	{
		
		Vec4f p = m_viewportMat * (varying_tri_light_view * bar);
		float depth = p[2];
		p = p / p[3];

		float index_x = std::max(float(0.0), std::min(float(m_width - 1), p[0]));
		float index_y = std::max(float(0.0), std::min(float(m_height - 1), p[1]));
		int idx = int(index_x) + int(index_y) * m_width;                       // index in the shadowbuffer array
		float shadow = 1.0;
		if (m_shadowBuffer && idx >=0 && idx <m_shadowBuffer->size())
		{
			shadow = 0.8 + 0.2 * (m_shadowBuffer->at(idx) < -depth + 0.05);  // magic coeff to avoid z-fighting
		}
		Vec3f bn = (varying_nrm * bar).normalize();
		Vec2f uv = varying_uv * bar;

		Vec3f reflection_direction = (bn * (bn * m_light_dir_local * 2.f) - m_light_dir_local).normalize();
        float specular = std::pow(std::max(reflection_direction.z, 0.f),
                                    m_model->specular(uv));
        float diffuse = std::max(0.f, bn * m_light_dir_local);

        color = m_model->diffuse(uv);
		color[0] *= m_colorRGBA[0];
		color[1] *= m_colorRGBA[1];
		color[2] *= m_colorRGBA[2];
		color[3] *= m_colorRGBA[3];

		for (int i = 0; i < 3; ++i)
		{
			int orgColor = 0;
			float floatColor = (m_ambient_coefficient * color[i] + shadow * (m_diffuse_coefficient * diffuse + m_specular_coefficient * specular) * color[i] * m_light_color[i]);
			if (floatColor==floatColor)
			{
				orgColor=int(floatColor);
			}
			color[i] = std::min(orgColor, 255);
		}

		return false;
	}
};



TinyRenderObjectInstance::TinyRenderObjectInstance()
	: m_model(0),
	  m_object_segmentation_uid(-1),
	  m_doubleSided(false)
{
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




TinyRenderObjectInstance::~TinyRenderObjectInstance()
{
	
}

static bool equals(const Vec4f& vA, const Vec4f& vB)
{
	return false;
}

static void clipEdge(const mat<4, 3, float>& triangleIn, int vertexIndexA, int vertexIndexB, std::vector<Vec4f>& vertices)
{
	Vec4f v0New = triangleIn.col(vertexIndexA);
	Vec4f v1New = triangleIn.col(vertexIndexB);

	bool v0Inside = v0New[3] > 0.f && v0New[2] > -v0New[3];
	bool v1Inside = v1New[3] > 0.f && v1New[2] > -v1New[3];

	if (v0Inside && v1Inside)
	{
	}
	else if (v0Inside || v1Inside)
	{
		float d0 = v0New[2] + v0New[3];
		float d1 = v1New[2] + v1New[3];
		float factor = 1.0 / (d1 - d0);
		Vec4f newVertex = (v0New * d1 - v1New * d0) * factor;
		if (v0Inside)
		{
			v1New = newVertex;
		}
		else
		{
			v0New = newVertex;
		}
	}
	else
	{
		return;
	}

	if (vertices.size() == 0 || !(equals(vertices[vertices.size() - 1], v0New)))
	{
		vertices.push_back(v0New);
	}

	vertices.push_back(v1New);
}

static bool clipTriangleAgainstNearplane(const mat<4, 3, float>& triangleIn, std::vector<mat<4, 3, float> >& clippedTrianglesOut)
{
	//discard triangle if all vertices are behind near-plane
	if (triangleIn[3][0] < 0 && triangleIn[3][1] < 0 && triangleIn[3][2] < 0)
	{
		return true;
	}

	//accept triangle if all vertices are in front of the near-plane
	if (triangleIn[3][0] >= 0 && triangleIn[3][1] >= 0 && triangleIn[3][2] >= 0)
	{
		clippedTrianglesOut.push_back(triangleIn);
		return false;
	}

	std::vector<Vec4f> vertices;
	vertices.reserve(5);
	
	clipEdge(triangleIn, 0, 1, vertices);
	clipEdge(triangleIn, 1, 2, vertices);
	clipEdge(triangleIn, 2, 0, vertices);

	if (vertices.size() < 3)
		return true;

	if (equals(vertices[0], vertices[vertices.size() - 1]))
	{
		vertices.pop_back();
	}

	//create a fan of triangles
	for (int i = 1; i < vertices.size() - 1; i++)
	{
		mat<4, 3, float> vtx;
		vtx.set_col(0, vertices[0]);
		vtx.set_col(1, vertices[i]);
		vtx.set_col(2, vertices[i + 1]);
		clippedTrianglesOut.push_back(vtx);

	}
	return true;
}



void TinyRenderer::renderObject(int width, int height, TinyRenderObjectInstance& object_instance, RenderBuffers& render_buffers )
{

	Vec3f light_dir_local = Vec3f(object_instance.m_lightDirWorld[0], object_instance.m_lightDirWorld[1], object_instance.m_lightDirWorld[2]);
	Vec3f light_color = Vec3f(object_instance.m_lightColor[0], object_instance.m_lightColor[1], object_instance.m_lightColor[2]);
	float light_distance = object_instance.m_lightDistance;
	Model* model = object_instance.m_model;
	if (0 == model)
		return;

	//discard invisible objects (zero alpha)
	if (model->getColorRGBA()[3] == 0)
		return;

	object_instance.m_viewportMatrix = viewport(0, 0, width, height);
	std::vector<float>* shadowBufferPtr = 0;//object_instance.m_shadowBuffer;

	{
		// light target is set to be the origin, and the up direction is set to be vertical up.
		Matrix lightViewMatrix = lookat(light_dir_local * light_distance, Vec3f(0.0, 0.0, 0.0), Vec3f(0.0, 0.0, 1.0));
		Matrix lightModelViewMatrix = lightViewMatrix * object_instance.m_modelMatrix;
		Matrix modelViewMatrix = object_instance.m_viewMatrix * object_instance.m_modelMatrix;
		Vec3f localScaling(object_instance.m_localScaling[0], object_instance.m_localScaling[1], object_instance.m_localScaling[2]);
		Matrix viewMatrixInv = object_instance.m_viewMatrix.invert();
		TinyRender::Vec3f P(viewMatrixInv[0][3], viewMatrixInv[1][3], viewMatrixInv[2][3]);

		Shader shader(model, light_dir_local, light_color, modelViewMatrix, lightModelViewMatrix, object_instance.m_projectionMatrix, object_instance.m_modelMatrix, object_instance.m_viewportMatrix, localScaling, model->getColorRGBA(), width, height, shadowBufferPtr, object_instance.m_lightAmbientCoeff, object_instance.m_lightDiffuseCoeff, object_instance.m_lightSpecularCoeff);

		{
			

			for (int i = 0; i < model->nfaces(); i++)
			{
				for (int j = 0; j < 3; j++)
				{
					shader.vertex(i, j);
				}

				if (!object_instance.m_doubleSided)
				{
					// backface culling
					TinyRender::Vec3f v0(shader.world_tri.col(0)[0], shader.world_tri.col(0)[1], shader.world_tri.col(0)[2]);
					TinyRender::Vec3f v1(shader.world_tri.col(1)[0], shader.world_tri.col(1)[1], shader.world_tri.col(1)[2]);
					TinyRender::Vec3f v2(shader.world_tri.col(2)[0], shader.world_tri.col(2)[1], shader.world_tri.col(2)[2]);
					TinyRender::Vec3f N = TinyRender::cross((v1 - v0),(v2 - v0));
					if (TinyRender::dot((v0 - P),(N)) >= 0)
						continue;
				}

				std::vector<mat<4, 3, float> > clippedTriangles;
				clippedTriangles.reserve(3);
				
				bool hasClipped = clipTriangleAgainstNearplane(shader.varying_tri, clippedTriangles);

				if (hasClipped)
				{
					for (int t = 0; t < clippedTriangles.size(); t++)
					{
						triangleClipped(clippedTriangles[t], shader.varying_tri, shader, render_buffers, object_instance.m_viewportMatrix, object_instance.m_object_segmentation_uid);
					}
				}
				else
				{
					triangle(shader.varying_tri, shader, render_buffers, object_instance.m_viewportMatrix, object_instance.m_object_segmentation_uid);
				}
			}
		}
	}
	return;
}

