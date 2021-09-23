#ifndef TINY_RENDERER_H
#define TINY_RENDERER_H

#include "geometry.h"
#include "model.h"
#include <vector>
#include "tgaimage.h"

struct TinyRenderObjectInstance
{
	//Camera
	TinyRender::Matrix m_viewMatrix;
	TinyRender::Matrix m_projectionMatrix;
	TinyRender::Matrix m_viewportMatrix;
	TinyRender::Vec3f m_localScaling;
	TinyRender::Vec3f m_lightDirWorld;
	TinyRender::Vec3f m_lightColor;
	float m_lightDistance;
	float m_lightAmbientCoeff;
	float m_lightDiffuseCoeff;
	float m_lightSpecularCoeff;
		
	TinyRender::Matrix m_modelMatrix;
	int m_object_segmentation_uid;
	bool m_doubleSided;


	//Model (vertices, indices, textures, shader)
	TinyRender::Model* m_model;
	//class IShader* m_shader; todo(erwincoumans) expose the shader, for now we use a default shader

	TinyRenderObjectInstance();
	virtual ~TinyRenderObjectInstance();

	void loadModel(const char* fileName, struct CommonFileIOInterface* fileIO);
	void createCube(float HalfExtentsX, float HalfExtentsY, float HalfExtentsZ, struct CommonFileIOInterface* fileIO=0);
	void registerMeshShape(const float* vertices, int numVertices, const int* indices, int numIndices, const float rgbaColor[4],
						   unsigned char* textureImage = 0, int textureWidth = 0, int textureHeight = 0);

	void registerMesh2(std::vector<TinyRender::Vec3f>& vertices, std::vector<TinyRender::Vec3f>& normals, std::vector<int>& indices, struct CommonFileIOInterface* fileIO);

	
};


class TinyRenderer
{
public:
	static void renderObject(int width, int height, TinyRenderObjectInstance& object_instance, struct RenderBuffers& render_buffers );
};

#endif  // TINY_RENDERER_Hbla
