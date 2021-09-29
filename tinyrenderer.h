#ifndef TDS_TINY_RENDERER_H
#define TDS_TINY_RENDERER_H

#include <map>
#include <vector>

#include "geometry.h"
#include "model.h"
#include "renderbuffers.h"
#include "tgaimage.h"

struct TinyRenderObjectInstance {
  
  int m_mesh_uid;

  // Camera
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


  TinyRenderObjectInstance();
  virtual ~TinyRenderObjectInstance();

  void loadModel(const char* fileName, struct CommonFileIOInterface* fileIO);
  void createCube(float HalfExtentsX, float HalfExtentsY, float HalfExtentsZ,
                  struct CommonFileIOInterface* fileIO = 0);
  void registerMeshShape(const float* vertices, int numVertices,
                         const int* indices, int numIndices,
                         const float rgbaColor[4],
                         unsigned char* textureImage = 0, int textureWidth = 0,
                         int textureHeight = 0);

  void registerMesh2(std::vector<TinyRender::Vec3f>& vertices,
                     std::vector<TinyRender::Vec3f>& normals,
                     std::vector<int>& indices,
                     struct CommonFileIOInterface* fileIO);
};

class TinySceneRenderer {
  int m_guid;

  std::map<int, TinyRender::Model*> m_models;
  std::map<int, TinyRenderObjectInstance*> m_object_instances;

 public:
  TinySceneRenderer();

  virtual ~TinySceneRenderer();

  int create_mesh(const std::vector<double>& vertices,
                  const std::vector<double>& normals,
                  const std::vector<double>& uvs,
                  const std::vector<int>& indices,
                  const std::vector<unsigned char>& texture, int texture_width,
                  int texture_height, float texture_scaling);

  int create_cube(const std::vector<double>& half_extents,
                  const std::vector<unsigned char>& texture, int texture_width,
                  int texture_height, float texture_scaling);

  int create_capsule(float radius, float half_height, int up_axis,
                     const std::vector<unsigned char>& texture,
                     int texture_width, int texture_height);

  void set_object_position(int instance_uid,
                           const std::vector<float>& position);

  void set_object_orientation(int instance_uid,
                              const std::vector<float>& orientation);

  void set_object_color(int instance_uid, const std::vector<float>& color);

  void set_object_segmentation_uid(int instance_uid,
                                   int object_segmentation_uid);

  int get_object_segmentation_uid(int instance_uid) const;

  void set_object_double_sided(int instance_uid, bool double_sided);

  void set_object_local_scaling(int instance_uid,
                                const std::vector<float>& local_scaling);

  int create_object_instance(int model_uid);

  RenderBuffers get_camera_image_py(int width, int height,
                                    const std::vector<int>& objects,
                                    const std::vector<float>& viewMatrix,
                                    const std::vector<float>& projectionMatrix);

  void get_camera_image(const std::vector<int>& objects,
                        const std::vector<float>& viewMatrix,
                        const std::vector<float>& projectionMatrix,
                        RenderBuffers& buffers);

  
  void delete_mesh(int mesh_uid);

  void delete_instance(int instance_uid);

  void renderObject(int width, int height,
                           TinyRenderObjectInstance& object_instance,
                           struct RenderBuffers& render_buffers);

  static std::vector<float> compute_view_matrix(
      const std::vector<float>& cameraPosition,
      const std::vector<float>& cameraTargetPosition,
      const std::vector<float>& cameraUp);

  static std::vector<float> compute_projection_matrix(float left, float right,
                                                      float bottom, float top,
                                                      float nearVal,
                                                      float farVal);
};

#endif  // TDS_TINY_RENDERER_H
