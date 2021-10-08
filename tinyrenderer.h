#ifndef TDS_TINY_RENDERER_H
#define TDS_TINY_RENDERER_H

#include <map>
#include <vector>

#include "geometry.h"
#include "model.h"
#include "renderbuffers.h"
#include "tgaimage.h"

struct TinyRenderCamera {
  TinyRender::Matrix m_viewMatrix;
  TinyRender::Matrix m_projectionMatrix;
  TinyRender::Matrix m_viewportMatrix;
  int m_viewWidth;
  int m_viewHeight;

  TinyRenderCamera(int viewWidth = 640, int viewHeight = 480, float near = 0.01,
                   float far = 1000, float hfov = 58, float vfov = 45,
                   const std::vector<float>& position = {1, 1, 1},
                   const std::vector<float>& target = {0, 0, 0},
                   const std::vector<float>& up = {0, 0, 1});

  TinyRenderCamera(int viewWidth, int viewHeight,
                   const std::vector<float>& viewMatrix,
                   const std::vector<float>& projectionMatrix);

  virtual ~TinyRenderCamera();
};

struct TinyRenderLight {
  TinyRender::Vec3f m_dirWorld;
  TinyRender::Vec3f m_color;
  TinyRender::Vec3f m_shadowmap_center;
  float m_distance;
  float m_ambientCoeff;
  float m_diffuseCoeff;
  float m_specularCoeff;
  bool m_has_shadow;
  float m_shadow_coefficient;

  TinyRenderLight(
      const std::vector<float>& direction = {0.57735, 0.57735, 0.57735},
    const std::vector<float>& color = {1, 1, 1}, 
    const std::vector<float>& shadowmap_center = {0,0,0},
    float distance = 10.0, float ambient = 0.6, float diffuse = 0.35, 
    float specular = 0.05, bool has_shadow = true, float shadow_coefficient=0.4);
    

  virtual ~TinyRenderLight();
};

struct TinyRenderObjectInstance {
  int m_mesh_uid;
  TinyRender::Vec3f m_localScaling;
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

  int create_object_instance(int mesh_uid);

  RenderBuffers get_camera_image_py(const std::vector<int>& objects,
                                    const TinyRenderLight& light,
                                    const TinyRenderCamera& camera);

  void get_camera_image(const std::vector<int>& objects,
                        const TinyRenderLight& light,
                        const TinyRenderCamera& camera,
                        RenderBuffers& buffers);


  void delete_mesh(int mesh_uid);

  void delete_instance(int instance_uid);

  void renderObject(const TinyRenderLight& light,
                    const TinyRenderCamera& camera,
                    const TinyRenderObjectInstance& object_instance,
                    struct RenderBuffers& render_buffers);

  void renderObjectDepth(  const TinyRenderLight& light,
                           const TinyRenderCamera& camera,
                           const TinyRenderObjectInstance& object_instance,
                           struct RenderBuffers& render_buffers);

  static std::vector<float> compute_view_matrix(
      const std::vector<float>& cameraPosition,
      const std::vector<float>& cameraTargetPosition,
      const std::vector<float>& cameraUp);

  static std::vector<float>  compute_view_matrix_from_yaw_pitch_roll(const float cameraTargetPosition[3], float distance, 
	float yaw, float pitch, float roll, int upAxis);

  static std::vector<float> compute_view_matrix_from_positions(const float cameraPosition[3], const float cameraTargetPosition[3], 
	const float cameraUp[3]);


  static std::vector<float> compute_projection_matrix(float hfov, float vfov,
                                                      float near, float far);
  
  static std::vector<float> compute_projection_matrix2(float left, float right, float bottom, float top, float nearVal, float farVal);
  static std::vector<float> compute_projection_matrix_fov(float fov, float aspect, float nearVal, float farVal);

  static TinyRender::Vec3f quatRotate(const TinyRender::Vec4f& rotation, const TinyRender::Vec3f& v);
  static TinyRender::Vec4f inverse(const TinyRender::Vec4f& rotation);
  static void setEulerZYX(	const float& yawZ, const float& pitchY, const float& rollX, TinyRender::Vec4f& euler);
  static TinyRender::Vec4f quatMul3(const TinyRender::Vec4f& q, const TinyRender::Vec3f& w);
  static TinyRender::Vec4f quatMul4(const TinyRender::Vec4f& q, const TinyRender::Vec4f& w);

};

#endif  // TDS_TINY_RENDERER_H
