import pytinyrenderer
import math

scene = pytinyrenderer.TinySceneRenderer()

class TextureRGB888:
  def __init__(self):
    self.pixels = [
            255,0,0,#red, green, blue
            0,255,0,
            0,0,255,
            255,255,255]
    self.width = 2
    self.height= 2
  

texture = TextureRGB888()

capx_model = scene.create_capsule(0.1,0.4,0, texture.pixels, texture.width, texture.height)
capy_model = scene.create_capsule(0.1,0.4,1, [255,0,0,0,255,0,0,0,255,255,255,255],2,2)
capz_model = scene.create_capsule(0.1,0.4,2, [255,0,0,0,255,0,0,0,255,255,255,255],2,2)

cube = scene.create_cube([0.5,0.5,0.03], texture.pixels, texture.width, texture.height, 16.)


NEAR_PLANE = 0.01
FAR_PLANE = 1000
HFOV = 58.0
VFOV = 45.0
left=-math.tan(math.pi*HFOV/360.0)*NEAR_PLANE
right=math.tan(math.pi*HFOV/360.0)*NEAR_PLANE
bottom=-math.tan(math.pi*VFOV/360.0)*NEAR_PLANE
top=math.tan(math.pi*VFOV/360.0)*NEAR_PLANE
projection_matrix = pytinyrenderer.compute_projection_matrix( left, right, bottom, top, NEAR_PLANE, FAR_PLANE)

up = [0.,0.,1.]
eye = [2.,-4.,11.]
target = [0.,0.,1.]
view_matrix = pytinyrenderer.compute_view_matrix(eye, target, up)

capsulex_instance = scene.create_object_instance(capx_model)
capsuley_instance = scene.create_object_instance(capy_model)
capsulez_instance = scene.create_object_instance(capz_model)

img = scene.get_camera_image(6,4,[capsulex_instance,capsuley_instance,capsulez_instance], view_matrix, projection_matrix)
print("img.rgb=",img.rgb)
print("done!")



