import pytinyrenderer
import math

#only used for showing the image
import numpy as np
from numpngw import write_apng


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
capy_model = scene.create_capsule(0.1,0.4,1, texture.pixels, texture.width, texture.height)
capz_model = scene.create_capsule(0.1,0.4,2, texture.pixels, texture.width, texture.height)

cube_model = scene.create_cube([0.5,0.5,0.03], texture.pixels, texture.width, texture.height, 16.)
cube_instance = scene.create_object_instance(cube_model)
scene.set_object_position(cube_instance, [0,0,-0.5])

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
eye = [2.,4.,1.]
target = [0.,0.,0.]
view_matrix = pytinyrenderer.compute_view_matrix(eye, target, up)

capsulex_instance = scene.create_object_instance(capx_model)
capsuley_instance = scene.create_object_instance(capy_model)
capsulez_instance = scene.create_object_instance(capz_model)

images=[]

img = scene.get_camera_image(640,480,[capsulex_instance], view_matrix, projection_matrix)
rgb_array = np.reshape(np.array(img.rgb,dtype=np.uint8), (img.height, img.width, -1))
images.append(rgb_array)

img = scene.get_camera_image(640,480,[capsuley_instance], view_matrix, projection_matrix)
rgb_array = np.reshape(np.array(img.rgb,dtype=np.uint8), (img.height, img.width, -1))
images.append(rgb_array)

img = scene.get_camera_image(640,480,[capsulez_instance], view_matrix, projection_matrix)
rgb_array = np.reshape(np.array(img.rgb,dtype=np.uint8), (img.height, img.width, -1))
images.append(rgb_array)

img = scene.get_camera_image(640,480,[capsulex_instance,capsuley_instance,capsulez_instance,cube_instance], view_matrix, projection_matrix)
rgb_array = np.reshape(np.array(img.rgb,dtype=np.uint8), (img.height, img.width, -1))
images.append(rgb_array)

write_apng('tinyanim10_tds.png', images, delay=500)