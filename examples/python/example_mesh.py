import pytinyrenderer
import numpy as np
from numpngw import write_apng
from PIL import Image

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
  
  def load(self, filename):
    im_frame = Image.open(filename).convert("RGB")
    np_frame = np.array(im_frame.getdata(), dtype=np.uint8)
    
    self.width = im_frame.width
    self.height = im_frame.height
    np_frame = np.reshape(np.array(np_frame), (self.height, self.width, -1))
    
    np_frame = np_frame[:, :, :3].flatten()
    
    self.pixels = np_frame.tolist()
    
  

texture = TextureRGB888()
texture.load("tex256.png")


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



vertices =np.array([[100.000000, -100.000000, 0.000000], 
            [100.000000, 100.000000, 0.000000],
            [-100.000000, 100.000000, 0.000000], 
            [-100.000000,  -100.000000, 0.000000]])#.flatten().tolist()
#vertices = vertices*0.01
normals = np.array([[0.000000, 0.000000, 1.000000], [0.000000, 0.000000, 1.000000],
           [0.000000, 0.000000, 1.000000], [0.000000, 0.000000, 1.000000]])#.flatten().tolist()

uvs = np.array([[1.000000, 0.000000], [1.000000, 1.000000],
 [0.000000, 1.000000], [0.000000, 0.000000]])#.flatten().tolist()

indices = [
    0,   1,    2,
    0,    2,    3]
    

plane_model = scene.create_mesh(vertices.flatten().tolist(), normals.flatten().tolist(), uvs.flatten().tolist(), indices, texture.pixels, texture.width, texture.height, 1.)

plane_instance = scene.create_object_instance(plane_model)
#scene.set_object_position(plane_instance,[-10,-2,-131])
scene.set_object_orientation(plane_instance,[0,0,0,1])
#scene.set_object_local_scaling(plane_instance,[0.01,0.01,0.01])

img = scene.get_camera_image(640,480,[plane_instance], view_matrix, projection_matrix)
rgb_array = np.reshape(np.array(img.rgb,dtype=np.uint8), (img.height, img.width, -1))

images=[]
images.append(rgb_array)

write_apng('tinyanim2_tds.png', images, delay=20)

