#include <vector>
#include <limits>
#include <iostream>
#include "tgaimage.h"
#include "model.h"
#include "geometry.h"
#include "our_gl.h"
#include "tinyrenderer.h"
#define _USE_MATH_DEFINES
#include <math.h>

using namespace TinyRender;



int main(int argc, char** argv) {

    TinySceneRenderer scene;
    std::vector<unsigned char> texture = {  255,0,0,//red,
            0,255,0,//green
            0,0,255,//, blue
            255,255,255 //white
    };
    int texwidth = 2;
    int texheight = 2;
    auto capsulex_model = scene.create_capsule(0.1,0.5,0,texture, texwidth, texheight);
    auto capsuley_model = scene.create_capsule(0.1,0.5,1,texture, texwidth, texheight);
    auto capsulez_model = scene.create_capsule(0.1,0.5,2,texture, texwidth, texheight);
	
    std::vector<unsigned char> white_texture = {  255,255,255}; //white
    std::vector<double> half_extents = {100,100,0.5};
    auto cube_model = scene.create_cube(half_extents, white_texture, 1, 1, 16.);
	
	std::vector<double> vertices = {-0.500000, -0.500000, 0.500000,
			0.500000, -0.500000, 0.500000,
			-0.500000,	 0.500000, 0.500000,
		0.500000, 0.500000, 0.500000};
	std::vector<double> normals = {0.000000, 0.000000, 1.000000,
				0.000000, 0.000000, 1.000000,
				0.000000, 0.000000, 1.000000,
		0.000000, 0.000000, 1.000000};


	std::vector<double> uvs = {0.000000, 0.000000,
			1.000000, 0.000000,
			0.000000, 1.000000,
		1.000000, 1.000000};

	std::vector<int> indices = {0,1,2, 2,1,3};

	int plane_model = scene.create_mesh(vertices, normals, uvs, indices, texture, 1, 1, 16.);
	
    int view_width = 640;
    int view_height = 480;
    float nearPlane = 0.01;
    float farPlane = 100;
    float fov = 60;
    float aspect = float(view_width) / view_height;
    

    auto projMatrix = TinySceneRenderer::compute_projection_matrix_fov(fov, aspect, nearPlane, farPlane);

	float cameraTargetPosition[3] = {0, 0, 0};
	float cameraUp[3] = {0, 0, 1};
	float cameraPos[3] = {1, 1, 3};
	
	float pitch = -20.0;
	float yaw = 30;
	float roll = 0;
	int upAxisIndex = 2;
	float camDistance = 10;
	
	
	auto viewMatrix = TinySceneRenderer::compute_view_matrix_from_yaw_pitch_roll(cameraTargetPosition, camDistance,  yaw, pitch, roll, upAxisIndex);

    TinyRenderCamera camera(view_width, view_height, viewMatrix, projMatrix);

    TinyRenderLight light;
    light.m_distance = 10;

    //int capsulex_instance = scene.create_object_instance(capsulex_model);
    //int capsuley_instance = scene.create_object_instance(capsulex_model);
	int capsulez_instance = scene.create_object_instance(capsulez_model);
    int capsulez_instance2 = scene.create_object_instance(capsulez_model);
    
    std::vector<float> cap_pos2={0,-2,0};
    

    std::vector<float> cap_pos={0,0,0};
    scene.set_object_position(capsulez_instance, cap_pos);

    
    scene.set_object_position(capsulez_instance2, cap_pos2);
    //scene.set_object_position(capsulez_instance, cap_pos);

    int cube_instance = scene.create_object_instance(cube_model);
	std::vector<float> local_scaling={1,1,1};
	scene.set_object_local_scaling(cube_instance,local_scaling);


    std::vector<float> cube_pos={0,0,-1};
    scene.set_object_position(cube_instance, cube_pos);

	std::vector<int> instances = {capsulez_instance, capsulez_instance2,  cube_instance };
    //std::vector<int> instances = {cube_instance };

    RenderBuffers buffers(view_width, view_height);
    scene.get_camera_image(instances, light, camera, buffers);

    TGAImage img(view_width, view_height, 3);
    for (int w=0;w<view_width;w++)
    {
        for (int h=0;h<view_height;h++)
        {
            unsigned char red = buffers.rgb[3*(w+buffers.m_width*h)+0];
            unsigned char green = buffers.rgb[3*(w+buffers.m_width*h)+1];
            unsigned char blue = buffers.rgb[3*(w+buffers.m_width*h)+2];
            TGAColor rgb(red,green,blue);
            img.set(w,h,rgb);
        }
    }
    img.write_tga_file("tinyrenderer_cpp.tga");

    return 0;
}
