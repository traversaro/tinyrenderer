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

    std::vector<double> half_extents = {0.5,0.5,0.03};
    auto cube_model = scene.create_cube(half_extents, texture, texwidth, texheight, 16.);

        
    double NEAR_PLANE = 0.01;
    double FAR_PLANE = 1000;
    double HFOV = 58.0;
    double VFOV = 45.0;
    double left=-tan(M_PI*HFOV/360.0)*NEAR_PLANE;
    double right=tan(M_PI*HFOV/360.0)*NEAR_PLANE;
    double bottom=-tan(M_PI*VFOV/360.0)*NEAR_PLANE;
    double top=tan(M_PI*VFOV/360.0)*NEAR_PLANE;
    auto projection_matrix = TinySceneRenderer::compute_projection_matrix( left, right, bottom, top, NEAR_PLANE, FAR_PLANE);


    std::vector<float> up = {0.,0.,1.};
    std::vector<float> eye = {2.,4.,1.};
    std::vector<float> target = {0.,0.,0.};
    auto view_matrix = TinySceneRenderer::compute_view_matrix(eye, target, up);

    int capsulex_instance = scene.create_object_instance(capsulex_model);
    int capsuley_instance = scene.create_object_instance(capsuley_model);
    int capsulez_instance = scene.create_object_instance(capsulez_model);

    std::vector<int> instances = {capsulex_instance};
    RenderBuffers buffers(256,256);
    scene.get_camera_image(instances, view_matrix, projection_matrix, buffers);

    return 0;
}
