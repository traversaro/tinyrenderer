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
    int view_width = 256;
    int view_height = 256;
    std::vector<float> up = {0., 0., 1.};
    std::vector<float> eye = {2., 4., 1.};
    std::vector<float> target = {0., 0., 0.};
    TinyRenderCamera camera(view_width, view_height, NEAR_PLANE, FAR_PLANE,
                            HFOV, VFOV, eye, target, up);
    TinyRenderLight light;

    int capsulex_instance = scene.create_object_instance(capsulex_model);
    int capsuley_instance = scene.create_object_instance(capsuley_model);
    int capsulez_instance = scene.create_object_instance(capsulez_model);

    std::vector<int> instances = {capsulex_instance};
    RenderBuffers buffers(view_width, view_height);
    scene.get_camera_image(instances, light, camera, buffers);

    return 0;
}
