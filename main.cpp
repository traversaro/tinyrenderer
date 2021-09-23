#include <vector>
#include <limits>
#include <iostream>
#include "tgaimage.h"
#include "model.h"
#include "geometry.h"
#include "our_gl.h"
#include "TinyRenderer.h"

using namespace TinyRender;

int main(int argc, char** argv) {
    if (2>argc) {
        std::cerr << "Usage: " << argv[0] << " obj/model.obj" << std::endl;
        return 1;
    }

    int width=640;
    int height = 480;

    RenderBuffers render_buffers(width, height);
    
    //clear the color buffer
	TGAColor clearColor;
	clearColor.bgra[0] = 255;
	clearColor.bgra[1] = 255;
	clearColor.bgra[2] = 255;
	clearColor.bgra[3] = 255;

    float projM[16] = {1.0825316905975342, 0.0, 0.0, 0.0, 0.0, 1.7320507764816284, 0.0, 0.0, 0.0, 0.0, -1.0002000331878662, -1.0, 0.0, 0.0, -0.020002000033855438, 0.0};
    float viewM[16] = {0.8660255074501038, -0.1710100620985031, 0.469846248626709, 0.0, 0.4999999701976776, 0.29619815945625305, -0.8137977123260498, 0.0, 1.4901161193847656e-08, 0.9396926760673523, 0.3420201241970062, 0.0, -1.019299400439877e-08, -0.0, -2.0, 1.0};

    float nearPlane = projM[14] / (projM[10] - 1);
	float farPlane = projM[14] / (projM[10] + 1);


    for (int x=0;x<width;x++)
    {
        for (int y=0;y<height;y++)
        {
            render_buffers.rgb[3*(x+y*width)+0]= clearColor[0];
            render_buffers.rgb[3*(x+y*width)+1]= clearColor[1];
            render_buffers.rgb[3*(x+y*width)+2]= clearColor[2];
            render_buffers.zbuffer[x + y * width] = -farPlane;
        }
    }
    int bodyUniqueId = 2;
    int linkIndex= -1;

    for (int m=1; m<argc; m++) {
        TinyRenderObjectInstance* tinyObj = new TinyRenderObjectInstance();
        tinyObj->m_model = new Model(argv[m]);
        tinyObj->m_doubleSided = true;
        for (int i=0;i<4;i++)
        {
            TinyRender::Vec4f p;
            TinyRender::Vec4f v;
            for (int j=0;j<4;j++)
            {
                p[j] = projM[i*4+j];
                v[j] = viewM[i*4+j];
            }
            tinyObj->m_projectionMatrix.set_col(i,p);
            tinyObj->m_viewMatrix.set_col(i,v);
        }
        tinyObj->m_lightDirWorld.x = 0.5773502;
        tinyObj->m_lightDirWorld.y= 0.5773502;
        tinyObj->m_lightDirWorld.z = 0.5773502;

        tinyObj->m_lightDistance = 2;
        tinyObj->m_viewportMatrix = viewport(0, 0, width, height);
        float color[4] = {1,1,1,1};
        tinyObj->m_model->setColorRGBA(color);
        
        
        TinyRenderer::renderObject(width, height, *tinyObj,render_buffers);

        delete tinyObj->m_model;
    }

    //copy to TGAImage
    TGAImage img(width, height,TGAImage::RGB);
    for (int w=0;w<width;w++)
    {
        for (int h=0;h<height;h++)
        {
            unsigned red = render_buffers.rgb[3*(w+h*width)+0];
            unsigned green = render_buffers.rgb[3*(w+h*width)+1];
            unsigned blue = render_buffers.rgb[3*(w+h*width)+2];
            TGAColor c(red,green,blue,255);
            img.set(w,h,c);
        }
    }
    
    img.write_tga_file("framebuffer3.tga");

    return 0;
}
