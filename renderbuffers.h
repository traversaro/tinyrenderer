#ifndef RENDER_BUFFERS_H
#define RENDER_BUFFERS_H

#include <vector>

struct RenderBuffers {
  int m_width;
  int m_height;
  std::vector<unsigned char> rgb;
  std::vector<float> zbuffer;
  std::vector<int> segmentation_mask;

  RenderBuffers(int width, int height) : m_width(width), m_height(height) {
    rgb.resize(width * height * 3);  // red-green-blue
    zbuffer.resize(width * height);
    segmentation_mask.resize(width * height);
  }
};

#endif  // RENDER_BUFFERS_H
