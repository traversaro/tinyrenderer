#ifndef RENDER_BUFFERS_H
#define RENDER_BUFFERS_H

#include <vector>

struct RenderBuffers {
  int m_width;
  int m_height;
  std::vector<unsigned char> rgb;
  std::vector<float> zbuffer;
  std::vector<float> shadow_buffer;
  std::vector<int> shadow_uids;
  std::vector<int> segmentation_mask;

  RenderBuffers(int width, int height) {
    resize(width, height);
  }

  void resize(int width, int height) {
    m_width = width;
    m_height = height;
    rgb.resize(width * height * 3);  // red-green-blue
    zbuffer.resize(width * height);
    segmentation_mask.resize(width * height);
    shadow_buffer.resize(width * height);
    shadow_uids.resize(width * height);
  }
  void clear(float farPlane)
  {
    int width = m_width;
    int height = m_height;
    for (int x = 0; x < width; x++) {
    for (int y = 0; y < height; y++) {
        rgb[3 * (x + y * width) + 0] = 255;
        rgb[3 * (x + y * width) + 1] = 255;
        rgb[3 * (x + y * width) + 2] = 255;
        zbuffer[x + y * width] = -farPlane;
        shadow_buffer[x + y * width] = -1e30f;
        shadow_uids[x + y * width] = -1;
    }
    }
  }
};

#endif  // RENDER_BUFFERS_H
