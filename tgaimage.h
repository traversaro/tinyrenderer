#ifndef __IMAGE_H__
#define __IMAGE_H__

#include <fstream>

#pragma pack(push, 1)
struct TGA_Header2 {
  char idlength;
  char colormaptype;
  char datatypecode;
  short colormaporigin;
  short colormaplength;
  char colormapdepth;
  short x_origin;
  short y_origin;
  short width;
  short height;
  char bitsperpixel;
  char imagedescriptor;
};
#pragma pack(pop)

struct TGAColor2 {
  unsigned char bgra[4];
  unsigned char bytespp;

  TGAColor2() : bytespp(1) {
    for (int i = 0; i < 4; i++) bgra[i] = 0;
  }

  TGAColor2(unsigned char R, unsigned char G, unsigned char B,
           unsigned char A = 255)
      : bytespp(4) {
    bgra[0] = B;
    bgra[1] = G;
    bgra[2] = R;
    bgra[3] = A;
  }

  TGAColor2(unsigned char v) : bytespp(1) {
    for (int i = 0; i < 4; i++) bgra[i] = 0;
    bgra[0] = v;
  }

  TGAColor2(const unsigned char *p, unsigned char bpp) : bytespp(bpp) {
    for (int i = 0; i < (int)bpp; i++) {
      bgra[i] = p[i];
    }
    for (int i = bpp; i < 4; i++) {
      bgra[i] = 0;
    }
  }

  unsigned char &operator[](const int i) { return bgra[i]; }

  TGAColor2 operator*(float intensity) const {
    TGAColor2 res = *this;
    intensity = (intensity > 1.f ? 1.f : (intensity < 0.f ? 0.f : intensity));
    for (int i = 0; i < 4; i++) res.bgra[i] = bgra[i] * intensity;
    return res;
  }
};

class TGAImage2 {
 protected:
  unsigned char *data;
  int width;
  int height;
  int bytespp;

  bool load_rle_data(std::ifstream &in);
  bool unload_rle_data(std::ofstream &out) const;

 public:
  enum Format { GRAYSCALE = 1, RGB = 3, RGBA = 4 };

  TGAImage2();
  TGAImage2(int w, int h, int bpp);
  TGAImage2(const TGAImage2 &img);
  bool read_tga_file(const char *filename);
  bool write_tga_file(const char *filename, bool rle = true) const;
  bool flip_horizontally();
  bool flip_vertically();
  bool scale(int w, int h);
  TGAColor2 get(int x, int y) const;

  bool set(int x, int y, TGAColor2 &c);
  bool set(int x, int y, const TGAColor2 &c);
  ~TGAImage2();
  TGAImage2 &operator=(const TGAImage2 &img);
  int get_width() const;
  int get_height() const;
  int get_bytespp() const;
  unsigned char *buffer();
  void clear();
};

#endif  //__IMAGE_H__
