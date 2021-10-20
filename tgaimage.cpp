#include "tgaimage.h"

#include <math.h>
#include <string.h>
#include <time.h>

#include <fstream>
#include <iostream>

TGAImage2::TGAImage2() : data(NULL), width(0), height(0), bytespp(0) {}

TGAImage2::TGAImage2(int w, int h, int bpp)
    : data(NULL), width(w), height(h), bytespp(bpp) {
  unsigned long nbytes = width * height * bytespp;
  data = new unsigned char[nbytes];
  // memset(data, 0, nbytes);
}

TGAImage2::TGAImage2(const TGAImage2 &img)
    : data(NULL), width(img.width), height(img.height), bytespp(img.bytespp) {
  unsigned long nbytes = width * height * bytespp;
  data = new unsigned char[nbytes];
  memcpy(data, img.data, nbytes);
}

TGAImage2::~TGAImage2() {
  if (data) delete[] data;
}

TGAImage2 &TGAImage2::operator=(const TGAImage2 &img) {
  if (this != &img) {
    if (data) delete[] data;
    width = img.width;
    height = img.height;
    bytespp = img.bytespp;
    unsigned long nbytes = width * height * bytespp;
    data = new unsigned char[nbytes];
    memcpy(data, img.data, nbytes);
  }
  return *this;
}

bool TGAImage2::read_tga_file(const char *filename) {
  if (data) delete[] data;
  data = NULL;
  std::ifstream in;
  in.open(filename, std::ios::binary);
  if (!in.is_open()) {
    std::cerr << "can't open file " << filename << "\n";
    in.close();
    return false;
  }
  TGA_Header2 header;
  in.read((char *)&header, sizeof(header));
  if (!in.good()) {
    in.close();
    std::cerr << "an error occured while reading the header\n";
    return false;
  }
  width = header.width;
  height = header.height;
  bytespp = header.bitsperpixel >> 3;
  if (width <= 0 || height <= 0 ||
      (bytespp != GRAYSCALE && bytespp != RGB && bytespp != RGBA)) {
    in.close();
    std::cerr << "bad bpp (or width/height) value\n";
    return false;
  }
  unsigned long nbytes = bytespp * width * height;
  data = new unsigned char[nbytes];
  if (3 == header.datatypecode || 2 == header.datatypecode) {
    in.read((char *)data, nbytes);
    if (!in.good()) {
      in.close();
      std::cerr << "an error occured while reading the data\n";
      return false;
    }
  } else if (10 == header.datatypecode || 11 == header.datatypecode) {
    if (!load_rle_data(in)) {
      in.close();
      std::cerr << "an error occured while reading the data\n";
      return false;
    }
  } else {
    in.close();
    std::cerr << "unknown file format " << (int)header.datatypecode << "\n";
    return false;
  }
  if (!(header.imagedescriptor & 0x20)) {
    flip_vertically();
  }
  if (header.imagedescriptor & 0x10) {
    flip_horizontally();
  }
  std::cerr << width << "x" << height << "/" << bytespp * 8 << "\n";
  in.close();
  return true;
}

bool TGAImage2::load_rle_data(std::ifstream &in) {
  unsigned long pixelcount = width * height;
  unsigned long currentpixel = 0;
  unsigned long currentbyte = 0;
  TGAColor2 colorbuffer;
  do {
    unsigned char chunkheader = 0;
    chunkheader = in.get();
    if (!in.good()) {
      std::cerr << "an error occured while reading the data\n";
      return false;
    }
    if (chunkheader < 128) {
      chunkheader++;
      for (int i = 0; i < chunkheader; i++) {
        in.read((char *)colorbuffer.bgra, bytespp);
        if (!in.good()) {
          std::cerr << "an error occured while reading the header\n";
          return false;
        }
        for (int t = 0; t < bytespp; t++)
          data[currentbyte++] = colorbuffer.bgra[t];
        currentpixel++;
        if (currentpixel > pixelcount) {
          std::cerr << "Too many pixels read\n";
          return false;
        }
      }
    } else {
      chunkheader -= 127;
      in.read((char *)colorbuffer.bgra, bytespp);
      if (!in.good()) {
        std::cerr << "an error occured while reading the header\n";
        return false;
      }
      for (int i = 0; i < chunkheader; i++) {
        for (int t = 0; t < bytespp; t++)
          data[currentbyte++] = colorbuffer.bgra[t];
        currentpixel++;
        if (currentpixel > pixelcount) {
          std::cerr << "Too many pixels read\n";
          return false;
        }
      }
    }
  } while (currentpixel < pixelcount);
  return true;
}

bool TGAImage2::write_tga_file(const char *filename, bool rle) const {
  unsigned char developer_area_ref[4] = {0, 0, 0, 0};
  unsigned char extension_area_ref[4] = {0, 0, 0, 0};
  unsigned char footer[18] = {'T', 'R', 'U', 'E', 'V', 'I', 'S', 'I', 'O',
                              'N', '-', 'X', 'F', 'I', 'L', 'E', '.', '\0'};
  std::ofstream out;
  out.open(filename, std::ios::binary);
  if (!out.is_open()) {
    std::cerr << "can't open file " << filename << "\n";
    out.close();
    return false;
  }
  TGA_Header2 header;
  memset((void *)&header, 0, sizeof(header));
  header.bitsperpixel = bytespp << 3;
  header.width = width;
  header.height = height;
  header.datatypecode =
      (bytespp == GRAYSCALE ? (rle ? 11 : 3) : (rle ? 10 : 2));
  header.imagedescriptor = 0x20;  // top-left origin
  out.write((char *)&header, sizeof(header));
  if (!out.good()) {
    out.close();
    std::cerr << "can't dump the tga file\n";
    return false;
  }
  if (!rle) {
    out.write((char *)data, width * height * bytespp);
    if (!out.good()) {
      std::cerr << "can't unload raw data\n";
      out.close();
      return false;
    }
  } else {
    if (!unload_rle_data(out)) {
      out.close();
      std::cerr << "can't unload rle data\n";
      return false;
    }
  }
  out.write((char *)developer_area_ref, sizeof(developer_area_ref));
  if (!out.good()) {
    std::cerr << "can't dump the tga file\n";
    out.close();
    return false;
  }
  out.write((char *)extension_area_ref, sizeof(extension_area_ref));
  if (!out.good()) {
    std::cerr << "can't dump the tga file\n";
    out.close();
    return false;
  }
  out.write((char *)footer, sizeof(footer));
  if (!out.good()) {
    std::cerr << "can't dump the tga file\n";
    out.close();
    return false;
  }
  out.close();
  return true;
}

// TODO: it is not necessary to break a raw chunk for two equal pixels (for the
// matter of the resulting size)
bool TGAImage2::unload_rle_data(std::ofstream &out) const {
  const unsigned char max_chunk_length = 128;
  unsigned long npixels = width * height;
  unsigned long curpix = 0;
  while (curpix < npixels) {
    unsigned long chunkstart = curpix * bytespp;
    unsigned long curbyte = curpix * bytespp;
    unsigned char run_length = 1;
    bool raw = true;
    while (curpix + run_length < npixels && run_length < max_chunk_length) {
      bool succ_eq = true;
      for (int t = 0; succ_eq && t < bytespp; t++) {
        succ_eq = (data[curbyte + t] == data[curbyte + t + bytespp]);
      }
      curbyte += bytespp;
      if (1 == run_length) {
        raw = !succ_eq;
      }
      if (raw && succ_eq) {
        run_length--;
        break;
      }
      if (!raw && !succ_eq) {
        break;
      }
      run_length++;
    }
    curpix += run_length;
    out.put(raw ? run_length - 1 : run_length + 127);
    if (!out.good()) {
      std::cerr << "can't dump the tga file\n";
      return false;
    }
    out.write((char *)(data + chunkstart),
              (raw ? run_length * bytespp : bytespp));
    if (!out.good()) {
      std::cerr << "can't dump the tga file\n";
      return false;
    }
  }
  return true;
}

TGAColor2 TGAImage2::get(int x, int y) const {
  if (x < 0) {
    x = 0;
  }
  if (y < 0) {
    y = 0;
  }
  if (x >= width) {
    x = width - 1;
  }
  if (y >= height) {
    y = height - 1;
  }

  if (!data || x < 0 || y < 0 || x >= width || y >= height) {
    return TGAColor2(128.f, 128.f, 128.f, 255.f);
  }
  return TGAColor2(data + (x + y * width) * bytespp, bytespp);
}

bool TGAImage2::set(int x, int y, TGAColor2 &c) {
  if (!data || x < 0 || y < 0 || x >= width || y >= height) {
    return false;
  }
  memcpy(data + (x + y * width) * bytespp, c.bgra, bytespp);
  return true;
}

bool TGAImage2::set(int x, int y, const TGAColor2 &c) {
  if (!data || x < 0 || y < 0 || x >= width || y >= height) {
    return false;
  }
  memcpy(data + (x + y * width) * bytespp, c.bgra, bytespp);
  return true;
}

int TGAImage2::get_bytespp() const { return bytespp; }

int TGAImage2::get_width() const { return width; }

int TGAImage2::get_height() const { return height; }

bool TGAImage2::flip_horizontally() {
  if (!data) return false;
  int half = width >> 1;
  for (int i = 0; i < half; i++) {
    for (int j = 0; j < height; j++) {
      TGAColor2 c1 = get(i, j);
      TGAColor2 c2 = get(width - 1 - i, j);
      set(i, j, c2);
      set(width - 1 - i, j, c1);
    }
  }
  return true;
}

bool TGAImage2::flip_vertically() {
  if (!data) return false;
  unsigned long bytes_per_line = width * bytespp;
  unsigned char *line = new unsigned char[bytes_per_line];
  int half = height >> 1;
  for (int j = 0; j < half; j++) {
    unsigned long l1 = j * bytes_per_line;
    unsigned long l2 = (height - 1 - j) * bytes_per_line;
    memmove((void *)line, (void *)(data + l1), bytes_per_line);
    memmove((void *)(data + l1), (void *)(data + l2), bytes_per_line);
    memmove((void *)(data + l2), (void *)line, bytes_per_line);
  }
  delete[] line;
  return true;
}

unsigned char *TGAImage2::buffer() { return data; }

void TGAImage2::clear() { memset((void *)data, 0, width * height * bytespp); }

bool TGAImage2::scale(int w, int h) {
  if (w <= 0 || h <= 0 || !data) return false;
  unsigned char *tdata = new unsigned char[w * h * bytespp];
  int nscanline = 0;
  int oscanline = 0;
  int erry = 0;
  unsigned long nlinebytes = w * bytespp;
  unsigned long olinebytes = width * bytespp;
  for (int j = 0; j < height; j++) {
    int errx = width - w;
    int nx = -bytespp;
    int ox = -bytespp;
    for (int i = 0; i < width; i++) {
      ox += bytespp;
      errx += w;
      while (errx >= (int)width) {
        errx -= width;
        nx += bytespp;
        memcpy(tdata + nscanline + nx, data + oscanline + ox, bytespp);
      }
    }
    erry += h;
    oscanline += olinebytes;
    while (erry >= (int)height) {
      if (erry >= (int)height << 1)  // it means we jump over a scanline
        memcpy(tdata + nscanline + nlinebytes, tdata + nscanline, nlinebytes);
      erry -= height;
      nscanline += nlinebytes;
    }
  }
  delete[] data;
  data = tdata;
  width = w;
  height = h;
  return true;
}
