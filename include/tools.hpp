#ifndef TOOLS_HPP_
#define TOOLS_HPP_

#include "typedata.hpp"

namespace deepRL {


/**
 * Convert pixel_t (NTSC) to RGB values.
 * Each value range [0,255]
 */
const std::array<int, 3> PixelToRGB(const pixel_t& pixel);

/**
 * Convert RGB values to a grayscale value [0,255].
 */
uint8_t RGBToGrayscale(const std::array<int, 3>& rgb);

uint8_t PixelToGrayscale(const pixel_t pixel);

FrameDataSp PreprocessScreen(const ALEScreen& raw_screen);

template <typename Dtype>
bool HasBlobSize(
    const caffe::Blob<Dtype>& blob,
    const int num,
    const int channels,
    const int height,
    const int width) {
  return blob.num() == num &&
      blob.channels() == channels &&
      blob.height() == height &&
      blob.width() == width;
}

}


#endif /* TOOLS_HPP_ */

