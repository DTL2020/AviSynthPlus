// Avisynth v2.5.  Copyright 2002 Ben Rudiak-Gould et al.
// http://avisynth.nl

// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA, or visit
// http://www.gnu.org/copyleft/gpl.html .
//
// Linking Avisynth statically or dynamically with other modules is making a
// combined work based on Avisynth.  Thus, the terms and conditions of the GNU
// General Public License cover the whole combination.
//
// As a special exception, the copyright holders of Avisynth give you
// permission to link Avisynth with independent modules that communicate with
// Avisynth solely through the interfaces defined in avisynth.h, regardless of the license
// terms of these independent modules, and to copy and distribute the
// resulting combined work under terms of your choice, provided that
// every copy of the combined work is accompanied by a complete copy of
// the source code of Avisynth (the version of Avisynth used to produce the
// combined work), being distributed under the terms of the GNU General
// Public License plus this exception.  An independent module is a module
// which is not derived from or based on Avisynth, such as 3rd-party filters,
// import and export plugins, or graphical user interfaces.

#include <avs/config.h>
#include "../core/internal.h"

#include <avs/alignment.h>
#include <avs/minmax.h>

#include "resample_avx512.h"
//------- 512 bit float Horizontals

// Safe quad lane partial load with AVX512
// Read exactly N pixels (where N mod 4 is the template parameter), avoiding
// - reading beyond the end of the source buffer.
// - avoid NaN contamination by padding with zeros.
template <int Nmod4>
AVS_FORCEINLINE static __m512 _mm512_load_partial_safe_4_m128(const float* src_ptr_offsetted1, const float* src_ptr_offsetted2, const float* src_ptr_offsetted3, const float* src_ptr_offsetted4) {
  __m128 s1, s2, s3, s4;
  switch (Nmod4) {
  case 1:
    s1 = _mm_set_ps(0.0f, 0.0f, 0.0f, src_ptr_offsetted1[0]);
    s2 = _mm_set_ps(0.0f, 0.0f, 0.0f, src_ptr_offsetted2[0]);
    s3 = _mm_set_ps(0.0f, 0.0f, 0.0f, src_ptr_offsetted3[0]);
    s4 = _mm_set_ps(0.0f, 0.0f, 0.0f, src_ptr_offsetted4[0]);
    // ideally: movss
    break;
  case 2:
    s1 = _mm_set_ps(0.0f, 0.0f, src_ptr_offsetted1[1], src_ptr_offsetted1[0]);
    s2 = _mm_set_ps(0.0f, 0.0f, src_ptr_offsetted2[1], src_ptr_offsetted2[0]);
    s3 = _mm_set_ps(0.0f, 0.0f, src_ptr_offsetted3[1], src_ptr_offsetted3[0]);
    s4 = _mm_set_ps(0.0f, 0.0f, src_ptr_offsetted4[1], src_ptr_offsetted4[0]);
    // ideally: movsd
    break;
  case 3:
    s1 = _mm_set_ps(0.0f, src_ptr_offsetted1[2], src_ptr_offsetted1[1], src_ptr_offsetted1[0]);
    s2 = _mm_set_ps(0.0f, src_ptr_offsetted2[2], src_ptr_offsetted2[1], src_ptr_offsetted2[0]);
    s3 = _mm_set_ps(0.0f, src_ptr_offsetted3[2], src_ptr_offsetted3[1], src_ptr_offsetted3[0]);
    s4 = _mm_set_ps(0.0f, src_ptr_offsetted4[2], src_ptr_offsetted4[1], src_ptr_offsetted4[0]);
    // ideally: movss + movsd + shuffle or movsd + insert
    break;
  case 0:
    s1 = _mm_set_ps(src_ptr_offsetted1[3], src_ptr_offsetted1[2], src_ptr_offsetted1[1], src_ptr_offsetted1[0]);
    s2 = _mm_set_ps(src_ptr_offsetted2[3], src_ptr_offsetted2[2], src_ptr_offsetted2[1], src_ptr_offsetted2[0]);
    s3 = _mm_set_ps(src_ptr_offsetted3[3], src_ptr_offsetted3[2], src_ptr_offsetted3[1], src_ptr_offsetted3[0]);
    s4 = _mm_set_ps(src_ptr_offsetted4[3], src_ptr_offsetted4[2], src_ptr_offsetted4[1], src_ptr_offsetted4[0]);
    // ideally: movups
    break;
  default:
    s1 = _mm_setzero_ps(); // n/a cannot happen
    s2 = _mm_setzero_ps();
    s3 = _mm_setzero_ps();
    s4 = _mm_setzero_ps();
  }
  __m512 result = _mm512_castps128_ps512(s1); // Cast the first __m128 to __m512
  result = _mm512_insertf32x4(result, s2, 1); // Insert the second __m128 at position 1
  result = _mm512_insertf32x4(result, s3, 2); // Insert the third __m128 at position 2
  result = _mm512_insertf32x4(result, s4, 3); // Insert the fourth __m128 at position 3
  return result;
}




// Processes a horizontal resampling kernel of up to four coefficients for float pixel types.
// Supports BilinearResize, BicubicResize, or sinc with up to 2 taps (filter size <= 4).
// AVX512 optimization loads and processes four float coefficients and sixteen pixels simultaneously.
// The 'filtersizemod4' template parameter (0-3) helps optimize for different filter sizes modulo 4.
// This AVX512 requires only filter_size_alignment of 4.
template<int filtersizemod4>
void resize_h_planar_float_avx512_transpose_vstripe_ks4(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
  assert(filtersizemod4 >= 0 && filtersizemod4 <= 3);

  const int filter_size = program->filter_size; // aligned, practically the coeff table stride

  src_pitch /= sizeof(float);
  dst_pitch /= sizeof(float);

  float* src = (float*)src8;
  float* dst = (float*)dst8;

  const float* AVS_RESTRICT current_coeff = (const float* AVS_RESTRICT)program->pixel_coefficient_float;

  constexpr int PIXELS_AT_A_TIME = 16; // Process sixteen pixels in parallel using AVX512 (4x4 using m128 lanes)

  // 'source_overread_beyond_targetx' indicates if the filter kernel can read beyond the target width.
  // Even if the filter alignment allows larger reads, our safety boundary for unaligned loads starts at 4 pixels back
  // from the target width, as we load 4 floats at once conceptually with our safe load.
  const int width_safe_mod = (program->safelimit_4_pixels.overread_possible ? program->safelimit_4_pixels.source_overread_beyond_targetx : width) / PIXELS_AT_A_TIME * PIXELS_AT_A_TIME;

  // Preconditions:
  assert(program->filter_size_real <= 4); // We preload all relevant coefficients (up to 4) before the height loop.

  // 'target_size_alignment' ensures we can safely access coefficients using offsets like
  // 'filter_size * 7' when processing 8 H pixels at a time or
  // 'filter_size * 15' when processing 16 H pixels at a time
  assert(program->target_size_alignment >= 16); // Adjusted for 16 pixels
  assert(FRAME_ALIGN >= 64); // Adjusted for 16 pixels AviSynth+ default

  // Ensure that coefficient loading beyond the valid target size is safe for 4x4 float loads.
  assert(program->filter_size_alignment >= 4);

  int x = 0;

  // This 'auto' lambda construct replaces the need of templates
  auto do_h_float_core = [&](auto partial_load) {
    // Load up to 4x4 coefficients at once before the height loop.
    // Pre-loading and transposing coefficients keeps register usage efficient.
    // Assumes 'filter_size_aligned' is at least 4.

    // Coefficients for the source pixel offset (for src_ptr + begin1 [0..3], begin5 [0..3], begin9 [0..3], begin13 [0..3])
    __m512 coef_1_5_9_13 = _mm512_load_4_m128(current_coeff + filter_size * 0, current_coeff + filter_size * 4, current_coeff + filter_size * 8, current_coeff + filter_size * 12);
    __m512 coef_2_6_10_14 = _mm512_load_4_m128(current_coeff + filter_size * 1, current_coeff + filter_size * 5, current_coeff + filter_size * 9, current_coeff + filter_size * 13);
    __m512 coef_3_7_11_15 = _mm512_load_4_m128(current_coeff + filter_size * 2, current_coeff + filter_size * 6, current_coeff + filter_size * 10, current_coeff + filter_size * 14);
    __m512 coef_4_8_12_16 = _mm512_load_4_m128(current_coeff + filter_size * 3, current_coeff + filter_size * 7, current_coeff + filter_size * 11, current_coeff + filter_size * 15);

    _MM_TRANSPOSE16_LANE4_PS(coef_1_5_9_13, coef_2_6_10_14, coef_3_7_11_15, coef_4_8_12_16);

    float* AVS_RESTRICT dst_ptr = dst + x;
    const float* src_ptr = src;

    // Pixel offsets for the current target x-positions.
    // Even for x >= width, these offsets are guaranteed to be within the allocated 'target_size_alignment'.
    const int begin1 = program->pixel_offset[x + 0];
    const int begin2 = program->pixel_offset[x + 1];
    const int begin3 = program->pixel_offset[x + 2];
    const int begin4 = program->pixel_offset[x + 3];
    const int begin5 = program->pixel_offset[x + 4];
    const int begin6 = program->pixel_offset[x + 5];
    const int begin7 = program->pixel_offset[x + 6];
    const int begin8 = program->pixel_offset[x + 7];
    const int begin9 = program->pixel_offset[x + 8];
    const int begin10 = program->pixel_offset[x + 9];
    const int begin11 = program->pixel_offset[x + 10];
    const int begin12 = program->pixel_offset[x + 11];
    const int begin13 = program->pixel_offset[x + 12];
    const int begin14 = program->pixel_offset[x + 13];
    const int begin15 = program->pixel_offset[x + 14];
    const int begin16 = program->pixel_offset[x + 15];

    for (int y = 0; y < height; y++)
    {
      __m512 data_1_5_9_13;
      __m512 data_2_6_10_14;
      __m512 data_3_7_11_15;
      __m512 data_4_8_12_16;

      if constexpr (partial_load) {
        // In the potentially unsafe zone (near the right edge of the image), we use a safe loading function
        // to prevent reading beyond the allocated source scanline.

        data_1_5_9_13 = _mm512_load_partial_safe_4_m128<filtersizemod4>(src_ptr + begin1, src_ptr + begin5, src_ptr + begin9, src_ptr + begin13);
        data_2_6_10_14 = _mm512_load_partial_safe_4_m128<filtersizemod4>(src_ptr + begin2, src_ptr + begin6, src_ptr + begin10, src_ptr + begin14);
        data_3_7_11_15 = _mm512_load_partial_safe_4_m128<filtersizemod4>(src_ptr + begin3, src_ptr + begin7, src_ptr + begin11, src_ptr + begin15);
        data_4_8_12_16 = _mm512_load_partial_safe_4_m128<filtersizemod4>(src_ptr + begin4, src_ptr + begin8, src_ptr + begin12, src_ptr + begin16);
      }
      else {
        // In the safe zone, we can directly load 4 pixels at a time for each of the four lanes.
        data_1_5_9_13 = _mm512_loadu_4_m128(src_ptr + begin1, src_ptr + begin5, src_ptr + begin9, src_ptr + begin13);
        data_2_6_10_14 = _mm512_loadu_4_m128(src_ptr + begin2, src_ptr + begin6, src_ptr + begin10, src_ptr + begin14);
        data_3_7_11_15 = _mm512_loadu_4_m128(src_ptr + begin3, src_ptr + begin7, src_ptr + begin11, src_ptr + begin15);
        data_4_8_12_16 = _mm512_loadu_4_m128(src_ptr + begin4, src_ptr + begin8, src_ptr + begin12, src_ptr + begin16);
      }

      _MM_TRANSPOSE16_LANE4_PS(data_1_5_9_13, data_2_6_10_14, data_3_7_11_15, data_4_8_12_16);

      __m512 result = _mm512_mul_ps(data_1_5_9_13, coef_1_5_9_13);
      result = _mm512_fmadd_ps(data_2_6_10_14, coef_2_6_10_14, result);
      result = _mm512_fmadd_ps(data_3_7_11_15, coef_3_7_11_15, result);
      result = _mm512_fmadd_ps(data_4_8_12_16, coef_4_8_12_16, result);

      _mm512_store_ps(dst_ptr, result);

      dst_ptr += dst_pitch;
      src_ptr += src_pitch;
    } // y
    current_coeff += filter_size * 16; // Move to the next set of coefficients for the next 16 output pixels
    }; // end of lambda

  // Process the 'safe zone' where direct full unaligned loads are acceptable.
  for (; x < width_safe_mod; x += PIXELS_AT_A_TIME)
  {
    do_h_float_core(std::false_type{}); // partial_load == false, use direct _mm512_loadu_4_m128
  }

  // Process the potentially 'unsafe zone' near the image edge, using safe loading.
  for (; x < width; x += PIXELS_AT_A_TIME)
  {
    do_h_float_core(std::true_type{}); // partial_load == true, use the safer '_mm512_load_partial_safe_4_m128'
  }
}

// Instantiate them
template void resize_h_planar_float_avx512_transpose_vstripe_ks4<0>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resize_h_planar_float_avx512_transpose_vstripe_ks4<1>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resize_h_planar_float_avx512_transpose_vstripe_ks4<2>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resize_h_planar_float_avx512_transpose_vstripe_ks4<3>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);

/* Universal function supporting 2 ways of processing depending on the max offset of the source samples to read in the resampling program :
1. For high upsampling ratios it uses low read (single 8 float source samples) and permute-transpose before V-fma
2. For downsample and no-resize convolution - use each input sequence gathering by direct addressing
*/
template<int filtersizemod4>
void resize_h_planar_float_avx512_gather_permutex_vstripe_ks4(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel)
{
  assert(filtersizemod4 >= 0 && filtersizemod4 <= 3);

  const int filter_size = program->filter_size; // aligned, practically the coeff table stride

  src_pitch /= sizeof(float);
  dst_pitch /= sizeof(float);

  float* src = (float*)src8;
  float* dst = (float*)dst8;

  const float* AVS_RESTRICT current_coeff = (const float* AVS_RESTRICT)program->pixel_coefficient_float;

  constexpr int PIXELS_AT_A_TIME = 16; // Process sixteen pixels in parallel using AVX512 (4x4 using m128 lanes)

  // 'source_overread_beyond_targetx' indicates if the filter kernel can read beyond the target width.
  // Even if the filter alignment allows larger reads, our safety boundary for unaligned loads starts at 4 pixels back
  // from the target width, as we load 4 floats at once conceptually with our safe load.
  const int width_safe_mod = (program->safelimit_4_pixels.overread_possible ? program->safelimit_4_pixels.source_overread_beyond_targetx : width) / PIXELS_AT_A_TIME * PIXELS_AT_A_TIME;

  // Preconditions:
  assert(program->filter_size_real <= 4); // We preload all relevant coefficients (up to 4) before the height loop.

  // 'target_size_alignment' ensures we can safely access coefficients using offsets like
  // 'filter_size * 7' when processing 8 H pixels at a time or
  // 'filter_size * 15' when processing 16 H pixels at a time
  assert(program->target_size_alignment >= 16); // Adjusted for 16 pixels
  assert(FRAME_ALIGN >= 64); // Adjusted for 16 pixels AviSynth+ default

  // Ensure that coefficient loading beyond the valid target size is safe for 4x4 float loads.
  assert(program->filter_size_alignment >= 4);

  bool bDoGather = false;
  // Analyse input resampling program to select method of processing
  for (int x = 0; x < width - 16; x += 16) // -16 to save from vector overrread at program->pixel_offset[x + 15 + 3]; ?
  {
    int start_off = program->pixel_offset[x + 0];
    int end_off = program->pixel_offset[x + 15];
    if ((end_off - start_off) + (program->filter_size_real - 1) > 32) bDoGather = true;

    start_off = program->pixel_offset[x + 1];
    end_off = program->pixel_offset[x + 15 + 1];
    if ((end_off - start_off) + (program->filter_size_real - 1) > 32) bDoGather = true;

    start_off = program->pixel_offset[x + 2];
    end_off = program->pixel_offset[x + 15 + 2];
    if ((end_off - start_off) + (program->filter_size_real - 1) > 32) bDoGather = true;

    start_off = program->pixel_offset[x + 3];
    end_off = program->pixel_offset[x + 15 + 3];
    if ((end_off - start_off) + (program->filter_size_real - 1) > 32) bDoGather = true;
  }

  int x = 0;

  if (bDoGather)
  {
    // This 'auto' lambda construct replaces the need of templates
    auto do_h_float_core = [&](auto partial_load) {
      // Load up to 4x4 coefficients at once before the height loop.
      // Pre-loading and transposing coefficients keeps register usage efficient.
      // Assumes 'filter_size_aligned' is at least 4.

      // Coefficients for the source pixel offset (for src_ptr + begin1 [0..3], begin5 [0..3], begin9 [0..3], begin13 [0..3])
      __m512 coef_1_5_9_13 = _mm512_load_4_m128(current_coeff + filter_size * 0, current_coeff + filter_size * 4, current_coeff + filter_size * 8, current_coeff + filter_size * 12);
      __m512 coef_2_6_10_14 = _mm512_load_4_m128(current_coeff + filter_size * 1, current_coeff + filter_size * 5, current_coeff + filter_size * 9, current_coeff + filter_size * 13);
      __m512 coef_3_7_11_15 = _mm512_load_4_m128(current_coeff + filter_size * 2, current_coeff + filter_size * 6, current_coeff + filter_size * 10, current_coeff + filter_size * 14);
      __m512 coef_4_8_12_16 = _mm512_load_4_m128(current_coeff + filter_size * 3, current_coeff + filter_size * 7, current_coeff + filter_size * 11, current_coeff + filter_size * 15);

      _MM_TRANSPOSE16_LANE4_PS(coef_1_5_9_13, coef_2_6_10_14, coef_3_7_11_15, coef_4_8_12_16);

      float* AVS_RESTRICT dst_ptr = dst + x;
      const float* src_ptr = src;

      // Pixel offsets for the current target x-positions.
      // Even for x >= width, these offsets are guaranteed to be within the allocated 'target_size_alignment'.
      const int begin1 = program->pixel_offset[x + 0];
      const int begin2 = program->pixel_offset[x + 1];
      const int begin3 = program->pixel_offset[x + 2];
      const int begin4 = program->pixel_offset[x + 3];
      const int begin5 = program->pixel_offset[x + 4];
      const int begin6 = program->pixel_offset[x + 5];
      const int begin7 = program->pixel_offset[x + 6];
      const int begin8 = program->pixel_offset[x + 7];
      const int begin9 = program->pixel_offset[x + 8];
      const int begin10 = program->pixel_offset[x + 9];
      const int begin11 = program->pixel_offset[x + 10];
      const int begin12 = program->pixel_offset[x + 11];
      const int begin13 = program->pixel_offset[x + 12];
      const int begin14 = program->pixel_offset[x + 13];
      const int begin15 = program->pixel_offset[x + 14];
      const int begin16 = program->pixel_offset[x + 15];

      for (int y = 0; y < height; y++)
      {
        __m512 data_1_5_9_13;
        __m512 data_2_6_10_14;
        __m512 data_3_7_11_15;
        __m512 data_4_8_12_16;

        if constexpr (partial_load) {
          // In the potentially unsafe zone (near the right edge of the image), we use a safe loading function
          // to prevent reading beyond the allocated source scanline.

          data_1_5_9_13 = _mm512_load_partial_safe_4_m128<filtersizemod4>(src_ptr + begin1, src_ptr + begin5, src_ptr + begin9, src_ptr + begin13);
          data_2_6_10_14 = _mm512_load_partial_safe_4_m128<filtersizemod4>(src_ptr + begin2, src_ptr + begin6, src_ptr + begin10, src_ptr + begin14);
          data_3_7_11_15 = _mm512_load_partial_safe_4_m128<filtersizemod4>(src_ptr + begin3, src_ptr + begin7, src_ptr + begin11, src_ptr + begin15);
          data_4_8_12_16 = _mm512_load_partial_safe_4_m128<filtersizemod4>(src_ptr + begin4, src_ptr + begin8, src_ptr + begin12, src_ptr + begin16);
        }
        else {
          // In the safe zone, we can directly load 4 pixels at a time for each of the four lanes.
          data_1_5_9_13 = _mm512_loadu_4_m128(src_ptr + begin1, src_ptr + begin5, src_ptr + begin9, src_ptr + begin13);
          data_2_6_10_14 = _mm512_loadu_4_m128(src_ptr + begin2, src_ptr + begin6, src_ptr + begin10, src_ptr + begin14);
          data_3_7_11_15 = _mm512_loadu_4_m128(src_ptr + begin3, src_ptr + begin7, src_ptr + begin11, src_ptr + begin15);
          data_4_8_12_16 = _mm512_loadu_4_m128(src_ptr + begin4, src_ptr + begin8, src_ptr + begin12, src_ptr + begin16);
        }

        _MM_TRANSPOSE16_LANE4_PS(data_1_5_9_13, data_2_6_10_14, data_3_7_11_15, data_4_8_12_16);

        __m512 result = _mm512_mul_ps(data_1_5_9_13, coef_1_5_9_13);
        result = _mm512_fmadd_ps(data_2_6_10_14, coef_2_6_10_14, result);
        result = _mm512_fmadd_ps(data_3_7_11_15, coef_3_7_11_15, result);
        result = _mm512_fmadd_ps(data_4_8_12_16, coef_4_8_12_16, result);

        _mm512_store_ps(dst_ptr, result);

        dst_ptr += dst_pitch;
        src_ptr += src_pitch;
      } // y
      current_coeff += filter_size * 16; // Move to the next set of coefficients for the next 16 output pixels
    }; // end of lambda

  // Process the 'safe zone' where direct full unaligned loads are acceptable.
    for (; x < width_safe_mod; x += PIXELS_AT_A_TIME)
    {
      do_h_float_core(std::false_type{}); // partial_load == false, use direct _mm512_loadu_4_m128
    }

    // Process the potentially 'unsafe zone' near the image edge, using safe loading.
    for (; x < width; x += PIXELS_AT_A_TIME)
    {
      do_h_float_core(std::true_type{}); // partial_load == true, use the safer '_mm512_load_partial_safe_4_m128'
    }
  } 
  else // if(bDoGather)
  {
    for (int x = 0; x < width; x += 16)
    {
      // prepare coefs in transposed V-form
      __m512 coef_r0 = _mm512_load_4_m128(current_coeff + filter_size * 0, current_coeff + filter_size * 4, current_coeff + filter_size * 8, current_coeff + filter_size * 12);
      __m512 coef_r1 = _mm512_load_4_m128(current_coeff + filter_size * 1, current_coeff + filter_size * 5, current_coeff + filter_size * 9, current_coeff + filter_size * 13);
      __m512 coef_r2 = _mm512_load_4_m128(current_coeff + filter_size * 2, current_coeff + filter_size * 6, current_coeff + filter_size * 10, current_coeff + filter_size * 14);
      __m512 coef_r3 = _mm512_load_4_m128(current_coeff + filter_size * 3, current_coeff + filter_size * 7, current_coeff + filter_size * 11, current_coeff + filter_size * 15);

      _MM_TRANSPOSE16_LANE4_PS(coef_r0, coef_r1, coef_r2, coef_r3);

      // convert resampling program in H-form into permuting indexes for src transposition in V-form
      int iStart = program->pixel_offset[x + 0];

      __m512i perm_0 = _mm512_set_epi32(
        program->pixel_offset[x + 15] - iStart,
        program->pixel_offset[x + 14] - iStart,
        program->pixel_offset[x + 13] - iStart,
        program->pixel_offset[x + 12] - iStart,
        program->pixel_offset[x + 11] - iStart,
        program->pixel_offset[x + 10] - iStart,
        program->pixel_offset[x + 9] - iStart,
        program->pixel_offset[x + 8] - iStart,
        program->pixel_offset[x + 7] - iStart,
        program->pixel_offset[x + 6] - iStart,
        program->pixel_offset[x + 5] - iStart,
        program->pixel_offset[x + 4] - iStart,
        program->pixel_offset[x + 3] - iStart,
        program->pixel_offset[x + 2] - iStart,
        program->pixel_offset[x + 1] - iStart,
        0);

      __m512i one_epi32 = _mm512_set1_epi32(1);
      __m512i perm_1 = _mm512_add_epi32(perm_0, one_epi32);
      one_epi32 = _mm512_set1_epi32(program->pixel_offset[x + 2] - program->pixel_offset[x + 1]);
      __m512i perm_2 = _mm512_add_epi32(perm_1, one_epi32);
      one_epi32 = _mm512_set1_epi32(program->pixel_offset[x + 3] - program->pixel_offset[x + 2]);
      __m512i perm_3 = _mm512_add_epi32(perm_2, one_epi32);

      float* AVS_RESTRICT dst_ptr = dst + x;
      const float* src_ptr = src + program->pixel_offset[x + 0]; // all permute offsets relative to this start offset

      for (int y = 0; y < height; y++) // single row proc
      {
        __m512 data_src = _mm512_loadu_ps(src_ptr);
        __m512 data_src2 = _mm512_loadu_ps(src_ptr + 16); // not always needed for upscale also can cause end of buffer overread - need to add limitation (special end of buffer processing ?)

        __m512 data_0 = _mm512_permutex2var_ps(data_src, perm_0, data_src2);
        __m512 data_1 = _mm512_permutex2var_ps(data_src, perm_1, data_src2);
        __m512 data_2 = _mm512_permutex2var_ps(data_src, perm_2, data_src2);
        __m512 data_3 = _mm512_permutex2var_ps(data_src, perm_3, data_src2);

        __m512 result0 = _mm512_mul_ps(data_0, coef_r0);
        __m512 result1 = _mm512_mul_ps(data_2, coef_r2);

        result0 = _mm512_fmadd_ps(data_1, coef_r1, result0);
        result1 = _mm512_fmadd_ps(data_3, coef_r3, result1);

        _mm512_store_ps(dst_ptr, _mm512_add_ps(result0, result1));

        dst_ptr += dst_pitch;
        src_ptr += src_pitch;
      }

      current_coeff += filter_size * 16;
    }
  }
}

template void resize_h_planar_float_avx512_gather_permutex_vstripe_ks4<0>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resize_h_planar_float_avx512_gather_permutex_vstripe_ks4<1>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resize_h_planar_float_avx512_gather_permutex_vstripe_ks4<2>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
template void resize_h_planar_float_avx512_gather_permutex_vstripe_ks4<3>(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);


#if 0 // DTL version
// Transpose-based
// process kernel size from up to 4 - BilinearResize, BicubicResize or sinc up to taps=2
void resize_h_planar_float_avx512_transpose_vstripe_ks4(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel) {
    int filter_size = program->filter_size;

    const float* AVS_RESTRICT current_coeff;

    src_pitch = src_pitch / sizeof(float);
    dst_pitch = dst_pitch / sizeof(float);

    float* src = (float*)src8;
    float* dst = (float*)dst8;

    current_coeff = (const float* AVS_RESTRICT)program->pixel_coefficient_float;

    // this 16xfloat works, since AviSynth aligns scanlines to 64 bytes.
    for (int x = 0; x < width; x += 16) // is it safe to read by 16 floats = 64 bytes ?
    {
        __m512 c1_c5_c9_c13 = _mm512_load_4_m128(current_coeff + filter_size * 0, current_coeff + filter_size * 4, current_coeff + filter_size * 8, current_coeff + filter_size * 12);
        __m512 c2_c6_c10_c14 = _mm512_load_4_m128(current_coeff + filter_size * 1, current_coeff + filter_size * 5, current_coeff + filter_size * 9, current_coeff + filter_size * 13);
        __m512 c3_c7_c11_c15 = _mm512_load_4_m128(current_coeff + filter_size * 2, current_coeff + filter_size * 6, current_coeff + filter_size * 10, current_coeff + filter_size * 14);
        __m512 c4_c8_c12_c16 = _mm512_load_4_m128(current_coeff + filter_size * 3, current_coeff + filter_size * 7, current_coeff + filter_size * 11, current_coeff + filter_size * 15);

        _MM_TRANSPOSE16_LANE4_PS(c1_c5_c9_c13, c2_c6_c10_c14, c3_c7_c11_c15, c4_c8_c12_c16);

        float* AVS_RESTRICT dst_ptr = dst + x;
        const float* src_ptr = src;

        for (int y = 0; y < height; y++)
        {
            __m512 d1_d5_d9_d13 = _mm512_loadu_4_m128(src_ptr + program->pixel_offset[x + 0], src_ptr + program->pixel_offset[x + 4], src_ptr + program->pixel_offset[x + 8], src_ptr + program->pixel_offset[x + 12]);
            __m512 d2_d6_d10_d14 = _mm512_loadu_4_m128(src_ptr + program->pixel_offset[x + 1], src_ptr + program->pixel_offset[x + 5], src_ptr + program->pixel_offset[x + 9], src_ptr + program->pixel_offset[x + 13]);
            __m512 d3_d7_d11_d15 = _mm512_loadu_4_m128(src_ptr + program->pixel_offset[x + 2], src_ptr + program->pixel_offset[x + 6], src_ptr + program->pixel_offset[x + 10], src_ptr + program->pixel_offset[x + 14]);
            __m512 d4_d8_d12_d16 = _mm512_loadu_4_m128(src_ptr + program->pixel_offset[x + 3], src_ptr + program->pixel_offset[x + 7], src_ptr + program->pixel_offset[x + 11], src_ptr + program->pixel_offset[x + 15]);

            _MM_TRANSPOSE16_LANE4_PS(d1_d5_d9_d13, d2_d6_d10_d14, d3_d7_d11_d15, d4_d8_d12_d16);

            __m512 result = _mm512_mul_ps(d1_d5_d9_d13, c1_c5_c9_c13);
            result = _mm512_fmadd_ps(d2_d6_d10_d14, c2_c6_c10_c14, result);
            result = _mm512_fmadd_ps(d3_d7_d11_d15, c3_c7_c11_c15, result);
            result = _mm512_fmadd_ps(d4_d8_d12_d16, c4_c8_c12_c16, result);

            _mm512_store_ps(dst_ptr, result);

            dst_ptr += dst_pitch;
            src_ptr += src_pitch;
        }
        current_coeff += filter_size * 16;
    }

}
#endif

#if 0
void resize_h_planar_float_avx512_permutex_vstripe_ks4(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel)
{

  // assert - check if max pixel_offset is not above single load of 16 src floats (or need several loads and more complex permute program)

#ifdef _DEBUG
  for (int x = 0; x < width; x += 16)
  {
    int start_off = program->pixel_offset[x + 0];
    int end_off = program->pixel_offset[x + 15];
    assert((end_off - start_off) > 15);
  }
#endif

  int filter_size = program->filter_size;

  const float* AVS_RESTRICT current_coeff;
  __m512i one_epi32 = _mm512_set1_epi32(1);

  src_pitch = src_pitch / sizeof(float);
  dst_pitch = dst_pitch / sizeof(float);

  float* src = (float*)src8;
  float* dst = (float*)dst8;

  current_coeff = (const float* AVS_RESTRICT)program->pixel_coefficient_float;

  for (int x = 0; x < width; x += 16)
  {
    // prepare coefs in transposed V-form
    __m512 coef_r0 = _mm512_load_4_m128(current_coeff + filter_size * 0, current_coeff + filter_size * 4, current_coeff + filter_size * 8, current_coeff + filter_size * 12);
    __m512 coef_r1 = _mm512_load_4_m128(current_coeff + filter_size * 1, current_coeff + filter_size * 5, current_coeff + filter_size * 9, current_coeff + filter_size * 13);
    __m512 coef_r2 = _mm512_load_4_m128(current_coeff + filter_size * 2, current_coeff + filter_size * 6, current_coeff + filter_size * 10, current_coeff + filter_size * 14);
    __m512 coef_r3 = _mm512_load_4_m128(current_coeff + filter_size * 3, current_coeff + filter_size * 7, current_coeff + filter_size * 11, current_coeff + filter_size * 15);

    _MM_TRANSPOSE16_LANE4_PS(coef_r0, coef_r1, coef_r2, coef_r3);

    // convert resampling program in H-form into permuting indexes for src transposition in V-form
    int iStart = program->pixel_offset[x + 0];
    __m512i perm_0 = _mm512_set_epi32(program->pixel_offset[x + 15] - iStart, program->pixel_offset[x + 14] - iStart, program->pixel_offset[x + 13] - iStart, program->pixel_offset[x + 12] - iStart, program->pixel_offset[x + 11] - iStart, program->pixel_offset[x + 10] - iStart, program->pixel_offset[x + 9] - iStart, program->pixel_offset[x + 8] - iStart, \
      program->pixel_offset[x + 7] - iStart, program->pixel_offset[x + 6] - iStart, program->pixel_offset[x + 5] - iStart, program->pixel_offset[x + 4] - iStart, program->pixel_offset[x + 3] - iStart, program->pixel_offset[x + 2] - iStart, program->pixel_offset[x + 1] - iStart, 0);
    __m512i perm_1 = _mm512_add_epi32(perm_0, one_epi32);
    __m512i perm_2 = _mm512_add_epi32(perm_1, one_epi32);
    __m512i perm_3 = _mm512_add_epi32(perm_2, one_epi32);

    float* AVS_RESTRICT dst_ptr = dst + x;
    const float* src_ptr = src + program->pixel_offset[x + 0]; // all permute offsets relative to this start offset

#if 0
    for (int y = 0; y < height; y++) // single row proc
    {
      __m512 data_src = _mm512_loadu_ps(src_ptr);

      __m512 data_0 = _mm512_permutexvar_ps(perm_0, data_src);
      __m512 data_1 = _mm512_permutexvar_ps(perm_1, data_src);
      __m512 data_2 = _mm512_permutexvar_ps(perm_2, data_src);
      __m512 data_3 = _mm512_permutexvar_ps(perm_3, data_src);

      __m512 result0 = _mm512_mul_ps(data_0, coef_r0);
      __m512 result1 = _mm512_mul_ps(data_2, coef_r2);

      result0 = _mm512_fmadd_ps(data_1, coef_r1, result0);
      result1 = _mm512_fmadd_ps(data_3, coef_r3, result1);

      _mm512_store_ps(dst_ptr, _mm512_add_ps(result0, result1));

      dst_ptr += dst_pitch;
      src_ptr += src_pitch;
    }
#endif

    const int height_mod2 = (height / 2) * 2; // Process pairs of rows for better efficiency
    // dual-rows not worst in performance - may be left for the future better memory performance and compute performance hosts
    for (int y = 0; y < height_mod2; y+=2)
    {
      __m512 data_src = _mm512_loadu_ps(src_ptr);
      __m512 data_src_2 = _mm512_loadu_ps(src_ptr + src_pitch);

      __m512 data_0 = _mm512_permutexvar_ps(perm_0, data_src);
      __m512 data_1 = _mm512_permutexvar_ps(perm_1, data_src);
      __m512 data_2 = _mm512_permutexvar_ps(perm_2, data_src);
      __m512 data_3 = _mm512_permutexvar_ps(perm_3, data_src);

      __m512 data_0_2 = _mm512_permutexvar_ps(perm_0, data_src_2);
      __m512 data_1_2 = _mm512_permutexvar_ps(perm_1, data_src_2);
      __m512 data_2_2 = _mm512_permutexvar_ps(perm_2, data_src_2);
      __m512 data_3_2 = _mm512_permutexvar_ps(perm_3, data_src_2);

      __m512 result0 = _mm512_mul_ps(data_0, coef_r0);
      __m512 result1 = _mm512_mul_ps(data_0_2, coef_r0);

      result0 = _mm512_fmadd_ps(data_1, coef_r1, result0);
      result1 = _mm512_fmadd_ps(data_1_2, coef_r1, result1);

      result0 = _mm512_fmadd_ps(data_2, coef_r2, result0);
      result1 = _mm512_fmadd_ps(data_2_2, coef_r2, result1);

      result0 = _mm512_fmadd_ps(data_3, coef_r3, result0);
      result1 = _mm512_fmadd_ps(data_3_2, coef_r3, result1);

      _mm512_store_ps(dst_ptr, result0);
      _mm512_store_ps(dst_ptr + dst_pitch, result1);

      dst_ptr += dst_pitch * 2;
      src_ptr += src_pitch * 2;
    }

    if (height > height_mod2) // last row
    {
      __m512 data_src = _mm512_loadu_ps(src_ptr);

      __m512 data_0 = _mm512_permutexvar_ps(perm_0, data_src);
      __m512 data_1 = _mm512_permutexvar_ps(perm_1, data_src);
      __m512 data_2 = _mm512_permutexvar_ps(perm_2, data_src);
      __m512 data_3 = _mm512_permutexvar_ps(perm_3, data_src);

      __m512 result0 = _mm512_mul_ps(data_0, coef_r0);
      __m512 result1 = _mm512_mul_ps(data_2, coef_r2);

      result0 = _mm512_fmadd_ps(data_1, coef_r1, result0);
      result1 = _mm512_fmadd_ps(data_3, coef_r3, result1);

      _mm512_store_ps(dst_ptr, _mm512_add_ps(result0, result1));
    }

    current_coeff += filter_size * 16;
  }
}
#endif
void resize_h_planar_float_avx512_permutex_vstripe_ks8(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel)
{
  // assert - check if max pixel_offset is not above single load of 16 src floats (or need several loads and more complex permute program)

#ifdef _DEBUG
  for (int x = 0; x < width; x += 16)
  {
    int start_off = program->pixel_offset[x + 0];
    int end_off = program->pixel_offset[x + 15];
    assert((end_off - start_off) > 15);
  }
#endif

  int filter_size = program->filter_size;

  const float* AVS_RESTRICT current_coeff;
  __m512i one_epi32 = _mm512_set1_epi32(1);

  src_pitch = src_pitch / sizeof(float);
  dst_pitch = dst_pitch / sizeof(float);

  float* src = (float*)src8;
  float* dst = (float*)dst8;

  current_coeff = (const float* AVS_RESTRICT)program->pixel_coefficient_float;

  for (int x = 0; x < width; x += 16)
  {
    // prepare coefs in transposed V-form, use gathering - not very slow until TRANSPOSE8_ is designed

    __m512i offsets = _mm512_set_epi32(filter_size * 7, filter_size * 6, filter_size * 5, filter_size * 4, filter_size * 3, filter_size * 2, filter_size * 1, filter_size * 0, \
                                       filter_size * 7, filter_size * 6, filter_size * 5, filter_size * 4, filter_size * 3, filter_size * 2, filter_size * 1, filter_size * 0 );

    __m512 coef_r0 = _mm512_i32gather_ps(offsets, current_coeff, 4);

    offsets = _mm512_add_epi32(offsets, one_epi32);
    __m512 coef_r1 = _mm512_i32gather_ps(offsets, current_coeff, 4);

    offsets = _mm512_add_epi32(offsets, one_epi32);
    __m512 coef_r2 = _mm512_i32gather_ps(offsets, current_coeff, 4);

    offsets = _mm512_add_epi32(offsets, one_epi32);
    __m512 coef_r3 = _mm512_i32gather_ps(offsets, current_coeff, 4);

    offsets = _mm512_add_epi32(offsets, one_epi32);
    __m512 coef_r4 = _mm512_i32gather_ps(offsets, current_coeff, 4);

    offsets = _mm512_add_epi32(offsets, one_epi32);
    __m512 coef_r5 = _mm512_i32gather_ps(offsets, current_coeff, 4);

    offsets = _mm512_add_epi32(offsets, one_epi32);
    __m512 coef_r6 = _mm512_i32gather_ps(offsets, current_coeff, 4);

    offsets = _mm512_add_epi32(offsets, one_epi32);
    __m512 coef_r7 = _mm512_i32gather_ps(offsets, current_coeff, 4);


    // convert resampling program in H-form into permuting indexes for src transposition in V-form
    int iStart = program->pixel_offset[x + 0];
    __m512i perm_0 = _mm512_set_epi32(program->pixel_offset[x + 15] - iStart, program->pixel_offset[x + 14] - iStart, program->pixel_offset[x + 13] - iStart, program->pixel_offset[x + 12] - iStart, program->pixel_offset[x + 11] - iStart, program->pixel_offset[x + 10] - iStart, program->pixel_offset[x + 9] - iStart, program->pixel_offset[x + 8] - iStart, \
      program->pixel_offset[x + 7] - iStart, program->pixel_offset[x + 6] - iStart, program->pixel_offset[x + 5] - iStart, program->pixel_offset[x + 4] - iStart, program->pixel_offset[x + 3] - iStart, program->pixel_offset[x + 2] - iStart, program->pixel_offset[x + 1] - iStart, 0);
    __m512i perm_1 = _mm512_add_epi32(perm_0, one_epi32);
    __m512i perm_2 = _mm512_add_epi32(perm_1, one_epi32);
    __m512i perm_3 = _mm512_add_epi32(perm_2, one_epi32);
    __m512i perm_4 = _mm512_add_epi32(perm_3, one_epi32);
    __m512i perm_5 = _mm512_add_epi32(perm_4, one_epi32);
    __m512i perm_6 = _mm512_add_epi32(perm_5, one_epi32);
    __m512i perm_7 = _mm512_add_epi32(perm_6, one_epi32);

    float* AVS_RESTRICT dst_ptr = dst + x;
    const float* src_ptr = src + program->pixel_offset[x + 0]; // all permute offsets relative to this start offset
#if 0
    for (int y = 0; y < height; y++) // single row proc
    {
      __m512 data_src = _mm512_loadu_ps(src_ptr);

      __m512 data_0 = _mm512_permutexvar_ps(perm_0, data_src);
      __m512 data_1 = _mm512_permutexvar_ps(perm_1, data_src);
      __m512 data_2 = _mm512_permutexvar_ps(perm_2, data_src);
      __m512 data_3 = _mm512_permutexvar_ps(perm_3, data_src);
      __m512 data_4 = _mm512_permutexvar_ps(perm_4, data_src);
      __m512 data_5 = _mm512_permutexvar_ps(perm_5, data_src);
      __m512 data_6 = _mm512_permutexvar_ps(perm_6, data_src);
      __m512 data_7 = _mm512_permutexvar_ps(perm_7, data_src);

      __m512 result0 = _mm512_mul_ps(data_0, coef_r0);
      __m512 result1 = _mm512_mul_ps(data_4, coef_r4);

      result0 = _mm512_fmadd_ps(data_1, coef_r1, result0);
      result1 = _mm512_fmadd_ps(data_5, coef_r5, result1);

      result0 = _mm512_fmadd_ps(data_2, coef_r2, result0);
      result1 = _mm512_fmadd_ps(data_6, coef_r6, result1);

      result0 = _mm512_fmadd_ps(data_3, coef_r3, result0);
      result1 = _mm512_fmadd_ps(data_7, coef_r7, result1);

      _mm512_store_ps(dst_ptr, _mm512_add_ps(result0, result1));

      dst_ptr += dst_pitch;
      src_ptr += src_pitch;
    }
#endif

    const int height_mod2 = (height / 2) * 2; // Process pairs of rows for better efficiency
    // dual-rows not worst in performance - may be left for the future better memory performance and compute performance hosts
    for (int y = 0; y < height_mod2; y += 2)
    {
      __m512 data_src = _mm512_loadu_ps(src_ptr);
      __m512 data_src_2 = _mm512_loadu_ps(src_ptr + src_pitch);

      __m512 data_0 = _mm512_permutexvar_ps(perm_0, data_src);
      __m512 data_1 = _mm512_permutexvar_ps(perm_1, data_src);
      __m512 data_2 = _mm512_permutexvar_ps(perm_2, data_src);
      __m512 data_3 = _mm512_permutexvar_ps(perm_3, data_src);
      __m512 data_4 = _mm512_permutexvar_ps(perm_4, data_src);
      __m512 data_5 = _mm512_permutexvar_ps(perm_5, data_src);
      __m512 data_6 = _mm512_permutexvar_ps(perm_6, data_src);
      __m512 data_7 = _mm512_permutexvar_ps(perm_7, data_src);

      __m512 data_0_2 = _mm512_permutexvar_ps(perm_0, data_src_2);
      __m512 data_1_2 = _mm512_permutexvar_ps(perm_1, data_src_2);
      __m512 data_2_2 = _mm512_permutexvar_ps(perm_2, data_src_2);
      __m512 data_3_2 = _mm512_permutexvar_ps(perm_3, data_src_2);
      __m512 data_4_2 = _mm512_permutexvar_ps(perm_4, data_src_2);
      __m512 data_5_2 = _mm512_permutexvar_ps(perm_5, data_src_2);
      __m512 data_6_2 = _mm512_permutexvar_ps(perm_6, data_src_2);
      __m512 data_7_2 = _mm512_permutexvar_ps(perm_7, data_src_2);

      __m512 result0 = _mm512_mul_ps(data_0, coef_r0);
      __m512 result1 = _mm512_mul_ps(data_0_2, coef_r0);

      result0 = _mm512_fmadd_ps(data_1, coef_r1, result0);
      result1 = _mm512_fmadd_ps(data_1_2, coef_r1, result1);

      result0 = _mm512_fmadd_ps(data_2, coef_r2, result0);
      result1 = _mm512_fmadd_ps(data_2_2, coef_r2, result1);

      result0 = _mm512_fmadd_ps(data_3, coef_r3, result0);
      result1 = _mm512_fmadd_ps(data_3_2, coef_r3, result1);

      result0 = _mm512_fmadd_ps(data_4, coef_r4, result0);
      result1 = _mm512_fmadd_ps(data_4_2, coef_r4, result1);

      result0 = _mm512_fmadd_ps(data_5, coef_r5, result0);
      result1 = _mm512_fmadd_ps(data_5_2, coef_r5, result1);

      result0 = _mm512_fmadd_ps(data_6, coef_r6, result0);
      result1 = _mm512_fmadd_ps(data_6_2, coef_r6, result1);

      result0 = _mm512_fmadd_ps(data_7, coef_r7, result0);
      result1 = _mm512_fmadd_ps(data_7_2, coef_r7, result1);

      _mm512_store_ps(dst_ptr, result0);
      _mm512_store_ps(dst_ptr + dst_pitch, result1);

      dst_ptr += dst_pitch * 2;
      src_ptr += src_pitch * 2;
    }

    if (height > height_mod2) // last row
    {
      __m512 data_src = _mm512_loadu_ps(src_ptr);

      __m512 data_0 = _mm512_permutexvar_ps(perm_0, data_src);
      __m512 data_1 = _mm512_permutexvar_ps(perm_1, data_src);
      __m512 data_2 = _mm512_permutexvar_ps(perm_2, data_src);
      __m512 data_3 = _mm512_permutexvar_ps(perm_3, data_src);
      __m512 data_4 = _mm512_permutexvar_ps(perm_4, data_src);
      __m512 data_5 = _mm512_permutexvar_ps(perm_5, data_src);
      __m512 data_6 = _mm512_permutexvar_ps(perm_6, data_src);
      __m512 data_7 = _mm512_permutexvar_ps(perm_7, data_src);

      __m512 result0 = _mm512_mul_ps(data_0, coef_r0);
      __m512 result1 = _mm512_mul_ps(data_4, coef_r4);

      result0 = _mm512_fmadd_ps(data_1, coef_r1, result0);
      result1 = _mm512_fmadd_ps(data_5, coef_r5, result1);

      result0 = _mm512_fmadd_ps(data_2, coef_r2, result0);
      result1 = _mm512_fmadd_ps(data_6, coef_r6, result1);

      result0 = _mm512_fmadd_ps(data_3, coef_r3, result0);
      result1 = _mm512_fmadd_ps(data_7, coef_r7, result1);

      _mm512_store_ps(dst_ptr, _mm512_add_ps(result0, result1));
    }

    current_coeff += filter_size * 16;
  }
}

#if 0
void resize_h_planar_float_avx512_permutex_vstripe_ks16(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel)
{
  // assert - check if max pixel_offset is not above single load of 16 src floats (or need several loads and more complex permute program)
#ifdef _DEBUG
  for (int x = 0; x < width; x += 16)
  {
    int start_off = program->pixel_offset[x + 0];
    int end_off = program->pixel_offset[x + 15];
    assert((end_off - start_off) > 15);
  }
#endif

  int filter_size = program->filter_size;

  const float* AVS_RESTRICT current_coeff;
  __m512i one_epi32 = _mm512_set1_epi32(1);

  src_pitch = src_pitch / sizeof(float);
  dst_pitch = dst_pitch / sizeof(float);

  float* src = (float*)src8;
  float* dst = (float*)dst8;

  current_coeff = (const float* AVS_RESTRICT)program->pixel_coefficient_float;

  for (int x = 0; x < width; x += 16)
  {
    // prepare coefs in transposed V-form, use gathering - not very slow until TRANSPOSE8_ is designed

    __m512i offsets = _mm512_set_epi32(filter_size * 15, filter_size * 14, filter_size * 13, filter_size * 12, filter_size * 11, filter_size * 10, filter_size * 9, filter_size * 8, \
      filter_size * 7, filter_size * 6, filter_size * 5, filter_size * 4, filter_size * 3, filter_size * 2, filter_size * 1, filter_size * 0);

    __m512 coef_r0 = _mm512_i32gather_ps(offsets, current_coeff, 4);

    offsets = _mm512_add_epi32(offsets, one_epi32);
    __m512 coef_r1 = _mm512_i32gather_ps(offsets, current_coeff, 4);

    offsets = _mm512_add_epi32(offsets, one_epi32);
    __m512 coef_r2 = _mm512_i32gather_ps(offsets, current_coeff, 4);

    offsets = _mm512_add_epi32(offsets, one_epi32);
    __m512 coef_r3 = _mm512_i32gather_ps(offsets, current_coeff, 4);

    offsets = _mm512_add_epi32(offsets, one_epi32);
    __m512 coef_r4 = _mm512_i32gather_ps(offsets, current_coeff, 4);

    offsets = _mm512_add_epi32(offsets, one_epi32);
    __m512 coef_r5 = _mm512_i32gather_ps(offsets, current_coeff, 4);

    offsets = _mm512_add_epi32(offsets, one_epi32);
    __m512 coef_r6 = _mm512_i32gather_ps(offsets, current_coeff, 4);

    offsets = _mm512_add_epi32(offsets, one_epi32);
    __m512 coef_r7 = _mm512_i32gather_ps(offsets, current_coeff, 4);

    offsets = _mm512_add_epi32(offsets, one_epi32);
    __m512 coef_r8 = _mm512_i32gather_ps(offsets, current_coeff, 4);

    offsets = _mm512_add_epi32(offsets, one_epi32);
    __m512 coef_r9 = _mm512_i32gather_ps(offsets, current_coeff, 4);

    offsets = _mm512_add_epi32(offsets, one_epi32);
    __m512 coef_r10 = _mm512_i32gather_ps(offsets, current_coeff, 4);

    offsets = _mm512_add_epi32(offsets, one_epi32);
    __m512 coef_r11 = _mm512_i32gather_ps(offsets, current_coeff, 4);

    offsets = _mm512_add_epi32(offsets, one_epi32);
    __m512 coef_r12 = _mm512_i32gather_ps(offsets, current_coeff, 4);

    offsets = _mm512_add_epi32(offsets, one_epi32);
    __m512 coef_r13 = _mm512_i32gather_ps(offsets, current_coeff, 4);

    offsets = _mm512_add_epi32(offsets, one_epi32);
    __m512 coef_r14 = _mm512_i32gather_ps(offsets, current_coeff, 4);

    offsets = _mm512_add_epi32(offsets, one_epi32);
    __m512 coef_r15 = _mm512_i32gather_ps(offsets, current_coeff, 4);


    // convert resampling program in H-form into permuting indexes for src transposition in V-form
    int iStart = program->pixel_offset[x + 0];
    __m512i perm_0 = _mm512_set_epi32(program->pixel_offset[x + 15] - iStart, program->pixel_offset[x + 14] - iStart, program->pixel_offset[x + 13] - iStart, program->pixel_offset[x + 12] - iStart, program->pixel_offset[x + 11] - iStart, program->pixel_offset[x + 10] - iStart, program->pixel_offset[x + 9] - iStart, program->pixel_offset[x + 8] - iStart, \
      program->pixel_offset[x + 7] - iStart, program->pixel_offset[x + 6] - iStart, program->pixel_offset[x + 5] - iStart, program->pixel_offset[x + 4] - iStart, program->pixel_offset[x + 3] - iStart, program->pixel_offset[x + 2] - iStart, program->pixel_offset[x + 1] - iStart, 0);
    __m512i perm_1 = _mm512_add_epi32(perm_0, one_epi32);
    __m512i perm_2 = _mm512_add_epi32(perm_1, one_epi32);
    __m512i perm_3 = _mm512_add_epi32(perm_2, one_epi32);
    __m512i perm_4 = _mm512_add_epi32(perm_3, one_epi32);
    __m512i perm_5 = _mm512_add_epi32(perm_4, one_epi32);
    __m512i perm_6 = _mm512_add_epi32(perm_5, one_epi32);
    __m512i perm_7 = _mm512_add_epi32(perm_6, one_epi32);
    __m512i perm_8 = _mm512_add_epi32(perm_7, one_epi32);
    __m512i perm_9 = _mm512_add_epi32(perm_8, one_epi32);
    __m512i perm_10 = _mm512_add_epi32(perm_9, one_epi32);
    __m512i perm_11 = _mm512_add_epi32(perm_10, one_epi32);
    __m512i perm_12 = _mm512_add_epi32(perm_11, one_epi32);
    __m512i perm_13 = _mm512_add_epi32(perm_12, one_epi32);
    __m512i perm_14 = _mm512_add_epi32(perm_13, one_epi32);
    __m512i perm_15 = _mm512_add_epi32(perm_14, one_epi32); // to do : test if better to add one_epi32 in the loop and only store perm_0 complex to fill dataword

    float* AVS_RESTRICT dst_ptr = dst + x;
    const float* src_ptr = src + program->pixel_offset[x + 0]; // all permute offsets relative to this start offset

    for (int y = 0; y < height; y++) // single row proc
    {
      __m512 data_src = _mm512_loadu_ps(src_ptr);

      __m512 data_0 = _mm512_permutexvar_ps(perm_0, data_src);
      __m512 data_1 = _mm512_permutexvar_ps(perm_1, data_src);
      __m512 data_2 = _mm512_permutexvar_ps(perm_2, data_src);
      __m512 data_3 = _mm512_permutexvar_ps(perm_3, data_src);
      __m512 data_4 = _mm512_permutexvar_ps(perm_4, data_src);
      __m512 data_5 = _mm512_permutexvar_ps(perm_5, data_src);
      __m512 data_6 = _mm512_permutexvar_ps(perm_6, data_src);
      __m512 data_7 = _mm512_permutexvar_ps(perm_7, data_src);
      __m512 data_8 = _mm512_permutexvar_ps(perm_8, data_src);
      __m512 data_9 = _mm512_permutexvar_ps(perm_9, data_src);
      __m512 data_10 = _mm512_permutexvar_ps(perm_10, data_src);
      __m512 data_11 = _mm512_permutexvar_ps(perm_11, data_src);
      __m512 data_12 = _mm512_permutexvar_ps(perm_12, data_src);
      __m512 data_13 = _mm512_permutexvar_ps(perm_13, data_src);
      __m512 data_14 = _mm512_permutexvar_ps(perm_14, data_src);
      __m512 data_15 = _mm512_permutexvar_ps(perm_15, data_src);

      __m512 result0 = _mm512_mul_ps(data_0, coef_r0);
      __m512 result1 = _mm512_mul_ps(data_8, coef_r8);

      result0 = _mm512_fmadd_ps(data_1, coef_r1, result0);
      result1 = _mm512_fmadd_ps(data_9, coef_r9, result1);

      result0 = _mm512_fmadd_ps(data_2, coef_r2, result0);
      result1 = _mm512_fmadd_ps(data_10, coef_r10, result1);

      result0 = _mm512_fmadd_ps(data_3, coef_r3, result0);
      result1 = _mm512_fmadd_ps(data_11, coef_r11, result1);

      result0 = _mm512_fmadd_ps(data_4, coef_r4, result0);
      result1 = _mm512_fmadd_ps(data_12, coef_r12, result1);

      result0 = _mm512_fmadd_ps(data_5, coef_r5, result0);
      result1 = _mm512_fmadd_ps(data_13, coef_r13, result1);

      result0 = _mm512_fmadd_ps(data_6, coef_r6, result0);
      result1 = _mm512_fmadd_ps(data_14, coef_r14, result1);

      result0 = _mm512_fmadd_ps(data_7, coef_r7, result0);
      result1 = _mm512_fmadd_ps(data_15, coef_r15, result1);

      _mm512_store_ps(dst_ptr, _mm512_add_ps(result0, result1));

      dst_ptr += dst_pitch;
      src_ptr += src_pitch;
    }

    current_coeff += filter_size * 16;
  }
}
#endif



//-------- 512 bit float Verticals

void resize_v_avx512_planar_float(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel)
{
  AVS_UNUSED(bits_per_pixel);

  const int filter_size = program->filter_size;
  const float* AVS_RESTRICT current_coeff = program->pixel_coefficient_float;

  const float* src = (const float*)src8;
  float* AVS_RESTRICT dst = (float*)dst8;
  dst_pitch = dst_pitch / sizeof(float);
  src_pitch = src_pitch / sizeof(float);

  const int kernel_size = program->filter_size_real; // not the aligned
  const int kernel_size_mod2 = (kernel_size / 2) * 2; // Process pairs of rows for better efficiency
  const bool notMod2 = kernel_size_mod2 < kernel_size;

  for (int y = 0; y < target_height; y++) {
    int offset = program->pixel_offset[y];
    const float* src_ptr = src + offset * src_pitch;

    // 64 byte 16 floats (AVX512 register holds 16 floats)
    // no need for wmod8, alignment is safe 32 bytes at least - is it safe for 64 bytes ?
    for (int x = 0; x < width; x += 16) {
      __m512 result_single = _mm512_setzero_ps();
      __m512 result_single_2 = _mm512_setzero_ps();

      const float* AVS_RESTRICT src2_ptr = src_ptr + x; // __restrict here

      // Process pairs of rows for better efficiency (2 coeffs/cycle)
      // two result variables for potential parallel operation
      int i = 0;
      for (; i < kernel_size_mod2; i += 2) {
        __m512 coeff_even = _mm512_set1_ps(current_coeff[i]);
        __m512 coeff_odd = _mm512_set1_ps(current_coeff[i + 1]);

        __m512 src_even = _mm512_loadu_ps(src2_ptr);
        __m512 src_odd = _mm512_loadu_ps(src2_ptr + src_pitch);

        result_single = _mm512_fmadd_ps(src_even, coeff_even, result_single);
        result_single_2 = _mm512_fmadd_ps(src_odd, coeff_odd, result_single_2);

        src2_ptr += 2 * src_pitch;
      }

      result_single = _mm512_add_ps(result_single, result_single_2);

      // Process the last odd row if needed
      if (notMod2) {
        __m512 coeff = _mm512_set1_ps(current_coeff[i]);
        __m512 src_val = _mm512_loadu_ps(src2_ptr);
        result_single = _mm512_fmadd_ps(src_val, coeff, result_single);
      }

      _mm512_store_ps(dst + x, result_single);
    }

    dst += dst_pitch;
    current_coeff += filter_size;
  }
}

