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

// ConvertPlanar (c) 2005 by Klaus Post


#include <avs/alignment.h>
#ifdef _MSC_VER
    #include <intrin.h>
#else
    #include <x86intrin.h>
#endif
#include <immintrin.h>

#include "convert_planar_avx2.h"
#include "..\convert_helper.h"

#ifndef _mm256_set_m128i
#define _mm256_set_m128i(v0, v1) _mm256_insertf128_si256(_mm256_castsi128_si256(v1), (v0), 1)
#endif

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4309)
#endif

// packed rgb helper
static AVS_FORCEINLINE __m256i convert_yuv_to_rgb_avx2_core(const __m256i& px0189, const __m256i& px23AB, const __m256i& px45CD, const __m256i& px67EF, const __m256i& zero, const __m256i& matrix, const __m256i& round_mask_plus_rgb_offset) {
  //int b = (((int)m[0] * Y + (int)m[1] * U + (int)m[ 2] * V + 4096 + rgb_offset)>>13);

  //px01 - xx xx 00 V1 00 U1 00 Y1 xx xx 00 V0 00 U0 00 Y0

  auto low_lo = _mm256_madd_epi16(px0189, matrix); //xx*0 + v1*m2 | u1*m1 + y1*m0 | xx*0 + v0*m2 | u0*m1 + y0*m0
  auto low_hi = _mm256_madd_epi16(px23AB, matrix); //xx*0 + v3*m2 | u3*m1 + y3*m0 | xx*0 + v2*m2 | u2*m1 + y2*m0
  auto high_lo = _mm256_madd_epi16(px45CD, matrix);
  auto high_hi = _mm256_madd_epi16(px67EF, matrix);

  auto low_v = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(low_lo), _mm256_castsi256_ps(low_hi), _MM_SHUFFLE(3, 1, 3, 1))); // v3*m2 | v2*m2 | v1*m2 | v0*m2
  auto high_v = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(high_lo), _mm256_castsi256_ps(high_hi), _MM_SHUFFLE(3, 1, 3, 1)));

  auto low_yu = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(low_lo), _mm256_castsi256_ps(low_hi), _MM_SHUFFLE(2, 0, 2, 0))); // u3*m1 + y3*m0 | u2*m1 + y2*m0 | u1*m1 + y1*m0 | u0*m1 + y0*m0
  auto high_yu = _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(high_lo), _mm256_castsi256_ps(high_hi), _MM_SHUFFLE(2, 0, 2, 0)));

  auto t_lo = _mm256_add_epi32(low_v, low_yu); // v3*m2 + u3*m1 + y3*m0...
  auto t_hi = _mm256_add_epi32(high_v, high_yu);

  // v3*m2 + u3*m1 + y3*m0 + 4096 + rgb_offset
  t_lo = _mm256_add_epi32(t_lo, round_mask_plus_rgb_offset);
  t_hi = _mm256_add_epi32(t_hi, round_mask_plus_rgb_offset);

  t_lo = _mm256_srai_epi32(t_lo, 13); // (v3*m2 + u3*m1 + y3*m0 + 4096) >> 13...
  t_hi = _mm256_srai_epi32(t_hi, 13);

  auto result = _mm256_packs_epi32(t_lo, t_hi);
  result = _mm256_packus_epi16(result, zero); //00 00 00 00 00 00 00 00 b15 b14 b14 b12 b11 b10 b9 b8 00 00 00 00 00 00 00 00 b7 b6 b5 b4 b3 b2 b1 b0
  return result;
}

template<int rgb_pixel_step, bool hasAlpha>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("avx2")))
#endif
void convert_yv24_to_rgb_avx2(BYTE* dstp, const BYTE* srcY, const BYTE* srcU, const BYTE* srcV, const BYTE* srcA, size_t dst_pitch, size_t src_pitch_y, size_t src_pitch_uv, size_t src_pitch_a, size_t width, size_t height, const ConversionMatrix& matrix)
{
  dstp += dst_pitch * (height - 1);  // We start at last line

  size_t mod16_width = rgb_pixel_step == 3 ? width / 16 * 16 : width;
  // for rgb32 target we can process pixels beyond width, but we have 64bit alignment at target 16 pixels to 16*4=64 rgb pixels
  // if alignment would only be 32 bytes, we'd do the last cycle to process only 8 source pixels

  auto matrix_b = _mm256_set_epi16(0, matrix.v_b, matrix.u_b, matrix.y_b, 0, matrix.v_b, matrix.u_b, matrix.y_b, 0, matrix.v_b, matrix.u_b, matrix.y_b, 0, matrix.v_b, matrix.u_b, matrix.y_b);
  auto matrix_g = _mm256_set_epi16(0, matrix.v_g, matrix.u_g, matrix.y_g, 0, matrix.v_g, matrix.u_g, matrix.y_g, 0, matrix.v_g, matrix.u_g, matrix.y_g, 0, matrix.v_g, matrix.u_g, matrix.y_g);
  auto matrix_r = _mm256_set_epi16(0, matrix.v_r, matrix.u_r, matrix.y_r, 0, matrix.v_r, matrix.u_r, matrix.y_r, 0, matrix.v_r, matrix.u_r, matrix.y_r, 0, matrix.v_r, matrix.u_r, matrix.y_r);

  auto zero128 = _mm_setzero_si128();
  auto zero = _mm256_setzero_si256();

  // .13 bit frac integer arithmetic
  int round_mask_plus_rgb_offset_i = (1 << 12) + (matrix.offset_rgb << 13);
  auto round_mask_plus_rgb_offset = _mm256_set1_epi32(round_mask_plus_rgb_offset_i);
  auto offset = _mm256_set_epi16(0, -128, -128, matrix.offset_y, 0, -128, -128, matrix.offset_y, 0, -128, -128, matrix.offset_y, 0, -128, -128, matrix.offset_y);

  // 16 YUV(A) pixels --> 4x16 RGB quads = 64 bytes. Avisynth's alignment is 64 fortunately.

  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < mod16_width; x += 16) {
      __m128i src_y = _mm_load_si128(reinterpret_cast<const __m128i*>(srcY + x)); //Y15 .. Y7 Y6 Y5 Y4 Y3 Y2 Y1 Y0
      __m128i src_u = _mm_load_si128(reinterpret_cast<const __m128i*>(srcU + x)); //U15 .. U7 U6 U5 U4 U3 U2 U1 U0
      __m128i src_v = _mm_load_si128(reinterpret_cast<const __m128i*>(srcV + x)); //V15 .. V7 V6 V5 V4 V3 V2 V1 V0
      [[maybe_unused]] __m128i src_a;
      if constexpr(hasAlpha)
        src_a = _mm_load_si128(reinterpret_cast<const __m128i*>(srcA + x)); //A15 .. A7 A6 A5 A4 A3 A2 A1 A0

      __m128i t1_lo = _mm_unpacklo_epi8(src_y, src_u); //U7 Y7 U6 Y6 U5 Y5 U4 Y4 U3 Y3 U2 Y2 U1 Y1 U0 Y0
      __m128i t1_hi = _mm_unpackhi_epi8(src_y, src_u); //U15 Y15 U14 Y14 U13 Y13 U12 Y12 U11 Y11 U10 Y10 U9 Y9 U8 Y8
      __m128i t2_lo = _mm_unpacklo_epi8(src_v, zero128);  //00 V7 00 V6 00 V5 00 V4 00 V3 00 V2 00 V1 00 V0
      __m128i t2_hi = _mm_unpackhi_epi8(src_v, zero128);  //00 V15 00 V14 00 V13 00 V12 00 V11 00 V10 00 V9 00 V8

      __m256i t1 = _mm256_set_m128i(t1_hi, t1_lo);
      __m256i t2 = _mm256_set_m128i(t2_hi, t2_lo);
      t1 = _mm256_permute4x64_epi64(t1, (0 << 0) | (2 << 2) | (1 << 4) | (3 << 6));
      t2 = _mm256_permute4x64_epi64(t2, (0 << 0) | (2 << 2) | (1 << 4) | (3 << 6));

      __m256i low = _mm256_unpacklo_epi16(t1, t2); //xx V11 U11 Y11 xx V10 U10 Y10 xx V9 U9 Y9 xx V8 U8 Y8    xx V3 U3 Y3 xx V2 U2 Y2 xx V1 U1 Y1 xx V0 U0 Y0
      __m256i high = _mm256_unpackhi_epi16(t1, t2); //xx V15 U15 Y15 xx V14 U14 Y14 xx V13 U13 Y13 xx V12 U12 Y12    xx V7 U7 Y7 xx V6 U6 Y6 xx V5 U5 Y5 xx V4 U4 Y4

      __m256i px0189 = _mm256_unpacklo_epi8(low, zero);  //xx xx 00 V1 00 U1 00 Y1 xx xx 00 V0 00 U0 00 Y0
      __m256i px23AB = _mm256_unpackhi_epi8(low, zero);  //xx xx 00 V3 00 U3 00 Y3 xx xx 00 V2 00 U2 00 Y2
      __m256i px45CD = _mm256_unpacklo_epi8(high, zero); //xx xx 00 V5 00 U5 00 Y5 xx xx 00 V4 00 U4 00 Y4
      __m256i px67EF = _mm256_unpackhi_epi8(high, zero); //xx xx 00 V7 00 U7 00 Y7 xx xx 00 V6 00 U6 00 Y6

      px0189 = _mm256_add_epi16(px0189, offset);
      px23AB = _mm256_add_epi16(px23AB, offset);
      px45CD = _mm256_add_epi16(px45CD, offset);
      px67EF = _mm256_add_epi16(px67EF, offset);

      __m256i result_b = convert_yuv_to_rgb_avx2_core(px0189, px23AB, px45CD, px67EF, zero, matrix_b, round_mask_plus_rgb_offset); // b15..0
      __m256i result_g = convert_yuv_to_rgb_avx2_core(px0189, px23AB, px45CD, px67EF, zero, matrix_g, round_mask_plus_rgb_offset); // g15..0
      __m256i result_r = convert_yuv_to_rgb_avx2_core(px0189, px23AB, px45CD, px67EF, zero, matrix_r, round_mask_plus_rgb_offset); // r15..0

      __m256i result_bg = _mm256_unpacklo_epi8(result_b, result_g); //g15 b15 g14 b14 g13 b13 g12 b12 g11 b11 g10 b10 g9 b9 g8 b8 | g7 b7 g6 b6 g5 b5 g4 b4 g3 b3 g2 b2 g1 b1 g0 b0
      __m256i alpha;
      if constexpr(hasAlpha) {
        __m128i a_lo = _mm_unpacklo_epi8(src_a, zero128);  //00 A7 00 A6 00 A5 00 A4 00 A3 00 A2 00 A1 00 A0
        __m128i a_hi = _mm_unpackhi_epi8(src_a, zero128);  //00 A15 00 A14 00 A13 00 A12 00 A11 00 A10 00 A9 00 A8
        alpha = _mm256_set_m128i(a_hi, a_lo); // a15 .. a0  at low of each m128i part
        alpha = _mm256_permute4x64_epi64(alpha, (0 << 0) | (2 << 2) | (1 << 4) | (3 << 6));
      }
      else
        alpha = _mm256_cmpeq_epi32(result_r, result_r); // FF FF FF FF ... default alpha transparent

      __m256i result_ra = _mm256_unpacklo_epi8(result_r, alpha);       //a7 r7 a6 r6 a5 r5 a4 r4 a3 r3 a2 r2 a1 r1 a0 r0

      __m256i result_lo = _mm256_unpacklo_epi16(result_bg, result_ra); // a11 r11 g11 b11 | a10 r10 g10 b10 | a9 r9 g9 b9 | a8 r8 g8 b8  |   a3 r3 g3 b3 | a2 r2 g2 b2 | a1 r1 g1 b1 | a0 r0 g0 b0
      __m256i result_hi = _mm256_unpackhi_epi16(result_bg, result_ra);
      // note: actual pixel indexes are different from the numbers in comments, they are kept to be follow more easily
      // Initial _mm256_permute4x64_epi64 ensures that the order here is properly argb7..argb0 and argb15..argb8

      if constexpr (rgb_pixel_step == 4) {
        //rgb32
        _mm256_store_si256(reinterpret_cast<__m256i*>(dstp + x * 4), result_lo);
        _mm256_store_si256(reinterpret_cast<__m256i*>(dstp + x * 4 + 32), result_hi);
      }
      else {
        // rgb24
        // 16*4 bytes to 16*3 bytes
        __m256i perm10 = _mm256_set_epi32(0, 0, 6, 5, 4, 2, 1, 0);
        __m256i shuffle_lo = _mm256_set_epi8(
            0, 0, 0, 0, 14, 13, 12, 10, 9, 8, 6, 5, 4, 2, 1, 0,
            0, 0, 0, 0, 14, 13, 12, 10, 9, 8, 6, 5, 4, 2, 1, 0
        );
        __m128i shuffle_hi_lo = _mm_set_epi8(9, 8, 6, 5,  4,  2,  1,  0, 0, 0, 0, 0, 0, 0, 0, 0);

        __m128i result_hi_lo = _mm256_extracti128_si256(result_hi, 0); // aBbBgBrB aAbAgArA a9b9g9r9 a8b8g8r8
        __m128i result_hi_hi = _mm256_extracti128_si256(result_hi, 1); // aFbFgFrF aEbEgErE aDbDgDrD aCbCgCrC
        
        __m256i result_lo_reorg = _mm256_shuffle_epi8(result_lo, shuffle_lo);
        // x  x  x  x  b7g7r7 b6g6r6 b5g5r5 b4g4r4 x  x  x  x  b3g3r3 b2g2r2 b1g1r1 b0g0r0
        result_lo_reorg = _mm256_permutevar8x32_epi32(result_lo_reorg, perm10);
        // x  x  x  x  x x x x b7g7r7 b6g6r6 b5g5r5 b4g4r4 b3g3r3 b2g2r2 b1g1r1 b0g0r0

        __m128i result_hi_lo_reorg = _mm_shuffle_epi8(result_hi_lo, shuffle_hi_lo);
        // gA rA b9 g9 r9 b8 g8 r8 x x x x x x x x

        __m256i dummy_y0 = _mm256_undefined_si256();
        auto result_hi_lo_reorg_2 = _mm256_inserti128_si256(dummy_y0, result_hi_lo_reorg, 1);
        //                                                                      
        // gA rA b9 g9|r9 b8 g8 r8|x x x  x|x x  x x|x  x x x |x x x  x|x x  x x|x  x x x   // result_hi_lo_reorg_2
        // x  x  x  x  x  x  x  x  b7g7r7 b6g6r6 b5g5r5 b4g4r4 b3g3r3 b2g2r2 b1g1r1 b0g0r0  // result_lo_reorg
        auto result_0_15 = _mm256_blend_epi32(result_lo_reorg, result_hi_lo_reorg_2, 0xC0); // 11000000
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dstp + x * 3), result_0_15); // not necessarily 32 bytes aligned

        // => x  x  x  x  x  x  x  x  x  x  x  x  bB gB rB bA
        __m128i shuffle_hi_lo_2 = _mm_set_epi8(0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 14, 13, 12, 10);
        __m128i xmm4 = _mm_shuffle_epi8(result_hi_lo, shuffle_hi_lo_2);

        // => bF gF rF bE gE rE bD gD rD bC gC rC x  x  x  x
        __m128i shuffle_hi_hi = _mm_set_epi8(14, 13, 12, 10, 9, 8, 6, 5, 4, 2, 1, 0, 0x80, 0x80, 0x80, 0x80);
        __m128i xmm5 = _mm_shuffle_epi8(result_hi_hi, shuffle_hi_hi);

        // => bF gF rF bE gE rE bD gD rD bC gC rC bB gB rB bA
        __m128i result_16_23 = _mm_or_si128(xmm4, xmm5);
        _mm_store_si128(reinterpret_cast<__m128i*>(dstp + x * 3 + 32), result_16_23);
#if 0
        // Intel compiler can cope with it 100% optimized. No store, just shuffles and blends as above.
        alignas(32) BYTE temp[64];
        _mm256_store_si256(reinterpret_cast<__m256i*>(temp), result_lo);
        _mm256_store_si256(reinterpret_cast<__m256i*>(temp + 32), result_hi);
        for (int i = 0; i < 16; ++i) {
          dstp[(x + i) * 3 + 0] = temp[i * 4 + 0];
          dstp[(x + i) * 3 + 1] = temp[i * 4 + 1];
          dstp[(x + i) * 3 + 2] = temp[i * 4 + 2];
        }
#endif
      }
    }

    if constexpr (rgb_pixel_step == 3) {
      // for rgb32 (pixel_step == 4) we processed full width and more, including padded bytes
      for (size_t x = mod16_width; x < width; ++x) {
        int Y = srcY[x] + matrix.offset_y;
        int U = srcU[x] - 128;
        int V = srcV[x] - 128;
        int b = (((int)matrix.y_b * Y + (int)matrix.u_b * U + (int)matrix.v_b * V + round_mask_plus_rgb_offset_i) >> 13);
        int g = (((int)matrix.y_g * Y + (int)matrix.u_g * U + (int)matrix.v_g * V + round_mask_plus_rgb_offset_i) >> 13);
        int r = (((int)matrix.y_r * Y + (int)matrix.u_r * U + (int)matrix.v_r * V + round_mask_plus_rgb_offset_i) >> 13);
        dstp[x * rgb_pixel_step + 0] = PixelClip(b);
        dstp[x * rgb_pixel_step + 1] = PixelClip(g);
        dstp[x * rgb_pixel_step + 2] = PixelClip(r);
        if constexpr (rgb_pixel_step == 4) { // n/a
          dstp[x * 4 + 3] = 255;
        }
      }
    }
    dstp -= dst_pitch;
    srcY += src_pitch_y;
    srcU += src_pitch_uv;
    srcV += src_pitch_uv;
    if constexpr(hasAlpha)
      srcA += src_pitch_a;
  }
}

//instantiate
//template<int rgb_pixel_step, bool hasAlpha>
template void convert_yv24_to_rgb_avx2<3, false>(BYTE* dstp, const BYTE* srcY, const BYTE* srcU, const BYTE* srcV, const BYTE* srcA, size_t dst_pitch, size_t src_pitch_y, size_t src_pitch_uv, size_t src_pitch_a, size_t width, size_t height, const ConversionMatrix& matrix);
template void convert_yv24_to_rgb_avx2<4, false>(BYTE* dstp, const BYTE* srcY, const BYTE* srcU, const BYTE* srcV, const BYTE* srcA, size_t dst_pitch, size_t src_pitch_y, size_t src_pitch_uv, size_t src_pitch_a, size_t width, size_t height, const ConversionMatrix& matrix);
template void convert_yv24_to_rgb_avx2<3, true>(BYTE* dstp, const BYTE* srcY, const BYTE* srcU, const BYTE* srcV, const BYTE* srcA, size_t dst_pitch, size_t src_pitch_y, size_t src_pitch_uv, size_t src_pitch_a, size_t width, size_t height, const ConversionMatrix& matrix);
template void convert_yv24_to_rgb_avx2<4, true>(BYTE* dstp, const BYTE* srcY, const BYTE* srcU, const BYTE* srcV, const BYTE* srcA, size_t dst_pitch, size_t src_pitch_y, size_t src_pitch_uv, size_t src_pitch_a, size_t width, size_t height, const ConversionMatrix& matrix);


template<int bits_per_pixel>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("avx2")))
#endif
void convert_planarrgb_to_yuv_uint16_avx2(BYTE* (&dstp)[3], int(&dstPitch)[3], const BYTE* (&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix& m)
{
  // generic for 10-16 bit uint16 
  // originally made only for 16 bits where unsigned 16 arithmetic makes things difficult

  __m256  half_f = _mm256_set1_ps((float)(1u << (bits_per_pixel - 1)));
  __m128i limit = _mm_set1_epi16((short)((1 << bits_per_pixel) - 1)); // 255
  __m256 offset_f = _mm256_set1_ps(m.offset_y_f);
  __m256i offset_rgb = _mm256_set1_epi32(m.offset_rgb);

  const bool has_offset_rgb = 0 != m.offset_rgb_f;

  //__m128i zero = _mm_setzero_si128();

  const int rowsize = width * sizeof(uint16_t);
  for (int yy = 0; yy < height; yy++) {
    for (int x = 0; x < rowsize; x += 8 * sizeof(uint16_t)) {
      __m256 g, b, r;
      // uint16_t: load 16 bytes: 8 pixels

      __m128i gi = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp[0] + x));
      __m128i bi = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp[1] + x));
      __m128i ri = _mm_load_si128(reinterpret_cast<const __m128i*>(srcp[2] + x));
      
      __m256i gi32 = _mm256_cvtepu16_epi32(gi);
      __m256i bi32 = _mm256_cvtepu16_epi32(bi);
      __m256i ri32 = _mm256_cvtepu16_epi32(ri);
      if (has_offset_rgb) {
        bi32 = _mm256_add_epi32(bi32, offset_rgb);
        gi32 = _mm256_add_epi32(gi32, offset_rgb);
        ri32 = _mm256_add_epi32(ri32, offset_rgb);
      }
      g = _mm256_cvtepi32_ps(gi32);
      b = _mm256_cvtepi32_ps(bi32);
      r = _mm256_cvtepi32_ps(ri32);
      /*
      int Y = m.offset_y + (int)(((sum_t)m.y_b * b + (sum_t)m.y_g * g + (sum_t)m.y_r * r + 16384)>>15);
      int U = half + (int)(((sum_t)m.u_b * b + (sum_t)m.u_g * g + (sum_t)m.u_r * r + 16384) >> 15);
      int V = half + (int)(((sum_t)m.v_b * b + (sum_t)m.v_g * g + (sum_t)m.v_r * r + 16384) >> 15);
      */
      // *Y*
      {
        auto mat_r = _mm256_set1_ps(m.y_r_f);
        auto mat_g = _mm256_set1_ps(m.y_g_f);
        auto mat_b = _mm256_set1_ps(m.y_b_f);
        __m256 y = _mm256_fmadd_ps(r, mat_r, _mm256_fmadd_ps(g, mat_g, _mm256_fmadd_ps(b, mat_b, offset_f)));
        __m256i yi = _mm256_cvtps_epi32(y); // no extra rounding, cvtps rounds to nearest
        yi = _mm256_packus_epi32(yi, _mm256_setzero_si256()); // 16x uint16_t
        __m128i res = _mm256_castsi256_si128(_mm256_permute4x64_epi64(yi, 0xD8));
        if constexpr (bits_per_pixel < 16) // albeit 10-14 bit have another function, make this general
          res = _mm_min_epi16(res, limit); // clamp 10,12,14 bit
        _mm_store_si128(reinterpret_cast<__m128i*>(dstp[0] + x), res);
      }
      // *U*
      {
        auto mat_r = _mm256_set1_ps(m.u_r_f);
        auto mat_g = _mm256_set1_ps(m.u_g_f);
        auto mat_b = _mm256_set1_ps(m.u_b_f);
        __m256 y = _mm256_fmadd_ps(r, mat_r, _mm256_fmadd_ps(g, mat_g, _mm256_fmadd_ps(b, mat_b, half_f)));
        __m256i yi = _mm256_cvtps_epi32(y); // no extra rounding, cvtps rounds to nearest
        yi = _mm256_packus_epi32(yi, _mm256_setzero_si256()); // 16x uint16_t
        __m128i res = _mm256_castsi256_si128(_mm256_permute4x64_epi64(yi, 0xD8));
        if constexpr (bits_per_pixel < 16) // albeit 10-14 bit have another function, make this general
          res = _mm_min_epi16(res, limit); // clamp 10,12,14 bit
        _mm_store_si128(reinterpret_cast<__m128i*>(dstp[1] + x), res);
      }
      // *V*
      {
        auto mat_r = _mm256_set1_ps(m.v_r_f);
        auto mat_g = _mm256_set1_ps(m.v_g_f);
        auto mat_b = _mm256_set1_ps(m.v_b_f);
        __m256 y = _mm256_fmadd_ps(r, mat_r, _mm256_fmadd_ps(g, mat_g, _mm256_fmadd_ps(b, mat_b, half_f)));
        __m256i yi = _mm256_cvtps_epi32(y); // no extra rounding, cvtps rounds to nearest
        yi = _mm256_packus_epi32(yi, _mm256_setzero_si256()); // 16x uint16_t
        __m128i res = _mm256_castsi256_si128(_mm256_permute4x64_epi64(yi, 0xD8));
        if constexpr (bits_per_pixel < 16) // albeit 10-14 bit have another function, make this general
          res = _mm_min_epi16(res, limit); // clamp 10,12,14 bit
        _mm_store_si128(reinterpret_cast<__m128i*>(dstp[2] + x), res);
      }
    }
    srcp[0] += srcPitch[0];
    srcp[1] += srcPitch[1];
    srcp[2] += srcPitch[2];
    dstp[0] += dstPitch[0];
    dstp[1] += dstPitch[1];
    dstp[2] += dstPitch[2];
  }
}

// Instantiate them
template void convert_planarrgb_to_yuv_uint16_avx2<10>(BYTE* (&dstp)[3], int(&dstPitch)[3], const BYTE* (&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix& m);
template void convert_planarrgb_to_yuv_uint16_avx2<12>(BYTE* (&dstp)[3], int(&dstPitch)[3], const BYTE* (&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix& m);
template void convert_planarrgb_to_yuv_uint16_avx2<14>(BYTE* (&dstp)[3], int(&dstPitch)[3], const BYTE* (&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix& m);
template void convert_planarrgb_to_yuv_uint16_avx2<16>(BYTE* (&dstp)[3], int(&dstPitch)[3], const BYTE* (&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix& m);

template<typename pixel_t, int bits_per_pixel>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("avx2")))
#endif
void convert_yuv_to_planarrgb_uint8_14_avx2(BYTE *(&dstp)[3], int(&dstPitch)[3], const BYTE *(&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix &m)
{
  // 8 bit        uint8_t
  // 10,12,14 bit uint16_t (signed range)
  const __m256i half = _mm256_set1_epi16((short)(1 << (bits_per_pixel - 1)));  // 128
  const __m256i limit = _mm256_set1_epi16((short)((1 << bits_per_pixel) - 1)); // 255
  const __m256i offset = _mm256_set1_epi16((short)(m.offset_y));

  // to be able to use it as signed 16 bit in madd; 4096+(16<<13) would not fit into i16
  // multiplier is 4096 instead of 1:
  // original   : 1      * 4096
  // needed     : 1      * (4096 + offset_rgb<<13)  (4096 + 131072 overflows i16)
  // changed to : 4096   * (1 + offset_rgb>>(13-1)
  const int round_scale = 4096; // 1 << 12
  const int round_mask_plus_rgb_offset_scaled_i = (4096 + (m.offset_rgb << 13)) / round_scale;
  const __m256i m256i_round_scale = _mm256_set1_epi16(round_scale);

  __m256i zero = _mm256_setzero_si256();

  const __m256i m_uy_G = _mm256_set1_epi32(int((static_cast<uint16_t>(m.y_g) << 16) | static_cast<uint16_t>(m.u_g))); // y and u
  const __m256i m_vR_G = _mm256_set1_epi32(int((static_cast<uint16_t>(round_mask_plus_rgb_offset_scaled_i) << 16) | static_cast<uint16_t>(m.v_g))); // rounding 13 bit >> 1 and v

  const __m256i m_uy_B = _mm256_set1_epi32(int((static_cast<uint16_t>(m.y_b) << 16) | static_cast<uint16_t>(m.u_b))); // y and u
  const __m256i m_vR_B = _mm256_set1_epi32(int((static_cast<uint16_t>(round_mask_plus_rgb_offset_scaled_i) << 16) | static_cast<uint16_t>(m.v_b))); // rounding 13 bit >> 1 and v

  const __m256i m_uy_R = _mm256_set1_epi32(int((static_cast<uint16_t>(m.y_r) << 16) | static_cast<uint16_t>(m.u_r))); // y and u
  const __m256i m_vR_R = _mm256_set1_epi32(int((static_cast<uint16_t>(round_mask_plus_rgb_offset_scaled_i) << 16) | static_cast<uint16_t>(m.v_r))); // rounding 13 bit >> 1 and v

  const int rowsize = width * sizeof(pixel_t);
  for (int yy = 0; yy < height; yy++) {
    // if not mod16 then still no trouble, process non-visible pixels, we have 32 byte aligned in avs+
//    for (int x = 0; x < rowsize; x += 8 * sizeof(pixel_t)) {
    for (int x = 0; x < rowsize; x += 32 * sizeof(pixel_t)) { // rows are 64bytes aligned for AVX512 ?
      __m256i y, u, v;
      __m256i y_2, u_2, v_2;

      if constexpr (sizeof(pixel_t) == 1) {
        __m256i y_32 = _mm256_load_si256(reinterpret_cast<const __m256i *>(srcp[0] + x));
        __m256i u_32 = _mm256_load_si256(reinterpret_cast<const __m256i *>(srcp[1] + x));
        __m256i v_32 = _mm256_load_si256(reinterpret_cast<const __m256i *>(srcp[2] + x));
        y = _mm256_unpacklo_epi8(y_32, zero);
        y_2 = _mm256_unpackhi_epi8(y_32, zero);

        u = _mm256_unpacklo_epi8(u_32, zero);
        u_2 = _mm256_unpackhi_epi8(u_32, zero);

        v = _mm256_unpacklo_epi8(v_32, zero);
        v_2 = _mm256_unpackhi_epi8(v_32, zero);

      }
      else { // uint16_t pixels, 14 bits OK, but 16 bit pixels are unsigned, cannot madd
        y = _mm256_load_si256(reinterpret_cast<const __m256i *>(srcp[0] + x));
        u = _mm256_load_si256(reinterpret_cast<const __m256i *>(srcp[1] + x));
        v = _mm256_load_si256(reinterpret_cast<const __m256i *>(srcp[2] + x));
        y_2 = _mm256_load_si256(reinterpret_cast<const __m256i *>(srcp[0] + x + 32));
        u_2 = _mm256_load_si256(reinterpret_cast<const __m256i *>(srcp[1] + x + 32));
        v_2 = _mm256_load_si256(reinterpret_cast<const __m256i *>(srcp[2] + x + 32));
      }

      y = _mm256_adds_epi16(y, offset); // offset is negative
      u = _mm256_subs_epi16(u, half);
      v = _mm256_subs_epi16(v, half);

      y_2 = _mm256_adds_epi16(y_2, offset); // offset is negative
      u_2 = _mm256_subs_epi16(u_2, half);
      v_2 = _mm256_subs_epi16(v_2, half);

      /*
      // int b = (((int64_t)matrix.y_b * Y + (int64_t)matrix.u_b * U + (int64_t)matrix.v_b * V + 4096)>>13);
      // int g = (((int64_t)matrix.y_g * Y + (int64_t)matrix.u_g * U + (int64_t)matrix.v_g * V + 4096)>>13);
      // int r = (((int64_t)matrix.y_r * Y + (int64_t)matrix.u_r * U + (int64_t)matrix.v_r * V + 4096)>>13);
      */
      // Need1:  (m.y_b   m.u_b )     (m.y_b   m.u_b)     (m.y_b   m.u_b)     (m.y_b   m.u_b)   8x16 bit
      //         (  y3      u3  )     (  y2      u2 )     (  y1      u1 )     (   y0     u0 )   8x16 bit
      // res1=  (y_b*y3 + u_b*u3)   ...                                                         4x32 bit
      // Need2:  (m.v_b   round')     (m.y_b   round')     (m.y_b   round')     (m.y_b   round')
      //         (  v3     4096 )     (  v2     4096 )     (  v1     4096 )     (  v0     4096 )
      // res2=  (yv_b*v3 + round' )  ...  round' = round + rgb_offset

      // *G* ----------------
      const __m256i uy0123 = _mm256_unpacklo_epi16(u, y);
      const __m256i xv0123 = _mm256_unpacklo_epi16(v, m256i_round_scale);

      const __m256i uy0123_2 = _mm256_unpacklo_epi16(u_2, y_2);
      const __m256i xv0123_2 = _mm256_unpacklo_epi16(v_2, m256i_round_scale);

      const __m256i uy4567 = _mm256_unpackhi_epi16(u, y);
      const __m256i xv4567 = _mm256_unpackhi_epi16(v, m256i_round_scale);

      const __m256i uy4567_2 = _mm256_unpackhi_epi16(u_2, y_2);
      const __m256i xv4567_2 = _mm256_unpackhi_epi16(v_2, m256i_round_scale);

//      const __m256i uy0123 = _mm256_unpacklo_epi16(u, y);
//      res1_lo = _mm256_madd_epi16(m_uy_G, uy0123);
//      const __m256i xv0123 = _mm256_unpacklo_epi16(v, m256i_round_scale);
//      res2_lo = _mm256_madd_epi16(m_vR_G, xv0123);
//      __m256i g_lo = _mm256_srai_epi32(_mm256_add_epi32(res1, res2), 13);
      __m256i g_lo = _mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(m_uy_G, uy0123), _mm256_madd_epi16(m_vR_G, xv0123)), 13);
      __m256i g_lo_2 = _mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(m_uy_G, uy0123_2), _mm256_madd_epi16(m_vR_G, xv0123_2)), 13);

      __m256i g_hi = _mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(m_uy_G, uy4567), _mm256_madd_epi16(m_vR_G, xv4567)), 13);
      __m256i g_hi_2 = _mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(m_uy_G, uy4567_2), _mm256_madd_epi16(m_vR_G, xv4567_2)), 13);

      __m256i g = _mm256_packs_epi32(g_lo, g_hi); // 2x4x32 -> 2x4xuint16_t
      __m256i g_2 = _mm256_packs_epi32(g_lo_2, g_hi_2); // 2x4x32 -> 2x4xuint16_t
      if constexpr (sizeof(pixel_t) == 1) {
        g = _mm256_packus_epi16(g, g_2);   // 32x uint16_t -> 32x uint_8
        _mm256_store_si256(reinterpret_cast<__m256i *>(dstp[0] + x), g);
      }
      else {
        g = _mm256_max_epi16(_mm256_min_epi16(g, limit), zero); // clamp 10,12,14 bit
        g_2 = _mm256_max_epi16(_mm256_min_epi16(g_2, limit), zero); // clamp 10,12,14 bit
        _mm256_store_si256(reinterpret_cast<__m256i *>(dstp[0] + x), g);
        _mm256_store_si256(reinterpret_cast<__m256i *>(dstp[0] + x + 32), g_2);
      }

      // *B* ----------------
      __m256i b_lo = _mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(m_uy_B, uy0123), _mm256_madd_epi16(m_vR_B, xv0123)), 13);
      __m256i b_lo_2 = _mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(m_uy_B, uy0123_2), _mm256_madd_epi16(m_vR_B, xv0123_2)), 13);

      __m256i b_hi = _mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(m_uy_B, uy4567), _mm256_madd_epi16(m_vR_B, xv4567)), 13);
      __m256i b_hi_2 = _mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(m_uy_B, uy4567_2), _mm256_madd_epi16(m_vR_B, xv4567_2)), 13);

      __m256i b = _mm256_packs_epi32(b_lo, b_hi); // 2x4x32 -> 2x4xuint16_t
      __m256i b_2 = _mm256_packs_epi32(b_lo_2, b_hi_2); // 2x4x32 -> 2x4xuint16_t

      if constexpr (sizeof(pixel_t) == 1) {
        b = _mm256_packus_epi16(b, b_2);   // 32x uint16_t -> 32x uint_8
        _mm256_store_si256(reinterpret_cast<__m256i *>(dstp[1] + x), b);
      }
      else {
        b = _mm256_max_epi16(_mm256_min_epi16(b, limit), zero); // clamp 10,12,14 bit
        b_2 = _mm256_max_epi16(_mm256_min_epi16(b_2, limit), zero); // clamp 10,12,14 bit
        _mm256_store_si256(reinterpret_cast<__m256i *>(dstp[1] + x), b);
        _mm256_store_si256(reinterpret_cast<__m256i *>(dstp[1] + x + 32), b_2);
      }

      // *R* ----------------
      __m256i r_lo = _mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(m_uy_R, uy0123), _mm256_madd_epi16(m_vR_R, xv0123)), 13);
      __m256i r_lo_2 = _mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(m_uy_R, uy0123_2), _mm256_madd_epi16(m_vR_R, xv0123_2)), 13);

      __m256i r_hi = _mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(m_uy_R, uy4567), _mm256_madd_epi16(m_vR_R, xv4567)), 13);
      __m256i r_hi_2 = _mm256_srai_epi32(_mm256_add_epi32(_mm256_madd_epi16(m_uy_R, uy4567_2), _mm256_madd_epi16(m_vR_R, xv4567_2)), 13);

      __m256i r = _mm256_packs_epi32(r_lo, r_hi); // 2x4x32 -> 2x4xuint16_t
      __m256i r_2 = _mm256_packs_epi32(r_lo_2, r_hi_2); // 2x4x32 -> 2x4xuint16_t

      if constexpr (sizeof(pixel_t) == 1) {
        r = _mm256_packus_epi16(r, r_2);   // 32x uint16_t -> 32x uint_8
        _mm256_store_si256(reinterpret_cast<__m256i *>(dstp[2] + x), r);
      }
      else {
        r = _mm256_max_epi16(_mm256_min_epi16(r, limit), zero); // clamp 10,12,14 bit
        r_2 = _mm256_max_epi16(_mm256_min_epi16(r_2, limit), zero); // clamp 10,12,14 bit
        _mm256_store_si256(reinterpret_cast<__m256i *>(dstp[2] + x), r);
        _mm256_store_si256(reinterpret_cast<__m256i *>(dstp[2] + x + 32), r_2);
      }

    }
    srcp[0] += srcPitch[0];
    srcp[1] += srcPitch[1];
    srcp[2] += srcPitch[2];
    dstp[0] += dstPitch[0];
    dstp[1] += dstPitch[1];
    dstp[2] += dstPitch[2];
  }
}

//instantiate
//template<typename pixel_t, int bits_per_pixel>
template void convert_yuv_to_planarrgb_uint8_14_avx2<uint8_t, 8>(BYTE *(&dstp)[3], int(&dstPitch)[3], const BYTE *(&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix &m);
template void convert_yuv_to_planarrgb_uint8_14_avx2<uint16_t, 10>(BYTE *(&dstp)[3], int(&dstPitch)[3], const BYTE *(&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix &m);
template void convert_yuv_to_planarrgb_uint8_14_avx2<uint16_t, 12>(BYTE *(&dstp)[3], int(&dstPitch)[3], const BYTE *(&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix &m);
template void convert_yuv_to_planarrgb_uint8_14_avx2<uint16_t, 14>(BYTE *(&dstp)[3], int(&dstPitch)[3], const BYTE *(&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix &m);

template<typename pixel_t, int bits_per_pixel>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("avx2")))
#endif
void convert_yuv_to_planarrgb_uint8_14_tops_avx2(BYTE *(&dstp)[3], int(&dstPitch)[3], const BYTE *(&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix &m)
{
  // 8 bit        uint8_t
  // 10,12,14 bit uint16_t (signed range)
  const __m256i half = _mm256_set1_epi16((short)(1 << (bits_per_pixel - 1)));  // 128
  const __m256i limit = _mm256_set1_epi16((short)((1 << bits_per_pixel) - 1)); // 255
  const __m256i offset = _mm256_set1_epi16((short)(m.offset_y));

  float* dstp0 = (float*)dstp[0];
  float* dstp1 = (float*)dstp[1];
  float* dstp2 = (float*)dstp[2];
  int dstpitch0 = dstPitch[0] / sizeof(float);
  int dstpitch1 = dstPitch[1] / sizeof(float);
  int dstpitch2 = dstPitch[2] / sizeof(float);

  // to be able to use it as signed 16 bit in madd; 4096+(16<<13) would not fit into i16
  // multiplier is 4096 instead of 1:
  // original   : 1      * 4096
  // needed     : 1      * (4096 + offset_rgb<<13)  (4096 + 131072 overflows i16)
  // changed to : 4096   * (1 + offset_rgb>>(13-1)
  const int round_scale = 4096; // 1 << 12
  const int round_mask_plus_rgb_offset_scaled_i = (4096 + (m.offset_rgb << 13)) / round_scale;
  const __m256i m256i_round_scale = _mm256_set1_epi16(round_scale);

  __m256i zero = _mm256_setzero_si256();

  const __m256i m_uy_G = _mm256_set1_epi32(int((static_cast<uint16_t>(m.y_g) << 16) | static_cast<uint16_t>(m.u_g))); // y and u
  const __m256i m_vR_G = _mm256_set1_epi32(int((static_cast<uint16_t>(round_mask_plus_rgb_offset_scaled_i) << 16) | static_cast<uint16_t>(m.v_g))); // rounding 13 bit >> 1 and v

  const __m256i m_uy_B = _mm256_set1_epi32(int((static_cast<uint16_t>(m.y_b) << 16) | static_cast<uint16_t>(m.u_b))); // y and u
  const __m256i m_vR_B = _mm256_set1_epi32(int((static_cast<uint16_t>(round_mask_plus_rgb_offset_scaled_i) << 16) | static_cast<uint16_t>(m.v_b))); // rounding 13 bit >> 1 and v

  const __m256i m_uy_R = _mm256_set1_epi32(int((static_cast<uint16_t>(m.y_r) << 16) | static_cast<uint16_t>(m.u_r))); // y and u
  const __m256i m_vR_R = _mm256_set1_epi32(int((static_cast<uint16_t>(round_mask_plus_rgb_offset_scaled_i) << 16) | static_cast<uint16_t>(m.v_r))); // rounding 13 bit >> 1 and v

  // convert to float32
  bits_conv_constants d;
 
  // void get_bits_conv_constants(bits_conv_constants& d, bool use_chroma, bool fulls, bool fulld, int srcBitDepth, int dstBitDepth)
  get_bits_conv_constants(d, false, true, true, bits_per_pixel, 32); // fulls=fulld=true (default for RGB ?)

  const __m256i m256_src_offset_epi32 = _mm256_set1_epi32(d.src_offset_i * 8192);// in 2^13 scaled domain , not used for  fulls=fulld=true (default for RGB ?)
  const __m256 m256_mul_factor = _mm256_set1_ps(d.mul_factor / 8192); // to replace srai(13) in output fma
  const __m256 m256_dst_offset = _mm256_set1_ps(d.dst_offset); // not used with  fulls=fulld=true (default for RGB ?)

  const int rowsize = width * sizeof(pixel_t);
  for (int yy = 0; yy < height; yy++) {
    // if not mod16 then still no trouble, process non-visible pixels, we have 32 byte aligned in avs+
    int x_ps = 0;
//    for (int x = 0; x < rowsize; x += 8 * sizeof(pixel_t)) {
    for (int x = 0; x < rowsize; x += 32 * sizeof(pixel_t)) { // rows are 64bytes aligned for AVX512 ?
      __m256i y, u, v;
      __m256i y_2, u_2, v_2;

/*      //DEBUG
      uint16_t* pdst = (uint16_t*)(const_cast<uint8_t*>(srcp[0]));
      for (int i = 0; i < 255; i++)
        pdst[i] = (uint16_t)i;*/

      if constexpr (sizeof(pixel_t) == 1) {
        __m256i y_32 = _mm256_load_si256(reinterpret_cast<const __m256i *>(srcp[0] + x));
        __m256i u_32 = _mm256_load_si256(reinterpret_cast<const __m256i *>(srcp[1] + x));
        __m256i v_32 = _mm256_load_si256(reinterpret_cast<const __m256i *>(srcp[2] + x));
        y = _mm256_unpacklo_epi8(y_32, zero);
        y_2 = _mm256_unpackhi_epi8(y_32, zero);

        u = _mm256_unpacklo_epi8(u_32, zero);
        u_2 = _mm256_unpackhi_epi8(u_32, zero);

        v = _mm256_unpacklo_epi8(v_32, zero);
        v_2 = _mm256_unpackhi_epi8(v_32, zero);

      }
      else { // uint16_t pixels, 14 bits OK, but 16 bit pixels are unsigned, cannot madd
        y = _mm256_load_si256(reinterpret_cast<const __m256i *>(srcp[0] + x));
        u = _mm256_load_si256(reinterpret_cast<const __m256i *>(srcp[1] + x));
        v = _mm256_load_si256(reinterpret_cast<const __m256i *>(srcp[2] + x));
        y_2 = _mm256_load_si256(reinterpret_cast<const __m256i *>(srcp[0] + x + 32));
        u_2 = _mm256_load_si256(reinterpret_cast<const __m256i *>(srcp[1] + x + 32));
        v_2 = _mm256_load_si256(reinterpret_cast<const __m256i *>(srcp[2] + x + 32));
      }

      y = _mm256_adds_epi16(y, offset); // offset is negative
      u = _mm256_subs_epi16(u, half);
      v = _mm256_subs_epi16(v, half);

      y_2 = _mm256_adds_epi16(y_2, offset); // offset is negative
      u_2 = _mm256_subs_epi16(u_2, half);
      v_2 = _mm256_subs_epi16(v_2, half);

      /*
      // int b = (((int64_t)matrix.y_b * Y + (int64_t)matrix.u_b * U + (int64_t)matrix.v_b * V + 4096)>>13);
      // int g = (((int64_t)matrix.y_g * Y + (int64_t)matrix.u_g * U + (int64_t)matrix.v_g * V + 4096)>>13);
      // int r = (((int64_t)matrix.y_r * Y + (int64_t)matrix.u_r * U + (int64_t)matrix.v_r * V + 4096)>>13);
      */
      // Need1:  (m.y_b   m.u_b )     (m.y_b   m.u_b)     (m.y_b   m.u_b)     (m.y_b   m.u_b)   8x16 bit
      //         (  y3      u3  )     (  y2      u2 )     (  y1      u1 )     (   y0     u0 )   8x16 bit
      // res1=  (y_b*y3 + u_b*u3)   ...                                                         4x32 bit
      // Need2:  (m.v_b   round')     (m.y_b   round')     (m.y_b   round')     (m.y_b   round')
      //         (  v3     4096 )     (  v2     4096 )     (  v1     4096 )     (  v0     4096 )
      // res2=  (yv_b*v3 + round' )  ...  round' = round + rgb_offset

      // *G* ----------------
      const __m256i uy0123 = _mm256_unpacklo_epi16(u, y);
      const __m256i xv0123 = _mm256_unpacklo_epi16(v, m256i_round_scale);

      const __m256i uy0123_2 = _mm256_unpacklo_epi16(u_2, y_2);
      const __m256i xv0123_2 = _mm256_unpacklo_epi16(v_2, m256i_round_scale);

      const __m256i uy4567 = _mm256_unpackhi_epi16(u, y);
      const __m256i xv4567 = _mm256_unpackhi_epi16(v, m256i_round_scale);

      const __m256i uy4567_2 = _mm256_unpackhi_epi16(u_2, y_2);
      const __m256i xv4567_2 = _mm256_unpackhi_epi16(v_2, m256i_round_scale);

      //      const __m256i uy0123 = _mm256_unpacklo_epi16(u, y);
      //      res1_lo = _mm256_madd_epi16(m_uy_G, uy0123);
      //      const __m256i xv0123 = _mm256_unpacklo_epi16(v, m256i_round_scale);
      //      res2_lo = _mm256_madd_epi16(m_vR_G, xv0123);
      //      __m256i g_lo = _mm256_srai_epi32(_mm256_add_epi32(res1, res2), 13);
      // need to divide to 2^13 - add this to mult coeff
      __m256i g_lo = _mm256_add_epi32(_mm256_madd_epi16(m_uy_G, uy0123), _mm256_madd_epi16(m_vR_G, xv0123));
      __m256i g_lo_2 = _mm256_add_epi32(_mm256_madd_epi16(m_uy_G, uy0123_2), _mm256_madd_epi16(m_vR_G, xv0123_2));

      __m256i g_hi = _mm256_add_epi32(_mm256_madd_epi16(m_uy_G, uy4567), _mm256_madd_epi16(m_vR_G, xv4567));
      __m256i g_hi_2 = _mm256_add_epi32(_mm256_madd_epi16(m_uy_G, uy4567_2), _mm256_madd_epi16(m_vR_G, xv4567_2));

      // are we need it for narrow->narrow (or no range mapping change ?) , uncomment in future if add full control for fulls/fulld
/*      g_lo = _mm256_sub_epi32(g_lo, m256_src_offset_epi32); // offset is 2^13 scaled to keep precision better
      g_lo_2 = _mm256_sub_epi32(g_lo_2, m256_src_offset_epi32);
      g_hi = _mm256_sub_epi32(g_hi, m256_src_offset_epi32);
      g_hi_2 = _mm256_sub_epi32(g_hi_2, m256_src_offset_epi32);*/

      __m256 g_lo_ps = _mm256_cvtepi32_ps(g_lo);
      __m256 g_lo_2_ps = _mm256_cvtepi32_ps(g_lo_2);
      __m256 g_hi_ps = _mm256_cvtepi32_ps(g_hi);
      __m256 g_hi_2_ps = _mm256_cvtepi32_ps(g_hi_2);

      //shuffle to linear
      __m256 g_0_ps, g_1_ps, g_2_ps, g_3_ps;
      if constexpr (sizeof(pixel_t) == 1)
      {
        g_0_ps = _mm256_permute2f128_ps(g_lo_ps, g_hi_ps, 0x20);
        g_1_ps = _mm256_permute2f128_ps(g_lo_2_ps, g_hi_2_ps, 0x20);
        g_2_ps = _mm256_permute2f128_ps(g_lo_ps, g_hi_ps, 0x31);
        g_3_ps = _mm256_permute2f128_ps(g_lo_2_ps, g_hi_2_ps, 0x31);
      }
      else
      {
        g_0_ps = _mm256_permute2f128_ps(g_lo_ps, g_hi_ps, 0x20);
        g_1_ps = _mm256_permute2f128_ps(g_lo_ps, g_hi_ps, 0x31);
        g_2_ps = _mm256_permute2f128_ps(g_lo_2_ps, g_hi_2_ps, 0x20);
        g_3_ps = _mm256_permute2f128_ps(g_lo_2_ps, g_hi_2_ps, 0x31);
      }

/*      g_0_ps = _mm256_fmadd_ps(g_0_ps, m256_mul_factor, m256_dst_offset); // for fulls=fulld dst_offset is zero, can do mul only
      g_1_ps = _mm256_fmadd_ps(g_1_ps, m256_mul_factor, m256_dst_offset);
      g_2_ps = _mm256_fmadd_ps(g_2_ps, m256_mul_factor, m256_dst_offset);
      g_3_ps = _mm256_fmadd_ps(g_3_ps, m256_mul_factor, m256_dst_offset);*/
      g_0_ps = _mm256_mul_ps(g_0_ps, m256_mul_factor); 
      g_1_ps = _mm256_mul_ps(g_1_ps, m256_mul_factor);
      g_2_ps = _mm256_mul_ps(g_2_ps, m256_mul_factor);
      g_3_ps = _mm256_mul_ps(g_3_ps, m256_mul_factor);

      _mm256_store_ps((dstp0 + x_ps + 0), g_0_ps);
      _mm256_store_ps((dstp0 + x_ps + 8), g_1_ps);
      _mm256_store_ps((dstp0 + x_ps + 16), g_2_ps);
      _mm256_store_ps((dstp0 + x_ps + 24), g_3_ps);

      // *B* ----------------
      __m256i b_lo = _mm256_add_epi32(_mm256_madd_epi16(m_uy_B, uy0123), _mm256_madd_epi16(m_vR_B, xv0123));
      __m256i b_lo_2 = _mm256_add_epi32(_mm256_madd_epi16(m_uy_B, uy0123_2), _mm256_madd_epi16(m_vR_B, xv0123_2));

      __m256i b_hi = _mm256_add_epi32(_mm256_madd_epi16(m_uy_B, uy4567), _mm256_madd_epi16(m_vR_B, xv4567));
      __m256i b_hi_2 = _mm256_add_epi32(_mm256_madd_epi16(m_uy_B, uy4567_2), _mm256_madd_epi16(m_vR_B, xv4567_2));

      // are we need it for narrow->narrow (or no range mapping change ?) 
/*      b_lo = _mm256_sub_epi32(b_lo, m256_src_offset_epi32); // offset is 2^13 scaled to keep precision better
      b_lo_2 = _mm256_sub_epi32(b_lo_2, m256_src_offset_epi32);
      b_hi = _mm256_sub_epi32(b_hi, m256_src_offset_epi32);
      b_hi_2 = _mm256_sub_epi32(b_hi_2, m256_src_offset_epi32);*/

      __m256 b_lo_ps = _mm256_cvtepi32_ps(b_lo);
      __m256 b_lo_2_ps = _mm256_cvtepi32_ps(b_lo_2);
      __m256 b_hi_ps = _mm256_cvtepi32_ps(b_hi);
      __m256 b_hi_2_ps = _mm256_cvtepi32_ps(b_hi_2);

      //shuffle to linear
      __m256 b_0_ps, b_1_ps, b_2_ps, b_3_ps;
      if constexpr (sizeof(pixel_t) == 1)
      {
        b_0_ps = _mm256_permute2f128_ps(b_lo_ps, b_hi_ps, 0x20);
        b_1_ps = _mm256_permute2f128_ps(b_lo_2_ps, b_hi_2_ps, 0x20);
        b_2_ps = _mm256_permute2f128_ps(b_lo_ps, b_hi_ps, 0x31);
        b_3_ps = _mm256_permute2f128_ps(b_lo_2_ps, b_hi_2_ps, 0x31);
      }
      else
      {
        b_0_ps = _mm256_permute2f128_ps(b_lo_ps, b_hi_ps, 0x20);
        b_1_ps = _mm256_permute2f128_ps(b_lo_ps, b_hi_ps, 0x31);
        b_2_ps = _mm256_permute2f128_ps(b_lo_2_ps, b_hi_2_ps, 0x20);
        b_3_ps = _mm256_permute2f128_ps(b_lo_2_ps, b_hi_2_ps, 0x31);
      }

/*      b_0_ps = _mm256_fmadd_ps(b_0_ps, m256_mul_factor, m256_dst_offset);
      b_1_ps = _mm256_fmadd_ps(b_1_ps, m256_mul_factor, m256_dst_offset);
      b_2_ps = _mm256_fmadd_ps(b_2_ps, m256_mul_factor, m256_dst_offset);
      b_3_ps = _mm256_fmadd_ps(b_3_ps, m256_mul_factor, m256_dst_offset);*/
      b_0_ps = _mm256_mul_ps(b_0_ps, m256_mul_factor);
      b_1_ps = _mm256_mul_ps(b_1_ps, m256_mul_factor);
      b_2_ps = _mm256_mul_ps(b_2_ps, m256_mul_factor);
      b_3_ps = _mm256_mul_ps(b_3_ps, m256_mul_factor);

      _mm256_store_ps((dstp1 + x_ps + 0), b_0_ps);
      _mm256_store_ps((dstp1 + x_ps + 8), b_1_ps);
      _mm256_store_ps((dstp1 + x_ps + 16), b_2_ps);
      _mm256_store_ps((dstp1 + x_ps + 24), b_3_ps);

      // *R* ----------------
      __m256i r_lo = _mm256_add_epi32(_mm256_madd_epi16(m_uy_R, uy0123), _mm256_madd_epi16(m_vR_R, xv0123));
      __m256i r_lo_2 = _mm256_add_epi32(_mm256_madd_epi16(m_uy_R, uy0123_2), _mm256_madd_epi16(m_vR_R, xv0123_2));

      __m256i r_hi = _mm256_add_epi32(_mm256_madd_epi16(m_uy_R, uy4567), _mm256_madd_epi16(m_vR_R, xv4567));
      __m256i r_hi_2 = _mm256_add_epi32(_mm256_madd_epi16(m_uy_R, uy4567_2), _mm256_madd_epi16(m_vR_R, xv4567_2));

      // are we need it for narrow->narrow (or no range mapping change ?) 
/*      r_lo = _mm256_sub_epi32(r_lo, m256_src_offset_epi32); // offset is 2^13 scaled to keep precision better
      r_lo_2 = _mm256_sub_epi32(r_lo_2, m256_src_offset_epi32);
      r_hi = _mm256_sub_epi32(r_hi, m256_src_offset_epi32);
      r_hi_2 = _mm256_sub_epi32(r_hi_2, m256_src_offset_epi32);*/

      __m256 r_lo_ps = _mm256_cvtepi32_ps(r_lo);
      __m256 r_lo_2_ps = _mm256_cvtepi32_ps(r_lo_2);
      __m256 r_hi_ps = _mm256_cvtepi32_ps(r_hi);
      __m256 r_hi_2_ps = _mm256_cvtepi32_ps(r_hi_2);

      //shuffle to linear
      __m256 r_0_ps, r_1_ps, r_2_ps, r_3_ps;
      if constexpr (sizeof(pixel_t) == 1)
      {
        r_0_ps = _mm256_permute2f128_ps(r_lo_ps, r_hi_ps, 0x20);
        r_1_ps = _mm256_permute2f128_ps(r_lo_2_ps, r_hi_2_ps, 0x20);
        r_2_ps = _mm256_permute2f128_ps(r_lo_ps, r_hi_ps, 0x31);
        r_3_ps = _mm256_permute2f128_ps(r_lo_2_ps, r_hi_2_ps, 0x31);
      }
      else
      {
        r_0_ps = _mm256_permute2f128_ps(r_lo_ps, r_hi_ps, 0x20);
        r_1_ps = _mm256_permute2f128_ps(r_lo_ps, r_hi_ps, 0x31);
        r_2_ps = _mm256_permute2f128_ps(r_lo_2_ps, r_hi_2_ps, 0x20);
        r_3_ps = _mm256_permute2f128_ps(r_lo_2_ps, r_hi_2_ps, 0x31);
      }
/*      r_0_ps = _mm256_fmadd_ps(r_0_ps, m256_mul_factor, m256_dst_offset);
      r_1_ps = _mm256_fmadd_ps(r_1_ps, m256_mul_factor, m256_dst_offset);
      r_2_ps = _mm256_fmadd_ps(r_2_ps, m256_mul_factor, m256_dst_offset);
      r_3_ps = _mm256_fmadd_ps(r_3_ps, m256_mul_factor, m256_dst_offset);*/
      r_0_ps = _mm256_mul_ps(r_0_ps, m256_mul_factor);
      r_1_ps = _mm256_mul_ps(r_1_ps, m256_mul_factor);
      r_2_ps = _mm256_mul_ps(r_2_ps, m256_mul_factor);
      r_3_ps = _mm256_mul_ps(r_3_ps, m256_mul_factor);

      _mm256_store_ps((dstp2 + x_ps + 0), r_0_ps);
      _mm256_store_ps((dstp2 + x_ps + 8), r_1_ps);
      _mm256_store_ps((dstp2 + x_ps + 16), r_2_ps);
      _mm256_store_ps((dstp2 + x_ps + 24), r_3_ps);

      x_ps += 32; // fixed step for floats output
    }
    srcp[0] += srcPitch[0];
    srcp[1] += srcPitch[1];
    srcp[2] += srcPitch[2];
    dstp0 += dstpitch0;
    dstp1 += dstpitch1;
    dstp2 += dstpitch2;
  }
}

//instantiate
//template<typename pixel_t, int bits_per_pixel>
template void convert_yuv_to_planarrgb_uint8_14_tops_avx2<uint8_t, 8>(BYTE *(&dstp)[3], int(&dstPitch)[3], const BYTE *(&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix &m);
template void convert_yuv_to_planarrgb_uint8_14_tops_avx2<uint16_t, 10>(BYTE *(&dstp)[3], int(&dstPitch)[3], const BYTE *(&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix &m);
template void convert_yuv_to_planarrgb_uint8_14_tops_avx2<uint16_t, 12>(BYTE *(&dstp)[3], int(&dstPitch)[3], const BYTE *(&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix &m);
template void convert_yuv_to_planarrgb_uint8_14_tops_avx2<uint16_t, 14>(BYTE *(&dstp)[3], int(&dstPitch)[3], const BYTE *(&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix &m);


#if defined(GCC) || defined(CLANG)
__attribute__((__target__("avx2"))) // FMA too
#endif
void convert_yuv_to_planarrgb_float_avx2(BYTE *(&dstp)[3], int(&dstPitch)[3], const BYTE *(&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix &m)
{
  // 32 bit float
  __m256  offset_f = _mm256_set1_ps(m.offset_y_f);
  __m256  offset_rgb = _mm256_set1_ps(m.offset_rgb_f);

  const bool has_offset_rgb = 0 != m.offset_rgb_f;

  const __m256 mat_yG = _mm256_set1_ps(m.y_g_f);
  const __m256 mat_uG = _mm256_set1_ps(m.u_g_f);
  const __m256 mat_vG = _mm256_set1_ps(m.v_g_f);

  const __m256 mat_yB = _mm256_set1_ps(m.y_b_f);
  const __m256 mat_uB = _mm256_set1_ps(m.u_b_f);
  const __m256 mat_vB = _mm256_set1_ps(m.v_b_f);

  const __m256 mat_yR = _mm256_set1_ps(m.y_r_f);
  const __m256 mat_uR = _mm256_set1_ps(m.u_r_f);
  const __m256 mat_vR = _mm256_set1_ps(m.v_r_f);

  const int rowsize = width * sizeof(float);
  for (int yy = 0; yy < height; yy++) {
//    for (int x = 0; x < rowsize; x += 4 * sizeof(float)) {
    for (int x = 0; x < rowsize; x += 8 * sizeof(float)) { // may be unroll to 16 or compiler will do it ?
      __m256 resG, resB, resR;
      __m256 y, u, v;

      // float: load 32 bytes: 8 pixels
      y = _mm256_load_ps(reinterpret_cast<const float *>(srcp[0] + x));
      u = _mm256_load_ps(reinterpret_cast<const float *>(srcp[1] + x));
      v = _mm256_load_ps(reinterpret_cast<const float *>(srcp[2] + x));
      y = _mm256_add_ps(y, offset_f); // offset is negative*/

      // *G*
      resG = _mm256_mul_ps(y, mat_yG); // mul_y
      resG = _mm256_fmadd_ps(u, mat_uG, resG); // mul_u
      resG = _mm256_fmadd_ps(v, mat_vG, resG); // mul_v
      if (has_offset_rgb)
        resG = _mm256_add_ps(resG, offset_rgb);
      // no clamp
      _mm256_store_ps(reinterpret_cast<float *>(dstp[0] + x), resG);
      // *B*
      resB = _mm256_mul_ps(y, mat_yB); // mul_y
      resB = _mm256_fmadd_ps(u, mat_uB, resB); // mul_u
      resB = _mm256_fmadd_ps(v, mat_vB, resB); // mul_v
      if (has_offset_rgb)
        resB = _mm256_add_ps(resB, offset_rgb);
      // no clamp
      _mm256_store_ps(reinterpret_cast<float *>(dstp[1] + x), resB);
      // *R*
      resR = _mm256_mul_ps(y, mat_yR); // mul_y
      resR = _mm256_fmadd_ps(u, mat_uR, resR); // mul_u
      resR = _mm256_fmadd_ps(v, mat_vR, resR); // mul_v
      if (has_offset_rgb)
        resR = _mm256_add_ps(resR, offset_rgb);
      // no clamp
      _mm256_store_ps(reinterpret_cast<float *>(dstp[2] + x), resR);
    }
    srcp[0] += srcPitch[0];
    srcp[1] += srcPitch[1];
    srcp[2] += srcPitch[2];
    dstp[0] += dstPitch[0];
    dstp[1] += dstPitch[1];
    dstp[2] += dstPitch[2];
  }
}

template<int bits_per_pixel>
#if defined(GCC) || defined(CLANG)
__attribute__((__target__("avx2")))
#endif
void convert_yuv_to_planarrgb_uint16_avx2(BYTE *(&dstp)[3], int(&dstPitch)[3], const BYTE *(&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix &m)
{
  // 16 bit uint16_t (unsigned range)
  const __m256  half_f = _mm256_set1_ps((float)(1u << (bits_per_pixel - 1))); // 8-16 bit chroma has shift
  const __m256i limit = _mm256_set1_epi16((short)((1 << bits_per_pixel) - 1)); // 255
  const __m256  offset_f = _mm256_set1_ps(m.offset_y_f);
  const __m256  offset_rgb = _mm256_set1_ps(m.offset_rgb_f);

  const __m256 mat_yG = _mm256_set1_ps(m.y_g_f);
  const __m256 mat_uG = _mm256_set1_ps(m.u_g_f);
  const __m256 mat_vG = _mm256_set1_ps(m.v_g_f);

  const __m256 mat_yB = _mm256_set1_ps(m.y_b_f);
  const __m256 mat_uB = _mm256_set1_ps(m.u_b_f);
  const __m256 mat_vB = _mm256_set1_ps(m.v_b_f);

  const __m256 mat_yR = _mm256_set1_ps(m.y_r_f);
  const __m256 mat_uR = _mm256_set1_ps(m.u_r_f);
  const __m256 mat_vR = _mm256_set1_ps(m.v_r_f);

  const bool has_offset_rgb = 0 != m.offset_rgb_f;

  const __m256i zero = _mm256_setzero_si256();

  const int rowsize = width * sizeof(uint16_t);
  for (int yy = 0; yy < height; yy++) {
//    for (int x = 0; x < rowsize; x += 4 * sizeof(uint16_t)) {
    for (int x = 0; x < rowsize; x += 16 * sizeof(uint16_t)) {
      __m256 resG, resG_2;
      __m256i resiG, resiG_2;

      __m256 resB, resB_2;
      __m256i resiB, resiB_2;

      __m256 resR, resR_2;
      __m256i resiR, resiR_2;

      __m256 y, u, v;
      __m256 y_2, u_2, v_2;

      // uint16_t: load 32 bytes: 16 pixels
      __m256i yi_16 = _mm256_load_si256(reinterpret_cast<const __m256i *>(srcp[0] + x));
      __m256i ui_16 = _mm256_load_si256(reinterpret_cast<const __m256i *>(srcp[1] + x));
      __m256i vi_16 = _mm256_load_si256(reinterpret_cast<const __m256i *>(srcp[2] + x));

      y = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(yi_16, zero));
      u = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(ui_16, zero));
      v = _mm256_cvtepi32_ps(_mm256_unpacklo_epi16(vi_16, zero));

      y_2 = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(yi_16, zero));
      u_2 = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(ui_16, zero));
      v_2 = _mm256_cvtepi32_ps(_mm256_unpackhi_epi16(vi_16, zero));

      y = _mm256_add_ps(y, offset_f); // offset is negative
      u = _mm256_sub_ps(u, half_f);
      v = _mm256_sub_ps(v, half_f);

      y_2 = _mm256_add_ps(y_2, offset_f); // offset is negative
      u_2 = _mm256_sub_ps(u_2, half_f);
      v_2 = _mm256_sub_ps(v_2, half_f);

      // *G*
      resG = _mm256_mul_ps(y, mat_yG); // mul_y
      resG = _mm256_fmadd_ps(u, mat_uG, resG); // mul_u
      resG = _mm256_fmadd_ps(v, mat_vG, resG); // mul_v
      if (has_offset_rgb)
        resG = _mm256_add_ps(resG, offset_rgb);

      resG_2 = _mm256_mul_ps(y_2, mat_yG); // mul_y
      resG_2 = _mm256_fmadd_ps(u_2, mat_uG, resG_2); // mul_u
      resG_2 = _mm256_fmadd_ps(v_2, mat_vG, resG_2); // mul_v
      if (has_offset_rgb)
        resG_2 = _mm256_add_ps(resG_2, offset_rgb);

      resiG = _mm256_cvtps_epi32(resG); // no extra rounding, cvtps rounds to nearest
      resiG_2 = _mm256_cvtps_epi32(resG_2); // no extra rounding, cvtps rounds to nearest

      resiG = _mm256_packus_epi32(resiG, resiG_2);
      if constexpr (bits_per_pixel < 16) // albeit 10-14 bit have another function, make this general
        resiG = _mm256_min_epi16(resiG, limit); // clamp 10,12,14 bit
      _mm256_store_si256(reinterpret_cast<__m256i *>(dstp[0] + x), resiG);

      // *B*
      resB = _mm256_mul_ps(y, mat_yB); // mul_y
      resB = _mm256_fmadd_ps(u, mat_uB, resB); // mul_u
      resB = _mm256_fmadd_ps(v, mat_vB, resB); // mul_v
      if (has_offset_rgb)
        resB = _mm256_add_ps(resB, offset_rgb);

      resB_2 = _mm256_mul_ps(y_2, mat_yB); // mul_y
      resB_2 = _mm256_fmadd_ps(u_2, mat_uB, resB_2); // mul_u
      resB_2 = _mm256_fmadd_ps(v_2, mat_vB, resB_2); // mul_v
      if (has_offset_rgb)
        resB_2 = _mm256_add_ps(resB_2, offset_rgb);

      resiB = _mm256_cvtps_epi32(resB); // no extra rounding, cvtps rounds to nearest
      resiB_2 = _mm256_cvtps_epi32(resB_2); // no extra rounding, cvtps rounds to nearest

      resiB = _mm256_packus_epi32(resiB, resiB_2);
      if constexpr (bits_per_pixel < 16) // albeit 10-14 bit have another function, make this general
        resiB = _mm256_min_epi16(resiB, limit); // clamp 10,12,14 bit
      _mm256_store_si256(reinterpret_cast<__m256i *>(dstp[1] + x), resiB);

      // *R*
      resR = _mm256_mul_ps(y, mat_yR); // mul_y
      resR = _mm256_fmadd_ps(u, mat_uR, resR); // mul_u
      resR = _mm256_fmadd_ps(v, mat_vR, resR); // mul_v
      if (has_offset_rgb)
        resR = _mm256_add_ps(resR, offset_rgb);

      resR_2 = _mm256_mul_ps(y_2, mat_yR); // mul_y
      resR_2 = _mm256_fmadd_ps(u_2, mat_uR, resR_2); // mul_u
      resR_2 = _mm256_fmadd_ps(v_2, mat_vR, resR_2); // mul_v
      if (has_offset_rgb)
        resR_2 = _mm256_add_ps(resR_2, offset_rgb);

      resiR = _mm256_cvtps_epi32(resR); // no extra rounding, cvtps rounds to nearest
      resiR_2 = _mm256_cvtps_epi32(resR_2); // no extra rounding, cvtps rounds to nearest

      resiR = _mm256_packus_epi32(resiR, resiR_2);
      if constexpr (bits_per_pixel < 16) // albeit 10-14 bit have another function, make this general
        resiR = _mm256_min_epi16(resiR, limit); // clamp 10,12,14 bit
      _mm256_store_si256(reinterpret_cast<__m256i *>(dstp[2] + x), resiR);

    }
    srcp[0] += srcPitch[0];
    srcp[1] += srcPitch[1];
    srcp[2] += srcPitch[2];
    dstp[0] += dstPitch[0];
    dstp[1] += dstPitch[1];
    dstp[2] += dstPitch[2];
  }
}

//instantiate
//template<int bits_per_pixel>
template void convert_yuv_to_planarrgb_uint16_avx2<16>(BYTE* (&dstp)[3], int(&dstPitch)[3], const BYTE* (&srcp)[3], const int(&srcPitch)[3], int width, int height, const ConversionMatrix& m);

#ifdef _MSC_VER
#pragma warning(pop)
#endif
