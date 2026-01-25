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

#ifndef __Resample_AVX512_H__
#define __Resample_AVX512_H__

#include <avisynth.h>
#include "../resample_functions.h"

// transpose and hi/lo unpack of resampling program for permute-based H-resizers
// allocate and fill pixel_coefficient_AVX512_H coeffs buffer of the ResamplingProgram
void resize_prepare_coeffs_AVX512_H(ResamplingProgram* p, IScriptEnvironment* env, int iSamplesInTheGroup, int iGroupsCount);

// generic - non specialized filter size
void resizer_h_avx512_generic_uint8_t(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel); // PF debug
template<bool lessthan16bit>
void resizer_h_avx512_generic_uint16_t(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel); // PF debug

// 8bit samples
template<bool bVNNI>
void resize_h_planar_uint8_avx512_permutex_vstripe_mpz_ks4(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);

template<bool bVNNI>
void resize_h_planar_uint8_avx512_permutex_vstripe_mpz_ks8(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);

void resize_h_planar_uint8_avx512_permutex_vstripe_2s32_ks8(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);

template<bool bVNNI>
void resize_h_planar_uint8_avx512_permutex_vstripe_mpz_ks16(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);

template<bool bVNNI>
void resize_h_planar_uint8_avx512_permutex_vstripe_2s32_ks64(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);

template<bool bVNNI>
void resize_h_planar_uint8_avx512_permutex_vstripe_mpz_2s32_ks64(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);

// 16bit samples
template<bool lessthan16bit, bool bVNNI>
void resize_h_planar_uint16_avx512_permutex_vstripe_mp_2s32_ks4(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);

template<bool lessthan16bit, bool bVNNI>
void resize_h_planar_uint16_avx512_permutex_vstripe_mp_2s32_ks8(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);

template<bool lessthan16bit, bool bVNNI>
void resize_h_planar_uint16_avx512_permutex_vstripe_mp_4s16_ks8(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);

template<bool lessthan16bit, bool bVNNI>
void resize_h_planar_uint16_avx512_permutex_vstripe_mp_ks16(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);

template<bool lessthan16bit, bool bVNNI>
void resize_h_planar_uint16_avx512_permutex_vstripe_mp_4s16_ks48(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);

// float32 samples
void resize_h_planar_float_avx512_transpose_vstripe_ks4(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
void resize_h_planar_float_avx512_permutex_vstripe_ks4(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);

void resize_h_planar_float_avx512_transpose_vstripe_ks8(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
void resize_h_planar_float_avx512_permutex_vstripe_ks8(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
void resize_h_planar_float_avx512_permutex_vstripe_2s8_ks8(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);

void resize_h_planar_float_avx512_permutex_vstripe_ks16(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
void resize_h_planar_float_avx512_permutex_vstripe_2s8_ks16(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);

// Vertical

void resize_v_avx512_planar_float(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel);
void resize_v_avx512_planar_float_w_sr(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel);

// uint8_t
// AVX512base - F + BW
void resize_v_avx512_planar_uint8_t_w_sr(BYTE* AVS_RESTRICT dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel);

// AVX512fast - F + VBMI + VNNI
void resize_v_avx512fast_planar_uint8_t_w(BYTE* AVS_RESTRICT dst, const BYTE* src, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel);

// uint16_t
template<bool lessthan16bit>
void resize_v_avx512_planar_uint16_t_w_sr(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel);

// AVX512base or AVX512fast based on bVNNI
template<bool lessthan16bit, bool bVNNI>
void resize_v_avx512_planar_uint16_t_w(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel);

void resizer_h_avx512_generic_float_pix16_sub4_ks_4_8_16(BYTE * dst8, const BYTE * src8, int dst_pitch, int src_pitch, ResamplingProgram * program, int width, int height, int bits_per_pixel);

#endif // __Resample_AVX512_H__
