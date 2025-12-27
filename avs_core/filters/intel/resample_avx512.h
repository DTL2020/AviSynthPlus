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

bool resize_h_planar_float_avx512_gather_permutex_vstripe_ks4_check(ResamplingProgram* program);
void resize_h_planar_float_avx512_transpose_vstripe_ks4(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
void resize_h_planar_float_avx512_permutex_vstripe_ks4(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);

//bool resize_h_planar_float_avx512_gather_permutex_vstripe_ks8_check(ResamplingProgram* program);
bool resize_h_planar_float_avx512_gather_permutex_vstripe_ks8_check(ResamplingProgram* program, int iSamplesPerPass);
void resize_h_planar_float_avx512_transpose_vstripe_ks8(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
void resize_h_planar_float_avx512_permutex_vstripe_ks8(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
void resize_h_planar_float_avx512_permutex_vstripe_2s8_ks8(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);

void resize_v_avx512_planar_float(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel);
void resize_v_avx512_planar_float_w_sr(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int target_height, int bits_per_pixel);

void resizer_h_avx512_generic_float_pix16_sub4_ks_4_8_16(BYTE * dst8, const BYTE * src8, int dst_pitch, int src_pitch, ResamplingProgram * program, int width, int height, int bits_per_pixel);

bool resize_h_planar_float_avx512_gather_permutex_vstripe_ks16_check(ResamplingProgram* program, int iSamplesPerPass);
bool resize_h_planar_float_avx512_gather_permutex_vstripe_check(ResamplingProgram* program, int iSamplesInTheGroup, int iSourceSamplesToAcsess, int iKernelSize);
void resize_h_planar_float_avx512_permutex_vstripe_ks16(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);
void resize_h_planar_float_avx512_permutex_vstripe_s8_ks16(BYTE* dst8, const BYTE* src8, int dst_pitch, int src_pitch, ResamplingProgram* program, int width, int height, int bits_per_pixel);

#endif // __Resample_AVX512_H__
