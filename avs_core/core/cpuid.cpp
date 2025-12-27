// Avisynth v1.0 beta.  Copyright 2000 Ben Rudiak-Gould.
// http://www.math.berkeley.edu/~benrg/avisynth.html

//	VirtualDub - Video processing and capture application
//	Copyright (C) 1998-2000 Avery Lee
//
//	This program is free software; you can redistribute it and/or modify
//	it under the terms of the GNU General Public License as published by
//	the Free Software Foundation; either version 2 of the License, or
//	(at your option) any later version.
//
//	This program is distributed in the hope that it will be useful,
//	but WITHOUT ANY WARRANTY; without even the implied warranty of
//	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//	GNU General Public License for more details.
//
//	You should have received a copy of the GNU General Public License
//	along with this program; if not, write to the Free Software
//	Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

#include <avs/cpuid.h>
#include <avs/config.h>
#include <cstdint>
#include <cstddef>

// Neon/Dotprod request in WinAPI ...
#if defined(ARM64) && defined(AVS_WINDOWS)
#include <Windows.h>
#include <processthreadsapi.h>
#endif

// --- Platform-specific headers for x86/x64 ---
#if defined(X86_32) || defined(X86_64)

#ifdef AVS_WINDOWS
#include <intrin.h> // MSVC/Clang-CL
#else
// Non-Windows (GCC, Clang-GNU, etc.)
#include <x86intrin.h>
#include <cpuid.h>
#undef __cpuid

static inline void __cpuid(int cpuinfo[4], int leaf) {
  unsigned int eax, ebx, ecx, edx;
  // for deeper leaves __get_cpuid is not enough
  __get_cpuid_count(leaf, 0, &eax, &ebx, &ecx, &edx);
  cpuinfo[0] = eax;
  cpuinfo[1] = ebx;
  cpuinfo[2] = ecx;
  cpuinfo[3] = edx;
}
#endif // AVS_WINDOWS

#endif // defined(X86_32) || defined(X86_64)

// --- Platform-specific headers for ARM64 ---
#if defined(ARM64)
#if defined(AVS_LINUX) || defined(AVS_BSD)
// HWCAP values are needed for Linux/BSD
#include <sys/auxv.h>
// Note: <asm/hwcap.h> may be required on some systems, 
// but AT_HWCAP and values like HWCAP_DOTPROD are often found in sys/auxv.h or defined by toolchain.
// We assume standard GNU/Clang behavior where flags like HWCAP_DOTPROD are available.
#include <asm/hwcap.h> 
#elif defined(AVS_MACOS)
// macOS/Apple Silicon uses sysctl for features
#include <sys/types.h>
#include <sys/sysctl.h>
#endif
#endif

#define IS_BIT_SET(bitfield, bit) ((bitfield) & (1<<(bit)) ? true : false)

#if defined(X86_32) || defined(X86_64)
static uint32_t get_xcr0()
{
    uint32_t xcr0;
    // _XCR_XFEATURE_ENABLED_MASK: 0
#if defined(GCC) || defined(CLANG)
    __asm__("xgetbv" : "=a" (xcr0) : "c" (0) : "%edx");
#else
    xcr0 = (uint32_t)_xgetbv(0);
#endif
    return xcr0;
}

// --- Helper for __cpuid_count/ex (required for Leaf 4 sub-leaves) ---
static void __cpuid_count_wrapper(int info[4], int leaf, int subleaf) {
#ifdef AVS_WINDOWS
  // MSVC uses __cpuidex
  __cpuidex(info, leaf, subleaf);
#elif defined(GCC) || defined(CLANG)
  // GCC/Clang uses __cpuid_count
  __cpuid_count(leaf, subleaf, info[0], info[1], info[2], info[3]);
#else
  // Fallback or error if no intrinsic is available
  info[0] = info[1] = info[2] = info[3] = 0;
#endif
}

// Helper function to determine the AVX10 version (e.g., 10.2)
// Returns the minor version (0 for none, 1 for 10.1, 2 for 10.2, etc.)
static int get_avx10_minor_version() {
  int info[4];
  int max_sub_leaf_7;

  // 1. Check max CPUID Leaf 7 sub-leaf to ensure Sub-leaf 1 is available.
  // Call Leaf 7, Sub-leaf 0. Max Sub-leaf is returned in EAX.
  __cpuid_count_wrapper(info, 7, 0);
  max_sub_leaf_7 = info[0]; // EAX holds the Max Sub-leaf

  // If max sub-leaf is less than 1, we cannot query the AVX10 version.
  if (max_sub_leaf_7 < 1) {
    return 0; // AVX10 version not supported or enumerable
  }

  // 2. Query CPUID Leaf 7, Sub-leaf 1 for the AVX10 Version (Major/Minor)
  __cpuid_count_wrapper(info, 7, 1);

  // EAX[31:24] = Major Version
  int major_version = (info[0] >> 24) & 0xFF;

  // EAX[23:16] = Minor Version
  int minor_version = (info[0] >> 16) & 0xFF;

  // AVX10 has a major version of 10.
  // This check confirms the version returned is for AVX10.
  if (major_version == 10) {
    return minor_version;
  }

  return 0; // Not an AVX10 major version
}
#endif // defined(X86_32) || defined(X86_64)

// ------------------------------------------------------------------
// Core ARMv8 Feature Detection Function
// ------------------------------------------------------------------
#if defined(ARM64)
static int64_t ARMCheckForExtensions()
{
  int64_t result = 0;

  // Tier 1: CPUF_ARM_NEON (Mandatory for AArch64)
  // We can assume NEON for any successful ARM64 build.
  result |= CPUF_ARM_NEON;

#if defined(AVS_LINUX) || defined(AVS_BSD)

  // Linux/BSD HWCAP detection (uses AT_HWCAP/AT_HWCAP2)
  // HWCAP_NEON (Basic NEON) is covered by the assumption above.
  unsigned long hwcap = getauxval(AT_HWCAP);

  // Tier 2: CPUF_ARM_DOTPROD (Dot Product)
  if (hwcap & HWCAP_DOTPROD) {
    result |= CPUF_ARM_DOTPROD;
  }

  // Tier 3: CPUF_ARM_SVE2
  // Check for SVE2. Note: SVE support often implies SVE2 is available on newer chips.
  // The official SVE2 flag is often checked on hwcap2 on older systems, but is moved to hwcap
  // on newer kernels. We check the most reliable ones.
  // Note: On newer kernels, HWCAP2 is deprecated in favor of HWCAP
  if (hwcap & HWCAP_SVE2) {
    result |= CPUF_ARM_SVE2;
  }
  else if (hwcap & HWCAP_SVE) {
    // If only SVE is found, SVE2 may be implied or the next best thing.
    // However, for strict SVE2 code path, we only rely on the SVE2 flag if available.
    // For now, let's keep it simple and only check for SVE2, as it's the target.
    // Future: Could add a distinct CPUF_ARM_SVE if needed.
  }

#elif defined(AVS_MACOS)

  // macOS (Apple Silicon) detection via sysctlbyname.
  int value = 0;
  size_t len = sizeof(value);

  // Tier 2: CPUF_ARM_DOTPROD (Dot Product)
  // Check for the DOTPROD feature, which is guaranteed on modern Apple Silicon.
  // The sysctl function returns 0 on success and populates 'value'.
  if (sysctlbyname("hw.optional.dotprod", &value, &len, NULL, 0) == 0 && value) {
    result |= CPUF_ARM_DOTPROD;
  }

  // Tier 3: CPUF_ARM_SVE2 (SVE/SVE2 is not currently (2025) implemented by Apple)
  // No check required here as the hardware does not support it.

#elif defined(AVS_WINDOWS)

  // Windows ARM64 detection using IsProcessorFeaturePresent
  // Note: Windows ARM does not currently (late 2025) expose SVE/SVE2 features via WinAPI flags.

  // Tier 2: CPUF_ARM_DOTPROD (Dot Product)
#if defined(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE)
  if (IsProcessorFeaturePresent(PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE)) {
    result |= CPUF_ARM_DOTPROD;
  }
#elif defined(PF_ARM_V8_DOTPROD_INSTRUCTIONS_AVAILABLE)
  if (IsProcessorFeaturePresent(PF_ARM_V8_DOTPROD_INSTRUCTIONS_AVAILABLE)) {
    result |= CPUF_ARM_DOTPROD;
  }
#endif
#else
  // Fallback for other ARM systems (e.g., Windows ARM64)
  // If no specific OS mechanism is available, we only rely on the mandatory NEON flag.
#endif

  return result;
}
#endif // defined(ARM64)


// ------------------------------------------------------------------
// Core x86/x64 Feature Detection Function
// (Refactored for cleaner architecture switch)
// ------------------------------------------------------------------
#if defined(X86_32) || defined(X86_64)
static int64_t X86CheckForExtensions()
{
  int64_t result = 0;
  int cpuinfo[4];

#if defined(X86_32) || defined(X86_64)
  // Check CPUID Leaf 1
  __cpuid(cpuinfo, 1);
  if (IS_BIT_SET(cpuinfo[3], 0))
    result |= CPUF_FPU;
  if (IS_BIT_SET(cpuinfo[3], 23))
    result |= CPUF_MMX;
  if (IS_BIT_SET(cpuinfo[3], 25))
    result |= CPUF_SSE | CPUF_INTEGER_SSE;
  if (IS_BIT_SET(cpuinfo[3], 26))
    result |= CPUF_SSE2;
  if (IS_BIT_SET(cpuinfo[2], 0))
    result |= CPUF_SSE3;
  if (IS_BIT_SET(cpuinfo[2], 9))
    result |= CPUF_SSSE3;
  if (IS_BIT_SET(cpuinfo[2], 19))
    result |= CPUF_SSE4_1;
  if (IS_BIT_SET(cpuinfo[2], 20))
    result |= CPUF_SSE4_2;
  if (IS_BIT_SET(cpuinfo[2], 22))
    result |= CPUF_MOVBE;
  if (IS_BIT_SET(cpuinfo[2], 23))
    result |= CPUF_POPCNT;
  if (IS_BIT_SET(cpuinfo[2], 25))
    result |= CPUF_AES;
  if (IS_BIT_SET(cpuinfo[2], 29))
    result |= CPUF_F16C;

  // AVX and XCR0 Check (Needed for AVX2 and AVX-512)
  bool xgetbv_supported = IS_BIT_SET(cpuinfo[2], 27);
  bool avx_supported = IS_BIT_SET(cpuinfo[2], 28);
  if (xgetbv_supported && avx_supported)
  {
    uint32_t xgetbv0_32 = get_xcr0();

    // Check OS support for AVX (XMM and YMM state)
    if ((xgetbv0_32 & 0x6u) == 0x6u) {
      result |= CPUF_AVX;
      if (IS_BIT_SET(cpuinfo[2], 12)) result |= CPUF_FMA3;

      __cpuid_count_wrapper(cpuinfo, 7, 0);
      if (IS_BIT_SET(cpuinfo[1], 5)) result |= CPUF_AVX2;
    }

    // Check OS support for AVX-512 (OPMASK, ZMM0-ZMM15, ZMM16-ZMM31 states)
    if ((xgetbv0_32 & (0x7u << 5)) && (xgetbv0_32 & (0x3u << 1)))
    {
      // Leaf 7, Sub-leaf 0 results are already in cpuinfo (from AVX2 check)

      // --- EBX: Core Base Features & Specialized Features ---
      if (IS_BIT_SET(cpuinfo[1], 16)) result |= CPUF_AVX512F;
      if (IS_BIT_SET(cpuinfo[1], 17)) result |= CPUF_AVX512DQ;
      if (IS_BIT_SET(cpuinfo[1], 21)) result |= CPUF_AVX512IFMA;
      if (IS_BIT_SET(cpuinfo[1], 26)) result |= CPUF_AVX512PF;
      if (IS_BIT_SET(cpuinfo[1], 27)) result |= CPUF_AVX512ER;
      if (IS_BIT_SET(cpuinfo[1], 28)) result |= CPUF_AVX512CD;
      if (IS_BIT_SET(cpuinfo[1], 30)) result |= CPUF_AVX512BW;
      if (IS_BIT_SET(cpuinfo[1], 31)) result |= CPUF_AVX512VL;

      // --- ECX: ICL/RKL Features (VBMI, VNNI, Cryptography) ---
      if (IS_BIT_SET(cpuinfo[2], 1)) result |= CPUF_AVX512VBMI;
      if (IS_BIT_SET(cpuinfo[2], 6)) result |= CPUF_AVX512VBMI2;

      /* Deprecated:
       * AVX-512 Vector Pair to Two Intersect (VP2INTERSECT) instruction set
       * has been found to be slower than alternative implementations using
       * existing instructions. Newer CPUs may not implement this feature.
      if (IS_BIT_SET(cpuinfo[2], 2)) result |= CPUF_AVX512VP2INTERSECT;
       */

      const bool has_avx512_crypto =
        IS_BIT_SET(cpuinfo[2], 7) ||
        IS_BIT_SET(cpuinfo[2], 8) ||
        IS_BIT_SET(cpuinfo[2], 9);
      /* Avisynth don't use crypto features.
      if (IS_BIT_SET(cpuinfo[2], 8)) result |= CPUF_AVX512GFNI;
      if (IS_BIT_SET(cpuinfo[2], 9)) result |= CPUF_AVX512VAES;
      if (IS_BIT_SET(cpuinfo[2], 10)) result |= CPUF_AVX512VPCLMULQDQ;
      */
      if (IS_BIT_SET(cpuinfo[2], 11)) result |= CPUF_AVX512VNNI;
      if (IS_BIT_SET(cpuinfo[2], 12)) result |= CPUF_AVX512BITALG;
      if (IS_BIT_SET(cpuinfo[2], 14)) result |= CPUF_AVX512VPOPCNTDQ;

      // --- AVX-512 (Leaf 7, Sub-leaf 1) ---
      __cpuid_count_wrapper(cpuinfo, 7, 1);

      // EAX:

      /* Deprecated:
       * AVX-512 4-way VNNI with Word Granularity (4VNNIW) and
       * AVX-512 4-way Fused Multiply-Add Single precision (4FMAPS)
       * have been deprecated and replaced by more versatile instructions
       * in AVX10. Newer CPUs may not implement these features.
      if (IS_BIT_SET(cpuinfo[0], 4)) result |= CPUF_AVX5124VNNIW;
      if (IS_BIT_SET(cpuinfo[0], 5)) result |= CPUF_AVX5124FMAPS;
      */

      // EDX:
      if (IS_BIT_SET(cpuinfo[3], 16)) result |= CPUF_AVX512FP16;
      if (IS_BIT_SET(cpuinfo[3], 17)) result |= CPUF_AVX512BF16;

      int avx10_minor = get_avx10_minor_version(); // 0 if no AVX10

      // --- Composite Feature Flags and AVX10 ---

      constexpr int64_t avx512_base_mask =
        CPUF_AVX512F |
        CPUF_AVX512CD |
        CPUF_AVX512BW |
        CPUF_AVX512DQ |
        CPUF_AVX512VL;

      constexpr int64_t avx512_fast_core_mask =
        CPUF_AVX512VBMI |
        CPUF_AVX512VNNI |
        CPUF_AVX512VBMI2 |
        CPUF_AVX512BITALG |
        CPUF_AVX512VPOPCNTDQ;

      // 1. Check for Base AVX-512
      if ((result & avx512_base_mask) == avx512_base_mask) {
        result |= CPUF_AVX512_BASE;

        // 2. Check for AVX10
        if (avx10_minor >= 1) {
          // AVX10.x automatically sets the FAST flag and the AVX10 flag
          // AVX10 version is queried separately from GetCPUFlags.
          //result |= CPUF_AVX10; // not yet, RFU
          result |= CPUF_AVX512_FAST;
        }
        // 3. Check for AVX512_FAST (Pre-AVX10 minimum Ice Lake)
        // Base + Core ICL + Crypto (to distinguish from older server CPUs)
        else if (((result & avx512_fast_core_mask) == avx512_fast_core_mask) && has_avx512_crypto)
        {
          result |= CPUF_AVX512_FAST;
        }
      }

      // GCC/clang compiler flags for matching CPUF_AVX512_FAST:
      //" -mfma -mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl -mavx512vnni -mavx512vbmi -mavx512vbmi2 -mavx512bitalg -mavx512vpopcntdq "
    }
  }
#else
  result |= CPUF_FORCE;

  return result;
#endif

#if defined(X86_32) || defined(X86_64)
  // Check CPUID Extended Leaf 0x80000001 (for 3DNow! and FMA4)
  __cpuid(cpuinfo, 0x80000000);
  if (cpuinfo[0] >= 0x80000001)
  {
    __cpuid(cpuinfo, 0x80000001);

    if (IS_BIT_SET(cpuinfo[3], 31))
      result |= CPUF_3DNOW;

    if (IS_BIT_SET(cpuinfo[3], 30))
      result |= CPUF_3DNOW_EXT;

    if (IS_BIT_SET(cpuinfo[3], 22))
      result |= CPUF_INTEGER_SSE;

    if (result & CPUF_AVX) {
      if (IS_BIT_SET(cpuinfo[2], 16))
        result |= CPUF_FMA4;
    }
  }
#endif

  return result;
}
#endif // defined(X86_32) || defined(X86_64)

// ------------------------------------------------------------------
// Master Feature Detection Function (Dispatcher)
// ------------------------------------------------------------------
static int64_t CPUCheckForExtensions()
{
  int64_t result = 0;

#if defined(X86_32) || defined(X86_64)
  result |= X86CheckForExtensions();
#elif defined(ARM64)
  result |= ARMCheckForExtensions();
#else
  // Fallback for architectures without specific detection implemented
#endif

  return result;
}

// ------------------------------------------------------------------
// Core L2 Cache Detection Function
// ------------------------------------------------------------------
static size_t DetectL2CacheSize()
{
#if defined(X86_32) || defined(X86_64)
  int info[4];

  // -------------------------------------------------------
  // 1. PRIMARY METHOD: Deterministic Cache Parameters (Leaf 4)
  //    (Modern, cross-vendor, and supports topology)
  // -------------------------------------------------------
  for (int i = 0; ; ++i) {
    // We use the helper that supports sub-leaves (i)
    __cpuid_count_wrapper(info, 0x4, i);

    // EAX[4:0] = Cache Type: 1=Data, 2=Instruction, 3=Unified
    int cache_type = info[0] & 0x1F;

    // EAX[7:5] = Cache Level: 1=L1, 2=L2, 3=L3, ...
    int cache_level = (info[0] >> 5) & 0b111;

    // Check for end of list (type 0)
    if (cache_type == 0) {
      break;
    }

    // We look for Cache Level 2, regardless of whether it's reported as Unified (3) or Instruction (2).
    if (cache_level == 2) {
      // Cache Size (Bytes) = (Ways + 1) * (Partitions + 1) * (Line Size + 1) * (Sets + 1)

      size_t line_size = (info[1] & 0xFFF) + 1;           // EBX[11:0]
      size_t partitions = ((info[1] >> 12) & 0x3FF) + 1;   // EBX[21:12]
      size_t ways = ((info[1] >> 22) & 0x3FF) + 1;   // EBX[31:22]
      size_t sets = (size_t)info[2] + 1;             // ECX[31:0]

      return ways * partitions * line_size * sets;
    }
  }

  // -------------------------------------------------------
  // 2. FALLBACK METHOD: AMD Extended Cache (Leaf 80000006)
  //    (Legacy method, but reliable for older AMD CPUs)
  // -------------------------------------------------------
  __cpuid(info, 0x80000000);
  // Check if the CPU supports extended leaf 80000006
  if (info[0] >= 0x80000006) {
    __cpuid(info, 0x80000006);
    // ECX[31:16] is L2 cache size in KB. Convert to bytes.
    return (size_t)(info[2] >> 16) * 1024;
  }

  // 3. If neither method worked, return 0.
  return 0;
#elif defined(ARM64)
  // Cache detection on ARM is highly vendor/OS-specific.
  // Returning 0 is a safe default for a portable cross-platform implementation.
  return 0;
#else
  return 0;
#endif
}

class _CPUFlags
{
private:
  size_t L2CacheSize; // in bytes
  int64_t lCPUExtensionsAvailable;
  _CPUFlags() {
    lCPUExtensionsAvailable = CPUCheckForExtensions();
    L2CacheSize = DetectL2CacheSize();
  }

public:
  static _CPUFlags& getInstance() {
    static _CPUFlags theInstance;
    return theInstance;
  }

  int GetCPUFlags() {
    return lCPUExtensionsAvailable & 0xFFFFFFFF;
  }

  int64_t GetCPUFlagsEx() {
    return lCPUExtensionsAvailable;
  }

  void SetCPUFlags(int64_t new_flags) {
    lCPUExtensionsAvailable = new_flags;
  }

  size_t GetL2CacheSize() {
    return L2CacheSize;
  }
};

int GetCPUFlags() {
  return _CPUFlags::getInstance().GetCPUFlags();
}

int64_t GetCPUFlagsEx() {
  return _CPUFlags::getInstance().GetCPUFlagsEx();
}

size_t GetL2CacheSize() {
  return _CPUFlags::getInstance().GetL2CacheSize();
}
