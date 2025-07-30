// Sistema de intrinsics otimizados para operações de bitboard
// Performance crítica: operações básicas de bitboard ultra-rápidas

use crate::types::Bitboard;

// ============================================================================
// INTRINSICS DE ALTA PERFORMANCE PARA BITBOARDS
// ============================================================================

/// Conta o número de bits setados (popcount) usando intrinsics quando disponível
#[inline(always)]
pub fn popcount(bb: Bitboard) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("popcnt") {
            unsafe {
                std::arch::x86_64::_popcnt64(bb as i64) as u32
            }
        } else {
            bb.count_ones()
        }
    }
    #[cfg(target_arch = "x86")]
    {
        if is_x86_feature_detected!("popcnt") {
            unsafe {
                let low = bb as u32;
                let high = (bb >> 32) as u32;
                std::arch::x86::_popcnt32(low as i32) as u32 +
                    std::arch::x86::_popcnt32(high as i32) as u32
            }
        } else {
            bb.count_ones()
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        bb.count_ones()
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")))]
    {
        bb.count_ones()
    }
}

/// Encontra o índice do bit menos significativo (LSB) usando intrinsics
#[inline(always)]
pub fn trailing_zeros(bb: Bitboard) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("bmi1") {
            unsafe {
                std::arch::x86_64::_tzcnt_u64(bb) as u32
            }
        } else {
            bb.trailing_zeros()
        }
    }
    #[cfg(target_arch = "x86")]
    {
        if is_x86_feature_detected!("bmi1") {
            unsafe {
                if bb == 0 {
                    64
                } else {
                    let low = bb as u32;
                    if low != 0 {
                        std::arch::x86::_tzcnt_u32(low)
                    } else {
                        32 + std::arch::x86::_tzcnt_u32((bb >> 32) as u32)
                    }
                }
            }
        } else {
            bb.trailing_zeros()
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    {
        bb.trailing_zeros()
    }
}

/// Encontra o índice do bit mais significativo (MSB) usando intrinsics
#[inline(always)]
pub fn leading_zeros(bb: Bitboard) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("lzcnt") {
            unsafe {
                std::arch::x86_64::_lzcnt_u64(bb) as u32
            }
        } else {
            bb.leading_zeros()
        }
    }
    #[cfg(target_arch = "x86")]
    {
        if is_x86_feature_detected!("lzcnt") {
            unsafe {
                if bb == 0 {
                    64
                } else {
                    let high = (bb >> 32) as u32;
                    if high != 0 {
                        std::arch::x86::_lzcnt_u32(high)
                    } else {
                        32 + std::arch::x86::_lzcnt_u32(bb as u32)
                    }
                }
            }
        } else {
            bb.leading_zeros()
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    {
        bb.leading_zeros()
    }
}

/// Remove e retorna o LSB (pop LSB) usando intrinsics para máxima performance
#[inline(always)]
pub fn pop_lsb(bb: &mut Bitboard) -> u32 {
    let lsb_index = trailing_zeros(*bb);
    *bb &= *bb - 1; // Remove o LSB
    lsb_index
}

/// Isola o LSB (retorna apenas o bit menos significativo)
#[inline(always)]
pub fn isolate_lsb(bb: Bitboard) -> Bitboard {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("bmi1") {
            unsafe {
                std::arch::x86_64::_blsi_u64(bb)
            }
        } else {
            bb & bb.wrapping_neg()
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        bb & bb.wrapping_neg()
    }
}

/// Reset do LSB (remove o bit menos significativo) usando intrinsics
#[inline(always)]
pub fn reset_lsb(bb: Bitboard) -> Bitboard {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("bmi1") {
            unsafe {
                std::arch::x86_64::_blsr_u64(bb)
            }
        } else {
            bb & (bb - 1)
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        bb & (bb - 1)
    }
}

/// Verifica se o bitboard tem apenas um bit setado (é potência de 2)
#[inline(always)]
pub fn is_single_bit(bb: Bitboard) -> bool {
    bb != 0 && (bb & (bb - 1)) == 0
}

/// Verifica se o bitboard está vazio
#[inline(always)]
pub fn is_empty(bb: Bitboard) -> bool {
    bb == 0
}

/// Verifica se o bitboard não está vazio
#[inline(always)]
pub fn is_not_empty(bb: Bitboard) -> bool {
    bb != 0
}

// ============================================================================
// OPERAÇÕES AVANÇADAS DE BITBOARD COM INTRINSICS
// ============================================================================

/// Paraleliza operações bit por bit usando PEXT/PDEP quando disponível
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn parallel_extract(source: Bitboard, mask: Bitboard) -> Bitboard {
    if is_x86_feature_detected!("bmi2") {
        unsafe {
            std::arch::x86_64::_pext_u64(source, mask)
        }
    } else {
        // Fallback manual para CPUs sem BMI2
        let mut result = 0u64;
        let mut src = source;
        let mut msk = mask;
        let mut bit_pos = 0;

        while msk != 0 {
            if (src & 1) != 0 {
                result |= 1u64 << bit_pos;
                bit_pos += 1;
            }
            src >>= 1;
            msk &= msk - 1;
        }
        result
    }
}

/// Paraleliza depósito de bits usando PDEP quando disponível
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn parallel_deposit(source: Bitboard, mask: Bitboard) -> Bitboard {
    if is_x86_feature_detected!("bmi2") {
        unsafe {
            std::arch::x86_64::_pdep_u64(source, mask)
        }
    } else {
        // Fallback manual para CPUs sem BMI2
        let mut result = 0u64;
        let mut src = source;
        let mut msk = mask;

        while msk != 0 {
            let lsb = msk & msk.wrapping_neg();
            if (src & 1) != 0 {
                result |= lsb;
            }
            src >>= 1;
            msk &= msk - 1;
        }
        result
    }
}

// ============================================================================
// FUNÇÕES DE UTILIDADE PARA BITBOARDS
// ============================================================================

/// Itera sobre todos os bits setados em um bitboard de forma eficiente
pub struct BitboardIterator {
    bb: Bitboard,
}

impl BitboardIterator {
    #[inline(always)]
    pub fn new(bb: Bitboard) -> Self {
        Self { bb }
    }
}

impl Iterator for BitboardIterator {
    type Item = u8;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.bb == 0 {
            None
        } else {
            let square = trailing_zeros(self.bb) as u8;
            self.bb = reset_lsb(self.bb);
            Some(square)
        }
    }
}

/// Trait para operações de bitboard otimizadas
pub trait BitboardOps {
    fn iter_squares(self) -> BitboardIterator;
    fn popcount_fast(self) -> u32;
    fn lsb_fast(self) -> u32;
    fn msb_fast(self) -> u32;
    fn is_single_bit_fast(self) -> bool;
    fn isolate_lsb_fast(self) -> Bitboard;
    fn reset_lsb_fast(self) -> Bitboard;
}

impl BitboardOps for Bitboard {
    #[inline(always)]
    fn iter_squares(self) -> BitboardIterator {
        BitboardIterator::new(self)
    }

    #[inline(always)]
    fn popcount_fast(self) -> u32 {
        popcount(self)
    }

    #[inline(always)]
    fn lsb_fast(self) -> u32 {
        trailing_zeros(self)
    }

    #[inline(always)]
    fn msb_fast(self) -> u32 {
        63 - leading_zeros(self)
    }

    #[inline(always)]
    fn is_single_bit_fast(self) -> bool {
        is_single_bit(self)
    }

    #[inline(always)]
    fn isolate_lsb_fast(self) -> Bitboard {
        isolate_lsb(self)
    }

    #[inline(always)]
    fn reset_lsb_fast(self) -> Bitboard {
        reset_lsb(self)
    }
}
// ============================================================================
// FUNÇÕES DE DETECÇÃO DE FEATURES
// ============================================================================

/// Verifica se as extensões BMI1/BMI2 estão disponíveis
pub fn has_bmi_support() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("bmi1") && is_x86_feature_detected!("bmi2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Verifica se POPCNT está disponível
pub fn has_popcnt_support() -> bool {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        is_x86_feature_detected!("popcnt")
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    {
        false
    }
}

// ============================================================================
// ADVANCED CPU OPTIMIZATIONS - SIMD AND PARALLEL OPERATIONS
// ============================================================================

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD-optimized batch operations for multiple bitboards
#[cfg(target_arch = "x86_64")]
pub struct SimdBitboardProcessor {
    pub avx2_support: bool,
    pub sse42_support: bool,
}

#[cfg(target_arch = "x86_64")]
impl SimdBitboardProcessor {
    pub fn new() -> Self {
        Self {
            avx2_support: is_x86_feature_detected!("avx2"),
            sse42_support: is_x86_feature_detected!("sse4.2"),
        }
    }
    
    /// Process 4 bitboards simultaneously using AVX2
    #[inline(always)]
    pub unsafe fn batch_popcount_avx2(&self, bitboards: &[Bitboard; 4]) -> [u32; 4] {
        if self.avx2_support {
            let values = _mm256_loadu_si256(bitboards.as_ptr() as *const __m256i);
            
            // Use POPCNT on each 64-bit lane
            let mut results = [0u32; 4];
            let extracted = std::mem::transmute::<__m256i, [u64; 4]>(values);
            
            for i in 0..4 {
                results[i] = _popcnt64(extracted[i] as i64) as u32;
            }
            
            results
        } else {
            [
                popcount(bitboards[0]),
                popcount(bitboards[1]),
                popcount(bitboards[2]),
                popcount(bitboards[3]),
            ]
        }
    }
    
    /// Parallel XOR operations on multiple bitboards
    #[inline(always)]
    pub unsafe fn batch_xor_avx2(&self, a: &[Bitboard; 4], b: &[Bitboard; 4]) -> [Bitboard; 4] {
        if self.avx2_support {
            let va = _mm256_loadu_si256(a.as_ptr() as *const __m256i);
            let vb = _mm256_loadu_si256(b.as_ptr() as *const __m256i);
            let result = _mm256_xor_si256(va, vb);
            
            std::mem::transmute::<__m256i, [u64; 4]>(result)
        } else {
            [a[0] ^ b[0], a[1] ^ b[1], a[2] ^ b[2], a[3] ^ b[3]]
        }
    }
    
    /// Parallel AND operations on multiple bitboards
    #[inline(always)]
    pub unsafe fn batch_and_avx2(&self, a: &[Bitboard; 4], b: &[Bitboard; 4]) -> [Bitboard; 4] {
        if self.avx2_support {
            let va = _mm256_loadu_si256(a.as_ptr() as *const __m256i);
            let vb = _mm256_loadu_si256(b.as_ptr() as *const __m256i);
            let result = _mm256_and_si256(va, vb);
            
            std::mem::transmute::<__m256i, [u64; 4]>(result)
        } else {
            [a[0] & b[0], a[1] & b[1], a[2] & b[2], a[3] & b[3]]
        }
    }
}

// ============================================================================
// CACHE-FRIENDLY DATA STRUCTURES AND ALGORITHMS
// ============================================================================

/// Cache-aligned structure for hot data
#[repr(align(64))] // Cache line size
pub struct CacheAlignedData<T> {
    pub data: T,
}

impl<T> CacheAlignedData<T> {
    pub fn new(data: T) -> Self {
        Self { data }
    }
}

/// Memory prefetch hints for better cache performance
#[inline(always)]
pub fn prefetch_read<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
    }
}

#[inline(always)]
pub fn prefetch_write<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T1);
    }
}

// ============================================================================
// BRANCH PREDICTION HINTS
// ============================================================================

/// Likely branch prediction hint (placeholder for stable Rust)
#[inline(always)]
pub fn likely(condition: bool) -> bool {
    condition
}

/// Unlikely branch prediction hint (placeholder for stable Rust)
#[inline(always)]
pub fn unlikely(condition: bool) -> bool {
    condition
}

// ============================================================================
// FAST MATH OPERATIONS
// ============================================================================

/// Fast integer square root using bit manipulation
#[inline(always)]
pub fn fast_sqrt_u32(n: u32) -> u32 {
    if n == 0 { return 0; }
    
    let mut shift = 2;
    let mut nshifted = n >> shift;
    
    while nshifted != 0 && nshifted != n {
        shift += 2;
        nshifted = n >> shift;
    }
    shift -= 2;
    
    let mut result = 0u32;
    while shift >= 0 {
        result <<= 1;
        let candidate = result + 1;
        if candidate * candidate <= (n >> shift) {
            result = candidate;
        }
        if shift == 0 { break; }
        shift -= 2;
    }
    
    result
}

/// Fast division by constant using multiplication
#[inline(always)]
pub fn fast_div_by_constant(dividend: u32, divisor: u32) -> u32 {
    match divisor {
        1 => dividend,
        2 => dividend >> 1,
        4 => dividend >> 2,
        8 => dividend >> 3,
        16 => dividend >> 4,
        32 => dividend >> 5,
        64 => dividend >> 6,
        _ => dividend / divisor, // Fallback to normal division
    }
}

// ============================================================================
// OPTIMIZED HASH FUNCTIONS
// ============================================================================

/// Fast hash function using bit manipulation
#[inline(always)]
pub fn fast_hash_64(mut key: u64) -> u64 {
    key = (!key).wrapping_add(key << 21); // key = (key << 21) - key - 1;
    key = key ^ (key >> 24);
    key = (key.wrapping_add(key << 3)).wrapping_add(key << 8); // key * 265
    key = key ^ (key >> 14);
    key = (key.wrapping_add(key << 2)).wrapping_add(key << 4); // key * 21
    key = key ^ (key >> 28);
    key = key.wrapping_add(key << 31);
    key
}

/// FNV-1a hash function optimized for speed
#[inline(always)]
pub fn fnv1a_hash(data: &[u8]) -> u64 {
    const FNV_PRIME: u64 = 1099511628211;
    const FNV_OFFSET_BASIS: u64 = 14695981039346656037;
    
    let mut hash = FNV_OFFSET_BASIS;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

// ============================================================================
// CPU FEATURE DETECTION AND INITIALIZATION
// ============================================================================

pub struct CpuFeatures {
    pub popcnt: bool,
    pub bmi1: bool,
    pub bmi2: bool,
    pub lzcnt: bool,
    pub avx2: bool,
    pub sse42: bool,
    pub aes: bool,
    pub pclmul: bool,
}

impl CpuFeatures {
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                popcnt: is_x86_feature_detected!("popcnt"),
                bmi1: is_x86_feature_detected!("bmi1"),
                bmi2: is_x86_feature_detected!("bmi2"),
                lzcnt: is_x86_feature_detected!("lzcnt"),
                avx2: is_x86_feature_detected!("avx2"),
                sse42: is_x86_feature_detected!("sse4.2"),
                aes: is_x86_feature_detected!("aes"),
                pclmul: is_x86_feature_detected!("pclmulqdq"),
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self {
                popcnt: false,
                bmi1: false,
                bmi2: false,
                lzcnt: false,
                avx2: false,
                sse42: false,
                aes: false,
                pclmul: false,
            }
        }
    }
    
    pub fn log_features(&self) {
        println!("info string CPU Features Detected:");
        println!("info string - POPCNT: {}", self.popcnt);
        println!("info string - BMI1: {}", self.bmi1);
        println!("info string - BMI2: {}", self.bmi2);
        println!("info string - LZCNT: {}", self.lzcnt);
        println!("info string - AVX2: {}", self.avx2);
        println!("info string - SSE4.2: {}", self.sse42);
        println!("info string - AES: {}", self.aes);
        println!("info string - PCLMUL: {}", self.pclmul);
    }
}

/// Global CPU features instance
static mut CPU_FEATURES: Option<CpuFeatures> = None;
static INIT_FEATURES: std::sync::Once = std::sync::Once::new();

pub fn get_cpu_features() -> &'static CpuFeatures {
    unsafe {
        INIT_FEATURES.call_once(|| {
            CPU_FEATURES = Some(CpuFeatures::detect());
        });
        CPU_FEATURES.as_ref().unwrap()
    }
}

/// Enhanced initialization with detailed CPU information
pub fn init_intrinsics() {
    let features = get_cpu_features();
    features.log_features();
    
    // Additional CPU info
    println!("info string Cache line size: 64 bytes");
    println!("info string Target architecture: {}", std::env::consts::ARCH);
    
    #[cfg(target_arch = "x86_64")]
    {
        // Detect CPU model if possible
        println!("info string Optimizations: x86_64 with intrinsics");
        if features.avx2 {
            println!("info string SIMD: AVX2 vectorization enabled");
        } else if features.sse42 {
            println!("info string SIMD: SSE4.2 vectorization enabled");
        }
    }
}