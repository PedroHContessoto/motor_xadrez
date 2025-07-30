// Sistema de lazy evaluation para otimização de performance
// Evita cálculos desnecessários através de memoization e avaliação sob demanda

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use crate::types::Bitboard;
use crate::board::Board;
use crate::intrinsics::{fast_hash_64, CacheAlignedData};

// ============================================================================
// LAZY EVALUATION CACHE SYSTEM
// ============================================================================

/// Cache entry com timestamp para aging
#[derive(Clone, Debug)]
struct LazyEvalEntry<T> {
    value: T,
    timestamp: u64,
    access_count: u32,
}

impl<T> LazyEvalEntry<T> {
    fn new(value: T, timestamp: u64) -> Self {
        Self { value, timestamp, access_count: 1 }
    }
    
    fn access(&mut self, timestamp: u64) -> &T {
        self.access_count += 1;
        self.timestamp = timestamp;
        &self.value
    }
}

/// Cache LRU com aging automático
pub struct LazyEvalCache<K, V> 
where 
    K: Hash + Eq + Clone,
    V: Clone,
{
    cache: HashMap<K, LazyEvalEntry<V>>,
    max_size: usize,
    current_time: u64,
    hits: u64,
    misses: u64,
}

impl<K, V> LazyEvalCache<K, V> 
where 
    K: Hash + Eq + Clone,
    V: Clone,
{
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::with_capacity(max_size),
            max_size,
            current_time: 0,
            hits: 0,
            misses: 0,
        }
    }
    
    pub fn get_or_compute<F>(&mut self, key: K, compute_fn: F) -> V
    where
        F: FnOnce() -> V,
    {
        self.current_time += 1;
        
        // Cache hit
        if let Some(entry) = self.cache.get_mut(&key) {
            self.hits += 1;
            return entry.access(self.current_time).clone();
        }
        
        // Cache miss - compute value
        self.misses += 1;
        let value = compute_fn();
        
        // Insert into cache
        self.insert(key, value.clone());
        value
    }
    
    fn insert(&mut self, key: K, value: V) {
        // Clean cache if full
        if self.cache.len() >= self.max_size {
            self.evict_old_entries();
        }
        
        let entry = LazyEvalEntry::new(value, self.current_time);
        self.cache.insert(key, entry);
    }
    
    fn evict_old_entries(&mut self) {
        let evict_count = self.max_size / 4; // Remove 25% of entries
        
        // Collect entries with their keys and scores for eviction
        let mut entries: Vec<(K, u64, u32)> = self.cache.iter()
            .map(|(k, v)| (k.clone(), self.current_time - v.timestamp, v.access_count))
            .collect();
        
        // Sort by age (older first) and access count (less accessed first)
        entries.sort_by(|a, b| {
            let age_cmp = b.1.cmp(&a.1); // Older entries first
            if age_cmp == std::cmp::Ordering::Equal {
                a.2.cmp(&b.2) // Less accessed first
            } else {
                age_cmp
            }
        });
        
        // Remove oldest/least accessed entries
        for (key, _, _) in entries.iter().take(evict_count) {
            self.cache.remove(key);
        }
    }
    
    pub fn clear(&mut self) {
        self.cache.clear();
        self.current_time = 0;
        self.hits = 0;
        self.misses = 0;
    }
    
    pub fn stats(&self) -> (f64, usize, u64, u64) {
        let hit_rate = if self.hits + self.misses > 0 {
            self.hits as f64 / (self.hits + self.misses) as f64
        } else {
            0.0
        };
        (hit_rate, self.cache.len(), self.hits, self.misses)
    }
}

// ============================================================================
// SPECIALIZED LAZY EVALUATORS
// ============================================================================

/// Chave para cache de avaliação de mobilidade
#[derive(Hash, Eq, PartialEq, Clone)]
struct MobilityKey {
    piece_bb: Bitboard,
    occupied: Bitboard,
    piece_type: u8,
}

impl MobilityKey {
    fn new(piece_bb: Bitboard, occupied: Bitboard, piece_type: u8) -> Self {
        Self { piece_bb, occupied, piece_type }
    }
}

/// Chave para cache de safety do rei
#[derive(Hash, Eq, PartialEq, Clone)]
struct KingSafetyKey {
    king_pos: u8,
    enemy_pieces: Bitboard,
    pawn_shield: Bitboard,
}

/// Lazy evaluator para mobilidade de peças
pub struct LazyMobilityEvaluator {
    cache: CacheAlignedData<LazyEvalCache<MobilityKey, i32>>,
}

impl LazyMobilityEvaluator {
    pub fn new() -> Self {
        Self {
            cache: CacheAlignedData::new(LazyEvalCache::new(16384)), // 16K entries
        }
    }
    
    pub fn evaluate_piece_mobility(&mut self, piece_bb: Bitboard, occupied: Bitboard, piece_type: u8) -> i32 {
        let key = MobilityKey::new(piece_bb, occupied, piece_type);
        
        // Create a temporary self reference to avoid borrowing issues
        let mobility_fn = Self::compute_mobility_static;
        self.cache.data.get_or_compute(key, || {
            mobility_fn(piece_bb, occupied, piece_type)
        })
    }
    
    fn compute_mobility(&self, piece_bb: Bitboard, occupied: Bitboard, piece_type: u8) -> i32 {
        Self::compute_mobility_static(piece_bb, occupied, piece_type)
    }
    
    fn compute_mobility_static(piece_bb: Bitboard, occupied: Bitboard, piece_type: u8) -> i32 {
        // Implementação específica de mobilidade baseada no tipo de peça
        match piece_type {
            1 => Self::compute_knight_mobility_static(piece_bb, occupied),
            2 => Self::compute_bishop_mobility_static(piece_bb, occupied),
            3 => Self::compute_rook_mobility_static(piece_bb, occupied),
            4 => Self::compute_queen_mobility_static(piece_bb, occupied),
            _ => 0,
        }
    }
    
    fn compute_knight_mobility(&self, knight_bb: Bitboard, occupied: Bitboard) -> i32 {
        Self::compute_knight_mobility_static(knight_bb, occupied)
    }
    
    fn compute_knight_mobility_static(knight_bb: Bitboard, occupied: Bitboard) -> i32 {
        let mut mobility = 0;
        let mut bb = knight_bb;
        
        while bb != 0 {
            let sq = bb.trailing_zeros() as u8;
            bb &= bb - 1;
            
            // Simplified knight mobility calculation
            let attacks = crate::moves::knight::get_knight_attacks_lookup(sq);
            let free_squares = attacks & !occupied;
            mobility += free_squares.count_ones() as i32;
        }
        
        mobility * 4 // Weight factor
    }
    
    fn compute_bishop_mobility(&self, bishop_bb: Bitboard, occupied: Bitboard) -> i32 {
        Self::compute_bishop_mobility_static(bishop_bb, occupied)
    }
    
    fn compute_bishop_mobility_static(bishop_bb: Bitboard, occupied: Bitboard) -> i32 {
        let mut mobility = 0;
        let mut bb = bishop_bb;
        
        while bb != 0 {
            let sq = bb.trailing_zeros() as u8;
            bb &= bb - 1;
            
            let attacks = crate::moves::magic_bitboards::get_bishop_attacks_magic(sq, occupied);
            mobility += attacks.count_ones() as i32;
        }
        
        mobility * 3
    }
    
    fn compute_rook_mobility(&self, rook_bb: Bitboard, occupied: Bitboard) -> i32 {
        Self::compute_rook_mobility_static(rook_bb, occupied)
    }
    
    fn compute_rook_mobility_static(rook_bb: Bitboard, occupied: Bitboard) -> i32 {
        let mut mobility = 0;
        let mut bb = rook_bb;
        
        while bb != 0 {
            let sq = bb.trailing_zeros() as u8;
            bb &= bb - 1;
            
            let attacks = crate::moves::magic_bitboards::get_rook_attacks_magic(sq, occupied);
            mobility += attacks.count_ones() as i32;
        }
        
        mobility * 2
    }
    
    fn compute_queen_mobility(&self, queen_bb: Bitboard, occupied: Bitboard) -> i32 {
        Self::compute_queen_mobility_static(queen_bb, occupied)
    }
    
    fn compute_queen_mobility_static(queen_bb: Bitboard, occupied: Bitboard) -> i32 {
        let mut mobility = 0;
        let mut bb = queen_bb;
        
        while bb != 0 {
            let sq = bb.trailing_zeros() as u8;
            bb &= bb - 1;
            
            let bishop_attacks = crate::moves::magic_bitboards::get_bishop_attacks_magic(sq, occupied);
            let rook_attacks = crate::moves::magic_bitboards::get_rook_attacks_magic(sq, occupied);
            let queen_attacks = bishop_attacks | rook_attacks;
            mobility += queen_attacks.count_ones() as i32;
        }
        
        mobility * 1
    }
    
    pub fn clear_cache(&mut self) {
        self.cache.data.clear();
    }
    
    pub fn get_stats(&self) -> (f64, usize, u64, u64) {
        self.cache.data.stats()
    }
}

/// Lazy evaluator para segurança do rei
pub struct LazyKingSafetyEvaluator {
    cache: CacheAlignedData<LazyEvalCache<KingSafetyKey, i32>>,
}

impl LazyKingSafetyEvaluator {
    pub fn new() -> Self {
        Self {
            cache: CacheAlignedData::new(LazyEvalCache::new(8192)), // 8K entries
        }
    }
    
    pub fn evaluate_king_safety(&mut self, king_pos: u8, enemy_pieces: Bitboard, pawn_shield: Bitboard) -> i32 {
        let key = KingSafetyKey { king_pos, enemy_pieces, pawn_shield };
        
        let safety_fn = Self::compute_king_safety_static;
        self.cache.data.get_or_compute(key, || {
            safety_fn(king_pos, enemy_pieces, pawn_shield)
        })
    }
    
    fn compute_king_safety(&self, king_pos: u8, enemy_pieces: Bitboard, pawn_shield: Bitboard) -> i32 {
        Self::compute_king_safety_static(king_pos, enemy_pieces, pawn_shield)
    }
    
    fn compute_king_safety_static(king_pos: u8, enemy_pieces: Bitboard, pawn_shield: Bitboard) -> i32 {
        let mut safety = 0;
        let _king_rank = king_pos / 8;
        let king_file = king_pos % 8;
        
        // Pawn shield evaluation
        let shield_strength = pawn_shield.count_ones() as i32;
        safety += shield_strength * 20;
        
        // Open files near king penalty
        for file_offset in -1..=1 {
            let target_file = king_file as i8 + file_offset;
            if target_file >= 0 && target_file < 8 {
                let file_mask = 0x0101010101010101u64 << target_file;
                if (pawn_shield & file_mask) == 0 {
                    safety -= 30; // Open file penalty
                }
            }
        }
        
        // Enemy piece pressure
        let king_area = Self::get_king_area_static(king_pos);
        let pressure = (enemy_pieces & king_area).count_ones() as i32;
        safety -= pressure * 15;
        
        safety
    }
    
    fn get_king_area(&self, king_pos: u8) -> Bitboard {
        Self::get_king_area_static(king_pos)
    }
    
    fn get_king_area_static(king_pos: u8) -> Bitboard {
        let king_rank = king_pos / 8;
        let king_file = king_pos % 8;
        let mut area = 0u64;
        
        // 3x3 area around king
        for rank_offset in -1..=1 {
            for file_offset in -1..=1 {
                let target_rank = king_rank as i8 + rank_offset;
                let target_file = king_file as i8 + file_offset;
                
                if target_rank >= 0 && target_rank < 8 && target_file >= 0 && target_file < 8 {
                    let square = (target_rank * 8 + target_file) as u8;
                    area |= 1u64 << square;
                }
            }
        }
        
        area
    }
    
    pub fn clear_cache(&mut self) {
        self.cache.data.clear();
    }
    
    pub fn get_stats(&self) -> (f64, usize, u64, u64) {
        self.cache.data.stats()
    }
}

// ============================================================================
// GLOBAL LAZY EVALUATION MANAGER
// ============================================================================

pub struct LazyEvaluationManager {
    pub mobility_evaluator: LazyMobilityEvaluator,
    pub king_safety_evaluator: LazyKingSafetyEvaluator,
}

impl LazyEvaluationManager {
    pub fn new() -> Self {
        Self {
            mobility_evaluator: LazyMobilityEvaluator::new(),
            king_safety_evaluator: LazyKingSafetyEvaluator::new(),
        }
    }
    
    pub fn clear_all_caches(&mut self) {
        self.mobility_evaluator.clear_cache();
        self.king_safety_evaluator.clear_cache();
    }
    
    pub fn print_stats(&self) {
        let (mob_hit_rate, mob_size, mob_hits, mob_misses) = self.mobility_evaluator.get_stats();
        let (king_hit_rate, king_size, king_hits, king_misses) = self.king_safety_evaluator.get_stats();
        
        println!("info string Lazy Evaluation Cache Stats:");
        println!("info string - Mobility: {:.1}% hit rate, {} entries, {} hits, {} misses", 
                 mob_hit_rate * 100.0, mob_size, mob_hits, mob_misses);
        println!("info string - King Safety: {:.1}% hit rate, {} entries, {} hits, {} misses", 
                 king_hit_rate * 100.0, king_size, king_hits, king_misses);
    }
}

/// Global instance for lazy evaluation
static mut LAZY_EVAL_MANAGER: Option<LazyEvaluationManager> = None;
static INIT_LAZY_EVAL: std::sync::Once = std::sync::Once::new();

pub fn get_lazy_eval_manager() -> &'static mut LazyEvaluationManager {
    unsafe {
        INIT_LAZY_EVAL.call_once(|| {
            LAZY_EVAL_MANAGER = Some(LazyEvaluationManager::new());
        });
        LAZY_EVAL_MANAGER.as_mut().unwrap()
    }
}