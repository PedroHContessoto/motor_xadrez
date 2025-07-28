// Cache especializado para posições de mate
use crate::types::Move;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Entrada do cache de mate
#[derive(Debug, Clone)]
pub struct CachedMateResult {
    pub mate_in: Option<u8>,
    pub best_moves: Vec<Move>,
    pub evaluation: i32,
    pub is_forced: bool,
    pub search_depth: u8,
}

/// Cache especializado para mate usando OnceLock
pub struct MateCache;

static MATE_CACHE: OnceLock<Mutex<HashMap<u64, CachedMateResult>>> = OnceLock::new();

impl MateCache {
    /// Obtém resultado do cache
    pub fn get(zobrist_hash: u64) -> Option<CachedMateResult> {
        let cache = MATE_CACHE.get_or_init(|| Mutex::new(HashMap::with_capacity(10000)));
        
        if let Ok(cache) = cache.try_lock() {
            cache.get(&zobrist_hash).cloned()
        } else {
            None
        }
    }
    
    /// Armazena resultado no cache
    pub fn store(
        zobrist_hash: u64,
        mate_in: Option<u8>,
        best_moves: Vec<Move>,
        evaluation: i32,
        is_forced: bool,
        search_depth: u8
    ) {
        let cache = MATE_CACHE.get_or_init(|| Mutex::new(HashMap::with_capacity(10000)));
        
        if let Ok(mut cache) = cache.try_lock() {
            if cache.len() >= 10000 {
                cache.clear(); // LRU simples: limpa quando cheio
            }
            
            cache.insert(zobrist_hash, CachedMateResult {
                mate_in,
                best_moves,
                evaluation,
                is_forced,
                search_depth,
            });
        }
    }
    
    /// Limpa o cache
    pub fn clear() {
        let cache = MATE_CACHE.get_or_init(|| Mutex::new(HashMap::with_capacity(10000)));
        
        if let Ok(mut cache) = cache.try_lock() {
            cache.clear();
        }
    }
}