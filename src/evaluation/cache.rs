// Cache de avaliação para evitar recálculos desnecessários
use crate::{board::Board, types::Color};
use std::collections::HashMap;

/// Cache LRU simples para avaliações
pub struct EvaluationCache {
    cache: HashMap<u64, CachedEvaluation>,
    max_size: usize,
    hits: u64,
    misses: u64,
}

#[derive(Debug, Clone, Copy)]
struct CachedEvaluation {
    eval: i32,
    depth: u8, // Profundidade de busca quando foi avaliado
    eval_type: EvaluationType, // Tipo de avaliação
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum EvaluationType {
    Full,     // Avaliação completa (depth <= 4)
    Partial,  // Avaliação parcial (depth 5-7)
    Fast,     // Avaliação rápida (depth >= 8)
}

impl EvaluationCache {
    pub fn new(max_size: usize) -> Self {
        EvaluationCache {
            cache: HashMap::with_capacity(max_size),
            max_size,
            hits: 0,
            misses: 0,
        }
    }

    /// Busca avaliação no cache com hierarquia por profundidade
    pub fn get(&mut self, hash: u64, requested_depth: u8) -> Option<i32> {
        if let Some(cached) = self.cache.get(&hash) {
            // Verifica se a avaliação cached é adequada para a profundidade solicitada
            let requested_type = depth_to_eval_type(requested_depth);
            
            // Cache hit se o tipo é igual ou mais detalhado
            let is_adequate = match (cached.eval_type, requested_type) {
                (EvaluationType::Full, _) => true,  // Full serve para qualquer profundidade
                (EvaluationType::Partial, EvaluationType::Partial) => true,
                (EvaluationType::Partial, EvaluationType::Fast) => true,
                (EvaluationType::Fast, EvaluationType::Fast) => true,
                _ => false, // Cached é menos detalhado que o necessário
            };
            
            if is_adequate {
                self.hits += 1;
                Some(cached.eval)
            } else {
                self.misses += 1;
                None
            }
        } else {
            self.misses += 1;
            None
        }
    }

    /// Armazena avaliação no cache com tipo hierárquico
    pub fn store(&mut self, hash: u64, eval: i32, depth: u8) {
        let eval_type = depth_to_eval_type(depth);
        
        // Verifica se devemos sobrescrever entrada existente
        if let Some(existing) = self.cache.get(&hash) {
            // Só substitui se a nova avaliação é mais detalhada ou igual profundidade
            let should_replace = match (existing.eval_type, eval_type) {
                (EvaluationType::Fast, EvaluationType::Partial) => true,
                (EvaluationType::Fast, EvaluationType::Full) => true,
                (EvaluationType::Partial, EvaluationType::Full) => true,
                (same_type, new_type) if same_type == new_type => depth <= existing.depth, // Profundidade menor é melhor
                _ => false,
            };
            
            if !should_replace {
                return; // Mantém a entrada mais valiosa
            }
        }
        
        // Se cache está cheio, remove entradas estrategicamente
        if self.cache.len() >= self.max_size {
            self.smart_cache_cleanup();
        }

        self.cache.insert(hash, CachedEvaluation { eval, depth, eval_type });
    }
    
    /// Limpeza inteligente do cache (prioriza avaliações Full)
    fn smart_cache_cleanup(&mut self) {
        let mut entries: Vec<(u64, CachedEvaluation)> = self.cache.iter()
            .map(|(&k, &v)| (k, v))
            .collect();
        
        // Ordena por prioridade: Fast primeiro (menos valiosos), depois profundidade alta
        entries.sort_by(|a, b| {
            match (a.1.eval_type, b.1.eval_type) {
                (EvaluationType::Fast, EvaluationType::Fast) => b.1.depth.cmp(&a.1.depth),
                (EvaluationType::Fast, _) => std::cmp::Ordering::Less, // Fast vai primeiro (remove)
                (_, EvaluationType::Fast) => std::cmp::Ordering::Greater,
                (EvaluationType::Partial, EvaluationType::Partial) => b.1.depth.cmp(&a.1.depth),
                (EvaluationType::Partial, EvaluationType::Full) => std::cmp::Ordering::Less,
                (EvaluationType::Full, EvaluationType::Partial) => std::cmp::Ordering::Greater,
                (EvaluationType::Full, EvaluationType::Full) => b.1.depth.cmp(&a.1.depth),
            }
        });
        
        // Remove 25% das entradas menos valiosas
        let remove_count = self.max_size / 4;
        for i in 0..remove_count.min(entries.len()) {
            self.cache.remove(&entries[i].0);
        }
    }

    /// Limpa o cache
    pub fn clear(&mut self) {
        self.cache.clear();
        self.hits = 0;
        self.misses = 0;
    }

    /// Estatísticas do cache
    pub fn stats(&self) -> (u64, u64, f64) {
        let total = self.hits + self.misses;
        let hit_rate = if total > 0 { 
            self.hits as f64 / total as f64 * 100.0 
        } else { 
            0.0 
        };
        (self.hits, self.misses, hit_rate)
    }

    /// Tamanho atual do cache
    pub fn size(&self) -> usize {
        self.cache.len()
    }
}

/// Cache hierárquico por componente para avaliação incremental
lazy_static::lazy_static! {
    static ref EVAL_CACHE: std::sync::Mutex<EvaluationCache> = 
        std::sync::Mutex::new(EvaluationCache::new(10000)); // Reduzido de 500k para 10k
    
    // Caches especializados por componente (incrementais)
    static ref MATERIAL_CACHE: std::sync::Mutex<std::collections::HashMap<u64, i32>> = 
        std::sync::Mutex::new(std::collections::HashMap::with_capacity(1000)); // Reduzido de 100k para 1k
    
    static ref PAWN_CACHE: std::sync::Mutex<std::collections::HashMap<u64, i32>> = 
        std::sync::Mutex::new(std::collections::HashMap::with_capacity(1000)); // Reduzido de 50k para 1k
    
    static ref KING_SAFETY_CACHE: std::sync::Mutex<std::collections::HashMap<u64, i32>> = 
        std::sync::Mutex::new(std::collections::HashMap::with_capacity(1000)); // Reduzido de 30k para 1k
    
    static ref MOBILITY_CACHE: std::sync::Mutex<std::collections::HashMap<u64, i32>> = 
        std::sync::Mutex::new(std::collections::HashMap::with_capacity(1000)); // Reduzido de 20k para 1k
}

/// Converte profundidade de busca para tipo de avaliação
fn depth_to_eval_type(depth: u8) -> EvaluationType {
    match depth {
        0..=4 => EvaluationType::Full,
        5..=7 => EvaluationType::Partial, 
        _ => EvaluationType::Fast,
    }
}

/// Interface pública para cache de avaliação com profundidade
pub fn get_cached_evaluation_with_depth(board: &Board, depth: u8) -> Option<i32> {
    if let Ok(mut cache) = EVAL_CACHE.try_lock() {
        cache.get(board.zobrist_hash, depth)
    } else {
        None
    }
}

/// Interface pública para cache de avaliação (compatibilidade)
pub fn get_cached_evaluation(board: &Board) -> Option<i32> {
    get_cached_evaluation_with_depth(board, 0) // Assume avaliação completa
}

/// Armazena avaliação no cache com profundidade
pub fn store_evaluation_with_depth(board: &Board, eval: i32, depth: u8) {
    if let Ok(mut cache) = EVAL_CACHE.try_lock() {
        cache.store(board.zobrist_hash, eval, depth);
    }
}

/// Armazena avaliação no cache (compatibilidade)
pub fn store_evaluation(board: &Board, eval: i32) {
    store_evaluation_with_depth(board, eval, 0) // Assume avaliação completa
}

/// Limpa cache de avaliação
pub fn clear_evaluation_cache() {
    if let Ok(mut cache) = EVAL_CACHE.try_lock() {
        cache.clear();
    }
}

/// Cache incremental para material + PST
pub fn get_cached_material(material_hash: u64) -> Option<i32> {
    if let Ok(cache) = MATERIAL_CACHE.try_lock() {
        cache.get(&material_hash).copied()
    } else {
        None
    }
}

pub fn store_cached_material(material_hash: u64, eval: i32) {
    if let Ok(mut cache) = MATERIAL_CACHE.try_lock() {
        if cache.len() >= 100000 {
            cache.clear(); // LRU simples
        }
        cache.insert(material_hash, eval);
    }
}

/// Cache incremental para estrutura de peões  
pub fn get_cached_pawn_structure(pawn_hash: u64) -> Option<i32> {
    if let Ok(cache) = PAWN_CACHE.try_lock() {
        cache.get(&pawn_hash).copied()
    } else {
        None
    }
}

pub fn store_cached_pawn_structure(pawn_hash: u64, eval: i32) {
    if let Ok(mut cache) = PAWN_CACHE.try_lock() {
        if cache.len() >= 50000 {
            cache.clear();
        }
        cache.insert(pawn_hash, eval);
    }
}

/// Cache incremental para segurança do rei
pub fn get_cached_king_safety(king_hash: u64) -> Option<i32> {
    if let Ok(cache) = KING_SAFETY_CACHE.try_lock() {
        cache.get(&king_hash).copied()
    } else {
        None
    }
}

pub fn store_cached_king_safety(king_hash: u64, eval: i32) {
    if let Ok(mut cache) = KING_SAFETY_CACHE.try_lock() {
        if cache.len() >= 30000 {
            cache.clear();
        }
        cache.insert(king_hash, eval);
    }
}

/// Cache incremental para mobilidade
pub fn get_cached_mobility(mobility_hash: u64) -> Option<i32> {
    if let Ok(cache) = MOBILITY_CACHE.try_lock() {
        cache.get(&mobility_hash).copied()
    } else {
        None
    }
}

pub fn store_cached_mobility(mobility_hash: u64, eval: i32) {
    if let Ok(mut cache) = MOBILITY_CACHE.try_lock() {
        if cache.len() >= 20000 {
            cache.clear();
        }
        cache.insert(mobility_hash, eval);
    }
}

/// Obtém estatísticas do cache
pub fn get_cache_stats() -> String {
    if let Ok(cache) = EVAL_CACHE.try_lock() {
        let (hits, misses, hit_rate) = cache.stats();
        format!("Cache: {} hits, {} misses, {:.1}% hit rate, {} entradas", 
                hits, misses, hit_rate, cache.size())
    } else {
        "Cache: Indisponível".to_string()
    }
}