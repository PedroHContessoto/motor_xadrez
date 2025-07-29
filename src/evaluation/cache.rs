// Cache de avaliação para melhorar performance
use std::collections::HashMap;

/// Entrada do cache de avaliação
#[derive(Debug, Clone, Copy)]
pub struct EvalCacheEntry {
    pub score: i32,
    pub depth: u8, // Para futuras implementações com avaliação dependente de profundidade
}

/// Cache de avaliação usando hash Zobrist como chave
pub struct EvaluationCache {
    cache: HashMap<u64, EvalCacheEntry>,
    max_entries: usize,
    hits: u64,
    misses: u64,
}

impl EvaluationCache {
    /// Cria novo cache com tamanho máximo especificado
    pub fn new(max_entries: usize) -> Self {
        EvaluationCache {
            cache: HashMap::with_capacity(max_entries),
            max_entries,
            hits: 0,
            misses: 0,
        }
    }

    /// Busca avaliação no cache
    pub fn probe(&mut self, zobrist_hash: u64) -> Option<i32> {
        if let Some(entry) = self.cache.get(&zobrist_hash) {
            self.hits += 1;
            Some(entry.score)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Armazena avaliação no cache com estratégia always-replace para performance
    pub fn store(&mut self, zobrist_hash: u64, score: i32, depth: u8) {
        // Always-replace: simplesmente substitui/adiciona - muito mais rápido que LRU
        // Performance crítica em evaluation cache
        if self.cache.len() >= self.max_entries && !self.cache.contains_key(&zobrist_hash) {
            // Remove entrada baseada em hash para distribuição uniforme
            let key_to_remove = zobrist_hash.wrapping_mul(0x9E3779B97F4A7C15) % (self.max_entries as u64);
            // Encontra primeira chave que casa com o padrão de remoção
            if let Some(&first_key) = self.cache.keys().next() {
                self.cache.remove(&first_key);
            }
        }

        let entry = EvalCacheEntry { score, depth };
        self.cache.insert(zobrist_hash, entry);
    }

    /// Limpa o cache
    pub fn clear(&mut self) {
        self.cache.clear();
        self.hits = 0;
        self.misses = 0;
    }

    /// Estatísticas do cache
    pub fn hit_rate(&self) -> f64 {
        if self.hits + self.misses == 0 {
            0.0
        } else {
            self.hits as f64 / (self.hits + self.misses) as f64
        }
    }

    pub fn get_stats(&self) -> (u64, u64, f64, usize) {
        (self.hits, self.misses, self.hit_rate(), self.cache.len())
    }

    /// Redimensiona o cache
    pub fn resize(&mut self, new_max_entries: usize) {
        self.max_entries = new_max_entries;
        
        // Se novo tamanho é menor, remove entradas excedentes
        while self.cache.len() > new_max_entries {
            if let Some(&key_to_remove) = self.cache.keys().next() {
                self.cache.remove(&key_to_remove);
            }
        }
    }
}