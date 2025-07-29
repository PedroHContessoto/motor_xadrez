// Cache de avaliação para melhorar performance
use std::sync::atomic::{AtomicU64, Ordering};

/// Entrada do cache de avaliação
#[derive(Debug, Clone, Copy, Default)]
pub struct EvalCacheEntry {
    pub zobrist_hash: u64,
    pub score: i32,
    pub depth: u8,
}

/// Cache de avaliação otimizado com array fixed-size e sem locks
pub struct EvaluationCache {
    entries: Vec<EvalCacheEntry>,
    size_mask: usize,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl EvaluationCache {
    /// Cria novo cache com tamanho especificado em MB
    pub fn new(size_mb: usize) -> Self {
        // Calcular número de entradas baseado no tamanho em MB
        let size_bytes = size_mb * 1024 * 1024;
        let entry_size = std::mem::size_of::<EvalCacheEntry>();
        let mut size = size_bytes / entry_size;
        
        // Garantir que é power of 2 para masking eficiente
        size = size.next_power_of_two();
        let size_mask = size - 1;

        EvaluationCache {
            entries: vec![EvalCacheEntry::default(); size],
            size_mask,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Busca avaliação no cache (thread-safe, lock-free)
    pub fn probe(&self, zobrist_hash: u64) -> Option<i32> {
        let index = (zobrist_hash as usize) & self.size_mask;
        let entry = self.entries[index];
        
        if entry.zobrist_hash == zobrist_hash {
            self.hits.fetch_add(1, Ordering::Relaxed);
            Some(entry.score)
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Armazena avaliação no cache (thread-safe, always-replace)
    pub fn store(&mut self, zobrist_hash: u64, score: i32, depth: u8) {
        let index = (zobrist_hash as usize) & self.size_mask;
        self.entries[index] = EvalCacheEntry {
            zobrist_hash,
            score,
            depth,
        };
    }

    /// Limpa o cache
    pub fn clear(&mut self) {
        for entry in &mut self.entries {
            *entry = EvalCacheEntry::default();
        }
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }

    /// Estatísticas do cache
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        if hits + misses == 0 {
            0.0
        } else {
            hits as f64 / (hits + misses) as f64
        }
    }

    pub fn get_stats(&self) -> (u64, u64, f64, usize) {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let used_entries = self.entries.iter()
            .filter(|entry| entry.zobrist_hash != 0)
            .count();
        (hits, misses, self.hit_rate(), used_entries)
    }

    /// Prefill cache com zero para melhor performance inicial
    pub fn prefill(&mut self) {
        // Força alocação de toda a memória para evitar page faults
        for entry in &mut self.entries {
            entry.zobrist_hash = 0;
        }
    }
}