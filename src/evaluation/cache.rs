// Cache de avaliação para melhorar performance
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

/// Entrada do cache de avaliação
#[derive(Debug, Clone, Copy, Default)]
pub struct EvalCacheEntry {
    pub zobrist_hash: u64,
    pub score: i32,
    pub depth: u8,
    pub flags: u8,  // Novo: flags para tipo de entrada (exact, lowerbound, upperbound)
    pub age: u8,    // Novo: idade da entrada para replacement
}

impl EvalCacheEntry {
    /// Cria nova entrada
    pub fn new(zobrist_hash: u64, score: i32, depth: u8) -> Self {
        EvalCacheEntry {
            zobrist_hash,
            score,
            depth,
            flags: 0,
            age: 0,
        }
    }

    /// Verifica se a entrada é válida
    pub fn is_valid(&self) -> bool {
        self.zobrist_hash != 0
    }

    /// Atualiza idade da entrada
    pub fn update_age(&mut self, current_age: u8) {
        self.age = current_age;
    }
}

/// Políticas de substituição do cache
#[derive(Debug, Clone, Copy)]
pub enum ReplacementPolicy {
    AlwaysReplace,
    DepthPreferred,
    AgePreferred,
    Combined,
}

/// Cache de avaliação otimizado com array fixed-size e estatísticas avançadas
pub struct EvaluationCache {
    entries: Vec<EvalCacheEntry>,
    size_mask: usize,
    hits: AtomicU64,
    misses: AtomicU64,
    collisions: AtomicU64,
    writes: AtomicU64,
    current_age: RwLock<u8>,
    replacement_policy: ReplacementPolicy,
}

impl EvaluationCache {
    /// Cria novo cache com tamanho especificado em MB
    pub fn new(size_mb: usize) -> Self {
        Self::with_policy(size_mb, ReplacementPolicy::Combined)
    }

    /// Cria cache com política de substituição específica
    pub fn with_policy(size_mb: usize, policy: ReplacementPolicy) -> Self {
        // Calcular número de entradas baseado no tamanho em MB
        let size_bytes = size_mb.saturating_mul(1024 * 1024);
        let entry_size = std::mem::size_of::<EvalCacheEntry>();
        let mut size = size_bytes / entry_size;

        // Garantir que é power of 2 para masking eficiente
        size = size.next_power_of_two();
        let size_mask = size - 1;

        let mut cache = EvaluationCache {
            entries: vec![EvalCacheEntry::default(); size],
            size_mask,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            collisions: AtomicU64::new(0),
            writes: AtomicU64::new(0),
            current_age: RwLock::new(0),
            replacement_policy: policy,
        };

        // Prefill para melhor performance inicial
        cache.prefill();
        cache
    }

    /// Busca avaliação no cache (thread-safe, lock-free)
    pub fn probe(&self, zobrist_hash: u64) -> Option<i32> {
        let index = self.get_index(zobrist_hash);
        let entry = self.entries[index];

        if entry.zobrist_hash == zobrist_hash {
            self.hits.fetch_add(1, Ordering::Relaxed);
            Some(entry.score)
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Busca avaliação com informações extras
    pub fn probe_with_info(&self, zobrist_hash: u64) -> Option<(i32, u8, u8)> {
        let index = self.get_index(zobrist_hash);
        let entry = self.entries[index];

        if entry.zobrist_hash == zobrist_hash {
            self.hits.fetch_add(1, Ordering::Relaxed);
            Some((entry.score, entry.depth, entry.flags))
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Armazena avaliação no cache com política de substituição
    pub fn store(&mut self, zobrist_hash: u64, score: i32, depth: u8) {
        self.store_with_flags(zobrist_hash, score, depth, 0);
    }

    /// Armazena avaliação com flags específicas
    pub fn store_with_flags(&mut self, zobrist_hash: u64, score: i32, depth: u8, flags: u8) {
        let index = self.get_index(zobrist_hash);
        let existing_entry = &self.entries[index];

        // Verifica se deve substituir baseado na política
        let should_replace = match self.replacement_policy {
            ReplacementPolicy::AlwaysReplace => true,
            ReplacementPolicy::DepthPreferred => {
                existing_entry.zobrist_hash == 0 ||
                    depth >= existing_entry.depth ||
                    existing_entry.zobrist_hash == zobrist_hash
            },
            ReplacementPolicy::AgePreferred => {
                let current_age = *self.current_age.read().unwrap();
                existing_entry.zobrist_hash == 0 ||
                    existing_entry.age < current_age ||
                    existing_entry.zobrist_hash == zobrist_hash
            },
            ReplacementPolicy::Combined => {
                self.should_replace_combined(existing_entry, zobrist_hash, depth)
            },
        };

        if should_replace {
            if existing_entry.zobrist_hash != 0 && existing_entry.zobrist_hash != zobrist_hash {
                self.collisions.fetch_add(1, Ordering::Relaxed);
            }

            let current_age = *self.current_age.read().unwrap();
            self.entries[index] = EvalCacheEntry {
                zobrist_hash,
                score,
                depth,
                flags,
                age: current_age,
            };
            self.writes.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Política de substituição combinada
    fn should_replace_combined(&self, existing: &EvalCacheEntry, new_hash: u64, new_depth: u8) -> bool {
        // Sempre substitui entradas vazias ou mesma posição
        if existing.zobrist_hash == 0 || existing.zobrist_hash == new_hash {
            return true;
        }

        let current_age = *self.current_age.read().unwrap();
        let age_diff = current_age.saturating_sub(existing.age);

        // Pontuação baseada em profundidade e idade
        let existing_score = (existing.depth as i32) * 2 - (age_diff as i32);
        let new_score = (new_depth as i32) * 2;

        new_score >= existing_score
    }

    /// Limpa o cache
    pub fn clear(&mut self) {
        for entry in &mut self.entries {
            *entry = EvalCacheEntry::default();
        }
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.collisions.store(0, Ordering::Relaxed);
        self.writes.store(0, Ordering::Relaxed);
        *self.current_age.write().unwrap() = 0;
    }

    /// Incrementa idade global (para ser chamado a cada nova busca)
    pub fn increment_age(&self) {
        let mut age = self.current_age.write().unwrap();
        *age = age.wrapping_add(1);
    }

    /// Obtém índice no cache
    #[inline(always)]
    fn get_index(&self, zobrist_hash: u64) -> usize {
        (zobrist_hash as usize) & self.size_mask
    }

    /// Estatísticas do cache
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }

    /// Estatísticas completas
    pub fn get_stats(&self) -> (u64, u64, f64, usize) {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let used_entries = self.entries.iter()
            .filter(|entry| entry.is_valid())
            .count();
        (hits, misses, self.hit_rate(), used_entries)
    }

    /// Estatísticas detalhadas
    pub fn get_detailed_stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            collisions: self.collisions.load(Ordering::Relaxed),
            writes: self.writes.load(Ordering::Relaxed),
            hit_rate: self.hit_rate(),
            used_entries: self.entries.iter().filter(|e| e.is_valid()).count(),
            total_entries: self.entries.len(),
            avg_depth: self.calculate_avg_depth(),
            memory_usage_mb: self.memory_usage_mb(),
        }
    }

    /// Calcula profundidade média das entradas
    fn calculate_avg_depth(&self) -> f32 {
        let (sum, count) = self.entries.iter()
            .filter(|e| e.is_valid())
            .fold((0u64, 0u64), |(sum, count), entry| {
                (sum + entry.depth as u64, count + 1)
            });

        if count == 0 {
            0.0
        } else {
            sum as f32 / count as f32
        }
    }

    /// Uso de memória em MB
    pub fn memory_usage_mb(&self) -> f32 {
        let bytes = self.entries.len() * std::mem::size_of::<EvalCacheEntry>();
        bytes as f32 / (1024.0 * 1024.0)
    }

    /// Prefill cache com zero para melhor performance inicial
    pub fn prefill(&mut self) {
        // Força alocação de toda a memória para evitar page faults
        for (i, entry) in self.entries.iter_mut().enumerate() {
            // Padrão alternado para melhor distribuição inicial
            entry.zobrist_hash = 0;
            entry.score = 0;
            entry.depth = 0;
            entry.flags = 0;
            entry.age = 0;
        }
    }

    /// Redimensiona o cache (cria novo cache)
    pub fn resize(&mut self, new_size_mb: usize) {
        let policy = self.replacement_policy;
        *self = Self::with_policy(new_size_mb, policy);
    }

    /// Define nova política de substituição
    pub fn set_replacement_policy(&mut self, policy: ReplacementPolicy) {
        self.replacement_policy = policy;
    }

    /// Obtém taxa de colisão
    pub fn collision_rate(&self) -> f64 {
        let writes = self.writes.load(Ordering::Relaxed);
        let collisions = self.collisions.load(Ordering::Relaxed);
        if writes == 0 {
            0.0
        } else {
            collisions as f64 / writes as f64
        }
    }

    /// Limpa entradas antigas
    pub fn clear_old_entries(&mut self, age_threshold: u8) {
        let current_age = *self.current_age.read().unwrap();
        let mut cleared = 0;

        for entry in &mut self.entries {
            if entry.is_valid() {
                let age_diff = current_age.saturating_sub(entry.age);
                if age_diff > age_threshold {
                    *entry = EvalCacheEntry::default();
                    cleared += 1;
                }
            }
        }

        if cleared > 0 {
            log::debug!("Cleared {} old cache entries", cleared);
        }
    }
}

/// Estatísticas detalhadas do cache
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub collisions: u64,
    pub writes: u64,
    pub hit_rate: f64,
    pub used_entries: usize,
    pub total_entries: usize,
    pub avg_depth: f32,
    pub memory_usage_mb: f32,
}

impl CacheStats {
    /// Formata estatísticas para display
    pub fn format_display(&self) -> String {
        format!(
            "Cache Stats:\n\
             Hits: {} ({:.1}%)\n\
             Misses: {}\n\
             Collisions: {} ({:.1}% of writes)\n\
             Writes: {}\n\
             Used: {}/{} entries ({:.1}%)\n\
             Avg Depth: {:.1}\n\
             Memory: {:.1} MB",
            self.hits,
            self.hit_rate * 100.0,
            self.misses,
            self.collisions,
            (self.collisions as f64 / self.writes.max(1) as f64) * 100.0,
            self.writes,
            self.used_entries,
            self.total_entries,
            (self.used_entries as f64 / self.total_entries as f64) * 100.0,
            self.avg_depth,
            self.memory_usage_mb
        )
    }
}

/// Cache compartilhado thread-safe (opcional)
pub struct SharedEvaluationCache {
    cache: RwLock<EvaluationCache>,
}

impl SharedEvaluationCache {
    pub fn new(size_mb: usize) -> Self {
        SharedEvaluationCache {
            cache: RwLock::new(EvaluationCache::new(size_mb)),
        }
    }

    pub fn probe(&self, zobrist_hash: u64) -> Option<i32> {
        self.cache.read().unwrap().probe(zobrist_hash)
    }

    pub fn store(&self, zobrist_hash: u64, score: i32, depth: u8) {
        self.cache.write().unwrap().store(zobrist_hash, score, depth);
    }

    pub fn clear(&self) {
        self.cache.write().unwrap().clear();
    }

    pub fn get_stats(&self) -> CacheStats {
        self.cache.read().unwrap().get_detailed_stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_basic_operations() {
        let mut cache = EvaluationCache::new(1); // 1MB cache

        // Test store and probe
        cache.store(12345, 100, 5);
        assert_eq!(cache.probe(12345), Some(100));
        assert_eq!(cache.probe(99999), None);

        // Test stats
        let (hits, misses, _, _) = cache.get_stats();
        assert_eq!(hits, 1);
        assert_eq!(misses, 1);
    }

    #[test]
    fn test_cache_replacement_policies() {
        // Test different replacement policies
        let mut cache = EvaluationCache::with_policy(1, ReplacementPolicy::DepthPreferred);

        cache.store(12345, 100, 5);
        cache.store(12345, 200, 3); // Lower depth, should not replace
        assert_eq!(cache.probe(12345), Some(100));

        cache.store(12345, 300, 7); // Higher depth, should replace
        assert_eq!(cache.probe(12345), Some(300));
    }

    #[test]
    fn test_cache_collisions() {
        let mut cache = EvaluationCache::new(1);
        let stats_before = cache.get_detailed_stats();

        // Force collision by using hashes that map to same index
        let hash1 = 12345;
        let hash2 = hash1 + (cache.entries.len() as u64); // Will map to same index

        cache.store(hash1, 100, 5);
        cache.store(hash2, 200, 5);

        let stats_after = cache.get_detailed_stats();
        assert!(stats_after.collisions > stats_before.collisions);
    }
}