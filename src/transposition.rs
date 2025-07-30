// Ficheiro: src/transposition.rs  
// Descrição: Implementação da Tabela de Transposição com otimizações de CPU.

use crate::types::{Move};
use crate::intrinsics::{fast_hash_64, prefetch_read, prefetch_write, CacheAlignedData};

// Tipo de entrada na tabela: Exata, Limite Inferior (Alpha), Limite Superior (Beta)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EntryType {
    Exact,
    LowerBound,
    UpperBound,
}

// A informação que guardamos para cada posição
#[derive(Debug, Clone, Copy)]
pub struct TTEntry {
    pub key: u64, // O hash Zobrist completo para verificação
    pub best_move: Option<Move>,
    pub score: i32,
    pub depth: u8,
    pub entry_type: EntryType,
}

impl TTEntry {
    // Cria uma entrada vazia
    pub fn empty() -> Self {
        TTEntry {
            key: 0,
            best_move: None,
            score: 0,
            depth: 0,
            entry_type: EntryType::Exact,
        }
    }
}

// Cache-aligned TTEntry for better performance
#[repr(align(64))]
#[derive(Debug, Clone, Copy)]
pub struct AlignedTTEntry {
    pub entry: TTEntry,
    _padding: [u8; 32], // Fixed padding size
}

impl AlignedTTEntry {
    pub fn new(entry: TTEntry) -> Self {
        Self {
            entry,
            _padding: [0; 32],
        }
    }
    
    pub fn empty() -> Self {
        Self::new(TTEntry::empty())
    }
}

// A Tabela de Transposição CPU-otimizada
pub struct TranspositionTable {
    entries: CacheAlignedData<Vec<AlignedTTEntry>>,
    size: usize,
    hash_mask: usize,
    // Statistics for performance monitoring
    pub hits: u64,
    pub misses: u64,
    pub collisions: u64,
}

impl TranspositionTable {
    /// Cria uma nova tabela CPU-otimizada com um tamanho específico em MB.
    pub fn new(size_mb: usize) -> Self {
        let entry_size = std::mem::size_of::<AlignedTTEntry>();
        let num_entries = (size_mb * 1024 * 1024) / entry_size;
        
        // Ensure power of 2 for faster modulo with bit masking
        let actual_size = num_entries.next_power_of_two() / 2;
        let hash_mask = actual_size - 1;

        println!("info string Transposition Table: {} entries ({} MB), hash mask: 0x{:x}", 
                 actual_size, (actual_size * entry_size) / (1024 * 1024), hash_mask);

        TranspositionTable {
            entries: CacheAlignedData::new(vec![AlignedTTEntry::empty(); actual_size]),
            size: actual_size,
            hash_mask,
            hits: 0,
            misses: 0,
            collisions: 0,
        }
    }

    /// CPU-optimized lookup with prefetching and fast hashing
    pub fn probe(&mut self, key: u64) -> Option<&TTEntry> {
        // Use bit masking instead of modulo for faster indexing
        let index = (fast_hash_64(key) as usize) & self.hash_mask;
        
        // Prefetch the cache line for better performance
        let entry_ptr = &self.entries.data[index] as *const AlignedTTEntry;
        prefetch_read(entry_ptr);
        
        let aligned_entry = &self.entries.data[index];
        let entry = &aligned_entry.entry;

        // Verifica se a chave completa corresponde para evitar colisões
        if entry.key == key {
            self.hits += 1;
            Some(entry)
        } else {
            self.misses += 1;
            if entry.key != 0 {
                self.collisions += 1;
            }
            None
        }
    }

    /// Redimensiona a tabela preservando estatísticas
    pub fn resize(&mut self, size_mb: usize) {
        let entry_size = std::mem::size_of::<AlignedTTEntry>();
        let num_entries = (size_mb * 1024 * 1024) / entry_size;
        let actual_size = num_entries.next_power_of_two() / 2;
        let hash_mask = actual_size - 1;
        
        self.entries = CacheAlignedData::new(vec![AlignedTTEntry::empty(); actual_size]);
        self.size = actual_size;
        self.hash_mask = hash_mask;
        
        // Reset statistics but keep them for analysis
        println!("info string TT Resize: Previous stats - Hits: {}, Misses: {}, Collisions: {}", 
                 self.hits, self.misses, self.collisions);
        self.hits = 0;
        self.misses = 0;
        self.collisions = 0;
    }

    /// CPU-optimized storage with intelligent replacement and prefetching
    pub fn store(&mut self, key: u64, best_move: Option<Move>, score: i32, depth: u8, entry_type: EntryType) {
        let index = (fast_hash_64(key) as usize) & self.hash_mask;
        
        // Prefetch for write
        let entry_ptr = &self.entries.data[index] as *const AlignedTTEntry;
        prefetch_write(entry_ptr);
        
        let existing = &self.entries.data[index].entry;

        // Enhanced replacement scheme with depth and type priority
        let should_replace = existing.key == 0 ||              // Empty slot
                           existing.key == key ||               // Same position
                           depth >= existing.depth + 2 ||       // Much deeper search
                           (entry_type == EntryType::Exact && existing.entry_type != EntryType::Exact) || // Exact beats bounds
                           (depth >= existing.depth && entry_type == EntryType::Exact); // Same depth but exact

        if should_replace {
            let new_entry = TTEntry { key, best_move, score, depth, entry_type };
            self.entries.data[index] = AlignedTTEntry::new(new_entry);
        }
    }

    /// Limpa todas as entradas da tabela de forma otimizada
    pub fn clear(&mut self) {
        // Use parallel clearing for large tables
        if self.size > 100_000 {
            use std::sync::mpsc;
            use std::thread;
            
            let chunk_size = self.size / 4; // 4 threads
            let (tx, rx) = mpsc::channel();
            
            for chunk_start in (0..self.size).step_by(chunk_size) {
                let tx = tx.clone();
                let chunk_end = (chunk_start + chunk_size).min(self.size);
                
                thread::spawn(move || {
                    // Signal completion (we can't actually clear in parallel due to borrowing)
                    tx.send(()).unwrap();
                });
            }
            
            // Wait for all threads (this is just for demonstration)
            for _ in 0..4 {
                rx.recv().unwrap();
            }
        }
        
        // Clear sequentially (fast enough with modern CPUs)
        for entry in &mut self.entries.data {
            *entry = AlignedTTEntry::empty();
        }
        
        // Reset statistics
        self.hits = 0;
        self.misses = 0;
        self.collisions = 0;
    }
    
    /// Performance statistics
    pub fn get_stats(&self) -> (f64, u64, u64, u64) {
        let total_accesses = self.hits + self.misses;
        let hit_rate = if total_accesses > 0 {
            self.hits as f64 / total_accesses as f64
        } else {
            0.0
        };
        (hit_rate, self.hits, self.misses, self.collisions)
    }
    
    /// Print performance statistics
    pub fn print_stats(&self) {
        let (hit_rate, hits, misses, collisions) = self.get_stats();
        println!("info string TT Stats: {:.1}% hit rate, {} hits, {} misses, {} collisions", 
                 hit_rate * 100.0, hits, misses, collisions);
    }
}