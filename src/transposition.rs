// Ficheiro: src/transposition.rs
// Descrição: Implementação da Tabela de Transposição.

use crate::types::{Move};

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

// A Tabela de Transposição em si
pub struct TranspositionTable {
    entries: Vec<TTEntry>,
    size: usize,
}

impl TranspositionTable {
    /// Cria uma nova tabela com um tamanho específico em MB.
    pub fn new(size_mb: usize) -> Self {
        let entry_size = std::mem::size_of::<TTEntry>();
        let num_entries = (size_mb * 1024 * 1024) / entry_size;

        println!("INFO: Tabela de Transposição com {} entradas (aprox. {} MB).", num_entries, size_mb);

        TranspositionTable {
            entries: vec![TTEntry::empty(); num_entries],
            size: num_entries,
        }
    }

    /// Procura uma entrada na tabela.
    pub fn probe(&self, key: u64) -> Option<&TTEntry> {
        let index = (key as usize) % self.size;
        let entry = &self.entries[index];

        // Verifica se a chave completa corresponde para evitar colisões
        if entry.key == key {
            Some(entry)
        } else {
            None
        }
    }

    /// Guarda uma nova entrada na tabela com esquema de substituição inteligente.
    pub fn store(&mut self, key: u64, best_move: Option<Move>, score: i32, depth: u8, entry_type: EntryType) {
        let index = (key as usize) % self.size;
        let existing = &self.entries[index];
        
        // Esquema de substituição depth-preferred:
        // Só substitui se a nova entrada tiver maior profundidade, for exata, ou a entrada atual estiver vazia
        if existing.key == 0 || depth >= existing.depth || entry_type == EntryType::Exact {
            self.entries[index] = TTEntry { key, best_move, score, depth, entry_type };
        }
    }
}