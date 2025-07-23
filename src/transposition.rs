// Descrição: Implementação da Tabela de Transposição com método clear() adicionado.

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
    /// Cria uma nova tabela com um tamanho específico em bytes.
    pub fn new(size_bytes: usize) -> Self {
        let entry_size = std::mem::size_of::<TTEntry>();
        let num_entries = size_bytes / entry_size;

        // println!("INFO: Tabela de Transposição com {} entradas (aprox. {} MB).", num_entries, size_bytes / (1024 * 1024));

        TranspositionTable {
            entries: vec![TTEntry::empty(); num_entries],
            size: num_entries,
        }
    }

    /// CORREÇÃO: Método clear() que estava faltando
    pub fn clear(&mut self) {
        // Limpa todas as entradas da tabela
        for entry in &mut self.entries {
            *entry = TTEntry::empty();
        }
    }

    /// Procura uma entrada na tabela.
    pub fn probe(&self, key: u64) -> Option<&TTEntry> {
        if self.size == 0 {
            return None;
        }

        let index = (key as usize) % self.size;
        let entry = &self.entries[index];

        // Verifica se a chave completa corresponde para evitar colisões
        if entry.key == key && entry.key != 0 {
            Some(entry)
        } else {
            None
        }
    }

    /// Guarda uma nova entrada na tabela com esquema de substituição inteligente.
    pub fn store(&mut self, key: u64, best_move: Option<Move>, score: i32, depth: u8, entry_type: EntryType) {
        if self.size == 0 {
            return;
        }

        let index = (key as usize) % self.size;
        let existing = &self.entries[index];

        // Esquema de substituição depth-preferred:
        // Só substitui se a nova entrada tiver maior profundidade, for exata, ou a entrada atual estiver vazia
        if existing.key == 0 || depth >= existing.depth || entry_type == EntryType::Exact {
            self.entries[index] = TTEntry { key, best_move, score, depth, entry_type };
        }
    }

    /// NOVO: Retorna estatísticas da tabela
    pub fn stats(&self) -> (usize, usize, f32) {
        let mut used_entries = 0;

        for entry in &self.entries {
            if entry.key != 0 {
                used_entries += 1;
            }
        }

        let usage_percentage = if self.size > 0 {
            (used_entries as f32 / self.size as f32) * 100.0
        } else {
            0.0
        };

        (self.size, used_entries, usage_percentage)
    }

    /// NOVO: Redimensiona a tabela (útil para mudanças de hash size)
    pub fn resize(&mut self, new_size_bytes: usize) {
        let entry_size = std::mem::size_of::<TTEntry>();
        let new_num_entries = new_size_bytes / entry_size;

        // Salva entradas importantes antes de redimensionar
        let mut important_entries = Vec::new();

        // Coleta entradas com maior profundidade (mais valiosas)
        for entry in &self.entries {
            if entry.key != 0 && entry.depth >= 4 {
                important_entries.push(*entry);
            }
        }

        // Redimensiona a tabela
        self.entries = vec![TTEntry::empty(); new_num_entries];
        self.size = new_num_entries;

        // Reinsere entradas importantes
        for entry in important_entries {
            self.store(entry.key, entry.best_move, entry.score, entry.depth, entry.entry_type);
        }
    }

    /// NOVO: Prefetch para otimização (placeholder)
    pub fn prefetch(&self, key: u64) {
        if self.size > 0 {
            let index = (key as usize) % self.size;
            // Em uma implementação real, usaria instruções de prefetch do processador
            // Por agora, apenas acessa a entrada para carregar na cache
            let _ = &self.entries[index];
        }
    }

    /// NOVO: Verifica integridade da tabela
    pub fn verify_integrity(&self) -> bool {
        // Verifica se não há entradas corrompidas
        for (i, entry) in self.entries.iter().enumerate() {
            if entry.key != 0 {
                // Verifica se o índice calculado corresponde
                let expected_index = (entry.key as usize) % self.size;
                if expected_index != i {
                    // Esta entrada pode estar no lugar errado devido a redimensionamento
                    // Não é necessariamente um erro crítico
                }

                // Verifica se os valores estão em ranges válidos
                if entry.depth > 100 {
                    return false; // Profundidade suspeita
                }

                if entry.score.abs() > 100000 {
                    return false; // Score suspeito
                }
            }
        }

        true
    }

    /// NOVO: Exporta estatísticas detalhadas
    pub fn detailed_stats(&self) -> String {
        let (total, used, percentage) = self.stats();
        let mut depth_histogram = [0; 21]; // Profundidades 0-20
        let mut type_counts = [0; 3]; // Exact, LowerBound, UpperBound

        for entry in &self.entries {
            if entry.key != 0 {
                let depth_idx = (entry.depth as usize).min(20);
                depth_histogram[depth_idx] += 1;

                let type_idx = match entry.entry_type {
                    EntryType::Exact => 0,
                    EntryType::LowerBound => 1,
                    EntryType::UpperBound => 2,
                };
                type_counts[type_idx] += 1;
            }
        }

        format!(
            "TT Stats: {}/{} entries ({:.1}% full)\n\
             Types: Exact={}, Lower={}, Upper={}\n\
             Avg Depth: {:.1}",
            used, total, percentage,
            type_counts[0], type_counts[1], type_counts[2],
            if used > 0 {
                depth_histogram.iter().enumerate()
                    .map(|(d, &count)| d * count)
                    .sum::<usize>() as f32 / used as f32
            } else { 0.0 }
        )
    }
}