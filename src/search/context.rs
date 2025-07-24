use crate::types::{Move, PieceKind};

#[derive(Debug)]
pub struct SearchContext {
    // Killer moves: 2 killers por profundidade
    killer_moves: [[Option<Move>; 2]; 64], // Expandido para 64 profundidades
    // History heuristic: [piece_type][from][to]
    history: [[[i32; 64]; 64]; 6], // Por tipo de peça
    // Counter moves: resposta eficaz ao último movimento
    counter_moves: [[Option<Move>; 64]; 64], // [from][to] -> counter_move
    pub nodes_searched: u64,
    // Novo para Fase 2: Rastrear PV para time management
    pub prev_best_move: Option<Move>,
    // Stop flag para timeout graceful
    pub should_stop: bool,
    // Rastrear último movimento para counter-moves
    last_moves: Vec<Move>, // Stack de movimentos recentes
    // Principal Variation tracking
    pub pv_table: [[Option<Move>; 64]; 64], // [depth][ply]
    pub pv_length: [usize; 64], // Length of PV at each depth
    // Search information for detailed logging
    pub current_depth: u8,
    pub current_move_number: usize,
    pub total_moves_at_depth: usize,
}

impl SearchContext {
    pub fn new() -> Self {
        SearchContext {
            killer_moves: [[None; 2]; 64],
            history: [[[0; 64]; 64]; 6], // Inicializa com zeros para todos os tipos de peça
            counter_moves: [[None; 64]; 64],
            nodes_searched: 0,
            prev_best_move: None,
            should_stop: false,
            last_moves: Vec::with_capacity(128), // Capacidade para ply profundo
            pv_table: [[None; 64]; 64],
            pv_length: [0; 64],
            current_depth: 0,
            current_move_number: 0,
            total_moves_at_depth: 0,
        }
    }

    /// Adiciona killer move - movimento não-captura que causou cutoff
    pub fn add_killer(&mut self, mv: Move, depth: u8) {
        let depth_idx = depth as usize;
        if depth_idx < 64 {
            // Só adiciona se não for captura (killers são quiet moves)
            if mv.promotion.is_none() && !mv.is_en_passant {
                // Move o killer atual para segunda posição
                if self.killer_moves[depth_idx][0] != Some(mv) {
                    self.killer_moves[depth_idx][1] = self.killer_moves[depth_idx][0];
                    self.killer_moves[depth_idx][0] = Some(mv);
                }
            }
        }
    }

    pub fn is_killer(&self, mv: Move, depth: u8) -> bool {
        let depth_idx = depth as usize;
        if depth_idx < 64 {
            self.killer_moves[depth_idx].contains(&Some(mv))
        } else {
            false
        }
    }

    /// Atualiza history heuristic por tipo de peça
    pub fn update_history(&mut self, mv: Move, piece_type: PieceKind, depth: u8, is_good: bool) {
        let piece_idx = piece_type as usize;
        let from_idx = mv.from as usize;
        let to_idx = mv.to as usize;
        
        if piece_idx < 6 && from_idx < 64 && to_idx < 64 {
            let bonus = (depth as i32) * (depth as i32);
            if is_good {
                // Movimento bom: aumenta score
                self.history[piece_idx][from_idx][to_idx] += bonus;
                // Limite máximo para evitar overflow
                if self.history[piece_idx][from_idx][to_idx] > 10000 {
                    self.history[piece_idx][from_idx][to_idx] = 10000;
                }
            } else {
                // Movimento ruim: diminui score
                self.history[piece_idx][from_idx][to_idx] -= bonus / 2;
                if self.history[piece_idx][from_idx][to_idx] < -5000 {
                    self.history[piece_idx][from_idx][to_idx] = -5000;
                }
            }
        }
    }

    pub fn get_history_score(&self, mv: Move, piece_type: PieceKind) -> i32 {
        let piece_idx = piece_type as usize;
        let from_idx = mv.from as usize;
        let to_idx = mv.to as usize;
        
        if piece_idx < 6 && from_idx < 64 && to_idx < 64 {
            self.history[piece_idx][from_idx][to_idx]
        } else {
            0
        }
    }
    
    /// Counter moves: armazena resposta eficaz ao movimento anterior
    pub fn add_counter_move(&mut self, prev_move: Move, counter_move: Move) {
        let from_idx = prev_move.from as usize;
        let to_idx = prev_move.to as usize;
        
        if from_idx < 64 && to_idx < 64 {
            self.counter_moves[from_idx][to_idx] = Some(counter_move);
        }
    }

    pub fn get_counter_move(&self, prev_move: Move) -> Option<Move> {
        let from_idx = prev_move.from as usize;
        let to_idx = prev_move.to as usize;
        
        if from_idx < 64 && to_idx < 64 {
            self.counter_moves[from_idx][to_idx]
        } else {
            None
        }
    }

    /// Gerencia stack de movimentos para counter-moves
    pub fn push_move(&mut self, mv: Move) {
        self.last_moves.push(mv);
    }

    pub fn pop_move(&mut self) -> Option<Move> {
        self.last_moves.pop()
    }

    pub fn get_last_move(&self) -> Option<Move> {
        self.last_moves.last().copied()
    }

    /// Decay history scores periodicamente para esquecer informação antiga
    pub fn age_history_scores(&mut self) {
        for piece in 0..6 {
            for from in 0..64 {
                for to in 0..64 {
                    self.history[piece][from][to] = (self.history[piece][from][to] * 7) / 8;
                }
            }
        }
    }

    /// Reset killers para nova busca
    pub fn clear_killers(&mut self) {
        self.killer_moves = [[None; 2]; 64];
    }

    /// Principal Variation management
    pub fn update_pv(&mut self, ply: usize, best_move: Move) {
        if ply < 64 {
            self.pv_table[ply][ply] = Some(best_move);
            
            // Copy PV from child node
            for i in (ply + 1)..64 {
                if ply + 1 < 64 && i < 64 {
                    self.pv_table[ply][i] = self.pv_table[ply + 1][i];
                    if self.pv_table[ply][i].is_none() {
                        break;
                    }
                } else {
                    break;
                }
            }
            
            // Update PV length
            self.pv_length[ply] = 1;
            if ply + 1 < 64 {
                self.pv_length[ply] += self.pv_length[ply + 1];
            }
        }
    }

    pub fn get_pv(&self, ply: usize) -> Vec<Move> {
        let mut pv = Vec::new();
        if ply < 64 {
            for i in ply..64 {
                if let Some(mv) = self.pv_table[ply][i] {
                    pv.push(mv);
                } else {
                    break;
                }
            }
        }
        pv
    }

    pub fn clear_pv(&mut self) {
        self.pv_table = [[None; 64]; 64];
        self.pv_length = [0; 64];
    }

    /// Format PV as string for UCI output
    pub fn format_pv(&self, ply: usize) -> String {
        let pv = self.get_pv(ply);
        pv.iter()
            .map(|mv| format!("{}", mv))
            .collect::<Vec<_>>()
            .join(" ")
    }

}