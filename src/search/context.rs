use crate::types::Move;

#[derive(Debug)]
pub struct SearchContext {
    killer_moves: [[Option<Move>; 2]; 32],
    history: [[i32; 64]; 64],
    pub nodes_searched: u64,
    // Novo para Fase 2: Rastrear PV para time management
    pub prev_best_move: Option<Move>,
    // Stop flag para timeout graceful
    pub should_stop: bool,
}

impl SearchContext {
    pub fn new() -> Self {
        SearchContext {
            killer_moves: [[None; 2]; 32],
            history: [[0; 64]; 64],
            nodes_searched: 0,
            prev_best_move: None,
            should_stop: false,
        }
    }

    pub fn add_killer(&mut self, mv: Move, depth: u8) {
        let depth_idx = depth as usize;
        if depth_idx < 32 {
            self.killer_moves[depth_idx][1] = self.killer_moves[depth_idx][0];
            self.killer_moves[depth_idx][0] = Some(mv);
        }
    }

    pub fn is_killer(&self, mv: Move, depth: u8) -> bool {
        let depth_idx = depth as usize;
        if depth_idx < 32 {
            self.killer_moves[depth_idx].contains(&Some(mv))
        } else {
            false
        }
    }

    pub fn update_history(&mut self, mv: Move, depth: u8) {
        self.history[mv.from as usize][mv.to as usize] += (depth as i32) * (depth as i32);
    }

    pub fn get_history_score(&self, mv: Move) -> i32 {
        self.history[mv.from as usize][mv.to as usize]
    }
    
    pub fn get_last_move(&self) -> Option<Move> {
        // Simplified implementation - can be expanded later
        None
    }
}