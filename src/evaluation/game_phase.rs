// Detecção de fase do jogo
use crate::board::Board;

#[derive(Debug, Clone, Copy)]
pub enum GamePhase {
    Opening,
    Middlegame,
    Endgame,
}

pub fn detect_game_phase(board: &Board) -> GamePhase {
    let mut game_phase_score = 0;
    
    // Pontuação baseada no material pesado
    game_phase_score += (board.knights.count_ones() as i32) * 1;
    game_phase_score += (board.bishops.count_ones() as i32) * 1;
    game_phase_score += (board.rooks.count_ones() as i32) * 2;
    game_phase_score += (board.queens.count_ones() as i32) * 4;
    
    if game_phase_score > 20 {
        GamePhase::Opening
    } else if game_phase_score > 10 {
        GamePhase::Middlegame
    } else {
        GamePhase::Endgame
    }
}

/// Estrutura para informações avançadas da fase do jogo
#[derive(Debug, Clone, Copy)]
pub struct GamePhaseInfo {
    pub phase: GamePhase,
    pub material_count: u32,
    pub phase_value: f32,
}

/// Versão avançada de detecção de fase do jogo
pub fn detect_game_phase_advanced(board: &Board) -> GamePhaseInfo {
    let total_pieces = (board.white_pieces | board.black_pieces).count_ones();
    let phase = detect_game_phase(board);
    let phase_value = match phase {
        GamePhase::Opening => 0.0,
        GamePhase::Middlegame => 0.5,
        GamePhase::Endgame => 1.0,
    };
    
    GamePhaseInfo {
        phase,
        material_count: total_pieces,
        phase_value,
    }
}

/// Interpola valores baseado na fase do jogo
pub fn interpolate_phase_i32(opening: i32, endgame: i32, phase_info: &GamePhaseInfo) -> i32 {
    let ratio = phase_info.phase_value;
    ((1.0 - ratio) * opening as f32 + ratio * endgame as f32) as i32
}