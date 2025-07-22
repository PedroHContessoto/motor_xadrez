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