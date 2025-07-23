// Mobilidade e estratégia avançada de torres
use crate::types::{Color, Bitboard};
use super::{MobilityContext, MobilityResult, GamePhase, utils::*};

/// Avaliação avançada de mobilidade de torres
pub fn evaluate_rook_mobility_advanced(context: &MobilityContext) -> i32 {
    let mut total_score = 0;
    let rooks = context.board.rooks & context.our_pieces;
    
    if rooks == 0 { return 0; }
    
    let rook_squares = get_set_bits(rooks);
    
    for &rook_sq in &rook_squares {
        let file = rook_sq % 8;
        let rank = rook_sq / 8;
        
        // Mobilidade básica
        let attacks = crate::moves::sliding::get_rook_attacks(rook_sq, context.all_pieces);
        let legal_squares = attacks & !context.our_pieces;
        let safe_squares = legal_squares & !context.enemy_attacked_squares;
        
        total_score += legal_squares.count_ones() as i32 * 2;
        total_score += safe_squares.count_ones() as i32 * 3;
        
        // Bônus por arquivo aberto
        if is_open_file(file, context) {
            total_score += match context.phase {
                GamePhase::Opening => 8,
                GamePhase::MiddleGame => 15,
                GamePhase::Endgame => 10,
            };
            
            // Bônus extra se penetra no território inimigo
            let enemy_territory = if context.enemy_color == Color::White {
                rank >= 5
            } else {
                rank <= 2
            };
            
            if enemy_territory {
                total_score += 10;
            }
        }
        
        // Bônus por arquivo semi-aberto
        else if is_semi_open_file(file, context.color, context) {
            total_score += match context.phase {
                GamePhase::Opening => 5,
                GamePhase::MiddleGame => 8,
                GamePhase::Endgame => 6,
            };
        }
        
        // Bônus por controlar fileira importante
        if rank == 1 || rank == 6 { // 2ª ou 7ª fileira
            let enemy_pawns_on_rank = get_rank_mask(rank) & (context.board.pawns & context.enemy_pieces);
            if enemy_pawns_on_rank != 0 {
                total_score += 8;
            }
        }
        
        // Torres dobradas
        let file_mask = get_file_mask(file);
        let friendly_rooks_on_file = (context.board.rooks & context.our_pieces & file_mask).count_ones();
        if friendly_rooks_on_file > 1 {
            total_score += 12; // Torres dobradas em arquivo
        }
        
        // Suporte a peões passados
        let our_pawns = context.board.pawns & context.our_pieces;
        let passed_pawns_file = file_mask & our_pawns;
        if passed_pawns_file != 0 {
            // Verifica se é realmente peão passado (simplificado)
            total_score += 8;
        }
    }
    
    total_score
}