// Mobilidade e estratégia avançada da rainha
use crate::types::{Color, Bitboard};
use super::{MobilityContext, MobilityResult, GamePhase, utils::*};
use crate::evaluation::utils as eval_utils;

/// Avaliação avançada de mobilidade da rainha
pub fn evaluate_queen_mobility_advanced(context: &MobilityContext) -> i32 {
    let mut total_score = 0;
    let queens = context.board.queens & context.our_pieces;

    if queens == 0 { return 0; }

    let queen_squares = get_set_bits(queens);

    for &queen_sq in &queen_squares {
        // Mobilidade combinada (bispo + torre)
        let bishop_attacks = crate::moves::sliding::get_bishop_attacks(queen_sq, context.all_pieces);
        let rook_attacks = crate::moves::sliding::get_rook_attacks(queen_sq, context.all_pieces);
        let total_attacks = bishop_attacks | rook_attacks;

        let legal_squares = total_attacks & !context.our_pieces;
        let safe_squares = legal_squares & !context.enemy_attacked_squares;

        // Mobilidade básica (reduzida - rainha já é muito valiosa)
        total_score += legal_squares.count_ones() as i32;
        total_score += safe_squares.count_ones() as i32;

        // Penalidade por desenvolvimento prematuro
        if context.phase == GamePhase::Opening {
            let back_rank = if context.color == Color::White {
                queen_sq >= 0 && queen_sq <= 7
            } else {
                queen_sq >= 56 && queen_sq <= 63
            };

            if !back_rank {
                total_score -= 15; // Penalidade por sair cedo
            }
        }

        // Bônus por centralização (moderado)
        if is_central_square(queen_sq) {
            total_score += match context.phase {
                GamePhase::Opening => 0,  // Não queremos centralizar cedo
                GamePhase::MiddleGame => 8,
                GamePhase::Endgame => 12,
            };
        }

        // Detecta ameaças múltiplas
        let enemy_pieces = legal_squares & context.enemy_pieces;
        if enemy_pieces.count_ones() >= 2 {
            total_score += 10; // Bônus por atacar múltiplas peças
        }

        // Atividade no final do jogo
        if context.phase == GamePhase::Endgame {
            // Rainha ativa próxima ao rei inimigo
            let enemy_king = context.board.kings & context.enemy_pieces;
            if enemy_king != 0 {
                let king_sq = enemy_king.trailing_zeros() as u8;
                let distance = eval_utils::calculate_square_distance(queen_sq, king_sq);
                if distance <= 3 {
                    total_score += (4 - distance) * 3;
                }
            }
        }
    }

    total_score
}

