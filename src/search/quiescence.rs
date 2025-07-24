
// CORREÇÕES PARA quiescence.rs

use crate::{board::Board, evaluation, transposition::TranspositionTable, types::{Move, Color}};
use super::{SearchContext, order_moves, see::see};

const MATE_VALUE: i32 = 99999;
const SEE_THRESHOLD: i32 = -20; // Era -50, agora menos permissivo

/// Quiescence Search - busca táticas até posição "quieta"
pub fn quiescence_search(
    board: &Board,
    mut alpha: i32,
    beta: i32,
    tt: &mut TranspositionTable,
    context: &mut SearchContext
) -> i32 {
    quiescence_search_with_ply(board, alpha, beta, 0, 8, tt, context) // Aumentado para 8
}


/// Quiescence search com limite de ply para prevenir recursão infinita
/// Quiescence search com limite de ply para prevenir recursão infinita
fn quiescence_search_with_ply(
    board: &Board,
    mut alpha: i32,
    beta: i32,
    ply: i32,
    max_ply: i32,
    tt: &mut TranspositionTable,
    context: &mut SearchContext
) -> i32 {
    // Termina se atingiu limite de ply ou stop flag
    if ply >= max_ply || context.should_stop {
        return evaluation::evaluate(board);
    }

    let in_check = board.is_king_in_check(board.to_move);

    // Stand pat - avaliação da posição quieta
    let stand_pat = evaluation::evaluate(board);

    // Beta cutoff - apenas se não estamos em xeque
    if !in_check && stand_pat >= beta {
        return beta;
    }

    // Atualiza alpha
    if !in_check && alpha < stand_pat {
        alpha = stand_pat;
    }

    // CORREÇÃO: Delta pruning ajustado para considerar melhor as capturas
    if !in_check && stand_pat + 900 < alpha { // Era 700
        return alpha;
    }

    // Gera capturas e movimentos de xeque
    let mut tactical_moves = Vec::new();

    if in_check {
        // Em xeque: considera todos os moves legais
        tactical_moves.extend(board.generate_legal_moves());
    } else {
        // CORREÇÃO: Gerar capturas de forma mais eficiente e completa
        let all_moves = board.generate_legal_moves();

        for mv in all_moves {
            // Inclui TODAS as capturas
            if board.is_capture(mv) {
                tactical_moves.push(mv);
            }
            // Inclui promoções (mesmo não-capturas)
            else if mv.promotion.is_some() {
                tactical_moves.push(mv);
            }
            // Inclui checks
            else if gives_check_fast(board, mv) {
                tactical_moves.push(mv);
            }
        }
    }

    // CORREÇÃO: Filtro SEE menos agressivo
    if !in_check {
        tactical_moves.retain(|&mv| {
            if board.is_capture(mv) {
                // SEMPRE considerar capturas de peças valiosas
                if let Some(captured_piece) = board.get_piece_on_square(mv.to) {
                    match captured_piece {
                        crate::types::PieceKind::Queen => return true,
                        crate::types::PieceKind::Rook => return true,
                        crate::types::PieceKind::Bishop | crate::types::PieceKind::Knight => {
                            // Para peças menores, usar SEE mais permissivo
                            return see(board, mv) >= -50;
                        }
                        _ => {}
                    }
                }
                see(board, mv) >= SEE_THRESHOLD
            } else {
                true // Mantém checks e promoções
            }
        });
    }

    // Ordena movimentos (capturas boas primeiro)
    let ordered_moves = order_moves(board, tactical_moves, tt, context, 0);

    // CORREÇÃO: Sempre avaliar pelo menos as primeiras capturas
    let min_moves_to_try = if in_check {
        ordered_moves.len()
    } else {
        ordered_moves.len().min(10) // Limita mas garante que vemos as principais
    };

    for (idx, mv) in ordered_moves.iter().take(min_moves_to_try).enumerate() {
        // Valida legalidade
        if !board.is_legal_move(*mv) {
            continue;
        }

        // Faz movimento e busca recursivamente
        let mut temp_board = *board;
        temp_board.make_move(*mv);

        let score = -quiescence_search_with_ply(&temp_board, -beta, -alpha, ply + 1, max_ply, tt, context);

        // Beta cutoff
        if score >= beta {
            return beta;
        }

        // Atualiza alpha
        if score > alpha {
            alpha = score;
        }
    }

    alpha
}

/// Verifica rapidamente se um movimento dá xeque (sem fazer o movimento)
/// Versão simplificada - pode dar falsos negativos, mas sem falsos positivos
pub fn gives_check_fast(board: &Board, mv: Move) -> bool {
    let piece_kind = board.get_piece_on_square(mv.from);
    if piece_kind.is_none() {
        return false;
    }

    let enemy_king_bb = board.kings & if board.to_move == Color::White {
        board.black_pieces
    } else {
        board.white_pieces
    };

    if enemy_king_bb == 0 {
        return false;
    }

    let enemy_king_square = enemy_king_bb.trailing_zeros() as u8;

    // Verifica se a peça movida pode atacar o rei inimigo da nova posição
    match piece_kind.unwrap() {
        crate::types::PieceKind::Knight => {
            let attacks = crate::moves::knight::get_knight_attacks_lookup(mv.to);
            (attacks & (1u64 << enemy_king_square)) != 0
        },
        crate::types::PieceKind::Bishop | crate::types::PieceKind::Queen => {
            // Simula ocupação após movimento
            let mut temp_occ = board.white_pieces | board.black_pieces;
            temp_occ &= !(1u64 << mv.from); // Remove peça da origem
            temp_occ |= 1u64 << mv.to;      // Adiciona na destino

            let attacks = crate::moves::sliding::get_bishop_attacks(mv.to, temp_occ);
            (attacks & (1u64 << enemy_king_square)) != 0
        },
        crate::types::PieceKind::Rook => {
            // This variable will hold the result of whether the king is attacked
            let king_is_attacked: bool;

            if piece_kind.unwrap() == crate::types::PieceKind::Queen {
                // Rainha também verifica ataques de torre
                let mut temp_occ = board.white_pieces | board.black_pieces;
                temp_occ &= !(1u64 << mv.from);
                temp_occ |= 1u64 << mv.to;

                let attacks = crate::moves::sliding::get_rook_attacks(mv.to, temp_occ);
                king_is_attacked = (attacks & (1u64 << enemy_king_square)) != 0;
            } else {
                let mut temp_occ = board.white_pieces | board.black_pieces;
                temp_occ &= !(1u64 << mv.from);
                temp_occ |= 1u64 << mv.to;

                let attacks = crate::moves::sliding::get_rook_attacks(mv.to, temp_occ);
                king_is_attacked = (attacks & (1u64 << enemy_king_square)) != 0;
            }
            king_is_attacked
        },
        crate::types::PieceKind::Pawn => {
            // Ataques de peão
            let rank_diff = (mv.to / 8) as i8 - (enemy_king_square / 8) as i8;
            let file_diff = (mv.to % 8) as i8 - (enemy_king_square % 8) as i8;

            if board.to_move == Color::White {
                rank_diff == 1 && file_diff.abs() == 1
            } else {
                rank_diff == -1 && file_diff.abs() == 1
            }
        },
        _ => false, // Rei não dá xeque direto normalmente
    }
}

/// Versão alternativa de quiescence para uso em análise
pub fn quiescence_search_depth_limited(
    board: &Board,
    mut alpha: i32,
    beta: i32,
    depth_left: i32,
    tt: &mut TranspositionTable,
    context: &mut SearchContext
) -> i32 {
    if depth_left <= 0 {
        return evaluation::evaluate(board);
    }

    quiescence_search(board, alpha, beta, tt, context)
}