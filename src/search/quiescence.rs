use crate::{board::Board, evaluation, transposition::TranspositionTable, types::Move};
use super::{SearchContext, order_moves, see::see};
use super::MATE_VALUE;  // Novo import para mate score

pub fn quiescence_search(board: &Board, mut alpha: i32, beta: i32, tt: &mut TranspositionTable, context: &mut SearchContext) -> i32 {
    // Novo: Detecte mate/stalemate cedo (como em pvs)
    let legal_moves = board.generate_legal_moves();  // Gere early para check
    if legal_moves.is_empty() {
        if board.is_king_in_check(board.to_move) {
            return -MATE_VALUE;  // Mate: negativo
        } else {
            return 0;  // Stalemate
        }
    }

    let stand_pat = evaluation::evaluate(board);

    if stand_pat >= beta {
        return beta;
    }

    if alpha < stand_pat {
        alpha = stand_pat;
    }

    if stand_pat + 900 < alpha {
        return alpha;
    }

    let mut captures: Vec<Move> = Vec::new();
    captures.extend(crate::moves::pawn::generate_pawn_captures(board));
    captures.extend(crate::moves::knight::generate_knight_moves(board).into_iter().filter(|mv| board.is_capture(*mv)));
    if !board.is_king_in_check(board.to_move) {
        for mv in legal_moves {  // Re-use legal_moves para efficiency
            if !board.is_capture(mv) {
                let mut temp_board = *board;
                temp_board.make_move(mv);
                if temp_board.is_king_in_check(!board.to_move) {
                    captures.push(mv);
                }
            }
        }
    }

    let ordered_captures = order_moves(board, captures, tt, context, 0);

    for mv in ordered_captures {
        if see(board, mv) < 0 { continue; }

        if !board.is_legal_move(mv) { continue; }
        let mut temp_board = *board;
        temp_board.make_move(mv);
        let score = -quiescence_search(&temp_board, -beta, -alpha, tt, context);
        if score >= beta {
            return beta;
        }
        if score > alpha {
            alpha = score;
        }
    }

    alpha
}