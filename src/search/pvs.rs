use std::time::Instant;
use crate::{board::Board, evaluation, transposition::{TranspositionTable, EntryType}, types::Move};
use super::{SearchContext, quiescence_search, order_moves};
// Novo: Import constants de mod.rs
use super::{MATE_VALUE, FUTILITY_MARGIN, LMR_MIN_DEPTH, LMR_MIN_MOVES, LMR_REDUCTION};

pub fn pvs_search(board: &Board, depth: u8, mut alpha: i32, mut beta: i32, tt: &mut TranspositionTable, context: &mut SearchContext, start_time: Instant, max_time_ms: u64, is_pv_node: bool) -> i32 {
    context.nodes_searched += 1;

    if context.nodes_searched % 4096 == 0 {
        if start_time.elapsed().as_millis() as u64 > max_time_ms {
            return 0;
        }
    }

    let original_alpha = alpha;

    if let Some(entry) = tt.probe(board.zobrist_hash) {
        if entry.depth >= depth {
            match entry.entry_type {
                EntryType::Exact => return entry.score,
                EntryType::LowerBound => alpha = alpha.max(entry.score),
                EntryType::UpperBound => beta = beta.min(entry.score),
            }
            if alpha >= beta {
                return entry.score;
            }
        }
    }

    if board.is_draw_by_50_moves() || board.is_draw_by_insufficient_material() {
        return 0;
    }

    if depth >= 3 && !is_pv_node && !board.is_king_in_check(board.to_move) {
        let mut null_board = *board;
        null_board.to_move = !null_board.to_move;
        null_board.en_passant_target = None;
        let temp_null_score = pvs_search(&null_board, depth - 3, -beta, -beta + 1, tt, context, start_time, max_time_ms, false);
        let null_score = -temp_null_score;
        if null_score >= beta {
            return beta;
        }
    }

    if depth == 0 {
        return quiescence_search(board, alpha, beta, tt, context);
    }

    let mut best_move = None;
    let legal_moves = board.generate_legal_moves();

    if legal_moves.is_empty() {
        if board.is_king_in_check(board.to_move) {
            return -(MATE_VALUE - depth as i32);
        } else {
            return 0;
        }
    }

    let ordered_moves = order_moves(board, legal_moves, tt, context, depth);
    let mut best_score = -50000;
    let mut moves_searched = 0;

    // Novo: Compute static_eval cedo para uso em extensions e futility
    let static_eval = evaluation::evaluate(board);

    for mv in ordered_moves {
        let mut temp_board = *board;
        temp_board.make_move(mv);

        // Futility Pruning ajustado Fase 1: Desabilita em captures/checks
        if depth <= 3 && !is_pv_node && !board.is_capture(mv) && !board.is_king_in_check(board.to_move) && !temp_board.is_king_in_check(!board.to_move) {
            let futility_margin = FUTILITY_MARGIN[depth as usize];
            if static_eval + futility_margin <= alpha {
                continue;
            }
        }

        // Extensões Fase 2: +1 para recaptures e singular
        let mut extension = if temp_board.is_king_in_check(!board.to_move) || mv.promotion.is_some() { 1 } else { 0 };
        extension += if is_recapture(board, mv) { 1 } else { 0 }; // Novo: Recapture
        if is_pv_node && moves_searched == 0 && depth >= 5 { // Singular básico
            let singular_beta = alpha - 50; // Margin
            let singular_score = pvs_search(&temp_board, depth / 2, singular_beta - 1, singular_beta, tt, context, start_time, max_time_ms, false);
            if singular_score < singular_beta {
                extension += 1;
            }
        }
        // Novo Fix: Substitua condição problemática por check-based (evita scope error)
        extension += if temp_board.is_king_in_check(!board.to_move) && static_eval.abs() > 5000 { 1 } else { 0 };  // Extend near mate via check e eval high

        let mut score;

        if moves_searched == 0 {
            let first_depth = if depth > 1 { depth - 1 + extension } else { extension };
            let first_score = pvs_search(&temp_board, first_depth, -beta, -alpha, tt, context, start_time, max_time_ms, is_pv_node);
            score = -first_score;
        } else {
            let mut reduction = 0;
            if depth >= LMR_MIN_DEPTH && moves_searched >= LMR_MIN_MOVES && !is_pv_node
                && !board.is_capture(mv) && !temp_board.is_king_in_check(!board.to_move) {
                reduction = LMR_REDUCTION;
            }
            // Ajuste Fase 1: No reduction em PV/captures/checks
            if is_pv_node || board.is_capture(mv) || temp_board.is_king_in_check(!board.to_move) {
                reduction = 0;
            }

            let search_depth = if depth > 1 + reduction { depth - 1 - reduction + extension } else { 0 };
            let temp_score = pvs_search(&temp_board, search_depth, -alpha - 1, -alpha, tt, context, start_time, max_time_ms, false);
            score = -temp_score;

            if score > alpha && is_pv_node {
                let full_depth = if depth > 1 { depth - 1 + extension } else { extension };
                let full_score = pvs_search(&temp_board, full_depth, -beta, -alpha, tt, context, start_time, max_time_ms, true);
                score = -full_score;
            }
        }

        moves_searched += 1;

        if score > best_score {
            best_score = score;
            best_move = Some(mv);
            if !board.is_capture(mv) {
                context.update_history(mv, depth);
            }
        }

        alpha = alpha.max(score);
        if alpha >= beta {
            if !board.is_capture(mv) {
                context.add_killer(mv, depth);
            }
            break;
        }
    }

    let entry_type = if best_score <= original_alpha { EntryType::UpperBound }
    else if best_score >= beta { EntryType::LowerBound }
    else { EntryType::Exact };

    tt.store(board.zobrist_hash, best_move, best_score, depth, entry_type);
    best_score
}

// Helper novo Fase 2
fn is_recapture(_board: &Board, mv: Move) -> bool {
    // TODO: Rastrear prev_target; por agora, assume false ou implemente context com last_move
    false // Placeholder; adicione lógica real
}