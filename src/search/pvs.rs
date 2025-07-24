use std::time::Instant;
use crate::{board::Board, evaluation, transposition::{TranspositionTable, EntryType}, types::Move};
use super::{SearchContext, quiescence::quiescence_search, ordering::order_moves, see::see_threshold};

const MATE_VALUE: i32 = 99999;
const FUTILITY_MARGIN: [i32; 8] = [0, 300, 500, 900, 1200, 1500, 1800, 2100];
const LMR_MIN_DEPTH: u8 = 3;
const LMR_MIN_MOVES: usize = 4;

/// Principal Variation Search com melhorias para táticas
pub fn pvs_search(
    board: &Board,
    depth: u8,
    mut alpha: i32,
    mut beta: i32,
    tt: &mut TranspositionTable,
    context: &mut SearchContext,
    start_time: Instant,
    max_time_ms: u64,
    is_pv_node: bool
) -> i32 {
    pvs_search_internal(board, depth, alpha, beta, tt, context, start_time, max_time_ms, is_pv_node, 0)
}

/// Internal PVS search with ply tracking
fn pvs_search_internal(
    board: &Board,
    depth: u8,
    mut alpha: i32,
    mut beta: i32,
    tt: &mut TranspositionTable,
    context: &mut SearchContext,
    start_time: Instant,
    max_time_ms: u64,
    is_pv_node: bool,
    ply: usize
) -> i32 {
    context.nodes_searched += 1;

    // CRÍTICO: Check time muito mais frequente (era 2047, agora 255)
    if context.nodes_searched & 1023 == 0 {
        if start_time.elapsed().as_millis() as u64 > max_time_ms {
            context.should_stop = true;
            return 0; // HARD BREAK - para imediatamente
        }
    }

    if context.should_stop {
        return 0;
    }

    // CORREÇÃO: Sempre fazer avaliação para detectar problemas táticos
    let static_eval = evaluation::evaluate(board);

    // CRÍTICO: Lazy evaluation - só avaliação completa nas folhas
    if depth == 0 {
        return quiescence_search(board, alpha, beta, tt, context);
    }

    if depth > 64 {
        return static_eval; // Proteção contra recursão infinita
    }

    let original_alpha = alpha;
    let mut tt_move: Option<Move> = None;

    if let Some(entry) = tt.probe(board.zobrist_hash) {
        tt_move = entry.best_move;
        if entry.depth >= depth && !is_pv_node {
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

    // CORREÇÃO: IID mais agressivo para encontrar movimentos táticos
    if is_pv_node && depth >= 4 && tt_move.is_none() {
        let iid_depth = depth.saturating_sub(2);
        pvs_search_internal(
            board, iid_depth, alpha, beta, tt, context, start_time, max_time_ms, false, ply
        );
        if let Some(entry) = tt.probe(board.zobrist_hash) {
            tt_move = entry.best_move;
        }
    }

    let in_check = board.is_king_in_check(board.to_move);

    // CORREÇÃO: Razoring menos agressivo
    if !is_pv_node && !in_check && depth <= 2 {
        let razor_margin = 500 + 150 * depth as i32; // Era 300 + 100
        if static_eval + razor_margin < alpha {
            let razor_score = quiescence_search(board, alpha, beta, tt, context);
            if razor_score <= alpha {
                return razor_score;
            }
        }
    }

    // CORREÇÃO: Null Move com verificação mais cuidadosa
    if depth >= 3 && !is_pv_node && !in_check && static_eval >= beta {
        // Verifica se há peças penduradas antes de fazer null move
        if !has_hanging_pieces(board) {
            let mut null_board = *board;
            null_board.to_move = !null_board.to_move;
            null_board.en_passant_target = None;

            let r = 2 + depth / 4; // Redução menos agressiva
            let null_depth = depth.saturating_sub(r);

            let null_score = -pvs_search_internal(
                &null_board, null_depth, -beta, -beta + 1, tt, context, start_time, max_time_ms, false, ply + 1
            );

            if null_score >= beta && null_score < MATE_VALUE - 100 {
                return beta;
            }
        }
    }

    let legal_moves = board.generate_legal_moves();

    if legal_moves.is_empty() {
        return if in_check { -(MATE_VALUE - ply as i32) } else { 0 };
    }

    // Passa o tt_move para a função de ordenação
    let ordered_moves = order_moves(board, legal_moves, tt, context, depth);
    let mut best_move = None;
    let mut best_score = -50000;
    let mut moves_searched = 0;
    let mut tried_moves = Vec::with_capacity(ordered_moves.len());

    for mv in &ordered_moves {
        let mut temp_board = *board;
        temp_board.make_move(*mv);

        context.push_move(*mv);

        let gives_check = temp_board.is_king_in_check(!board.to_move);
        let is_capture = board.is_capture(*mv);

        // CORREÇÃO: Extended Futility Pruning menos agressivo
        if depth <= 4 && !is_pv_node && !in_check && !gives_check && !is_capture {
            let futility_value = static_eval + FUTILITY_MARGIN[depth as usize];
            if futility_value <= alpha {
                if !mv.promotion.is_some() {
                    moves_searched += 1;
                    context.pop_move();
                    continue;
                }
            }
        }

        // CORREÇÃO: SEE Pruning apenas para capturas ruins
        if !is_pv_node && is_capture && depth <= 3 && moves_searched > 3 {
            if !see_threshold(board, *mv, -100) { // Era -200
                context.pop_move();
                continue;
            }
        }

        // Extensions mais agressivas para movimentos táticos
        let mut extension = 0;
        if gives_check { extension += 1; }
        if mv.promotion.is_some() { extension += 1; }
        if is_capture && see_threshold(board, *mv, 0) { extension += 1; } // Nova extensão
        extension = extension.min(2);

        let mut score;

        if moves_searched == 0 {
            let first_depth = depth.saturating_sub(1) + extension;
            score = -pvs_search_internal(&temp_board, first_depth, -beta, -alpha, tt, context, start_time, max_time_ms, is_pv_node, ply + 1);
        } else {
            // LMR menos agressivo
            let mut reduction: u8 = 0;
            if depth >= LMR_MIN_DEPTH && moves_searched >= LMR_MIN_MOVES && !is_pv_node
                && !is_capture && !gives_check && !in_check && extension == 0 {

                reduction = if depth <= 8 { 1 } else { 2 };
                reduction = reduction.min(depth.saturating_sub(3));
            }

            let search_depth = depth.saturating_sub(1 + reduction) + extension;
            score = -pvs_search_internal(&temp_board, search_depth, -alpha - 1, -alpha, tt, context, start_time, max_time_ms, false, ply + 1);

            if score > alpha && (is_pv_node || reduction > 0) {
                let full_depth = depth.saturating_sub(1) + extension;
                score = -pvs_search_internal(&temp_board, full_depth, -beta, -alpha, tt, context, start_time, max_time_ms, is_pv_node, ply + 1);
            }
        }

        moves_searched += 1;
        tried_moves.push(*mv);
        context.pop_move();

        if score > best_score {
            best_score = score;
            best_move = Some(*mv);

            if is_pv_node {
                context.update_pv(ply, *mv);
            }
        }

        alpha = alpha.max(score);

        if alpha >= beta {
            use super::ordering::update_context_on_cutoff;
            update_context_on_cutoff(context, board, *mv, depth, &tried_moves);
            break;
        }
    }

    let entry_type = if best_score <= original_alpha {
        EntryType::UpperBound
    } else if best_score >= beta {
        EntryType::LowerBound
    } else {
        EntryType::Exact
    };

    tt.store(board.zobrist_hash, best_move, best_score, depth, entry_type);
    best_score
}

/// Detecta se movimento é uma recaptura
fn is_recapture(board: &Board, mv: Move, context: &SearchContext) -> bool {
    if let Some(last_move) = context.get_last_move() {
        board.is_capture(mv) && mv.to == last_move.to
    } else {
        false
    }
}

fn has_hanging_pieces(board: &Board) -> bool {
    let our_color = board.to_move;
    let our_pieces = if our_color == crate::types::Color::White {
        board.white_pieces
    } else {
        board.black_pieces
    };

    let valuable_pieces = (board.knights | board.bishops | board.rooks | board.queens) & our_pieces;

    let mut bb = valuable_pieces;
    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        bb &= bb - 1;

        if board.is_square_attacked_by(sq, !our_color) &&
            !board.is_square_attacked_by(sq, our_color) {
            return true;
        }
    }

    false
}

/// Versão simplificada para análise
pub fn analyze_position(board: &Board, depth: u8, tt: &mut TranspositionTable) -> (i32, Option<Move>) {
    let mut context = SearchContext::new();
    let start_time = Instant::now();
    let max_time = 30000; // 30 segundos

    let score = pvs_search(board, depth, -50000, 50000, tt, &mut context, start_time, max_time, true);

    let best_move = if let Some(entry) = tt.probe(board.zobrist_hash) {
        entry.best_move
    } else {
        None
    };

    (score, best_move)
}

#[allow(dead_code)]
fn has_hanging_pieces_simple(board: &Board) -> bool {
    let our_valuables = (board.knights | board.bishops | board.rooks | board.queens) &
        if board.to_move == crate::types::Color::White { board.white_pieces } else { board.black_pieces };

    let mut bb = our_valuables;
    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        bb &= bb - 1;

        if board.is_square_attacked_by(sq, !board.to_move) {
            return true;
        }
    }
    false
}