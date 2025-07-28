use std::time::Instant;
use crate::{board::Board, evaluation, transposition::{TranspositionTable, EntryType}, types::Move};
use super::{SearchContext, quiescence::quiescence_search, ordering::order_moves, see::see_threshold};

const MATE_VALUE: i32 = 99999;
const FUTILITY_MARGIN: [i32; 8] = [0, 200, 350, 600, 900, 1200, 1500, 1800];
const REVERSE_FUTILITY_MARGIN: [i32; 8] = [0, 120, 240, 360, 480, 600, 720, 840];
const LMR_MIN_DEPTH: u8 = 2;
const LMR_MIN_MOVES: usize = 3;
const PROBCUT_DEPTH: u8 = 5;
const PROBCUT_MARGIN: i32 = 200;
const SINGULAR_EXTENSION_DEPTH: u8 = 6;
const ASPIRATION_WINDOW: i32 = 25;

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

    if context.nodes_searched & 2047 == 0 {
        if start_time.elapsed().as_millis() as u64 > max_time_ms {
            context.should_stop = true;
        }
    }

    if context.should_stop {
        return 0;
    }

    if depth > 64 {
        return evaluation::evaluate(board);
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

    // IID (Internal Iterative Deepening) para melhorar a ordenação quando não há lance da TT
    if is_pv_node && depth >= 6 && tt_move.is_none() {
        let iid_depth = if depth > 8 { depth - 4 } else { depth - 2 }; // Redução adaptativa
        pvs_search_internal(
            board, iid_depth, alpha, beta, tt, context, start_time, max_time_ms, false, ply
        );
        // Após a busca IID, a TT deve ter um lance para esta posição
        if let Some(entry) = tt.probe(board.zobrist_hash) {
            tt_move = entry.best_move;
        }
    }

    if depth == 0 {
        return quiescence_search(board, alpha, beta, tt, context);
    }

    let in_check = board.is_king_in_check(board.to_move);
    let static_eval = if !in_check { evaluation::evaluate(board) } else { -MATE_VALUE / 2 };

    // Reverse Futility Pruning (Static Null Move Pruning)
    if !is_pv_node && !in_check && depth <= 7 && static_eval != -MATE_VALUE / 2 {
        let rfp_margin = REVERSE_FUTILITY_MARGIN[depth as usize];
        if static_eval - rfp_margin >= beta {
            return static_eval - rfp_margin;
        }
    }

    // Razoring (melhorada)
    if !is_pv_node && !in_check && depth <= 3 && static_eval + FUTILITY_MARGIN[depth as usize] < alpha {
        let razor_score = quiescence_search(board, alpha, beta, tt, context);
        if razor_score <= alpha {
            return razor_score;
        }
    }

    // ProbCut
    if !is_pv_node && depth >= PROBCUT_DEPTH && static_eval >= beta {
        let probcut_beta = beta + PROBCUT_MARGIN;
        let probcut_moves = board.generate_legal_moves().into_iter()
            .filter(|mv| board.is_capture(*mv) || gives_check_fast(board, *mv))
            .take(3)
            .collect::<Vec<_>>();
            
        for mv in probcut_moves {
            let mut temp_board = *board;
            temp_board.make_move(mv);
            let probcut_score = -pvs_search_internal(
                &temp_board, depth - 4, -probcut_beta, -probcut_beta + 1, 
                tt, context, start_time, max_time_ms, false, ply + 1
            );
            if probcut_score >= probcut_beta {
                return probcut_score;
            }
        }
    }

    // Null Move Pruning (melhorada)
    if depth >= 2 && !is_pv_node && !in_check && static_eval >= beta && has_non_pawn_material(board) {
        let mut null_board = *board;
        null_board.to_move = !null_board.to_move;
        null_board.en_passant_target = None;
        null_board.halfmove_clock += 1;

        // Redução adaptativa baseada na profundidade e margem de avaliação
        let eval_margin = static_eval - beta;
        let base_reduction = if depth <= 6 { 3 } else { 4 };
        let eval_reduction = (eval_margin / 200).min(2) as u8;
        let r = base_reduction + eval_reduction;
        let null_depth = depth.saturating_sub(r);

        let null_score = -pvs_search_internal(
            &null_board, null_depth, -beta, -beta + 1, tt, context, start_time, max_time_ms, false, ply + 1
        );

        if null_score >= beta {
            // Verificação apenas em profundidades mais altas
            if depth < 14 || null_score >= MATE_VALUE - 100 {
                return beta;
            }
            let verify_depth = depth.saturating_sub(r + 2);
            let verify_score = pvs_search_internal(
                board, verify_depth, beta - 1, beta, tt, context, start_time, max_time_ms, false, ply
            );
            if verify_score >= beta {
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

    // Multi-cut pruning
    if !is_pv_node && depth >= 8 && moves_searched >= 3 {
        let mut cut_count = 0;
        const MC_MOVES_TO_TRY: usize = 6;

        for (i, mv) in ordered_moves.iter().take(MC_MOVES_TO_TRY).enumerate() {
            if i >= moves_searched { break; }

            // Usa copy-make para performance
            let temp_board = board.make_move_copy(*mv);

            let score = -pvs_search_internal(
                &temp_board, depth - 3, -alpha - 1, -alpha, tt, context, start_time, max_time_ms, false, ply + 1
            );

            if score > alpha {
                cut_count += 1;
                if cut_count >= 3 {
                    return beta;
                }
            }
        }
    }

    for mv in &ordered_moves {
        // Usa copy-make para performance
        let temp_board = board.make_move_copy(*mv);

        context.push_move(*mv);

        let gives_check = temp_board.is_king_in_check(!board.to_move);
        let is_capture = board.is_capture(*mv);

        // Extended Futility Pruning (melhorada)
        if depth <= 6 && !is_pv_node && !in_check && !gives_check && moves_searched > 0 {
            let futility_value = static_eval + FUTILITY_MARGIN[depth as usize];
            if futility_value <= alpha && !is_capture && mv.promotion.is_none() {
                // Exceção para movimentos que podem melhorar a posição significativamente
                if !is_killer_or_counter_move(*mv, context, depth) {
                    moves_searched += 1;
                    context.pop_move();
                    continue;
                }
            }
        }

        // Late Move Count Pruning
        if depth <= 8 && !is_pv_node && !in_check && !gives_check && !is_capture 
            && mv.promotion.is_none() && moves_searched >= late_move_count_threshold(depth) {
            moves_searched += 1;
            context.pop_move();
            continue;
        }

        // SEE Pruning
        if !is_pv_node && is_capture && depth <= 4 && moves_searched > 0 {
            if !see_threshold(board, *mv, -200) {
                context.pop_move();
                continue;
            }
        }

        // Extensions
        let mut extension = 0;
        if gives_check { extension += 1; }
        if mv.promotion.is_some() { extension += 1; }
        if is_recapture(board, *mv, context) { extension += 1; }
        extension = extension.min(2);

        let mut score;

        if moves_searched == 0 {
            let first_depth = depth.saturating_sub(1) + extension;
            score = -pvs_search_internal(&temp_board, first_depth, -beta, -alpha, tt, context, start_time, max_time_ms, is_pv_node, ply + 1);
        } else {
            let mut reduction: u8 = 0;
            if depth >= LMR_MIN_DEPTH && moves_searched >= LMR_MIN_MOVES && !is_pv_node
                && !is_capture && !gives_check && !in_check && extension == 0 {

                // Fórmula logarítmica mais sofisticada para LMR
                let log_depth = (depth as f32).ln();
                let log_moves = (moves_searched as f32).ln();
                let base_reduction = (log_depth * log_moves / 2.5) as u8;
                
                reduction = base_reduction.max(1).min(depth.saturating_sub(1));

                // Ajustes baseados em heurísticas
                if let Some(piece) = board.get_piece_on_square(mv.from) {
                    let history = context.get_history_score(*mv, piece);
                    
                    // Reduz menos para movimentos com boa história
                    if history > 1000 {
                        reduction = reduction.saturating_sub(1);
                    } else if history < -1500 {
                        reduction += 1;
                    }
                }
                
                // Reduz menos no nó PV
                if is_pv_node {
                    reduction = reduction.saturating_sub(1);
                }
                
                // Reduz menos se a posição é tática
                if is_tactical_position(board) {
                    reduction = reduction.saturating_sub(1);
                }
                
                // Limitações finais
                reduction = reduction.min(depth.saturating_sub(1)).max(1);
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

/// Verifica se há material não-peão (necessário para null move)
fn has_non_pawn_material(board: &Board) -> bool {
    let our_pieces = if board.to_move == crate::types::Color::White { 
        board.white_pieces 
    } else { 
        board.black_pieces 
    };
    
    (our_pieces & (board.knights | board.bishops | board.rooks | board.queens)) != 0
}

/// Detecção rápida de check sem fazer o movimento
fn gives_check_fast(board: &Board, mv: Move) -> bool {
    // Implementação simplificada - pode ser melhorada
    let mut temp_board = *board;
    temp_board.make_move(mv);
    temp_board.is_king_in_check(!board.to_move)
}

/// Verifica se movimento é killer ou counter-move
fn is_killer_or_counter_move(mv: Move, context: &SearchContext, depth: u8) -> bool {
    context.is_killer(mv, depth) || {
        if let Some(last_move) = context.get_last_move() {
            if let Some(counter) = context.get_counter_move(last_move) {
                counter == mv
            } else {
                false
            }
        } else {
            false
        }
    }
}

/// Threshold para Late Move Count Pruning
fn late_move_count_threshold(depth: u8) -> usize {
    match depth {
        1 => 6,
        2 => 8,
        3 => 12,
        4 => 16,
        5 => 20,
        6 => 24,
        7 => 28,
        _ => 32,
    }
}

/// Detecta posições táticas que precisam de busca mais profunda
fn is_tactical_position(board: &Board) -> bool {
    // Verifica se há peças penduradas ou em xeque
    let in_check = board.is_king_in_check(board.to_move);
    if in_check {
        return true;
    }
    
    // Conta peças atacadas
    let our_pieces = if board.to_move == crate::types::Color::White { 
        board.white_pieces 
    } else { 
        board.black_pieces 
    };
    
    let valuable_pieces = (board.knights | board.bishops | board.rooks | board.queens) & our_pieces;
    let mut attacked_count = 0;
    let mut bb = valuable_pieces;
    
    while bb != 0 && attacked_count < 3 {
        let sq = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        
        if board.is_square_attacked_by(sq, !board.to_move) {
            attacked_count += 1;
        }
    }
    
    attacked_count >= 2
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