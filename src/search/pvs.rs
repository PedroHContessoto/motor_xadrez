// Principal Variation Search aprimorado para táticas
use std::time::Instant;
use crate::{board::Board, evaluation, transposition::{TranspositionTable, EntryType}, types::Move};
use super::{SearchContext, quiescence::quiescence_search, ordering::order_moves, see::see_threshold};

const MATE_VALUE: i32 = 99999;
const FUTILITY_MARGIN: [i32; 4] = [0, 200, 350, 600]; // Ainda menos agressivo - reduzido
const LMR_MIN_DEPTH: u8 = 3;
const LMR_MIN_MOVES: usize = 4;
const LMR_REDUCTION: u8 = 1;

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
    context.nodes_searched += 1;


    // Proteção contra stack overflow - limite absoluto de profundidade
    if depth > 8 {
        // Depth limit reached
        return evaluation::evaluate(board);
    }

    // Verificação de tempo - seta flag ao invés de retornar imediatamente
    if context.nodes_searched % 1024 == 0 {
        if start_time.elapsed().as_millis() as u64 > max_time_ms {
            context.should_stop = true;
        }
    }
    
    // Se should_stop foi setado, termina gracefully
    if context.should_stop {
        return evaluation::evaluate(board);
    }

    let original_alpha = alpha;

    // Transposition Table lookup
    if let Some(entry) = tt.probe(board.zobrist_hash) {
        if entry.depth >= depth {
            match entry.entry_type {
                EntryType::Exact => {
                    return entry.score;
                }
                EntryType::LowerBound => alpha = alpha.max(entry.score),
                EntryType::UpperBound => beta = beta.min(entry.score),
            }
            if alpha >= beta {
                return entry.score;
            }
        }
    }

    // Draw detection
    if board.is_draw_by_50_moves() || board.is_draw_by_insufficient_material() {
        return 0;
    }

    // Null Move Pruning melhorado (com verificação de zugzwang)
    if depth >= 3 && !is_pv_node && !board.is_king_in_check(board.to_move) {
        let static_eval = evaluation::evaluate(board);
        
        // Não faz null move se posição é muito tática
        if static_eval.abs() < 8000 {
            let mut null_board = *board;
            null_board.to_move = !null_board.to_move;
            null_board.en_passant_target = None;
            
            let null_depth = if depth > 6 { depth - 4 } else { depth - 3 }; // Adaptive reduction
            let temp_null_score = pvs_search(&null_board, null_depth, -beta, -beta + 1, tt, context, start_time, max_time_ms, false);
            let null_score = -temp_null_score;
            
            if null_score >= beta {
                // Verification search para evitar zugzwang
                if depth < 6 || null_score >= MATE_VALUE - 100 {
                    return beta;
                } else {
                    let verify_score = pvs_search(board, depth - 4, beta - 1, beta, tt, context, start_time, max_time_ms, false);
                    if verify_score >= beta {
                        return beta;
                    }
                }
            }
        }
    }

    // Quiescence search no leaf nodes
    if depth == 0 {
        return quiescence_search(board, alpha, beta, tt, context);
    }

    // Move generation
    let legal_moves = board.generate_legal_moves();
    
    if legal_moves.is_empty() {
        if board.is_king_in_check(board.to_move) {
            return -(MATE_VALUE - depth as i32); // Checkmate
        } else {
            return 0; // Stalemate
        }
    }

    let in_check = board.is_king_in_check(board.to_move);
    let static_eval = if !in_check { evaluation::evaluate(board) } else { -MATE_VALUE / 2 };
    
    let ordered_moves = order_moves(board, legal_moves, tt, context, depth);
    let mut best_move = None;
    let mut best_score = -50000;
    let mut moves_searched = 0;


    for mv in &ordered_moves {
        let mut temp_board = *board;
        temp_board.make_move(*mv);
        
        let gives_check = temp_board.is_king_in_check(!board.to_move);
        let is_capture = board.is_capture(*mv);
        
        // Futility Pruning aprimorado - desabilitado para táticas e peças penduradas
        if depth <= 3 && !is_pv_node && !is_capture && !in_check && !gives_check && moves_searched > 0 {
            let futility_margin = FUTILITY_MARGIN[depth as usize];
            if static_eval + futility_margin <= alpha && !has_hanging_pieces_simple(board) {
                continue;
            }
        }
        
        // SEE Pruning - poda capturas muito perdedoras em nós não-PV
        if !is_pv_node && is_capture && depth <= 4 && moves_searched > 0 {
            if !see_threshold(board, *mv, -200) { // Muito perdedor
                continue;
            }
        }

        // Extensions
        let mut extension = 0;
        
        if gives_check {
            extension += 1; // Check extension
        }
        
        if mv.promotion.is_some() {
            extension += 1; // Promotion extension
        }
        
        if is_recapture(board, *mv, context) {
            extension += 1; // Recapture extension
        }
        
        // Limita extensões para evitar explosion
        extension = extension.min(2);

        let mut score;

        if moves_searched == 0 {
            // Primeira jogada: busca completa
            let first_depth = if depth > 1 { depth - 1 + extension } else { extension };
            let first_score = pvs_search(&temp_board, first_depth, -beta, -alpha, tt, context, start_time, max_time_ms, is_pv_node);
            score = -first_score;
        } else {
            // Late Move Reduction aprimorado
            let mut reduction = 0;
            
            if depth >= LMR_MIN_DEPTH && moves_searched >= LMR_MIN_MOVES && !is_pv_node 
                && !is_capture && !gives_check && !in_check && extension == 0 {
                
                reduction = LMR_REDUCTION;
                
                // Reduce mais para moves com história baixa
                if context.get_history_score(*mv) < 0 {
                    reduction += 1;
                }
                
                // Reduce menos próximo de mate
                if static_eval.abs() > MATE_VALUE / 4 {
                    reduction = reduction.saturating_sub(1);
                }
            }

            // PVS null window search
            let search_depth = if depth > 1 + reduction { depth - 1 - reduction + extension } else { 0 };
            let temp_score = pvs_search(&temp_board, search_depth, -alpha - 1, -alpha, tt, context, start_time, max_time_ms, false);
            score = -temp_score;

            // Re-search com janela completa se necessário
            if score > alpha && (is_pv_node || reduction > 0) {
                let full_depth = if depth > 1 { depth - 1 + extension } else { extension };
                let full_score = pvs_search(&temp_board, full_depth, -beta, -alpha, tt, context, start_time, max_time_ms, is_pv_node);
                score = -full_score;
            }
        }

        moves_searched += 1;

        // Update best score
        if score > best_score {
            best_score = score;
            best_move = Some(*mv);
            
            if !is_capture {
                context.update_history(*mv, depth);
            }
        }

        alpha = alpha.max(score);
        
        // Beta cutoff
        if alpha >= beta {
            if !is_capture {
                context.add_killer(*mv, depth);
            }
            break;
        }
    }

    // Store na Transposition Table
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
    // Verifica se captura na mesma casa do último movimento inimigo
    if let Some(last_move) = context.get_last_move() {
        board.is_capture(mv) && mv.to == last_move.to
    } else {
        false
    }
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

/// Função auxiliar para detectar peças penduradas (versão simplificada)
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