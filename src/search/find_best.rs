use std::time::Instant;
use std::io::{self, Write};
use crate::board::Board;
use crate::evaluation;
use crate::transposition::TranspositionTable;
use super::{SearchContext, aspiration_search, pvs_search};

pub fn find_best_move(board: &Board, max_depth: u8, tt: &mut TranspositionTable) -> Option<(crate::types::Move, i32)> { // Use full path para Move
    find_best_move_with_time(board, max_depth, 5000, tt)
}

pub fn find_best_move_with_time(board: &Board, max_depth: u8, mut max_time_ms: u64, tt: &mut TranspositionTable) -> Option<(crate::types::Move, i32)> {
    let start_time = Instant::now();
    let mut context = SearchContext::new();
    let mut best_move = None;
    let mut best_score = 0;
    let mut prev_score = 0;
    let mut stable_count = 0;

    // Clear PV table for new search
    context.clear_pv();

    // Garante que sempre temos um movimento de fallback
    let legal_moves = board.generate_legal_moves();
    if legal_moves.is_empty() {
        return None; // Mate/stalemate
    }
    let mut fallback_move = legal_moves[0];
    let mut fallback_score = evaluation::evaluate(board);

    // Detecta se é posição tática para ajuste de tempo
    let is_tactical_position = detect_tactical_position(board);
    let tactical_time_multiplier = if is_tactical_position { 1.5 } else { 1.0 };
    let effective_time_limit = (max_time_ms as f32 * tactical_time_multiplier) as u64;

    for depth in 1..=max_depth { // CORREÇÃO: Remove limite artificial de depth 8
        let elapsed = start_time.elapsed().as_millis() as u64;

        // Update search context for logging
        context.current_depth = depth;
        context.nodes_searched = 0; // Reset for this depth

        // Time management adaptado para posições táticas
        if depth > 12 && elapsed > effective_time_limit {
            // Em posições táticas, permite mais tempo até depth 18
            if !is_tactical_position || depth > 18 {
                break;
            }
        }

        // Timeout absoluto mais generoso para posições táticas
        let absolute_limit = if is_tactical_position {
            max_time_ms * 4 // 4x mais tempo em posições táticas críticas
        } else {
            max_time_ms * 3
        };

        if elapsed > absolute_limit {
            break;
        }

        // Reset stop flag para cada profundidade
        context.should_stop = false;

        // Removed search start info to keep logs clean

        let score = if depth > 2 {
            aspiration_search(board, depth, prev_score, tt, &mut context, start_time, max_time_ms)
        } else {
            pvs_search(board, depth, -50000, 50000, tt, &mut context, start_time, max_time_ms, true)
        };

        // Se parou por timeout, usa o que temos
        if context.should_stop {
            break;
        }

        if let Some(entry) = tt.probe(board.zobrist_hash) {
            if let Some(mv) = entry.best_move {
                if board.is_legal_move(mv) {
                    if Some(mv) == context.prev_best_move {
                        stable_count += 1;
                        if stable_count >= 3 {
                            max_time_ms = (max_time_ms * 3) / 4; // Redução menor
                        }
                    } else {
                        stable_count = 0;
                    }
                    context.prev_best_move = Some(mv);

                    best_move = Some(mv);
                    best_score = score;
                    fallback_move = mv; // Atualiza fallback
                    fallback_score = score;
                } else {
                    // Move da TT não é legal, mas ainda podemos usar o score
                    if best_move.is_none() {
                        best_move = Some(fallback_move);
                        best_score = score;
                    }
                    // Não imprime info se move não é legal
                    continue;
                }

                // Só imprime se temos um move legal válido
                let nps = if elapsed > 0 { context.nodes_searched * 1000 / elapsed } else { 0 }; // Limpado parens
                let time_ms = elapsed;

                let display_score = score.clamp(-10000, 10000);

                // Format Principal Variation
                let pv_string = if context.pv_length[0] > 0 {
                    context.format_pv(0)
                } else {
                    format!("{}", mv)
                };

                // Show thinking line
                println!("info depth {} score cp {} nodes {} nps {} time {} pv {}",
                         depth, display_score, context.nodes_searched, nps, time_ms, pv_string);

                // Removed extra logging to keep output clean

                io::stdout().flush().ok();
            }
        }

        prev_score = score;
    }

    // Garante que sempre retornamos um movimento válido
    if best_move.is_none() {
        best_move = Some(fallback_move);
        best_score = fallback_score;
    }

    best_move.map(|mv| (mv, best_score))
}


/// Detecta se a posição atual é tática (precisa de mais tempo/profundidade)
fn detect_tactical_position(board: &Board) -> bool {
    let our_color = board.to_move;
    let enemy_color = !our_color;

    // 1. Estamos em xeque?
    if board.is_king_in_check(our_color) {
        return true;
    }

    // 2. Há peças penduradas (atacadas sem defesa)?
    let our_pieces = if our_color == crate::types::Color::White {
        board.white_pieces
    } else {
        board.black_pieces
    };

    let our_valuables = (board.knights | board.bishops | board.rooks | board.queens) & our_pieces;
    let mut hanging_count = 0;
    let mut attacked_count = 0;

    let mut bb = our_valuables;
    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        bb &= bb - 1;

        if board.is_square_attacked_by(sq, enemy_color) {
            attacked_count += 1;
            if !board.is_square_attacked_by(sq, our_color) {
                hanging_count += 1;
            }
        }
    }

    // 3. Muitas peças atacadas indica complexidade tática
    if hanging_count > 0 || attacked_count > 2 {
        return true;
    }

    // 4. Verifica se o inimigo também tem peças penduradas (oportunidades táticas)
    let enemy_pieces = if enemy_color == crate::types::Color::White {
        board.white_pieces
    } else {
        board.black_pieces
    };

    let enemy_valuables = (board.knights | board.bishops | board.rooks | board.queens) & enemy_pieces;
    let mut enemy_hanging = 0;

    let mut enemy_bb = enemy_valuables;
    while enemy_bb != 0 {
        let sq = enemy_bb.trailing_zeros() as u8;
        enemy_bb &= enemy_bb - 1;

        if board.is_square_attacked_by(sq, our_color) &&
            !board.is_square_attacked_by(sq, enemy_color) {
            enemy_hanging += 1;
        }
    }

    if enemy_hanging > 0 {
        return true;
    }

    // 5. Posições de final com poucos peões são complexas
    let total_pieces = (board.white_pieces | board.black_pieces).count_ones();
    let total_pawns = board.pawns.count_ones();

    if total_pieces <= 10 && total_pawns <= 4 {
        return true; // Finais técnicos complexos
    }

    false
}