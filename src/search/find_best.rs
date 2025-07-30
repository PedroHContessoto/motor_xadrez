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

    // Enhanced position complexity evaluation for better time management
    let tactical_level = evaluate_position_complexity(board);
    let tactical_time_multiplier = match tactical_level {
        3 => 1.8,   // Very tactical positions need more time
        2 => 1.4,   // Moderately tactical positions
        1 => 1.15,  // Slightly tactical positions
        _ => 1.0    // Normal positions
    };
    
    // Dynamic time adjustment based on game phase
    let game_phase_multiplier = evaluate_game_phase_time_factor(board);
    let combined_multiplier = tactical_time_multiplier * game_phase_multiplier;
    let _effective_time_limit = (max_time_ms as f32 * combined_multiplier) as u64;

    // DEBUG: Log inicial detalhado
    println!("DEBUG: Starting search - max_time_ms: {}, tactical_level: {}, multiplier: {:.2}",
             max_time_ms, tactical_level, tactical_time_multiplier);

    for depth in 1..=max_depth { // CORREÇÃO: Remove limite artificial de depth 8
        let iteration_start = std::time::Instant::now();
        let elapsed = start_time.elapsed().as_millis() as u64;

        // Update search context for logging
        context.current_depth = depth;
        context.nodes_searched = 0; // Reset for this depth

        // DEBUG: Log início da iteração
        println!("DEBUG: Starting depth {}, total_elapsed: {}ms", depth, elapsed);

        // Enhanced time management with adaptive allocation
        let base_timeout = max_time_ms / 20;  // More generous base allocation
        let max_timeout = max_time_ms / 5;    // Higher maximum per iteration
        
        // Apply combined multipliers for more intelligent time allocation
        let adjusted_timeout = (base_timeout as f32 * combined_multiplier) as u64;
        let final_timeout = adjusted_timeout.min(max_timeout);
        
        // Adaptive timeout based on search instability
        let stability_factor = if stable_count >= 2 { 0.8 } else { 1.2 };
        let dynamic_timeout = (final_timeout as f32 * stability_factor) as u64;

        // Depth limit aumentado para melhor jogo
        let max_safe_depth = match tactical_level {
            3 => 12,  // Posições táticas complexas
            2 => 10,  // Posições moderadamente táticas
            1 => 9,   // Posições simples
            _ => 8    // Posições normais
        };

        if depth > max_safe_depth {
            println!("DEBUG: Depth limit reached at {}", depth);
            break;
        }

        if depth > 8 && elapsed > dynamic_timeout {
            println!("DEBUG: Time limit reached at depth {}, elapsed: {}ms, limit: {}ms",
                     depth, elapsed, dynamic_timeout);
            break;
        }

        // Hard timeout absoluto
        let absolute_limit = (max_time_ms * 3) / 4;  // Menos conservador: 3/4 do tempo
        if elapsed > absolute_limit {
            println!("DEBUG: HARD TIMEOUT at {}ms (limit: {}ms)", elapsed, absolute_limit);
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

        // DEBUG: Log tempo da iteração
        let iteration_time = iteration_start.elapsed().as_millis();
        let total_elapsed = start_time.elapsed().as_millis();
        println!("DEBUG: Depth {} completed in {}ms, total: {}ms, nodes: {}",
                 depth, iteration_time, total_elapsed, context.nodes_searched);

        // Se parou por timeout, usa o que temos
        if context.should_stop {
            println!("DEBUG: Search stopped by timeout flag at depth {}", depth);
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


/// Avalia complexidade da posição em níveis (0-3)
fn evaluate_position_complexity(board: &Board) -> u8 {
    let our_color = board.to_move;
    let enemy_color = !our_color;
    let mut complexity_score = 0u8;

    // 1. Fatores de complexidade imediata (+2 pontos cada)
    if board.is_king_in_check(our_color) {
        complexity_score += 2; // Xeque = alta complexidade
    }

    // 2. Análise de peças atacadas/penduradas
    let our_pieces = if our_color == crate::types::Color::White {
        board.white_pieces
    } else {
        board.black_pieces
    };
    let enemy_pieces = if enemy_color == crate::types::Color::White {
        board.white_pieces
    } else {
        board.black_pieces
    };

    let our_valuables = (board.knights | board.bishops | board.rooks | board.queens) & our_pieces;
    let enemy_valuables = (board.knights | board.bishops | board.rooks | board.queens) & enemy_pieces;

    let mut hanging_count = 0;
    let mut attacked_count = 0;
    let mut enemy_hanging = 0;

    // Analisa nossas peças
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

    // Analisa peças inimigas
    let mut enemy_bb = enemy_valuables;
    while enemy_bb != 0 {
        let sq = enemy_bb.trailing_zeros() as u8;
        enemy_bb &= enemy_bb - 1;

        if board.is_square_attacked_by(sq, our_color) &&
            !board.is_square_attacked_by(sq, enemy_color) {
            enemy_hanging += 1;
        }
    }

    // 3. Pontuação baseada em ameaças
    if hanging_count > 0 {
        complexity_score += 2; // Temos peças penduradas = muito complexo
    }
    if enemy_hanging > 0 {
        complexity_score += 1; // Podemos ganhar material = complexo
    }
    if attacked_count > 2 {
        complexity_score += 1; // Muitas peças sob ataque = complexo
    }

    // 4. Análise da fase do jogo
    let total_pieces = (board.white_pieces | board.black_pieces).count_ones();
    let total_pawns = board.pawns.count_ones();
    let queens_on_board = board.queens.count_ones();

    // Finais técnicos complexos
    if total_pieces <= 8 && total_pawns <= 3 {
        complexity_score += 1;
    }

    // Meio-jogo com muitas peças = potencial tático
    if total_pieces > 20 && queens_on_board >= 2 {
        complexity_score += 1;
    }

    // 5. Limita o score máximo a 3
    complexity_score.min(3)
}

/// Evaluates game phase factor for time allocation
fn evaluate_game_phase_time_factor(board: &Board) -> f32 {
    let total_pieces = (board.white_pieces | board.black_pieces).count_ones();
    let total_pawns = board.pawns.count_ones();
    let queens_on_board = board.queens.count_ones();
    let total_material = board.knights.count_ones() * 3 + 
                        board.bishops.count_ones() * 3 + 
                        board.rooks.count_ones() * 5 + 
                        board.queens.count_ones() * 9;
    
    match total_pieces {
        // Opening: Many pieces, need more time for complex calculations
        26..=32 => {
            if queens_on_board >= 2 && total_material >= 60 {
                1.3 // Complex opening positions
            } else {
                1.1 // Normal opening
            }
        },
        // Middlegame: Peak complexity, most time needed
        16..=25 => {
            if queens_on_board >= 1 && total_material >= 40 {
                1.4 // Complex middlegame
            } else {
                1.2 // Normal middlegame
            }
        },
        // Endgame: Technical positions may need precision
        8..=15 => {
            if total_pawns <= 4 && queens_on_board == 0 {
                1.2 // Technical endgame (need precision)
            } else if total_pawns >= 6 {
                1.0 // Pawn endgame (usually straightforward)
            } else {
                1.1 // Normal endgame
            }
        },
        // Late endgame: Usually faster but some positions are tricky
        _ => {
            if total_pawns >= 2 {
                1.1 // Pawn races can be complex
            } else {
                0.9 // Simple material endgames
            }
        }
    }
}