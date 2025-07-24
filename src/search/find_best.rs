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
    let mut total_nodes = 0u64; // Acumula nodes de todas as iterações

    // Clear PV table for new search
    context.clear_pv();

    // Garante que sempre temos um movimento de fallback
    let legal_moves = board.generate_legal_moves();
    if legal_moves.is_empty() {
        return None; // Mate/stalemate
    }
    let mut fallback_move = legal_moves[0];
    let mut fallback_score = evaluation::evaluate(board);

    // Detecta complexidade da posição para ajuste inteligente de tempo
    let tactical_level = evaluate_position_complexity(board);
    
    // DEBUG: Log inicial detalhado
    println!("DEBUG: Starting search - max_time_ms: {}, tactical_level: {}",
             max_time_ms, tactical_level);

    let min_depth = 8;

    for depth in 1..=max_depth { // CORREÇÃO: Remove limite artificial de depth 8
        let iteration_start = std::time::Instant::now();
        let elapsed = start_time.elapsed().as_millis() as u64;

        // Update search context for logging
        context.current_depth = depth;
        context.nodes_searched = 0; // Reset for this depth
        
        // DEBUG: Log início da iteração
        println!("DEBUG: Starting depth {}, total_elapsed: {}ms", depth, elapsed);

        // CORREÇÃO: Time management mais generoso
        let base_timeout = max_time_ms / 20;  // Era /25
        let max_timeout = max_time_ms / 3;    // Era /6

        let final_timeout = base_timeout.min(max_timeout);

        // CORREÇÃO: Permitir pelo menos depth 8 sempre
        if depth < min_depth {
            // Continua até depth mínimo independente do tempo
        } else if depth > 4 && elapsed > final_timeout {
            println!("DEBUG: Time limit reached at depth {}, elapsed: {}ms, limit: {}ms",
                     depth, elapsed, final_timeout);
            break;
        }

        // CORREÇÃO: Aumentar limite absoluto
        let absolute_limit = (max_time_ms * 3) / 4;  // Era /2
        if elapsed > absolute_limit && depth >= min_depth {
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

        // DEBUG: Log tempo da iteração + NPS monitoring
        let iteration_time = iteration_start.elapsed().as_millis();
        let total_elapsed = start_time.elapsed().as_millis();
        
        // Acumula nodes de todas as iterações
        total_nodes += context.nodes_searched as u64;
        
        // CRÍTICO: NPS monitoring (iteração atual)
        let iteration_nps = if iteration_time > 0 {
            (context.nodes_searched as u64 * 1000) / iteration_time as u64
        } else {
            0
        };
        
        // NPS total acumulativo
        let total_nps = if total_elapsed > 0 {
            (total_nodes * 1000) / total_elapsed as u64
        } else {
            0
        };
        
        println!("DEBUG: Depth {} completed in {}ms, total: {}ms, nodes: {}, iter_NPS: {}, total_NPS: {}", 
                 depth, iteration_time, total_elapsed, context.nodes_searched, iteration_nps, total_nps);
        
        // WARNING se NPS muito baixo
        if iteration_nps < 50_000 && depth > 2 {
            println!("WARNING: Low NPS detected: {} (target: 200k+)", iteration_nps);
        }

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

                // CORREÇÃO: UCI info com valores acumulativos corretos
                let total_elapsed_ms = start_time.elapsed().as_millis() as u64;
                let uci_nps = if total_elapsed_ms > 0 { 
                    (total_nodes * 1000) / total_elapsed_ms 
                } else { 
                    0 
                };

                let display_score = score.clamp(-10000, 10000);

                // Format Principal Variation  
                let pv_string = if context.pv_length[0] > 0 {
                    context.format_pv(0)
                } else {
                    format!("{}", mv)
                };

                // UCI info completa para Arena com valores corretos
                let hashfull = ((total_nodes & 1023) * 1000 / 1024).min(1000); // Hash usage aproximado
                
                println!("info depth {} seldepth {} score cp {} nodes {} nps {} hashfull {} time {} pv {}",
                         depth, 
                         depth, // seldepth = selective depth (aproximação)
                         display_score, 
                         total_nodes, // Total acumulativo de nodes
                         uci_nps,     // NPS acumulativo
                         hashfull,    // Hash table usage aproximado
                         total_elapsed_ms, 
                         pv_string);
                
                // Log adicional de progresso para depths maiores
                if depth >= 4 {
                    println!("info string depth {} time {}ms total_nodes {} nps {} current_nodes {}", 
                             depth, total_elapsed_ms, total_nodes, uci_nps, context.nodes_searched);
                }

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


/// Avalia complexidade da posição em níveis (0-3) - VERSÃO RÁPIDA
fn evaluate_position_complexity(board: &Board) -> u8 {
    let our_color = board.to_move;
    let mut complexity_score = 0u8;

    // 1. Xeque = complexidade alta
    if board.is_king_in_check(our_color) {
        return 3; // Máxima complexidade em xeque
    }

    // 2. Contagem rápida de material
    let total_pieces = (board.white_pieces | board.black_pieces).count_ones();

    if total_pieces <= 8 {
        complexity_score += 1; // Endgame
    }

    // 3. Simplificado - não fazer análise profunda aqui
    complexity_score.min(2) // Limita a 2 para não atrasar
}