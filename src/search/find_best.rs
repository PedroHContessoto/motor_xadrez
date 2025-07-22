use std::time::Instant;
use std::io::{self, Write};
use crate::board::Board;
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

    for depth in 1..=max_depth {
        let elapsed = start_time.elapsed().as_millis() as u64;

        if depth > 3 && elapsed * 3 > max_time_ms {
            break;
        }
        if elapsed > max_time_ms {
            break;
        }

        let score = if depth > 2 {
            aspiration_search(board, depth, prev_score, tt, &mut context, start_time, max_time_ms)
        } else {
            pvs_search(board, depth, -50000, 50000, tt, &mut context, start_time, max_time_ms, true)
        };

        if let Some(entry) = tt.probe(board.zobrist_hash) {
            if let Some(mv) = entry.best_move {
                if board.is_legal_move(mv) {
                    if Some(mv) == context.prev_best_move {
                        stable_count += 1;
                        if stable_count >= 2 {
                            max_time_ms /= 2;
                        }
                    } else {
                        stable_count = 0;
                    }
                    context.prev_best_move = Some(mv);

                    best_move = Some(mv);
                    best_score = score;
                } else {
                    let legal_moves = board.generate_legal_moves();
                    if let Some(fallback_mv) = legal_moves.first() {
                        best_move = Some(*fallback_mv);
                        best_score = score;
                    }
                }
                let nps = if elapsed > 0 { context.nodes_searched * 1000 / elapsed } else { 0 }; // Limpado parens
                let time_ms = elapsed;

                let display_score = score.clamp(-10000, 10000);
                println!("info depth {} score cp {} nodes {} nps {} time {} pv {}",
                         depth, display_score, context.nodes_searched, nps, time_ms, mv);
                io::stdout().flush().ok();
            }
        }

        prev_score = score;
    }

    best_move.map(|mv| (mv, best_score))
}