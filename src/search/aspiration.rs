use std::time::Instant;
use crate::board::Board;
use crate::transposition::TranspositionTable;
use super::{SearchContext, pvs_search};

pub fn aspiration_search(board: &Board, depth: u8, prev_score: i32, tt: &mut TranspositionTable, context: &mut SearchContext, start_time: Instant, max_time_ms: u64) -> i32 {
    let mut window = 25;
    let mut alpha = prev_score - window;
    let mut beta = prev_score + window;

    loop {
        let score = pvs_search(board, depth, alpha, beta, tt, context, start_time, max_time_ms, true);

        if score <= alpha {
            alpha = -50000;
            window *= 2;
        } else if score >= beta {
            beta = 50000;
            window *= 2;
        } else {
            return score;
        }

        if window > 1000 {
            return pvs_search(board, depth, -50000, 50000, tt, context, start_time, max_time_ms, true);
        }
    }
}