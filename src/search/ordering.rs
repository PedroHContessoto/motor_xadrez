use crate::{board::Board, transposition::TranspositionTable, types::{Move, PieceKind}};
use super::SearchContext;
use super::see::see;

pub fn order_moves(board: &Board, moves: Vec<Move>, tt: &TranspositionTable, context: &SearchContext, depth: u8) -> Vec<Move> {
    let tt_move = if let Some(entry) = tt.probe(board.zobrist_hash) {
        entry.best_move.filter(|mv| board.is_legal_move(*mv))
    } else {
        None
    };

    let mut scored_moves = moves.into_iter().map(|mv| {
        // Novo: Inicialize score condicionalmente para evitar warning
        let score = if Some(mv) == tt_move {
            100_000
        } else if mv.is_castling {
            15_000
        } else if board.is_capture(mv) {
            let from_piece = board.get_piece_on_square(mv.from).unwrap_or(PieceKind::Pawn);
            let to_piece = board.get_piece_on_square(mv.to).unwrap_or(PieceKind::Pawn);
            super::PIECE_VALUES[to_piece as usize] * 10 - super::PIECE_VALUES[from_piece as usize] + 10_000 + see(board, mv) * 100
        } else if context.is_killer(mv, depth) {
            9_000
        } else {
            context.get_history_score(mv)
        };
        (mv, score)
    }).collect::<Vec<(Move, i32)>>();

    scored_moves.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    scored_moves.into_iter().map(|(mv, _)| mv).collect()
}