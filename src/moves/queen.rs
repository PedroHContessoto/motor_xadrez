// Ficheiro: src/moves/queen.rs
// Descrição: Lógica para gerar os lances da Dama.

use crate::{board::Board, types::{Move, Color}};

/// Gera todos os lances pseudo-legais para a dama do jogador atual.
pub fn generate_queen_moves(board: &Board) -> Vec<Move> {
    let mut moves = Vec::with_capacity(32); // Pre-aloca para reduzir realocações
    let our_pieces = if board.to_move == Color::White { board.white_pieces } else { board.black_pieces };
    let mut our_queens = board.queens & our_pieces;

    // Direções da dama: 8 direções (rook + bishop)
    let queen_directions = [8, 1, -8, -1, 9, 7, -9, -7];

    while our_queens != 0 {
        let from_sq = our_queens.trailing_zeros() as u8;
        for &direction in &queen_directions {
            // Reutiliza a lógica de raios do módulo de peças deslizantes.
            super::sliding::generate_ray_moves(&mut moves, board, from_sq, direction);
        }
        our_queens &= our_queens - 1;
    }

    moves
}
