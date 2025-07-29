// Ficheiro: src/moves/queen.rs
// Descrição: Lógica otimizada para gerar lances da Dama com funcionalidades avançadas

use crate::{board::Board, types::{Move, Color, Bitboard}};
use super::magic_bitboards::{get_queen_attacks_magic, get_rook_attacks_magic, get_bishop_attacks_magic};

/// Gera todos os lances pseudo-legais para a dama do jogador atual usando magic bitboards.
pub fn generate_queen_moves(board: &Board) -> Vec<Move> {
    let mut moves = Vec::with_capacity(32); // Pre-aloca para reduzir realocações
    let our_pieces = if board.to_move == Color::White { board.white_pieces } else { board.black_pieces };
    let all_pieces = board.white_pieces | board.black_pieces;
    let mut our_queens = board.queens & our_pieces;

    while our_queens != 0 {
        let from_sq = our_queens.trailing_zeros() as u8;
        
        // Usa magic bitboards para calcular ataques de rainha (ultra rápido)
        let attacks = get_queen_attacks_magic(from_sq, all_pieces);
        
        // Filtra movimentos válidos (exclui nossas próprias peças)
        let mut valid_moves = attacks & !our_pieces;
        
        while valid_moves != 0 {
            let to_sq = valid_moves.trailing_zeros() as u8;
            moves.push(Move { 
                from: from_sq, 
                to: to_sq, 
                promotion: None, 
                is_castling: false, 
                is_en_passant: false 
            });
            valid_moves &= valid_moves - 1;
        }
        
        our_queens &= our_queens - 1;
    }

    moves
}
