use crate::{board::Board, types::{Color, PieceKind, Bitboard, Move}}; // Adicionado Move
use super::PIECE_VALUES; // Agora de mod.rs

pub fn see(board: &Board, mv: Move) -> i32 {
    let target = mv.to;
    let mut side = board.to_move;
    let mut gain = [0i32; 32];
    let mut d = 0;
    let mut occ = board.white_pieces | board.black_pieces;
    let mut attackers = get_attackers(board, target, Color::White) | get_attackers(board, target, Color::Black);

    let mut value = PIECE_VALUES[board.get_piece_on_square(target).unwrap_or(PieceKind::Pawn) as usize];
    gain[d] = value;

    while let Some(att_sq) = get_least_valuable_attacker(board, attackers, side, occ) {
        d += 1;
        value = PIECE_VALUES[board.get_piece_on_square(att_sq).unwrap() as usize] - gain[d - 1];
        gain[d] = value.max(0);

        occ ^= 1u64 << att_sq;
        attackers = get_attackers(board, target, Color::White) | get_attackers(board, target, Color::Black);
        side = !side;
    }

    while d > 0 {
        gain[d - 1] = -gain[d].min(-gain[d - 1]);
        d -= 1;
    }

    gain[0]
}

fn get_attackers(board: &Board, target: u8, color: Color) -> Bitboard {
    let mut attackers = 0u64;

    attackers |= board.is_square_attacked_by(target, color) as u64 * (1u64 << target);
    attackers
}

fn get_least_valuable_attacker(board: &Board, attackers: Bitboard, side: Color, _occ: Bitboard) -> Option<u8> {  // Prefix _occ para warning
    let mut min_val = i32::MAX;
    let mut min_sq = None;
    let mut bb = attackers & if side == Color::White { board.white_pieces } else { board.black_pieces };

    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        let kind = board.get_piece_on_square(sq).unwrap();
        let val = PIECE_VALUES[kind as usize];
        if val < min_val {
            min_val = val;
            min_sq = Some(sq);
        }
    }
    min_sq
}