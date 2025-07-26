// Consts movidas para aqui (pub) para visibilidade global no módulo search
pub const PIECE_VALUES: [i32; 6] = [100, 320, 330, 500, 900, 20000];
pub const MATE_VALUE: i32 = 99999;
pub const LMR_MIN_DEPTH: u8 = 3;
pub const LMR_MIN_MOVES: usize = 4;
pub const LMR_REDUCTION: u8 = 2;
pub const FUTILITY_MARGIN: [i32; 4] = [0, 300, 500, 800];

pub use context::SearchContext;
pub use ordering::{order_moves, update_context_on_cutoff};
pub use see::see;
pub use find_best::{find_best_move, find_best_move_with_time};
pub use aspiration::aspiration_search;
pub use pvs::pvs_search;
pub use quiescence::quiescence_search;

mod context;
mod ordering;
mod see;
mod find_best;
mod aspiration;
mod pvs;
mod quiescence;
mod mate_detection;

use crate::{board::Board, types::{Color, PieceKind}};

/// Detecta se a posição atual é tática (precisa de mais análise)
pub fn is_tactical_position(board: &Board) -> bool {
    // 1. Xeque
    if board.is_king_in_check(board.to_move) {
        return true;
    }

    // 2. Capturas de peças valiosas disponíveis
    let moves = board.generate_legal_moves();
    for mv in &moves {
        if board.is_capture(*mv) {
            if let Some(captured) = board.get_piece_on_square(mv.to) {
                if PIECE_VALUES[captured as usize] >= 300 { // Peça valiosa
                    return true;
                }
            }
        }
    }

    // 3. Peças penduradas
    let our_color = board.to_move;
    let our_pieces = if our_color == Color::White { board.white_pieces } else { board.black_pieces };
    let valuable = (board.queens | board.rooks | board.bishops | board.knights) & our_pieces;

    let mut bb = valuable;
    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        bb &= bb - 1;

        if board.is_square_attacked_by(sq, !our_color) &&
            !board.is_square_attacked_by(sq, our_color) {
            return true; // Peça pendurada
        }
    }

    // 4. Rei exposto (fora da primeira/última fileira)
    let king_bb = board.kings & our_pieces;
    if king_bb != 0 {
        let king_sq = king_bb.trailing_zeros() as u8;
        let king_rank = king_sq / 8;

        match our_color {
            Color::White => {
                if king_rank > 1 { return true; } // Rei branco exposto
            },
            Color::Black => {
                if king_rank < 6 { return true; } // Rei preto exposto
            }
        }
    }

    false
}