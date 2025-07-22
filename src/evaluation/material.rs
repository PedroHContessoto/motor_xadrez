// Avaliação de material e tabelas de posição
use crate::{board::Board, types::{Color, PieceKind, Bitboard}};
use super::game_phase::GamePhase;

pub const MATERIAL_VALUES: [i32; 6] = [100, 320, 330, 500, 900, 20000];

// Tabelas de posição (PST) - copiadas do código original
const PAWN_TABLE: [i32; 64] = [0,0,0,0,0,0,0,0,50,50,50,50,50,50,50,50,10,10,20,30,30,20,10,10,5,5,10,25,25,10,5,5,0,0,0,20,20,0,0,0,5,-5,-10,0,0,-10,-5,5,5,10,10,-20,-20,10,10,5,0,0,0,0,0,0,0,0];
const KNIGHT_TABLE: [i32; 64] = [-50,-40,-30,-30,-30,-30,-40,-50,-40,-20,0,0,0,0,-20,-40,-30,0,10,15,15,10,0,-30,-30,5,15,20,20,15,5,-30,-30,0,15,20,20,15,0,-30,-30,5,10,15,15,10,5,-30,-40,-20,0,5,5,0,-20,-40,-50,-40,-30,-30,-30,-30,-40,-50];
const BISHOP_TABLE: [i32; 64] = [-20,-10,-10,-10,-10,-10,-10,-20,-10,0,0,0,0,0,0,-10,-10,0,5,10,10,5,0,-10,-10,5,5,10,10,5,5,-10,-10,0,10,10,10,10,0,-10,-10,10,10,10,10,10,10,-10,-10,5,0,0,0,0,5,-10,-20,-10,-10,-10,-10,-10,-10,-20];
const ROOK_TABLE: [i32; 64] = [0,0,0,0,0,0,0,0,5,10,10,10,10,10,10,5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,0,0,0,5,5,0,0,0];
const QUEEN_TABLE: [i32; 64] = [-20,-10,-10,-5,-5,-10,-10,-20,-10,0,0,0,0,0,0,-10,-10,0,5,5,5,5,0,-10,-5,0,5,5,5,5,0,-5,0,0,5,5,5,5,0,-5,-10,5,5,5,5,5,0,-10,-10,0,5,0,0,0,0,-10,-20,-10,-10,-5,-5,-10,-10,-20];
const KING_TABLE_MIDGAME: [i32; 64] = [-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-20,-30,-30,-40,-40,-30,-30,-20,-10,-20,-20,-20,-20,-20,-20,-10,20,20,0,0,0,0,20,20,20,30,10,0,0,10,30,20];
const KING_TABLE_ENDGAME: [i32; 64] = [-50,-40,-30,-20,-20,-30,-40,-50,-30,-20,-10,0,0,-10,-20,-30,-30,-10,20,30,30,20,-10,-30,-30,-10,30,40,40,30,-10,-30,-30,-10,30,40,40,30,-10,-30,-30,-10,20,30,30,20,-10,-30,-30,-30,0,0,0,0,-30,-30,-50,-30,-30,-30,-30,-30,-30,-50];

// Constantes para centro
const CENTRAL_SQUARES: Bitboard = (1u64 << 27) | (1u64 << 28) | (1u64 << 35) | (1u64 << 36); // d4, e4, d5, e5
const EXTENDED_CENTER: Bitboard = 0x00003C3C3C3C0000; // c3-f3 to c6-f6

pub fn evaluate_material_and_pst(board: &Board, color: Color, game_phase: &GamePhase) -> i32 {
    let mut score = 0;
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    
    // Escolhe tabela do rei baseada na fase
    let king_table = match game_phase {
        GamePhase::Endgame => &KING_TABLE_ENDGAME,
        _ => &KING_TABLE_MIDGAME,
    };

    // Avalia cada tipo de peça
    score += evaluate_piece_type(board.pawns & pieces, &PAWN_TABLE, PieceKind::Pawn, color);
    score += evaluate_piece_type(board.knights & pieces, &KNIGHT_TABLE, PieceKind::Knight, color);
    score += evaluate_piece_type(board.bishops & pieces, &BISHOP_TABLE, PieceKind::Bishop, color);
    score += evaluate_piece_type(board.rooks & pieces, &ROOK_TABLE, PieceKind::Rook, color);
    score += evaluate_piece_type(board.queens & pieces, &QUEEN_TABLE, PieceKind::Queen, color);
    score += evaluate_piece_type(board.kings & pieces, king_table, PieceKind::King, color);
    
    // Bônus por controle do centro (ajustado por fase)
    score += evaluate_center_control(board, color, game_phase);
    
    score
}

fn evaluate_piece_type(mut piece_bb: Bitboard, pst: &'static [i32; 64], kind: PieceKind, color: Color) -> i32 {
    let mut score = 0;
    while piece_bb != 0 {
        let sq = piece_bb.trailing_zeros() as usize;
        piece_bb &= piece_bb - 1;
        let positional_score = if color == Color::White { pst[sq] } else { pst[sq ^ 56] };
        score += MATERIAL_VALUES[kind as usize] + positional_score;
    }
    score
}

fn evaluate_center_control(board: &Board, color: Color, game_phase: &GamePhase) -> i32 {
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let mut score = 0;
    
    // Bônus ajustado por fase do jogo
    let center_multiplier = match game_phase {
        GamePhase::Opening => 1.5,
        _ => 1.0,
    };
    
    // Peões no centro
    let pawns_in_center = (board.pawns & pieces & CENTRAL_SQUARES).count_ones() as i32 * (20.0 * center_multiplier) as i32;
    let pawns_in_extended_center = (board.pawns & pieces & EXTENDED_CENTER).count_ones() as i32 * (10.0 * center_multiplier) as i32;
    
    // Cavalos no centro (mais importante na abertura)
    let knights_in_center = (board.knights & pieces & CENTRAL_SQUARES).count_ones() as i32 * (15.0 * center_multiplier) as i32;
    let knights_in_extended_center = (board.knights & pieces & EXTENDED_CENTER).count_ones() as i32 * (8.0 * center_multiplier) as i32;
    
    // Bispos no centro
    let bishops_in_center = (board.bishops & pieces & CENTRAL_SQUARES).count_ones() as i32 * (12.0 * center_multiplier) as i32;
    
    score += pawns_in_center + pawns_in_extended_center;
    score += knights_in_center + knights_in_extended_center;
    score += bishops_in_center;
    
    score
}