// Avaliação de material e tabelas de posição
use crate::{board::Board, types::{Color, PieceKind, Bitboard}};
use super::game_phase::GamePhase;

// Extension trait para funções auxiliares do Board
trait BoardExt {
    fn has_many_attacks(&self) -> bool;
    fn has_hanging_pieces(&self, color: Color) -> bool;
    fn has_tactical_threats(&self, color: Color) -> bool;
    fn get_valuable_pieces(&self, color: Color) -> Bitboard;
}

impl BoardExt for Board {
    fn has_many_attacks(&self) -> bool {
        // Detecta posições táticas pela densidade de peças
        (self.white_pieces | self.black_pieces).count_ones() > 20
    }

    fn has_hanging_pieces(&self, color: Color) -> bool {
        let our_valuables = (self.knights | self.bishops | self.rooks | self.queens) &
            if color == Color::White { self.white_pieces } else { self.black_pieces };

        let mut bb = our_valuables;
        while bb != 0 {
            let sq = bb.trailing_zeros() as u8;
            bb &= bb - 1;

            if self.is_square_attacked_by(sq, !color) {
                return true;
            }
        }
        false
    }

    fn has_tactical_threats(&self, color: Color) -> bool {
        let valuable_pieces = self.get_valuable_pieces(color);
        let mut threatened_count = 0;

        let mut bb = valuable_pieces;
        while bb != 0 {
            let sq = bb.trailing_zeros() as u8;
            bb &= bb - 1;

            if self.is_square_attacked_by(sq, !color) {
                threatened_count += 1;
            }
        }

        threatened_count > 1
    }

    fn get_valuable_pieces(&self, color: Color) -> Bitboard {
        let pieces = if color == Color::White { self.white_pieces } else { self.black_pieces };
        (self.queens | self.rooks | self.bishops | self.knights) & pieces
    }
}

pub const MATERIAL_VALUES: [i32; 6] = [100, 320, 330, 500, 900, 20000];

// Tabelas de posição (PST) - copiadas do código original
const PAWN_TABLE: [i32; 64] = [0,0,0,0,0,0,0,0,50,50,50,50,50,50,50,50,10,10,20,30,30,20,10,10,5,5,10,25,25,10,5,5,0,0,0,20,20,0,0,0,5,-5,-10,0,0,-10,-5,5,5,10,10,-20,-20,10,10,5,0,0,0,0,0,0,0,0];
// Tabela de cavalo aprimorada: -50 corners, -30 edges, +30 centro
const KNIGHT_TABLE: [i32; 64] = [
    -50,-40,-30,-25,-25,-30,-40,-50,
    -40,-20, -5,  0,  0, -5,-20,-40,
    -30, -5, 10, 15, 15, 10, -5,-30,
    -25,  0, 15, 30, 30, 15,  0,-25,
    -25,  0, 15, 30, 30, 15,  0,-25,
    -30, -5, 10, 15, 15, 10, -5,-30,
    -40,-20, -5,  0,  0, -5,-20,-40,
    -50,-40,-30,-25,-25,-30,-40,-50
];
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
        GamePhase::EarlyEndgame | GamePhase::Endgame | GamePhase::LateEndgame | GamePhase::PureEndgame | GamePhase::TheoreticalEndgame => &KING_TABLE_ENDGAME,
        _ => &KING_TABLE_MIDGAME,
    };

    // Avalia cada tipo de peça
    score += evaluate_piece_type(board.pawns & pieces, &PAWN_TABLE, PieceKind::Pawn, color);
    score += evaluate_knights_enhanced(board, board.knights & pieces, color, game_phase); // Versão melhorada para cavalos
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

    // Bônus ajustado por fase do jogo e situação tática
    let center_multiplier = match game_phase {
        GamePhase::Opening => if board.has_many_attacks() { 1.0 } else { 1.2 }, // Era 1.5, reduzido se tático
        GamePhase::EarlyMiddlegame => 1.1,
        GamePhase::Middlegame => 1.0,
        GamePhase::LateMiddlegame => 0.9,
        GamePhase::EarlyEndgame | GamePhase::Endgame | GamePhase::LateEndgame | GamePhase::PureEndgame | GamePhase::TheoreticalEndgame => 0.8, // Reduzido de 1.0 para 0.8
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

/// Avaliação aprimorada de cavalos com outposts e mobilidade
fn evaluate_knights_enhanced(board: &Board, mut knight_bb: Bitboard, color: Color, game_phase: &GamePhase) -> i32 {
    let mut score = 0;

    while knight_bb != 0 {
        let sq = knight_bb.trailing_zeros() as usize;
        knight_bb &= knight_bb - 1;

        // Material + PST base
        let positional_score = if color == Color::White { KNIGHT_TABLE[sq] } else { KNIGHT_TABLE[sq ^ 56] };
        score += MATERIAL_VALUES[PieceKind::Knight as usize] + positional_score;

        // Bônus por outpost (posição segura em território inimigo)
        if is_outpost(sq as u8, color) && defended_by_own_pawn(board, sq as u8, color) {
            let outpost_bonus = match game_phase {
                GamePhase::Opening => 20,
                GamePhase::EarlyMiddlegame => 25,
                GamePhase::Middlegame => 30,
                GamePhase::LateMiddlegame => 35,
                GamePhase::EarlyEndgame => 38,
                GamePhase::Endgame => 40,
                GamePhase::LateEndgame => 42,
                GamePhase::PureEndgame => 45, // Mais valioso no endgame
                GamePhase::TheoreticalEndgame => 50, // Máximo no final teórico
            };
            score += outpost_bonus;
        }

        // Mobilidade do cavalo (casas seguras)
        score += evaluate_knight_mobility_safe(board, sq as u8, color);
    }

    score
}

/// Verifica se uma casa é um outpost (ranks 4-5 para brancas, sem ataques de peões inimigos)
fn is_outpost(square: u8, color: Color) -> bool {
    let rank = square / 8;

    match color {
        Color::White => {
            // Ranks 4-5 (0-indexed: 3-4) são considerados outposts
            if rank < 3 || rank > 4 {
                return false;
            }

            // Verifica se não há peões inimigos que podem atacar
            !can_enemy_pawns_attack(square, Color::Black)
        },
        Color::Black => {
            // Ranks 5-4 (0-indexed: 4-3) são considerados outposts  
            if rank < 3 || rank > 4 {
                return false;
            }

            !can_enemy_pawns_attack(square, Color::White)
        }
    }
}

/// Verifica se peões inimigos podem atacar uma casa
fn can_enemy_pawns_attack(square: u8, enemy_color: Color) -> bool {
    let file = square % 8;
    let rank = square / 8;

    match enemy_color {
        Color::White => {
            // Peões brancos atacam diagonalmente para cima
            if rank == 0 { return false; }

            // Verifica files adjacentes - versão conservadora (assume que pode haver peões)
            if file > 0 && rank < 7 { return true; }
            if file < 7 && rank < 7 { return true; }
            false
        },
        Color::Black => {
            // Peões pretos atacam diagonalmente para baixo
            if rank == 7 { return false; }

            if file > 0 && rank > 0 { return true; }
            if file < 7 && rank > 0 { return true; }
            false
        }
    }
}

/// Verifica se cavalo está defendido por peão próprio
fn defended_by_own_pawn(board: &Board, square: u8, color: Color) -> bool {
    let our_pawns = board.pawns & if color == Color::White { board.white_pieces } else { board.black_pieces };

    let file = square % 8;
    let rank = square / 8;

    match color {
        Color::White => {
            // Peões brancos defendem de baixo
            if rank == 0 { return false; }

            let defend_rank = rank - 1;
            if file > 0 {
                let defend_sq = defend_rank * 8 + file - 1;
                if (1u64 << defend_sq) & our_pawns != 0 { return true; }
            }
            if file < 7 {
                let defend_sq = defend_rank * 8 + file + 1;
                if (1u64 << defend_sq) & our_pawns != 0 { return true; }
            }
            false
        },
        Color::Black => {
            // Peões pretos defendem de cima
            if rank == 7 { return false; }

            let defend_rank = rank + 1;
            if file > 0 {
                let defend_sq = defend_rank * 8 + file - 1;
                if (1u64 << defend_sq) & our_pawns != 0 { return true; }
            }
            if file < 7 {
                let defend_sq = defend_rank * 8 + file + 1;
                if (1u64 << defend_sq) & our_pawns != 0 { return true; }
            }
            false
        }
    }
}

/// Avalia mobilidade segura do cavalo (casas não atacadas pelo inimigo)
fn evaluate_knight_mobility_safe(board: &Board, square: u8, color: Color) -> i32 {
    let enemy_color = !color;
    let knight_moves = crate::moves::knight::get_knight_attacks_lookup(square);

    let mut safe_squares = 0;
    let mut moves_bb = knight_moves;

    while moves_bb != 0 {
        let target_sq = moves_bb.trailing_zeros() as u8;
        moves_bb &= moves_bb - 1;

        // Conta apenas casas não atacadas pelo inimigo
        if !board.is_square_attacked_by(target_sq, enemy_color) {
            safe_squares += 1;
        }
    }

    // +5 por cada casa segura
    safe_squares * 5
}