// utils.rs - Funções auxiliares para avaliação
use crate::{types::{Bitboard, Color}, board::Board};

/// Extrai todas as posições de bits definidos em um bitboard
pub fn get_set_bits(mut bb: Bitboard) -> Vec<u8> {
    let mut squares = Vec::new();
    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        squares.push(sq);
        bb &= bb - 1; // Remove o bit mais baixo
    }
    squares
}

/// Calcula a distância de Manhattan entre duas casas
pub fn manhattan_distance(sq1: u8, sq2: u8) -> i32 {
    let rank1 = (sq1 / 8) as i32;
    let file1 = (sq1 % 8) as i32;
    let rank2 = (sq2 / 8) as i32;
    let file2 = (sq2 % 8) as i32;

    (rank1 - rank2).abs() + (file1 - file2).abs()
}

/// Calcula a distância Chebyshev (máximo de rank/file) entre duas casas
pub fn calculate_square_distance(sq1: u8, sq2: u8) -> i32 {
    let rank1 = (sq1 / 8) as i32;
    let file1 = (sq1 % 8) as i32;
    let rank2 = (sq2 / 8) as i32;
    let file2 = (sq2 % 8) as i32;

    let rank_diff = (rank1 - rank2).abs();
    let file_diff = (file1 - file2).abs();

    rank_diff.max(file_diff)
}

/// Verifica se uma casa está na borda do tabuleiro
pub fn is_edge_square(sq: u8) -> bool {
    let rank = sq / 8;
    let file = sq % 8;
    rank == 0 || rank == 7 || file == 0 || file == 7
}

/// Verifica se uma casa está no centro (4 casas centrais)
pub fn is_center_square(sq: u8) -> bool {
    let rank = sq / 8;
    let file = sq % 8;
    (rank == 3 || rank == 4) && (file == 3 || file == 4)
}

/// Verifica se uma casa está no centro expandido (16 casas centrais)
pub fn is_extended_center_square(sq: u8) -> bool {
    let rank = sq / 8;
    let file = sq % 8;
    rank >= 2 && rank <= 5 && file >= 2 && file <= 5
}

/// Máscara de arquivo para uma casa
pub fn get_file_mask(sq: u8) -> Bitboard {
    let file = sq % 8;
    0x0101010101010101u64 << file
}

/// Máscara de rank para uma casa
pub fn get_rank_mask(sq: u8) -> Bitboard {
    let rank = sq / 8;
    0xFFu64 << (rank * 8)
}

/// Bitboard com todas as casas diagonalmente adjacentes
pub fn get_diagonal_mask(sq: u8) -> Bitboard {
    let rank = (sq / 8) as i32;
    let file = (sq % 8) as i32;
    let mut mask = 0u64;

    // Diagonal principal (cima-direita e baixo-esquerda)
    for i in 1..8 {
        let new_rank = rank + i;
        let new_file = file + i;
        if new_rank < 8 && new_file < 8 {
            mask |= 1u64 << (new_rank * 8 + new_file);
        }

        let new_rank = rank - i;
        let new_file = file - i;
        if new_rank >= 0 && new_file >= 0 {
            mask |= 1u64 << (new_rank * 8 + new_file);
        }
    }

    // Diagonal secundária (cima-esquerda e baixo-direita)
    for i in 1..8 {
        let new_rank = rank + i;
        let new_file = file - i;
        if new_rank < 8 && new_file >= 0 {
            mask |= 1u64 << (new_rank * 8 + new_file);
        }

        let new_rank = rank - i;
        let new_file = file + i;
        if new_rank >= 0 && new_file < 8 {
            mask |= 1u64 << (new_rank * 8 + new_file);
        }
    }

    mask
}

/// Conta bits definidos em um bitboard (população)
pub fn popcount(bb: Bitboard) -> u32 {
    bb.count_ones()
}

/// Encontra o bit mais baixo definido (LSB)
pub fn lsb(bb: Bitboard) -> u8 {
    bb.trailing_zeros() as u8
}

/// Remove o bit mais baixo definido
pub fn pop_lsb(bb: &mut Bitboard) -> u8 {
    let sq = lsb(*bb);
    *bb &= *bb - 1;
    sq
}

// === FUNÇÕES CONSOLIDADAS ===

/// FUNÇÃO CONSOLIDADA: Verifica se peão é passado (melhor implementação)
pub fn is_passed_pawn(pawn_sq: u8, pawn_color: Color, our_pawns: Bitboard, enemy_pawns: Bitboard) -> bool {
    let file = pawn_sq % 8;
    let rank = pawn_sq / 8;

    // Verifica arquivos adjacentes e o próprio arquivo
    for check_file in (file.saturating_sub(1))..=(file.saturating_add(1)).min(7) {
        let file_mask = 0x0101010101010101u64 << check_file;
        let file_pawns = enemy_pawns & file_mask;

        if file_pawns != 0 {
            let enemy_pawn_squares = get_set_bits(file_pawns);
            for enemy_sq in enemy_pawn_squares {
                let enemy_rank = enemy_sq / 8;

                let blocks_advancement = if pawn_color == Color::White {
                    enemy_rank > rank
                } else {
                    enemy_rank < rank
                };

                if blocks_advancement {
                    return false;
                }
            }
        }
    }
    true
}

/// FUNÇÃO CONSOLIDADA: Ataques de peões (melhor implementação)
pub fn compute_pawn_attacks(pawns: Bitboard, color: Color) -> Bitboard {
    const NOT_A_FILE: Bitboard = 0xfefefefefefefefe;
    const NOT_H_FILE: Bitboard = 0x7f7f7f7f7f7f7f7f;

    if color == Color::White {
        let left_attacks = (pawns & NOT_A_FILE) << 7;
        let right_attacks = (pawns & NOT_H_FILE) << 9;
        left_attacks | right_attacks
    } else {
        let left_attacks = (pawns & NOT_H_FILE) >> 7;
        let right_attacks = (pawns & NOT_A_FILE) >> 9;
        left_attacks | right_attacks
    }
}

/// FUNÇÃO CONSOLIDADA: Verifica se peão pode atacar casa específica
pub fn can_pawn_attack_square(pawn_sq: u8, target_sq: u8, color: Color) -> bool {
    let pawn_rank = pawn_sq / 8;
    let pawn_file = pawn_sq % 8;
    let target_rank = target_sq / 8;
    let target_file = target_sq % 8;

    if color == Color::White {
        target_rank > pawn_rank &&
            (target_file as i8 - pawn_file as i8).abs() == 1
    } else {
        target_rank < pawn_rank &&
            (target_file as i8 - pawn_file as i8).abs() == 1
    }
}

/// FUNÇÃO CONSOLIDADA: Verifica se casa é outpost (melhor implementação)
pub fn is_outpost(square: u8, color: Color, enemy_pawns: Bitboard) -> bool {
    let rank = square / 8;
    let file = square % 8;

    // Outposts são normalmente entre 4ª e 6ª fileiras
    let valid_ranks = if color == Color::White {
        rank >= 3 && rank <= 5
    } else {
        rank >= 2 && rank <= 4
    };

    if !valid_ranks {
        return false;
    }

    // Verifica se não há peões inimigos que podem atacar esta casa
    for adj_file in (file.saturating_sub(1))..=(file.saturating_add(1)).min(7) {
        if adj_file == file { continue; }

        let file_mask = get_file_mask_from_file(adj_file);
        let file_pawns = enemy_pawns & file_mask;
        if file_pawns != 0 {
            let pawn_squares = get_set_bits(file_pawns);
            for pawn_sq in pawn_squares {
                if can_pawn_attack_square(pawn_sq, square, !color) {
                    return false;
                }
            }
        }
    }
    true
}

/// FUNÇÃO CONSOLIDADA: Máscara de arquivo por número de arquivo
pub fn get_file_mask_from_file(file: u8) -> Bitboard {
    0x0101010101010101u64 << file
}


/// FUNÇÃO CONSOLIDADA: Distância de rei (usada em endgame)
pub fn king_distance(king1: u8, king2: u8) -> u8 {
    let file1 = king1 % 8;
    let rank1 = king1 / 8;
    let file2 = king2 % 8;
    let rank2 = king2 / 8;

    let file_diff = (file1 as i8 - file2 as i8).abs() as u8;
    let rank_diff = (rank1 as i8 - rank2 as i8).abs() as u8;

    file_diff.max(rank_diff)
}