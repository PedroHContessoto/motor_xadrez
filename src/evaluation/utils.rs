// utils.rs - Funções auxiliares para avaliação
use crate::types::Bitboard;

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