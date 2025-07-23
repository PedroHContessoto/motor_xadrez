// Estrutura de peões - peões passados, conectados, etc.
use crate::{board::Board, types::{Color, Bitboard}};
use super::material::MATERIAL_VALUES;

// Bônus para peões passados com base na sua fileira (rank)
const PASSED_PAWN_BONUS: [i32; 8] = [0, 10, 20, 30, 50, 75, 100, 0];

// Máscaras de bits para cada coluna
const FILE_MASKS: [Bitboard; 8] = [
    0x0101010101010101, 0x0202020202020202, 0x0404040404040404, 0x0808080808080808,
    0x1010101010101010, 0x2020202020202020, 0x4040404040404040, 0x8080808080808080,
];

// Máscaras para peões passados (inicializadas no init)
static mut PASSED_PAWN_MASKS: [[Bitboard; 64]; 2] = [[0; 64]; 2];
static mut MASKS_INITIALIZED: bool = false;

/// Inicializa as máscaras de peões passados (deve ser chamada uma vez)
pub fn init_pawn_masks() {
    unsafe {
        if MASKS_INITIALIZED {
            return;
        }
        
        for sq in 0..64 {
            let rank = sq / 8;
            let file = sq % 8;

            // Máscara para Brancas (peões à frente e adjacentes)
            let mut white_mask: Bitboard = 0;
            for r in (rank + 1)..8 {
                white_mask |= FILE_MASKS[file] & (0xFF << (r * 8));
                if file > 0 { white_mask |= FILE_MASKS[file - 1] & (0xFF << (r * 8)); }
                if file < 7 { white_mask |= FILE_MASKS[file + 1] & (0xFF << (r * 8)); }
            }
            PASSED_PAWN_MASKS[Color::White as usize][sq] = white_mask;

            // Máscara para Pretas
            let mut black_mask: Bitboard = 0;
            for r in 0..rank {
                black_mask |= FILE_MASKS[file] & (0xFF << (r * 8));
                if file > 0 { black_mask |= FILE_MASKS[file - 1] & (0xFF << (r * 8)); }
                if file < 7 { black_mask |= FILE_MASKS[file + 1] & (0xFF << (r * 8)); }
            }
            PASSED_PAWN_MASKS[Color::Black as usize][sq] = black_mask;
        }
        
        MASKS_INITIALIZED = true;
    }
}

/// Avalia estrutura de peões completa
pub fn evaluate_pawn_structure(board: &Board, color: Color) -> i32 {
    let mut score = 0;
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let our_pawns = board.pawns & pieces;
    
    // Inicializa máscaras se necessário
    unsafe {
        if !MASKS_INITIALIZED {
            init_pawn_masks();
        }
    }
    
    // Material básico + PST para peões
    score += evaluate_pawn_material_pst(our_pawns, color);
    
    // Peões passados
    score += evaluate_passed_pawns(board, our_pawns, color);
    
    // Peões conectados/duplos
    score += evaluate_pawn_connections(our_pawns);
    
    // Peões isolados e atrasados
    score -= evaluate_pawn_weaknesses(our_pawns, board, color);
    
    score
}

/// Material básico e PST para peões
fn evaluate_pawn_material_pst(mut pawn_bb: Bitboard, color: Color) -> i32 {
    let mut score = 0;
    
    // Tabela PST para peões (copiada do código original)
    const PAWN_TABLE: [i32; 64] = [
        0,0,0,0,0,0,0,0,
        50,50,50,50,50,50,50,50,
        10,10,20,30,30,20,10,10,
        5,5,10,25,25,10,5,5,
        0,0,0,20,20,0,0,0,
        5,-5,-10,0,0,-10,-5,5,
        5,10,10,-20,-20,10,10,5,
        0,0,0,0,0,0,0,0
    ];
    
    while pawn_bb != 0 {
        let sq = pawn_bb.trailing_zeros() as usize;
        pawn_bb &= pawn_bb - 1;
        
        score += MATERIAL_VALUES[0]; // Peão = 100
        score += if color == Color::White { PAWN_TABLE[sq] } else { PAWN_TABLE[sq ^ 56] };
    }
    
    score
}

/// Avalia peões passados
fn evaluate_passed_pawns(board: &Board, mut pawn_bb: Bitboard, color: Color) -> i32 {
    let mut score = 0;
    let enemy_pawns = if color == Color::White { 
        board.pawns & board.black_pieces 
    } else { 
        board.pawns & board.white_pieces 
    };
    let color_idx = color as usize;

    while pawn_bb != 0 {
        let sq = pawn_bb.trailing_zeros() as usize;
        pawn_bb &= pawn_bb - 1;

        unsafe {
            if (PASSED_PAWN_MASKS[color_idx][sq] & enemy_pawns) == 0 {
                // É um peão passado!
                let rank = sq / 8;
                let bonus = if color == Color::White {
                    PASSED_PAWN_BONUS[rank]
                } else {
                    PASSED_PAWN_BONUS[7 - rank]
                };
                
                score += bonus;
                
                // Bônus extra se próximo da promoção
                let promotion_rank = if color == Color::White { 7 } else { 0 };
                let distance_to_promotion = if color == Color::White { 
                    7 - rank 
                } else { 
                    rank 
                };
                
                if distance_to_promotion <= 2 {
                    score += bonus; // Dobra bônus se muito próximo
                }
            }
        }
    }

    score
}

/// Avalia conexões entre peões (corrente de peões)
fn evaluate_pawn_connections(pawn_bb: Bitboard) -> i32 {
    let mut score = 0;
    let mut temp_bb = pawn_bb;
    
    while temp_bb != 0 {
        let sq = temp_bb.trailing_zeros() as usize;
        temp_bb &= temp_bb - 1;
        
        let file = sq % 8;
        let rank = sq / 8;
        
        // Verifica peões conectados diagonalmente (cadeia)
        let diagonal_supports = [
            if file > 0 && rank > 0 { Some((rank - 1) * 8 + (file - 1)) } else { None },
            if file < 7 && rank > 0 { Some((rank - 1) * 8 + (file + 1)) } else { None },
        ];
        
        for support_sq in diagonal_supports.iter().flatten() {
            if (pawn_bb & (1u64 << support_sq)) != 0 {
                score += 8; // Bônus por suporte diagonal
            }
        }
        
        // Verifica peões adjacentes na mesma linha
        let adjacent_pawns = [
            if file > 0 { Some(rank * 8 + (file - 1)) } else { None },
            if file < 7 { Some(rank * 8 + (file + 1)) } else { None },
        ];
        
        for adj_sq in adjacent_pawns.iter().flatten() {
            if (pawn_bb & (1u64 << adj_sq)) != 0 {
                score += 5; // Pequeno bônus por peões lado a lado
            }
        }
    }
    
    score
}

/// Penaliza fraquezas na estrutura de peões
fn evaluate_pawn_weaknesses(pawn_bb: Bitboard, board: &Board, color: Color) -> i32 {
    let mut penalty = 0;
    let mut temp_bb = pawn_bb;
    
    // Conta peões por coluna para detectar duplos
    let mut pawns_per_file = [0u8; 8];
    let mut file_bb = pawn_bb;
    while file_bb != 0 {
        let sq = file_bb.trailing_zeros() as usize;
        file_bb &= file_bb - 1;
        pawns_per_file[sq % 8] += 1;
    }
    
    while temp_bb != 0 {
        let sq = temp_bb.trailing_zeros() as usize;
        temp_bb &= temp_bb - 1;
        
        let file = sq % 8;
        
        // Penaliza peões duplos
        if pawns_per_file[file] > 1 {
            penalty += 15 * (pawns_per_file[file] - 1) as i32;
        }
        
        // Penaliza peões isolados (sem peões adjacentes)
        let has_adjacent_pawns = 
            (file > 0 && pawns_per_file[file - 1] > 0) ||
            (file < 7 && pawns_per_file[file + 1] > 0);
        
        if !has_adjacent_pawns {
            penalty += 20; // Peão isolado
        }
        
        // Penaliza peões atrasados (sem suporte e não pode avançar)
        if is_backward_pawn(sq, pawn_bb, board, color) {
            penalty += 15;
        }
    }
    
    penalty
}

/// Verifica se um peão é atrasado
fn is_backward_pawn(pawn_square: usize, our_pawns: Bitboard, board: &Board, color: Color) -> bool {
    let file = pawn_square % 8;
    let rank = pawn_square / 8;
    
    // Verifica se há peões de suporte nas colunas adjacentes atrás
    let support_rank = if color == Color::White {
        if rank == 0 { return false; } // Não pode ter suporte atrás na primeira linha
        rank - 1
    } else {
        if rank == 7 { return false; }
        rank + 1
    };
    
    let has_support = 
        (file > 0 && (our_pawns & (1u64 << (support_rank * 8 + file - 1))) != 0) ||
        (file < 7 && (our_pawns & (1u64 << (support_rank * 8 + file + 1))) != 0);
    
    if has_support {
        return false; // Tem suporte, não é atrasado
    }
    
    // Verifica se pode avançar seguramente
    let advance_square = if color == Color::White {
        if rank >= 7 { return false; }
        (rank + 1) * 8 + file
    } else {
        if rank == 0 { return false; }
        (rank - 1) * 8 + file
    };
    
    // Se casa à frente está ocupada ou atacada por peão inimigo, é atrasado
    let all_pieces = board.white_pieces | board.black_pieces;
    if (all_pieces & (1u64 << advance_square)) != 0 {
        return true; // Bloqueado
    }
    
    // Verifica se a casa é atacada por peões inimigos
    let enemy_color = !color;
    let enemy_pawns = if enemy_color == Color::White {
        board.pawns & board.white_pieces
    } else {
        board.pawns & board.black_pieces
    };
    
    // Simula ataques de peões inimigos à casa de avanço
    attacks_square_pawn(advance_square, enemy_pawns, enemy_color)
}

/// Verifica se peões atacam uma casa específica
fn attacks_square_pawn(target_square: usize, pawn_bb: Bitboard, pawn_color: Color) -> bool {
    let target_file = target_square % 8;
    let target_rank = target_square / 8;
    
    // Casas de onde peões podem atacar o alvo
    let attacker_squares = if pawn_color == Color::White {
        // Peões brancos atacam de baixo
        let mut attackers = Vec::new();
        if target_rank > 0 {
            if target_file > 0 { attackers.push((target_rank - 1) * 8 + target_file - 1); }
            if target_file < 7 { attackers.push((target_rank - 1) * 8 + target_file + 1); }
        }
        attackers
    } else {
        // Peões pretos atacam de cima
        let mut attackers = Vec::new();
        if target_rank < 7 {
            if target_file > 0 { attackers.push((target_rank + 1) * 8 + target_file - 1); }
            if target_file < 7 { attackers.push((target_rank + 1) * 8 + target_file + 1); }
        }
        attackers
    };
    
    // Verifica se há peões nessas casas
    attacker_squares.iter().any(|&sq| (pawn_bb & (1u64 << sq)) != 0)
}

/// Função auxiliar para converter bitboard em vetor de casas
pub fn get_set_bits_simple(mut bitboard: Bitboard) -> Vec<u8> {
    let mut squares = Vec::new();
    while bitboard != 0 {
        let sq = bitboard.trailing_zeros() as u8;
        bitboard &= bitboard - 1;
        squares.push(sq);
    }
    squares
}