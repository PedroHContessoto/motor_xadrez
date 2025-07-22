// Mobilidade segura - considera ataques inimigos
use crate::{board::Board, types::{Color, PieceKind}};

// Pesos para mobilidade por tipo de peça
const MOBILITY_WEIGHTS: [i32; 6] = [0, 4, 4, 2, 1, 0]; // [pawn, knight, bishop, rook, queen, king]

/// Avalia mobilidade das peças (versão melhorada com mobilidade "segura")
pub fn evaluate_mobility(board: &Board, color: Color) -> i32 {
    let mut score = 0;
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let enemy_color = !color;
    let all_pieces = board.white_pieces | board.black_pieces;
    
    // Pré-computa casas atacadas por inimigos (para mobilidade segura)
    let enemy_attacked_squares = compute_enemy_attacked_squares(board, enemy_color);
    
    // Mobilidade dos cavalos
    score += evaluate_knight_mobility(board, pieces, enemy_attacked_squares);
    
    // Mobilidade dos bispos
    score += evaluate_bishop_mobility(board, pieces, all_pieces, enemy_attacked_squares);
    
    // Mobilidade das torres
    score += evaluate_rook_mobility(board, pieces, all_pieces, enemy_attacked_squares);
    
    // Mobilidade das rainhas
    score += evaluate_queen_mobility(board, pieces, all_pieces, enemy_attacked_squares);
    
    score
}

/// Mobilidade dos cavalos
fn evaluate_knight_mobility(board: &Board, our_pieces: crate::types::Bitboard, enemy_attacked: crate::types::Bitboard) -> i32 {
    let mut score = 0;
    let mut knights = board.knights & our_pieces;
    
    while knights != 0 {
        let sq = knights.trailing_zeros() as usize;
        knights &= knights - 1;
        
        let attacks = crate::moves::knight::get_knight_attacks_lookup(sq as u8);
        let legal_squares = attacks & !our_pieces; // Não pode ir para casa própria
        let safe_squares = legal_squares & !enemy_attacked; // Mobilidade segura
        
        // Pontuação: Total de casas + bônus por casas seguras
        let total_mobility = legal_squares.count_ones() as i32;
        let safe_mobility = safe_squares.count_ones() as i32;
        
        score += total_mobility * MOBILITY_WEIGHTS[PieceKind::Knight as usize];
        score += safe_mobility * 2; // Bônus por mobilidade segura
    }
    
    score
}

/// Mobilidade dos bispos
fn evaluate_bishop_mobility(board: &Board, our_pieces: crate::types::Bitboard, all_pieces: crate::types::Bitboard, enemy_attacked: crate::types::Bitboard) -> i32 {
    let mut score = 0;
    let mut bishops = board.bishops & our_pieces;
    
    while bishops != 0 {
        let sq = bishops.trailing_zeros() as usize;
        bishops &= bishops - 1;
        
        let attacks = crate::moves::sliding::get_bishop_attacks(sq as u8, all_pieces);
        let legal_squares = attacks & !our_pieces;
        let safe_squares = legal_squares & !enemy_attacked;
        
        let total_mobility = legal_squares.count_ones() as i32;
        let safe_mobility = safe_squares.count_ones() as i32;
        
        score += total_mobility * MOBILITY_WEIGHTS[PieceKind::Bishop as usize];
        score += safe_mobility * 2;
    }
    
    score
}

/// Mobilidade das torres
fn evaluate_rook_mobility(board: &Board, our_pieces: crate::types::Bitboard, all_pieces: crate::types::Bitboard, enemy_attacked: crate::types::Bitboard) -> i32 {
    let mut score = 0;
    let mut rooks = board.rooks & our_pieces;
    
    while rooks != 0 {
        let sq = rooks.trailing_zeros() as usize;
        rooks &= rooks - 1;
        
        let attacks = crate::moves::sliding::get_rook_attacks(sq as u8, all_pieces);
        let legal_squares = attacks & !our_pieces;
        let safe_squares = legal_squares & !enemy_attacked;
        
        let total_mobility = legal_squares.count_ones() as i32;
        let safe_mobility = safe_squares.count_ones() as i32;
        
        score += total_mobility * MOBILITY_WEIGHTS[PieceKind::Rook as usize];
        score += safe_mobility * 3; // Torres valorizam mais mobilidade segura
    }
    
    score
}

/// Mobilidade das rainhas
fn evaluate_queen_mobility(board: &Board, our_pieces: crate::types::Bitboard, all_pieces: crate::types::Bitboard, enemy_attacked: crate::types::Bitboard) -> i32 {
    let mut score = 0;
    let mut queens = board.queens & our_pieces;
    
    while queens != 0 {
        let sq = queens.trailing_zeros() as usize;
        queens &= queens - 1;
        
        let bishop_attacks = crate::moves::sliding::get_bishop_attacks(sq as u8, all_pieces);
        let rook_attacks = crate::moves::sliding::get_rook_attacks(sq as u8, all_pieces);
        let attacks = bishop_attacks | rook_attacks;
        let legal_squares = attacks & !our_pieces;
        let safe_squares = legal_squares & !enemy_attacked;
        
        let total_mobility = legal_squares.count_ones() as i32;
        let safe_mobility = safe_squares.count_ones() as i32;
        
        score += total_mobility * MOBILITY_WEIGHTS[PieceKind::Queen as usize];
        score += safe_mobility * 1; // Rainha é menos sensível (já tem muito valor)
        
        // Penaliza rainha desenvolvida muito cedo (se muitas peças ainda no back rank)
        if is_early_game(board) {
            score -= 10; // Pequena penalidade por rainha ativa cedo
        }
    }
    
    score
}

/// Computa aproximação de casas atacadas por inimigos
/// Esta é uma versão otimizada - em engines mais avançados, use bitboards pré-computados
fn compute_enemy_attacked_squares(board: &Board, enemy_color: Color) -> crate::types::Bitboard {
    let mut attacked = 0u64;
    let enemy_pieces = if enemy_color == Color::White { board.white_pieces } else { board.black_pieces };
    let all_pieces = board.white_pieces | board.black_pieces;
    
    // Ataques de peões (mais comuns)
    let enemy_pawns = board.pawns & enemy_pieces;
    attacked |= compute_pawn_attacks(enemy_pawns, enemy_color);
    
    // Ataques de cavalos
    let mut knights = board.knights & enemy_pieces;
    while knights != 0 {
        let sq = knights.trailing_zeros() as u8;
        knights &= knights - 1;
        attacked |= crate::moves::knight::get_knight_attacks_lookup(sq);
    }
    
    // Ataques de rei
    let enemy_king = board.kings & enemy_pieces;
    if enemy_king != 0 {
        let king_sq = enemy_king.trailing_zeros() as u8;
        attacked |= crate::moves::king::get_king_attacks_lookup(king_sq);
    }
    
    // Ataques de peças deslizantes (simplificado)
    let mut sliders = (board.bishops | board.rooks | board.queens) & enemy_pieces;
    while sliders != 0 {
        let sq = sliders.trailing_zeros() as u8;
        sliders &= sliders - 1;
        
        let piece_bb = 1u64 << sq;
        if (board.bishops & piece_bb) != 0 || (board.queens & piece_bb) != 0 {
            attacked |= crate::moves::sliding::get_bishop_attacks(sq, all_pieces);
        }
        if (board.rooks & piece_bb) != 0 || (board.queens & piece_bb) != 0 {
            attacked |= crate::moves::sliding::get_rook_attacks(sq, all_pieces);
        }
    }
    
    attacked
}

/// Computa ataques de peões para uma cor
fn compute_pawn_attacks(pawns: crate::types::Bitboard, color: Color) -> crate::types::Bitboard {
    const NOT_A_FILE: crate::types::Bitboard = 0xfefefefefefefefe;
    const NOT_H_FILE: crate::types::Bitboard = 0x7f7f7f7f7f7f7f7f;
    
    if color == Color::White {
        // Brancas: ataques para cima
        let left_attacks = (pawns & NOT_A_FILE) << 7;
        let right_attacks = (pawns & NOT_H_FILE) << 9;
        left_attacks | right_attacks
    } else {
        // Pretas: ataques para baixo
        let left_attacks = (pawns & NOT_H_FILE) >> 7;
        let right_attacks = (pawns & NOT_A_FILE) >> 9;
        left_attacks | right_attacks
    }
}

/// Verifica se ainda é início de jogo (muitas peças no back rank)
fn is_early_game(board: &Board) -> bool {
    let back_ranks = 0xFF | 0xFF00000000000000; // Rank 1 e 8
    let pieces_on_back_rank = (board.white_pieces | board.black_pieces) & back_ranks;
    pieces_on_back_rank.count_ones() > 10 // Mais de 10 peças ainda nos ranks iniciais
}