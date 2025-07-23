// Mobilidade e segurança do rei
use crate::types::{Color, Bitboard};
use super::{MobilityContext, MobilityResult, GamePhase, utils::*};
use crate::evaluation::utils as eval_utils;

/// Avaliação avançada de mobilidade e segurança do rei
pub fn evaluate_king_mobility_advanced(context: &MobilityContext) -> i32 {
    let mut total_score = 0;
    let kings = context.board.kings & context.our_pieces;
    
    if kings == 0 { return 0; }
    
    let king_sq = kings.trailing_zeros() as u8;
    
    match context.phase {
        GamePhase::Opening | GamePhase::MiddleGame => {
            // No início/meio-jogo, priorize segurança
            total_score += evaluate_king_safety(king_sq, context);
        }
        GamePhase::Endgame => {
            // No final, o rei deve ser ativo
            total_score += evaluate_king_activity(king_sq, context);
        }
    }
    
    total_score
}

/// Avalia segurança do rei
fn evaluate_king_safety(king_sq: u8, context: &MobilityContext) -> i32 {
    let mut safety_score = 0;
    
    // Escudo de peões
    let pawn_shield = evaluate_pawn_shield(king_sq, context);
    safety_score += pawn_shield;
    
    // Distância de peças inimigas perigosas
    let threat_distance = evaluate_threat_distance(king_sq, context);
    safety_score += threat_distance;
    
    // Mobilidade limitada é boa na abertura/meio-jogo
    let king_attacks = crate::moves::king::get_king_attacks_lookup(king_sq);
    let safe_squares = king_attacks & !context.enemy_attacked_squares & !context.our_pieces;
    let mobility = safe_squares.count_ones() as i32;
    
    // Muito pouca mobilidade é ruim (pode ser mate)
    if mobility <= 1 {
        safety_score -= 15;
    } else if mobility <= 3 {
        safety_score += 5; // Mobilidade controlada é ok
    }
    
    safety_score
}

/// Avalia atividade do rei no final
fn evaluate_king_activity(king_sq: u8, context: &MobilityContext) -> i32 {
    let mut activity_score = 0;
    
    // Mobilidade é importante no final
    let king_attacks = crate::moves::king::get_king_attacks_lookup(king_sq);
    let legal_squares = king_attacks & !context.our_pieces;
    let safe_squares = legal_squares & !context.enemy_attacked_squares;
    
    activity_score += safe_squares.count_ones() as i32 * 3;
    
    // Centralização no final
    if is_central_square(king_sq) {
        activity_score += 15;
    } else if is_extended_center(king_sq) {
        activity_score += 8;
    }
    
    // Proximidade com peões para suporte/bloqueio
    let our_pawns = context.board.pawns & context.our_pieces;
    let pawns_nearby = count_pieces_in_radius(king_sq, our_pawns, 2);
    activity_score += pawns_nearby * 2;
    
    // Oposição com rei inimigo (melhorado)
    let enemy_king = context.board.kings & context.enemy_pieces;
    if enemy_king != 0 {
        let enemy_king_sq = enemy_king.trailing_zeros() as u8;
        if has_opposition(king_sq, enemy_king_sq) {
            activity_score += 15;
        }
    }
    
    activity_score
}

/// Avalia escudo de peões
fn evaluate_pawn_shield(king_sq: u8, context: &MobilityContext) -> i32 {
    let mut shield_score = 0;
    let our_pawns = context.board.pawns & context.our_pieces;
    
    // Casas de proteção na frente do rei
    let protection_squares = get_pawn_shield_squares(king_sq, context.color);
    
    for &shield_sq in &protection_squares {
        let shield_bb = 1u64 << shield_sq;
        if (our_pawns & shield_bb) != 0 {
            shield_score += 8; // Bônus por peão protetor
        } else {
            shield_score -= 5; // Penalidade por buraco no escudo
        }
    }
    
    shield_score
}

/// Obtém casas do escudo de peões
fn get_pawn_shield_squares(king_sq: u8, color: Color) -> Vec<u8> {
    let file = king_sq % 8;
    let rank = king_sq / 8;
    let mut shield_squares = Vec::new();
    
    let shield_rank = if color == Color::White {
        if rank < 7 { rank + 1 } else { return shield_squares; }
    } else {
        if rank > 0 { rank - 1 } else { return shield_squares; }
    };
    
    // Três casas na frente do rei
    for shield_file in (file.saturating_sub(1))..=(file.saturating_add(1)).min(7) {
        shield_squares.push(shield_rank * 8 + shield_file);
    }
    
    shield_squares
}

/// Avalia distância de ameaças
fn evaluate_threat_distance(king_sq: u8, context: &MobilityContext) -> i32 {
    let mut threat_score = 0;
    
    // Distância de rainhas inimigas
    let enemy_queens = context.board.queens & context.enemy_pieces;
    if enemy_queens != 0 {
        let queen_squares = get_set_bits(enemy_queens);
        for &queen_sq in &queen_squares {
            let distance = eval_utils::calculate_square_distance(king_sq, queen_sq);
            if distance <= 4 {
                threat_score -= (5 - distance) * 4;
            }
        }
    }
    
    // Distância de torres inimigas
    let enemy_rooks = context.board.rooks & context.enemy_pieces;
    let rook_squares = get_set_bits(enemy_rooks);
    for &rook_sq in &rook_squares {
        let distance = eval_utils::calculate_square_distance(king_sq, rook_sq);
        if distance <= 3 {
            threat_score -= (4 - distance) * 2;
        }
    }
    
    threat_score
}

/// Conta peças em raio específico
fn count_pieces_in_radius(center_sq: u8, pieces: Bitboard, radius: i32) -> i32 {
    let piece_squares = get_set_bits(pieces);
    let mut count = 0;
    
    for &piece_sq in &piece_squares {
        if eval_utils::calculate_square_distance(center_sq, piece_sq) <= radius {
            count += 1;
        }
    }
    
    count
}

/// Verifica oposição entre reis
fn has_opposition(our_king: u8, enemy_king: u8) -> bool {
    let file_diff = ((our_king % 8) as i32 - (enemy_king % 8) as i32).abs();
    let rank_diff = ((our_king / 8) as i32 - (enemy_king / 8) as i32).abs();
    
    // Oposição direta
    (file_diff == 0 && rank_diff == 2) || (file_diff == 2 && rank_diff == 0) ||
    // Oposição diagonal
    (file_diff == 2 && rank_diff == 2)
}

