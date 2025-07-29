// Mobilidade e estratégia avançada da rainha com padrões táticos
use crate::types::{Color, Bitboard};
use super::{MobilityContext, MobilityResult, GamePhase, utils::*};
use crate::evaluation::utils as eval_utils;
use crate::moves::magic_bitboards::{get_queen_attacks_magic, get_rook_attacks_magic, get_bishop_attacks_magic};

/// Avaliação avançada de mobilidade da rainha com detecção tática
pub fn evaluate_queen_mobility_advanced(context: &MobilityContext) -> i32 {
    let mut total_score = 0;
    let queens = context.board.queens & context.our_pieces;

    if queens == 0 { return 0; }

    let queen_squares = get_set_bits(queens);

    for &queen_sq in &queen_squares {
        // Usa magic bitboards para performance máxima
        let total_attacks = get_queen_attacks_magic(queen_sq, context.all_pieces);
        let legal_squares = total_attacks & !context.our_pieces;
        let safe_squares = legal_squares & !context.enemy_attacked_squares;

        // === MOBILIDADE BÁSICA === 
        let mobility_count = legal_squares.count_ones() as i32;
        let safe_mobility = safe_squares.count_ones() as i32;
        
        // Peso ajustado por fase do jogo
        let mobility_weight = match context.phase {
            GamePhase::Opening => 1,    // Mobilidade menos importante na abertura
            GamePhase::MiddleGame => 2, // Mais importante no meio-jogo
            GamePhase::Endgame => 3,    // Crucial no endgame
        };
        
        total_score += mobility_count * mobility_weight;
        total_score += safe_mobility * (mobility_weight + 1);

        // === ANÁLISE POSICIONAL ===
        total_score += evaluate_queen_position(context, queen_sq);
        
        // === DETECÇÃO TÁTICA ===
        total_score += detect_queen_tactics(context, queen_sq, total_attacks);
        
        // === SEGURANÇA E DESENVOLVIMENTO ===
        total_score += evaluate_queen_safety(context, queen_sq);
        
        // === COORDENAÇÃO COM OUTRAS PEÇAS ===
        total_score += evaluate_queen_coordination(context, queen_sq, total_attacks);
    }

    // Limitação para evitar que rainha domine a avaliação
    total_score.clamp(-80, 80)
}

/// Avalia posicionamento estratégico da rainha
fn evaluate_queen_position(context: &MobilityContext, queen_sq: u8) -> i32 {
    let mut score = 0;
    
    // Centralização progressiva
    let centrality = get_centrality_score(queen_sq);
    score += match context.phase {
        GamePhase::Opening => -(centrality / 2), // Penalidade por centralizar cedo
        GamePhase::MiddleGame => centrality,     // Bônus moderado
        GamePhase::Endgame => centrality * 2,   // Muito importante no endgame
    };
    
    // Controle de diagonais longas
    let long_diagonals = 0x8142241818244281u64; // a1-h8 e h1-a8
    if (1u64 << queen_sq) & long_diagonals != 0 {
        score += match context.phase {
            GamePhase::Opening => 2,
            GamePhase::MiddleGame => 8,
            GamePhase::Endgame => 5,
        };
    }
    
    // Posição na 7ª/2ª fileira (penetração)
    let rank = queen_sq / 8;
    let is_penetrating_rank = (context.color == Color::White && rank == 6) || 
                             (context.color == Color::Black && rank == 1);
    
    if is_penetrating_rank {
        score += match context.phase {
            GamePhase::Opening => -5,  // Muito cedo
            GamePhase::MiddleGame => 15, // Excelente pressão
            GamePhase::Endgame => 20,   // Dominação
        };
    }
    
    score
}

/// Detecta padrões táticos da rainha
fn detect_queen_tactics(context: &MobilityContext, queen_sq: u8, attacks: Bitboard) -> i32 {
    let mut tactical_score = 0;
    let enemy_king = context.board.kings & context.enemy_pieces;
    
    if enemy_king == 0 { return 0; }
    
    let enemy_king_sq = enemy_king.trailing_zeros() as u8;
    
    // === XEQUES ===
    if (attacks & enemy_king) != 0 {
        tactical_score += match context.phase {
            GamePhase::Opening => 8,
            GamePhase::MiddleGame => 15,
            GamePhase::Endgame => 25,
        };
    }
    
    // === FORKS (ataca rei + outra peça) ===
    if (attacks & enemy_king) != 0 {
        let valuable_targets = (context.board.queens | context.board.rooks | 
                               context.board.bishops | context.board.knights) & context.enemy_pieces;
        let fork_count = (attacks & valuable_targets).count_ones();
        
        if fork_count > 0 {
            tactical_score += (fork_count as i32) * 40; // Fork é devastador!
        }
    }
    
    // === PINS E SKEWERS ===
    tactical_score += detect_queen_pins(context, queen_sq);
    
    // === ATAQUES MÚLTIPLOS ===
    let attacked_pieces = attacks & context.enemy_pieces;
    let attack_count = attacked_pieces.count_ones();
    
    if attack_count >= 3 {
        tactical_score += (attack_count as i32 - 2) * 8; // Pressão múltipla
    }
    
    // === CONTROLE DE CASAS CRÍTICAS ===
    let critical_squares = get_critical_squares_around_king(enemy_king_sq);
    let controlled_critical = (attacks & critical_squares).count_ones() as i32;
    tactical_score += controlled_critical * 6;
    
    tactical_score
}

/// Detecta pins e skewers da rainha
fn detect_queen_pins(context: &MobilityContext, queen_sq: u8) -> i32 {
    let mut pin_score = 0;
    let enemy_king = context.board.kings & context.enemy_pieces;
    
    if enemy_king == 0 { return 0; }
    
    let enemy_king_sq = enemy_king.trailing_zeros() as u8;
    
    // Verifica pins nas 8 direções da rainha
    let directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ];
    
    for &(dr, df) in &directions {
        pin_score += check_pin_in_direction(context, queen_sq, enemy_king_sq, dr, df);
    }
    
    pin_score
}

/// Verifica pin numa direção específica
fn check_pin_in_direction(context: &MobilityContext, queen_sq: u8, king_sq: u8, dr: i8, df: i8) -> i32 {
    let queen_rank = queen_sq / 8;
    let queen_file = queen_sq % 8;
    let king_rank = king_sq / 8;
    let king_file = king_sq % 8;
    
    // Verifica se rei está na direção correta
    let rank_diff = (king_rank as i8) - (queen_rank as i8);
    let file_diff = (king_file as i8) - (queen_file as i8);
    
    if rank_diff.signum() != dr || file_diff.signum() != df {
        return 0;
    }
    
    // Conta peças entre rainha e rei
    let mut pieces_between = 0;
    let mut enemy_piece_value = 0;
    
    let mut current_rank = queen_rank as i8 + dr;
    let mut current_file = queen_file as i8 + df;
    
    while current_rank != king_rank as i8 || current_file != king_file as i8 {
        if current_rank < 0 || current_rank >= 8 || current_file < 0 || current_file >= 8 {
            break;
        }
        
        let current_sq = (current_rank * 8 + current_file) as u8;
        let sq_bb = 1u64 << current_sq;
        
        if (context.all_pieces & sq_bb) != 0 {
            pieces_between += 1;
            
            // Se é peça inimiga, guarda o valor
            if (context.enemy_pieces & sq_bb) != 0 {
                enemy_piece_value = get_piece_value_on_square(context, current_sq);
            }
        }
        
        current_rank += dr;
        current_file += df;
    }
    
    // Pin detectado: exatamente 1 peça entre rainha e rei, e é peça inimiga
    if pieces_between == 1 && enemy_piece_value > 0 {
        return match enemy_piece_value {
            100 => 15,   // Peão pinado
            320 => 25,   // Cavalo pinado  
            330 => 25,   // Bispo pinado
            500 => 35,   // Torre pinada
            900 => 50,   // Rainha pinada (skewer)
            _ => 10,
        };
    }
    
    0
}

/// Avalia segurança da rainha
fn evaluate_queen_safety(context: &MobilityContext, queen_sq: u8) -> i32 {
    let mut safety_score = 0;
    
    // Penalidade se rainha está atacada
    if (1u64 << queen_sq) & context.enemy_attacked_squares != 0 {
        // Verifica por que tipo de peça está sendo atacada
        let attacked_by_pawn = is_attacked_by_piece_type(context, queen_sq, context.board.pawns);
        let attacked_by_minor = is_attacked_by_piece_type(context, queen_sq, 
                                                         context.board.knights | context.board.bishops);
        
        if attacked_by_pawn {
            safety_score -= 30; // Muito ruim ser atacada por peão
        } else if attacked_by_minor {
            safety_score -= 20; // Ruim ser atacada por peça menor
        } else {
            safety_score -= 10; // Atacada por peça igual ou maior
        }
    }
    
    // Desenvolvimento prematuro na abertura
    if context.phase == GamePhase::Opening {
        let back_rank = match context.color {
            Color::White => queen_sq <= 7,
            Color::Black => queen_sq >= 56,
        };
        
        if !back_rank {
            // Verifica se saiu muito cedo
            let moves_played = count_developed_pieces(context);
            if moves_played < 6 { // Menos de 6 peças desenvolvidas
                safety_score -= 25; // Penalidade severa
            }
        }
    }
    
    safety_score
}

/// Avalia coordenação da rainha com outras peças
fn evaluate_queen_coordination(context: &MobilityContext, queen_sq: u8, attacks: Bitboard) -> i32 {
    let mut coordination_score = 0;
    
    // Suporte de peças menores
    let our_minor_pieces = (context.board.knights | context.board.bishops) & context.our_pieces;
    let mut minor_bb = our_minor_pieces;
    
    while minor_bb != 0 {
        let piece_sq = minor_bb.trailing_zeros() as u8;
        minor_bb &= minor_bb - 1;
        
        // Verifica se peça menor protege a rainha
        if piece_attacks_square(context, piece_sq, queen_sq) {
            coordination_score += 8;
        }
        
        // Verifica se rainha e peça menor atacam o mesmo alvo
        let piece_attacks = get_piece_attacks(context, piece_sq);
        let shared_targets = (attacks & piece_attacks & context.enemy_pieces).count_ones();
        coordination_score += (shared_targets as i32) * 3;
    }
    
    // Coordenação com torres
    let our_rooks = context.board.rooks & context.our_pieces;
    if our_rooks != 0 {
        let rook_attacks = get_rook_attacks_magic(our_rooks.trailing_zeros() as u8, context.all_pieces);
        let shared_files_ranks = (attacks & rook_attacks).count_ones() as i32;
        coordination_score += shared_files_ranks * 2;
    }
    
    coordination_score
}

/// Funções auxiliares
fn get_centrality_score(square: u8) -> i32 {
    let rank = (square / 8) as i32;
    let file = (square % 8) as i32;
    let center_distance = ((rank - 3).abs() + (file - 3).abs()) as i32;
    (7 - center_distance).max(0)
}

fn get_critical_squares_around_king(king_sq: u8) -> Bitboard {
    let king_attacks = crate::moves::king::get_king_attacks_lookup(king_sq);
    king_attacks | (1u64 << king_sq)
}

fn get_piece_value_on_square(context: &MobilityContext, square: u8) -> i32 {
    let sq_bb = 1u64 << square;
    
    if context.board.pawns & sq_bb != 0 { 100 }
    else if context.board.knights & sq_bb != 0 { 320 }
    else if context.board.bishops & sq_bb != 0 { 330 }
    else if context.board.rooks & sq_bb != 0 { 500 }
    else if context.board.queens & sq_bb != 0 { 900 }
    else { 0 }
}

fn is_attacked_by_piece_type(context: &MobilityContext, square: u8, piece_type_bb: Bitboard) -> bool {
    // Implementação simplificada - pode ser melhorada
    (1u64 << square) & context.enemy_attacked_squares != 0 && 
    piece_type_bb & context.enemy_pieces != 0
}

fn count_developed_pieces(context: &MobilityContext) -> u32 {
    let back_rank = match context.color {
        Color::White => 0xFF,
        Color::Black => 0xFF00000000000000,
    };
    
    let minor_pieces = (context.board.knights | context.board.bishops) & context.our_pieces;
    let developed = minor_pieces & !back_rank;
    developed.count_ones()
}

fn piece_attacks_square(context: &MobilityContext, piece_sq: u8, target_sq: u8) -> bool {
    let attacks = get_piece_attacks(context, piece_sq);
    (attacks & (1u64 << target_sq)) != 0
}

fn get_piece_attacks(context: &MobilityContext, square: u8) -> Bitboard {
    let sq_bb = 1u64 << square;
    
    if context.board.knights & sq_bb != 0 {
        crate::moves::knight::get_knight_attacks_lookup(square)
    } else if context.board.bishops & sq_bb != 0 {
        get_bishop_attacks_magic(square, context.all_pieces)
    } else if context.board.rooks & sq_bb != 0 {
        get_rook_attacks_magic(square, context.all_pieces)
    } else if context.board.queens & sq_bb != 0 {
        get_queen_attacks_magic(square, context.all_pieces)
    } else {
        0
    }
}

