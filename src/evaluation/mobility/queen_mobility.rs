// Sistema de mobilidade e estratégia avançada da rainha - Versão revolucionária
use crate::types::{Color, Bitboard, PieceKind};
use super::{MobilityContext, GamePhase, utils::*};
use crate::evaluation::utils as eval_utils;

const QUEEN_MOBILITY_MULTIPLIER: i32 = 6;
const QUEEN_SAFETY_MULTIPLIER: i32 = 4;
const QUEEN_INFILTRATION_BONUS: i32 = 350;
const QUEEN_MATE_NETWORK_BONUS: i32 = 250;
const QUEEN_TARGET_PRIORITY_BONUS: i32 = 180;
const QUEEN_COORDINATION_BONUS: i32 = 120;
const QUEEN_EXPOSURE_PENALTY: i32 = 200;
const QUEEN_CRITICAL_CONTROL_BONUS: i32 = 300;
const QUEEN_TACTICAL_DOMINANCE_BONUS: i32 = 400;

/// Peça values for target prioritization
const PIECE_VALUES: [i32; 7] = [0, 100, 320, 330, 500, 900, 20000]; // None, P, N, B, R, Q, K

/// Avaliação revolucionária de mobilidade da rainha
pub fn evaluate_queen_mobility_advanced(context: &MobilityContext) -> i32 {
    let mut total_score = 0;
    let queens = context.board.queens & context.our_pieces;

    if queens == 0 { return 0; }

    let queen_squares = get_set_bits(queens);

    for &queen_sq in &queen_squares {
        // === ANÁLISE BÁSICA DE MOBILIDADE ===
        let queen_attacks = generate_queen_attacks(queen_sq, context.all_pieces);
        let legal_squares = queen_attacks & !context.our_pieces;
        let safe_squares = legal_squares & !context.enemy_attacked_squares;
        let dangerous_squares = legal_squares & context.enemy_attacked_squares;

        // Mobilidade básica com multiplicadores poderosos
        total_score += (legal_squares.count_ones() as i32) * QUEEN_MOBILITY_MULTIPLIER;
        total_score += (safe_squares.count_ones() as i32) * QUEEN_SAFETY_MULTIPLIER;
        total_score -= (dangerous_squares.count_ones() as i32) * 3; // Penalidade por quadrados perigosos

        // === SISTEMA DE PRIORIZAÇÃO DE ALVOS POR VALOR ===
        total_score += evaluate_target_prioritization(queen_sq, context, &queen_attacks);

        // === DETECÇÃO DE MATES EM REDE (Q+N, Q+B patterns) ===
        total_score += detect_mate_networks(queen_sq, context);

        // === ANÁLISE DE INFILTRAÇÃO EM POSIÇÕES INIMIGAS ===
        total_score += analyze_enemy_infiltration(queen_sq, context);

        // === COORDENAÇÃO COM OUTRAS PEÇAS PESADAS ===
        total_score += evaluate_heavy_piece_coordination(queen_sq, context);

        // === PENALIDADES DINÂMICAS POR EXPOSIÇÃO PREMATURA ===
        total_score -= calculate_exposure_penalties(queen_sq, context);

        // === BÔNUS POR CONTROLE DE DIAGONAIS/COLUNAS CRÍTICAS ===
        total_score += evaluate_critical_control(queen_sq, context, &queen_attacks);

        // === DOMINÂNCIA TÁTICA ABSOLUTA ===
        total_score += evaluate_tactical_dominance(queen_sq, context, &queen_attacks);

        // === PRESSÃO SOBRE O REI INIMIGO ===
        total_score += evaluate_king_pressure(queen_sq, context, &queen_attacks);

        // === CONTROLE DE QUADRADOS CHAVE ===
        total_score += evaluate_key_square_control(queen_sq, context, &queen_attacks);
    }

    total_score
}

/// Gera ataques completos da rainha (bispo + torre)
fn generate_queen_attacks(square: u8, blockers: Bitboard) -> Bitboard {
    crate::moves::sliding::get_bishop_attacks(square, blockers) |
    crate::moves::sliding::get_rook_attacks(square, blockers)
}

/// Sistema de priorização de alvos por valor da peça
fn evaluate_target_prioritization(queen_sq: u8, context: &MobilityContext, queen_attacks: &Bitboard) -> i32 {
    let mut score = 0;
    let enemy_pieces = *queen_attacks & context.enemy_pieces;
    
    let mut bb = enemy_pieces;
    let mut high_value_targets = 0;
    
    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        
        // Determina o tipo da peça atacada
        if (context.board.queens & (1u64 << sq)) != 0 {
            score += PIECE_VALUES[5] / 10; // Queen
            high_value_targets += 3;
        } else if (context.board.rooks & (1u64 << sq)) != 0 {
            score += PIECE_VALUES[4] / 8; // Rook
            high_value_targets += 2;
        } else if (context.board.bishops & (1u64 << sq)) != 0 || (context.board.knights & (1u64 << sq)) != 0 {
            score += PIECE_VALUES[2] / 6; // Minor piece
            high_value_targets += 1;
        } else if (context.board.pawns & (1u64 << sq)) != 0 {
            score += PIECE_VALUES[1] / 4; // Pawn
        }
    }
    
    // Bônus massivo por atacar múltiplos alvos valiosos
    if high_value_targets >= 2 {
        score += QUEEN_TARGET_PRIORITY_BONUS * high_value_targets;
    }
    
    score
}

/// Detecção de padrões de mate em rede com cavalos e bispos
fn detect_mate_networks(queen_sq: u8, context: &MobilityContext) -> i32 {
    let mut score = 0;
    let enemy_king = context.board.kings & context.enemy_pieces;
    
    if enemy_king == 0 { return 0; }
    
    let king_sq = enemy_king.trailing_zeros() as u8;
    let distance_to_king = eval_utils::calculate_square_distance(queen_sq, king_sq);
    
    if distance_to_king > 4 { return 0; }
    
    let our_knights = context.board.knights & context.our_pieces;
    let our_bishops = context.board.bishops & context.our_pieces;
    
    // Q+N mate networks
    if our_knights != 0 {
        let mut knights_bb = our_knights;
        while knights_bb != 0 {
            let knight_sq = knights_bb.trailing_zeros() as u8;
            knights_bb &= knights_bb - 1;
            
            let knight_distance = eval_utils::calculate_square_distance(knight_sq, king_sq);
            if knight_distance <= 3 {
                // Verifica se rainha e cavalo controlam quadrados complementares ao redor do rei
                let knight_attacks = crate::moves::knight::get_knight_attacks_lookup(knight_sq);
                let queen_attacks = generate_queen_attacks(queen_sq, context.all_pieces);
                
                let combined_control = knight_attacks | queen_attacks;
                let king_area = get_king_area(king_sq);
                let controlled_king_area = (combined_control & king_area).count_ones();
                
                if controlled_king_area >= 6 {
                    score += QUEEN_MATE_NETWORK_BONUS;
                }
            }
        }
    }
    
    // Q+B mate networks
    if our_bishops != 0 {
        let mut bishops_bb = our_bishops;
        while bishops_bb != 0 {
            let bishop_sq = bishops_bb.trailing_zeros() as u8;
            bishops_bb &= bishops_bb - 1;
            
            let bishop_distance = eval_utils::calculate_square_distance(bishop_sq, king_sq);
            if bishop_distance <= 4 {
                let bishop_attacks = crate::moves::sliding::get_bishop_attacks(bishop_sq, context.all_pieces);
                let queen_attacks = generate_queen_attacks(queen_sq, context.all_pieces);
                
                // Verifica controle de diagonais longas convergindo no rei
                if (bishop_attacks & queen_attacks & get_king_area(king_sq)).count_ones() >= 3 {
                    score += QUEEN_MATE_NETWORK_BONUS * 2 / 3;
                }
            }
        }
    }
    
    score
}

/// Análise de infiltração em território inimigo
fn analyze_enemy_infiltration(queen_sq: u8, context: &MobilityContext) -> i32 {
    let rank = queen_sq / 8;
    let enemy_territory = if context.color == Color::White {
        rank >= 5 // 6th, 7th, 8th ranks
    } else {
        rank <= 2 // 3rd, 2nd, 1st ranks
    };
    
    if !enemy_territory { return 0; }
    
    let mut infiltration_score = QUEEN_INFILTRATION_BONUS;
    
    // Bônus extra por profundidade da infiltração
    let depth = if context.color == Color::White {
        rank - 4
    } else {
        3 - rank
    };
    
    infiltration_score += (depth as i32) * 100;
    
    // Bônus por ter suporte de outras peças na infiltração
    let queen_attacks = generate_queen_attacks(queen_sq, context.all_pieces);
    let supported_squares = queen_attacks & context.our_attacked_squares;
    infiltration_score += (supported_squares.count_ones() as i32) * 15;
    
    // Penalidade se a rainha está isolada em território inimigo
    let queen_area = get_extended_area(queen_sq, 2);
    let friendly_support = (queen_area & context.our_pieces).count_ones();
    if friendly_support == 1 { // Apenas a própria rainha
        infiltration_score -= 150;
    }
    
    infiltration_score
}

/// Coordenação com outras peças pesadas (torres e rainha)
fn evaluate_heavy_piece_coordination(queen_sq: u8, context: &MobilityContext) -> i32 {
    let mut score = 0;
    let our_rooks = context.board.rooks & context.our_pieces;
    let our_other_queens = (context.board.queens & context.our_pieces) & !(1u64 << queen_sq);
    
    // Coordenação com torres
    if our_rooks != 0 {
        let mut rooks_bb = our_rooks;
        while rooks_bb != 0 {
            let rook_sq = rooks_bb.trailing_zeros() as u8;
            rooks_bb &= rooks_bb - 1;
            
            // Verifica se estão na mesma linha ou coluna
            if queen_sq / 8 == rook_sq / 8 || queen_sq % 8 == rook_sq % 8 {
                score += QUEEN_COORDINATION_BONUS;
                
                // Bônus extra se controlam a mesma linha/coluna crítica
                let queen_attacks = generate_queen_attacks(queen_sq, context.all_pieces);
                let rook_attacks = crate::moves::sliding::get_rook_attacks(rook_sq, context.all_pieces);
                let shared_control = queen_attacks & rook_attacks & context.enemy_pieces;
                score += (shared_control.count_ones() as i32) * 30;
            }
        }
    }
    
    // Coordenação com outras rainhas (caso raro mas importante)
    if our_other_queens != 0 {
        let mut queens_bb = our_other_queens;
        while queens_bb != 0 {
            let other_queen_sq = queens_bb.trailing_zeros() as u8;
            queens_bb &= queens_bb - 1;
            
            let distance = eval_utils::calculate_square_distance(queen_sq, other_queen_sq);
            if distance <= 4 {
                score += QUEEN_COORDINATION_BONUS * 2; // Dupla de rainhas é devastadora
            }
        }
    }
    
    score
}

/// Calcula penalidades dinâmicas por exposição prematura
fn calculate_exposure_penalties(queen_sq: u8, context: &MobilityContext) -> i32 {
    let mut penalty = 0;
    
    // Penalidade severa por desenvolvimento prematuro na abertura
    if context.phase == GamePhase::Opening {
        let back_rank = if context.color == Color::White {
            queen_sq >= 0 && queen_sq <= 7
        } else {
            queen_sq >= 56 && queen_sq <= 63
        };
        
        if !back_rank {
            penalty += QUEEN_EXPOSURE_PENALTY;
            
            // Penalidade extra se peças menores ainda não foram desenvolvidas
            let undeveloped_knights = count_undeveloped_pieces(context, PieceKind::Knight);
            let undeveloped_bishops = count_undeveloped_pieces(context, PieceKind::Bishop);
            penalty += (undeveloped_knights + undeveloped_bishops) * 50;
        }
    }
    
    // Penalidade por estar sob ataque sem proteção adequada
    if context.enemy_attacked_squares & (1u64 << queen_sq) != 0 {
        let defenders = count_defenders(queen_sq, context);
        let attackers = count_attackers(queen_sq, context);
        
        if attackers > defenders {
            penalty += QUEEN_EXPOSURE_PENALTY / 2;
        }
    }
    
    // Penalidade por estar muito avançada sem suporte
    let queen_area = get_extended_area(queen_sq, 2);
    let friendly_support = (queen_area & context.our_pieces).count_ones();
    if friendly_support <= 2 && context.phase != GamePhase::Endgame {
        penalty += 100;
    }
    
    penalty
}

/// Avalia controle de diagonais e colunas críticas
fn evaluate_critical_control(queen_sq: u8, context: &MobilityContext, queen_attacks: &Bitboard) -> i32 {
    let mut score = 0;
    
    // Identifica linhas e colunas críticas
    let critical_files = get_critical_files(context);
    let critical_ranks = get_critical_ranks(context);
    let critical_diagonals = get_critical_diagonals(context);
    
    let queen_file = get_file_mask(queen_sq % 8);
    let queen_rank = 0xFFu64 << ((queen_sq / 8) * 8);
    
    // Bônus por controlar arquivo crítico
    if (critical_files & queen_file) != 0 {
        let file_control = count_file_control(queen_sq % 8, context);
        score += QUEEN_CRITICAL_CONTROL_BONUS + file_control * 20;
    }
    
    // Bônus por controlar linha crítica
    if (critical_ranks & queen_rank) != 0 {
        let rank_control = count_rank_control(queen_sq / 8, context);
        score += QUEEN_CRITICAL_CONTROL_BONUS + rank_control * 20;
    }
    
    // Bônus por controlar diagonais críticas
    let diagonal_control = (*queen_attacks & critical_diagonals).count_ones() as i32;
    score += diagonal_control * 40;
    
    score
}

/// Avalia dominância tática absoluta da rainha
fn evaluate_tactical_dominance(queen_sq: u8, context: &MobilityContext, queen_attacks: &Bitboard) -> i32 {
    let mut score = 0;
    
    // Conta quantas peças inimigas estão sob ataque direto
    let attacked_pieces = *queen_attacks & context.enemy_pieces;
    let pieces_under_attack = attacked_pieces.count_ones() as i32;
    
    if pieces_under_attack >= 3 {
        score += QUEEN_TACTICAL_DOMINANCE_BONUS;
        
        // Bônus extra por cada peça adicional atacada
        score += (pieces_under_attack - 3) * 100;
    }
    
    // Verifica se a rainha está criando ameaças múltiplas irresistíveis
    let valuable_targets = count_valuable_targets_under_attack(context, queen_attacks);
    if valuable_targets >= 2 {
        score += QUEEN_TACTICAL_DOMINANCE_BONUS / 2;
    }
    
    // Bônus por controlar quadrados de fuga do rei inimigo
    let enemy_king = context.board.kings & context.enemy_pieces;
    if enemy_king != 0 {
        let king_sq = enemy_king.trailing_zeros() as u8;
        let king_escape_squares = get_king_area(king_sq) & !context.enemy_pieces;
        let controlled_escapes = (*queen_attacks & king_escape_squares).count_ones() as i32;
        score += controlled_escapes * 60;
    }
    
    score
}

/// Avalia pressão sobre o rei inimigo
fn evaluate_king_pressure(queen_sq: u8, context: &MobilityContext, queen_attacks: &Bitboard) -> i32 {
    let enemy_king = context.board.kings & context.enemy_pieces;
    if enemy_king == 0 { return 0; }
    
    let king_sq = enemy_king.trailing_zeros() as u8;
    let distance = eval_utils::calculate_square_distance(queen_sq, king_sq);
    
    let mut pressure_score = 0;
    
    // Pressão baseada na distância
    if distance <= 3 {
        pressure_score += (4 - distance as i32) * 80;
    }
    
    // Bônus por atacar diretamente o rei
    if (*queen_attacks & enemy_king) != 0 {
        pressure_score += 200;
    }
    
    // Bônus por controlar quadrados ao redor do rei
    let king_area = get_king_area(king_sq);
    let controlled_king_area = (*queen_attacks & king_area).count_ones() as i32;
    pressure_score += controlled_king_area * 25;
    
    // Bônus extra se o rei inimigo não tem castling
    if !can_castle(context, !context.color) {
        pressure_score += 100;
    }
    
    pressure_score
}

/// Avalia controle de quadrados chave estratégicos
fn evaluate_key_square_control(queen_sq: u8, context: &MobilityContext, queen_attacks: &Bitboard) -> i32 {
    let mut score = 0;
    
    // Identifica quadrados chave baseados na estrutura de peões
    let key_squares = identify_key_squares(context);
    let controlled_key_squares = (*queen_attacks & key_squares).count_ones() as i32;
    score += controlled_key_squares * 45;
    
    // Bônus por controlar quadrados de invasão
    let invasion_squares = get_invasion_squares(context);
    let controlled_invasion = (*queen_attacks & invasion_squares).count_ones() as i32;
    score += controlled_invasion * 35;
    
    // Bônus por bloquear avanços de peões inimigos
    let blocked_pawns = count_blocked_enemy_pawns(context, queen_attacks);
    score += blocked_pawns * 30;
    
    score
}

// === FUNÇÕES AUXILIARES ===

fn get_king_area(king_sq: u8) -> Bitboard {
    let file = king_sq % 8;
    let rank = king_sq / 8;
    
    let mut area = 0u64;
    
    for dr in -1i8..=1 {
        for df in -1i8..=1 {
            let new_rank = rank as i8 + dr;
            let new_file = file as i8 + df;
            
            if new_rank >= 0 && new_rank < 8 && new_file >= 0 && new_file < 8 {
                area |= 1u64 << (new_rank * 8 + new_file);
            }
        }
    }
    
    area
}

fn get_extended_area(square: u8, radius: u8) -> Bitboard {
    let file = square % 8;
    let rank = square / 8;
    let mut area = 0u64;
    
    for dr in -(radius as i8)..=(radius as i8) {
        for df in -(radius as i8)..=(radius as i8) {
            let new_rank = rank as i8 + dr;
            let new_file = file as i8 + df;
            
            if new_rank >= 0 && new_rank < 8 && new_file >= 0 && new_file < 8 {
                area |= 1u64 << (new_rank * 8 + new_file);
            }
        }
    }
    
    area
}

fn count_undeveloped_pieces(context: &MobilityContext, piece_type: PieceKind) -> i32 {
    let pieces = match piece_type {
        PieceKind::Knight => context.board.knights & context.our_pieces,
        PieceKind::Bishop => context.board.bishops & context.our_pieces,
        _ => return 0,
    };
    
    let back_rank = if context.color == Color::White {
        0xFFu64 // 1st rank
    } else {
        0xFF00000000000000u64 // 8th rank
    };
    
    (pieces & back_rank).count_ones() as i32
}

fn count_defenders(square: u8, context: &MobilityContext) -> i32 {
    // Implementação simplificada - conta peças amigas que podem defender
    let defenders = context.our_attacked_squares & (1u64 << square);
    if defenders != 0 { 1 } else { 0 }
}

fn count_attackers(square: u8, context: &MobilityContext) -> i32 {
    // Implementação simplificada - conta se está sob ataque
    let attackers = context.enemy_attacked_squares & (1u64 << square);
    if attackers != 0 { 1 } else { 0 }
}

fn get_critical_files(context: &MobilityContext) -> Bitboard {
    // Arquivos com reis, torres, ou sem peões
    let enemy_king_bb = context.board.kings & context.enemy_pieces;
    if enemy_king_bb == 0 { return 0; }
    let king_file = get_file_mask(enemy_king_bb.trailing_zeros() as u8 % 8);
    let rook_files = get_rook_files(context);
    king_file | rook_files
}

fn get_critical_ranks(context: &MobilityContext) -> Bitboard {
    // 7ª e 8ª fileiras para brancos, 1ª e 2ª para pretos
    if context.color == Color::White {
        0xFF000000000000u64 | 0xFF00000000000000u64
    } else {
        0xFFu64 | 0xFF00u64
    }
}

fn get_critical_diagonals(context: &MobilityContext) -> Bitboard {
    // Diagonais longas e diagonais que passam pelo rei inimigo
    let enemy_king = context.board.kings & context.enemy_pieces;
    if enemy_king == 0 { return 0; }
    
    let king_sq = enemy_king.trailing_zeros() as u8;
    crate::moves::sliding::get_bishop_attacks(king_sq, 0) // Diagonal completa
}

fn count_file_control(file: u8, context: &MobilityContext) -> i32 {
    let file_mask = get_file_mask(file);
    (context.our_attacked_squares & file_mask).count_ones() as i32
}

fn count_rank_control(rank: u8, context: &MobilityContext) -> i32 {
    let rank_mask = 0xFFu64 << (rank * 8);
    (context.our_attacked_squares & rank_mask).count_ones() as i32
}

fn get_file_mask(file: u8) -> Bitboard {
    0x0101010101010101u64 << file
}

fn get_rook_files(context: &MobilityContext) -> Bitboard {
    let our_rooks = context.board.rooks & context.our_pieces;
    let mut files = 0u64;
    
    let mut bb = our_rooks;
    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        files |= get_file_mask(sq % 8);
    }
    
    files
}

fn count_valuable_targets_under_attack(context: &MobilityContext, queen_attacks: &Bitboard) -> i32 {
    let valuable_pieces = (context.board.queens | context.board.rooks | 
                          context.board.bishops | context.board.knights) & context.enemy_pieces;
    (*queen_attacks & valuable_pieces).count_ones() as i32
}

fn can_castle(_context: &MobilityContext, _color: Color) -> bool {
    // Implementação simplificada - assumir que pode não ter castling
    false
}

fn identify_key_squares(context: &MobilityContext) -> Bitboard {
    // Quadrados fracos na estrutura de peões inimiga
    let enemy_pawns = context.board.pawns & context.enemy_pieces;
    let mut key_squares = 0u64;
    
    // Simplificação: quadrados na 6ª fileira para brancos, 3ª para pretos
    if context.color == Color::White {
        key_squares |= 0xFF0000000000u64; // 6th rank
    } else {
        key_squares |= 0xFF0000u64; // 3rd rank
    }
    
    // Remove quadrados ocupados por peões inimigos
    key_squares & !enemy_pawns
}

fn get_invasion_squares(context: &MobilityContext) -> Bitboard {
    // Quadrados no território inimigo
    if context.color == Color::White {
        0xFFFF000000000000u64 // 7th and 8th ranks
    } else {
        0xFFFFu64 // 1st and 2nd ranks
    }
}

fn count_blocked_enemy_pawns(context: &MobilityContext, queen_attacks: &Bitboard) -> i32 {
    let enemy_pawns = context.board.pawns & context.enemy_pieces;
    let mut blocked = 0;
    
    let mut bb = enemy_pawns;
    while bb != 0 {
        let pawn_sq = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        
        // Verifica se a rainha bloqueia o avanço do peão
        let advance_sq = if context.color == Color::White {
            if pawn_sq >= 8 { pawn_sq - 8 } else { continue; }
        } else {
            if pawn_sq <= 55 { pawn_sq + 8 } else { continue; }
        };
        
        if (*queen_attacks & (1u64 << advance_sq)) != 0 {
            blocked += 1;
        }
    }
    
    blocked
}

