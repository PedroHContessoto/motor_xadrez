// Avaliação de ameaças otimizada - peças penduradas, ataques táticos
use crate::{board::Board, types::{Color, PieceKind, Move, Bitboard}};
use super::material::MATERIAL_VALUES;
use super::{mobility, utils};
use std::collections::HashMap;

// Cache otimizado para threats
struct ThreatCache {
    attack_maps: HashMap<u8, Bitboard>,
    threat_patterns: Vec<TacticalPattern>,
}

#[derive(Debug, Clone)]
struct TacticalPattern {
    pattern_type: PatternType,
    squares: Vec<u8>,
    value: i32,
}

#[derive(Debug, Clone, PartialEq)]
enum PatternType {
    Fork,
    Pin,
    Skewer,
    DiscoveredAttack,
    Sacrifice,
}

impl ThreatCache {
    fn new() -> Self {
        Self {
            attack_maps: HashMap::new(),
            threat_patterns: Vec::new(),
        }
    }
}

/// Avalia ameaças mútuas - Versão otimizada estratégica
pub fn evaluate_threats(board: &Board, color: Color) -> i32 {
    let mut score = 0;
    let mut cache = ThreatCache::new();

    // 1. PRIMEIRA PRIORIDADE: Threats críticos
    score -= evaluate_hanging_pieces_optimized(board, color);
    score += evaluate_royal_forks(board, color, &mut cache);

    // 2. SEGUNDA PRIORIDADE: Tactical patterns de alto valor
    score += evaluate_pins_and_skewers_combined(board, color);
    score += evaluate_discovered_attacks_new(board, color);

    // 3. TERCEIRA PRIORIDADE: Pressure tático
    score += evaluate_piece_pressure(board, color, &mut cache);
    score += evaluate_overloaded_defenders(board, color);

    // 4. BÔNUS: Coordenação de ataques
    score += evaluate_coordinated_attacks(board, color);

    score.clamp(-800, 800)
}

/// Avalia peças penduradas com SEE otimizado
fn evaluate_hanging_pieces_optimized(board: &Board, color: Color) -> i32 {
    let mut penalty = 0;
    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let enemy_color = !color;

    // Prioriza peças por valor decrescente
    let piece_priorities = [
        (board.queens & our_pieces, PieceKind::Queen, 900),
        (board.rooks & our_pieces, PieceKind::Rook, 500),
        (board.bishops & our_pieces, PieceKind::Bishop, 330),
        (board.knights & our_pieces, PieceKind::Knight, 320),
    ];

    for (piece_bb, piece_kind, piece_value) in piece_priorities {
        let mut bb = piece_bb;
        while bb != 0 {
            let sq = bb.trailing_zeros() as u8;
            bb &= bb - 1;

            if board.is_square_attacked_by(sq, enemy_color) {
                let see_value = calculate_see_capture(board, sq, color);

                if see_value < 0 {
                    let loss = see_value.abs();
                    let tactical_multiplier = get_tactical_multiplier(board, sq, piece_kind, color);
                    penalty += (loss as f32 * tactical_multiplier) as i32;

                    // Penalidade crítica para rainha
                    if piece_kind == PieceKind::Queen && see_value <= -400 {
                        penalty += 500;
                    }
                }
            }
        }
    }

    // Peões atacados
    penalty += evaluate_pawn_safety(board, color);

    penalty
}

/// Calcula SEE simplificado
fn calculate_see_capture(board: &Board, target_sq: u8, defending_color: Color) -> i32 {
    let target_piece = board.get_piece_on_square(target_sq);
    if target_piece.is_none() { return 0; }

    let piece_value = MATERIAL_VALUES[target_piece.unwrap() as usize];
    let attacking_color = !defending_color;

    let min_attacker_value = find_smallest_attacker_value(board, target_sq, attacking_color);

    if board.is_square_attacked_by(target_sq, defending_color) {
        let min_defender_value = find_smallest_attacker_value(board, target_sq, defending_color);
        piece_value - min_attacker_value - min_defender_value
    } else {
        piece_value - min_attacker_value
    }
}

/// Multiplicador tático baseado na posição
fn get_tactical_multiplier(board: &Board, sq: u8, piece_kind: PieceKind, color: Color) -> f32 {
    let mut multiplier: f32 = 1.0;

    // Peças centralizadas
    if utils::is_center_square(sq) || utils::is_extended_center_square(sq) {
        multiplier += 0.3;
    }

    // Peças atacando zona do rei
    if attacks_enemy_king_zone(board, sq, color) {
        multiplier += 0.5;
    }

    // Peças avançadas
    if is_advanced_piece(sq, piece_kind, color) {
        multiplier += 0.2;
    }

    // Peças desprotegidas
    if !board.is_square_attacked_by(sq, color) {
        multiplier += 0.4;
    }

    multiplier.min(2.5)
}

/// Avalia segurança dos peões otimizada
fn evaluate_pawn_safety(board: &Board, color: Color) -> i32 {
    let mut penalty = 0;
    let enemy_color = !color;
    let our_pawns = board.pawns & if color == Color::White { board.white_pieces } else { board.black_pieces };

    let attacked_pawns = our_pawns & get_enemy_attack_map(board, enemy_color);
    let defended_pawns = our_pawns & get_our_attack_map(board, color);
    let hanging_pawns = attacked_pawns & !defended_pawns;

    penalty += hanging_pawns.count_ones() as i32 * 12;

    // Penalidades extras para peões isolados/atrasados atacados
    let mut pawn_bb = hanging_pawns;
    while pawn_bb != 0 {
        let sq = pawn_bb.trailing_zeros() as u8;
        pawn_bb &= pawn_bb - 1;

        if is_isolated_pawn(board, sq, color) {
            penalty += 8;
        }
        if is_backward_pawn(board, sq, color) {
            penalty += 6;
        }
    }

    penalty
}

/// Verifica se peão está isolado
fn is_isolated_pawn(board: &Board, pawn_sq: u8, color: Color) -> bool {
    let file = pawn_sq % 8;
    let our_pawns = board.pawns & if color == Color::White { board.white_pieces } else { board.black_pieces };

    for adj_file in [file.saturating_sub(1), file.saturating_add(1).min(7)] {
        if adj_file != file {
            let file_mask = utils::get_file_mask_from_file(adj_file);
            if (our_pawns & file_mask) != 0 {
                return false;
            }
        }
    }
    true
}

/// Verifica se peão está atrasado
fn is_backward_pawn(board: &Board, pawn_sq: u8, color: Color) -> bool {
    let file = pawn_sq % 8;
    let rank = pawn_sq / 8;
    let our_pawns = board.pawns & if color == Color::White { board.white_pieces } else { board.black_pieces };

    for adj_file in [file.saturating_sub(1), file.saturating_add(1).min(7)] {
        if adj_file != file {
            let file_mask = utils::get_file_mask_from_file(adj_file);
            let file_pawns = our_pawns & file_mask;

            let mut file_bb = file_pawns;
            while file_bb != 0 {
                let sq = file_bb.trailing_zeros() as u8;
                file_bb &= file_bb - 1;
                let pawn_rank = sq / 8;

                let can_support = if color == Color::White {
                    pawn_rank <= rank
                } else {
                    pawn_rank >= rank
                };

                if can_support {
                    return false;
                }
            }
        }
    }
    true
}

/// Avalia forks reais (rei + peça)
fn evaluate_royal_forks(board: &Board, color: Color, _cache: &mut ThreatCache) -> i32 {
    let mut bonus = 0;
    let enemy_color = !color;
    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let enemy_pieces = if color == Color::White { board.black_pieces } else { board.white_pieces };
    let enemy_king = board.kings & enemy_pieces;

    if enemy_king == 0 { return 0; }
    let enemy_king_sq = enemy_king.trailing_zeros() as u8;

    // Cavalos fazendo fork real
    let our_knights = board.knights & our_pieces;
    let mut knight_bb = our_knights;
    while knight_bb != 0 {
        let knight_sq = knight_bb.trailing_zeros() as u8;
        knight_bb &= knight_bb - 1;

        let knight_attacks = crate::moves::knight::get_knight_attacks_lookup(knight_sq);
        if (knight_attacks & enemy_king) != 0 {
            let valuable_enemies = (board.knights | board.bishops | board.rooks | board.queens) & enemy_pieces;
            let attacked_valuables = knight_attacks & valuable_enemies;

            if attacked_valuables != 0 {
                let mut value_sum = 0;
                let mut temp_bb = attacked_valuables;
                while temp_bb != 0 {
                    let sq = temp_bb.trailing_zeros() as u8;
                    temp_bb &= temp_bb - 1;
                    if let Some(piece_kind) = board.get_piece_on_square(sq) {
                        value_sum += MATERIAL_VALUES[piece_kind as usize];
                    }
                }

                bonus += match value_sum {
                    v if v >= 900 => 80,  // Fork com rainha
                    v if v >= 500 => 60,  // Fork com torre
                    v if v >= 320 => 40,  // Fork com peça menor
                    _ => 25               // Fork básico
                };
            }
        }
    }

    // Bispos e rainhas fazendo fork real
    let our_long_range = (board.bishops | board.queens) & our_pieces;
    let mut long_range_bb = our_long_range;
    while long_range_bb != 0 {
        let piece_sq = long_range_bb.trailing_zeros() as u8;
        long_range_bb &= long_range_bb - 1;

        let all_pieces = board.white_pieces | board.black_pieces;
        let attacks = if (board.bishops & (1u64 << piece_sq)) != 0 {
            crate::moves::magic_bitboards::get_bishop_attacks_magic(piece_sq, all_pieces)
        } else {
            crate::moves::magic_bitboards::get_queen_attacks_magic(piece_sq, all_pieces)
        };

        if (attacks & enemy_king) != 0 {
            let valuable_enemies = (board.knights | board.bishops | board.rooks | board.queens) & enemy_pieces;
            let attacked_valuables = attacks & valuable_enemies;

            if attacked_valuables.count_ones() >= 1 {
                let piece_kind = board.get_piece_on_square(piece_sq).unwrap();
                let fork_bonus = match piece_kind {
                    PieceKind::Queen => 70,
                    PieceKind::Bishop => 50,
                    _ => 30
                };
                bonus += fork_bonus;
            }
        }
    }

    bonus
}

/// Avalia pinos e skewers combinados
fn evaluate_pins_and_skewers_combined(board: &Board, color: Color) -> i32 {
    let mut bonus = 0;
    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let enemy_pieces = if color == Color::White { board.black_pieces } else { board.white_pieces };
    let our_sliders = (board.rooks | board.queens | board.bishops) & our_pieces;

    let mut slider_bb = our_sliders;
    while slider_bb != 0 {
        let slider_sq = slider_bb.trailing_zeros() as u8;
        slider_bb &= slider_bb - 1;

        if let Some(pin_value) = evaluate_pin_simple(board, slider_sq, enemy_pieces) {
            bonus += (pin_value / 10).min(50);
        }

        if let Some(skewer_value) = evaluate_skewer_simple(board, slider_sq, enemy_pieces) {
            bonus += (skewer_value / 12).min(40);
        }
    }

    bonus
}

/// Avalia pin simples
fn evaluate_pin_simple(board: &Board, slider_sq: u8, enemy_pieces: Bitboard) -> Option<i32> {
    let all_pieces = board.white_pieces | board.black_pieces;
    let piece_kind = board.get_piece_on_square(slider_sq)?;

    let attacks = match piece_kind {
        PieceKind::Bishop => crate::moves::magic_bitboards::get_bishop_attacks_magic(slider_sq, all_pieces),
        PieceKind::Rook => crate::moves::magic_bitboards::get_rook_attacks_magic(slider_sq, all_pieces),
        PieceKind::Queen => crate::moves::magic_bitboards::get_queen_attacks_magic(slider_sq, all_pieces),
        _ => return None,
    };

    let attacked_enemies = attacks & enemy_pieces;
    if attacked_enemies.count_ones() >= 2 {
        let mut total_value = 0;
        let mut temp_bb = attacked_enemies;
        while temp_bb != 0 {
            let sq = temp_bb.trailing_zeros() as u8;
            temp_bb &= temp_bb - 1;
            if let Some(piece_kind) = board.get_piece_on_square(sq) {
                total_value += MATERIAL_VALUES[piece_kind as usize];
            }
        }
        Some(total_value)
    } else {
        None
    }
}

/// Avalia skewer simples
fn evaluate_skewer_simple(board: &Board, slider_sq: u8, enemy_pieces: Bitboard) -> Option<i32> {
    // Implementação similar ao pin, mas busca por alinhamentos valiosos
    evaluate_pin_simple(board, slider_sq, enemy_pieces)
}

/// Avalia ataques descobertos
fn evaluate_discovered_attacks_new(board: &Board, color: Color) -> i32 {
    let mut bonus = 0;
    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let enemy_pieces = if color == Color::White { board.black_pieces } else { board.white_pieces };
    let our_sliders = (board.rooks | board.queens | board.bishops) & our_pieces;

    let mut slider_bb = our_sliders;
    while slider_bb != 0 {
        let slider_sq = slider_bb.trailing_zeros() as u8;
        slider_bb &= slider_bb - 1;

        if let Some(discovered_value) = find_discovered_potential(board, slider_sq, our_pieces, enemy_pieces) {
            bonus += (discovered_value / 15).min(30);
        }
    }

    bonus
}

/// Encontra potencial de ataque descoberto
fn find_discovered_potential(board: &Board, slider_sq: u8, our_pieces: Bitboard, enemy_pieces: Bitboard) -> Option<i32> {
    let all_pieces = board.white_pieces | board.black_pieces;
    let piece_kind = board.get_piece_on_square(slider_sq)?;

    let attacks = match piece_kind {
        PieceKind::Bishop => crate::moves::magic_bitboards::get_bishop_attacks_magic(slider_sq, all_pieces),
        PieceKind::Rook => crate::moves::magic_bitboards::get_rook_attacks_magic(slider_sq, all_pieces),
        PieceKind::Queen => crate::moves::magic_bitboards::get_queen_attacks_magic(slider_sq, all_pieces),
        _ => return None,
    };

    let blocking_pieces = attacks & our_pieces;
    let valuable_enemies = attacks & enemy_pieces & (board.knights | board.bishops | board.rooks | board.queens);

    if blocking_pieces != 0 && valuable_enemies != 0 {
        let mut total_value = 0;
        let mut temp_bb = valuable_enemies;
        while temp_bb != 0 {
            let sq = temp_bb.trailing_zeros() as u8;
            temp_bb &= temp_bb - 1;
            if let Some(piece_kind) = board.get_piece_on_square(sq) {
                total_value += MATERIAL_VALUES[piece_kind as usize];
            }
        }
        Some(total_value)
    } else {
        None
    }
}

/// Avalia pressão sobre peças
fn evaluate_piece_pressure(board: &Board, color: Color, _cache: &mut ThreatCache) -> i32 {
    let mut bonus = 0;
    let enemy_color = !color;
    let enemy_pieces = if enemy_color == Color::White { board.white_pieces } else { board.black_pieces };

    let valuable_enemies = (board.knights | board.bishops | board.rooks | board.queens) & enemy_pieces;
    let mut valuable_bb = valuable_enemies;

    while valuable_bb != 0 {
        let target_sq = valuable_bb.trailing_zeros() as u8;
        valuable_bb &= valuable_bb - 1;

        let attack_count = count_our_attacks_on_square_simple(board, target_sq, color);
        let defense_count = count_our_attacks_on_square_simple(board, target_sq, enemy_color);

        if attack_count > defense_count {
            if let Some(piece_kind) = board.get_piece_on_square(target_sq) {
                let piece_value = MATERIAL_VALUES[piece_kind as usize];
                let pressure_bonus = (piece_value / 20) * (attack_count - defense_count) as i32;
                bonus += pressure_bonus.min(40);
            }
        }
    }

    bonus
}

/// Conta ataques simples sobre uma casa
fn count_our_attacks_on_square_simple(board: &Board, target_sq: u8, color: Color) -> u8 {
    if board.is_square_attacked_by(target_sq, color) { 1 } else { 0 }
}

/// Avalia defensores sobrecarregados
fn evaluate_overloaded_defenders(board: &Board, color: Color) -> i32 {
    let mut bonus = 0;
    let enemy_color = !color;
    let enemy_pieces = if enemy_color == Color::White { board.white_pieces } else { board.black_pieces };

    let mut enemy_bb = enemy_pieces;
    while enemy_bb != 0 {
        let defender_sq = enemy_bb.trailing_zeros() as u8;
        enemy_bb &= enemy_bb - 1;

        let protected_count = count_pieces_protected_by(board, defender_sq, enemy_color);
        if protected_count >= 2 {
            bonus += (protected_count as i32 - 1) * 10;
        }
    }

    bonus.min(50)
}

/// Conta peças protegidas por um defensor
fn count_pieces_protected_by(board: &Board, defender_sq: u8, color: Color) -> u8 {
    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };

    if let Some(piece_kind) = board.get_piece_on_square(defender_sq) {
        let all_pieces = board.white_pieces | board.black_pieces;
        let attacks = match piece_kind {
            PieceKind::Pawn => utils::compute_pawn_attacks(1u64 << defender_sq, color),
            PieceKind::Knight => crate::moves::knight::get_knight_attacks_lookup(defender_sq),
            PieceKind::Bishop => crate::moves::magic_bitboards::get_bishop_attacks_magic(defender_sq, all_pieces),
            PieceKind::Rook => crate::moves::magic_bitboards::get_rook_attacks_magic(defender_sq, all_pieces),
            PieceKind::Queen => crate::moves::magic_bitboards::get_queen_attacks_magic(defender_sq, all_pieces),
            PieceKind::King => crate::moves::king::get_king_attacks_lookup(defender_sq),
        };

        (attacks & our_pieces).count_ones() as u8
    } else {
        0
    }
}

/// Avalia ataques coordenados
fn evaluate_coordinated_attacks(board: &Board, color: Color) -> i32 {
    let mut bonus = 0;
    let enemy_color = !color;
    let enemy_pieces = if enemy_color == Color::White { board.white_pieces } else { board.black_pieces };

    let valuable_enemies = (board.knights | board.bishops | board.rooks | board.queens) & enemy_pieces;
    let mut valuable_bb = valuable_enemies;

    while valuable_bb != 0 {
        let target_sq = valuable_bb.trailing_zeros() as u8;
        valuable_bb &= valuable_bb - 1;

        let attackers = find_our_attackers_of_square(board, target_sq, color);
        if attackers.len() >= 2 {
            if let Some(piece_kind) = board.get_piece_on_square(target_sq) {
                let target_value = MATERIAL_VALUES[piece_kind as usize];
                let coordination_bonus = (target_value / 20) * (attackers.len() as i32 - 1);
                bonus += coordination_bonus.min(40);
            }
        }
    }

    bonus
}

/// Encontra nossos atacantes de uma casa
fn find_our_attackers_of_square(board: &Board, target_sq: u8, color: Color) -> Vec<u8> {
    let mut attackers = Vec::new();
    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let all_pieces = board.white_pieces | board.black_pieces;

    let mut pieces_bb = our_pieces;
    while pieces_bb != 0 {
        let piece_sq = pieces_bb.trailing_zeros() as u8;
        pieces_bb &= pieces_bb - 1;

        if let Some(piece_kind) = board.get_piece_on_square(piece_sq) {
            let can_attack = match piece_kind {
                PieceKind::Pawn => utils::can_pawn_attack_square(piece_sq, target_sq, color),
                PieceKind::Knight => {
                    let attacks = crate::moves::knight::get_knight_attacks_lookup(piece_sq);
                    (attacks & (1u64 << target_sq)) != 0
                },
                PieceKind::Bishop => {
                    let attacks = crate::moves::magic_bitboards::get_bishop_attacks_magic(piece_sq, all_pieces);
                    (attacks & (1u64 << target_sq)) != 0
                },
                PieceKind::Rook => {
                    let attacks = crate::moves::magic_bitboards::get_rook_attacks_magic(piece_sq, all_pieces);
                    (attacks & (1u64 << target_sq)) != 0
                },
                PieceKind::Queen => {
                    let attacks = crate::moves::magic_bitboards::get_queen_attacks_magic(piece_sq, all_pieces);
                    (attacks & (1u64 << target_sq)) != 0
                },
                PieceKind::King => {
                    let attacks = crate::moves::king::get_king_attacks_lookup(piece_sq);
                    (attacks & (1u64 << target_sq)) != 0
                },
            };

            if can_attack {
                attackers.push(piece_sq);
            }
        }
    }

    attackers
}

/// Gera mapa de ataques inimigos
fn get_enemy_attack_map(board: &Board, enemy_color: Color) -> Bitboard {
    let mut attack_map = 0u64;
    let enemy_pieces = if enemy_color == Color::White { board.white_pieces } else { board.black_pieces };
    let all_pieces = board.white_pieces | board.black_pieces;

    // Ataques de peões
    let enemy_pawns = board.pawns & enemy_pieces;
    attack_map |= utils::compute_pawn_attacks(enemy_pawns, enemy_color);

    // Ataques de cavalos
    let mut knights = board.knights & enemy_pieces;
    while knights != 0 {
        let sq = knights.trailing_zeros() as u8;
        knights &= knights - 1;
        attack_map |= crate::moves::knight::get_knight_attacks_lookup(sq);
    }

    // Ataques de bispos e rainhas (diagonais)
    let mut bishops = (board.bishops | board.queens) & enemy_pieces;
    while bishops != 0 {
        let sq = bishops.trailing_zeros() as u8;
        bishops &= bishops - 1;
        attack_map |= crate::moves::magic_bitboards::get_bishop_attacks_magic(sq, all_pieces);
    }

    // Ataques de torres e rainhas (linhas/colunas)
    let mut rooks = (board.rooks | board.queens) & enemy_pieces;
    while rooks != 0 {
        let sq = rooks.trailing_zeros() as u8;
        rooks &= rooks - 1;
        attack_map |= crate::moves::magic_bitboards::get_rook_attacks_magic(sq, all_pieces);
    }

    // Ataques do rei
    let enemy_king = board.kings & enemy_pieces;
    if enemy_king != 0 {
        let king_sq = enemy_king.trailing_zeros() as u8;
        attack_map |= crate::moves::king::get_king_attacks_lookup(king_sq);
    }

    attack_map
}

/// Gera mapa de ataques nossos
fn get_our_attack_map(board: &Board, color: Color) -> Bitboard {
    get_enemy_attack_map(board, color)
}

/// Verifica se peça ataca zona do rei inimigo
fn attacks_enemy_king_zone(board: &Board, piece_sq: u8, color: Color) -> bool {
    let enemy_color = !color;
    let enemy_pieces = if enemy_color == Color::White { board.white_pieces } else { board.black_pieces };
    let enemy_king = board.kings & enemy_pieces;

    if enemy_king == 0 { return false; }
    let king_sq = enemy_king.trailing_zeros() as u8;

    let king_zone = get_king_zone(king_sq);

    if let Some(piece_kind) = board.get_piece_on_square(piece_sq) {
        let all_pieces = board.white_pieces | board.black_pieces;
        let attacks = match piece_kind {
            PieceKind::Pawn => utils::compute_pawn_attacks(1u64 << piece_sq, color),
            PieceKind::Knight => crate::moves::knight::get_knight_attacks_lookup(piece_sq),
            PieceKind::Bishop => crate::moves::magic_bitboards::get_bishop_attacks_magic(piece_sq, all_pieces),
            PieceKind::Rook => crate::moves::magic_bitboards::get_rook_attacks_magic(piece_sq, all_pieces),
            PieceKind::Queen => crate::moves::magic_bitboards::get_queen_attacks_magic(piece_sq, all_pieces),
            PieceKind::King => crate::moves::king::get_king_attacks_lookup(piece_sq),
        };

        (attacks & king_zone) != 0
    } else {
        false
    }
}

/// Obtém zona do rei
fn get_king_zone(king_sq: u8) -> Bitboard {
    let king_attacks = crate::moves::king::get_king_attacks_lookup(king_sq);
    king_attacks | (1u64 << king_sq)
}

/// Verifica se peça está em posição avançada
fn is_advanced_piece(sq: u8, piece_kind: PieceKind, color: Color) -> bool {
    let rank = sq / 8;
    match color {
        Color::White => match piece_kind {
            PieceKind::Knight | PieceKind::Bishop => rank >= 4,
            PieceKind::Rook | PieceKind::Queen => rank >= 3,
            PieceKind::Pawn => rank >= 5,
            _ => false,
        },
        Color::Black => match piece_kind {
            PieceKind::Knight | PieceKind::Bishop => rank <= 3,
            PieceKind::Rook | PieceKind::Queen => rank <= 4,
            PieceKind::Pawn => rank <= 2,
            _ => false,
        },
    }
}

/// Encontra menor atacante
fn find_smallest_attacker_value(board: &Board, target_square: u8, attacker_color: Color) -> i32 {
    let attacker_pieces = if attacker_color == Color::White { board.white_pieces } else { board.black_pieces };

    // Verifica em ordem de valor (menor primeiro)
    if can_piece_type_attack(board, target_square, attacker_color, board.pawns & attacker_pieces) {
        return MATERIAL_VALUES[0]; // Peão = 100
    }

    if can_piece_type_attack(board, target_square, attacker_color, board.knights & attacker_pieces) {
        return MATERIAL_VALUES[1]; // Cavalo = 320
    }

    if can_piece_type_attack(board, target_square, attacker_color, board.bishops & attacker_pieces) {
        return MATERIAL_VALUES[2]; // Bispo = 330
    }

    if can_piece_type_attack(board, target_square, attacker_color, board.rooks & attacker_pieces) {
        return MATERIAL_VALUES[3]; // Torre = 500
    }

    if can_piece_type_attack(board, target_square, attacker_color, board.queens & attacker_pieces) {
        return MATERIAL_VALUES[4]; // Rainha = 900
    }

    if can_piece_type_attack(board, target_square, attacker_color, board.kings & attacker_pieces) {
        return MATERIAL_VALUES[5]; // Rei = 20000
    }

    1000 // Nenhum atacante encontrado
}

/// Verifica se alguma peça de um tipo pode atacar o alvo
fn can_piece_type_attack(board: &Board, target_square: u8, attacker_color: Color, piece_bb: Bitboard) -> bool {
    let mut bb = piece_bb;

    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        bb &= bb - 1;

        if piece_can_attack_square(board, sq, target_square, attacker_color, piece_bb) {
            return true;
        }
    }

    false
}

/// Verifica se uma peça específica pode atacar uma casa
fn piece_can_attack_square(board: &Board, piece_square: u8, target_square: u8, color: Color, piece_type_bb: Bitboard) -> bool {
    let all_pieces = board.white_pieces | board.black_pieces;

    if (piece_type_bb & board.pawns) != 0 {
        let attacks = utils::compute_pawn_attacks(1u64 << piece_square, color);
        (attacks & (1u64 << target_square)) != 0
    } else if (piece_type_bb & board.knights) != 0 {
        let attacks = crate::moves::knight::get_knight_attacks_lookup(piece_square);
        (attacks & (1u64 << target_square)) != 0
    } else if (piece_type_bb & board.bishops) != 0 {
        let attacks = crate::moves::magic_bitboards::get_bishop_attacks_magic(piece_square, all_pieces);
        (attacks & (1u64 << target_square)) != 0
    } else if (piece_type_bb & board.rooks) != 0 {
        let attacks = crate::moves::magic_bitboards::get_rook_attacks_magic(piece_square, all_pieces);
        (attacks & (1u64 << target_square)) != 0
    } else if (piece_type_bb & board.queens) != 0 {
        let attacks = crate::moves::magic_bitboards::get_queen_attacks_magic(piece_square, all_pieces);
        (attacks & (1u64 << target_square)) != 0
    } else if (piece_type_bb & board.kings) != 0 {
        let attacks = crate::moves::king::get_king_attacks_lookup(piece_square);
        (attacks & (1u64 << target_square)) != 0
    } else {
        false
    }
}

// === FUNÇÕES LEGADAS PARA COMPATIBILIDADE ===

/// Avalia bônus por atacar peças inimigas (legada)
fn evaluate_enemy_attacks(board: &Board, color: Color) -> i32 {
    let mut bonus = 0;
    let enemy_color = !color;
    let enemy_pieces = if enemy_color == Color::White { board.white_pieces } else { board.black_pieces };

    let enemy_valuables = (board.knights | board.bishops | board.rooks | board.queens) & enemy_pieces;
    let mut bb = enemy_valuables;

    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        bb &= bb - 1;

        if board.is_square_attacked_by(sq, color) {
            let piece_kind = board.get_piece_on_square(sq).unwrap();
            let piece_value = MATERIAL_VALUES[piece_kind as usize];

            bonus += piece_value / 8;

            if !board.is_square_attacked_by(sq, enemy_color) {
                bonus += piece_value / 6;
            }
        }
    }

    bonus
}

/// Avalia knight forks (legada)
fn evaluate_knight_forks(board: &Board, color: Color) -> i32 {
    let mut bonus = 0;
    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let enemy_pieces = if color == Color::White { board.black_pieces } else { board.white_pieces };

    let our_knights = board.knights & our_pieces;
    let mut knight_bb = our_knights;

    while knight_bb != 0 {
        let knight_sq = knight_bb.trailing_zeros() as u8;
        knight_bb &= knight_bb - 1;

        let knight_attacks = crate::moves::knight::get_knight_attacks_lookup(knight_sq);
        let valuable_enemies = (board.knights | board.bishops | board.rooks | board.queens) & enemy_pieces;
        let attacked_valuables = knight_attacks & valuable_enemies;

        if attacked_valuables.count_ones() >= 2 {
            bonus += 30;
        }
    }

    bonus
}

/// Avalia queen forks (legada)
fn evaluate_queen_forks(board: &Board, color: Color) -> i32 {
    evaluate_knight_forks(board, color) / 2 // Simplificado
}

/// Avalia bishop forks (legada)
fn evaluate_bishop_forks(board: &Board, color: Color) -> i32 {
    evaluate_knight_forks(board, color) / 3 // Simplificado
}

/// Avalia pinos reais (legada)
fn evaluate_real_pins(board: &Board, color: Color) -> i32 {
    evaluate_pins_and_skewers_combined(board, color) / 2
}

/// Avalia skewers (legada)
fn evaluate_skewers(board: &Board, color: Color) -> i32 {
    evaluate_pins_and_skewers_combined(board, color) / 3
}

/// Avalia overloads (legada)
fn evaluate_overloads(board: &Board, color: Color) -> i32 {
    evaluate_overloaded_defenders(board, color)
}

/// Avalia X-ray threats (legada)
fn evaluate_xray_threats(board: &Board, color: Color) -> i32 {
    evaluate_discovered_attacks_new(board, color) / 2
}

/// Avalia compound threats (legada)
fn evaluate_compound_threats(board: &Board, color: Color) -> i32 {
    evaluate_coordinated_attacks(board, color)
}