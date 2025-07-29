// Bônus de coordenação entre peças
use crate::types::{Color, Bitboard};
use super::{MobilityContext, utils::*};

/// Avalia coordenação e sinergia entre peças
pub fn evaluate_piece_coordination(context: &MobilityContext) -> i32 {
    let mut coordination_score = 0;

    // Baterias (torres/rainhas na mesma linha)
    coordination_score += evaluate_batteries(context);

    // Peças defendendo umas às outras
    coordination_score += evaluate_mutual_defense(context);

    // Controle combinado de casas importantes
    coordination_score += evaluate_combined_control(context);

    // Apoio a peões passados
    coordination_score += evaluate_passed_pawn_support(context);

    // Coordenação para ataque ao rei
    coordination_score += evaluate_king_attack_coordination(context);

    coordination_score
}

/// Avalia baterias (peças na mesma linha de ataque)
fn evaluate_batteries(context: &MobilityContext) -> i32 {
    let mut battery_score = 0;
    let heavy_pieces = (context.board.queens | context.board.rooks) & context.our_pieces;

    if heavy_pieces.count_ones() < 2 { return 0; }

    let piece_squares = get_set_bits(heavy_pieces);

    // Verifica baterias em arquivos
    for file in 0..8 {
        let file_mask = get_file_mask(file);
        let pieces_in_file = (heavy_pieces & file_mask).count_ones();
        if pieces_in_file >= 2 {
            battery_score += 15; // Bateria vertical

            // Bônus se arquivo está aberto/semi-aberto
            if is_open_file(file, context) {
                battery_score += 10;
            } else if is_semi_open_file(file, context.color, context) {
                battery_score += 5;
            }
        }
    }

    // Verifica baterias em fileiras
    for rank in 0..8 {
        let rank_mask = get_rank_mask(rank);
        let pieces_in_rank = (heavy_pieces & rank_mask).count_ones();
        if pieces_in_rank >= 2 {
            battery_score += 12; // Bateria horizontal

            // Bônus extra se é fileira importante (2ª/7ª)
            if rank == 1 || rank == 6 {
                battery_score += 8;
            }
        }
    }

    // Baterias diagonais (bispo + rainha)
    let diagonal_pieces = (context.board.queens | context.board.bishops) & context.our_pieces;
    if diagonal_pieces.count_ones() >= 2 {
        battery_score += evaluate_diagonal_batteries(diagonal_pieces, context);
    }

    battery_score
}

/// Avalia baterias diagonais
fn evaluate_diagonal_batteries(diagonal_pieces: Bitboard, context: &MobilityContext) -> i32 {
    let mut diagonal_score = 0;
    let piece_squares = get_set_bits(diagonal_pieces);

    for i in 0..piece_squares.len() {
        for j in (i+1)..piece_squares.len() {
            let sq1 = piece_squares[i];
            let sq2 = piece_squares[j];

            if are_on_same_diagonal(sq1, sq2) {
                diagonal_score += 10;

                // Bônus se ataca território inimigo
                let enemy_territory = get_enemy_territory(context.enemy_color);
                let sq1_attacks = crate::moves::magic_bitboards::get_bishop_attacks_magic(sq1, context.all_pieces);
                let sq2_attacks = crate::moves::magic_bitboards::get_bishop_attacks_magic(sq2, context.all_pieces);

                if (sq1_attacks & enemy_territory) != 0 && (sq2_attacks & enemy_territory) != 0 {
                    diagonal_score += 5;
                }
            }
        }
    }

    diagonal_score
}

/// Verifica se duas casas estão na mesma diagonal
fn are_on_same_diagonal(sq1: u8, sq2: u8) -> bool {
    let file_diff = ((sq1 % 8) as i32 - (sq2 % 8) as i32).abs();
    let rank_diff = ((sq1 / 8) as i32 - (sq2 / 8) as i32).abs();
    file_diff == rank_diff && file_diff > 0
}

/// Avalia defesa mútua entre peças
fn evaluate_mutual_defense(context: &MobilityContext) -> i32 {
    let mut defense_score = 0;
    let our_pieces_bb = context.our_pieces;
    let valuable_pieces = (context.board.queens | context.board.rooks | context.board.bishops | context.board.knights) & our_pieces_bb;

    let valuable_squares = get_set_bits(valuable_pieces);

    for &piece_sq in &valuable_squares {
        let defenders = count_defenders(piece_sq, context);

        // Bônus por peças valiosas bem defendidas
        if defenders >= 2 {
            defense_score += defenders * 2;
        } else if defenders == 1 {
            defense_score += 1;
        }

        // Penalidade por peças não defendidas
        if defenders == 0 && (context.enemy_attacked_squares & (1u64 << piece_sq)) != 0 {
            defense_score -= 8; // Peça pendurada
        }
    }

    defense_score
}

/// Conta quantas peças defendem uma casa
fn count_defenders(square: u8, context: &MobilityContext) -> i32 {
    let square_bb = 1u64 << square;

    // Verifica se nossa estrutura de ataque inclui esta casa
    (context.our_attacked_squares & square_bb != 0) as i32
}

/// Avalia controle combinado de casas importantes
fn evaluate_combined_control(context: &MobilityContext) -> i32 {
    let mut control_score = 0;

    // Casas centrais controladas por múltiplas peças
    let center_squares = 0x0000001818000000u64; // d4, e4, d5, e5
    let our_attacks = context.our_attacked_squares;

    let controlled_center = our_attacks & center_squares;
    let center_control_count = controlled_center.count_ones() as i32;
    control_score += center_control_count * 5;

    // Casas próximas ao rei inimigo
    let enemy_king = context.board.kings & context.enemy_pieces;
    if enemy_king != 0 {
        let king_sq = enemy_king.trailing_zeros() as u8;
        let king_area = crate::moves::king::get_king_attacks_lookup(king_sq);
        let controlled_king_area = (our_attacks & king_area).count_ones() as i32;
        control_score += controlled_king_area * 3;
    }

    // Casas de invasão no território inimigo
    let enemy_territory = get_enemy_territory(context.enemy_color);
    let invasion_squares = our_attacks & enemy_territory;
    control_score += (invasion_squares.count_ones() as i32) / 4;

    control_score
}

/// Avalia suporte a peões passados
fn evaluate_passed_pawn_support(context: &MobilityContext) -> i32 {
    let mut support_score = 0;
    let our_pawns = context.board.pawns & context.our_pieces;
    let enemy_pawns = context.board.pawns & context.enemy_pieces;

    if our_pawns == 0 { return 0; }

    let pawn_squares = get_set_bits(our_pawns);

    for &pawn_sq in &pawn_squares {
        // Verifica se é peão passado (simplificado)
        if super::super::utils::is_passed_pawn(pawn_sq, context.color, our_pawns, enemy_pawns) {
            let promotion_path = get_promotion_path(pawn_sq, context.color);

            // Conta quantas de nossas peças apoiam o caminho
            let supported_squares = promotion_path & context.our_attacked_squares;
            support_score += (supported_squares.count_ones() as i32) * 4;

            // Bônus especial se torres apoiam por trás
            let rook_support = has_rook_support_from_behind(pawn_sq, context);
            if rook_support {
                support_score += 8;
            }
        }
    }

    support_score
}

/// Verifica suporte de torre por trás do peão
fn has_rook_support_from_behind(pawn_sq: u8, context: &MobilityContext) -> bool {
    let file = pawn_sq % 8;
    let file_mask = get_file_mask(file);
    let our_rooks = (context.board.rooks | context.board.queens) & context.our_pieces;
    let rooks_on_file = our_rooks & file_mask;

    if rooks_on_file == 0 { return false; }

    let rook_squares = get_set_bits(rooks_on_file);
    let pawn_rank = pawn_sq / 8;

    for &rook_sq in &rook_squares {
        let rook_rank = rook_sq / 8;

        let behind_pawn = if context.color == Color::White {
            rook_rank < pawn_rank
        } else {
            rook_rank > pawn_rank
        };

        if behind_pawn {
            return true;
        }
    }

    false
}

/// Obtém caminho de promoção do peão
fn get_promotion_path(pawn_sq: u8, color: Color) -> Bitboard {
    let file = pawn_sq % 8;
    let file_mask = get_file_mask(file);
    let current_rank = pawn_sq / 8;

    let promotion_ranks = if color == Color::White {
        ((current_rank + 1)..8).collect::<Vec<_>>()
    } else {
        (0..current_rank).collect::<Vec<_>>()
    };

    let mut path = 0u64;
    for rank in promotion_ranks {
        path |= 1u64 << (rank * 8 + file);
    }

    path
}

/// Avalia coordenação para ataque ao rei
fn evaluate_king_attack_coordination(context: &MobilityContext) -> i32 {
    let enemy_king = context.board.kings & context.enemy_pieces;
    if enemy_king == 0 { return 0; }

    let king_sq = enemy_king.trailing_zeros() as u8;
    let king_area = get_extended_king_area(king_sq);

    let mut attack_score = 0;

    // Conta quantos tipos de peças atacam área do rei
    let attacking_pieces = count_piece_types_attacking_area(king_area, context);

    if attacking_pieces >= 2 {
        attack_score += attacking_pieces * 5;

        // Bônus extra se rainha participa do ataque
        let queen_attacks = if (context.board.queens & context.our_pieces) != 0 {
            let queens = get_set_bits(context.board.queens & context.our_pieces);
            queens.iter().any(|&queen_sq| {
                let queen_attacks = crate::moves::magic_bitboards::get_queen_attacks_magic(queen_sq, context.all_pieces);
                (queen_attacks & king_area) != 0
            })
        } else {
            false
        };

        if queen_attacks {
            attack_score += 10;
        }
    }

    attack_score
}

/// Obtém área estendida ao redor do rei
fn get_extended_king_area(king_sq: u8) -> Bitboard {
    let mut area = crate::moves::king::get_king_attacks_lookup(king_sq);
    area |= 1u64 << king_sq; // Inclui a casa do rei

    // Adiciona casas em raio 2
    let king_file = king_sq % 8;
    let king_rank = king_sq / 8;

    for file in (king_file.saturating_sub(2))..=(king_file.saturating_add(2)).min(7) {
        for rank in (king_rank.saturating_sub(2))..=(king_rank.saturating_add(2)).min(7) {
            area |= 1u64 << (rank * 8 + file);
        }
    }

    area
}

/// Conta tipos de peças atacando uma área
fn count_piece_types_attacking_area(area: Bitboard, context: &MobilityContext) -> i32 {
    let mut attacking_types = 0;

    // Verifica cada tipo de peça
    let piece_types = [
        (context.board.pawns, "pawn"),
        (context.board.knights, "knight"),
        (context.board.bishops, "bishop"),
        (context.board.rooks, "rook"),
        (context.board.queens, "queen"),
    ];

    for &(piece_bb, _piece_name) in &piece_types {
        let our_pieces = piece_bb & context.our_pieces;
        if our_pieces == 0 { continue; }

        let pieces = get_set_bits(our_pieces);
        let attacks_area = pieces.iter().any(|&piece_sq| {
            let attacks = match _piece_name {
                "pawn" => super::super::utils::compute_pawn_attacks(1u64 << piece_sq, context.color),
                "knight" => crate::moves::knight::get_knight_attacks_lookup(piece_sq),
                "bishop" => crate::moves::magic_bitboards::get_bishop_attacks_magic(piece_sq, context.all_pieces),
                "rook" => crate::moves::magic_bitboards::get_rook_attacks_magic(piece_sq, context.all_pieces),
                "queen" => crate::moves::magic_bitboards::get_queen_attacks_magic(piece_sq, context.all_pieces),
                _ => 0,
            };
            (attacks & area) != 0
        });

        if attacks_area {
            attacking_types += 1;
        }
    }

    attacking_types
}


/// Obtém território inimigo
fn get_enemy_territory(enemy_color: Color) -> Bitboard {
    if enemy_color == Color::White {
        0xFFFFFFFF00000000u64 // Ranks 5-8
    } else {
        0x00000000FFFFFFFFu64 // Ranks 1-4
    }
}