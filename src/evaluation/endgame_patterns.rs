// Padrões avançados de endgame - triangulação, oposição à distância, etc.
use crate::{board::Board, types::{Color, Bitboard}};
use super::utils as eval_utils;

/// Estrutura para análise avançada de padrões de endgame
#[derive(Debug, Clone, Copy)]
pub struct EndgamePatterns {
    pub king_opposition: i32,
    pub distant_opposition: i32,
    pub triangulation: i32,
    pub outflanking: i32,
    pub zugzwang_potential: i32,
    pub square_rule: i32,
}

impl EndgamePatterns {
    pub fn new() -> Self {
        EndgamePatterns {
            king_opposition: 0,
            distant_opposition: 0,
            triangulation: 0,
            outflanking: 0,
            zugzwang_potential: 0,
            square_rule: 0,
        }
    }

    pub fn total_score(&self) -> i32 {
        self.king_opposition + self.distant_opposition + self.triangulation +
            self.outflanking + self.zugzwang_potential + self.square_rule
    }
}

/// Avalia padrões avançados de endgame
pub fn evaluate_endgame_patterns(board: &Board, color: Color) -> EndgamePatterns {
    let mut patterns = EndgamePatterns::new();

    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let enemy_pieces = if color == Color::White { board.black_pieces } else { board.white_pieces };

    let our_king = board.kings & our_pieces;
    let enemy_king = board.kings & enemy_pieces;

    if our_king == 0 || enemy_king == 0 {
        return patterns;
    }

    let our_king_sq = our_king.trailing_zeros() as u8;
    let enemy_king_sq = enemy_king.trailing_zeros() as u8;

    // Avalia diferentes tipos de oposição
    patterns.king_opposition = evaluate_opposition_types(our_king_sq, enemy_king_sq);
    patterns.distant_opposition = evaluate_distant_opposition(our_king_sq, enemy_king_sq);

    // Avalia potencial de triangulação
    patterns.triangulation = evaluate_triangulation_potential(board, color, our_king_sq, enemy_king_sq);

    // Avalia outflanking (contorno)
    patterns.outflanking = evaluate_outflanking(our_king_sq, enemy_king_sq);

    // Avalia potencial de zugzwang
    patterns.zugzwang_potential = evaluate_zugzwang_potential(board, color);

    // Regra do quadrado para peões passados
    patterns.square_rule = evaluate_square_rule(board, color, our_king_sq, enemy_king_sq);

    patterns
}

/// Avalia diferentes tipos de oposição entre reis
fn evaluate_opposition_types(our_king: u8, enemy_king: u8) -> i32 {
    let our_file = our_king % 8;
    let our_rank = our_king / 8;
    let enemy_file = enemy_king % 8;
    let enemy_rank = enemy_king / 8;

    let file_diff = (our_file as i32 - enemy_file as i32).abs();
    let rank_diff = (our_rank as i32 - enemy_rank as i32).abs();

    // Oposição direta (horizontal/vertical) - valor corrigido conforme análise
    if (file_diff == 0 && rank_diff == 2) || (file_diff == 2 && rank_diff == 0) {
        return 150; // Aumentado de 25 para 150 - oposição é fundamental em finais
    }

    // Oposição diagonal - valor também aumentado
    if file_diff == 2 && rank_diff == 2 {
        return 80; // Aumentado de 20 para 80 - oposição diagonal também crucial
    }

    // Oposição próxima (1 casa de distância)
    if (file_diff == 0 && rank_diff == 1) || (file_diff == 1 && rank_diff == 0) {
        return -10; // Estar muito próximo pode ser ruim
    }

    0
}

/// Avalia oposição à distância
fn evaluate_distant_opposition(our_king: u8, enemy_king: u8) -> i32 {
    let our_file = our_king % 8;
    let our_rank = our_king / 8;
    let enemy_file = enemy_king % 8;
    let enemy_rank = enemy_king / 8;

    let file_diff = (our_file as i32 - enemy_file as i32).abs();
    let rank_diff = (our_rank as i32 - enemy_rank as i32).abs();

    // Oposição à distância horizontal
    if file_diff == 0 && rank_diff >= 4 && rank_diff % 2 == 0 {
        return 15 - (rank_diff - 4) * 2; // Menor valor para distâncias maiores
    }

    // Oposição à distância vertical
    if rank_diff == 0 && file_diff >= 4 && file_diff % 2 == 0 {
        return 15 - (file_diff - 4) * 2;
    }

    // Oposição à distância diagonal
    if file_diff == rank_diff && file_diff >= 4 && file_diff % 2 == 0 {
        return 12 - (file_diff - 4) * 2;
    }

    0
}

/// Avalia potencial de triangulação
fn evaluate_triangulation_potential(board: &Board, color: Color, our_king: u8, enemy_king: u8) -> i32 {
    // Triangulação é útil quando:
    // 1. Há poucas peças no tabuleiro
    // 2. O rei precisa perder tempo para forçar o oponente a se mover primeiro
    // 3. Há espaço para manobras

    let total_pieces = (board.white_pieces | board.black_pieces).count_ones();

    // Só é relevante em endgames muito simples
    if total_pieces > 6 {
        return 0;
    }

    let our_file = our_king % 8;
    let our_rank = our_king / 8;
    let enemy_file = enemy_king % 8;
    let enemy_rank = enemy_king / 8;

    // Verifica se há espaço para triangulação (rei tem pelo menos 3 casas livres)
    let king_area = crate::moves::king::get_king_attacks_lookup(our_king);
    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let enemy_pieces = if color == Color::White { board.black_pieces } else { board.white_pieces };
    let all_pieces = board.white_pieces | board.black_pieces;

    let free_squares = king_area & !(all_pieces);
    let mobility = free_squares.count_ones();

    if mobility < 3 {
        return 0; // Não há espaço suficiente para triangular
    }

    // Verifica se estamos em oposição ou próximos dela
    let distance = eval_utils::calculate_square_distance(our_king, enemy_king);

    if distance == 2 || distance == 3 {
        // Situação ideal para triangulação
        let space_bonus = (mobility as i32 - 3) * 2;
        return 8 + space_bonus;
    }

    0
}

/// Avalia potencial de outflanking (contorno do rei inimigo)
fn evaluate_outflanking(our_king: u8, enemy_king: u8) -> i32 {
    let our_file = our_king % 8;
    let our_rank = our_king / 8;
    let enemy_file = enemy_king % 8;
    let enemy_rank = enemy_king / 8;

    let file_diff = our_file as i32 - enemy_file as i32;
    let rank_diff = our_rank as i32 - enemy_rank as i32;

    // Outflanking horizontal
    if rank_diff.abs() <= 1 && file_diff.abs() >= 2 {
        let flank_advantage = file_diff.abs() - 1;
        return flank_advantage * 3;
    }

    // Outflanking vertical
    if file_diff.abs() <= 1 && rank_diff.abs() >= 2 {
        let flank_advantage = rank_diff.abs() - 1;
        return flank_advantage * 3;
    }

    0
}

/// Avalia potencial de zugzwang
fn evaluate_zugzwang_potential(board: &Board, color: Color) -> i32 {
    // Zugzwang é mais provável quando:
    // 1. Poucas peças no tabuleiro
    // 2. Posição fechada/bloqueada
    // 3. Ambos os reis têm mobilidade limitada

    let total_pieces = (board.white_pieces | board.black_pieces).count_ones();

    if total_pieces > 8 {
        return 0; // Zugzwang é raro em posições com muitas peças
    }

    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let enemy_pieces = if color == Color::White { board.black_pieces } else { board.white_pieces };

    let our_king = board.kings & our_pieces;
    let enemy_king = board.kings & enemy_pieces;

    if our_king == 0 || enemy_king == 0 {
        return 0;
    }

    let our_king_sq = our_king.trailing_zeros() as u8;
    let enemy_king_sq = enemy_king.trailing_zeros() as u8;

    // Calcula mobilidade dos reis
    let our_king_attacks = crate::moves::king::get_king_attacks_lookup(our_king_sq);
    let enemy_king_attacks = crate::moves::king::get_king_attacks_lookup(enemy_king_sq);

    let all_pieces = board.white_pieces | board.black_pieces;
    let our_mobility = (our_king_attacks & !our_pieces & !all_pieces).count_ones();
    let enemy_mobility = (enemy_king_attacks & !enemy_pieces & !all_pieces).count_ones();

    // Zugzwang é mais provável quando ambos têm mobilidade limitada
    if our_mobility <= 3 && enemy_mobility <= 3 {
        let mobility_factor = (6 - our_mobility - enemy_mobility) as i32;
        let piece_factor = (8 - total_pieces as i32) / 2;
        return mobility_factor + piece_factor;
    }

    0
}

/// Avalia regra do quadrado para peões passados
fn evaluate_square_rule(board: &Board, color: Color, our_king: u8, enemy_king: u8) -> i32 {
    let enemy_pawns = board.pawns & if color == Color::White { board.black_pieces } else { board.white_pieces };

    if enemy_pawns == 0 {
        return 0;
    }

    let mut square_rule_score = 0;
    let enemy_pawn_squares = eval_utils::get_set_bits(enemy_pawns);

    for &pawn_sq in &enemy_pawn_squares {
        // Verifica se é peão passado (simplificado)
        if is_likely_passed_pawn(pawn_sq, !color, board) {
            let pawn_file = pawn_sq % 8;
            let pawn_rank = pawn_sq / 8;

            // Calcula distância até a promoção
            let promotion_distance = if color == Color::White {
                pawn_rank as i32 // Peão preto, distância até rank 0
            } else {
                7 - pawn_rank as i32 // Peão branco, distância até rank 7
            };

            // Calcula se nosso rei pode alcançar o quadrado
            let promotion_sq = if color == Color::White {
                pawn_file // Rank 0
            } else {
                56 + pawn_file // Rank 7
            };

            let king_distance = eval_utils::calculate_square_distance(our_king, promotion_sq);

            // Regra do quadrado: se rei pode chegar antes do peão promover
            if king_distance <= promotion_distance {
                square_rule_score += 15; // Bônus por poder parar o peão
            } else {
                square_rule_score -= 20 * (king_distance - promotion_distance); // Penalidade
            }
        }
    }

    square_rule_score
}

/// Verifica se peão é provavelmente passado (versão simplificada)
fn is_likely_passed_pawn(pawn_sq: u8, pawn_color: Color, board: &Board) -> bool {
    let file = pawn_sq % 8;
    let rank = pawn_sq / 8;

    let enemy_pawns = board.pawns & if pawn_color == Color::White { board.black_pieces } else { board.white_pieces };

    // Verifica arquivos adjacentes e o próprio arquivo
    for check_file in (file.saturating_sub(1))..=(file.saturating_add(1)).min(7) {
        let file_mask = 0x0101010101010101u64 << check_file;
        let file_pawns = enemy_pawns & file_mask;

        if file_pawns != 0 {
            let pawn_squares = eval_utils::get_set_bits(file_pawns);
            for enemy_sq in pawn_squares {
                let enemy_rank = enemy_sq / 8;

                let blocks_advancement = if pawn_color == Color::White {
                    enemy_rank > rank // Peão inimigo está à frente
                } else {
                    enemy_rank < rank
                };

                if blocks_advancement {
                    return false; // Há um peão inimigo bloqueando
                }
            }
        }
    }

    true // Provavelmente passado
}

/// Avalia melhoria na atividade do rei baseada em padrões avançados
pub fn evaluate_king_activity_advanced(board: &Board, color: Color) -> i32 {
    let patterns = evaluate_endgame_patterns(board, color);

    // Aplica pesos aos padrões
    let mut activity_score = 0;

    activity_score += patterns.king_opposition;
    activity_score += patterns.distant_opposition;
    activity_score += patterns.triangulation;
    activity_score += patterns.outflanking;
    activity_score += patterns.zugzwang_potential;
    activity_score += patterns.square_rule;

    // Normaliza o score para não dominar outros fatores
    activity_score.clamp(-50, 50)
}