// Padrões avançados de endgame com foco em finais práticos
use crate::{board::Board, types::{Color, Bitboard, PieceKind}};
use super::utils as eval_utils;

/// Estrutura para análise avançada de padrões de endgame
#[derive(Debug, Clone, Copy, Default)]
pub struct EndgamePatterns {
    pub king_opposition: i32,
    pub distant_opposition: i32,
    pub triangulation: i32,
    pub outflanking: i32,
    pub zugzwang_potential: i32,
    pub square_rule: i32,
    pub king_activity: i32,
    pub pawn_promotion_race: i32,
    pub fortress_patterns: i32,
    pub rook_endgame_bonus: i32,
    pub bishop_endgame_bonus: i32,      // Novo
    pub knight_endgame_bonus: i32,      // Novo
    pub queen_endgame_bonus: i32,       // Novo
    pub breakthrough_patterns: i32,      // Novo
}

impl EndgamePatterns {
    pub fn new() -> Self {
        EndgamePatterns::default()
    }

    pub fn total_score(&self) -> i32 {
        self.king_opposition + self.distant_opposition + self.triangulation +
            self.outflanking + self.zugzwang_potential + self.square_rule +
            self.king_activity + self.pawn_promotion_race + self.fortress_patterns +
            self.rook_endgame_bonus + self.bishop_endgame_bonus +
            self.knight_endgame_bonus + self.queen_endgame_bonus +
            self.breakthrough_patterns
    }
}

/// Configuração para avaliação de endgame
pub struct EndgameConfig {
    pub evaluate_tablebases: bool,
    pub deep_king_analysis: bool,
    pub pawn_race_depth: i32,
}

impl Default for EndgameConfig {
    fn default() -> Self {
        EndgameConfig {
            evaluate_tablebases: false,
            deep_king_analysis: true,
            pawn_race_depth: 5,
        }
    }
}

/// Avalia padrões avançados de endgame
pub fn evaluate_endgame_patterns(board: &Board, color: Color) -> EndgamePatterns {
    evaluate_endgame_patterns_with_config(board, color, &EndgameConfig::default())
}

/// Avalia padrões avançados de endgame com configuração
pub fn evaluate_endgame_patterns_with_config(board: &Board, color: Color, config: &EndgameConfig) -> EndgamePatterns {
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

    // Análise básica de oposição de reis
    if config.deep_king_analysis {
        evaluate_king_patterns(&mut patterns, our_king_sq, enemy_king_sq);
    } else {
        patterns.king_opposition = evaluate_opposition_types(our_king_sq, enemy_king_sq);
    }

    // Avalia potencial de zugzwang
    patterns.zugzwang_potential = evaluate_zugzwang_potential(board, color);

    // Regra do quadrado para peões passados
    patterns.square_rule = evaluate_square_rule_advanced(board, color, our_king_sq, enemy_king_sq);

    // Atividade do rei aprimorada
    patterns.king_activity = evaluate_king_activity_enhanced(board, color, our_king_sq);

    // Corrida de promoção de peões
    patterns.pawn_promotion_race = evaluate_pawn_promotion_race_advanced(board, color, config.pawn_race_depth);

    // Padrões de fortaleza
    patterns.fortress_patterns = evaluate_fortress_patterns(board, color);

    // Avaliação específica por tipo de peça
    patterns.rook_endgame_bonus = evaluate_rook_endgame_patterns(board, color);
    patterns.bishop_endgame_bonus = evaluate_bishop_endgame_patterns(board, color);
    patterns.knight_endgame_bonus = evaluate_knight_endgame_patterns(board, color);
    patterns.queen_endgame_bonus = evaluate_queen_endgame_patterns(board, color);

    // Novos padrões
    patterns.breakthrough_patterns = evaluate_breakthrough_patterns(board, color);

    patterns
}

/// Avalia padrões relacionados aos reis (oposição, triangulação, etc)
fn evaluate_king_patterns(patterns: &mut EndgamePatterns, our_king_sq: u8, enemy_king_sq: u8) {
    patterns.king_opposition = evaluate_opposition_types(our_king_sq, enemy_king_sq);
    patterns.distant_opposition = evaluate_distant_opposition(our_king_sq, enemy_king_sq);
    patterns.triangulation = evaluate_triangulation_advanced(our_king_sq, enemy_king_sq);
    patterns.outflanking = evaluate_outflanking_advanced(our_king_sq, enemy_king_sq);
}

/// Avalia diferentes tipos de oposição entre reis
fn evaluate_opposition_types(our_king: u8, enemy_king: u8) -> i32 {
    let our_file = (our_king % 8) as i32;
    let our_rank = (our_king / 8) as i32;
    let enemy_file = (enemy_king % 8) as i32;
    let enemy_rank = (enemy_king / 8) as i32;

    let file_diff = (our_file - enemy_file).abs();
    let rank_diff = (our_rank - enemy_rank).abs();

    // Oposição direta (horizontal/vertical)
    if (file_diff == 0 && rank_diff == 2) || (file_diff == 2 && rank_diff == 0) {
        return 30; // Aumentado de 25
    }

    // Oposição diagonal
    if file_diff == 2 && rank_diff == 2 {
        return 25; // Aumentado de 20
    }

    // Oposição próxima (contato direto)
    if file_diff <= 1 && rank_diff <= 1 && (file_diff + rank_diff) > 0 {
        // Verifica quem tem a vez para determinar se é vantajoso
        return -5; // Pequena penalidade por proximidade sem oposição
    }

    // Oposição de cavaleiro (L-shape)
    if (file_diff == 1 && rank_diff == 2) || (file_diff == 2 && rank_diff == 1) {
        return 10; // Oposição de cavaleiro pode ser útil
    }

    0
}

/// Avalia oposição à distância
fn evaluate_distant_opposition(our_king: u8, enemy_king: u8) -> i32 {
    let our_file = (our_king % 8) as i32;
    let our_rank = (our_king / 8) as i32;
    let enemy_file = (enemy_king % 8) as i32;
    let enemy_rank = (enemy_king / 8) as i32;

    let file_diff = (our_file - enemy_file).abs();
    let rank_diff = (our_rank - enemy_rank).abs();

    // Oposição à distância horizontal
    if file_diff == 0 && rank_diff >= 4 && rank_diff % 2 == 0 {
        let distance_factor = ((8 - rank_diff) as f32 / 2.0) as i32;
        return 15 + distance_factor;
    }

    // Oposição à distância vertical
    if rank_diff == 0 && file_diff >= 4 && file_diff % 2 == 0 {
        let distance_factor = ((8 - file_diff) as f32 / 2.0) as i32;
        return 15 + distance_factor;
    }

    // Oposição à distância diagonal
    if file_diff == rank_diff && file_diff >= 4 && file_diff % 2 == 0 {
        let distance_factor = ((8 - file_diff) as f32 / 2.0) as i32;
        return 12 + distance_factor;
    }

    0
}

/// Avalia potencial de triangulação avançado
fn evaluate_triangulation_advanced(our_king: u8, enemy_king: u8) -> i32 {
    let distance = eval_utils::king_distance(our_king, enemy_king);

    // Triangulação é mais valiosa quando reis estão próximos
    if distance >= 2 && distance <= 4 {
        // Verifica se nosso rei tem mais mobilidade
        let our_mobility = calculate_king_mobility_squares(our_king);
        let enemy_mobility = calculate_king_mobility_squares(enemy_king);

        if our_mobility > enemy_mobility {
            return 15 + (our_mobility - enemy_mobility) * 3;
        }
    }

    0
}

/// Calcula número de casas disponíveis para o rei
fn calculate_king_mobility_squares(king_sq: u8) -> i32 {
    let king_attacks = crate::moves::king::get_king_attacks_lookup(king_sq);
    king_attacks.count_ones() as i32
}

/// Avalia potencial de triangulação (versão original para compatibilidade)
fn evaluate_triangulation_potential(board: &Board, color: Color, our_king: u8, enemy_king: u8) -> i32 {
    let total_pieces = (board.white_pieces | board.black_pieces).count_ones();

    if total_pieces > 6 {
        return 0;
    }

    let king_area = crate::moves::king::get_king_attacks_lookup(our_king);
    let all_pieces = board.white_pieces | board.black_pieces;
    let free_squares = king_area & !all_pieces;
    let mobility = free_squares.count_ones();

    if mobility < 3 {
        return 0;
    }

    let distance = eval_utils::calculate_square_distance(our_king, enemy_king);

    if distance == 2 || distance == 3 {
        let space_bonus = (mobility as i32 - 3) * 2;
        return 8 + space_bonus;
    }

    0
}

/// Avalia outflanking avançado
fn evaluate_outflanking_advanced(our_king: u8, enemy_king: u8) -> i32 {
    let our_file = (our_king % 8) as i32;
    let our_rank = (our_king / 8) as i32;
    let enemy_file = (enemy_king % 8) as i32;
    let enemy_rank = (enemy_king / 8) as i32;

    let file_diff = our_file - enemy_file;
    let rank_diff = our_rank - enemy_rank;

    let mut score = 0;

    // Outflanking horizontal
    if rank_diff.abs() <= 1 && file_diff.abs() >= 2 {
        let flank_advantage = file_diff.abs() - 1;
        score = flank_advantage * 4; // Aumentado de 3

        // Bônus se estamos mais próximos da borda (limitando rei inimigo)
        if (enemy_file == 0 || enemy_file == 7) && file_diff.abs() >= 3 {
            score += 10;
        }
    }

    // Outflanking vertical
    if file_diff.abs() <= 1 && rank_diff.abs() >= 2 {
        let flank_advantage = rank_diff.abs() - 1;
        score = flank_advantage * 4;

        // Bônus se estamos mais próximos da borda
        if (enemy_rank == 0 || enemy_rank == 7) && rank_diff.abs() >= 3 {
            score += 10;
        }
    }

    score
}

/// Avalia outflanking (versão original)
fn evaluate_outflanking(our_king: u8, enemy_king: u8) -> i32 {
    evaluate_outflanking_advanced(our_king, enemy_king)
}

/// Avalia potencial de zugzwang
fn evaluate_zugzwang_potential(board: &Board, color: Color) -> i32 {
    let total_pieces = (board.white_pieces | board.black_pieces).count_ones();

    if total_pieces > 8 {
        return 0;
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

        let mut score = mobility_factor + piece_factor;

        // Bônus se temos peões bloqueados (situação típica de zugzwang)
        let blocked_pawns = count_blocked_pawns(board);
        if blocked_pawns >= 2 {
            score += blocked_pawns * 3;
        }

        return score;
    }

    0
}

/// Conta peões bloqueados
fn count_blocked_pawns(board: &Board) -> i32 {
    let white_pawns = board.pawns & board.white_pieces;
    let black_pawns = board.pawns & board.black_pieces;
    let all_pieces = board.white_pieces | board.black_pieces;

    let white_advances = (white_pawns << 8) & all_pieces;
    let black_advances = (black_pawns >> 8) & all_pieces;

    (white_advances.count_ones() + black_advances.count_ones()) as i32
}

/// Avalia regra do quadrado avançada
fn evaluate_square_rule_advanced(board: &Board, color: Color, our_king: u8, enemy_king: u8) -> i32 {
    let mut score = 0;

    // Avalia peões passados inimigos
    let enemy_pawns = board.pawns & if color == Color::White { board.black_pieces } else { board.white_pieces };
    score -= evaluate_passed_pawns_square_rule(board, !color, enemy_pawns, our_king, true);

    // Avalia nossos peões passados
    let our_pawns = board.pawns & if color == Color::White { board.white_pieces } else { board.black_pieces };
    score += evaluate_passed_pawns_square_rule(board, color, our_pawns, enemy_king, false);

    score
}

/// Avalia regra do quadrado para conjunto de peões
fn evaluate_passed_pawns_square_rule(board: &Board, pawn_color: Color, pawns: Bitboard, king_sq: u8, is_enemy: bool) -> i32 {
    let mut score = 0;
    let enemy_pawns = board.pawns & if pawn_color == Color::White { board.black_pieces } else { board.white_pieces };
    let our_pawns = board.pawns & if pawn_color == Color::White { board.white_pieces } else { board.black_pieces };

    let mut pawn_bb = pawns;
    while pawn_bb != 0 {
        let pawn_sq = pawn_bb.trailing_zeros() as u8;
        pawn_bb &= pawn_bb - 1;

        if eval_utils::is_passed_pawn(pawn_sq, pawn_color, our_pawns, enemy_pawns) {
            let pawn_rank = pawn_sq / 8;
            let pawn_file = pawn_sq % 8;

            // Calcula distância até promoção
            let promotion_distance = if pawn_color == Color::White {
                7 - pawn_rank
            } else {
                pawn_rank
            };

            // Casa de promoção
            let promotion_sq = if pawn_color == Color::White {
                56 + pawn_file // Rank 8
            } else {
                pawn_file // Rank 1
            };

            let king_distance = eval_utils::king_distance(king_sq, promotion_sq) as i32;

            // Regra do quadrado com tempo
            let can_catch = if board.to_move == pawn_color {
                king_distance <= promotion_distance as i32 + 1
            } else {
                king_distance <= promotion_distance as i32
            };

            if can_catch {
                score += 15; // Pode parar o peão
            } else {
                // Peão não pode ser parado
                let unstoppable_bonus = match promotion_distance {
                    0 => 200,  // Promove na próxima
                    1 => 150,  // Muito próximo
                    2 => 100,  // Próximo
                    3 => 70,   // Médio
                    _ => 50,   // Distante
                };
                score += unstoppable_bonus;
            }
        }
    }

    if is_enemy {
        -score
    } else {
        score
    }
}

/// Avalia regra do quadrado (versão original)
fn evaluate_square_rule(board: &Board, color: Color, our_king: u8, enemy_king: u8) -> i32 {
    evaluate_square_rule_advanced(board, color, our_king, enemy_king)
}

/// Avaliação aprimorada da atividade do rei em endgames
fn evaluate_king_activity_enhanced(board: &Board, color: Color, king_sq: u8) -> i32 {
    let mut activity_score = 0;
    let total_pieces = (board.white_pieces | board.black_pieces).count_ones();

    if total_pieces > 12 {
        return 0;
    }

    let king_file = king_sq % 8;
    let king_rank = king_sq / 8;

    // Centralização progressiva baseada no número de peças
    let centralization_weight = ((14 - total_pieces) as f32 / 14.0 * 5.0) as i32;
    let center_distance = ((king_file as f32 - 3.5).abs() + (king_rank as f32 - 3.5).abs()) as i32;
    activity_score += (7 - center_distance) * centralization_weight;

    // Proximidade a peões
    activity_score += evaluate_king_pawn_proximity(board, color, king_sq);

    // Atividade relativa ao rei inimigo
    let enemy_king = board.kings & if color == Color::White { board.black_pieces } else { board.white_pieces };
    if enemy_king != 0 {
        let enemy_king_sq = enemy_king.trailing_zeros() as u8;
        activity_score += evaluate_relative_king_activity(king_sq, enemy_king_sq);
    }

    activity_score.clamp(0, 60)
}

/// Avalia proximidade do rei aos peões
fn evaluate_king_pawn_proximity(board: &Board, color: Color, king_sq: u8) -> i32 {
    let mut score = 0;
    let enemy_pawns = board.pawns & if color == Color::White { board.black_pieces } else { board.white_pieces };
    let our_pawns = board.pawns & if color == Color::White { board.white_pieces } else { board.black_pieces };

    // Proximidade a peões inimigos (atacar)
    let mut min_enemy_distance = 8;
    let mut pawn_bb = enemy_pawns;
    while pawn_bb != 0 {
        let pawn_sq = pawn_bb.trailing_zeros() as u8;
        pawn_bb &= pawn_bb - 1;

        let distance = eval_utils::king_distance(king_sq, pawn_sq);
        min_enemy_distance = min_enemy_distance.min(distance);
    }

    if min_enemy_distance < 8 {
        score += (8 - min_enemy_distance as i32) * 3;
    }

    // Suporte a peões passados próprios
    let mut pawn_bb = our_pawns;
    while pawn_bb != 0 {
        let pawn_sq = pawn_bb.trailing_zeros() as u8;
        pawn_bb &= pawn_bb - 1;

        if eval_utils::is_passed_pawn(pawn_sq, color, our_pawns, enemy_pawns) {
            let distance = eval_utils::king_distance(king_sq, pawn_sq);

            if distance <= 1 {
                score += 20;
            } else if distance <= 3 {
                score += 10;
            }
        }
    }

    score
}

/// Avalia atividade relativa entre reis
fn evaluate_relative_king_activity(our_king: u8, enemy_king: u8) -> i32 {
    let our_rank = our_king / 8;
    let enemy_rank = enemy_king / 8;
    let our_file = our_king % 8;
    let enemy_file = enemy_king % 8;

    // Nosso rei mais centralizado = bom
    let our_center = ((our_file as i32 - 3).abs() + (our_rank as i32 - 3).abs()).min(
        ((our_file as i32 - 4).abs() + (our_rank as i32 - 4).abs())
    );
    let enemy_center = ((enemy_file as i32 - 3).abs() + (enemy_rank as i32 - 3).abs()).min(
        ((enemy_file as i32 - 4).abs() + (enemy_rank as i32 - 4).abs())
    );

    (enemy_center - our_center) * 3
}

/// Avalia corridas de promoção avançadas
fn evaluate_pawn_promotion_race_advanced(board: &Board, color: Color, depth: i32) -> i32 {
    let mut race_score = 0;
    let our_pawns = board.pawns & if color == Color::White { board.white_pieces } else { board.black_pieces };
    let enemy_pawns = board.pawns & if color == Color::White { board.black_pieces } else { board.white_pieces };

    // Analisa cada peão passado
    let our_passed = find_all_passed_pawns(board, color, our_pawns, enemy_pawns);
    let enemy_passed = find_all_passed_pawns(board, !color, enemy_pawns, our_pawns);

    // Avalia corridas
    for &our_pawn in &our_passed {
        for &enemy_pawn in &enemy_passed {
            race_score += evaluate_single_race(board, color, our_pawn, enemy_pawn);
        }
    }

    // Avalia peões passados sem oposição
    for &pawn in &our_passed {
        if enemy_passed.is_empty() {
            race_score += evaluate_unopposed_passer(board, color, pawn);
        }
    }

    race_score.clamp(-100, 100)
}

/// Encontra todos os peões passados
fn find_all_passed_pawns(board: &Board, color: Color, our_pawns: Bitboard, enemy_pawns: Bitboard) -> Vec<u8> {
    let mut passed = Vec::new();
    let mut pawn_bb = our_pawns;

    while pawn_bb != 0 {
        let pawn_sq = pawn_bb.trailing_zeros() as u8;
        pawn_bb &= pawn_bb - 1;

        if eval_utils::is_passed_pawn(pawn_sq, color, our_pawns, enemy_pawns) {
            passed.push(pawn_sq);
        }
    }

    passed
}

/// Avalia uma corrida individual
fn evaluate_single_race(board: &Board, color: Color, our_pawn: u8, enemy_pawn: u8) -> i32 {
    let our_distance = promotion_distance(our_pawn, color);
    let enemy_distance = promotion_distance(enemy_pawn, !color);

    let tempo = if board.to_move == color { 1 } else { 0 };

    if our_distance < enemy_distance + tempo {
        30 // Ganhamos a corrida
    } else if our_distance == enemy_distance + tempo {
        0 // Empate
    } else {
        -30 // Perdemos a corrida
    }
}

/// Calcula distância até promoção
fn promotion_distance(pawn_sq: u8, color: Color) -> i32 {
    let rank = pawn_sq / 8;
    if color == Color::White {
        (7 - rank) as i32
    } else {
        rank as i32
    }
}

/// Avalia peão passado sem oposição
fn evaluate_unopposed_passer(board: &Board, color: Color, pawn_sq: u8) -> i32 {
    let distance = promotion_distance(pawn_sq, color);
    let base_value = match distance {
        0 => 100,
        1 => 80,
        2 => 60,
        3 => 40,
        4 => 25,
        _ => 15,
    };

    // Bônus se rei suporta
    let our_king = board.kings & if color == Color::White { board.white_pieces } else { board.black_pieces };
    if our_king != 0 {
        let king_sq = our_king.trailing_zeros() as u8;
        let king_distance = eval_utils::king_distance(king_sq, pawn_sq);
        if king_distance <= 2 {
            return base_value + 20;
        }
    }

    base_value
}

/// Avalia corrida de promoção (versão original)
fn evaluate_pawn_promotion_race(board: &Board, color: Color) -> i32 {
    evaluate_pawn_promotion_race_advanced(board, color, 3)
}

/// Detecta padrões de fortaleza
fn evaluate_fortress_patterns(board: &Board, color: Color) -> i32 {
    let mut fortress_score = 0;
    let total_pieces = (board.white_pieces | board.black_pieces).count_ones();

    if total_pieces > 10 {
        return 0;
    }

    // Detecta diferentes tipos de fortaleza
    fortress_score += detect_bishop_fortress(board, color);
    fortress_score += detect_rook_fortress(board, color);
    fortress_score += detect_knight_fortress(board, color);
    fortress_score += detect_opposite_bishops_fortress(board, color);

    fortress_score.clamp(0, 60)
}

/// Detecta fortaleza de bispo
fn detect_bishop_fortress(board: &Board, color: Color) -> i32 {
    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let our_bishops = board.bishops & our_pieces;

    if our_bishops == 0 {
        return 0;
    }

    let bishop_sq = our_bishops.trailing_zeros() as u8;
    let bishop_on_light = (bishop_sq / 8 + bishop_sq % 8) % 2 == 0;

    // Verifica se temos peões nas casas certas
    let our_pawns = board.pawns & our_pieces;
    let mut correct_pawns = 0;
    let mut blocked_pawns = 0;

    let mut pawns_bb = our_pawns;
    while pawns_bb != 0 {
        let pawn_sq = pawns_bb.trailing_zeros() as u8;
        pawns_bb &= pawns_bb - 1;

        let pawn_on_light = (pawn_sq / 8 + pawn_sq % 8) % 2 == 0;
        if pawn_on_light != bishop_on_light {
            correct_pawns += 1;

            // Verifica se peão está bloqueado
            let advance_sq = if color == Color::White {
                pawn_sq + 8
            } else {
                pawn_sq.saturating_sub(8)
            };

            if advance_sq < 64 && ((1u64 << advance_sq) & (board.white_pieces | board.black_pieces)) != 0 {
                blocked_pawns += 1;
            }
        }
    }

    if correct_pawns >= 3 && blocked_pawns >= 2 {
        25 // Fortaleza bem estabelecida
    } else if correct_pawns >= 2 {
        15 // Fortaleza parcial
    } else {
        0
    }
}

/// Detecta fortaleza de torre
fn detect_rook_fortress(board: &Board, color: Color) -> i32 {
    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let enemy_pieces = if color == Color::White { board.black_pieces } else { board.white_pieces };
    let our_rooks = board.rooks & our_pieces;

    if our_rooks == 0 {
        return 0;
    }

    let rook_sq = our_rooks.trailing_zeros() as u8;
    let rook_rank = rook_sq / 8;

    let enemy_king = board.kings & enemy_pieces;
    if enemy_king != 0 {
        let enemy_king_sq = enemy_king.trailing_zeros() as u8;
        let enemy_king_rank = enemy_king_sq / 8;

        // Torre na 7ª/2ª fileira cortando rei
        if color == Color::White && rook_rank == 6 && enemy_king_rank == 7 {
            return 30;
        } else if color == Color::Black && rook_rank == 1 && enemy_king_rank == 0 {
            return 30;
        }

        // Torre lateral cortando rei
        let rook_file = rook_sq % 8;
        let enemy_king_file = enemy_king_sq % 8;
        if (rook_file == 0 || rook_file == 7) &&
            (enemy_king_file >= 5 || enemy_king_file <= 2) {
            return 20;
        }
    }

    0
}

/// Detecta fortaleza de cavalo
fn detect_knight_fortress(board: &Board, color: Color) -> i32 {
    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let our_knights = board.knights & our_pieces;

    if our_knights == 0 {
        return 0;
    }

    let knight_sq = our_knights.trailing_zeros() as u8;
    let knight_file = knight_sq % 8;
    let knight_rank = knight_sq / 8;

    // Cavalo em outpost forte defendido
    if eval_utils::is_center_square(knight_sq) || eval_utils::is_extended_center_square(knight_sq) {
        // Verifica se está defendido por peão
        let our_pawns = board.pawns & our_pieces;
        let defended = is_defended_by_pawn(knight_sq, color, our_pawns);

        if defended {
            // Verifica se não pode ser atacado por peões inimigos
            let enemy_pawns = board.pawns & if color == Color::White { board.black_pieces } else { board.white_pieces };
            let can_be_attacked = can_be_attacked_by_enemy_pawns(knight_sq, color, enemy_pawns);

            if !can_be_attacked {
                return 25; // Cavalo eterno
            }
        }
    }

    0
}

/// Detecta fortaleza de bispos de cores opostas
fn detect_opposite_bishops_fortress(board: &Board, color: Color) -> i32 {
    let white_bishops = board.bishops & board.white_pieces;
    let black_bishops = board.bishops & board.black_pieces;

    if white_bishops.count_ones() != 1 || black_bishops.count_ones() != 1 {
        return 0;
    }

    let white_bishop_sq = white_bishops.trailing_zeros() as u8;
    let black_bishop_sq = black_bishops.trailing_zeros() as u8;

    let white_on_light = (white_bishop_sq / 8 + white_bishop_sq % 8) % 2 == 0;
    let black_on_light = (black_bishop_sq / 8 + black_bishop_sq % 8) % 2 == 0;

    if white_on_light != black_on_light {
        // Bispos de cores opostas - tendência para empate
        let material_imbalance = calculate_material_imbalance(board);

        if material_imbalance < 200 {
            return 30; // Forte tendência para empate
        } else if material_imbalance < 400 {
            return 15; // Alguma tendência para empate
        }
    }

    0
}

/// Calcula desequilíbrio material
fn calculate_material_imbalance(board: &Board) -> i32 {
    let white_material = crate::evaluation::material::evaluate_material_only(board);
    white_material.abs()
}

/// Verifica se está defendido por peão
fn is_defended_by_pawn(square: u8, color: Color, our_pawns: Bitboard) -> bool {
    let file = square % 8;
    let rank = square / 8;

    let defense_rank = if color == Color::White {
        rank.saturating_sub(1)
    } else {
        (rank + 1).min(7)
    };

    if file > 0 {
        let left_pawn = defense_rank * 8 + file - 1;
        if (1u64 << left_pawn) & our_pawns != 0 {
            return true;
        }
    }

    if file < 7 {
        let right_pawn = defense_rank * 8 + file + 1;
        if (1u64 << right_pawn) & our_pawns != 0 {
            return true;
        }
    }

    false
}

/// Verifica se pode ser atacado por peões inimigos
fn can_be_attacked_by_enemy_pawns(square: u8, our_color: Color, enemy_pawns: Bitboard) -> bool {
    let file = square % 8;
    let rank = square / 8;

    // Verifica todas as fileiras futuras de onde peões inimigos podem vir
    for enemy_rank in 0..8 {
        for adj_file in (file.saturating_sub(1))..=(file + 1).min(7) {
            if adj_file == file {
                continue;
            }

            let pawn_sq = enemy_rank * 8 + adj_file;
            if (1u64 << pawn_sq) & enemy_pawns != 0 {
                // Verifica se este peão pode eventualmente atacar nossa casa
                if our_color == Color::White && enemy_rank < rank {
                    return true;
                } else if our_color == Color::Black && enemy_rank > rank {
                    return true;
                }
            }
        }
    }

    false
}

/// Avaliação específica para finais de torre
fn evaluate_rook_endgame_patterns(board: &Board, color: Color) -> i32 {
    let mut rook_score = 0;

    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let enemy_pieces = if color == Color::White { board.black_pieces } else { board.white_pieces };

    let our_rooks = board.rooks & our_pieces;
    let enemy_rooks = board.rooks & enemy_pieces;

    if our_rooks == 0 {
        return 0;
    }

    // Avalia cada torre nossa
    let mut rook_bb = our_rooks;
    while rook_bb != 0 {
        let rook_sq = rook_bb.trailing_zeros() as u8;
        rook_bb &= rook_bb - 1;

        // Torre ativa
        rook_score += evaluate_rook_activity(rook_sq, color);

        // Torre atrás de peão passado
        rook_score += evaluate_rook_behind_passer(board, rook_sq, color);

        // Torre cortando rei
        if enemy_rooks == 0 {
            rook_score += evaluate_rook_cutting_king(board, rook_sq, color);
        }
    }

    // Finais específicos de torre
    if our_rooks.count_ones() == 1 && enemy_rooks.count_ones() == 1 {
        rook_score += evaluate_rook_vs_rook_endgame(board, color);
    }

    rook_score.clamp(0, 80)
}

/// Avalia atividade da torre
fn evaluate_rook_activity(rook_sq: u8, color: Color) -> i32 {
    let rook_rank = rook_sq / 8;
    let rook_file = rook_sq % 8;

    let mut score = 0;

    // Torre na 7ª/2ª fileira
    if (color == Color::White && rook_rank == 6) || (color == Color::Black && rook_rank == 1) {
        score += 25;
    }

    // Torre em coluna aberta (simplificado)
    score += 10;

    score
}

/// Avalia torre atrás de peão passado
fn evaluate_rook_behind_passer(board: &Board, rook_sq: u8, color: Color) -> i32 {
    let rook_file = rook_sq % 8;
    let rook_rank = rook_sq / 8;

    let our_pawns = board.pawns & if color == Color::White { board.white_pieces } else { board.black_pieces };
    let enemy_pawns = board.pawns & if color == Color::White { board.black_pieces } else { board.white_pieces };

    // Verifica peões na mesma coluna
    let file_mask = eval_utils::get_file_mask_from_file(rook_file);
    let file_pawns = our_pawns & file_mask;

    let mut pawn_bb = file_pawns;
    while pawn_bb != 0 {
        let pawn_sq = pawn_bb.trailing_zeros() as u8;
        pawn_bb &= pawn_bb - 1;

        let pawn_rank = pawn_sq / 8;

        if eval_utils::is_passed_pawn(pawn_sq, color, our_pawns, enemy_pawns) {
            // Torre atrás do peão passado
            if (color == Color::White && rook_rank < pawn_rank) ||
                (color == Color::Black && rook_rank > pawn_rank) {
                return 35;
            }
        }
    }

    0
}

/// Avalia torre cortando rei
fn evaluate_rook_cutting_king(board: &Board, rook_sq: u8, color: Color) -> i32 {
    let enemy_king = board.kings & if color == Color::White { board.black_pieces } else { board.white_pieces };

    if enemy_king == 0 {
        return 0;
    }

    let enemy_king_sq = enemy_king.trailing_zeros() as u8;
    let rook_file = rook_sq % 8;
    let rook_rank = rook_sq / 8;
    let king_file = enemy_king_sq % 8;
    let king_rank = enemy_king_sq / 8;

    // Torre na mesma linha ou coluna que o rei
    if rook_file == king_file || rook_rank == king_rank {
        return 20;
    }

    0
}

/// Avalia finais torre vs torre
fn evaluate_rook_vs_rook_endgame(board: &Board, color: Color) -> i32 {
    // Implementação simplificada
    // Em produção, usaria tablebases ou análise mais profunda
    0
}

/// Avalia padrões de finais de bispo
fn evaluate_bishop_endgame_patterns(board: &Board, color: Color) -> i32 {
    let mut score = 0;
    let our_bishops = board.bishops & if color == Color::White { board.white_pieces } else { board.black_pieces };

    if our_bishops == 0 {
        return 0;
    }

    // Bispos da mesma cor vs cores opostas
    let white_bishops = board.bishops & board.white_pieces;
    let black_bishops = board.bishops & board.black_pieces;

    if white_bishops.count_ones() == 1 && black_bishops.count_ones() == 1 {
        score += evaluate_bishop_color_complex(white_bishops, black_bishops);
    }

    // Bispo bom vs bispo mau
    score += evaluate_good_bad_bishop(board, color);

    score.clamp(-20, 40)
}

/// Avalia complexo de cores dos bispos
fn evaluate_bishop_color_complex(white_bishops: Bitboard, black_bishops: Bitboard) -> i32 {
    let white_bishop_sq = white_bishops.trailing_zeros() as u8;
    let black_bishop_sq = black_bishops.trailing_zeros() as u8;

    let white_on_light = (white_bishop_sq / 8 + white_bishop_sq % 8) % 2 == 0;
    let black_on_light = (black_bishop_sq / 8 + black_bishop_sq % 8) % 2 == 0;

    if white_on_light == black_on_light {
        0 // Mesma cor - normal
    } else {
        -10 // Cores opostas - tendência empate
    }
}

/// Avalia bispo bom vs mau
fn evaluate_good_bad_bishop(board: &Board, color: Color) -> i32 {
    let our_bishops = board.bishops & if color == Color::White { board.white_pieces } else { board.black_pieces };
    let our_pawns = board.pawns & if color == Color::White { board.white_pieces } else { board.black_pieces };

    if our_bishops == 0 {
        return 0;
    }

    let bishop_sq = our_bishops.trailing_zeros() as u8;
    let bishop_on_light = (bishop_sq / 8 + bishop_sq % 8) % 2 == 0;

    // Conta peões na cor do bispo (maus) vs cor oposta (bons)
    let mut pawns_on_bishop_color = 0;
    let mut pawns_on_opposite_color = 0;

    let mut pawn_bb = our_pawns;
    while pawn_bb != 0 {
        let pawn_sq = pawn_bb.trailing_zeros() as u8;
        pawn_bb &= pawn_bb - 1;

        let pawn_on_light = (pawn_sq / 8 + pawn_sq % 8) % 2 == 0;
        if pawn_on_light == bishop_on_light {
            pawns_on_bishop_color += 1;
        } else {
            pawns_on_opposite_color += 1;
        }
    }

    // Bônus por bispo "bom" (poucos peões na sua cor)
    let goodness = pawns_on_opposite_color - pawns_on_bishop_color;
    goodness * 5
}

/// Avalia padrões de finais de cavalo
fn evaluate_knight_endgame_patterns(board: &Board, color: Color) -> i32 {
    let mut score = 0;
    let our_knights = board.knights & if color == Color::White { board.white_pieces } else { board.black_pieces };

    if our_knights == 0 {
        return 0;
    }

    // Cavalos são melhores em posições fechadas
    let pawn_count = board.pawns.count_ones();
    if pawn_count > 10 {
        score += 10; // Posição fechada favorece cavalos
    }

    // Avalia cada cavalo
    let mut knight_bb = our_knights;
    while knight_bb != 0 {
        let knight_sq = knight_bb.trailing_zeros() as u8;
        knight_bb &= knight_bb - 1;

        // Cavalo centralizado em endgame
        if eval_utils::is_center_square(knight_sq) {
            score += 15;
        } else if eval_utils::is_extended_center_square(knight_sq) {
            score += 8;
        }

        // Cavalo em outpost
        let enemy_pawns = board.pawns & if color == Color::White { board.black_pieces } else { board.white_pieces };
        if eval_utils::is_outpost(knight_sq, color, enemy_pawns) {
            score += 20;
        }
    }

    score.clamp(0, 50)
}

/// Avalia padrões de finais de rainha
fn evaluate_queen_endgame_patterns(board: &Board, color: Color) -> i32 {
    let mut score = 0;
    let our_queens = board.queens & if color == Color::White { board.white_pieces } else { board.black_pieces };

    if our_queens == 0 {
        return 0;
    }

    // Rainha ativa em endgame
    let queen_sq = our_queens.trailing_zeros() as u8;

    // Mobilidade da rainha (simplificado)
    score += 10;

    // Rainha apoiando peões passados
    let our_pawns = board.pawns & if color == Color::White { board.white_pieces } else { board.black_pieces };
    let enemy_pawns = board.pawns & if color == Color::White { board.black_pieces } else { board.white_pieces };

    let passed_pawns = count_passed_pawns(our_pawns, enemy_pawns, color);
    if passed_pawns > 0 {
        score += passed_pawns * 10;
    }

    score.clamp(0, 40)
}

/// Conta peões passados
fn count_passed_pawns(our_pawns: Bitboard, enemy_pawns: Bitboard, color: Color) -> i32 {
    let mut count = 0;
    let mut pawn_bb = our_pawns;

    while pawn_bb != 0 {
        let pawn_sq = pawn_bb.trailing_zeros() as u8;
        pawn_bb &= pawn_bb - 1;

        if eval_utils::is_passed_pawn(pawn_sq, color, our_pawns, enemy_pawns) {
            count += 1;
        }
    }

    count
}

/// Avalia padrões de breakthrough (rompimento de peões)
fn evaluate_breakthrough_patterns(board: &Board, color: Color) -> i32 {
    let mut score = 0;
    let our_pawns = board.pawns & if color == Color::White { board.white_pieces } else { board.black_pieces };
    let enemy_pawns = board.pawns & if color == Color::White { board.black_pieces } else { board.white_pieces };

    // Detecta potencial de breakthrough
    score += detect_pawn_breakthrough(our_pawns, enemy_pawns, color);

    // Detecta maiorias de peões
    score += evaluate_pawn_majorities(our_pawns, enemy_pawns, color);

    score.clamp(-30, 50)
}

/// Detecta potencial de breakthrough de peões
fn detect_pawn_breakthrough(our_pawns: Bitboard, enemy_pawns: Bitboard, color: Color) -> i32 {
    let mut score = 0;

    // Verifica configurações típicas de breakthrough
    // (implementação simplificada - em produção seria mais complexa)

    // Conta peões avançados em grupo
    let advanced_rank = if color == Color::White { 5 } else { 2 }; // 6ª/3ª fileira
    let rank_mask = 0xFFu64 << (advanced_rank * 8);
    let advanced_pawns = (our_pawns & rank_mask).count_ones();

    if advanced_pawns >= 2 {
        // Verifica se há potencial de breakthrough
        let next_rank = if color == Color::White { 6 } else { 1 };
        let next_rank_mask = 0xFFu64 << (next_rank * 8);
        let blocking_pawns = (enemy_pawns & next_rank_mask).count_ones();

        if advanced_pawns > blocking_pawns {
            score += 20; // Potencial breakthrough
        }
    }

    score
}

/// Avalia maiorias de peões
fn evaluate_pawn_majorities(our_pawns: Bitboard, enemy_pawns: Bitboard, color: Color) -> i32 {
    let mut score = 0;

    // Divide o tabuleiro em alas
    const QUEENSIDE: Bitboard = 0x0707070707070707; // Colunas a-c
    const CENTER: Bitboard = 0x1818181818181818;    // Colunas d-e
    const KINGSIDE: Bitboard = 0xE0E0E0E0E0E0E0E0; // Colunas f-h

    // Conta peões por setor
    let our_queenside = (our_pawns & QUEENSIDE).count_ones() as i32;
    let enemy_queenside = (enemy_pawns & QUEENSIDE).count_ones() as i32;

    let our_kingside = (our_pawns & KINGSIDE).count_ones() as i32;
    let enemy_kingside = (enemy_pawns & KINGSIDE).count_ones() as i32;

    // Bônus por maioria de peões
    if our_queenside > enemy_queenside && enemy_queenside == 0 {
        score += 15; // Maioria sem oposição
    } else if our_queenside > enemy_queenside {
        score += 8; // Maioria com oposição
    }

    if our_kingside > enemy_kingside && enemy_kingside == 0 {
        score += 15;
    } else if our_kingside > enemy_kingside {
        score += 8;
    }

    score
}

/// Função para avaliação pública da atividade do rei
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

    // Normaliza o score
    activity_score.clamp(-50, 50)
}
