// Padrões avançados de endgame com foco em finais práticos
use crate::{board::Board, types::{Color, Bitboard, PieceKind}};
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
    pub king_activity: i32,
    pub pawn_promotion_race: i32,
    pub fortress_patterns: i32,
    pub rook_endgame_bonus: i32,
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
            king_activity: 0,
            pawn_promotion_race: 0,
            fortress_patterns: 0,
            rook_endgame_bonus: 0,
        }
    }

    pub fn total_score(&self) -> i32 {
        self.king_opposition + self.distant_opposition + self.triangulation +
            self.outflanking + self.zugzwang_potential + self.square_rule +
            self.king_activity + self.pawn_promotion_race + self.fortress_patterns +
            self.rook_endgame_bonus
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

    // Nova avaliação: Atividade do rei
    patterns.king_activity = evaluate_king_activity_enhanced(board, color, our_king_sq);

    // Nova avaliação: Corrida de promoção de peões
    patterns.pawn_promotion_race = evaluate_pawn_promotion_race(board, color);

    // Nova avaliação: Padrões de fortaleza
    patterns.fortress_patterns = evaluate_fortress_patterns(board, color);

    // Nova avaliação: Bônus específicos para finais de torre
    patterns.rook_endgame_bonus = evaluate_rook_endgame_patterns(board, color);

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

    // Oposição direta (horizontal/vertical)
    if (file_diff == 0 && rank_diff == 2) || (file_diff == 2 && rank_diff == 0) {
        return 25; // Oposição direta é muito valiosa
    }

    // Oposição diagonal
    if file_diff == 2 && rank_diff == 2 {
        return 20; // Oposição diagonal também é valiosa
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
        let enemy_pawns = board.pawns & if !color == Color::White { board.white_pieces } else { board.black_pieces };
        let our_pawns = board.pawns & if color == Color::White { board.white_pieces } else { board.black_pieces };
        if super::utils::is_passed_pawn(pawn_sq, !color, our_pawns, enemy_pawns) {
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

            let king_distance = super::utils::king_distance(our_king, promotion_sq) as i32;

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

/// Avaliação aprimorada da atividade do rei em endgames
fn evaluate_king_activity_enhanced(board: &Board, color: Color, king_sq: u8) -> i32 {
    let mut activity_score = 0;
    let total_pieces = (board.white_pieces | board.black_pieces).count_ones();
    
    // Só é relevante em endgames (≤12 peças)
    if total_pieces > 12 {
        return 0;
    }

    let king_file = king_sq % 8;
    let king_rank = king_sq / 8;
    
    // Atividade baseada na centralização (mais importante no endgame)
    let center_distance = ((king_file as f32 - 3.5).abs() + (king_rank as f32 - 3.5).abs()) as i32;
    activity_score += (7 - center_distance) * 3; // Máximo +21
    
    // Bônus por estar próximo de peões inimigos (pressão)
    let enemy_pawns = board.pawns & if color == Color::White { board.black_pieces } else { board.white_pieces };
    let mut min_pawn_distance = 8;
    
    let mut pawn_bb = enemy_pawns;
    while pawn_bb != 0 {
        let pawn_sq = pawn_bb.trailing_zeros() as u8;
        pawn_bb &= pawn_bb - 1;
        
        let pawn_file = pawn_sq % 8;
        let pawn_rank = pawn_sq / 8;
        let distance = ((king_file as i32 - pawn_file as i32).abs() + (king_rank as i32 - pawn_rank as i32).abs()) as u8;
        min_pawn_distance = min_pawn_distance.min(distance);
    }
    
    if min_pawn_distance < 8 {
        activity_score += (8 - min_pawn_distance as i32) * 2; // Máximo +14
    }
    
    // Bônus por suporte aos próprios peões passados
    let our_pawns = board.pawns & if color == Color::White { board.white_pieces } else { board.black_pieces };
    let mut pawn_bb = our_pawns;
    while pawn_bb != 0 {
        let pawn_sq = pawn_bb.trailing_zeros() as u8;
        pawn_bb &= pawn_bb - 1;
        
        let enemy_pawns = board.pawns & if color == Color::White { board.black_pieces } else { board.white_pieces };
        let our_pawns = board.pawns & if color == Color::White { board.white_pieces } else { board.black_pieces };
        if super::utils::is_passed_pawn(pawn_sq, color, our_pawns, enemy_pawns) {
            let pawn_file = pawn_sq % 8;
            let pawn_rank = pawn_sq / 8;
            let distance = ((king_file as i32 - pawn_file as i32).abs() + (king_rank as i32 - pawn_rank as i32).abs()) as u8;
            
            if distance <= 2 {
                activity_score += 15; // Rei próximo de peão passado
            } else if distance <= 4 {
                activity_score += 8;
            }
        }
    }
    
    activity_score.clamp(0, 40)
}

/// Avalia corridas de promoção de peões
fn evaluate_pawn_promotion_race(board: &Board, color: Color) -> i32 {
    let mut race_score = 0;
    let our_pawns = board.pawns & if color == Color::White { board.white_pieces } else { board.black_pieces };
    let enemy_pawns = board.pawns & if color == Color::White { board.black_pieces } else { board.white_pieces };
    
    let our_king = board.kings & if color == Color::White { board.white_pieces } else { board.black_pieces };
    let enemy_king = board.kings & if color == Color::White { board.black_pieces } else { board.white_pieces };
    
    if our_king == 0 || enemy_king == 0 {
        return 0;
    }
    
    let our_king_sq = our_king.trailing_zeros() as u8;
    let enemy_king_sq = enemy_king.trailing_zeros() as u8;
    
    // Analisa nossos peões passados
    let mut our_pawns_bb = our_pawns;
    while our_pawns_bb != 0 {
        let pawn_sq = our_pawns_bb.trailing_zeros() as u8;
        our_pawns_bb &= our_pawns_bb - 1;
        
        let enemy_pawns = board.pawns & if color == Color::White { board.black_pieces } else { board.white_pieces };
        let our_pawns = board.pawns & if color == Color::White { board.white_pieces } else { board.black_pieces };
        if super::utils::is_passed_pawn(pawn_sq, color, our_pawns, enemy_pawns) {
            let pawn_rank = pawn_sq / 8;
            let promotion_sq = if color == Color::White {
                pawn_sq + (7 - pawn_rank) * 8 // a8, b8, etc.
            } else {
                pawn_sq - pawn_rank * 8 // a1, b1, etc.
            };
            
            let moves_to_promote = if color == Color::White { 7 - pawn_rank } else { pawn_rank };
            let our_king_distance = super::utils::king_distance(our_king_sq, promotion_sq);
            let enemy_king_distance = super::utils::king_distance(enemy_king_sq, promotion_sq);
            
            // Se nosso rei consegue promover antes do inimigo chegar
            if our_king_distance + moves_to_promote < enemy_king_distance {
                race_score += 30; // Corrida ganha
            } else if our_king_distance + moves_to_promote == enemy_king_distance {
                // Empate - depende de quem tem a vez
                if board.to_move == color {
                    race_score += 15; // Temos a vez, vantagem
                } else {
                    race_score += 5; // Não temos a vez, desvantagem
                }
            }
        }
    }
    
    // Analisa peões passados inimigos (penalidade)
    let mut enemy_pawns_bb = enemy_pawns;
    while enemy_pawns_bb != 0 {
        let pawn_sq = enemy_pawns_bb.trailing_zeros() as u8;
        enemy_pawns_bb &= enemy_pawns_bb - 1;
        
        let enemy_pawns = board.pawns & if !color == Color::White { board.white_pieces } else { board.black_pieces };
        let our_pawns = board.pawns & if color == Color::White { board.white_pieces } else { board.black_pieces };
        if super::utils::is_passed_pawn(pawn_sq, !color, our_pawns, enemy_pawns) {
            let pawn_rank = pawn_sq / 8;
            let promotion_sq = if color == Color::Black { // Inimigo é White
                pawn_sq + (7 - pawn_rank) * 8
            } else { // Inimigo é Black
                pawn_sq - pawn_rank * 8
            };
            
            let moves_to_promote = if color == Color::Black { 7 - pawn_rank } else { pawn_rank };
            let our_king_distance = super::utils::king_distance(our_king_sq, promotion_sq);
            let enemy_king_distance = super::utils::king_distance(enemy_king_sq, promotion_sq);
            
            // Se inimigo consegue promover antes de nós chegarmos
            if enemy_king_distance + moves_to_promote < our_king_distance {
                race_score -= 25; // Perdemos a corrida
            }
        }
    }
    
    race_score.clamp(-50, 50)
}

/// Detecta padrões de fortaleza (posições defensivas)
fn evaluate_fortress_patterns(board: &Board, color: Color) -> i32 {
    let mut fortress_score = 0;
    let total_pieces = (board.white_pieces | board.black_pieces).count_ones();
    
    // Fortalezas são mais comuns em endgames simples
    if total_pieces > 8 {
        return 0;
    }
    
    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let enemy_pieces = if color == Color::White { board.black_pieces } else { board.white_pieces };
    
    // Fortaleza de bispo + peões na diagonal certa
    let our_bishops = board.bishops & our_pieces;
    if our_bishops != 0 {
        let bishop_sq = our_bishops.trailing_zeros() as u8;
        let bishop_on_light = (bishop_sq / 8 + bishop_sq % 8) % 2 == 0;
        
        // Verifica se temos peões nas casas da cor certa
        let our_pawns = board.pawns & our_pieces;
        let mut pawn_fortress_count = 0;
        
        let mut pawns_bb = our_pawns;
        while pawns_bb != 0 {
            let pawn_sq = pawns_bb.trailing_zeros() as u8;
            pawns_bb &= pawns_bb - 1;
            
            let pawn_on_light = (pawn_sq / 8 + pawn_sq % 8) % 2 == 0;
            if pawn_on_light == bishop_on_light {
                pawn_fortress_count += 1;
            }
        }
        
        if pawn_fortress_count >= 2 {
            fortress_score += 20; // Fortaleza de bispo bem estabelecida
        }
    }
    
    // Fortaleza de torre na 7ª/2ª fileira
    let our_rooks = board.rooks & our_pieces;
    if our_rooks != 0 {
        let rook_sq = our_rooks.trailing_zeros() as u8;
        let rook_rank = rook_sq / 8;
        
        let enemy_king = board.kings & enemy_pieces;
        if enemy_king != 0 {
            let enemy_king_sq = enemy_king.trailing_zeros() as u8;
            let enemy_king_rank = enemy_king_sq / 8;
            
            // Torre na 7ª fileira com rei inimigo na 8ª
            if color == Color::White && rook_rank == 6 && enemy_king_rank == 7 {
                fortress_score += 25;
            } else if color == Color::Black && rook_rank == 1 && enemy_king_rank == 0 {
                fortress_score += 25;
            }
        }
    }
    
    fortress_score.clamp(0, 45)
}

/// Avaliação específica para finais de torre
fn evaluate_rook_endgame_patterns(board: &Board, color: Color) -> i32 {
    let mut rook_score = 0;
    
    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let enemy_pieces = if color == Color::White { board.black_pieces } else { board.white_pieces };
    
    let our_rooks = board.rooks & our_pieces;
    let enemy_rooks = board.rooks & enemy_pieces;
    
    // Só é relevante se há torres
    if our_rooks == 0 {
        return 0;
    }
    
    let our_rook_sq = our_rooks.trailing_zeros() as u8;
    let our_rook_file = our_rook_sq % 8;
    let our_rook_rank = our_rook_sq / 8;
    
    // Torre ativa (7ª/2ª fileira)
    if (color == Color::White && our_rook_rank == 6) || (color == Color::Black && our_rook_rank == 1) {
        rook_score += 20;
    }
    
    // Torre atrás de peão passado próprio
    let our_pawns = board.pawns & our_pieces;
    let mut pawns_bb = our_pawns;
    while pawns_bb != 0 {
        let pawn_sq = pawns_bb.trailing_zeros() as u8;
        pawns_bb &= pawns_bb - 1;
        
        let enemy_pawns = board.pawns & if color == Color::White { board.black_pieces } else { board.white_pieces };
        let our_pawns = board.pawns & if color == Color::White { board.white_pieces } else { board.black_pieces };
        if super::utils::is_passed_pawn(pawn_sq, color, our_pawns, enemy_pawns) {
            let pawn_file = pawn_sq % 8;
            let pawn_rank = pawn_sq / 8;
            
            // Torre atrás do peão passado (muito bom)
            if our_rook_file == pawn_file {
                if (color == Color::White && our_rook_rank < pawn_rank) ||
                   (color == Color::Black && our_rook_rank > pawn_rank) {
                    rook_score += 30;
                }
            }
        }
    }
    
    // Torre cortando rei inimigo
    if enemy_rooks == 0 { // Só vale se inimigo não tem torre
        let enemy_king = board.kings & enemy_pieces;
        if enemy_king != 0 {
            let enemy_king_sq = enemy_king.trailing_zeros() as u8;
            let enemy_king_file = enemy_king_sq % 8;
            let enemy_king_rank = enemy_king_sq / 8;
            
            // Torre na mesma fileira ou coluna que o rei (pressão)
            if our_rook_file == enemy_king_file || our_rook_rank == enemy_king_rank {
                rook_score += 15;
            }
        }
    }
    
    rook_score.clamp(0, 65)
}

