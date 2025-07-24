// Mobilidade e estratégia avançada de cavalos
use crate::types::{Color, Bitboard};
use super::{MobilityContext, MobilityResult, GamePhase, utils::*};
use crate::evaluation::utils as eval_utils;
use super::super::pawn_structure::get_set_bits_simple;

/// Pesos para diferentes aspectos da estratégia de cavalos
#[derive(Debug, Clone, Copy)]
pub struct KnightWeights {
    pub mobility_per_square: [i32; 3],    // [opening, middlegame, endgame]
    pub safe_mobility_bonus: [i32; 3],    // Bônus por mobilidade segura
    pub outpost_bonus: [i32; 3],          // Bônus por outpost
    pub central_knight: [i32; 3],         // Bônus por cavalo central
    pub edge_penalty: [i32; 3],           // Penalidade por proximidade da borda
    pub fork_threat: [i32; 3],            // Bônus por ameaças de garfo
    pub defender_bonus: [i32; 3],         // Bônus por defender peças importantes
    pub blockade_bonus: [i32; 3],         // Bônus por bloquear peões passados
}

impl Default for KnightWeights {
    fn default() -> Self {
        KnightWeights {
            mobility_per_square: [4, 4, 3],      // Menos importante no final
            safe_mobility_bonus: [2, 3, 2],      // Pico no meio-jogo
            outpost_bonus: [15, 25, 20],         // Muito valioso no meio-jogo
            central_knight: [10, 15, 8],         // Centralização importante
            edge_penalty: [-5, -8, -3],          // Evitar bordas no meio-jogo
            fork_threat: [8, 15, 10],            // Táticas valiosas
            defender_bonus: [5, 8, 4],           // Defesa importante
            blockade_bonus: [8, 12, 20],        // Muito mais valioso no final
        }
    }
}

/// Resultado específico da análise de cavalos
#[derive(Debug, Clone, Copy)]
pub struct KnightAnalysis {
    pub mobility_score: i32,
    pub outpost_knights: i32,
    pub central_knights: i32,
    pub edge_penalty: i32,
    pub fork_threats: i32,
    pub defensive_value: i32,
    pub blockade_value: i32,
    pub tactical_opportunities: i32,
    pub mate_potential: i32,
}

impl KnightAnalysis {
    pub fn new() -> Self {
        KnightAnalysis {
            mobility_score: 0,
            outpost_knights: 0,
            central_knights: 0,
            edge_penalty: 0,
            fork_threats: 0,
            defensive_value: 0,
            blockade_value: 0,
            tactical_opportunities: 0,
            mate_potential: 0,
        }
    }
}

/// Avaliação avançada de mobilidade de cavalos
pub fn evaluate_knight_mobility_advanced(context: &MobilityContext) -> i32 {
    let weights = KnightWeights::default();
    let phase_idx = match context.phase {
        GamePhase::Opening => 0,
        GamePhase::MiddleGame => 1,
        GamePhase::Endgame => 2,
    };

    let analysis = analyze_knight_strategy(context);

    // Aplica pesos baseados na fase do jogo
    let mut score = 0;
    score += analysis.mobility_score * weights.mobility_per_square[phase_idx];
    score += analysis.outpost_knights * weights.outpost_bonus[phase_idx];
    score += analysis.central_knights * weights.central_knight[phase_idx];
    score += analysis.edge_penalty * weights.edge_penalty[phase_idx];
    score += analysis.fork_threats * weights.fork_threat[phase_idx];
    score += analysis.defensive_value * weights.defender_bonus[phase_idx];
    score += analysis.blockade_value * weights.blockade_bonus[phase_idx];
    score += analysis.tactical_opportunities;
    score += analysis.mate_potential; // Bônus direto para potencial de mate

    score
}

/// Análise completa da estratégia de cavalos
pub fn analyze_knight_strategy(context: &MobilityContext) -> KnightAnalysis {
    let mut analysis = KnightAnalysis::new();
    let knights = context.board.knights & context.our_pieces;

    if knights == 0 {
        return analysis; // Não há cavalos para analisar
    }

    let knight_squares = get_set_bits(knights);

    for &knight_sq in &knight_squares {
        // Análise de mobilidade básica
        let (mobility, safe_mobility) = calculate_knight_mobility(knight_sq, context);
        analysis.mobility_score += mobility + safe_mobility;

        // Verifica se está em outpost
        if is_outpost(knight_sq, context.color, context) {
            analysis.outpost_knights += 1;

            // Bônus extra se outpost está próximo ao rei inimigo
            if is_near_enemy_king(knight_sq, context) {
                analysis.outpost_knights += 1;
            }
        }

        // Verifica centralização
        if is_central_square(knight_sq) {
            analysis.central_knights += 1;
        } else if is_extended_center(knight_sq) {
            analysis.central_knights += 1; // Meio bônus para centro expandido
        }

        // Penalidade por proximidade da borda
        let edge_distance = calculate_edge_distance(knight_sq);
        if edge_distance <= 1 {
            analysis.edge_penalty += edge_distance - 2; // Penalidade negativa
        }

        // Detecta ameaças de garfo
        analysis.fork_threats += detect_fork_threats(knight_sq, context);

        // Avalia valor defensivo
        analysis.defensive_value += calculate_defensive_value(knight_sq, context);

        // Verifica bloqueio de peões passados
        analysis.blockade_value += evaluate_pawn_blockade(knight_sq, context);

        // Oportunidades táticas diversas
        analysis.tactical_opportunities += evaluate_tactical_opportunities(knight_sq, context);

        // Bônus especial para cavalos que atacam o rei inimigo em endgame
        if matches!(context.phase, GamePhase::Endgame) {
            if is_near_enemy_king(knight_sq, context) {
                analysis.mate_potential += 10;
            }
        }
    }

    analysis
}

/// Calcula mobilidade básica e segura do cavalo
fn calculate_knight_mobility(knight_sq: u8, context: &MobilityContext) -> (i32, i32) {
    let attacks = crate::moves::knight::get_knight_attacks_lookup(knight_sq);

    // Casas legais (não ocupadas por peças próprias)
    let legal_squares = attacks & !context.our_pieces;
    let total_mobility = legal_squares.count_ones() as i32;

    // Casas seguras (não atacadas pelo inimigo)
    let safe_squares = legal_squares & !context.enemy_attacked_squares;
    let safe_mobility = safe_squares.count_ones() as i32;

    // Bônus por atacar casas importantes
    let important_squares = calculate_important_squares(context);
    let attacking_important = (legal_squares & important_squares).count_ones() as i32;

    (total_mobility, safe_mobility * 2 + attacking_important)
}

/// Calcula casas importantes que o cavalo pode atacar
fn calculate_important_squares(context: &MobilityContext) -> Bitboard {
    let mut important = 0u64;

    // Casas próximas ao rei inimigo
    let enemy_king = context.board.kings & context.enemy_pieces;
    if enemy_king != 0 {
        let king_sq = enemy_king.trailing_zeros() as u8;
        important |= crate::moves::king::get_king_attacks_lookup(king_sq);
        important |= enemy_king; // O próprio rei
    }

    // Peças valiosas inimigas
    important |= context.board.queens & context.enemy_pieces;
    important |= context.board.rooks & context.enemy_pieces;

    // Casas centrais
    important |= 0x0000001818000000u64; // d4, e4, d5, e5

    important
}

/// Verifica se cavalo está próximo ao rei inimigo
fn is_near_enemy_king(knight_sq: u8, context: &MobilityContext) -> bool {
    let enemy_king = context.board.kings & context.enemy_pieces;
    if enemy_king == 0 { return false; }

    let king_sq = enemy_king.trailing_zeros() as u8;
    let distance = eval_utils::calculate_square_distance(knight_sq, king_sq);
    distance <= 3
}

/// Calcula distância de uma casa até a borda do tabuleiro
fn calculate_edge_distance(square: u8) -> i32 {
    let file = square % 8;
    let rank = square / 8;

    let dist_to_edge = [
        file,              // Distância à borda esquerda
        7 - file,          // Distância à borda direita
        rank,              // Distância à borda inferior
        7 - rank,          // Distância à borda superior
    ];

    *dist_to_edge.iter().min().unwrap() as i32
}


/// Detecta ameaças de garfo imediatas e a dois lances (melhorado)
fn detect_fork_threats(knight_sq: u8, context: &MobilityContext) -> i32 {
    let mut fork_score = 0;

    // 1. Garfos imediatos (já em posição)
    fork_score += detect_immediate_forks(knight_sq, context);

    // 2. Garfos a dois lances (novo)
    fork_score += detect_two_move_forks(knight_sq, context);

    // 3. Garfos potenciais (inimigo pode entrar em garfo)
    fork_score += count_potential_forks(knight_sq, context) / 2;

    fork_score
}

/// Detecta garfos imediatos
fn detect_immediate_forks(knight_sq: u8, context: &MobilityContext) -> i32 {
    let attacks = crate::moves::knight::get_knight_attacks_lookup(knight_sq);
    let enemy_pieces = context.enemy_pieces;

    // Conta peças valiosas inimigas atacadas
    let attacked_pieces = attacks & enemy_pieces;
    let attacked_count = attacked_pieces.count_ones();

    if attacked_count >= 2 {
        // Verifica se inclui peças valiosas
        let valuable_pieces = (context.board.queens | context.board.rooks | context.board.kings) & context.enemy_pieces;
        let attacked_valuable = (attacks & valuable_pieces).count_ones();

        // Análise mais sofisticada do valor do garfo
        let mut fork_value = 0;

        // Bônus base por garfo
        fork_value += 8;

        // Bônus por cada peça valiosa no garfo
        if (attacks & (context.board.queens & context.enemy_pieces)) != 0 {
            fork_value += 15; // Rainha no garfo
        }
        if (attacks & (context.board.rooks & context.enemy_pieces)) != 0 {
            fork_value += 10; // Torre no garfo
        }
        if (attacks & (context.board.kings & context.enemy_pieces)) != 0 {
            fork_value += 20; // Rei no garfo (xeque + outra peça)
        }

        // Bônus por número total de peças atacadas
        fork_value += (attacked_count as i32 - 2) * 3; // Bônus por peças extras além das duas básicas

        return fork_value;
    }

    0
}

/// Detecta garfos possíveis a dois lances
fn detect_two_move_forks(knight_sq: u8, context: &MobilityContext) -> i32 {
    let attacks = crate::moves::knight::get_knight_attacks_lookup(knight_sq);
    let valuable_pieces = get_valuable_enemy_pieces(context);

    // Se já há duas peças valiosas próximas, não precisa de análise a dois lances
    if (attacks & valuable_pieces).count_ones() >= 2 {
        return 0;
    }

    let legal_squares = attacks & !context.our_pieces;
    let safe_squares = filter_safe_squares(legal_squares, context);

    let total_score = calculate_two_move_fork_score(safe_squares, valuable_pieces, context);

    // Reduz o valor por ser especulativo
    total_score / 3
}

/// Obtém peças valiosas inimigas
fn get_valuable_enemy_pieces(context: &MobilityContext) -> Bitboard {
    (context.board.queens | context.board.rooks | context.board.kings) & context.enemy_pieces
}

/// Filtra casas seguras (não atacadas pelo inimigo)
fn filter_safe_squares(legal_squares: Bitboard, context: &MobilityContext) -> Vec<u8> {
    let mut safe_squares = Vec::new();
    let mut legal_bb = legal_squares;

    while legal_bb != 0 {
        let target_sq = legal_bb.trailing_zeros() as u8;
        legal_bb &= legal_bb - 1;

        // Se a casa não é atacada pelo inimigo, é segura
        if (context.enemy_attacked_squares & (1u64 << target_sq)) == 0 {
            safe_squares.push(target_sq);
        }
    }

    safe_squares
}

/// Calcula pontuação de garfos de dois movimentos
fn calculate_two_move_fork_score(safe_squares: Vec<u8>, valuable_pieces: Bitboard, context: &MobilityContext) -> i32 {
    let mut total_score = 0;

    for &target_sq in &safe_squares {
        let hypothetical_attacks = crate::moves::knight::get_knight_attacks_lookup(target_sq);
        let score = evaluate_hypothetical_fork(hypothetical_attacks, valuable_pieces, context);
        total_score += score;
    }

    total_score
}

/// Avalia um garfo hipotético
fn evaluate_hypothetical_fork(hypothetical_attacks: Bitboard, valuable_pieces: Bitboard, context: &MobilityContext) -> i32 {
    let would_attack_valuable = hypothetical_attacks & valuable_pieces;
    let valuable_count = would_attack_valuable.count_ones();

    if valuable_count >= 2 {
        // Garfo duplo de peças valiosas
        let base_quality = 12;
        let piece_bonus = calculate_piece_type_bonus(would_attack_valuable, context);
        return base_quality + piece_bonus;
    } else if valuable_count == 1 {
        // Uma peça valiosa + outras peças
        let total_attacked = (hypothetical_attacks & context.enemy_pieces).count_ones();
        if total_attacked >= 2 {
            return 6; // Garfo misto
        }
    }

    0
}

/// Calcula bônus baseado no tipo de peças no garfo
fn calculate_piece_type_bonus(attacked_valuable: Bitboard, context: &MobilityContext) -> i32 {
    let mut bonus = 0;

    if (attacked_valuable & context.board.queens) != 0 {
        bonus += 6; // Rainha no garfo
    }
    if (attacked_valuable & context.board.rooks) != 0 {
        bonus += 4; // Torre no garfo
    }
    if (attacked_valuable & context.board.kings) != 0 {
        bonus += 8; // Rei no garfo
    }

    bonus
}

/// Conta garfos potenciais
fn count_potential_forks(knight_sq: u8, context: &MobilityContext) -> i32 {
    let attacks = crate::moves::knight::get_knight_attacks_lookup(knight_sq);
    let mut potential_score = 0;

    // Verifica cada casa atacada
    let attacked_squares = get_set_bits(attacks & !context.our_pieces);

    for &attack_sq in &attacked_squares {
        // Se inimigo movesse para esta casa, quantas peças seriam atacadas?
        let hypothetical_attacks = crate::moves::knight::get_knight_attacks_lookup(attack_sq);
        let would_attack = (hypothetical_attacks & context.enemy_pieces).count_ones();

        if would_attack >= 2 {
            potential_score += would_attack as i32;
        }
    }

    potential_score
}

/// Calcula valor defensivo do cavalo
fn calculate_defensive_value(knight_sq: u8, context: &MobilityContext) -> i32 {
    let attacks = crate::moves::knight::get_knight_attacks_lookup(knight_sq);
    let mut defensive_value = 0;

    // Defende peças próprias valiosas
    let protected_pieces = attacks & context.our_pieces;
    let valuable_defended = protected_pieces &
        (context.board.queens | context.board.rooks | context.board.bishops);

    defensive_value += (valuable_defended.count_ones() as i32) * 3;

    // Defende casas próximas ao nosso rei
    let our_king = context.board.kings & context.our_pieces;
    if our_king != 0 {
        let king_sq = our_king.trailing_zeros() as u8;
        let king_area = crate::moves::king::get_king_attacks_lookup(king_sq);
        let defends_king_area = (attacks & king_area).count_ones() as i32;
        defensive_value += defends_king_area * 2;
    }

    // Controla casas importantes
    let center_squares = 0x0000001818000000u64;
    let controls_center = (attacks & center_squares).count_ones() as i32;
    defensive_value += controls_center;

    defensive_value
}

/// Avalia capacidade de bloquear peões passados (integrado com pawn_structure.rs)
fn evaluate_pawn_blockade(knight_sq: u8, context: &MobilityContext) -> i32 {
    let enemy_pawns = context.board.pawns & context.enemy_pieces;
    let our_pawns = context.board.pawns & context.our_pieces;
    if enemy_pawns == 0 { return 0; }

    let mut blockade_value = 0;
    let pawn_squares = get_set_bits_simple(enemy_pawns);

    for &pawn_sq in &pawn_squares {
        // Usa a função integrada de pawn_structure.rs para verificar peão passado
        if is_passed_pawn_integrated(pawn_sq as usize, context.enemy_color, our_pawns, enemy_pawns) {
            // Verifica se cavalo pode bloquear efetivamente o caminho
            let blocking_effectiveness = calculate_knight_blocking_power(knight_sq, pawn_sq as u8, context);

            if blocking_effectiveness > 0 {
                let distance_to_promotion = if context.enemy_color == Color::White {
                    7 - (pawn_sq / 8) // Distância para a 8ª fileira
                } else {
                    pawn_sq / 8      // Distância para a 1ª fileira
                };

                // Fórmula melhorada considerando:
                // 1. Efetividade do bloqueio (posição do cavalo)
                // 2. Urgência (proximidade da promoção)
                // 3. Fase do jogo (mais importante no final)
                let urgency_factor = (8 - distance_to_promotion).max(1);
                let phase_multiplier = match context.phase {
                    GamePhase::Endgame => 2.0,
                    GamePhase::MiddleGame => 1.5,
                    GamePhase::Opening => 1.0,
                };

                let base_value = blocking_effectiveness * urgency_factor as i32;
                blockade_value += (base_value as f32 * phase_multiplier) as i32;

                // Bônus adicional se o cavalo pode atacar múltiplas casas do caminho
                blockade_value += evaluate_path_control(knight_sq, pawn_sq as u8, context);
            }
        }
    }

    blockade_value
}

/// Versão integrada que usa a lógica de pawn_structure.rs
fn is_passed_pawn_integrated(pawn_sq: usize, pawn_color: Color, our_pawns: Bitboard, enemy_pawns: Bitboard) -> bool {
    // Usa a mesma lógica da função is_passed_pawn do arquivo pawn_structure.rs
    let file = pawn_sq % 8;
    let rank = pawn_sq / 8;

    // Verifica arquivos adjacentes e o próprio arquivo
    for check_file in (file.saturating_sub(1))..=(file.saturating_add(1)).min(7) {
        let file_mask = 0x0101010101010101u64 << check_file;
        let file_pawns = our_pawns & file_mask; // Nossos peões que podem bloquear

        if file_pawns != 0 {
            // Verifica se há peão nosso que pode bloquear
            let our_pawn_squares = get_set_bits_simple(file_pawns);
            for our_sq in our_pawn_squares {
                let our_rank = our_sq / 8;

                let blocks_advancement = if pawn_color == Color::White {
                    our_rank > (rank as u8) // Nosso peão está à frente do peão inimigo
                } else {
                    our_rank < (rank as u8) // Nosso peão está à frente do peão inimigo
                };

                if blocks_advancement {
                    return false;
                }
            }
        }
    }

    true
}

/// Calcula o poder de bloqueio do cavalo para um peão específico
fn calculate_knight_blocking_power(knight_sq: u8, pawn_sq: u8, context: &MobilityContext) -> i32 {
    let knight_attacks = crate::moves::knight::get_knight_attacks_lookup(knight_sq);
    let pawn_file = pawn_sq % 8;
    let pawn_rank = pawn_sq / 8;

    let mut blocking_power = 0;

    // Verifica se o cavalo pode atacar o peão diretamente
    if (knight_attacks & (1u64 << pawn_sq)) != 0 {
        blocking_power += 5; // Ataque direto
    }

    // Verifica se pode atacar casas no caminho para promoção
    let promotion_path: Vec<u8> = if context.enemy_color == Color::White {
        ((pawn_rank + 1)..8).map(|r| r * 8 + pawn_file).collect()
    } else {
        (0..pawn_rank).rev().map(|r| r * 8 + pawn_file).collect()
    };

    let mut path_control = 0;
    for &path_sq in &promotion_path {
        if (knight_attacks & (1u64 << path_sq)) != 0 {
            path_control += 1;
        }

        // Se pode alcançar a casa em dois movimentos
        if can_knight_reach_in_two_moves(knight_sq, path_sq, context) {
            path_control += 1;
        }
    }

    blocking_power += path_control * 2;

    // Distância física também importa
    let distance = eval_utils::calculate_square_distance(knight_sq, pawn_sq);
    if distance <= 3 {
        blocking_power += (4 - distance) as i32;
    }

    blocking_power
}

/// Avalia controle do caminho de promoção
fn evaluate_path_control(knight_sq: u8, pawn_sq: u8, context: &MobilityContext) -> i32 {
    let knight_attacks = crate::moves::knight::get_knight_attacks_lookup(knight_sq);
    let pawn_file = pawn_sq % 8;
    let pawn_rank = pawn_sq / 8;

    // Casas importantes no caminho
    let critical_squares: Vec<u8> = if context.enemy_color == Color::White {
        // Para peões brancos, casas críticas nas últimas fileiras
        vec![
            (6 * 8 + pawn_file), // 7ª fileira
            (7 * 8 + pawn_file), // 8ª fileira (promoção)
        ]
    } else {
        // Para peões pretos, casas críticas nas primeiras fileiras
        vec![
            (1 * 8 + pawn_file), // 2ª fileira
            pawn_file,           // 1ª fileira (promoção)
        ]
    };

    let mut control_bonus = 0;
    for &critical_sq in &critical_squares {
        if (knight_attacks & (1u64 << critical_sq)) != 0 {
            control_bonus += 3; // Controle direto de casa crítica
        }
    }

    control_bonus
}

/// Verifica se cavalo pode alcançar uma casa em dois movimentos
fn can_knight_reach_in_two_moves(knight_sq: u8, target_sq: u8, context: &MobilityContext) -> bool {
    let first_move_attacks = crate::moves::knight::get_knight_attacks_lookup(knight_sq);
    let safe_first_moves = first_move_attacks & !context.enemy_attacked_squares & !context.our_pieces;

    let mut can_reach = false;
    let mut temp_bb = safe_first_moves;
    while temp_bb != 0 {
        let intermediate_sq = temp_bb.trailing_zeros() as u8;
        temp_bb &= temp_bb - 1;

        let second_move_attacks = crate::moves::knight::get_knight_attacks_lookup(intermediate_sq);
        if (second_move_attacks & (1u64 << target_sq)) != 0 {
            can_reach = true;
            break;
        }
    }

    can_reach
}

/// Verifica se peão inimigo é passado (função legacy mantida para compatibilidade)
fn is_passed_pawn_enemy(pawn_sq: u8, pawn_color: Color, our_pawns: Bitboard, enemy_pawns: Bitboard) -> bool {
    is_passed_pawn_integrated(pawn_sq as usize, pawn_color, our_pawns, enemy_pawns)
}

/// Legacy: Verifica se peão inimigo é passado (versão simplificada)
fn is_passed_pawn_enemy_old(pawn_sq: u8, pawn_color: Color, our_pawns: Bitboard, enemy_pawns: Bitboard) -> bool {
    let file = pawn_sq % 8;
    let rank = pawn_sq / 8;

    // Verifica se nossos peões podem bloquear
    for check_file in (file.saturating_sub(1))..=(file.saturating_add(1)).min(7) {
        let file_mask = get_file_mask(check_file);
        let our_file_pawns = our_pawns & file_mask;

        if our_file_pawns != 0 {
            let our_pawn_squares = get_set_bits(our_file_pawns);
            for our_sq in our_pawn_squares {
                let our_rank = our_sq / 8;

                let can_block = if pawn_color == Color::White {
                    our_rank > rank // Nosso peão está à frente
                } else {
                    our_rank < rank
                };

                if can_block {
                    return false;
                }
            }
        }
    }

    true
}

/// Avalia oportunidades táticas diversas
fn evaluate_tactical_opportunities(knight_sq: u8, context: &MobilityContext) -> i32 {
    let mut tactical_score = 0;
    let attacks = crate::moves::knight::get_knight_attacks_lookup(knight_sq);

    // Ataques descobertos potenciais
    tactical_score += evaluate_discovered_attacks(knight_sq, context);

    // Suporte para ataques combinados
    tactical_score += evaluate_combo_attacks(knight_sq, context);

    // Controle de casas de fuga do rei inimigo
    let enemy_king = context.board.kings & context.enemy_pieces;
    if enemy_king != 0 {
        let king_sq = enemy_king.trailing_zeros() as u8;
        let king_escapes = crate::moves::king::get_king_attacks_lookup(king_sq);
        let blocks_escapes = (attacks & king_escapes).count_ones() as i32;
        tactical_score += blocks_escapes * 2;
    }

    // Preparação para avanços táticos
    if context.phase == GamePhase::MiddleGame {
        tactical_score += evaluate_tactical_preparation(knight_sq, context);
    }

    tactical_score
}

/// Avalia ataques descobertos potenciais
fn evaluate_discovered_attacks(_knight_sq: u8, _context: &MobilityContext) -> i32 {
    // Implementação simplificada - em versão completa, analisaria linhas de ataque
    // que podem ser abertas movendo o cavalo
    5 // Valor placeholder
}

/// Avalia ataques combinados com outras peças
fn evaluate_combo_attacks(knight_sq: u8, context: &MobilityContext) -> i32 {
    let attacks = crate::moves::knight::get_knight_attacks_lookup(knight_sq);
    let mut combo_score = 0;

    // Verifica sobreposição com ataques de nossas outras peças
    let our_other_attacks = context.our_attacked_squares & !attacks;
    let shared_targets = attacks & our_other_attacks;

    // Ataques duplos são valiosos
    let enemy_pieces_double_attacked = shared_targets & context.enemy_pieces;
    combo_score += (enemy_pieces_double_attacked.count_ones() as i32) * 3;

    combo_score
}

/// Avalia preparação tática
fn evaluate_tactical_preparation(_knight_sq: u8, context: &MobilityContext) -> i32 {
    // Considera posicionamento para futuros ataques baseados na estrutura de peões
    // e posição das peças inimigas
    let mut prep_score = 0;

    // Bônus se cavalo está bem posicionado para explorar fraquezas
    let enemy_weaknesses = identify_enemy_weaknesses(context);
    prep_score += enemy_weaknesses;

    prep_score
}

/// Identifica fraquezas na posição inimiga
fn identify_enemy_weaknesses(context: &MobilityContext) -> i32 {
    let mut weakness_score = 0;

    // Rei exposto
    let enemy_king = context.board.kings & context.enemy_pieces;
    if enemy_king != 0 {
        let king_sq = enemy_king.trailing_zeros() as u8;
        let king_safety = evaluate_king_exposure(king_sq, context);
        weakness_score += king_safety;
    }

    // Peças não defendidas
    let undefended = context.enemy_pieces & !context.enemy_attacked_squares;
    weakness_score += (undefended.count_ones() as i32) * 2;

    weakness_score / 3 // Normaliza o valor
}

/// Avalia exposição do rei inimigo
fn evaluate_king_exposure(king_sq: u8, context: &MobilityContext) -> i32 {
    let king_area = crate::moves::king::get_king_attacks_lookup(king_sq);

    // Conta casas ao redor do rei que não estão protegidas
    let unprotected_around_king = king_area & !context.enemy_attacked_squares;
    let exposure = unprotected_around_king.count_ones() as i32;

    // Conta nossas peças que podem atacar área do rei
    let our_attacks_on_king_area = king_area & context.our_attacked_squares;
    let pressure = our_attacks_on_king_area.count_ones() as i32;

    (exposure * 2) + pressure
}