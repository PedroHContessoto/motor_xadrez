// Mobilidade e estratégia avançada de bispos
use crate::types::{Color, Bitboard};
use super::{MobilityContext, MobilityResult, GamePhase, utils::*};
use super::super::pawn_structure; // Adicionado para integração

/// Pesos para diferentes aspectos da estratégia de bispos
#[derive(Debug, Clone, Copy)]
pub struct BishopWeights {
    pub mobility_per_square: [i32; 3],    // [opening, middlegame, endgame]
    pub long_diagonal: [i32; 3],          // Bônus por controle de diagonal longa
    pub bishop_pair: [i32; 3],            // Bônus por par de bispos
    pub fianchetto: [i32; 3],             // Bônus por fianchetto
    pub trapped_penalty: [i32; 3],        // Penalidade por bispo preso
    pub color_complex: [i32; 3],          // Bônus por dominar complexo de casas
    pub pin_potential: [i32; 3],          // Bônus por potencial de pregadura
    pub diagonal_dominance: [i32; 3],     // Bônus por dominância diagonal
}

impl Default for BishopWeights {
    fn default() -> Self {
        BishopWeights {
            mobility_per_square: [1, 2, 2],
            long_diagonal: [5, 6, 4],
            bishop_pair: [8, 8, 12], // Ajustado: reduzido no meio-jogo, aumentado no final
            fianchetto: [3, 4, 2],
            trapped_penalty: [-12, -15, -10], // Aumentado de [-10, -12, -8]
            color_complex: [3, 5, 6],
            pin_potential: [2, 4, 3],
            diagonal_dominance: [2, 3, 2],
        }
    }
}

/// Resultado específico da análise de bispos
#[derive(Debug, Clone, Copy)]
pub struct BishopAnalysis {
    pub mobility_score: i32,
    pub diagonal_control: i32,
    pub bishop_pair_bonus: i32,
    pub fianchetto_bonus: i32,
    pub trapped_penalty: i32,
    pub color_complex_bonus: i32,
    pub pin_threats: i32,
    pub long_range_influence: i32,
}

impl BishopAnalysis {
    pub fn new() -> Self {
        BishopAnalysis {
            mobility_score: 0,
            diagonal_control: 0,
            bishop_pair_bonus: 0,
            fianchetto_bonus: 0,
            trapped_penalty: 0,
            color_complex_bonus: 0,
            pin_threats: 0,
            long_range_influence: 0,
        }
    }
}

/// Avaliação avançada de mobilidade de bispos
pub fn evaluate_bishop_mobility_advanced(context: &MobilityContext) -> i32 {
    let weights = BishopWeights::default();
    let phase_idx = match context.phase {
        GamePhase::Opening => 0,
        GamePhase::MiddleGame => 1,
        GamePhase::Endgame => 2,
    };

    let analysis = analyze_bishop_strategy(context);

    // Aplica pesos baseados na fase do jogo
    let mut score = 0;
    score += analysis.mobility_score * weights.mobility_per_square[phase_idx];
    score += analysis.diagonal_control * weights.long_diagonal[phase_idx];
    score += analysis.bishop_pair_bonus * weights.bishop_pair[phase_idx];
    score += analysis.fianchetto_bonus * weights.fianchetto[phase_idx];
    score += analysis.trapped_penalty * weights.trapped_penalty[phase_idx];
    score += analysis.color_complex_bonus * weights.color_complex[phase_idx];
    score += analysis.pin_threats * weights.pin_potential[phase_idx];
    score += analysis.long_range_influence * weights.diagonal_dominance[phase_idx];

    score
}

/// Análise completa da estratégia de bispos
pub fn analyze_bishop_strategy(context: &MobilityContext) -> BishopAnalysis {
    let mut analysis = BishopAnalysis::new();
    let bishops = context.board.bishops & context.our_pieces;

    if bishops == 0 {
        return analysis; // Não há bispos para analisar
    }

    let bishop_count = bishops.count_ones();
    let bishop_squares = get_set_bits(bishops);

    // Bônus por par de bispos
    if bishop_count >= 2 {
        analysis.bishop_pair_bonus = 1;
        // Novo: Reduz bônus se estrutura fechada
        if !is_closed_position(context) {
            if has_bishops_on_different_colors(&bishop_squares) {
                analysis.bishop_pair_bonus += 1;
            }
        }
    }

    for &bishop_sq in &bishop_squares {
        // Análise de mobilidade básica
        let (mobility, safe_mobility) = calculate_bishop_mobility(bishop_sq, context);
        analysis.mobility_score += mobility + safe_mobility;

        // Controle diagonal
        analysis.diagonal_control += evaluate_diagonal_control(bishop_sq, context);

        // Verifica fianchetto
        if is_fianchetto_bishop(bishop_sq, context.color) {
            analysis.fianchetto_bonus += 1;
        }

        // Verifica se bispo está preso
        if is_trapped_bishop(bishop_sq, context) {
            analysis.trapped_penalty += 1;
        }

        // Avalia domínio do complexo de casas
        analysis.color_complex_bonus += evaluate_color_complex_control(bishop_sq, context);

        // Detecta potenciais pregaduras
        analysis.pin_threats += detect_pin_potential(bishop_sq, context);

        // Influência de longo alcance
        analysis.long_range_influence += evaluate_long_range_influence(bishop_sq, context);
    }

    analysis
}

/// Calcula mobilidade básica e segura do bispo
fn calculate_bishop_mobility(bishop_sq: u8, context: &MobilityContext) -> (i32, i32) {
    let attacks = crate::moves::magic_bitboards::get_bishop_attacks_magic(bishop_sq, context.all_pieces);

    // Casas legais (não ocupadas por peças próprias)
    let legal_squares = attacks & !context.our_pieces;
    let total_mobility = legal_squares.count_ones() as i32;

    // Casas seguras (não atacadas pelo inimigo)
    let safe_squares = legal_squares & !context.enemy_attacked_squares;
    let safe_mobility = safe_squares.count_ones() as i32;

    // Bônus por atacar casas centrais
    let center_squares = 0x0000001818000000u64; // d4, e4, d5, e5
    let attacking_center = (legal_squares & center_squares).count_ones() as i32;

    // Bônus por controlar casas próximas ao rei inimigo
    let enemy_king = context.board.kings & context.enemy_pieces;
    let king_area_bonus = if enemy_king != 0 {
        let king_sq = enemy_king.trailing_zeros() as u8;
        let king_area = crate::moves::king::get_king_attacks_lookup(king_sq);
        (legal_squares & king_area).count_ones() as i32
    } else {
        0
    };

    (total_mobility, safe_mobility + attacking_center + king_area_bonus)
}

/// Verifica se bispos estão em casas de cores diferentes
fn has_bishops_on_different_colors(bishop_squares: &[u8]) -> bool {
    if bishop_squares.len() < 2 {
        return false;
    }

    let colors: Vec<bool> = bishop_squares.iter()
        .map(|&sq| is_light_square(sq))
        .collect();

    colors.iter().any(|&x| x) && colors.iter().any(|&x| !x)
}

/// Verifica se casa é clara
fn is_light_square(square: u8) -> bool {
    let file = square % 8;
    let rank = square / 8;
    (file + rank) % 2 == 1
}

/// Avalia controle diagonal
fn evaluate_diagonal_control(bishop_sq: u8, context: &MobilityContext) -> i32 {
    let mut control_score = 0;
    let attacks = crate::moves::magic_bitboards::get_bishop_attacks_magic(bishop_sq, context.all_pieces);

    // Novo: Usar máscaras pré-computadas para diagonais principais
    const MAIN_DIAGONAL: Bitboard = 0x8040201008040201u64;
    const ANTI_DIAGONAL: Bitboard = 0x0102040810204080u64;

    if (1u64 << bishop_sq) & MAIN_DIAGONAL != 0 {
        let controls_main = (attacks & MAIN_DIAGONAL).count_ones() as i32;
        control_score += controls_main * 2;
    }

    if (1u64 << bishop_sq) & ANTI_DIAGONAL != 0 {
        let controls_anti = (attacks & ANTI_DIAGONAL).count_ones() as i32;
        control_score += controls_anti * 2;
    }

    // Diagonais longas gerais (7+ casas)
    let diagonal_length = calculate_diagonal_length(bishop_sq, attacks);
    if diagonal_length >= 7 {
        control_score += 5; // Bônus por diagonal longa
    }

    control_score
}

/// Calcula comprimento das diagonais controladas (OTIMIZADO - simples popcount das diagonais)
fn calculate_diagonal_length(bishop_sq: u8, attacks: Bitboard) -> i32 {
    // Usa popcount das casas atacadas para aproximar o comprimento das diagonais
    // Muito mais rápido que loops manuais
    let attack_count = attacks.count_ones() as i32;
    
    // Aproximação: se o bispo ataca muitas casas, tem diagonais longas
    if attack_count >= 10 {
        7 // Diagonal máxima
    } else if attack_count >= 7 {
        6
    } else if attack_count >= 5 {
        5
    } else {
        attack_count.min(7)
    }
}

/// Verifica se bispo está em fianchetto
fn is_fianchetto_bishop(bishop_sq: u8, color: Color) -> bool {
    let fianchetto_squares = if color == Color::White {
        [1, 6, 57, 62] // b1, g1, b8, g8
    } else {
        [1, 6, 57, 62]
    };

    fianchetto_squares.contains(&bishop_sq)
}

/// Verifica se bispo está preso
fn is_trapped_bishop(bishop_sq: u8, context: &MobilityContext) -> bool {
    let attacks = crate::moves::magic_bitboards::get_bishop_attacks_magic(bishop_sq, context.all_pieces);
    let legal_moves = attacks & !context.our_pieces;
    let safe_moves = legal_moves & !context.enemy_attacked_squares;

    // Novo: Verifica se bispo está bloqueado por peões próprios
    let is_light = is_light_square(bishop_sq);
    let our_pawns = context.board.pawns & context.our_pieces;
    let same_color_squares = get_squares_of_color(is_light);
    let blocked_pawns = (our_pawns & same_color_squares).count_ones() as i32;

    safe_moves.count_ones() <= 2 || blocked_pawns >= 3
}

/// Avalia domínio do complexo de casas da mesma cor
fn evaluate_color_complex_control(bishop_sq: u8, context: &MobilityContext) -> i32 {
    let is_light = is_light_square(bishop_sq);
    let mut control_score = 0;

    let attacks = crate::moves::magic_bitboards::get_bishop_attacks_magic(bishop_sq, context.all_pieces);
    let same_color_squares = get_squares_of_color(is_light);
    let controlled_same_color = (attacks & same_color_squares).count_ones() as i32;

    control_score += controlled_same_color;

    let central_same_color = same_color_squares & 0x0000001818000000u64;
    let controls_central_same_color = (attacks & central_same_color).count_ones() as i32;
    control_score += controls_central_same_color * 2;

    let our_pawns = context.board.pawns & context.our_pieces;
    let blocked_same_color = (our_pawns & same_color_squares).count_ones() as i32;
    control_score -= blocked_same_color;

    control_score
}

/// Obtém casas de uma cor específica
fn get_squares_of_color(is_light: bool) -> Bitboard {
    if is_light {
        0x55AA55AA55AA55AAu64 // Casas claras
    } else {
        0xAA55AA55AA55AA55u64 // Casas escuras
    }
}

/// Detecta potenciais pregaduras
fn detect_pin_potential(bishop_sq: u8, context: &MobilityContext) -> i32 {
    let mut pin_score = 0;
    let attacks = crate::moves::magic_bitboards::get_bishop_attacks_magic(bishop_sq, context.all_pieces);

    let valuable_enemies = (context.board.queens | context.board.rooks | context.board.kings) & context.enemy_pieces;
    let attacks_valuable = attacks & valuable_enemies;

    if attacks_valuable != 0 {
        let valuable_squares = get_set_bits(attacks_valuable);

        for &target_sq in &valuable_squares {
            if can_create_pin(bishop_sq, target_sq, context) {
                let target_bb = 1u64 << target_sq;
                if (context.board.kings & target_bb) != 0 {
                    pin_score += 15; // Pregadura no rei
                } else if (context.board.queens & target_bb) != 0 {
                    pin_score += 12; // Pregadura na rainha
                } else {
                    pin_score += 8; // Pregadura em torre
                }
            }
        }
    }

    pin_score
}

/// Verifica se pode criar pregadura entre bispo e peça alvo
fn can_create_pin(bishop_sq: u8, target_sq: u8, context: &MobilityContext) -> bool {
    let dir = get_diagonal_direction(bishop_sq, target_sq);
    if dir == 0 { return false; }

    let mut current_sq = target_sq as i32 + dir;

    while current_sq >= 0 && current_sq <= 63 && is_valid_diagonal_step(target_sq as i32, current_sq, dir) {
        let current_bb = 1u64 << current_sq;

        if (context.enemy_pieces & current_bb) != 0 {
            let is_valuable = (context.board.queens | context.board.rooks | context.board.kings) & current_bb;
            return is_valuable != 0;
        } else if (context.our_pieces & current_bb) != 0 {
            return false;
        }

        current_sq += dir;
    }

    false
}

/// Obtém direção diagonal entre duas casas
fn get_diagonal_direction(from: u8, to: u8) -> i32 {
    let file_diff = (to % 8) as i32 - (from % 8) as i32;
    let rank_diff = (to / 8) as i32 - (from / 8) as i32;

    if file_diff.abs() != rank_diff.abs() {
        return 0;
    }

    match (file_diff.signum(), rank_diff.signum()) {
        (1, 1) => 9,   // NE
        (1, -1) => -7, // SE
        (-1, 1) => 7,  // NW
        (-1, -1) => -9, // SW
        _ => 0,
    }
}

/// Verifica se movimento diagonal é válido
fn is_valid_diagonal_step(from: i32, to: i32, dir: i32) -> bool {
    let expected = from + dir;
    expected == to
}

/// Avalia influência de longo alcance
fn evaluate_long_range_influence(bishop_sq: u8, context: &MobilityContext) -> i32 {
    let attacks = crate::moves::magic_bitboards::get_bishop_attacks_magic(bishop_sq, context.all_pieces);
    let mut influence_score = 0;

    let enemy_territory = get_enemy_territory(context.enemy_color);
    let influence_on_enemy_territory = (attacks & enemy_territory).count_ones() as i32;
    influence_score += influence_on_enemy_territory;

    let passage_squares = get_important_passage_squares();
    let controls_passages = (attacks & passage_squares).count_ones() as i32;
    influence_score += controls_passages * 2;

    let enemy_pawns = context.board.pawns & context.enemy_pieces;
    let pressure_on_pawns = (attacks & enemy_pawns).count_ones() as i32;
    influence_score += pressure_on_pawns;

    influence_score
}

/// Obtém território inimigo (metade do tabuleiro)
fn get_enemy_territory(enemy_color: Color) -> Bitboard {
    if enemy_color == Color::White {
        0xFFFFFFFF00000000u64 // Ranks 5-8
    } else {
        0x00000000FFFFFFFFu64 // Ranks 1-4
    }
}

/// Obtém casas de passagem importantes
fn get_important_passage_squares() -> Bitboard {
    0x00003C3C3C3C0000u64
}

/// Novo: Verifica se posição é fechada
fn is_closed_position(context: &MobilityContext) -> bool {
    let center_pawns = context.board.pawns & 0x0000001818000000u64; // d4, e4, d5, e5
    center_pawns.count_ones() >= 3 // Posição fechada se >=3 peões centrais
}

/// Avalia bispo baseado em sua mobilidade atual vs potencial
pub fn evaluate_bishop_development(bishop_sq: u8, context: &MobilityContext) -> i32 {
    let mut development_score = 0;

    if context.phase != GamePhase::Opening {
        let initial_squares = if context.color == Color::White {
            [2, 5] // c1, f1
        } else {
            [58, 61] // c8, f8
        };

        if initial_squares.contains(&bishop_sq) {
            development_score -= 10;
        }
    }

    if context.phase == GamePhase::Opening {
        let central_influence = evaluate_central_influence(bishop_sq, context);
        development_score += central_influence;
    }

    development_score
}

/// Avalia influência central do bispo
fn evaluate_central_influence(bishop_sq: u8, context: &MobilityContext) -> i32 {
    let attacks = crate::moves::magic_bitboards::get_bishop_attacks_magic(bishop_sq, context.all_pieces);
    let center = 0x0000001818000000u64; // d4, e4, d5, e5
    let extended_center = 0x00003C3C3C3C0000u64; // c3-f3 até c6-f6

    let controls_center = (attacks & center).count_ones() as i32;
    let controls_extended = (attacks & extended_center).count_ones() as i32;

    controls_center * 3 + controls_extended
}