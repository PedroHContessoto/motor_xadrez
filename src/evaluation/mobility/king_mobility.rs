// Sistema revolucionário de mobilidade e atividade do rei com conceitos avançados de endgame
use crate::types::{Color, Bitboard};
use super::{MobilityContext, GamePhase, utils::*};
use crate::evaluation::utils as eval_utils;

const KING_SAFETY_MULTIPLIER: i32 = 8;
const KING_ACTIVITY_MULTIPLIER: i32 = 25; // Aumentado de 12 para 25 conforme análise
const OPPOSITION_BONUS: i32 = 200;
const DISTANT_OPPOSITION_BONUS: i32 = 150;
const SHOULDER_CHARGE_BONUS: i32 = 180;
const BODY_CHECK_BONUS: i32 = 160;
const TRIANGULATION_BONUS: i32 = 140;
const OUTFLANKING_BONUS: i32 = 220;
const PAWN_RACE_CRITICAL_BONUS: i32 = 300;
const CONJUGATE_SQUARE_BONUS: i32 = 190;
const BREAKTHROUGH_SUPPORT_BONUS: i32 = 250;
const ZUGZWANG_POTENTIAL_BONUS: i32 = 170;

/// Contexto avançado de análise do rei
#[derive(Debug, Clone)]
pub struct KingEndgameContext {
    pub our_king_sq: u8,
    pub enemy_king_sq: u8,
    pub our_pawns: Bitboard,
    pub enemy_pawns: Bitboard,
    pub pawn_race_active: bool,
    pub king_activity_phase: KingActivityPhase,
    pub material_balance: MaterialBalance,
    pub time_critical_zones: Vec<u8>,
    pub conjugate_squares: Vec<u8>,
    pub breakthrough_candidates: Vec<u8>,
}

#[derive(Debug, Clone, Copy)]
pub enum KingActivityPhase {
    SafetyFirst,      // Opening/Early middlegame
    Transitional,     // Late middlegame
    ActivePlay,       // Endgame
    PawnRaceMode,     // Critical pawn races
    TechnicalWin,     // Converting advantage
}

#[derive(Debug, Clone, Copy)]
pub enum MaterialBalance {
    Balanced,
    MinorAdvantage,
    MajorAdvantage,
    PawnEndgame,
    PieceEndgame,
}

/// Avaliação revolucionária de mobilidade e atividade do rei
pub fn evaluate_king_mobility_advanced(context: &MobilityContext) -> i32 {
    let mut total_score = 0;
    let kings = context.board.kings & context.our_pieces;

    if kings == 0 { return 0; }

    let king_sq = kings.trailing_zeros() as u8;
    let enemy_king = context.board.kings & context.enemy_pieces;
    
    if enemy_king == 0 { return 0; }
    
    let enemy_king_sq = enemy_king.trailing_zeros() as u8;

    // Cria contexto especializado de endgame
    let endgame_context = create_king_endgame_context(king_sq, enemy_king_sq, context);

    match context.phase {
        GamePhase::Opening | GamePhase::MiddleGame => {
            // === FASE DE SEGURANÇA ===
            total_score += evaluate_king_safety_advanced(king_sq, context) * KING_SAFETY_MULTIPLIER;
            
            // Detecção precoce de transição para endgame
            if is_approaching_endgame(context) {
                total_score += prepare_for_endgame_transition(king_sq, context);
            }
        }
        GamePhase::Endgame => {
            // === SISTEMA AVANÇADO DE ENDGAME ===
            
            // 1. Atividade relativa (comparação com rei adversário)
            total_score += evaluate_relative_king_activity(king_sq, enemy_king_sq, context);
            
            // 2. Oposição em todas suas formas
            total_score += evaluate_opposition_systems(&endgame_context);
            
            // 3. Body check e shoulder charge
            total_score += evaluate_body_check_shoulder_charge(&endgame_context);
            
            // 4. Análise de tempo crítico em corridas de peões
            total_score += evaluate_pawn_race_critical_time(&endgame_context, context);
            
            // 5. Quadrados conjugados e correspondentes
            total_score += evaluate_conjugate_corresponding_squares(&endgame_context);
            
            // 6. Triangulação e manobras de tempo
            total_score += evaluate_triangulation_potential(&endgame_context);
            
            // 7. Outflanking e infiltração
            total_score += evaluate_outflanking_infiltration(&endgame_context);
            
            // 8. Integração com pawn_structure para breakthrough
            total_score += evaluate_breakthrough_support(&endgame_context, context);
            
            // 9. Detecção de zugzwang
            total_score += evaluate_zugzwang_potential(&endgame_context, context);
            
            // 10. Atividade básica multiplicada
            total_score += evaluate_basic_king_activity(king_sq, context) * KING_ACTIVITY_MULTIPLIER;
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

// ============================================================================
// IMPLEMENTAÇÃO DAS FUNÇÕES AVANÇADAS DE ENDGAME
// ============================================================================

/// Cria contexto especializado de endgame
fn create_king_endgame_context(king_sq: u8, enemy_king_sq: u8, context: &MobilityContext) -> KingEndgameContext {
    let our_pawns = context.board.pawns & context.our_pieces;
    let enemy_pawns = context.board.pawns & context.enemy_pieces;
    
    // Detecta corridas de peões ativas
    let pawn_race_active = detect_pawn_races(our_pawns, enemy_pawns, context);
    
    // Determina fase de atividade do rei
    let king_activity_phase = determine_king_activity_phase(context);
    
    // Avalia balanço material
    let material_balance = assess_material_balance(context);
    
    // Identifica zonas críticas de tempo
    let time_critical_zones = identify_time_critical_zones(our_pawns, enemy_pawns, context);
    
    // Calcula quadrados conjugados
    let conjugate_squares = calculate_conjugate_squares(king_sq, enemy_king_sq, our_pawns, enemy_pawns);
    
    // Identifica candidatos a breakthrough
    let breakthrough_candidates = identify_breakthrough_candidates(our_pawns, enemy_pawns, context);
    
    KingEndgameContext {
        our_king_sq: king_sq,
        enemy_king_sq,
        our_pawns,
        enemy_pawns,
        pawn_race_active,
        king_activity_phase,
        material_balance,
        time_critical_zones,
        conjugate_squares,
        breakthrough_candidates,
    }
}

/// Detecta se há corridas de peões ativas
fn detect_pawn_races(our_pawns: Bitboard, enemy_pawns: Bitboard, context: &MobilityContext) -> bool {
    let our_passed = find_passed_pawns(our_pawns, enemy_pawns, context.color);
    let enemy_passed = find_passed_pawns(enemy_pawns, our_pawns, !context.color);
    
    // Se ambos os lados têm peões passados avançados, há corrida
    (our_passed.count_ones() > 0 && enemy_passed.count_ones() > 0) &&
        has_advanced_passed_pawns(our_passed, context.color) ||
        has_advanced_passed_pawns(enemy_passed, !context.color)
}

/// Encontra peões passados
fn find_passed_pawns(our_pawns: Bitboard, enemy_pawns: Bitboard, color: Color) -> Bitboard {
    let mut passed = 0u64;
    let mut pawns = our_pawns;
    
    while pawns != 0 {
        let sq = pawns.trailing_zeros() as u8;
        pawns &= pawns - 1;
        
        if is_passed_pawn_simple(sq, color, enemy_pawns) {
            passed |= 1u64 << sq;
        }
    }
    
    passed
}

/// Verifica se um peão é passado (versão simplificada)
fn is_passed_pawn_simple(pawn_sq: u8, color: Color, enemy_pawns: Bitboard) -> bool {
    let file = pawn_sq % 8;
    let rank = pawn_sq / 8;
    
    for check_file in (file.saturating_sub(1))..=(file.saturating_add(1)).min(7) {
        let file_mask = get_file_mask(check_file);
        let file_enemy_pawns = enemy_pawns & file_mask;
        
        if file_enemy_pawns != 0 {
            let mut enemy_pawns_in_file = file_enemy_pawns;
            while enemy_pawns_in_file != 0 {
                let enemy_sq = enemy_pawns_in_file.trailing_zeros() as u8;
                enemy_pawns_in_file &= enemy_pawns_in_file - 1;
                let enemy_rank = enemy_sq / 8;
                
                let blocks = if color == Color::White {
                    enemy_rank > rank
                } else {
                    enemy_rank < rank
                };
                
                if blocks { return false; }
            }
        }
    }
    
    true
}

/// Verifica se há peões passados avançados
fn has_advanced_passed_pawns(passed_pawns: Bitboard, color: Color) -> bool {
    let mut pawns = passed_pawns;
    
    while pawns != 0 {
        let sq = pawns.trailing_zeros() as u8;
        pawns &= pawns - 1;
        let rank = sq / 8;
        
        let is_advanced = if color == Color::White {
            rank >= 5  // 6ª fileira ou mais
        } else {
            rank <= 2  // 3ª fileira ou menos
        };
        
        if is_advanced { return true; }
    }
    
    false
}

/// Determina a fase de atividade do rei
fn determine_king_activity_phase(context: &MobilityContext) -> KingActivityPhase {
    match context.phase {
        GamePhase::Opening => KingActivityPhase::SafetyFirst,
        GamePhase::MiddleGame => {
            // Verifica se está transitando para endgame
            let total_pieces = (context.our_pieces | context.enemy_pieces).count_ones();
            if total_pieces <= 16 {
                KingActivityPhase::Transitional
            } else {
                KingActivityPhase::SafetyFirst
            }
        },
        GamePhase::Endgame => {
            // Verifica se há corridas de peões
            if detect_pawn_races(context.board.pawns & context.our_pieces, 
                                 context.board.pawns & context.enemy_pieces, context) {
                KingActivityPhase::PawnRaceMode
            } else if has_winning_material_advantage(context) {
                KingActivityPhase::TechnicalWin
            } else {
                KingActivityPhase::ActivePlay
            }
        },
    }
}

/// Verifica se tem vantagem material decisiva
fn has_winning_material_advantage(context: &MobilityContext) -> bool {
    let our_material = calculate_simple_material(context.our_pieces, &context.board);
    let enemy_material = calculate_simple_material(context.enemy_pieces, &context.board);
    
    our_material - enemy_material > 300  // Mais de 3 peões de vantagem
}

/// Calcula material simples
fn calculate_simple_material(pieces: Bitboard, board: &crate::board::Board) -> i32 {
    let pawns = (board.pawns & pieces).count_ones() as i32 * 100;
    let knights = (board.knights & pieces).count_ones() as i32 * 320;
    let bishops = (board.bishops & pieces).count_ones() as i32 * 330;
    let rooks = (board.rooks & pieces).count_ones() as i32 * 500;
    let queens = (board.queens & pieces).count_ones() as i32 * 900;
    
    pawns + knights + bishops + rooks + queens
}

/// Avalia balanço material
fn assess_material_balance(context: &MobilityContext) -> MaterialBalance {
    let our_material = calculate_simple_material(context.our_pieces, &context.board);
    let enemy_material = calculate_simple_material(context.enemy_pieces, &context.board);
    let diff = (our_material - enemy_material).abs();
    
    let total_pieces = (context.our_pieces | context.enemy_pieces).count_ones();
    let is_pawn_endgame = (context.board.knights | context.board.bishops | 
                          context.board.rooks | context.board.queens).count_ones() == 0;
    
    if is_pawn_endgame {
        MaterialBalance::PawnEndgame
    } else if total_pieces <= 8 {
        MaterialBalance::PieceEndgame
    } else if diff <= 100 {
        MaterialBalance::Balanced
    } else if diff <= 300 {
        MaterialBalance::MinorAdvantage
    } else {
        MaterialBalance::MajorAdvantage
    }
}

/// Identifica zonas críticas de tempo
fn identify_time_critical_zones(our_pawns: Bitboard, enemy_pawns: Bitboard, context: &MobilityContext) -> Vec<u8> {
    let mut zones = Vec::new();
    
    // Encontra peões passados que podem decidir o jogo
    let our_passed = find_passed_pawns(our_pawns, enemy_pawns, context.color);
    let enemy_passed = find_passed_pawns(enemy_pawns, our_pawns, !context.color);
    
    // Adiciona quadrados de promoção como zonas críticas
    let mut passed = our_passed | enemy_passed;
    while passed != 0 {
        let sq = passed.trailing_zeros() as u8;
        passed &= passed - 1;
        
        let promotion_sq = if context.color == Color::White {
            (sq % 8) + 56  // 8ª fileira
        } else {
            sq % 8  // 1ª fileira
        };
        
        zones.push(promotion_sq);
    }
    
    zones
}

/// Calcula quadrados conjugados
fn calculate_conjugate_squares(our_king: u8, enemy_king: u8, our_pawns: Bitboard, enemy_pawns: Bitboard) -> Vec<u8> {
    let mut conjugate = Vec::new();
    
    // Para endgames de peões, calcula quadrados correspondentes
    if (our_pawns | enemy_pawns).count_ones() <= 4 {
        // Algoritmo simplificado para quadrados conjugados
        let king_file = our_king % 8;
        let king_rank = our_king / 8;
        
        // Adiciona quadrados críticos baseados na posição dos peões
        for sq in 0..64 {
            let file = sq % 8;
            let rank = sq / 8;
            
            // Quadrados que mantêm oposição ou controle crítico
            if (file as i32 - king_file as i32).abs() <= 2 && 
               (rank as i32 - king_rank as i32).abs() <= 2 {
                conjugate.push(sq);
            }
        }
    }
    
    conjugate
}

/// Identifica candidatos a breakthrough
fn identify_breakthrough_candidates(our_pawns: Bitboard, enemy_pawns: Bitboard, context: &MobilityContext) -> Vec<u8> {
    let mut candidates = Vec::new();
    
    // Procura por peões que podem forçar breakthrough
    let mut pawns = our_pawns;
    while pawns != 0 {
        let sq = pawns.trailing_zeros() as u8;
        pawns &= pawns - 1;
        
        if can_force_breakthrough(sq, enemy_pawns, context) {
            candidates.push(sq);
        }
    }
    
    candidates
}

/// Verifica se peão pode forçar breakthrough
fn can_force_breakthrough(pawn_sq: u8, enemy_pawns: Bitboard, context: &MobilityContext) -> bool {
    let file = pawn_sq % 8;
    let rank = pawn_sq / 8;
    
    // Verifica se há espaço para avançar
    let advancement_path = if context.color == Color::White {
        ((rank + 1)..8).map(|r| r * 8 + file).collect::<Vec<_>>()
    } else {
        (0..rank).rev().map(|r| r * 8 + file).collect::<Vec<_>>()
    };
    
    // Se o caminho está livre de peões inimigos
    for &sq in &advancement_path {
        if (enemy_pawns & (1u64 << sq)) != 0 {
            return false;
        }
    }
    
    // Verifica se pode ser detido por peões laterais
    let mut can_be_stopped = false;
    for check_file in (file.saturating_sub(1))..=(file.saturating_add(1)).min(7) {
        if check_file == file { continue; }
        
        let file_mask = get_file_mask(check_file);
        if (enemy_pawns & file_mask) != 0 {
            can_be_stopped = true;
            break;
        }
    }
    
    !can_be_stopped
}

// ============================================================================
// FUNÇÕES DE AVALIAÇÃO AVANÇADA
// ============================================================================

/// Detecta se está se aproximando do endgame
fn is_approaching_endgame(context: &MobilityContext) -> bool {
    let total_pieces = (context.our_pieces | context.enemy_pieces).count_ones();
    let major_pieces = (context.board.queens | context.board.rooks).count_ones();
    
    total_pieces <= 20 || major_pieces <= 4
}

/// Prepara para transição de endgame
fn prepare_for_endgame_transition(king_sq: u8, context: &MobilityContext) -> i32 {
    let mut score = 0;
    
    // Bônus por centralização gradual
    if is_central_square(king_sq) {
        score += 10;
    } else if is_extended_center(king_sq) {
        score += 5;
    }
    
    // Bônus por aproximação com peões próprios
    let our_pawns = context.board.pawns & context.our_pieces;
    let pawns_nearby = count_pieces_in_radius(king_sq, our_pawns, 3);
    score += pawns_nearby * 3;
    
    score
}

/// Avalia segurança avançada do rei
fn evaluate_king_safety_advanced(king_sq: u8, context: &MobilityContext) -> i32 {
    let mut safety = evaluate_king_safety(king_sq, context);
    
    // Penalidades específicas por ataques diretos
    let enemy_queens = context.board.queens & context.enemy_pieces;
    if enemy_queens != 0 {
        let queen_squares = get_set_bits(enemy_queens);
        for &queen_sq in &queen_squares {
            let distance = eval_utils::calculate_square_distance(king_sq, queen_sq);
            if distance <= 3 {
                safety -= (4 - distance) * 8;
            }
        }
    }
    
    safety
}

/// Avalia atividade relativa do rei
fn evaluate_relative_king_activity(our_king: u8, enemy_king: u8, context: &MobilityContext) -> i32 {
    let mut activity = 0;
    
    // Compara centralização
    let our_centralization = calculate_king_centralization(our_king);
    let enemy_centralization = calculate_king_centralization(enemy_king);
    activity += (our_centralization - enemy_centralization) * 2;
    
    // Compara mobilidade
    let our_mobility = calculate_king_mobility_raw(our_king, context);
    let enemy_king_context = create_enemy_context(context);
    let enemy_mobility = calculate_king_mobility_raw(enemy_king, &enemy_king_context);
    activity += (our_mobility - enemy_mobility) * 3;
    
    // Compara proximidade com peões importantes
    let our_pawn_proximity = calculate_pawn_proximity(our_king, context.board.pawns & context.our_pieces);
    let enemy_pawn_proximity = calculate_pawn_proximity(enemy_king, context.board.pawns & context.enemy_pieces);
    activity += (our_pawn_proximity - enemy_pawn_proximity);
    
    activity
}

/// Calcula centralização do rei
fn calculate_king_centralization(king_sq: u8) -> i32 {
    let file = king_sq % 8;
    let rank = king_sq / 8;
    
    // Distância do centro (3.5, 3.5)
    let file_center_dist = ((file as f32) - 3.5).abs();
    let rank_center_dist = ((rank as f32) - 3.5).abs();
    let total_dist = file_center_dist + rank_center_dist;
    
    (10.0 - total_dist * 2.0) as i32
}

/// Calcula mobilidade bruta do rei
fn calculate_king_mobility_raw(king_sq: u8, context: &MobilityContext) -> i32 {
    let king_attacks = crate::moves::king::get_king_attacks_lookup(king_sq);
    let legal_moves = king_attacks & !context.our_pieces;
    let safe_moves = legal_moves & !context.enemy_attacked_squares;
    
    safe_moves.count_ones() as i32
}

/// Cria contexto para o rei inimigo
fn create_enemy_context(context: &MobilityContext) -> MobilityContext {
    MobilityContext {
        board: context.board,
        color: context.enemy_color,
        enemy_color: context.color,
        our_pieces: context.enemy_pieces,
        enemy_pieces: context.our_pieces,
        all_pieces: context.all_pieces,
        enemy_attacked_squares: context.our_attacked_squares,
        our_attacked_squares: context.enemy_attacked_squares,
        phase: context.phase,
        phase_info: context.phase_info.clone(),
        enemy_attack_cache: context.our_attack_cache.clone(),
        our_attack_cache: context.enemy_attack_cache.clone(),
    }
}

/// Calcula proximidade com peões
fn calculate_pawn_proximity(king_sq: u8, pawns: Bitboard) -> i32 {
    let mut proximity = 0;
    let mut pawn_bb = pawns;
    
    while pawn_bb != 0 {
        let pawn_sq = pawn_bb.trailing_zeros() as u8;
        pawn_bb &= pawn_bb - 1;
        
        let distance = eval_utils::calculate_square_distance(king_sq, pawn_sq);
        if distance <= 3 {
            proximity += 4 - distance;
        }
    }
    
    proximity
}

/// Avalia sistemas de oposição
fn evaluate_opposition_systems(endgame_ctx: &KingEndgameContext) -> i32 {
    let mut score = 0;
    
    // Oposição direta
    if has_direct_opposition(endgame_ctx.our_king_sq, endgame_ctx.enemy_king_sq) {
        score += OPPOSITION_BONUS;
    }
    
    // Oposição distante
    if has_distant_opposition(endgame_ctx.our_king_sq, endgame_ctx.enemy_king_sq) {
        score += DISTANT_OPPOSITION_BONUS;
    }
    
    // Oposição diagonal
    if has_diagonal_opposition(endgame_ctx.our_king_sq, endgame_ctx.enemy_king_sq) {
        score += OPPOSITION_BONUS / 2;
    }
    
    score
}

/// Verifica oposição direta
fn has_direct_opposition(our_king: u8, enemy_king: u8) -> bool {
    let file_diff = ((our_king % 8) as i32 - (enemy_king % 8) as i32).abs();
    let rank_diff = ((our_king / 8) as i32 - (enemy_king / 8) as i32).abs();
    
    (file_diff == 0 && rank_diff == 2) || (file_diff == 2 && rank_diff == 0)
}

/// Verifica oposição distante
fn has_distant_opposition(our_king: u8, enemy_king: u8) -> bool {
    let file_diff = ((our_king % 8) as i32 - (enemy_king % 8) as i32).abs();
    let rank_diff = ((our_king / 8) as i32 - (enemy_king / 8) as i32).abs();
    
    // Oposição com distância 4 ou 6
    (file_diff == 0 && (rank_diff == 4 || rank_diff == 6)) ||
    (rank_diff == 0 && (file_diff == 4 || file_diff == 6))
}

/// Verifica oposição diagonal
fn has_diagonal_opposition(our_king: u8, enemy_king: u8) -> bool {
    let file_diff = ((our_king % 8) as i32 - (enemy_king % 8) as i32).abs();
    let rank_diff = ((our_king / 8) as i32 - (enemy_king / 8) as i32).abs();
    
    file_diff == rank_diff && (file_diff == 2 || file_diff == 4)
}

/// Avalia body check e shoulder charge
fn evaluate_body_check_shoulder_charge(endgame_ctx: &KingEndgameContext) -> i32 {
    let mut score = 0;
    
    // Body check: rei bloqueia caminho do rei inimigo
    if can_body_check(endgame_ctx) {
        score += BODY_CHECK_BONUS;
    }
    
    // Shoulder charge: rei força o inimigo para o lado
    if can_shoulder_charge(endgame_ctx) {
        score += SHOULDER_CHARGE_BONUS;
    }
    
    score
}

/// Verifica se pode fazer body check
fn can_body_check(endgame_ctx: &KingEndgameContext) -> bool {
    // Simplificado: verifica se rei está entre o inimigo e seus peões
    if endgame_ctx.our_pawns == 0 { return false; }
    
    let king_file = endgame_ctx.our_king_sq % 8;
    let enemy_king_file = endgame_ctx.enemy_king_sq % 8;
    
    // Verifica se há peões importantes atrás do rei
    let mut pawns = endgame_ctx.our_pawns;
    while pawns != 0 {
        let pawn_sq = pawns.trailing_zeros() as u8;
        pawns &= pawns - 1;
        let pawn_file = pawn_sq % 8;
        
        // Se rei está entre inimigo e peão importante
        if (king_file > enemy_king_file && king_file < pawn_file) ||
           (king_file < enemy_king_file && king_file > pawn_file) {
            return true;
        }
    }
    
    false
}

/// Verifica se pode fazer shoulder charge
fn can_shoulder_charge(endgame_ctx: &KingEndgameContext) -> bool {
    // Shoulder charge é efetivo quando reis estão próximos
    let distance = eval_utils::calculate_square_distance(endgame_ctx.our_king_sq, endgame_ctx.enemy_king_sq);
    
    if distance <= 2 && endgame_ctx.our_pawns != 0 {
        // Verifica se há vantagem posicional para empurrar
        let our_king_rank = endgame_ctx.our_king_sq / 8;
        let enemy_king_rank = endgame_ctx.enemy_king_sq / 8;
        
        // Simplificado: vantagem se nosso rei está mais avançado
        return our_king_rank > enemy_king_rank;
    }
    
    false
}

/// Avalia tempo crítico em corridas
fn evaluate_pawn_race_critical_time(endgame_ctx: &KingEndgameContext, context: &MobilityContext) -> i32 {
    if !endgame_ctx.pawn_race_active {
        return 0;
    }
    
    let mut score = 0;
    
    // Calcula tempo para promoção dos nossos peões vs inimigos
    let our_promotion_time = calculate_minimum_promotion_time(endgame_ctx.our_pawns, context.color, endgame_ctx.our_king_sq);
    let enemy_promotion_time = calculate_minimum_promotion_time(endgame_ctx.enemy_pawns, !context.color, endgame_ctx.enemy_king_sq);
    
    // Bônus se chegamos primeiro
    if our_promotion_time < enemy_promotion_time {
        score += PAWN_RACE_CRITICAL_BONUS;
    } else if our_promotion_time == enemy_promotion_time {
        // Empate: verifica quem joga
        if context.board.to_move == context.color {
            score += PAWN_RACE_CRITICAL_BONUS / 2;
        }
    }
    
    score
}

/// Calcula tempo mínimo para promoção
fn calculate_minimum_promotion_time(pawns: Bitboard, color: Color, king_sq: u8) -> i32 {
    if pawns == 0 { return 999; }
    
    let mut min_time = 999;
    let mut pawn_bb = pawns;
    
    while pawn_bb != 0 {
        let pawn_sq = pawn_bb.trailing_zeros() as u8;
        pawn_bb &= pawn_bb - 1;
        
        let pawn_rank = pawn_sq / 8;
        let promotion_rank = if color == Color::White { 7 } else { 0 };
        
        // Tempo do peão para promoção
        let pawn_time = if color == Color::White {
            promotion_rank - pawn_rank
        } else {
            pawn_rank - promotion_rank
        } as i32;
        
        // Tempo do rei para apoiar (simplificado)
        let king_support_time = eval_utils::calculate_square_distance(king_sq, pawn_sq);
        
        let total_time = pawn_time.max(king_support_time);
        min_time = min_time.min(total_time);
    }
    
    min_time
}

/// Avalia quadrados conjugados
fn evaluate_conjugate_corresponding_squares(endgame_ctx: &KingEndgameContext) -> i32 {
    if endgame_ctx.conjugate_squares.is_empty() {
        return 0;
    }
    
    let mut score = 0;
    
    // Bônus se rei está em quadrado conjugado ideal
    if endgame_ctx.conjugate_squares.contains(&endgame_ctx.our_king_sq) {
        score += CONJUGATE_SQUARE_BONUS;
    }
    
    // Penalidade se inimigo está em posição ideal
    if endgame_ctx.conjugate_squares.contains(&endgame_ctx.enemy_king_sq) {
        score -= CONJUGATE_SQUARE_BONUS / 2;
    }
    
    score
}

/// Avalia potencial de triangulação
fn evaluate_triangulation_potential(endgame_ctx: &KingEndgameContext) -> i32 {
    // Triangulação é valiosa quando há zugzwang mútuo
    if can_create_zugzwang_situation(endgame_ctx) {
        // Verifica se temos espaço para triangular
        let triangulation_space = count_available_king_squares(endgame_ctx.our_king_sq, endgame_ctx);
        
        if triangulation_space >= 3 {
            return TRIANGULATION_BONUS;
        }
    }
    
    0
}

/// Verifica se pode criar situação de zugzwang
fn can_create_zugzwang_situation(endgame_ctx: &KingEndgameContext) -> bool {
    // Simplificado: zugzwang é comum em endgames de peões
    match endgame_ctx.material_balance {
        MaterialBalance::PawnEndgame => true,
        MaterialBalance::PieceEndgame => {
            // Em endgames de peças, menos comum mas possível
            (endgame_ctx.our_pawns | endgame_ctx.enemy_pawns).count_ones() <= 4
        },
        _ => false,
    }
}

/// Conta quadrados disponíveis para o rei
fn count_available_king_squares(king_sq: u8, endgame_ctx: &KingEndgameContext) -> i32 {
    let king_attacks = crate::moves::king::get_king_attacks_lookup(king_sq);
    let all_pieces = endgame_ctx.our_pawns | endgame_ctx.enemy_pawns;
    let available = king_attacks & !all_pieces;
    
    available.count_ones() as i32
}

/// Avalia outflanking e infiltração
fn evaluate_outflanking_infiltration(endgame_ctx: &KingEndgameContext) -> i32 {
    let mut score = 0;
    
    // Outflanking: contornar defesas inimigas
    if can_outflank(endgame_ctx) {
        score += OUTFLANKING_BONUS;
    }
    
    // Infiltração: penetrar em território inimigo
    if can_infiltrate(endgame_ctx) {
        score += OUTFLANKING_BONUS / 2;
    }
    
    score
}

/// Verifica se pode fazer outflanking
fn can_outflank(endgame_ctx: &KingEndgameContext) -> bool {
    if endgame_ctx.enemy_pawns == 0 { return false; }
    
    let king_file = endgame_ctx.our_king_sq % 8;
    let enemy_king_file = endgame_ctx.enemy_king_sq % 8;
    
    // Verifica se podemos contornar pela lateral
    let can_go_left = king_file > 0 && enemy_king_file > king_file;
    let can_go_right = king_file < 7 && enemy_king_file < king_file;
    
    can_go_left || can_go_right
}

/// Verifica se pode infiltrar
fn can_infiltrate(endgame_ctx: &KingEndgameContext) -> bool {
    let king_rank = endgame_ctx.our_king_sq / 8;
    let enemy_king_rank = endgame_ctx.enemy_king_sq / 8;
    
    // Simplificado: infiltração se rei está mais avançado que o inimigo
    king_rank > enemy_king_rank
}

/// Avalia suporte para breakthrough
fn evaluate_breakthrough_support(endgame_ctx: &KingEndgameContext, context: &MobilityContext) -> i32 {
    if endgame_ctx.breakthrough_candidates.is_empty() {
        return 0;
    }
    
    let mut score = 0;
    
    for &candidate_sq in &endgame_ctx.breakthrough_candidates {
        // Verifica proximidade do rei com candidato
        let distance = eval_utils::calculate_square_distance(endgame_ctx.our_king_sq, candidate_sq);
        
        if distance <= 2 {
            score += BREAKTHROUGH_SUPPORT_BONUS;
        } else if distance <= 4 {
            score += BREAKTHROUGH_SUPPORT_BONUS / 2;
        }
    }
    
    score
}

/// Avalia potencial de zugzwang
fn evaluate_zugzwang_potential(endgame_ctx: &KingEndgameContext, context: &MobilityContext) -> i32 {
    if !can_create_zugzwang_situation(endgame_ctx) {
        return 0;
    }
    
    let mut score = 0;
    
    // Verifica se temos mobilidade limitada (sinal de zugzwang)
    let our_mobility = calculate_king_mobility_raw(endgame_ctx.our_king_sq, context);
    let enemy_context = create_enemy_context(context);
    let enemy_mobility = calculate_king_mobility_raw(endgame_ctx.enemy_king_sq, &enemy_context);
    
    // Se o inimigo tem menos mobilidade, zugzwang favorece a nós
    if enemy_mobility < our_mobility {
        score += ZUGZWANG_POTENTIAL_BONUS;
    }
    
    score
}

/// Avalia atividade básica do rei
fn evaluate_basic_king_activity(king_sq: u8, context: &MobilityContext) -> i32 {
    evaluate_king_activity(king_sq, context)
}

