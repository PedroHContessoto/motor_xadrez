// Sistema revolucionário de mobilidade e estratégia de torres
use crate::types::{Color, Bitboard};
use super::{MobilityContext, GamePhase, utils::*};
use super::super::game_phase::{interpolate_phase_i32, GamePhaseInfo};
use crate::evaluation::utils as eval_utils;

// Constantes para avaliação avançada de torres
const ROOK_MOBILITY_MULTIPLIER: i32 = 4;
const SAFE_MOBILITY_BONUS: i32 = 6;
const OPEN_FILE_BONUS: [i32; 3] = [25, 35, 20]; // [opening, middlegame, endgame]
const SEMI_OPEN_FILE_BONUS: [i32; 3] = [15, 25, 12];
const SEVENTH_RANK_BONUS: i32 = 45;
const EIGHTH_RANK_BONUS: i32 = 25;
const DOUBLED_ROOKS_BONUS: i32 = 30;
const CONNECTED_ROOKS_BONUS: i32 = 20;
const BATTERY_FORMATION_BONUS: i32 = 35;
const PASSED_PAWN_SUPPORT_BONUS: i32 = 25;
const WEAK_SQUARE_CONTROL_BONUS: i32 = 15;
const TACTICAL_PIN_BONUS: i32 = 40;
const DISCOVERED_ATTACK_BONUS: i32 = 30;
const ROOK_LIFT_BONUS: i32 = 20;
const TRADE_INCENTIVE_BONUS: i32 = 15;

/// Contexto especializado para análise de torres
#[derive(Debug, Clone)]
pub struct RookContext {
    pub rook_sq: u8,
    pub file: u8,
    pub rank: u8,
    pub attacks: Bitboard,
    pub legal_moves: Bitboard,
    pub safe_moves: Bitboard,
    pub file_control: FileControlType,
    pub rank_influence: RankInfluenceType,
    pub tactical_role: TacticalRole,
    pub endgame_potential: EndgamePotential,
}

#[derive(Debug, Clone, Copy)]
pub enum FileControlType {
    OpenFile,          // Arquivo completamente aberto
    SemiOpen,          // Semi-aberto para nossa cor
    Contested,         // Disputado com torres inimigas
    Blocked,           // Bloqueado por peões
    Infiltrated,       // Torre penetrou em território inimigo
}

#[derive(Debug, Clone, Copy)]
pub enum RankInfluenceType {
    BackRank,          // Torre na primeira fileira
    SeventhRank,       // Torre na sétima fileira (território inimigo)
    EighthRank,        // Torre na oitava fileira (promoção/mate)
    CentralRanks,      // Torres nas fileiras centrais
    PassiveRank,       // Fileira sem influência especial
}

#[derive(Debug, Clone, Copy)]
pub enum TacticalRole {
    Attacker,          // Torre atacante principal
    Defender,          // Torre defensiva
    Support,           // Torre de suporte
    Battery,           // Torre em bateria (com outra torre/rainha)
    Lifter,            // Torre "lifted" para ataque lateral
    Pinning,           // Torre criando pins
    DiscoveredAttack,  // Torre preparando ataque descoberto
}

#[derive(Debug, Clone, Copy)]
pub enum EndgamePotential {
    Dominant,          // Torre dominante no endgame
    Active,            // Torre ativa
    Passive,           // Torre passiva
    Trapped,           // Torre restrita
    Converting,        // Torre convertendo vantagem
}

/// Avaliação revolucionária de mobilidade de torres
pub fn evaluate_rook_mobility_advanced(context: &MobilityContext) -> i32 {
    let mut total_score = 0;
    let rooks = context.board.rooks & context.our_pieces;

    if rooks == 0 { return 0; }

    // Análise de coordenação entre torres
    let coordination_bonus = evaluate_rook_coordination(rooks, context);
    total_score += coordination_bonus;

    // Avaliação individual de cada torre
    let rook_squares = get_set_bits(rooks);
    let mut rook_contexts = Vec::new();

    // Cria contextos especializados para cada torre
    for &rook_sq in &rook_squares {
        let rook_ctx = create_rook_context(rook_sq, context);
        total_score += evaluate_individual_rook(&rook_ctx, context);
        rook_contexts.push(rook_ctx);
    }

    // Análise de sinergia entre torres
    if rook_contexts.len() >= 2 {
        total_score += evaluate_rook_synergy(&rook_contexts, context);
    }

    // Bônus específicos por fase do jogo
    total_score += evaluate_phase_specific_rook_benefits(&rook_contexts, context);

    // Análise de potencial tático
    total_score += evaluate_rook_tactical_potential(&rook_contexts, context);

    total_score
}

/// Cria contexto especializado para uma torre
fn create_rook_context(rook_sq: u8, context: &MobilityContext) -> RookContext {
    let file = rook_sq % 8;
    let rank = rook_sq / 8;
    let attacks = crate::moves::sliding::get_rook_attacks(rook_sq, context.all_pieces);
    let legal_moves = attacks & !context.our_pieces;
    let safe_moves = legal_moves & !context.enemy_attacked_squares;

    RookContext {
        rook_sq,
        file,
        rank,
        attacks,
        legal_moves,
        safe_moves,
        file_control: determine_file_control(file, context),
        rank_influence: determine_rank_influence(rank, context),
        tactical_role: determine_tactical_role(rook_sq, attacks, context),
        endgame_potential: assess_endgame_potential(rook_sq, context),
    }
}

/// Avalia torre individual com sistema avançado
fn evaluate_individual_rook(rook_ctx: &RookContext, context: &MobilityContext) -> i32 {
    let mut score = 0;

    // === MOBILIDADE BÁSICA E SEGURA ===
    score += rook_ctx.legal_moves.count_ones() as i32 * ROOK_MOBILITY_MULTIPLIER;
    score += rook_ctx.safe_moves.count_ones() as i32 * SAFE_MOBILITY_BONUS;

    // === CONTROLE DE ARQUIVO ===
    score += evaluate_file_control_advanced(rook_ctx, context);

    // === INFLUÊNCIA DE FILEIRA ===
    score += evaluate_rank_influence_advanced(rook_ctx, context);

    // === PAPEL TÁTICO ===
    score += evaluate_tactical_role_contribution(rook_ctx, context);

    // === POTENCIAL DE ENDGAME ===
    score += evaluate_endgame_contribution(rook_ctx, context);

    // === ANÁLISE DE ALVOS ===
    score += evaluate_rook_targets(rook_ctx, context);

    // === CONTROLE DE CASAS FRACAS ===
    score += evaluate_weak_square_control(rook_ctx, context);

    score
}

/// Determina tipo de controle de arquivo
fn determine_file_control(file: u8, context: &MobilityContext) -> FileControlType {
    let file_mask = get_file_mask(file);
    let our_pawns_on_file = (context.board.pawns & context.our_pieces & file_mask).count_ones();
    let enemy_pawns_on_file = (context.board.pawns & context.enemy_pieces & file_mask).count_ones();
    let enemy_rooks_on_file = (context.board.rooks & context.enemy_pieces & file_mask).count_ones();

    if our_pawns_on_file == 0 && enemy_pawns_on_file == 0 {
        if enemy_rooks_on_file > 0 {
            FileControlType::Contested
        } else {
            FileControlType::OpenFile
        }
    } else if our_pawns_on_file == 0 && enemy_pawns_on_file > 0 {
        FileControlType::SemiOpen
    } else if our_pawns_on_file > 0 {
        FileControlType::Blocked
    } else {
        FileControlType::OpenFile
    }
}

/// Determina influência de fileira
fn determine_rank_influence(rank: u8, context: &MobilityContext) -> RankInfluenceType {
    match rank {
        0 => RankInfluenceType::BackRank,
        6 if context.color == Color::White => RankInfluenceType::SeventhRank,
        1 if context.color == Color::Black => RankInfluenceType::SeventhRank,
        7 if context.color == Color::White => RankInfluenceType::EighthRank,
        0 if context.color == Color::Black => RankInfluenceType::EighthRank,
        3..=4 => RankInfluenceType::CentralRanks,
        _ => RankInfluenceType::PassiveRank,
    }
}

/// Determina papel tático da torre
fn determine_tactical_role(rook_sq: u8, attacks: Bitboard, context: &MobilityContext) -> TacticalRole {
    // Simplificado por agora - pode ser expandido com análise mais complexa
    let enemy_pieces_attacked = (attacks & context.enemy_pieces).count_ones();
    let our_pieces_defended = (attacks & context.our_pieces).count_ones();

    if enemy_pieces_attacked > 2 {
        TacticalRole::Attacker
    } else if our_pieces_defended > 3 {
        TacticalRole::Defender
    } else if can_create_battery(rook_sq, context) {
        TacticalRole::Battery
    } else if can_create_pin(rook_sq, attacks, context) {
        TacticalRole::Pinning
    } else {
        TacticalRole::Support
    }
}

/// Avalia potencial de endgame
fn assess_endgame_potential(rook_sq: u8, context: &MobilityContext) -> EndgamePotential {
    let total_pieces = (context.our_pieces | context.enemy_pieces).count_ones();
    let is_endgame = total_pieces <= 16;

    if !is_endgame {
        return EndgamePotential::Active;
    }

    let file = rook_sq % 8;
    let rank = rook_sq / 8;
    let centralization = is_central_square(rook_sq) || is_extended_center(rook_sq);

    if centralization && rank >= 3 {
        EndgamePotential::Dominant
    } else if is_open_file(file, context) {
        EndgamePotential::Active
    } else {
        EndgamePotential::Passive
    }
}

/// Avalia controle de arquivo avançado
fn evaluate_file_control_advanced(rook_ctx: &RookContext, context: &MobilityContext) -> i32 {
    let mut score = 0;

    match rook_ctx.file_control {
        FileControlType::OpenFile => {
            score += interpolate_phase_i32(
                OPEN_FILE_BONUS[0],
                OPEN_FILE_BONUS[2],
                &context.phase_info
            );

            // Bônus extra por penetração
            if rook_ctx.rank >= 5 && context.color == Color::White ||
               rook_ctx.rank <= 2 && context.color == Color::Black {
                score += 20;
            }

            // Bônus por domínio completo do arquivo
            if dominates_entire_file(rook_ctx.file, context) {
                score += 25;
            }
        },
        FileControlType::SemiOpen => {
            score += interpolate_phase_i32(
                SEMI_OPEN_FILE_BONUS[0],
                SEMI_OPEN_FILE_BONUS[2],
                &context.phase_info
            );
        },
        FileControlType::Contested => {
            // Luta pelo controle do arquivo
            score += 10;
        },
        FileControlType::Infiltrated => {
            score += 30;
        },
        FileControlType::Blocked => {
            score -= 10;
        },
    }

    score
}

/// Avalia influência de fileira avançada
fn evaluate_rank_influence_advanced(rook_ctx: &RookContext, context: &MobilityContext) -> i32 {
    match rook_ctx.rank_influence {
        RankInfluenceType::SeventhRank => {
            let mut bonus = SEVENTH_RANK_BONUS;
            
            // Bônus extra se há peões inimigos na fileira
            let rank_mask = get_rank_mask(rook_ctx.rank);
            let enemy_pawns_on_rank = (context.board.pawns & context.enemy_pieces & rank_mask).count_ones();
            bonus += enemy_pawns_on_rank as i32 * 10;

            // Bônus por restringir rei inimigo
            if restricts_enemy_king(rook_ctx, context) {
                bonus += 20;
            }

            bonus
        },
        RankInfluenceType::EighthRank => EIGHTH_RANK_BONUS,
        RankInfluenceType::BackRank => {
            // Torre na primeira fileira - pode ser defensiva ou preparando castling
            if context.phase == GamePhase::Opening {
                10
            } else {
                -5 // Passiva demais no meio/endgame
            }
        },
        RankInfluenceType::CentralRanks => 15,
        RankInfluenceType::PassiveRank => 0,
    }
}

/// Avalia coordenação entre torres
fn evaluate_rook_coordination(rooks: Bitboard, context: &MobilityContext) -> i32 {
    if rooks.count_ones() < 2 {
        return 0;
    }

    let mut score = 0;
    let rook_squares = get_set_bits(rooks);

    // Torres dobradas no mesmo arquivo
    for file in 0..8 {
        let file_mask = get_file_mask(file);
        let rooks_on_file = (rooks & file_mask).count_ones();
        if rooks_on_file >= 2 {
            score += DOUBLED_ROOKS_BONUS;
            
            // Bônus extra se arquivo está aberto
            if is_open_file(file, context) {
                score += 20;
            }
        }
    }

    // Torres conectadas (se atacam mutuamente)
    if rook_squares.len() >= 2 {
        for i in 0..rook_squares.len() {
            for j in (i+1)..rook_squares.len() {
                if rooks_connected(rook_squares[i], rook_squares[j], context) {
                    score += CONNECTED_ROOKS_BONUS;
                }
            }
        }
    }

    score
}

/// Verifica se torres estão conectadas
fn rooks_connected(rook1: u8, rook2: u8, context: &MobilityContext) -> bool {
    let attacks1 = crate::moves::sliding::get_rook_attacks(rook1, context.all_pieces);
    let attacks2 = crate::moves::sliding::get_rook_attacks(rook2, context.all_pieces);
    
    (attacks1 & (1u64 << rook2)) != 0 || (attacks2 & (1u64 << rook1)) != 0
}

/// Avalia potencial tático das torres
fn evaluate_rook_tactical_potential(rook_contexts: &[RookContext], context: &MobilityContext) -> i32 {
    let mut score = 0;

    for rook_ctx in rook_contexts {
        // Pins táticos
        if can_create_pin(rook_ctx.rook_sq, rook_ctx.attacks, context) {
            score += TACTICAL_PIN_BONUS;
        }

        // Ataques descobertos
        if can_create_discovered_attack(rook_ctx, context) {
            score += DISCOVERED_ATTACK_BONUS;
        }

        // Rook lifts (torre elevada para ataque lateral)
        if is_rook_lift_position(rook_ctx, context) {
            score += ROOK_LIFT_BONUS;
        }

        // Incentivo para trocas quando vantajoso
        if should_trade_rooks(rook_ctx, context) {
            score += TRADE_INCENTIVE_BONUS;
        }
    }

    score
}

// ============================================================================
// FUNÇÕES AUXILIARES PARA ANÁLISE TÁTICA
// ============================================================================

/// Verifica se torre domina arquivo inteiro
fn dominates_entire_file(file: u8, context: &MobilityContext) -> bool {
    let file_mask = get_file_mask(file);
    let enemy_pieces_on_file = (context.enemy_pieces & file_mask).count_ones();
    enemy_pieces_on_file == 0
}

/// Verifica se torre restringe rei inimigo
fn restricts_enemy_king(rook_ctx: &RookContext, context: &MobilityContext) -> bool {
    let enemy_king = context.board.kings & context.enemy_pieces;
    if enemy_king == 0 { return false; }
    
    let king_sq = enemy_king.trailing_zeros() as u8;
    let king_rank = king_sq / 8;
    
    // Torre na 7ª fileira restringe rei na 8ª
    match context.color {
        Color::White => rook_ctx.rank == 6 && king_rank == 7,
        Color::Black => rook_ctx.rank == 1 && king_rank == 0,
    }
}

/// Verifica se pode criar bateria
fn can_create_battery(rook_sq: u8, context: &MobilityContext) -> bool {
    let file = rook_sq % 8;
    let rank = rook_sq / 8;
    
    // Verifica se há rainha ou outra torre no mesmo arquivo/fileira
    let file_mask = get_file_mask(file);
    let rank_mask = get_rank_mask(rank);
    
    let heavy_pieces = (context.board.rooks | context.board.queens) & context.our_pieces;
    let same_file_pieces = (heavy_pieces & file_mask).count_ones();
    let same_rank_pieces = (heavy_pieces & rank_mask).count_ones();
    
    same_file_pieces > 1 || same_rank_pieces > 1
}

/// Verifica se pode criar pin
fn can_create_pin(rook_sq: u8, attacks: Bitboard, context: &MobilityContext) -> bool {
    // Procura por peças inimigas valiosas que podem ser "pinned"
    let valuable_enemies = (context.board.queens | context.board.rooks | 
                           context.board.bishops | context.board.knights) & context.enemy_pieces;
    
    (attacks & valuable_enemies).count_ones() >= 2
}

/// Verifica se pode criar ataque descoberto
fn can_create_discovered_attack(rook_ctx: &RookContext, context: &MobilityContext) -> bool {
    // Simplificado - verifica se há peças próprias que podem se mover e descobrir ataque
    let our_pieces_on_line = rook_ctx.attacks & context.our_pieces;
    our_pieces_on_line.count_ones() >= 1
}

/// Verifica se é posição de rook lift
fn is_rook_lift_position(rook_ctx: &RookContext, context: &MobilityContext) -> bool {
    // Torre moveu-se da fileira inicial para uma posição de ataque lateral
    let initial_rank = if context.color == Color::White { 0 } else { 7 };
    rook_ctx.rank != initial_rank && rook_ctx.rank >= 2 && rook_ctx.rank <= 5
}

/// Verifica se deveria trocar torres
fn should_trade_rooks(rook_ctx: &RookContext, context: &MobilityContext) -> bool {
    // Incentiva trocas quando:
    // 1. Estamos em vantagem material
    // 2. Torre inimiga está mais ativa
    // 3. Endgame favorável para nós
    
    match context.phase {
        GamePhase::Endgame => {
            // No endgame, torres ativas são valiosas
            matches!(rook_ctx.endgame_potential, EndgamePotential::Dominant | EndgamePotential::Active)
        },
        _ => false,
    }
}

/// Avalia alvos da torre
fn evaluate_rook_targets(rook_ctx: &RookContext, context: &MobilityContext) -> i32 {
    let mut score = 0;
    
    let valuable_targets = (context.board.queens | context.board.rooks | 
                           context.board.bishops | context.board.knights) & context.enemy_pieces;
    
    let attacked_valuable = (rook_ctx.attacks & valuable_targets).count_ones();
    score += attacked_valuable as i32 * 5;
    
    // Ataques ao rei inimigo
    let enemy_king = context.board.kings & context.enemy_pieces;
    if enemy_king != 0 && (rook_ctx.attacks & enemy_king) != 0 {
        score += 15;
    }
    
    score
}

/// Avalia controle de casas fracas
fn evaluate_weak_square_control(rook_ctx: &RookContext, context: &MobilityContext) -> i32 {
    let mut score = 0;
    
    // Define casas fracas como casas importantes não defendidas pelo inimigo
    let important_squares = get_important_squares(context);
    let weak_squares = important_squares & !context.enemy_attacked_squares;
    let controlled_weak_squares = rook_ctx.attacks & weak_squares;
    
    score += controlled_weak_squares.count_ones() as i32 * WEAK_SQUARE_CONTROL_BONUS;
    
    score
}

/// Obtém casas importantes para controle
fn get_important_squares(context: &MobilityContext) -> Bitboard {
    let center_squares = 0x0000001818000000u64; // e4, d4, e5, d5
    let extended_center = 0x00003C3C3C3C0000u64; // c3-f6 área
    
    center_squares | extended_center
}

/// Avalia sinergia entre torres
fn evaluate_rook_synergy(rook_contexts: &[RookContext], context: &MobilityContext) -> i32 {
    let mut synergy = 0;
    
    // Torres em bateria no mesmo arquivo
    for i in 0..rook_contexts.len() {
        for j in (i+1)..rook_contexts.len() {
            if rook_contexts[i].file == rook_contexts[j].file {
                synergy += BATTERY_FORMATION_BONUS;
            }
        }
    }
    
    synergy
}

/// Avalia benefícios específicos por fase
fn evaluate_phase_specific_rook_benefits(rook_contexts: &[RookContext], context: &MobilityContext) -> i32 {
    let mut score = 0;
    
    for rook_ctx in rook_contexts {
        match context.phase {
            GamePhase::Opening => {
                // Penaliza desenvolvimento prematuro
                if rook_ctx.rank != 0 && rook_ctx.rank != 7 {
                    score -= 10;
                }
            },
            GamePhase::MiddleGame => {
                // Premia atividade e coordenação
                score += evaluate_middlegame_activity(rook_ctx, context);
            },
            GamePhase::Endgame => {
                // Premia centralização e atividade
                score += evaluate_endgame_activity(rook_ctx, context);
            },
        }
    }
    
    score
}

/// Avalia atividade no meio-jogo
fn evaluate_middlegame_activity(rook_ctx: &RookContext, context: &MobilityContext) -> i32 {
    let mut score = 0;
    
    // Premia torres ativas
    if rook_ctx.safe_moves.count_ones() >= 8 {
        score += 15;
    }
    
    // Premia controle de arquivos
    match rook_ctx.file_control {
        FileControlType::OpenFile | FileControlType::SemiOpen => score += 20,
        _ => {},
    }
    
    score
}

/// Avalia atividade no endgame
fn evaluate_endgame_activity(rook_ctx: &RookContext, context: &MobilityContext) -> i32 {
    let mut score = 0;
    
    // Centralização é crucial no endgame
    if is_central_square(rook_ctx.rook_sq) {
        score += 25;
    } else if is_extended_center(rook_ctx.rook_sq) {
        score += 15;
    }
    
    // Atividade contra peões inimigos
    let enemy_pawns = context.board.pawns & context.enemy_pieces;
    let attacked_pawns = (rook_ctx.attacks & enemy_pawns).count_ones();
    score += attacked_pawns as i32 * 8;
    
    score
}

/// Avalia contribuição tática específica
fn evaluate_tactical_role_contribution(rook_ctx: &RookContext, context: &MobilityContext) -> i32 {
    match rook_ctx.tactical_role {
        TacticalRole::Attacker => 20,
        TacticalRole::Battery => BATTERY_FORMATION_BONUS,
        TacticalRole::Pinning => TACTICAL_PIN_BONUS,
        TacticalRole::DiscoveredAttack => DISCOVERED_ATTACK_BONUS,
        TacticalRole::Lifter => ROOK_LIFT_BONUS,
        TacticalRole::Defender => 10,
        TacticalRole::Support => 5,
    }
}

/// Avalia contribuição para o endgame
fn evaluate_endgame_contribution(rook_ctx: &RookContext, context: &MobilityContext) -> i32 {
    if context.phase != GamePhase::Endgame {
        return 0;
    }
    
    match rook_ctx.endgame_potential {
        EndgamePotential::Dominant => 35,
        EndgamePotential::Active => 20,
        EndgamePotential::Converting => 30,
        EndgamePotential::Passive => -10,
        EndgamePotential::Trapped => -25,
    }
}