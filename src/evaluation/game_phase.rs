// Sistema revolucionário de detecção de fase com transições contínuas e múltiplos fatores
use crate::board::Board;
use crate::types::{Color, Bitboard};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GamePhase {
    Opening,
    EarlyMiddlegame,
    Middlegame,
    LateMiddlegame,
    EarlyEndgame,
    Endgame,
    LateEndgame,        // Nova fase: 5-8 peças
    PureEndgame,        // <5 peças
    TheoreticalEndgame, // Posições de tablebase conhecidas
}

/// Sistema de avaliação dinâmica com pesos ajustáveis
#[derive(Debug, Clone, Copy)]
pub struct DynamicEvaluation {
    pub material_weight: f32,      // 0.7-1.3
    pub positional_weight: f32,    // 0.5-1.5
    pub tactical_weight: f32,      // 0.8-2.0
    pub king_safety_weight: f32,   // 0.6-1.8
    pub development_weight: f32,   // 0.8-1.5
    pub complexity_factor: f32,    // Aumenta com táticas
    pub time_pressure_factor: f32, // Considera relógio
    pub imbalance_factor: f32,     // Para desequilíbrios materiais
}

impl Default for DynamicEvaluation {
    fn default() -> Self {
        Self {
            material_weight: 1.0,
            positional_weight: 1.0,
            tactical_weight: 1.0,
            king_safety_weight: 1.0,
            development_weight: 1.0,
            complexity_factor: 1.0,
            time_pressure_factor: 1.0,
            imbalance_factor: 1.0,
        }
    }
}

/// Contexto de avaliação multinível
#[derive(Debug, Clone)]
pub struct EvaluationContext {
    pub dynamic_eval: DynamicEvaluation,
    pub phase_info: GamePhaseInfo,
    pub material_factor: f32,
    pub development_factor: f32,
    pub pawn_structure_factor: f32,
    pub king_safety_factor: f32,
    pub rook_position_factor: f32,
    pub centralization_factor: f32,
    pub tactical_complexity: f32,
    pub positional_characteristics: PositionalCharacteristics,
}

/// Características posicionais específicas
#[derive(Debug, Clone)]
pub struct PositionalCharacteristics {
    pub has_isolated_queen_pawn: bool,
    pub has_carlsbad_structure: bool,
    pub has_pawn_chain: bool,
    pub has_weak_squares: bool,
    pub has_bishop_pair: [bool; 2], // [White, Black]
    pub has_rook_on_open_file: [bool; 2],
    pub castling_rights_lost: [bool; 2],
    pub material_imbalance_type: MaterialImbalanceType,
}

#[derive(Debug, Clone, Copy)]
pub enum MaterialImbalanceType {
    Balanced,
    MinorPieceImbalance,  // B vs N
    ExchangeImbalance,    // R vs B+N
    QueenImbalance,       // Q vs R+R or Q vs pieces
    PawnImbalance,        // Extra pawns
}

/// Estrutura avançada para informações da fase do jogo
#[derive(Debug, Clone, Copy)]
pub struct GamePhaseInfo {
    pub phase: GamePhase,
    pub material_count: u32,
    pub phase_value: f32,        // 0.0 (opening) to 1.0 (pure endgame)
    pub transition_smoothness: f32, // Measure of how gradual the transition is
    pub material_score: f32,     // Factor 1: 25%
    pub development_score: f32,  // Factor 2: 20%
    pub pawn_structure_score: f32, // Factor 3: 20%
    pub king_safety_score: f32,  // Factor 4: 15%
    pub rook_position_score: f32, // Factor 5: 10%
    pub centralization_score: f32, // Factor 6: 10%
    pub confidence_level: f32,   // How confident we are in this phase detection
}

const MATERIAL_WEIGHT: f32 = 0.25;
const DEVELOPMENT_WEIGHT: f32 = 0.20;
const PAWN_STRUCTURE_WEIGHT: f32 = 0.20;
const KING_SAFETY_WEIGHT: f32 = 0.15;
const ROOK_POSITION_WEIGHT: f32 = 0.10;
const CENTRALIZATION_WEIGHT: f32 = 0.10;

/// Backward compatibility
pub fn detect_game_phase(board: &Board) -> GamePhase {
    detect_game_phase_revolutionary(board).phase
}

/// Sistema revolucionário de detecção de fase
pub fn detect_game_phase_revolutionary(board: &Board) -> GamePhaseInfo {
    // Factor 1: Material total (25%)
    let material_score = calculate_material_factor(board);
    
    // Factor 2: Desenvolvimento das peças (20%)
    let development_score = calculate_development_factor(board);
    
    // Factor 3: Estrutura de peões (20%)
    let pawn_structure_score = calculate_pawn_structure_factor(board);
    
    // Factor 4: Segurança dos reis (15%)
    let king_safety_score = calculate_king_safety_factor(board);
    
    // Factor 5: Posição das torres (10%)
    let rook_position_score = calculate_rook_position_factor(board);
    
    // Factor 6: Centralização (10%)
    let centralization_score = calculate_centralization_factor(board);
    
    // Compute weighted phase value
    let phase_value = (material_score * MATERIAL_WEIGHT +
                      development_score * DEVELOPMENT_WEIGHT +
                      pawn_structure_score * PAWN_STRUCTURE_WEIGHT +
                      king_safety_score * KING_SAFETY_WEIGHT +
                      rook_position_score * ROOK_POSITION_WEIGHT +
                      centralization_score * CENTRALIZATION_WEIGHT).clamp(0.0, 1.0);
    
    // Determine discrete phase
    // Primeiro verifica critérios especiais baseados em contagem de peças
    let total_pieces = (board.white_pieces | board.black_pieces).count_ones();
    let phase = if total_pieces <= 4 {
        // Verifica se é posição teórica conhecida
        if is_theoretical_position(board) {
            GamePhase::TheoreticalEndgame
        } else {
            GamePhase::PureEndgame
        }
    } else if total_pieces <= 8 {
        GamePhase::LateEndgame
    } else {
        // Usa sistema híbrido: phase_value + piece count
        match phase_value {
            x if x < 0.10 => GamePhase::Opening,
            x if x < 0.25 => GamePhase::EarlyMiddlegame,
            x if x < 0.50 => GamePhase::Middlegame,
            x if x < 0.70 => GamePhase::LateMiddlegame,
            x if x < 0.85 => GamePhase::EarlyEndgame,
            x if x < 0.95 => {
                // Refina baseado em contagem de peças
                if total_pieces <= 12 { GamePhase::Endgame } else { GamePhase::EarlyEndgame }
            },
            _ => {
                // Para valores muito altos, usa contagem de peças
                if total_pieces <= 8 { GamePhase::LateEndgame } else { GamePhase::Endgame }
            }
        }
    };
    
    // Calculate transition smoothness and confidence
    let transition_smoothness = calculate_transition_smoothness(&[
        material_score, development_score, pawn_structure_score,
        king_safety_score, rook_position_score, centralization_score
    ]);
    
    let confidence_level = calculate_confidence_level(board, phase_value);
    
    GamePhaseInfo {
        phase,
        material_count: (board.white_pieces | board.black_pieces).count_ones(),
        phase_value,
        transition_smoothness,
        material_score,
        development_score,
        pawn_structure_score,
        king_safety_score,
        rook_position_score,
        centralization_score,
        confidence_level,
    }
}

/// Calcula fator material com consideração de qualidade
fn calculate_material_factor(board: &Board) -> f32 {
    let max_material = 78.0; // Starting material value
    
    // Standard piece values
    let knight_value = 3.0;
    let bishop_value = 3.2;
    let rook_value = 5.0;
    let queen_value = 9.0;
    
    let current_material = 
        (board.knights.count_ones() as f32) * knight_value +
        (board.bishops.count_ones() as f32) * bishop_value +
        (board.rooks.count_ones() as f32) * rook_value +
        (board.queens.count_ones() as f32) * queen_value;
    
    // Apply quality adjustments
    let quality_factor = calculate_material_quality(board);
    
    let material_ratio = (max_material - current_material) / max_material;
    (material_ratio * quality_factor).clamp(0.0, 1.0)
}

/// Calcula qualidade do material (não só quantidade)
fn calculate_material_quality(board: &Board) -> f32 {
    let mut quality = 1.0;
    
    // Bishop pair bonus
    let white_bishops = (board.bishops & board.white_pieces).count_ones();
    let black_bishops = (board.bishops & board.black_pieces).count_ones();
    
    if white_bishops >= 2 { quality += 0.1; }
    if black_bishops >= 2 { quality += 0.1; }
    
    // Rook on open file consideration
    quality += calculate_rook_quality_bonus(board);
    
    // Knight vs Bishop considerations
    quality += calculate_minor_piece_quality(board);
    
    quality.clamp(0.8, 1.3)
}

/// Calcula desenvolvimento das peças
fn calculate_development_factor(board: &Board) -> f32 {
    let mut development_score = 0.0;
    let mut total_pieces = 0.0;
    
    // Check knight development
    let white_knights = board.knights & board.white_pieces;
    let black_knights = board.knights & board.black_pieces;
    
    development_score += calculate_piece_development(white_knights, Color::White, "knight");
    development_score += calculate_piece_development(black_knights, Color::Black, "knight");
    total_pieces += 4.0; // 2 knights per side
    
    // Check bishop development  
    let white_bishops = board.bishops & board.white_pieces;
    let black_bishops = board.bishops & board.black_pieces;
    
    development_score += calculate_piece_development(white_bishops, Color::White, "bishop");
    development_score += calculate_piece_development(black_bishops, Color::Black, "bishop");
    total_pieces += 4.0; // 2 bishops per side
    
    // Check castling status
    development_score += calculate_castling_development(board);
    total_pieces += 2.0; // 2 kings
    
    if total_pieces > 0.0 {
        development_score / total_pieces
    } else {
        1.0
    }
}

/// Calcula desenvolvimento de um tipo de peça específico
fn calculate_piece_development(pieces: Bitboard, color: Color, piece_type: &str) -> f32 {
    if pieces == 0 { return 0.0; }
    
    let mut development = 0.0;
    let mut bb = pieces;
    
    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        
        let is_developed = match piece_type {
            "knight" => is_knight_developed(sq, color),
            "bishop" => is_bishop_developed(sq, color),
            _ => false,
        };
        
        if is_developed {
            development += 1.0;
        }
    }
    
    development
}

fn is_knight_developed(sq: u8, color: Color) -> bool {
    let starting_squares = if color == Color::White {
        [1, 6] // b1, g1
    } else {
        [57, 62] // b8, g8
    };
    
    !starting_squares.contains(&sq)
}

fn is_bishop_developed(sq: u8, color: Color) -> bool {
    let starting_squares = if color == Color::White {
        [2, 5] // c1, f1
    } else {
        [58, 61] // c8, f8
    };
    
    !starting_squares.contains(&sq)
}

/// Calcula fator de estrutura de peões
fn calculate_pawn_structure_factor(board: &Board) -> f32 {
    let white_pawns = board.pawns & board.white_pieces;
    let black_pawns = board.pawns & board.black_pieces;
    
    let total_pawns = white_pawns.count_ones() + black_pawns.count_ones();
    let max_pawns = 16.0;
    
    // Basic pawn count factor
    let pawn_count_factor = (max_pawns - total_pawns as f32) / max_pawns;
    
    // Structural complexity factor
    let structure_complexity = calculate_pawn_structure_complexity(board);
    
    (pawn_count_factor * 0.7 + structure_complexity * 0.3).clamp(0.0, 1.0)
}

/// Calcula complexidade da estrutura de peões
fn calculate_pawn_structure_complexity(board: &Board) -> f32 {
    let mut complexity = 0.0;
    
    // Count pawn islands
    complexity += count_pawn_islands(board) * 0.1;
    
    // Count passed pawns (reduces complexity as endgame approaches)
    complexity += count_passed_pawns(board) * 0.15;
    
    // Count doubled pawns
    complexity += count_doubled_pawns(board) * 0.05;
    
    // Count isolated pawns
    complexity += count_isolated_pawns(board) * 0.1;
    
    complexity.clamp(0.0, 1.0)
}

/// Calcula segurança dos reis
fn calculate_king_safety_factor(board: &Board) -> f32 {
    let white_king = board.kings & board.white_pieces;
    let black_king = board.kings & board.black_pieces;
    
    if white_king == 0 || black_king == 0 { return 1.0; }
    
    let white_king_sq = white_king.trailing_zeros() as u8;
    let black_king_sq = black_king.trailing_zeros() as u8;
    
    let white_safety = calculate_individual_king_safety(white_king_sq, Color::White, board);
    let black_safety = calculate_individual_king_safety(black_king_sq, Color::Black, board);
    
    // Lower safety = later phase (kings more active)
    1.0 - ((white_safety + black_safety) / 2.0).clamp(0.0, 1.0)
}

/// Calcula segurança individual do rei
fn calculate_individual_king_safety(king_sq: u8, color: Color, board: &Board) -> f32 {
    let mut safety = 0.0;
    
    // Check if king is still on back rank
    let back_rank = if color == Color::White {
        king_sq >= 0 && king_sq <= 7
    } else {
        king_sq >= 56 && king_sq <= 63
    };
    
    if back_rank {
        safety += 0.5;
    }
    
    // Check pawn shield
    safety += calculate_pawn_shield_safety(king_sq, color, board);
    
    // Check king exposure
    safety -= calculate_king_exposure(king_sq, board) * 0.3;
    
    safety.clamp(0.0, 1.0)
}

/// Calcula posicionamento das torres
fn calculate_rook_position_factor(board: &Board) -> f32 {
    let white_rooks = board.rooks & board.white_pieces;
    let black_rooks = board.rooks & board.black_pieces;
    
    if white_rooks == 0 && black_rooks == 0 { return 1.0; }
    
    let mut activity_score = 0.0;
    let mut rook_count = 0;
    
    // Evaluate white rooks
    let mut rooks_bb = white_rooks;
    while rooks_bb != 0 {
        let sq = rooks_bb.trailing_zeros() as u8;
        rooks_bb &= rooks_bb - 1;
        activity_score += calculate_rook_activity(sq, Color::White, board);
        rook_count += 1;
    }
    
    // Evaluate black rooks
    let mut rooks_bb = black_rooks;
    while rooks_bb != 0 {
        let sq = rooks_bb.trailing_zeros() as u8;
        rooks_bb &= rooks_bb - 1;
        activity_score += calculate_rook_activity(sq, Color::Black, board);
        rook_count += 1;
    }
    
    if rook_count > 0 {
        (activity_score / rook_count as f32).clamp(0.0, 1.0)
    } else {
        1.0
    }
}

/// Calcula atividade de uma torre
fn calculate_rook_activity(rook_sq: u8, color: Color, board: &Board) -> f32 {
    let mut activity: f32 = 0.0;
    
    // Check if on open file
    let file = rook_sq % 8;
    if is_open_file(file, board) {
        activity += 0.4;
    }
    
    // Check if on 7th/2nd rank
    let rank = rook_sq / 8;
    let seventh_rank = if color == Color::White { rank == 6 } else { rank == 1 };
    if seventh_rank {
        activity += 0.3;
    }
    
    // Check centralization
    if is_central_file(file) {
        activity += 0.2;
    }
    
    // Check if still on back rank (negative for activity)
    let back_rank = if color == Color::White { rank == 0 } else { rank == 7 };
    if back_rank {
        activity -= 0.2;
    }
    
    activity.clamp(0.0, 1.0)
}

/// Calcula centralização das peças
fn calculate_centralization_factor(board: &Board) -> f32 {
    let central_squares = 0x0000001818000000u64; // d4, e4, d5, e5
    let extended_center = 0x00003C3C3C3C0000u64; // c3-f3 to c6-f6
    
    let pieces = (board.knights | board.bishops | board.queens) & 
                 (board.white_pieces | board.black_pieces);
    
    let central_pieces = (pieces & central_squares).count_ones() as f32;
    let extended_central_pieces = (pieces & extended_center).count_ones() as f32;
    
    let total_pieces = pieces.count_ones() as f32;
    
    if total_pieces > 0.0 {
        (central_pieces * 0.6 + extended_central_pieces * 0.4) / total_pieces
    } else {
        0.0
    }
}

/// Calcula suavidade da transição
fn calculate_transition_smoothness(scores: &[f32]) -> f32 {
    if scores.len() < 2 { return 1.0; }
    
    let mut variance = 0.0;
    let mean: f32 = scores.iter().sum::<f32>() / scores.len() as f32;
    
    for &score in scores {
        variance += (score - mean).powi(2);
    }
    
    variance /= scores.len() as f32;
    
    // Lower variance = smoother transition
    1.0 - variance.sqrt().clamp(0.0, 1.0)
}

/// Calcula confiança na detecção de fase
fn calculate_confidence_level(board: &Board, phase_value: f32) -> f32 {
    let mut confidence: f32 = 1.0;
    
    // Reduce confidence in transition zones
    let transition_zones = [0.10, 0.25, 0.50, 0.70, 0.85, 0.95];
    for &zone in &transition_zones {
        let distance = (phase_value - zone).abs();
        if distance < 0.05 {
            confidence -= 0.2;
        }
    }
    
    // Increase confidence with clear indicators
    let total_pieces = (board.white_pieces | board.black_pieces).count_ones();
    if total_pieces <= 10 { confidence += 0.2; } // Clear endgame
    if total_pieces >= 28 { confidence += 0.2; } // Clear opening
    
    confidence.clamp(0.3, 1.0)
}

/// Backward compatibility
pub fn detect_game_phase_advanced(board: &Board) -> GamePhaseInfo {
    detect_game_phase_revolutionary(board)
}

/// Sistema de interpolação avançado
pub fn interpolate_phase_i32(opening: i32, endgame: i32, phase_info: &GamePhaseInfo) -> i32 {
    let ratio = phase_info.phase_value;
    
    // Apply smoothing function for better transitions
    let smoothed_ratio = smooth_transition(ratio);
    
    // Consider confidence level
    let confidence_adjusted_ratio = apply_confidence_adjustment(smoothed_ratio, phase_info.confidence_level);
    
    ((1.0 - confidence_adjusted_ratio) * opening as f32 + confidence_adjusted_ratio * endgame as f32) as i32
}

/// Função de suavização para transições
fn smooth_transition(ratio: f32) -> f32 {
    // Use sigmoid-like function for smoother transitions
    if ratio < 0.5 {
        2.0 * ratio * ratio
    } else {
        1.0 - 2.0 * (1.0 - ratio) * (1.0 - ratio)
    }
}

/// Ajusta ratio baseado na confiança
fn apply_confidence_adjustment(ratio: f32, confidence: f32) -> f32 {
    // When confidence is low, bias towards middle values
    if confidence < 0.7 {
        let bias_strength = 1.0 - confidence;
        ratio * (1.0 - bias_strength) + 0.5 * bias_strength
    } else {
        ratio
    }
}

/// Cria contexto de avaliação completo
pub fn create_evaluation_context(board: &Board) -> EvaluationContext {
    let phase_info = detect_game_phase_revolutionary(board);
    let dynamic_eval = calculate_dynamic_evaluation(&phase_info, board);
    let positional_characteristics = analyze_positional_characteristics(board);
    
    EvaluationContext {
        dynamic_eval,
        phase_info,
        material_factor: phase_info.material_score,
        development_factor: phase_info.development_score,
        pawn_structure_factor: phase_info.pawn_structure_score,
        king_safety_factor: phase_info.king_safety_score,
        rook_position_factor: phase_info.rook_position_score,
        centralization_factor: phase_info.centralization_score,
        tactical_complexity: calculate_tactical_complexity(board),
        positional_characteristics,
    }
}

/// Calcula avaliação dinâmica
fn calculate_dynamic_evaluation(phase_info: &GamePhaseInfo, board: &Board) -> DynamicEvaluation {
    let mut eval = DynamicEvaluation::default();
    
    // Adjust weights based on phase
    match phase_info.phase {
        GamePhase::Opening => {
            eval.development_weight = 1.3;
            eval.king_safety_weight = 1.2;
            eval.material_weight = 1.1;
        },
        GamePhase::Middlegame => {
            eval.tactical_weight = 1.4;
            eval.positional_weight = 1.2;
            eval.king_safety_weight = 1.3;
        },
        GamePhase::Endgame | GamePhase::PureEndgame => {
            eval.material_weight = 0.8;
            eval.king_safety_weight = 0.7;
            eval.positional_weight = 1.3;
        },
        _ => {}, // Use defaults for transition phases
    }
    
    // Adjust for tactical complexity
    eval.complexity_factor = calculate_tactical_complexity(board);
    
    // Adjust for material imbalances
    eval.imbalance_factor = calculate_imbalance_factor(board);
    
    eval
}

/// Analisa características posicionais
fn analyze_positional_characteristics(board: &Board) -> PositionalCharacteristics {
    PositionalCharacteristics {
        has_isolated_queen_pawn: detect_isolated_queen_pawn(board),
        has_carlsbad_structure: detect_carlsbad_structure(board),
        has_pawn_chain: detect_pawn_chains(board),
        has_weak_squares: detect_weak_squares(board),
        has_bishop_pair: [
            (board.bishops & board.white_pieces).count_ones() >= 2,
            (board.bishops & board.black_pieces).count_ones() >= 2,
        ],
        has_rook_on_open_file: [
            has_rook_on_open_file(board, Color::White),
            has_rook_on_open_file(board, Color::Black),
        ],
        castling_rights_lost: [false, false], // TODO: Implement based on actual castling rights
        material_imbalance_type: detect_material_imbalance_type(board),
    }
}

// === HELPER FUNCTIONS ===

fn calculate_rook_quality_bonus(board: &Board) -> f32 {
    let mut bonus = 0.0;
    
    for file in 0..8 {
        if is_open_file(file, board) {
            let white_rooks_on_file = count_rooks_on_file(file, board, Color::White);
            let black_rooks_on_file = count_rooks_on_file(file, board, Color::Black);
            bonus += (white_rooks_on_file + black_rooks_on_file) as f32 * 0.05;
        }
    }
    
    bonus.clamp(0.0, 0.3)
}

fn calculate_minor_piece_quality(board: &Board) -> f32 {
    // Simplified: bishop pair vs knight considerations
    let white_bishops = (board.bishops & board.white_pieces).count_ones();
    let black_bishops = (board.bishops & board.black_pieces).count_ones();
    
    let mut quality = 0.0;
    if white_bishops >= 2 { quality += 0.05; }
    if black_bishops >= 2 { quality += 0.05; }
    
    quality
}

fn calculate_castling_development(board: &Board) -> f32 {
    // Simplified implementation - check if kings moved from starting positions
    let white_king = board.kings & board.white_pieces;
    let black_king = board.kings & board.black_pieces;
    
    let mut development = 0.0;
    
    if white_king != 0 {
        let king_sq = white_king.trailing_zeros() as u8;
        if king_sq != 4 { development += 1.0; } // King moved from e1
    }
    
    if black_king != 0 {
        let king_sq = black_king.trailing_zeros() as u8;
        if king_sq != 60 { development += 1.0; } // King moved from e8
    }
    
    development
}

fn count_pawn_islands(board: &Board) -> f32 {
    // Simplified implementation
    let white_pawns = board.pawns & board.white_pieces;
    let black_pawns = board.pawns & board.black_pieces;
    
    let white_islands = count_islands_for_side(white_pawns);
    let black_islands = count_islands_for_side(black_pawns);
    
    (white_islands + black_islands) as f32
}

fn count_islands_for_side(pawns: Bitboard) -> u32 {
    let mut islands = 0;
    let mut prev_file_has_pawn = false;
    
    for file in 0..8 {
        let file_mask = 0x0101010101010101u64 << file;
        let file_has_pawn = (pawns & file_mask) != 0;
        
        if file_has_pawn && !prev_file_has_pawn {
            islands += 1;
        }
        prev_file_has_pawn = file_has_pawn;
    }
    
    islands
}

fn count_passed_pawns(board: &Board) -> f32 {
    // Simplified implementation
    let white_pawns = board.pawns & board.white_pieces;
    let black_pawns = board.pawns & board.black_pieces;
    
    let mut passed = 0;
    
    // Check white passed pawns
    let mut bb = white_pawns;
    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        if is_passed_pawn_simple(sq, Color::White, white_pawns, black_pawns) {
            passed += 1;
        }
    }
    
    // Check black passed pawns
    let mut bb = black_pawns;
    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        if is_passed_pawn_simple(sq, Color::Black, white_pawns, black_pawns) {
            passed += 1;
        }
    }
    
    passed as f32
}

fn count_doubled_pawns(board: &Board) -> f32 {
    let white_pawns = board.pawns & board.white_pieces;
    let black_pawns = board.pawns & board.black_pieces;
    
    let mut doubled = 0;
    
    for file in 0..8 {
        let file_mask = 0x0101010101010101u64 << file;
        let white_on_file = (white_pawns & file_mask).count_ones();
        let black_on_file = (black_pawns & file_mask).count_ones();
        
        if white_on_file > 1 { doubled += white_on_file - 1; }
        if black_on_file > 1 { doubled += black_on_file - 1; }
    }
    
    doubled as f32
}

fn count_isolated_pawns(board: &Board) -> f32 {
    // Simplified implementation
    let white_pawns = board.pawns & board.white_pieces;
    let black_pawns = board.pawns & board.black_pieces;
    
    let mut isolated = 0;
    
    for file in 0..8 {
        let file_mask = 0x0101010101010101u64 << file;
        let adjacent_files = get_adjacent_files_mask(file);
        
        if (white_pawns & file_mask) != 0 && (white_pawns & adjacent_files) == 0 {
            isolated += 1;
        }
        if (black_pawns & file_mask) != 0 && (black_pawns & adjacent_files) == 0 {
            isolated += 1;
        }
    }
    
    isolated as f32
}

fn calculate_pawn_shield_safety(king_sq: u8, color: Color, board: &Board) -> f32 {
    // Simplified pawn shield calculation
    let our_pawns = if color == Color::White {
        board.pawns & board.white_pieces
    } else {
        board.pawns & board.black_pieces
    };
    
    let king_file = king_sq % 8;
    let shield_files = [
        king_file.saturating_sub(1),
        king_file,
        (king_file + 1).min(7)
    ];
    
    let mut shield_strength = 0.0;
    for &file in &shield_files {
        let file_mask = 0x0101010101010101u64 << file;
        if (our_pawns & file_mask) != 0 {
            shield_strength += 0.33;
        }
    }
    
    shield_strength
}

fn calculate_king_exposure(king_sq: u8, board: &Board) -> f32 {
    // Simplified king exposure calculation
    let king_area = get_king_area_simple(king_sq);
    let total_pieces = board.white_pieces | board.black_pieces;
    let pieces_near_king = (king_area & total_pieces).count_ones();
    
    // More pieces near king = more exposed
    (pieces_near_king as f32 / 8.0).clamp(0.0, 1.0)
}

fn get_king_area_simple(king_sq: u8) -> Bitboard {
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

fn is_open_file(file: u8, board: &Board) -> bool {
    let file_mask = 0x0101010101010101u64 << file;
    (board.pawns & file_mask) == 0
}

fn is_central_file(file: u8) -> bool {
    file >= 2 && file <= 5 // c, d, e, f files
}

fn count_rooks_on_file(file: u8, board: &Board, color: Color) -> u32 {
    let rooks = if color == Color::White {
        board.rooks & board.white_pieces
    } else {
        board.rooks & board.black_pieces
    };
    
    let file_mask = 0x0101010101010101u64 << file;
    (rooks & file_mask).count_ones()
}

fn is_passed_pawn_simple(pawn_sq: u8, color: Color, our_pawns: Bitboard, enemy_pawns: Bitboard) -> bool {
    let file = pawn_sq % 8;
    let rank = pawn_sq / 8;
    
    // Get the mask for squares that would block this pawn
    let blocking_mask = if color == Color::White {
        // For white, check ranks above
        let mut mask = 0u64;
        for r in (rank + 1)..8 {
            for f in file.saturating_sub(1)..=(file + 1).min(7) {
                mask |= 1u64 << (r * 8 + f);
            }
        }
        mask
    } else {
        // For black, check ranks below
        let mut mask = 0u64;
        for r in 0..rank {
            for f in file.saturating_sub(1)..=(file + 1).min(7) {
                mask |= 1u64 << (r * 8 + f);
            }
        }
        mask
    };
    
    (enemy_pawns & blocking_mask) == 0
}

fn get_adjacent_files_mask(file: u8) -> Bitboard {
    let mut mask = 0u64;
    if file > 0 {
        mask |= 0x0101010101010101u64 << (file - 1);
    }
    if file < 7 {
        mask |= 0x0101010101010101u64 << (file + 1);
    }
    mask
}

fn calculate_tactical_complexity(board: &Board) -> f32 {
    // Simplified tactical complexity calculation
    let total_pieces = (board.white_pieces | board.black_pieces).count_ones() as f32;
    let max_pieces = 32.0;
    
    // More pieces = potentially more complex
    let piece_complexity = total_pieces / max_pieces;
    
    // Add factors for tactical pieces
    let tactical_pieces = (board.queens | board.rooks | board.bishops | board.knights).count_ones() as f32;
    let tactical_complexity = tactical_pieces / 20.0; // Max 20 tactical pieces
    
    ((piece_complexity + tactical_complexity) / 2.0).clamp(0.3, 2.0)
}

fn calculate_imbalance_factor(board: &Board) -> f32 {
    // Simplified material imbalance calculation
    let white_material = calculate_side_material(board, Color::White);
    let black_material = calculate_side_material(board, Color::Black);
    
    let total_material = white_material + black_material;
    if total_material > 0.0 {
        let imbalance = (white_material - black_material).abs() / total_material;
        1.0 + imbalance // 1.0 to 2.0 based on imbalance
    } else {
        1.0
    }
}

fn calculate_side_material(board: &Board, color: Color) -> f32 {
    let pieces = if color == Color::White {
        board.white_pieces
    } else {
        board.black_pieces
    };
    
    let knights = (board.knights & pieces).count_ones() as f32 * 3.0;
    let bishops = (board.bishops & pieces).count_ones() as f32 * 3.2;
    let rooks = (board.rooks & pieces).count_ones() as f32 * 5.0;
    let queens = (board.queens & pieces).count_ones() as f32 * 9.0;
    
    knights + bishops + rooks + queens
}

// === POSITIONAL PATTERN DETECTION ===

fn detect_isolated_queen_pawn(board: &Board) -> bool {
    // Simplified implementation
    for file in [3, 4] { // d and e files
        let file_mask = 0x0101010101010101u64 << file;
        let adjacent_files = get_adjacent_files_mask(file);
        
        if (board.pawns & file_mask) != 0 && (board.pawns & adjacent_files) == 0 {
            return true;
        }
    }
    false
}

fn detect_carlsbad_structure(board: &Board) -> bool {
    // Simplified detection for Carlsbad pawn structure
    let d_file = 0x0101010101010101u64 << 3;
    let e_file = 0x0101010101010101u64 << 4;
    
    let white_pawns = board.pawns & board.white_pieces;
    let black_pawns = board.pawns & board.black_pieces;
    
    // Look for typical Carlsbad characteristics
    (white_pawns & d_file) != 0 && (black_pawns & e_file) != 0
}

fn detect_pawn_chains(board: &Board) -> bool {
    // Simplified chain detection
    let white_pawns = board.pawns & board.white_pieces;
    let black_pawns = board.pawns & board.black_pieces;
    
    has_pawn_chain(white_pawns) || has_pawn_chain(black_pawns)
}

fn has_pawn_chain(pawns: Bitboard) -> bool {
    // Check for connected pawns (simplified)
    let mut bb = pawns;
    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        
        let file = sq % 8;
        let rank = sq / 8;
        
        // Check for supporting pawn
        for df in [-1i8, 1] {
            let support_file = file as i8 + df;
            let support_rank = rank as i8 - 1; // One rank behind
            
            if support_file >= 0 && support_file < 8 && support_rank >= 0 {
                let support_sq = (support_rank * 8 + support_file) as u8;
                if (pawns & (1u64 << support_sq)) != 0 {
                    return true;
                }
            }
        }
    }
    false
}

fn detect_weak_squares(board: &Board) -> bool {
    // Simplified weak square detection
    // Look for squares that can't be defended by pawns
    let white_pawns = board.pawns & board.white_pieces;
    let black_pawns = board.pawns & board.black_pieces;
    
    // Check central squares d4, e4, d5, e5
    let central_squares = [27, 28, 35, 36]; // d4, e4, d5, e5
    
    for &sq in &central_squares {
        if is_weak_square(sq, white_pawns) || is_weak_square(sq, black_pawns) {
            return true;
        }
    }
    
    false
}

fn is_weak_square(sq: u8, our_pawns: Bitboard) -> bool {
    let file = sq % 8;
    
    // Check if any pawn can defend this square
    for df in [-1i8, 1] {
        let pawn_file = file as i8 + df;
        if pawn_file >= 0 && pawn_file < 8 {
            let file_mask = 0x0101010101010101u64 << pawn_file;
            if (our_pawns & file_mask) != 0 {
                return false; // Can be defended
            }
        }
    }
    
    true // Weak square
}

fn has_rook_on_open_file(board: &Board, color: Color) -> bool {
    for file in 0..8 {
        if is_open_file(file, board) && count_rooks_on_file(file, board, color) > 0 {
            return true;
        }
    }
    
    false
}

fn detect_material_imbalance_type(board: &Board) -> MaterialImbalanceType {
    let white_material = calculate_detailed_material(board, Color::White);
    let black_material = calculate_detailed_material(board, Color::Black);
    
    // Compare material compositions
    if (white_material.0 - black_material.0).abs() > 0.5 {
        MaterialImbalanceType::MinorPieceImbalance
    } else if (white_material.1 - black_material.1).abs() >= 1.0 {
        MaterialImbalanceType::ExchangeImbalance
    } else if (white_material.2 - black_material.2).abs() >= 1.0 {
        MaterialImbalanceType::QueenImbalance
    } else if (white_material.3 - black_material.3).abs() >= 1.0 {
        MaterialImbalanceType::PawnImbalance
    } else {
        MaterialImbalanceType::Balanced
    }
}

fn calculate_detailed_material(board: &Board, color: Color) -> (f32, f32, f32, f32) {
    let pieces = if color == Color::White {
        board.white_pieces
    } else {
        board.black_pieces
    };
    
    let minor_pieces = ((board.knights & pieces).count_ones() + (board.bishops & pieces).count_ones()) as f32;
    let rooks = (board.rooks & pieces).count_ones() as f32;
    let queens = (board.queens & pieces).count_ones() as f32;
    let pawns = (board.pawns & pieces).count_ones() as f32;
    
    (minor_pieces, rooks, queens, pawns)
}

/// Verifica se a posição é um final teórico conhecido
fn is_theoretical_position(board: &Board) -> bool {
    let total_pieces = (board.white_pieces | board.black_pieces).count_ones();
    if total_pieces > 5 { return false; }
    
    let white_material = board.white_pieces & !board.kings;
    let black_material = board.black_pieces & !board.kings;
    let white_count = white_material.count_ones();
    let black_count = black_material.count_ones();
    
    // K vs K
    if white_count == 0 && black_count == 0 {
        return true;
    }
    
    // Material insuficiente: KN vs K, KB vs K (sem peões)
    if (white_count == 1 && black_count == 0) || (white_count == 0 && black_count == 1) {
        let stronger_material = if white_count > 0 { white_material } else { black_material };
        let is_knight = (stronger_material & board.knights) != 0;
        let is_bishop = (stronger_material & board.bishops) != 0;
        let no_pawns = board.pawns == 0;
        
        if no_pawns && (is_knight || is_bishop) {
            return true;
        }
    }
    
    // Finais teóricos básicos com material suficiente para mate
    if (white_count == 1 && black_count == 0) || (white_count == 0 && black_count == 1) {
        let stronger_material = if white_count > 0 { white_material } else { black_material };
        let has_queen = (stronger_material & board.queens) != 0;
        let has_rook = (stronger_material & board.rooks) != 0;
        let has_pawn = (stronger_material & board.pawns) != 0;
        
        if has_queen || has_rook || has_pawn {
            return true; // KQ vs K, KR vs K, KP vs K
        }
    }
    
    // Finais com duas peças: KBB vs K, KBN vs K, KNN vs K
    if (white_count == 2 && black_count == 0) || (white_count == 0 && black_count == 2) {
        return true;
    }
    
    false
}

/// Converte GamePhase para DetailedGamePhase (compatibilidade)
pub fn to_detailed_game_phase(phase: GamePhase) -> crate::evaluation::endgame::DetailedGamePhase {
    use crate::evaluation::endgame::DetailedGamePhase;
    
    match phase {
        GamePhase::Opening => DetailedGamePhase::Opening,
        GamePhase::EarlyMiddlegame => DetailedGamePhase::EarlyMiddlegame,
        GamePhase::Middlegame => DetailedGamePhase::Middlegame,
        GamePhase::LateMiddlegame => DetailedGamePhase::LateMiddlegame,
        GamePhase::EarlyEndgame => DetailedGamePhase::EarlyEndgame,
        GamePhase::Endgame => DetailedGamePhase::Endgame,
        GamePhase::LateEndgame => DetailedGamePhase::LateEndgame,
        GamePhase::PureEndgame => DetailedGamePhase::PureEndgame,
        GamePhase::TheoreticalEndgame => DetailedGamePhase::TheoreticalEndgame,
    }
}

/// Converte DetailedGamePhase para GamePhase (compatibilidade reversa)
pub fn from_detailed_game_phase(detailed_phase: crate::evaluation::endgame::DetailedGamePhase) -> GamePhase {
    use crate::evaluation::endgame::DetailedGamePhase;
    
    match detailed_phase {
        DetailedGamePhase::Opening => GamePhase::Opening,
        DetailedGamePhase::EarlyMiddlegame => GamePhase::EarlyMiddlegame,
        DetailedGamePhase::Middlegame => GamePhase::Middlegame,
        DetailedGamePhase::LateMiddlegame => GamePhase::LateMiddlegame,
        DetailedGamePhase::EarlyEndgame => GamePhase::EarlyEndgame,  
        DetailedGamePhase::Endgame => GamePhase::Endgame,
        DetailedGamePhase::LateEndgame => GamePhase::LateEndgame,
        DetailedGamePhase::PureEndgame => GamePhase::PureEndgame,
        DetailedGamePhase::TheoreticalEndgame => GamePhase::TheoreticalEndgame,
    }
}