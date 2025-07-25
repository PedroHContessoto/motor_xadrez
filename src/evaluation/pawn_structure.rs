// Sistema revolucionário integrado de estrutura e mobilidade de peões
use crate::{board::Board, types::{Color, Bitboard}};
use super::material::MATERIAL_VALUES;
use super::game_phase::{interpolate_phase_i32, GamePhaseInfo, detect_game_phase_advanced};

// Constantes avançadas para avaliação de peões
const PASSED_PAWN_BONUS: [i32; 8] = [0, 15, 25, 40, 65, 100, 150, 200];
const DOUBLED_PAWN_PENALTY: [i32; 3] = [20, 25, 15]; // [opening, middlegame, endgame]
const ISOLATED_PAWN_PENALTY: [i32; 3] = [15, 20, 25];
const BACKWARD_PAWN_PENALTY: [i32; 3] = [12, 18, 22];
const CONNECTED_PAWN_BONUS: [i32; 3] = [8, 12, 6];
const PAWN_CHAIN_BONUS: [i32; 3] = [10, 15, 8];
const PAWN_STORM_BONUS: [i32; 3] = [18, 25, 8];
const ADVANCED_PASSED_MULTIPLIER: [i32; 8] = [0, 1, 1, 2, 3, 5, 8, 12];
const KING_DISTANCE_FACTOR: i32 = 3;
const ENEMY_KING_DISTANCE_FACTOR: i32 = 2;

// Bônus para padrões específicos de peões
const PAWN_MAJORITY_BONUS: i32 = 25;
const PAWN_MINORITY_ATTACK_BONUS: i32 = 20;
const PAWN_BREAKTHROUGH_BONUS: i32 = 35;
const PAWN_LEVER_BONUS: i32 = 15;
const PAWN_OUTPOST_SUPPORT_BONUS: i32 = 12;
const PAWN_WEAKNESS_COMPENSATION: i32 = 8;

// Máscaras de bits para cada coluna (otimizadas)
const FILE_MASKS: [Bitboard; 8] = [
    0x0101010101010101, 0x0202020202020202, 0x0404040404040404, 0x0808080808080808,
    0x1010101010101010, 0x2020202020202020, 0x4040404040404040, 0x8080808080808080,
];

// Máscaras para peões passados (thread-safe com Once)
use std::sync::Once;
static INIT: Once = Once::new();
static mut PASSED_PAWN_MASKS: [[Bitboard; 64]; 2] = [[0; 64]; 2];

/// Estrutura avançada para análise completa de peões
#[derive(Debug, Clone)]
pub struct AdvancedPawnAnalysis {
    // Estrutura básica
    pub passed_pawns: Vec<PassedPawnInfo>,
    pub doubled_pawns: i32,
    pub isolated_pawns: i32,
    pub backward_pawns: i32,
    pub connected_pawns: i32,
    pub pawn_chains: i32,
    
    // Análise tática
    pub pawn_storms: i32,
    pub pawn_levers: i32,
    pub pawn_breaks: i32,
    pub minority_attacks: i32,
    pub majority_potential: i32,
    
    // Dinâmica posicional
    pub king_safety_contribution: i32,
    pub piece_coordination_support: i32,
    pub endgame_potential: i32,
    pub space_control: i32,
    
    // Mobilidade integrada
    pub pawn_mobility: i32,
    pub tactical_mobility: i32,
    pub strategic_advances: i32,
    
    // Avaliação final
    pub total_structural_score: i32,
    pub total_dynamic_score: i32,
}

/// Informação detalhada sobre peão passado
#[derive(Debug, Clone)]
pub struct PassedPawnInfo {
    pub square: u8,
    pub file: u8,
    pub rank: u8,
    pub distance_to_promotion: i32,
    pub is_protected: bool,
    pub is_advanced: bool,
    pub is_unstoppable: bool,
    pub king_support_distance: i32,
    pub enemy_king_distance: i32,
    pub mobility_value: i32,
    pub breakthrough_potential: i32,
}

/// Tipos de configuração de peões
#[derive(Debug, Clone, Copy)]
pub enum PawnFormation {
    Phalanx,        // Peões lado a lado
    Chain,          // Cadeia diagonal
    Storm,          // Ataque de peões
    Blockade,       // Bloqueio de peões
    Majority,       // Maioria de peões
    Minority,       // Ataque de minoria
    Isolated,       // Ilhas isoladas
}

/// Inicializa as máscaras de peões passados (thread-safe)
pub fn init_pawn_masks() {
    INIT.call_once(|| {
        unsafe {
            for sq in 0..64 {
                let rank = sq / 8;
                let file = sq % 8;

                // Máscara para Brancas (peões à frente e adjacentes)
                let mut white_mask: Bitboard = 0;
                for r in (rank + 1)..8 {
                    white_mask |= FILE_MASKS[file] & (0xFF << (r * 8));
                    if file > 0 { white_mask |= FILE_MASKS[file - 1] & (0xFF << (r * 8)); }
                    if file < 7 { white_mask |= FILE_MASKS[file + 1] & (0xFF << (r * 8)); }
                }
                PASSED_PAWN_MASKS[Color::White as usize][sq] = white_mask;

                // Máscara para Pretas
                let mut black_mask: Bitboard = 0;
                for r in 0..rank {
                    black_mask |= FILE_MASKS[file] & (0xFF << (r * 8));
                    if file > 0 { black_mask |= FILE_MASKS[file - 1] & (0xFF << (r * 8)); }
                    if file < 7 { black_mask |= FILE_MASKS[file + 1] & (0xFF << (r * 8)); }
                }
                PASSED_PAWN_MASKS[Color::Black as usize][sq] = black_mask;
            }
        }
    });
}

/// Avaliação revolucionária integrada de estrutura de peões
pub fn evaluate_pawn_structure_advanced(board: &Board, color: Color) -> AdvancedPawnAnalysis {
    init_pawn_masks();
    
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let our_pawns = board.pawns & pieces;
    let enemy_pawns = board.pawns & if color == Color::White { board.black_pieces } else { board.white_pieces };
    
    if our_pawns == 0 {
        return AdvancedPawnAnalysis::new();
    }

    let phase_info = detect_game_phase_advanced(board);
    let mut analysis = AdvancedPawnAnalysis::new();

    // === ANÁLISE ESTRUTURAL BÁSICA ===
    analysis.passed_pawns = evaluate_passed_pawns_advanced(our_pawns, enemy_pawns, color, board);
    analysis.doubled_pawns = evaluate_doubled_pawns_advanced(our_pawns, &phase_info);
    analysis.isolated_pawns = evaluate_isolated_pawns_advanced(our_pawns, &phase_info);
    analysis.backward_pawns = evaluate_backward_pawns_advanced(our_pawns, enemy_pawns, color, board, &phase_info);
    analysis.connected_pawns = evaluate_connected_pawns_advanced(our_pawns, &phase_info);
    analysis.pawn_chains = evaluate_pawn_chains_advanced(our_pawns, color, &phase_info);

    // === ANÁLISE TÁTICA ===
    analysis.pawn_storms = evaluate_pawn_storms_advanced(our_pawns, enemy_pawns, color, board, &phase_info);
    analysis.pawn_levers = evaluate_pawn_levers(our_pawns, enemy_pawns, color);
    analysis.pawn_breaks = evaluate_pawn_breaks(our_pawns, enemy_pawns, color, board);
    analysis.minority_attacks = evaluate_minority_attacks(our_pawns, enemy_pawns, color);
    analysis.majority_potential = evaluate_majority_potential(our_pawns, enemy_pawns, color);

    // === DINÂMICA POSICIONAL ===
    analysis.king_safety_contribution = evaluate_king_safety_contribution(our_pawns, board, color);
    analysis.piece_coordination_support = evaluate_piece_coordination_support(our_pawns, board, color);
    analysis.endgame_potential = evaluate_endgame_potential(our_pawns, enemy_pawns, color, board);
    analysis.space_control = evaluate_space_control(our_pawns, color);

    // === MOBILIDADE INTEGRADA ===
    analysis.pawn_mobility = evaluate_pawn_mobility_integrated(our_pawns, enemy_pawns, color, board);
    analysis.tactical_mobility = evaluate_tactical_mobility(our_pawns, enemy_pawns, color, board);
    analysis.strategic_advances = evaluate_strategic_advances(our_pawns, enemy_pawns, color, board);

    // === CÁLCULO FINAL ===
    analysis.calculate_total_scores(&phase_info);
    
    analysis
}

/// Função de compatibilidade com o sistema antigo
pub fn evaluate_pawn_structure(board: &Board, color: Color) -> i32 {
    let analysis = evaluate_pawn_structure_advanced(board, color);
    analysis.total_structural_score + analysis.total_dynamic_score
}

impl AdvancedPawnAnalysis {
    pub fn new() -> Self {
        AdvancedPawnAnalysis {
            passed_pawns: Vec::new(),
            doubled_pawns: 0,
            isolated_pawns: 0,
            backward_pawns: 0,
            connected_pawns: 0,
            pawn_chains: 0,
            pawn_storms: 0,
            pawn_levers: 0,
            pawn_breaks: 0,
            minority_attacks: 0,
            majority_potential: 0,
            king_safety_contribution: 0,
            piece_coordination_support: 0,
            endgame_potential: 0,
            space_control: 0,
            pawn_mobility: 0,
            tactical_mobility: 0,
            strategic_advances: 0,
            total_structural_score: 0,
            total_dynamic_score: 0,
        }
    }

    /// Calcula scores totais com interpolação de fase
    pub fn calculate_total_scores(&mut self, phase_info: &GamePhaseInfo) {
        // Score estrutural (penalidades/bônus básicos)
        self.total_structural_score = 
            self.passed_pawns.iter().map(|p| p.mobility_value + p.breakthrough_potential).sum::<i32>() +
            self.connected_pawns + self.pawn_chains - 
            self.doubled_pawns - self.isolated_pawns - self.backward_pawns;

        // Score dinâmico (aspectos táticos e posicionais)
        self.total_dynamic_score = 
            self.pawn_storms + self.pawn_levers + self.pawn_breaks +
            self.minority_attacks + self.majority_potential +
            self.king_safety_contribution + self.piece_coordination_support +
            self.endgame_potential + self.space_control +
            self.pawn_mobility + self.tactical_mobility + self.strategic_advances;

        // Aplica interpolação de fase para ajustar importância
        self.total_structural_score = interpolate_phase_i32(
            self.total_structural_score,
            (self.total_structural_score as f32 * 1.2) as i32, // Mais importante no endgame
            phase_info
        );
    }
}

/// Avaliação avançada de peões passados
fn evaluate_passed_pawns_advanced(our_pawns: Bitboard, enemy_pawns: Bitboard, color: Color, board: &Board) -> Vec<PassedPawnInfo> {
    let mut passed_pawns = Vec::new();
    let mut pawns = our_pawns;
    
    // Localiza reis para cálculo de distâncias
    let our_king = board.kings & if color == Color::White { board.white_pieces } else { board.black_pieces };
    let enemy_king = board.kings & if color == Color::White { board.black_pieces } else { board.white_pieces };
    
    let our_king_sq = if our_king != 0 { our_king.trailing_zeros() as u8 } else { 64 };
    let enemy_king_sq = if enemy_king != 0 { enemy_king.trailing_zeros() as u8 } else { 64 };

    while pawns != 0 {
        let sq = pawns.trailing_zeros() as usize;
        pawns &= pawns - 1;

        unsafe {
            if (PASSED_PAWN_MASKS[color as usize][sq] & enemy_pawns) == 0 {
                let pawn_info = create_passed_pawn_info(sq as u8, color, our_king_sq, enemy_king_sq, board);
                passed_pawns.push(pawn_info);
            }
        }
    }

    passed_pawns
}

/// Cria informação detalhada sobre peão passado
fn create_passed_pawn_info(sq: u8, color: Color, our_king_sq: u8, enemy_king_sq: u8, board: &Board) -> PassedPawnInfo {
    let file = sq % 8;
    let rank = sq / 8;
    
    let distance_to_promotion = if color == Color::White {
        7 - rank as i32
    } else {
        rank as i32
    };

    let is_advanced = distance_to_promotion <= 3;
    let is_protected = is_pawn_protected(sq, color, board);
    let is_unstoppable = is_pawn_unstoppable(sq, color, enemy_king_sq, board);
    
    let king_support_distance = if our_king_sq < 64 {
        calculate_square_distance(sq, our_king_sq)
    } else { 99 };
    
    let enemy_king_distance = if enemy_king_sq < 64 {
        calculate_square_distance(sq, enemy_king_sq)
    } else { 99 };

    let base_bonus = PASSED_PAWN_BONUS[rank as usize];
    let advancement_multiplier = ADVANCED_PASSED_MULTIPLIER[rank as usize];
    let king_factor = if king_support_distance < enemy_king_distance { 2 } else { 1 };
    
    let mobility_value = base_bonus * advancement_multiplier * king_factor;
    let breakthrough_potential = if is_unstoppable { 
        PAWN_BREAKTHROUGH_BONUS + distance_to_promotion * 10 
    } else { 0 };

    PassedPawnInfo {
        square: sq,
        file,
        rank,
        distance_to_promotion,
        is_protected,
        is_advanced,
        is_unstoppable,
        king_support_distance,
        enemy_king_distance,
        mobility_value,
        breakthrough_potential,
    }
}

/// Avaliação avançada de peões dobrados
fn evaluate_doubled_pawns_advanced(our_pawns: Bitboard, phase_info: &GamePhaseInfo) -> i32 {
    let mut penalty = 0;
    
    for file in 0..8 {
        let file_mask = FILE_MASKS[file];
        let pawns_on_file = (our_pawns & file_mask).count_ones();
        
        if pawns_on_file > 1 {
            let doubled_penalty = interpolate_phase_i32(
                DOUBLED_PAWN_PENALTY[0],
                DOUBLED_PAWN_PENALTY[2],
                phase_info
            );
            penalty += doubled_penalty * (pawns_on_file - 1) as i32;
        }
    }
    
    penalty
}

/// Avaliação avançada de peões isolados
fn evaluate_isolated_pawns_advanced(our_pawns: Bitboard, phase_info: &GamePhaseInfo) -> i32 {
    let mut penalty = 0;
    let mut pawns = our_pawns;
    
    while pawns != 0 {
        let sq = pawns.trailing_zeros() as u8;
        pawns &= pawns - 1;
        
        let file = sq % 8;
        
        // Verifica arquivos adjacentes
        let left_file = if file > 0 { FILE_MASKS[file as usize - 1] } else { 0 };
        let right_file = if file < 7 { FILE_MASKS[file as usize + 1] } else { 0 };
        
        let adjacent_pawns = our_pawns & (left_file | right_file);
        
        if adjacent_pawns == 0 {
            let isolated_penalty = interpolate_phase_i32(
                ISOLATED_PAWN_PENALTY[0],
                ISOLATED_PAWN_PENALTY[2],
                phase_info
            );
            penalty += isolated_penalty;
        }
    }
    
    penalty
}

/// Avaliação avançada de peões atrasados
fn evaluate_backward_pawns_advanced(our_pawns: Bitboard, enemy_pawns: Bitboard, color: Color, board: &Board, phase_info: &GamePhaseInfo) -> i32 {
    let mut penalty = 0;
    let mut pawns = our_pawns;
    
    while pawns != 0 {
        let sq = pawns.trailing_zeros() as u8;
        pawns &= pawns - 1;
        
        if is_backward_pawn_advanced(sq, our_pawns, enemy_pawns, color, board) {
            let backward_penalty = interpolate_phase_i32(
                BACKWARD_PAWN_PENALTY[0],
                BACKWARD_PAWN_PENALTY[2],
                phase_info
            );
            penalty += backward_penalty;
        }
    }
    
    penalty
}

/// Avaliação avançada de peões conectados
fn evaluate_connected_pawns_advanced(our_pawns: Bitboard, phase_info: &GamePhaseInfo) -> i32 {
    let mut bonus = 0;
    let mut pawns = our_pawns;
    
    while pawns != 0 {
        let sq = pawns.trailing_zeros() as u8;
        pawns &= pawns - 1;
        
        let file = sq % 8;
        let rank = sq / 8;
        
        // Verifica peões conectados diagonalmente (suporte)
        let diagonal_supports = [
            if file > 0 && rank > 0 { Some((rank - 1) * 8 + (file - 1)) } else { None },
            if file < 7 && rank > 0 { Some((rank - 1) * 8 + (file + 1)) } else { None },
        ];
        
        for support_sq in diagonal_supports.iter().flatten() {
            if (our_pawns & (1u64 << support_sq)) != 0 {
                let connected_bonus = interpolate_phase_i32(
                    CONNECTED_PAWN_BONUS[0],
                    CONNECTED_PAWN_BONUS[2],
                    phase_info
                );
                bonus += connected_bonus;
            }
        }
    }
    
    bonus
}

/// Avaliação avançada de cadeias de peões
fn evaluate_pawn_chains_advanced(our_pawns: Bitboard, color: Color, phase_info: &GamePhaseInfo) -> i32 {
    let mut bonus = 0;
    let mut chain_length = 0;
    let mut pawns = our_pawns;
    
    while pawns != 0 {
        let sq = pawns.trailing_zeros() as u8;
        pawns &= pawns - 1;
        
        if is_in_pawn_chain(sq, our_pawns, color) {
            chain_length += 1;
        }
    }
    
    if chain_length >= 2 {
        let chain_bonus = interpolate_phase_i32(
            PAWN_CHAIN_BONUS[0],
            PAWN_CHAIN_BONUS[2],
            phase_info
        );
        bonus += chain_bonus * chain_length;
    }
    
    bonus
}

/// Avaliação avançada de tempestades de peões
fn evaluate_pawn_storms_advanced(our_pawns: Bitboard, enemy_pawns: Bitboard, color: Color, board: &Board, phase_info: &GamePhaseInfo) -> i32 {
    let enemy_king = board.kings & if color == Color::White { board.black_pieces } else { board.white_pieces };
    if enemy_king == 0 { return 0; }

    let enemy_king_sq = enemy_king.trailing_zeros() as u8;
    let enemy_king_file = enemy_king_sq % 8;
    
    let mut storm_value = 0;
    let mut pawns = our_pawns;
    
    while pawns != 0 {
        let sq = pawns.trailing_zeros() as u8;
        pawns &= pawns - 1;
        
        let file = sq % 8;
        let rank = sq / 8;
        
        // Peões próximos ao rei inimigo são mais valiosos
        let file_distance = (file as i8 - enemy_king_file as i8).abs();
        if file_distance <= 2 {
            let advancement = if color == Color::White { rank } else { 7 - rank };
            let proximity_bonus = (3 - file_distance) * (advancement as i8 + 1);
            
            storm_value += proximity_bonus as i32;
        }
    }
    
    let storm_bonus = interpolate_phase_i32(
        PAWN_STORM_BONUS[0],
        PAWN_STORM_BONUS[2],
        phase_info
    );
    
    (storm_value * storm_bonus) / 8
}

/// Avaliação de alavancas de peões
fn evaluate_pawn_levers(our_pawns: Bitboard, enemy_pawns: Bitboard, color: Color) -> i32 {
    let mut bonus = 0;
    let mut pawns = our_pawns;
    
    while pawns != 0 {
        let sq = pawns.trailing_zeros() as u8;
        pawns &= pawns - 1;
        
        if can_create_pawn_lever(sq, enemy_pawns, color) {
            bonus += PAWN_LEVER_BONUS;
        }
    }
    
    bonus
}

/// Avaliação de quebras de peões
fn evaluate_pawn_breaks(our_pawns: Bitboard, enemy_pawns: Bitboard, color: Color, board: &Board) -> i32 {
    // Analisa potencial para quebrar estrutura inimiga
    let mut breaks = 0;
    let all_pieces = board.white_pieces | board.black_pieces;
    
    let mut pawns = our_pawns;
    while pawns != 0 {
        let sq = pawns.trailing_zeros() as u8;
        pawns &= pawns - 1;
        
        if can_create_pawn_break(sq, enemy_pawns, color, all_pieces) {
            breaks += 1;
        }
    }
    
    breaks * 15
}

/// Avaliação de ataques de minoria
fn evaluate_minority_attacks(our_pawns: Bitboard, enemy_pawns: Bitboard, color: Color) -> i32 {
    let our_queenside_pawns = count_pawns_in_files(our_pawns, 0, 3);
    let enemy_queenside_pawns = count_pawns_in_files(enemy_pawns, 0, 3);
    
    if our_queenside_pawns < enemy_queenside_pawns && our_queenside_pawns >= 2 {
        PAWN_MINORITY_ATTACK_BONUS
    } else {
        0
    }
}

/// Avaliação de potencial de maioria
fn evaluate_majority_potential(our_pawns: Bitboard, enemy_pawns: Bitboard, color: Color) -> i32 {
    let our_kingside_pawns = count_pawns_in_files(our_pawns, 4, 7);
    let enemy_kingside_pawns = count_pawns_in_files(enemy_pawns, 4, 7);
    
    if our_kingside_pawns > enemy_kingside_pawns {
        PAWN_MAJORITY_BONUS
    } else {
        0
    }
}

/// Avaliação de contribuição para segurança do rei
fn evaluate_king_safety_contribution(our_pawns: Bitboard, board: &Board, color: Color) -> i32 {
    let our_king = board.kings & if color == Color::White { board.white_pieces } else { board.black_pieces };
    if our_king == 0 { return 0; }

    let king_sq = our_king.trailing_zeros() as u8;
    let king_file = king_sq % 8;
    
    let mut safety_bonus = 0;
    
    // Escudo de peões nas fileiras próximas ao rei
    for shield_file in (king_file.saturating_sub(1))..=(king_file.saturating_add(1)).min(7) {
        let file_mask = FILE_MASKS[shield_file as usize];
        if (our_pawns & file_mask) != 0 {
            safety_bonus += 8;
        }
    }
    
    safety_bonus
}

/// Avaliação de suporte para coordenação de peças
fn evaluate_piece_coordination_support(our_pawns: Bitboard, board: &Board, color: Color) -> i32 {
    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let our_knights = board.knights & our_pieces;
    let our_bishops = board.bishops & our_pieces;
    
    let mut support_bonus = 0;
    
    // Suporte para outposts de cavalos
    let mut knights = our_knights;
    while knights != 0 {
        let knight_sq = knights.trailing_zeros() as u8;
        knights &= knights - 1;
        
        if is_supported_by_pawn(knight_sq, our_pawns, color) {
            support_bonus += PAWN_OUTPOST_SUPPORT_BONUS;
        }
    }
    
    support_bonus
}

/// Avaliação de potencial de endgame
fn evaluate_endgame_potential(our_pawns: Bitboard, enemy_pawns: Bitboard, color: Color, board: &Board) -> i32 {
    let total_pieces = (board.white_pieces | board.black_pieces).count_ones();
    if total_pieces > 16 { return 0; } // Não é endgame ainda
    
    let mut endgame_score = 0;
    
    // Peões avançados são mais valiosos no endgame
    let mut pawns = our_pawns;
    while pawns != 0 {
        let sq = pawns.trailing_zeros() as u8;
        pawns &= pawns - 1;
        
        let rank = sq / 8;
        let advancement = if color == Color::White { rank } else { 7 - rank };
        
        if advancement >= 4 {
            endgame_score += advancement as i32 * 5;
        }
    }
    
    endgame_score
}

/// Avaliação de controle de espaço
fn evaluate_space_control(our_pawns: Bitboard, color: Color) -> i32 {
    let mut space = 0;
    let mut pawns = our_pawns;
    
    while pawns != 0 {
        let sq = pawns.trailing_zeros() as u8;
        pawns &= pawns - 1;
        
        let rank = sq / 8;
        
        // Peões avançados controlam mais espaço
        let advancement = if color == Color::White { rank } else { 7 - rank };
        if advancement >= 3 {
            space += advancement as i32 * 2;
        }
    }
    
    space
}

/// Avaliação de mobilidade integrada de peões
fn evaluate_pawn_mobility_integrated(our_pawns: Bitboard, enemy_pawns: Bitboard, color: Color, board: &Board) -> i32 {
    let mut mobility = 0;
    let all_pieces = board.white_pieces | board.black_pieces;
    let mut pawns = our_pawns;
    
    while pawns != 0 {
        let sq = pawns.trailing_zeros() as u8;
        pawns &= pawns - 1;
        
        mobility += calculate_pawn_mobility(sq, color, all_pieces);
    }
    
    mobility * 2
}

/// Avaliação de mobilidade tática
fn evaluate_tactical_mobility(our_pawns: Bitboard, enemy_pawns: Bitboard, color: Color, board: &Board) -> i32 {
    let mut tactical_value = 0;
    let enemy_pieces = if color == Color::White { board.black_pieces } else { board.white_pieces };
    let mut pawns = our_pawns;
    
    while pawns != 0 {
        let sq = pawns.trailing_zeros() as u8;
        pawns &= pawns - 1;
        
        // Capturas possíveis
        let pawn_attacks = get_pawn_attacks(sq, color);
        let possible_captures = pawn_attacks & enemy_pieces;
        tactical_value += possible_captures.count_ones() as i32 * 5;
    }
    
    tactical_value
}

/// Avaliação de avanços estratégicos
fn evaluate_strategic_advances(our_pawns: Bitboard, enemy_pawns: Bitboard, color: Color, board: &Board) -> i32 {
    let mut strategic_value = 0;
    let all_pieces = board.white_pieces | board.black_pieces;
    let mut pawns = our_pawns;
    
    while pawns != 0 {
        let sq = pawns.trailing_zeros() as u8;
        pawns &= pawns - 1;
        
        if can_advance_strategically(sq, color, all_pieces, enemy_pawns) {
            strategic_value += 8;
        }
    }
    
    strategic_value
}

// ============================================================================
// FUNÇÕES AUXILIARES
// ============================================================================

/// Calcula distância entre duas casas
fn calculate_square_distance(sq1: u8, sq2: u8) -> i32 {
    let file1 = sq1 % 8;
    let rank1 = sq1 / 8;
    let file2 = sq2 % 8;
    let rank2 = sq2 / 8;
    
    let file_diff = (file1 as i8 - file2 as i8).abs();
    let rank_diff = (rank1 as i8 - rank2 as i8).abs();
    
    file_diff.max(rank_diff) as i32
}

/// Verifica se peão está protegido
fn is_pawn_protected(sq: u8, color: Color, board: &Board) -> bool {
    let our_pawns = board.pawns & if color == Color::White { board.white_pieces } else { board.black_pieces };
    let file = sq % 8;
    let rank = sq / 8;
    
    let protection_squares = match color {
        Color::White => {
            let mut squares = Vec::new();
            if rank > 0 {
                if file > 0 { squares.push((rank - 1) * 8 + file - 1); }
                if file < 7 { squares.push((rank - 1) * 8 + file + 1); }
            }
            squares
        },
        Color::Black => {
            let mut squares = Vec::new();
            if rank < 7 {
                if file > 0 { squares.push((rank + 1) * 8 + file - 1); }
                if file < 7 { squares.push((rank + 1) * 8 + file + 1); }
            }
            squares
        },
    };
    
    protection_squares.iter().any(|&sq| (our_pawns & (1u64 << sq)) != 0)
}

/// Verifica se peão é imparável
fn is_pawn_unstoppable(sq: u8, color: Color, enemy_king_sq: u8, board: &Board) -> bool {
    if enemy_king_sq >= 64 { return true; }
    
    let rank = sq / 8;
    let promotion_rank = if color == Color::White { 7 } else { 0 };
    let distance_to_promotion = (rank as i8 - promotion_rank as i8).abs();
    let king_distance = calculate_square_distance(sq, enemy_king_sq);
    
    distance_to_promotion < (king_distance - 1) as i8
}

/// Verifica se peão está atrasado (versão avançada)
fn is_backward_pawn_advanced(sq: u8, our_pawns: Bitboard, enemy_pawns: Bitboard, color: Color, board: &Board) -> bool {
    let file = sq % 8;
    let rank = sq / 8;
    
    // Verifica se peões aliados nos arquivos adjacentes estão mais avançados
    let adjacent_files = [
        if file > 0 { Some(file - 1) } else { None },
        if file < 7 { Some(file + 1) } else { None },
    ];
    
    let mut has_advanced_neighbors = false;
    for adj_file_opt in &adjacent_files {
        if let Some(adj_file) = adj_file_opt {
            let file_mask = FILE_MASKS[*adj_file as usize];
            let file_pawns = our_pawns & file_mask;
            
            if file_pawns != 0 {
                let mut neighbor_pawns = file_pawns;
                while neighbor_pawns != 0 {
                    let neighbor_sq = neighbor_pawns.trailing_zeros() as u8;
                    neighbor_pawns &= neighbor_pawns - 1;
                    let neighbor_rank = neighbor_sq / 8;
                    
                    let is_more_advanced = match color {
                        Color::White => neighbor_rank > rank,
                        Color::Black => neighbor_rank < rank,
                    };
                    
                    if is_more_advanced {
                        has_advanced_neighbors = true;
                        break;
                    }
                }
            }
        }
    }
    
    if !has_advanced_neighbors { return false; }
    
    // Verifica se casa à frente está controlada por inimigo
    let advance_sq = match color {
        Color::White => if rank >= 7 { return false; } else { sq + 8 },
        Color::Black => if rank <= 0 { return false; } else { sq - 8 },
    };
    
    let enemy_pawn_attacks = compute_all_pawn_attacks(enemy_pawns, !color);
    (enemy_pawn_attacks & (1u64 << advance_sq)) != 0
}

/// Verifica se peão está em cadeia
fn is_in_pawn_chain(sq: u8, our_pawns: Bitboard, color: Color) -> bool {
    let file = sq % 8;
    let rank = sq / 8;
    
    // Verifica suporte diagonal
    let support_squares = match color {
        Color::White => {
            let mut squares = Vec::new();
            if rank > 0 {
                if file > 0 { squares.push((rank - 1) * 8 + file - 1); }
                if file < 7 { squares.push((rank - 1) * 8 + file + 1); }
            }
            squares
        },
        Color::Black => {
            let mut squares = Vec::new();
            if rank < 7 {
                if file > 0 { squares.push((rank + 1) * 8 + file - 1); }
                if file < 7 { squares.push((rank + 1) * 8 + file + 1); }
            }
            squares
        },
    };
    
    support_squares.iter().any(|&sq| (our_pawns & (1u64 << sq)) != 0)
}

/// Verifica se pode criar alavanca
fn can_create_pawn_lever(sq: u8, enemy_pawns: Bitboard, color: Color) -> bool {
    let file = sq % 8;
    let rank = sq / 8;
    
    let target_squares = match color {
        Color::White => {
            let mut squares = Vec::new();
            if rank < 7 {
                if file > 0 { squares.push((rank + 1) * 8 + file - 1); }
                if file < 7 { squares.push((rank + 1) * 8 + file + 1); }
            }
            squares
        },
        Color::Black => {
            let mut squares = Vec::new();
            if rank > 0 {
                if file > 0 { squares.push((rank - 1) * 8 + file - 1); }
                if file < 7 { squares.push((rank - 1) * 8 + file + 1); }
            }
            squares
        },
    };
    
    target_squares.iter().any(|&target_sq| (enemy_pawns & (1u64 << target_sq)) != 0)
}

/// Verifica se pode criar quebra
fn can_create_pawn_break(sq: u8, enemy_pawns: Bitboard, color: Color, all_pieces: Bitboard) -> bool {
    let advance_sq = match color {
        Color::White => if sq >= 56 { return false; } else { sq + 8 },
        Color::Black => if sq <= 7 { return false; } else { sq - 8 },
    };
    
    // Casa à frente deve estar livre
    if (all_pieces & (1u64 << advance_sq)) != 0 { return false; }
    
    // Deve haver peões inimigos nas proximidades para "quebrar"
    let file = sq % 8;
    let target_rank = advance_sq / 8;
    
    for check_file in (file.saturating_sub(1))..=(file.saturating_add(1)).min(7) {
        let target_sq = target_rank * 8 + check_file;
        if (enemy_pawns & (1u64 << target_sq)) != 0 {
            return true;
        }
    }
    
    false
}

/// Conta peões em range de arquivos
fn count_pawns_in_files(pawns: Bitboard, start_file: u8, end_file: u8) -> u32 {
    let mut count = 0;
    for file in start_file..=end_file {
        count += (pawns & FILE_MASKS[file as usize]).count_ones();
    }
    count
}

/// Verifica se peça está suportada por peão
fn is_supported_by_pawn(piece_sq: u8, our_pawns: Bitboard, color: Color) -> bool {
    let file = piece_sq % 8;
    let rank = piece_sq / 8;
    
    let support_squares = match color {
        Color::White => {
            let mut squares = Vec::new();
            if rank > 0 {
                if file > 0 { squares.push((rank - 1) * 8 + file - 1); }
                if file < 7 { squares.push((rank - 1) * 8 + file + 1); }
            }
            squares
        },
        Color::Black => {
            let mut squares = Vec::new();
            if rank < 7 {
                if file > 0 { squares.push((rank + 1) * 8 + file - 1); }
                if file < 7 { squares.push((rank + 1) * 8 + file + 1); }
            }
            squares
        },
    };
    
    support_squares.iter().any(|&sq| (our_pawns & (1u64 << sq)) != 0)
}

/// Calcula mobilidade de peão específico
fn calculate_pawn_mobility(sq: u8, color: Color, all_pieces: Bitboard) -> i32 {
    let rank = sq / 8;
    let mut mobility = 0;
    
    // Avanço de uma casa
    let one_step = match color {
        Color::White => if rank >= 7 { return 0; } else { sq + 8 },
        Color::Black => if rank <= 0 { return 0; } else { sq - 8 },
    };
    
    if (all_pieces & (1u64 << one_step)) == 0 {
        mobility += 1;
        
        // Avanço duplo para peões iniciais
        let can_double = match color {
            Color::White => rank == 1,
            Color::Black => rank == 6,
        };
        
        if can_double {
            let two_steps = match color {
                Color::White => sq + 16,
                Color::Black => sq - 16,
            };
            
            if (all_pieces & (1u64 << two_steps)) == 0 {
                mobility += 1;
            }
        }
    }
    
    mobility
}

/// Obtém ataques de peão
fn get_pawn_attacks(sq: u8, color: Color) -> Bitboard {
    let file = sq % 8;
    let rank = sq / 8;
    
    match color {
        Color::White => {
            if rank >= 7 { return 0; }
            let mut attacks = 0u64;
            if file > 0 { attacks |= 1u64 << (sq + 7); }
            if file < 7 { attacks |= 1u64 << (sq + 9); }
            attacks
        },
        Color::Black => {
            if rank <= 0 { return 0; }
            let mut attacks = 0u64;
            if file < 7 { attacks |= 1u64 << (sq - 7); }
            if file > 0 { attacks |= 1u64 << (sq - 9); }
            attacks
        },
    }
}

/// Computa todos os ataques de peões
fn compute_all_pawn_attacks(pawns: Bitboard, color: Color) -> Bitboard {
    const NOT_A_FILE: Bitboard = 0xfefefefefefefefe;
    const NOT_H_FILE: Bitboard = 0x7f7f7f7f7f7f7f7f;

    match color {
        Color::White => {
            let left_attacks = (pawns & NOT_A_FILE) << 7;
            let right_attacks = (pawns & NOT_H_FILE) << 9;
            left_attacks | right_attacks
        },
        Color::Black => {
            let left_attacks = (pawns & NOT_H_FILE) >> 7;
            let right_attacks = (pawns & NOT_A_FILE) >> 9;
            left_attacks | right_attacks
        },
    }
}

/// Verifica se pode avançar estrategicamente
fn can_advance_strategically(sq: u8, color: Color, all_pieces: Bitboard, enemy_pawns: Bitboard) -> bool {
    let advance_sq = match color {
        Color::White => if sq >= 56 { return false; } else { sq + 8 },
        Color::Black => if sq <= 7 { return false; } else { sq - 8 },
    };
    
    // Casa à frente deve estar livre
    if (all_pieces & (1u64 << advance_sq)) == 0 {
        // Verifica se avanço melhora posição (não atacado por peões inimigos)
        let enemy_attacks = compute_all_pawn_attacks(enemy_pawns, !color);
        return (enemy_attacks & (1u64 << advance_sq)) == 0;
    }
    
    false
}

/// Função auxiliar para converter bitboard em vetor de casas
pub fn get_set_bits_simple(mut bitboard: Bitboard) -> Vec<u8> {
    let mut squares = Vec::new();
    while bitboard != 0 {
        let sq = bitboard.trailing_zeros() as u8;
        bitboard &= bitboard - 1;
        squares.push(sq);
    }
    squares
}