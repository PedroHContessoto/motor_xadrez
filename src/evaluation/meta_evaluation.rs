// Sistema Meta-Avaliação - Detecção Avançada de Vulnerabilidades Posicionais
use crate::{board::Board, types::{Color, Bitboard, Move}};
use super::{
    king_safety::*, threats::*, pawn_structure::*, 
    game_phase::{GamePhaseInfo, detect_game_phase_advanced},
    mobility::MobilityContext
};

// ============================================================================
// META-EVALUATION LAYER - ANÁLISE ESTRATÉGICA PROFUNDA DE VULNERABILIDADES
// ============================================================================

/// Configuração para análise meta-avaliativa
#[derive(Debug, Clone)]
pub struct MetaEvaluationConfig {
    pub vulnerability_weight: f32,      // Peso das vulnerabilidades
    pub strategic_depth: u8,            // Profundidade da análise estratégica
    pub tactical_horizon: u8,           // Horizonte tático a considerar
    pub positional_accuracy: f32,       // Precisão da análise posicional
    pub threat_sensitivity: f32,        // Sensibilidade a ameaças
    pub defensive_focus: f32,           // Foco em aspectos defensivos
}

impl Default for MetaEvaluationConfig {
    fn default() -> Self {
        MetaEvaluationConfig {
            vulnerability_weight: 1.2,
            strategic_depth: 4,
            tactical_horizon: 3,
            positional_accuracy: 0.85,
            threat_sensitivity: 1.1,
            defensive_focus: 1.0,
        }
    }
}

/// Tipos de vulnerabilidades detectadas
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VulnerabilityType {
    // Vulnerabilidades Táticas
    PinnedPieces,               // Peças pregadas
    OverloadedPieces,           // Peças sobrecarregadas
    LooseHangingPieces,         // Peças soltas/penduradas
    TacticalMotifs,             // Motivos táticos (garfo, espeto, etc.)
    
    // Vulnerabilidades Posicionais
    WeakSquares,                // Casas fracas
    PawnWeaknesses,             // Fraquezas na estrutura de peões
    PieceCoordination,          // Má coordenação de peças
    KingSafety,                 // Segurança do rei comprometida
    
    // Vulnerabilidades Estratégicas
    PoorDevelopment,            // Desenvolvimento pobre
    BadPiecePositioning,        // Posicionamento ruim de peças
    StrategicMisalignment,      // Desalinhamento estratégico
    EndgameWeaknesses,          // Fraquezas de final
}

/// Severidade da vulnerabilidade
#[derive(Debug, Clone, Copy, PartialEq, Ord, PartialOrd, Eq)]
pub enum VulnerabilitySeverity {
    Minor,      // Vulnerabilidade menor
    Moderate,   // Vulnerabilidade moderada
    Serious,    // Vulnerabilidade séria
    Critical,   // Vulnerabilidade crítica
}

/// Informação detalhada sobre uma vulnerabilidade
#[derive(Debug, Clone)]
pub struct VulnerabilityInfo {
    pub vulnerability_type: VulnerabilityType,
    pub severity: VulnerabilitySeverity,
    pub affected_squares: Vec<u8>,         // Casas afetadas
    pub involved_pieces: Vec<u8>,          // Peças envolvidas
    pub exploitation_moves: Vec<Move>,     // Movimentos que exploram
    pub defense_suggestions: Vec<Move>,    // Sugestões defensivas
    pub impact_score: i32,                 // Impacto numérico
    pub description: String,               // Descrição da vulnerabilidade
}

impl VulnerabilityInfo {
    pub fn new(vuln_type: VulnerabilityType, severity: VulnerabilitySeverity) -> Self {
        VulnerabilityInfo {
            vulnerability_type: vuln_type,
            severity,
            affected_squares: Vec::new(),
            involved_pieces: Vec::new(),
            exploitation_moves: Vec::new(),
            defense_suggestions: Vec::new(),
            impact_score: 0,
            description: String::new(),
        }
    }
}

/// Resultado completo da meta-avaliação
#[derive(Debug, Clone)]
pub struct MetaEvaluationResult {
    pub vulnerabilities: Vec<VulnerabilityInfo>,
    pub total_vulnerability_score: i32,
    pub strategic_assessment: StrategicAssessment,
    pub defensive_recommendations: Vec<Move>,
    pub positional_trends: PositionalTrends,
    pub meta_score: i32,
}

/// Avaliação estratégica geral
#[derive(Debug, Clone)]
pub struct StrategicAssessment {
    pub king_safety_rating: f32,           // 0.0-1.0
    pub piece_activity_rating: f32,        // 0.0-1.0  
    pub pawn_structure_rating: f32,        // 0.0-1.0
    pub tactical_alertness: f32,           // 0.0-1.0
    pub endgame_preparation: f32,          // 0.0-1.0
    pub overall_health: f32,               // 0.0-1.0
}

/// Tendências posicionais detectadas
#[derive(Debug, Clone)]
pub struct PositionalTrends {
    pub improving_factors: Vec<String>,    // Fatores que melhoram
    pub deteriorating_factors: Vec<String>, // Fatores que pioram
    pub stable_elements: Vec<String>,      // Elementos estáveis
    pub critical_transitions: Vec<String>, // Transições críticas
}

impl MetaEvaluationResult {
    pub fn new() -> Self {
        MetaEvaluationResult {
            vulnerabilities: Vec::new(),
            total_vulnerability_score: 0,
            strategic_assessment: StrategicAssessment {
                king_safety_rating: 0.5,
                piece_activity_rating: 0.5,
                pawn_structure_rating: 0.5,
                tactical_alertness: 0.5,
                endgame_preparation: 0.5,
                overall_health: 0.5,
            },
            defensive_recommendations: Vec::new(),
            positional_trends: PositionalTrends {
                improving_factors: Vec::new(),
                deteriorating_factors: Vec::new(),
                stable_elements: Vec::new(),
                critical_transitions: Vec::new(),
            },
            meta_score: 0,
        }
    }
}

/// Sistema principal de meta-avaliação
pub struct MetaEvaluator {
    config: MetaEvaluationConfig,
}

impl MetaEvaluator {
    pub fn new(config: MetaEvaluationConfig) -> Self {
        MetaEvaluator { config }
    }

    /// Análise completa de vulnerabilidades posicionais
    pub fn analyze_position(&self, board: &Board, color: Color) -> MetaEvaluationResult {
        let mut result = MetaEvaluationResult::new();
        let phase_info = detect_game_phase_advanced(board);
        let mobility_context = MobilityContext::new(board, color);

        // === ANÁLISE DE VULNERABILIDADES TÁTICAS ===
        result.vulnerabilities.extend(self.detect_tactical_vulnerabilities(board, color, &mobility_context));

        // === ANÁLISE DE VULNERABILIDADES POSICIONAIS ===
        result.vulnerabilities.extend(self.detect_positional_vulnerabilities(board, color, &phase_info));

        // === ANÁLISE DE VULNERABILIDADES ESTRATÉGICAS ===
        result.vulnerabilities.extend(self.detect_strategic_vulnerabilities(board, color, &phase_info));

        // === AVALIAÇÃO ESTRATÉGICA GERAL ===
        result.strategic_assessment = self.assess_strategic_situation(board, color, &result.vulnerabilities, &phase_info);

        // === ANÁLISE DE TENDÊNCIAS POSICIONAIS ===
        result.positional_trends = self.analyze_positional_trends(board, color, &result.vulnerabilities, &phase_info);

        // === RECOMENDAÇÕES DEFENSIVAS ===
        result.defensive_recommendations = self.generate_defensive_recommendations(board, color, &result.vulnerabilities);

        // === CÁLCULO DE SCORES ===
        result.total_vulnerability_score = self.calculate_vulnerability_score(&result.vulnerabilities);
        result.meta_score = self.calculate_meta_score(&result);

        result
    }

    /// Detecta vulnerabilidades táticas
    fn detect_tactical_vulnerabilities(&self, board: &Board, color: Color, context: &MobilityContext) -> Vec<VulnerabilityInfo> {
        let mut vulnerabilities = Vec::new();

        // Detecção de peças pregadas
        vulnerabilities.extend(self.detect_pinned_pieces(board, color, context));

        // Detecção de peças sobrecarregadas
        vulnerabilities.extend(self.detect_overloaded_pieces(board, color, context));

        // Detecção de peças soltas
        vulnerabilities.extend(self.detect_hanging_pieces(board, color, context));

        // Detecção de motivos táticos
        vulnerabilities.extend(self.detect_tactical_motifs(board, color, context));

        vulnerabilities
    }

    /// Detecta peças pregadas
    fn detect_pinned_pieces(&self, board: &Board, color: Color, context: &MobilityContext) -> Vec<VulnerabilityInfo> {
        let mut vulnerabilities = Vec::new();
        let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
        let enemy_pieces = if color == Color::White { board.black_pieces } else { board.white_pieces };
        
        // Localiza nosso rei
        let our_king = board.kings & our_pieces;
        if our_king == 0 { return vulnerabilities; }
        let king_sq = our_king.trailing_zeros() as u8;

        // Verifica peças inimigas deslizantes que podem pregar
        let enemy_sliding = (board.bishops | board.rooks | board.queens) & enemy_pieces;
        let enemy_sliding_squares = self.get_set_bits(enemy_sliding);

        for &sliding_sq in &enemy_sliding_squares {
            let pinned_pieces = self.find_pieces_pinned_by(sliding_sq, king_sq, board, color);
            
            for pinned_sq in pinned_pieces {
                let mut vuln = VulnerabilityInfo::new(VulnerabilityType::PinnedPieces, VulnerabilitySeverity::Moderate);
                vuln.affected_squares = vec![pinned_sq];
                vuln.involved_pieces = vec![pinned_sq, sliding_sq];
                vuln.impact_score = -25; // Peça pregada tem mobilidade reduzida
                vuln.description = format!("Peça em {} pregada por peça em {}", self.square_name(pinned_sq), self.square_name(sliding_sq));
                vulnerabilities.push(vuln);
            }
        }

        vulnerabilities
    }

    /// Encontra peças pregadas por uma peça deslizante específica
    fn find_pieces_pinned_by(&self, attacker_sq: u8, king_sq: u8, board: &Board, color: Color) -> Vec<u8> {
        let mut pinned = Vec::new();
        let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
        let attacker_bit = 1u64 << attacker_sq;

        // Determina tipo da peça atacante
        let is_bishop_like = (board.bishops | board.queens) & attacker_bit != 0;
        let is_rook_like = (board.rooks | board.queens) & attacker_bit != 0;

        if is_bishop_like {
            pinned.extend(self.find_pinned_on_diagonals(attacker_sq, king_sq, board, our_pieces));
        }
        if is_rook_like {
            pinned.extend(self.find_pinned_on_ranks_files(attacker_sq, king_sq, board, our_pieces));
        }

        pinned
    }

    /// Encontra pregos nas diagonais
    fn find_pinned_on_diagonals(&self, attacker_sq: u8, king_sq: u8, board: &Board, our_pieces: Bitboard) -> Vec<u8> {
        let mut pinned = Vec::new();
        let directions = [-9, -7, 7, 9]; // Direções diagonais

        for &direction in &directions {
            if let Some(pinned_sq) = self.find_pinned_in_direction(attacker_sq, king_sq, direction, board, our_pieces) {
                pinned.push(pinned_sq);
            }
        }

        pinned
    }

    /// Encontra pregos nas fileiras e colunas
    fn find_pinned_on_ranks_files(&self, attacker_sq: u8, king_sq: u8, board: &Board, our_pieces: Bitboard) -> Vec<u8> {
        let mut pinned = Vec::new();
        let directions = [-8, -1, 1, 8]; // Direções de fileira/coluna

        for &direction in &directions {
            if let Some(pinned_sq) = self.find_pinned_in_direction(attacker_sq, king_sq, direction, board, our_pieces) {
                pinned.push(pinned_sq);
            }
        }

        pinned
    }

    /// Encontra prego em uma direção específica
    fn find_pinned_in_direction(&self, attacker_sq: u8, king_sq: u8, direction: i8, board: &Board, our_pieces: Bitboard) -> Option<u8> {
        let mut current_sq = attacker_sq as i8;
        let mut pieces_found = 0;
        let mut potential_pinned = None;

        loop {
            current_sq += direction;
            
            if current_sq < 0 || current_sq >= 64 {
                break;
            }

            if !self.is_valid_direction_move(attacker_sq, current_sq as u8, direction) {
                break;
            }

            let current_bit = 1u64 << current_sq;

            if current_sq as u8 == king_sq {
                // Encontrou o rei - se há exatamente uma peça no meio, está pregada
                return if pieces_found == 1 { potential_pinned } else { None };
            }

            if (board.white_pieces | board.black_pieces) & current_bit != 0 {
                pieces_found += 1;
                if pieces_found == 1 && (our_pieces & current_bit) != 0 {
                    potential_pinned = Some(current_sq as u8);
                } else if pieces_found > 1 {
                    break; // Mais de uma peça - não é prego válido
                }
            }
        }

        None
    }

    /// Verifica movimento válido de direção (sem wrap-around)
    fn is_valid_direction_move(&self, from: u8, to: u8, direction: i8) -> bool {
        let from_file = from % 8;
        let from_rank = from / 8;
        let to_file = to % 8;
        let to_rank = to / 8;

        match direction {
            -1 | 7 | -9 => from_file > 0 && to_file == from_file - 1,  // Movendo para esquerda
            1 | 9 | -7 => from_file < 7 && to_file == from_file + 1,   // Movendo para direita
            -8 => to_file == from_file && to_rank == from_rank - 1,    // Movendo para baixo
            8 => to_file == from_file && to_rank == from_rank + 1,     // Movendo para cima
            _ => true,
        }
    }

    /// Detecta peças sobrecarregadas
    fn detect_overloaded_pieces(&self, board: &Board, color: Color, context: &MobilityContext) -> Vec<VulnerabilityInfo> {
        let mut vulnerabilities = Vec::new();
        let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };

        // Analisa cada uma de nossas peças para sobrecarga
        let piece_squares = self.get_set_bits(our_pieces);

        for &piece_sq in &piece_squares {
            let piece_bit = 1u64 << piece_sq;
            
            // Conta responsabilidades defensivas da peça
            let defensive_duties = self.count_defensive_duties(piece_sq, board, color);
            
            // Se a peça tem muitas responsabilidades, pode estar sobrecarregada
            if defensive_duties >= 3 {
                let severity = match defensive_duties {
                    3 => VulnerabilitySeverity::Minor,
                    4 => VulnerabilitySeverity::Moderate,
                    _ => VulnerabilitySeverity::Serious,
                };

                let mut vuln = VulnerabilityInfo::new(VulnerabilityType::OverloadedPieces, severity);
                vuln.affected_squares = vec![piece_sq];
                vuln.involved_pieces = vec![piece_sq];
                vuln.impact_score = -(defensive_duties as i32 * 8);
                vuln.description = format!("Peça em {} sobrecarregada com {} responsabilidades", 
                    self.square_name(piece_sq), defensive_duties);
                vulnerabilities.push(vuln);
            }
        }

        vulnerabilities
    }

    /// Conta responsabilidades defensivas de uma peça
    fn count_defensive_duties(&self, piece_sq: u8, board: &Board, color: Color) -> usize {
        let mut duties = 0;
        let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
        let enemy_pieces = if color == Color::White { board.black_pieces } else { board.white_pieces };

        // 1. Defende o rei?
        let our_king = board.kings & our_pieces;
        if our_king != 0 {
            let king_sq = our_king.trailing_zeros() as u8;
            if self.piece_defends_square(piece_sq, king_sq, board) {
                duties += 1;
            }
        }

        // 2. Defende peças valiosas?
        let valuable_pieces = (board.queens | board.rooks) & our_pieces;
        let valuable_squares = self.get_set_bits(valuable_pieces);
        for &valuable_sq in &valuable_squares {
            if valuable_sq != piece_sq && self.piece_defends_square(piece_sq, valuable_sq, board) {
                duties += 1;
            }
        }

        // 3. Bloqueia ameaças?
        if self.piece_blocks_threats(piece_sq, board, color) {
            duties += 1;
        }

        // 4. Controla casas críticas?
        if self.piece_controls_critical_squares(piece_sq, board, color) {
            duties += 1;
        }

        duties
    }

    /// Verifica se peça defende uma casa específica
    fn piece_defends_square(&self, piece_sq: u8, defended_sq: u8, board: &Board) -> bool {
        let piece_bit = 1u64 << piece_sq;
        let defended_bit = 1u64 << defended_sq;

        // Determina tipo da peça e calcula seus ataques
        let attacks = if (board.pawns & piece_bit) != 0 {
            // Ataques de peão (precisaríamos da cor, mas simplificado)
            0u64 // Simplificado por ora
        } else if (board.knights & piece_bit) != 0 {
            crate::moves::knight::get_knight_attacks_lookup(piece_sq)
        } else if (board.bishops & piece_bit) != 0 {
            crate::moves::sliding::get_bishop_attacks(piece_sq, board.white_pieces | board.black_pieces)
        } else if (board.rooks & piece_bit) != 0 {
            crate::moves::sliding::get_rook_attacks(piece_sq, board.white_pieces | board.black_pieces)
        } else if (board.queens & piece_bit) != 0 {
            let bishop_attacks = crate::moves::sliding::get_bishop_attacks(piece_sq, board.white_pieces | board.black_pieces);
            let rook_attacks = crate::moves::sliding::get_rook_attacks(piece_sq, board.white_pieces | board.black_pieces);
            bishop_attacks | rook_attacks
        } else if (board.kings & piece_bit) != 0 {
            crate::moves::king::get_king_attacks_lookup(piece_sq)
        } else {
            0u64
        };

        (attacks & defended_bit) != 0
    }

    /// Verifica se peça bloqueia ameaças
    fn piece_blocks_threats(&self, piece_sq: u8, board: &Board, color: Color) -> bool {
        // Simplificado: verifica se remoção da peça exporia o rei
        let our_king = board.kings & if color == Color::White { board.white_pieces } else { board.black_pieces };
        if our_king == 0 { return false; }
        let king_sq = our_king.trailing_zeros() as u8;

        // Simula remoção da peça e verifica se o rei fica em xeque
        let mut temp_board = *board;
        let piece_bit = 1u64 << piece_sq;
        
        // Remove a peça temporariamente
        temp_board.white_pieces &= !piece_bit;
        temp_board.black_pieces &= !piece_bit;
        temp_board.pawns &= !piece_bit;
        temp_board.knights &= !piece_bit;
        temp_board.bishops &= !piece_bit;
        temp_board.rooks &= !piece_bit;
        temp_board.queens &= !piece_bit;
        temp_board.kings &= !piece_bit;

        // Verifica se o rei ficaria em xeque
        temp_board.is_king_in_check(color)
    }

    /// Verifica se peça controla casas críticas
    fn piece_controls_critical_squares(&self, piece_sq: u8, board: &Board, color: Color) -> bool {
        let piece_bit = 1u64 << piece_sq;
        
        // Calcula ataques da peça
        let attacks = if (board.knights & piece_bit) != 0 {
            crate::moves::knight::get_knight_attacks_lookup(piece_sq)
        } else if (board.bishops & piece_bit) != 0 {
            crate::moves::sliding::get_bishop_attacks(piece_sq, board.white_pieces | board.black_pieces)
        } else if (board.rooks & piece_bit) != 0 {
            crate::moves::sliding::get_rook_attacks(piece_sq, board.white_pieces | board.black_pieces)
        } else if (board.queens & piece_bit) != 0 {
            let bishop_attacks = crate::moves::sliding::get_bishop_attacks(piece_sq, board.white_pieces | board.black_pieces);
            let rook_attacks = crate::moves::sliding::get_rook_attacks(piece_sq, board.white_pieces | board.black_pieces);
            bishop_attacks | rook_attacks
        } else {
            return false;
        };

        // Verifica se controla centro ou casas importantes
        let center = 0x0000001818000000u64; // d4, e4, d5, e5
        let extended_center = 0x00003C3C3C3C0000u64;
        
        (attacks & (center | extended_center)).count_ones() >= 2
    }

    /// Detecta peças soltas/penduradas
    fn detect_hanging_pieces(&self, board: &Board, color: Color, context: &MobilityContext) -> Vec<VulnerabilityInfo> {
        let mut vulnerabilities = Vec::new();
        let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };

        // Analisa cada uma de nossas peças
        let piece_squares = self.get_set_bits(our_pieces);

        for &piece_sq in &piece_squares {
            if self.is_piece_hanging(piece_sq, board, color, context) {
                let piece_value = self.get_piece_value(piece_sq, board);
                let severity = match piece_value {
                    100..=350 => VulnerabilitySeverity::Minor,
                    351..=550 => VulnerabilitySeverity::Moderate,
                    551..=950 => VulnerabilitySeverity::Serious,
                    _ => VulnerabilitySeverity::Critical,
                };

                let mut vuln = VulnerabilityInfo::new(VulnerabilityType::LooseHangingPieces, severity);
                vuln.affected_squares = vec![piece_sq];
                vuln.involved_pieces = vec![piece_sq];
                vuln.impact_score = -(piece_value / 2); // Penalidade por peça solta
                vuln.description = format!("Peça em {} está solta/pendurada", self.square_name(piece_sq));
                vulnerabilities.push(vuln);
            }
        }

        vulnerabilities
    }

    /// Verifica se peça está pendurada
    fn is_piece_hanging(&self, piece_sq: u8, board: &Board, color: Color, context: &MobilityContext) -> bool {
        // Peça está pendurada se:
        // 1. É atacada pelo inimigo
        // 2. Não está defendida adequadamente
        // 3. O valor dos defensores é menor que o valor da peça

        let piece_bit = 1u64 << piece_sq;
        
        // 1. Verifica se está sendo atacada
        if (context.enemy_attacked_squares & piece_bit) == 0 {
            return false; // Não está sendo atacada
        }

        // 2. Calcula valor dos atacantes e defensores
        let piece_value = self.get_piece_value(piece_sq, board);
        let attackers_value = self.calculate_attackers_value(piece_sq, board, !color);
        let defenders_value = self.calculate_defenders_value(piece_sq, board, color);

        // 3. Se atacantes valem mais ou defensores são insuficientes, está pendurada
        attackers_value > 0 && (defenders_value == 0 || attackers_value > defenders_value + piece_value)
    }

    /// Calcula valor das peças atacantes
    fn calculate_attackers_value(&self, target_sq: u8, board: &Board, attacking_color: Color) -> i32 {
        let mut total_value = 0;
        let attackers = self.get_attackers_of_square(target_sq, board, attacking_color);
        
        for attacker_sq in attackers {
            total_value += self.get_piece_value(attacker_sq, board);
        }

        total_value
    }

    /// Calcula valor das peças defensoras
    fn calculate_defenders_value(&self, target_sq: u8, board: &Board, defending_color: Color) -> i32 {
        let mut total_value = 0;
        let defenders = self.get_defenders_of_square(target_sq, board, defending_color);
        
        for defender_sq in defenders {
            total_value += self.get_piece_value(defender_sq, board);
        }

        total_value
    }

    /// Obtém atacantes de uma casa
    fn get_attackers_of_square(&self, target_sq: u8, board: &Board, attacking_color: Color) -> Vec<u8> {
        let mut attackers = Vec::new();
        let attacking_pieces = if attacking_color == Color::White { board.white_pieces } else { board.black_pieces };
        let target_bit = 1u64 << target_sq;

        // Verifica cada tipo de peça
        self.find_piece_attackers(target_sq, board.pawns & attacking_pieces, board, &mut attackers, attacking_color);
        self.find_piece_attackers(target_sq, board.knights & attacking_pieces, board, &mut attackers, attacking_color);
        self.find_piece_attackers(target_sq, board.bishops & attacking_pieces, board, &mut attackers, attacking_color);
        self.find_piece_attackers(target_sq, board.rooks & attacking_pieces, board, &mut attackers, attacking_color);
        self.find_piece_attackers(target_sq, board.queens & attacking_pieces, board, &mut attackers, attacking_color);
        self.find_piece_attackers(target_sq, board.kings & attacking_pieces, board, &mut attackers, attacking_color);

        attackers
    }

    /// Encontra atacantes de um tipo específico de peça
    fn find_piece_attackers(&self, target_sq: u8, pieces: Bitboard, board: &Board, attackers: &mut Vec<u8>, color: Color) {
        let piece_squares = self.get_set_bits(pieces);
        
        for &piece_sq in &piece_squares {
            if self.piece_attacks_square(piece_sq, target_sq, board, color) {
                attackers.push(piece_sq);
            }
        }
    }

    /// Verifica se peça ataca uma casa específica
    fn piece_attacks_square(&self, piece_sq: u8, target_sq: u8, board: &Board, color: Color) -> bool {
        let piece_bit = 1u64 << piece_sq;
        let target_bit = 1u64 << target_sq;

        if (board.pawns & piece_bit) != 0 {
            let pawn_attacks = if color == Color::White {
                let attacks_left = if piece_sq % 8 > 0 { 1u64 << (piece_sq + 7) } else { 0 };
                let attacks_right = if piece_sq % 8 < 7 { 1u64 << (piece_sq + 9) } else { 0 };
                attacks_left | attacks_right
            } else {
                let attacks_left = if piece_sq % 8 > 0 { 1u64 << (piece_sq - 9) } else { 0 };
                let attacks_right = if piece_sq % 8 < 7 { 1u64 << (piece_sq - 7) } else { 0 };
                attacks_left | attacks_right
            };
            (pawn_attacks & target_bit) != 0
        } else if (board.knights & piece_bit) != 0 {
            let attacks = crate::moves::knight::get_knight_attacks_lookup(piece_sq);
            (attacks & target_bit) != 0
        } else if (board.bishops & piece_bit) != 0 {
            let attacks = crate::moves::sliding::get_bishop_attacks(piece_sq, board.white_pieces | board.black_pieces);
            (attacks & target_bit) != 0
        } else if (board.rooks & piece_bit) != 0 {
            let attacks = crate::moves::sliding::get_rook_attacks(piece_sq, board.white_pieces | board.black_pieces);
            (attacks & target_bit) != 0
        } else if (board.queens & piece_bit) != 0 {
            let bishop_attacks = crate::moves::sliding::get_bishop_attacks(piece_sq, board.white_pieces | board.black_pieces);
            let rook_attacks = crate::moves::sliding::get_rook_attacks(piece_sq, board.white_pieces | board.black_pieces);
            ((bishop_attacks | rook_attacks) & target_bit) != 0
        } else if (board.kings & piece_bit) != 0 {
            let attacks = crate::moves::king::get_king_attacks_lookup(piece_sq);
            (attacks & target_bit) != 0
        } else {
            false
        }
    }

    /// Obtém defensores de uma casa
    fn get_defenders_of_square(&self, target_sq: u8, board: &Board, defending_color: Color) -> Vec<u8> {
        // Similar ao get_attackers_of_square mas para defensores
        self.get_attackers_of_square(target_sq, board, defending_color)
    }

    /// Obtém valor de uma peça
    fn get_piece_value(&self, piece_sq: u8, board: &Board) -> i32 {
        let piece_bit = 1u64 << piece_sq;
        
        if (board.pawns & piece_bit) != 0 { 100 }
        else if (board.knights & piece_bit) != 0 { 320 }
        else if (board.bishops & piece_bit) != 0 { 330 }
        else if (board.rooks & piece_bit) != 0 { 500 }
        else if (board.queens & piece_bit) != 0 { 900 }
        else if (board.kings & piece_bit) != 0 { 20000 }
        else { 0 }
    }

    /// Detecta motivos táticos
    fn detect_tactical_motifs(&self, board: &Board, color: Color, context: &MobilityContext) -> Vec<VulnerabilityInfo> {
        let mut vulnerabilities = Vec::new();

        // Detecta garfos possíveis
        vulnerabilities.extend(self.detect_fork_vulnerabilities(board, color, context));

        // Detecta espetos possíveis
        vulnerabilities.extend(self.detect_skewer_vulnerabilities(board, color, context));

        // Detecta ataques descobertos
        vulnerabilities.extend(self.detect_discovered_attack_vulnerabilities(board, color, context));

        vulnerabilities
    }

    /// Detecta vulnerabilidades a garfos
    fn detect_fork_vulnerabilities(&self, board: &Board, color: Color, context: &MobilityContext) -> Vec<VulnerabilityInfo> {
        let mut vulnerabilities = Vec::new();
        let enemy_pieces = if color == Color::White { board.black_pieces } else { board.white_pieces };
        let enemy_knights = board.knights & enemy_pieces;

        // Verifica garfos de cavalo inimigo
        let enemy_knight_squares = self.get_set_bits(enemy_knights);
        
        for &knight_sq in &enemy_knight_squares {
            let knight_attacks = crate::moves::knight::get_knight_attacks_lookup(knight_sq);
            let our_pieces_attacked = knight_attacks & context.our_pieces;
            
            if our_pieces_attacked.count_ones() >= 2 {
                let attacked_squares = self.get_set_bits(our_pieces_attacked);
                let total_value: i32 = attacked_squares.iter()
                    .map(|&sq| self.get_piece_value(sq, board))
                    .sum();

                if total_value >= 600 { // Garfo valioso
                    let mut vuln = VulnerabilityInfo::new(VulnerabilityType::TacticalMotifs, VulnerabilitySeverity::Serious);
                    vuln.affected_squares = attacked_squares;
                    vuln.involved_pieces = vec![knight_sq];
                    vuln.impact_score = -(total_value / 3);
                    vuln.description = format!("Vulnerabilidade a garfo de cavalo em {}", self.square_name(knight_sq));
                    vulnerabilities.push(vuln);
                }
            }
        }

        vulnerabilities
    }

    /// Detecta vulnerabilidades a espetos
    fn detect_skewer_vulnerabilities(&self, board: &Board, color: Color, context: &MobilityContext) -> Vec<VulnerabilityInfo> {
        let mut vulnerabilities = Vec::new();
        let enemy_pieces = if color == Color::White { board.black_pieces } else { board.white_pieces };
        let enemy_sliding = (board.bishops | board.rooks | board.queens) & enemy_pieces;

        let enemy_sliding_squares = self.get_set_bits(enemy_sliding);

        for &sliding_sq in &enemy_sliding_squares {
            let skewer_targets = self.find_skewer_targets(sliding_sq, board, color);
            
            for (front_sq, back_sq, combined_value) in skewer_targets {
                if combined_value >= 800 { // Espeto valioso
                    let mut vuln = VulnerabilityInfo::new(VulnerabilityType::TacticalMotifs, VulnerabilitySeverity::Serious);
                    vuln.affected_squares = vec![front_sq, back_sq];
                    vuln.involved_pieces = vec![sliding_sq];
                    vuln.impact_score = -(combined_value / 4);
                    vuln.description = format!("Vulnerabilidade a espeto: peças em {} e {}", 
                        self.square_name(front_sq), self.square_name(back_sq));
                    vulnerabilities.push(vuln);
                }
            }
        }

        vulnerabilities
    }

    /// Encontra alvos para espeto
    fn find_skewer_targets(&self, sliding_sq: u8, board: &Board, color: Color) -> Vec<(u8, u8, i32)> {
        let mut targets = Vec::new();
        let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
        let sliding_bit = 1u64 << sliding_sq;

        // Determina direções baseado no tipo da peça
        let directions = if (board.bishops & sliding_bit) != 0 {
            vec![-9, -7, 7, 9] // Diagonais
        } else if (board.rooks & sliding_bit) != 0 {
            vec![-8, -1, 1, 8] // Fileiras/colunas
        } else if (board.queens & sliding_bit) != 0 {
            vec![-9, -8, -7, -1, 1, 7, 8, 9] // Todas as direções
        } else {
            vec![]
        };

        for direction in directions {
            if let Some((front_sq, back_sq, value)) = self.find_skewer_in_direction(sliding_sq, direction, board, our_pieces) {
                targets.push((front_sq, back_sq, value));
            }
        }

        targets
    }

    /// Encontra espeto em uma direção específica
    fn find_skewer_in_direction(&self, sliding_sq: u8, direction: i8, board: &Board, our_pieces: Bitboard) -> Option<(u8, u8, i32)> {
        let mut current_sq = sliding_sq as i8;
        let mut first_piece = None;
        let all_pieces = board.white_pieces | board.black_pieces;

        loop {
            current_sq += direction;
            
            if current_sq < 0 || current_sq >= 64 {
                break;
            }

            if !self.is_valid_direction_move(sliding_sq, current_sq as u8, direction) {
                break;
            }

            let current_bit = 1u64 << current_sq;

            if (all_pieces & current_bit) != 0 {
                if (our_pieces & current_bit) != 0 {
                    // Nossa peça
                    if first_piece.is_none() {
                        first_piece = Some(current_sq as u8);
                    } else {
                        // Segunda peça nossa - espeto possível
                        let front_sq = first_piece.unwrap();
                        let back_sq = current_sq as u8;
                        let front_value = self.get_piece_value(front_sq, board);
                        let back_value = self.get_piece_value(back_sq, board);
                        
                        // Espeto é efetivo se a peça da frente vale menos que a de trás
                        if front_value < back_value {
                            return Some((front_sq, back_sq, front_value + back_value));
                        } else {
                            break;
                        }
                    }
                } else {
                    // Peça inimiga bloqueia
                    break;
                }
            }
        }

        None
    }

    /// Detecta vulnerabilidades a ataques descobertos
    fn detect_discovered_attack_vulnerabilities(&self, board: &Board, color: Color, context: &MobilityContext) -> Vec<VulnerabilityInfo> {
        let mut vulnerabilities = Vec::new();
        
        // Verifica se temos peças que podem ser movidas para revelar ataques
        let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
        let enemy_pieces = if color == Color::White { board.black_pieces } else { board.white_pieces };
        
        // Procura por configurações de ataque descoberto
        let our_sliding = (board.bishops | board.rooks | board.queens) & our_pieces;
        let our_sliding_squares = self.get_set_bits(our_sliding);

        for &sliding_sq in &our_sliding_squares {
            let discovered_opportunities = self.find_discovered_attack_opportunities(sliding_sq, board, color);
            
            for (blocking_piece, target_value) in discovered_opportunities {
                if target_value >= 500 { // Oportunidade valiosa
                    let mut vuln = VulnerabilityInfo::new(VulnerabilityType::TacticalMotifs, VulnerabilitySeverity::Moderate);
                    vuln.affected_squares = vec![blocking_piece];
                    vuln.involved_pieces = vec![sliding_sq, blocking_piece];
                    vuln.impact_score = target_value / 5; // Bônus por oportunidade tática
                    vuln.description = format!("Oportunidade de ataque descoberto movendo peça em {}", 
                        self.square_name(blocking_piece));
                    vulnerabilities.push(vuln);
                }
            }
        }

        vulnerabilities
    }

    /// Encontra oportunidades de ataque descoberto
    fn find_discovered_attack_opportunities(&self, sliding_sq: u8, board: &Board, color: Color) -> Vec<(u8, i32)> {
        let mut opportunities = Vec::new();
        let sliding_bit = 1u64 << sliding_sq;
        let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
        let enemy_pieces = if color == Color::White { board.black_pieces } else { board.white_pieces };

        // Determina direções da peça deslizante
        let directions = if (board.bishops & sliding_bit) != 0 {
            vec![-9, -7, 7, 9]
        } else if (board.rooks & sliding_bit) != 0 {
            vec![-8, -1, 1, 8]
        } else if (board.queens & sliding_bit) != 0 {
            vec![-9, -8, -7, -1, 1, 7, 8, 9]
        } else {
            vec![]
        };

        for direction in directions {
            if let Some((blocking_piece, target_value)) = self.find_discovered_in_direction(sliding_sq, direction, board, our_pieces, enemy_pieces) {
                opportunities.push((blocking_piece, target_value));
            }
        }

        opportunities
    }

    /// Encontra ataque descoberto em uma direção
    fn find_discovered_in_direction(&self, sliding_sq: u8, direction: i8, board: &Board, our_pieces: Bitboard, enemy_pieces: Bitboard) -> Option<(u8, i32)> {
        let mut current_sq = sliding_sq as i8;
        let mut blocking_piece = None;
        let all_pieces = board.white_pieces | board.black_pieces;

        loop {
            current_sq += direction;
            
            if current_sq < 0 || current_sq >= 64 {
                break;
            }

            if !self.is_valid_direction_move(sliding_sq, current_sq as u8, direction) {
                break;
            }

            let current_bit = 1u64 << current_sq;

            if (all_pieces & current_bit) != 0 {
                if (our_pieces & current_bit) != 0 && blocking_piece.is_none() {
                    // Nossa peça que pode bloquear
                    blocking_piece = Some(current_sq as u8);
                } else if (enemy_pieces & current_bit) != 0 && blocking_piece.is_some() {
                    // Peça inimiga que seria atacada
                    let target_value = self.get_piece_value(current_sq as u8, board);
                    return Some((blocking_piece.unwrap(), target_value));
                } else {
                    break; // Configuração não permite ataque descoberto
                }
            }
        }

        None
    }

    /// Detecta vulnerabilidades posicionais
    fn detect_positional_vulnerabilities(&self, board: &Board, color: Color, phase_info: &GamePhaseInfo) -> Vec<VulnerabilityInfo> {
        let mut vulnerabilities = Vec::new();

        // Detecta casas fracas
        vulnerabilities.extend(self.detect_weak_squares(board, color));

        // Detecta fraquezas na estrutura de peões
        vulnerabilities.extend(self.detect_pawn_weaknesses(board, color));

        // Detecta problemas de coordenação
        vulnerabilities.extend(self.detect_coordination_issues(board, color));

        // Detecta problemas de segurança do rei
        vulnerabilities.extend(self.detect_king_safety_issues(board, color, phase_info));

        vulnerabilities
    }

    /// Detecta casas fracas
    fn detect_weak_squares(&self, board: &Board, color: Color) -> Vec<VulnerabilityInfo> {
        let mut vulnerabilities = Vec::new();
        let our_pawns = board.pawns & if color == Color::White { board.white_pieces } else { board.black_pieces };
        let enemy_pieces = if color == Color::White { board.black_pieces } else { board.white_pieces };

        // Verifica casas que não podem ser defendidas por peões
        for square in 0..64 {
            if self.is_weak_square(square, color, our_pawns) {
                // Verifica se inimigo pode ocupar esta casa com vantagem
                if self.can_enemy_exploit_weak_square(square, board, color) {
                    let mut vuln = VulnerabilityInfo::new(VulnerabilityType::WeakSquares, VulnerabilitySeverity::Minor);
                    vuln.affected_squares = vec![square];
                    vuln.impact_score = -15;
                    vuln.description = format!("Casa fraca em {}", self.square_name(square));
                    vulnerabilities.push(vuln);
                }
            }
        }

        vulnerabilities
    }

    /// Verifica se uma casa é fraca
    fn is_weak_square(&self, square: u8, color: Color, our_pawns: Bitboard) -> bool {
        let file = square % 8;
        let rank = square / 8;

        // Casa fraca se nossos peões não podem defendê-la
        let adjacent_files = [file.saturating_sub(1), file, file.saturating_add(1).min(7)];
        
        for &adj_file in &adjacent_files {
            if adj_file == file { continue; }
            
            let file_pawns = our_pawns & (0x0101010101010101u64 << adj_file);
            if file_pawns != 0 {
                // Verifica se algum peão neste arquivo pode avançar para defender
                let pawn_squares = self.get_set_bits(file_pawns);
                for &pawn_sq in &pawn_squares {
                    let pawn_rank = pawn_sq / 8;
                    if color == Color::White {
                        if pawn_rank < rank { // Peão pode avançar para defender
                            return false;
                        }
                    } else {
                        if pawn_rank > rank { // Peão pode avançar para defender
                            return false;
                        }
                    }
                }
            }
        }

        true
    }

    /// Verifica se inimigo pode explorar casa fraca
    fn can_enemy_exploit_weak_square(&self, square: u8, board: &Board, color: Color) -> bool {
        let enemy_pieces = if color == Color::White { board.black_pieces } else { board.white_pieces };
        let enemy_knights = board.knights & enemy_pieces;
        let enemy_bishops = board.bishops & enemy_pieces;

        // Cavalos podem ocupar casas fracas efetivamente
        let knight_squares = self.get_set_bits(enemy_knights);
        for &knight_sq in &knight_squares {
            let knight_attacks = crate::moves::knight::get_knight_attacks_lookup(knight_sq);
            if (knight_attacks & (1u64 << square)) != 0 {
                return true;
            }
        }

        // Bispos podem pressionar casas fracas
        let bishop_squares = self.get_set_bits(enemy_bishops);
        for &bishop_sq in &bishop_squares {
            let bishop_attacks = crate::moves::sliding::get_bishop_attacks(bishop_sq, board.white_pieces | board.black_pieces);
            if (bishop_attacks & (1u64 << square)) != 0 {
                return true;
            }
        }

        false
    }

    /// Detecta fraquezas na estrutura de peões
    fn detect_pawn_weaknesses(&self, board: &Board, color: Color) -> Vec<VulnerabilityInfo> {
        let mut vulnerabilities = Vec::new();
        let our_pawns = board.pawns & if color == Color::White { board.white_pieces } else { board.black_pieces };
        let enemy_pawns = board.pawns & if color == Color::White { board.black_pieces } else { board.white_pieces };

        // Detecta peões isolados
        vulnerabilities.extend(self.detect_isolated_pawns(our_pawns));

        // Detecta peões dobrados
        vulnerabilities.extend(self.detect_doubled_pawns(our_pawns));

        // Detecta peões atrasados
        vulnerabilities.extend(self.detect_backward_pawns(our_pawns, enemy_pawns, color));

        vulnerabilities
    }

    /// Detecta peões isolados
    fn detect_isolated_pawns(&self, our_pawns: Bitboard) -> Vec<VulnerabilityInfo> {
        let mut vulnerabilities = Vec::new();

        for file in 0..8 {
            let file_mask = 0x0101010101010101u64 << file;
            let file_pawns = our_pawns & file_mask;
            
            if file_pawns != 0 {
                // Verifica se há peões nos arquivos adjacentes
                let left_file = if file > 0 { 0x0101010101010101u64 << (file - 1) } else { 0 };
                let right_file = if file < 7 { 0x0101010101010101u64 << (file + 1) } else { 0 };
                
                if (our_pawns & (left_file | right_file)) == 0 {
                    // Peão isolado encontrado
                    let pawn_squares = self.get_set_bits(file_pawns);
                    for pawn_sq in pawn_squares {
                        let mut vuln = VulnerabilityInfo::new(VulnerabilityType::PawnWeaknesses, VulnerabilitySeverity::Minor);
                        vuln.affected_squares = vec![pawn_sq];
                        vuln.impact_score = -20;
                        vuln.description = format!("Peão isolado em {}", self.square_name(pawn_sq));
                        vulnerabilities.push(vuln);
                    }
                }
            }
        }

        vulnerabilities
    }

    /// Detecta peões dobrados
    fn detect_doubled_pawns(&self, our_pawns: Bitboard) -> Vec<VulnerabilityInfo> {
        let mut vulnerabilities = Vec::new();

        for file in 0..8 {
            let file_mask = 0x0101010101010101u64 << file;
            let file_pawns = our_pawns & file_mask;
            
            if file_pawns.count_ones() > 1 {
                // Peões dobrados encontrados
                let pawn_squares = self.get_set_bits(file_pawns);
                let mut vuln = VulnerabilityInfo::new(VulnerabilityType::PawnWeaknesses, VulnerabilitySeverity::Minor);
                vuln.affected_squares = pawn_squares.clone();
                vuln.impact_score = -15 * (pawn_squares.len() as i32 - 1);
                vuln.description = format!("Peões dobrados no arquivo {}", (b'a' + file as u8) as char);
                vulnerabilities.push(vuln);
            }
        }

        vulnerabilities
    }

    /// Detecta peões atrasados
    fn detect_backward_pawns(&self, our_pawns: Bitboard, enemy_pawns: Bitboard, color: Color) -> Vec<VulnerabilityInfo> {
        let mut vulnerabilities = Vec::new();
        let pawn_squares = self.get_set_bits(our_pawns);

        for &pawn_sq in &pawn_squares {
            if self.is_backward_pawn(pawn_sq, our_pawns, enemy_pawns, color) {
                let mut vuln = VulnerabilityInfo::new(VulnerabilityType::PawnWeaknesses, VulnerabilitySeverity::Minor);
                vuln.affected_squares = vec![pawn_sq];
                vuln.impact_score = -25;
                vuln.description = format!("Peão atrasado em {}", self.square_name(pawn_sq));
                vulnerabilities.push(vuln);
            }
        }

        vulnerabilities
    }

    /// Verifica se peão está atrasado
    fn is_backward_pawn(&self, pawn_sq: u8, our_pawns: Bitboard, enemy_pawns: Bitboard, color: Color) -> bool {
        let file = pawn_sq % 8;
        let rank = pawn_sq / 8;

        // Peão está atrasado se:
        // 1. Não pode ser defendido por peões adjacentes
        // 2. Sua casa de avanço é controlada pelo inimigo
        // 3. Peões adjacentes estão mais avançados

        let adjacent_files = [file.saturating_sub(1), file.saturating_add(1).min(7)];
        
        for &adj_file in &adjacent_files {
            if adj_file == file { continue; }
            
            let adj_file_pawns = our_pawns & (0x0101010101010101u64 << adj_file);
            if adj_file_pawns != 0 {
                let adj_pawn_squares = self.get_set_bits(adj_file_pawns);
                for &adj_pawn_sq in &adj_pawn_squares {
                    let adj_rank = adj_pawn_sq / 8;
                    
                    // Verifica se peão adjacente pode defender
                    if color == Color::White {
                        if adj_rank >= rank { // Peão adjacente não está atrasado
                            return false;
                        }
                    } else {
                        if adj_rank <= rank { // Peão adjacente não está atrasado
                            return false;
                        }
                    }
                }
            }
        }

        // Verifica se casa de avanço está controlada pelo inimigo
        let advance_sq = if color == Color::White { pawn_sq + 8 } else { pawn_sq - 8 };
        if advance_sq < 64 {
            // Simplificado: assume que se há peão inimigo atacando, está controlada
            let enemy_pawn_attacks = self.get_pawn_attacks_to_square(advance_sq, !color, enemy_pawns);
            if enemy_pawn_attacks > 0 {
                return true;
            }
        }

        false
    }

    /// Calcula ataques de peão a uma casa
    fn get_pawn_attacks_to_square(&self, square: u8, attacking_color: Color, pawns: Bitboard) -> u32 {
        let file = square % 8;
        let rank = square / 8;
        let mut attacks = 0;

        if attacking_color == Color::White {
            // Peões brancos atacam de baixo
            if rank > 0 {
                if file > 0 {
                    let attacking_sq = square - 9;
                    if (pawns & (1u64 << attacking_sq)) != 0 {
                        attacks += 1;
                    }
                }
                if file < 7 {
                    let attacking_sq = square - 7;
                    if (pawns & (1u64 << attacking_sq)) != 0 {
                        attacks += 1;
                    }
                }
            }
        } else {
            // Peões pretos atacam de cima
            if rank < 7 {
                if file > 0 {
                    let attacking_sq = square + 7;
                    if (pawns & (1u64 << attacking_sq)) != 0 {
                        attacks += 1;
                    }
                }
                if file < 7 {
                    let attacking_sq = square + 9;
                    if (pawns & (1u64 << attacking_sq)) != 0 {
                        attacks += 1;
                    }
                }
            }
        }

        attacks
    }

    /// Detecta problemas de coordenação
    fn detect_coordination_issues(&self, board: &Board, color: Color) -> Vec<VulnerabilityInfo> {
        let mut vulnerabilities = Vec::new();

        // Detecta peças mal coordenadas
        vulnerabilities.extend(self.detect_uncoordinated_pieces(board, color));

        // Detecta falta de desenvolvimento
        vulnerabilities.extend(self.detect_development_issues(board, color));

        vulnerabilities
    }

    /// Detecta peças mal coordenadas
    fn detect_uncoordinated_pieces(&self, board: &Board, color: Color) -> Vec<VulnerabilityInfo> {
        let mut vulnerabilities = Vec::new();
        let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
        
        // Analisa cada peça para coordenação
        let piece_squares = self.get_set_bits(our_pieces);
        
        for &piece_sq in &piece_squares {
            let coordination_score = self.calculate_piece_coordination(piece_sq, board, color);
            
            if coordination_score < -20 { // Mal coordenada
                let severity = if coordination_score < -40 {
                    VulnerabilitySeverity::Moderate
                } else {
                    VulnerabilitySeverity::Minor
                };

                let mut vuln = VulnerabilityInfo::new(VulnerabilityType::PieceCoordination, severity);
                vuln.affected_squares = vec![piece_sq];
                vuln.impact_score = coordination_score;
                vuln.description = format!("Peça mal coordenada em {}", self.square_name(piece_sq));
                vulnerabilities.push(vuln);
            }
        }

        vulnerabilities
    }

    /// Calcula score de coordenação de uma peça
    fn calculate_piece_coordination(&self, piece_sq: u8, board: &Board, color: Color) -> i32 {
        let mut score = 0;
        let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
        
        // Verifica se a peça colabora com outras
        let piece_attacks = self.get_piece_attacks(piece_sq, board);
        let ally_squares = self.get_set_bits(our_pieces);
        
        for &ally_sq in &ally_squares {
            if ally_sq == piece_sq { continue; }
            
            let ally_attacks = self.get_piece_attacks(ally_sq, board);
            let shared_control = (piece_attacks & ally_attacks).count_ones();
            
            if shared_control > 0 {
                score += shared_control as i32 * 3; // Bônus por controle compartilhado
            }
        }

        // Penalidade se a peça está isolada
        let distance_to_allies = self.calculate_average_distance_to_allies(piece_sq, board, color);
        if distance_to_allies > 4.0 {
            score -= 20; // Peça isolada
        }

        score
    }

    /// Obtém ataques de uma peça
    fn get_piece_attacks(&self, piece_sq: u8, board: &Board) -> Bitboard {
        let piece_bit = 1u64 << piece_sq;
        
        if (board.pawns & piece_bit) != 0 {
            // Simplificado para peões
            0u64
        } else if (board.knights & piece_bit) != 0 {
            crate::moves::knight::get_knight_attacks_lookup(piece_sq)
        } else if (board.bishops & piece_bit) != 0 {
            crate::moves::sliding::get_bishop_attacks(piece_sq, board.white_pieces | board.black_pieces)
        } else if (board.rooks & piece_bit) != 0 {
            crate::moves::sliding::get_rook_attacks(piece_sq, board.white_pieces | board.black_pieces)
        } else if (board.queens & piece_bit) != 0 {
            let bishop_attacks = crate::moves::sliding::get_bishop_attacks(piece_sq, board.white_pieces | board.black_pieces);
            let rook_attacks = crate::moves::sliding::get_rook_attacks(piece_sq, board.white_pieces | board.black_pieces);
            bishop_attacks | rook_attacks
        } else if (board.kings & piece_bit) != 0 {
            crate::moves::king::get_king_attacks_lookup(piece_sq)
        } else {
            0u64
        }
    }

    /// Calcula distância média para aliados
    fn calculate_average_distance_to_allies(&self, piece_sq: u8, board: &Board, color: Color) -> f32 {
        let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
        let ally_squares = self.get_set_bits(our_pieces);
        
        if ally_squares.len() <= 1 { return 0.0; }
        
        let mut total_distance = 0.0;
        let mut count = 0;
        
        for &ally_sq in &ally_squares {
            if ally_sq != piece_sq {
                total_distance += self.chebyshev_distance(piece_sq, ally_sq) as f32;
                count += 1;
            }
        }
        
        if count > 0 { total_distance / count as f32 } else { 0.0 }
    }

    /// Calcula distância de Chebyshev (distância do rei)
    fn chebyshev_distance(&self, sq1: u8, sq2: u8) -> u8 {
        let file1 = sq1 % 8;
        let rank1 = sq1 / 8;
        let file2 = sq2 % 8;
        let rank2 = sq2 / 8;
        
        let file_diff = (file1 as i8 - file2 as i8).abs() as u8;
        let rank_diff = (rank1 as i8 - rank2 as i8).abs() as u8;
        
        file_diff.max(rank_diff)
    }

    /// Detecta problemas de desenvolvimento
    fn detect_development_issues(&self, board: &Board, color: Color) -> Vec<VulnerabilityInfo> {
        let mut vulnerabilities = Vec::new();
        
        // Verifica se peças menores ainda estão nas casas iniciais
        let undeveloped_pieces = self.find_undeveloped_pieces(board, color);
        
        if undeveloped_pieces.len() >= 2 {
            let mut vuln = VulnerabilityInfo::new(VulnerabilityType::PoorDevelopment, VulnerabilitySeverity::Moderate);
            vuln.affected_squares = undeveloped_pieces.clone();
            vuln.impact_score = -(undeveloped_pieces.len() as i32 * 15);
            vuln.description = format!("{} peças ainda não desenvolvidas", undeveloped_pieces.len());
            vulnerabilities.push(vuln);
        }

        vulnerabilities
    }

    /// Encontra peças não desenvolvidas
    fn find_undeveloped_pieces(&self, board: &Board, color: Color) -> Vec<u8> {
        let mut undeveloped = Vec::new();
        let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
        
        let initial_squares = if color == Color::White {
            vec![1, 2, 5, 6] // b1, c1, f1, g1 (cavalos e bispos)
        } else {
            vec![57, 58, 61, 62] // b8, c8, f8, g8
        };

        for &sq in &initial_squares {
            let sq_bit = 1u64 << sq;
            if (our_pieces & sq_bit) != 0 && ((board.knights | board.bishops) & sq_bit) != 0 {
                undeveloped.push(sq);
            }
        }

        undeveloped
    }

    /// Detecta problemas de segurança do rei
    fn detect_king_safety_issues(&self, board: &Board, color: Color, phase_info: &GamePhaseInfo) -> Vec<VulnerabilityInfo> {
        let mut vulnerabilities = Vec::new();
        let our_king = board.kings & if color == Color::White { board.white_pieces } else { board.black_pieces };
        
        if our_king == 0 { return vulnerabilities; }
        let king_sq = our_king.trailing_zeros() as u8;

        // Avalia segurança do rei baseado na fase do jogo
        let safety_score = self.evaluate_king_safety_detailed(king_sq, board, color, phase_info);
        
        if safety_score < -30 {
            let severity = match safety_score {
                -30..-20 => VulnerabilitySeverity::Minor,
                -50..-30 => VulnerabilitySeverity::Moderate,
                -80..-50 => VulnerabilitySeverity::Serious,
                _ => VulnerabilitySeverity::Critical,
            };

            let mut vuln = VulnerabilityInfo::new(VulnerabilityType::KingSafety, severity);
            vuln.affected_squares = vec![king_sq];
            vuln.impact_score = safety_score;
            vuln.description = format!("Segurança do rei comprometida (score: {})", safety_score);
            vulnerabilities.push(vuln);
        }

        vulnerabilities
    }

    /// Avalia segurança do rei detalhadamente
    fn evaluate_king_safety_detailed(&self, king_sq: u8, board: &Board, color: Color, phase_info: &GamePhaseInfo) -> i32 {
        let mut safety_score = 0;

        // 1. Escudo de peões
        safety_score += self.evaluate_pawn_shield_safety(king_sq, board, color);

        // 2. Peças atacantes inimigas próximas
        safety_score -= self.count_enemy_attackers_near_king(king_sq, board, color) * 10;

        // 3. Casas fracas ao redor do rei
        safety_score -= self.count_weak_squares_around_king(king_sq, board, color) * 8;

        // 4. Abertura de linhas/colunas
        safety_score -= self.evaluate_open_lines_to_king(king_sq, board, color) * 12;

        // 5. Fase do jogo (rei mais vulnerável no meio-jogo)
        match phase_info.phase {
            super::game_phase::GamePhase::Middlegame | super::game_phase::GamePhase::LateMiddlegame => {
                safety_score = (safety_score as f32 * 1.5) as i32; // Amplifica vulnerabilidades
            },
            _ => {},
        }

        safety_score
    }

    /// Avalia escudo de peões do rei
    fn evaluate_pawn_shield_safety(&self, king_sq: u8, board: &Board, color: Color) -> i32 {
        let mut shield_score = 0;
        let our_pawns = board.pawns & if color == Color::White { board.white_pieces } else { board.black_pieces };
        let king_file = king_sq % 8;
        let king_rank = king_sq / 8;

        // Verifica peões nos arquivos adjacentes
        for file_offset in -1..=1i8 {
            let check_file = king_file as i8 + file_offset;
            if check_file < 0 || check_file > 7 { continue; }

            let file_mask = 0x0101010101010101u64 << check_file;
            let file_pawns = our_pawns & file_mask;

            if file_pawns == 0 {
                // Arquivo sem peões - vulnerabilidade
                shield_score -= 20;
            } else {
                // Verifica distância do peão mais próximo
                let pawn_squares = self.get_set_bits(file_pawns);
                let closest_distance = pawn_squares.iter()
                    .map(|&pawn_sq| self.chebyshev_distance(king_sq, pawn_sq))
                    .min()
                    .unwrap_or(8);

                if closest_distance > 2 {
                    shield_score -= (closest_distance as i32 - 2) * 5;
                }
            }
        }

        shield_score
    }

    /// Conta atacantes inimigos próximos ao rei
    fn count_enemy_attackers_near_king(&self, king_sq: u8, board: &Board, color: Color) -> i32 {
        let enemy_pieces = if color == Color::White { board.black_pieces } else { board.white_pieces };
        let king_area = crate::moves::king::get_king_attacks_lookup(king_sq);
        let extended_king_area = self.get_extended_king_area(king_sq);

        let mut attackers = 0;

        // Conta peças inimigas que atacam a área do rei
        let enemy_squares = self.get_set_bits(enemy_pieces);
        for &enemy_sq in &enemy_squares {
            let enemy_attacks = self.get_piece_attacks(enemy_sq, board);
            if (enemy_attacks & (king_area | extended_king_area)) != 0 {
                attackers += 1;
            }
        }

        attackers
    }

    /// Obtém área expandida ao redor do rei
    fn get_extended_king_area(&self, king_sq: u8) -> Bitboard {
        let king_attacks = crate::moves::king::get_king_attacks_lookup(king_sq);
        let mut extended_area = king_attacks;

        // Expande uma casa adicional
        let king_area_squares = self.get_set_bits(king_attacks);
        for &sq in &king_area_squares {
            extended_area |= crate::moves::king::get_king_attacks_lookup(sq);
        }

        extended_area
    }

    /// Conta casas fracas ao redor do rei
    fn count_weak_squares_around_king(&self, king_sq: u8, board: &Board, color: Color) -> i32 {
        let our_pawns = board.pawns & if color == Color::White { board.white_pieces } else { board.black_pieces };
        let king_area = crate::moves::king::get_king_attacks_lookup(king_sq);
        let king_area_squares = self.get_set_bits(king_area);

        let mut weak_squares = 0;
        for &sq in &king_area_squares {
            if self.is_weak_square(sq, color, our_pawns) {
                weak_squares += 1;
            }
        }

        weak_squares
    }

    /// Avalia linhas abertas em direção ao rei
    fn evaluate_open_lines_to_king(&self, king_sq: u8, board: &Board, color: Color) -> i32 {
        let mut open_lines = 0;
        let enemy_pieces = if color == Color::White { board.black_pieces } else { board.white_pieces };
        let enemy_heavy = (board.rooks | board.queens) & enemy_pieces;

        if enemy_heavy == 0 { return 0; }

        let king_file = king_sq % 8;
        let king_rank = king_sq / 8;

        // Verifica arquivo do rei
        let file_mask = 0x0101010101010101u64 << king_file;
        if (board.pawns & file_mask).count_ones() <= 1 {
            // Arquivo semi-aberto ou aberto
            let file_heavy = enemy_heavy & file_mask;
            if file_heavy != 0 {
                open_lines += 1;
            }
        }

        // Verifica fileira do rei
        let rank_mask = 0xFFu64 << (king_rank * 8);
        if (board.pawns & rank_mask).count_ones() <= 2 {
            // Fileira relativamente aberta
            let rank_heavy = enemy_heavy & rank_mask;
            if rank_heavy != 0 {
                open_lines += 1;
            }
        }

        open_lines
    }

    /// Detecta vulnerabilidades estratégicas
    fn detect_strategic_vulnerabilities(&self, board: &Board, color: Color, phase_info: &GamePhaseInfo) -> Vec<VulnerabilityInfo> {
        let mut vulnerabilities = Vec::new();

        // Detecta posicionamento ruim de peças
        vulnerabilities.extend(self.detect_bad_piece_positioning(board, color, phase_info));

        // Detecta desalinhamento estratégico
        vulnerabilities.extend(self.detect_strategic_misalignment(board, color, phase_info));

        // Detecta fraquezas de final
        vulnerabilities.extend(self.detect_endgame_weaknesses(board, color, phase_info));

        vulnerabilities
    }

    /// Detecta posicionamento ruim de peças
    fn detect_bad_piece_positioning(&self, board: &Board, color: Color, phase_info: &GamePhaseInfo) -> Vec<VulnerabilityInfo> {
        let mut vulnerabilities = Vec::new();
        let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };

        // Analisa posicionamento de cada tipo de peça
        self.analyze_piece_positioning(board, color, phase_info, &mut vulnerabilities);

        vulnerabilities
    }

    /// Analisa posicionamento específico de peças
    fn analyze_piece_positioning(&self, board: &Board, color: Color, phase_info: &GamePhaseInfo, vulnerabilities: &mut Vec<VulnerabilityInfo>) {
        let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };

        // Analisa posicionamento de cavalos
        let our_knights = board.knights & our_pieces;
        let knight_squares = self.get_set_bits(our_knights);
        
        for &knight_sq in &knight_squares {
            let positioning_score = self.evaluate_knight_positioning(knight_sq, board, color, phase_info);
            if positioning_score < -25 {
                let mut vuln = VulnerabilityInfo::new(VulnerabilityType::BadPiecePositioning, VulnerabilitySeverity::Minor);
                vuln.affected_squares = vec![knight_sq];
                vuln.impact_score = positioning_score;
                vuln.description = format!("Cavalo mal posicionado em {}", self.square_name(knight_sq));
                vulnerabilities.push(vuln);
            }
        }

        // Analisa posicionamento de bispos
        let our_bishops = board.bishops & our_pieces;
        let bishop_squares = self.get_set_bits(our_bishops);
        
        for &bishop_sq in &bishop_squares {
            let positioning_score = self.evaluate_bishop_positioning(bishop_sq, board, color, phase_info);
            if positioning_score < -25 {
                let mut vuln = VulnerabilityInfo::new(VulnerabilityType::BadPiecePositioning, VulnerabilitySeverity::Minor);
                vuln.affected_squares = vec![bishop_sq];
                vuln.impact_score = positioning_score;
                vuln.description = format!("Bispo mal posicionado em {}", self.square_name(bishop_sq));
                vulnerabilities.push(vuln);
            }
        }
    }

    /// Avalia posicionamento de cavalo
    fn evaluate_knight_positioning(&self, knight_sq: u8, board: &Board, color: Color, phase_info: &GamePhaseInfo) -> i32 {
        let mut score = 0;

        // Cavalos devem estar centralizados
        let centrality = self.calculate_centrality(knight_sq);
        score += centrality * 3;

        // Cavalos em outposts são valiosos
        if self.is_outpost_square(knight_sq, board, color) {
            score += 30;
        }

        // Penalidade se cavalo está na borda
        if self.is_edge_square_simple(knight_sq) {
            score -= 20;
        }

        // Cavalos devem atacar casas importantes
        let knight_attacks = crate::moves::knight::get_knight_attacks_lookup(knight_sq);
        let important_squares = self.get_important_squares_simple();
        let attacks_important = (knight_attacks & important_squares).count_ones() as i32;
        score += attacks_important * 5;

        score
    }

    /// Avalia posicionamento de bispo
    fn evaluate_bishop_positioning(&self, bishop_sq: u8, board: &Board, color: Color, phase_info: &GamePhaseInfo) -> i32 {
        let mut score = 0;

        // Bispos precisam de diagonais longas
        let bishop_attacks = crate::moves::sliding::get_bishop_attacks(bishop_sq, board.white_pieces | board.black_pieces);
        let diagonal_length = bishop_attacks.count_ones() as i32;
        score += diagonal_length * 2;

        // Penalidade se bispo está bloqueado por próprios peões
        let our_pawns = board.pawns & if color == Color::White { board.white_pieces } else { board.black_pieces };
        let blocked_by_pawns = (bishop_attacks & our_pawns).count_ones() as i32;
        score -= blocked_by_pawns * 8;

        // Bônus se bispo ataca casas centrais
        let center = 0x0000001818000000u64;
        if (bishop_attacks & center) != 0 {
            score += 15;
        }

        score
    }

    /// Calcula centralidade de uma casa
    fn calculate_centrality(&self, square: u8) -> i32 {
        let file = square % 8;
        let rank = square / 8;
        
        let file_centrality = 7 - (3.5 - file as f32).abs() as i32;
        let rank_centrality = 7 - (3.5 - rank as f32).abs() as i32;
        
        file_centrality + rank_centrality
    }

    /// Verifica se casa é outpost
    fn is_outpost_square(&self, square: u8, board: &Board, color: Color) -> bool {
        let file = square % 8;
        let rank = square / 8;
        let enemy_pawns = board.pawns & if color == Color::White { board.black_pieces } else { board.white_pieces };

        // Outpost deve estar em território inimigo ou neutro
        let valid_rank = if color == Color::White {
            rank >= 4
        } else {
            rank <= 3
        };

        if !valid_rank { return false; }

        // Não deve ser atacável por peões inimigos
        for adj_file in (file.saturating_sub(1))..=(file.saturating_add(1)).min(7) {
            if adj_file == file { continue; }
            
            let file_mask = 0x0101010101010101u64 << adj_file;
            let file_pawns = enemy_pawns & file_mask;
            
            if file_pawns != 0 {
                let pawn_squares = self.get_set_bits(file_pawns);
                for &pawn_sq in &pawn_squares {
                    if self.pawn_can_attack_square(pawn_sq, square, !color) {
                        return false;
                    }
                }
            }
        }

        true
    }

    /// Verifica se peão pode atacar casa
    fn pawn_can_attack_square(&self, pawn_sq: u8, target_sq: u8, pawn_color: Color) -> bool {
        let pawn_file = pawn_sq % 8;
        let pawn_rank = pawn_sq / 8;
        let target_file = target_sq % 8;
        let target_rank = target_sq / 8;

        let file_diff = (target_file as i8 - pawn_file as i8).abs();
        if file_diff != 1 { return false; }

        if pawn_color == Color::White {
            target_rank > pawn_rank
        } else {
            target_rank < pawn_rank
        }
    }

    /// Verifica se casa está na borda
    fn is_edge_square_simple(&self, square: u8) -> bool {
        let file = square % 8;
        let rank = square / 8;
        file == 0 || file == 7 || rank == 0 || rank == 7
    }

    /// Obtém casas importantes simples
    fn get_important_squares_simple(&self) -> Bitboard {
        0x0000001818000000u64 // Centro simples
    }

    /// Detecta desalinhamento estratégico
    fn detect_strategic_misalignment(&self, board: &Board, color: Color, phase_info: &GamePhaseInfo) -> Vec<VulnerabilityInfo> {
        let mut vulnerabilities = Vec::new();

        // Verifica se estratégia está alinhada com a fase do jogo
        let strategy_alignment = self.evaluate_strategy_phase_alignment(board, color, phase_info);
        
        if strategy_alignment < -30 {
            let mut vuln = VulnerabilityInfo::new(VulnerabilityType::StrategicMisalignment, VulnerabilitySeverity::Moderate);
            vuln.impact_score = strategy_alignment;
            vuln.description = "Estratégia desalinhada com a fase do jogo".to_string();
            vulnerabilities.push(vuln);
        }

        vulnerabilities
    }

    /// Avalia alinhamento entre estratégia e fase do jogo
    fn evaluate_strategy_phase_alignment(&self, board: &Board, color: Color, phase_info: &GamePhaseInfo) -> i32 {
        let mut alignment_score = 0;

        match phase_info.phase {
            super::game_phase::GamePhase::Opening => {
                // Na abertura, deve focar em desenvolvimento
                alignment_score += self.evaluate_opening_principles(board, color);
            },
            super::game_phase::GamePhase::Middlegame | super::game_phase::GamePhase::LateMiddlegame => {
                // No meio-jogo, deve focar em atividade e ataques
                alignment_score += self.evaluate_middlegame_activity(board, color);
            },
            super::game_phase::GamePhase::Endgame | super::game_phase::GamePhase::PureEndgame => {
                // No final, deve focar em rei ativo e promoção
                alignment_score += self.evaluate_endgame_principles(board, color);
            },
            _ => {},
        }

        alignment_score
    }

    /// Avalia princípios de abertura
    fn evaluate_opening_principles(&self, board: &Board, color: Color) -> i32 {
        let mut score = 0;
        let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };

        // Desenvolvimento de peças menores
        let developed_pieces = self.count_developed_pieces(board, color);
        score += developed_pieces * 10;

        // Segurança do rei (roque)
        if self.has_king_castled(board, color) {
            score += 20;
        }

        // Não mover mesma peça múltiplas vezes (simplificado)
        // Controle do centro
        let center_control = self.evaluate_center_control(board, color);
        score += center_control;

        score
    }

    /// Conta peças desenvolvidas
    fn count_developed_pieces(&self, board: &Board, color: Color) -> i32 {
        let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
        let initial_squares = if color == Color::White {
            vec![1, 2, 5, 6] // b1, c1, f1, g1
        } else {
            vec![57, 58, 61, 62] // b8, c8, f8, g8
        };

        let mut developed = 0;
        for &sq in &initial_squares {
            let sq_bit = 1u64 << sq;
            if (our_pieces & sq_bit) == 0 {
                // Peça saiu da casa inicial
                developed += 1;
            }
        }

        developed
    }

    /// Verifica se rei fez roque
    fn has_king_castled(&self, board: &Board, color: Color) -> bool {
        let our_king = board.kings & if color == Color::White { board.white_pieces } else { board.black_pieces };
        if our_king == 0 { return false; }
        
        let king_sq = our_king.trailing_zeros() as u8;
        let castled_squares = if color == Color::White {
            [2, 6] // c1, g1
        } else {
            [58, 62] // c8, g8
        };

        castled_squares.contains(&king_sq)
    }

    /// Avalia controle do centro
    fn evaluate_center_control(&self, board: &Board, color: Color) -> i32 {
        let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
        let center = 0x0000001818000000u64; // d4, e4, d5, e5
        
        let mut control_score = 0;
        
        // Conta peões no centro
        let center_pawns = (board.pawns & our_pieces & center).count_ones() as i32;
        control_score += center_pawns * 15;

        // Conta ataques ao centro (simplificado)
        let piece_squares = self.get_set_bits(our_pieces);
        for &piece_sq in &piece_squares {
            let attacks = self.get_piece_attacks(piece_sq, board);
            let center_attacks = (attacks & center).count_ones() as i32;
            control_score += center_attacks * 3;
        }

        control_score
    }

    /// Avalia atividade no meio-jogo
    fn evaluate_middlegame_activity(&self, board: &Board, color: Color) -> i32 {
        let mut score = 0;
        let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };

        // Mobilidade das peças
        let total_mobility = self.calculate_total_mobility(board, color);
        score += total_mobility / 3;

        // Pressão sobre peças inimigas
        let pressure_score = self.calculate_pressure_on_enemy(board, color);
        score += pressure_score;

        // Controle de casas importantes
        let important_control = self.calculate_important_square_control(board, color);
        score += important_control;

        score
    }

    /// Calcula mobilidade total
    fn calculate_total_mobility(&self, board: &Board, color: Color) -> i32 {
        let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
        let piece_squares = self.get_set_bits(our_pieces);
        
        let mut total_mobility = 0;
        for &piece_sq in &piece_squares {
            let attacks = self.get_piece_attacks(piece_sq, board);
            let legal_moves = attacks & !(our_pieces); // Não pode capturar próprias peças
            total_mobility += legal_moves.count_ones() as i32;
        }

        total_mobility
    }

    /// Calcula pressão sobre inimigo
    fn calculate_pressure_on_enemy(&self, board: &Board, color: Color) -> i32 {
        let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
        let enemy_pieces = if color == Color::White { board.black_pieces } else { board.white_pieces };
        let piece_squares = self.get_set_bits(our_pieces);
        
        let mut pressure = 0;
        for &piece_sq in &piece_squares {
            let attacks = self.get_piece_attacks(piece_sq, board);
            let attacks_enemy = (attacks & enemy_pieces).count_ones() as i32;
            pressure += attacks_enemy * 5;
        }

        pressure
    }

    /// Calcula controle de casas importantes
    fn calculate_important_square_control(&self, board: &Board, color: Color) -> i32 {
        let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
        let important = self.get_important_squares_simple();
        let piece_squares = self.get_set_bits(our_pieces);
        
        let mut control = 0;
        for &piece_sq in &piece_squares {
            let attacks = self.get_piece_attacks(piece_sq, board);
            let controls_important = (attacks & important).count_ones() as i32;
            control += controls_important * 3;
        }

        control
    }

    /// Avalia princípios de final
    fn evaluate_endgame_principles(&self, board: &Board, color: Color) -> i32 {
        let mut score = 0;

        // Rei deve ser ativo no final
        let king_activity = self.evaluate_king_activity_endgame(board, color);
        score += king_activity;

        // Peões passados são cruciais
        let passed_pawns_value = self.evaluate_passed_pawns_endgame(board, color);
        score += passed_pawns_value;

        // Centralização de peças
        let centralization = self.evaluate_piece_centralization(board, color);
        score += centralization;

        score
    }

    /// Avalia atividade do rei no final
    fn evaluate_king_activity_endgame(&self, board: &Board, color: Color) -> i32 {
        let our_king = board.kings & if color == Color::White { board.white_pieces } else { board.black_pieces };
        if our_king == 0 { return 0; }
        
        let king_sq = our_king.trailing_zeros() as u8;
        
        // Rei deve estar centralizado no final - valor aumentado para finais
        let centrality = self.calculate_centrality(king_sq);
        centrality * 15  // Aumentado de 3 para 15 conforme análise
    }

    /// Avalia peões passados no final
    fn evaluate_passed_pawns_endgame(&self, board: &Board, color: Color) -> i32 {
        let our_pawns = board.pawns & if color == Color::White { board.white_pieces } else { board.black_pieces };
        let enemy_pawns = board.pawns & if color == Color::White { board.black_pieces } else { board.white_pieces };
        
        let mut score = 0;
        let pawn_squares = self.get_set_bits(our_pawns);
        
        for &pawn_sq in &pawn_squares {
            if self.is_passed_pawn_simple(pawn_sq, color, our_pawns, enemy_pawns) {
                let rank = pawn_sq / 8;
                let advancement = if color == Color::White { rank } else { 7 - rank };
                score += advancement as i32 * 10; // Peões mais avançados valem mais
            }
        }

        score
    }

    /// Verifica se peão é passado (simplificado)
    fn is_passed_pawn_simple(&self, pawn_sq: u8, color: Color, our_pawns: Bitboard, enemy_pawns: Bitboard) -> bool {
        let file = pawn_sq % 8;
        let rank = pawn_sq / 8;

        // Verifica se há peões inimigos no arquivo ou adjacentes que podem bloquear
        for check_file in (file.saturating_sub(1))..=(file.saturating_add(1)).min(7) {
            let file_mask = 0x0101010101010101u64 << check_file;
            let enemy_file_pawns = enemy_pawns & file_mask;
            
            if enemy_file_pawns != 0 {
                let enemy_pawn_squares = self.get_set_bits(enemy_file_pawns);
                for &enemy_pawn_sq in &enemy_pawn_squares {
                    let enemy_rank = enemy_pawn_sq / 8;
                    
                    if color == Color::White {
                        if enemy_rank > rank { // Peão inimigo à frente
                            return false;
                        }
                    } else {
                        if enemy_rank < rank { // Peão inimigo à frente
                            return false;
                        }
                    }
                }
            }
        }

        true
    }

    /// Avalia centralização de peças
    fn evaluate_piece_centralization(&self, board: &Board, color: Color) -> i32 {
        let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
        let piece_squares = self.get_set_bits(our_pieces);
        
        let mut centralization_score = 0;
        for &piece_sq in &piece_squares {
            let centrality = self.calculate_centrality(piece_sq);
            centralization_score += centrality;
        }

        centralization_score / 4 // Normaliza
    }

    /// Detecta fraquezas de final
    fn detect_endgame_weaknesses(&self, board: &Board, color: Color, phase_info: &GamePhaseInfo) -> Vec<VulnerabilityInfo> {
        let mut vulnerabilities = Vec::new();

        // Só analisa se estivermos próximos ou no final
        match phase_info.phase {
            super::game_phase::GamePhase::LateMiddlegame | 
            super::game_phase::GamePhase::EarlyEndgame |
            super::game_phase::GamePhase::Endgame |
            super::game_phase::GamePhase::PureEndgame => {
                
                // Verifica se rei está passivo
                if self.is_king_passive_endgame(board, color) {
                    let mut vuln = VulnerabilityInfo::new(VulnerabilityType::EndgameWeaknesses, VulnerabilitySeverity::Serious);
                    vuln.description = "Rei passivo no final".to_string();
                    vuln.impact_score = -75; // CORRIGIDO: Aumentado de -25 para -75
                    vulnerabilities.push(vuln);
                }

                // Verifica falta de peões passados
                if !self.has_passed_pawns(board, color) {
                    let mut vuln = VulnerabilityInfo::new(VulnerabilityType::EndgameWeaknesses, VulnerabilitySeverity::Minor);
                    vuln.description = "Ausência de peões passados".to_string();
                    vuln.impact_score = -20;
                    vulnerabilities.push(vuln);
                }
            },
            _ => {},
        }

        vulnerabilities
    }

    /// Verifica se rei está passivo no final
    fn is_king_passive_endgame(&self, board: &Board, color: Color) -> bool {
        let our_king = board.kings & if color == Color::White { board.white_pieces } else { board.black_pieces };
        if our_king == 0 { return true; }
        
        let king_sq = our_king.trailing_zeros() as u8;
        let centrality = self.calculate_centrality(king_sq);
        
        centrality < 8 // Se centralidade é baixa, rei está passivo
    }

    /// Verifica se tem peões passados
    fn has_passed_pawns(&self, board: &Board, color: Color) -> bool {
        let our_pawns = board.pawns & if color == Color::White { board.white_pieces } else { board.black_pieces };
        let enemy_pawns = board.pawns & if color == Color::White { board.black_pieces } else { board.white_pieces };
        let pawn_squares = self.get_set_bits(our_pawns);
        
        for &pawn_sq in &pawn_squares {
            if self.is_passed_pawn_simple(pawn_sq, color, our_pawns, enemy_pawns) {
                return true;
            }
        }

        false
    }

    /// Avalia situação estratégica
    fn assess_strategic_situation(&self, board: &Board, color: Color, vulnerabilities: &[VulnerabilityInfo], phase_info: &GamePhaseInfo) -> StrategicAssessment {
        let mut assessment = StrategicAssessment {
            king_safety_rating: 0.5,
            piece_activity_rating: 0.5,
            pawn_structure_rating: 0.5,
            tactical_alertness: 0.5,
            endgame_preparation: 0.5,
            overall_health: 0.5,
        };

        // Avalia segurança do rei
        assessment.king_safety_rating = self.calculate_king_safety_rating(board, color, vulnerabilities);

        // Avalia atividade das peças
        assessment.piece_activity_rating = self.calculate_piece_activity_rating(board, color);

        // Avalia estrutura de peões
        assessment.pawn_structure_rating = self.calculate_pawn_structure_rating(board, color, vulnerabilities);

        // Avalia alerta tático
        assessment.tactical_alertness = self.calculate_tactical_alertness(vulnerabilities);

        // Avalia preparação para o final
        assessment.endgame_preparation = self.calculate_endgame_preparation(board, color, phase_info);

        // Calcula saúde geral
        assessment.overall_health = (
            assessment.king_safety_rating +
            assessment.piece_activity_rating +
            assessment.pawn_structure_rating +
            assessment.tactical_alertness +
            assessment.endgame_preparation
        ) / 5.0;

        assessment
    }

    /// Calcula rating de segurança do rei
    fn calculate_king_safety_rating(&self, board: &Board, color: Color, vulnerabilities: &[VulnerabilityInfo]) -> f32 {
        let our_king = board.kings & if color == Color::White { board.white_pieces } else { board.black_pieces };
        if our_king == 0 { return 0.0; }
        
        let king_sq = our_king.trailing_zeros() as u8;
        let mut safety_score = 50; // Base score

        // Aplica penalidades baseadas em vulnerabilidades
        for vuln in vulnerabilities {
            if vuln.vulnerability_type == VulnerabilityType::KingSafety {
                safety_score += vuln.impact_score;
            }
        }

        // Normaliza para 0.0-1.0
        (safety_score.max(0).min(100) as f32) / 100.0
    }

    /// Calcula rating de atividade das peças
    fn calculate_piece_activity_rating(&self, board: &Board, color: Color) -> f32 {
        let mobility = self.calculate_total_mobility(board, color);
        let coordination = self.calculate_total_coordination(board, color);
        
        let total_score = mobility + coordination;
        let normalized = (total_score as f32 / 100.0).min(1.0).max(0.0);
        
        normalized
    }

    /// Calcula coordenação total
    fn calculate_total_coordination(&self, board: &Board, color: Color) -> i32 {
        let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
        let piece_squares = self.get_set_bits(our_pieces);
        
        let mut total_coordination = 0;
        for &piece_sq in &piece_squares {
            total_coordination += self.calculate_piece_coordination(piece_sq, board, color);
        }

        total_coordination / piece_squares.len().max(1) as i32
    }

    /// Calcula rating da estrutura de peões
    fn calculate_pawn_structure_rating(&self, board: &Board, color: Color, vulnerabilities: &[VulnerabilityInfo]) -> f32 {
        let mut structure_score = 50; // Base score

        // Aplica penalidades por fraquezas de peões
        for vuln in vulnerabilities {
            if vuln.vulnerability_type == VulnerabilityType::PawnWeaknesses {
                structure_score += vuln.impact_score / 2; // Reduz impacto
            }
        }

        // Bônus por peões passados
        if self.has_passed_pawns(board, color) {
            structure_score += 20;
        }

        (structure_score.max(0).min(100) as f32) / 100.0
    }

    /// Calcula alerta tático
    fn calculate_tactical_alertness(&self, vulnerabilities: &[VulnerabilityInfo]) -> f32 {
        let mut alertness_score = 50;

        // Penaliza vulnerabilidades táticas
        for vuln in vulnerabilities {
            match vuln.vulnerability_type {
                VulnerabilityType::PinnedPieces |
                VulnerabilityType::OverloadedPieces |
                VulnerabilityType::LooseHangingPieces |
                VulnerabilityType::TacticalMotifs => {
                    alertness_score += vuln.impact_score / 3;
                },
                _ => {},
            }
        }

        (alertness_score.max(0).min(100) as f32) / 100.0
    }

    /// Calcula preparação para final
    fn calculate_endgame_preparation(&self, board: &Board, color: Color, phase_info: &GamePhaseInfo) -> f32 {
        let mut preparation_score = 50;

        // Avalia baseado na fase do jogo
        match phase_info.phase {
            super::game_phase::GamePhase::Endgame | super::game_phase::GamePhase::PureEndgame => {
                preparation_score += self.evaluate_endgame_principles(board, color) / 2;
            },
            super::game_phase::GamePhase::LateMiddlegame => {
                preparation_score += self.evaluate_endgame_principles(board, color) / 4;
            },
            _ => {},
        }

        (preparation_score.max(0).min(100) as f32) / 100.0
    }

    /// Analisa tendências posicionais
    fn analyze_positional_trends(&self, board: &Board, color: Color, vulnerabilities: &[VulnerabilityInfo], phase_info: &GamePhaseInfo) -> PositionalTrends {
        let mut trends = PositionalTrends {
            improving_factors: Vec::new(),
            deteriorating_factors: Vec::new(),
            stable_elements: Vec::new(),
            critical_transitions: Vec::new(),
        };

        // Analisa fatores que melhoram
        if self.has_good_development(board, color) {
            trends.improving_factors.push("Desenvolvimento sólido".to_string());
        }

        if self.has_active_pieces(board, color) {
            trends.improving_factors.push("Peças ativas".to_string());
        }

        // Analisa fatores que deterioram
        let critical_vulns = vulnerabilities.iter()
            .filter(|v| v.severity >= VulnerabilitySeverity::Serious)
            .count();
            
        if critical_vulns > 0 {
            trends.deteriorating_factors.push(format!("{} vulnerabilidades sérias", critical_vulns));
        }

        // Analisa elementos estáveis
        if self.has_solid_pawn_structure(board, color) {
            trends.stable_elements.push("Estrutura de peões sólida".to_string());
        }

        // Analisa transições críticas
        self.analyze_critical_transitions(board, color, phase_info, &mut trends);

        trends
    }

    /// Verifica se tem bom desenvolvimento
    fn has_good_development(&self, board: &Board, color: Color) -> bool {
        let developed = self.count_developed_pieces(board, color);
        developed >= 2
    }

    /// Verifica se tem peças ativas
    fn has_active_pieces(&self, board: &Board, color: Color) -> bool {
        let mobility = self.calculate_total_mobility(board, color);
        mobility >= 30 // Threshold arbitrário
    }

    /// Verifica se tem estrutura de peões sólida
    fn has_solid_pawn_structure(&self, board: &Board, color: Color) -> bool {
        let our_pawns = board.pawns & if color == Color::White { board.white_pieces } else { board.black_pieces };
        let enemy_pawns = board.pawns & if color == Color::White { board.black_pieces } else { board.white_pieces };
        
        // Simplificado: se não há muitos peões dobrados/isolados
        let isolated_count = self.count_isolated_pawns(our_pawns);
        let doubled_count = self.count_doubled_pawns(our_pawns);
        
        isolated_count + doubled_count <= 2
    }

    /// Conta peões isolados
    fn count_isolated_pawns(&self, our_pawns: Bitboard) -> usize {
        let mut isolated_count = 0;

        for file in 0..8 {
            let file_mask = 0x0101010101010101u64 << file;
            let file_pawns = our_pawns & file_mask;
            
            if file_pawns != 0 {
                let left_file = if file > 0 { 0x0101010101010101u64 << (file - 1) } else { 0 };
                let right_file = if file < 7 { 0x0101010101010101u64 << (file + 1) } else { 0 };
                
                if (our_pawns & (left_file | right_file)) == 0 {
                    isolated_count += file_pawns.count_ones() as usize;
                }
            }
        }

        isolated_count
    }

    /// Conta peões dobrados
    fn count_doubled_pawns(&self, our_pawns: Bitboard) -> usize {
        let mut doubled_count = 0;

        for file in 0..8 {
            let file_mask = 0x0101010101010101u64 << file;
            let file_pawns = our_pawns & file_mask;
            let pawn_count = file_pawns.count_ones() as usize;
            
            if pawn_count > 1 {
                doubled_count += pawn_count - 1; // Excesso além de 1
            }
        }

        doubled_count
    }

    /// Analisa transições críticas
    fn analyze_critical_transitions(&self, board: &Board, color: Color, phase_info: &GamePhaseInfo, trends: &mut PositionalTrends) {
        match phase_info.phase {
            super::game_phase::GamePhase::Opening => {
                if self.is_transitioning_to_middlegame(board) {
                    trends.critical_transitions.push("Transição para meio-jogo".to_string());
                }
            },
            super::game_phase::GamePhase::LateMiddlegame => {
                if self.is_transitioning_to_endgame(board) {
                    trends.critical_transitions.push("Transição para final iminente".to_string());
                }
            },
            _ => {},
        }
    }

    /// Verifica transição para meio-jogo
    fn is_transitioning_to_middlegame(&self, board: &Board) -> bool {
        let total_pieces = (board.white_pieces | board.black_pieces).count_ones();
        total_pieces <= 28 // Threshold arbitrário
    }

    /// Verifica transição para final
    fn is_transitioning_to_endgame(&self, board: &Board) -> bool {
        let queens = board.queens.count_ones();
        let major_pieces = (board.queens | board.rooks).count_ones();
        
        queens <= 1 || major_pieces <= 4
    }

    /// Gera recomendações defensivas
    fn generate_defensive_recommendations(&self, board: &Board, color: Color, vulnerabilities: &[VulnerabilityInfo]) -> Vec<Move> {
        let mut recommendations = Vec::new();

        // Para cada vulnerabilidade crítica, tenta gerar defesas
        for vuln in vulnerabilities {
            if vuln.severity >= VulnerabilitySeverity::Serious {
                let defensive_moves = self.suggest_defensive_moves(vuln, board, color);
                recommendations.extend(defensive_moves);
            }
        }

        // Limita número de recomendações
        recommendations.truncate(5);
        recommendations
    }

    /// Sugere movimentos defensivos para uma vulnerabilidade
    fn suggest_defensive_moves(&self, vuln: &VulnerabilityInfo, board: &Board, color: Color) -> Vec<Move> {
        let mut defensive_moves = Vec::new();

        match vuln.vulnerability_type {
            VulnerabilityType::PinnedPieces => {
                // Sugere mover o rei ou interromper o prego
                defensive_moves.extend(self.suggest_pin_defenses(vuln, board, color));
            },
            VulnerabilityType::LooseHangingPieces => {
                // Sugere defender ou mover a peça
                defensive_moves.extend(self.suggest_hanging_piece_defenses(vuln, board, color));
            },
            VulnerabilityType::KingSafety => {
                // Sugere melhorar segurança do rei
                defensive_moves.extend(self.suggest_king_safety_improvements(vuln, board, color));
            },
            _ => {
                // Defesas genéricas
            },
        }

        defensive_moves
    }

    /// Sugere defesas contra pregos
    fn suggest_pin_defenses(&self, vuln: &VulnerabilityInfo, board: &Board, color: Color) -> Vec<Move> {
        let mut defenses = Vec::new();

        if let Some(&pinned_sq) = vuln.affected_squares.first() {
            // Movimento simples: mover a peça pregada (se legal)
            let simple_moves = self.generate_simple_moves_for_piece(pinned_sq, board);
            defenses.extend(simple_moves.into_iter().take(2)); // Limita a 2 movimentos
        }

        defenses
    }

    /// Sugere defesas para peças penduradas
    fn suggest_hanging_piece_defenses(&self, vuln: &VulnerabilityInfo, board: &Board, color: Color) -> Vec<Move> {
        let mut defenses = Vec::new();

        if let Some(&hanging_sq) = vuln.affected_squares.first() {
            // Tenta defender a peça ou movê-la
            let defensive_moves = self.generate_defensive_moves_for_piece(hanging_sq, board, color);
            defenses.extend(defensive_moves.into_iter().take(2));
        }

        defenses
    }

    /// Sugere melhorias na segurança do rei
    fn suggest_king_safety_improvements(&self, vuln: &VulnerabilityInfo, board: &Board, color: Color) -> Vec<Move> {
        let mut improvements = Vec::new();

        let our_king = board.kings & if color == Color::White { board.white_pieces } else { board.black_pieces };
        if our_king != 0 {
            let king_sq = our_king.trailing_zeros() as u8;
            
            // Sugere movimentos do rei para casas mais seguras
            let king_moves = self.generate_safer_king_moves(king_sq, board, color);
            improvements.extend(king_moves.into_iter().take(2));
        }

        improvements
    }

    /// Gera movimentos simples para uma peça
    fn generate_simple_moves_for_piece(&self, piece_sq: u8, board: &Board) -> Vec<Move> {
        let mut moves = Vec::new();
        let piece_attacks = self.get_piece_attacks(piece_sq, board);
        let legal_targets = piece_attacks & !(board.white_pieces | board.black_pieces);
        
        let target_squares = self.get_set_bits(legal_targets);
        for &target_sq in target_squares.iter().take(3) {
            moves.push(Move { from: piece_sq, to: target_sq, promotion: None, is_castling: false, is_en_passant: false });
        }

        moves
    }

    /// Gera movimentos defensivos para uma peça
    fn generate_defensive_moves_for_piece(&self, piece_sq: u8, board: &Board, color: Color) -> Vec<Move> {
        // Simplificado: retorna movimentos similares aos simples
        self.generate_simple_moves_for_piece(piece_sq, board)
    }

    /// Gera movimentos mais seguros para o rei
    fn generate_safer_king_moves(&self, king_sq: u8, board: &Board, color: Color) -> Vec<Move> {
        let mut safe_moves = Vec::new();
        let king_attacks = crate::moves::king::get_king_attacks_lookup(king_sq);
        let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
        let legal_targets = king_attacks & !our_pieces;
        
        let target_squares = self.get_set_bits(legal_targets);
        for &target_sq in target_squares.iter().take(2) {
            // Verifica se casa é mais segura (simplificado)
            if self.is_square_safer(target_sq, king_sq, board, color) {
                safe_moves.push(Move { from: king_sq, to: target_sq, promotion: None, is_castling: false, is_en_passant: false });
            }
        }

        safe_moves
    }

    /// Verifica se casa é mais segura
    fn is_square_safer(&self, new_sq: u8, current_sq: u8, board: &Board, color: Color) -> bool {
        // Simplificado: verifica se nova casa está mais longe da borda
        let current_centrality = self.calculate_centrality(current_sq);
        let new_centrality = self.calculate_centrality(new_sq);
        
        new_centrality > current_centrality
    }

    /// Calcula score de vulnerabilidade
    fn calculate_vulnerability_score(&self, vulnerabilities: &[VulnerabilityInfo]) -> i32 {
        vulnerabilities.iter()
            .map(|v| v.impact_score)
            .sum()
    }

    /// Calcula score meta final
    fn calculate_meta_score(&self, result: &MetaEvaluationResult) -> i32 {
        let mut meta_score = 0;

        // Score baseado em vulnerabilidades
        meta_score += result.total_vulnerability_score;

        // Score baseado em avaliação estratégica
        let strategic_bonus = (result.strategic_assessment.overall_health * 100.0) as i32 - 50;
        meta_score += strategic_bonus;

        // Penaliza muitas vulnerabilidades críticas
        let critical_count = result.vulnerabilities.iter()
            .filter(|v| v.severity == VulnerabilitySeverity::Critical)
            .count() as i32;
        meta_score -= critical_count * 50;

        // Aplicação do peso de configuração
        meta_score = (meta_score as f32 * self.config.vulnerability_weight) as i32;

        meta_score
    }

    /// Utilitário: converte bitboard para vetor de casas
    fn get_set_bits(&self, bitboard: Bitboard) -> Vec<u8> {
        let mut squares = Vec::new();
        let mut bb = bitboard;
        
        while bb != 0 {
            let sq = bb.trailing_zeros() as u8;
            squares.push(sq);
            bb &= bb - 1;
        }
        
        squares
    }

    /// Utilitário: converte casa para nome (e.g., 0 -> "a1")
    fn square_name(&self, square: u8) -> String {
        let file = (square % 8) as u8 + b'a';
        let rank = (square / 8) + 1;
        format!("{}{}", file as char, rank)
    }
}

/// Interface pública para meta-avaliação
pub fn evaluate_position_vulnerabilities(board: &Board, color: Color) -> MetaEvaluationResult {
    let config = MetaEvaluationConfig::default();
    let evaluator = MetaEvaluator::new(config);
    evaluator.analyze_position(board, color)
}

/// Interface simplificada que retorna apenas o score COM CACHE
pub fn meta_evaluate_position(board: &Board, color: Color) -> i32 {
    use std::collections::HashMap;
    use std::sync::Mutex;
    
    lazy_static::lazy_static! {
        static ref META_CACHE: Mutex<HashMap<(u64, Color), i32>> = 
            Mutex::new(HashMap::with_capacity(5000));
    }
    
    let cache_key = (board.zobrist_hash, color);
    
    // Verifica cache primeiro
    if let Ok(cache) = META_CACHE.try_lock() {
        if let Some(&cached_result) = (*cache).get(&cache_key) {
            return cached_result;
        }
    }
    
    // Cálculo original completo (mantendo toda funcionalidade)
    let result = evaluate_position_vulnerabilities(board, color);
    let meta_score = result.meta_score;
    
    // Armazena no cache
    if let Ok(mut cache) = META_CACHE.try_lock() {
        if (*cache).len() >= 5000 {
            (*cache).clear(); // LRU simples: limpa quando cheio
        }
        (*cache).insert(cache_key, meta_score);
    }
    
    meta_score
}

/// Interface para análise rápida de vulnerabilidades críticas
pub fn detect_critical_vulnerabilities(board: &Board, color: Color) -> Vec<VulnerabilityInfo> {
    let result = evaluate_position_vulnerabilities(board, color);
    result.vulnerabilities.into_iter()
        .filter(|v| v.severity >= VulnerabilitySeverity::Serious)
        .collect()
}