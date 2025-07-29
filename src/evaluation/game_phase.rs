// Detecção de fase do jogo com análise avançada
use crate::board::Board;
use crate::types::{PieceKind, Color};

/// Fases do jogo
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GamePhase {
    Opening,
    Middlegame,
    Endgame,
}

impl GamePhase {
    /// Retorna descrição textual da fase
    pub fn description(&self) -> &'static str {
        match self {
            GamePhase::Opening => "Opening",
            GamePhase::Middlegame => "Middlegame",
            GamePhase::Endgame => "Endgame",
        }
    }

    /// Verifica se é fase inicial (opening/middlegame)
    pub fn is_early_game(&self) -> bool {
        matches!(self, GamePhase::Opening | GamePhase::Middlegame)
    }

    /// Verifica se é endgame
    pub fn is_endgame(&self) -> bool {
        matches!(self, GamePhase::Endgame)
    }
}

/// Estrutura para informações avançadas da fase do jogo
#[derive(Debug, Clone, Copy)]
pub struct GamePhaseInfo {
    pub phase: GamePhase,
    pub material_count: u32,
    pub phase_value: f32,
    pub piece_count: u32,
    pub pawn_count: u32,
    pub minor_count: u32,
    pub major_count: u32,
    pub queen_count: u32,
    pub opening_score: i32,
    pub endgame_score: i32,
    pub complexity: PositionComplexity,
}

impl GamePhaseInfo {
    /// Cria nova estrutura com valores padrão
    pub fn new() -> Self {
        GamePhaseInfo {
            phase: GamePhase::Opening,
            material_count: 32,
            phase_value: 0.0,
            piece_count: 32,
            pawn_count: 16,
            minor_count: 8,
            major_count: 4,
            queen_count: 2,
            opening_score: 24,
            endgame_score: 0,
            complexity: PositionComplexity::Normal,
        }
    }

    /// Interpola valor entre abertura e final baseado na fase
    pub fn interpolate(&self, opening_value: i32, endgame_value: i32) -> i32 {
        interpolate_phase_i32(opening_value, endgame_value, self)
    }

    /// Retorna peso para abertura (0.0 a 1.0)
    pub fn opening_weight(&self) -> f32 {
        1.0 - self.phase_value
    }

    /// Retorna peso para final (0.0 a 1.0)
    pub fn endgame_weight(&self) -> f32 {
        self.phase_value
    }

    /// Verifica se está em transição entre fases
    pub fn is_transitioning(&self) -> bool {
        self.phase_value > 0.2 && self.phase_value < 0.8
    }
}

impl Default for GamePhaseInfo {
    fn default() -> Self {
        Self::new()
    }
}

/// Complexidade da posição
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionComplexity {
    VerySimple,  // Finais básicos (KPK, KRK, etc)
    Simple,      // Poucos tipos de peças
    Normal,      // Posição típica
    Complex,     // Muitas peças e possibilidades
    VeryComplex, // Posição muito tática/dinâmica
}

impl PositionComplexity {
    /// Retorna fator de complexidade (0.0 a 1.0)
    pub fn factor(&self) -> f32 {
        match self {
            PositionComplexity::VerySimple => 0.0,
            PositionComplexity::Simple => 0.25,
            PositionComplexity::Normal => 0.5,
            PositionComplexity::Complex => 0.75,
            PositionComplexity::VeryComplex => 1.0,
        }
    }
}

// Constantes para cálculo de fase
const PAWN_PHASE_WEIGHT: i32 = 0;
const KNIGHT_PHASE_WEIGHT: i32 = 1;
const BISHOP_PHASE_WEIGHT: i32 = 1;
const ROOK_PHASE_WEIGHT: i32 = 2;
const QUEEN_PHASE_WEIGHT: i32 = 4;

const TOTAL_PHASE_WEIGHT: i32 = 16 * PAWN_PHASE_WEIGHT +
    4 * KNIGHT_PHASE_WEIGHT +
    4 * BISHOP_PHASE_WEIGHT +
    4 * ROOK_PHASE_WEIGHT +
    2 * QUEEN_PHASE_WEIGHT;

/// Detecta fase do jogo (versão simples para compatibilidade)
pub fn detect_game_phase(board: &Board) -> GamePhase {
    let info = detect_game_phase_advanced(board);
    info.phase
}

/// Versão avançada de detecção de fase do jogo
pub fn detect_game_phase_advanced(board: &Board) -> GamePhaseInfo {
    let mut info = GamePhaseInfo::new();

    // Conta peças
    count_pieces(board, &mut info);

    // Calcula score de fase baseado em material
    calculate_phase_scores(&mut info);

    // Determina fase e valor de interpolação
    determine_phase(&mut info);

    // Analisa complexidade da posição
    analyze_complexity(board, &mut info);

    info
}

/// Conta todas as peças no tabuleiro
fn count_pieces(board: &Board, info: &mut GamePhaseInfo) {
    info.piece_count = (board.white_pieces | board.black_pieces).count_ones();
    info.material_count = info.piece_count;

    info.pawn_count = board.pawns.count_ones();
    info.minor_count = (board.knights | board.bishops).count_ones();
    info.major_count = board.rooks.count_ones();
    info.queen_count = board.queens.count_ones();
}

/// Calcula scores de fase baseados no material
fn calculate_phase_scores(info: &mut GamePhaseInfo) {
    // Score de abertura baseado em peças não-peões
    info.opening_score = (info.minor_count as i32) * KNIGHT_PHASE_WEIGHT +
        (info.major_count as i32) * ROOK_PHASE_WEIGHT +
        (info.queen_count as i32) * QUEEN_PHASE_WEIGHT;

    // Score de endgame (inverso do opening)
    info.endgame_score = TOTAL_PHASE_WEIGHT - info.opening_score;
}

/// Determina a fase atual e valor de interpolação
fn determine_phase(info: &mut GamePhaseInfo) {
    // Calcula valor de fase (0.0 = opening, 1.0 = endgame)
    info.phase_value = info.endgame_score as f32 / TOTAL_PHASE_WEIGHT as f32;

    // Determina fase discreta com histerese
    if info.opening_score > 20 {
        info.phase = GamePhase::Opening;
    } else if info.opening_score > 10 {
        info.phase = GamePhase::Middlegame;
    } else {
        info.phase = GamePhase::Endgame;
    }

    // Ajustes especiais
    apply_special_phase_rules(info);
}

/// Aplica regras especiais para determinação de fase
fn apply_special_phase_rules(info: &mut GamePhaseInfo) {
    // Se não há rainhas, tendemos mais para endgame
    if info.queen_count == 0 {
        info.phase_value = (info.phase_value + 0.2).min(1.0);

        if info.opening_score <= 12 {
            info.phase = GamePhase::Endgame;
        }
    }

    // Com muitos peões e poucas peças, é endgame
    if info.piece_count <= 14 && info.pawn_count >= 8 {
        info.phase = GamePhase::Endgame;
        info.phase_value = (info.phase_value + 0.1).min(1.0);
    }

    // Posições muito simplificadas
    if info.piece_count <= 8 {
        info.phase = GamePhase::Endgame;
        info.phase_value = 1.0;
    }
}

/// Analisa complexidade da posição
fn analyze_complexity(board: &Board, info: &mut GamePhaseInfo) {
    let mut complexity_score = 0;

    // Fatores que aumentam complexidade
    if info.queen_count > 0 {
        complexity_score += 20;
    }

    // Desequilíbrio material
    let material_imbalance = calculate_material_imbalance(board);
    if material_imbalance > 200 {
        complexity_score += 15;
    }

    // Estrutura de peões
    let pawn_tension = calculate_pawn_tension(board);
    complexity_score += pawn_tension * 5;

    // Peças ativas
    let piece_activity = estimate_piece_activity(board);
    complexity_score += piece_activity;

    // Posições táticas (checks, peças penduradas)
    if has_tactical_features(board) {
        complexity_score += 25;
    }

    // Determina complexidade baseado no score
    info.complexity = match complexity_score {
        0..=10 => PositionComplexity::VerySimple,
        11..=25 => PositionComplexity::Simple,
        26..=50 => PositionComplexity::Normal,
        51..=75 => PositionComplexity::Complex,
        _ => PositionComplexity::VeryComplex,
    };
}

/// Calcula desequilíbrio material entre os lados
fn calculate_material_imbalance(board: &Board) -> i32 {
    let white_material = calculate_material_value(board, Color::White);
    let black_material = calculate_material_value(board, Color::Black);
    (white_material - black_material).abs()
}

/// Calcula valor material de um lado
fn calculate_material_value(board: &Board, color: Color) -> i32 {
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };

    let pawns = (board.pawns & pieces).count_ones() as i32 * 100;
    let knights = (board.knights & pieces).count_ones() as i32 * 320;
    let bishops = (board.bishops & pieces).count_ones() as i32 * 330;
    let rooks = (board.rooks & pieces).count_ones() as i32 * 500;
    let queens = (board.queens & pieces).count_ones() as i32 * 900;

    pawns + knights + bishops + rooks + queens
}

/// Calcula tensão entre peões (possíveis capturas)
fn calculate_pawn_tension(board: &Board) -> i32 {
    let white_pawns = board.pawns & board.white_pieces;
    let black_pawns = board.pawns & board.black_pieces;

    // Ataques de peões brancos
    let white_attacks_left = ((white_pawns & 0xFEFEFEFEFEFEFEFE) << 7) & black_pawns;
    let white_attacks_right = ((white_pawns & 0x7F7F7F7F7F7F7F7F) << 9) & black_pawns;

    // Ataques de peões pretos
    let black_attacks_left = ((black_pawns & 0x7F7F7F7F7F7F7F7F) >> 7) & white_pawns;
    let black_attacks_right = ((black_pawns & 0xFEFEFEFEFEFEFEFE) >> 9) & white_pawns;

    let total_tension = white_attacks_left.count_ones() +
        white_attacks_right.count_ones() +
        black_attacks_left.count_ones() +
        black_attacks_right.count_ones();

    total_tension as i32
}

/// Estima atividade das peças (heurística simples)
fn estimate_piece_activity(board: &Board) -> i32 {
    let mut activity = 0;

    // Cavalos centralizados
    let center_squares = 0x00003C3C3C3C0000u64;
    let central_knights = (board.knights & center_squares).count_ones() as i32;
    activity += central_knights * 5;

    // Torres em colunas abertas (aproximação)
    let rooks_on_7th = (board.rooks & 0x00FF000000000000u64).count_ones() +
        (board.rooks & 0x000000000000FF00u64).count_ones();
    activity += rooks_on_7th as i32 * 10;

    activity
}

/// Verifica se há características táticas na posição
fn has_tactical_features(board: &Board) -> bool {
    // Check
    if board.is_king_in_check(board.to_move) {
        return true;
    }

    // Peças sem defesa (verificação simplificada)
    for color in [Color::White, Color::Black] {
        let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
        let valuable = (board.knights | board.bishops | board.rooks | board.queens) & pieces;

        // Se há peças valiosas, assume que pode haver táticas
        if valuable.count_ones() >= 3 {
            return true;
        }
    }

    false
}

/// Interpola valores baseado na fase do jogo
pub fn interpolate_phase_i32(opening: i32, endgame: i32, phase_info: &GamePhaseInfo) -> i32 {
    let opening_weight = phase_info.opening_weight();
    let endgame_weight = phase_info.endgame_weight();

    ((opening as f32 * opening_weight) + (endgame as f32 * endgame_weight)) as i32
}

/// Interpola valores f32 baseado na fase
pub fn interpolate_phase_f32(opening: f32, endgame: f32, phase_info: &GamePhaseInfo) -> f32 {
    let opening_weight = phase_info.opening_weight();
    let endgame_weight = phase_info.endgame_weight();

    (opening * opening_weight) + (endgame * endgame_weight)
}

/// Detecta transições específicas de fase
pub fn detect_phase_transition(board: &Board) -> PhaseTransition {
    let info = detect_game_phase_advanced(board);

    match (info.phase, info.is_transitioning()) {
        (GamePhase::Opening, true) => PhaseTransition::OpeningToMiddlegame,
        (GamePhase::Middlegame, true) if info.phase_value > 0.5 => PhaseTransition::MiddlegameToEndgame,
        (GamePhase::Middlegame, true) => PhaseTransition::EarlyMiddlegame,
        (GamePhase::Endgame, _) if info.piece_count <= 10 => PhaseTransition::LateEndgame,
        _ => PhaseTransition::Stable,
    }
}

/// Tipos de transição entre fases
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhaseTransition {
    Stable,
    OpeningToMiddlegame,
    EarlyMiddlegame,
    MiddlegameToEndgame,
    LateEndgame,
}

impl PhaseTransition {
    /// Verifica se está em transição
    pub fn is_transitioning(&self) -> bool {
        !matches!(self, PhaseTransition::Stable)
    }

    /// Retorna fator de ajuste para avaliação durante transições
    pub fn adjustment_factor(&self) -> f32 {
        match self {
            PhaseTransition::Stable => 1.0,
            PhaseTransition::OpeningToMiddlegame => 0.9,
            PhaseTransition::EarlyMiddlegame => 0.95,
            PhaseTransition::MiddlegameToEndgame => 0.9,
            PhaseTransition::LateEndgame => 1.1,
        }
    }
}

/// Detecta tipo específico de endgame
pub fn detect_endgame_type(board: &Board) -> Option<EndgameType> {
    let info = detect_game_phase_advanced(board);

    if !info.phase.is_endgame() {
        return None;
    }

    let white_pieces = board.white_pieces;
    let black_pieces = board.black_pieces;

    // Contagem detalhada
    let white_pawns = (board.pawns & white_pieces).count_ones();
    let black_pawns = (board.pawns & black_pieces).count_ones();
    let white_knights = (board.knights & white_pieces).count_ones();
    let black_knights = (board.knights & black_pieces).count_ones();
    let white_bishops = (board.bishops & white_pieces).count_ones();
    let black_bishops = (board.bishops & black_pieces).count_ones();
    let white_rooks = (board.rooks & white_pieces).count_ones();
    let black_rooks = (board.rooks & black_pieces).count_ones();
    let white_queens = (board.queens & white_pieces).count_ones();
    let black_queens = (board.queens & black_pieces).count_ones();

    // Detecta tipos específicos
    match info.piece_count {
        2 => Some(EndgameType::KingVsKing),
        3 => {
            if white_pawns + black_pawns == 1 {
                Some(EndgameType::KPK)
            } else if white_rooks + black_rooks == 1 {
                Some(EndgameType::KRK)
            } else if white_queens + black_queens == 1 {
                Some(EndgameType::KQK)
            } else {
                Some(EndgameType::MinorPiece)
            }
        },
        4..=6 => {
            if white_rooks == 1 && black_rooks == 1 && white_pawns + black_pawns > 0 {
                Some(EndgameType::RookEndgame)
            } else if white_bishops == 1 && black_bishops == 1 {
                if is_opposite_colored_bishops(board) {
                    Some(EndgameType::OppositeColoredBishops)
                } else {
                    Some(EndgameType::SameColoredBishops)
                }
            } else if white_queens == 1 && black_queens == 1 {
                Some(EndgameType::QueenEndgame)
            } else {
                Some(EndgameType::MinorPiece)
            }
        },
        _ => {
            if white_pawns + black_pawns >= 6 {
                Some(EndgameType::PawnEndgame)
            } else {
                Some(EndgameType::General)
            }
        }
    }
}

/// Tipos específicos de endgame
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndgameType {
    KingVsKing,
    KPK,  // King and Pawn vs King
    KRK,  // King and Rook vs King
    KQK,  // King and Queen vs King
    RookEndgame,
    PawnEndgame,
    MinorPiece,
    OppositeColoredBishops,
    SameColoredBishops,
    QueenEndgame,
    General,
}

impl EndgameType {
    /// Retorna importância de conhecimento específico do endgame
    pub fn specificity_bonus(&self) -> i32 {
        match self {
            EndgameType::KPK => 50,
            EndgameType::KRK => 40,
            EndgameType::KQK => 30,
            EndgameType::OppositeColoredBishops => 35,
            EndgameType::RookEndgame => 25,
            EndgameType::PawnEndgame => 20,
            _ => 0,
        }
    }
}

/// Verifica se há bispos de cores opostas
fn is_opposite_colored_bishops(board: &Board) -> bool {
    let white_bishops = board.bishops & board.white_pieces;
    let black_bishops = board.bishops & board.black_pieces;

    if white_bishops.count_ones() != 1 || black_bishops.count_ones() != 1 {
        return false;
    }

    let white_bishop_sq = white_bishops.trailing_zeros() as u8;
    let black_bishop_sq = black_bishops.trailing_zeros() as u8;

    let white_on_light = (white_bishop_sq / 8 + white_bishop_sq % 8) % 2 == 0;
    let black_on_light = (black_bishop_sq / 8 + black_bishop_sq % 8) % 2 == 0;

    white_on_light != black_on_light
}
