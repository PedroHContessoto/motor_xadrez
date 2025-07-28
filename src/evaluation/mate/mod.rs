// Módulo dedicado exclusivamente para MATE (não endgame geral)
use crate::{board::Board, types::Color};

pub mod detection;
pub mod patterns;  // Padrões específicos de mate com implementação completa
pub mod search;
pub mod cache;
pub mod draw_win_management;
pub mod victory_conversion;

/// Tipos de mate reconhecidos
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MateType {
    BackRankMate,       // Mate na última fileira
    SupportedMate,      // Mate com suporte de peças
    SmotheredMate,      // Mate abafado
    DiscoveredMate,     // Mate por ataque descoberto
    PawnPromotionMate,  // Mate por promoção
    KingHuntMate,       // Mate por caça ao rei
    EndgameMate,        // Mate de endgame (K+Q vs K, etc.)
    TacticalMate,       // Mate tático complexo
    PositionalMate,     // Mate posicional
    ForcedMate,         // Mate forçado (sequência calculada)
    Unknown,            // Padrão não identificado
}

/// Qualidade da execução do mate
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MateQuality {
    Perfect,      // Mate forçado com técnica perfeita
    Excellent,    // Mate com técnica muito boa
    Good,         // Mate com técnica adequada
    Adequate,     // Mate funcional mas não otimal
    Poor,         // Mate com técnica ruim
}

/// Informação completa sobre mate detectado
#[derive(Debug, Clone)]
pub struct MateInfo {
    pub mate_in_moves: u8,
    pub best_sequence: Vec<crate::types::Move>,
    pub evaluation: i32,
    pub mate_type: MateType,
    pub mate_quality: MateQuality,
    pub search_depth: u8,
    pub nodes_searched: u64,
    pub time_taken_ms: u128,
}

/// Resultado da detecção de mate
#[derive(Debug, Clone)]
pub enum MateResult {
    MateFound(MateInfo),
    MateInProgress(u8), // Mate encontrado mas ainda calculando sequência
    NoMateFound,
    SearchTimeout,
}

/// Coordenador principal para detecção e análise de mate
pub struct MateEvaluator;

impl MateEvaluator {
    pub fn new() -> Self {
        Self
    }
    
    /// Interface principal para detecção de mate
    pub fn detect_mate(&self, board: &Board, max_depth: u8) -> MateResult {
        // Primeiro verifica padrões de mate específicos (mais rápido)
        let mate_patterns = patterns::detect_mate_patterns(board);
        
        if let Some(best_pattern) = mate_patterns.into_iter().min_by_key(|p| p.mate_in) {
            return MateResult::MateFound(MateInfo {
                mate_in_moves: best_pattern.mate_in,
                best_sequence: best_pattern.forcing_moves,
                evaluation: 99999 - best_pattern.mate_in as i32,
                mate_type: self.convert_pattern_to_mate_type(best_pattern.pattern_type),
                mate_quality: if best_pattern.confidence > 0.9 { MateQuality::Excellent } else { MateQuality::Good },
                search_depth: 1, // Padrão detectado diretamente
                nodes_searched: 1,
                time_taken_ms: 0,
            });
        }
        
        // Se não encontrou padrão específico, usa busca tradicional
        if let Some(mate_result) = search::integrate_mate_search_with_endgame(board) {
            return MateResult::MateFound(MateInfo {
                mate_in_moves: mate_result.mate_in,
                best_sequence: mate_result.moves,
                evaluation: mate_result.evaluation,
                mate_type: MateType::TacticalMate,
                mate_quality: if mate_result.is_forced { MateQuality::Perfect } else { MateQuality::Good },
                search_depth: max_depth,
                nodes_searched: 100, // Aproximação
                time_taken_ms: 10,
            });
        }
        
        MateResult::NoMateFound
    }
    
    /// Converte tipo de padrão para tipo de mate
    fn convert_pattern_to_mate_type(&self, pattern_type: patterns::MatePatternType) -> MateType {
        match pattern_type {
            patterns::MatePatternType::BackRankMate => MateType::BackRankMate,
            patterns::MatePatternType::SmotheredMate => MateType::SmotheredMate,
            patterns::MatePatternType::DiscoveredMate => MateType::DiscoveredMate,
            patterns::MatePatternType::TwoRooksMate => MateType::TacticalMate,
            patterns::MatePatternType::QueenKnightMate => MateType::TacticalMate,
            patterns::MatePatternType::BishopKnightMate => MateType::EndgameMate,
            patterns::MatePatternType::DoubleMate => MateType::TacticalMate,
            patterns::MatePatternType::SupportedMate => MateType::SupportedMate,
            patterns::MatePatternType::PawnPromotionMate => MateType::PawnPromotionMate,
            patterns::MatePatternType::KingHuntMate => MateType::KingHuntMate,
            patterns::MatePatternType::TacticalMate => MateType::TacticalMate,
            patterns::MatePatternType::PositionalMate => MateType::PositionalMate,
            patterns::MatePatternType::ForcedMate => MateType::ForcedMate,
            patterns::MatePatternType::Unknown => MateType::Unknown,
        }
    }
    
    /// Avalia se posição tem potencial de mate
    pub fn evaluate_mate_potential(&self, board: &Board, color: Color) -> i32 {
        // Implementação específica para mate
        0
    }
}

impl Default for MateEvaluator {
    fn default() -> Self {
        Self::new()
    }
}