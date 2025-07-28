// Módulo coordenador principal para avaliação de finais
use crate::{board::Board, types::Color};

pub mod theoretical;
pub mod practical;
pub mod patterns;
pub mod tablebase;
pub mod mate_evaluators;
pub mod mate_search;

use theoretical::*;
use practical::*;
use patterns::*;

/// Tipos de finais reconhecidos
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EndgameType {
    // Finais teóricos básicos
    KQvsK,
    KRvsK,
    KPvsK,
    KBBvsK,
    KBNvsK,
    KNNvsK,
    
    // Finais de torres
    KRPvsKR,
    KRvsKR,
    
    // Finais de bispos
    KBPvsKB,
    KBvsKB,
    
    // Finais de cavalos
    KNPvsKN,
    KNvsKN,
    
    // Finais de peões
    KPvsKP,
    KPPvsKP,
    
    // Finais práticos
    QueenEndgame,
    RookEndgame,
    MinorPieceEndgame,
    PawnEndgame,
    
    // Não reconhecido
    Unknown,
}

/// Fases detalhadas de jogo conforme proposta da análise
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DetailedGamePhase {
    Opening,
    EarlyMiddlegame,
    Middlegame,
    LateMiddlegame,
    EarlyEndgame,     // 12-16 peças
    Endgame,          // 8-12 peças  
    LateEndgame,      // 5-8 peças
    PureEndgame,      // <5 peças
    TheoreticalEndgame // Posições de tablebase
}

/// Resultado da avaliação de final
#[derive(Debug, Clone)]
pub struct EndgameEvaluation {
    pub score: i32,
    pub endgame_type: EndgameType,
    pub phase: DetailedGamePhase,
    pub is_theoretical: bool,
    pub mate_distance: Option<u8>,
    pub key_concepts: Vec<String>,
}

/// Coordenador principal da avaliação de finais
pub struct EndgameEvaluator {
    pub theoretical: TheoreticalEvaluator,
    pub practical: PracticalEvaluator,
    pub patterns: MatePatternDetector, // TODO: Implementar quando necessário
}

impl EndgameEvaluator {
    pub fn new() -> Self {
        Self {
            theoretical: TheoreticalEvaluator::new(),
            practical: PracticalEvaluator::new(),
            patterns: MatePatternDetector::new(), // TODO: Implementar quando necessário
        }
    }
    
    /// Avaliação principal que coordena todos os módulos
    pub fn evaluate(&self, board: &Board) -> Option<EndgameEvaluation> {
        let phase = self.detect_game_phase(board);
        let endgame_type = self.classify_endgame(board);
        
        // Primeiro tenta avaliação teórica exata
        if let Some(theoretical_eval) = self.theoretical.evaluate(board, endgame_type) {
            return Some(EndgameEvaluation {
                score: theoretical_eval.score,
                endgame_type,
                phase,
                is_theoretical: true,
                mate_distance: theoretical_eval.mate_distance,
                key_concepts: theoretical_eval.concepts,
            });
        }
        
        // Depois tenta padrões de mate
        if let Some(mate_pattern) = self.patterns.detect_mate_pattern(board) {
            return Some(EndgameEvaluation {
                score: mate_pattern.evaluation,
                endgame_type,
                phase,
                is_theoretical: false,
                mate_distance: Some(mate_pattern.mate_in),
                key_concepts: vec![format!("{:?}", mate_pattern.pattern_type)],
            });
        }
        
        // Por último usa avaliação prática
        if matches!(phase, DetailedGamePhase::LateEndgame | DetailedGamePhase::PureEndgame | DetailedGamePhase::TheoreticalEndgame) {
            let practical_eval = self.practical.evaluate(board, endgame_type, phase);
            return Some(EndgameEvaluation {
                score: practical_eval.score,
                endgame_type,
                phase,
                is_theoretical: false,
                mate_distance: None,
                key_concepts: practical_eval.concepts,
            });
        }
        
        None
    }
    
    /// Detecta fase detalhada do jogo
    fn detect_game_phase(&self, board: &Board) -> DetailedGamePhase {
        let total_pieces = (board.white_pieces | board.black_pieces).count_ones();
        let total_pawns = board.pawns.count_ones();
        let queens = board.queens.count_ones();
        let rooks = board.rooks.count_ones();
        let minors = (board.bishops | board.knights).count_ones();
        
        match total_pieces {
            0..=4 => {
                // Verifica se é posição teórica conhecida
                if self.is_theoretical_position(board) {
                    DetailedGamePhase::TheoreticalEndgame
                } else {
                    DetailedGamePhase::PureEndgame
                }
            },
            5..=8 => DetailedGamePhase::LateEndgame,
            9..=12 => DetailedGamePhase::Endgame,
            13..=16 => DetailedGamePhase::EarlyEndgame,
            17..=24 => {
                if queens >= 2 || (queens >= 1 && rooks >= 2) {
                    DetailedGamePhase::LateMiddlegame
                } else {
                    DetailedGamePhase::EarlyEndgame
                }
            },
            25..=28 => {
                if total_pawns <= 4 {
                    DetailedGamePhase::LateMiddlegame
                } else {
                    DetailedGamePhase::Middlegame
                }
            },
            29..=30 => DetailedGamePhase::EarlyMiddlegame,
            _ => DetailedGamePhase::Opening,
        }
    }
    
    /// Classifica o tipo de final
    fn classify_endgame(&self, board: &Board) -> EndgameType {
        let white_pieces = board.white_pieces;
        let black_pieces = board.black_pieces;
        
        // Conta peças por tipo
        let white_queens = (board.queens & white_pieces).count_ones();
        let black_queens = (board.queens & black_pieces).count_ones();
        let white_rooks = (board.rooks & white_pieces).count_ones();
        let black_rooks = (board.rooks & black_pieces).count_ones();
        let white_bishops = (board.bishops & white_pieces).count_ones();
        let black_bishops = (board.bishops & black_pieces).count_ones();
        let white_knights = (board.knights & white_pieces).count_ones();
        let black_knights = (board.knights & black_pieces).count_ones();
        let white_pawns = (board.pawns & white_pieces).count_ones();
        let black_pawns = (board.pawns & black_pieces).count_ones();
        
        // Remove reis para contagem de material
        let white_material = white_pieces & !board.kings;
        let black_material = black_pieces & !board.kings;
        let white_material_count = white_material.count_ones();
        let black_material_count = black_material.count_ones();
        
        // Finais teóricos básicos
        if white_material_count == 1 && black_material_count == 0 {
            if white_queens == 1 { return EndgameType::KQvsK; }
            if white_rooks == 1 { return EndgameType::KRvsK; }
            if white_pawns == 1 { return EndgameType::KPvsK; }
        }
        
        if black_material_count == 1 && white_material_count == 0 {
            if black_queens == 1 { return EndgameType::KQvsK; }
            if black_rooks == 1 { return EndgameType::KRvsK; }
            if black_pawns == 1 { return EndgameType::KPvsK; }
        }
        
        if white_material_count == 2 && black_material_count == 0 {
            if white_bishops == 2 { return EndgameType::KBBvsK; }
            if white_knights == 2 { return EndgameType::KNNvsK; }
            if white_bishops == 1 && white_knights == 1 { return EndgameType::KBNvsK; }
        }
        
        if black_material_count == 2 && white_material_count == 0 {
            if black_bishops == 2 { return EndgameType::KBBvsK; }
            if black_knights == 2 { return EndgameType::KNNvsK; }
            if black_bishops == 1 && black_knights == 1 { return EndgameType::KBNvsK; }
        }
        
        // Finais práticos por categoria principal
        if white_queens > 0 || black_queens > 0 {
            return EndgameType::QueenEndgame;
        }
        
        if white_rooks > 0 || black_rooks > 0 {
            return EndgameType::RookEndgame;
        }
        
        if white_bishops > 0 || black_bishops > 0 || white_knights > 0 || black_knights > 0 {
            return EndgameType::MinorPieceEndgame;
        }
        
        if white_pawns > 0 || black_pawns > 0 {
            return EndgameType::PawnEndgame;
        }
        
        EndgameType::Unknown
    }
    
    /// Verifica se é posição teórica conhecida
    fn is_theoretical_position(&self, board: &Board) -> bool {
        let total_pieces = (board.white_pieces | board.black_pieces).count_ones();
        if total_pieces > 5 { return false; }
        
        let endgame_type = self.classify_endgame(board);
        matches!(endgame_type, 
            EndgameType::KQvsK | EndgameType::KRvsK | EndgameType::KPvsK |
            EndgameType::KBBvsK | EndgameType::KBNvsK | EndgameType::KNNvsK
        )
    }
}

impl Default for EndgameEvaluator {
    fn default() -> Self {
        Self::new()
    }
}