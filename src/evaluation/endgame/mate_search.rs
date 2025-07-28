// Consolidated mate search functionality moved from victory_conversion.rs
use crate::{
    board::Board, 
    types::{Color, Move},
    evaluation,
};
use std::collections::HashMap;

/// Sequência de mate detectada
#[derive(Debug, Clone)]
pub struct MateSequence {
    pub moves: Vec<Move>,
    pub mate_in: u8,
    pub evaluation: i32,
    pub is_forced: bool,
}

/// Sistema de busca rápida de mate para conversão de vitória
pub struct QuickMateSearcher {
    max_depth: u8,
}

impl QuickMateSearcher {
    pub fn new() -> Self {
        Self {
            max_depth: 5, // Conservative depth to avoid false positives
        }
    }

    /// Sistema de detecção rápida de mate otimizado
    pub fn quick_mate_detection(&self, board: &Board, depth: u8) -> Option<MateSequence> {
        // Critérios rigorosos para buscar mate - evita falsos positivos
        let evaluation = evaluation::evaluate_with_depth(board, 0);
        let total_pieces = (board.white_pieces | board.black_pieces).count_ones();
        let opponent_in_check = board.is_king_in_check(!board.to_move);
        
        // Só busca mate se:
        // 1. Vantagem massive (>1500cp) OU
        // 2. Oponente já em xeque OU  
        // 3. Muito poucas peças no tabuleiro (<8)
        if evaluation.abs() < 1500 && !opponent_in_check && total_pieces > 8 {
            return None;
        }
        
        // Limita profundidade para busca segura
        let safe_depth = depth.min(self.max_depth).min(3);
        
        // Busca mate em profundidades crescentes
        for search_depth in 1..=safe_depth {
            if let Some(sequence) = self.search_mate_at_depth(board, search_depth) {
                if self.validate_mate_sequence(board, &sequence) {
                    return Some(sequence);
                }
            }
        }
        
        None
    }

    /// Busca mate numa profundidade específica
    fn search_mate_at_depth(&self, board: &Board, depth: u8) -> Option<MateSequence> {
        if depth == 0 {
            return None;
        }

        let moves = board.generate_legal_moves();
        if moves.is_empty() {
            return None; // Stalemate ou mate (mas não é nossa vez)
        }

        // Ordena movimentos por prioridade para mate
        let mut ordered_moves = moves;
        ordered_moves.sort_by(|a, b| {
            let score_a = self.evaluate_move_for_mate(board, *a);
            let score_b = self.evaluate_move_for_mate(board, *b);
            score_b.cmp(&score_a)
        });

        for mv in ordered_moves {
            let mut test_board = *board;
            let _undo = test_board.make_move_fast(mv);

            if depth == 1 {
                // Verifica se é mate imediato
                if test_board.is_checkmate() {
                    return Some(MateSequence {
                        moves: vec![mv],
                        mate_in: 1,
                        evaluation: 99999 - 1, // MATE_VALUE - distance
                        is_forced: true,
                    });
                }
            } else {
                // Busca recursiva
                if let Some(mut continuation) = self.search_mate_at_depth(&test_board, depth - 1) {
                    // Verifica se o oponente tem apenas uma resposta (forçado)
                    let opponent_moves = test_board.generate_legal_moves();
                    if opponent_moves.len() <= 2 || test_board.is_king_in_check(test_board.to_move) {
                        let mut sequence = vec![mv];
                        sequence.extend(continuation.moves);
                        
                        return Some(MateSequence {
                            moves: sequence,
                            mate_in: depth,
                            evaluation: 99999 - depth as i32,
                            is_forced: opponent_moves.len() <= 1,
                        });
                    }
                }
            }
        }

        None
    }

    /// Avalia movimento para busca de mate
    fn evaluate_move_for_mate(&self, board: &Board, mv: Move) -> i32 {
        let mut score = 0;
        
        // Prioriza xeques
        let mut test_board = *board;
        let _undo = test_board.make_move_fast(mv);
        if test_board.is_king_in_check(!board.to_move) {
            score += 1000;
        }

        // Prioriza capturas
        if board.is_capture(mv) {
            score += 500;
        }

        // Prioriza promoções
        if mv.promotion.is_some() {
            score += 300;
        }

        // Prioriza movimentos que restringem o rei inimigo
        score += self.evaluate_king_restriction(&test_board, !board.to_move);

        score
    }

    /// Avalia restrição do rei inimigo
    fn evaluate_king_restriction(&self, board: &Board, king_color: Color) -> i32 {
        let king_pos = board.kings & if king_color == Color::White { 
            board.white_pieces 
        } else { 
            board.black_pieces 
        };

        if king_pos == 0 { return 0; }

        let king_sq = king_pos.trailing_zeros() as u8;
        let king_moves = crate::moves::king::get_king_attacks_lookup(king_sq);
        
        // Aproximação: conta casas ocupadas ou atacadas
        let blocked_squares = king_moves & (board.white_pieces | board.black_pieces);
        
        // Menos casas disponíveis = mais restrição
        8 - (king_moves & !blocked_squares).count_ones() as i32
    }

    /// Valida se uma sequência de movimentos é realmente um mate forçado
    fn validate_mate_sequence(&self, board: &Board, sequence: &MateSequence) -> bool {
        if sequence.moves.is_empty() {
            return false;
        }
        
        let mut test_board = *board;
        
        // Testa a sequência completa
        for (i, &mv) in sequence.moves.iter().enumerate() {
            // Verifica se o movimento é legal
            let legal_moves = test_board.generate_legal_moves();
            if !legal_moves.contains(&mv) {
                return false;
            }
            
            // Executa o movimento
            let _undo = test_board.make_move_fast(mv);
            
            // Se é o último movimento, deve ser mate
            if i == sequence.moves.len() - 1 {
                return test_board.is_checkmate();
            }
            
            // Se não é o último movimento, verifica se é forçado
            if !test_board.is_king_in_check(test_board.to_move) {
                // Se não dá xeque, verifica se há poucas opções válidas
                let responses = test_board.generate_legal_moves();
                if responses.len() > 3 {
                    return false; // Muitas opções = não forçado
                }
            }
        }
        
        false
    }
}

impl Default for QuickMateSearcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Integração com o sistema de avaliação de endgame
pub fn integrate_mate_search_with_endgame(board: &Board) -> Option<MateSequence> {
    // Primeiro tenta endgames teóricos
    if let Some(endgame_result) = super::mate_evaluators::evaluate_theoretical_endgame(board) {
        if let Some(mate_distance) = endgame_result.estimated_moves_to_mate {
            // Se o endgame teórico prevê mate próximo, tenta encontrar a sequência
            if mate_distance <= 5 {
                let searcher = QuickMateSearcher::new();
                return searcher.quick_mate_detection(board, mate_distance.min(3));
            }
        }
    }
    
    None
}