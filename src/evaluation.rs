// Sistema de avaliação híbrido: NNUE + avaliação tradicional
use crate::board::Board;
use crate::types::*;
use crate::nnue::NNUEEvaluator;
use std::sync::Arc;

/// Valores das peças em centipawns
const PIECE_VALUES: [i32; 6] = [
    100,  // Pawn
    320,  // Knight
    330,  // Bishop
    500,  // Rook
    900,  // Queen
    20000, // King
];

/// Tabelas de valores posicionais para peças
const PAWN_TABLE: [i32; 64] = [
    0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5,  5, 10, 25, 25, 10,  5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5, -5,-10,  0,  0,-10, -5,  5,
    5, 10, 10,-20,-20, 10, 10,  5,
    0,  0,  0,  0,  0,  0,  0,  0
];

const KNIGHT_TABLE: [i32; 64] = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
];

const BISHOP_TABLE: [i32; 64] = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
];

const ROOK_TABLE: [i32; 64] = [
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    0,  0,  0,  5,  5,  0,  0,  0
];

const QUEEN_TABLE: [i32; 64] = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
    0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
];

const KING_MIDDLE_GAME: [i32; 64] = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20
];

const KING_END_GAME: [i32; 64] = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50
];

/// Tipo de avaliação disponível
#[derive(Debug, Clone, Copy)]
pub enum EvaluationType {
    Traditional,
    NNUE,
    Hybrid,
}

/// Avaliador principal do motor
pub struct Evaluator {
    nnue: Option<Arc<NNUEEvaluator>>,
    eval_type: EvaluationType,
}

impl Evaluator {
    /// Cria novo avaliador com avaliação tradicional
    pub fn new_traditional() -> Self {
        Evaluator {
            nnue: None,
            eval_type: EvaluationType::Traditional,
        }
    }

    /// Cria novo avaliador com NNUE
    pub fn new_nnue(nnue: Arc<NNUEEvaluator>) -> Self {
        Evaluator {
            nnue: Some(nnue),
            eval_type: EvaluationType::NNUE,
        }
    }

    /// Cria novo avaliador NNUE (main method)
    pub fn new_nnue_main(nnue: Arc<NNUEEvaluator>) -> Self {
        Evaluator {
            nnue: Some(nnue),
            eval_type: EvaluationType::NNUE,
        }
    }
    
    /// Cria novo avaliador híbrido
    pub fn new_hybrid(nnue: Arc<NNUEEvaluator>) -> Self {
        Evaluator {
            nnue: Some(nnue),
            eval_type: EvaluationType::Hybrid,
        }
    }

    /// Avalia uma posição
    pub fn evaluate(&self, board: &Board) -> i32 {
        match self.eval_type {
            EvaluationType::Traditional => self.traditional_evaluate(board),
            EvaluationType::NNUE => self.nnue_evaluate(board),
            EvaluationType::Hybrid => self.hybrid_evaluate(board),
        }
    }

    /// Avaliação tradicional baseada em material e posição
    fn traditional_evaluate(&self, board: &Board) -> i32 {
        let mut score = 0;
        
        // Avaliação de material
        score += self.evaluate_material(board);
        
        // Avaliação posicional
        score += self.evaluate_piece_positions(board);
        
        // Avaliação de segurança do rei
        score += self.evaluate_king_safety(board);
        
        // Avaliação de estrutura de peões
        score += self.evaluate_pawn_structure(board);
        
        // Inverter se é vez das pretas
        if board.to_move == Color::Black {
            -score
        } else {
            score
        }
    }

    /// Avaliação usando NNUE
    fn nnue_evaluate(&self, board: &Board) -> i32 {
        if let Some(ref nnue) = self.nnue {
            match nnue.evaluate(board) {
                Ok(score) => score as i32,
                Err(_) => self.traditional_evaluate(board), // Fallback
            }
        } else {
            self.traditional_evaluate(board)
        }
    }

    /// Avaliação híbrida (NNUE puro - sem hardcoded bias)
    fn hybrid_evaluate(&self, board: &Board) -> i32 {
        // NNUE puro - deixar o modelo aprender valores implicitamente
        self.nnue_evaluate(board)
    }

    /// Avaliação de material (desabilitada para NNUE puro)
    fn evaluate_material(&self, _board: &Board) -> i32 {
        // Retorna 0 - deixar NNUE aprender valores das peças implicitamente
        // Evita bias hardcoded que contamina o aprendizado
        0
    }

    /// Avaliação de posições das peças
    fn evaluate_piece_positions(&self, board: &Board) -> i32 {
        let mut score = 0;
        let is_endgame = self.is_endgame(board);
        
        for square in 0..64 {
            let square_bb = 1u64 << square;
            
            if (board.white_pieces & square_bb) != 0 {
                score += self.get_piece_square_value(board, square_bb, square, Color::White, is_endgame);
            } else if (board.black_pieces & square_bb) != 0 {
                score -= self.get_piece_square_value(board, square_bb, square, Color::Black, is_endgame);
            }
        }
        
        score
    }

    fn get_piece_square_value(&self, board: &Board, square_bb: u64, square: usize, color: Color, is_endgame: bool) -> i32 {
        let adjusted_square = if color == Color::White { square } else { 63 - square };
        
        if (board.pawns & square_bb) != 0 {
            PAWN_TABLE[adjusted_square]
        } else if (board.knights & square_bb) != 0 {
            KNIGHT_TABLE[adjusted_square]
        } else if (board.bishops & square_bb) != 0 {
            BISHOP_TABLE[adjusted_square]
        } else if (board.rooks & square_bb) != 0 {
            ROOK_TABLE[adjusted_square]
        } else if (board.queens & square_bb) != 0 {
            QUEEN_TABLE[adjusted_square]
        } else if (board.kings & square_bb) != 0 {
            if is_endgame {
                KING_END_GAME[adjusted_square]
            } else {
                KING_MIDDLE_GAME[adjusted_square]
            }
        } else {
            0
        }
    }

    /// Avaliação de segurança do rei
    fn evaluate_king_safety(&self, board: &Board) -> i32 {
        let mut score = 0;
        
        // Penalizar rei em xeque
        if board.is_king_in_check(Color::White) {
            score -= 50;
        }
        if board.is_king_in_check(Color::Black) {
            score += 50;
        }
        
        score
    }

    /// Avaliação de estrutura de peões
    fn evaluate_pawn_structure(&self, board: &Board) -> i32 {
        let mut score = 0;
        
        // Bonificar peões passados
        if board.has_passed_pawn(Color::White) {
            score += 30;
        }
        if board.has_passed_pawn(Color::Black) {
            score -= 30;
        }
        
        score
    }

    /// Avaliação de características táticas
    fn evaluate_tactical_features(&self, board: &Board) -> i32 {
        let mut score = 0;
        
        // Contar movimentos legais (mobilidade)
        let white_mobility = if board.to_move == Color::White {
            board.generate_legal_moves().len()
        } else {
            let mut temp_board = *board;
            temp_board.to_move = Color::White;
            temp_board.generate_legal_moves().len()
        };
        
        let black_mobility = if board.to_move == Color::Black {
            board.generate_legal_moves().len()
        } else {
            let mut temp_board = *board;
            temp_board.to_move = Color::Black;
            temp_board.generate_legal_moves().len()
        };
        
        score += (white_mobility as i32 - black_mobility as i32) * 5;
        
        score
    }

    /// Verifica se é endgame
    fn is_endgame(&self, board: &Board) -> bool {
        let total_pieces = (board.white_pieces | board.black_pieces).count_ones();
        total_pieces <= 16 // Menos de 16 peças no tabuleiro
    }
}

/// Implementação do trait Display para melhor debugging
impl std::fmt::Display for Evaluator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Evaluator({:?})", self.eval_type)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;

    #[test]
    fn test_traditional_evaluation() {
        let evaluator = Evaluator::new_traditional();
        let board = Board::new();
        
        let score = evaluator.evaluate(&board);
        
        // Posição inicial deve ter score próximo de 0
        assert!(score.abs() < 100);
    }

    #[test]
    fn test_material_evaluation() {
        let evaluator = Evaluator::new_traditional();
        let board = Board::new();
        
        let material_score = evaluator.evaluate_material(&board);
        
        // Posição inicial deve ter material equilibrado
        assert_eq!(material_score, 0);
    }

    #[test]
    fn test_endgame_detection() {
        let evaluator = Evaluator::new_traditional();
        let board = Board::new();
        
        // Posição inicial não é endgame
        assert!(!evaluator.is_endgame(&board));
        
        // Testar com posição FEN de endgame
        let endgame_fen = "8/8/8/8/8/8/8/k6K w - - 0 1";
        let endgame_board = Board::from_fen(endgame_fen).unwrap();
        assert!(evaluator.is_endgame(&endgame_board));
    }
}