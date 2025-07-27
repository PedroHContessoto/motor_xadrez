// Avaliadores práticos para finais complexos
use crate::{board::Board, types::Color};
use super::{EndgameType, DetailedGamePhase};

/// Resultado da avaliação prática
#[derive(Debug, Clone)]
pub struct PracticalResult {
    pub score: i32,
    pub concepts: Vec<String>,
}

/// Avaliador de finais práticos
pub struct PracticalEvaluator {}

impl PracticalEvaluator {
    pub fn new() -> Self {
        Self {}
    }
    
    /// Avalia finais práticos
    pub fn evaluate(&self, board: &Board, endgame_type: EndgameType, phase: DetailedGamePhase) -> PracticalResult {
        match endgame_type {
            EndgameType::QueenEndgame => self.evaluate_queen_endgame(board, phase),
            EndgameType::RookEndgame => self.evaluate_rook_endgame(board, phase),
            EndgameType::MinorPieceEndgame => self.evaluate_minor_piece_endgame(board, phase),
            EndgameType::PawnEndgame => self.evaluate_pawn_endgame(board, phase),
            _ => PracticalResult {
                score: 0,
                concepts: vec!["Unknown endgame".to_string()],
            },
        }
    }
    
    fn evaluate_queen_endgame(&self, board: &Board, phase: DetailedGamePhase) -> PracticalResult {
        let mut score = 0;
        let mut concepts = vec!["Queen endgame".to_string()];
        
        // Avaliação básica com foco na atividade
        score += self.evaluate_piece_activity(board, Color::White) * 2;
        score -= self.evaluate_piece_activity(board, Color::Black) * 2;
        
        concepts.push("Centralization important".to_string());
        concepts.push("King safety crucial".to_string());
        
        PracticalResult { score, concepts }
    }
    
    fn evaluate_rook_endgame(&self, board: &Board, phase: DetailedGamePhase) -> PracticalResult {
        let mut score = 0;
        let mut concepts = vec!["Rook endgame".to_string()];
        
        // Princípios de finais de torre
        score += self.evaluate_rook_activity(board, Color::White);
        score -= self.evaluate_rook_activity(board, Color::Black);
        
        // Torres na 7ª fileira
        score += self.evaluate_rook_on_seventh(board, Color::White) * 50;
        score -= self.evaluate_rook_on_seventh(board, Color::Black) * 50;
        
        concepts.push("Active rook important".to_string());
        concepts.push("Rook on 7th rank".to_string());
        concepts.push("Cut off enemy king".to_string());
        
        PracticalResult { score, concepts }
    }
    
    fn evaluate_minor_piece_endgame(&self, board: &Board, phase: DetailedGamePhase) -> PracticalResult {
        let mut score = 0;
        let mut concepts = vec!["Minor piece endgame".to_string()];
        
        // Atividade das peças menores
        score += self.evaluate_minor_piece_activity(board, Color::White);
        score -= self.evaluate_minor_piece_activity(board, Color::Black);
        
        concepts.push("Piece coordination".to_string());
        concepts.push("King support".to_string());
        
        PracticalResult { score, concepts }
    }
    
    fn evaluate_pawn_endgame(&self, board: &Board, phase: DetailedGamePhase) -> PracticalResult {
        let mut score = 0;
        let mut concepts = vec!["Pawn endgame".to_string()];
        
        // Conceitos fundamentais de finais de peões
        score += self.evaluate_pawn_endgame_concepts(board, Color::White);
        score -= self.evaluate_pawn_endgame_concepts(board, Color::Black);
        
        concepts.push("Opposition critical".to_string());
        concepts.push("Passed pawn advantage".to_string());
        concepts.push("King activity".to_string());
        concepts.push("Pawn structure".to_string());
        
        PracticalResult { score, concepts }
    }
    
    // ========== FUNÇÕES AUXILIARES ==========
    
    fn evaluate_piece_activity(&self, board: &Board, color: Color) -> i32 {
        // Avaliação básica de atividade das peças
        let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
        let king = board.kings & pieces;
        
        if king == 0 { return 0; }
        
        let king_sq = king.trailing_zeros() as u8;
        let centrality = self.calculate_centrality(king_sq);
        
        centrality * 15 // Valor aumentado conforme análise anterior
    }
    
    fn evaluate_rook_activity(&self, board: &Board, color: Color) -> i32 {
        let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
        let rooks = board.rooks & pieces;
        let mut activity = 0;
        
        let mut rooks_bb = rooks;
        while rooks_bb != 0 {
            let rook_sq = rooks_bb.trailing_zeros() as u8;
            rooks_bb &= rooks_bb - 1;
            
            // Torre ativa em fileiras/colunas abertas
            activity += self.evaluate_rook_position(rook_sq, board);
        }
        
        activity
    }
    
    fn evaluate_rook_on_seventh(&self, board: &Board, color: Color) -> i32 {
        let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
        let rooks = board.rooks & pieces;
        let seventh_rank = if color == Color::White { 6 } else { 1 };
        
        let mut count = 0;
        let mut rooks_bb = rooks;
        while rooks_bb != 0 {
            let rook_sq = rooks_bb.trailing_zeros() as u8;
            rooks_bb &= rooks_bb - 1;
            
            if rook_sq / 8 == seventh_rank {
                count += 1;
            }
        }
        
        count
    }
    
    fn evaluate_minor_piece_activity(&self, board: &Board, color: Color) -> i32 {
        let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
        let minors = (board.bishops | board.knights) & pieces;
        let mut activity = 0;
        
        let mut minors_bb = minors;
        while minors_bb != 0 {
            let piece_sq = minors_bb.trailing_zeros() as u8;
            minors_bb &= minors_bb - 1;
            
            activity += self.calculate_centrality(piece_sq) * 5;
        }
        
        activity
    }
    
    fn evaluate_pawn_endgame_concepts(&self, board: &Board, color: Color) -> i32 {
        let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
        let enemy_pieces = if color == Color::White { board.black_pieces } else { board.white_pieces };
        
        let our_king = board.kings & pieces;
        let enemy_king = board.kings & enemy_pieces;
        let our_pawns = board.pawns & pieces;
        
        if our_king == 0 || enemy_king == 0 { return 0; }
        
        let our_king_sq = our_king.trailing_zeros() as u8;
        let enemy_king_sq = enemy_king.trailing_zeros() as u8;
        
        let mut score = 0;
        
        // 1. Oposição (valores corretos da análise)
        score += self.evaluate_opposition_comprehensive(our_king_sq, enemy_king_sq);
        
        // 2. Atividade do rei  
        score += self.calculate_centrality(our_king_sq) * 15;
        
        // 3. Peões passados
        score += self.evaluate_passed_pawns(our_pawns, enemy_pieces, color) * 100;
        
        // 4. Estrutura de peões
        score += self.evaluate_pawn_structure_endgame(our_pawns, color);
        
        score
    }
    
    fn evaluate_rook_position(&self, rook_sq: u8, board: &Board) -> i32 {
        let mut activity = 0;
        
        // Centralização
        activity += self.calculate_centrality(rook_sq) * 3;
        
        // Fileiras/colunas abertas (simplificado)
        let rook_file = rook_sq % 8;
        let pawns_on_file = self.count_pawns_on_file(board, rook_file);
        if pawns_on_file <= 1 {
            activity += 30; // Coluna aberta/semi-aberta
        }
        
        activity
    }
    
    fn evaluate_opposition_comprehensive(&self, our_king_sq: u8, enemy_king_sq: u8) -> i32 {
        let file_diff = ((our_king_sq % 8) as i8 - (enemy_king_sq % 8) as i8).abs();
        let rank_diff = ((our_king_sq / 8) as i8 - (enemy_king_sq / 8) as i8).abs();
        
        // Valores corretos conforme análise
        if (file_diff == 0 && rank_diff == 2) || (file_diff == 2 && rank_diff == 0) {
            150 // DIRECT_OPPOSITION
        } else if file_diff == 2 && rank_diff == 2 {
            80  // DIAGONAL_OPPOSITION
        } else if (file_diff == 0 && rank_diff > 2) || (file_diff > 2 && rank_diff == 0) {
            100 // DISTANT_OPPOSITION
        } else if (file_diff == 1 && rank_diff == 2) || (file_diff == 2 && rank_diff == 1) {
            60  // KNIGHT_OPPOSITION
        } else {
            0
        }
    }
    
    fn calculate_centrality(&self, square: u8) -> i32 {
        let file = square % 8;
        let rank = square / 8;
        let file_centrality = 7 - (3.5 - file as f32).abs() as i32;
        let rank_centrality = 7 - (3.5 - rank as f32).abs() as i32;
        file_centrality + rank_centrality
    }
    
    fn evaluate_passed_pawns(&self, pawns: u64, enemy_pieces: u64, color: Color) -> i32 {
        let mut passed_count = 0;
        let mut pawns_bb = pawns;
        
        while pawns_bb != 0 {
            let pawn_sq = pawns_bb.trailing_zeros() as u8;
            pawns_bb &= pawns_bb - 1;
            
            if self.is_passed_pawn_simple(pawn_sq, color, pawns, enemy_pieces) {
                passed_count += 1;
            }
        }
        
        passed_count
    }
    
    fn evaluate_pawn_structure_endgame(&self, pawns: u64, color: Color) -> i32 {
        let mut structure_score = 0;
        
        // Peões conectados
        structure_score += self.count_connected_pawns(pawns) * 20;
        
        // Evita peões isolados
        structure_score -= self.count_isolated_pawns(pawns) * 30;
        
        structure_score
    }
    
    fn count_pawns_on_file(&self, board: &Board, file: u8) -> u32 {
        let mut count = 0;
        for rank in 0..8 {
            let square = file + rank * 8;
            if board.pawns & (1u64 << square) != 0 {
                count += 1;
            }
        }
        count
    }
    
    fn is_passed_pawn_simple(&self, pawn_sq: u8, color: Color, our_pawns: u64, enemy_pieces: u64) -> bool {
        let pawn_file = pawn_sq % 8;
        let pawn_rank = pawn_sq / 8;
        
        let direction = if color == Color::White { 1i8 } else { -1i8 };
        let start_rank = if color == Color::White { (pawn_rank + 1) as i8 } else { (pawn_rank as i8) - 1 };
        let end_rank = if color == Color::White { 7 } else { 0 };
        
        // Verifica se há peões inimigos bloqueando
        let mut rank = start_rank;
        while (color == Color::White && rank <= end_rank as i8) || 
              (color == Color::Black && rank >= end_rank as i8) {
            for file_offset in -1..=1 {
                let check_file = (pawn_file as i8 + file_offset).clamp(0, 7) as u8;
                let check_square = check_file + (rank as u8) * 8;
                if enemy_pieces & (1u64 << check_square) != 0 {
                    return false;
                }
            }
            rank += direction;
        }
        
        true
    }
    
    fn count_connected_pawns(&self, pawns: u64) -> i32 {
        let mut connected = 0;
        let mut pawns_bb = pawns;
        
        while pawns_bb != 0 {
            let pawn_sq = pawns_bb.trailing_zeros() as u8;
            pawns_bb &= pawns_bb - 1;
            
            let pawn_file = pawn_sq % 8;
            let pawn_rank = pawn_sq / 8;
            
            // Verifica peões adjacentes
            for file_offset in [-1i8, 1] {
                let adj_file = pawn_file as i8 + file_offset;
                if adj_file >= 0 && adj_file <= 7 {
                    let adj_square = adj_file as u8 + pawn_rank * 8;
                    if pawns & (1u64 << adj_square) != 0 {
                        connected += 1;
                        break;
                    }
                }
            }
        }
        
        connected / 2 // Evita dupla contagem
    }
    
    fn count_isolated_pawns(&self, pawns: u64) -> i32 {
        let mut isolated = 0;
        let mut pawns_bb = pawns;
        
        while pawns_bb != 0 {
            let pawn_sq = pawns_bb.trailing_zeros() as u8;
            pawns_bb &= pawns_bb - 1;
            
            let pawn_file = pawn_sq % 8;
            let mut has_support = false;
            
            // Verifica peões em fileiras adjacentes
            for file_offset in [-1i8, 1] {
                let adj_file = pawn_file as i8 + file_offset;
                if adj_file >= 0 && adj_file <= 7 {
                    for rank in 0..8 {
                        let adj_square = adj_file as u8 + rank * 8;
                        if pawns & (1u64 << adj_square) != 0 {
                            has_support = true;
                            break;
                        }
                    }
                    if has_support { break; }
                }
            }
            
            if !has_support {
                isolated += 1;
            }
        }
        
        isolated
    }
}

impl Default for PracticalEvaluator {
    fn default() -> Self {
        Self::new()
    }
}