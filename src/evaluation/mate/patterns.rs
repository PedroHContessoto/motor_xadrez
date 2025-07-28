// Padrões específicos de mate (não confundir com padrões de endgame)
use crate::{
    board::Board, 
    types::{Color, Move, Bitboard}
};

/// Padrões de mate específicos
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MatePatternType {
    BackRankMate,       // Mate na última fileira
    SupportedMate,      // Mate com suporte de peças
    SmotheredMate,      // Mate abafado
    DiscoveredMate,     // Mate por ataque descoberto
    PawnPromotionMate,  // Mate por promoção
    KingHuntMate,       // Mate por caça ao rei
    TwoRooksMate,       // Mate com duas torres
    QueenKnightMate,    // Mate dama + cavalo
    BishopKnightMate,   // Mate bispo + cavalo
    DoubleMate,         // Mate duplo
    TacticalMate,       // Mate tático complexo
    PositionalMate,     // Mate posicional
    ForcedMate,         // Mate forçado (sequência calculada)
    Unknown,            // Padrão não identificado
}

/// Informação detalhada sobre um padrão de mate detectado
#[derive(Debug, Clone)]
pub struct MatePatternInfo {
    pub pattern_type: MatePatternType,
    pub key_squares: Vec<u8>,           // Casas críticas para o padrão
    pub forcing_moves: Vec<Move>,       // Movimentos que forçam o mate
    pub victim_king_sq: u8,             // Posição do rei sendo mateado
    pub attacker_pieces: Vec<u8>,       // Posições das peças atacantes
    pub mate_in: u8,                    // Distância do mate
    pub confidence: f32,                // Confiança na detecção (0.0-1.0)
}

/// Detector de padrões de mate com implementação completa
pub struct MatePatternDetector {
    // Configurações internas do detector
}

impl MatePatternDetector {
    pub fn new() -> Self {
        Self {}
    }
    
    /// Detecta todos os padrões de mate possíveis na posição
    pub fn detect_patterns(&self, board: &Board) -> Vec<MatePatternInfo> {
        let mut patterns = Vec::new();
        
        // Detecta diferentes tipos de padrões
        if let Some(pattern) = self.detect_back_rank_mate(board) {
            patterns.push(pattern);
        }
        
        if let Some(pattern) = self.detect_smothered_mate(board) {
            patterns.push(pattern);
        }
        
        if let Some(pattern) = self.detect_two_rooks_mate(board) {
            patterns.push(pattern);
        }
        
        if let Some(pattern) = self.detect_queen_knight_mate(board) {
            patterns.push(pattern);
        }
        
        if let Some(pattern) = self.detect_discovered_mate(board) {
            patterns.push(pattern);
        }
        
        patterns
    }
    
    /// Detecta padrão de mate na posição (interface legada)
    pub fn detect_pattern(&self, board: &Board) -> Option<MatePatternType> {
        let patterns = self.detect_patterns(board);
        patterns.into_iter().min_by_key(|p| p.mate_in).map(|p| p.pattern_type)
    }
    
    /// Detecta mate do corredor (back-rank mate)
    pub fn detect_back_rank_mate(&self, board: &Board) -> Option<MatePatternInfo> {
        let enemy_color = !board.to_move;
        let enemy_pieces = if enemy_color == Color::White { board.white_pieces } else { board.black_pieces };
        let our_heavy_pieces = if board.to_move == Color::White { 
            board.white_pieces & (board.queens | board.rooks) 
        } else { 
            board.black_pieces & (board.queens | board.rooks) 
        };
        
        let enemy_king = board.kings & enemy_pieces;
        if enemy_king == 0 { return None; }
        
        let enemy_king_sq = enemy_king.trailing_zeros() as u8;
        let king_rank = enemy_king_sq / 8;
        
        // Verifica se rei está na primeira ou última fileira
        let is_back_rank = king_rank == 0 || king_rank == 7;
        if !is_back_rank { return None; }
        
        // Verifica se rei está preso por seus próprios peões
        if !self.is_king_trapped_by_pawns(board, enemy_king_sq, enemy_color) {
            return None;
        }
        
        // Verifica se temos peça pesada na fileira do rei
        let rank_mask = 0xFFu64 << (king_rank * 8);
        let our_attackers_on_rank = our_heavy_pieces & rank_mask;
        
        if our_attackers_on_rank == 0 { return None; }
        
        // Calcula casas de escape do rei
        let escape_squares = self.calculate_king_escape_squares(board, enemy_king_sq);
        
        // Se não há escape, é mate do corredor!
        if escape_squares == 0 {
            let attacker_squares = self.get_set_bits(our_attackers_on_rank);
            let key_squares = vec![enemy_king_sq];
            
            return Some(MatePatternInfo {
                pattern_type: MatePatternType::BackRankMate,
                key_squares,
                forcing_moves: vec![], // TODO: calcular movimentos específicos
                victim_king_sq: enemy_king_sq,
                attacker_pieces: attacker_squares,
                mate_in: 1,
                confidence: 0.95,
            });
        }
        
        None
    }
    
    /// Detecta mate afogado (smothered mate)
    pub fn detect_smothered_mate(&self, board: &Board) -> Option<MatePatternInfo> {
        let enemy_color = !board.to_move;
        let enemy_pieces = if enemy_color == Color::White { board.white_pieces } else { board.black_pieces };
        let our_knights = if board.to_move == Color::White { 
            board.white_pieces & board.knights 
        } else { 
            board.black_pieces & board.knights 
        };
        
        if our_knights == 0 { return None; }
        
        let enemy_king = board.kings & enemy_pieces;
        if enemy_king == 0 { return None; }
        
        let enemy_king_sq = enemy_king.trailing_zeros() as u8;
        
        // Verifica se rei está completamente cercado por suas próprias peças
        let king_moves = crate::moves::king::get_king_attacks_lookup(enemy_king_sq);
        let blocked_by_own_pieces = king_moves & enemy_pieces;
        
        // Para mate afogado, rei deve estar quase completamente bloqueado
        if blocked_by_own_pieces.count_ones() < 6 { return None; }
        
        // Verifica se algum cavalo pode dar xeque mate
        let knight_squares = self.get_set_bits(our_knights);
        for &knight_sq in &knight_squares {
            let knight_attacks = crate::moves::knight::get_knight_attacks_lookup(knight_sq);
            
            // Se cavalo ataca o rei e rei não pode escapar
            if (knight_attacks & enemy_king) != 0 {
                let remaining_escape = king_moves & !blocked_by_own_pieces & !knight_attacks;
                
                if remaining_escape == 0 {
                    return Some(MatePatternInfo {
                        pattern_type: MatePatternType::SmotheredMate,
                        key_squares: vec![enemy_king_sq],
                        forcing_moves: vec![], // TODO: calcular movimento do cavalo
                        victim_king_sq: enemy_king_sq,
                        attacker_pieces: vec![knight_sq],
                        mate_in: 1,
                        confidence: 0.90,
                    });
                }
            }
        }
        
        None
    }
    
    /// Detecta mate com duas torres
    pub fn detect_two_rooks_mate(&self, board: &Board) -> Option<MatePatternInfo> {
        let our_rooks = if board.to_move == Color::White { 
            board.white_pieces & board.rooks 
        } else { 
            board.black_pieces & board.rooks 
        };
        
        if our_rooks.count_ones() < 2 { return None; }
        
        let enemy_color = !board.to_move;
        let enemy_pieces = if enemy_color == Color::White { board.white_pieces } else { board.black_pieces };
        let enemy_king = board.kings & enemy_pieces;
        
        if enemy_king == 0 { return None; }
        
        let enemy_king_sq = enemy_king.trailing_zeros() as u8;
        let king_file = enemy_king_sq % 8;
        let king_rank = enemy_king_sq / 8;
        
        let rook_squares = self.get_set_bits(our_rooks);
        
        // Verifica padrão clássico: uma torre cortando fileira, outra cortando coluna
        for i in 0..rook_squares.len() {
            for j in (i+1)..rook_squares.len() {
                let rook1_sq = rook_squares[i];
                let rook2_sq = rook_squares[j];
                
                let rook1_file = rook1_sq % 8;
                let rook1_rank = rook1_sq / 8;
                let rook2_file = rook2_sq % 8;
                let rook2_rank = rook2_sq / 8;
                
                // Torre 1 na mesma fileira, Torre 2 na mesma coluna (ou vice-versa)
                let pattern1 = (rook1_rank == king_rank && rook2_file == king_file);
                let pattern2 = (rook1_file == king_file && rook2_rank == king_rank);
                
                if pattern1 || pattern2 {
                    // Verifica se rei não tem escape
                    let escape_squares = self.calculate_king_escape_squares(board, enemy_king_sq);
                    let rook1_attacks = self.get_rook_attacks(rook1_sq, board);
                    let rook2_attacks = self.get_rook_attacks(rook2_sq, board);
                    
                    let attacked_escapes = escape_squares & (rook1_attacks | rook2_attacks);
                    
                    if attacked_escapes.count_ones() >= escape_squares.count_ones() {
                        return Some(MatePatternInfo {
                            pattern_type: MatePatternType::TwoRooksMate,
                            key_squares: vec![enemy_king_sq],
                            forcing_moves: vec![],
                            victim_king_sq: enemy_king_sq,
                            attacker_pieces: vec![rook1_sq, rook2_sq],
                            mate_in: if (rook1_attacks | rook2_attacks) & enemy_king != 0 { 1 } else { 2 },
                            confidence: 0.85,
                        });
                    }
                }
            }
        }
        
        None
    }
    
    /// Detecta mate com dama + cavalo
    pub fn detect_queen_knight_mate(&self, board: &Board) -> Option<MatePatternInfo> {
        let our_queens = if board.to_move == Color::White { 
            board.white_pieces & board.queens 
        } else { 
            board.black_pieces & board.queens 
        };
        let our_knights = if board.to_move == Color::White { 
            board.white_pieces & board.knights 
        } else { 
            board.black_pieces & board.knights 
        };
        
        if our_queens == 0 || our_knights == 0 { return None; }
        
        let enemy_color = !board.to_move;
        let enemy_pieces = if enemy_color == Color::White { board.white_pieces } else { board.black_pieces };
        let enemy_king = board.kings & enemy_pieces;
        
        if enemy_king == 0 { return None; }
        
        let enemy_king_sq = enemy_king.trailing_zeros() as u8;
        let queen_sq = our_queens.trailing_zeros() as u8;
        let knight_squares = self.get_set_bits(our_knights);
        
        // Verifica combinação dama + cavalo
        let queen_attacks = self.get_queen_attacks(queen_sq, board);
        
        for &knight_sq in &knight_squares {
            let knight_attacks = crate::moves::knight::get_knight_attacks_lookup(knight_sq);
            let combined_attacks = queen_attacks | knight_attacks;
            
            // Se dama e cavalo controlam casas de escape do rei
            let escape_squares = self.calculate_king_escape_squares(board, enemy_king_sq);
            let controlled_escapes = escape_squares & combined_attacks;
            
            if controlled_escapes.count_ones() >= escape_squares.count_ones() {
                // Verifica se há xeque direto
                let in_check = (combined_attacks & enemy_king) != 0;
                
                if in_check {
                    return Some(MatePatternInfo {
                        pattern_type: MatePatternType::QueenKnightMate,
                        key_squares: vec![enemy_king_sq],
                        forcing_moves: vec![],
                        victim_king_sq: enemy_king_sq,
                        attacker_pieces: vec![queen_sq, knight_sq],
                        mate_in: 1,
                        confidence: 0.80,
                    });
                }
            }
        }
        
        None
    }
    
    /// Detecta mate por descoberta
    pub fn detect_discovered_mate(&self, board: &Board) -> Option<MatePatternInfo> {
        let our_pieces = if board.to_move == Color::White { board.white_pieces } else { board.black_pieces };
        let enemy_color = !board.to_move;
        let enemy_pieces = if enemy_color == Color::White { board.white_pieces } else { board.black_pieces };
        let enemy_king = board.kings & enemy_pieces;
        
        if enemy_king == 0 { return None; }
        
        let enemy_king_sq = enemy_king.trailing_zeros() as u8;
        
        // Procura por peças que podem descobrir xeque
        let our_piece_squares = self.get_set_bits(our_pieces & !board.kings);
        
        for &piece_sq in &our_piece_squares {
            // Simula movimento da peça e verifica se descobriu mate
            if self.piece_can_discover_mate(board, piece_sq, enemy_king_sq) {
                return Some(MatePatternInfo {
                    pattern_type: MatePatternType::DiscoveredMate,
                    key_squares: vec![enemy_king_sq, piece_sq],
                    forcing_moves: vec![], // TODO: calcular movimento específico
                    victim_king_sq: enemy_king_sq,
                    attacker_pieces: vec![piece_sq],
                    mate_in: 1,
                    confidence: 0.75,
                });
            }
        }
        
        None
    }
    
    // ========== FUNÇÕES AUXILIARES ==========
    
    /// Verifica se rei está preso por seus próprios peões
    fn is_king_trapped_by_pawns(&self, board: &Board, king_sq: u8, king_color: Color) -> bool {
        let king_moves = crate::moves::king::get_king_attacks_lookup(king_sq);
        let own_pawns = board.pawns & if king_color == Color::White { board.white_pieces } else { board.black_pieces };
        
        let blocked_by_pawns = king_moves & own_pawns;
        blocked_by_pawns.count_ones() >= 3 // Pelo menos 3 casas bloqueadas por peões
    }
    
    /// Calcula casas de escape disponíveis para o rei
    fn calculate_king_escape_squares(&self, board: &Board, king_sq: u8) -> Bitboard {
        let king_moves = crate::moves::king::get_king_attacks_lookup(king_sq);
        let all_pieces = board.white_pieces | board.black_pieces;
        
        king_moves & !all_pieces // Casas livres onde rei pode ir
    }
    
    /// Obtém ataques da torre
    fn get_rook_attacks(&self, rook_sq: u8, board: &Board) -> Bitboard {
        crate::moves::sliding::get_rook_attacks(rook_sq, board.white_pieces | board.black_pieces)
    }
    
    /// Obtém ataques da dama
    fn get_queen_attacks(&self, queen_sq: u8, board: &Board) -> Bitboard {
        let all_pieces = board.white_pieces | board.black_pieces;
        crate::moves::sliding::get_rook_attacks(queen_sq, all_pieces) |
        crate::moves::sliding::get_bishop_attacks(queen_sq, all_pieces)
    }
    
    /// Verifica se peça pode descobrir mate ao se mover
    fn piece_can_discover_mate(&self, board: &Board, piece_sq: u8, enemy_king_sq: u8) -> bool {
        // Implementação simplificada - verifica se peça está na linha rei-atacante
        let our_heavy_pieces = if board.to_move == Color::White { 
            board.white_pieces & (board.queens | board.rooks | board.bishops) 
        } else { 
            board.black_pieces & (board.queens | board.rooks | board.bishops) 
        };
        
        let heavy_squares = self.get_set_bits(our_heavy_pieces);
        
        for &heavy_sq in &heavy_squares {
            if self.is_piece_blocking_line(piece_sq, heavy_sq, enemy_king_sq) {
                return true;
            }
        }
        
        false
    }
    
    /// Verifica se uma peça está bloqueando linha entre atacante e rei
    fn is_piece_blocking_line(&self, blocking_sq: u8, attacker_sq: u8, target_sq: u8) -> bool {
        // Verifica se as três casas estão alinhadas
        let blocking_file = blocking_sq % 8;
        let blocking_rank = blocking_sq / 8;
        let attacker_file = attacker_sq % 8;
        let attacker_rank = attacker_sq / 8;
        let target_file = target_sq % 8;
        let target_rank = target_sq / 8;
        
        // Linha horizontal
        if blocking_rank == attacker_rank && attacker_rank == target_rank {
            let min_file = attacker_file.min(target_file);
            let max_file = attacker_file.max(target_file);
            return blocking_file > min_file && blocking_file < max_file;
        }
        
        // Linha vertical
        if blocking_file == attacker_file && attacker_file == target_file {
            let min_rank = attacker_rank.min(target_rank);
            let max_rank = attacker_rank.max(target_rank);
            return blocking_rank > min_rank && blocking_rank < max_rank;
        }
        
        // Diagonal (simplificado)
        let file_diff1 = (blocking_file as i8 - attacker_file as i8).abs();
        let rank_diff1 = (blocking_rank as i8 - attacker_rank as i8).abs();
        let file_diff2 = (attacker_file as i8 - target_file as i8).abs();
        let rank_diff2 = (attacker_rank as i8 - target_rank as i8).abs();
        
        if file_diff1 == rank_diff1 && file_diff2 == rank_diff2 {
            // TODO: verificação mais precisa de diagonal
            return true;
        }
        
        false
    }
    
    /// Obtém lista de casas setadas em um bitboard
    fn get_set_bits(&self, bitboard: Bitboard) -> Vec<u8> {
        let mut squares = Vec::new();
        let mut bb = bitboard;
        
        while bb != 0 {
            let sq = bb.trailing_zeros() as u8;
            squares.push(sq);
            bb &= bb - 1; // Remove o bit menos significativo
        }
        
        squares
    }
}

impl Default for MatePatternDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Função de conveniência para detectar padrões de mate
pub fn detect_mate_patterns(board: &Board) -> Vec<MatePatternInfo> {
    let detector = MatePatternDetector::new();
    detector.detect_patterns(board)
}

/// Verifica se existe algum padrão de mate na posição
pub fn has_mate_pattern(board: &Board) -> bool {
    !detect_mate_patterns(board).is_empty()
}

/// Obtém o melhor padrão de mate (menor distância)
pub fn get_best_mate_pattern(board: &Board) -> Option<MatePatternInfo> {
    let patterns = detect_mate_patterns(board);
    patterns.into_iter().min_by_key(|p| p.mate_in)
}