// Avaliadores específicos para finais teóricos - implementação completa conforme análise
use crate::{board::Board, types::{Color, Bitboard}};
use std::sync::OnceLock;

/// Resultado da avaliação teórica
#[derive(Debug, Clone)]
pub struct TheoreticalResult {
    pub score: i32,
    pub mate_distance: Option<u8>,
    pub concepts: Vec<String>,
}

/// Tipo de função avaliadora para finais teóricos
type EvalFunction = fn(&TheoreticalEvaluator, &Board) -> Option<TheoreticalResult>;

/// Lookup table para finais teóricos conhecidos - evita recálculo desnecessário
static ENDGAME_KNOWLEDGE: OnceLock<Vec<(super::EndgameType, EvalFunction)>> = OnceLock::new();

/// Resultado direto de lookup para posições conhecidas
#[derive(Debug, Clone)]
pub struct EndgameLookupResult {
    pub result: i32,
    pub mate_in: Option<u8>,
    pub is_theoretical_draw: bool,
    pub winning_side: Option<Color>,
}

/// Avaliador de finais teóricos com sistema de lookup interno
pub struct TheoreticalEvaluator {
    // Cache interno para posições já calculadas
}

impl TheoreticalEvaluator {
    pub fn new() -> Self {
        // Inicializa a lookup table na primeira chamada
        Self::init_endgame_knowledge();
        Self {}
    }
    
    /// Inicializa a lookup table com finais teóricos conhecidos
    fn init_endgame_knowledge() {
        ENDGAME_KNOWLEDGE.get_or_init(|| {
            vec![
                (super::EndgameType::KQvsK, TheoreticalEvaluator::evaluate_kq_vs_k),
                (super::EndgameType::KRvsK, TheoreticalEvaluator::evaluate_kr_vs_k),  
                (super::EndgameType::KPvsK, TheoreticalEvaluator::evaluate_kp_vs_k),
                (super::EndgameType::KBBvsK, TheoreticalEvaluator::evaluate_kbb_vs_k),
                (super::EndgameType::KBNvsK, TheoreticalEvaluator::evaluate_kbn_vs_k),
                // Adicionar mais finais conforme necessário
            ]
        });
    }
    
    /// Sistema de lookup rápido para finais conhecidos
    pub fn quick_lookup(&self, board: &Board) -> Option<EndgameLookupResult> {
        let total_pieces = (board.white_pieces | board.black_pieces).count_ones();
        
        // Só usa lookup para finais com poucas peças
        if total_pieces > 7 {
            return None;
        }
        
        // Verifica posições de empate teórico conhecidas
        if let Some(draw_result) = self.check_theoretical_draws(board) {
            return Some(draw_result);
        }
        
        // Verifica mates forçados em posições específicas
        if let Some(mate_result) = self.check_forced_mates(board) {
            return Some(mate_result);
        }
        
        None
    }
    
    /// Verifica empates teóricos conhecidos (KvK, KBvK, KNvK, etc.)
    fn check_theoretical_draws(&self, board: &Board) -> Option<EndgameLookupResult> {
        let white_material = board.white_pieces & !board.kings;
        let black_material = board.black_pieces & !board.kings;
        let white_count = white_material.count_ones();
        let black_count = black_material.count_ones();
        
        // K vs K
        if white_count == 0 && black_count == 0 {
            return Some(EndgameLookupResult {
                result: 0,
                mate_in: None,
                is_theoretical_draw: true,
                winning_side: None,
            });
        }
        
        // KN vs K ou KB vs K (sem peões)
        if (white_count == 1 && black_count == 0) || (white_count == 0 && black_count == 1) {
            let stronger_material = if white_count > 0 { white_material } else { black_material };
            let is_knight = (stronger_material & board.knights) != 0;
            let is_bishop = (stronger_material & board.bishops) != 0;
            let no_pawns = board.pawns == 0;
            
            if no_pawns && (is_knight || is_bishop) {
                return Some(EndgameLookupResult {
                    result: 0,
                    mate_in: None,
                    is_theoretical_draw: true,
                    winning_side: None,
                });
            }
        }
        
        // KN vs KN (geralmente empate)
        if white_count == 1 && black_count == 1 && board.pawns == 0 {
            let white_knights = (white_material & board.knights).count_ones();
            let black_knights = (black_material & board.knights).count_ones();
            
            if white_knights == 1 && black_knights == 1 {
                return Some(EndgameLookupResult {
                    result: 0,
                    mate_in: None,
                    is_theoretical_draw: true,
                    winning_side: None,
                });
            }
        }
        
        None
    }
    
    /// Verifica mates forçados em posições específicas conhecidas
    fn check_forced_mates(&self, board: &Board) -> Option<EndgameLookupResult> {
        // KQ vs K - sempre ganha se rei fraco está na borda
        if let Some(result) = self.check_kq_vs_k_forced_mate(board) {
            return Some(result);
        }
        
        // KR vs K - sempre ganha se rei fraco está na borda
        if let Some(result) = self.check_kr_vs_k_forced_mate(board) {
            return Some(result);
        }
        
        None
    }
    
    /// Verifica mate forçado específico para KQ vs K
    fn check_kq_vs_k_forced_mate(&self, board: &Board) -> Option<EndgameLookupResult> {
        let (strong_side, strong_pieces, weak_pieces) = self.get_material_sides(board)?;
        
        let strong_material = strong_pieces & !board.kings;
        let weak_material = weak_pieces & !board.kings;
        let strong_queens = (board.queens & strong_pieces).count_ones();
        
        if strong_material.count_ones() != 1 || weak_material.count_ones() != 0 || strong_queens != 1 {
            return None;
        }
        
        let weak_king = board.kings & weak_pieces;
        if weak_king == 0 { return None; }
        
        let weak_king_sq = weak_king.trailing_zeros() as u8;
        let distance_to_edge = self.calculate_distance_to_edge(weak_king_sq);
        
        // Se rei está na borda, mate é muito próximo
        if distance_to_edge == 0 {
            let mate_distance = if self.is_corner_square(weak_king_sq) { 1 } else { 3 };
            return Some(EndgameLookupResult {
                result: if strong_side == Color::White { 95000 } else { -95000 },
                mate_in: Some(mate_distance),
                is_theoretical_draw: false,
                winning_side: Some(strong_side),
            });
        }
        
        None
    }
    
    /// Verifica mate forçado específico para KR vs K  
    fn check_kr_vs_k_forced_mate(&self, board: &Board) -> Option<EndgameLookupResult> {
        let (strong_side, strong_pieces, weak_pieces) = self.get_material_sides(board)?;
        
        let strong_material = strong_pieces & !board.kings;
        let weak_material = weak_pieces & !board.kings;
        let strong_rooks = (board.rooks & strong_pieces).count_ones();
        
        if strong_material.count_ones() != 1 || weak_material.count_ones() != 0 || strong_rooks != 1 {
            return None;
        }
        
        let weak_king = board.kings & weak_pieces;
        if weak_king == 0 { return None; }
        
        let weak_king_sq = weak_king.trailing_zeros() as u8;
        let distance_to_edge = self.calculate_distance_to_edge(weak_king_sq);
        
        // Se rei está na borda e torre está bem posicionada, mate é próximo
        if distance_to_edge == 0 {
            return Some(EndgameLookupResult {
                result: if strong_side == Color::White { 90000 } else { -90000 },
                mate_in: Some(5), // KR vs K pode demorar mais
                is_theoretical_draw: false,
                winning_side: Some(strong_side),
            });
        }
        
        None
    }
    
    /// Avalia final teórico se aplicável
    pub fn evaluate(&self, board: &Board, endgame_type: super::EndgameType) -> Option<TheoreticalResult> {
        use super::EndgameType;
        
        match endgame_type {
            EndgameType::KQvsK => self.evaluate_kq_vs_k(board),
            EndgameType::KRvsK => self.evaluate_kr_vs_k(board),
            EndgameType::KPvsK => self.evaluate_kp_vs_k(board),
            EndgameType::KBBvsK => self.evaluate_kbb_vs_k(board),
            EndgameType::KBNvsK => self.evaluate_kbn_vs_k(board),
            _ => None,
        }
    }
    
    /// KQ vs K - deve forçar mate em máximo 10 movimentos
    fn evaluate_kq_vs_k(&self, board: &Board) -> Option<TheoreticalResult> {
        let (strong_side, strong_pieces, weak_pieces) = self.get_material_sides(board)?;
        
        // Verifica se realmente é KQ vs K
        let strong_material = strong_pieces & !board.kings;
        let weak_material = weak_pieces & !board.kings;
        let strong_queens = (board.queens & strong_pieces).count_ones();
        
        if strong_material.count_ones() != 1 || weak_material.count_ones() != 0 || strong_queens != 1 {
            return None;
        }
        
        let evaluation = self.evaluate_kq_vs_k_position(board, strong_side);
        let mate_distance = self.calculate_kq_vs_k_mate_distance(board, strong_side);
        
        Some(TheoreticalResult {
            score: evaluation,
            mate_distance: Some(mate_distance),
            concepts: vec![
                "Force king to edge".to_string(),
                "King and queen cooperation".to_string(),
                "Control escape squares".to_string(),
            ],
        })
    }
    
    fn evaluate_kq_vs_k_position(&self, board: &Board, strong_side: Color) -> i32 {
        let strong_pieces = if strong_side == Color::White { board.white_pieces } else { board.black_pieces };
        let weak_pieces = if strong_side == Color::White { board.black_pieces } else { board.white_pieces };
        
        let strong_king = board.kings & strong_pieces;
        let weak_king = board.kings & weak_pieces;
        let queen = board.queens & strong_pieces;
        
        if strong_king == 0 || weak_king == 0 || queen == 0 {
            return 0;
        }
        
        let strong_king_sq = strong_king.trailing_zeros() as u8;
        let weak_king_sq = weak_king.trailing_zeros() as u8;
        let queen_sq = queen.trailing_zeros() as u8;
        
        let mut score = 9000; // Base winning score (era +900 apenas por material)
        
        // 1. Força rei fraco para a borda (-50 conforme análise)
        let weak_king_distance_to_edge = self.calculate_distance_to_edge(weak_king_sq);
        score += (7 - weak_king_distance_to_edge as i32) * 150; // Penalidade por rei longe da borda
        
        // 2. Bônus por controle de casas centrais (+30 conforme análise)
        score += self.evaluate_queen_central_control(queen_sq, weak_king_sq) * 30;
        
        // 3. Bônus por aproximação dos reis (+40 conforme análise)
        let king_distance = self.calculate_distance(strong_king_sq, weak_king_sq);
        score += (7 - king_distance as i32) * 40;
        
        // 4. Dama deve estar a distância ideal (nem muito longe, nem muito perto)
        let queen_king_distance = self.calculate_distance(queen_sq, weak_king_sq);
        if queen_king_distance == 2 || queen_king_distance == 3 {
            score += 50; // Distância ideal
        } else if queen_king_distance > 5 {
            score -= 30; // Muito longe
        }
        
        // 5. Bônus especial se rei está na borda
        if weak_king_distance_to_edge == 0 {
            score += 200;
            // Se na corner, ainda melhor
            if self.is_corner_square(weak_king_sq) {
                score += 300;
            }
        }
        
        // Score final guia para técnica correta conforme análise
        if strong_side == Color::White { score } else { -score }
    }
    
    fn calculate_kq_vs_k_mate_distance(&self, board: &Board, strong_side: Color) -> u8 {
        let weak_pieces = if strong_side == Color::White { board.black_pieces } else { board.white_pieces };
        let weak_king = board.kings & weak_pieces;
        
        if weak_king == 0 { return 10; }
        
        let weak_king_sq = weak_king.trailing_zeros() as u8;
        let distance_to_edge = self.calculate_distance_to_edge(weak_king_sq);
        
        // Estimate: cada casa de distância da borda = ~1.5 movimentos
        (distance_to_edge as f32 * 1.5).ceil() as u8 + 2
    }
    
    /// KR vs K - deve forçar mate em máximo 16 movimentos  
    fn evaluate_kr_vs_k(&self, board: &Board) -> Option<TheoreticalResult> {
        let (strong_side, strong_pieces, weak_pieces) = self.get_material_sides(board)?;
        
        // Verifica se realmente é KR vs K
        let strong_material = strong_pieces & !board.kings;
        let weak_material = weak_pieces & !board.kings;
        let strong_rooks = (board.rooks & strong_pieces).count_ones();
        
        if strong_material.count_ones() != 1 || weak_material.count_ones() != 0 || strong_rooks != 1 {
            return None;
        }
        
        let evaluation = self.evaluate_kr_vs_k_position(board, strong_side);
        let mate_distance = self.calculate_kr_vs_k_mate_distance(board, strong_side);
        
        Some(TheoreticalResult {
            score: evaluation,
            mate_distance: Some(mate_distance),
            concepts: vec![
                "Cut off enemy king".to_string(),
                "Drive king to edge".to_string(),
                "Ladder mate technique".to_string(),
            ],
        })
    }
    
    fn evaluate_kr_vs_k_position(&self, board: &Board, strong_side: Color) -> i32 {
        let strong_pieces = if strong_side == Color::White { board.white_pieces } else { board.black_pieces };
        let weak_pieces = if strong_side == Color::White { board.black_pieces } else { board.white_pieces };
        
        let strong_king = board.kings & strong_pieces;
        let weak_king = board.kings & weak_pieces;
        let rook = board.rooks & strong_pieces;
        
        if strong_king == 0 || weak_king == 0 || rook == 0 {
            return 0;
        }
        
        let strong_king_sq = strong_king.trailing_zeros() as u8;
        let weak_king_sq = weak_king.trailing_zeros() as u8;
        let rook_sq = rook.trailing_zeros() as u8;
        
        let mut score = 5000; // Base winning score
        
        // 1. Força rei fraco para a borda
        let weak_king_distance_to_edge = self.calculate_distance_to_edge(weak_king_sq);
        score += (7 - weak_king_distance_to_edge as i32) * 100;
        
        // 2. Torre deve cortar rei (técnica de corte)
        if self.rook_cuts_off_king(rook_sq, weak_king_sq) {
            score += 300; // Bônus grande por cortar
        }
        
        // 3. Rei forte deve apoiar
        let king_distance = self.calculate_distance(strong_king_sq, weak_king_sq);
        score += (7 - king_distance as i32) * 60;
        
        // 4. Torre não deve estar muito longe
        let rook_distance = self.calculate_distance(rook_sq, weak_king_sq);
        if rook_distance <= 4 {
            score += 50;
        }
        
        // 5. Bônus especial na borda
        if weak_king_distance_to_edge == 0 {
            score += 400;
        }
        
        if strong_side == Color::White { score } else { -score }
    }
    
    fn calculate_kr_vs_k_mate_distance(&self, board: &Board, strong_side: Color) -> u8 {
        let weak_pieces = if strong_side == Color::White { board.black_pieces } else { board.white_pieces };
        let weak_king = board.kings & weak_pieces;
        
        if weak_king == 0 { return 16; }
        
        let weak_king_sq = weak_king.trailing_zeros() as u8;
        let distance_to_edge = self.calculate_distance_to_edge(weak_king_sq);
        
        // KR vs K: ~2 movimentos por casa de distância
        (distance_to_edge as f32 * 2.0).ceil() as u8 + 3
    }
    
    /// KP vs K - avalia promoção e oposição  
    fn evaluate_kp_vs_k(&self, board: &Board) -> Option<TheoreticalResult> {
        let (strong_side, strong_pieces, weak_pieces) = self.get_material_sides(board)?;
        
        // Verifica se realmente é KP vs K
        let strong_material = strong_pieces & !board.kings;
        let weak_material = weak_pieces & !board.kings;
        let strong_pawns = (board.pawns & strong_pieces).count_ones();
        
        if strong_material.count_ones() != 1 || weak_material.count_ones() != 0 || strong_pawns != 1 {
            return None;
        }
        
        let evaluation = self.evaluate_kp_vs_k_position(board, strong_side);
        
        Some(TheoreticalResult {
            score: evaluation,
            mate_distance: None, // Pode não ser mate, apenas promoção
            concepts: vec![
                "Opposition".to_string(),
                "Square rule".to_string(),
                "King activity".to_string(),
                "Pawn promotion".to_string(),
            ],
        })
    }
    
    fn evaluate_kp_vs_k_position(&self, board: &Board, strong_side: Color) -> i32 {
        let strong_pieces = if strong_side == Color::White { board.white_pieces } else { board.black_pieces };
        let weak_pieces = if strong_side == Color::White { board.black_pieces } else { board.white_pieces };
        
        let strong_king = board.kings & strong_pieces;
        let weak_king = board.kings & weak_pieces;
        let pawn = board.pawns & strong_pieces;
        
        if strong_king == 0 || weak_king == 0 || pawn == 0 {
            return 0;
        }
        
        let strong_king_sq = strong_king.trailing_zeros() as u8;
        let weak_king_sq = weak_king.trailing_zeros() as u8;
        let pawn_sq = pawn.trailing_zeros() as u8;
        
        let mut score = 300; // Base advantage
        
        // 1. Regra do quadrado
        if self.passes_square_rule(pawn_sq, weak_king_sq, strong_side) {
            score += 800; // Peão promove livremente
        }
        
        // 2. Oposição é crucial - valores corretos conforme análise
        let opposition_score = self.evaluate_opposition_comprehensive(strong_king_sq, weak_king_sq);
        score += opposition_score;
        
        // 3. Rei ativo apoiando peão (+50 conforme análise)
        let king_pawn_distance = self.calculate_distance(strong_king_sq, pawn_sq);
        score += (4 - king_pawn_distance as i32) * 50;
        
        // 4. Peão passado avançado (+100 conforme análise)
        let pawn_rank = pawn_sq / 8;
        let advancement = if strong_side == Color::White { pawn_rank } else { 7 - pawn_rank };
        score += advancement as i32 * 100;
        
        // Total guia para vitória técnica conforme análise
        if strong_side == Color::White { score } else { -score }
    }
    
    /// KBB vs K - força rei para canto correto
    fn evaluate_kbb_vs_k(&self, board: &Board) -> Option<TheoreticalResult> {
        let (strong_side, strong_pieces, weak_pieces) = self.get_material_sides(board)?;
        
        let strong_bishops = (board.bishops & strong_pieces).count_ones();
        let strong_material = (strong_pieces & !board.kings).count_ones();
        let weak_material = (weak_pieces & !board.kings).count_ones();
        
        if strong_bishops != 2 || strong_material != 2 || weak_material != 0 {
            return None;
        }
        
        let evaluation = self.evaluate_kbb_vs_k_position(board, strong_side);
        
        Some(TheoreticalResult {
            score: evaluation,
            mate_distance: Some(15), // Máximo 19 movimentos
            concepts: vec![
                "Force to correct corner".to_string(),
                "Bishop coordination".to_string(),
                "King support".to_string(),
            ],
        })
    }
    
    fn evaluate_kbb_vs_k_position(&self, board: &Board, strong_side: Color) -> i32 {
        let weak_pieces = if strong_side == Color::White { board.black_pieces } else { board.white_pieces };
        let weak_king = board.kings & weak_pieces;
        
        if weak_king == 0 { return 0; }
        
        let weak_king_sq = weak_king.trailing_zeros() as u8;
        let mut score = 3000;
        
        // Força para qualquer canto primeiro
        let distance_to_corner = self.calculate_distance_to_nearest_corner(weak_king_sq);
        score += (7 - distance_to_corner as i32) * 150;
        
        if strong_side == Color::White { score } else { -score }
    }
    
    /// KBN vs K - padrão W de Deletang
    fn evaluate_kbn_vs_k(&self, board: &Board) -> Option<TheoreticalResult> {
        let (strong_side, strong_pieces, weak_pieces) = self.get_material_sides(board)?;
        
        let strong_bishops = (board.bishops & strong_pieces).count_ones();
        let strong_knights = (board.knights & strong_pieces).count_ones();
        let strong_material = (strong_pieces & !board.kings).count_ones();
        let weak_material = (weak_pieces & !board.kings).count_ones();
        
        if strong_bishops != 1 || strong_knights != 1 || strong_material != 2 || weak_material != 0 {
            return None;
        }
        
        let evaluation = self.evaluate_kbn_vs_k_position(board, strong_side);
        
        Some(TheoreticalResult {
            score: evaluation,
            mate_distance: Some(25), // Máximo 33 movimentos
            concepts: vec![
                "Deletang W pattern".to_string(),
                "Force to correct corner".to_string(),
                "Bishop and knight coordination".to_string(),
            ],
        })
    }
    
    fn evaluate_kbn_vs_k_position(&self, board: &Board, strong_side: Color) -> i32 {
        let weak_pieces = if strong_side == Color::White { board.black_pieces } else { board.white_pieces };
        let weak_king = board.kings & weak_pieces;
        
        if weak_king == 0 { return 0; }
        
        let weak_king_sq = weak_king.trailing_zeros() as u8;
        let mut score = 2000;
        
        // Determina cor do bispo e canto correto
        // Implementação simplificada - força para canto mais próximo
        let distance_to_corner = self.calculate_distance_to_nearest_corner(weak_king_sq);
        score += (7 - distance_to_corner as i32) * 100;
        
        if strong_side == Color::White { score } else { -score }
    }
    
    // ========== FUNÇÕES AUXILIARES ==========
    
    fn get_material_sides(&self, board: &Board) -> Option<(Color, Bitboard, Bitboard)> {
        let white_material = (board.white_pieces & !board.kings).count_ones();
        let black_material = (board.black_pieces & !board.kings).count_ones();
        
        if white_material > black_material {
            Some((Color::White, board.white_pieces, board.black_pieces))
        } else if black_material > white_material {
            Some((Color::Black, board.black_pieces, board.white_pieces))
        } else {
            None
        }
    }
    
    fn calculate_distance_to_edge(&self, square: u8) -> u8 {
        let file = square % 8;
        let rank = square / 8;
        let dist_to_files = file.min(7 - file);
        let dist_to_ranks = rank.min(7 - rank);
        dist_to_files.min(dist_to_ranks)
    }
    
    fn calculate_distance(&self, sq1: u8, sq2: u8) -> u8 {
        let file1 = sq1 % 8;
        let rank1 = sq1 / 8;
        let file2 = sq2 % 8;
        let rank2 = sq2 / 8;
        
        let file_diff = (file1 as i8 - file2 as i8).abs() as u8;
        let rank_diff = (rank1 as i8 - rank2 as i8).abs() as u8;
        
        file_diff.max(rank_diff)
    }
    
    fn calculate_distance_to_nearest_corner(&self, square: u8) -> u8 {
        let file = square % 8;
        let rank = square / 8;
        
        let corners = [(0, 0), (0, 7), (7, 0), (7, 7)];
        corners.iter()
            .map(|(cf, cr)| {
                let file_diff = (file as i8 - *cf as i8).abs() as u8;
                let rank_diff = (rank as i8 - *cr as i8).abs() as u8;
                file_diff.max(rank_diff)
            })
            .min()
            .unwrap_or(7)
    }
    
    fn is_corner_square(&self, square: u8) -> bool {
        matches!(square, 0 | 7 | 56 | 63)
    }
    
    fn evaluate_queen_central_control(&self, queen_sq: u8, enemy_king_sq: u8) -> i32 {
        // Simplified: dama controlando centro e próxima do rei inimigo
        let distance = self.calculate_distance(queen_sq, enemy_king_sq);
        (8 - distance as i32).max(0)
    }
    
    fn rook_cuts_off_king(&self, rook_sq: u8, king_sq: u8) -> bool {
        let rook_file = rook_sq % 8;
        let rook_rank = rook_sq / 8;
        let king_file = king_sq % 8;
        let king_rank = king_sq / 8;
        
        // Torre corta se está na mesma fileira/coluna e limita movimentação
        (rook_file == king_file && self.calculate_distance_to_edge(king_sq) <= 2) ||
        (rook_rank == king_rank && self.calculate_distance_to_edge(king_sq) <= 2)
    }
    
    fn passes_square_rule(&self, pawn_sq: u8, king_sq: u8, pawn_color: Color) -> bool {
        let pawn_file = pawn_sq % 8;
        let pawn_rank = pawn_sq / 8;
        
        let promotion_rank = if pawn_color == Color::White { 7 } else { 0 };
        let distance_to_promotion = (pawn_rank as i8 - promotion_rank as i8).abs() as u8;
        
        // Distância do rei inimigo até a casa de promoção
        let promotion_square = pawn_file + promotion_rank * 8;
        let king_distance_to_promotion = self.calculate_distance(king_sq, promotion_square);
        
        // Se peão chega primeiro, promove
        distance_to_promotion < king_distance_to_promotion
    }
    
    fn evaluate_opposition_comprehensive(&self, our_king_sq: u8, enemy_king_sq: u8) -> i32 {
        let file_diff = ((our_king_sq % 8) as i8 - (enemy_king_sq % 8) as i8).abs();
        let rank_diff = ((our_king_sq / 8) as i8 - (enemy_king_sq / 8) as i8).abs();
        
        // Valores corretos conforme análise
        if (file_diff == 0 && rank_diff == 2) || (file_diff == 2 && rank_diff == 0) {
            150 // DIRECT_OPPOSITION - aumentado de 25 para 150
        } else if file_diff == 2 && rank_diff == 2 {
            80  // DIAGONAL_OPPOSITION - novo
        } else if (file_diff == 0 && rank_diff > 2) || (file_diff > 2 && rank_diff == 0) {
            100 // DISTANT_OPPOSITION - novo  
        } else if (file_diff == 1 && rank_diff == 2) || (file_diff == 2 && rank_diff == 1) {
            60  // KNIGHT_OPPOSITION - novo
        } else {
            0
        }
    }
}

impl Default for TheoreticalEvaluator {
    fn default() -> Self {
        Self::new()
    }
}