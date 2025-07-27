// Sistema avançado para evitar empates quando ganhando e buscar empates quando perdendo

use crate::{board::Board, evaluation, types::{Color, Move, PieceKind}};
use super::endgame_patterns::{evaluate_endgame_patterns, EndgamePatterns};
use super::material::MATERIAL_VALUES;

/// Contexto para decisões de empate/vitória
#[derive(Debug, Clone)]
pub struct DrawDecisionContext {
    pub evaluation: i32,
    pub material_advantage: i32,
    pub winning_threshold: i32,
    pub losing_threshold: i32,
    pub move_count: u16,
    pub endgame_type: EndgameType,
}

impl DrawDecisionContext {
    pub fn new(board: &Board, evaluation: i32) -> Self {
        let material_eval = MaterialEval::from_board(board);
        let endgame_type = identify_endgame_type(board);
        
        DrawDecisionContext {
            evaluation,
            material_advantage: material_eval.white_material - material_eval.black_material,
            winning_threshold: 150,  // +1.5 pawns para considerar posição ganhadora
            losing_threshold: -150,  // -1.5 pawns para considerar posição perdedora
            move_count: (board.halfmove_clock / 2) + 1,
            endgame_type,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EndgameType {
    WinningEndgame,      // KQ vs K, KR vs K, etc.
    DrawishEndgame,      // KB vs K, KN vs K, etc.
    ComplexEndgame,      // Posições complexas
    Unknown,
}

/// Sistema principal de gerenciamento de empate/vitória
pub struct DrawWinManager {
    contempt_factor: i32,
    dynamic_contempt: bool,
    endgame_knowledge: EndgameKnowledge,
}

impl DrawWinManager {
    pub fn new() -> Self {
        DrawWinManager {
            contempt_factor: 20,  // Valor base de desprezo por empate
            dynamic_contempt: true,
            endgame_knowledge: EndgameKnowledge::new(),
        }
    }

    /// Cria manager com contempt personalizado
    pub fn with_contempt(contempt: i32) -> Self {
        DrawWinManager {
            contempt_factor: contempt,
            dynamic_contempt: true,
            endgame_knowledge: EndgameKnowledge::new(),
        }
    }

    /// Avalia se deve aceitar ou evitar empate
    pub fn evaluate_draw_decision(&self, board: &Board, context: &DrawDecisionContext) -> DrawDecision {
        // 1. Análise de material
        let material_eval = self.evaluate_material_situation(board);
        
        // 2. Análise de endgame
        let endgame_eval = self.endgame_knowledge.evaluate_position(board);
        
        // 3. Decisão baseada na situação
        if context.evaluation > context.winning_threshold {
            // Estamos ganhando - evitar empate agressivamente
            self.avoid_draw_when_winning(board, &material_eval, &endgame_eval)
        } else if context.evaluation < context.losing_threshold {
            // Estamos perdendo - buscar empate ativamente
            self.seek_draw_when_losing(board, &material_eval, &endgame_eval)
        } else {
            // Posição equilibrada - decisão normal
            DrawDecision::Normal
        }
    }

    /// Evita empate quando está ganhando
    fn avoid_draw_when_winning(&self, board: &Board, material: &MaterialEval, endgame: &EndgameEval) -> DrawDecision {
        // Verifica se o endgame é teoricamente ganho
        if endgame.is_theoretical_win() {
            // Continua jogando para a vitória
            return DrawDecision::AvoidDraw {
                contempt: self.contempt_factor * 3,  // Triplo desprezo
                prefer_complexity: true,
            };
        }

        // Se material é suficiente para ganhar
        if material.can_force_win() {
            return DrawDecision::AvoidDraw {
                contempt: self.contempt_factor * 2,
                prefer_complexity: false,
            };
        }

        // Posição ganhadora mas precisa cuidado
        DrawDecision::PlayForWin {
            risk_tolerance: 0.3,  // Baixa tolerância a risco
            time_pressure_factor: 1.5,  // Mais tempo para converter
        }
    }

    /// Busca empate quando está perdendo
    fn seek_draw_when_losing(&self, board: &Board, material: &MaterialEval, endgame: &EndgameEval) -> DrawDecision {
        // Se posição permite forçar empate
        if self.can_force_draw(board) {
            return DrawDecision::ForceDraw {
                methods: vec![
                    DrawMethod::Repetition,
                    DrawMethod::Perpetual,
                    DrawMethod::Stalemate,
                ],
                urgency: DrawUrgency::Immediate,
            };
        }

        // Busca simplificação para empate teórico
        if material.leads_to_theoretical_draw() {
            return DrawDecision::SimplifyToDraw {
                target_pieces: material.pieces_to_trade(),
                avoid_tactics: true,
            };
        }

        // Complicar posição e buscar chances
        DrawDecision::Complicate {
            seek_tactics: true,
            time_trouble_tricks: board.halfmove_clock > 80,
        }
    }

    /// Verifica se pode forçar empate
    fn can_force_draw(&self, board: &Board) -> bool {
        // 1. Repetição tripla possível
        if self.can_force_repetition(board) {
            return true;
        }

        // 2. Xeque perpétuo disponível
        if self.has_perpetual_check(board) {
            return true;
        }

        // 3. Afogamento forçado
        if self.can_force_stalemate(board) {
            return true;
        }

        false
    }

    /// Verifica possibilidade de repetição forçada
    fn can_force_repetition(&self, board: &Board) -> bool {
        // Implementação simplificada - pode ser expandida
        let moves = board.generate_legal_moves();
        
        for mv in &moves {
            let mut test_board = *board;
            let _undo = test_board.make_move_fast(*mv);
            
            // Verifica se movimento resulta em posição repetitiva
            if self.leads_to_repetition(&test_board) {
                return true;
            }
        }
        false
    }

    /// Verifica xeque perpétuo disponível
    fn has_perpetual_check(&self, board: &Board) -> bool {
        let moves = board.generate_legal_moves();
        let mut check_moves = 0;
        
        for mv in &moves {
            let mut test_board = *board;
            let _undo = test_board.make_move_fast(*mv);
            
            if test_board.is_king_in_check(!board.to_move) {
                check_moves += 1;
                if check_moves >= 2 {
                    return true; // Pelo menos 2 xeques disponíveis
                }
            }
        }
        false
    }

    /// Verifica possibilidade de afogamento forçado
    fn can_force_stalemate(&self, board: &Board) -> bool {
        // Detecta se oponente tem poucas opções
        let mut temp_board = *board;
        temp_board.to_move = !temp_board.to_move;
        let opponent_moves = temp_board.generate_legal_moves();
        
        // Se oponente tem <= 2 movimentos, pode ser possível forçar stalemate
        opponent_moves.len() <= 2 && !temp_board.is_king_in_check(temp_board.to_move)
    }

    /// Sistema de pontuação para movimentos considerando empate/vitória
    pub fn score_move_for_draw_strategy(&self, mv: Move, board: &Board, strategy: &DrawDecision) -> i32 {
        match strategy {
            DrawDecision::AvoidDraw { contempt, prefer_complexity } => {
                let mut score = 0;
                
                // Penaliza movimentos que simplificam desnecessariamente
                if board.is_capture(mv) && !self.is_favorable_trade(mv, board) {
                    score -= 50;
                }
                
                // Bônus para movimentos que mantêm tensão
                if self.maintains_tension(mv, board) {
                    score += 100;
                }
                
                // Bônus para complexidade se preferido
                if *prefer_complexity && self.increases_complexity(mv, board) {
                    score += 150;
                }
                
                // Adiciona contempt factor
                score + contempt
            },
            
            DrawDecision::ForceDraw { methods, urgency } => {
                let mut score = 0;
                
                // Prioriza movimentos que levam aos métodos desejados
                for method in methods {
                    if self.move_leads_to_draw_method(mv, board, *method) {
                        score += match urgency {
                            DrawUrgency::Immediate => 1000,
                            DrawUrgency::Soon => 500,
                            DrawUrgency::Eventually => 200,
                        };
                    }
                }
                
                score
            },
            
            DrawDecision::SimplifyToDraw { target_pieces, .. } => {
                let mut score = 0;
                
                // Bônus para trocas das peças alvo
                if board.is_capture(mv) {
                    if let Some(captured) = board.get_piece_with_color(mv.to) {
                        if target_pieces.contains(&captured.piece_type()) {
                            score += 500;
                        }
                    }
                }
                
                score
            },
            
            DrawDecision::Complicate { seek_tactics, .. } => {
                let mut score = 0;
                
                if *seek_tactics && self.creates_tactical_complications(mv, board) {
                    score += 300;
                }
                
                score
            },
            
            _ => 0,
        }
    }

    /// Avalia situação material
    fn evaluate_material_situation(&self, board: &Board) -> MaterialEval {
        MaterialEval::from_board(board)
    }

    // === FUNÇÕES AUXILIARES PARA ANÁLISE DE MOVIMENTOS ===

    fn leads_to_repetition(&self, _board: &Board) -> bool {
        // Implementação simplificada - seria necessário histórico de posições
        false
    }

    fn is_favorable_trade(&self, mv: Move, board: &Board) -> bool {
        if let Some(captured) = board.get_piece_with_color(mv.to) {
            if let Some(attacker) = board.get_piece_with_color(mv.from) {
                let captured_value = MATERIAL_VALUES[captured.piece_type() as usize];
                let attacker_value = MATERIAL_VALUES[attacker.piece_type() as usize];
                return captured_value >= attacker_value;
            }
        }
        false
    }

    fn maintains_tension(&self, mv: Move, board: &Board) -> bool {
        // Verifica se movimento mantém peças atacantes próximas ao rei inimigo
        let enemy_king = if board.to_move == Color::White {
            board.kings & board.black_pieces
        } else {
            board.kings & board.white_pieces
        };
        
        if enemy_king == 0 { return false; }
        
        let king_sq = enemy_king.trailing_zeros() as u8;
        let distance_to_king = ((mv.to % 8) as i32 - (king_sq % 8) as i32).abs() +
                              ((mv.to / 8) as i32 - (king_sq / 8) as i32).abs();
        
        distance_to_king <= 3 // Movimento mantém pressão próxima ao rei
    }

    fn increases_complexity(&self, mv: Move, board: &Board) -> bool {
        // Verifica se movimento aumenta número de ameaças mútuas
        let mut test_board = *board;
        let _undo = test_board.make_move_fast(mv);
        
        let original_attacks = self.count_mutual_attacks(board);
        let new_attacks = self.count_mutual_attacks(&test_board);
        
        new_attacks > original_attacks
    }

    fn count_mutual_attacks(&self, board: &Board) -> u32 {
        // Conta ataques mútuos entre peças valiosas
        let valuable_white = (board.knights | board.bishops | board.rooks | board.queens) & board.white_pieces;
        let valuable_black = (board.knights | board.bishops | board.rooks | board.queens) & board.black_pieces;
        
        let mut attacks = 0;
        
        // Conta peças brancas atacando peças pretas valiosas
        let mut bb = valuable_white;
        while bb != 0 {
            let sq = bb.trailing_zeros() as u8;
            bb &= bb - 1;
            
            let attacked_enemies = self.get_piece_attacks(board, sq) & valuable_black;
            attacks += attacked_enemies.count_ones();
        }
        
        attacks
    }

    fn get_piece_attacks(&self, board: &Board, square: u8) -> u64 {
        // Implementação simplificada - retorna ataques da peça na casa
        if let Some(piece) = board.get_piece_with_color(square) {
            let occupancy = board.white_pieces | board.black_pieces;
            match piece.piece_type() {
                PieceKind::Queen => {
                    crate::moves::magic_bitboards::get_queen_attacks_magic(square, occupancy)
                },
                PieceKind::Rook => {
                    crate::moves::magic_bitboards::get_rook_attacks_magic(square, occupancy)
                },
                PieceKind::Bishop => {
                    crate::moves::magic_bitboards::get_bishop_attacks_magic(square, occupancy)
                },
                PieceKind::Knight => {
                    crate::moves::knight::get_knight_attacks_lookup(square)
                },
                PieceKind::King => {
                    crate::moves::king::get_king_attacks_lookup(square)
                },
                PieceKind::Pawn => {
                    // Implementação simplificada de ataques de peão
                    self.get_pawn_attacks_simple(square, piece.color)
                },
            }
        } else {
            0
        }
    }

    /// Implementação simplificada de ataques de peão
    fn get_pawn_attacks_simple(&self, square: u8, color: Color) -> u64 {
        let file = square % 8;
        let rank = square / 8;
        let mut attacks = 0u64;
        
        match color {
            Color::White => {
                if rank < 7 {
                    if file > 0 {
                        attacks |= 1u64 << ((rank + 1) * 8 + file - 1);
                    }
                    if file < 7 {
                        attacks |= 1u64 << ((rank + 1) * 8 + file + 1);
                    }
                }
            },
            Color::Black => {
                if rank > 0 {
                    if file > 0 {
                        attacks |= 1u64 << ((rank - 1) * 8 + file - 1);
                    }
                    if file < 7 {
                        attacks |= 1u64 << ((rank - 1) * 8 + file + 1);
                    }
                }
            },
        }
        
        attacks
    }

    fn move_leads_to_draw_method(&self, mv: Move, board: &Board, method: DrawMethod) -> bool {
        match method {
            DrawMethod::Repetition => self.leads_to_repetition_after_move(mv, board),
            DrawMethod::Perpetual => self.leads_to_perpetual_after_move(mv, board),
            DrawMethod::Stalemate => self.leads_to_stalemate_after_move(mv, board),
            DrawMethod::InsufficientMaterial => self.leads_to_insufficient_material(mv, board),
            DrawMethod::FiftyMoveRule => board.halfmove_clock >= 90, // Próximo da regra dos 50
        }
    }

    fn leads_to_repetition_after_move(&self, _mv: Move, _board: &Board) -> bool {
        // Implementação seria baseada no histórico de posições
        false
    }

    fn leads_to_perpetual_after_move(&self, mv: Move, board: &Board) -> bool {
        let mut test_board = *board;
        let _undo = test_board.make_move_fast(mv);
        
        // Verifica se dá xeque e permite repetição
        test_board.is_king_in_check(!board.to_move) && 
        self.has_perpetual_check(&test_board)
    }

    fn leads_to_stalemate_after_move(&self, mv: Move, board: &Board) -> bool {
        let mut test_board = *board;
        let _undo = test_board.make_move_fast(mv);
        
        // Troca turno para ver se oponente fica em stalemate
        test_board.to_move = !test_board.to_move;
        let opponent_moves = test_board.generate_legal_moves();
        
        opponent_moves.is_empty() && !test_board.is_king_in_check(test_board.to_move)
    }

    fn leads_to_insufficient_material(&self, mv: Move, board: &Board) -> bool {
        if !board.is_capture(mv) { return false; }
        
        let mut test_board = *board;
        let _undo = test_board.make_move_fast(mv);
        
        self.has_insufficient_material(&test_board)
    }

    fn has_insufficient_material(&self, board: &Board) -> bool {
        let white_pieces = board.white_pieces;
        let black_pieces = board.black_pieces;
        
        // K vs K
        if (white_pieces & !(board.kings)).count_ones() == 0 && 
           (black_pieces & !(board.kings)).count_ones() == 0 {
            return true;
        }
        
        // KB vs K ou KN vs K
        let white_minors = (board.knights | board.bishops) & white_pieces;
        let black_minors = (board.knights | board.bishops) & black_pieces;
        
        if white_minors.count_ones() == 1 && (white_pieces & !(board.kings | white_minors)).count_ones() == 0 &&
           (black_pieces & !(board.kings)).count_ones() == 0 {
            return true;
        }
        
        if black_minors.count_ones() == 1 && (black_pieces & !(board.kings | black_minors)).count_ones() == 0 &&
           (white_pieces & !(board.kings)).count_ones() == 0 {
            return true;
        }
        
        false
    }

    fn creates_tactical_complications(&self, mv: Move, board: &Board) -> bool {
        let mut test_board = *board;
        let _undo = test_board.make_move_fast(mv);
        
        // Verifica se move cria forks, pins, skewers
        self.has_tactical_motifs(&test_board)
    }

    fn has_tactical_motifs(&self, _board: &Board) -> bool {
        // Implementação simplificada - pode ser expandida
        // Verificaria forks, pins, skewers, etc.
        false
    }
}

/// Integração com o sistema de busca
impl DrawWinManager {
    /// Ajusta alpha/beta baseado na estratégia de empate
    pub fn adjust_search_windows(&self, alpha: i32, beta: i32, strategy: &DrawDecision) -> (i32, i32) {
        match strategy {
            DrawDecision::AvoidDraw { contempt, .. } => {
                // Aumenta alpha para rejeitar empates
                (alpha + contempt, beta)
            },
            DrawDecision::ForceDraw { .. } => {
                // Janela em torno de 0 para aceitar empates
                (-50, 50)
            },
            _ => (alpha, beta),
        }
    }

    /// Filtra movimentos baseado na estratégia
    pub fn filter_moves(&self, moves: Vec<Move>, board: &Board, strategy: &DrawDecision) -> Vec<Move> {
        match strategy {
            DrawDecision::SimplifyToDraw { .. } => {
                // Prioriza trocas
                moves.into_iter()
                    .filter(|mv| board.is_capture(*mv) || self.leads_to_simplification(*mv, board))
                    .collect()
            },
            DrawDecision::Complicate { .. } => {
                // Evita trocas simples
                moves.into_iter()
                    .filter(|mv| !board.is_capture(*mv) || self.creates_imbalance(*mv, board))
                    .collect()
            },
            _ => moves,
        }
    }

    fn leads_to_simplification(&self, mv: Move, board: &Board) -> bool {
        // Verifica se movimento leva a trocas ou simplificação
        board.is_capture(mv) || self.enables_trade_sequence(mv, board)
    }

    fn creates_imbalance(&self, mv: Move, board: &Board) -> bool {
        // Verifica se troca cria desequilíbrio material
        if let Some(captured) = board.get_piece_with_color(mv.to) {
            if let Some(attacker) = board.get_piece_with_color(mv.from) {
                // Troca desigual cria desequilíbrio
                return captured.piece_type() != attacker.piece_type();
            }
        }
        false
    }

    fn enables_trade_sequence(&self, _mv: Move, _board: &Board) -> bool {
        // Implementação simplificada
        false
    }
}

/// Tipos de decisão sobre empate
#[derive(Debug, Clone)]
pub enum DrawDecision {
    AvoidDraw {
        contempt: i32,
        prefer_complexity: bool,
    },
    ForceDraw {
        methods: Vec<DrawMethod>,
        urgency: DrawUrgency,
    },
    SimplifyToDraw {
        target_pieces: Vec<PieceKind>,
        avoid_tactics: bool,
    },
    PlayForWin {
        risk_tolerance: f32,
        time_pressure_factor: f32,
    },
    Complicate {
        seek_tactics: bool,
        time_trouble_tricks: bool,
    },
    Normal,
}

#[derive(Debug, Clone, Copy)]
pub enum DrawMethod {
    Repetition,
    Perpetual,
    Stalemate,
    InsufficientMaterial,
    FiftyMoveRule,
}

#[derive(Debug, Clone, Copy)]
pub enum DrawUrgency {
    Immediate,
    Soon,
    Eventually,
}

/// Conhecimento de endgame para decisões
struct EndgameKnowledge {
    winning_endgames: Vec<EndgamePattern>,
    drawing_endgames: Vec<EndgamePattern>,
}

impl EndgameKnowledge {
    fn new() -> Self {
        EndgameKnowledge {
            winning_endgames: vec![
                EndgamePattern::KQvsK,
                EndgamePattern::KRvsK,
                EndgamePattern::KBBvsK,
                EndgamePattern::KBNvsK,
                EndgamePattern::KPvsK,
            ],
            drawing_endgames: vec![
                EndgamePattern::KvsK,
                EndgamePattern::KBvsK,
                EndgamePattern::KNvsK,
                EndgamePattern::KBvsKB_SameColor,
                EndgamePattern::KNvsKN,
            ],
        }
    }

    fn evaluate_position(&self, board: &Board) -> EndgameEval {
        // Identifica tipo de endgame
        let pattern = self.identify_endgame_pattern(board);
        
        EndgameEval {
            pattern,
            is_winning: self.winning_endgames.contains(&pattern),
            is_drawing: self.drawing_endgames.contains(&pattern),
            confidence: self.calculate_confidence(board, pattern),
        }
    }

    fn identify_endgame_pattern(&self, board: &Board) -> EndgamePattern {
        let white_pieces = board.white_pieces;
        let black_pieces = board.black_pieces;
        
        let white_count = white_pieces.count_ones();
        let black_count = black_pieces.count_ones();
        
        // K vs K
        if white_count == 1 && black_count == 1 {
            return EndgamePattern::KvsK;
        }
        
        // KQ vs K
        if white_count == 2 && black_count == 1 && (board.queens & white_pieces).count_ones() == 1 {
            return EndgamePattern::KQvsK;
        }
        if black_count == 2 && white_count == 1 && (board.queens & black_pieces).count_ones() == 1 {
            return EndgamePattern::KQvsK;
        }
        
        // KR vs K
        if white_count == 2 && black_count == 1 && (board.rooks & white_pieces).count_ones() == 1 {
            return EndgamePattern::KRvsK;
        }
        if black_count == 2 && white_count == 1 && (board.rooks & black_pieces).count_ones() == 1 {
            return EndgamePattern::KRvsK;
        }
        
        // Adicionar mais padrões...
        EndgamePattern::Complex
    }

    fn calculate_confidence(&self, _board: &Board, pattern: EndgamePattern) -> f32 {
        match pattern {
            EndgamePattern::KQvsK | EndgamePattern::KRvsK => 0.95,
            EndgamePattern::KBBvsK | EndgamePattern::KBNvsK => 0.85,
            EndgamePattern::KPvsK => 0.75,
            EndgamePattern::KvsK | EndgamePattern::KBvsK | EndgamePattern::KNvsK => 0.99,
            _ => 0.5,
        }
    }
}

// Estruturas auxiliares
#[derive(Debug, Clone)]
pub struct MaterialEval {
    pub white_material: i32,
    pub black_material: i32,
    pub piece_imbalance: i32,
}

impl MaterialEval {
    fn from_board(board: &Board) -> Self {
        let white_material = Self::calculate_material(board, Color::White);
        let black_material = Self::calculate_material(board, Color::Black);
        
        MaterialEval {
            white_material,
            black_material,
            piece_imbalance: (white_material - black_material).abs(),
        }
    }
    
    fn calculate_material(board: &Board, color: Color) -> i32 {
        let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
        
        let pawns = (board.pawns & pieces).count_ones() as i32 * MATERIAL_VALUES[0];
        let knights = (board.knights & pieces).count_ones() as i32 * MATERIAL_VALUES[1];
        let bishops = (board.bishops & pieces).count_ones() as i32 * MATERIAL_VALUES[2];
        let rooks = (board.rooks & pieces).count_ones() as i32 * MATERIAL_VALUES[3];
        let queens = (board.queens & pieces).count_ones() as i32 * MATERIAL_VALUES[4];
        
        pawns + knights + bishops + rooks + queens
    }

    fn can_force_win(&self) -> bool {
        let diff = (self.white_material - self.black_material).abs();
        diff >= 300 && self.piece_imbalance < 100
    }

    fn leads_to_theoretical_draw(&self) -> bool {
        // Material insuficiente para mate
        let total_material = self.white_material + self.black_material;
        total_material < 600 // Menos que torre para cada lado
    }

    fn pieces_to_trade(&self) -> Vec<PieceKind> {
        // Simplifica retornando peças maiores primeiro
        vec![PieceKind::Queen, PieceKind::Rook, PieceKind::Bishop, PieceKind::Knight]
    }
}

#[derive(Debug, Clone)]
struct EndgameEval {
    pattern: EndgamePattern,
    is_winning: bool,
    is_drawing: bool,
    confidence: f32,
}

impl EndgameEval {
    fn is_theoretical_win(&self) -> bool {
        self.is_winning && self.confidence > 0.9
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum EndgamePattern {
    KQvsK,
    KRvsK,
    KBBvsK,
    KBNvsK,
    KPvsK,
    KvsK,
    KBvsK,
    KNvsK,
    KBvsKB_SameColor,
    KNvsKN,
    Complex,
}

/// Identifica tipo de endgame baseado no material
fn identify_endgame_type(board: &Board) -> EndgameType {
    let total_pieces = (board.white_pieces | board.black_pieces).count_ones();
    
    if total_pieces <= 5 {
        // Analisa se é endgame teórico
        let knowledge = EndgameKnowledge::new();
        let pattern = knowledge.identify_endgame_pattern(board);
        
        if knowledge.winning_endgames.contains(&pattern) {
            EndgameType::WinningEndgame
        } else if knowledge.drawing_endgames.contains(&pattern) {
            EndgameType::DrawishEndgame
        } else {
            EndgameType::ComplexEndgame
        }
    } else if total_pieces <= 10 {
        EndgameType::ComplexEndgame
    } else {
        EndgameType::Unknown
    }
}

/// Extensão do Board para obter peça completa na casa (incluindo cor)
impl Board {
    pub fn get_piece_with_color(&self, square: u8) -> Option<crate::types::Piece> {
        let square_bit = 1u64 << square;
        
        if (self.white_pieces & square_bit) != 0 {
            // Peça branca
            if (self.pawns & square_bit) != 0 {
                Some(crate::types::Piece::new(PieceKind::Pawn, Color::White))
            } else if (self.rooks & square_bit) != 0 {
                Some(crate::types::Piece::new(PieceKind::Rook, Color::White))
            } else if (self.knights & square_bit) != 0 {
                Some(crate::types::Piece::new(PieceKind::Knight, Color::White))
            } else if (self.bishops & square_bit) != 0 {
                Some(crate::types::Piece::new(PieceKind::Bishop, Color::White))
            } else if (self.queens & square_bit) != 0 {
                Some(crate::types::Piece::new(PieceKind::Queen, Color::White))
            } else if (self.kings & square_bit) != 0 {
                Some(crate::types::Piece::new(PieceKind::King, Color::White))
            } else {
                None
            }
        } else if (self.black_pieces & square_bit) != 0 {
            // Peça preta
            if (self.pawns & square_bit) != 0 {
                Some(crate::types::Piece::new(PieceKind::Pawn, Color::Black))
            } else if (self.rooks & square_bit) != 0 {
                Some(crate::types::Piece::new(PieceKind::Rook, Color::Black))
            } else if (self.knights & square_bit) != 0 {
                Some(crate::types::Piece::new(PieceKind::Knight, Color::Black))
            } else if (self.bishops & square_bit) != 0 {
                Some(crate::types::Piece::new(PieceKind::Bishop, Color::Black))
            } else if (self.queens & square_bit) != 0 {
                Some(crate::types::Piece::new(PieceKind::Queen, Color::Black))
            } else if (self.kings & square_bit) != 0 {
                Some(crate::types::Piece::new(PieceKind::King, Color::Black))
            } else {
                None
            }
        } else {
            None
        }
    }
}