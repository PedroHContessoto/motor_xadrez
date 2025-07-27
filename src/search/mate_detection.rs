// Sistema avançado de detecção de mate em N movimentos integrado com endgame_patterns
use crate::{
    board::Board, 
    types::{Color, Move, Piece, PieceKind}, 
    transposition::{TranspositionTable, EntryType},
    evaluation::endgame_patterns::{evaluate_endgame_patterns, evaluate_king_activity_advanced}
};
use std::time::Instant;

// Constantes para detecção de mate
const MATE_VALUE: i32 = 100000;
const MAX_MATE_DISTANCE: u8 = 30; // Máximo de movimentos para buscar mate
const MATE_SEARCH_TIME_LIMIT: u128 = 2000; // 2 segundos em milissegundos

/// Estrutura para informações de mate detectado com análise de endgame
#[derive(Debug, Clone)]
pub struct MateInfo {
    pub mate_in_moves: u8,
    pub best_sequence: Vec<Move>,
    pub evaluation: i32,
    pub search_depth: u8,
    pub nodes_searched: u64,
    pub time_taken_ms: u128,
    pub endgame_quality: EndgameQuality,
    pub mate_pattern: MatePattern,
}

/// Qualidade da análise de endgame para o mate
#[derive(Debug, Clone, Copy)]
pub enum EndgameQuality {
    Perfect,      // Mate forçado com técnica perfeita
    Excellent,    // Mate com técnica muito boa
    Good,         // Mate com técnica adequada
    Adequate,     // Mate funcional mas não otimal
    Poor,         // Mate com técnica ruim
}

/// Padrão de mate detectado
#[derive(Debug, Clone, Copy)]
pub enum MatePattern {
    BackRankMate,       // Mate na última fileira
    SupportedMate,      // Mate com suporte de peças
    SmotheredMate,      // Mate abafado
    DiscoveredMate,     // Mate por ataque descoberto
    PawnPromotionMate,  // Mate por promoção
    KingHuntMate,       // Mate por caça ao rei
    EndgameMate,        // Mate de endgame (K+Q vs K, etc.)
    TacticalMate,       // Mate tático complexo
    PositionalMate,     // Mate posicional
    Unknown,            // Padrão não identificado
}

/// Resultado da detecção de mate
#[derive(Debug, Clone)]
pub enum MateResult {
    MateFound(MateInfo),
    MateInProgress(u8), // Mate encontrado mas ainda calculando sequência
    NoMateFound,
    SearchTimeout,
}

/// Detector avançado de mate
pub struct MateDetector {
    nodes_searched: u64,
    start_time: Instant,
    time_limit_ms: u128,
}

impl MateDetector {
    pub fn new() -> Self {
        MateDetector {
            nodes_searched: 0,
            start_time: Instant::now(),
            time_limit_ms: MATE_SEARCH_TIME_LIMIT,
        }
    }

    /// Detecta mate em N movimentos com análise integrada de endgame
    pub fn detect_mate(&mut self, board: &Board, max_depth: u8, tt: &mut TranspositionTable) -> MateResult {
        self.start_time = Instant::now();
        self.nodes_searched = 0;

        // Análise prévia de endgame para guiar a busca
        let endgame_analysis = self.analyze_endgame_position(board);
        
        // Busca rápida para mates imediatos (M1)
        if let Some(mate_move) = self.find_mate_in_one(board) {
            let mate_pattern = self.identify_mate_pattern(board, &[mate_move]);
            let endgame_quality = self.assess_endgame_quality(board, &[mate_move], &endgame_analysis);
            
            let mate_info = MateInfo {
                mate_in_moves: 1,
                best_sequence: vec![mate_move],
                evaluation: MATE_VALUE - 1,
                search_depth: 1,
                nodes_searched: self.nodes_searched,
                time_taken_ms: self.start_time.elapsed().as_millis(),
                endgame_quality,
                mate_pattern,
            };
            return MateResult::MateFound(mate_info);
        }

        // Busca progressiva para mates mais distantes
        for depth in 2..=max_depth.min(MAX_MATE_DISTANCE) {
            if self.start_time.elapsed().as_millis() > self.time_limit_ms {
                return MateResult::SearchTimeout;
            }

            if let Some(mate_info) = self.search_mate_at_depth(board, depth, tt, &endgame_analysis) {
                return MateResult::MateFound(mate_info);
            }
        }

        MateResult::NoMateFound
    }

    /// Busca mate específico em uma profundidade com análise de endgame
    fn search_mate_at_depth(&mut self, board: &Board, depth: u8, tt: &mut TranspositionTable, endgame_analysis: &EndgameAnalysis) -> Option<MateInfo> {
        let mut best_sequence = Vec::new();
        let alpha = -MATE_VALUE;
        let beta = MATE_VALUE;

        let score = self.mate_search(board, depth, alpha, beta, &mut best_sequence, tt);

        if self.is_mate_score(score) {
            let mate_distance = self.mate_distance_from_score(score);
            let mate_pattern = self.identify_mate_pattern(board, &best_sequence);
            let endgame_quality = self.assess_endgame_quality(board, &best_sequence, endgame_analysis);
            
            Some(MateInfo {
                mate_in_moves: mate_distance,
                best_sequence,
                evaluation: score,
                search_depth: depth,
                nodes_searched: self.nodes_searched,
                time_taken_ms: self.start_time.elapsed().as_millis(),
                endgame_quality,
                mate_pattern,
            })
        } else {
            None
        }
    }

    /// Busca algoritmo específico para mate
    fn mate_search(&mut self, board: &Board, depth: u8, mut alpha: i32, beta: i32, 
                   pv: &mut Vec<Move>, tt: &mut TranspositionTable) -> i32 {
        self.nodes_searched += 1;

        // Verificação de tempo limite
        if self.nodes_searched % 1000 == 0 && self.start_time.elapsed().as_millis() > self.time_limit_ms {
            return 0;
        }

        // Verificação de mate/empate imediato
        if board.is_checkmate() {
            return -MATE_VALUE + (MAX_MATE_DISTANCE as i32 - depth as i32);
        }

        if board.is_stalemate() || board.halfmove_clock >= 100 {
            return 0;
        }

        // Caso base da recursão
        if depth == 0 {
            return self.evaluate_mate_potential(board);
        }

        // Consulta tabela de transposição
        if let Some(entry) = tt.probe(board.zobrist_hash) {
            if entry.depth >= depth {
                match entry.entry_type {
                    EntryType::Exact => return entry.score,
                    EntryType::LowerBound if entry.score >= beta => return entry.score,
                    EntryType::UpperBound if entry.score <= alpha => return entry.score,
                    _ => {}
                }
            }
        }

        let moves = self.generate_ordered_moves(board);
        if moves.is_empty() {
            return if board.is_king_in_check(board.to_move) {
                -MATE_VALUE + (MAX_MATE_DISTANCE as i32 - depth as i32)
            } else {
                0
            };
        }

        let mut best_score = -MATE_VALUE;
        let mut best_move = None;
        let mut local_pv = Vec::new();

        for mv in moves {
            let mut new_board = *board;
            let _undo_info = new_board.make_move_fast(mv);

            let mut child_pv = Vec::new();
            let score = -self.mate_search(&new_board, depth - 1, -beta, -alpha, &mut child_pv, tt);

            if score > best_score {
                best_score = score;
                best_move = Some(mv);
                local_pv.clear();
                local_pv.push(mv);
                local_pv.extend(child_pv);
            }

            alpha = alpha.max(score);
            if alpha >= beta {
                break; // Corte beta
            }
        }

        // Atualiza PV principal
        if !local_pv.is_empty() {
            pv.clear();
            pv.extend(local_pv);
        }

        // Armazena na tabela de transposição
        let entry_type = if best_score <= alpha {
            EntryType::UpperBound
        } else if best_score >= beta {
            EntryType::LowerBound
        } else {
            EntryType::Exact
        };

        tt.store(board.zobrist_hash, best_move, best_score, depth, entry_type);

        best_score
    }

    /// Encontra mate em 1 movimento (otimizado)
    fn find_mate_in_one(&mut self, board: &Board) -> Option<Move> {
        let moves = board.generate_legal_moves();
        
        for mv in moves {
            let mut test_board = *board;
            let _undo_info = test_board.make_move_fast(mv);
            self.nodes_searched += 1;
            if test_board.is_checkmate() {
                return Some(mv);
            }
        }
        None
    }

    /// Gera movimentos ordenados para busca de mate
    fn generate_ordered_moves(&self, board: &Board) -> Vec<Move> {
        let mut moves = board.generate_legal_moves();
        
        // Ordena movimentos priorizando:
        // 1. Xeques
        // 2. Capturas de peças valiosas
        // 3. Movimentos para o centro
        moves.sort_by(|a, b| {
            let score_a = self.evaluate_move_for_mate(board, *a);
            let score_b = self.evaluate_move_for_mate(board, *b);
            score_b.cmp(&score_a)
        });

        moves
    }

    /// Avalia movimento para busca de mate
    fn evaluate_move_for_mate(&self, board: &Board, mv: Move) -> i32 {
        let mut score = 0;
        
        // Prioriza xeques
        let mut test_board = *board;
        let _undo_info = test_board.make_move_fast(mv);
        if test_board.is_king_in_check(!board.to_move) {
            score += 1000;
        }

        // Prioriza capturas
        if board.piece_at(mv.to).is_some() {
            score += 500;
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
        let safe_squares = king_moves & !board.get_attacked_squares(!king_color);
        
        // Menos casas seguras = mais restrição
        10 - safe_squares.count_ones() as i32
    }

    /// Avalia potencial de mate na posição com integração de endgame
    fn evaluate_mate_potential(&self, board: &Board) -> i32 {
        let enemy_color = !board.to_move;
        let enemy_king_pos = board.kings & if enemy_color == Color::White { 
            board.white_pieces 
        } else { 
            board.black_pieces 
        };

        if enemy_king_pos == 0 { return 0; }

        let king_sq = enemy_king_pos.trailing_zeros() as u8;
        let mut mate_potential = 0;

        // === ANÁLISE BÁSICA DE MATE ===
        // Rei próximo da borda
        let rank = king_sq / 8;
        let file = king_sq % 8;
        if rank == 0 || rank == 7 || file == 0 || file == 7 {
            mate_potential += 50;
        }

        // Rei com pouca mobilidade
        let king_moves = crate::moves::king::get_king_attacks_lookup(king_sq);
        let safe_squares = king_moves & !board.get_attacked_squares(!enemy_color);
        mate_potential += (8 - safe_squares.count_ones() as i32) * 10;

        // Peças atacantes próximas
        let our_pieces = if board.to_move == Color::White { 
            board.white_pieces 
        } else { 
            board.black_pieces 
        };
        
        let attacking_pieces = (board.queens | board.rooks) & our_pieces;
        let mut attackers = attacking_pieces;
        while attackers != 0 {
            let attacker_sq = attackers.trailing_zeros() as u8;
            attackers &= attackers - 1;
            
            let distance = ((king_sq % 8) as i32 - (attacker_sq % 8) as i32).abs() +
                          ((king_sq / 8) as i32 - (attacker_sq / 8) as i32).abs();
            
            if distance <= 3 {
                mate_potential += 20;
            }
        }

        // === INTEGRAÇÃO COM ANÁLISE DE ENDGAME ===
        let endgame_patterns = evaluate_endgame_patterns(board, board.to_move);
        
        // Fatores de endgame que ajudam no mate
        mate_potential += endgame_patterns.king_opposition / 2; // Oposição facilita mate
        mate_potential += endgame_patterns.outflanking / 2;    // Outflanking ajuda
        mate_potential -= endgame_patterns.zugzwang_potential; // Zugzwang pode ser ruim para nós
        
        // Atividade do rei atacante
        let king_activity = evaluate_king_activity_advanced(board, board.to_move);
        mate_potential += king_activity / 3;

        // Análise específica de endgame para mate
        let total_pieces = (board.white_pieces | board.black_pieces).count_ones();
        if total_pieces <= 10 {
            // Em endgames, prioriza centralização do rei atacante
            let our_king = board.kings & our_pieces;
            if our_king != 0 {
                let our_king_sq = our_king.trailing_zeros() as u8;
                let our_king_centralization = self.evaluate_king_centralization(our_king_sq);
                mate_potential += our_king_centralization;
            }
        }

        mate_potential
    }

    /// Verifica se é um score de mate
    fn is_mate_score(&self, score: i32) -> bool {
        score.abs() > MATE_VALUE - 1000
    }

    /// Calcula distância do mate a partir do score
    fn mate_distance_from_score(&self, score: i32) -> u8 {
        if score > 0 {
            ((MATE_VALUE - score) as u8).max(1)
        } else {
            ((MATE_VALUE + score) as u8).max(1)
        }
    }
    /// Analisa posição de endgame para guiar busca de mate
    fn analyze_endgame_position(&self, board: &Board) -> EndgameAnalysis {
        let total_pieces = (board.white_pieces | board.black_pieces).count_ones();
        let patterns = evaluate_endgame_patterns(board, board.to_move);
        let king_activity = evaluate_king_activity_advanced(board, board.to_move);
        
        EndgameAnalysis {
            piece_count: total_pieces,
            endgame_patterns: patterns,
            king_activity_score: king_activity,
            is_theoretical_endgame: total_pieces <= 7,
            mate_potential_score: self.calculate_advanced_mate_potential(board),
        }
    }
    
    /// Calcula potencial avançado de mate
    fn calculate_advanced_mate_potential(&self, board: &Board) -> i32 {
        let mut potential = 0;
        
        // Análise de material
        let our_pieces = if board.to_move == Color::White { board.white_pieces } else { board.black_pieces };
        let enemy_pieces = if board.to_move == Color::White { board.black_pieces } else { board.white_pieces };
        
        let our_queens = (board.queens & our_pieces).count_ones();
        let our_rooks = (board.rooks & our_pieces).count_ones();
        let enemy_queens = (board.queens & enemy_pieces).count_ones();
        let enemy_rooks = (board.rooks & enemy_pieces).count_ones();
        
        // Material suficiente para mate?
        if our_queens > 0 || our_rooks >= 2 || (our_rooks >= 1 && enemy_queens == 0 && enemy_rooks == 0) {
            potential += 100;
        }
        
        potential
    }
    
    /// Identifica padrão de mate
    fn identify_mate_pattern(&self, board: &Board, sequence: &[Move]) -> MatePattern {
        if sequence.is_empty() {
            return MatePattern::Unknown;
        }
        
        let total_pieces = (board.white_pieces | board.black_pieces).count_ones();
        
        // Endgame teórico
        if total_pieces <= 7 {
            return MatePattern::EndgameMate;
        }
        
        // Análise do movimento final
        let final_move = sequence[sequence.len() - 1];
        let piece = board.piece_at(final_move.from);
        
        match piece {
            Some(p) if p.piece_type() == PieceKind::Queen => MatePattern::SupportedMate,
            Some(p) if p.piece_type() == PieceKind::Rook => {
                if final_move.to / 8 == 0 || final_move.to / 8 == 7 {
                    MatePattern::BackRankMate
                } else {
                    MatePattern::SupportedMate
                }
            },
            Some(p) if p.piece_type() == PieceKind::Knight => MatePattern::SmotheredMate,
            Some(p) if p.piece_type() == PieceKind::Pawn => MatePattern::PawnPromotionMate,
            _ => {
                if sequence.len() >= 5 {
                    MatePattern::KingHuntMate
                } else {
                    MatePattern::TacticalMate
                }
            }
        }
    }
    
    /// Avalia qualidade da técnica de endgame
    fn assess_endgame_quality(&self, board: &Board, sequence: &[Move], analysis: &EndgameAnalysis) -> EndgameQuality {
        if sequence.is_empty() {
            return EndgameQuality::Poor;
        }
        
        let mut quality_score = 0;
        
        // Eficiência da sequência
        match sequence.len() {
            1..=3 => quality_score += 30,
            4..=7 => quality_score += 20,
            8..=12 => quality_score += 10,
            _ => quality_score -= 10,
        }
        
        // Qualidade dos padrões de endgame
        if analysis.endgame_patterns.total_score() > 20 {
            quality_score += 25;
        } else if analysis.endgame_patterns.total_score() > 0 {
            quality_score += 10;
        }
        
        // Atividade do rei
        if analysis.king_activity_score > 30 {
            quality_score += 20;
        } else if analysis.king_activity_score > 0 {
            quality_score += 10;
        }
        
        // Classificação final
        match quality_score {
            70.. => EndgameQuality::Perfect,
            50..=69 => EndgameQuality::Excellent,
            30..=49 => EndgameQuality::Good,
            10..=29 => EndgameQuality::Adequate,
            _ => EndgameQuality::Poor,
        }
    }
    
    /// Avalia centralização do rei
    fn evaluate_king_centralization(&self, king_sq: u8) -> i32 {
        let file = king_sq % 8;
        let rank = king_sq / 8;
        
        // Centro absoluto (d4, d5, e4, e5)
        if (file == 3 || file == 4) && (rank == 3 || rank == 4) {
            return 25;
        }
        
        // Centro estendido
        if file >= 2 && file <= 5 && rank >= 2 && rank <= 5 {
            return 15;
        }
        
        // Bordas são ruins para rei atacante
        if file == 0 || file == 7 || rank == 0 || rank == 7 {
            return -10;
        }
        
        5
    }
}

/// Estrutura para análise de endgame
#[derive(Debug, Clone)]
struct EndgameAnalysis {
    piece_count: u32,
    endgame_patterns: crate::evaluation::endgame_patterns::EndgamePatterns,
    king_activity_score: i32,
    is_theoretical_endgame: bool,
    mate_potential_score: i32,
}

// ============================================================================
// SISTEMA INTEGRADO DE LOGGING PARA ARENA (PADRÃO UCI)
// ============================================================================

/// Interface principal para detecção e logging integrado de mates
pub fn detect_and_log_mate(board: &Board, max_depth: u8, tt: &mut TranspositionTable) -> MateResult {
    let mut detector = MateDetector::new();
    let result = detector.detect_mate(board, max_depth, tt);
    
    // Sistema de logging integrado com padrão UCI
    match &result {
        MateResult::MateFound(info) => {
            ArenaLogger::log_mate_found(info);
        },
        MateResult::MateInProgress(moves) => {
            ArenaLogger::log_mate_in_progress(*moves);
        },
        MateResult::NoMateFound => {
            // Silencioso - apenas debug interno se necessário
        },
        MateResult::SearchTimeout => {
            ArenaLogger::log_search_timeout();
        },
    }
    
    result
}

/// Sistema integrado de logging para Arena (compatível com UCI)
struct ArenaLogger;

impl ArenaLogger {
    /// Log principal para mate encontrado - formato integrado
    fn log_mate_found(info: &MateInfo) {
        // === LOG PRINCIPAL UCI COMPATÍVEL ===
        Self::log_uci_mate_info(info);
        
        // === INFORMAÇÕES DETALHADAS ===
        Self::log_detailed_mate_analysis(info);
    }
    
    /// Log UCI compatível com formato padrão
    fn log_uci_mate_info(info: &MateInfo) {
        // Formato UCI padrão para mate (similar ao usado em find_best.rs)
        let mate_score = if info.mate_in_moves <= 1 {
            format!("mate {}", info.mate_in_moves)
        } else {
            format!("mate {}", info.mate_in_moves)
        };
        
        let nps = if info.time_taken_ms > 0 {
            (info.nodes_searched as f64 / info.time_taken_ms as f64 * 1000.0) as u64
        } else {
            0
        };
        
        let pv_string = if !info.best_sequence.is_empty() {
            info.best_sequence.iter()
                .map(|m| Self::format_move_uci(*m))
                .collect::<Vec<_>>()
                .join(" ")
        } else {
            String::new()
        };
        
        // Log principal UCI compatível
        println!("info depth {} score {} nodes {} nps {} time {} pv {}",
                info.search_depth, mate_score, info.nodes_searched, 
                nps, info.time_taken_ms, pv_string);
    }
    
    /// Log detalhado da análise (usando formato info string)
    fn log_detailed_mate_analysis(info: &MateInfo) {
        // Mensagem principal do mate
        let mate_msg = match info.mate_in_moves {
            1 => "MATE EM 1! Movimento decisivo detectado",
            2 => "MATE EM 2! Sequência forçada encontrada",
            3..=5 => "Mate tático detectado",
            6..=10 => "Mate estratégico encontrado",
            _ => "Mate profundo calculado",
        };
        
        println!("info string {}", mate_msg);
        
        // Análise do padrão
        let pattern_desc = Self::get_pattern_description(info.mate_pattern);
        println!("info string Padrão: {}", pattern_desc);
        
        // Qualidade da técnica
        let quality_desc = Self::get_quality_description(info.endgame_quality);
        println!("info string Qualidade: {}", quality_desc);
        
        // Performance da busca
        println!("info string Busca: {}ms, {} nós, prof. {}", 
                info.time_taken_ms, Self::format_nodes_compact(info.nodes_searched), info.search_depth);
    }
    
    /// Obtém descrição do padrão de mate
    fn get_pattern_description(pattern: MatePattern) -> &'static str {
        match pattern {
            MatePattern::BackRankMate => "Mate na última fileira",
            MatePattern::SupportedMate => "Mate com suporte de peças",
            MatePattern::SmotheredMate => "Mate abafado",
            MatePattern::DiscoveredMate => "Mate por ataque descoberto",
            MatePattern::PawnPromotionMate => "Mate por promoção",
            MatePattern::KingHuntMate => "Caça ao rei",
            MatePattern::EndgameMate => "Mate de endgame",
            MatePattern::TacticalMate => "Mate tático",
            MatePattern::PositionalMate => "Mate posicional",
            MatePattern::Unknown => "Padrão único",
        }
    }
    
    /// Obtém descrição da qualidade
    fn get_quality_description(quality: EndgameQuality) -> &'static str {
        match quality {
            EndgameQuality::Perfect => "Perfeita (100%)",
            EndgameQuality::Excellent => "Excelente (85%)",
            EndgameQuality::Good => "Boa (70%)",
            EndgameQuality::Adequate => "Adequada (55%)",
            EndgameQuality::Poor => "Fraca (30%)",
        }
    }
    
    
    /// Log para mate em progresso
    fn log_mate_in_progress(moves: u8) {
        println!("info string Mate M{} detectado - calculando sequência...", moves);
    }
    
    /// Log para timeout de busca
    fn log_search_timeout() {
        println!("info string Busca de mate interrompida por tempo limite");
    }
    
    // === FUNÇÕES AUXILIARES UCI ===
    
    /// Formatação UCI padrão de movimentos
    fn format_move_uci(mv: Move) -> String {
        let from_file = (mv.from % 8) as u8 + b'a';
        let from_rank = (mv.from / 8) + 1;
        let to_file = (mv.to % 8) as u8 + b'a';
        let to_rank = (mv.to / 8) + 1;
        
        format!("{}{}{}{}", 
                from_file as char, from_rank,
                to_file as char, to_rank)
    }
    
    /// Formata número de nós de forma compacta
    fn format_nodes_compact(nodes: u64) -> String {
        if nodes >= 1_000_000 {
            format!("{:.1}M", nodes as f64 / 1_000_000.0)
        } else if nodes >= 1_000 {
            format!("{:.1}K", nodes as f64 / 1_000.0)
        } else {
            nodes.to_string()
        }
    }
}


/// Extensão do Board para get_attacked_squares
impl Board {
    fn get_attacked_squares(&self, color: Color) -> u64 {
        crate::evaluation::mobility::compute_attacked_squares(self, color)
    }
    
    /// Obtém peça na posição especificada
    fn piece_at(&self, square: u8) -> Option<crate::types::Piece> {
        let square_bit = 1u64 << square;
        
        if (self.white_pieces & square_bit) != 0 {
            // Peça branca
            if (self.pawns & square_bit) != 0 {
                Some(Piece::new(PieceKind::Pawn, Color::White))
            } else if (self.rooks & square_bit) != 0 {
                Some(Piece::new(PieceKind::Rook, Color::White))
            } else if (self.knights & square_bit) != 0 {
                Some(Piece::new(PieceKind::Knight, Color::White))
            } else if (self.bishops & square_bit) != 0 {
                Some(Piece::new(PieceKind::Bishop, Color::White))
            } else if (self.queens & square_bit) != 0 {
                Some(Piece::new(PieceKind::Queen, Color::White))
            } else if (self.kings & square_bit) != 0 {
                Some(Piece::new(PieceKind::King, Color::White))
            } else {
                None
            }
        } else if (self.black_pieces & square_bit) != 0 {
            // Peça preta
            if (self.pawns & square_bit) != 0 {
                Some(Piece::new(PieceKind::Pawn, Color::Black))
            } else if (self.rooks & square_bit) != 0 {
                Some(Piece::new(PieceKind::Rook, Color::Black))
            } else if (self.knights & square_bit) != 0 {
                Some(Piece::new(PieceKind::Knight, Color::Black))
            } else if (self.bishops & square_bit) != 0 {
                Some(Piece::new(PieceKind::Bishop, Color::Black))
            } else if (self.queens & square_bit) != 0 {
                Some(Piece::new(PieceKind::Queen, Color::Black))
            } else if (self.kings & square_bit) != 0 {
                Some(Piece::new(PieceKind::King, Color::Black))
            } else {
                None
            }
        } else {
            None
        }
    }
}