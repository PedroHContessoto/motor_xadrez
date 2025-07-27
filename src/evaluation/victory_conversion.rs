// Sistema especializado para converter vantagens em vitórias e detectar mates eficientemente

use crate::{
    board::Board,
    types::{Move, Color, PieceKind},
    evaluation,
    search::SearchContext,
};
use std::collections::HashMap;

/// Sistema principal de conversão de vitórias
pub struct VictoryConversionSystem {
    mate_patterns: MatePatternDatabase,
    technique_evaluator: TechniqueEvaluator,
    progress_tracker: ProgressTracker,
}

impl VictoryConversionSystem {
    pub fn new() -> Self {
        VictoryConversionSystem {
            mate_patterns: MatePatternDatabase::new(),
            technique_evaluator: TechniqueEvaluator::new(),
            progress_tracker: ProgressTracker::new(),
        }
    }

    /// Avalia e guia a conversão de uma vantagem em vitória
    pub fn evaluate_winning_plan(&mut self, board: &Board, advantage: i32) -> WinningPlan {
        // 1. Identifica tipo de vantagem
        let advantage_type = self.classify_advantage(board, advantage);
        
        // 2. Seleciona plano apropriado
        let plan = match advantage_type {
            AdvantageType::Material(diff) => self.create_material_conversion_plan(board, diff),
            AdvantageType::Positional => self.create_positional_conversion_plan(board),
            AdvantageType::Tactical => self.create_tactical_conversion_plan(board),
            AdvantageType::Mixed => self.create_hybrid_conversion_plan(board),
        };

        // 3. Avalia progresso
        self.progress_tracker.update_progress(board, &plan);

        plan
    }

    /// Classifica tipo de vantagem
    fn classify_advantage(&self, board: &Board, advantage: i32) -> AdvantageType {
        let material_diff = evaluate_material(board);
        let material_ratio = material_diff.abs() as f32 / advantage.abs().max(1) as f32;

        if material_ratio > 0.7 {
            AdvantageType::Material(material_diff)
        } else if self.has_tactical_motifs(board) {
            AdvantageType::Tactical
        } else if material_ratio < 0.3 {
            AdvantageType::Positional
        } else {
            AdvantageType::Mixed
        }
    }

    /// Cria plano para converter vantagem material
    fn create_material_conversion_plan(&self, board: &Board, material_diff: i32) -> WinningPlan {
        let mut plan = WinningPlan::new();

        // Fase 1: Simplificação controlada
        if material_diff > 500 {
            plan.add_phase(ConversionPhase::Simplification {
                target_piece_count: 10,
                avoid_complications: true,
                time_limit: 20,
            });
        }

        // Fase 2: Criação de peão passado
        if self.can_create_passed_pawn(board) {
            plan.add_phase(ConversionPhase::PassedPawnCreation {
                target_files: self.identify_pawn_break_files(board),
                support_with_king: true,
            });
        }

        // Fase 3: Conversão técnica
        plan.add_phase(ConversionPhase::TechnicalConversion {
            endgame_type: self.identify_target_endgame(board, material_diff),
            technique_requirements: vec![
                Technique::KingActivity,
                Technique::PawnPromotion,
                Technique::PieceCoordination,
            ],
        });

        // Fase 4: Mate
        plan.add_phase(ConversionPhase::MateExecution {
            pattern_type: self.mate_patterns.suggest_pattern(board),
            max_moves: 50,
        });

        plan
    }

    /// Cria plano para vantagem posicional
    fn create_positional_conversion_plan(&self, board: &Board) -> WinningPlan {
        let mut plan = WinningPlan::new();

        // Fase 1: Restrição de espaço
        plan.add_phase(ConversionPhase::SpaceRestriction {
            target_area: self.identify_restriction_area(board),
            maintain_pressure: true,
        });

        // Fase 2: Criação de fraquezas
        plan.add_phase(ConversionPhase::PassedPawnCreation {
            target_files: self.identify_pawn_break_files(board),
            support_with_king: false,
        });

        // Fase 3: Conversão material
        plan.add_phase(ConversionPhase::TechnicalConversion {
            endgame_type: EndgameType::Complex,
            technique_requirements: vec![
                Technique::SpaceAdvantage,
                Technique::PieceCoordination,
                Technique::TimeManagement,
            ],
        });

        plan
    }

    /// Cria plano para vantagem tática
    fn create_tactical_conversion_plan(&self, board: &Board) -> WinningPlan {
        let mut plan = WinningPlan::new();

        // Execução tática imediata
        plan.add_phase(ConversionPhase::MateExecution {
            pattern_type: self.mate_patterns.suggest_pattern(board),
            max_moves: 10,
        });

        plan
    }

    /// Cria plano híbrido
    fn create_hybrid_conversion_plan(&self, board: &Board) -> WinningPlan {
        let mut plan = WinningPlan::new();

        // Combina elementos posicionais e materiais
        plan.add_phase(ConversionPhase::SpaceRestriction {
            target_area: self.identify_restriction_area(board),
            maintain_pressure: true,
        });

        plan.add_phase(ConversionPhase::Simplification {
            target_piece_count: 12,
            avoid_complications: false,
            time_limit: 15,
        });

        plan.add_phase(ConversionPhase::TechnicalConversion {
            endgame_type: EndgameType::Complex,
            technique_requirements: vec![
                Technique::KingActivity,
                Technique::SpaceAdvantage,
                Technique::PieceCoordination,
            ],
        });

        plan
    }

    /// Sistema de detecção rápida de mate
    pub fn quick_mate_detection(&self, board: &Board, depth: u8) -> Option<MateSequence> {
        // IMPORTANTE: Só busca mate se a posição realmente justifica
        let evaluation = evaluation::evaluate(board);
        
        // Critérios muito mais restritivos para buscar mate:
        // 1. Vantagem massive (>1500cp) OU
        // 2. Oponente já em xeque OU  
        // 3. Muito poucas peças no tabuleiro (<10)
        let total_pieces = (board.white_pieces | board.black_pieces).count_ones();
        let opponent_in_check = board.is_king_in_check(!board.to_move);
        
        if evaluation < 1500 && !opponent_in_check && total_pieces > 10 {
            return None;
        }
        
        // Só busca mate até profundidade 3 para evitar falsos positivos
        let safe_depth = depth.min(3);
        
        // 1. Verifica padrões de mate conhecidos primeiro (mais confiáveis)
        if let Some(pattern) = self.mate_patterns.detect_pattern(board) {
            if let Some(sequence) = self.execute_mate_pattern(board, pattern, safe_depth) {
                // Valida a sequência antes de retornar
                if self.validate_mate_sequence(board, &sequence) {
                    return Some(sequence);
                }
            }
        }

        // 2. Busca especializada apenas se critérios muito restritivos forem atendidos
        if evaluation > 2000 || (opponent_in_check && total_pieces <= 8) {
            if let Some(sequence) = self.specialized_mate_search(board, safe_depth) {
                // Valida antes de retornar
                if self.validate_mate_sequence(board, &sequence) {
                    return Some(sequence);
                }
            }
        }
        
        None
    }

    /// Busca especializada otimizada para mates (versão conservadora)
    fn specialized_mate_search(&self, board: &Board, max_depth: u8) -> Option<MateSequence> {
        // Apenas procura mates muito curtos (1-2 movimentos) para evitar falsos positivos
        let conservative_depth = max_depth.min(2);
        let mut cache = HashMap::new();
        
        // Verifica primeiro mate em 1
        if let Some(sequence) = self.search_mate_at_depth(board, 1, &mut cache) {
            return Some(sequence);
        }
        
        // Só procura mate em 2 se há evidência forte (oponente em xeque ou pouquíssimas peças)
        if conservative_depth >= 2 {
            let total_pieces = (board.white_pieces | board.black_pieces).count_ones();
            if board.is_king_in_check(!board.to_move) || total_pieces <= 6 {
                if let Some(sequence) = self.search_mate_at_depth(board, 2, &mut cache) {
                    return Some(sequence);
                }
            }
        }

        None
    }

    /// Busca mate em profundidade específica
    fn search_mate_at_depth(&self, board: &Board, depth: u8, cache: &mut HashMap<u64, i32>) -> Option<MateSequence> {
        if depth == 0 {
            return None;
        }

        // Verifica cache
        if let Some(&cached_result) = cache.get(&board.zobrist_hash) {
            if cached_result != 0 {
                return Some(MateSequence {
                    moves: Vec::new(),
                    mate_in: depth,
                });
            }
        }

        let moves = board.generate_legal_moves();
        
        for mv in moves {
            let mut test_board = *board;
            let _undo = test_board.make_move_fast(mv);
            
            // Verifica mate imediato
            if test_board.is_checkmate() {
                return Some(MateSequence {
                    moves: vec![mv],
                    mate_in: 1,
                });
            }
            
            // Busca recursiva
            if depth > 1 {
                if let Some(mut sequence) = self.search_mate_at_depth(&test_board, depth - 1, cache) {
                    sequence.moves.insert(0, mv);
                    sequence.mate_in = depth;
                    return Some(sequence);
                }
            }
        }

        // Armazena no cache
        cache.insert(board.zobrist_hash, 0);
        None
    }

    /// Executa padrão de mate
    fn execute_mate_pattern(&self, board: &Board, pattern: &MatePattern, depth: u8) -> Option<MateSequence> {
        (pattern.execution_function)(board).map(|moves| MateSequence {
            moves,
            mate_in: depth,
        })
    }

    /// Avalia técnica de conversão
    pub fn evaluate_conversion_technique(&self, board: &Board, moves: &[Move]) -> TechniqueScore {
        self.technique_evaluator.evaluate_sequence(board, moves)
    }
    
    /// Valida rigorosamente se uma sequência é realmente mate forçado
    fn validate_mate_sequence(&self, board: &Board, sequence: &MateSequence) -> bool {
        if sequence.moves.is_empty() {
            return false;
        }
        
        let mut test_board = *board;
        
        // Executa cada movimento da sequência
        for (i, &mv) in sequence.moves.iter().enumerate() {
            // Verifica se movimento é legal
            let legal_moves = test_board.generate_legal_moves();
            if !legal_moves.contains(&mv) {
                return false;
            }
            
            // Executa movimento
            let _undo = test_board.make_move_fast(mv);
            
            // Se é último movimento, deve resultar em mate
            if i == sequence.moves.len() - 1 {
                return test_board.is_checkmate();
            }
            
            // Movimento intermediário deve dar xeque ou ser praticamente forçado
            if !test_board.is_king_in_check(test_board.to_move) {
                // Se não dá xeque, a resposta deve ser muito limitada
                let responses = test_board.generate_legal_moves();
                if responses.len() > 2 {
                    return false; // Muitas opções = não é forçado
                }
            }
        }
        
        false
    }

    // === FUNÇÕES AUXILIARES ===

    fn has_tactical_motifs(&self, board: &Board) -> bool {
        // Verifica xeques, capturas forçadas, ameaças imediatas
        let moves = board.generate_legal_moves();
        
        for mv in &moves {
            let mut test_board = *board;
            let _undo = test_board.make_move_fast(*mv);
            
            if test_board.is_king_in_check(!board.to_move) {
                return true;
            }
            
            if board.is_capture(*mv) {
                if let Some(captured) = board.get_piece_with_color(mv.to) {
                    if captured.piece_type() != PieceKind::Pawn {
                        return true;
                    }
                }
            }
        }
        
        false
    }

    fn can_create_passed_pawn(&self, board: &Board) -> bool {
        // Verifica se há potencial para criar peão passado
        let our_pawns = board.pawns & if board.to_move == Color::White {
            board.white_pieces
        } else {
            board.black_pieces
        };

        our_pawns.count_ones() >= 2
    }

    fn identify_pawn_break_files(&self, board: &Board) -> Vec<u8> {
        let mut files = Vec::new();
        
        for file in 0..8 {
            if self.can_break_on_file(board, file) {
                files.push(file);
            }
        }
        
        files
    }

    fn can_break_on_file(&self, board: &Board, file: u8) -> bool {
        // Verifica se há oportunidade de avanço na coluna
        let file_mask = 0x0101010101010101u64 << file;
        let our_pawns = board.pawns & if board.to_move == Color::White {
            board.white_pieces
        } else {
            board.black_pieces
        };

        (our_pawns & file_mask).count_ones() >= 1
    }

    fn identify_target_endgame(&self, board: &Board, material_diff: i32) -> EndgameType {
        let total_pieces = (board.white_pieces | board.black_pieces).count_ones();
        
        if total_pieces <= 8 {
            if (board.queens & (board.white_pieces | board.black_pieces)).count_ones() > 0 {
                EndgameType::QueenEndgame
            } else if (board.rooks & (board.white_pieces | board.black_pieces)).count_ones() > 0 {
                EndgameType::RookEndgame
            } else if (board.pawns & (board.white_pieces | board.black_pieces)).count_ones() > 2 {
                EndgameType::PawnEndgame
            } else {
                EndgameType::MinorPieceEndgame
            }
        } else {
            EndgameType::Complex
        }
    }

    fn identify_restriction_area(&self, board: &Board) -> RestrictedArea {
        // Identifica área para restringir o rei inimigo
        let enemy_king = board.kings & if board.to_move == Color::White {
            board.black_pieces
        } else {
            board.white_pieces
        };

        if enemy_king == 0 {
            return RestrictedArea {
                min_file: 0,
                max_file: 7,
                min_rank: 0,
                max_rank: 7,
            };
        }

        let king_sq = enemy_king.trailing_zeros() as u8;
        let king_file = king_sq % 8;
        let king_rank = king_sq / 8;

        // Cria área restritiva em torno do rei
        RestrictedArea {
            min_file: king_file.saturating_sub(2),
            max_file: (king_file + 2).min(7),
            min_rank: king_rank.saturating_sub(2),
            max_rank: (king_rank + 2).min(7),
        }
    }
}

/// Plano estruturado para converter vantagem em vitória
#[derive(Debug, Clone)]
pub struct WinningPlan {
    phases: Vec<ConversionPhase>,
    current_phase: usize,
    progress: f32,
    time_allocated: u32,
}

impl WinningPlan {
    fn new() -> Self {
        WinningPlan {
            phases: Vec::new(),
            current_phase: 0,
            progress: 0.0,
            time_allocated: 0,
        }
    }

    fn add_phase(&mut self, phase: ConversionPhase) {
        self.phases.push(phase);
    }

    /// Obtém próximo movimento recomendado pelo plano
    pub fn get_recommended_move(&self, board: &Board, candidates: &[Move]) -> Option<Move> {
        if let Some(phase) = self.phases.get(self.current_phase) {
            phase.evaluate_moves(board, candidates)
        } else {
            None
        }
    }

    /// Atualiza progresso do plano
    pub fn update_progress(&mut self, board: &Board) {
        if let Some(phase) = self.phases.get(self.current_phase) {
            let phase_progress = phase.calculate_progress(board);
            
            if phase_progress >= 1.0 {
                // Fase completa, avança para próxima
                self.current_phase += 1;
                self.progress = (self.current_phase as f32) / (self.phases.len() as f32);
            }
        }
    }

    /// Verifica se plano está completo
    pub fn is_complete(&self) -> bool {
        self.current_phase >= self.phases.len()
    }

    /// Obtém descrição da fase atual
    pub fn current_phase_description(&self) -> String {
        if let Some(phase) = self.phases.get(self.current_phase) {
            match phase {
                ConversionPhase::Simplification { .. } => "Simplificação controlada".to_string(),
                ConversionPhase::PassedPawnCreation { .. } => "Criação de peão passado".to_string(),
                ConversionPhase::TechnicalConversion { .. } => "Conversão técnica".to_string(),
                ConversionPhase::MateExecution { .. } => "Execução de mate".to_string(),
                ConversionPhase::SpaceRestriction { .. } => "Restrição de espaço".to_string(),
            }
        } else {
            "Plano completo".to_string()
        }
    }
}

/// Fases de conversão de vitória
#[derive(Debug, Clone)]
pub enum ConversionPhase {
    Simplification {
        target_piece_count: u32,
        avoid_complications: bool,
        time_limit: u32,
    },
    PassedPawnCreation {
        target_files: Vec<u8>,
        support_with_king: bool,
    },
    TechnicalConversion {
        endgame_type: EndgameType,
        technique_requirements: Vec<Technique>,
    },
    MateExecution {
        pattern_type: MatePatternType,
        max_moves: u32,
    },
    SpaceRestriction {
        target_area: RestrictedArea,
        maintain_pressure: bool,
    },
}

impl ConversionPhase {
    /// Avalia movimentos para esta fase
    fn evaluate_moves(&self, board: &Board, candidates: &[Move]) -> Option<Move> {
        let scored_moves: Vec<(Move, i32)> = candidates.iter()
            .map(|&mv| (mv, self.score_move_for_phase(board, mv)))
            .collect();

        scored_moves.into_iter()
            .max_by_key(|(_, score)| *score)
            .map(|(mv, _)| mv)
    }

    /// Pontua movimento para a fase atual
    fn score_move_for_phase(&self, board: &Board, mv: Move) -> i32 {
        match self {
            ConversionPhase::Simplification { avoid_complications, .. } => {
                let mut score = 0;
                
                // Favorece trocas
                if board.is_capture(mv) {
                    score += 100;
                    
                    // Bônus para trocar peças maiores
                    if let Some(captured) = board.get_piece_with_color(mv.to) {
                        score += match captured.piece_type() {
                            PieceKind::Queen => 200,
                            PieceKind::Rook => 150,
                            PieceKind::Bishop | PieceKind::Knight => 100,
                            _ => 50,
                        };
                    }
                }
                
                // Evita complicações
                if *avoid_complications && !creates_tactics(board, mv) {
                    score += 50;
                }
                
                score
            },
            
            ConversionPhase::PassedPawnCreation { target_files, .. } => {
                let mut score = 0;
                
                // Favorece avanços de peão nas colunas alvo
                let from_file = mv.from % 8;
                let to_file = mv.to % 8;
                
                if target_files.contains(&from_file) || target_files.contains(&to_file) {
                    if is_pawn_move(board, mv) {
                        score += 150;
                        
                        // Bônus para avanços
                        let advance = if board.to_move == Color::White {
                            (mv.to / 8) as i32 - (mv.from / 8) as i32
                        } else {
                            (mv.from / 8) as i32 - (mv.to / 8) as i32
                        };
                        
                        score += advance * 50;
                    }
                }
                
                score
            },
            
            ConversionPhase::MateExecution { pattern_type, .. } => {
                // Delega para sistema de padrões de mate
                evaluate_mate_move(board, mv, *pattern_type)
            },
            
            ConversionPhase::SpaceRestriction { target_area, .. } => {
                let mut score = 0;
                
                // Favorece movimentos que restringem espaço
                if mv.to >= target_area.min_rank * 8 + target_area.min_file &&
                   mv.to <= target_area.max_rank * 8 + target_area.max_file {
                    score += 100;
                }
                
                // Verifica se move restringe mobilidade inimiga
                if restricts_enemy_mobility(board, mv) {
                    score += 75;
                }
                
                score
            },
            
            ConversionPhase::TechnicalConversion { technique_requirements, .. } => {
                let mut score = 0;
                
                for technique in technique_requirements {
                    score += evaluate_technique_move(board, mv, *technique);
                }
                
                score
            },
        }
    }

    /// Calcula progresso da fase
    fn calculate_progress(&self, board: &Board) -> f32 {
        match self {
            ConversionPhase::Simplification { target_piece_count, .. } => {
                let current_pieces = (board.white_pieces | board.black_pieces).count_ones();
                if current_pieces <= *target_piece_count {
                    1.0
                } else {
                    1.0 - (current_pieces as f32 - *target_piece_count as f32).max(0.0) / 20.0
                }
            },
            
            ConversionPhase::PassedPawnCreation { target_files, .. } => {
                // Verifica se há peão passado nas colunas alvo
                let has_passed = target_files.iter()
                    .any(|&file| has_passed_pawn_on_file(board, file));
                
                if has_passed { 1.0 } else { 0.0 }
            },
            
            ConversionPhase::MateExecution { .. } => {
                // Verifica se mate está próximo
                if board.is_checkmate() {
                    1.0
                } else if board.is_king_in_check(!board.to_move) {
                    0.8
                } else {
                    0.0
                }
            },
            
            _ => 0.5, // Progresso genérico
        }
    }
}

/// Base de dados de padrões de mate
struct MatePatternDatabase {
    patterns: Vec<MatePattern>,
}

impl MatePatternDatabase {
    fn new() -> Self {
        MatePatternDatabase {
            patterns: Self::initialize_patterns(),
        }
    }

    fn initialize_patterns() -> Vec<MatePattern> {
        vec![
            // Back rank mate
            MatePattern {
                name: "Back Rank Mate",
                pattern_type: MatePatternType::BackRank,
                required_pieces: vec![PieceKind::Rook],
                key_squares: vec![],
                execution_function: execute_back_rank_mate,
            },
            
            // Smothered mate
            MatePattern {
                name: "Smothered Mate",
                pattern_type: MatePatternType::Smothered,
                required_pieces: vec![PieceKind::Knight],
                key_squares: vec![],
                execution_function: execute_smothered_mate,
            },
            
            // Two rooks mate
            MatePattern {
                name: "Two Rooks Mate",
                pattern_type: MatePatternType::TwoRooks,
                required_pieces: vec![PieceKind::Rook, PieceKind::Rook],
                key_squares: vec![],
                execution_function: execute_two_rooks_mate,
            },
            
            // Queen and king mate
            MatePattern {
                name: "Queen and King Mate",
                pattern_type: MatePatternType::QueenKing,
                required_pieces: vec![PieceKind::Queen, PieceKind::King],
                key_squares: vec![],
                execution_function: execute_queen_king_mate,
            },
        ]
    }

    fn detect_pattern(&self, board: &Board) -> Option<&MatePattern> {
        self.patterns.iter()
            .find(|pattern| pattern.matches_position(board))
    }

    fn suggest_pattern(&self, board: &Board) -> MatePatternType {
        if let Some(pattern) = self.detect_pattern(board) {
            pattern.pattern_type
        } else {
            MatePatternType::General
        }
    }
}

/// Padrão de mate
struct MatePattern {
    name: &'static str,
    pattern_type: MatePatternType,
    required_pieces: Vec<PieceKind>,
    key_squares: Vec<u8>,
    execution_function: fn(&Board) -> Option<Vec<Move>>,
}

impl MatePattern {
    fn matches_position(&self, board: &Board) -> bool {
        // Verifica se as peças necessárias estão presentes
        let our_pieces = if board.to_move == Color::White {
            board.white_pieces
        } else {
            board.black_pieces
        };

        for &piece_kind in &self.required_pieces {
            let piece_bb = match piece_kind {
                PieceKind::Queen => board.queens,
                PieceKind::Rook => board.rooks,
                PieceKind::Bishop => board.bishops,
                PieceKind::Knight => board.knights,
                PieceKind::King => board.kings,
                PieceKind::Pawn => board.pawns,
            };
            
            if (piece_bb & our_pieces).count_ones() == 0 {
                return false;
            }
        }

        true
    }
}

/// Tipos de padrão de mate
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MatePatternType {
    BackRank,
    Smothered,
    TwoRooks,
    QueenKing,
    BishopKnight,
    Ladder,
    Arabian,
    Greek,
    Anastasia,
    General,
}

/// Sistema de rastreamento de progresso
struct ProgressTracker {
    position_history: Vec<u64>,
    material_history: Vec<i32>,
    evaluation_history: Vec<i32>,
    move_count: u32,
}

impl ProgressTracker {
    fn new() -> Self {
        ProgressTracker {
            position_history: Vec::new(),
            material_history: Vec::new(),
            evaluation_history: Vec::new(),
            move_count: 0,
        }
    }

    fn update_progress(&mut self, board: &Board, _plan: &WinningPlan) {
        self.position_history.push(board.zobrist_hash);
        self.material_history.push(evaluate_material(board));
        self.evaluation_history.push(evaluation::evaluate(board));
        self.move_count += 1;

        // Detecta falta de progresso
        if self.is_stuck() {
            println!("info string Warning: Lack of progress detected in winning position");
        }
    }

    fn is_stuck(&self) -> bool {
        // Verifica se houve progresso nos últimos 10 movimentos
        if self.evaluation_history.len() < 10 {
            return false;
        }

        let recent_evals = &self.evaluation_history[self.evaluation_history.len() - 10..];
        let variation = recent_evals.iter()
            .max()
            .unwrap_or(&0) - recent_evals.iter()
            .min()
            .unwrap_or(&0);

        variation < 50 // Pouca variação indica falta de progresso
    }
}

/// Avaliador de técnica
struct TechniqueEvaluator {
    criteria: Vec<TechniqueCriterion>,
}

impl TechniqueEvaluator {
    fn new() -> Self {
        TechniqueEvaluator {
            criteria: vec![
                TechniqueCriterion::Efficiency,
                TechniqueCriterion::Safety,
                TechniqueCriterion::Simplicity,
                TechniqueCriterion::ForcePlay,
            ],
        }
    }

    fn evaluate_sequence(&self, board: &Board, moves: &[Move]) -> TechniqueScore {
        let mut score = TechniqueScore::default();

        for criterion in &self.criteria {
            score.add_criterion_score(*criterion, self.evaluate_criterion(board, moves, *criterion));
        }

        score
    }

    fn evaluate_criterion(&self, board: &Board, moves: &[Move], criterion: TechniqueCriterion) -> f32 {
        match criterion {
            TechniqueCriterion::Efficiency => {
                // Menos movimentos = mais eficiente
                1.0 / (moves.len() as f32).max(1.0)
            },
            TechniqueCriterion::Safety => {
                // Evita contra-jogo
                self.evaluate_safety(board, moves)
            },
            TechniqueCriterion::Simplicity => {
                // Prefere linhas diretas
                self.evaluate_simplicity(moves)
            },
            TechniqueCriterion::ForcePlay => {
                // Força respostas do oponente
                self.evaluate_forcing_nature(board, moves)
            },
        }
    }

    fn evaluate_safety(&self, _board: &Board, _moves: &[Move]) -> f32 {
        // Implementação simplificada
        0.8
    }

    fn evaluate_simplicity(&self, moves: &[Move]) -> f32 {
        // Penaliza ramificações complexas
        let unique_pieces = moves.iter()
            .map(|mv| mv.from)
            .collect::<std::collections::HashSet<_>>()
            .len();
        
        1.0 / (unique_pieces as f32).max(1.0)
    }

    fn evaluate_forcing_nature(&self, board: &Board, moves: &[Move]) -> f32 {
        // Conta checks e capturas forçadas
        let forcing_moves = moves.iter()
            .filter(|&&mv| {
                let mut temp = *board;
                let _undo = temp.make_move_fast(mv);
                temp.is_king_in_check(!board.to_move) || board.is_capture(mv)
            })
            .count();
        
        forcing_moves as f32 / moves.len().max(1) as f32
    }
}

// Estruturas auxiliares
#[derive(Debug, Clone, Copy)]
enum AdvantageType {
    Material(i32),
    Positional,
    Tactical,
    Mixed,
}

#[derive(Debug, Clone, Copy)]
enum Technique {
    KingActivity,
    PawnPromotion,
    PieceCoordination,
    SpaceAdvantage,
    TimeManagement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum TechniqueCriterion {
    Efficiency,
    Safety,
    Simplicity,
    ForcePlay,
}

#[derive(Debug, Clone, Copy)]
enum EndgameType {
    PawnEndgame,
    RookEndgame,
    QueenEndgame,
    MinorPieceEndgame,
    Complex,
}

#[derive(Debug, Default)]
struct TechniqueScore {
    total: f32,
    breakdown: HashMap<TechniqueCriterion, f32>,
}

impl TechniqueScore {
    fn add_criterion_score(&mut self, criterion: TechniqueCriterion, score: f32) {
        self.breakdown.insert(criterion, score);
        self.total += score;
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RestrictedArea {
    min_file: u8,
    max_file: u8,
    min_rank: u8,
    max_rank: u8,
}

#[derive(Debug, Clone)]
pub struct MateSequence {
    pub moves: Vec<Move>,
    pub mate_in: u8,
}

// ===== FUNÇÕES AUXILIARES =====

fn creates_tactics(board: &Board, mv: Move) -> bool {
    let mut temp = *board;
    let _undo = temp.make_move_fast(mv);
    
    // Verifica se cria ameaças táticas
    let moves = temp.generate_legal_moves();
    moves.iter().any(|&m| {
        temp.is_capture(m) || {
            let mut temp2 = temp;
            let _undo2 = temp2.make_move_fast(m);
            temp2.is_king_in_check(!temp.to_move)
        }
    })
}

fn is_pawn_move(board: &Board, mv: Move) -> bool {
    let from_bb = 1u64 << mv.from;
    (board.pawns & from_bb) != 0
}

fn has_passed_pawn_on_file(board: &Board, file: u8) -> bool {
    let file_mask = 0x0101010101010101u64 << file;
    let our_pawns = board.pawns & if board.to_move == Color::White {
        board.white_pieces
    } else {
        board.black_pieces
    };
    let enemy_pawns = board.pawns & if board.to_move == Color::White {
        board.black_pieces
    } else {
        board.white_pieces
    };

    // Verifica se há peão nosso na coluna
    if (our_pawns & file_mask) == 0 {
        return false;
    }

    // Verifica se não há peão inimigo bloqueando
    let adjacent_files = if file > 0 && file < 7 {
        (0x0101010101010101u64 << (file - 1)) | (0x0101010101010101u64 << (file + 1))
    } else if file == 0 {
        0x0101010101010101u64 << 1
    } else {
        0x0101010101010101u64 << 6
    };

    // Simplificação: considera passado se não há peão inimigo na coluna ou adjacentes
    (enemy_pawns & (file_mask | adjacent_files)) == 0
}

fn restricts_enemy_mobility(board: &Board, mv: Move) -> bool {
    // Conta mobilidade inimiga antes e depois do movimento
    let original_mobility = count_enemy_mobility(board);
    
    let mut temp = *board;
    let _undo = temp.make_move_fast(mv);
    temp.to_move = !temp.to_move; // Troca turno para contar mobilidade inimiga
    
    let new_mobility = count_enemy_mobility(&temp);
    
    new_mobility < original_mobility
}

fn count_enemy_mobility(board: &Board) -> u32 {
    board.generate_legal_moves().len() as u32
}

fn evaluate_technique_move(board: &Board, mv: Move, technique: Technique) -> i32 {
    match technique {
        Technique::KingActivity => {
            // Favorece movimentos de rei em direção ao centro
            if is_king_move(board, mv) {
                let centrality = calculate_centrality(mv.to);
                centrality * 10
            } else {
                0
            }
        },
        Technique::PawnPromotion => {
            // Favorece avanços de peão
            if is_pawn_move(board, mv) {
                let advance_bonus = if board.to_move == Color::White {
                    (mv.to / 8) as i32 * 10
                } else {
                    (7 - mv.to / 8) as i32 * 10
                };
                advance_bonus
            } else {
                0
            }
        },
        Technique::PieceCoordination => {
            // Favorece movimentos que coordenam peças
            if improves_piece_coordination(board, mv) {
                50
            } else {
                0
            }
        },
        Technique::SpaceAdvantage => {
            // Favorece expansão territorial
            if expands_territory(board, mv) {
                30
            } else {
                0
            }
        },
        Technique::TimeManagement => {
            // Favorece movimentos forçados
            if is_forcing_move(board, mv) {
                40
            } else {
                0
            }
        },
    }
}

fn is_king_move(board: &Board, mv: Move) -> bool {
    let from_bb = 1u64 << mv.from;
    (board.kings & from_bb) != 0
}

fn calculate_centrality(square: u8) -> i32 {
    let file = square % 8;
    let rank = square / 8;
    let center_dist = ((file as i32 - 3).abs() + (rank as i32 - 3).abs()) / 2;
    4 - center_dist
}

fn improves_piece_coordination(board: &Board, mv: Move) -> bool {
    // Implementação simplificada
    let mut temp = *board;
    let _undo = temp.make_move_fast(mv);
    
    // Verifica se peça fica mais próxima de outras peças aliadas
    let our_pieces = if board.to_move == Color::White {
        temp.white_pieces
    } else {
        temp.black_pieces
    };
    
    let piece_bb = 1u64 << mv.to;
    let adjacent_mask = get_adjacent_squares_mask(mv.to);
    
    (our_pieces & adjacent_mask).count_ones() >= 2
}

fn get_adjacent_squares_mask(square: u8) -> u64 {
    let file = square % 8;
    let rank = square / 8;
    let mut mask = 0u64;
    
    for df in -1..=1 {
        for dr in -1..=1 {
            if df == 0 && dr == 0 { continue; }
            
            let new_file = file as i32 + df;
            let new_rank = rank as i32 + dr;
            
            if new_file >= 0 && new_file < 8 && new_rank >= 0 && new_rank < 8 {
                mask |= 1u64 << (new_rank * 8 + new_file);
            }
        }
    }
    
    mask
}

fn expands_territory(board: &Board, mv: Move) -> bool {
    // Implementação simplificada: verifica se move avança em direção ao território inimigo
    let our_color = board.to_move;
    let advance = match our_color {
        Color::White => (mv.to / 8) as i32 - (mv.from / 8) as i32,
        Color::Black => (mv.from / 8) as i32 - (mv.to / 8) as i32,
    };
    
    advance > 0
}

fn is_forcing_move(board: &Board, mv: Move) -> bool {
    if board.is_capture(mv) {
        return true;
    }
    
    let mut temp = *board;
    let _undo = temp.make_move_fast(mv);
    
    temp.is_king_in_check(!board.to_move)
}

fn evaluate_material(board: &Board) -> i32 {
    let white_material = count_material(board, Color::White);
    let black_material = count_material(board, Color::Black);
    white_material - black_material
}

fn count_material(board: &Board, color: Color) -> i32 {
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    
    let queens = (board.queens & pieces).count_ones() as i32 * 900;
    let rooks = (board.rooks & pieces).count_ones() as i32 * 500;
    let bishops = (board.bishops & pieces).count_ones() as i32 * 330;
    let knights = (board.knights & pieces).count_ones() as i32 * 320;
    let pawns = (board.pawns & pieces).count_ones() as i32 * 100;
    
    queens + rooks + bishops + knights + pawns
}

// Funções de execução de padrões de mate (implementações básicas)
fn execute_back_rank_mate(board: &Board) -> Option<Vec<Move>> {
    // Implementação simplificada para mate na última fileira
    let moves = board.generate_legal_moves();
    
    for mv in moves {
        let mut temp = *board;
        let _undo = temp.make_move_fast(mv);
        
        if temp.is_checkmate() {
            let target_rank = if board.to_move == Color::White { 7 } else { 0 };
            if mv.to / 8 == target_rank {
                return Some(vec![mv]);
            }
        }
    }
    
    None
}

fn execute_smothered_mate(_board: &Board) -> Option<Vec<Move>> {
    // Implementação stub para mate abafado
    None
}

fn execute_two_rooks_mate(_board: &Board) -> Option<Vec<Move>> {
    // Implementação stub para mate com duas torres
    None
}

fn execute_queen_king_mate(_board: &Board) -> Option<Vec<Move>> {
    // Implementação stub para mate com dama e rei
    None
}

fn evaluate_mate_move(board: &Board, mv: Move, pattern: MatePatternType) -> i32 {
    let mut score = 0;
    
    // Verifica se move dá xeque
    let mut temp = *board;
    let _undo = temp.make_move_fast(mv);
    
    if temp.is_checkmate() {
        score += 10000; // Mate imediato
    } else if temp.is_king_in_check(!board.to_move) {
        score += 500; // Xeque
    }
    
    // Bônus específico por padrão
    match pattern {
        MatePatternType::BackRank => {
            if is_back_rank_relevant(board, mv) {
                score += 200;
            }
        },
        MatePatternType::Smothered => {
            if is_knight_move(board, mv) {
                score += 150;
            }
        },
        _ => {},
    }
    
    score
}

fn is_back_rank_relevant(board: &Board, mv: Move) -> bool {
    let target_rank = if board.to_move == Color::White { 7 } else { 0 };
    mv.to / 8 == target_rank
}

fn is_knight_move(board: &Board, mv: Move) -> bool {
    let from_bb = 1u64 << mv.from;
    (board.knights & from_bb) != 0
}