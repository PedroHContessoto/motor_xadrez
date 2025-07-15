// Self-play system for training NNUE
use crate::board::Board;
use crate::types::*;
use crate::nnue::{NNUEEvaluator, NNUETrainer, NNUEConfig, GameResult, game_result_to_training_value};
use rand::Rng;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Configuração para self-play
#[derive(Debug, Clone)]
pub struct SelfPlayConfig {
    pub games_per_iteration: usize,
    pub max_game_length: usize,
    pub exploration_rate: f32,
    pub temperature: f32,
    pub batch_size: usize,
    pub curriculum_learning: bool,
    pub quiet_positions_only: bool,
    pub nnue_config: NNUEConfig,
    pub gpu_batch_size: usize,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        let nnue_config = NNUEConfig::default();
        let batch_size = nnue_config.batch_size;
        let gpu_batch_size = if nnue_config.use_gpu { 512 } else { 64 };
        
        SelfPlayConfig {
            games_per_iteration: 1000,
            max_game_length: 400,
            exploration_rate: 0.1,
            temperature: 1.0,
            batch_size,
            curriculum_learning: true,
            quiet_positions_only: false, // Começa false, será ativado nas primeiras iterações
            nnue_config,
            gpu_batch_size,
        }
    }
}

/// Dados de uma posição de treino
#[derive(Debug, Clone)]
pub struct TrainingPosition {
    pub board: Board,
    pub result: GameResult,
    pub side_to_move: Color,
}

/// Motor de self-play
pub struct SelfPlayEngine {
    evaluators: Vec<Arc<NNUEEvaluator>>, // Pool de evaluators para threads paralelas
    trainer: Arc<Mutex<NNUETrainer>>,
    config: SelfPlayConfig,
}

impl SelfPlayEngine {
    pub fn new(config: SelfPlayConfig) -> candle_core::Result<Self> {
        let num_threads = rayon::current_num_threads();
        
        // Criar pool de evaluators com GPU config (um por thread)
        let mut evaluators = Vec::with_capacity(num_threads);
        for _ in 0..num_threads {
            evaluators.push(Arc::new(NNUEEvaluator::new_with_config(config.nnue_config.clone())?));
        }
        
        let trainer = Arc::new(Mutex::new(NNUETrainer::new()?));
        
        println!("SelfPlayEngine initialized with {} threads, GPU: {}", 
                num_threads, config.nnue_config.use_gpu);
        
        Ok(SelfPlayEngine {
            evaluators,
            trainer,
            config,
        })
    }

    /// Executa uma iteração completa de self-play
    pub fn run_iteration(&self, iteration: usize) -> Result<f32, Box<dyn std::error::Error + Send + Sync>> {
        println!("Starting self-play iteration {}", iteration);
        let start_time = Instant::now();
        
        // Aplicar curriculum learning
        let mut config = self.config.clone();
        if config.curriculum_learning {
            // Primeiras 20 iterações: foco em quiet positions
            config.quiet_positions_only = iteration <= 20;
            // Decaimento da temperatura ao longo das iterações
            config.temperature = (1.0 - (iteration as f32 / 100.0)).max(0.2);
            // Redução da exploration rate
            config.exploration_rate = (0.1 - (iteration as f32 / 200.0)).max(0.02);
        }
        
        // Gerar games em paralelo com config atualizada
        let training_positions = self.generate_training_games_with_config(&config)?;
        
        println!("Generated {} training positions in {:?}", 
                training_positions.len(), start_time.elapsed());
        
        // Treinar modelo com LR scheduler
        let loss = self.train_model_with_iteration(&training_positions, iteration)?;
        
        println!("Iteration {} completed in {:?}, loss: {:.4}", 
                iteration, start_time.elapsed(), loss);
        
        Ok(loss)
    }

    /// Gera games de treino usando self-play
    fn generate_training_games(&self) -> Result<Vec<TrainingPosition>, Box<dyn std::error::Error + Send + Sync>> {
        self.generate_training_games_with_config(&self.config)
    }
    
    /// Gera games de treino usando self-play com config específica
    fn generate_training_games_with_config(&self, config: &SelfPlayConfig) -> Result<Vec<TrainingPosition>, Box<dyn std::error::Error + Send + Sync>> {
        let evaluators = &self.evaluators;
        
        let games: Vec<_> = (0..config.games_per_iteration)
            .into_par_iter()
            .map(|_game_id| {
                let thread_id = rayon::current_thread_index().unwrap_or(0);
                let evaluator_id = thread_id % evaluators.len();
                let evaluator = evaluators[evaluator_id].clone();
                self.play_game_with_evaluator_and_config(evaluator, config)
            })
            .collect::<Result<Vec<_>, _>>()?;
        
        let mut all_positions = Vec::new();
        for game_positions in games {
            all_positions.extend(game_positions);
        }
        
        Ok(all_positions)
    }

    /// Joga uma partida completa (versão original para single-thread)
    fn play_game(&self) -> Result<Vec<TrainingPosition>, Box<dyn std::error::Error + Send + Sync>> {
        let evaluator = self.evaluators[0].clone();
        self.play_game_with_evaluator(evaluator)
    }
    
    /// Joga uma partida completa com evaluator específico (para parallelização)
    fn play_game_with_evaluator(&self, evaluator: Arc<NNUEEvaluator>) -> Result<Vec<TrainingPosition>, Box<dyn std::error::Error + Send + Sync>> {
        self.play_game_with_evaluator_and_config(evaluator, &self.config)
    }
    
    /// Joga uma partida completa com evaluator e config específicos
    fn play_game_with_evaluator_and_config(&self, evaluator: Arc<NNUEEvaluator>, config: &SelfPlayConfig) -> Result<Vec<TrainingPosition>, Box<dyn std::error::Error + Send + Sync>> {
        let mut board = Board::new();
        let mut positions = Vec::new();
        let mut move_count = 0;
        
        while !board.is_game_over() && move_count < config.max_game_length {
            // Filtrar quiet positions no curriculum learning
            if config.quiet_positions_only && !self.is_quiet_position(&board) {
                // Pular posições não-quiet durante curriculum learning
                let legal_moves = board.generate_legal_moves();
                if legal_moves.is_empty() {
                    break;
                }
                // Só fazer movimento sem salvar posição
                let chosen_move = self.select_move_with_evaluator_and_config(&board, &legal_moves, evaluator.clone(), config)?;
                board.make_move(chosen_move);
                move_count += 1;
                continue;
            }
            
            // Salvar posição atual (quiet ou não, dependendo do curriculum)
            positions.push((board, board.to_move));
            
            // Escolher movimento
            let legal_moves = board.generate_legal_moves();
            if legal_moves.is_empty() {
                break;
            }
            
            let chosen_move = self.select_move_with_evaluator_and_config(&board, &legal_moves, evaluator.clone(), config)?;
            board.make_move(chosen_move);
            move_count += 1;
        }
        
        // Determinar resultado
        let result = self.determine_game_result(&board);
        
        // Converter posições para dados de treino
        let training_positions: Vec<TrainingPosition> = positions
            .into_iter()
            .map(|(board, side_to_move)| TrainingPosition {
                board,
                result,
                side_to_move,
            })
            .collect();
        
        Ok(training_positions)
    }

    /// Seleciona movimento usando NNUE + softmax temperature sampling com batch evaluation
    fn select_move(&self, board: &Board, legal_moves: &[Move]) -> Result<Move, Box<dyn std::error::Error + Send + Sync>> {
        let evaluator = self.evaluators[0].clone();
        self.select_move_with_evaluator(board, legal_moves, evaluator)
    }
    
    /// Seleciona movimento com evaluator específico (para parallelização)
    fn select_move_with_evaluator(&self, board: &Board, legal_moves: &[Move], evaluator: Arc<NNUEEvaluator>) -> Result<Move, Box<dyn std::error::Error + Send + Sync>> {
        self.select_move_with_evaluator_and_config(board, legal_moves, evaluator, &self.config)
    }
    
    /// Seleciona movimento com evaluator e config específicos
    fn select_move_with_evaluator_and_config(&self, board: &Board, legal_moves: &[Move], evaluator: Arc<NNUEEvaluator>, config: &SelfPlayConfig) -> Result<Move, Box<dyn std::error::Error + Send + Sync>> {
        let mut rng = rand::thread_rng();
        
        // Exploração aleatória diminui com o tempo
        if rng.gen::<f32>() < config.exploration_rate {
            let random_idx = rng.gen_range(0..legal_moves.len());
            return Ok(legal_moves[random_idx]);
        }
        
        // Batch evaluation para maior eficiência
        let scores = self.batch_evaluate_moves_with_evaluator(board, legal_moves, evaluator)?;
        
        // Softmax sampling com temperature
        let chosen_idx = self.sample_with_softmax_and_config(&scores, &mut rng, config)?;
        Ok(legal_moves[chosen_idx])
    }
    
    /// Avalia todos os movimentos em batch para melhor performance
    fn batch_evaluate_moves(&self, board: &Board, legal_moves: &[Move]) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
        let evaluator = self.evaluators[0].clone();
        self.batch_evaluate_moves_with_evaluator(board, legal_moves, evaluator)
    }
    
    /// Avalia todos os movimentos em batch com evaluator específico (GPU optimized)
    fn batch_evaluate_moves_with_evaluator(&self, board: &Board, legal_moves: &[Move], evaluator: Arc<NNUEEvaluator>) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
        
        if legal_moves.is_empty() {
            return Ok(Vec::new());
        }
        
        // Use GPU batch evaluation for efficiency
        if legal_moves.len() >= 4 && self.config.nnue_config.use_gpu {
            // Prepare all board positions after moves
            let mut temp_boards = Vec::with_capacity(legal_moves.len());
            for &mv in legal_moves {
                let mut temp_board = *board;
                temp_board.make_move(mv);
                temp_boards.push(temp_board);
            }
            
            // GPU batch evaluation
            let batch_scores = evaluator.evaluate_batch(&temp_boards)
                .map_err(|e| format!("GPU batch evaluation error: {}", e))?;
            
            // Adjust scores for side to move
            let mut scores = Vec::with_capacity(legal_moves.len());
            for &raw_score in &batch_scores {
                let adjusted_score = if board.to_move == Color::White { 
                    raw_score 
                } else { 
                    -raw_score 
                };
                scores.push(adjusted_score);
            }
            
            Ok(scores)
        } else {
            // Fallback to single evaluations for small batches or CPU
            let mut scores = Vec::with_capacity(legal_moves.len());
            for &mv in legal_moves {
                let mut temp_board = *board;
                temp_board.make_move(mv);
                
                let score = evaluator.evaluate(&temp_board)
                    .map_err(|e| format!("Single evaluation error: {}", e))?;
                
                let adjusted_score = if board.to_move == Color::White { score } else { -score };
                scores.push(adjusted_score);
            }
            
            Ok(scores)
        }
    }
    
    /// Sample usando softmax temperature
    fn sample_with_softmax(&self, scores: &[f32], rng: &mut rand::rngs::ThreadRng) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        self.sample_with_softmax_and_config(scores, rng, &self.config)
    }
    
    /// Sample usando softmax temperature com config específica
    fn sample_with_softmax_and_config(&self, scores: &[f32], rng: &mut rand::rngs::ThreadRng, config: &SelfPlayConfig) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        if scores.is_empty() {
            return Err("Empty scores vector".into());
        }
        
        // Aplicar temperature e normalizar
        let temperature = config.temperature.max(0.01); // Evitar divisão por zero
        let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        let mut exp_scores = Vec::with_capacity(scores.len());
        let mut sum = 0.0f32;
        
        for &score in scores {
            let exp_val = ((score - max_score) / temperature).exp();
            exp_scores.push(exp_val);
            sum += exp_val;
        }
        
        // Normalizar para probabilidades
        for exp_score in &mut exp_scores {
            *exp_score /= sum;
        }
        
        // Sample usando probabilidades
        let r = rng.gen::<f32>();
        let mut cumulative = 0.0f32;
        
        for (i, &prob) in exp_scores.iter().enumerate() {
            cumulative += prob;
            if r < cumulative {
                return Ok(i);
            }
        }
        
        // Fallback para último movimento
        Ok(scores.len() - 1)
    }
    
    /// Verifica se uma posição é "quiet" (sem xeque, sem capturas óbvias, sem hanging pieces)
    fn is_quiet_position(&self, board: &Board) -> bool {
        // Posição não é quiet se há xeque
        if board.is_king_in_check(board.to_move) {
            return false;
        }
        
        // Posição não é quiet se há capturas imediatas disponíveis
        let legal_moves = board.generate_legal_moves();
        for &mv in &legal_moves {
            if self.is_capture_move(board, mv) {
                return false;
            }
            
            // Check for hanging pieces: if after move, the destination square is attacked
            if self.post_move_attacked(board, mv) {
                return false;
            }
        }
        
        true
    }
    
    /// Verifica se após um movimento, a casa de destino fica atacada (hanging piece)
    fn post_move_attacked(&self, board: &Board, mv: Move) -> bool {
        let mut temp_board = *board;
        temp_board.make_move(mv);
        
        // Check if the destination square is attacked by opponent
        let opponent_color = if board.to_move == Color::White { Color::Black } else { Color::White };
        temp_board.is_square_attacked_by(mv.to, opponent_color)
    }
    
    /// Verifica se um movimento é uma captura (incluindo en passant e promoções)
    fn is_capture_move(&self, board: &Board, mv: Move) -> bool {
        let to_square = mv.to as usize;
        let to_bb = 1u64 << to_square;
        
        // Verifica se há peça oponente na casa de destino
        let opponent_pieces = if board.to_move == Color::White {
            board.black_pieces
        } else {
            board.white_pieces
        };
        
        // Captura normal
        if (opponent_pieces & to_bb) != 0 {
            return true;
        }
        
        // En passant
        if mv.is_en_passant {
            return true;
        }
        
        // Promoções são consideradas táticas (não quiet)
        if mv.promotion.is_some() {
            return true;
        }
        
        false
    }
    
    /// Espelha o tabuleiro horizontalmente para data augmentation
    fn mirror_board_horizontal(&self, board: &Board) -> Option<Board> {
        let mut mirrored = *board;
        
        // Espelhar todas as bitboards
        mirrored.white_pieces = Self::mirror_bitboard(board.white_pieces);
        mirrored.black_pieces = Self::mirror_bitboard(board.black_pieces);
        mirrored.pawns = Self::mirror_bitboard(board.pawns);
        mirrored.knights = Self::mirror_bitboard(board.knights);
        mirrored.bishops = Self::mirror_bitboard(board.bishops);
        mirrored.rooks = Self::mirror_bitboard(board.rooks);
        mirrored.queens = Self::mirror_bitboard(board.queens);
        mirrored.kings = Self::mirror_bitboard(board.kings);
        
        // Recomputar zobrist hash
        mirrored.zobrist_hash = mirrored.compute_zobrist_hash();
        
        Some(mirrored)
    }
    
    /// Espelha uma bitboard horizontalmente com bit operations otimizadas
    fn mirror_bitboard(bb: u64) -> u64 {
        // Flip horizontal usando bit manipulation otimizada
        let mut result = bb;
        // Swap files A<->H, B<->G, C<->F, D<->E
        result = ((result & 0x0F0F0F0F0F0F0F0F) << 4) | ((result & 0xF0F0F0F0F0F0F0F0) >> 4);
        result = ((result & 0x3333333333333333) << 2) | ((result & 0xCCCCCCCCCCCCCCCC) >> 2);
        result = ((result & 0x5555555555555555) << 1) | ((result & 0xAAAAAAAAAAAAAAAA) >> 1);
        result
    }
    
    /// Reverte os bits de um byte
    fn reverse_byte(byte: u8) -> u8 {
        let mut result = 0u8;
        for i in 0..8 {
            if (byte >> i) & 1 == 1 {
                result |= 1 << (7 - i);
            }
        }
        result
    }
    
    /// Espelha o tabuleiro verticalmente
    fn mirror_board_vertical(&self, board: &Board) -> Option<Board> {
        let mut mirrored = *board;
        
        // Espelhar todas as bitboards verticalmente (flip ranks)
        mirrored.white_pieces = Self::flip_ranks(board.white_pieces);
        mirrored.black_pieces = Self::flip_ranks(board.black_pieces);
        mirrored.pawns = Self::flip_ranks(board.pawns);
        mirrored.knights = Self::flip_ranks(board.knights);
        mirrored.bishops = Self::flip_ranks(board.bishops);
        mirrored.rooks = Self::flip_ranks(board.rooks);
        mirrored.queens = Self::flip_ranks(board.queens);
        mirrored.kings = Self::flip_ranks(board.kings);
        
        // Recomputar zobrist hash
        mirrored.zobrist_hash = mirrored.compute_zobrist_hash();
        
        Some(mirrored)
    }
    
    /// Rotaciona o tabuleiro 180 graus
    fn rotate_board_180(&self, board: &Board) -> Option<Board> {
        let mut rotated = *board;
        
        // Rotação 180° = flip vertical + horizontal
        rotated.white_pieces = Self::flip_ranks(Self::mirror_bitboard(board.white_pieces));
        rotated.black_pieces = Self::flip_ranks(Self::mirror_bitboard(board.black_pieces));
        rotated.pawns = Self::flip_ranks(Self::mirror_bitboard(board.pawns));
        rotated.knights = Self::flip_ranks(Self::mirror_bitboard(board.knights));
        rotated.bishops = Self::flip_ranks(Self::mirror_bitboard(board.bishops));
        rotated.rooks = Self::flip_ranks(Self::mirror_bitboard(board.rooks));
        rotated.queens = Self::flip_ranks(Self::mirror_bitboard(board.queens));
        rotated.kings = Self::flip_ranks(Self::mirror_bitboard(board.kings));
        
        // Recomputar zobrist hash
        rotated.zobrist_hash = rotated.compute_zobrist_hash();
        
        Some(rotated)
    }
    
    /// Flip vertical (troca ranks 1-8 por 8-1) com bit operations otimizadas
    fn flip_ranks(bb: u64) -> u64 {
        // Flip vertical usando byte reversal otimizado
        let mut result = bb;
        // Swap rank pairs
        result = ((result & 0x00000000FFFFFFFF) << 32) | ((result & 0xFFFFFFFF00000000) >> 32);
        result = ((result & 0x0000FFFF0000FFFF) << 16) | ((result & 0xFFFF0000FFFF0000) >> 16);
        result = ((result & 0x00FF00FF00FF00FF) << 8)  | ((result & 0xFF00FF00FF00FF00) >> 8);
        result
    }
    
    /// Rotaciona o tabuleiro 90 graus no sentido horário
    fn rotate_board_90(&self, board: &Board) -> Option<Board> {
        let mut rotated = *board;
        
        // Rotação 90° horário: (x,y) -> (y, 7-x)
        rotated.white_pieces = Self::rotate_bitboard_90(board.white_pieces);
        rotated.black_pieces = Self::rotate_bitboard_90(board.black_pieces);
        rotated.pawns = Self::rotate_bitboard_90(board.pawns);
        rotated.knights = Self::rotate_bitboard_90(board.knights);
        rotated.bishops = Self::rotate_bitboard_90(board.bishops);
        rotated.rooks = Self::rotate_bitboard_90(board.rooks);
        rotated.queens = Self::rotate_bitboard_90(board.queens);
        rotated.kings = Self::rotate_bitboard_90(board.kings);
        
        rotated.zobrist_hash = rotated.compute_zobrist_hash();
        Some(rotated)
    }
    
    /// Rotaciona o tabuleiro 270 graus no sentido horário
    fn rotate_board_270(&self, board: &Board) -> Option<Board> {
        let mut rotated = *board;
        
        // Rotação 270° horário: (x,y) -> (7-y, x)
        rotated.white_pieces = Self::rotate_bitboard_270(board.white_pieces);
        rotated.black_pieces = Self::rotate_bitboard_270(board.black_pieces);
        rotated.pawns = Self::rotate_bitboard_270(board.pawns);
        rotated.knights = Self::rotate_bitboard_270(board.knights);
        rotated.bishops = Self::rotate_bitboard_270(board.bishops);
        rotated.rooks = Self::rotate_bitboard_270(board.rooks);
        rotated.queens = Self::rotate_bitboard_270(board.queens);
        rotated.kings = Self::rotate_bitboard_270(board.kings);
        
        rotated.zobrist_hash = rotated.compute_zobrist_hash();
        Some(rotated)
    }
    
    /// Flip de cores (troca brancas por pretas)
    fn color_flip(&self, board: &Board) -> Option<Board> {
        let mut flipped = *board;
        flipped.flip_colors();
        Some(flipped)
    }
    
    /// Rotaciona bitboard 90 graus horário com lookup table otimizada
    fn rotate_bitboard_90(bb: u64) -> u64 {
        // Use lookup table para rotação 90° mais eficiente
        let mut result = 0u64;
        let mut temp_bb = bb;
        
        // Process 8 squares at a time for better performance
        while temp_bb != 0 {
            let square = temp_bb.trailing_zeros() as usize;
            temp_bb &= temp_bb - 1; // Clear lowest set bit
            
            let file = square % 8;
            let rank = square / 8;
            let new_square = file * 8 + (7 - rank);
            result |= 1u64 << new_square;
        }
        result
    }
    
    /// Rotaciona bitboard 270 graus horário com otimização
    fn rotate_bitboard_270(bb: u64) -> u64 {
        let mut result = 0u64;
        let mut temp_bb = bb;
        
        // Process efficiently by clearing set bits
        while temp_bb != 0 {
            let square = temp_bb.trailing_zeros() as usize;
            temp_bb &= temp_bb - 1; // Clear lowest set bit
            
            let file = square % 8;
            let rank = square / 8;
            let new_square = (7 - file) * 8 + rank;
            result |= 1u64 << new_square;
        }
        result
    }

    /// Determina resultado final do jogo
    fn determine_game_result(&self, board: &Board) -> GameResult {
        if board.is_checkmate() {
            match board.to_move {
                Color::White => GameResult::BlackWins,
                Color::Black => GameResult::WhiteWins,
            }
        } else {
            GameResult::Draw
        }
    }

    /// Treina o modelo com posições geradas
    fn train_model(&self, positions: &[TrainingPosition]) -> Result<f32, Box<dyn std::error::Error + Send + Sync>> {
        self.train_model_with_iteration(positions, 0)
    }
    
    /// Treina o modelo com posições geradas e iteração específica (para LR scheduler)
    fn train_model_with_iteration(&self, positions: &[TrainingPosition], iteration: usize) -> Result<f32, Box<dyn std::error::Error + Send + Sync>> {
        let mut trainer = self.trainer.lock().unwrap();
        
        // Atualizar learning rate baseado na iteração
        trainer.update_learning_rate(iteration)
            .map_err(|e| format!("LR scheduler error: {}", e))?;
        
        trainer.clear_training_data();
        
        // Split 90/10 para train/validation
        let split_idx = (positions.len() as f32 * 0.9) as usize;
        let (train_positions, val_positions) = positions.split_at(split_idx);
        
        // Adicionar dados de treino com augmentation
        for position in train_positions {
            let training_value = game_result_to_training_value(position.result, position.side_to_move);
            
            // Posição original
            trainer.add_training_data(position.board, training_value);
            
            // Augmentations: múltiplas transformações para 8x mais data
            if let Some(mirrored_h) = self.mirror_board_horizontal(&position.board) {
                trainer.add_training_data(mirrored_h, training_value);
            }
            
            if let Some(mirrored_v) = self.mirror_board_vertical(&position.board) {
                trainer.add_training_data(mirrored_v, training_value);
            }
            
            if let Some(rotated_180) = self.rotate_board_180(&position.board) {
                trainer.add_training_data(rotated_180, training_value);
            }
            
            if let Some(rotated_90) = self.rotate_board_90(&position.board) {
                trainer.add_training_data(rotated_90, training_value);
            }
            
            if let Some(rotated_270) = self.rotate_board_270(&position.board) {
                trainer.add_training_data(rotated_270, training_value);
            }
            
            if let Some(color_flipped) = self.color_flip(&position.board) {
                // Inverter training_value para color flip
                trainer.add_training_data(color_flipped, 1.0 - training_value);
            }
            
            // Combine color flip + horizontal mirror
            if let Some(color_flipped) = self.color_flip(&position.board) {
                if let Some(combined) = self.mirror_board_horizontal(&color_flipped) {
                    trainer.add_training_data(combined, 1.0 - training_value);
                }
            }
            
            // Cap augmentation to 8x to avoid memory issues
            // Only add extra combines if memory allows
            let data_size = trainer.training_data.len();
            if data_size < 50000 { // Cap if mem high
                // Add color+90 combine
                if let Some(color_flipped) = self.color_flip(&position.board) {
                    if let Some(color_90) = self.rotate_board_90(&color_flipped) {
                        trainer.add_training_data(color_90, 1.0 - training_value);
                    }
                }
            }
        }
        
        // Treinar em batches
        let train_loss = trainer.train_batch(self.config.batch_size)
            .map_err(|e| format!("Training error: {}", e))?;
        
        // Calcular validation loss
        let val_loss = self.compute_validation_loss(&mut trainer, val_positions)?;
        
        // Early stopping: se validation loss aumenta por 5 iterações consecutivas (safe)
        const PATIENCE: usize = 5;
        
        if val_loss < trainer.best_val_loss {
            trainer.best_val_loss = val_loss;
            trainer.patience_counter = 0;
        } else {
            trainer.patience_counter += 1;
        }
        
        if trainer.patience_counter >= PATIENCE && iteration > 20 {
            println!("Early stopping triggered at iteration {} (val_loss={:.4})", 
                    iteration, val_loss);
            // Note: Early stopping logic would need to be handled by caller
        }
        
        // Save model after training
        let model_path = format!("iter_{}.safetensors", iteration);
        if let Err(e) = trainer.save_model(&model_path) {
            println!("Warning: Could not save model {}: {}", model_path, e);
        }
        
        // Save as last.safetensors for next session
        if let Err(e) = trainer.save_model("last.safetensors") {
            println!("Warning: Could not save last.safetensors: {}", e);
        }
        
        // Log progress a cada 10 iterações
        if iteration % 10 == 0 {
            println!("Iteration {}: train_loss={:.4}, val_loss={:.4}, positions={}", 
                    iteration, train_loss, val_loss, positions.len());
        }
        
        Ok(train_loss)
    }
    
    /// Calcula validation loss para monitorar overfitting
    fn compute_validation_loss(&self, trainer: &mut crate::nnue::NNUETrainer, val_positions: &[TrainingPosition]) -> Result<f32, Box<dyn std::error::Error + Send + Sync>> {
        if val_positions.is_empty() {
            return Ok(0.0);
        }
        
        // Adicionar validation data (sem augmentation)
        let original_data = trainer.get_training_data_copy();
        trainer.clear_training_data();
        
        for position in val_positions {
            let training_value = game_result_to_training_value(position.result, position.side_to_move);
            trainer.add_training_data(position.board, training_value);
        }
        
        // Calcular validation loss (sem training)
        let val_loss = trainer.compute_loss_only(self.config.batch_size.min(val_positions.len()))
            .map_err(|e| format!("Validation error: {}", e))?;
        
        // Restaurar training data
        trainer.restore_training_data(original_data);
        
        Ok(val_loss)
    }

    /// Executa múltiplas iterações de treino
    pub fn train_iterations(&self, num_iterations: usize) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        for iteration in 1..=num_iterations {
            let loss = self.run_iteration(iteration)?;
            
            // Log progresso
            if iteration % 10 == 0 {
                println!("Iteration {}/{}, Average loss: {:.4}", 
                        iteration, num_iterations, loss);
            }
        }
        
        Ok(())
    }

    /// Testa o modelo atual contra baseline
    pub fn benchmark_model(&self) -> Result<f32, Box<dyn std::error::Error + Send + Sync>> {
        let test_games = 100;
        let mut wins = 0;
        let mut draws = 0;
        
        for _ in 0..test_games {
            let result = self.play_benchmark_game()?;
            match result {
                GameResult::WhiteWins => wins += 1,
                GameResult::Draw => draws += 1,
                GameResult::BlackWins => {},
            }
        }
        
        let win_rate = wins as f32 / test_games as f32;
        let draw_rate = draws as f32 / test_games as f32;
        
        println!("Benchmark results: {:.1}% wins, {:.1}% draws", 
                win_rate * 100.0, draw_rate * 100.0);
        
        Ok(win_rate + draw_rate * 0.5)
    }

    /// Joga um jogo de benchmark contra baseline simples
    fn play_benchmark_game(&self) -> Result<GameResult, Box<dyn std::error::Error + Send + Sync>> {
        let mut board = Board::new();
        let mut move_count = 0;
        
        while !board.is_game_over() && move_count < 200 {
            let legal_moves = board.generate_legal_moves();
            if legal_moves.is_empty() {
                break;
            }
            
            let chosen_move = if board.to_move == Color::White {
                // NNUE jogando com brancas
                let evaluator = self.evaluators[0].clone();
                self.select_move_with_evaluator(&board, &legal_moves, evaluator)?
            } else {
                // Baseline simples para pretas (movimento aleatório)
                let mut rng = rand::thread_rng();
                legal_moves[rng.gen_range(0..legal_moves.len())]
            };
            
            board.make_move(chosen_move);
            move_count += 1;
        }
        
        Ok(self.determine_game_result(&board))
    }
}

/// Função principal para executar self-play
pub fn run_selfplay_training(iterations: usize) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let config = SelfPlayConfig::default();
    let engine = SelfPlayEngine::new(config)?;
    
    println!("Starting NNUE self-play training for {} iterations", iterations);
    
    // Benchmark inicial
    let initial_score = engine.benchmark_model()?;
    println!("Initial benchmark score: {:.3}", initial_score);
    
    // Executar treino
    engine.train_iterations(iterations)?;
    
    // Benchmark final
    let final_score = engine.benchmark_model()?;
    println!("Final benchmark score: {:.3}", final_score);
    println!("Improvement: {:.3}", final_score - initial_score);
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_selfplay_game() {
        let config = SelfPlayConfig {
            games_per_iteration: 1,
            max_game_length: 50,
            exploration_rate: 0.5,
            temperature: 1.0,
            batch_size: 16,
        };
        
        let engine = SelfPlayEngine::new(config).unwrap();
        let positions = engine.play_game().unwrap();
        
        assert!(!positions.is_empty());
        assert!(positions.len() <= 50);
    }

    #[test]
    fn test_move_selection() {
        let config = SelfPlayConfig::default();
        let engine = SelfPlayEngine::new(config).unwrap();
        
        let board = Board::new();
        let legal_moves = board.generate_legal_moves();
        
        let chosen_move = engine.select_move(&board, &legal_moves).unwrap();
        assert!(legal_moves.contains(&chosen_move));
    }
}