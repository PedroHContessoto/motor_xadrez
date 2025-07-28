use std::time::Instant;
use crate::{board::Board, evaluation, transposition::{TranspositionTable, EntryType}, types::Move};
use super::{SearchContext, quiescence::quiescence_search, ordering::order_moves, see::{see_threshold, see}};

const MATE_VALUE: i32 = 99999;
const FUTILITY_MARGIN: [i32; 8] = [0, 150, 250, 400, 600, 800, 1000, 1200]; // Mais agressivo
const REVERSE_FUTILITY_MARGIN: [i32; 8] = [0, 100, 200, 300, 400, 500, 600, 700]; // Mais agressivo
// === LMR AGRESSIVO E ADAPTATIVO (FÓRMULA LOGARÍTMICA) ===
const LMR_MIN_DEPTH: u8 = 1; // Mais agressivo - LMR desde profundidade 1
const LMR_MIN_MOVES: usize = 1; // Mais agressivo - LMR desde o segundo movimento
const PROBCUT_DEPTH: u8 = 5;
const PROBCUT_MARGIN: i32 = 200;
const SINGULAR_EXTENSION_DEPTH: u8 = 6;
const ASPIRATION_WINDOW: i32 = 25;

/// Principal Variation Search com melhorias para táticas
pub fn pvs_search(
    board: &Board,
    depth: u8,
    mut alpha: i32,
    mut beta: i32,
    tt: &mut TranspositionTable,
    context: &mut SearchContext,
    start_time: Instant,
    max_time_ms: u64,
    is_pv_node: bool
) -> i32 {
    pvs_search_internal(board, depth, alpha, beta, tt, context, start_time, max_time_ms, is_pv_node, 0)
}

/// Internal PVS search with ply tracking
fn pvs_search_internal(
    board: &Board,
    depth: u8,
    mut alpha: i32,
    mut beta: i32,
    tt: &mut TranspositionTable,
    context: &mut SearchContext,
    start_time: Instant,
    max_time_ms: u64,
    is_pv_node: bool,
    ply: usize
) -> i32 {
    // Proteção absoluta contra explosão de nós (reduzido de 50 para 40)
    if ply > 40 {
        return crate::evaluation::evaluate(board);
    }
    
    context.nodes_searched += 1;

    if context.nodes_searched & 2047 == 0 {
        if start_time.elapsed().as_millis() as u64 > max_time_ms {
            context.should_stop = true;
        }
    }

    if context.should_stop {
        return 0;
    }

    if depth > 64 {
        return evaluation::evaluate_with_depth(board, depth.saturating_add(ply as u8));
    }

    let original_alpha = alpha;
    let mut tt_move: Option<Move> = None;

    if let Some(entry) = tt.probe(board.zobrist_hash) {
        tt_move = entry.best_move;
        if entry.depth >= depth && !is_pv_node {
            match entry.entry_type {
                EntryType::Exact => return entry.score,
                EntryType::LowerBound => alpha = alpha.max(entry.score),
                EntryType::UpperBound => beta = beta.min(entry.score),
            }
            if alpha >= beta {
                return entry.score;
            }
        }
    }

    if board.is_draw_by_50_moves() || board.is_draw_by_insufficient_material() {
        return 0;
    }

    // IID (Internal Iterative Deepening) para melhorar a ordenação quando não há lance da TT
    if is_pv_node && depth >= 6 && tt_move.is_none() {
        let iid_depth = if depth > 8 { depth - 4 } else { depth - 2 }; // Redução adaptativa
        pvs_search_internal(
            board, iid_depth, alpha, beta, tt, context, start_time, max_time_ms, false, ply
        );
        // Após a busca IID, a TT deve ter um lance para esta posição
        if let Some(entry) = tt.probe(board.zobrist_hash) {
            tt_move = entry.best_move;
        }
    }

    if depth == 0 {
        return quiescence_search(board, alpha, beta, tt, context);
    }

    let in_check = board.is_king_in_check(board.to_move);
    let static_eval = if !in_check { evaluation::evaluate_with_depth(board, depth.saturating_add(ply as u8)) } else { -MATE_VALUE / 2 };

    // Reverse Futility Pruning (Static Null Move Pruning)
    if !is_pv_node && !in_check && depth <= 7 && static_eval != -MATE_VALUE / 2 {
        let rfp_margin = REVERSE_FUTILITY_MARGIN[depth as usize];
        if static_eval - rfp_margin >= beta {
            return static_eval - rfp_margin;
        }
    }

    // Razoring (melhorada)
    if !is_pv_node && !in_check && depth <= 3 && static_eval + FUTILITY_MARGIN[depth as usize] < alpha {
        let razor_score = quiescence_search(board, alpha, beta, tt, context);
        if razor_score <= alpha {
            return razor_score;
        }
    }

    // === PROBCUT APRIMORADO ===
    if !is_pv_node && depth >= PROBCUT_DEPTH && static_eval >= beta && beta < MATE_VALUE - 100 {
        let probcut_beta = beta + PROBCUT_MARGIN;
        let probcut_depth = depth - 4;
        
        // Gera movimentos táticos para ProbCut
        let mut probcut_moves = board.generate_legal_moves().into_iter()
            .filter(|mv| {
                board.is_capture(*mv) || 
                gives_check_fast(board, *mv) ||
                mv.promotion.is_some()
            })
            .collect::<Vec<_>>();
            
        // Ordena movimentos por SEE para ProbCut
        probcut_moves.sort_by(|a, b| {
            let see_a = if board.is_capture(*a) { see(board, *a) } else { 0 };
            let see_b = if board.is_capture(*b) { see(board, *b) } else { 0 };
            see_b.cmp(&see_a)
        });
            
        for mv in probcut_moves.iter().take(6) { // Aumentado para 6 movimentos
            let mut temp_board = *board;
            let _undo_info = temp_board.make_move_fast(*mv);
            
            let probcut_score = -pvs_search_internal(
                &temp_board, probcut_depth, -probcut_beta, -probcut_beta + 1, 
                tt, context, start_time, max_time_ms, false, ply + 1
            );
            
            if probcut_score >= probcut_beta {
                // Verificação adicional para reduzir falsos positivos
                if depth >= 8 {
                    let verify_score = -pvs_search_internal(
                        &temp_board, probcut_depth + 1, -probcut_beta, -probcut_beta + 1,
                        tt, context, start_time, max_time_ms, false, ply + 1
                    );
                    if verify_score >= probcut_beta {
                        return probcut_score;
                    }
                } else {
                    return probcut_score;
                }
            }
        }
    }

    // === NULL MOVE PRUNING AVANÇADO COM VERIFICAÇÃO ===
    if should_try_null_move(board, depth, is_pv_node, in_check, static_eval, beta, ply, context) {
        let null_move_result = perform_advanced_null_move_search(
            board, depth, alpha, beta, static_eval, tt, context, start_time, max_time_ms, ply
        );
        
        match null_move_result {
            NullMoveResult::Cutoff(score) => return score,
            NullMoveResult::Continue => {}, // Continua busca normal
            NullMoveResult::ThreatDetected => {
                // Threat detected - reduz menos os próximos movimentos
                context.set_threat_detected(true);
            }
        }
    }

    // OTIMIZAÇÃO: Verificação rápida de mate/stalemate sem gerar todos os movimentos
    // Só gera movimentos se realmente precisar para a busca
    let legal_moves = if in_check {
        // Em xeque: precisa verificar se há movimentos legais (mate?)
        let moves = board.generate_legal_moves();
        if moves.is_empty() {
            return -(MATE_VALUE - ply as i32); // Mate
        }
        moves
    } else {
        // Não em xeque: assume que há movimentos legais (stalemate é raro)
        // Só gera quando realmente precisar para ordenação
        board.generate_legal_moves()
    };

    // Verificação adicional para stalemate apenas se não em xeque
    if !in_check && legal_moves.is_empty() {
        return 0; // Stalemate
    }

    // === SINGULAR EXTENSIONS ===
    let mut singular_extension = 0;
    // Extensões singulares mais restritivas
    if depth >= SINGULAR_EXTENSION_DEPTH && tt_move.is_some() && !is_pv_node && ply <= 15 && depth <= 8 {
        if let Some(singular_ext) = evaluate_singular_extension(
            board, tt_move.unwrap(), depth, alpha, beta, tt, context, start_time, max_time_ms, ply
        ) {
            singular_extension = singular_ext;
        }
    }

    // Passa o tt_move para a função de ordenação
    let ordered_moves = order_moves(board, legal_moves, tt, context, depth);
    let mut best_move = None;
    let mut best_score = -50000;
    let mut moves_searched = 0;
    let mut tried_moves = Vec::with_capacity(ordered_moves.len());

    // === MULTI-CUT PRUNING APRIMORADO ===
    // Reativando multicut pruning 
    // Detecta posições onde múltiplos movimentos causam beta cutoff
    if should_try_multicut_enhanced(depth, is_pv_node, in_check, &ordered_moves, static_eval, beta) {
        let multicut_result = evaluate_multicut_pruning_enhanced(
            board, &ordered_moves, depth, alpha, beta, static_eval, tt, context, start_time, max_time_ms, ply
        );
        
        if multicut_result >= 3 {
            // Múltiplos movimentos causam cutoff - posição é muito boa
            return beta;
        } else if multicut_result >= 2 {
            // Reduz futility margins devido à posição tática boa
            // (implementação futura)
        }
    }

    for mv in &ordered_moves {
        let mut temp_board = *board;
        let _undo_info = temp_board.make_move_fast(*mv);

        context.push_move(*mv);

        let gives_check = temp_board.is_king_in_check(!board.to_move);
        let is_capture = board.is_capture(*mv);

        // Extended Futility Pruning (melhorada)
        if depth <= 6 && !is_pv_node && !in_check && !gives_check && moves_searched > 0 {
            let futility_value = static_eval + FUTILITY_MARGIN[depth as usize];
            if futility_value <= alpha && !is_capture && mv.promotion.is_none() {
                // Exceção para movimentos que podem melhorar a posição significativamente
                if !is_killer_or_counter_move(*mv, context, depth) {
                    moves_searched += 1;
                    context.pop_move();
                    continue;
                }
            }
        }

        // Late Move Count Pruning
        if depth <= 8 && !is_pv_node && !in_check && !gives_check && !is_capture 
            && mv.promotion.is_none() && moves_searched >= late_move_count_threshold(depth) {
            moves_searched += 1;
            context.pop_move();
            continue;
        }

        // SEE Pruning
        if !is_pv_node && is_capture && depth <= 4 && moves_searched > 0 {
            if !see_threshold(board, *mv, -200) {
                context.pop_move();
                continue;
            }
        }

        // === SISTEMA DE EXTENSÕES COM ORÇAMENTO CONTROLADO ===
        
        // Define orçamento máximo baseado na profundidade atual
        let extension_budget = if ply > 25 {
            0 // Sem extensões em PLY muito alto
        } else if ply > 20 {
            1 // Orçamento mínimo
        } else if ply > 15 {
            2 // Orçamento baixo
        } else if ply > 10 {
            3 // Orçamento médio
        } else {
            4 // Orçamento alto apenas em PLY baixo
        };
        
        // Calcula extensões por prioridade (não acumula, escolhe a melhor)
        let mut extension_candidates = Vec::new();
        
        // 1. PRIORIDADE MÁXIMA: Checks em posições críticas de mate
        if gives_check && in_check && depth <= 1 && ply <= 10 {
            extension_candidates.push(("mate_threat", 2));
        }
        
        // 2. PRIORIDADE ALTA: TT move singular
        if Some(*mv) == tt_move && singular_extension > 0 {
            extension_candidates.push(("singular", singular_extension as i32));
        }
        
        // 3. PRIORIDADE ALTA: Checks básicos
        if gives_check {
            extension_candidates.push(("check", 1));
        }
        
        // 4. PRIORIDADE MÉDIA: Promoções
        if mv.promotion.is_some() {
            extension_candidates.push(("promotion", 1));
        }
        
        // 5. PRIORIDADE BAIXA: Recapturas
        if is_recapture(board, *mv, context) {
            extension_candidates.push(("recapture", 1));
        }
        
        // Escolhe a extensão de maior prioridade que cabe no orçamento
        let mut extension = 0i32;
        for (_name, ext_value) in extension_candidates {
            if ext_value <= extension_budget {
                extension = extension.max(ext_value);
            }
        }
        
        // Aplica o orçamento final e converte para u8
        extension = extension.min(extension_budget);
        let extension = extension.max(0) as u8;

        let mut score;

        if moves_searched == 0 {
            let first_depth = depth.saturating_sub(1) + extension;
            score = -pvs_search_internal(&temp_board, first_depth, -beta, -alpha, tt, context, start_time, max_time_ms, is_pv_node, ply + 1);
        } else {
            // === LMR AVANÇADO COM FÓRMULA LOGARÍTMICA E AJUSTES ADAPTATIVOS ===
            let mut reduction: u8 = 0;
            if depth >= LMR_MIN_DEPTH && moves_searched >= LMR_MIN_MOVES && !is_pv_node
                && !is_capture && !gives_check && !in_check && extension == 0 
                && !board.is_king_in_check(board.to_move) {  // Desabilita LMR em posições de xeque

                // Fórmula logarítmica moderna MAIS AGRESSIVA (mais precisa que tabela)
                let log_depth = (depth as f32).ln();
                let log_moves = (moves_searched as f32).ln();
                let mut base_reduction = (log_depth * log_moves / 1.5) as u8; // Mais agressivo - divisor reduzido de 2.0 para 1.5
                
                // === AJUSTES ADAPTATIVOS AVANÇADOS ===
                
                // 1. Histórico de movimentos (mais refinado)
                if let Some(piece_kind) = board.get_piece_on_square(mv.from) {
                    let history = context.get_history_score(*mv, piece_kind);
                    
                    if history > 3000 {
                        base_reduction = base_reduction.saturating_sub(2); // Movimento excepcional
                    } else if history > 1000 {
                        base_reduction = base_reduction.saturating_sub(1); // Movimento bom
                    } else if history < -3000 {
                        base_reduction += 2; // Movimento muito ruim
                    } else if history < -1000 {
                        base_reduction += 1; // Movimento ruim
                    }
                }
                
                // 2. Ajuste para killer moves (reduz menos)
                if context.is_killer(*mv, depth) {
                    base_reduction = base_reduction.saturating_sub(1);
                }
                
                // 3. Counter-moves (movimentos que refutam o anterior)
                if let Some(last_move) = context.get_last_move() {
                    if let Some(counter) = context.get_counter_move(last_move) {
                        if counter == *mv {
                            base_reduction = base_reduction.saturating_sub(1);
                        }
                    }
                }
                
                // 4. Redução baseada na complexidade posicional
                let tactical_level = evaluate_tactical_complexity(board);
                match tactical_level {
                    3 => base_reduction = base_reduction.saturating_sub(2), // Posição muito tática
                    2 => base_reduction = base_reduction.saturating_sub(1), // Posição tática
                    0 => base_reduction += 1, // Posição muito calma
                    _ => {} // Posição normal
                }
                
                // 5. Ajuste para peças específicas
                let from_sq_bit = 1u64 << mv.from;
                if (board.knights & from_sq_bit) != 0 || (board.bishops & from_sq_bit) != 0 {
                    if tactical_level >= 2 {
                        base_reduction = base_reduction.saturating_sub(1);
                    }
                }
                
                // 6. Penalização para movimentos muito tardios
                if moves_searched >= 20 {
                    base_reduction += 1;
                } else if moves_searched >= 40 {
                    base_reduction += 2;
                }
                
                // 7. Ajuste para endgame (menos redução)
                let total_pieces = (board.white_pieces | board.black_pieces).count_ones();
                if total_pieces <= 12 {
                    base_reduction = base_reduction.saturating_sub(1);
                }
                
                // 8. Ajuste para PV nodes
                if is_pv_node {
                    base_reduction = base_reduction.saturating_sub(1);
                }
                
                // 9. Limitações finais
                reduction = base_reduction.clamp(1, depth.saturating_sub(1).max(1));
            }

            let search_depth = depth.saturating_sub(1 + reduction) + extension;
            score = -pvs_search_internal(&temp_board, search_depth, -alpha - 1, -alpha, tt, context, start_time, max_time_ms, false, ply + 1);

            if score > alpha && (is_pv_node || reduction > 0) {
                let full_depth = depth.saturating_sub(1) + extension;
                score = -pvs_search_internal(&temp_board, full_depth, -beta, -alpha, tt, context, start_time, max_time_ms, is_pv_node, ply + 1);
            }
        }

        moves_searched += 1;
        tried_moves.push(*mv);
        context.pop_move();

        if score > best_score {
            best_score = score;
            best_move = Some(*mv);

            if is_pv_node {
                context.update_pv(ply, *mv);
            }
        }

        alpha = alpha.max(score);

        if alpha >= beta {
            use super::ordering::update_context_on_cutoff;
            update_context_on_cutoff(context, board, *mv, depth, &tried_moves);
            break;
        }
    }

    let entry_type = if best_score <= original_alpha {
        EntryType::UpperBound
    } else if best_score >= beta {
        EntryType::LowerBound
    } else {
        EntryType::Exact
    };

    tt.store(board.zobrist_hash, best_move, best_score, depth, entry_type);
    best_score
}

/// Detecta se movimento é uma recaptura
fn is_recapture(board: &Board, mv: Move, context: &SearchContext) -> bool {
    if let Some(last_move) = context.get_last_move() {
        board.is_capture(mv) && mv.to == last_move.to
    } else {
        false
    }
}

/// Verifica se há material não-peão (necessário para null move)
fn has_non_pawn_material(board: &Board) -> bool {
    let our_pieces = if board.to_move == crate::types::Color::White { 
        board.white_pieces 
    } else { 
        board.black_pieces 
    };
    
    (our_pieces & (board.knights | board.bishops | board.rooks | board.queens)) != 0
}

/// Detecção rápida de check sem fazer o movimento
fn gives_check_fast(board: &Board, mv: Move) -> bool {
    // Implementação simplificada - pode ser melhorada
    let mut temp_board = *board;
    let _undo_info = temp_board.make_move_fast(mv);
    temp_board.is_king_in_check(!board.to_move)
}

/// Verifica se movimento é killer ou counter-move
fn is_killer_or_counter_move(mv: Move, context: &SearchContext, depth: u8) -> bool {
    context.is_killer(mv, depth) || {
        if let Some(last_move) = context.get_last_move() {
            if let Some(counter) = context.get_counter_move(last_move) {
                counter == mv
            } else {
                false
            }
        } else {
            false
        }
    }
}

/// Threshold para Late Move Count Pruning - SUPER AGRESSIVO
fn late_move_count_threshold(depth: u8) -> usize {
    // Threshold MUITO mais agressivo para late moves
    match depth {
        1 => 2,   // Extremamente agressivo - só 2 movimentos!
        2 => 3,   // Muito agressivo
        3 => 5,   // Reduzido ainda mais  
        4 => 8,   // Significativamente reduzido
        5 => 12,  // Reduzido
        6 => 16,  // Reduzido
        7 => 20,  // Reduzido
        _ => 24,  // Reduzido
    }
}

// ============================================================================
// FUNÇÕES AUXILIARES PARA LMR LOGARÍTMICO (SUBSTITUI TABELA ANTIGA)
// ============================================================================

/// Avalia complexidade tática da posição (0-3)
fn evaluate_tactical_complexity(board: &Board) -> u8 {
    let mut complexity = 0u8;
    
    // 1. Verifica xeques
    if board.is_king_in_check(board.to_move) {
        complexity += 2;
    }
    
    // 2. Peças atacadas
    let our_pieces = if board.to_move == crate::types::Color::White { 
        board.white_pieces 
    } else { 
        board.black_pieces 
    };
    
    let valuable_pieces = (board.queens | board.rooks | board.bishops | board.knights) & our_pieces;
    let mut attacked_valuable = 0;
    let mut bb = valuable_pieces;
    
    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        
        if board.is_square_attacked_by(sq, !board.to_move) {
            attacked_valuable += 1;
        }
    }
    
    if attacked_valuable >= 2 {
        complexity += 2;
    } else if attacked_valuable >= 1 {
        complexity += 1;
    }
    
    // 3. Densidade de peças (posições congestionadas são mais táticas)
    let total_pieces = (board.white_pieces | board.black_pieces).count_ones();
    if total_pieces >= 28 {
        complexity += 1;
    }
    
    complexity.min(3)
}

/// Detecta posições táticas que precisam de busca mais profunda
fn is_tactical_position(board: &Board) -> bool {
    // Verifica se há peças penduradas ou em xeque
    let in_check = board.is_king_in_check(board.to_move);
    if in_check {
        return true;
    }
    
    // Conta peças atacadas
    let our_pieces = if board.to_move == crate::types::Color::White { 
        board.white_pieces 
    } else { 
        board.black_pieces 
    };
    
    let valuable_pieces = (board.knights | board.bishops | board.rooks | board.queens) & our_pieces;
    let mut attacked_count = 0;
    let mut bb = valuable_pieces;
    
    while bb != 0 && attacked_count < 3 {
        let sq = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        
        if board.is_square_attacked_by(sq, !board.to_move) {
            attacked_count += 1;
        }
    }
    
    attacked_count >= 2
}

/// Versão simplificada para análise
pub fn analyze_position(board: &Board, depth: u8, tt: &mut TranspositionTable) -> (i32, Option<Move>) {
    let mut context = SearchContext::new();
    let start_time = Instant::now();
    let max_time = 30000; // 30 segundos

    let score = pvs_search(board, depth, -50000, 50000, tt, &mut context, start_time, max_time, true);

    let best_move = if let Some(entry) = tt.probe(board.zobrist_hash) {
        entry.best_move
    } else {
        None
    };

    (score, best_move)
}

#[allow(dead_code)]
fn has_hanging_pieces_simple(board: &Board) -> bool {
    let our_valuables = (board.knights | board.bishops | board.rooks | board.queens) &
        if board.to_move == crate::types::Color::White { board.white_pieces } else { board.black_pieces };

    let mut bb = our_valuables;
    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        bb &= bb - 1;

        if board.is_square_attacked_by(sq, !board.to_move) {
            return true;
        }
    }
    false
}

// ============================================================================
// NULL MOVE PRUNING AVANÇADO COM VERIFICAÇÃO E THREAT DETECTION
// ============================================================================

/// Resultado do null move search avançado
#[derive(Debug, Clone, Copy, PartialEq)]
enum NullMoveResult {
    Cutoff(i32),        // Beta cutoff - pode retornar
    Continue,           // Continua busca normal
    ThreatDetected,     // Threat detectada - ajusta busca
}

/// Configuração avançada para null move
struct NullMoveConfig {
    base_reduction: u8,
    eval_reduction: u8,
    verification_depth: u8,
    threat_threshold: i32,
    use_verification: bool,
    use_double_null: bool,
}

/// Verifica se devemos tentar null move com critérios avançados
fn should_try_null_move(
    board: &Board,
    depth: u8,
    is_pv_node: bool,
    in_check: bool,
    static_eval: i32,
    beta: i32,
    ply: usize,
    context: &SearchContext
) -> bool {
    // Condições básicas
    if depth < 2 || is_pv_node || in_check || static_eval < beta {
        return false;
    }

    // Não fazer null move se já fizemos um recentemente (double null move prevention)
    if context.consecutive_null_moves() >= 1 {
        return false;
    }

    // Verifica se temos material suficiente para null move
    if !has_sufficient_material_for_null_move(board) {
        return false;
    }

    // Não fazer null move em posições táticas críticas
    if is_zugzwang_sensitive_position(board) {
        return false;
    }

    // Verifica se está próximo ao mate (null move pode esconder mate threats)
    if static_eval >= MATE_VALUE - 100 || static_eval <= -MATE_VALUE + 100 {
        return false;
    }

    // Considera histórico de threats
    if context.recent_threat_detected() && depth <= 4 {
        return false;
    }

    true
}

/// Executa null move search avançado com verificação adaptativa
fn perform_advanced_null_move_search(
    board: &Board,
    depth: u8,
    alpha: i32,
    beta: i32,
    static_eval: i32,
    tt: &mut TranspositionTable,
    context: &mut SearchContext,
    start_time: Instant,
    max_time_ms: u64,
    ply: usize
) -> NullMoveResult {
    // Calcula configuração adaptativa
    let config = calculate_null_move_config(board, depth, static_eval, beta);

    // Prepara null move board
    let mut null_board = *board;
    null_board.to_move = !null_board.to_move;
    null_board.en_passant_target = None;
    null_board.halfmove_clock += 1;

    // Atualiza contexto para null move
    context.increment_null_moves();
    
    // Primeiro null move search
    let null_depth = depth.saturating_sub(config.base_reduction + config.eval_reduction);
    let null_score = -pvs_search_internal(
        &null_board, null_depth, -beta, -beta + 1, tt, context, start_time, max_time_ms, false, ply + 1
    );
    
    context.decrement_null_moves();

    if null_score < beta {
        return NullMoveResult::Continue;
    }

    // Beta cutoff detectado - agora vem a verificação avançada
    
    // 1. Verificação simples para profundidades baixas
    if depth < 12 || !config.use_verification {
        return NullMoveResult::Cutoff(beta);
    }

    // 2. Verificação para detectar threats e zugzwang
    let verification_result = perform_null_move_verification(
        board, &config, alpha, beta, null_score, tt, context, start_time, max_time_ms, ply
    );

    // 3. Double null move para posições críticas
    if config.use_double_null && verification_result == NullMoveResult::Cutoff(beta) {
        let double_null_result = perform_double_null_move(
            &null_board, depth, beta, tt, context, start_time, max_time_ms, ply
        );
        
        if double_null_result < beta {
            return NullMoveResult::ThreatDetected;
        }
    }

    verification_result
}

/// Calcula configuração adaptativa do null move baseada na posição
fn calculate_null_move_config(board: &Board, depth: u8, static_eval: i32, beta: i32) -> NullMoveConfig {
    let eval_margin = static_eval - beta;
    
    // Base reduction adaptativa
    let base_reduction = if depth <= 6 {
        3
    } else if depth <= 12 {
        4
    } else {
        4 + (depth - 12) / 6 // Redução maior para profundidades altas
    };

    // Eval reduction baseada na margem
    let eval_reduction = ((eval_margin / 200).min(3).max(0)) as u8;

    // Verification depth
    let verification_depth = if depth >= 16 {
        depth.saturating_sub(base_reduction + 3)
    } else if depth >= 8 {
        depth.saturating_sub(base_reduction + 2)
    } else {
        0
    };

    // Threat threshold adaptativo
    let threat_threshold = if is_endgame_position(board) {
        50  // Mais sensível em endgame
    } else {
        150 // Menos sensível em middlegame
    };

    NullMoveConfig {
        base_reduction,
        eval_reduction,
        verification_depth,
        threat_threshold,
        use_verification: depth >= 8,
        use_double_null: depth >= 14 && eval_margin >= 400,
    }
}

/// Verificação avançada do null move para detectar threats
fn perform_null_move_verification(
    board: &Board,
    config: &NullMoveConfig,
    alpha: i32,
    beta: i32,
    null_score: i32,
    tt: &mut TranspositionTable,
    context: &mut SearchContext,
    start_time: Instant,
    max_time_ms: u64,
    ply: usize
) -> NullMoveResult {
    if config.verification_depth == 0 {
        return NullMoveResult::Cutoff(beta);
    }

    // Busca de verificação com janela reduzida
    let verify_score = pvs_search_internal(
        board, config.verification_depth, beta - 1, beta, tt, context, start_time, max_time_ms, false, ply
    );

    if verify_score >= beta {
        // Verificação confirma cutoff
        return NullMoveResult::Cutoff(beta);
    }

    // Analisa diferença entre null move e verificação
    let threat_margin = null_score - verify_score;
    
    if threat_margin >= config.threat_threshold {
        // Threat significativa detectada
        return NullMoveResult::ThreatDetected;
    }

    // Threat menor ou zugzwang detectado
    NullMoveResult::Continue
}

/// Double null move para detectar threats ocultas
fn perform_double_null_move(
    null_board: &Board,
    depth: u8,
    beta: i32,
    tt: &mut TranspositionTable,
    context: &mut SearchContext,
    start_time: Instant,
    max_time_ms: u64,
    ply: usize
) -> i32 {
    let mut double_null_board = *null_board;
    double_null_board.to_move = !double_null_board.to_move;
    double_null_board.halfmove_clock += 1;

    let double_null_depth = depth.saturating_sub(6);
    
    if double_null_depth <= 0 {
        return beta + 1; // Assume no threat
    }

    context.increment_null_moves();
    let score = -pvs_search_internal(
        &double_null_board, double_null_depth, -beta, -beta + 1, 
        tt, context, start_time, max_time_ms, false, ply + 2
    );
    context.decrement_null_moves();

    score
}

/// Verifica se posição tem material suficiente para null move
fn has_sufficient_material_for_null_move(board: &Board) -> bool {
    let our_pieces = if board.to_move == crate::types::Color::White { 
        board.white_pieces 
    } else { 
        board.black_pieces 
    };
    
    let valuable_pieces = (board.knights | board.bishops | board.rooks | board.queens) & our_pieces;
    
    // Precisa ter pelo menos uma peça valiosa além do rei
    valuable_pieces != 0
}

/// Detecta posições sensíveis a zugzwang
fn is_zugzwang_sensitive_position(board: &Board) -> bool {
    let total_pieces = (board.white_pieces | board.black_pieces).count_ones();
    
    // Endgames com poucos peões são sensíveis a zugzwang
    if total_pieces <= 10 {
        let total_pawns = board.pawns.count_ones();
        if total_pawns <= 4 {
            return true;
        }
    }
    
    // Posições de rei + peão vs rei
    if total_pieces <= 4 {
        return true;
    }
    
    // Verifica padrões específicos de zugzwang
    detect_specific_zugzwang_patterns(board)
}

/// Detecta padrões específicos de zugzwang
fn detect_specific_zugzwang_patterns(board: &Board) -> bool {
    let our_color = board.to_move;
    let our_pieces = if our_color == crate::types::Color::White { 
        board.white_pieces 
    } else { 
        board.black_pieces 
    };
    
    // Rei + bispo de cor errada + peão vs rei
    let our_bishops = board.bishops & our_pieces;
    let our_pawns = board.pawns & our_pieces;
    
    if our_bishops.count_ones() == 1 && our_pawns.count_ones() <= 2 {
        // Verifica se é bispo de cor errada (implementação simplificada)
        return true;
    }
    
    false
}

/// Verifica se é posição de endgame
fn is_endgame_position(board: &Board) -> bool {
    let total_pieces = (board.white_pieces | board.black_pieces).count_ones();
    let major_pieces = (board.queens | board.rooks).count_ones();
    
    total_pieces <= 12 || major_pieces <= 2
}

// ============================================================================
// SINGULAR EXTENSIONS - EXTENSÕES QUANDO MOVIMENTO É CLARAMENTE SUPERIOR
// ============================================================================

/// Constantes para Singular Extensions
const SINGULAR_MARGIN: i32 = 64;           // Margem para considerar movimento singular
const SINGULAR_SEARCH_DEPTH_REDUCTION: u8 = 4; // Aumentado de 3 para 4 - menos agressivo
const SINGULAR_MAX_EXTENSION: u8 = 1;      // Extensão máxima
const MULTICUT_DEPTH: u8 = 8;              // Profundidade mínima para multi-cut

/// Avalia se movimento merece extensão singular
fn evaluate_singular_extension(
    board: &Board,
    tt_move: Move,
    depth: u8,
    alpha: i32,
    beta: i32,
    tt: &mut TranspositionTable,
    context: &mut SearchContext,
    start_time: Instant,
    max_time_ms: u64,
    ply: usize
) -> Option<u8> {
    // Verifica se tempo permite busca adicional
    if start_time.elapsed().as_millis() as u64 > max_time_ms / 2 {
        return None;
    }
    
    // Obtém score da TT para este movimento
    let tt_score = if let Some(entry) = tt.probe(board.zobrist_hash) {
        if entry.depth >= depth.saturating_sub(2) {
            entry.score
        } else {
            return None;
        }
    } else {
        return None;
    };
    
    // Calcula bounds para busca singular
    let singular_beta = (tt_score - SINGULAR_MARGIN).max(alpha);
    let search_depth = depth.saturating_sub(SINGULAR_SEARCH_DEPTH_REDUCTION);
    
    if search_depth <= 0 {
        return None;
    }
    
    // Busca excluindo o movimento da TT para ver se outros movimentos são bons
    let excluded_score = search_without_move(
        board, tt_move, search_depth, alpha, singular_beta, 
        tt, context, start_time, max_time_ms, ply
    )?;
    
    // Analisa resultado para determinar tipo de extensão
    if excluded_score < singular_beta {
        // Movimento é singular - outros movimentos são significativamente piores
        Some(SINGULAR_MAX_EXTENSION)
    } else if excluded_score >= beta && depth >= MULTICUT_DEPTH {
        // Multi-cut detectado - vários movimentos são bons, pode podar
        // Retorna extensão negativa (redução) em alguns casos extremos
        None
    } else {
        // Movimento não é suficientemente singular
        None
    }
}

/// Busca excluindo movimento específico para teste de singularidade
fn search_without_move(
    board: &Board,
    excluded_move: Move,
    depth: u8,
    alpha: i32,
    beta: i32,
    tt: &mut TranspositionTable,
    context: &mut SearchContext,
    start_time: Instant,
    max_time_ms: u64,
    ply: usize
) -> Option<i32> {
    if depth == 0 {
        return Some(quiescence_search(board, alpha, beta, tt, context));
    }
    
    // Gera movimentos excluindo o movimento da TT
    let legal_moves = board.generate_legal_moves();
    let mut moves_to_search = Vec::new();
    
    for mv in legal_moves {
        if mv != excluded_move {
            moves_to_search.push(mv);
        }
    }
    
    if moves_to_search.is_empty() {
        return Some(alpha); // Sem movimentos alternativos
    }
    
    // Ordena movimentos (simplificado para performance)
    let ordered_moves = order_moves_simple(board, moves_to_search, context);
    
    let mut best_score = -50000;
    let mut moves_searched = 0;
    
    for mv in &ordered_moves {
        // Limita número de movimentos para performance
        if moves_searched >= 8 {
            break;
        }
        
        // Verifica timeout
        if context.should_stop || start_time.elapsed().as_millis() as u64 > max_time_ms * 3 / 4 {
            break;
        }
        
        let mut temp_board = *board;
        let _undo_info = temp_board.make_move_fast(*mv);
        
        context.push_move(*mv);
        
        let score = if moves_searched == 0 {
            -pvs_search_internal(&temp_board, depth - 1, -beta, -alpha, tt, context, start_time, max_time_ms, false, ply + 1)
        } else {
            // Scout search com janela zero
            let scout_score = -pvs_search_internal(&temp_board, depth - 1, -alpha - 1, -alpha, tt, context, start_time, max_time_ms, false, ply + 1);
            if scout_score > alpha && scout_score < beta {
                -pvs_search_internal(&temp_board, depth - 1, -beta, -alpha, tt, context, start_time, max_time_ms, false, ply + 1)
            } else {
                scout_score
            }
        };
        
        context.pop_move();
        moves_searched += 1;
        
        best_score = best_score.max(score);
        
        if score >= beta {
            return Some(score); // Beta cutoff
        }
    }
    
    Some(best_score)
}

/// Ordenação simplificada para busca singular (performance)
fn order_moves_simple(board: &Board, moves: Vec<Move>, context: &SearchContext) -> Vec<Move> {
    let mut scored_moves: Vec<(Move, i32)> = moves.into_iter().map(|mv| {
        let mut score = 0;
        
        // Prioriza capturas
        if board.is_capture(mv) {
            score += 1000;
            // Adiciona valor MVV-LVA simplificado
            if let Some(captured) = board.get_piece_on_square(mv.to) {
                score += get_piece_value_simple(captured) * 10;
            }
            if let Some(attacker) = board.get_piece_on_square(mv.from) {
                score -= get_piece_value_simple(attacker);
            }
        }
        
        // Prioriza promoções
        if mv.promotion.is_some() {
            score += 900;
        }
        
        // Killer moves
        if context.is_killer(mv, 0) { // Usa depth 0 como aproximação
            score += 500;
        }
        
        // Historical score simplificado
        if let Some(piece) = board.get_piece_on_square(mv.from) {
            score += context.get_history_score(mv, piece) / 100;
        }
        
        (mv, score)
    }).collect();
    
    // Ordena por score (maior primeiro)
    scored_moves.sort_by(|a, b| b.1.cmp(&a.1));
    
    scored_moves.into_iter().map(|(mv, _)| mv).collect()
}

/// Valores simplificados para ordenação MVV-LVA
fn get_piece_value_simple(piece: crate::types::PieceKind) -> i32 {
    match piece {
        crate::types::PieceKind::Pawn => 1,
        crate::types::PieceKind::Knight => 3,
        crate::types::PieceKind::Bishop => 3,
        crate::types::PieceKind::Rook => 5,
        crate::types::PieceKind::Queen => 9,
        crate::types::PieceKind::King => 100,
    }
}

// ============================================================================
// MULTI-CUT PRUNING - PODA QUANDO MÚLTIPLOS MOVIMENTOS SÃO BONS
// ============================================================================

// ============================================================================ 
// CONSTANTES PARA MULTI-CUT APRIMORADO (SUBSTITUEM AS ANTIGAS)
// ============================================================================
const MULTICUT_CUTOFF_THRESHOLD: usize = 3;
const MULTICUT_SEARCH_DEPTH: u8 = 3;
const MULTICUT_MARGIN: i32 = 100;

/// Multi-cut pruning aprimorado com análise de padrões
fn evaluate_advanced_multicut(
    board: &Board,
    moves: &[Move],
    depth: u8,
    alpha: i32,
    beta: i32,
    tt: &mut TranspositionTable,
    context: &mut SearchContext,
    start_time: Instant,
    max_time_ms: u64,
    ply: usize
) -> MulticutAdvancedResult {
    let mut result = MulticutAdvancedResult::new();
    
    // Análise em duas fases para melhor precisão
    let phase1_moves = moves.len().min(4);
    let phase2_moves = moves.len().min(8);
    
    // Fase 1: Teste rápido com redução maior
    for &mv in moves.iter().take(phase1_moves) {
        if test_multicut_move_phase1(board, mv, depth, beta, tt, context, start_time, max_time_ms, ply) {
            result.phase1_cutoffs.push(mv);
        }
    }
    
    // Se Fase 1 não encontrou cutoffs suficientes, não continua
    if result.phase1_cutoffs.len() < 2 {
        return result;
    }
    
    // Fase 2: Teste mais preciso com redução menor
    for &mv in moves.iter().take(phase2_moves) {
        if test_multicut_move_phase2(board, mv, depth, alpha, beta, tt, context, start_time, max_time_ms, ply) {
            result.phase2_cutoffs.push(mv);
        }
        
        // Para se já confirmamos multi-cut
        if result.phase2_cutoffs.len() >= MULTICUT_CUTOFF_THRESHOLD {
            result.confirmed = true;
            break;
        }
    }
    
    result
}

#[derive(Debug)]
struct MulticutAdvancedResult {
    phase1_cutoffs: Vec<Move>,
    phase2_cutoffs: Vec<Move>,
    confirmed: bool,
}

impl MulticutAdvancedResult {
    fn new() -> Self {
        MulticutAdvancedResult {
            phase1_cutoffs: Vec::new(),
            phase2_cutoffs: Vec::new(),
            confirmed: false,
        }
    }
}

/// Teste de multi-cut Fase 1 (redução maior, mais rápido)
fn test_multicut_move_phase1(
    board: &Board,
    mv: Move,
    depth: u8,
    beta: i32,
    tt: &mut TranspositionTable,
    context: &mut SearchContext,
    start_time: Instant,
    max_time_ms: u64,
    ply: usize
) -> bool {
    let mut temp_board = *board;
    let _undo_info = temp_board.make_move_fast(mv);
    
    context.push_move(mv);
    
    let search_depth = depth.saturating_sub(4); // Redução maior
    let test_beta = beta + 50; // Margem menor para fase 1
    
    let score = -pvs_search_internal(
        &temp_board, 
        search_depth, 
        -test_beta, 
        -test_beta + 1, 
        tt, 
        context, 
        start_time, 
        max_time_ms, 
        false, 
        ply + 1
    );
    
    context.pop_move();
    
    score >= test_beta
}

/// Teste de multi-cut Fase 2 (redução menor, mais preciso)
fn test_multicut_move_phase2(
    board: &Board,
    mv: Move,
    depth: u8,
    alpha: i32,
    beta: i32,
    tt: &mut TranspositionTable,
    context: &mut SearchContext,
    start_time: Instant,
    max_time_ms: u64,
    ply: usize
) -> bool {
    let mut temp_board = *board;
    let _undo_info = temp_board.make_move_fast(mv);
    
    context.push_move(mv);
    
    let search_depth = depth.saturating_sub(MULTICUT_SEARCH_DEPTH); // Redução padrão
    let test_beta = beta + MULTICUT_MARGIN;
    
    let score = -pvs_search_internal(
        &temp_board, 
        search_depth, 
        -test_beta, 
        -alpha - 1, 
        tt, 
        context, 
        start_time, 
        max_time_ms, 
        false, 
        ply + 1
    );
    
    context.pop_move();
    
    score >= test_beta
}

/// Multi-cut adaptativo baseado na posição
fn evaluate_adaptive_multicut(
    board: &Board,
    moves: &[Move],
    depth: u8,
    alpha: i32,
    beta: i32,
    static_eval: i32,
    tt: &mut TranspositionTable,
    context: &mut SearchContext,
    start_time: Instant,
    max_time_ms: u64,
    ply: usize
) -> bool {
    // Ajusta parâmetros baseado na avaliação estática
    let eval_margin = static_eval - beta;
    
    let (test_depth, test_moves, cutoff_threshold) = if eval_margin > 200 {
        // Posição muito boa - pode ser mais agressivo
        (depth.saturating_sub(2), 6, 2)
    } else if eval_margin > 100 {
        // Posição boa - parâmetros padrão
        (depth.saturating_sub(3), 8, 3)
    } else {
        // Posição marginal - mais conservativo
        (depth.saturating_sub(4), 10, 4)
    };
    
    let mut cutoffs = 0;
    let test_beta = beta + (eval_margin / 4); // Ajusta margem baseado na posição
    
    for &mv in moves.iter().take(test_moves) {
        if start_time.elapsed().as_millis() as u64 > max_time_ms / 4 {
            break;
        }
        
        let mut temp_board = *board;
        let _undo_info = temp_board.make_move_fast(mv);
        
        context.push_move(mv);
        
        let score = -pvs_search_internal(
            &temp_board, 
            test_depth, 
            -test_beta, 
            -test_beta + 1, 
            tt, 
            context, 
            start_time, 
            max_time_ms, 
            false, 
            ply + 1
        );
        
        context.pop_move();
        
        if score >= test_beta {
            cutoffs += 1;
            if cutoffs >= cutoff_threshold {
                return true; // Multi-cut confirmado
            }
        }
    }
    
    false
}

/// Análise de padrões para multi-cut (detecta tipos de posição favoráveis)
fn analyze_multicut_position_patterns(board: &Board, static_eval: i32, depth: u8) -> MulticutPositionType {
    let total_pieces = (board.white_pieces | board.black_pieces).count_ones();
    let material_advantage = static_eval.abs();
    
    if material_advantage > 300 && total_pieces <= 16 {
        MulticutPositionType::WinningEndgame
    } else if material_advantage > 200 && depth >= 10 {
        MulticutPositionType::DominantPosition
    } else if static_eval > 150 && total_pieces >= 20 {
        MulticutPositionType::TacticalAdvantage
    } else {
        MulticutPositionType::Normal
    }
}

#[derive(Debug, PartialEq)]
enum MulticutPositionType {
    WinningEndgame,     // Endgame vencedor - multi-cut muito provável
    DominantPosition,   // Posição dominante - multi-cut provável
    TacticalAdvantage,  // Vantagem tática - multi-cut possível
    Normal,             // Posição normal - multi-cut improvável
}

// ============================================================================
// MULTI-CUT PRUNING APRIMORADO - FUNÇÕES AUXILIARES
// ============================================================================

/// Verifica se devemos tentar multi-cut pruning aprimorado
fn should_try_multicut_enhanced(
    depth: u8, 
    is_pv_node: bool, 
    in_check: bool, 
    moves: &[Move], 
    static_eval: i32, 
    beta: i32
) -> bool {
    if depth < 6 || is_pv_node || in_check || moves.len() < 8 {
        return false;
    }
    
    // Só tenta se avaliação estática indica posição boa
    if static_eval < beta + 100 {
        return false;
    }
    
    true
}

/// Avalia multi-cut pruning de forma aprimorada e eficiente
fn evaluate_multicut_pruning_enhanced(
    board: &Board,
    moves: &[Move],
    depth: u8,
    alpha: i32,
    beta: i32,
    static_eval: i32,
    tt: &mut TranspositionTable,
    context: &mut SearchContext,
    start_time: Instant,
    max_time_ms: u64,
    ply: usize
) -> usize {
    let search_depth = depth.saturating_sub(3);
    let margin = if static_eval > beta + 200 { 150 } else { 100 };
    let test_beta = beta + margin;
    
    let mut cutoff_count = 0;
    let max_moves_to_test = if depth >= 10 { 8 } else { 6 };
    
    for &mv in moves.iter().take(max_moves_to_test) {
        // Timeout check otimizado
        if cutoff_count == 0 && start_time.elapsed().as_millis() as u64 > max_time_ms / 4 {
            break;
        }
        
        if context.should_stop {
            break;
        }
        
        // Filtra movimentos táticos prioritários para multi-cut
        let is_tactical = board.is_capture(mv) || 
                         gives_check_fast(board, mv) || 
                         mv.promotion.is_some();
        
        if !is_tactical {
            continue;
        }
        
        let mut temp_board = *board;
        let _undo_info = temp_board.make_move_fast(mv);
        
        context.push_move(mv);
        
        let score = -pvs_search_internal(
            &temp_board, 
            search_depth, 
            -test_beta, 
            -test_beta + 1, 
            tt, 
            context, 
            start_time, 
            max_time_ms, 
            false, 
            ply + 1
        );
        
        context.pop_move();
        
        if score >= test_beta {
            cutoff_count += 1;
            
            // Para cedo se já confirmou multi-cut
            if cutoff_count >= 3 {
                break;
            }
        }
    }
    
    cutoff_count
}