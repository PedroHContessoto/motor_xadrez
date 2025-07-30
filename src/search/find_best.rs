use std::time::Instant;
use std::io::{self, Write};
use crate::board::Board;
use crate::evaluation;
use crate::evaluation::mate::draw_win_management::{DrawWinManager, DrawDecisionContext, DrawDecision};
use crate::evaluation::mate::victory_conversion::{VictoryConversionSystem, WinningPlan, MateSequence};
use crate::transposition::TranspositionTable;
use crate::{profile, count};
use super::{SearchContext, aspiration_search, pvs_search, parallel::{parallel_search, ParallelConfig}};

pub fn find_best_move(board: &Board, max_depth: u8, tt: &mut TranspositionTable) -> Option<(crate::types::Move, i32)> {
    find_best_move_with_time(board, max_depth, 5000, tt)
}

/// Interface pública que escolhe entre busca paralela ou single-thread
pub fn find_best_move_with_time(board: &Board, max_depth: u8, max_time_ms: u64, _tt: &mut TranspositionTable) -> Option<(crate::types::Move, i32)> {
    // Por enquanto usa apenas busca single-thread até resolvermos os tipos
    if let Some(mv) = find_best_move_single_thread(board, max_depth, max_time_ms) {
        Some((mv, 0)) // Retorna tupla para manter compatibilidade
    } else {
        None
    }
}

/// Busca single-thread (original)
pub fn find_best_move_single_thread(board: &Board, max_depth: u8, mut max_time_ms: u64) -> Option<crate::types::Move> {
    let mut tt = TranspositionTable::new(64); // 64MB local
    let _timer = crate::profiling::PROFILER.start_timer("find_best_move_total");
    count!("search_calls");
    
    let start_time = Instant::now();
    let mut context = SearchContext::new();
    let mut best_move = None;
    let mut best_score = 0;
    let mut prev_score = 0;
    let mut stable_count = 0;
    
    // === SISTEMA DE TIME MANAGEMENT APRIMORADO ===
    let time_allocation_factor = 0.4;
    let panic_time_factor = 0.8;
    let stable_move_time_reduction = 0.7;
    let allocated_time = (max_time_ms as f64 * time_allocation_factor) as u64;
    let soft_limit = (max_time_ms as f64 * 0.6) as u64;
    let hard_limit = (max_time_ms as f64 * panic_time_factor) as u64;
    
    // === ASPIRATION WINDOWS ADAPTATIVAS ===
    let mut aspiration_enabled = true;
    let mut window_size = 25;
    let max_window_size = 500;
    let aspiration_depth_threshold = 4;

    // Clear PV table for new search
    context.clear_pv();

    // Garante que sempre temos um movimento de fallback
    let legal_moves = profile!("move_generation", {
        board.generate_legal_moves()
    });
    count!("legal_moves_generated", legal_moves.len() as u64);
    if legal_moves.is_empty() {
        return None; // Mate/stalemate
    }
    let mut fallback_move = legal_moves[0];
    let mut fallback_score = profile!("evaluation", {
        evaluation::evaluate_with_depth(board, 0) // Root node - avaliação completa
    });

    // === SISTEMA DE GESTÃO DE TEMPO ADAPTATIVO APRIMORADO ===
    let moves_played = (board.halfmove_clock / 2) + 1; // Aproximação do número de movimentos
    let complexity = profile!("position_complexity_analysis", {
        evaluate_position_complexity(board)
    });
    let adaptive_time_limit = profile!("time_management_calculation", {
        calculate_adaptive_time_management(board, complexity, moves_played, max_time_ms)
    });
    
    // === SISTEMA DE GESTÃO DE EMPATE/VITÓRIA ===
    let draw_win_manager = DrawWinManager::new();
    let initial_eval = evaluation::evaluate_with_depth(board, 0); // Root evaluation
    let draw_context = DrawDecisionContext::new(board, initial_eval);
    let draw_strategy = profile!("draw_win_decision", {
        draw_win_manager.evaluate_draw_decision(board, &draw_context)
    });
    
    // === SISTEMA DE CONVERSÃO DE VITÓRIA ===
    let mut victory_system = VictoryConversionSystem::new();
    let mut winning_plan = if initial_eval > 150 {
        // Posição ganhadora - cria plano de conversão
        Some(profile!("victory_plan_creation", {
            victory_system.evaluate_winning_plan(board, initial_eval)
        }))
    } else {
        None
    };
    
    // === DETECÇÃO RÁPIDA DE MATE (APENAS EM POSIÇÕES CRÍTICAS) ===
    // Só busca mate se realmente justificado (vantagem >1000cp ou oponente em xeque)
    if initial_eval > 1000 || board.is_king_in_check(!board.to_move) {
        if let Some(mate_sequence) = profile!("mate_detection", {
            victory_system.quick_mate_detection(board, 10)
        }) {
            // Validação adicional: verifica se realmente é mate forçado
            if mate_sequence.mate_in <= 10 && profile!("mate_sequence_validation", {
                is_forced_mate_sequence(board, &mate_sequence.moves)
            }) {
                count!("mates_found");
                println!("info string MATE DETECTADO em {} movimentos!", mate_sequence.mate_in);
                if let Some(first_move) = mate_sequence.moves.first() {
                    return Some(*first_move); // Só retorna o movimento, não a tupla
                }
            }
        }
    }
    
    // Atualiza limites baseado no tempo adaptativo
    let adapted_soft_limit = (adaptive_time_limit as f64 * 0.6) as u64;
    let adapted_hard_limit = (adaptive_time_limit as f64 * panic_time_factor) as u64;
    
    // Log inicial removido para terminal limpo

    for depth in 1..=max_depth {
        let _depth_timer = crate::profiling::PROFILER.start_timer(&format!("search_depth_{}", depth));
        count!("depth_iterations");
        
        let iteration_start = std::time::Instant::now();
        let elapsed = start_time.elapsed().as_millis() as u64;

        // Update search context for logging
        context.current_depth = depth;
        context.nodes_searched = 0; // Reset for this depth
        
        // === TIME MANAGEMENT ADAPTATIVO ===
        // Verifica se deve continuar iteração baseado em estimativas
        if depth > 1 {
            let estimated_time = estimate_iteration_time(depth, elapsed);
            if elapsed + estimated_time > adapted_soft_limit {
                break;
            }
        }
        
        // Hard timeout absoluto adaptativo
        if elapsed > adapted_hard_limit {
            break;
        }
        
        // Depth limit baseado na complexidade da posição
        let max_safe_depth = match complexity {
            PositionComplexity::Critical => 16,  // Posições críticas - máxima profundidade
            PositionComplexity::Tactical => 14,  // Posições táticas - alta profundidade
            PositionComplexity::Complex => 12,   // Posições complexas - profundidade média-alta
            PositionComplexity::Normal => 10,    // Posições normais - profundidade normal
            PositionComplexity::Simple => 8,     // Posições simples - profundidade reduzida
        };
        
        if depth > max_safe_depth {
            break;
        }

        // Reset stop flag para cada profundidade
        context.should_stop = false;

        // === BUSCA COM ASPIRATION WINDOWS ADAPTATIVAS E DRAW/WIN MANAGEMENT ===
        let mut alpha = -50000;
        let mut beta = 50000;
        let mut aspiration_fails = 0;
        let mut iteration_score;
        
        // Ajusta janelas baseado na estratégia de empate/vitória
        let (adjusted_alpha, adjusted_beta) = draw_win_manager.adjust_search_windows(alpha, beta, &draw_strategy);
        alpha = adjusted_alpha;
        beta = adjusted_beta;
        
        // Configura aspiration windows se profundidade suficiente
        if aspiration_enabled && depth >= aspiration_depth_threshold && prev_score != 0 {
            alpha = prev_score - window_size;
            beta = prev_score + window_size;
            
            // Reaplica ajustes de draw/win se necessário
            let (readjusted_alpha, readjusted_beta) = draw_win_manager.adjust_search_windows(alpha, beta, &draw_strategy);
            alpha = readjusted_alpha;
            beta = readjusted_beta;
        }
        
        // Loop de aspiration windows
        loop {
            iteration_score = if depth >= aspiration_depth_threshold && aspiration_enabled {
                profile!("aspiration_search", {
                    aspiration_search(board, depth, (alpha + beta) / 2, &mut tt, &mut context, start_time, max_time_ms)
                })
            } else {
                profile!("pvs_search", {
                    pvs_search(board, depth, alpha, beta, &mut tt, &mut context, start_time, max_time_ms, true)
                })
            };
            count!("search_iterations");
            
            if context.should_stop {
                break;
            }
            
            // Analisa resultado da aspiration window
            if aspiration_enabled && depth >= aspiration_depth_threshold {
                if iteration_score <= alpha {
                    // Fail low
                    aspiration_fails += 1;
                    beta = (alpha + beta) / 2;
                    alpha = iteration_score - window_size * (1 << aspiration_fails.min(4));
                    
                    // Aspiration fail-low (removido log para terminal limpo)
                    
                } else if iteration_score >= beta {
                    // Fail high
                    aspiration_fails += 1;
                    alpha = (alpha + beta) / 2;
                    beta = iteration_score + window_size * (1 << aspiration_fails.min(4));
                    
                    // Aspiration fail-high (removido log para terminal limpo)
                    
                } else {
                    // Sucesso
                    break;
                }
                
                // Evita loops infinitos em aspiration
                if aspiration_fails > 6 {
                    alpha = -50000;
                    beta = 50000;
                }
            } else {
                break;
            }
        }
        
        let score = iteration_score;

        // Log tempo da iteração (removido para terminal limpo)
        let _iteration_time = iteration_start.elapsed().as_millis();
        let _total_elapsed = start_time.elapsed().as_millis();

        // Se parou por timeout, usa o que temos
        if context.should_stop {
            break;
        }

        if let Some(entry) = profile!("transposition_table_probe", {
            tt.probe(board.zobrist_hash)
        }) {
            if let Some(mv) = entry.best_move {
                if board.is_legal_move(mv) {
                    if Some(mv) == context.prev_best_move {
                        stable_count += 1;
                        if stable_count >= 3 {
                            // Aplica redução de tempo para movimento estável
                            max_time_ms = (max_time_ms as f64 * stable_move_time_reduction) as u64;
                        }
                    } else {
                        stable_count = 0;
                        // Restaura tempo original se movimento mudou
                        max_time_ms = (allocated_time as f64 / time_allocation_factor) as u64;
                    }
                    context.prev_best_move = Some(mv);

                    best_move = Some(mv);
                    best_score = score;
                    fallback_move = mv; // Atualiza fallback
                    fallback_score = score;
                    
                    // Atualiza plano de vitória se aplicável
                    if let Some(ref mut plan) = winning_plan {
                        plan.update_progress(board);
                    }
                } else {
                    // Move da TT não é legal, mas ainda podemos usar o score
                    if best_move.is_none() {
                        best_move = Some(fallback_move);
                        best_score = score;
                    }
                    // Não imprime info se move não é legal
                    continue;
                }

                // Só imprime se temos um move legal válido
                let nps = if elapsed > 0 { context.nodes_searched * 1000 / elapsed } else { 0 }; // Limpado parens
                let time_ms = elapsed;

                let display_score = score.clamp(-10000, 10000);

                // Format Principal Variation
                let pv_string = if context.pv_length[0] > 0 {
                    context.format_pv(0)
                } else {
                    format!("{}", mv)
                };

                // Detecção e formatação limpa de mate
                let score_output = format_mate_score_clean(display_score);
                
                // Log principal UCI detalhado e profissional  
                println!("info depth {} {} nodes {} nps {} time {} pv {}",
                         depth, score_output, context.nodes_searched, nps, time_ms, pv_string);
                
                // Log adicional removido para terminal limpo

                io::stdout().flush().ok();
            }
        }

        prev_score = score;
        
        // === ATUALIZA WINDOW SIZE PARA PRÓXIMA ITERAÇÃO ===
        window_size = calculate_next_aspiration_window(score, depth, aspiration_fails, window_size, max_window_size);
    }

    // Garante que sempre retornamos um movimento válido
    if best_move.is_none() {
        best_move = Some(fallback_move);
        best_score = fallback_score;
    }

    // VALIDAÇÃO FINAL: Garante que o movimento é legal
    if let Some(mv) = best_move {
        if !legal_moves.contains(&mv) {
            eprintln!("WARNING: Best move {} is not legal! Using fallback.", mv);
            // Força usar primeiro movimento legal
            best_move = Some(legal_moves[0]);
        }
    }

    best_move
}

// ============================================================================
// FUNÇÕES AUXILIARES PARA DETECÇÃO DE MATE
// ============================================================================

/// Formata score de mate de forma limpa e profissional
fn format_mate_score_clean(score: i32) -> String {
    const MATE_THRESHOLD: i32 = 9000;
    
    if score > MATE_THRESHOLD {
        // Mate favorável - calcula distância correta do mate
        let mate_distance = calculate_mate_distance_accurate(score);
        format!("score mate {}", mate_distance)
    } else if score < -MATE_THRESHOLD {
        // Mate contra nós
        let mate_distance = calculate_mate_distance_accurate(-score);
        format!("score mate -{}", mate_distance)
    } else {
        // Score normal em centipawns
        format!("score cp {}", score)
    }
}

/// Calcula distância precisa do mate baseado no score
fn calculate_mate_distance_accurate(mate_score: i32) -> u8 {
    const MATE_VALUE: i32 = 99999; // Uniformizado com outros módulos
    
    // Fórmula padrão UCI: distância = (MATE_VALUE - score)
    // Mas ajustada para ser sempre positiva e realista
    let raw_distance = MATE_VALUE - mate_score;
    
    // Garante que seja um número realista (1-15 movimentos)
    if raw_distance > 0 && raw_distance <= 15 {
        raw_distance as u8
    } else {
        // Para scores muito altos, assume mate em 1
        1
    }
}


/// Níveis de complexidade da posição para gestão de tempo
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PositionComplexity {
    Simple = 0,     // Posições simples, endgames básicos
    Normal = 1,     // Posições normais
    Complex = 2,    // Posições com múltiplas ameaças
    Tactical = 3,   // Posições com ameaças táticas diretas
    Critical = 4,   // Posições críticas (mate, material hanging)
}

/// Avalia complexidade da posição em níveis (0-4) para gestão adaptativa de tempo
fn evaluate_position_complexity(board: &Board) -> PositionComplexity {
    let our_color = board.to_move;
    let enemy_color = !our_color;
    let mut complexity_score = 0u8;

    // 1. Fatores de complexidade imediata (+2 pontos cada)
    if board.is_king_in_check(our_color) {
        complexity_score += 2; // Xeque = alta complexidade
    }

    // 2. Análise de peças atacadas/penduradas
    let our_pieces = if our_color == crate::types::Color::White {
        board.white_pieces
    } else {
        board.black_pieces
    };
    let enemy_pieces = if enemy_color == crate::types::Color::White {
        board.white_pieces
    } else {
        board.black_pieces
    };

    let our_valuables = (board.knights | board.bishops | board.rooks | board.queens) & our_pieces;
    let enemy_valuables = (board.knights | board.bishops | board.rooks | board.queens) & enemy_pieces;
    
    let mut hanging_count = 0;
    let mut attacked_count = 0;
    let mut enemy_hanging = 0;

    // Analisa nossas peças
    let mut bb = our_valuables;
    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        bb &= bb - 1;

        if board.is_square_attacked_by(sq, enemy_color) {
            attacked_count += 1;
            if !board.is_square_attacked_by(sq, our_color) {
                hanging_count += 1;
            }
        }
    }

    // Analisa peças inimigas
    let mut enemy_bb = enemy_valuables;
    while enemy_bb != 0 {
        let sq = enemy_bb.trailing_zeros() as u8;
        enemy_bb &= enemy_bb - 1;

        if board.is_square_attacked_by(sq, our_color) &&
            !board.is_square_attacked_by(sq, enemy_color) {
            enemy_hanging += 1;
        }
    }

    // 3. Pontuação baseada em ameaças
    if hanging_count > 0 {
        complexity_score += 2; // Temos peças penduradas = muito complexo
    }
    if enemy_hanging > 0 {
        complexity_score += 1; // Podemos ganhar material = complexo
    }
    if attacked_count > 2 {
        complexity_score += 1; // Muitas peças sob ataque = complexo
    }

    // 4. Análise da fase do jogo
    let total_pieces = (board.white_pieces | board.black_pieces).count_ones();
    let total_pawns = board.pawns.count_ones();
    let queens_on_board = board.queens.count_ones();

    // Finais técnicos complexos
    if total_pieces <= 8 && total_pawns <= 3 {
        complexity_score += 1;
    }
    
    // Meio-jogo com muitas peças = potencial tático
    if total_pieces > 20 && queens_on_board >= 2 {
        complexity_score += 1;
    }

    // 5. Converte para enum baseado no score final
    match complexity_score {
        0 => PositionComplexity::Simple,
        1 => PositionComplexity::Normal, 
        2 => PositionComplexity::Complex,
        3 => PositionComplexity::Tactical,
        _ => PositionComplexity::Critical, // 4+
    }
}

// ============================================================================
// SISTEMA DE GESTÃO DE TEMPO ADAPTATIVO APRIMORADO
// ============================================================================

/// Calcula gestão de tempo adaptativa baseada na complexidade e fase do jogo
fn calculate_adaptive_time_management(
    board: &Board, 
    complexity: PositionComplexity, 
    moves_played: u16, 
    base_time: u64
) -> u64 {
    let base_time_f64 = get_base_time(moves_played, base_time);
    
    // Sistema de multiplicadores dinâmicos conforme proposta
    let multiplier = match complexity {
        PositionComplexity::Critical => 3.5,  // 250% mais tempo para posições críticas
        PositionComplexity::Tactical => 2.5,  // 150% mais tempo para posições táticas
        PositionComplexity::Complex => 2.0,   // 100% mais tempo para posições complexas
        PositionComplexity::Normal => 1.0,    // Tempo normal
        PositionComplexity::Simple => 0.7,    // 30% menos tempo para posições simples
    };
    
    // Sistema de bônus adaptativo para situações especiais
    let special_bonus = calculate_special_bonus(board, complexity);
    
    let final_time = (base_time_f64 * multiplier + special_bonus as f64) as u64;
    
    // Limita o tempo máximo para evitar timeouts extremos
    let max_allowed_time = base_time * 5; // Máximo 5x o tempo base
    final_time.min(max_allowed_time)
}

/// Calcula tempo base baseado na fase do jogo (número de movimentos)
fn get_base_time(moves_played: u16, original_time: u64) -> f64 {
    let base_factor = match moves_played {
        1..=10 => 1.2,     // Abertura: ligeiramente mais tempo
        11..=20 => 1.0,    // Meio-jogo inicial: tempo normal
        21..=40 => 1.1,    // Meio-jogo: tempo normal+
        41..=60 => 1.3,    // Final de jogo: mais tempo para precisão
        _ => 1.4,          // Finais longos: máximo tempo para precisão
    };
    
    original_time as f64 * base_factor
}

/// Calcula bônus especial para situações críticas
fn calculate_special_bonus(board: &Board, complexity: PositionComplexity) -> u64 {
    let mut bonus = 0u64;
    
    // Bônus para situações especiais críticas
    match complexity {
        PositionComplexity::Critical => {
            // Verificação de mate iminente
            if board.is_king_in_check(board.to_move) {
                bonus += 2000; // 2 segundos extras para escapar do xeque
            }
            
            // Material hanging (peças não defendidas)
            let hanging_material = calculate_hanging_material_value(board);
            if hanging_material > 300 {
                bonus += 1500; // 1.5 segundos para salvar material
            }
            
            // Ameaças de mate em poucas jogadas
            if detect_immediate_mate_threats(board) {
                bonus += 5000; // 5 segundos para defender mate (aumentado)
            }
            
            // Posições de mate detectadas - tempo extra para encontrar sequência precisa
            let eval = crate::evaluation::evaluate_with_depth(board, 0);
            if eval.abs() > 5000 {  // Likely mate position
                bonus += 4000; // 4 segundos para calcular mate preciso
            }
        },
        
        PositionComplexity::Tactical => {
            // Combinações táticas complexas
            if count_tactical_motifs(board) >= 2 {
                bonus += 1000; // 1 segundo para calcular táticas
            }
        },
        
        PositionComplexity::Complex => {
            // Múltiplas ameaças simultâneas
            let threat_count = count_simultaneous_threats(board);
            if threat_count >= 3 {
                bonus += 800; // 0.8 segundos para múltiplas ameaças
            }
        },
        
        _ => {}, // Normal e Simple não recebem bônus especial
    }
    
    // Bônus para finais teóricos complexos
    let total_pieces = (board.white_pieces | board.black_pieces).count_ones();
    if total_pieces <= 7 && complexity != PositionComplexity::Simple {
        bonus += 500; // 0.5 segundos extras para finais técnicos
    }
    
    bonus
}

// ============================================================================
// FUNÇÕES AUXILIARES PARA DETECÇÃO DE SITUAÇÕES ESPECIAIS
// ============================================================================

/// Calcula valor do material "hanging" (não defendido)
fn calculate_hanging_material_value(board: &Board) -> i32 {
    let our_color = board.to_move;
    let enemy_color = !our_color;
    let our_pieces = if our_color == crate::types::Color::White {
        board.white_pieces
    } else {
        board.black_pieces
    };
    
    let piece_values = [100, 320, 330, 500, 900, 20000]; // P, N, B, R, Q, K
    let mut hanging_value = 0;
    
    // Verifica cada tipo de peça
    let piece_types = [
        (board.pawns, 0),
        (board.knights, 1), 
        (board.bishops, 2),
        (board.rooks, 3),
        (board.queens, 4),
    ];
    
    for (piece_bb, piece_type) in piece_types {
        let our_pieces_of_type = piece_bb & our_pieces;
        let mut pieces = our_pieces_of_type;
        
        while pieces != 0 {
            let sq = pieces.trailing_zeros() as u8;
            pieces &= pieces - 1;
            
            // Verifica se a peça está atacada e não defendida
            if board.is_square_attacked_by(sq, enemy_color) &&
               !board.is_square_attacked_by(sq, our_color) {
                hanging_value += piece_values[piece_type];
            }
        }
    }
    
    hanging_value
}

/// Detecta ameaças de mate imediatas
fn detect_immediate_mate_threats(board: &Board) -> bool {
    // Verifica se o oponente pode dar mate em 1-2 movimentos
    let enemy_moves = board.generate_legal_moves();
    
    for mv in &enemy_moves {
        let mut test_board = *board;
        let _undo = test_board.make_move_fast(*mv);
        
        // Xeque que pode levar a mate
        if test_board.is_king_in_check(!board.to_move) {
            let escape_moves = test_board.generate_legal_moves();
            if escape_moves.len() <= 2 { // Poucas opções de escape
                return true;
            }
        }
    }
    
    false
}

/// Conta motivos táticos na posição
fn count_tactical_motifs(board: &Board) -> u8 {
    let mut motifs = 0;
    
    // Pins
    if has_pin_motifs(board) {
        motifs += 1;
    }
    
    // Forks
    if has_fork_opportunities(board) {
        motifs += 1;
    }
    
    // Descobertas
    if has_discovered_attack_potential(board) {
        motifs += 1;
    }
    
    // Deflections/Decoys
    if has_deflection_motifs(board) {
        motifs += 1;
    }
    
    motifs
}

/// Conta ameaças simultâneas
fn count_simultaneous_threats(board: &Board) -> u8 {
    let mut threats = 0;
    let moves = board.generate_legal_moves();
    
    for mv in &moves {
        let mut test_board = *board;
        let _undo = test_board.make_move_fast(*mv);
        
        // Conta tipos diferentes de ameaças
        if test_board.is_king_in_check(!board.to_move) {
            threats += 1; // Xeque
        }
        
        if board.is_capture(*mv) {
            threats += 1; // Captura
        }
        
        // Ameaça a peça valiosa
        if threatens_valuable_piece(&test_board, !board.to_move) {
            threats += 1;
        }
    }
    
    threats.min(5) // Limita para evitar overflow
}

// Funções auxiliares simples para motivos táticos
fn has_pin_motifs(_board: &Board) -> bool { false } // Implementação simplificada
fn has_fork_opportunities(_board: &Board) -> bool { false }
fn has_discovered_attack_potential(_board: &Board) -> bool { false }
fn has_deflection_motifs(_board: &Board) -> bool { false }

fn threatens_valuable_piece(board: &Board, color: crate::types::Color) -> bool {
    let pieces = if color == crate::types::Color::White {
        board.white_pieces
    } else {
        board.black_pieces
    };
    
    let valuable = (board.queens | board.rooks) & pieces;
    let attacked_by_enemy = crate::evaluation::mobility::compute_attacked_squares(board, !color);
    valuable.count_ones() > 0 && 
    (valuable & attacked_by_enemy).count_ones() > 0
}

// ============================================================================
// FUNÇÕES AUXILIARES PARA TIME MANAGEMENT E ASPIRATION WINDOWS APRIMORADAS
// ============================================================================

/// Estima tempo necessário para próxima iteração baseado no histórico
fn estimate_iteration_time(depth: u8, elapsed_total: u64) -> u64 {
    if depth <= 1 {
        return 100; // Estimativa conservadora para primeiras profundidades
    }
    
    // Fator de ramificação estimado baseado na profundidade
    let branching_factor = if depth <= 4 {
        2.0 // Abertura/meio-jogo inicial
    } else if depth <= 8 {
        2.5 // Meio-jogo
    } else {
        3.0 // Profundidades altas
    };
    
    // Tempo da última iteração estimado
    let estimated_last_iteration = elapsed_total / depth as u64;
    
    // Projeta próxima iteração
    (estimated_last_iteration as f64 * branching_factor) as u64
}

/// Calcula próximo tamanho da aspiration window de forma adaptativa
fn calculate_next_aspiration_window(
    score: i32, 
    depth: u8, 
    fails: usize, 
    current_window: i32, 
    max_window: i32
) -> i32 {
    let base_window = 25;
    
    // Fator baseado na profundidade
    let depth_factor = if depth > 10 { 1.3 } else { 1.0 };
    
    // Fator baseado no número de fails
    let fail_factor = if fails > 0 { 
        1.0 + (fails as f64 * 0.3) 
    } else { 
        0.8 // Reduz janela se não houve fails
    };
    
    // Fator baseado no score (scores extremos precisam de janelas maiores)
    let score_factor = if score.abs() > 500 { 
        1.5 
    } else if score.abs() > 200 { 
        1.2 
    } else { 
        1.0 
    };
    
    // Calcula novo tamanho
    let new_window = (current_window as f64 * fail_factor * score_factor * depth_factor) as i32;
    
    // Limita entre mínimo e máximo
    new_window.max(base_window).min(max_window)
}

/// Configuração avançada do sistema de busca com gestão de tempo adaptativa
#[derive(Debug, Clone)]
pub struct SearchConfig {
    pub time_allocation_factor: f64,
    pub panic_time_factor: f64,
    pub stable_move_time_reduction: f64,
    pub aspiration_enabled: bool,
    pub initial_aspiration_window: i32,
    pub max_aspiration_window: i32,
    pub aspiration_depth_threshold: u8,
    pub max_depth_tactical: u8,
    pub max_depth_normal: u8,
    
    // === NOVOS PARÂMETROS DE GESTÃO DE TEMPO ADAPTATIVA ===
    pub adaptive_time_enabled: bool,
    pub critical_time_multiplier: f64,   // Para posições críticas
    pub tactical_time_multiplier: f64,   // Para posições táticas
    pub complex_time_multiplier: f64,    // Para posições complexas
    pub simple_time_multiplier: f64,     // Para posições simples
    pub max_time_extension_factor: f64,  // Limite máximo de extensão
    
    // Bônus especiais (em milissegundos)
    pub check_escape_bonus: u64,         // Bônus para escapar do xeque
    pub material_save_bonus: u64,        // Bônus para salvar material
    pub mate_defense_bonus: u64,         // Bônus para defender mate
    pub tactical_calculation_bonus: u64, // Bônus para cálculos táticos
    pub endgame_precision_bonus: u64,    // Bônus para precisão em finais
}

impl Default for SearchConfig {
    fn default() -> Self {
        SearchConfig {
            time_allocation_factor: 0.4,
            panic_time_factor: 0.8,
            stable_move_time_reduction: 0.7,
            aspiration_enabled: true,
            initial_aspiration_window: 25,
            max_aspiration_window: 500,
            aspiration_depth_threshold: 4,
            max_depth_tactical: 14,
            max_depth_normal: 9,
            
            // === CONFIGURAÇÕES DE GESTÃO DE TEMPO ADAPTATIVA ===
            adaptive_time_enabled: true,
            critical_time_multiplier: 3.5,   // Conforme proposta
            tactical_time_multiplier: 2.5,   // Conforme proposta
            complex_time_multiplier: 2.0,    // Conforme proposta
            simple_time_multiplier: 0.7,     // Conforme proposta
            max_time_extension_factor: 5.0,  // Máximo 5x o tempo original
            
            // Bônus especiais conforme implementação
            check_escape_bonus: 2000,        // 2 segundos para escapar xeque
            material_save_bonus: 1500,       // 1.5 segundos para salvar material
            mate_defense_bonus: 3000,        // 3 segundos para defender mate
            tactical_calculation_bonus: 1000, // 1 segundo para táticas
            endgame_precision_bonus: 500,    // 0.5 segundos para finais
        }
    }
}

// Para compatibilidade, mantém o nome antigo como alias
pub type EnhancedSearchConfig = SearchConfig;

/// Valida se uma sequência de movimentos é realmente um mate forçado
fn is_forced_mate_sequence(board: &Board, moves: &[crate::types::Move]) -> bool {
    if moves.is_empty() {
        return false;
    }
    
    let mut test_board = *board;
    
    // Testa a sequência completa
    for (i, &mv) in moves.iter().enumerate() {
        // Verifica se o movimento é legal
        let legal_moves = test_board.generate_legal_moves();
        if !legal_moves.contains(&mv) {
            return false;
        }
        
        // Executa o movimento
        let _undo = test_board.make_move_fast(mv);
        
        // Se é o último movimento, deve ser mate
        if i == moves.len() - 1 {
            return test_board.is_checkmate();
        }
        
        // Se não é o último movimento, verifica se força resposta
        // (deve dar xeque ou ser única opção razoável)
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