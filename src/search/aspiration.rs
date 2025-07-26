use std::time::Instant;
use crate::board::Board;
use crate::transposition::TranspositionTable;
use super::{SearchContext, pvs_search};

// ============================================================================
// DYNAMIC ASPIRATION WINDOWS - SISTEMA AVANÇADO E ADAPTATIVO
// ============================================================================

/// Configuração dinâmica das aspiration windows
#[derive(Debug, Clone)]
struct AspirationConfig {
    initial_window: i32,
    max_window: i32,
    expansion_factor: f32,
    reduction_factor: f32,
    stability_threshold: i32,
    max_fails: usize,
    adaptive_factor: f32,
}

/// Histórico de performance das aspiration windows
#[derive(Debug, Clone)]
struct AspirationHistory {
    recent_fails: Vec<bool>,        // Últimos resultados (true = fail)
    average_window_size: f32,       // Tamanho médio das janelas
    score_volatility: f32,          // Volatilidade dos scores
    position_complexity: f32,       // Complexidade da posição (0.0-1.0)
    last_successful_window: i32,    // Última janela que funcionou
}

/// Resultado detalhado da aspiration search
#[derive(Debug)]
pub struct AspirationResult {
    pub score: i32,
    pub window_size: i32,
    pub searches_performed: usize,
    pub time_spent: u64,
    pub failed_high: usize,
    pub failed_low: usize,
}

impl AspirationHistory {
    fn new() -> Self {
        Self {
            recent_fails: Vec::with_capacity(10),
            average_window_size: 25.0,
            score_volatility: 0.0,
            position_complexity: 0.5,
            last_successful_window: 25,
        }
    }

    fn update_fail(&mut self, failed: bool) {
        self.recent_fails.push(failed);
        if self.recent_fails.len() > 10 {
            self.recent_fails.remove(0);
        }
    }

    fn fail_rate(&self) -> f32 {
        if self.recent_fails.is_empty() {
            return 0.0;
        }
        self.recent_fails.iter().filter(|&&x| x).count() as f32 / self.recent_fails.len() as f32
    }

    fn update_window_size(&mut self, size: i32) {
        self.average_window_size = self.average_window_size * 0.9 + size as f32 * 0.1;
        if size > 0 {
            self.last_successful_window = size;
        }
    }

    fn update_score_volatility(&mut self, score_change: i32) {
        let volatility = score_change.abs() as f32;
        self.score_volatility = self.score_volatility * 0.8 + volatility * 0.2;
    }
}

/// Aspiration search avançado com janelas dinâmicas e adaptativas
pub fn dynamic_aspiration_search(
    board: &Board, 
    depth: u8, 
    prev_score: i32, 
    tt: &mut TranspositionTable, 
    context: &mut SearchContext, 
    start_time: Instant, 
    max_time_ms: u64,
    history: &mut AspirationHistory
) -> AspirationResult {
    let search_start = Instant::now();
    
    // Calcula configuração adaptativa baseada no histórico
    let config = calculate_aspiration_config(board, depth, prev_score, history);
    
    // Detecta complexidade da posição para ajuste dinâmico
    let position_complexity = analyze_position_complexity(board, context);
    history.position_complexity = position_complexity;
    
    // Ajusta janela inicial baseada na complexidade e histórico
    let initial_window = calculate_initial_window(&config, prev_score, history);
    
    let mut alpha = prev_score - initial_window;
    let mut beta = prev_score + initial_window;
    let mut current_window = initial_window;
    let mut searches_performed = 0;
    let mut failed_high = 0;
    let mut failed_low = 0;
    
    println!("info string Aspiration: window={} complexity={:.2} volatility={:.1}", 
             initial_window, position_complexity, history.score_volatility);

    loop {
        searches_performed += 1;
        
        // Verifica timeout antes de cada busca
        if start_time.elapsed().as_millis() as u64 > max_time_ms * 2 / 3 {
            // Se já gastamos 2/3 do tempo, faz busca completa
            let score = pvs_search(board, depth, -50000, 50000, tt, context, start_time, max_time_ms, true);
            history.update_fail(true); // Marca como fail porque precisou expandir completamente
            history.update_window_size(-1); // Indica falha completa
            
            return AspirationResult {
                score,
                window_size: 50000,
                searches_performed,
                time_spent: search_start.elapsed().as_millis() as u64,
                failed_high,
                failed_low,
            };
        }

        let score = pvs_search(board, depth, alpha, beta, tt, context, start_time, max_time_ms, true);

        // Analisa resultado e decide próximo passo
        if score <= alpha {
            // Fail low - expande para baixo
            failed_low += 1;
            alpha = calculate_fail_low_bound(prev_score, current_window, &config);
            current_window = expand_window_intelligently(current_window, &config, history, false);
            
            println!("info string Aspiration fail low: score={} new_alpha={} window={}", 
                     score, alpha, current_window);
                     
        } else if score >= beta {
            // Fail high - expande para cima  
            failed_high += 1;
            beta = calculate_fail_high_bound(prev_score, current_window, &config);
            current_window = expand_window_intelligently(current_window, &config, history, true);
            
            println!("info string Aspiration fail high: score={} new_beta={} window={}", 
                     score, beta, current_window);
                     
        } else {
            // Sucesso! Score dentro da janela
            history.update_fail(false);
            history.update_window_size(current_window);
            history.update_score_volatility(score - prev_score);
            
            println!("info string Aspiration success: score={} window={} searches={}", 
                     score, current_window, searches_performed);
            
            return AspirationResult {
                score,
                window_size: current_window,
                searches_performed,
                time_spent: search_start.elapsed().as_millis() as u64,
                failed_high,
                failed_low,
            };
        }

        // Proteção contra loops infinitos ou janelas muito grandes
        if current_window >= config.max_window || searches_performed >= config.max_fails {
            let final_score = pvs_search(board, depth, -50000, 50000, tt, context, start_time, max_time_ms, true);
            history.update_fail(true);
            history.update_window_size(-1);
            
            println!("info string Aspiration timeout: final_search score={} searches={}", 
                     final_score, searches_performed + 1);
            
            return AspirationResult {
                score: final_score,
                window_size: 50000,
                searches_performed: searches_performed + 1,
                time_spent: search_start.elapsed().as_millis() as u64,
                failed_high,
                failed_low,
            };
        }
    }
}

/// Calcula configuração adaptativa baseada no contexto
fn calculate_aspiration_config(
    board: &Board, 
    depth: u8, 
    _prev_score: i32, 
    history: &AspirationHistory
) -> AspirationConfig {
    let base_window = if depth <= 4 {
        15  // Janelas menores para profundidades baixas
    } else if depth <= 8 {
        25  // Janela padrão
    } else {
        35  // Janelas maiores para profundidades altas
    };

    // Ajusta baseado na volatilidade do score
    let volatility_factor = if history.score_volatility > 100.0 {
        1.5  // Mais volatilidade = janelas maiores
    } else if history.score_volatility < 20.0 {
        0.7  // Menos volatilidade = janelas menores
    } else {
        1.0
    };

    // Ajusta baseado na taxa de falhas
    let fail_rate_factor = if history.fail_rate() > 0.6 {
        1.4  // Muitas falhas = janelas maiores
    } else if history.fail_rate() < 0.2 {
        0.8  // Poucas falhas = janelas menores
    } else {
        1.0
    };

    // Ajusta baseado na complexidade da posição
    let complexity_factor = 0.7 + history.position_complexity * 0.6;

    let adjusted_window = (base_window as f32 * volatility_factor * fail_rate_factor * complexity_factor) as i32;

    // Considera fase do jogo
    let total_pieces = (board.white_pieces | board.black_pieces).count_ones();
    let endgame_factor = if total_pieces <= 12 {
        0.8  // Endgame - janelas menores
    } else {
        1.0
    };

    let final_window = (adjusted_window as f32 * endgame_factor) as i32;

    AspirationConfig {
        initial_window: final_window.max(10).min(100),
        max_window: if depth >= 12 { 400 } else { 300 },
        expansion_factor: if history.fail_rate() > 0.5 { 2.5 } else { 2.0 },
        reduction_factor: 0.8,
        stability_threshold: 15,
        max_fails: if depth >= 10 { 6 } else { 4 },
        adaptive_factor: complexity_factor,
    }
}

/// Analisa complexidade da posição para aspiration tuning
fn analyze_position_complexity(board: &Board, context: &SearchContext) -> f32 {
    let mut complexity = 0.0;

    // 1. Número de peças (mais peças = mais complexidade)
    let piece_count = (board.white_pieces | board.black_pieces).count_ones() as f32;
    complexity += (piece_count / 32.0) * 0.3;

    // 2. Atividade tática
    if board.is_king_in_check(board.to_move) {
        complexity += 0.4;
    }

    // 3. Threats recentes
    if context.recent_threat_detected() {
        complexity += 0.3;
    }

    // 4. Material desequilibrado
    let white_material = count_material(board, crate::types::Color::White);
    let black_material = count_material(board, crate::types::Color::Black);
    let material_imbalance = (white_material - black_material).abs() as f32;
    complexity += (material_imbalance / 500.0).min(0.3);

    // 5. Densidade de peças no centro
    let center_squares = 0x0000001818000000u64; // e4, e5, d4, d5
    let center_pieces = (board.white_pieces | board.black_pieces) & center_squares;
    complexity += (center_pieces.count_ones() as f32 / 4.0) * 0.2;

    complexity.min(1.0)
}

/// Conta material total para uma cor
fn count_material(board: &Board, color: crate::types::Color) -> i32 {
    let pieces = if color == crate::types::Color::White { 
        board.white_pieces 
    } else { 
        board.black_pieces 
    };

    let pawns = (board.pawns & pieces).count_ones() as i32 * 100;
    let knights = (board.knights & pieces).count_ones() as i32 * 320;
    let bishops = (board.bishops & pieces).count_ones() as i32 * 330;
    let rooks = (board.rooks & pieces).count_ones() as i32 * 500;
    let queens = (board.queens & pieces).count_ones() as i32 * 900;

    pawns + knights + bishops + rooks + queens
}

/// Calcula janela inicial baseada na configuração e histórico
fn calculate_initial_window(
    config: &AspirationConfig, 
    prev_score: i32, 
    history: &AspirationHistory
) -> i32 {
    let mut window = config.initial_window;

    // Ajusta baseado na estabilidade do score
    if history.score_volatility < 10.0 {
        window = (window as f32 * 0.7) as i32;  // Score estável = janela menor
    } else if history.score_volatility > 80.0 {
        window = (window as f32 * 1.4) as i32;  // Score instável = janela maior
    }

    // Ajusta baseado no último sucesso
    if history.last_successful_window > 0 {
        let success_factor = history.last_successful_window as f32 / 50.0;
        window = (window as f32 * (0.8 + success_factor * 0.4)) as i32;
    }

    // Ajusta para scores extremos
    if prev_score.abs() > 500 {
        window = (window as f32 * 1.3) as i32;  // Scores extremos = janelas maiores
    }

    window.max(8).min(150)
}

/// Expande janela de forma inteligente baseada no padrão de falhas
fn expand_window_intelligently(
    current_window: i32, 
    config: &AspirationConfig, 
    history: &AspirationHistory, 
    failed_high: bool
) -> i32 {
    // Fator de expansão baseado no histórico
    let mut expansion = config.expansion_factor;
    
    // Se falhamos na mesma direção recentemente, expande mais agressivamente
    if history.recent_fails.len() >= 2 {
        let recent_pattern = &history.recent_fails[history.recent_fails.len()-2..];
        if recent_pattern.iter().all(|&x| x) {
            expansion *= 1.3;  // Padrão de falhas = expansão mais agressiva
        }
    }

    // Ajusta expansão baseada na complexidade
    expansion *= config.adaptive_factor;

    // Diferenciação entre fail high e fail low
    let adjusted_expansion = if failed_high {
        expansion * 1.1  // Fail high tipicamente precisa de mais expansão
    } else {
        expansion * 0.9  // Fail low pode ser mais conservador
    };

    let new_window = (current_window as f32 * adjusted_expansion) as i32;
    new_window.min(config.max_window)
}

/// Calcula bound para fail low
fn calculate_fail_low_bound(prev_score: i32, current_window: i32, config: &AspirationConfig) -> i32 {
    // Estratégia: expande mais conservadoramente para baixo
    let expansion = (current_window as f32 * config.expansion_factor * 0.8) as i32;
    (prev_score - expansion).max(-49000)
}

/// Calcula bound para fail high  
fn calculate_fail_high_bound(prev_score: i32, current_window: i32, config: &AspirationConfig) -> i32 {
    // Estratégia: expande mais agressivamente para cima
    let expansion = (current_window as f32 * config.expansion_factor * 1.2) as i32;
    (prev_score + expansion).min(49000)
}

/// Wrapper para compatibilidade com a interface existente
pub fn aspiration_search(
    board: &Board, 
    depth: u8, 
    prev_score: i32, 
    tt: &mut TranspositionTable, 
    context: &mut SearchContext, 
    start_time: Instant, 
    max_time_ms: u64
) -> i32 {
    // Cria histórico simples para esta busca
    let mut history = AspirationHistory::new();
    
    // Se for primeira busca ou profundidade baixa, usa janela conservadora
    if depth <= 3 || prev_score == 0 {
        return pvs_search(board, depth, -50000, 50000, tt, context, start_time, max_time_ms, true);
    }
    
    let result = dynamic_aspiration_search(board, depth, prev_score, tt, context, start_time, max_time_ms, &mut history);
    result.score
}