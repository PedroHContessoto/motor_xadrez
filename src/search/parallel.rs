// Lazy SMP - Paralelização simples e eficiente para motores de xadrez
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use crate::{board::Board, types::Move, transposition::TranspositionTable};
use super::{SearchContext, find_best::find_best_move_single_thread};

/// Resultado de busca paralela
#[derive(Debug, Clone)]
pub struct ParallelSearchResult {
    pub best_move: Option<Move>,
    pub score: i32,
    pub depth: u8,
    pub nodes: u64,
    pub time_ms: u64,
    pub thread_id: usize,
}

/// Configuração para busca paralela
pub struct ParallelConfig {
    pub num_threads: usize,
    pub max_time_ms: u64,
    pub target_depth: u8,
    pub use_shared_tt: bool, // Compartilhar tabela de transposição
}

impl Default for ParallelConfig {
    fn default() -> Self {
        ParallelConfig {
            num_threads: num_cpus::get().min(8), // Máximo 8 threads
            max_time_ms: 5000,
            target_depth: 12,
            use_shared_tt: true,
        }
    }
}

/// Busca paralela usando Lazy SMP
pub fn parallel_search(
    board: &Board,
    config: ParallelConfig,
) -> Option<Move> {
    if config.num_threads <= 1 {
        // Fallback para busca single-thread
        return find_best_move_single_thread(board, config.target_depth, config.max_time_ms);
    }

    let start_time = Instant::now();
    let board_arc = Arc::new(*board);
    let best_result = Arc::new(Mutex::new(None::<ParallelSearchResult>));
    let should_stop = Arc::new(Mutex::new(false));

    // Cria tabela de transposição compartilhada ou individual por thread
    let shared_tt = if config.use_shared_tt {
        Some(Arc::new(Mutex::new(TranspositionTable::new(64)))) // 64MB compartilhado
    } else {
        None
    };

    let mut handles = Vec::new();

    // Lança threads de busca
    for thread_id in 0..config.num_threads {
        let board_clone = Arc::clone(&board_arc);
        let best_result_clone = Arc::clone(&best_result);
        let should_stop_clone = Arc::clone(&should_stop);
        let shared_tt_clone = shared_tt.clone();
        let config_clone = config.clone();

        let handle = thread::spawn(move || {
            search_thread(
                thread_id,
                board_clone,
                best_result_clone,
                should_stop_clone,
                shared_tt_clone,
                config_clone,
                start_time,
            )
        });

        handles.push(handle);
    }

    // Thread principal monitora tempo
    let timeout_handle = {
        let should_stop_clone = Arc::clone(&should_stop);
        let max_time_ms = config.max_time_ms;
        
        thread::spawn(move || {
            thread::sleep(Duration::from_millis(max_time_ms));
            *should_stop_clone.lock().unwrap() = true;
        })
    };

    // Aguarda threads terminarem
    for handle in handles {
        let _ = handle.join();
    }

    // Para thread de timeout
    *should_stop.lock().unwrap() = true;
    let _ = timeout_handle.join();

    // Retorna melhor resultado
    let result = best_result.lock().unwrap()
        .as_ref()
        .map(|result| result.best_move)
        .flatten();
    result
}

/// Thread individual de busca (Lazy SMP)
fn search_thread(
    thread_id: usize,
    board: Arc<Board>,
    best_result: Arc<Mutex<Option<ParallelSearchResult>>>,
    should_stop: Arc<Mutex<bool>>,
    shared_tt: Option<Arc<Mutex<TranspositionTable>>>,
    config: ParallelConfig,
    start_time: Instant,
) {
    // Cada thread usa profundidade ligeiramente diferente (Lazy SMP)
    let thread_depth = calculate_thread_depth(thread_id, config.target_depth);
    let thread_time = calculate_thread_time(thread_id, config.max_time_ms);

    let mut iteration_depth = 1;
    let mut _nodes_searched = 0;

    // Iterative deepening até a profundidade alvo ou timeout
    while iteration_depth <= thread_depth {
        // Verifica se deve parar
        if *should_stop.lock().unwrap() {
            break;
        }

        let iteration_start = Instant::now();
        
        // Busca iterativa
        let current_best = find_best_move_single_thread(&board, iteration_depth, thread_time / 4);

        if let Some(mv) = current_best {
            // Atualiza resultado global se é melhor
            let result = ParallelSearchResult {
                best_move: Some(mv),
                score: 0, // Simplificado - poderia ter score real
                depth: iteration_depth,
                nodes: _nodes_searched,
                time_ms: start_time.elapsed().as_millis() as u64,
                thread_id,
            };

            let mut global_best = best_result.lock().unwrap();
            let should_update = match global_best.as_ref() {
                None => true,
                Some(current) => {
                    // Prefere profundidade maior, ou thread 0 em caso de empate
                    iteration_depth > current.depth || 
                    (iteration_depth == current.depth && thread_id == 0)
                }
            };

            if should_update {
                *global_best = Some(result);
            }
        }

        iteration_depth += 1;

        // Controle de tempo por iteração
        if iteration_start.elapsed().as_millis() > (thread_time / 2) as u128 {
            break;
        }
    }
}

/// Calcula profundidade específica para cada thread (Lazy SMP)
fn calculate_thread_depth(thread_id: usize, base_depth: u8) -> u8 {
    match thread_id {
        0 => base_depth,          // Thread principal: profundidade alvo
        1 => base_depth + 1,      // Thread 1: +1 profundidade  
        2 => base_depth.saturating_sub(1), // Thread 2: -1 profundidade
        3 => base_depth + 2,      // Thread 3: +2 profundidade
        4 => base_depth.saturating_sub(2), // Thread 4: -2 profundidade
        _ => {
            // Threads adicionais: variação aleatória controlada
            let variation = ((thread_id - 5) % 4) as i8 - 2; // -2, -1, 0, +1
            if variation >= 0 {
                base_depth + variation as u8
            } else {
                base_depth.saturating_sub((-variation) as u8)
            }
        }
    }
}

/// Calcula tempo específico para cada thread
fn calculate_thread_time(thread_id: usize, base_time_ms: u64) -> u64 {
    match thread_id {
        0 => base_time_ms,                    // Thread principal: tempo total
        1 => (base_time_ms as f64 * 0.8) as u64, // 80% do tempo
        2 => (base_time_ms as f64 * 1.2) as u64, // 120% do tempo  
        3 => (base_time_ms as f64 * 0.6) as u64, // 60% do tempo
        _ => (base_time_ms as f64 * 0.9) as u64,  // 90% do tempo para outras
    }
}

impl Clone for ParallelConfig {
    fn clone(&self) -> Self {
        ParallelConfig {
            num_threads: self.num_threads,
            max_time_ms: self.max_time_ms,
            target_depth: self.target_depth,
            use_shared_tt: self.use_shared_tt,
        }
    }
}