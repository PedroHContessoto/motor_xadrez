// Ficheiro: src/main.rs
// Descrição: Ponto de entrada com o loop principal do protocolo UCI.

use motor_xadrez::{Board, Move};
use motor_xadrez::evaluation;
use motor_xadrez::search;
use motor_xadrez::transposition::TranspositionTable;
use std::io;
use std::time::Instant;

// Estrutura para gerenciar tempos de forma inteligente
#[derive(Debug)]
struct TimeManager {
    wtime: Option<u64>,    // Tempo restante das brancas em ms
    btime: Option<u64>,    // Tempo restante das pretas em ms
    winc: Option<u64>,     // Incremento por lance das brancas em ms
    binc: Option<u64>,     // Incremento por lance das pretas em ms
    movetime: Option<u64>, // Tempo fixo por lance em ms
    depth: Option<u8>,     // Profundidade fixa
    nodes: Option<u64>,    // Número máximo de nós
    infinite: bool,        // Busca infinita
}

impl TimeManager {
    fn new() -> Self {
        TimeManager {
            wtime: None,
            btime: None,
            winc: None,
            binc: None,
            movetime: None,
            depth: None,
            nodes: None,
            infinite: false,
        }
    }
    
    // Calcula o tempo ótimo para este lance baseado na situação
    fn calculate_time_for_move(&self, board: &Board, moves_played: u16) -> u64 {
        if let Some(movetime) = self.movetime {
            return movetime;
        }
        
        if self.infinite {
            return u64::MAX;
        }
        
        let (my_time, my_inc) = if board.to_move == motor_xadrez::types::Color::White {
            (self.wtime, self.winc)
        } else {
            (self.btime, self.binc)
        };
        
        if let Some(time_left) = my_time {
            // Gestão mais agressiva do tempo - usar mais tempo disponível
            let base_divisor = if moves_played < 15 {
                // Abertura: ainda conservador mas não muito
                20  // Era 40, agora 20 (usar mais tempo)
            } else if moves_played < 35 {
                // Meio-jogo: usar bastante tempo para posições complexas
                12  // Era 25, agora 12 (muito mais tempo)
            } else {
                // Final: usar tempo substancial
                15  // Era 30, agora 15 (mais tempo)
            };
            
            let base_time = time_left / base_divisor;
            let increment_bonus = my_inc.unwrap_or(0).saturating_mul(2) / 3; // Usar mais incremento
            let time_with_increment = base_time + increment_bonus;
            
            // Limites mais generosos
            let min_time = if time_left > 10000 { 500 } else { 200 }; // Mínimo maior
            let max_time = time_left / 2; // Pode usar até metade do tempo
            
            time_with_increment.max(min_time).min(max_time)
        } else {
            8000 // 8 segundos por defeito (era 5)
        }
    }
    
    fn get_max_depth(&self) -> u8 {
        self.depth.unwrap_or(50) // Profundidade máxima de 50
    }
}

fn main() {
    // Inicializa as dependências do motor
    motor_xadrez::evaluation::pawn_structure::init_pawn_masks();
    let mut board = Board::new();
    let mut tt = TranspositionTable::new(16); // 16 MB
    let mut moves_played = 0u16;

    // Loop principal que espera por comandos da GUI
    loop {
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let commands: Vec<&str> = input.trim().split_whitespace().collect();

        if let Some(&command) = commands.get(0) {
            match command {
                "uci" => {
                    println!("id name MotorXadrez 2.0");
                    println!("id author Pedro Contessoto");
                    
                    // Opções UCI configuráveis
                    println!("option name Hash type spin default 16 min 1 max 1024");
                    println!("option name Threads type spin default 1 min 1 max 1");
                    println!("option name Ponder type check default false");
                    println!("option name MultiPV type spin default 1 min 1 max 5");
                    
                    println!("uciok");
                }
                "isready" => {
                    println!("readyok");
                }
                "position" => {
                    moves_played = handle_position_command(&mut board, &commands);
                }
                "go" => {
                    handle_go_command(&board, &mut tt, &commands, moves_played);
                }
                "setoption" => {
                    handle_setoption_command(&commands);
                }
                "stop" => {
                    // Para a busca atual (para implementar futuramente com threading)
                    println!("bestmove (none)");
                }
                "quit" => {
                    break; // Sai do loop e termina o programa
                }
                _ => {
                    // Ignora comandos desconhecidos
                }
            }
        }
    }
}

/// Processa o comando "position" e retorna o número de lances jogados
fn handle_position_command(board: &mut Board, commands: &[&str]) -> u16 {
    let mut move_start_index = 0;
    let mut moves_count = 0u16;

    if commands.get(1) == Some(&"startpos") {
        *board = Board::new();
        move_start_index = 2;
    } else if commands.get(1) == Some(&"fen") {
        // Encontra o início da string FEN
        let fen_parts: Vec<&str> = commands.iter().skip(2).take_while(|&&c| c != "moves").cloned().collect();
        let fen = fen_parts.join(" ");
        if let Ok(new_board) = Board::from_fen(&fen) {
            *board = new_board;
            // Extrai o número de lances do FEN se possível
            if let Some(fullmove_str) = fen_parts.last() {
                if let Ok(fullmove) = fullmove_str.parse::<u16>() {
                    moves_count = (fullmove - 1) * 2;
                    if board.to_move == motor_xadrez::types::Color::Black {
                        moves_count += 1;
                    }
                }
            }
        }
        move_start_index = 2 + fen_parts.len();
    }

    // Se houver lances após a posição, aplica-os
    if commands.get(move_start_index) == Some(&"moves") {
        for move_str in commands.iter().skip(move_start_index + 1) {
            if let Some(mv) = parse_move(board, move_str) {
                board.make_move(mv);
                moves_count += 1;
            }
        }
    }
    
    moves_count
}

/// Processa o comando "go" com gestão inteligente de tempo
fn handle_go_command(board: &Board, tt: &mut TranspositionTable, commands: &[&str], moves_played: u16) {
    let mut time_manager = TimeManager::new();
    
    // Processa os parâmetros do comando go
    let mut i = 1;
    while i < commands.len() {
        match commands[i] {
            "wtime" => {
                if i + 1 < commands.len() {
                    time_manager.wtime = commands[i + 1].parse().ok();
                    i += 1;
                }
            },
            "btime" => {
                if i + 1 < commands.len() {
                    time_manager.btime = commands[i + 1].parse().ok();
                    i += 1;
                }
            },
            "winc" => {
                if i + 1 < commands.len() {
                    time_manager.winc = commands[i + 1].parse().ok();
                    i += 1;
                }
            },
            "binc" => {
                if i + 1 < commands.len() {
                    time_manager.binc = commands[i + 1].parse().ok();
                    i += 1;
                }
            },
            "movetime" => {
                if i + 1 < commands.len() {
                    time_manager.movetime = commands[i + 1].parse().ok();
                    i += 1;
                }
            },
            "depth" => {
                if i + 1 < commands.len() {
                    time_manager.depth = commands[i + 1].parse().ok();
                    i += 1;
                }
            },
            "nodes" => {
                if i + 1 < commands.len() {
                    time_manager.nodes = commands[i + 1].parse().ok();
                    i += 1;
                }
            },
            "infinite" => {
                time_manager.infinite = true;
            },
            _ => {}
        }
        i += 1;
    }
    
    // Calcula o tempo para este lance
    let time_for_move = time_manager.calculate_time_for_move(board, moves_played);
    let max_depth = time_manager.get_max_depth();
    
    // Força flush para Arena ver imediatamente
    use std::io::{self, Write};
    
    if let Some((best_move, final_score)) = search::find_best_move_with_time(board, max_depth, time_for_move, tt) {
        println!("bestmove {}", best_move);
        io::stdout().flush().ok();
    } else {
        // Fallback se não encontrar nenhuma jogada
        let legal_moves = board.generate_legal_moves();
        if let Some(fallback_move) = legal_moves.first() {
            println!("bestmove {}", fallback_move);
        } else {
            println!("bestmove (none)");
        }
        io::stdout().flush().ok();
    }
}

/// Processa o comando "setoption"
fn handle_setoption_command(commands: &[&str]) {
    if commands.len() >= 5 && commands[1] == "name" && commands[3] == "value" {
        let option_name = commands[2];
        let option_value = commands[4];
        
        match option_name {
            "Hash" => {
                if let Ok(size) = option_value.parse::<usize>() {
                    println!("info string Hash table size set to {} MB", size);
                    // TODO: Redimensionar a tabela de transposição
                }
            },
            "Threads" => {
                if let Ok(threads) = option_value.parse::<usize>() {
                    println!("info string Thread count set to {}", threads);
                    // TODO: Implementar busca multi-threaded
                }
            },
            "Ponder" => {
                let ponder_enabled = option_value == "true";
                println!("info string Pondering {}", if ponder_enabled { "enabled" } else { "disabled" });
            },
            _ => {
                println!("info string Unknown option: {}", option_name);
            }
        }
    }
}

/// Função auxiliar para converter uma string (ex: "e2e4") num objeto Move
fn parse_move(board: &Board, move_str: &str) -> Option<Move> {
    let legal_moves = board.generate_legal_moves();
    for mv in legal_moves {
        if mv.to_string() == *move_str {
            return Some(mv);
        }
    }
    None
}