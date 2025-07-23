// main.rs - VERSÃO CORRIGIDA
// Principais correções: Time management mais conservador

use std::io::{self, BufRead, Write};
use std::time::Instant;
use motor_xadrez::{Board, types::Move};
use motor_xadrez::transposition::TranspositionTable;
use motor_xadrez::search::{find_best_move_with_time, SearchContext};

struct TimeManager {
    wtime: Option<u64>,
    btime: Option<u64>,
    winc: Option<u64>,
    binc: Option<u64>,
    movetime: Option<u64>,
    depth: Option<u8>,
    infinite: bool,
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
            infinite: false,
        }
    }

    /// CORREÇÃO 1: Time management mais conservador
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
            let tactical_factors = self.analyze_position_complexity(board);

            // CORREÇÃO: Divisores mais conservadores para permitir maior profundidade
            let mut base_divisor = if moves_played < 15 {
                40  // Era 20, agora 40 (mais conservador na abertura)
            } else if moves_played < 35 {
                30  // Era 12, agora 30 (mais conservador no meio-jogo)
            } else {
                35  // Era 15, agora 35 (mais conservador no final)
            };

            // Ajustes mais moderados para posições táticas
            if tactical_factors.is_tactical {
                base_divisor = (base_divisor as f32 * 0.85) as u64; // Era 0.8, agora 0.85
                if tactical_factors.has_hanging_pieces {
                    base_divisor = base_divisor.saturating_sub(2); // Era 3, agora 2
                }
                if tactical_factors.in_check {
                    base_divisor = base_divisor.saturating_sub(2); // Mantido
                }
            }

            // CORREÇÃO: Divisor mínimo mais alto para garantir tempo suficiente
            base_divisor = base_divisor.max(20); // Era 15, agora 20

            let base_time = time_left / base_divisor;
            let increment_bonus = my_inc.unwrap_or(0) / 3; // Era /2, agora /3 (mais conservador)
            let mut time_with_increment = base_time + increment_bonus;

            // Bônus mais moderado para posições críticas
            if tactical_factors.is_critical {
                time_with_increment = (time_with_increment as f32 * 1.15) as u64; // Era 1.2, agora 1.15
            }

            // CORREÇÃO: Limites mais generosos mas realistas
            let min_time = if time_left > 30000 { 1500 } else { 800 }; // Aumentado
            let max_time = if tactical_factors.is_tactical {
                time_left / 6 // Era /5, agora /6 (mais conservador)
            } else {
                time_left / 10 // Era /8, agora /10 (muito mais conservador)
            };

            time_with_increment.max(min_time).min(max_time)
        } else {
            15000 // Era 12000, agora 15 segundos por defeito
        }
    }

    /// CORREÇÃO 2: Profundidade máxima adaptativa mais alta
    fn get_adaptive_depth(&self, board: &Board) -> u8 {
        if let Some(fixed_depth) = self.depth {
            return fixed_depth;
        }

        let tactical_factors = self.analyze_position_complexity(board);

        // CORREÇÃO: Profundidades mais altas por padrão
        let mut max_depth = 80u8; // Era 64, agora 80

        // Ajustes mais moderados baseados na complexidade
        if tactical_factors.is_critical {
            max_depth = 100; // Era 80, agora 100
        } else if tactical_factors.is_tactical {
            max_depth = 90; // Era 72, agora 90
        }

        // Não limita tanto em finais simples
        if tactical_factors.piece_density < 0.25 && !tactical_factors.is_tactical {
            max_depth = 70; // Era 60, agora 70
        }

        max_depth
    }

    fn analyze_position_complexity(&self, board: &Board) -> TacticalFactors {
        let mut factors = TacticalFactors::new();

        // Análise básica de complexidade
        let total_pieces = (board.white_pieces | board.black_pieces).count_ones();
        factors.piece_density = total_pieces as f32 / 32.0;

        // Verifica se está em xeque
        factors.in_check = board.is_king_in_check(board.to_move);

        // Análise de peças penduradas (simplificada)
        factors.has_hanging_pieces = self.has_hanging_pieces_simple(board);

        // Determina se é posição tática
        factors.is_tactical = factors.in_check || factors.has_hanging_pieces ||
            self.has_tactical_motifs(board);

        // Determina se é posição crítica
        factors.is_critical = factors.is_tactical && (factors.in_check ||
            factors.piece_density < 0.4);

        factors
    }

    fn has_hanging_pieces_simple(&self, board: &Board) -> bool {
        // Implementação simplificada
        let our_color = board.to_move;
        let enemy_color = !our_color;
        let our_pieces = if our_color == motor_xadrez::types::Color::White {
            board.white_pieces
        } else {
            board.black_pieces
        };

        let our_valuables = (board.knights | board.bishops | board.rooks | board.queens) & our_pieces;
        let mut bb = our_valuables;

        while bb != 0 {
            let sq = bb.trailing_zeros() as u8;
            bb &= bb - 1;

            if board.is_square_attacked_by(sq, enemy_color) &&
                !board.is_square_attacked_by(sq, our_color) {
                return true;
            }
        }

        false
    }

    fn has_tactical_motifs(&self, board: &Board) -> bool {
        // Verifica motivos táticos básicos
        let legal_moves = board.generate_legal_moves();

        for mv in legal_moves.iter().take(10) { // Verifica apenas os primeiros 10 movimentos
            if board.is_capture(*mv) {
                return true;
            }

            // Verifica se dá xeque
            let mut temp_board = *board;
            temp_board.make_move(*mv);
            if temp_board.is_king_in_check(!board.to_move) {
                return true;
            }
        }

        false
    }
}

#[derive(Debug)]
struct TacticalFactors {
    is_tactical: bool,
    is_critical: bool,
    in_check: bool,
    has_hanging_pieces: bool,
    piece_density: f32,
}

impl TacticalFactors {
    fn new() -> Self {
        TacticalFactors {
            is_tactical: false,
            is_critical: false,
            in_check: false,
            has_hanging_pieces: false,
            piece_density: 1.0,
        }
    }
}

fn main() {
    // Inicialização
    motor_xadrez::evaluation::pawn_structure::init_pawn_masks();

    let mut board = Board::new();
    let mut tt = TranspositionTable::new(128 * 1024 * 1024); // 128 MB
    let mut time_manager = TimeManager::new();
    let mut moves_played = 0u16;

    println!("id name MotorXadrez 2.0");
    println!("id author Pedro Contessoto");
    println!("option name Hash type spin default 16 min 1 max 1024");
    println!("option name Threads type spin default 1 min 1 max 1");
    println!("option name Ponder type check default false");
    println!("option name MultiPV type spin default 1 min 1 max 5");
    println!("option name OwnBook type check default true");
    println!("uciok");

    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = line.unwrap().trim().to_string();
        let parts: Vec<&str> = line.split_whitespace().collect();

        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "uci" => {
                println!("id name MotorXadrez 2.0");
                println!("id author Pedro Contessoto");
                println!("option name Hash type spin default 16 min 1 max 1024");
                println!("option name Threads type spin default 1 min 1 max 1");
                println!("option name Ponder type check default false");
                println!("option name MultiPV type spin default 1 min 1 max 5");
                println!("option name OwnBook type check default true");
                println!("uciok");
            },
            "isready" => {
                println!("readyok");
            },
            "ucinewgame" => {
                board = Board::new();
                tt.clear();
                moves_played = 0;
            },
            "position" => {
                if parts.len() > 1 {
                    if parts[1] == "startpos" {
                        board = Board::new();
                        moves_played = 0;

                        if parts.len() > 2 && parts[2] == "moves" {
                            for move_str in &parts[3..] {
                                if let Some(mv) = parse_move(&board, move_str) {
                                    board.make_move(mv);
                                    moves_played += 1;
                                }
                            }
                        }
                    }
                }
            },
            "go" => {
                // Reset time manager
                time_manager = TimeManager::new();

                // Parse go command
                let mut i = 1;
                while i < parts.len() {
                    match parts[i] {
                        "wtime" => {
                            if i + 1 < parts.len() {
                                time_manager.wtime = parts[i + 1].parse().ok();
                                i += 2;
                            } else {
                                i += 1;
                            }
                        },
                        "btime" => {
                            if i + 1 < parts.len() {
                                time_manager.btime = parts[i + 1].parse().ok();
                                i += 2;
                            } else {
                                i += 1;
                            }
                        },
                        "winc" => {
                            if i + 1 < parts.len() {
                                time_manager.winc = parts[i + 1].parse().ok();
                                i += 2;
                            } else {
                                i += 1;
                            }
                        },
                        "binc" => {
                            if i + 1 < parts.len() {
                                time_manager.binc = parts[i + 1].parse().ok();
                                i += 2;
                            } else {
                                i += 1;
                            }
                        },
                        "movetime" => {
                            if i + 1 < parts.len() {
                                time_manager.movetime = parts[i + 1].parse().ok();
                                i += 2;
                            } else {
                                i += 1;
                            }
                        },
                        "depth" => {
                            if i + 1 < parts.len() {
                                time_manager.depth = parts[i + 1].parse().ok();
                                i += 2;
                            } else {
                                i += 1;
                            }
                        },
                        "infinite" => {
                            time_manager.infinite = true;
                            i += 1;
                        },
                        _ => {
                            i += 1;
                        }
                    }
                }

                // Verifica livro de aberturas primeiro
                let opening_book = motor_xadrez::opening_book::OpeningBook::new();
                if let Some((book_move, opening_name)) = opening_book.get_move(&board) {
                    println!("info string Usando livro: {}", opening_name);
                    println!("bestmove {}", book_move);
                    continue;
                }

                println!("info string Saindo do livro de aberturas - calculando...");

                // Calcula tempo e profundidade
                let max_time = time_manager.calculate_time_for_move(&board, moves_played);
                let max_depth = time_manager.get_adaptive_depth(&board);

                // Busca o melhor movimento
                if let Some((best_move, _score)) = find_best_move_with_time(&board, max_depth, max_time, &mut tt) {
                    println!("bestmove {}", best_move);
                } else {
                    // CORREÇÃO 3: Melhor tratamento quando não há movimento
                    let legal_moves = board.generate_legal_moves();
                    if !legal_moves.is_empty() {
                        println!("bestmove {}", legal_moves[0]);
                    } else {
                        println!("bestmove (none)");
                    }
                }
            },
            "setoption" => {
                if parts.len() >= 5 && parts[1] == "name" {
                    match parts[2] {
                        "Hash" => {
                            if parts[3] == "value" {
                                if let Ok(size_mb) = parts[4].parse::<usize>() {
                                    let size_bytes = size_mb * 1024 * 1024;
                                    tt = TranspositionTable::new(size_bytes);
                                    println!("info string Hash table size set to {} MB", size_mb);
                                }
                            }
                        },
                        "Ponder" => {
                            if parts[3] == "value" {
                                match parts[4] {
                                    "true" => println!("info string Pondering enabled"),
                                    "false" => println!("info string Pondering disabled"),
                                    _ => {}
                                }
                            }
                        },
                        _ => {}
                    }
                }
            },
            "stop" => {
                // Para a busca atual (se houver)
                println!("bestmove (none)");
            },
            "quit" => {
                break;
            },
            _ => {}
        }

        io::stdout().flush().unwrap();
    }
}

fn parse_move(board: &Board, move_str: &str) -> Option<Move> {
    let legal_moves = board.generate_legal_moves();
    for mv in legal_moves {
        let mv_str = mv.to_string();
        if mv_str == move_str {
            return Some(mv);
        }
        // Tenta sem a notação de captura 'x'
        let clean_move_str = move_str.replace("x", "").replace("+", "").replace("#", "");
        if mv_str == clean_move_str {
            return Some(mv);
        }
    }
    None
}
