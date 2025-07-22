// Ficheiro: src/main.rs
// Descrição: Ponto de entrada com o loop principal do protocolo UCI.

use motor_xadrez::{Board, Move};
use motor_xadrez::evaluation;
use motor_xadrez::search;
use motor_xadrez::transposition::TranspositionTable;
use std::io;

fn main() {
    // Inicializa as dependências do motor
    evaluation::init_evaluation_masks();
    let mut board = Board::new();
    let mut tt = TranspositionTable::new(16); // 16 MB

    // Loop principal que espera por comandos da GUI
    loop {
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let commands: Vec<&str> = input.trim().split_whitespace().collect();

        if let Some(&command) = commands.get(0) {
            match command {
                "uci" => {
                    println!("id name MeuMotorRust 1.0");
                    println!("id author OProgramador");
                    // Adicionar opções aqui no futuro, se quisermos (ex: tamanho da TT)
                    println!("uciok");
                }
                "isready" => {
                    println!("readyok");
                }
                "position" => {
                    handle_position_command(&mut board, &commands);
                }
                "go" => {
                    handle_go_command(&board, &mut tt);
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

/// Processa o comando "position"
fn handle_position_command(board: &mut Board, commands: &[&str]) {
    let mut move_start_index = 0;

    if commands.get(1) == Some(&"startpos") {
        *board = Board::new();
        move_start_index = 2;
    } else if commands.get(1) == Some(&"fen") {
        // Encontra o início da string FEN
        // Exemplo: position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 moves e2e4
        let fen_parts: Vec<&str> = commands.iter().skip(2).take_while(|&&c| c != "moves").cloned().collect();
        let fen = fen_parts.join(" ");
        if let Ok(new_board) = Board::from_fen(&fen) {
            *board = new_board;
        }
        move_start_index = 2 + fen_parts.len();
    }

    // Se houver lances após a posição, aplica-os
    if commands.get(move_start_index) == Some(&"moves") {
        for move_str in commands.iter().skip(move_start_index + 1) {
            if let Some(mv) = parse_move(board, move_str) {
                board.make_move(mv);
            }
        }
    }
}

/// Processa o comando "go"
fn handle_go_command(board: &Board, tt: &mut TranspositionTable) {
    // Por agora, usamos uma profundidade fixa. No futuro, podemos ler parâmetros como "depth" ou "movetime".
    let depth = 6;

    if let Some((best_move, _score)) = search::find_best_move(board, depth, tt) {
        // O comando mais importante: envia a melhor jogada de volta para a GUI
        println!("bestmove {}", best_move);
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