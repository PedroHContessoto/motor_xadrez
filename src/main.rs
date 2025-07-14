// Ficheiro: src/main.rs
// Descrição: Ponto de entrada principal da aplicação.

mod board;
mod types;
mod moves;
mod perft; // Declara o nosso novo módulo de teste

use board::Board;

fn main() {
    let board = Board::new();

    println!("--- A INICIAR TESTE PERFT ---");
    
    // Executa o Perft para as profundidades 1 e 2.
    // Os resultados devem ser 20 e 400.
    perft::run_perft(&board, 1);
    perft::run_perft(&board, 2);
    
    // Descomente as linhas abaixo para testes mais profundos (podem demorar mais).
    perft::run_perft(&board, 3);
    perft::run_perft(&board, 4); 


    // Descomente as linhas abaixo para testes mais profundos (podem demorar mais).
    perft::run_perft(&board, 5);
    perft::run_perft(&board, 6); 
    
    
}
