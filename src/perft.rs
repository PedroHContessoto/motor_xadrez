// Ficheiro: src/perft.rs
// Descrição: Contém a lógica para o teste de performance (Perft),
// uma ferramenta para verificar a corretude da geração de lances.

use crate::board::Board;
use rayon::prelude::*;

/// Função principal do Perft que inicia o teste.
pub fn run_perft(board: &Board, depth: u8) {
    println!("A executar Perft para profundidade {}", depth);
    let start_time = std::time::Instant::now();
    let nodes = perft_driver(board, depth);
    let duration = start_time.elapsed();
    println!("Nós totais: {}", nodes);
    println!("Tempo decorrido: {:?}", duration);
}

/// Função recursiva que percorre a árvore de lances.
fn perft_driver(board: &Board, depth: u8) -> u64 {
    if depth == 0 {
        return 1; // Chegamos a uma folha da árvore de busca.
    }

    // Otimização: para profundidade 1, conta movimentos diretamente
    if depth == 1 {
        let moves = board.generate_all_moves();
        return moves.into_iter()
            .filter(|&mv| {
                let mut new_board = *board;
                new_board.make_move(mv);
                !new_board.is_king_in_check(board.to_move)
            })
            .count() as u64;
    }

    let moves = board.generate_all_moves();
    
    // Para profundidades maiores que 3, usa paralelização
    if depth > 3 {
        moves.into_par_iter()
            .map(|mv| {
                let mut new_board = *board;
                new_board.make_move(mv);
                
                if !new_board.is_king_in_check(board.to_move) {
                    perft_driver(&new_board, depth - 1)
                } else {
                    0
                }
            })
            .sum()
    } else {
        // Para profundidades menores, usa processamento sequencial (evita overhead)
        moves.into_iter()
            .map(|mv| {
                let mut new_board = *board;
                new_board.make_move(mv);
                
                if !new_board.is_king_in_check(board.to_move) {
                    perft_driver(&new_board, depth - 1)
                } else {
                    0
                }
            })
            .sum()
    }
}
