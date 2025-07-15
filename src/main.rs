// Motor Xadrez - High-Performance Chess Engine
// Optimized for maximum performance while maintaining correctness

mod board;
mod types;
mod moves;
mod perft;

use board::Board;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    
    if args.len() > 1 {
        match args[1].as_str() {
            "benchmark" => run_benchmark(),
            "--full" => run_full_test(),
            _ => run_quick_test(),
        }
    } else {
        run_quick_test();
    }
}

fn run_quick_test() {
    let board = Board::new();
    println!("Motor Xadrez - Performance Chess Engine");
    println!("Validando corretude (profundidades 1-5)...\n");
    
    let expected = [(1, 20), (2, 400), (3, 8902), (4, 197281), (5, 4865609)];
    
    for &(depth, expected_nodes) in &expected {
        let start = std::time::Instant::now();
        let nodes = perft::perft_driver(&board, depth);
        let elapsed = start.elapsed();
        let nps = (nodes as f64 / elapsed.as_secs_f64()) as u64;
        
        let status = if nodes == expected_nodes { "✓" } else { "✗" };
        println!("Perft({}): {} {} nodes ({:.3}s, {} NPS)", 
                depth, status, nodes, elapsed.as_secs_f64(), format_nps(nps));
    }
    
    println!("\nUso: ./motor_xadrez [benchmark|--full]");
}

fn run_full_test() {
    let board = Board::new();
    println!("Motor Xadrez - Teste Completo (profundidades 1-6)\n");
    
    let expected = [(1, 20), (2, 400), (3, 8902), (4, 197281), (5, 4865609), (6, 119060324)];
    
    for &(depth, expected_nodes) in &expected {
        let start = std::time::Instant::now();
        let nodes = perft::perft_driver(&board, depth);
        let elapsed = start.elapsed();
        let nps = (nodes as f64 / elapsed.as_secs_f64()) as u64;
        
        let status = if nodes == expected_nodes { "✓" } else { "✗" };
        println!("Perft({}): {} {} nodes ({:.3}s, {} NPS)", 
                depth, status, nodes, elapsed.as_secs_f64(), format_nps(nps));
    }
}

fn run_benchmark() {
    let board = Board::new();
    println!("Motor Xadrez - Benchmark de Performance\n");
    
    // Move generation benchmark
    println!("Testando geração de movimentos...");
    let start = std::time::Instant::now();
    let mut _total = 0;
    for _ in 0..100_000 {
        _total += board.generate_all_moves().len();
    }
    let elapsed = start.elapsed();
    let moves_per_sec = (100_000.0 / elapsed.as_secs_f64()) as u64;
    println!("Geração: {} calls/sec\n", format_nps(moves_per_sec));
    
    // Perft benchmark
    println!("Benchmark Perft:");
    for depth in 1..=6 {
        let start = std::time::Instant::now();
        let nodes = perft::perft_driver(&board, depth);
        let elapsed = start.elapsed();
        let nps = (nodes as f64 / elapsed.as_secs_f64()) as u64;
        
        println!("Depth {}: {} nodes ({:.3}s, {} NPS)", 
                depth, nodes, elapsed.as_secs_f64(), format_nps(nps));
                
        if elapsed.as_secs() > 5 { break; } // Stop if taking too long
    }
}

fn format_nps(nps: u64) -> String {
    if nps >= 1_000_000 {
        format!("{:.1}M", nps as f64 / 1_000_000.0)
    } else if nps >= 1_000 {
        format!("{:.1}K", nps as f64 / 1_000.0)
    } else {
        nps.to_string()
    }
}
