// find_best.rs - VERSÃO CORRIGIDA
// Busca iterativa com melhor gestão de tempo e profundidade

use std::time::Instant;
use crate::{board::Board, evaluation, transposition::TranspositionTable, types::Move};
use super::{SearchContext, aspiration::aspiration_search, pvs::pvs_search, ordering::order_root_moves};

/// Encontra o melhor movimento com limite de tempo
pub fn find_best_move_with_time(
    board: &Board,
    max_depth: u8,
    mut max_time_ms: u64,
    tt: &mut TranspositionTable
) -> Option<(Move, i32)> {
    let start_time = Instant::now();
    let mut context = SearchContext::new();
    let mut best_move = None;
    let mut best_score = 0;
    let mut prev_score = 0;
    let mut stable_count = 0;

    let legal_moves = board.generate_legal_moves();
    if legal_moves.is_empty() {
        return None;
    }
    let mut fallback_move = legal_moves[0];
    let mut fallback_score = evaluation::evaluate(board);

    let is_tactical_position = detect_tactical_position(board);
    // CORREÇÃO 1: Multiplicador de tempo mais moderado
    let tactical_time_multiplier = if is_tactical_position { 1.2 } else { 1.0 }; // Era 1.3, agora 1.2
    let effective_time_limit = (max_time_ms as f32 * tactical_time_multiplier) as u64;

    // CRÍTICO: PROFUNDIDADE MÍNIMA ABSOLUTA 7 (não pode parar antes)
    const MIN_DEPTH_ABSOLUTE: u8 = 7;

    // CORREÇÃO 2: Profundidade mínima mais alta e limite maior
    for depth in 1..=max_depth.min(50) { // Era 32, agora 50
        let elapsed = start_time.elapsed().as_millis() as u64;
        
        // Só permite parar por tempo se JÁ atingiu profundidade mínima
        if depth >= MIN_DEPTH_ABSOLUTE {
            if elapsed > effective_time_limit {
                if !is_tactical_position || depth > 14 {
                    break;
                }
            }
            
            // Timeout absoluto mais generoso (só após depth mínima)
            let absolute_limit = if is_tactical_position {
                max_time_ms.saturating_mul(3)
            } else {
                max_time_ms.saturating_mul(2)
            };

            if elapsed > absolute_limit {
                break;
            }
        }
        
        // TIMEOUT EXTREMO: Só para se passar muito tempo E já ter depth mínima
        if elapsed > max_time_ms.saturating_mul(10) && depth >= MIN_DEPTH_ABSOLUTE {
            break;
        }

        // Usa aspiration search para profundidades maiores
        let score = if depth >= 4 && best_move.is_some() {
            aspiration_search(board, depth, prev_score, tt, &mut context, start_time, max_time_ms)
        } else {
            pvs_search(board, depth, -50000, 50000, tt, &mut context, start_time, max_time_ms, true)
        };

        // Verifica se a busca foi interrompida
        if context.should_stop {
            break;
        }

        // Atualiza melhor movimento
        if let Some(entry) = tt.probe(board.zobrist_hash) {
            if let Some(mv) = entry.best_move {
                if board.is_legal_move(mv) {
                    best_move = Some(mv);
                    best_score = score;
                    fallback_move = mv;
                    fallback_score = score;
                }
            }
        }

        // CORREÇÃO 5: Critério de estabilidade mais relaxado
        if (score - prev_score).abs() < 20 { // Era 15, agora 20
            stable_count += 1;
        } else {
            stable_count = 0;
        }

        // CORREÇÃO 6: Para busca se muito estável e profundidade razoável (após mínima)
        if stable_count >= 3 && depth >= 12 && depth >= MIN_DEPTH_ABSOLUTE {
            break;
        }

        prev_score = score;

        // Informações de debug
        if let Some(entry) = tt.probe(board.zobrist_hash) {
            if let Some(mv) = entry.best_move {
                let pv_str = extract_pv(board, tt, 5);
                println!("info depth {} score cp {} nodes {} nps {} time {} pv {}",
                         depth,
                         score,
                         context.nodes_searched,
                         if elapsed > 0 { context.nodes_searched * 1000 / elapsed } else { 0 },
                         elapsed,
                         pv_str
                );
            }
        }

        // CORREÇÃO 7: Gestão de tempo mais inteligente (SÓ APÓS profundidade mínima)
        if depth >= MIN_DEPTH_ABSOLUTE {
            let time_per_depth = if depth > 1 { elapsed / (depth as u64) } else { elapsed };
            let estimated_next_time = time_per_depth * 3; // Estima próxima profundidade

            if elapsed + estimated_next_time > effective_time_limit {
                break;
            }
        }
    }

    // Retorna o melhor movimento encontrado
    if let Some(mv) = best_move {
        Some((mv, best_score))
    } else {
        Some((fallback_move, fallback_score))
    }
}

/// Versão simplificada sem limite de tempo
pub fn find_best_move(board: &Board, depth: u8, tt: &mut TranspositionTable) -> Option<(Move, i32)> {
    find_best_move_with_time(board, depth, u64::MAX, tt)
}

/// Detecta se a posição é tática
fn detect_tactical_position(board: &Board) -> bool {
    // 1. Verifica se está em xeque
    if board.is_king_in_check(board.to_move) {
        return true;
    }

    // 2. Verifica se há peças penduradas
    if has_hanging_pieces(board) {
        return true;
    }

    // 3. Verifica se há capturas vantajosas
    let legal_moves = board.generate_legal_moves();
    for mv in legal_moves.iter().take(15) { // Verifica apenas os primeiros 15
        if board.is_capture(*mv) {
            // Usa SEE simplificado
            if see_simple(board, *mv) > 0 {
                return true;
            }
        }

        // Verifica se dá xeque
        let mut temp_board = *board;
        temp_board.make_move(*mv);
        if temp_board.is_king_in_check(!board.to_move) {
            return true;
        }
    }

    // 4. Verifica densidade de peças (posições abertas são mais táticas)
    let total_pieces = (board.white_pieces | board.black_pieces).count_ones();
    if total_pieces < 20 { // Menos de 20 peças = posição aberta
        return true;
    }

    false
}

/// Verifica se há peças penduradas
fn has_hanging_pieces(board: &Board) -> bool {
    let our_color = board.to_move;
    let enemy_color = !our_color;
    let our_pieces = if our_color == crate::types::Color::White {
        board.white_pieces
    } else {
        board.black_pieces
    };

    // Verifica peças valiosas
    let our_valuables = (board.knights | board.bishops | board.rooks | board.queens) & our_pieces;
    let mut bb = our_valuables;

    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        bb &= bb - 1;

        // Se atacada pelo inimigo e não defendida por nós
        if board.is_square_attacked_by(sq, enemy_color) &&
            !board.is_square_attacked_by(sq, our_color) {
            return true;
        }
    }

    false
}

/// SEE simplificado
fn see_simple(board: &Board, mv: Move) -> i32 {
    if !board.is_capture(mv) {
        return 0;
    }

    // Valores das peças
    const PIECE_VALUES: [i32; 6] = [100, 320, 330, 500, 900, 20000];

    let attacker_piece = board.get_piece_on_square(mv.from);
    let victim_piece = board.get_piece_on_square(mv.to);

    if let (Some(attacker), Some(victim)) = (attacker_piece, victim_piece) {
        let gain = PIECE_VALUES[victim as usize];
        let loss = PIECE_VALUES[attacker as usize];

        // Estimativa simples: ganho - perda se não defendida
        if board.is_square_attacked_by(mv.to, !board.to_move) {
            gain - loss
        } else {
            gain
        }
    } else {
        0
    }
}

/// Extrai linha principal da transposition table
fn extract_pv(board: &Board, tt: &TranspositionTable, max_depth: usize) -> String {
    let mut pv = Vec::new();
    let mut current_board = *board;

    for _ in 0..max_depth {
        if let Some(entry) = tt.probe(current_board.zobrist_hash) {
            if let Some(mv) = entry.best_move {
                if current_board.is_legal_move(mv) {
                    pv.push(mv.to_string());
                    current_board.make_move(mv);
                } else {
                    break;
                }
            } else {
                break;
            }
        } else {
            break;
        }
    }

    pv.join(" ")
}

/// Ordena movimentos para busca root
fn order_moves_for_root(board: &Board, moves: Vec<Move>) -> Vec<Move> {
    order_root_moves(board, moves)
}