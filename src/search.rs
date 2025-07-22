// Ficheiro: src/search.rs
// Descrição: Versão corrigida e final da lógica de busca Alfa-Beta.

use crate::{board::Board, evaluation, transposition::*, types::{Move, Color, PieceKind}};
use std::time::Instant;

// Estrutura para manter o contexto da busca
#[derive(Debug)]
struct SearchContext {
    killer_moves: [[Option<Move>; 2]; 32], // 2 killers por profundidade (até profundidade 32)
    history: [[i32; 64]; 64], // Tabela de histórico [from][to]
}

impl SearchContext {
    fn new() -> Self {
        SearchContext {
            killer_moves: [[None; 2]; 32],
            history: [[0; 64]; 64],
        }
    }
    
    fn add_killer(&mut self, mv: Move, depth: u8) {
        let depth_idx = depth as usize;
        if depth_idx < 32 {
            // Move o killer anterior para a segunda posição
            self.killer_moves[depth_idx][1] = self.killer_moves[depth_idx][0];
            self.killer_moves[depth_idx][0] = Some(mv);
        }
    }
    
    fn is_killer(&self, mv: Move, depth: u8) -> bool {
        let depth_idx = depth as usize;
        if depth_idx < 32 {
            self.killer_moves[depth_idx].contains(&Some(mv))
        } else {
            false
        }
    }
    
    fn update_history(&mut self, mv: Move, depth: u8) {
        self.history[mv.from as usize][mv.to as usize] += (depth as i32) * (depth as i32);
    }
    
    fn get_history_score(&self, mv: Move) -> i32 {
        self.history[mv.from as usize][mv.to as usize]
    }
}

const PIECE_VALUES: [i32; 6] = [100, 320, 330, 500, 900, 20000]; // Usamos valores maiores para MVV-LVA
const MATE_VALUE: i32 = 99999; // Valor base para mate (maior que rainha*10)

fn order_moves(board: &Board, moves: Vec<Move>, tt: &TranspositionTable, context: &SearchContext, depth: u8) -> Vec<Move> {
    // Tenta obter a melhor jogada da TT para pesquisá-la primeiro
    let tt_move = if let Some(entry) = tt.probe(board.zobrist_hash) {
        entry.best_move
    } else {
        None
    };

    let mut scored_moves = moves.into_iter().map(|mv| {
        let mut score = 0;
        if Some(mv) == tt_move {
            score = 100_000; // Prioridade máxima para a jogada da TT
        } else if board.is_capture(mv) {
            let from_piece = board.get_piece_on_square(mv.from).unwrap_or(PieceKind::Pawn);
            let to_piece = board.get_piece_on_square(mv.to).unwrap_or(PieceKind::Pawn);
            score = PIECE_VALUES[to_piece as usize] * 10 - PIECE_VALUES[from_piece as usize] + 10_000;
        } else if context.is_killer(mv, depth) {
            score = 9_000; // Prioridade alta para killer moves
        } else {
            score = context.get_history_score(mv); // Pontuação do histórico
        }
        (mv, score)
    }).collect::<Vec<(Move, i32)>>();

    scored_moves.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    scored_moves.into_iter().map(|(mv, _)| mv).collect()
}

pub fn find_best_move(board: &Board, max_depth: u8, tt: &mut TranspositionTable) -> Option<(Move, i32)> {
    find_best_move_with_time(board, max_depth, 5000, tt) // 5 segundos por defeito
}

pub fn find_best_move_with_time(board: &Board, max_depth: u8, max_time_ms: u64, tt: &mut TranspositionTable) -> Option<(Move, i32)> {
    let start_time = Instant::now();
    let mut context = SearchContext::new();
    let mut best_move = None;
    let mut best_score = 0;
    
    // Iterative deepening
    for depth in 1..=max_depth {
        if start_time.elapsed().as_millis() as u64 > max_time_ms {
            break;
        }
        
        let score = alphabeta(board, depth, i32::MIN + 1, i32::MAX - 1, board.to_move == Color::White, tt, &mut context);
        
        // Obtém a melhor jogada da tabela de transposição após cada iteração
        if let Some(entry) = tt.probe(board.zobrist_hash) {
            if let Some(mv) = entry.best_move {
                best_move = Some(mv);
                best_score = score;
                println!("Profundidade {}: melhor jogada {}, pontuação {}", depth, mv, score);
            }
        }
        
        // Para no caso de mate forçado encontrado
        if score.abs() > MATE_VALUE - 100 {
            break;
        }
    }

    let duration = start_time.elapsed();
    println!("Tempo de busca: {:?}", duration);

    best_move.map(|mv| (mv, best_score))
}

fn alphabeta(board: &Board, depth: u8, mut alpha: i32, mut beta: i32, is_maximizing: bool, tt: &mut TranspositionTable, context: &mut SearchContext) -> i32 {
    let original_alpha = alpha;
    if let Some(entry) = tt.probe(board.zobrist_hash) {
        if entry.depth >= depth {
            match entry.entry_type {
                EntryType::Exact => return entry.score,
                EntryType::LowerBound => alpha = alpha.max(entry.score),
                EntryType::UpperBound => beta = beta.min(entry.score),
            }
            if alpha >= beta {
                return entry.score;
            }
        }
    }

    // Checa draws não-terminais cedo (sem gerar moves)
    if board.is_draw_by_50_moves() || board.is_draw_by_insufficient_material() {
        return 0;
    }

    // Null Move Pruning
    if depth >= 3 && !board.is_king_in_check(board.to_move) && !is_maximizing {
        let mut null_board = *board;
        null_board.to_move = !null_board.to_move;
        null_board.en_passant_target = None;
        let null_score = -alphabeta(&null_board, depth - 3, -beta, -beta + 1, !is_maximizing, tt, context);
        if null_score >= beta {
            return beta;
        }
    }

    if depth == 0 {
        return quiescence(board, alpha, beta, is_maximizing, tt, context);
    }

    let mut best_move = None;
    let legal_moves = board.generate_legal_moves();
    if legal_moves.is_empty() {
        if board.is_king_in_check(board.to_move) {
            // Checkmate: current player perde
            let mate_score = MATE_VALUE - depth as i32; // Prefere mates mais rápidos
            return if board.to_move == Color::White { -mate_score } else { mate_score };
        } else {
            // Stalemate
            return 0;
        }
    }
    let ordered_moves = order_moves(board, legal_moves, tt, context, depth);

    if is_maximizing {
        let mut max_eval = i32::MIN;
        for mv in ordered_moves {
            let mut temp_board = *board;
            temp_board.make_move(mv);
            
            // Extensões de busca
            let extension = if temp_board.is_king_in_check(!board.to_move) || mv.promotion.is_some() { 1 } else { 0 };
            
            let eval = alphabeta(&temp_board, depth - 1 + extension, alpha, beta, false, tt, context);
            if eval > max_eval {
                max_eval = eval;
                best_move = Some(mv);
                // Atualiza histórico para movimentos bons
                if !board.is_capture(mv) {
                    context.update_history(mv, depth);
                }
            }
            alpha = alpha.max(eval);
            if beta <= alpha { 
                // Beta cutoff - adiciona killer move se não for captura
                if !board.is_capture(mv) {
                    context.add_killer(mv, depth);
                }
                break; 
            }
        }
        let entry_type = if max_eval <= original_alpha { EntryType::UpperBound }
        else if max_eval >= beta { EntryType::LowerBound }
        else { EntryType::Exact };
        tt.store(board.zobrist_hash, best_move, max_eval, depth, entry_type);
        max_eval
    } else {
        let mut min_eval = i32::MAX;
        for mv in ordered_moves {
            let mut temp_board = *board;
            temp_board.make_move(mv);
            
            // Extensões de busca
            let extension = if temp_board.is_king_in_check(!board.to_move) || mv.promotion.is_some() { 1 } else { 0 };
            
            let eval = alphabeta(&temp_board, depth - 1 + extension, alpha, beta, true, tt, context);
            if eval < min_eval {
                min_eval = eval;
                best_move = Some(mv);
                // Atualiza histórico para movimentos bons
                if !board.is_capture(mv) {
                    context.update_history(mv, depth);
                }
            }
            beta = beta.min(eval);
            if beta <= alpha { 
                // Beta cutoff - adiciona killer move se não for captura
                if !board.is_capture(mv) {
                    context.add_killer(mv, depth);
                }
                break; 
            }
        }
        let entry_type = if min_eval <= original_alpha { EntryType::UpperBound }
        else if min_eval >= beta { EntryType::LowerBound }
        else { EntryType::Exact };
        tt.store(board.zobrist_hash, best_move, min_eval, depth, entry_type);
        min_eval
    }
}

// Nova função: Quiescence Search (busca só capturas para evitar horizon effect)
fn quiescence(board: &Board, mut alpha: i32, mut beta: i32, is_maximizing: bool, tt: &mut TranspositionTable, context: &mut SearchContext) -> i32 {
    let stand_pat = evaluation::evaluate(board);
    
    // Delta pruning - não avalia capturas que não podem melhorar alpha/beta
    if is_maximizing {
        if stand_pat + 900 < alpha { // 900 = valor da rainha, maior peça capturável
            return stand_pat;
        }
        alpha = alpha.max(stand_pat);
    } else {
        if stand_pat - 900 > beta {
            return stand_pat;
        }
        beta = beta.min(stand_pat);
    }
    if alpha >= beta {
        return stand_pat;
    }

    // Gera só capturas (pseudo-legais, filtra legais)
    let mut captures: Vec<Move> = Vec::new();
    captures.extend(crate::moves::pawn::generate_pawn_captures(board));
    captures.extend(crate::moves::knight::generate_knight_moves(board).into_iter().filter(|mv| board.is_capture(*mv)));
    // Adicionar movimentos de xeque se não estiver em xeque
    if !board.is_king_in_check(board.to_move) {
        // Para simplicidade, vamos usar todos os movimentos e filtrar os que dão xeque
        let all_moves = board.generate_legal_moves();
        for mv in all_moves {
            if !board.is_capture(mv) {
                let mut temp_board = *board;
                temp_board.make_move(mv);
                if temp_board.is_king_in_check(!board.to_move) {
                    captures.push(mv); // Adiciona movimentos que dão xeque
                }
            }
        }
    }

    let ordered_captures = order_moves(board, captures, tt, context, 0); // Usa mesma ordenação MVV-LVA

    if is_maximizing {
        let mut max_eval = stand_pat;
        for mv in ordered_captures {
            if !board.is_legal_move(mv) { continue; }
            let mut temp_board = *board;
            temp_board.make_move(mv);
            let eval = quiescence(&temp_board, alpha, beta, false, tt, context);
            max_eval = max_eval.max(eval);
            alpha = alpha.max(eval);
            if beta <= alpha { break; }
        }
        max_eval
    } else {
        let mut min_eval = stand_pat;
        for mv in ordered_captures {
            if !board.is_legal_move(mv) { continue; }
            let mut temp_board = *board;
            temp_board.make_move(mv);
            let eval = quiescence(&temp_board, alpha, beta, true, tt, context);
            min_eval = min_eval.min(eval);
            beta = beta.min(eval);
            if beta <= alpha { break; }
        }
        min_eval
    }
}