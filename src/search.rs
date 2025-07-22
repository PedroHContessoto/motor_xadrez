// Ficheiro: src/search.rs
// Descrição: Versão corrigida e final da lógica de busca Alfa-Beta.

use crate::{board::Board, evaluation, transposition::*, types::{Move, PieceKind}};
use std::time::Instant;

// Estrutura para manter o contexto da busca
#[derive(Debug)]
struct SearchContext {
    killer_moves: [[Option<Move>; 2]; 32], // 2 killers por profundidade (até profundidade 32)
    history: [[i32; 64]; 64], // Tabela de histórico [from][to]
    nodes_searched: u64, // Contador de nós pesquisados
}

impl SearchContext {
    fn new() -> Self {
        SearchContext {
            killer_moves: [[None; 2]; 32],
            history: [[0; 64]; 64],
            nodes_searched: 0,
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

// Constantes para Late Move Reduction
const LMR_MIN_DEPTH: u8 = 3;
const LMR_MIN_MOVES: usize = 4;
const LMR_REDUCTION: u8 = 1;

// Constantes para Futility Pruning
const FUTILITY_MARGIN: [i32; 4] = [0, 200, 300, 500];

fn order_moves(board: &Board, moves: Vec<Move>, tt: &TranspositionTable, context: &SearchContext, depth: u8) -> Vec<Move> {
    // Tenta obter a melhor jogada da TT para pesquisá-la primeiro
    let tt_move = if let Some(entry) = tt.probe(board.zobrist_hash) {
        if let Some(mv) = entry.best_move {
            // Valida se o move da TT é legal
            if board.is_legal_move(mv) {
                Some(mv)
            } else {
                None // Ignora move ilegal da TT
            }
        } else {
            None
        }
    } else {
        None
    };

    let mut scored_moves = moves.into_iter().map(|mv| {
        let mut score = 0;
        if Some(mv) == tt_move {
            score = 100_000; // Prioridade máxima para a jogada da TT
        } else if mv.is_castling {
            score = 15_000; // Alta prioridade para castling
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
    let mut prev_score = 0;
    
    // Iterative deepening com aspiration windows
    for depth in 1..=max_depth {
        let elapsed = start_time.elapsed().as_millis() as u64;
        
        // Para se não tiver tempo suficiente para a próxima profundidade
        // Estima que a próxima profundidade levará ~3x mais tempo
        if depth > 3 && elapsed * 3 > max_time_ms {
            break;
        }
        
        // Para só se ultrapassou o tempo limite
        if elapsed > max_time_ms {
            break;
        }
        
        let score = if depth > 2 {
            // Usa aspiration windows para profundidades maiores
            aspiration_search(board, depth, prev_score, tt, &mut context, start_time, max_time_ms)
        } else {
            // Busca completa para as primeiras profundidades
            pvs_search(board, depth, -50000, 50000, tt, &mut context, start_time, max_time_ms, true)
        };
        
        // Obtém a melhor jogada da tabela de transposição após cada iteração
        if let Some(entry) = tt.probe(board.zobrist_hash) {
            if let Some(mv) = entry.best_move {
                // IMPORTANTE: Valida se o move da TT é legal na posição atual
                if board.is_legal_move(mv) {
                    best_move = Some(mv);
                    best_score = score;
                } else {
                    // Move ilegal da TT - usa fallback
                    let legal_moves = board.generate_legal_moves();
                    if let Some(fallback_mv) = legal_moves.first() {
                        best_move = Some(*fallback_mv);
                        best_score = score;
                    }
                }
                let nps = if start_time.elapsed().as_millis() > 0 {
                    (context.nodes_searched as f64 / (start_time.elapsed().as_millis() as f64 / 1000.0)) as u64
                } else {
                    0
                };
                let time_ms = start_time.elapsed().as_millis() as u64;
                
                // UCI info output - formato exato que Arena espera
                // Debug: mostrar score real
                if score.abs() > 10000 {
                    println!("info string SCORE ALTO: {} em profundidade {}", score, depth);
                }
                let display_score = score.clamp(-10000, 10000);
                println!("info depth {} score cp {} nodes {} nps {} time {} pv {}", 
                        depth, display_score, context.nodes_searched, nps, time_ms, mv);
                
                // Força flush do stdout para Arena ver imediatamente  
                use std::io::{self, Write};
                io::stdout().flush().ok();
            }
        }
        
        prev_score = score;
        
        
        // Para no caso de mate forçado encontrado - remover para debug
        // if score.abs() > MATE_VALUE - 1000 {
        //     break;
        // }
    }

    let duration = start_time.elapsed();
    let nps = if duration.as_millis() > 0 {
        (context.nodes_searched as f64 / (duration.as_millis() as f64 / 1000.0)) as u64
    } else {
        0
    };
    
    // Final summary removido para Arena

    best_move.map(|mv| (mv, best_score))
}

// Aspiration Windows: busca com janelas mais estreitas para melhor performance
fn aspiration_search(board: &Board, depth: u8, prev_score: i32, tt: &mut TranspositionTable, context: &mut SearchContext, start_time: Instant, max_time_ms: u64) -> i32 {
    let mut window = 25;
    let mut alpha = prev_score - window;
    let mut beta = prev_score + window;
    
    loop {
        let score = pvs_search(board, depth, alpha, beta, tt, context, start_time, max_time_ms, true);
        
        if score <= alpha {
            // Falha em alpha - amplia janela para baixo
            alpha = -50000;
            window *= 2;
        } else if score >= beta {
            // Falha em beta - amplia janela para cima
            beta = 50000;
            window *= 2;
        } else {
            // Sucesso - score dentro da janela
            return score;
        }
        
        // Limite de segurança para evitar loops infinitos
        if window > 1000 {
            return pvs_search(board, depth, -50000, 50000, tt, context, start_time, max_time_ms, true);
        }
    }
}

// Principal Variation Search - versão mais eficiente do alpha-beta (negamax puro)
fn pvs_search(board: &Board, depth: u8, mut alpha: i32, mut beta: i32, tt: &mut TranspositionTable, context: &mut SearchContext, start_time: Instant, max_time_ms: u64, is_pv_node: bool) -> i32 {
    context.nodes_searched += 1;
    
    // Verificação de tempo
    if context.nodes_searched % 4096 == 0 {
        if start_time.elapsed().as_millis() as u64 > max_time_ms {
            return 0; // Timeout
        }
    }
    
    let original_alpha = alpha;
    
    // Transposition Table lookup
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

    // Null Move Pruning (apenas para non-PV nodes)
    if depth >= 3 && !is_pv_node && !board.is_king_in_check(board.to_move) {
        let mut null_board = *board;
        null_board.to_move = !null_board.to_move;
        null_board.en_passant_target = None;
        let temp_null_score = pvs_search(&null_board, depth - 3, -beta, -beta + 1, tt, context, start_time, max_time_ms, false);
        let null_score = -temp_null_score;
        if null_score >= beta {
            return beta;
        }
    }

    if depth == 0 {
        return quiescence_search(board, alpha, beta, tt, context);
    }

    let mut best_move = None;
    let legal_moves = board.generate_legal_moves();
    
    if legal_moves.is_empty() {
        if board.is_king_in_check(board.to_move) {
            // Checkmate: current player perde - sempre retorna negativo (perda)
            return -(MATE_VALUE - depth as i32);
        } else {
            // Stalemate
            return 0;
        }
    }
    
    let ordered_moves = order_moves(board, legal_moves, tt, context, depth);
    let mut best_score = -50000; // Negamax: sempre maximiza
    let mut moves_searched = 0;

    for mv in ordered_moves {
        let mut temp_board = *board;
        temp_board.make_move(mv);
        
        // Futility Pruning - poda movimentos obviamente ruins em profundidades rasas
        if depth <= 3 && !is_pv_node && !board.is_capture(mv) && !board.is_king_in_check(board.to_move) {
            let static_eval = evaluation::evaluate(board); // Agora já é relativo ao jogador atual
            let futility_margin = FUTILITY_MARGIN[depth as usize];
            if static_eval + futility_margin <= alpha {
                continue;
            }
        }
        
        // Extensões de busca
        let extension = if temp_board.is_king_in_check(!board.to_move) || mv.promotion.is_some() { 1 } else { 0 };
        
        let mut score;
        
        if moves_searched == 0 {
            // Primeira jogada: busca completa
            let first_depth = if depth > 1 { depth - 1 + extension } else { extension };
            let first_score = pvs_search(&temp_board, first_depth, -beta, -alpha, tt, context, start_time, max_time_ms, is_pv_node);
            score = -first_score;
        } else {
            // Late Move Reduction
            let mut reduction = 0;
            if depth >= LMR_MIN_DEPTH && moves_searched >= LMR_MIN_MOVES && !is_pv_node 
                && !board.is_capture(mv) && !temp_board.is_king_in_check(!board.to_move) {
                reduction = LMR_REDUCTION;
            }
            
            // Primeiro tenta busca com janela nula (PVS)
            let search_depth = if depth > 1 + reduction { depth - 1 - reduction + extension } else { 0 };
            let temp_score = pvs_search(&temp_board, search_depth, -alpha - 1, -alpha, tt, context, start_time, max_time_ms, false);
            score = -temp_score;
            
            // Se falhou e é uma busca PV, re-busca com janela completa
            if score > alpha && is_pv_node {
                let full_depth = if depth > 1 { depth - 1 + extension } else { extension };
                let full_score = pvs_search(&temp_board, full_depth, -beta, -alpha, tt, context, start_time, max_time_ms, true);
                score = -full_score;
            }
        }
        
        moves_searched += 1;
        
        // Negamax: sempre maximiza o score (após negação)
        if score > best_score {
            best_score = score;
            best_move = Some(mv);
            if !board.is_capture(mv) {
                context.update_history(mv, depth);
            }
        }
        
        alpha = alpha.max(score);
        if alpha >= beta {
            if !board.is_capture(mv) {
                context.add_killer(mv, depth);
            }
            break; // Beta cutoff
        }
    }

    let entry_type = if best_score <= original_alpha { EntryType::UpperBound }
    else if best_score >= beta { EntryType::LowerBound }
    else { EntryType::Exact };
    
    tt.store(board.zobrist_hash, best_move, best_score, depth, entry_type);
    best_score
}

// Quiescence Search melhorada (busca só capturas para evitar horizon effect)
fn quiescence_search(board: &Board, mut alpha: i32, beta: i32, tt: &mut TranspositionTable, context: &mut SearchContext) -> i32 {
    let stand_pat = evaluation::evaluate(board); // Já é relativo ao jogador atual
    
    if stand_pat >= beta {
        return beta;
    }
    
    if alpha < stand_pat {
        alpha = stand_pat;
    }
    
    // Delta pruning - não avalia capturas que não podem melhorar alpha
    if stand_pat + 900 < alpha {
        return alpha;
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

    for mv in ordered_captures {
        if !board.is_legal_move(mv) { continue; }
        let mut temp_board = *board;
        temp_board.make_move(mv);
        let score = -quiescence_search(&temp_board, -beta, -alpha, tt, context);
        if score >= beta {
            return beta;
        }
        if score > alpha {
            alpha = score;
        }
    }
    
    alpha
}
