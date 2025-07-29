// Módulo principal de avaliação modular
// Descrição: Coordena todas as subfunções de avaliação

pub mod material;
pub mod king_safety;
pub mod threats;
pub mod mobility;
pub mod pawn_structure;
pub mod game_phase;
pub mod cache;
mod utils;
mod endgame_patterns;

use crate::{board::Board, types::Color};
use cache::EvaluationCache;
use std::sync::Mutex;

// Cache global de avaliação (thread-safe)
lazy_static::lazy_static! {
    static ref EVAL_CACHE: Mutex<EvaluationCache> = Mutex::new(EvaluationCache::new(16384)); // 16K entradas
}

/// Função principal de avaliação com cache (interface pública)
pub fn evaluate(board: &Board) -> i32 {
    // Tenta buscar no cache primeiro
    if let Ok(mut cache) = EVAL_CACHE.lock() {
        if let Some(cached_score) = cache.probe(board.zobrist_hash) {
            return cached_score;
        }
    }

    // Se não encontrou no cache, calcula normalmente
    let score = evaluate_uncached(board);

    // Armazena no cache
    if let Ok(mut cache) = EVAL_CACHE.lock() {
        cache.store(board.zobrist_hash, score, 0);
    }

    score
}

/// Função de avaliação sem cache (interna)
fn evaluate_uncached(board: &Board) -> i32 {
    let game_phase = game_phase::detect_game_phase(board);
    let phase_info = game_phase::detect_game_phase_advanced(board);

    let white_score = evaluate_color(board, Color::White, &game_phase);
    let black_score = evaluate_color(board, Color::Black, &game_phase);

    let mut final_score = white_score - black_score;

    // Adiciona tempo/iniciativa
    final_score += evaluate_tempo(board);

    // Material Safety Net: penaliza avaliações excessivamente otimistas
    final_score = apply_material_safety_net(board, final_score);

    // Retorna relativo ao jogador atual
    if board.to_move == Color::White {
        final_score
    } else {
        -final_score
    }
}

/// Função para limpar o cache de avaliação
pub fn clear_eval_cache() {
    if let Ok(mut cache) = EVAL_CACHE.lock() {
        cache.clear();
    }
}

/// Função para obter estatísticas do cache
pub fn get_eval_cache_stats() -> (u64, u64, f64, usize) {
    if let Ok(cache) = EVAL_CACHE.lock() {
        cache.get_stats()
    } else {
        (0, 0, 0.0, 0)
    }
}

/// Avalia cor específica
fn evaluate_color(board: &Board, color: Color, game_phase: &game_phase::GamePhase) -> i32 {
    let mut score = 0;

    // Material + PST
    score += material::evaluate_material_and_pst(board, color, game_phase);

    // Estrutura de peões (incluindo passados) - CAP: ±120
    score += pawn_structure::evaluate_pawn_structure(board, color).clamp(-120, 120);

    // Mobilidade segura - CAP: ±100
    score += mobility::evaluate_mobility(board, color).clamp(-100, 100);

    // Segurança do rei (aprimorada) - CAP: ±150
    score += king_safety::evaluate_king_safety(board, color, game_phase).clamp(-150, 150);

    // NOVO: Avaliação de ameaças (peças penduradas, ataques) - CAP: ±80
    score += threats::evaluate_threats(board, color).clamp(-80, 80);

    // Avaliações específicas por fase
    match game_phase {
        game_phase::GamePhase::Opening => {
            score += evaluate_development(board, color);
        },
        game_phase::GamePhase::Endgame => {
            score += evaluate_king_activity(board, color);
            // Nova avaliação: padrões avançados de endgame
            let endgame_patterns = endgame_patterns::evaluate_endgame_patterns(board, color);
            score += endgame_patterns.total_score().clamp(-100, 100);
        },
        _ => {}
    }

    score
}

/// Avalia o tempo (iniciativa)
fn evaluate_tempo(board: &Board) -> i32 {
    let current_moves = board.generate_legal_moves().len() as i32;

    let mut temp_board = *board;
    temp_board.to_move = !temp_board.to_move;
    let opponent_moves = temp_board.generate_legal_moves().len() as i32;

    ((current_moves - opponent_moves) * 2).clamp(-50, 50)
}

/// Avalia desenvolvimento na abertura
fn evaluate_development(board: &Board, color: Color) -> i32 {
    let mut score = 0;
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let back_rank = if color == Color::White { 0xFF } else { 0xFF00000000000000 };

    // Penaliza peças ainda no rank inicial
    let knights_undeveloped = (board.knights & pieces & back_rank).count_ones() as i32 * -15;
    let bishops_undeveloped = (board.bishops & pieces & back_rank).count_ones() as i32 * -15;

    // Avaliação de castling melhorada (do código original)
    score += evaluate_castling(board, color);

    score + knights_undeveloped + bishops_undeveloped
}

/// Avalia castling
fn evaluate_castling(board: &Board, color: Color) -> i32 {
    let mut score = 0;

    if color == Color::White {
        let white_king = board.kings & board.white_pieces;
        if white_king != 0 {
            let king_square = white_king.trailing_zeros();
            if king_square == 6 || king_square == 2 { // g1 ou c1 (castling feito)
                score += 50;
            } else if king_square == 4 { // Ainda em e1
                if board.castling_rights & 0x03 == 0 {
                    score -= 30; // Perdeu castling sem fazer
                } else {
                    score += 10; // Ainda pode fazer
                }
            }
        }
    } else {
        let black_king = board.kings & board.black_pieces;
        if black_king != 0 {
            let king_square = black_king.trailing_zeros();
            if king_square == 62 || king_square == 58 { // g8 ou c8
                score += 50;
            } else if king_square == 60 { // Ainda em e8
                if board.castling_rights & 0x0C == 0 {
                    score -= 30;
                } else {
                    score += 10;
                }
            }
        }
    }

    score
}

/// Avalia atividade do rei no endgame
fn evaluate_king_activity(board: &Board, color: Color) -> i32 {
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let king_bb = board.kings & pieces;

    if king_bb == 0 {
        return 0;
    }

    let king_square = king_bb.trailing_zeros() as usize;
    let rank = king_square / 8;
    let file = king_square % 8;

    // Bônus por rei centralizado no endgame
    let centralization_bonus = match (rank, file) {
        (3, 3) | (3, 4) | (4, 3) | (4, 4) => 40, // Centro
        (2, 2) | (2, 3) | (2, 4) | (2, 5) |
        (3, 2) | (3, 5) | (4, 2) | (4, 5) |
        (5, 2) | (5, 3) | (5, 4) | (5, 5) => 25, // Próximo ao centro
        _ => 0
    };

    // Mobilidade do rei
    let king_mobility = crate::moves::king::get_king_attacks_lookup(king_square as u8)
        .count_ones() as i32 * 5;

    centralization_bonus + king_mobility
}

/// Material Safety Net: previne avaliações excessivamente otimistas
fn apply_material_safety_net(board: &Board, mut score: i32) -> i32 {
    // Calcula diferença material real
    let white_material = calculate_raw_material(board, Color::White);
    let black_material = calculate_raw_material(board, Color::Black);
    let material_diff = white_material - black_material;
    
    // Se a avaliação é muito mais otimista que o material, aplica penalty
    let score_vs_material_diff = score - material_diff;
    
    if score_vs_material_diff.abs() > 200 {
        // Avaliação posicional muito extrema (>200cp vs material)
        let penalty = (score_vs_material_diff.abs() - 200) / 3;
        
        if score_vs_material_diff > 0 {
            score -= penalty; // Reduz avaliação otimista excessiva
        } else {
            score += penalty; // Reduz avaliação pessimista excessiva
        }
    }
    
    score
}

/// Calcula material bruto (sem PST ou bônus)
fn calculate_raw_material(board: &Board, color: Color) -> i32 {
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    
    let pawns = (board.pawns & pieces).count_ones() as i32 * 100;
    let knights = (board.knights & pieces).count_ones() as i32 * 320;
    let bishops = (board.bishops & pieces).count_ones() as i32 * 330;
    let rooks = (board.rooks & pieces).count_ones() as i32 * 500;
    let queens = (board.queens & pieces).count_ones() as i32 * 900;
    
    pawns + knights + bishops + rooks + queens
}