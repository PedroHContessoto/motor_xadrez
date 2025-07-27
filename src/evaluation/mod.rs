// Módulo principal de avaliação modular
// Descrição: Coordena todas as subfunções de avaliação

pub mod material;
pub mod king_safety;
pub mod threats;
pub mod mobility;
pub mod pawn_structure;
pub mod game_phase;
mod utils;
pub mod endgame_patterns;
pub mod meta_evaluation;
pub mod draw_win_management;
pub mod victory_conversion;
pub mod cache;

use crate::{board::Board, types::Color, profile, count};
use cache::{get_cached_evaluation_with_depth, store_evaluation_with_depth};


/// Função principal de avaliação (interface pública)
pub fn evaluate(board: &Board) -> i32 {
    evaluate_with_depth(board, 0) // Profundidade 0 = avaliação completa (compatibilidade)
}

/// Avaliação com lazy loading baseado na profundidade de busca
pub fn evaluate_with_depth(board: &Board, depth: u8) -> i32 {
    count!("evaluations");
    
    // Verifica cache primeiro com profundidade
    if let Some(cached_eval) = profile!("cache_lookup", {
        get_cached_evaluation_with_depth(board, depth)
    }) {
        count!("cache_hits");
        return cached_eval;
    }
    
    count!("cache_misses");
    
    // OTIMIZAÇÃO: Avaliação rápida para nós folha ou profundidade baixa
    if depth >= 8 { // Nós próximos das folhas (profundidade alta na busca)
        return evaluate_fast(board);
    }
    
    let game_phase = profile!("game_phase_detection", {
        game_phase::detect_game_phase(board)
    });

    let white_score = profile!("evaluate_white", {
        evaluate_color_with_depth(board, Color::White, &game_phase, depth)
    });
    let black_score = profile!("evaluate_black", {
        evaluate_color_with_depth(board, Color::Black, &game_phase, depth)
    });

    let mut final_score = white_score - black_score;

    // Componentes caros apenas em nós importantes (profundidade baixa)
    if depth <= 4 {
        // Adiciona tempo/iniciativa (só quando realmente relevante)
        if should_evaluate_tempo(board, &game_phase, final_score) {
            final_score += profile!("tempo_evaluation", {
                evaluate_tempo_fast(board) // Versão rápida sem legal moves
            });
        }

        // Material Safety Net: penaliza avaliações excessivamente otimistas
        final_score = profile!("material_safety_net", {
            apply_material_safety_net(board, final_score)
        });
    }

    // Retorna relativo ao jogador atual
    let final_eval = if board.to_move == Color::White {
        final_score
    } else {
        -final_score
    };
    
    // Armazena no cache com profundidade
    profile!("cache_store", {
        store_evaluation_with_depth(board, final_eval, depth)
    });
    
    final_eval
}

/// Avaliação rápida para nós folha (apenas material + PST)
fn evaluate_fast(board: &Board) -> i32 {
    let game_phase = game_phase::detect_game_phase(board);
    
    let white_material = material::evaluate_material_and_pst(board, Color::White, &game_phase);
    let black_material = material::evaluate_material_and_pst(board, Color::Black, &game_phase);
    
    let score = white_material - black_material;
    
    if board.to_move == Color::White {
        score
    } else {
        -score
    }
}

/// Decide se deve calcular tempo/iniciativa (costly: 80k calls!)
fn should_evaluate_tempo(board: &Board, game_phase: &game_phase::GamePhase, current_score: i32) -> bool {
    // 1. Só em posições equilibradas (onde tempo realmente importa)
    if current_score.abs() > 200 {
        return false; // Em vantagens grandes, tempo é irrelevante
    }
    
    // 2. Principalmente no middlegame (onde iniciativa é crucial)
    if matches!(game_phase, 
        game_phase::GamePhase::EarlyMiddlegame | 
        game_phase::GamePhase::Middlegame
    ) {
        return true;
    }
    
    // 3. Na abertura, só se houver muitas peças desenvolvidas
    if matches!(game_phase, game_phase::GamePhase::Opening) {
        let total_pieces = (board.white_pieces | board.black_pieces).count_ones();
        return total_pieces <= 28; // Posições desenvolvidas
    }
    
    false // Endgames: tempo menos relevante
}

/// Decide se deve fazer mobility evaluation completa (160k calls!)
fn should_perform_mobility_evaluation(board: &Board, game_phase: &game_phase::GamePhase) -> bool {
    // 1. Sempre no middlegame (mobilidade crucial)
    if matches!(game_phase, 
        game_phase::GamePhase::Middlegame | 
        game_phase::GamePhase::LateMiddlegame
    ) {
        return true;
    }
    
    // 2. Em posições com peças menores ativas
    let minor_pieces = (board.knights | board.bishops).count_ones();
    if minor_pieces >= 3 {
        return true;
    }
    
    // 3. Em endgames com peças ativas
    if matches!(game_phase, game_phase::GamePhase::EarlyEndgame) {
        let active_pieces = (board.queens | board.rooks).count_ones();
        return active_pieces >= 2;
    }
    
    false // Abertura inicial e endgames simples: usa basic_mobility
}

/// Mobilidade básica e rápida (sem análise detalhada)
fn evaluate_basic_mobility(board: &Board, color: Color) -> i32 {
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    
    // Conta desenvolvimento simples
    let back_rank = if color == Color::White { 0xFF } else { 0xFF00000000000000 };
    let developed = (pieces & !back_rank & !(board.pawns | board.kings)).count_ones() as i32;
    
    // Bônus simples por desenvolvimento
    developed * 4 // Aproximação rápida da mobilidade
}


/// Decide se deve fazer threats evaluation baseado na posição tática
fn should_perform_threats_evaluation(board: &Board, game_phase: &game_phase::GamePhase) -> bool {
    // MUITO mais seletivo: só quando realmente há potencial tático
    
    // 1. Nunca na abertura (desenvolvimento mais importante)
    if matches!(game_phase, game_phase::GamePhase::Opening) {
        return false;
    }
    
    // 2. Só no middlegame se há pelo menos uma rainha no tabuleiro
    if matches!(game_phase, 
        game_phase::GamePhase::EarlyMiddlegame | 
        game_phase::GamePhase::Middlegame | 
        game_phase::GamePhase::LateMiddlegame
    ) {
        return board.queens.count_ones() >= 1;
    }
    
    // 3. No endgame, só se há peças suficientes para táticas complexas
    let total_pieces = (board.white_pieces | board.black_pieces).count_ones();
    if total_pieces >= 16 && (board.queens | board.rooks).count_ones() >= 2 {
        return true;
    }
    
    // 4. Posições com desequilíbrio material significativo (onde ameaças importam mais)
    let white_major = (board.queens | board.rooks) & board.white_pieces;
    let black_major = (board.queens | board.rooks) & board.black_pieces;
    let major_imbalance = (white_major.count_ones() as i32 - black_major.count_ones() as i32).abs();
    
    if major_imbalance >= 2 {
        return true; // Desequilíbrio tático significativo
    }
    
    false // Na maioria dos casos, pula threats completamente
}

/// Decide se deve fazer meta-evaluation baseado na posição e contexto
fn should_perform_meta_evaluation(board: &Board, current_score: i32, game_phase: &game_phase::GamePhase) -> bool {
    // 1. Só em middlegame/endgame (não na abertura onde é menos relevante)
    if matches!(game_phase, game_phase::GamePhase::Opening) {
        return false;
    }
    
    // 2. Em posições equilibradas (onde meta-evaluation tem mais impacto)
    if current_score.abs() <= 150 {
        return true;
    }
    
    // 3. Em posições com desequilíbrio material (táticas complexas)
    let white_material = calculate_raw_material(board, Color::White);
    let black_material = calculate_raw_material(board, Color::Black);
    let material_imbalance = (white_material - black_material).abs();
    
    // Se há desequilíbrio significativo mas não é uma vantagem esmagadora
    if material_imbalance >= 100 && current_score.abs() <= 300 {
        return true;
    }
    
    // 4. Poucas peças no tabuleiro (endgames complexos)
    let total_pieces = (board.white_pieces | board.black_pieces).count_ones();
    if total_pieces <= 16 {
        return true;
    }
    
    false // Nos outros casos, pula meta-evaluation
}

/// Avalia cor específica
fn evaluate_color(board: &Board, color: Color, game_phase: &game_phase::GamePhase) -> i32 {
    evaluate_color_with_depth(board, color, game_phase, 0)
}

/// Avalia cor específica com lazy evaluation baseado na profundidade
fn evaluate_color_with_depth(board: &Board, color: Color, game_phase: &game_phase::GamePhase, depth: u8) -> i32 {
    let mut score = 0;

    // SEMPRE: Material + PST (base da avaliação)
    score += profile!("material_and_pst", {
        material::evaluate_material_and_pst(board, color, game_phase)
    });

    // PROFUNDIDADE <= 6: Componentes estruturais importantes
    if depth <= 6 {
        // Estrutura de peões (incluindo passados) - CAP: ±120
        score += profile!("pawn_structure", {
            pawn_structure::evaluate_pawn_structure(board, color).clamp(-120, 120)
        });

        // Segurança do rei (crítica) - CAP: ±150
        score += profile!("king_safety", {
            king_safety::evaluate_king_safety(board, color, game_phase).clamp(-150, 150)
        });
    }

    // PROFUNDIDADE <= 4: Componentes táticos (caros)
    if depth <= 4 {
        // Mobilidade segura - CAP: ±100 (otimizada: 160k calls!)
        if should_perform_mobility_evaluation(board, game_phase) {
            score += profile!("mobility", {
                mobility::evaluate_mobility(board, color).clamp(-100, 100)
            });
        } else {
            // Mobilidade simplificada (desenvolvimento básico)
            score += evaluate_basic_mobility(board, color);
        }

        // Avaliação de ameaças (peças penduradas, ataques) - CAP: ±80
        if should_perform_threats_evaluation(board, game_phase) {
            score += profile!("threats", {
                threats::evaluate_threats(board, color).clamp(-80, 80)
            });
        }
    }

    // PROFUNDIDADE <= 2: Meta-evaluation (muito cara)
    if depth <= 2 {
        // META-EVALUATION: Análise avançada de vulnerabilidades - CAP: ±60
        if should_perform_meta_evaluation(board, score, game_phase) {
            score += profile!("meta_evaluation", {
                meta_evaluation::meta_evaluate_position(board, color).clamp(-60, 60)
            });
        }
    }

    // PROFUNDIDADE == 0: Avaliações específicas por fase (root e PV nodes)
    if depth == 0 {
        match game_phase {
            game_phase::GamePhase::Opening => {
                score += evaluate_development(board, color);
            },
            game_phase::GamePhase::EarlyMiddlegame | game_phase::GamePhase::Middlegame | game_phase::GamePhase::LateMiddlegame => {
                // Middlegame focus on tactics and piece coordination
            },
            game_phase::GamePhase::EarlyEndgame | game_phase::GamePhase::Endgame | game_phase::GamePhase::PureEndgame => {
                score += evaluate_king_activity(board, color);
            },
        }
    }

    score
}

/// Avalia o tempo (iniciativa) usando mobilidade aproximada (OTIMIZADA - sem legal moves)
fn evaluate_tempo_fast(board: &Board) -> i32 {
    use std::collections::HashMap;
    use std::sync::Mutex;
    
    lazy_static::lazy_static! {
        static ref TEMPO_CACHE: Mutex<HashMap<u64, i32>> = 
            Mutex::new(HashMap::with_capacity(1000));
    }
    
    // Verifica cache primeiro
    if let Ok(cache) = TEMPO_CACHE.try_lock() {
        if let Some(&cached_tempo) = (*cache).get(&board.zobrist_hash) {
            return cached_tempo;
        }
    }
    
    // OTIMIZAÇÃO: Usa aproximação de mobilidade em vez de legal moves (muito mais rápido!)
    let current_mobility = approximate_mobility(board, board.to_move);
    let opponent_mobility = approximate_mobility(board, !board.to_move);

    let tempo = ((current_mobility - opponent_mobility) / 2).clamp(-50, 50);
    
    // Armazena no cache
    if let Ok(mut cache) = TEMPO_CACHE.try_lock() {
        if (*cache).len() >= 1000 {
            (*cache).clear(); // LRU simples: limpa quando cheio
        }
        (*cache).insert(board.zobrist_hash, tempo);
    }
    
    tempo
}

/// Aproxima mobilidade sem gerar movimentos legais (muito mais rápido)
fn approximate_mobility(board: &Board, color: Color) -> i32 {
    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let all_pieces = board.white_pieces | board.black_pieces;
    let mut mobility = 0;
    
    // Cavalos: aproximação simples
    let knights = board.knights & our_pieces;
    let mut knight_bb = knights;
    while knight_bb != 0 {
        let sq = knight_bb.trailing_zeros() as u8;
        knight_bb &= knight_bb - 1;
        let attacks = crate::moves::knight::get_knight_attacks_lookup(sq);
        mobility += (attacks & !our_pieces).count_ones() as i32;
    }
    
    // Bispos: aproximação diagonal  
    let bishops = board.bishops & our_pieces;
    let mut bishop_bb = bishops;
    while bishop_bb != 0 {
        let sq = bishop_bb.trailing_zeros() as u8;
        bishop_bb &= bishop_bb - 1;
        let attacks = crate::moves::sliding::get_bishop_attacks(sq, all_pieces);
        mobility += (attacks & !our_pieces).count_ones() as i32;
    }
    
    // Torres: aproximação horizontal/vertical
    let rooks = board.rooks & our_pieces;
    let mut rook_bb = rooks;
    while rook_bb != 0 {
        let sq = rook_bb.trailing_zeros() as u8;
        rook_bb &= rook_bb - 1;
        let attacks = crate::moves::sliding::get_rook_attacks(sq, all_pieces);
        mobility += (attacks & !our_pieces).count_ones() as i32;
    }
    
    // Rainhas: bispo + torre
    let queens = board.queens & our_pieces;
    let mut queen_bb = queens;
    while queen_bb != 0 {
        let sq = queen_bb.trailing_zeros() as u8;
        queen_bb &= queen_bb - 1;
        let bishop_attacks = crate::moves::sliding::get_bishop_attacks(sq, all_pieces);
        let rook_attacks = crate::moves::sliding::get_rook_attacks(sq, all_pieces);
        mobility += ((bishop_attacks | rook_attacks) & !our_pieces).count_ones() as i32;
    }
    
    // Peões: aproximação simples (movimentos à frente + capturas)
    let pawns = board.pawns & our_pieces;
    mobility += pawns.count_ones() as i32 * 2; // Aproximação: ~2 movimentos por peão
    
    mobility
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

/// Avalia castling com valores aumentados
fn evaluate_castling(board: &Board, color: Color) -> i32 {
    let mut score = 0;

    if color == Color::White {
        let white_king = board.kings & board.white_pieces;
        if white_king != 0 {
            let king_square = white_king.trailing_zeros();
            if king_square == 6 || king_square == 2 { // g1 ou c1 (castling feito)
                score += 80; // Aumentado de 50 -> 80
            } else if king_square == 4 { // Ainda em e1
                if board.castling_rights & 0x03 == 0 {
                    score -= 60; // Aumentado de -30 -> -60 (penalidade por perder roque)
                } else {
                    score += 25; // Aumentado de 10 -> 25 (incentivo para fazer roque)
                }
            }
        }
    } else {
        let black_king = board.kings & board.black_pieces;
        if black_king != 0 {
            let king_square = black_king.trailing_zeros();
            if king_square == 62 || king_square == 58 { // g8 ou c8
                score += 80; // Aumentado de 50 -> 80
            } else if king_square == 60 { // Ainda em e8
                if board.castling_rights & 0x0C == 0 {
                    score -= 60; // Aumentado de -30 -> -60
                } else {
                    score += 25; // Aumentado de 10 -> 25
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
    
    // Mais conservador com peças valiosas: reduzido de 200 -> 150
    if score_vs_material_diff.abs() > 150 {
        // Avaliação posicional muito extrema (>150cp vs material)
        let penalty = (score_vs_material_diff.abs() - 150) / 2; // Penalty mais forte: /2 em vez de /3
        
        if score_vs_material_diff > 0 {
            score -= penalty; // Reduz avaliação otimista excessiva
        } else {
            score += penalty; // Reduz avaliação pessimista excessiva
        }
    }
    
    // Safety check adicional: se diferença material é extrema (>300cp), limita score posicional
    if material_diff.abs() > 300 {
        let material_dominance = material_diff.signum();
        let max_positional_compensation = 200; // Máximo que posição pode compensar material
        
        if material_dominance > 0 && score < material_diff - max_positional_compensation {
            score = material_diff - max_positional_compensation;
        } else if material_dominance < 0 && score > material_diff + max_positional_compensation {
            score = material_diff + max_positional_compensation;
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