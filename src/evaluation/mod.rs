// Módulo principal de avaliação modular
// Descrição: Coordena todas as subfunções de avaliação com melhor estrutura e performance

pub mod material;
pub mod king_safety;
pub mod threats;
pub mod mobility;
pub mod mobility_cache;
pub mod pawn_structure;
pub mod game_phase;
pub mod cache;
mod utils;
mod endgame_patterns;

use crate::{board::Board, types::Color};
use crate::lazy_eval::get_lazy_eval_manager;
use cache::EvaluationCache;
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};

// Constantes de configuração
const DEFAULT_CACHE_SIZE_MB: usize = 32;
const LAZY_EVAL_THRESHOLD: i32 = 800;
const MAX_EVAL_SCORE: i32 = 10000;

// Flag global para lazy evaluation
static LAZY_EVAL_ENABLED: AtomicBool = AtomicBool::new(true);

// Cache de avaliação thread-local para melhor performance
thread_local! {
    static EVAL_CACHE: RefCell<EvaluationCache> = RefCell::new(EvaluationCache::new(DEFAULT_CACHE_SIZE_MB));
}

/// Estrutura para armazenar componentes da avaliação
#[derive(Debug, Clone, Copy, Default)]
pub struct EvaluationBreakdown {
    pub material: i32,
    pub pawn_structure: i32,
    pub mobility: i32,
    pub king_safety: i32,
    pub threats: i32,
    pub development: i32,
    pub endgame: i32,
    pub tempo: i32,
}

impl EvaluationBreakdown {
    /// Calcula o score total
    pub fn total(&self) -> i32 {
        self.material + self.pawn_structure + self.mobility +
            self.king_safety + self.threats + self.development +
            self.endgame + self.tempo
    }
}

/// Configuração de avaliação
pub struct EvaluationConfig {
    pub lazy_eval: bool,
    pub cache_enabled: bool,
    pub detailed_breakdown: bool,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        EvaluationConfig {
            lazy_eval: true,
            cache_enabled: true,
            detailed_breakdown: false,
        }
    }
}

/// Função principal de avaliação com cache (interface pública)
pub fn evaluate(board: &Board) -> i32 {
    evaluate_with_config(board, &EvaluationConfig::default())
}

/// Função de avaliação com configuração customizada
pub fn evaluate_with_config(board: &Board, config: &EvaluationConfig) -> i32 {
    // Verifica cache se habilitado
    if config.cache_enabled {
        if let Some(cached_score) = probe_cache(board.zobrist_hash) {
            return cached_score;
        }
    }

    // Calcula avaliação
    let score = if config.lazy_eval && LAZY_EVAL_ENABLED.load(Ordering::Relaxed) {
        evaluate_lazy(board)
    } else {
        evaluate_full(board)
    };

    // Armazena no cache se habilitado
    if config.cache_enabled {
        store_in_cache(board.zobrist_hash, score);
    }

    score
}

/// Função de avaliação detalhada que retorna breakdown
pub fn evaluate_detailed(board: &Board) -> (i32, EvaluationBreakdown) {
    let breakdown = evaluate_with_breakdown(board);
    let score = breakdown.total();

    // Ajusta para perspectiva do jogador atual
    let final_score = if board.to_move == Color::White { score } else { -score };
    (final_score, breakdown)
}

/// Avaliação lazy com early cutoff otimizado
fn evaluate_lazy(board: &Board) -> i32 {
    // Step 1: Material rápido
    let material_score = material::evaluate_material_only(board);

    // Early cutoff para diferenças enormes
    if material_score.abs() >= LAZY_EVAL_THRESHOLD {
        return apply_final_adjustments(board, material_score);
    }

    // Step 2: Avaliação incremental baseada na complexidade
    let position_complexity = assess_position_complexity(board);

    match position_complexity {
        PositionComplexity::Simple => {
            // Posições simples: apenas material + PST
            evaluate_simple_position(board)
        },
        PositionComplexity::Moderate => {
            // Posições moderadas: adiciona estrutura de peões e mobilidade
            evaluate_moderate_position(board)
        },
        PositionComplexity::Complex => {
            // Posições complexas: avaliação completa
            evaluate_full(board)
        }
    }
}

/// Avaliação completa sem lazy eval
fn evaluate_full(board: &Board) -> i32 {
    let breakdown = evaluate_with_breakdown(board);
    let mut final_score = breakdown.total();

    // Material Safety Net
    final_score = apply_material_safety_net(board, final_score);

    // Ajusta para perspectiva do jogador atual
    if board.to_move == Color::White {
        final_score
    } else {
        -final_score
    }
}

/// Avaliação com breakdown detalhado
fn evaluate_with_breakdown(board: &Board) -> EvaluationBreakdown {
    let game_phase = game_phase::detect_game_phase(board);
    let phase_info = game_phase::detect_game_phase_advanced(board);

    let mut white_breakdown = evaluate_color_detailed(board, Color::White, &game_phase);
    let mut black_breakdown = evaluate_color_detailed(board, Color::Black, &game_phase);

    // Calcula diferenças para cada componente
    let mut breakdown = EvaluationBreakdown {
        material: white_breakdown.material - black_breakdown.material,
        pawn_structure: white_breakdown.pawn_structure - black_breakdown.pawn_structure,
        mobility: white_breakdown.mobility - black_breakdown.mobility,
        king_safety: white_breakdown.king_safety - black_breakdown.king_safety,
        threats: white_breakdown.threats - black_breakdown.threats,
        development: white_breakdown.development - black_breakdown.development,
        endgame: white_breakdown.endgame - black_breakdown.endgame,
        tempo: evaluate_tempo_advanced(board, &phase_info),
    };

    breakdown
}

/// Tipos de complexidade de posição
#[derive(Debug, Clone, Copy)]
enum PositionComplexity {
    Simple,
    Moderate,
    Complex,
}

/// Avalia a complexidade da posição
fn assess_position_complexity(board: &Board) -> PositionComplexity {
    let total_pieces = (board.white_pieces | board.black_pieces).count_ones();
    let has_queens = board.queens != 0;
    let pawn_moves = count_pawn_breaks(board);
    let is_endgame = total_pieces <= 14;

    // Verifica fatores táticos
    let has_hanging = has_hanging_pieces_quick(board);
    let has_checks = board.is_king_in_check(board.to_move);

    if is_endgame && !has_queens {
        PositionComplexity::Simple
    } else if has_hanging || has_checks || pawn_moves > 2 {
        PositionComplexity::Complex
    } else {
        PositionComplexity::Moderate
    }
}

/// Avaliação para posições simples
fn evaluate_simple_position(board: &Board) -> i32 {
    let game_phase = game_phase::detect_game_phase(board);

    let white_score = material::evaluate_material_and_pst(board, Color::White, &game_phase);
    let black_score = material::evaluate_material_and_pst(board, Color::Black, &game_phase);

    let mut score = white_score - black_score;

    // Adiciona avaliação básica de peões passados em endgames
    if matches!(game_phase, game_phase::GamePhase::Endgame) {
        score += evaluate_passed_pawns_simple(board);
    }

    apply_final_adjustments(board, score)
}

/// Avaliação para posições de complexidade moderada
fn evaluate_moderate_position(board: &Board) -> i32 {
    let game_phase = game_phase::detect_game_phase(board);

    let mut white_score = 0;
    let mut black_score = 0;

    // Material + PST
    white_score += material::evaluate_material_and_pst(board, Color::White, &game_phase);
    black_score += material::evaluate_material_and_pst(board, Color::Black, &game_phase);

    // Estrutura de peões básica
    white_score += pawn_structure::evaluate_pawn_structure(board, Color::White).clamp(-100, 100);
    black_score += pawn_structure::evaluate_pawn_structure(board, Color::Black).clamp(-100, 100);

    // Mobilidade básica
    white_score += mobility::evaluate_mobility(board, Color::White).clamp(-80, 80);
    black_score += mobility::evaluate_mobility(board, Color::Black).clamp(-80, 80);

    let mut score = white_score - black_score;
    score += evaluate_tempo(board);

    apply_final_adjustments(board, score)
}

/// Avalia cor específica com breakdown detalhado
fn evaluate_color_detailed(board: &Board, color: Color, game_phase: &game_phase::GamePhase) -> EvaluationBreakdown {
    let mut breakdown = EvaluationBreakdown::default();

    // Material + PST
    breakdown.material = material::evaluate_material_and_pst(board, color, game_phase);

    // Estrutura de peões
    breakdown.pawn_structure = pawn_structure::evaluate_pawn_structure(board, color).clamp(-120, 120);

    // Mobilidade
    breakdown.mobility = mobility::evaluate_mobility(board, color).clamp(-100, 100);

    // Segurança do rei
    breakdown.king_safety = king_safety::evaluate_king_safety(board, color, game_phase).clamp(-150, 150);

    // Ameaças
    breakdown.threats = threats::evaluate_threats(board, color).clamp(-80, 80);

    // Desenvolvimento/Endgame específicos por fase
    match game_phase {
        game_phase::GamePhase::Opening => {
            breakdown.development = evaluate_development_advanced(board, color);
        },
        game_phase::GamePhase::Endgame => {
            breakdown.endgame = evaluate_endgame_advanced(board, color);
        },
        _ => {
            // Middlegame: balance entre desenvolvimento e preparação para endgame
            breakdown.development = evaluate_development_advanced(board, color) / 2;
            breakdown.endgame = evaluate_king_activity(board, color) / 2;
        }
    }

    breakdown
}

/// Avalia cor específica (versão simplificada para compatibilidade)
fn evaluate_color(board: &Board, color: Color, game_phase: &game_phase::GamePhase) -> i32 {
    let breakdown = evaluate_color_detailed(board, color, game_phase);
    breakdown.total()
}

/// Avalia o tempo/iniciativa com análise avançada
fn evaluate_tempo_advanced(board: &Board, phase_info: &game_phase::GamePhaseInfo) -> i32 {
    let current_moves = board.generate_legal_moves().len() as i32;

    let mut temp_board = *board;
    temp_board.to_move = !temp_board.to_move;
    let opponent_moves = temp_board.generate_legal_moves().len() as i32;

    let mobility_diff = current_moves - opponent_moves;

    // Ajusta importância do tempo baseado na fase
    let phase_multiplier = match phase_info.phase {
        game_phase::GamePhase::Opening => 1.5,
        game_phase::GamePhase::Middlegame => 1.2,
        game_phase::GamePhase::Endgame => 0.8,
    };

    ((mobility_diff as f32 * phase_multiplier * 2.0) as i32).clamp(-50, 50)
}

/// Avalia o tempo (versão básica para compatibilidade)
fn evaluate_tempo(board: &Board) -> i32 {
    let phase_info = game_phase::detect_game_phase_advanced(board);
    evaluate_tempo_advanced(board, &phase_info)
}

/// Avalia desenvolvimento avançado na abertura
fn evaluate_development_advanced(board: &Board, color: Color) -> i32 {
    let mut score = 0;
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let back_rank = if color == Color::White { 0xFF } else { 0xFF00000000000000 };

    // Penaliza peças menores não desenvolvidas
    let knights_undeveloped = (board.knights & pieces & back_rank).count_ones() as i32 * -15;
    let bishops_undeveloped = (board.bishops & pieces & back_rank).count_ones() as i32 * -15;

    score += knights_undeveloped + bishops_undeveloped;

    // Avaliação de castling melhorada
    score += evaluate_castling_advanced(board, color);

    // Bônus por controle central precoce
    score += evaluate_early_center_control(board, color);

    // Penaliza rainhas desenvolvidas muito cedo
    score += evaluate_early_queen_development(board, color);

    score.clamp(-100, 100)
}

/// Avalia desenvolvimento na abertura (versão básica)
fn evaluate_development(board: &Board, color: Color) -> i32 {
    evaluate_development_advanced(board, color)
}

/// Avalia castling com análise avançada
fn evaluate_castling_advanced(board: &Board, color: Color) -> i32 {
    let mut score = 0;
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let king_bb = board.kings & pieces;

    if king_bb == 0 {
        return 0;
    }

    let king_square = king_bb.trailing_zeros();
    let starting_king_square = if color == Color::White { 4 } else { 60 };
    let kingside_castle = if color == Color::White { 6 } else { 62 };
    let queenside_castle = if color == Color::White { 2 } else { 58 };

    if king_square == kingside_castle || king_square == queenside_castle {
        // Castling feito
        score += 50;

        // Bônus extra por castling kingside (geralmente mais seguro)
        if king_square == kingside_castle {
            score += 10;
        }
    } else if king_square == starting_king_square {
        // Ainda não fez castling
        let castle_rights = if color == Color::White {
            board.castling_rights & 0x03
        } else {
            board.castling_rights & 0x0C
        };

        if castle_rights == 0 {
            // Perdeu direitos de roque
            score -= 40;
        } else {
            // Ainda pode fazer roque
            score += 10;

            // Penaliza se muitas peças já foram movidas (roque tardio)
            let development = count_developed_pieces(board, color);
            if development > 4 {
                score -= 20;
            }
        }
    } else {
        // Rei moveu mas não fez roque
        score -= 25;
    }

    score
}

/// Avalia castling (versão básica)
fn evaluate_castling(board: &Board, color: Color) -> i32 {
    evaluate_castling_advanced(board, color)
}

/// Conta peças desenvolvidas
fn count_developed_pieces(board: &Board, color: Color) -> i32 {
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let back_rank = if color == Color::White { 0xFF } else { 0xFF00000000000000 };

    let total_minors = (board.knights | board.bishops) & pieces;
    let developed_minors = total_minors & !back_rank;

    developed_minors.count_ones() as i32
}

/// Avalia controle central precoce
fn evaluate_early_center_control(board: &Board, color: Color) -> i32 {
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let center_squares = 0x00003C3C3C3C0000u64; // Centro expandido

    let our_control = (pieces & center_squares).count_ones() as i32;
    our_control * 3
}

/// Avalia desenvolvimento precoce da rainha
fn evaluate_early_queen_development(board: &Board, color: Color) -> i32 {
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let back_rank = if color == Color::White { 0xFF } else { 0xFF00000000000000 };
    let queen = board.queens & pieces;

    if queen == 0 {
        return 0;
    }

    // Penaliza rainha desenvolvida antes das peças menores
    if (queen & back_rank) == 0 {
        let minors_developed = count_developed_pieces(board, color);
        if minors_developed < 2 {
            return -25; // Rainha saiu muito cedo
        }
    }

    0
}

/// Avalia endgame avançado
fn evaluate_endgame_advanced(board: &Board, color: Color) -> i32 {
    let mut score = 0;

    // Atividade do rei
    score += evaluate_king_activity_advanced(board, color);

    // Padrões avançados de endgame
    let endgame_patterns = endgame_patterns::evaluate_endgame_patterns(board, color);
    score += endgame_patterns.total_score().clamp(-100, 100);

    // Avaliação de finais específicos
    score += evaluate_specific_endgames(board, color);

    score
}

/// Avalia atividade do rei no endgame com melhorias
fn evaluate_king_activity_advanced(board: &Board, color: Color) -> i32 {
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let king_bb = board.kings & pieces;

    if king_bb == 0 {
        return 0;
    }

    let king_square = king_bb.trailing_zeros() as usize;
    let rank = king_square / 8;
    let file = king_square % 8;

    let mut score = 0;

    // Centralização
    let center_distance = ((rank as i32 - 3).abs() + (file as i32 - 4).abs()).max(
        ((rank as i32 - 4).abs() + (file as i32 - 3).abs())
    );
    score += (6 - center_distance) * 5;

    // Mobilidade do rei
    let king_mobility = crate::moves::king::get_king_attacks_lookup(king_square as u8)
        .count_ones() as i32;
    score += king_mobility * 4;

    // Proximidade a peões passados
    score += evaluate_king_support_passed_pawns(board, color, king_square as u8);

    score
}

/// Avalia atividade do rei (versão básica)
fn evaluate_king_activity(board: &Board, color: Color) -> i32 {
    evaluate_king_activity_advanced(board, color)
}

/// Avalia suporte do rei a peões passados
fn evaluate_king_support_passed_pawns(board: &Board, color: Color, king_sq: u8) -> i32 {
    let mut score = 0;
    let our_pawns = board.pawns & if color == Color::White { board.white_pieces } else { board.black_pieces };
    let enemy_pawns = board.pawns & if color == Color::White { board.black_pieces } else { board.white_pieces };

    let mut pawn_bb = our_pawns;
    while pawn_bb != 0 {
        let pawn_sq = pawn_bb.trailing_zeros() as u8;
        pawn_bb &= pawn_bb - 1;

        if utils::is_passed_pawn(pawn_sq, color, our_pawns, enemy_pawns) {
            let distance = utils::king_distance(king_sq, pawn_sq);
            if distance <= 1 {
                score += 20;
            } else if distance <= 3 {
                score += 10;
            }
        }
    }

    score
}

/// Avalia finais específicos (KPK, KRK, etc)
fn evaluate_specific_endgames(board: &Board, color: Color) -> i32 {
    let total_pieces = (board.white_pieces | board.black_pieces).count_ones();

    if total_pieces > 7 {
        return 0; // Não é endgame específico
    }

    // Detecta e avalia finais específicos
    if is_kpk_endgame(board) {
        return evaluate_kpk_endgame(board, color);
    }

    if is_krk_endgame(board) {
        return evaluate_krk_endgame(board, color);
    }

    0
}

/// Verifica se é final KPK
fn is_kpk_endgame(board: &Board) -> bool {
    let total_pieces = (board.white_pieces | board.black_pieces).count_ones();
    let pawns = board.pawns.count_ones();
    let kings = board.kings.count_ones();

    total_pieces == 3 && pawns == 1 && kings == 2
}

/// Avalia final KPK
fn evaluate_kpk_endgame(board: &Board, color: Color) -> i32 {
    // Implementação simplificada - em produção usaria tablebase
    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let has_pawn = (board.pawns & our_pieces) != 0;

    if has_pawn {
        50 // Vantagem para quem tem o peão
    } else {
        -50
    }
}

/// Verifica se é final KRK
fn is_krk_endgame(board: &Board) -> bool {
    let total_pieces = (board.white_pieces | board.black_pieces).count_ones();
    let rooks = board.rooks.count_ones();
    let kings = board.kings.count_ones();

    total_pieces == 3 && rooks == 1 && kings == 2
}

/// Avalia final KRK
fn evaluate_krk_endgame(board: &Board, color: Color) -> i32 {
    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let has_rook = (board.rooks & our_pieces) != 0;

    if has_rook {
        500 // Torre vale aproximadamente isso
    } else {
        -500
    }
}

/// Material Safety Net melhorado
fn apply_material_safety_net(board: &Board, mut score: i32) -> i32 {
    let white_material = calculate_raw_material(board, Color::White);
    let black_material = calculate_raw_material(board, Color::Black);
    let material_diff = white_material - black_material;

    let score_vs_material = score - material_diff;

    // Aplica correção mais suave
    if score_vs_material.abs() > 200 {
        let correction_factor = 1.0 - (score_vs_material.abs() - 200) as f32 / 1000.0;
        let correction_factor = correction_factor.max(0.5); // Nunca reduz mais que 50%

        if score_vs_material > 0 {
            score = material_diff + ((score - material_diff) as f32 * correction_factor) as i32;
        } else {
            score = material_diff - ((material_diff - score) as f32 * correction_factor) as i32;
        }
    }

    score
}

/// Calcula material bruto
fn calculate_raw_material(board: &Board, color: Color) -> i32 {
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };

    let pawns = (board.pawns & pieces).count_ones() as i32 * 100;
    let knights = (board.knights & pieces).count_ones() as i32 * 320;
    let bishops = (board.bishops & pieces).count_ones() as i32 * 330;
    let rooks = (board.rooks & pieces).count_ones() as i32 * 500;
    let queens = (board.queens & pieces).count_ones() as i32 * 900;

    pawns + knights + bishops + rooks + queens
}

/// Aplica ajustes finais à avaliação
fn apply_final_adjustments(board: &Board, mut score: i32) -> i32 {
    // Limite máximo
    score = score.clamp(-MAX_EVAL_SCORE, MAX_EVAL_SCORE);

    // Ajusta para perspectiva do jogador
    if board.to_move == Color::White {
        score
    } else {
        -score
    }
}

/// Conta possíveis pawn breaks
fn count_pawn_breaks(board: &Board) -> i32 {
    let white_pawns = board.pawns & board.white_pieces;
    let black_pawns = board.pawns & board.black_pieces;

    let white_attacks = utils::compute_pawn_attacks(white_pawns, Color::White);
    let black_attacks = utils::compute_pawn_attacks(black_pawns, Color::Black);

    let potential_captures = (white_attacks & black_pawns).count_ones() +
        (black_attacks & white_pawns).count_ones();

    potential_captures as i32
}

/// Verifica rapidamente se há peças penduradas
fn has_hanging_pieces_quick(board: &Board) -> bool {
    for color in [Color::White, Color::Black] {
        let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
        let valuable = (board.knights | board.bishops | board.rooks | board.queens) & pieces;

        if valuable.count_ones() > 0 {
            // Verificação simplificada - apenas conta ataques
            let mut bb = valuable;
            while bb != 0 {
                let sq = bb.trailing_zeros() as u8;
                bb &= bb - 1;

                if board.is_square_attacked_by(sq, !color) {
                    return true;
                }
            }
        }
    }
    false
}

/// Avalia peões passados de forma simplificada
fn evaluate_passed_pawns_simple(board: &Board) -> i32 {
    let mut score = 0;

    for color in [Color::White, Color::Black] {
        let our_pawns = board.pawns & if color == Color::White { board.white_pieces } else { board.black_pieces };
        let enemy_pawns = board.pawns & if color == Color::White { board.black_pieces } else { board.white_pieces };

        let mut pawn_bb = our_pawns;
        while pawn_bb != 0 {
            let pawn_sq = pawn_bb.trailing_zeros() as u8;
            pawn_bb &= pawn_bb - 1;

            if utils::is_passed_pawn(pawn_sq, color, our_pawns, enemy_pawns) {
                let rank = pawn_sq / 8;
                let bonus = if color == Color::White {
                    (rank * rank) as i32 * 5
                } else {
                    ((7 - rank) * (7 - rank)) as i32 * 5
                };

                if color == Color::White {
                    score += bonus;
                } else {
                    score -= bonus;
                }
            }
        }
    }

    score
}

// === Funções públicas de controle ===

/// Habilita ou desabilita lazy evaluation globalmente
pub fn set_lazy_eval_enabled(enabled: bool) {
    LAZY_EVAL_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Função para limpar o cache de avaliação
pub fn clear_eval_cache() {
    EVAL_CACHE.with(|cache| {
        if let Ok(mut cache_ref) = cache.try_borrow_mut() {
            cache_ref.clear();
        }
    });
}

/// Função para obter estatísticas do cache
pub fn get_eval_cache_stats() -> (u64, u64, f64, usize) {
    EVAL_CACHE.with(|cache| {
        if let Ok(cache_ref) = cache.try_borrow() {
            cache_ref.get_stats()
        } else {
            (0, 0, 0.0, 0)
        }
    })
}

/// Redimensiona o cache de avaliação
pub fn resize_eval_cache(size_mb: usize) {
    EVAL_CACHE.with(|cache| {
        if let Ok(mut cache_ref) = cache.try_borrow_mut() {
            *cache_ref = EvaluationCache::new(size_mb);
        }
    });
}

// === Funções auxiliares privadas ===

/// Busca no cache
fn probe_cache(hash: u64) -> Option<i32> {
    EVAL_CACHE.with(|cache| {
        if let Ok(cache_ref) = cache.try_borrow() {
            cache_ref.probe(hash)
        } else {
            None
        }
    })
}

/// Armazena no cache
fn store_in_cache(hash: u64, score: i32) {
    EVAL_CACHE.with(|cache| {
        if let Ok(mut cache_ref) = cache.try_borrow_mut() {
            cache_ref.store(hash, score, 0);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluation_symmetry() {
        // Testa se a avaliação é simétrica
        // TODO: Implementar testes
    }

    #[test]
    fn test_cache_functionality() {
        // Testa funcionalidade do cache
        // TODO: Implementar testes
    }
}