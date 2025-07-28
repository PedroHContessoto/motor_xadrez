// Sistema avançado de ordenação de movimentos com múltiplas heurísticas
use crate::{board::Board, transposition::TranspositionTable, types::{Move, PieceKind, Color}};
use super::{SearchContext, see::see};

// === CONSTANTES PARA ORDENAÇÃO AVANÇADA ===
const PIECE_VALUES: [i32; 6] = [100, 320, 330, 500, 900, 20000];

// Pesos para diferentes tipos de movimentos
const SCORE_TT_MOVE: i32 = 10_000_000;          // TT move tem prioridade absoluta
const SCORE_GOOD_CAPTURE_BASE: i32 = 8_000_000; // Capturas boas
const SCORE_EQUAL_CAPTURE: i32 = 6_000_000;     // Capturas iguais
const SCORE_KILLER_1: i32 = 4_000_000;          // Primeiro killer
const SCORE_KILLER_2: i32 = 3_000_000;          // Segundo killer
const SCORE_COUNTER_MOVE: i32 = 2_000_000;      // Counter move
const SCORE_CASTLING: i32 = 5_000_000;          // Castling (aumentado significativamente)
const SCORE_PROMOTION_BASE: i32 = 1_000_000;    // Promoções
const SCORE_HISTORY_BASE: i32 = 500_000;        // Histórico de movimentos
const SCORE_BAD_CAPTURE_BASE: i32 = 100_000;    // Capturas ruins (mas ainda considera)

/// Sistema avançado de ordenação com múltiplas heurísticas
pub fn order_moves(
    board: &Board,
    moves: Vec<Move>,
    tt: &TranspositionTable,
    context: &SearchContext,
    depth: u8
) -> Vec<Move> {
    // === 1. IDENTIFICAÇÃO DE MOVIMENTOS ESPECIAIS ===
    
    // TT Move (prioridade absoluta)
    let tt_move = if let Some(entry) = tt.probe(board.zobrist_hash) {
        entry.best_move.filter(|mv| board.is_legal_move(*mv))
    } else {
        None
    };
    
    // Killer moves para esta profundidade
    let killer1 = context.get_killer_move(depth, 0);
    let killer2 = context.get_killer_move(depth, 1);
    
    // Counter move (resposta ao último movimento)
    let counter_move = context.get_last_move()
        .and_then(|last_mv| context.get_counter_move(last_mv));
    
    // === 2. SCORING AVANÇADO DE MOVIMENTOS ===
    
    let mut scored_moves = moves.into_iter().map(|mv| {
        let score = calculate_move_score(
            mv, 
            board, 
            context, 
            tt_move, 
            killer1, 
            killer2, 
            counter_move,
            depth
        );
        (mv, score)
    }).collect::<Vec<_>>();
    
    // === 3. ORDENAÇÃO OTIMIZADA ===
    
    // Ordenação estável por score (maior primeiro) com proteção contra overflow
    scored_moves.sort_by(|a, b| {
        // Protege contra valores inválidos que podem causar crash
        let score_a = a.1.max(i32::MIN + 1).min(i32::MAX - 1);
        let score_b = b.1.max(i32::MIN + 1).min(i32::MAX - 1);
        score_b.cmp(&score_a)
    });
    
    // Retorna apenas os movimentos ordenados
    scored_moves.into_iter().map(|(mv, _)| mv).collect()
}

/// Calcula score abrangente para um movimento
fn calculate_move_score(
    mv: Move,
    board: &Board,
    context: &SearchContext,
    tt_move: Option<Move>,
    killer1: Option<Move>,
    killer2: Option<Move>,
    counter_move: Option<Move>,
    depth: u8
) -> i32 {
    // === 1. TT MOVE (PRIORIDADE ABSOLUTA) ===
    if Some(mv) == tt_move {
        return SCORE_TT_MOVE;
    }
    
    // === 2. CAPTURAS COM ANÁLISE SEE AVANÇADA ===
    if board.is_capture(mv) {
        return score_capture_advanced(mv, board);
    }
    
    // === 3. PROMOÇÕES ===
    if let Some(promotion) = mv.promotion {
        return score_promotion(promotion);
    }
    
    // === 4. MOVIMENTOS ESPECIAIS NÃO-CAPTURA ===
    
    // Killer moves
    if Some(mv) == killer1 {
        return SCORE_KILLER_1;
    }
    if Some(mv) == killer2 {
        return SCORE_KILLER_2;
    }
    
    // Counter move
    if Some(mv) == counter_move {
        return SCORE_COUNTER_MOVE;
    }
    
    // Castling
    if mv.is_castling {
        return SCORE_CASTLING;
    }
    
    // === 5. HISTÓRICO E HEURÍSTICAS AVANÇADAS ===
    
    let mut quiet_score: i32 = 0;
    
    // Histórico de movimentos
    if let Some(piece) = board.get_piece_on_square(mv.from) {
        let history = context.get_history_score(mv, piece);
        let history_contribution = history.saturating_mul(SCORE_HISTORY_BASE).checked_div(10000).unwrap_or(0);
        quiet_score = quiet_score.saturating_add(history_contribution);
    }
    
    // Heurísticas posicionais
    quiet_score = quiet_score.saturating_add(score_positional_heuristics(mv, board, depth));
    
    quiet_score
}

/// Score avançado para capturas com análise SEE
fn score_capture_advanced(mv: Move, board: &Board) -> i32 {
    let see_value = see(board, mv);
    let captured_value = get_captured_piece_value(board, mv);
    let attacker_value = get_attacking_piece_value(board, mv);
    
    if see_value > 0 {
        // Captura boa - usa MVV-LVA otimizado com proteção contra overflow
        SCORE_GOOD_CAPTURE_BASE.saturating_add(captured_value.saturating_mul(10)).saturating_sub(attacker_value)
    } else if see_value == 0 {
        // Captura igual - ainda prioritária
        SCORE_EQUAL_CAPTURE.saturating_add(captured_value)
    } else {
        // Captura ruim mas ainda considerável
        SCORE_BAD_CAPTURE_BASE.saturating_add(see_value)
    }
}

/// Score para promoções
fn score_promotion(promotion: PieceKind) -> i32 {
    match promotion {
        PieceKind::Queen => SCORE_PROMOTION_BASE.saturating_add(900),
        PieceKind::Rook => SCORE_PROMOTION_BASE.saturating_add(500),
        PieceKind::Bishop => SCORE_PROMOTION_BASE.saturating_add(330),
        PieceKind::Knight => SCORE_PROMOTION_BASE.saturating_add(320),
        _ => SCORE_PROMOTION_BASE,
    }
}

/// Heurísticas posicionais avançadas e modernas
fn score_positional_heuristics(mv: Move, board: &Board, depth: u8) -> i32 {
    let mut score = 0;
    
    // === DESENVOLVIMENTO E ABERTURA ===
    if is_development_move(board, mv) {
        let development_bonus = if depth <= 4 { 30000 } else { 15000 };
        score += development_bonus;
        
        // Bônus extra para desenvolvimento para centro
        if controls_center(mv) {
            score += 10000;
        }
    }
    
    // === CONTROLE DE CENTRO REFINADO ===
    let center_score = evaluate_center_control(mv, board);
    score += center_score;
    
    // === MOVIMENTOS DEFENSIVOS APRIMORADOS ===
    if is_defensive_move(board, mv) {
        let defensive_urgency = evaluate_defensive_urgency(board, mv);
        score = score.saturating_add(8000).saturating_add(defensive_urgency);
    }
    
    // === CHECKS E AMEAÇAS ===
    if gives_check_heuristic(board, mv) {
        score += 12000; // Aumentado de 8000
        
        // Discovered checks são especialmente perigosos
        if is_discovered_check(board, mv) {
            score += 8000; // Aumentado de 5000
        }
        
        // Checks que levam a mate threats
        if could_lead_to_mate_threat(board, mv) {
            score += 15000;
        }
    }
    
    // === AMEAÇAS TÁTICAS ===
    if creates_tactical_threat(board, mv) {
        score += 6000;
    }
    
    // === MOBILIDADE DE PEÇAS ===
    let mobility_bonus = evaluate_piece_mobility_gain(board, mv);
    score += mobility_bonus;
    
    // === ESTRUTURA DE PEÕES ===
    if improves_pawn_structure(board, mv) {
        score += 4000;
    }
    
    // === PENALIZAÇÕES ===
    if weakens_king_safety(board, mv) {
        let safety_penalty = evaluate_king_safety_impact(board, mv);
        score -= safety_penalty;
    }
    
    // Penaliza movimentos que isolam peças próprias
    if isolates_own_piece(board, mv) {
        score -= 8000;
    }
    
    // === AJUSTES POR PROFUNDIDADE E FASE DO JOGO ===
    let depth_factor = if depth <= 3 {
        0.6 // Reduz importância posicional em busca tática
    } else if depth <= 6 {
        0.8
    } else {
        1.0 // Importância total em busca profunda
    };
    
    // Ajuste por fase do jogo
    let total_pieces = (board.white_pieces | board.black_pieces).count_ones();
    let game_phase_factor = if total_pieces <= 12 {
        1.2 // Endgame - posição é mais importante
    } else if total_pieces >= 28 {
        0.7 // Abertura - tática é mais importante
    } else {
        1.0 // Meio-jogo
    };
    
    (score as f32 * depth_factor * game_phase_factor) as i32
}


/// Verifica se é counter-move (refuta último movimento)
fn is_counter_move(mv: Move, context: &SearchContext) -> bool {
    if let Some(last_move) = context.get_last_move() {
        // Verifica se é um counter-move registrado
        if let Some(counter) = context.get_counter_move(last_move) {
            return counter == mv;
        }
    }
    false
}

/// Heurística rápida para detectar checks (sem fazer o movimento)
fn gives_check_heuristic(board: &Board, mv: Move) -> bool {
    // Usa a função da quiescence (pode ser otimizada)
    super::quiescence::gives_check_fast(board, mv)
}

/// Detecta movimentos de desenvolvimento na abertura
fn is_development_move(board: &Board, mv: Move) -> bool {
    // Só considera se ainda na abertura (muitas peças no back rank)
    let back_ranks = 0xFF | 0xFF00000000000000;
    let pieces_on_back = (board.white_pieces | board.black_pieces) & back_ranks;

    if pieces_on_back.count_ones() < 10 {
        return false; // Já não é abertura
    }

    // Verifica se move uma peça menor do back rank
    let from_bb = 1u64 << mv.from;
    let is_from_back_rank = (back_ranks & from_bb) != 0;

    if !is_from_back_rank {
        return false;
    }

    // Só cavalos e bispos são "desenvolvimento"
    let piece_kind = board.get_piece_on_square(mv.from);
    matches!(piece_kind, Some(PieceKind::Knight) | Some(PieceKind::Bishop))
}

/// Verifica se move controla casas centrais
fn controls_center(mv: Move) -> bool {
    let center_squares = [27, 28, 35, 36]; // d4, e4, d5, e5
    let extended_center = [
        19, 20, 21, 22, // c3-f3
        27, 28, 29, 30, // c4-f4
        35, 36, 37, 38, // c5-f5
        43, 44, 45, 46, // c6-f6
    ];

    // Casa de destino no centro
    if center_squares.contains(&(mv.to as usize)) {
        return true;
    }

    // Casa de destino no centro estendido
    if extended_center.contains(&(mv.to as usize)) {
        return true;
    }

    false
}

/// Detecta movimentos defensivos - defendem peças valiosas atacadas
fn is_defensive_move(board: &Board, mv: Move) -> bool {
    let our_color = board.to_move;
    let enemy_color = !our_color;
    let our_pieces = if our_color == Color::White { board.white_pieces } else { board.black_pieces };

    // Analisa peças valiosas próprias atacadas pelo inimigo
    let our_valuables = (board.knights | board.bishops | board.rooks | board.queens) & our_pieces;
    let mut bb = our_valuables;

    while bb != 0 {
        let valuable_sq = bb.trailing_zeros() as u8;
        bb &= bb - 1;

        // Se peça valiosa está atacada pelo inimigo
        if board.is_square_attacked_by(valuable_sq, enemy_color) {
            // Verifica se o movimento defende esta peça

            // 1. Movimento bloqueia o ataque (interposes)
            if mv.to != valuable_sq && blocks_attack_to_square(board, mv, valuable_sq, enemy_color) {
                return true;
            }

            // 2. Movimento move a peça atacada para segurança
            if mv.from == valuable_sq && !would_be_attacked_after_move(board, mv, enemy_color) {
                return true;
            }

            // 3. Movimento defende a peça (adiciona defensor)
            if mv.to != valuable_sq && defends_square_after_move(board, mv, valuable_sq) {
                return true;
            }

            // 4. Movimento captura o atacante
            if board.is_capture(mv) && was_attacking_valuable_piece(board, mv.to, valuable_sq, enemy_color) {
                return true;
            }
        }
    }

    false
}

/// Verifica se movimento bloqueia ataque a uma casa específica
fn blocks_attack_to_square(board: &Board, mv: Move, target_sq: u8, enemy_color: Color) -> bool {
    // Simula o movimento e verifica se ainda há ataque à casa
    let mut temp_board = *board;
    let _undo_info = temp_board.make_move_fast(mv);

    // Se após o movimento a casa não está mais atacada, bloqueou o ataque
    !temp_board.is_square_attacked_by(target_sq, enemy_color)
}

/// Verifica se após o movimento a peça não estará atacada
fn would_be_attacked_after_move(board: &Board, mv: Move, enemy_color: Color) -> bool {
    let mut temp_board = *board;
    let _undo_info = temp_board.make_move_fast(mv);

    temp_board.is_square_attacked_by(mv.to, enemy_color)
}

/// Verifica se após o movimento a casa estará defendida
fn defends_square_after_move(board: &Board, mv: Move, defended_sq: u8) -> bool {
    let mut temp_board = *board;
    let _undo_info = temp_board.make_move_fast(mv);

    // Verifica se a peça movida agora defende a casa
    temp_board.is_square_attacked_by(defended_sq, temp_board.to_move)
}

/// Verifica se a peça capturada estava atacando uma peça valiosa
fn was_attacking_valuable_piece(board: &Board, captured_sq: u8, valuable_sq: u8, enemy_color: Color) -> bool {
    // Verifica se removendo a peça da casa capturada, ela para de atacar a peça valiosa
    let captured_piece_bb = 1u64 << captured_sq;
    let enemy_pieces = if enemy_color == Color::White { board.white_pieces } else { board.black_pieces };

    if (captured_piece_bb & enemy_pieces) == 0 {
        return false; // Não é peça inimiga
    }

    // Simula remoção da peça e verifica se o ataque para
    let mut temp_board = *board;

    // Remove a peça capturada de todos os bitboards relevantes
    temp_board.white_pieces &= !captured_piece_bb;
    temp_board.black_pieces &= !captured_piece_bb;
    temp_board.pawns &= !captured_piece_bb;
    temp_board.knights &= !captured_piece_bb;
    temp_board.bishops &= !captured_piece_bb;
    temp_board.rooks &= !captured_piece_bb;
    temp_board.queens &= !captured_piece_bb;
    temp_board.kings &= !captured_piece_bb;

    // Se após remoção a casa não está mais atacada, a peça capturada era atacante
    let was_attacked_before = board.is_square_attacked_by(valuable_sq, enemy_color);
    let is_attacked_after = temp_board.is_square_attacked_by(valuable_sq, enemy_color);

    was_attacked_before && !is_attacked_after
}

/// Ordenação específica para quiescence search (só capturas + checks)
pub fn order_tactical_moves(
    board: &Board,
    moves: Vec<Move>,
    tt: &TranspositionTable,
    _context: &SearchContext
) -> Vec<Move> {
    let tt_move = if let Some(entry) = tt.probe(board.zobrist_hash) {
        entry.best_move.filter(|mv| board.is_legal_move(*mv))
    } else {
        None
    };

    let mut scored_moves = moves.into_iter().map(|mv| {
        let mut score = 0;

        if Some(mv) == tt_move {
            score = 50_000; // TT move
        } else if board.is_capture(mv) {
            // Usa SEE para ordenar capturas
            let see_value = see(board, mv);
            score = 10_000 + see_value; // Capturas boas primeiro
        } else {
            // Checks e outras táticas
            score = 1_000;
        }

        (mv, score)
    }).collect::<Vec<(Move, i32)>>();

    scored_moves.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    scored_moves.into_iter().map(|(mv, _)| mv).collect()
}

/// Versão rápida para root moves (sem context)
pub fn order_root_moves(board: &Board, moves: Vec<Move>) -> Vec<Move> {
    let mut scored_moves = moves.into_iter().map(|mv| {
        let mut score = 0;

        if board.is_capture(mv) {
            let see_value = see(board, mv);
            if see_value > 0 {
                score = 50_000 + see_value;
            } else if see_value == 0 {
                score = 40_000;
            } else {
                score = 10_000 + see_value;
            }
        } else if mv.promotion.is_some() {
            // Promoções têm alta prioridade na root
            score = match mv.promotion {
                Some(PieceKind::Queen) => 25_000,
                Some(PieceKind::Rook) => 23_000,
                Some(PieceKind::Bishop) => 21_000,
                Some(PieceKind::Knight) => 19_000,
                _ => 17_000,
            };
        } else if mv.is_castling {
            score = 15_000;
        } else if is_development_move(board, mv) {
            score = 8_000;
        } else if controls_center(mv) {
            score = 5_000;
        }

        (mv, score)
    }).collect::<Vec<(Move, i32)>>();

    scored_moves.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    scored_moves.into_iter().map(|(mv, _)| mv).collect()
}

/// Funções auxiliares para integração com o sistema de busca

/// Atualiza context após um cutoff (beta-cutoff)
pub fn update_context_on_cutoff(
    context: &mut SearchContext,
    board: &Board,
    best_move: Move,
    depth: u8,
    tried_moves: &[Move]
) {
    // 1. Adiciona killer move se não for captura
    if !board.is_capture(best_move) && best_move.promotion.is_none() {
        context.add_killer(best_move, depth);
    }

    // 2. Atualiza history heuristic
    if let Some(piece) = board.get_piece_on_square(best_move.from) {
        // Movimento que causou cutoff é bom
        context.update_history(best_move, piece, depth, true);
    }

    // 3. Penaliza movimentos que foram tentados antes do cutoff
    for &tried_move in tried_moves {
        if tried_move != best_move {
            if let Some(piece) = board.get_piece_on_square(tried_move.from) {
                context.update_history(tried_move, piece, depth, false);
            }
        }
    }

    // 4. Adiciona counter-move se há movimento anterior
    if let Some(prev_move) = context.get_last_move() {
        context.add_counter_move(prev_move, best_move);
    }
}

/// Detecta se captura envolve promoção (captura + promoção)
pub fn is_promotion_capture(board: &Board, mv: Move) -> bool {
    mv.promotion.is_some() && board.is_capture(mv)
}

/// Obtém valor da peça capturada para MVV-LVA
fn get_captured_piece_value(board: &Board, mv: Move) -> i32 {
    let target_bb = 1u64 << mv.to;
    
    if (target_bb & board.queens) != 0 { return 900; }
    if (target_bb & board.rooks) != 0 { return 500; }
    if (target_bb & board.bishops) != 0 { return 330; }
    if (target_bb & board.knights) != 0 { return 320; }
    if (target_bb & board.pawns) != 0 { return 100; }
    
    0 // En passant ou erro
}

/// Obtém valor da peça atacante para MVV-LVA
fn get_attacking_piece_value(board: &Board, mv: Move) -> i32 {
    let from_bb = 1u64 << mv.from;
    
    if (from_bb & board.queens) != 0 { return 900; }
    if (from_bb & board.rooks) != 0 { return 500; }
    if (from_bb & board.bishops) != 0 { return 330; }
    if (from_bb & board.knights) != 0 { return 320; }
    if (from_bb & board.pawns) != 0 { return 100; }
    if (from_bb & board.kings) != 0 { return 20000; }
    
    0
}

/// Detecta checks descobertos (mais perigosos)
fn is_discovered_check(board: &Board, mv: Move) -> bool {
    let king_pos = if board.to_move == Color::White {
        (board.kings & board.black_pieces).trailing_zeros() as u8
    } else {
        (board.kings & board.white_pieces).trailing_zeros() as u8
    };
    
    // Verifica se mover a peça de 'from' expõe uma linha de ataque ao rei inimigo
    let from_to_king = attacks_between(mv.from, king_pos);
    let our_sliding_pieces = if board.to_move == Color::White {
        (board.bishops | board.rooks | board.queens) & board.white_pieces
    } else {
        (board.bishops | board.rooks | board.queens) & board.black_pieces
    };
    
    (from_to_king & our_sliding_pieces) != 0
}

/// Calcula bitboard de ataques entre duas casas (simplificado)
fn attacks_between(from: u8, to: u8) -> u64 {
    // Implementação simplificada - pode ser melhorada com magic bitboards
    let from_rank = from / 8;
    let from_file = from % 8;
    let to_rank = to / 8;
    let to_file = to % 8;
    
    // Mesma linha/coluna/diagonal
    if from_rank == to_rank || from_file == to_file || 
       (from_rank as i8 - to_rank as i8).abs() == (from_file as i8 - to_file as i8).abs() {
        // Simplificação: retorna bitboard das casas intermediárias
        let mut result = 0u64;
        let rank_diff = (to_rank as i8 - from_rank as i8).signum();
        let file_diff = (to_file as i8 - from_file as i8).signum();
        
        let mut current_rank = from_rank as i8 + rank_diff;
        let mut current_file = from_file as i8 + file_diff;
        
        while current_rank != to_rank as i8 || current_file != to_file as i8 {
            if current_rank >= 0 && current_rank < 8 && current_file >= 0 && current_file < 8 {
                result |= 1u64 << (current_rank * 8 + current_file);
            }
            current_rank += rank_diff;
            current_file += file_diff;
        }
        
        result
    } else {
        0
    }
}

/// Verifica se movimento melhora estrutura de peões
fn improves_pawn_structure(board: &Board, mv: Move) -> bool {
    let from_bb = 1u64 << mv.from;
    
    // Só para movimentos de peão
    if (from_bb & board.pawns) == 0 {
        return false;
    }
    
    let to_file = mv.to % 8;
    let our_pawns = board.pawns & if board.to_move == Color::White {
        board.white_pieces
    } else {
        board.black_pieces
    };
    
    // Heurísticas simples:
    // 1. Avança peão central
    if to_file >= 3 && to_file <= 4 {
        return true;
    }
    
    // 2. Conecta peões isolados
    let left_file = if to_file > 0 { to_file - 1 } else { to_file };
    let right_file = if to_file < 7 { to_file + 1 } else { to_file };
    
    let adjacent_files_mask = file_mask(left_file) | file_mask(right_file);
    if (our_pawns & adjacent_files_mask) != 0 {
        return true;
    }
    
    false
}

/// Verifica se movimento enfraquece segurança do rei
fn weakens_king_safety(board: &Board, mv: Move) -> bool {
    let from_bb = 1u64 << mv.from;
    let our_king = board.kings & if board.to_move == Color::White {
        board.white_pieces
    } else {
        board.black_pieces
    };
    
    if our_king == 0 {
        return false;
    }
    
    let king_pos = our_king.trailing_zeros() as u8;
    let king_file = king_pos % 8;
    let king_rank = king_pos / 8;
    
    // Verifica se está movendo peça defensiva próxima ao rei
    let to_king_distance = distance(mv.from, king_pos);
    
    if to_king_distance <= 2 {
        // Move peça defensiva para longe do rei
        let after_king_distance = distance(mv.to, king_pos);
        if after_king_distance > to_king_distance + 1 {
            return true;
        }
        
        // Remove defensor de peão na frente do rei (com proteção contra underflow)
        if board.to_move == Color::White && king_rank < 7 && mv.from / 8 == king_rank + 1 {
            let file_diff = (mv.from % 8) as i8 - king_file as i8;
            if file_diff.abs() <= 1 {
                return true;
            }
        } else if board.to_move == Color::Black && king_rank > 0 && mv.from / 8 == king_rank.saturating_sub(1) {
            let file_diff = (mv.from % 8) as i8 - king_file as i8;
            if file_diff.abs() <= 1 {
                return true;
            }
        }
    }
    
    false
}

/// Calcula distância entre duas casas
fn distance(sq1: u8, sq2: u8) -> u8 {
    let rank1 = sq1 / 8;
    let file1 = sq1 % 8;
    let rank2 = sq2 / 8;
    let file2 = sq2 % 8;
    
    let rank_diff = (rank1 as i8 - rank2 as i8).abs() as u8;
    let file_diff = (file1 as i8 - file2 as i8).abs() as u8;
    
    rank_diff.max(file_diff)
}

/// Gera máscara de arquivo
fn file_mask(file: u8) -> u64 {
    0x0101010101010101u64 << file
}

// ============================================================================
// FUNÇÕES AUXILIARES APRIMORADAS PARA ORDENAÇÃO MODERNA
// ============================================================================

/// Avalia controle de centro de forma mais refinada
fn evaluate_center_control(mv: Move, board: &Board) -> i32 {
    let to_file = mv.to % 8;
    let to_rank = mv.to / 8;
    
    let mut score = 0;
    
    // Centro absoluto (d4, d5, e4, e5) - pontuação máxima
    if (to_file == 3 || to_file == 4) && (to_rank == 3 || to_rank == 4) {
        score += 20000;
    }
    // Centro estendido (c3-f6) - pontuação média
    else if to_file >= 2 && to_file <= 5 && to_rank >= 2 && to_rank <= 5 {
        score += 12000;
    }
    // Centro amplo - pontuação baixa
    else if to_file >= 1 && to_file <= 6 && to_rank >= 1 && to_rank <= 6 {
        score += 6000;
    }
    
    // Bônus adicional se a peça que move é um peão ou cavalo (peças que se beneficiam de centralização)
    if let Some(piece) = board.get_piece_on_square(mv.from) {
        match piece {
            crate::types::PieceKind::Pawn => score += score / 2,
            crate::types::PieceKind::Knight => score += score / 3,
            _ => {}
        }
    }
    
    score
}

/// Avalia urgência defensiva de um movimento
fn evaluate_defensive_urgency(board: &Board, mv: Move) -> i32 {
    let our_color = board.to_move;
    let enemy_color = !our_color;
    let our_pieces = if our_color == crate::types::Color::White { board.white_pieces } else { board.black_pieces };
    
    let mut urgency = 0;
    
    // Conta peças valiosas atacadas pelo inimigo
    let our_valuables = (board.knights | board.bishops | board.rooks | board.queens) & our_pieces;
    let mut attacked_value = 0;
    let mut bb = our_valuables;
    
    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        
        if board.is_square_attacked_by(sq, enemy_color) {
            // Adiciona valor da peça atacada
            if let Some(piece) = board.get_piece_on_square(sq) {
                attacked_value += match piece {
                    crate::types::PieceKind::Queen => 900,
                    crate::types::PieceKind::Rook => 500,
                    crate::types::PieceKind::Bishop => 330,
                    crate::types::PieceKind::Knight => 320,
                    _ => 0,
                };
            }
        }
    }
    
    // Urgência baseada no valor das peças atacadas
    urgency += (attacked_value / 10).min(10000);
    
    // Urgência extra se rei está em xeque
    if board.is_king_in_check(our_color) {
        urgency += 5000;
    }
    
    urgency
}

/// Verifica se movimento pode levar a ameaça de mate
fn could_lead_to_mate_threat(board: &Board, mv: Move) -> bool {
    // Implementação simplificada - pode ser expandida
    let mut temp_board = *board;
    let _undo_info = temp_board.make_move_fast(mv);
    
    // Verifica se após o movimento, o rei inimigo está em situação crítica
    let enemy_color = !board.to_move;
    let enemy_king = temp_board.kings & if enemy_color == crate::types::Color::White {
        temp_board.white_pieces
    } else {
        temp_board.black_pieces
    };
    
    if enemy_king == 0 { return false; }
    
    let king_sq = enemy_king.trailing_zeros() as u8;
    let king_attacks = crate::moves::king::get_king_attacks_lookup(king_sq);
    let safe_squares = king_attacks & !crate::evaluation::mobility::compute_attacked_squares(&temp_board, board.to_move);
    
    // Se rei tem 2 ou menos casas seguras, pode ser ameaça de mate
    safe_squares.count_ones() <= 2
}

/// Verifica se movimento cria ameaça tática
fn creates_tactical_threat(board: &Board, mv: Move) -> bool {
    // Implementação básica - procura por forks, pins, skewers
    let mut temp_board = *board;
    let _undo_info = temp_board.make_move_fast(mv);
    
    let enemy_color = !board.to_move;
    let enemy_pieces = if enemy_color == crate::types::Color::White {
        temp_board.white_pieces
    } else {
        temp_board.black_pieces
    };
    
    // Conta peças inimigas atacadas após o movimento
    let enemy_valuables = (temp_board.knights | temp_board.bishops | temp_board.rooks | temp_board.queens) & enemy_pieces;
    let mut attacked_count = 0;
    let mut bb = enemy_valuables;
    
    while bb != 0 && attacked_count < 3 {
        let sq = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        
        if temp_board.is_square_attacked_by(sq, board.to_move) {
            attacked_count += 1;
        }
    }
    
    // Se ataca 2+ peças valiosas, é ameaça tática (possível fork)
    attacked_count >= 2
}

/// Avalia ganho de mobilidade da peça
fn evaluate_piece_mobility_gain(board: &Board, mv: Move) -> i32 {
    if let Some(piece) = board.get_piece_on_square(mv.from) {
        match piece {
            crate::types::PieceKind::Knight => {
                // Cavalos se beneficiam de posições centrais
                let to_file = mv.to % 8;
                let to_rank = mv.to / 8;
                let center_distance = ((to_file as i32 - 4).abs() + (to_rank as i32 - 4).abs()) as u8;
                (8 - center_distance as i32) * 300
            },
            crate::types::PieceKind::Bishop => {
                // Bispos se beneficiam de diagonais longas
                evaluate_diagonal_mobility(mv.to) * 200
            },
            crate::types::PieceKind::Rook => {
                // Torres se beneficiam de fileiras e colunas abertas
                evaluate_file_rank_mobility(board, mv.to) * 150
            },
            crate::types::PieceKind::Queen => {
                // Rainha se beneficia de posições ativas
                evaluate_queen_activity(board, mv.to) * 100
            },
            _ => 0,
        }
    } else {
        0
    }
}

/// Verifica se movimento isola peça própria
fn isolates_own_piece(board: &Board, mv: Move) -> bool {
    // Implementação simplificada - verifica se move peça para casa sem apoio
    let mut temp_board = *board;
    let _undo_info = temp_board.make_move_fast(mv);
    
    let our_color = board.to_move;
    
    // Verifica se a peça movida está atacada e não defendida
    temp_board.is_square_attacked_by(mv.to, !our_color) &&
    !temp_board.is_square_attacked_by(mv.to, our_color)
}

/// Avalia impacto na segurança do rei de forma mais precisa
fn evaluate_king_safety_impact(board: &Board, mv: Move) -> i32 {
    let our_color = board.to_move;
    let our_king = board.kings & if our_color == crate::types::Color::White {
        board.white_pieces
    } else {
        board.black_pieces
    };
    
    if our_king == 0 { return 0; }
    
    let king_pos = our_king.trailing_zeros() as u8;
    let king_distance_before = distance(mv.from, king_pos);
    let king_distance_after = distance(mv.to, king_pos);
    
    let mut penalty = 0;
    
    // Penaliza movimento de defensores próximos ao rei
    if king_distance_before <= 2 && king_distance_after > king_distance_before + 1 {
        penalty += 15000;
        
        // Penalidade extra se rei não fez castling (simplificação - verifica se rei está na posição inicial)
        let initial_king_pos = if our_color == crate::types::Color::White { 4 } else { 60 };
        if king_pos == initial_king_pos {
            penalty += 5000;
        }
    }
    
    // Penaliza abertura de linhas de ataque ao rei
    if opens_attack_line_to_king(board, mv, king_pos) {
        penalty += 20000;
    }
    
    penalty
}

/// Funções auxiliares para mobilidade
fn evaluate_diagonal_mobility(square: u8) -> i32 {
    let file = square % 8;
    let rank = square / 8;
    
    // Diagonais longas são melhores
    let main_diagonal = if file == rank { 8 - (file as i32 - 4).abs() } else { 0 };
    let anti_diagonal = if file + rank == 7 { 8 - (file as i32 - 4).abs() } else { 0 };
    
    main_diagonal.max(anti_diagonal)
}

fn evaluate_file_rank_mobility(board: &Board, square: u8) -> i32 {
    let file = square % 8;
    let rank = square / 8;
    
    let mut mobility = 0;
    
    // Avalia abertura da coluna
    let file_mask = 0x0101010101010101u64 << file;
    let pawns_in_file = (board.pawns & file_mask).count_ones();
    mobility += (8 - pawns_in_file as i32) * 2;
    
    // Avalia abertura da fileira
    let rank_mask = 0xFFu64 << (rank * 8);
    let pieces_in_rank = ((board.white_pieces | board.black_pieces) & rank_mask).count_ones();
    mobility += (8 - pieces_in_rank as i32);
    
    mobility
}

fn evaluate_queen_activity(board: &Board, square: u8) -> i32 {
    let file = square % 8;
    let rank = square / 8;
    
    let mut activity = 0;
    
    // Rainha ativa no centro
    let center_bonus = 8 - ((file as i32 - 4).abs() + (rank as i32 - 4).abs());
    activity += center_bonus * 2;
    
    // Rainha avançada no território inimigo
    let enemy_territory = if board.to_move == crate::types::Color::White {
        rank >= 5
    } else {
        rank <= 2
    };
    
    if enemy_territory {
        activity += 5;
    }
    
    activity
}

/// Verifica se movimento abre linha de ataque ao rei
fn opens_attack_line_to_king(board: &Board, mv: Move, king_pos: u8) -> bool {
    // Implementação simplificada - verifica se remove peça que bloqueia linha
    let from_to_king = attacks_between(mv.from, king_pos);
    
    // Se há linha entre 'from' e rei, verifica se há atacantes inimigos atrás
    if from_to_king != 0 {
        let enemy_pieces = if board.to_move == crate::types::Color::White {
            board.black_pieces
        } else {
            board.white_pieces
        };
        
        let enemy_sliders = (board.bishops | board.rooks | board.queens) & enemy_pieces;
        
        // Verifica se há peças inimigas deslizantes que podem atacar o rei pela linha
        (from_to_king & enemy_sliders) != 0
    } else {
        false
    }
}