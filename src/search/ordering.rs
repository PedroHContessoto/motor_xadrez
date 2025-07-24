// Move ordering aprimorado com SEE e counter-moves
use crate::{board::Board, transposition::TranspositionTable, types::{Move, PieceKind, Color}};
use super::{SearchContext, see::see};

const PIECE_VALUES: [i32; 6] = [100, 320, 330, 500, 900, 20000];

/// Ordena movimentos para maximizar cutoffs (melhor primeiro)
pub fn order_moves(
    board: &Board,
    moves: Vec<Move>,
    tt: &TranspositionTable,
    context: &SearchContext,
    depth: u8
) -> Vec<Move> {
    // Move da TT tem prioridade máxima (se legal)
    let tt_move = if let Some(entry) = tt.probe(board.zobrist_hash) {
        entry.best_move.filter(|mv| board.is_legal_move(*mv))
    } else {
        None
    };

    let mut scored_moves = moves.into_iter().map(|mv| {
        let mut score = 0;

        // 1. TT Move (prioridade máxima)
        if Some(mv) == tt_move {
            score = 1_000_000; // Aumentar prioridade
        }
        // 2. Capturas (ordenadas por SEE + MVV-LVA)
        else if board.is_capture(mv) {
            let see_value = see(board, mv);
            if see_value > 0 {
                score = 900_000 + see_value;
            } else if see_value == 0 {
                score = 800_000;
            } else {
                score = 100_000 + see_value; // Ainda considera capturas ruins
            }
        }
        // 3. Castling (desenvolvimento seguro)
        else if mv.is_castling {
            score = 15_000;
        }
        // 4. Promoções (prioridade muito alta, especialmente promoção para dama)
        else if mv.promotion.is_some() {
            score = match mv.promotion {
                Some(PieceKind::Queen) => 30_000,
                Some(PieceKind::Rook) => 28_000,
                Some(PieceKind::Bishop) => 26_000,
                Some(PieceKind::Knight) => 24_000,
                _ => 22_000,
            };
        }
        // 5. Movimentos defensivos (nova prioridade alta)
        else if is_defensive_move(board, mv) {
            score = 20_000;
        }
        // 6. Killers (moves que causaram cutoffs)
        else if context.is_killer(mv, depth) {
            score = 700_000; // Aumentar importância
        }
        // 7. Counter-moves (refutam último movimento inimigo)
        else if is_counter_move(mv, context) {
            score = 600_000;
        }
        // 8. Checks (podem causar táticas)
        else if gives_check_heuristic(board, mv) {
            score = 500_000;
        }
        // 9. CRÍTICO: Penaliza movimentos de rei no meio-jogo
        else if board.get_piece_on_square(mv.from) == Some(PieceKind::King) {
            // Verifica se não é final de jogo (heurística simples)
            let total_pieces = (board.white_pieces | board.black_pieces).count_ones();
            if total_pieces > 10 { // Não é endgame
                score = 50_000; // Muito baixa prioridade

                // Exceção para roque - ainda é bom
                if mv.is_castling {
                    score = 750_000; // Alta prioridade para roque
                }
            } else {
                // No endgame, rei ativo é bom
                score = 400_000;
            }
        }
        // 10. Histórico (moves que foram bons antes)
        else {
            if let Some(piece) = board.get_piece_on_square(mv.from) {
                let history_score = context.get_history_score(mv, piece);
                score = 300_000 + history_score.clamp(-50_000, 50_000);
            } else {
                score = 300_000;
            }

            // Bônus por desenvolvimento na abertura
            if is_development_move(board, mv) {
                score += 5_000;
            }

            // Bônus por controle de centro
            if controls_center(mv) {
                score += 2_000;
            }
        }

        (mv, score)
    }).collect::<Vec<(Move, i32)>>();

    // Ordena por pontuação (maior primeiro)
    scored_moves.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    scored_moves.into_iter().map(|(mv, _)| mv).collect()
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
    temp_board.make_move(mv);

    // Se após o movimento a casa não está mais atacada, bloqueou o ataque
    !temp_board.is_square_attacked_by(target_sq, enemy_color)
}

/// Verifica se após o movimento a peça não estará atacada
fn would_be_attacked_after_move(board: &Board, mv: Move, enemy_color: Color) -> bool {
    let mut temp_board = *board;
    temp_board.make_move(mv);

    temp_board.is_square_attacked_by(mv.to, enemy_color)
}

/// Verifica se após o movimento a casa estará defendida
fn defends_square_after_move(board: &Board, mv: Move, defended_sq: u8) -> bool {
    let mut temp_board = *board;
    temp_board.make_move(mv);

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