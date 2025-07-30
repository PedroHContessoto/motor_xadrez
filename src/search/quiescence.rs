// Quiescence Search aprimorada com SEE filtering
use crate::{board::Board, evaluation, transposition::TranspositionTable, types::{Move, Color}};
use super::{SearchContext, order_moves, see::see};

const MATE_VALUE: i32 = 99999;
const SEE_THRESHOLD: i32 = -15; // Optimized threshold for better tactical play
const DELTA_MARGIN: i32 = 900; // Reduced from 950 for more aggressive play
const MAX_QUIESCENCE_PLY: i32 = 6; // Increased from 4 for deeper tactical search

/// Enhanced Quiescence Search with improved tactical detection
pub fn quiescence_search(
    board: &Board,
    mut alpha: i32,
    beta: i32,
    tt: &mut TranspositionTable,
    context: &mut SearchContext
) -> i32 {
    quiescence_search_with_ply(board, alpha, beta, 0, MAX_QUIESCENCE_PLY, tt, context)
}

/// Quiescence search com limite de ply para prevenir recursão infinita
fn quiescence_search_with_ply(
    board: &Board,
    mut alpha: i32,
    beta: i32,
    ply: i32,
    max_ply: i32,
    tt: &mut TranspositionTable,
    context: &mut SearchContext
) -> i32 {

    // Termina se atingiu limite de ply ou stop flag
    if ply >= max_ply || context.should_stop {
        return evaluation::evaluate(board);
    }
    let in_check = board.is_king_in_check(board.to_move);

    // Stand pat - avaliação da posição quieta
    let stand_pat = evaluation::evaluate(board);

    // Beta cutoff
    if !in_check && stand_pat >= beta {
        return beta;
    }

    // Atualiza alpha
    if !in_check && alpha < stand_pat {
        alpha = stand_pat;
    }

    // Enhanced Delta pruning with tactical awareness
    // Exception: no delta pruning in check or when material imbalance is large
    if !in_check && stand_pat + DELTA_MARGIN < alpha {
        // Don't prune if there are hanging pieces that might be captured
        let has_tactical_potential = has_hanging_pieces_quick(board) || 
                                   has_discovered_attack_potential(board);
        if !has_tactical_potential {
            return alpha;
        }
    }

    // Gera capturas e movimentos de xeque
    let mut tactical_moves = Vec::new();

    if in_check {
        // Em xeque: considera todos os moves legais
        tactical_moves.extend(board.generate_legal_moves());
    } else {
        // Não em xeque: só capturas e checks
        // Capturas de peão (mais eficiente)
        tactical_moves.extend(crate::moves::pawn::generate_pawn_captures(board));

        // Capturas de outras peças
        let all_moves = board.generate_legal_moves();
        for mv in all_moves {
            if board.is_capture(mv) {
                tactical_moves.push(mv);
            } else {
                // Verifica se é check sem fazer o movimento (otimização)
                if gives_check_fast(board, mv) {
                    tactical_moves.push(mv);
                }
            }
        }
    }

    // Enhanced SEE filtering with tactical exceptions
    if !in_check {
        tactical_moves.retain(|&mv| {
            if board.is_capture(mv) {
                let captured_value = get_captured_piece_value_quick(board, mv);
                let see_value = see(board, mv);
                
                // Always consider high-value captures
                if captured_value >= 500 { // Rook or Queen
                    return true;
                }
                
                // Consider promotion captures
                if mv.promotion.is_some() {
                    return true;
                }
                
                // Consider captures that might be part of tactical sequences
                if see_value >= SEE_THRESHOLD || is_potential_tactical_capture(board, mv) {
                    return true;
                }
                
                false
            } else {
                true // Keep all checks
            }
        });
    }

    // Ordena movimentos (capturas boas primeiro)
    let ordered_moves = order_moves(board, tactical_moves, tt, context, 0);

    for mv in ordered_moves {
        // Valida legalidade
        if !board.is_legal_move(mv) {
            continue;
        }

        // Faz movimento e busca recursivamente (usa copy-make)
        let temp_board = board.make_move_copy(mv);

        let score = -quiescence_search_with_ply(&temp_board, -beta, -alpha, ply + 1, max_ply, tt, context);

        // Beta cutoff
        if score >= beta {
            return beta;
        }

        // Atualiza alpha
        if score > alpha {
            alpha = score;
        }
    }

    alpha
}

/// Verifica rapidamente se um movimento dá xeque (sem fazer o movimento)
/// Versão simplificada - pode dar falsos negativos, mas sem falsos positivos
pub fn gives_check_fast(board: &Board, mv: Move) -> bool {
    let piece_kind = board.get_piece_on_square(mv.from);
    if piece_kind.is_none() {
        return false;
    }

    let enemy_king_bb = board.kings & if board.to_move == Color::White {
        board.black_pieces
    } else {
        board.white_pieces
    };

    if enemy_king_bb == 0 {
        return false;
    }

    let enemy_king_square = enemy_king_bb.trailing_zeros() as u8;

    // Verifica se a peça movida pode atacar o rei inimigo da nova posição
    match piece_kind.unwrap() {
        crate::types::PieceKind::Knight => {
            let attacks = crate::moves::knight::get_knight_attacks_lookup(mv.to);
            (attacks & (1u64 << enemy_king_square)) != 0
        },
        crate::types::PieceKind::Bishop | crate::types::PieceKind::Queen => {
            // Simula ocupação após movimento
            let mut temp_occ = board.white_pieces | board.black_pieces;
            temp_occ &= !(1u64 << mv.from); // Remove peça da origem
            temp_occ |= 1u64 << mv.to;      // Adiciona na destino

            let attacks = crate::moves::magic_bitboards::get_bishop_attacks_magic(mv.to, temp_occ);
            (attacks & (1u64 << enemy_king_square)) != 0
        },
        crate::types::PieceKind::Rook => {
            // Similar para torre
            let mut temp_occ = board.white_pieces | board.black_pieces;
            temp_occ &= !(1u64 << mv.from);
            temp_occ |= 1u64 << mv.to;

            let attacks = crate::moves::magic_bitboards::get_rook_attacks_magic(mv.to, temp_occ);
            (attacks & (1u64 << enemy_king_square)) != 0
        },
        crate::types::PieceKind::Pawn => {
            // Ataques de peão
            let rank_diff = (mv.to / 8) as i8 - (enemy_king_square / 8) as i8;
            let file_diff = (mv.to % 8) as i8 - (enemy_king_square % 8) as i8;

            if board.to_move == Color::White {
                rank_diff == 1 && file_diff.abs() == 1
            } else {
                rank_diff == -1 && file_diff.abs() == 1
            }
        },
        _ => false, // Rei não dá xeque direto normalmente
    }
}

/// Quick check for hanging pieces without full SEE calculation
fn has_hanging_pieces_quick(board: &Board) -> bool {
    let enemy_color = !board.to_move;
    let our_pieces = if board.to_move == Color::White { 
        board.white_pieces 
    } else { 
        board.black_pieces 
    };
    
    // Check valuable pieces under attack
    let valuable_pieces = (board.knights | board.bishops | board.rooks | board.queens) & our_pieces;
    let mut bb = valuable_pieces;
    
    // Check up to 3 pieces for performance
    for _ in 0..3 {
        if bb == 0 { break; }
        let sq = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        
        if board.is_square_attacked_by(sq, enemy_color) {
            return true;
        }
    }
    
    false
}

/// Quick check for discovered attack potential
fn has_discovered_attack_potential(board: &Board) -> bool {
    let our_color = board.to_move;
    let enemy_king_pos = if our_color == Color::White {
        let enemy_kings = board.kings & board.black_pieces;
        if enemy_kings == 0 { return false; }
        enemy_kings.trailing_zeros() as u8
    } else {
        let enemy_kings = board.kings & board.white_pieces;
        if enemy_kings == 0 { return false; }
        enemy_kings.trailing_zeros() as u8
    };
    
    // Quick check for sliding pieces that could create discovered attacks
    let our_sliding = if our_color == Color::White {
        (board.bishops | board.rooks | board.queens) & board.white_pieces
    } else {
        (board.bishops | board.rooks | board.queens) & board.black_pieces
    };
    
    // Simple heuristic: if we have sliding pieces, there might be discovered attack potential
    our_sliding.count_ones() >= 2
}

/// Quick piece value lookup for captures
fn get_captured_piece_value_quick(board: &Board, mv: Move) -> i32 {
    let target_bb = 1u64 << mv.to;
    
    if (target_bb & board.queens) != 0 { return 900; }
    if (target_bb & board.rooks) != 0 { return 500; }
    if (target_bb & board.bishops) != 0 { return 330; }
    if (target_bb & board.knights) != 0 { return 320; }
    if (target_bb & board.pawns) != 0 { return 100; }
    
    0
}

/// Check if capture might be part of tactical sequence
fn is_potential_tactical_capture(board: &Board, mv: Move) -> bool {
    // Captures that attack the enemy king area are often tactical
    let enemy_color = !board.to_move;
    let enemy_king_bb = board.kings & if enemy_color == Color::White {
        board.white_pieces
    } else {
        board.black_pieces
    };
    
    if enemy_king_bb == 0 { return false; }
    
    let enemy_king_sq = enemy_king_bb.trailing_zeros() as u8;
    let king_rank = enemy_king_sq / 8;
    let king_file = enemy_king_sq % 8;
    let capture_rank = mv.to / 8;
    let capture_file = mv.to % 8;
    
    // Capture near enemy king (within 2 squares) might be tactical
    let rank_diff = (capture_rank as i8 - king_rank as i8).abs();
    let file_diff = (capture_file as i8 - king_file as i8).abs();
    
    rank_diff <= 2 && file_diff <= 2
}

/// Versão alternativa de quiescence para uso em análise
pub fn quiescence_search_depth_limited(
    board: &Board,
    mut alpha: i32,
    beta: i32,
    depth_left: i32,
    tt: &mut TranspositionTable,
    context: &mut SearchContext
) -> i32 {
    if depth_left <= 0 {
        return evaluation::evaluate(board);
    }

    quiescence_search(board, alpha, beta, tt, context)
}