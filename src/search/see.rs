// Static Exchange Evaluation - versão corrigida e otimizada
use crate::{board::Board, types::{Color, PieceKind, Move}};
use super::PIECE_VALUES;

/// Static Exchange Evaluation - calcula ganho/perda líquida de uma captura
/// Retorna valor positivo se vantajoso, negativo se perdedor
pub fn see(board: &Board, mv: Move) -> i32 {
    // Se não é captura, retorna 0 (sem exchange)
    if !board.is_capture(mv) {
        return 0;
    }
    
    let target = mv.to;
    let attacker_kind = board.get_piece_on_square(mv.from).unwrap();
    let victim_kind = board.get_piece_on_square(target);
    
    if victim_kind.is_none() {
        return 0; // En passant ou erro
    }
    
    let victim_value = PIECE_VALUES[victim_kind.unwrap() as usize];
    let attacker_value = PIECE_VALUES[attacker_kind as usize];
    
    // Ganho inicial: valor da peça capturada
    let mut gain = victim_value;
    
    // Simula recapturas em sequência
    let mut temp_board = *board;
    temp_board.make_move(mv);
    
    let recapture_value = see_recapture(&temp_board, target, !board.to_move, attacker_value, 0);
    
    gain - recapture_value.max(0)
}

/// Versão recursiva para calcular recapturas (com limite de profundidade)
fn see_recapture(board: &Board, target_square: u8, side_to_move: Color, last_attacker_value: i32, depth: u8) -> i32 {
    // Limita recursão para evitar stack overflow
    if depth > 10 {
        return 0;
    }
    // Encontra a peça menos valiosa que pode recapturar
    if let Some((recapture_sq, recapture_kind)) = find_least_valuable_attacker(board, target_square, side_to_move) {
        let recapture_value = PIECE_VALUES[recapture_kind as usize];
        
        // Executa a recaptura
        let mut temp_board = *board;
        let recapture_move = Move {
            from: recapture_sq,
            to: target_square,
            promotion: None,
            is_castling: false,
            is_en_passant: false,
        };
        
        temp_board.make_move(recapture_move);
        
        // Valor ganho: peça atacada menos valor da próxima recaptura
        let next_recapture = see_recapture(&temp_board, target_square, !side_to_move, recapture_value, depth + 1);
        
        // Retorna o melhor entre recapturar ou não
        (last_attacker_value - next_recapture.max(0)).max(0)
    } else {
        // Não há recaptura possível
        0
    }
}

/// Encontra o atacante menos valioso que pode atacar uma casa
fn find_least_valuable_attacker(board: &Board, target: u8, color: Color) -> Option<(u8, PieceKind)> {
    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    
    // Ordem de prioridade: Peão, Cavalo, Bispo, Torre, Rainha, Rei
    let piece_types = [
        (board.pawns & our_pieces, PieceKind::Pawn),
        (board.knights & our_pieces, PieceKind::Knight),
        (board.bishops & our_pieces, PieceKind::Bishop),
        (board.rooks & our_pieces, PieceKind::Rook),
        (board.queens & our_pieces, PieceKind::Queen),
        (board.kings & our_pieces, PieceKind::King),
    ];
    
    for (mut piece_bb, kind) in piece_types {
        while piece_bb != 0 {
            let sq = piece_bb.trailing_zeros() as u8;
            piece_bb &= piece_bb - 1;
            
            if can_attack_square(board, sq, target, kind) {
                return Some((sq, kind));
            }
        }
    }
    
    None
}

/// Verifica se uma peça específica pode atacar uma casa
fn can_attack_square(board: &Board, from_square: u8, target_square: u8, piece_kind: PieceKind) -> bool {
    let all_pieces = board.white_pieces | board.black_pieces;
    
    match piece_kind {
        PieceKind::Pawn => {
            // Ataques de peão (diagonal)
            let color = if (board.white_pieces & (1u64 << from_square)) != 0 { Color::White } else { Color::Black };
            pawn_can_attack(from_square, target_square, color)
        },
        PieceKind::Knight => {
            let attacks = crate::moves::knight::get_knight_attacks_lookup(from_square);
            (attacks & (1u64 << target_square)) != 0
        },
        PieceKind::Bishop => {
            let attacks = crate::moves::sliding::get_bishop_attacks(from_square, all_pieces);
            (attacks & (1u64 << target_square)) != 0
        },
        PieceKind::Rook => {
            let attacks = crate::moves::sliding::get_rook_attacks(from_square, all_pieces);
            (attacks & (1u64 << target_square)) != 0
        },
        PieceKind::Queen => {
            let bishop_attacks = crate::moves::sliding::get_bishop_attacks(from_square, all_pieces);
            let rook_attacks = crate::moves::sliding::get_rook_attacks(from_square, all_pieces);
            ((bishop_attacks | rook_attacks) & (1u64 << target_square)) != 0
        },
        PieceKind::King => {
            let attacks = crate::moves::king::get_king_attacks_lookup(from_square);
            (attacks & (1u64 << target_square)) != 0
        },
    }
}

/// Verifica ataques de peão
fn pawn_can_attack(pawn_square: u8, target_square: u8, pawn_color: Color) -> bool {
    let pawn_rank = pawn_square / 8;
    let pawn_file = pawn_square % 8;
    let target_rank = target_square / 8;
    let target_file = target_square % 8;
    
    match pawn_color {
        Color::White => {
            // Brancas atacam para cima
            target_rank == pawn_rank + 1 && 
            (target_file == pawn_file + 1 || target_file + 1 == pawn_file) && 
            pawn_file < 8 && target_file < 8
        },
        Color::Black => {
            // Pretas atacam para baixo  
            pawn_rank > 0 && target_rank == pawn_rank - 1 &&
            (target_file == pawn_file + 1 || target_file + 1 == pawn_file) &&
            pawn_file < 8 && target_file < 8
        }
    }
}

/// SEE simplificado para usar em move ordering
/// Retorna true se captura é boa/neutra, false se perdedora
pub fn see_capture_good(board: &Board, mv: Move) -> bool {
    see(board, mv) >= 0
}

/// Threshold SEE - para otimização em pruning
/// Retorna true se SEE >= threshold
pub fn see_threshold(board: &Board, mv: Move, threshold: i32) -> bool {
    see(board, mv) >= threshold
}