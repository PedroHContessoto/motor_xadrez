// Segurança do rei aprimorada - considera ataques inimigos
use crate::{board::Board, types::{Color, Bitboard}};
use super::game_phase::GamePhase;

const CENTRAL_SQUARES: Bitboard = (1u64 << 27) | (1u64 << 28) | (1u64 << 35) | (1u64 << 36);

/// Avalia a segurança do rei (versão aprimorada)
pub fn evaluate_king_safety(board: &Board, color: Color, game_phase: &GamePhase) -> i32 {
    if matches!(game_phase, GamePhase::Endgame) {
        return 0; // Segurança menos importante no final
    }

    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let king_bb = board.kings & pieces;

    if king_bb == 0 {
        return 0;
    }

    let king_square = king_bb.trailing_zeros() as usize;
    let rank = king_square / 8;
    let file = king_square % 8;
    let mut score = 0;

    // 1. Shield de peões (código original mantido)
    score += evaluate_pawn_shield(board, color, king_square, rank, file);

    // 2. Penaliza rei no centro durante meio-jogo
    if (CENTRAL_SQUARES & king_bb) != 0 {
        score -= 50;
    }

    // 3. NOVO: Conta número de atacantes inimigos ao rei
    score -= evaluate_enemy_attackers(board, color, king_square);

    // 4. NOVO: Tropismo - Penaliza proximidade de peças inimigas
    score -= evaluate_tropism(board, color, king_square);

    // 5. Penalidade inteligente por rei exposto no meio-jogo
    if !matches!(game_phase, GamePhase::Endgame) {
        let king_rank = rank;
        let king_file = file;

        // Penalidade progressiva por rei exposto (mais balanceada)
        let exposed_penalty = match color {
            Color::White => {
                if king_rank > 1 {
                    // Penalidade crescente: -30 na 3ª, -60 na 4ª, -100 na 5ª, etc.
                    -(30 + (king_rank as i32 - 2) * 30)
                } else { 0 }
            },
            Color::Black => {
                if king_rank < 6 {
                    // Penalidade similar para pretas
                    -(30 + (5 - king_rank as i32) * 30)
                } else { 0 }
            }
        };

        score += exposed_penalty;

        // Penalidade por não ter feito roque quando necessário
        if !has_castled(board, color) && !can_castle(board, color) {
            score -= 60; // Penalidade moderada mas significativa
        }

        // Penalidade por rei no centro (apenas em posições muito expostas)
        if king_file >= 3 && king_file <= 4 && (king_rank >= 3 && king_rank <= 4) {
            score -= 40; // Rei no centro = perigoso mas não extremo
        }
    }

    score
}

fn evaluate_pawn_shield(board: &Board, color: Color, king_square: usize, rank: usize, file: usize) -> i32 {
    let mut score = 0;
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };

    let pawn_shield_files = [file.saturating_sub(1), file, (file + 1).min(7)];
    let pawn_shield_ranks = if color == Color::White {
        [rank + 1, rank + 2] // Fileiras na frente para brancas
    } else {
        [rank.saturating_sub(1), rank.saturating_sub(2)] // Fileiras na frente para pretas
    };

    for &shield_file in &pawn_shield_files {
        for &shield_rank in &pawn_shield_ranks {
            if shield_rank < 8 {
                let shield_square = shield_rank * 8 + shield_file;
                if shield_square < 64 {
                    let square_bb = 1u64 << shield_square;
                    if (board.pawns & pieces & square_bb) != 0 {
                        score += 15; // Bônus por peão protetor
                    } else {
                        score -= 20; // Penalidade por peão ausente
                    }
                }
            }
        }
    }

    score
}

fn evaluate_enemy_attackers(board: &Board, color: Color, king_square: usize) -> i32 {
    let enemy_color = !color;
    let mut penalty = 0;

    // Verifica se o rei está atacado
    if board.is_square_attacked_by(king_square as u8, enemy_color) {
        let enemy_pieces = if enemy_color == Color::White { board.white_pieces } else { board.black_pieces };

        // Conta atacantes por tipo (peso diferenciado por perigosidade)
        let pawn_attackers = count_piece_attackers(board, king_square as u8, enemy_color, board.pawns & enemy_pieces);
        let knight_attackers = count_piece_attackers(board, king_square as u8, enemy_color, board.knights & enemy_pieces);
        let bishop_attackers = count_piece_attackers(board, king_square as u8, enemy_color, board.bishops & enemy_pieces);
        let rook_attackers = count_piece_attackers(board, king_square as u8, enemy_color, board.rooks & enemy_pieces);
        let queen_attackers = count_piece_attackers(board, king_square as u8, enemy_color, board.queens & enemy_pieces);

        // Penalidades graduadas por tipo
        penalty += pawn_attackers * 10;    // Peões: baixa ameaça
        penalty += knight_attackers * 25;  // Cavalos: alta (forks)
        penalty += bishop_attackers * 20;  // Bispos: média-alta
        penalty += rook_attackers * 30;    // Torres: alta
        penalty += queen_attackers * 50;   // Rainha: crítica

        // Bônus por múltiplos atacantes (ataques combinados são perigosos)
        let total_attackers = pawn_attackers + knight_attackers + bishop_attackers + rook_attackers + queen_attackers;
        if total_attackers > 1 {
            penalty += total_attackers * 15; // Ex: 3 atacantes = +45 penalty extra
        }
    }

    penalty
}

fn count_piece_attackers(board: &Board, target_square: u8, attacker_color: Color, piece_bb: Bitboard) -> i32 {
    let mut count = 0;
    let mut bb = piece_bb;

    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        bb &= bb - 1;

        // Verifica se esta peça específica ataca o alvo
        if piece_attacks_square(board, sq, target_square, attacker_color) {
            count += 1;
        }
    }

    count
}

fn piece_attacks_square(board: &Board, piece_square: u8, target_square: u8, color: Color) -> bool {
    // Determina o tipo de peça e verifica ataque
    let piece_bb = 1u64 << piece_square;

    if (board.pawns & piece_bb) != 0 {
        // Ataque de peão
        let pawn_attacks = if color == Color::White {
            // Brancas: ataques diagonais para cima
            let left_attack = if piece_square % 8 > 0 { Some(piece_square + 7) } else { None };
            let right_attack = if piece_square % 8 < 7 { Some(piece_square + 9) } else { None };
            [left_attack, right_attack]
        } else {
            // Pretas: ataques diagonais para baixo
            let left_attack = if piece_square % 8 > 0 && piece_square >= 9 { Some(piece_square - 9) } else { None };
            let right_attack = if piece_square % 8 < 7 && piece_square >= 7 { Some(piece_square - 7) } else { None };
            [left_attack, right_attack]
        };

        pawn_attacks.iter().any(|&attack| attack == Some(target_square))
    } else if (board.knights & piece_bb) != 0 {
        // Ataque de cavalo
        let knight_attacks = crate::moves::knight::get_knight_attacks_lookup(piece_square);
        (knight_attacks & (1u64 << target_square)) != 0
    } else if (board.bishops & piece_bb) != 0 || (board.queens & piece_bb) != 0 {
        // Ataque de bispo ou rainha (diagonal)
        let all_pieces = board.white_pieces | board.black_pieces;
        let bishop_attacks = crate::moves::sliding::get_bishop_attacks(piece_square, all_pieces);
        (bishop_attacks & (1u64 << target_square)) != 0
    } else if (board.rooks & piece_bb) != 0 || (board.queens & piece_bb) != 0 {
        // Ataque de torre ou rainha (horizontal/vertical)
        let all_pieces = board.white_pieces | board.black_pieces;
        let rook_attacks = crate::moves::sliding::get_rook_attacks(piece_square, all_pieces);
        (rook_attacks & (1u64 << target_square)) != 0
    } else if (board.kings & piece_bb) != 0 {
        // Ataque de rei
        let king_attacks = crate::moves::king::get_king_attacks_lookup(piece_square);
        (king_attacks & (1u64 << target_square)) != 0
    } else {
        false
    }
}

fn evaluate_tropism(board: &Board, color: Color, king_square: usize) -> i32 {
    let enemy_color = !color;
    let enemy_pieces = if enemy_color == Color::White { board.white_pieces } else { board.black_pieces };

    // Casas ao redor do rei (1 e 2 quadrados de distância)
    let king_attacks = crate::moves::king::get_king_attacks_lookup(king_square as u8);

    // Extensão para 2 quadrados (simples aproximação)
    let extended_attacks = king_attacks |
        (king_attacks << 8) | (king_attacks >> 8) |  // Cima/baixo
        (king_attacks << 1) | (king_attacks >> 1);   // Esquerda/direita

    // Conta peças inimigas próximas
    let enemy_near_king = (extended_attacks & enemy_pieces).count_ones() as i32;

    // Penalidade por proximidade (peças próximas = pressão)
    enemy_near_king * 8
}

/// Verifica se o rei já fez roque (heurística baseada na posição)
fn has_castled(board: &Board, color: Color) -> bool {
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let king_bb = board.kings & pieces;

    if king_bb == 0 {
        return false;
    }

    let king_square = king_bb.trailing_zeros() as u8;

    match color {
        Color::White => {
            // Rei branco fez roque se está na casa 6 (g1) ou 2 (c1)
            king_square == 6 || king_square == 2
        },
        Color::Black => {
            // Rei preto fez roque se está na casa 62 (g8) ou 58 (c8)
            king_square == 62 || king_square == 58
        }
    }
}

/// Verifica se ainda pode fazer roque (heurística)
fn can_castle(board: &Board, color: Color) -> bool {
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let king_bb = board.kings & pieces;

    if king_bb == 0 {
        return false;
    }

    let king_square = king_bb.trailing_zeros() as u8;

    match color {
        Color::White => {
            // Rei branco pode fazer roque se está na casa inicial (4 = e1)
            if king_square != 4 {
                return false;
            }

            // Verifica se as torres estão nas posições iniciais
            let rooks = board.rooks & pieces;
            let king_rook = (rooks & (1u64 << 7)) != 0; // h1
            let queen_rook = (rooks & (1u64 << 0)) != 0; // a1

            king_rook || queen_rook
        },
        Color::Black => {
            // Rei preto pode fazer roque se está na casa inicial (60 = e8)
            if king_square != 60 {
                return false;
            }

            // Verifica se as torres estão nas posições iniciais
            let rooks = board.rooks & pieces;
            let king_rook = (rooks & (1u64 << 63)) != 0; // h8
            let queen_rook = (rooks & (1u64 << 56)) != 0; // a8

            king_rook || queen_rook
        }
    }
}