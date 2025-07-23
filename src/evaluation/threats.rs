// threats.rs - VERSÃO CORRIGIDA
// Avaliação de ameaças - peças penduradas, ataques táticos com penalidades mais severas

use crate::{board::Board, types::{Color, PieceKind}};
use super::material::MATERIAL_VALUES;

/// Avalia ameaças mútuas entre as cores
pub fn evaluate_threats(board: &Board, color: Color) -> i32 {
    let mut score = 0;

    // Penaliza nossas peças atacadas sem defesa adequada
    score -= evaluate_hanging_pieces(board, color);

    // Bônus por atacar peças inimigas
    score += evaluate_enemy_attacks(board, color);

    // Avalia pins e descobertos
    score += evaluate_pins_and_discoveries(board, color);

    score.clamp(-300, 300) // Reduzido de -800/800 para -300/300
}

/// CORREÇÃO 1: Penaliza peças próprias atacadas com penalidades mais severas
fn evaluate_hanging_pieces(board: &Board, color: Color) -> i32 {
    let mut penalty = 0;
    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let enemy_color = !color;

    // Analisa peças valiosas (não peões/rei - estes têm análise específica)
    let our_valuables = (board.knights | board.bishops | board.rooks | board.queens) & our_pieces;
    let mut bb = our_valuables;

    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        bb &= bb - 1;

        // CORREÇÃO CRÍTICA: Usa nova função que detecta ameaças de 2-3 tempos
        let threat_level = evaluate_square_threat_level(board, sq, enemy_color);
        
        if threat_level > 0 {
            let piece_kind = board.get_piece_on_square(sq).unwrap();
            let piece_value = MATERIAL_VALUES[piece_kind as usize];

            // Penalidade baseada no nível de ameaça
            let base_penalty = match threat_level {
                1 => piece_value / 4,        // Ameaça de 1 tempo (ataque direto)
                2 => piece_value / 2,        // Ameaça de 2 tempos (pode ser atacada facilmente)
                3 => (piece_value * 3) / 4,  // Ameaça de 3 tempos (provável perda)
                _ => piece_value,            // Ameaça crítica (perda certa)
            };

            penalty += base_penalty;

            // CORREÇÃO: Penalidades extras para cavalos avançados vulneráveis
            if piece_kind == PieceKind::Knight {
                if is_advanced_knight(sq, color) {
                    // Cavalo avançado é especialmente vulnerável
                    penalty += piece_value / 2;
                    
                    // CORREÇÃO ESPECIAL: Penalidade massiva para cavalos em e5/d5/c5/f5 sem escape
                    let file = sq % 8;
                    let rank = sq / 8;
                    if rank == 4 && file >= 2 && file <= 5 { // Ranks centrais (e5, d5, etc.)
                        if !has_knight_escape_squares(board, sq, color) {
                            penalty += piece_value; // Penalidade total - cavalos presos são perdidos
                        }
                    }
                }
                
                if !has_support(board, sq, color) {
                    penalty += piece_value / 3; // Cavalo sem suporte
                }
            }

            // Penalidades extras por tipo de peça
            match piece_kind {
                PieceKind::Queen => penalty += 200,  // Rainha em perigo
                PieceKind::Rook => penalty += 100,   // Torre em perigo
                PieceKind::Bishop | PieceKind::Knight => penalty += 50, // Peças menores
                _ => {}
            }
        }
    }

    // CORREÇÃO 2: Penalidades mais severas para peões desprotegidos
    let our_pawns = board.pawns & our_pieces;
    let mut pawn_bb = our_pawns;
    while pawn_bb != 0 {
        let sq = pawn_bb.trailing_zeros() as u8;
        pawn_bb &= pawn_bb - 1;

        if board.is_square_attacked_by(sq, enemy_color) && !board.is_square_attacked_by(sq, color) {
            penalty += 25; // Era 15, agora 25

            // Penalidade extra para peões avançados desprotegidos
            let rank = sq / 8;
            if (color == Color::White && rank >= 4) || (color == Color::Black && rank <= 3) {
                penalty += 15; // Peão avançado desprotegido
            }
        }
    }

    penalty
}

/// Bônus por atacar peças inimigas
fn evaluate_enemy_attacks(board: &Board, color: Color) -> i32 {
    let mut bonus = 0;
    let enemy_color = !color;
    let enemy_pieces = if enemy_color == Color::White { board.white_pieces } else { board.black_pieces };

    // Analisa peças inimigas valiosas que atacamos
    let enemy_valuables = (board.knights | board.bishops | board.rooks | board.queens) & enemy_pieces;
    let mut bb = enemy_valuables;

    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        bb &= bb - 1;

        if board.is_square_attacked_by(sq, color) {
            let piece_kind = board.get_piece_on_square(sq).unwrap();
            let piece_value = MATERIAL_VALUES[piece_kind as usize];

            // Bônus reduzido por atacar peças valiosas
            bonus += piece_value / 10; // Reduzido de /5 para /10

            // Bônus extra se não defendida (muito reduzido)
            if !board.is_square_attacked_by(sq, enemy_color) {
                bonus += piece_value / 8; // Reduzido de /3 para /8

                // Bônus extra por peças valiosas indefesas (reduzido)
                match piece_kind {
                    PieceKind::Queen => bonus += 30,   // Reduzido de 150 para 30
                    PieceKind::Rook => bonus += 20,    // Reduzido de 80 para 20
                    PieceKind::Bishop | PieceKind::Knight => bonus += 10, // Reduzido de 40 para 10
                    _ => {}
                }
            }
        }
    }

    // Bônus por forks de cavalo
    bonus += evaluate_knight_forks(board, color);

    // CORREÇÃO 4: Bônus reduzido por atacar peões (para não superestimar)
    let enemy_pawns = board.pawns & enemy_pieces;
    let mut pawn_bb = enemy_pawns;
    while pawn_bb != 0 {
        let sq = pawn_bb.trailing_zeros() as u8;
        pawn_bb &= pawn_bb - 1;

        if board.is_square_attacked_by(sq, color) {
            bonus += 2; // Era 3, agora 2
        }
    }

    bonus
}

/// CORREÇÃO 5: Avalia bônus por forks de cavalo com verificação de segurança
fn evaluate_knight_forks(board: &Board, color: Color) -> i32 {
    let mut bonus = 0;
    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let enemy_pieces = if color == Color::White { board.black_pieces } else { board.white_pieces };

    // Analisa cada cavalo nosso
    let our_knights = board.knights & our_pieces;
    let mut knight_bb = our_knights;

    while knight_bb != 0 {
        let knight_sq = knight_bb.trailing_zeros() as u8;
        knight_bb &= knight_bb - 1;

        // Obtém ataques do cavalo
        let knight_attacks = crate::moves::knight::get_knight_attacks_lookup(knight_sq);

        // Conta peças inimigas valiosas atacadas
        let valuable_enemies = (board.knights | board.bishops | board.rooks | board.queens) & enemy_pieces;
        let attacked_valuables = knight_attacks & valuable_enemies;

        if attacked_valuables.count_ones() >= 2 {
            // Calcula valor das peças atacadas
            let mut attacked_value = 0;
            let mut temp_bb = attacked_valuables;

            while temp_bb != 0 {
                let sq = temp_bb.trailing_zeros() as u8;
                temp_bb &= temp_bb - 1;

                if let Some(piece_kind) = board.get_piece_on_square(sq) {
                    attacked_value += MATERIAL_VALUES[piece_kind as usize];
                }
            }

            // CORREÇÃO: Verifica se o cavalo sobrevive ao fork
            let knight_survives = !board.is_square_attacked_by(knight_sq, !color) ||
                board.is_square_attacked_by(knight_sq, color);

            // Bônus baseado no valor e segurança
            if attacked_value > 300 {
                let base_bonus = if knight_survives { 15 } else { 8 }; // Reduzido de 40/20 para 15/8
                bonus += base_bonus;
            } else {
                let base_bonus = if knight_survives { 8 } else { 4 }; // Reduzido de 20/10 para 8/4
                bonus += base_bonus;
            }
        }

        // CORREÇÃO: Bônus especial por fork rei + peça com verificação de segurança
        let enemy_king = board.kings & enemy_pieces;
        if knight_attacks & enemy_king != 0 && knight_attacks & valuable_enemies != 0 {
            let knight_survives = !board.is_square_attacked_by(knight_sq, !color) ||
                board.is_square_attacked_by(knight_sq, color);

            let base_bonus = if knight_survives { 25 } else { 12 }; // Reduzido de 80/40 para 25/12
            bonus += base_bonus;
        }
    }

    bonus
}

/// Encontra o valor do menor atacante inimigo
fn find_smallest_attacker_value(board: &Board, target_square: u8, attacker_color: Color) -> i32 {
    let attacker_pieces = if attacker_color == Color::White { board.white_pieces } else { board.black_pieces };

    // Verifica em ordem de valor (menor primeiro)

    // Peões (100)
    if can_piece_type_attack(board, target_square, attacker_color, board.pawns & attacker_pieces) {
        return MATERIAL_VALUES[0]; // Peão = 100
    }

    // Cavalos/Bispos (320/330)
    if can_piece_type_attack(board, target_square, attacker_color, board.knights & attacker_pieces) {
        return MATERIAL_VALUES[1]; // Cavalo = 320
    }

    if can_piece_type_attack(board, target_square, attacker_color, board.bishops & attacker_pieces) {
        return MATERIAL_VALUES[2]; // Bispo = 330
    }

    // Torres (500)
    if can_piece_type_attack(board, target_square, attacker_color, board.rooks & attacker_pieces) {
        return MATERIAL_VALUES[3]; // Torre = 500
    }

    // Rainhas (900)
    if can_piece_type_attack(board, target_square, attacker_color, board.queens & attacker_pieces) {
        return MATERIAL_VALUES[4]; // Rainha = 900
    }

    // Rei (20000 - só em situações extremas)
    if can_piece_type_attack(board, target_square, attacker_color, board.kings & attacker_pieces) {
        return MATERIAL_VALUES[5];
    }

    // Nenhum atacante encontrado (não deveria acontecer se is_square_attacked_by retornou true)
    1000
}

/// Verifica se alguma peça de um tipo pode atacar o alvo
fn can_piece_type_attack(board: &Board, target_square: u8, attacker_color: Color, piece_bb: crate::types::Bitboard) -> bool {
    let mut bb = piece_bb;

    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        bb &= bb - 1;

        // Usa lógica similar do king_safety para verificar ataques
        if piece_can_attack_square(board, sq, target_square, attacker_color, piece_bb) {
            return true;
        }
    }

    false
}

/// Verifica se cavalo está em posição avançada (ranks 5-7 para brancas, 2-4 para pretas)
fn is_advanced_knight(square: u8, color: Color) -> bool {
    let rank = square / 8;
    match color {
        Color::White => rank >= 4, // Ranks 5-8 (0-indexed: 4-7)
        Color::Black => rank <= 3, // Ranks 1-4 (0-indexed: 0-3)
    }
}

/// Verifica se peça tem suporte de peões ou outras peças próximas
fn has_support(board: &Board, square: u8, color: Color) -> bool {
    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };

    // Verifica suporte de peões
    let pawn_support = has_pawn_support(board, square, color);
    if pawn_support {
        return true;
    }

    // Verifica peças adjacentes (cavalos, bispos, torres próximas)
    let adjacent_squares = get_adjacent_squares(square);
    for adj_sq in adjacent_squares {
        if (1u64 << adj_sq) & our_pieces != 0 {
            return true;
        }
    }

    false
}

/// Verifica suporte específico de peões
fn has_pawn_support(board: &Board, square: u8, color: Color) -> bool {
    let our_pawns = board.pawns & if color == Color::White { board.white_pieces } else { board.black_pieces };

    let file = square % 8;
    let rank = square / 8;

    // Posições onde peões podem defender esta casa
    let support_squares = match color {
        Color::White => {
            // Peões brancos defendem de baixo (rank anterior)
            if rank > 0 {
                let mut squares = Vec::new();
                if file > 0 { squares.push((rank - 1) * 8 + file - 1); }
                if file < 7 { squares.push((rank - 1) * 8 + file + 1); }
                squares
            } else {
                Vec::new()
            }
        },
        Color::Black => {
            // Peões pretos defendem de cima (rank posterior)
            if rank < 7 {
                let mut squares = Vec::new();
                if file > 0 { squares.push((rank + 1) * 8 + file - 1); }
                if file < 7 { squares.push((rank + 1) * 8 + file + 1); }
                squares
            } else {
                Vec::new()
            }
        }
    };

    for support_sq in support_squares {
        if (1u64 << support_sq) & our_pawns != 0 {
            return true;
        }
    }

    false
}

/// Obtém casas adjacentes (8 direções) para uma casa
fn get_adjacent_squares(square: u8) -> Vec<u8> {
    let mut adjacent = Vec::new();
    let file = square % 8;
    let rank = square / 8;

    for dr in -1..=1i8 {
        for df in -1..=1i8 {
            if dr == 0 && df == 0 { continue; }

            let new_rank = rank as i8 + dr;
            let new_file = file as i8 + df;

            if new_rank >= 0 && new_rank <= 7 && new_file >= 0 && new_file <= 7 {
                adjacent.push((new_rank as u8) * 8 + (new_file as u8));
            }
        }
    }

    adjacent
}

/// Verifica se uma peça específica pode atacar uma casa (versão simplificada)
fn piece_can_attack_square(board: &Board, piece_square: u8, target_square: u8, color: Color, piece_type_bb: crate::types::Bitboard) -> bool {
    let piece_bb = 1u64 << piece_square;
    let all_pieces = board.white_pieces | board.black_pieces;

    // Determina o tipo baseado no bitboard
    if (piece_type_bb & board.pawns) != 0 {
        // Peão
        pawn_attacks_square(piece_square, target_square, color)
    } else if (piece_type_bb & board.knights) != 0 {
        // Cavalo
        let attacks = crate::moves::knight::get_knight_attacks_lookup(piece_square);
        (attacks & (1u64 << target_square)) != 0
    } else if (piece_type_bb & board.bishops) != 0 {
        // Bispo
        let attacks = crate::moves::sliding::get_bishop_attacks(piece_square, all_pieces);
        (attacks & (1u64 << target_square)) != 0
    } else if (piece_type_bb & board.rooks) != 0 {
        // Torre
        let attacks = crate::moves::sliding::get_rook_attacks(piece_square, all_pieces);
        (attacks & (1u64 << target_square)) != 0
    } else if (piece_type_bb & board.queens) != 0 {
        // Rainha (bispo + torre)
        let bishop_attacks = crate::moves::sliding::get_bishop_attacks(piece_square, all_pieces);
        let rook_attacks = crate::moves::sliding::get_rook_attacks(piece_square, all_pieces);
        ((bishop_attacks | rook_attacks) & (1u64 << target_square)) != 0
    } else if (piece_type_bb & board.kings) != 0 {
        // Rei
        let attacks = crate::moves::king::get_king_attacks_lookup(piece_square);
        (attacks & (1u64 << target_square)) != 0
    } else {
        false
    }
}

/// Verifica se peão pode atacar casa específica
fn pawn_attacks_square(pawn_square: u8, target_square: u8, pawn_color: Color) -> bool {
    if pawn_color == Color::White {
        // Brancas: ataques diagonais para cima
        let left_attack = if pawn_square % 8 > 0 && pawn_square + 7 < 64 { Some(pawn_square + 7) } else { None };
        let right_attack = if pawn_square % 8 < 7 && pawn_square + 9 < 64 { Some(pawn_square + 9) } else { None };
        [left_attack, right_attack].iter().any(|&attack| attack == Some(target_square))
    } else {
        // Pretas: ataques diagonais para baixo
        let left_attack = if pawn_square % 8 > 0 && pawn_square >= 9 { Some(pawn_square - 9) } else { None };
        let right_attack = if pawn_square % 8 < 7 && pawn_square >= 7 { Some(pawn_square - 7) } else { None };
        [left_attack, right_attack].iter().any(|&attack| attack == Some(target_square))
    }
}

/// Avalia pins e ataques descobertos (versão básica)
fn evaluate_pins_and_discoveries(board: &Board, color: Color) -> i32 {
    // Esta é uma implementação básica - pode ser expandida
    let mut bonus = 0;

    // Procura por pins básicos (peça inimiga "pinada" ao rei)
    let enemy_color = !color;
    let enemy_king_bb = board.kings & if enemy_color == Color::White { board.white_pieces } else { board.black_pieces };

    if enemy_king_bb != 0 {
        let enemy_king_square = enemy_king_bb.trailing_zeros() as u8;

        // Verifica se temos torres/rainhas/bispos que podem "pin"
        let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
        let our_sliders = (board.rooks | board.queens | board.bishops) & our_pieces;

        let mut slider_bb = our_sliders;
        while slider_bb != 0 {
            let slider_sq = slider_bb.trailing_zeros() as u8;
            slider_bb &= slider_bb - 1;

            // Verifica se está na mesma linha/diagonal que o rei inimigo
            if can_pin_king(slider_sq, enemy_king_square, board) {
                bonus += 30; // Era 25, agora 30
            }
        }
    }

    bonus
}

/// Verifica se uma peça pode potencialmente fazer pin no rei
fn can_pin_king(slider_square: u8, king_square: u8, board: &Board) -> bool {
    // Verifica se estão na mesma linha, coluna ou diagonal
    let slider_rank = slider_square / 8;
    let slider_file = slider_square % 8;
    let king_rank = king_square / 8;
    let king_file = king_square % 8;

    let same_rank = slider_rank == king_rank;
    let same_file = slider_file == king_file;
    let same_diagonal = (slider_rank as i8 - king_rank as i8).abs() == (slider_file as i8 - king_file as i8).abs();

    // Verifica se a peça pode se mover nessa direção (torre vs bispo)
    let piece_bb = 1u64 << slider_square;
    let is_rook_like = (board.rooks | board.queens) & piece_bb != 0;
    let is_bishop_like = (board.bishops | board.queens) & piece_bb != 0;

    (same_rank || same_file) && is_rook_like || same_diagonal && is_bishop_like
}

/// NOVA FUNÇÃO: Avalia nível de ameaça para uma casa específica (detecta ameaças de 2-3 tempos)
fn evaluate_square_threat_level(board: &Board, target_square: u8, attacking_color: Color) -> i32 {
    let mut threat_level = 0;

    // 1. Verifica ataque direto (1 tempo)
    if board.is_square_attacked_by(target_square, attacking_color) {
        threat_level = 4; // Ameaça máxima - ataque direto
    }

    // 2. Verifica ameaças de 2 tempos (peças que podem atacar em 1 movimento)
    threat_level = threat_level.max(check_two_move_threats(board, target_square, attacking_color));

    // 3. Verifica ameaças de 3 tempos (desenvolvimento + ataque)
    threat_level = threat_level.max(check_three_move_threats(board, target_square, attacking_color));

    threat_level
}

/// Verifica ameaças que podem ser executadas em 2 movimentos
fn check_two_move_threats(board: &Board, target_square: u8, attacking_color: Color) -> i32 {
    let mut max_threat = 0;
    let attacking_pieces = if attacking_color == Color::White { board.white_pieces } else { board.black_pieces };

    // Verifica peões que podem avançar e atacar
    let enemy_pawns = board.pawns & attacking_pieces;
    max_threat = max_threat.max(check_pawn_advance_threats(board, target_square, enemy_pawns, attacking_color));

    // Verifica peças que podem se mover para atacar diretamente
    max_threat = max_threat.max(check_piece_development_threats(board, target_square, attacking_color));

    max_threat
}

/// Verifica ameaças de desenvolvimento (3 movimentos)
fn check_three_move_threats(board: &Board, target_square: u8, attacking_color: Color) -> i32 {
    let attacking_pieces = if attacking_color == Color::White { board.white_pieces } else { board.black_pieces };
    
    // Verifica se há peças não desenvolvidas que podem eventualmente atacar
    let undeveloped_pieces = get_undeveloped_pieces(board, attacking_color);
    
    if undeveloped_pieces > 2 {
        // Muitas peças não desenvolvidas = ameaça futura
        return 1; 
    }
    
    0
}

/// Verifica ameaças de peões que podem avançar
fn check_pawn_advance_threats(board: &Board, target_square: u8, enemy_pawns: crate::types::Bitboard, attacking_color: Color) -> i32 {
    let mut max_threat = 0;
    let target_file = target_square % 8;
    let target_rank = target_square / 8;
    
    let mut pawn_bb = enemy_pawns;
    while pawn_bb != 0 {
        let pawn_sq = pawn_bb.trailing_zeros() as u8;
        pawn_bb &= pawn_bb - 1;
        
        let pawn_file = pawn_sq % 8;
        let pawn_rank = pawn_sq / 8;
        let file_diff = (target_file as i8 - pawn_file as i8).abs();
        
        // Verifica se peão pode eventualmente atacar o alvo
        match attacking_color {
            Color::Black => {
                // Peões pretos se movem para baixo
                if pawn_rank > target_rank && file_diff == 1 {
                    let moves_needed = pawn_rank - target_rank;
                    if moves_needed <= 2 {
                        max_threat = max_threat.max(3 - moves_needed as i32); // Mais próximo = maior ameaça
                    }
                }
            },
            Color::White => {
                // Peões brancos se movem para cima
                if pawn_rank < target_rank && file_diff == 1 {
                    let moves_needed = target_rank - pawn_rank;
                    if moves_needed <= 2 {
                        max_threat = max_threat.max(3 - moves_needed as i32);
                    }
                }
            }
        }
    }
    
    max_threat
}

/// Verifica ameaças de desenvolvimento de peças
fn check_piece_development_threats(board: &Board, target_square: u8, attacking_color: Color) -> i32 {
    let attacking_pieces = if attacking_color == Color::White { board.white_pieces } else { board.black_pieces };
    let all_pieces = board.white_pieces | board.black_pieces;
    
    // Verifica bispos que podem desenvolver e atacar
    let bishops = board.bishops & attacking_pieces;
    let mut bishop_bb = bishops;
    while bishop_bb != 0 {
        let bishop_sq = bishop_bb.trailing_zeros() as u8;
        bishop_bb &= bishop_bb - 1;
        
        // Verifica se bispo pode eventualmente atacar através de desenvolvimento
        if can_bishop_eventually_attack(bishop_sq, target_square, all_pieces) {
            return 2; // Ameaça de 2 tempos
        }
    }
    
    // Verifica cavalos
    let knights = board.knights & attacking_pieces;
    let mut knight_bb = knights;
    while knight_bb != 0 {
        let knight_sq = knight_bb.trailing_zeros() as u8;
        knight_bb &= knight_bb - 1;
        
        if can_knight_eventually_attack(knight_sq, target_square) {
            return 2;
        }
    }
    
    0
}

/// Verifica se bispo pode eventualmente atacar uma casa
fn can_bishop_eventually_attack(bishop_sq: u8, target_sq: u8, all_pieces: crate::types::Bitboard) -> bool {
    let bishop_file = bishop_sq % 8;
    let bishop_rank = bishop_sq / 8;
    let target_file = target_sq % 8;
    let target_rank = target_sq / 8;
    
    // Verifica se estão na mesma diagonal
    let rank_diff = (target_rank as i8 - bishop_rank as i8).abs();
    let file_diff = (target_file as i8 - bishop_file as i8).abs();
    
    if rank_diff == file_diff {
        // Mesma diagonal - verifica se há obstruções removíveis
        let obstructions = count_pieces_between_diagonal(bishop_sq, target_sq, all_pieces);
        return obstructions <= 2; // Pode ser desenvolvido em 1-2 movimentos
    }
    
    false
}

/// Verifica se cavalo pode eventualmente atacar
fn can_knight_eventually_attack(knight_sq: u8, target_sq: u8) -> bool {
    // Cavalo pode atacar qualquer casa em no máximo 3 movimentos
    let knight_file = knight_sq % 8;
    let knight_rank = knight_sq / 8;
    let target_file = target_sq % 8;
    let target_rank = target_sq / 8;
    
    let file_diff = (target_file as i8 - knight_file as i8).abs();
    let rank_diff = (target_rank as i8 - knight_rank as i8).abs();
    
    // Heurística: se está relativamente próximo, pode ser uma ameaça
    file_diff + rank_diff <= 4
}

/// Conta peças entre duas casas na diagonal
fn count_pieces_between_diagonal(from: u8, to: u8, all_pieces: crate::types::Bitboard) -> i32 {
    let from_file = from % 8;
    let from_rank = from / 8;
    let to_file = to % 8;
    let to_rank = to / 8;
    
    let file_dir = if to_file > from_file { 1 } else { -1 };
    let rank_dir = if to_rank > from_rank { 1 } else { -1 };
    
    let mut count = 0;
    let mut current_file = from_file as i8 + file_dir;
    let mut current_rank = from_rank as i8 + rank_dir;
    
    while current_file != to_file as i8 && current_rank != to_rank as i8 {
        let sq = (current_rank * 8 + current_file) as u8;
        if (all_pieces & (1u64 << sq)) != 0 {
            count += 1;
        }
        current_file += file_dir;
        current_rank += rank_dir;
    }
    
    count
}

/// Conta peças não desenvolvidas
fn get_undeveloped_pieces(board: &Board, color: Color) -> i32 {
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let back_rank = if color == Color::White { 0xFF } else { 0xFF00000000000000 };
    
    let undeveloped_knights = (board.knights & pieces & back_rank).count_ones() as i32;
    let undeveloped_bishops = (board.bishops & pieces & back_rank).count_ones() as i32;
    
    undeveloped_knights + undeveloped_bishops
}

/// NOVA FUNÇÃO: Verifica se cavalo tem casas de escape
fn has_knight_escape_squares(board: &Board, knight_sq: u8, color: Color) -> bool {
    let knight_attacks = crate::moves::knight::get_knight_attacks_lookup(knight_sq);
    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let enemy_color = !color;
    
    // Verifica cada casa que o cavalo pode ir
    let mut escape_bb = knight_attacks;
    while escape_bb != 0 {
        let escape_sq = escape_bb.trailing_zeros() as u8;
        escape_bb &= escape_bb - 1;
        
        // Verifica se a casa está livre ou pode capturar
        if (our_pieces & (1u64 << escape_sq)) == 0 {
            // Casa não ocupada por nós - verifica se é segura
            if !board.is_square_attacked_by(escape_sq, enemy_color) {
                return true; // Encontrou casa de escape segura
            }
        }
    }
    
    false // Nenhuma casa de escape segura
}
