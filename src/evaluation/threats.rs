// Avaliação de ameaças - peças penduradas, ataques táticos
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
    
    score.clamp(-500, 500) // Limita para estabilidade
}

/// Penaliza peças próprias atacadas (peças penduradas)
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
        
        if board.is_square_attacked_by(sq, enemy_color) {
            let piece_kind = board.get_piece_on_square(sq).unwrap();
            let piece_value = MATERIAL_VALUES[piece_kind as usize];
            
            // Determina valor do menor atacante inimigo
            let min_attacker_value = find_smallest_attacker_value(board, sq, enemy_color);
            
            if min_attacker_value < piece_value {
                // Peça pode ser capturada por menor valor
                let vulnerability = piece_value - min_attacker_value;
                
                // Verifica se a peça está defendida
                if board.is_square_attacked_by(sq, color) {
                    // Defendida - penalidade maior (era /4, agora /2)
                    penalty += vulnerability / 2;
                } else {
                    // Desprotegida - penalidade triplicada (era /2, agora full * 1.5)
                    let base_penalty = (vulnerability as f32 * 1.5) as i32;
                    penalty += base_penalty;
                    
                    // Penalidade extra para rainha pendurada
                    if piece_kind == PieceKind::Queen {
                        penalty += 300;
                    }
                    
                    // Penalidade especial para cavalos avançados sem suporte
                    if piece_kind == PieceKind::Knight {
                        if is_advanced_knight(sq, color) && !has_support(board, sq, color) {
                            penalty += vulnerability / 2 + 50;
                        }
                    }
                }
            }
        }
    }
    
    // Verifica peões atacados também (menor prioridade)
    let our_pawns = board.pawns & our_pieces;
    let mut pawn_bb = our_pawns;
    while pawn_bb != 0 {
        let sq = pawn_bb.trailing_zeros() as u8;
        pawn_bb &= pawn_bb - 1;
        
        if board.is_square_attacked_by(sq, enemy_color) && !board.is_square_attacked_by(sq, color) {
            penalty += 15; // Peão desprotegido
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
            
            // Bônus baseado no valor da peça atacada
            bonus += piece_value / 6; // Ex: cavalo atacado = +50
            
            // Bônus extra se não defendida
            if !board.is_square_attacked_by(sq, enemy_color) {
                bonus += piece_value / 4; // Peça indefesa = bônus maior
            }
        }
    }
    
    // NOVO: Bônus por forks de cavalo
    bonus += evaluate_knight_forks(board, color);
    
    // Pequeno bônus por atacar peões inimigos (reduzido de 5 para 3)
    let enemy_pawns = board.pawns & enemy_pieces;
    let mut pawn_bb = enemy_pawns;
    while pawn_bb != 0 {
        let sq = pawn_bb.trailing_zeros() as u8;
        pawn_bb &= pawn_bb - 1;
        
        if board.is_square_attacked_by(sq, color) {
            bonus += 3; // Reduzido de 5 para 3
        }
    }
    
    bonus
}

/// Avalia bônus por forks de cavalo (atacar duas peças valiosas simultaneamente)
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
            
            // Bônus significativo por fork em peças valiosas
            if attacked_value > 300 {
                bonus += 30; // Fork tático valioso
            } else {
                bonus += 15; // Fork menor
            }
        }
        
        // Bônus especial por fork rei + peça
        let enemy_king = board.kings & enemy_pieces;
        if knight_attacks & enemy_king != 0 && knight_attacks & valuable_enemies != 0 {
            bonus += 50; // Fork real é muito valioso
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
                bonus += 25; // Bônus por possível pin
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