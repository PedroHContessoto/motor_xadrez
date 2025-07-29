// Avaliação de ameaças - peças penduradas, ataques táticos
use crate::{board::Board, types::{Color, PieceKind, Move}};
use super::material::MATERIAL_VALUES;

/// Avalia ameaças mútuas entre as cores - Estrutura modular aprimorada
pub fn evaluate_threats(board: &Board, color: Color) -> i32 {
    let mut score = 0;

    // 1. Penalidades por peças penduradas
    score -= evaluate_hanging_pieces(board, color);

    // 2. Bônus por atacar peças inimigas
    score += evaluate_enemy_attacks(board, color);

    // 3. Forks táticos (cavalos, rainhas, bispos)
    score += evaluate_knight_forks(board, color);
    score += evaluate_queen_forks(board, color);
    score += evaluate_bishop_forks(board, color);

    // 4. Pinos reais com peça intermediária detectada
    score += evaluate_real_pins(board, color);

    // 5. Skewers (alinhamento inverso)
    score += evaluate_skewers(board, color);

    // 6. Overloads (defensores sobrecarregados)
    score += evaluate_overloads(board, color);

    // 7. X-ray threats / removal of guard (ameaças descobertas)
    score += evaluate_xray_threats(board, color);

    // 8. Compound threats (múltiplas ameaças simultâneas)
    score += evaluate_compound_threats(board, color);

    score.clamp(-600, 600) // Aumentado para acomodar novas funcionalidades
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

    // Forks são avaliados na função principal - removido daqui para evitar duplicação

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
            // Verifica se o fork é viável com SEE antes de aplicar bônus
            let mut viable_targets = 0;
            let mut attacked_value = 0;
            let mut temp_bb = attacked_valuables;

            while temp_bb != 0 {
                let sq = temp_bb.trailing_zeros() as u8;
                temp_bb &= temp_bb - 1;

                // Aplica filtro SEE para verificar se o ataque é viável
                if is_attack_viable_see(board, knight_sq, sq) {
                    viable_targets += 1;
                    if let Some(piece_kind) = board.get_piece_on_square(sq) {
                        attacked_value += MATERIAL_VALUES[piece_kind as usize];
                    }
                }
            }

            // Fork só é válido se pelo menos 2 ataques passam no SEE
            if viable_targets >= 2 {
                // Bônus significativo por fork em peças valiosas
                if attacked_value > 300 {
                    bonus += 30; // Fork tático valioso
                } else {
                    bonus += 15; // Fork menor
                }
            }
        }

        // Bônus especial por fork rei + peça (rei sempre é alvo válido)
        let enemy_king = board.kings & enemy_pieces;
        if knight_attacks & enemy_king != 0 && knight_attacks & valuable_enemies != 0 {
            // Verifica se pelo menos uma peça valiosa passa no SEE
            let mut viable_royal_fork = false;
            let mut temp_bb = knight_attacks & valuable_enemies;
            while temp_bb != 0 {
                let sq = temp_bb.trailing_zeros() as u8;
                temp_bb &= temp_bb - 1;
                if is_attack_viable_see(board, knight_sq, sq) {
                    viable_royal_fork = true;
                    break;
                }
            }

            if viable_royal_fork {
                bonus += 25; // Fork real (reduzido de 50 -> 25)
            }
        }
    }

    bonus
}

/// Avalia bônus por forks de rainha (atacar duas peças valiosas simultaneamente)
fn evaluate_queen_forks(board: &Board, color: Color) -> i32 {
    let mut bonus = 0;
    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let enemy_pieces = if color == Color::White { board.black_pieces } else { board.white_pieces };

    // Analisa cada rainha nossa
    let our_queens = board.queens & our_pieces;
    let mut queen_bb = our_queens;

    while queen_bb != 0 {
        let queen_sq = queen_bb.trailing_zeros() as u8;
        queen_bb &= queen_bb - 1;

        // Obtém ataques da rainha usando magic bitboards
        let all_pieces = board.white_pieces | board.black_pieces;
        let queen_attacks = crate::moves::magic_bitboards::get_queen_attacks_magic(queen_sq, all_pieces);

        // Conta peças inimigas valiosas atacadas
        let valuable_enemies = (board.knights | board.bishops | board.rooks | board.queens) & enemy_pieces;
        let attacked_valuables = queen_attacks & valuable_enemies;

        if attacked_valuables.count_ones() >= 2 {
            // Verifica se o fork é viável com SEE antes de aplicar bônus
            let mut viable_targets = 0;
            let mut attacked_value = 0;
            let mut temp_bb = attacked_valuables;

            while temp_bb != 0 {
                let sq = temp_bb.trailing_zeros() as u8;
                temp_bb &= temp_bb - 1;

                // Aplica filtro SEE para verificar se o ataque é viável
                if is_attack_viable_see(board, queen_sq, sq) {
                    viable_targets += 1;
                    if let Some(piece_kind) = board.get_piece_on_square(sq) {
                        attacked_value += MATERIAL_VALUES[piece_kind as usize];
                    }
                }
            }

            // Fork só é válido se pelo menos 2 ataques passam no SEE
            if viable_targets >= 2 {
                // Bônus significativo por fork em peças valiosas
                if attacked_value > 600 {
                    bonus += 40; // Fork de rainha muito valioso
                } else if attacked_value > 300 {
                    bonus += 25; // Fork de rainha moderado
                }
            }
        }

        // Bônus especial por fork rei + peça com rainha (rei sempre é alvo válido)
        let enemy_king = board.kings & enemy_pieces;
        if queen_attacks & enemy_king != 0 && queen_attacks & valuable_enemies != 0 {
            // Verifica se pelo menos uma peça valiosa passa no SEE
            let mut viable_royal_fork = false;
            let mut temp_bb = queen_attacks & valuable_enemies;
            while temp_bb != 0 {
                let sq = temp_bb.trailing_zeros() as u8;
                temp_bb &= temp_bb - 1;
                if is_attack_viable_see(board, queen_sq, sq) {
                    viable_royal_fork = true;
                    break;
                }
            }

            if viable_royal_fork {
                bonus += 35; // Fork real com rainha (reduzido de 70 -> 35)
            }
        }
    }

    bonus
}

/// Avalia bônus por forks de bispo (atacar duas peças valiosas simultaneamente)
fn evaluate_bishop_forks(board: &Board, color: Color) -> i32 {
    let mut bonus = 0;
    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let enemy_pieces = if color == Color::White { board.black_pieces } else { board.white_pieces };

    // Analisa cada bispo nosso
    let our_bishops = board.bishops & our_pieces;
    let mut bishop_bb = our_bishops;

    while bishop_bb != 0 {
        let bishop_sq = bishop_bb.trailing_zeros() as u8;
        bishop_bb &= bishop_bb - 1;

        // Obtém ataques do bispo usando magic bitboards
        let all_pieces = board.white_pieces | board.black_pieces;
        let bishop_attacks = crate::moves::magic_bitboards::get_bishop_attacks_magic(bishop_sq, all_pieces);

        // Conta peças inimigas valiosas atacadas
        let valuable_enemies = (board.knights | board.bishops | board.rooks | board.queens) & enemy_pieces;
        let attacked_valuables = bishop_attacks & valuable_enemies;

        if attacked_valuables.count_ones() >= 2 {
            // Verifica se o fork é viável com SEE antes de aplicar bônus
            let mut viable_targets = 0;
            let mut attacked_value = 0;
            let mut temp_bb = attacked_valuables;

            while temp_bb != 0 {
                let sq = temp_bb.trailing_zeros() as u8;
                temp_bb &= temp_bb - 1;

                // Aplica filtro SEE para verificar se o ataque é viável
                if is_attack_viable_see(board, bishop_sq, sq) {
                    viable_targets += 1;
                    if let Some(piece_kind) = board.get_piece_on_square(sq) {
                        attacked_value += MATERIAL_VALUES[piece_kind as usize];
                    }
                }
            }

            // Fork só é válido se pelo menos 2 ataques passam no SEE
            if viable_targets >= 2 {
                // Bônus por fork em peças valiosas
                if attacked_value > 600 {
                    bonus += 25; // Fork de bispo muito valioso
                } else if attacked_value > 300 {
                    bonus += 15; // Fork de bispo moderado
                }
            }
        }

        // Bônus especial por fork rei + peça com bispo (rei sempre é alvo válido)
        let enemy_king = board.kings & enemy_pieces;
        if bishop_attacks & enemy_king != 0 && bishop_attacks & valuable_enemies != 0 {
            // Verifica se pelo menos uma peça valiosa passa no SEE
            let mut viable_royal_fork = false;
            let mut temp_bb = bishop_attacks & valuable_enemies;
            while temp_bb != 0 {
                let sq = temp_bb.trailing_zeros() as u8;
                temp_bb &= temp_bb - 1;
                if is_attack_viable_see(board, bishop_sq, sq) {
                    viable_royal_fork = true;
                    break;
                }
            }

            if viable_royal_fork {
                bonus += 45; // Fork real com bispo é muito bom
            }
        }

        // Bônus específico por fork de torres (bispos são especializados nisso)
        let enemy_rooks = board.rooks & enemy_pieces;
        let attacked_rooks = bishop_attacks & enemy_rooks;
        if attacked_rooks.count_ones() >= 2 {
            // Verifica se pelo menos 2 torres passam no SEE
            let mut viable_rook_attacks = 0;
            let mut temp_bb = attacked_rooks;
            while temp_bb != 0 {
                let sq = temp_bb.trailing_zeros() as u8;
                temp_bb &= temp_bb - 1;
                if is_attack_viable_see(board, bishop_sq, sq) {
                    viable_rook_attacks += 1;
                }
            }

            if viable_rook_attacks >= 2 {
                bonus += 35; // Fork duplo de torres com bispo
            }
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
        super::utils::can_pawn_attack_square(piece_square, target_square, color)
    } else if (piece_type_bb & board.knights) != 0 {
        // Cavalo
        let attacks = crate::moves::knight::get_knight_attacks_lookup(piece_square);
        (attacks & (1u64 << target_square)) != 0
    } else if (piece_type_bb & board.bishops) != 0 {
        // Bispo - usando magic bitboards
        let attacks = crate::moves::magic_bitboards::get_bishop_attacks_magic(piece_square, all_pieces);
        (attacks & (1u64 << target_square)) != 0
    } else if (piece_type_bb & board.rooks) != 0 {
        // Torre - usando magic bitboards
        let attacks = crate::moves::magic_bitboards::get_rook_attacks_magic(piece_square, all_pieces);
        (attacks & (1u64 << target_square)) != 0
    } else if (piece_type_bb & board.queens) != 0 {
        // Rainha - usando magic bitboards
        let attacks = crate::moves::magic_bitboards::get_queen_attacks_magic(piece_square, all_pieces);
        (attacks & (1u64 << target_square)) != 0
    } else if (piece_type_bb & board.kings) != 0 {
        // Rei
        let attacks = crate::moves::king::get_king_attacks_lookup(piece_square);
        (attacks & (1u64 << target_square)) != 0
    } else {
        false
    }
}


// Função evaluate_pins_and_discoveries removida - substituída por funções específicas na função principal

/// Avalia pinos reais com detecção de peça intermediária
fn evaluate_real_pins(board: &Board, color: Color) -> i32 {
    let mut bonus = 0;
    let enemy_color = !color;
    let enemy_king_bb = board.kings & if enemy_color == Color::White { board.white_pieces } else { board.black_pieces };

    if enemy_king_bb == 0 { return 0; }
    let enemy_king_sq = enemy_king_bb.trailing_zeros() as u8;

    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let enemy_pieces = if enemy_color == Color::White { board.white_pieces } else { board.black_pieces };
    let our_sliders = (board.rooks | board.queens | board.bishops) & our_pieces;

    let mut slider_bb = our_sliders;
    while slider_bb != 0 {
        let slider_sq = slider_bb.trailing_zeros() as u8;
        slider_bb &= slider_bb - 1;

        // Verificar se pode formar linha com o rei inimigo
        if let Some(direction) = get_pin_direction(slider_sq, enemy_king_sq) {
            let pinned_piece = find_pinned_piece(board, slider_sq, enemy_king_sq, direction, enemy_pieces);

            if let Some((pinned_sq, piece_kind)) = pinned_piece {
                // Pin real detectado!
                let piece_value = MATERIAL_VALUES[piece_kind as usize];

                // Bônus baseado no valor da peça pinada (reduzido 50%)
                let pin_bonus = match piece_value {
                    v if v >= 900 => 40,  // Rainha pinada (era 80 -> 40)
                    v if v >= 500 => 25,  // Torre pinada (era 50 -> 25)
                    v if v >= 300 => 15,  // Cavalo/Bispo pinado (era 30 -> 15)
                    _ => 8                // Peão pinado (era 15 -> 8)
                };

                bonus += pin_bonus;

                // Bônus extra se a peça pinada não pode se mover sem expor o rei
                if !can_pinned_piece_move_safely(board, pinned_sq, enemy_king_sq, direction) {
                    bonus += pin_bonus / 2; // +50% se totalmente imobilizada
                }
            }
        }
    }

    bonus
}

/// Avalia skewers (alinhamento inverso: peça valiosa -> peça menor)
fn evaluate_skewers(board: &Board, color: Color) -> i32 {
    let mut bonus = 0;
    let enemy_color = !color;
    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let enemy_pieces = if enemy_color == Color::White { board.white_pieces } else { board.black_pieces };
    let our_sliders = (board.rooks | board.queens | board.bishops) & our_pieces;

    let mut slider_bb = our_sliders;
    while slider_bb != 0 {
        let slider_sq = slider_bb.trailing_zeros() as u8;
        slider_bb &= slider_bb - 1;

        // Procura por skewers em todas as direções
        for direction in &[(0, 1), (1, 0), (1, 1), (1, -1), (0, -1), (-1, 0), (-1, -1), (-1, 1)] {
            if let Some((front_piece_sq, front_value, back_piece_sq, back_value)) =
                find_skewer_targets(board, slider_sq, *direction, enemy_pieces) {

                // Skewer válido: peça valiosa na frente, menor atrás
                if front_value > back_value && front_value >= 500 { // Só skewers valiosos
                    let skewer_bonus = match front_value {
                        v if v >= 900 => 60,  // Rainha na frente
                        v if v >= 500 => 40,  // Torre na frente
                        _ => 20               // Outras peças
                    };

                    bonus += skewer_bonus;

                    // Bônus extra se é o rei na frente (skewer absoluto)
                    if front_value >= 20000 {
                        bonus += 40; // Skewer absoluto
                    }
                }
            }
        }
    }

    bonus
}

/// Determina direção do pin entre slider e rei
fn get_pin_direction(slider_sq: u8, king_sq: u8) -> Option<(i8, i8)> {
    let slider_rank = slider_sq / 8;
    let slider_file = slider_sq % 8;
    let king_rank = king_sq / 8;
    let king_file = king_sq % 8;

    let rank_diff = king_rank as i8 - slider_rank as i8;
    let file_diff = king_file as i8 - slider_file as i8;

    // Verifica se estão alinhados
    if rank_diff == 0 && file_diff != 0 {
        // Mesma linha (horizontal)
        Some((0, file_diff.signum()))
    } else if file_diff == 0 && rank_diff != 0 {
        // Mesma coluna (vertical)
        Some((rank_diff.signum(), 0))
    } else if rank_diff.abs() == file_diff.abs() && rank_diff != 0 {
        // Mesma diagonal
        Some((rank_diff.signum(), file_diff.signum()))
    } else {
        None
    }
}

/// Encontra peça pinada entre slider e rei
fn find_pinned_piece(board: &Board, slider_sq: u8, king_sq: u8, direction: (i8, i8), enemy_pieces: crate::types::Bitboard) -> Option<(u8, PieceKind)> {
    let mut current_sq = slider_sq;
    let mut pieces_found = 0;
    let mut pinned_piece: Option<(u8, PieceKind)> = None;

    loop {
        // Move na direção
        let new_rank = (current_sq / 8) as i8 + direction.0;
        let new_file = (current_sq % 8) as i8 + direction.1;

        if new_rank < 0 || new_rank > 7 || new_file < 0 || new_file > 7 {
            break;
        }

        current_sq = (new_rank as u8) * 8 + (new_file as u8);

        if current_sq == king_sq {
            // Chegamos ao rei - pin válido apenas se encontramos exatamente 1 peça
            return if pieces_found == 1 { pinned_piece } else { None };
        }

        // Verifica se há peça nesta casa
        let sq_bb = 1u64 << current_sq;
        if (board.white_pieces | board.black_pieces) & sq_bb != 0 {
            pieces_found += 1;

            if pieces_found == 1 && enemy_pieces & sq_bb != 0 {
                // Primeira peça encontrada e é inimiga - candidata a pinada
                if let Some(piece_kind) = board.get_piece_on_square(current_sq) {
                    pinned_piece = Some((current_sq, piece_kind));
                }
            } else if pieces_found > 1 {
                // Mais de uma peça no caminho - não é pin
                break;
            }
        }
    }

    None
}

/// Encontra alvos de skewer na direção especificada
fn find_skewer_targets(board: &Board, slider_sq: u8, direction: (i8, i8), enemy_pieces: crate::types::Bitboard) -> Option<(u8, i32, u8, i32)> {
    let mut current_sq = slider_sq;
    let mut first_piece: Option<(u8, i32)> = None;

    loop {
        // Move na direção
        let new_rank = (current_sq / 8) as i8 + direction.0;
        let new_file = (current_sq % 8) as i8 + direction.1;

        if new_rank < 0 || new_rank > 7 || new_file < 0 || new_file > 7 {
            break;
        }

        current_sq = (new_rank as u8) * 8 + (new_file as u8);

        // Verifica se há peça nesta casa
        let sq_bb = 1u64 << current_sq;
        if (board.white_pieces | board.black_pieces) & sq_bb != 0 {
            if enemy_pieces & sq_bb != 0 {
                // Peça inimiga encontrada
                if let Some(piece_kind) = board.get_piece_on_square(current_sq) {
                    let piece_value = MATERIAL_VALUES[piece_kind as usize];

                    if first_piece.is_none() {
                        // Primeira peça inimiga
                        first_piece = Some((current_sq, piece_value));
                    } else {
                        // Segunda peça inimiga - possível skewer
                        let (front_sq, front_value) = first_piece.unwrap();
                        return Some((front_sq, front_value, current_sq, piece_value));
                    }
                }
            } else {
                // Peça nossa bloqueia o caminho
                break;
            }
        }
    }

    None
}

/// Verifica se peça pinada pode se mover sem expor o rei
fn can_pinned_piece_move_safely(board: &Board, pinned_sq: u8, king_sq: u8, pin_direction: (i8, i8)) -> bool {
    // Simplificado: se a peça pode se mover na direção do pin, não está totalmente imobilizada
    // Uma implementação mais complexa testaria todos os movimentos legais

    let piece_kind = board.get_piece_on_square(pinned_sq);
    match piece_kind {
        Some(PieceKind::Bishop) => {
            // Bispo pode se mover na diagonal do pin
            pin_direction.0.abs() == pin_direction.1.abs()
        },
        Some(PieceKind::Rook) => {
            // Torre pode se mover na linha/coluna do pin
            pin_direction.0 == 0 || pin_direction.1 == 0
        },
        Some(PieceKind::Queen) => {
            // Rainha sempre pode se mover na direção do pin
            true
        },
        _ => {
            // Cavalos e peões geralmente ficam totalmente imobilizados
            false
        }
    }
}

/// Avalia overloads (defensores sobrecarregados)
fn evaluate_overloads(board: &Board, color: Color) -> i32 {
    let mut bonus = 0;
    let enemy_color = !color;
    let enemy_pieces = if enemy_color == Color::White { board.white_pieces } else { board.black_pieces };

    // Mapa de defensores: cada entrada [piece_sq] = Vec<defender_sq>
    let mut defended_by: std::collections::HashMap<u8, Vec<u8>> = std::collections::HashMap::new();

    // Primeiro, mapeia todos os defensores inimigos
    let mut defender_bb = enemy_pieces;
    while defender_bb != 0 {
        let defender_sq = defender_bb.trailing_zeros() as u8;
        defender_bb &= defender_bb - 1;

        // Encontra todas as peças que este defensor protege
        let defended_squares = find_defended_squares(board, defender_sq, enemy_color);

        for defended_sq in defended_squares {
            defended_by.entry(defended_sq).or_insert_with(Vec::new).push(defender_sq);
        }
    }

    // Agora identifica defensores sobrecarregados
    let mut overloaded_defenders: std::collections::HashSet<u8> = std::collections::HashSet::new();

    for (_defended_sq, defenders) in &defended_by {
        for &defender_sq in defenders {
            // Conta quantas peças este defensor está protegendo
            let defense_count = defended_by.values()
                .filter(|defender_list| defender_list.contains(&defender_sq))
                .count();

            if defense_count >= 2 {
                overloaded_defenders.insert(defender_sq);
            }
        }
    }

    // Calcula bônus baseado nos defensores sobrecarregados
    for &overloaded_sq in &overloaded_defenders {
        // Conta quantas peças valiosas ele defende
        let mut valuable_defenses = 0;
        let mut total_defended_value = 0;

        for (defended_sq, defenders) in &defended_by {
            if defenders.contains(&overloaded_sq) {
                if let Some(piece_kind) = board.get_piece_on_square(*defended_sq) {
                    let piece_value = MATERIAL_VALUES[piece_kind as usize];
                    if piece_value >= 300 { // Só peças valiosas
                        valuable_defenses += 1;
                        total_defended_value += piece_value;
                    }
                }
            }
        }

        // Bônus proporcional ao overload
        if valuable_defenses >= 2 {
            let overload_bonus = match valuable_defenses {
                2 => 25,  // Defende 2 peças valiosas
                3 => 40,  // Defende 3 peças valiosas
                _ => 60,  // Defende 4+ peças valiosas
            };

            bonus += overload_bonus;

            // Bônus extra se defendemos múltiplas peças que ele protege
            let our_attacks_on_defended = count_our_attacks_on_defended_pieces(board, color, &defended_by, overloaded_sq);
            if our_attacks_on_defended >= 2 {
                bonus += overload_bonus / 2; // +50% se atacamos múltiplas peças que ele defende
            }
        }
    }

    bonus
}

/// Encontra todas as casas defendidas por uma peça
fn find_defended_squares(board: &Board, defender_sq: u8, defender_color: Color) -> Vec<u8> {
    let mut defended = Vec::new();
    let our_pieces = if defender_color == Color::White { board.white_pieces } else { board.black_pieces };

    // Determina o tipo da peça defensora
    let piece_kind = board.get_piece_on_square(defender_sq);
    if piece_kind.is_none() { return defended; }

    // Gera ataques da peça defensora
    let attacks = match piece_kind.unwrap() {
        PieceKind::Pawn => super::utils::compute_pawn_attacks(1u64 << defender_sq, defender_color),
        PieceKind::Knight => crate::moves::knight::get_knight_attacks_lookup(defender_sq),
        PieceKind::Bishop => crate::moves::magic_bitboards::get_bishop_attacks_magic(defender_sq, board.white_pieces | board.black_pieces),
        PieceKind::Rook => crate::moves::magic_bitboards::get_rook_attacks_magic(defender_sq, board.white_pieces | board.black_pieces),
        PieceKind::Queen => crate::moves::magic_bitboards::get_queen_attacks_magic(defender_sq, board.white_pieces | board.black_pieces),
        PieceKind::King => crate::moves::king::get_king_attacks_lookup(defender_sq),
    };

    // Filtra apenas peças nossas que estão sendo defendidas
    let defended_pieces = attacks & our_pieces;
    let mut piece_bb = defended_pieces;
    while piece_bb != 0 {
        let sq = piece_bb.trailing_zeros() as u8;
        piece_bb &= piece_bb - 1;
        defended.push(sq);
    }

    defended
}


/// Conta quantas peças defendidas por um defensor sobrecarregado nós atacamos
fn count_our_attacks_on_defended_pieces(board: &Board, color: Color, defended_by: &std::collections::HashMap<u8, Vec<u8>>, overloaded_defender: u8) -> usize {
    let mut count = 0;

    for (defended_sq, defenders) in defended_by {
        if defenders.contains(&overloaded_defender) {
            // Esta peça é defendida pelo defensor sobrecarregado
            if board.is_square_attacked_by(*defended_sq, color) {
                count += 1;
            }
        }
    }

    count
}

/// Verifica se um ataque é viável usando SEE (Static Exchange Evaluation)
fn is_attack_viable_see(board: &Board, attacker_sq: u8, target_sq: u8) -> bool {
    // Cria movimento simulado para SEE
    let attack_move = Move {
        from: attacker_sq,
        to: target_sq,
        promotion: None,
        is_castling: false,
        is_en_passant: false,
    };

    // Usa SEE do módulo search para avaliar a troca
    let see_value = crate::search::see(board, attack_move);

    // Ataque é viável se SEE >= 0 (não perdemos material)
    see_value >= 0
}

/// Avalia X-ray threats (ameaças descobertas por remoção de guarda)
fn evaluate_xray_threats(board: &Board, color: Color) -> i32 {
    let mut bonus = 0;
    let enemy_color = !color;
    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let enemy_pieces = if enemy_color == Color::White { board.white_pieces } else { board.black_pieces };
    let our_sliders = (board.rooks | board.queens | board.bishops) & our_pieces;

    let mut slider_bb = our_sliders;
    while slider_bb != 0 {
        let slider_sq = slider_bb.trailing_zeros() as u8;
        slider_bb &= slider_bb - 1;

        // Verifica ameaças X-ray em todas as direções
        for direction in &[(0, 1), (1, 0), (1, 1), (1, -1), (0, -1), (-1, 0), (-1, -1), (-1, 1)] {
            if let Some((guard_sq, target_sq, target_value)) =
                find_xray_threat(board, slider_sq, *direction, enemy_pieces) {

                // X-ray threat detectado: se a guarda sair, revelamos ataque
                let xray_bonus = match target_value {
                    v if v >= 900 => 35,  // X-ray na rainha
                    v if v >= 500 => 25,  // X-ray na torre
                    v if v >= 300 => 15,  // X-ray em cavalos/bispos
                    v if v >= 20000 => 50, // X-ray no rei (!!!)
                    _ => 5                // X-ray em peões
                };

                bonus += xray_bonus;

                // Bônus extra se a guarda também está atacada por nós
                if board.is_square_attacked_by(guard_sq, color) {
                    bonus += xray_bonus / 2; // +50% se podemos forçar a remoção
                }
            }
        }
    }

    bonus
}

/// Encontra ameaças X-ray na direção especificada
fn find_xray_threat(board: &Board, slider_sq: u8, direction: (i8, i8), enemy_pieces: crate::types::Bitboard) -> Option<(u8, u8, i32)> {
    let mut current_sq = slider_sq;
    let mut guard_piece: Option<u8> = None;

    loop {
        // Move na direção
        let new_rank = (current_sq / 8) as i8 + direction.0;
        let new_file = (current_sq % 8) as i8 + direction.1;

        if new_rank < 0 || new_rank > 7 || new_file < 0 || new_file > 7 {
            break;
        }

        current_sq = (new_rank as u8) * 8 + (new_file as u8);

        // Verifica se há peça nesta casa
        let sq_bb = 1u64 << current_sq;
        if (board.white_pieces | board.black_pieces) & sq_bb != 0 {
            if enemy_pieces & sq_bb != 0 {
                // Peça inimiga encontrada
                if guard_piece.is_none() {
                    // Primeira peça inimiga - candidata a guarda
                    guard_piece = Some(current_sq);
                } else {
                    // Segunda peça inimiga - possível alvo X-ray
                    if let Some(piece_kind) = board.get_piece_on_square(current_sq) {
                        let target_value = MATERIAL_VALUES[piece_kind as usize];
                        let guard_sq = guard_piece.unwrap();
                        return Some((guard_sq, current_sq, target_value));
                    }
                }
            } else {
                // Peça nossa bloqueia o X-ray
                break;
            }
        }
    }

    None
}

/// Avalia compound threats (múltiplas ameaças simultâneas)
fn evaluate_compound_threats(board: &Board, color: Color) -> i32 {
    let mut bonus = 0;
    let enemy_color = !color;
    let enemy_pieces = if enemy_color == Color::White { board.white_pieces } else { board.black_pieces };

    // Mapa de peças inimigas ameaçadas e quantas vezes
    let mut threat_count: std::collections::HashMap<u8, usize> = std::collections::HashMap::new();

    // Conta ameaças de cada tipo de peça nossa
    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };

    // 1. Conta ameaças de todas as nossas peças
    let mut piece_bb = our_pieces;
    while piece_bb != 0 {
        let piece_sq = piece_bb.trailing_zeros() as u8;
        piece_bb &= piece_bb - 1;

        let attacks = get_piece_attacks(board, piece_sq);
        let threatened_enemies = attacks & enemy_pieces;

        let mut threatened_bb = threatened_enemies;
        while threatened_bb != 0 {
            let threatened_sq = threatened_bb.trailing_zeros() as u8;
            threatened_bb &= threatened_bb - 1;

            *threat_count.entry(threatened_sq).or_insert(0) += 1;
        }
    }

    // 2. Calcula bônus por compound threats
    for (threatened_sq, threat_count_value) in threat_count {
        if threat_count_value >= 2 {
            if let Some(piece_kind) = board.get_piece_on_square(threatened_sq) {
                let piece_value = MATERIAL_VALUES[piece_kind as usize];

                // Bônus baseado no valor da peça e quantidade de ameaças
                let compound_bonus = match (piece_value, threat_count_value) {
                    (v, 2) if v >= 900 => 30,  // Rainha ameaçada por 2 peças
                    (v, 2) if v >= 500 => 20,  // Torre ameaçada por 2 peças
                    (v, 2) if v >= 300 => 15,  // Cavalo/Bispo ameaçado por 2 peças
                    (v, 3) if v >= 500 => 40,  // Peça valiosa ameaçada por 3+ peças
                    (v, n) if v >= 900 && n >= 3 => 50, // Rainha ameaçada por 3+ peças
                    _ => 5 * (threat_count_value - 1) as i32, // Bônus geral
                };

                bonus += compound_bonus;

                // Bônus especial se a peça não está defendida
                if !board.is_square_attacked_by(threatened_sq, enemy_color) {
                    bonus += compound_bonus / 2; // +50% se indefesa
                }
            }
        }
    }

    bonus
}

/// Obtém ataques de uma peça específica
fn get_piece_attacks(board: &Board, piece_sq: u8) -> crate::types::Bitboard {
    let piece_kind = board.get_piece_on_square(piece_sq);
    if piece_kind.is_none() { return 0; }

    let all_pieces = board.white_pieces | board.black_pieces;

    match piece_kind.unwrap() {
        PieceKind::Pawn => {
            let color = if (board.white_pieces & (1u64 << piece_sq)) != 0 { Color::White } else { Color::Black };
            super::utils::compute_pawn_attacks(1u64 << piece_sq, color)
        },
        PieceKind::Knight => crate::moves::knight::get_knight_attacks_lookup(piece_sq),
        PieceKind::Bishop => crate::moves::magic_bitboards::get_bishop_attacks_magic(piece_sq, all_pieces),
        PieceKind::Rook => crate::moves::magic_bitboards::get_rook_attacks_magic(piece_sq, all_pieces),
        PieceKind::Queen => crate::moves::magic_bitboards::get_queen_attacks_magic(piece_sq, all_pieces),
        PieceKind::King => crate::moves::king::get_king_attacks_lookup(piece_sq),
    }
}