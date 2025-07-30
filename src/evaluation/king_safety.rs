// Segurança do rei aprimorada - considera ataques inimigos
use crate::{board::Board, types::{Color, Bitboard, PieceKind}};
use super::game_phase::GamePhase;

// Constantes para máscaras de posição
const CENTRAL_SQUARES: Bitboard = (1u64 << 27) | (1u64 << 28) | (1u64 << 35) | (1u64 << 36);
const EXTENDED_CENTER: Bitboard = 0x00003C3C3C3C0000;

// Zonas de segurança do rei
const WHITE_KINGSIDE_ZONE: Bitboard = 0x00000000000000F0;
const WHITE_QUEENSIDE_ZONE: Bitboard = 0x000000000000000F;
const BLACK_KINGSIDE_ZONE: Bitboard = 0xF000000000000000;
const BLACK_QUEENSIDE_ZONE: Bitboard = 0x0F00000000000000;

// Pesos para diferentes fatores de segurança
const PAWN_SHIELD_VALUE: i32 = 15;
const MISSING_PAWN_PENALTY: i32 = 20;
const OPEN_FILE_PENALTY: i32 = 30;
const HALF_OPEN_FILE_PENALTY: i32 = 15;
const KING_EXPOSED_PENALTY: i32 = 40;
const CASTLING_BONUS: i32 = 50;
const LOST_CASTLING_PENALTY: i32 = 30;

/// Estrutura para análise detalhada de segurança do rei
#[derive(Debug, Clone, Copy, Default)]
pub struct KingSafetyAnalysis {
    pub pawn_shield_score: i32,
    pub attacker_score: i32,
    pub tropism_score: i32,
    pub exposure_score: i32,
    pub castling_score: i32,
    pub open_files_score: i32,
    pub total_attackers: i32,
    pub attack_weight: i32,
    pub defense_weight: i32,
}

impl KingSafetyAnalysis {
    /// Calcula score total de segurança
    pub fn total_score(&self) -> i32 {
        self.pawn_shield_score + self.attacker_score + self.tropism_score +
            self.exposure_score + self.castling_score + self.open_files_score
    }

    /// Retorna se o rei está em perigo
    pub fn is_king_unsafe(&self) -> bool {
        self.total_score() < -100 || self.total_attackers >= 3
    }
}

/// Configuração para avaliação de segurança do rei
pub struct KingSafetyConfig {
    pub detailed_analysis: bool,
    pub consider_piece_values: bool,
    pub dynamic_weights: bool,
}

impl Default for KingSafetyConfig {
    fn default() -> Self {
        KingSafetyConfig {
            detailed_analysis: true,
            consider_piece_values: true,
            dynamic_weights: true,
        }
    }
}

/// Avalia a segurança do rei (versão principal)
pub fn evaluate_king_safety(board: &Board, color: Color, game_phase: &GamePhase) -> i32 {
    evaluate_king_safety_with_config(board, color, game_phase, &KingSafetyConfig::default())
}

/// Avalia segurança do rei com configuração customizada
pub fn evaluate_king_safety_with_config(
    board: &Board,
    color: Color,
    game_phase: &GamePhase,
    config: &KingSafetyConfig
) -> i32 {
    // Segurança menos importante no final
    if matches!(game_phase, GamePhase::Endgame) {
        return 0;
    }

    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let king_bb = board.kings & pieces;

    if king_bb == 0 {
        return 0;
    }

    let king_square = king_bb.trailing_zeros() as usize;

    if config.detailed_analysis {
        let analysis = analyze_king_safety_detailed(board, color, king_square, game_phase);
        analysis.total_score()
    } else {
        evaluate_king_safety_simple(board, color, king_square, game_phase)
    }
}

/// Análise detalhada de segurança do rei
pub fn analyze_king_safety_detailed(
    board: &Board,
    color: Color,
    king_square: usize,
    game_phase: &GamePhase
) -> KingSafetyAnalysis {
    let mut analysis = KingSafetyAnalysis::default();

    let rank = king_square / 8;
    let file = king_square % 8;

    // 1. Shield de peões
    analysis.pawn_shield_score = evaluate_pawn_shield_advanced(board, color, king_square, rank, file);

    // 2. Arquivos abertos próximos ao rei
    analysis.open_files_score = evaluate_open_files_near_king(board, color, king_square);

    // 3. Conta atacantes inimigos
    let attacker_info = count_enemy_attackers_detailed(board, color, king_square);
    analysis.attacker_score = -attacker_info.0;
    analysis.total_attackers = attacker_info.1;
    analysis.attack_weight = attacker_info.2;

    // 4. Tropismo (proximidade de peças inimigas)
    analysis.tropism_score = -evaluate_tropism_advanced(board, color, king_square);

    // 5. Exposição do rei
    analysis.exposure_score = evaluate_king_exposure(board, color, king_square, rank, file, game_phase);

    // 6. Status de roque
    analysis.castling_score = evaluate_castling_status(board, color, king_square);

    // 7. Defesas disponíveis
    analysis.defense_weight = evaluate_king_defenses(board, color, king_square);

    // Ajusta scores baseado em defesas
    if analysis.defense_weight > 0 {
        let defense_factor = 1.0 - (analysis.defense_weight as f32 / 100.0).min(0.5);
        analysis.attacker_score = (analysis.attacker_score as f32 * defense_factor) as i32;
        analysis.tropism_score = (analysis.tropism_score as f32 * defense_factor) as i32;
    }

    analysis
}

/// Avaliação simples de segurança (para performance)
fn evaluate_king_safety_simple(
    board: &Board,
    color: Color,
    king_square: usize,
    game_phase: &GamePhase
) -> i32 {
    let mut score = 0;
    let rank = king_square / 8;
    let file = king_square % 8;

    // Shield básico
    score += evaluate_pawn_shield(board, color, king_square, rank, file);

    // Penaliza rei no centro
    if (CENTRAL_SQUARES & (1u64 << king_square)) != 0 {
        score -= 50;
    }

    // Conta atacantes básico
    score -= evaluate_enemy_attackers(board, color, king_square);

    // Exposição básica
    if !matches!(game_phase, GamePhase::Endgame) {
        score += evaluate_basic_exposure(color, rank);
    }

    score
}

/// Avalia shield de peões avançado
fn evaluate_pawn_shield_advanced(
    board: &Board,
    color: Color,
    king_square: usize,
    rank: usize,
    file: usize
) -> i32 {
    let mut score = 0;
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let our_pawns = board.pawns & pieces;
    let enemy_pawns = board.pawns & if color == Color::White { board.black_pieces } else { board.white_pieces };

    // Define zona do shield baseado na posição do rei
    let shield_files = get_shield_files(file);
    let shield_ranks = get_shield_ranks(rank, color);

    for &shield_file in &shield_files {
        for &shield_rank in &shield_ranks {
            if shield_rank < 8 && shield_file < 8 {
                let shield_square = shield_rank * 8 + shield_file;
                if shield_square < 64 {
                    let square_bb = 1u64 << shield_square;

                    // Avalia presença/ausência de peões no shield
                    if (our_pawns & square_bb) != 0 {
                        score += PAWN_SHIELD_VALUE;

                        // Bônus extra se o peão não foi movido (rank inicial)
                        if is_pawn_on_initial_rank(shield_square, color) {
                            score += 5;
                        }
                    } else {
                        score -= MISSING_PAWN_PENALTY;

                        // Penalidade extra se há peão inimigo nesta casa
                        if (enemy_pawns & square_bb) != 0 {
                            score -= 10;
                        }
                    }
                }
            }
        }
    }

    // Avalia estrutura geral do shield
    score += evaluate_shield_structure(board, color, king_square);

    score
}

/// Avalia shield de peões (versão original)
fn evaluate_pawn_shield(board: &Board, color: Color, king_square: usize, rank: usize, file: usize) -> i32 {
    evaluate_pawn_shield_advanced(board, color, king_square, rank, file)
}

/// Obtém arquivos relevantes para o shield
fn get_shield_files(king_file: usize) -> Vec<usize> {
    match king_file {
        0 => vec![0, 1, 2],           // Rei na coluna a
        7 => vec![5, 6, 7],           // Rei na coluna h
        _ => vec![king_file.saturating_sub(1), king_file, (king_file + 1).min(7)],
    }
}

/// Obtém ranks relevantes para o shield
fn get_shield_ranks(king_rank: usize, color: Color) -> Vec<usize> {
    if color == Color::White {
        vec![king_rank + 1, king_rank + 2].into_iter()
            .filter(|&r| r < 8)
            .collect()
    } else {
        vec![king_rank.saturating_sub(1), king_rank.saturating_sub(2)].into_iter()
            .filter(|&r| r < 8)
            .collect()
    }
}

/// Verifica se peão está no rank inicial
fn is_pawn_on_initial_rank(square: usize, color: Color) -> bool {
    let rank = square / 8;
    match color {
        Color::White => rank == 1,
        Color::Black => rank == 6,
    }
}

/// Avalia estrutura geral do shield
fn evaluate_shield_structure(board: &Board, color: Color, king_square: usize) -> i32 {
    let mut score = 0;
    let king_file = king_square % 8;

    // Penaliza buracos no shield
    let holes = count_shield_holes(board, color, king_square);
    score -= holes * 15;

    // Bônus por shield compacto (peões conectados)
    let connected = count_connected_shield_pawns(board, color, king_square);
    score += connected * 8;

    // Penaliza peões avançados demais no shield
    let advanced = count_advanced_shield_pawns(board, color, king_square);
    score -= advanced * 10;

    score
}

/// Conta buracos no shield de peões
fn count_shield_holes(board: &Board, color: Color, king_square: usize) -> i32 {
    let mut holes = 0;
    let our_pawns = board.pawns & if color == Color::White { board.white_pieces } else { board.black_pieces };
    let king_file = king_square % 8;

    for file_offset in -1i32..=1 {
        let file = (king_file as i32 + file_offset) as usize;
        if file >= 8 { continue; }

        let file_mask = super::utils::get_file_mask_from_file(file as u8);
        let file_pawns = our_pawns & file_mask;

        if file_pawns == 0 {
            holes += 1;
        }
    }

    holes
}

/// Conta peões conectados no shield
fn count_connected_shield_pawns(board: &Board, color: Color, king_square: usize) -> i32 {
    let mut connected = 0;
    let our_pawns = board.pawns & if color == Color::White { board.white_pieces } else { board.black_pieces };
    let king_zone = get_king_zone(king_square, color);
    let shield_pawns = our_pawns & king_zone;

    let mut pawn_bb = shield_pawns;
    while pawn_bb != 0 {
        let sq = pawn_bb.trailing_zeros() as u8;
        pawn_bb &= pawn_bb - 1;

        // Verifica conexões horizontais
        if sq % 8 > 0 && (shield_pawns & (1u64 << (sq - 1))) != 0 {
            connected += 1;
        }
        if sq % 8 < 7 && (shield_pawns & (1u64 << (sq + 1))) != 0 {
            connected += 1;
        }
    }

    connected / 2 // Divide por 2 para não contar duas vezes
}

/// Conta peões avançados no shield
fn count_advanced_shield_pawns(board: &Board, color: Color, king_square: usize) -> i32 {
    let mut advanced = 0;
    let our_pawns = board.pawns & if color == Color::White { board.white_pieces } else { board.black_pieces };
    let king_zone = get_king_zone(king_square, color);
    let shield_pawns = our_pawns & king_zone;

    let mut pawn_bb = shield_pawns;
    while pawn_bb != 0 {
        let sq = pawn_bb.trailing_zeros() as u8;
        pawn_bb &= pawn_bb - 1;

        let rank = sq / 8;
        if color == Color::White && rank >= 4 {
            advanced += 1;
        } else if color == Color::Black && rank <= 3 {
            advanced += 1;
        }
    }

    advanced
}

/// Obtém zona ao redor do rei
fn get_king_zone(king_square: usize, color: Color) -> Bitboard {
    let king_bb = 1u64 << king_square;
    let king_attacks = crate::moves::king::get_king_attacks_lookup(king_square as u8);

    // Inclui o quadrado do rei e todos adjacentes
    let basic_zone = king_bb | king_attacks;

    // Expande a zona na direção apropriada
    if color == Color::White {
        basic_zone | (basic_zone << 8) | (basic_zone << 16)
    } else {
        basic_zone | (basic_zone >> 8) | (basic_zone >> 16)
    }
}

/// Avalia arquivos abertos próximos ao rei
fn evaluate_open_files_near_king(board: &Board, color: Color, king_square: usize) -> i32 {
    let mut score = 0;
    let king_file = king_square % 8;
    let our_pawns = board.pawns & if color == Color::White { board.white_pieces } else { board.black_pieces };
    let enemy_pawns = board.pawns & if color == Color::White { board.black_pieces } else { board.white_pieces };

    // Verifica arquivos próximos ao rei
    for file_offset in -1i32..=1 {
        let file = (king_file as i32 + file_offset) as usize;
        if file >= 8 { continue; }

        let file_mask = super::utils::get_file_mask_from_file(file as u8);
        let our_file_pawns = our_pawns & file_mask;
        let enemy_file_pawns = enemy_pawns & file_mask;

        if our_file_pawns == 0 && enemy_file_pawns == 0 {
            // Arquivo completamente aberto
            score -= OPEN_FILE_PENALTY;

            // Penalidade extra se há torre inimiga neste arquivo
            if has_enemy_rook_on_file(board, color, file as u8) {
                score -= 20;
            }
        } else if our_file_pawns == 0 {
            // Arquivo semi-aberto (sem nossos peões)
            score -= HALF_OPEN_FILE_PENALTY;
        }
    }

    score
}

/// Verifica se há torre inimiga em arquivo específico
fn has_enemy_rook_on_file(board: &Board, our_color: Color, file: u8) -> bool {
    let enemy_pieces = if our_color == Color::White { board.black_pieces } else { board.white_pieces };
    let enemy_rooks_queens = (board.rooks | board.queens) & enemy_pieces;
    let file_mask = super::utils::get_file_mask_from_file(file);

    (enemy_rooks_queens & file_mask) != 0
}

/// Conta atacantes inimigos com análise detalhada
fn count_enemy_attackers_detailed(board: &Board, color: Color, king_square: usize) -> (i32, i32, i32) {
    let enemy_color = !color;
    let mut total_penalty = 0;
    let mut attacker_count = 0;
    let mut attack_weight = 0;

    if board.is_square_attacked_by(king_square as u8, enemy_color) {
        let enemy_pieces = if enemy_color == Color::White { board.white_pieces } else { board.black_pieces };

        // Analisa cada tipo de atacante
        let pawn_attackers = count_piece_type_attackers(board, king_square as u8, enemy_color, board.pawns & enemy_pieces);
        let knight_attackers = count_piece_type_attackers(board, king_square as u8, enemy_color, board.knights & enemy_pieces);
        let bishop_attackers = count_piece_type_attackers(board, king_square as u8, enemy_color, board.bishops & enemy_pieces);
        let rook_attackers = count_piece_type_attackers(board, king_square as u8, enemy_color, board.rooks & enemy_pieces);
        let queen_attackers = count_piece_type_attackers(board, king_square as u8, enemy_color, board.queens & enemy_pieces);

        // Calcula penalidades e pesos
        if pawn_attackers > 0 {
            total_penalty += pawn_attackers * 15;
            attack_weight += pawn_attackers * 2;
        }
        if knight_attackers > 0 {
            total_penalty += knight_attackers * 30;
            attack_weight += knight_attackers * 3;
        }
        if bishop_attackers > 0 {
            total_penalty += bishop_attackers * 25;
            attack_weight += bishop_attackers * 3;
        }
        if rook_attackers > 0 {
            total_penalty += rook_attackers * 40;
            attack_weight += rook_attackers * 5;
        }
        if queen_attackers > 0 {
            total_penalty += queen_attackers * 60;
            attack_weight += queen_attackers * 9;
        }

        attacker_count = pawn_attackers + knight_attackers + bishop_attackers + rook_attackers + queen_attackers;

        // Bônus não-linear por múltiplos atacantes
        if attacker_count >= 2 {
            let synergy_bonus = match attacker_count {
                2 => 20,
                3 => 50,
                4 => 100,
                _ => 150,
            };
            total_penalty += synergy_bonus;
        }

        // Penalidade extra se não há defesas adequadas
        let defenders = count_king_defenders(board, color, king_square as u8);
        if defenders < attacker_count {
            total_penalty += (attacker_count - defenders) * 25;
        }
    }

    (total_penalty, attacker_count, attack_weight)
}

/// Avalia atacantes inimigos (versão original)
fn evaluate_enemy_attackers(board: &Board, color: Color, king_square: usize) -> i32 {
    let (penalty, _, _) = count_enemy_attackers_detailed(board, color, king_square);
    penalty
}

/// Conta atacantes de um tipo específico
fn count_piece_type_attackers(board: &Board, target_square: u8, attacker_color: Color, piece_bb: Bitboard) -> i32 {
    let mut count = 0;
    let mut bb = piece_bb;

    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        bb &= bb - 1;

        if piece_attacks_square(board, sq, target_square, attacker_color) {
            count += 1;
        }
    }

    count
}

/// Verifica se peça específica ataca uma casa
fn piece_attacks_square(board: &Board, piece_square: u8, target_square: u8, color: Color) -> bool {
    let piece_bb = 1u64 << piece_square;
    let all_pieces = board.white_pieces | board.black_pieces;

    if (board.pawns & piece_bb) != 0 {
        can_pawn_attack(piece_square, target_square, color)
    } else if (board.knights & piece_bb) != 0 {
        let knight_attacks = crate::moves::knight::get_knight_attacks_lookup(piece_square);
        (knight_attacks & (1u64 << target_square)) != 0
    } else if (board.bishops & piece_bb) != 0 {
        let bishop_attacks = crate::moves::magic_bitboards::get_bishop_attacks_magic(piece_square, all_pieces);
        (bishop_attacks & (1u64 << target_square)) != 0
    } else if (board.rooks & piece_bb) != 0 {
        let rook_attacks = crate::moves::magic_bitboards::get_rook_attacks_magic(piece_square, all_pieces);
        (rook_attacks & (1u64 << target_square)) != 0
    } else if (board.queens & piece_bb) != 0 {
        let queen_attacks = crate::moves::magic_bitboards::get_queen_attacks_magic(piece_square, all_pieces);
        (queen_attacks & (1u64 << target_square)) != 0
    } else if (board.kings & piece_bb) != 0 {
        let king_attacks = crate::moves::king::get_king_attacks_lookup(piece_square);
        (king_attacks & (1u64 << target_square)) != 0
    } else {
        false
    }
}

/// Verifica se peão pode atacar casa
fn can_pawn_attack(pawn_square: u8, target_square: u8, color: Color) -> bool {
    let pawn_rank = pawn_square / 8;
    let pawn_file = pawn_square % 8;
    let target_rank = target_square / 8;
    let target_file = target_square % 8;

    if color == Color::White {
        // Brancas atacam diagonalmente para cima
        target_rank == pawn_rank + 1 &&
            (target_file as i8 - pawn_file as i8).abs() == 1
    } else {
        // Pretas atacam diagonalmente para baixo
        target_rank + 1 == pawn_rank &&
            (target_file as i8 - pawn_file as i8).abs() == 1
    }
}

/// Conta defensores do rei
fn count_king_defenders(board: &Board, color: Color, king_square: u8) -> i32 {
    let mut defenders = 0;
    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let king_zone = get_king_zone(king_square as usize, color);

    // Conta peças nossas que defendem a zona do rei
    let defending_pieces = our_pieces & !board.kings;
    let mut piece_bb = defending_pieces;

    while piece_bb != 0 {
        let sq = piece_bb.trailing_zeros() as u8;
        piece_bb &= piece_bb - 1;

        // Verifica se a peça defende alguma casa na zona do rei
        let piece_attacks = get_piece_attacks(board, sq, color);
        if (piece_attacks & king_zone) != 0 {
            defenders += 1;
        }
    }

    defenders
}

/// Obtém ataques de uma peça
fn get_piece_attacks(board: &Board, square: u8, color: Color) -> Bitboard {
    let piece_bb = 1u64 << square;
    let all_pieces = board.white_pieces | board.black_pieces;
    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };

    if (board.pawns & piece_bb) != 0 {
        get_pawn_attacks(square, color)
    } else if (board.knights & piece_bb) != 0 {
        crate::moves::knight::get_knight_attacks_lookup(square)
    } else if (board.bishops & piece_bb) != 0 {
        crate::moves::magic_bitboards::get_bishop_attacks_magic(square, all_pieces)
    } else if (board.rooks & piece_bb) != 0 {
        crate::moves::magic_bitboards::get_rook_attacks_magic(square, all_pieces)
    } else if (board.queens & piece_bb) != 0 {
        crate::moves::magic_bitboards::get_queen_attacks_magic(square, all_pieces)
    } else if (board.kings & piece_bb) != 0 {
        crate::moves::king::get_king_attacks_lookup(square)
    } else {
        0
    }
}

/// Obtém ataques de peão
fn get_pawn_attacks(square: u8, color: Color) -> Bitboard {
    let rank = square / 8;
    let file = square % 8;
    let mut attacks = 0u64;

    if color == Color::White && rank < 7 {
        if file > 0 {
            attacks |= 1u64 << (square + 7);
        }
        if file < 7 {
            attacks |= 1u64 << (square + 9);
        }
    } else if color == Color::Black && rank > 0 {
        if file > 0 {
            attacks |= 1u64 << (square - 9);
        }
        if file < 7 {
            attacks |= 1u64 << (square - 7);
        }
    }

    attacks
}

/// Avalia tropismo avançado
fn evaluate_tropism_advanced(board: &Board, color: Color, king_square: usize) -> i32 {
    let enemy_color = !color;
    let enemy_pieces = if enemy_color == Color::White { board.white_pieces } else { board.black_pieces };
    let mut tropism_penalty = 0;

    // Define zonas de distância ao redor do rei
    let close_zone = get_king_zone(king_square, color);
    let medium_zone = expand_zone(close_zone);
    let far_zone = expand_zone(medium_zone);

    // Avalia cada tipo de peça inimiga
    tropism_penalty += evaluate_piece_tropism(board.knights & enemy_pieces, king_square, 6, 4, 2);
    tropism_penalty += evaluate_piece_tropism(board.bishops & enemy_pieces, king_square, 5, 3, 2);
    tropism_penalty += evaluate_piece_tropism(board.rooks & enemy_pieces, king_square, 8, 5, 3);
    tropism_penalty += evaluate_piece_tropism(board.queens & enemy_pieces, king_square, 12, 8, 4);

    // Penalidade extra por concentração de peças
    let pieces_near_king = ((close_zone | medium_zone) & enemy_pieces).count_ones();
    if pieces_near_king >= 3 {
        tropism_penalty += (pieces_near_king as i32 - 2) * 15;
    }

    tropism_penalty
}

/// Avalia tropismo (versão original)
fn evaluate_tropism(board: &Board, color: Color, king_square: usize) -> i32 {
    evaluate_tropism_advanced(board, color, king_square)
}

/// Avalia tropismo de tipo específico de peça
fn evaluate_piece_tropism(
    piece_bb: Bitboard,
    king_square: usize,
    close_penalty: i32,
    medium_penalty: i32,
    far_penalty: i32
) -> i32 {
    let mut penalty = 0;
    let mut bb = piece_bb;

    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        bb &= bb - 1;

        let distance = super::utils::king_distance(sq, king_square as u8);

        penalty += match distance {
            0..=2 => close_penalty,
            3..=4 => medium_penalty,
            5..=6 => far_penalty,
            _ => 0,
        };
    }

    penalty
}

/// Expande uma zona
fn expand_zone(zone: Bitboard) -> Bitboard {
    let expanded = zone |
        (zone << 1) | (zone >> 1) |
        (zone << 8) | (zone >> 8) |
        (zone << 9) | (zone >> 9) |
        (zone << 7) | (zone >> 7);

    // Remove wrap-around nas bordas
    expanded & 0xFFFFFFFFFFFFFFFF
}

/// Avalia exposição do rei
fn evaluate_king_exposure(
    board: &Board,
    color: Color,
    king_square: usize,
    rank: usize,
    file: usize,
    game_phase: &GamePhase
) -> i32 {
    let mut score = 0;

    // Penalidade por rei no centro
    if (CENTRAL_SQUARES & (1u64 << king_square)) != 0 {
        score -= KING_EXPOSED_PENALTY;
    } else if (EXTENDED_CENTER & (1u64 << king_square)) != 0 {
        score -= KING_EXPOSED_PENALTY / 2;
    }

    // Penalidade progressiva por rei avançado
    if !matches!(game_phase, GamePhase::Endgame) {
        score += match color {
            Color::White => {
                match rank {
                    0..=1 => 0,      // Seguro
                    2 => -20,        // Ligeiramente exposto
                    3 => -40,        // Exposto
                    4 => -70,        // Muito exposto
                    _ => -100,       // Extremamente exposto
                }
            },
            Color::Black => {
                match rank {
                    6..=7 => 0,      // Seguro
                    5 => -20,        // Ligeiramente exposto
                    4 => -40,        // Exposto
                    3 => -70,        // Muito exposto
                    _ => -100,       // Extremamente exposto
                }
            }
        };
    }

    // Penalidade por rei na borda (menos rotas de fuga)
    if file == 0 || file == 7 {
        score -= 15;
    }
    if rank == 0 || rank == 7 {
        score -= 10;
    }

    score
}

/// Avalia exposição básica
fn evaluate_basic_exposure(color: Color, rank: usize) -> i32 {
    match color {
        Color::White => {
            if rank > 1 {
                -(30 + (rank as i32 - 2) * 30)
            } else { 0 }
        },
        Color::Black => {
            if rank < 6 {
                -(30 + (5 - rank as i32) * 30)
            } else { 0 }
        }
    }
}

/// Avalia status de castling
fn evaluate_castling_status(board: &Board, color: Color, king_square: usize) -> i32 {
    let mut score = 0;

    if has_castled(board, color) {
        score += CASTLING_BONUS;

        // Bônus extra por castling kingside
        let kingside_squares = if color == Color::White { [6] } else { [62] };
        if kingside_squares.contains(&king_square) {
            score += 10;
        }
    } else if can_castle(board, color) {
        // Ainda pode fazer castling
        score += 10;

        // Penalidade se já moveu muitas peças (castling tardio)
        let development = count_developed_pieces(board, color);
        if development > 4 {
            score -= 20;
        }
    } else {
        // Perdeu direitos de castling
        score -= LOST_CASTLING_PENALTY;

        // Penalidade extra se rei ainda está no centro
        if is_king_in_center(king_square) {
            score -= 20;
        }
    }

    score
}

/// Conta peças desenvolvidas
fn count_developed_pieces(board: &Board, color: Color) -> i32 {
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let back_rank = if color == Color::White { 0xFF } else { 0xFF00000000000000 };

    let minor_pieces = (board.knights | board.bishops) & pieces;
    let developed = minor_pieces & !back_rank;

    developed.count_ones() as i32
}

/// Verifica se rei está no centro
fn is_king_in_center(king_square: usize) -> bool {
    let file = king_square % 8;
    file >= 3 && file <= 4
}

/// Avalia defesas do rei
fn evaluate_king_defenses(board: &Board, color: Color, king_square: usize) -> i32 {
    let mut defense_score = 0;
    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };

    // Peças defensivas próximas
    let king_zone = get_king_zone(king_square, color);
    let defending_pieces = our_pieces & king_zone & !board.kings;

    // Cada peça defensora contribui
    let defenders = defending_pieces.count_ones() as i32;
    defense_score += defenders * 10;

    // Bônus por tipos específicos de defensores
    if (board.knights & defending_pieces) != 0 {
        defense_score += 15; // Cavalos são bons defensores
    }
    if (board.pawns & defending_pieces) != 0 {
        defense_score += 5 * (board.pawns & defending_pieces).count_ones() as i32;
    }

    // Casas de fuga disponíveis
    let escape_squares = count_escape_squares(board, color, king_square as u8);
    defense_score += escape_squares * 8;

    defense_score
}

/// Conta casas de fuga do rei
fn count_escape_squares(board: &Board, color: Color, king_square: u8) -> i32 {
    let king_moves = crate::moves::king::get_king_attacks_lookup(king_square);
    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let all_pieces = board.white_pieces | board.black_pieces;

    let mut safe_squares = 0;
    let mut moves_bb = king_moves & !our_pieces;

    while moves_bb != 0 {
        let sq = moves_bb.trailing_zeros() as u8;
        moves_bb &= moves_bb - 1;

        // Verifica se a casa não está atacada
        if !board.is_square_attacked_by(sq, !color) {
            safe_squares += 1;
        }
    }

    safe_squares
}

/// Verifica se o rei já fez roque
fn has_castled(board: &Board, color: Color) -> bool {
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let king_bb = board.kings & pieces;

    if king_bb == 0 {
        return false;
    }

    let king_square = king_bb.trailing_zeros() as u8;

    match color {
        Color::White => king_square == 6 || king_square == 2,    // g1 ou c1
        Color::Black => king_square == 62 || king_square == 58,  // g8 ou c8
    }
}

/// Verifica se ainda pode fazer roque
fn can_castle(board: &Board, color: Color) -> bool {
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let king_bb = board.kings & pieces;

    if king_bb == 0 {
        return false;
    }

    let king_square = king_bb.trailing_zeros() as u8;

    match color {
        Color::White => {
            if king_square != 4 { // e1
                return false;
            }

            // Verifica direitos de castling
            board.castling_rights & 0x03 != 0
        },
        Color::Black => {
            if king_square != 60 { // e8
                return false;
            }

            // Verifica direitos de castling
            board.castling_rights & 0x0C != 0
        }
    }
}

/// Análise específica para diferentes fases do jogo
pub fn analyze_king_safety_by_phase(
    board: &Board,
    color: Color,
    phase: &GamePhase
) -> KingSafetyAnalysis {
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let king_bb = board.kings & pieces;

    if king_bb == 0 {
        return KingSafetyAnalysis::default();
    }

    let king_square = king_bb.trailing_zeros() as usize;

    match phase {
        GamePhase::Opening => analyze_opening_king_safety(board, color, king_square),
        GamePhase::EarlyMiddlegame => analyze_opening_king_safety(board, color, king_square),
        GamePhase::Middlegame => analyze_middlegame_king_safety(board, color, king_square),
        GamePhase::LateMiddlegame => analyze_middlegame_king_safety(board, color, king_square),
        GamePhase::EarlyEndgame => KingSafetyAnalysis::default(), // Segurança menos importante no endgame
        GamePhase::Endgame => KingSafetyAnalysis::default(), 
        GamePhase::LateEndgame => KingSafetyAnalysis::default(),
        GamePhase::PureEndgame => KingSafetyAnalysis::default(),
        GamePhase::TheoreticalEndgame => KingSafetyAnalysis::default(),
    }
}

/// Análise de segurança na abertura
fn analyze_opening_king_safety(board: &Board, color: Color, king_square: usize) -> KingSafetyAnalysis {
    let mut analysis = KingSafetyAnalysis::default();

    // Na abertura, castling é crucial
    if !has_castled(board, color) {
        analysis.castling_score = -40;

        if !can_castle(board, color) {
            analysis.castling_score = -60;
        }
    } else {
        analysis.castling_score = 50;
    }

    // Desenvolvimento e controle central são importantes
    if is_king_in_center(king_square) {
        analysis.exposure_score = -50;
    }

    analysis
}

/// Análise de segurança no meio-jogo
fn analyze_middlegame_king_safety(board: &Board, color: Color, king_square: usize) -> KingSafetyAnalysis {
    // Análise completa para o meio-jogo
    analyze_king_safety_detailed(board, color, king_square, &GamePhase::Middlegame)
}

// ============================
// NOVOS MÉTODOS AVANÇADOS DE KING SAFETY
// ============================

/// Sistema avançado de avaliação de padrões de ataque ao rei
pub struct KingAttackPatterns {
    pub mating_net_pressure: i32,
    pub sacrificial_threats: i32,
    pub back_rank_weakness: i32,
    pub diagonal_pressure: i32,
    pub knight_fork_threats: i32,
    pub discovered_attack_potential: i32,
}

impl Default for KingAttackPatterns {
    fn default() -> Self {
        Self {
            mating_net_pressure: 0,
            sacrificial_threats: 0,
            back_rank_weakness: 0,
            diagonal_pressure: 0,
            knight_fork_threats: 0,
            discovered_attack_potential: 0,
        }
    }
}

/// Avalia padrões específicos de ataque ao rei
pub fn evaluate_king_attack_patterns(
    board: &Board,
    color: Color,
    king_square: usize
) -> KingAttackPatterns {
    let mut patterns = KingAttackPatterns::default();
    
    patterns.mating_net_pressure = evaluate_mating_net_pressure(board, color, king_square);
    patterns.sacrificial_threats = evaluate_sacrificial_threats(board, color, king_square);
    patterns.back_rank_weakness = evaluate_back_rank_weakness(board, color, king_square);
    patterns.diagonal_pressure = evaluate_diagonal_pressure(board, color, king_square);
    patterns.knight_fork_threats = evaluate_knight_fork_threats(board, color, king_square);
    patterns.discovered_attack_potential = evaluate_discovered_attack_potential(board, color, king_square);
    
    patterns
}

/// Avalia pressão de rede de mate
fn evaluate_mating_net_pressure(board: &Board, color: Color, king_square: usize) -> i32 {
    let enemy_color = !color;
    let enemy_pieces = if enemy_color == Color::White { board.white_pieces } else { board.black_pieces };
    let mut pressure = 0;
    
    // Verifica se há múltiplas peças coordenando ataque
    let king_zone = get_king_zone(king_square, color);
    let attacking_pieces = get_pieces_attacking_zone(board, enemy_color, king_zone);
    
    if attacking_pieces >= 3 {
        // Múltiplas peças atacando - possível rede de mate
        pressure += 60;
        
        // Verifica coordenação específica Dama + Torre
        if has_queen_rook_battery(board, enemy_color, king_square) {
            pressure += 40;
        }
        
        // Verifica coordenação Dama + Bispo
        if has_queen_bishop_battery(board, enemy_color, king_square) {
            pressure += 35;
        }
        
        // Dupla torre é muito perigosa
        if has_double_rook_attack(board, enemy_color, king_square) {
            pressure += 50;
        }
    }
    
    pressure
}

/// Avalia ameaças de sacrifício
fn evaluate_sacrificial_threats(board: &Board, color: Color, king_square: usize) -> i32 {
    let enemy_color = !color;
    let mut threat_score = 0;
    
    // Verifica sacrifícios típicos na posição do rei
    if can_sacrifice_on_h7_h2(board, color, king_square) {
        threat_score += 45; // Sacrifício clássico Bxh7+
    }
    
    if can_sacrifice_knight_fork(board, color, king_square) {
        threat_score += 30; // Sacrifício de cavalo para garfo
    }
    
    if can_sacrifice_exchange(board, color, king_square) {
        threat_score += 25; // Sacrifício de qualidade
    }
    
    // Verifica "sacrifícios gregos" (Bispo + Cavalo)
    if has_greek_gift_pattern(board, color, king_square) {
        threat_score += 55;
    }
    
    threat_score
}

/// Avalia fraqueza na primeira fileira
fn evaluate_back_rank_weakness(board: &Board, color: Color, king_square: usize) -> i32 {
    let rank = king_square / 8;
    let back_rank = if color == Color::White { 0 } else { 7 };
    
    if rank != back_rank {
        return 0; // Rei não está na primeira fileira
    }
    
    let mut weakness = 0;
    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let enemy_pieces = if color == Color::White { board.black_pieces } else { board.white_pieces };
    
    // Verifica se há torres inimigas na mesma fileira
    let back_rank_mask = if color == Color::White { 0xFF } else { 0xFF00000000000000 };
    let enemy_rooks_queens = (board.rooks | board.queens) & enemy_pieces;
    
    if (enemy_rooks_queens & back_rank_mask) != 0 {
        weakness += 40;
        
        // Pior ainda se não há casas de fuga
        let escape_squares = count_back_rank_escape_squares(board, color, king_square);
        if escape_squares == 0 {
            weakness += 60; // Mate de primeira fileira possível
        } else if escape_squares == 1 {
            weakness += 30;
        }
    }
    
    weakness
}

/// Conta casas de fuga na primeira fileira
fn count_back_rank_escape_squares(board: &Board, color: Color, king_square: usize) -> i32 {
    let king_file = king_square % 8;
    let back_rank = if color == Color::White { 0 } else { 7 };
    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    
    let mut escape_squares = 0;
    
    // Verifica casas adjacentes na primeira fileira
    for file_offset in -1i32..=1 {
        let target_file = king_file as i32 + file_offset;
        if target_file >= 0 && target_file < 8 && target_file != king_file as i32 {
            let target_square = back_rank * 8 + target_file as usize;
            let target_bb = 1u64 << target_square;
            
            // Casa deve estar livre de nossas peças
            if (our_pieces & target_bb) == 0 {
                // E não pode estar atacada pelo inimigo
                if !board.is_square_attacked_by(target_square as u8, !color) {
                    escape_squares += 1;
                }
            }
        }
    }
    
    escape_squares
}

/// Avalia pressão nas diagonais
fn evaluate_diagonal_pressure(board: &Board, color: Color, king_square: usize) -> i32 {
    let enemy_color = !color;
    let enemy_pieces = if enemy_color == Color::White { board.white_pieces } else { board.black_pieces };
    let enemy_bishops_queens = (board.bishops | board.queens) & enemy_pieces;
    
    let mut pressure = 0;
    let all_pieces = board.white_pieces | board.black_pieces;
    
    // Verifica cada bispo/dama inimiga
    let mut piece_bb = enemy_bishops_queens;
    while piece_bb != 0 {
        let sq = piece_bb.trailing_zeros() as u8;
        piece_bb &= piece_bb - 1;
        
        let attacks = crate::moves::magic_bitboards::get_bishop_attacks_magic(sq, all_pieces);
        
        // Se ataca o rei ou zona próxima
        if (attacks & (1u64 << king_square)) != 0 {
            pressure += 25;
            
            // Verifica se há peças nossas bloqueando
            let blockers = count_pieces_on_diagonal(board, color, sq as usize, king_square);
            if blockers <= 1 {
                pressure += 20; // Diagonal quase livre
            }
        }
        
        // Verifica pressão na zona do rei
        let king_zone = get_king_zone(king_square, color);
        if (attacks & king_zone) != 0 {
            pressure += 15;
        }
    }
    
    pressure
}

/// Conta peças numa diagonal entre duas casas
fn count_pieces_on_diagonal(board: &Board, color: Color, from_square: usize, to_square: usize) -> i32 {
    let all_pieces = board.white_pieces | board.black_pieces;
    let diagonal_mask = get_diagonal_between(from_square as u8, to_square as u8);
    
    (all_pieces & diagonal_mask).count_ones() as i32
}

/// Obtém máscara diagonal entre duas casas
fn get_diagonal_between(from: u8, to: u8) -> Bitboard {
    // Implementação simplificada - na prática seria mais complexa
    let from_rank = from / 8;
    let from_file = from % 8;
    let to_rank = to / 8;
    let to_file = to % 8;
    
    if (from_rank as i8 - to_rank as i8).abs() != (from_file as i8 - to_file as i8).abs() {
        return 0; // Não estão na mesma diagonal
    }
    
    let mut mask = 0u64;
    let rank_dir = if to_rank > from_rank { 1 } else { -1 };
    let file_dir = if to_file > from_file { 1 } else { -1 };
    
    let mut current_rank = from_rank as i8 + rank_dir;
    let mut current_file = from_file as i8 + file_dir;
    
    while current_rank != to_rank as i8 && current_file != to_file as i8 {
        if current_rank >= 0 && current_rank < 8 && current_file >= 0 && current_file < 8 {
            let square = current_rank as u8 * 8 + current_file as u8;
            mask |= 1u64 << square;
        }
        current_rank += rank_dir;
        current_file += file_dir;
    }
    
    mask
}

/// Avalia ameaças de garfo de cavalo
fn evaluate_knight_fork_threats(board: &Board, color: Color, king_square: usize) -> i32 {
    let enemy_color = !color;
    let enemy_pieces = if enemy_color == Color::White { board.white_pieces } else { board.black_pieces };
    let enemy_knights = board.knights & enemy_pieces;
    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    
    let mut threat_score = 0;
    
    let mut knight_bb = enemy_knights;
    while knight_bb != 0 {
        let knight_sq = knight_bb.trailing_zeros() as u8;
        knight_bb &= knight_bb - 1;
        
        let knight_attacks = crate::moves::knight::get_knight_attacks_lookup(knight_sq);
        
        // Verifica se o cavalo pode atacar o rei e outra peça valiosa simultaneamente
        if (knight_attacks & (1u64 << king_square)) != 0 {
            // Cavalo já ataca o rei
            let valuable_targets = (board.queens | board.rooks) & our_pieces;
            if (knight_attacks & valuable_targets) != 0 {
                threat_score += 45; // Garfo rei + peça valiosa
            }
        } else {
            // Verifica se o cavalo pode mover para atacar rei + peça valiosa
            let potential_forks = find_knight_fork_squares(board, color, king_square, knight_sq);
            threat_score += potential_forks * 15;
        }
    }
    
    threat_score
}

/// Encontra casas onde cavalo pode fazer garfo
fn find_knight_fork_squares(board: &Board, color: Color, king_square: usize, knight_square: u8) -> i32 {
    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let valuable_pieces = (board.queens | board.rooks | board.bishops) & our_pieces;
    let all_pieces = board.white_pieces | board.black_pieces;
    
    let mut fork_squares = 0;
    
    // Para cada casa que o cavalo pode alcançar em 1 movimento
    let knight_moves = crate::moves::knight::get_knight_attacks_lookup(knight_square);
    let mut moves_bb = knight_moves & !all_pieces; // Casas livres
    
    while moves_bb != 0 {
        let sq = moves_bb.trailing_zeros() as u8;
        moves_bb &= moves_bb - 1;
        
        let attacks_from_square = crate::moves::knight::get_knight_attacks_lookup(sq);
        
        // Se desta casa atacaria o rei E uma peça valiosa
        if (attacks_from_square & (1u64 << king_square)) != 0 && 
           (attacks_from_square & valuable_pieces) != 0 {
            fork_squares += 1;
        }
    }
    
    fork_squares
}

/// Avalia potencial de ataques descobertos
fn evaluate_discovered_attack_potential(board: &Board, color: Color, king_square: usize) -> i32 {
    let enemy_color = !color;
    let enemy_pieces = if enemy_color == Color::White { board.white_pieces } else { board.black_pieces };
    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    
    let mut threat_score = 0;
    let all_pieces = board.white_pieces | board.black_pieces;
    
    // Verifica linhas/diagonais do rei para peças inimigas que podem atacar descoberto
    let directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ];
    
    for &(rank_delta, file_delta) in &directions {
        if let Some(threat) = check_discovered_attack_line(
            board, color, king_square, rank_delta, file_delta
        ) {
            threat_score += threat;
        }
    }
    
    threat_score
}

/// Verifica ataques descobertos numa linha específica
fn check_discovered_attack_line(
    board: &Board,
    color: Color,
    king_square: usize,
    rank_delta: i8,
    file_delta: i8
) -> Option<i32> {
    let king_rank = king_square / 8;
    let king_file = king_square % 8;
    let enemy_color = !color;
    let enemy_pieces = if enemy_color == Color::White { board.white_pieces } else { board.black_pieces };
    
    let mut current_rank = king_rank as i8 + rank_delta;
    let mut current_file = king_file as i8 + file_delta;
    let mut pieces_in_line = Vec::new();
    
    // Percorre a linha/diagonal
    while current_rank >= 0 && current_rank < 8 && current_file >= 0 && current_file < 8 {
        let square = (current_rank as usize) * 8 + (current_file as usize);
        let square_bb = 1u64 << square;
        
        if (board.white_pieces | board.black_pieces) & square_bb != 0 {
            pieces_in_line.push(square);
        }
        
        current_rank += rank_delta;
        current_file += file_delta;
    }
    
    // Se há exatamente 2 peças na linha, verifica ataque descoberto
    if pieces_in_line.len() == 2 {
        let blocking_piece = pieces_in_line[0];
        let attacking_piece = pieces_in_line[1];
        
        // A peça que ataca deve ser inimiga e do tipo certo (torre/dama para linha, bispo/dama para diagonal)
        let attacking_piece_bb = 1u64 << attacking_piece;
        if (enemy_pieces & attacking_piece_bb) != 0 {
            let is_diagonal = rank_delta.abs() == file_delta.abs();
            let can_attack = if is_diagonal {
                (board.bishops | board.queens) & attacking_piece_bb != 0
            } else {
                (board.rooks | board.queens) & attacking_piece_bb != 0
            };
            
            if can_attack {
                return Some(30); // Ataque descoberto possível
            }
        }
    }
    
    None
}

// Métodos auxiliares para padrões específicos

/// Verifica bateria Dama + Torre
fn has_queen_rook_battery(board: &Board, color: Color, king_square: usize) -> bool {
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let queens = board.queens & pieces;
    let rooks = board.rooks & pieces;
    
    // Implementação simplificada - na prática seria mais detalhada
    queens != 0 && rooks != 0 && are_pieces_coordinated(queens, rooks, king_square)
}

/// Verifica bateria Dama + Bispo
fn has_queen_bishop_battery(board: &Board, color: Color, king_square: usize) -> bool {
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let queens = board.queens & pieces;
    let bishops = board.bishops & pieces;
    
    queens != 0 && bishops != 0 && are_pieces_coordinated(queens, bishops, king_square)
}

/// Verifica duplo ataque de torres
fn has_double_rook_attack(board: &Board, color: Color, king_square: usize) -> bool {
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let rooks = board.rooks & pieces;
    
    rooks.count_ones() >= 2 && 
    count_pieces_attacking_square(board, color, king_square, rooks) >= 2
}

/// Verifica se peças estão coordenadas
fn are_pieces_coordinated(pieces1: Bitboard, pieces2: Bitboard, target_square: usize) -> bool {
    // Implementação simplificada
    pieces1 != 0 && pieces2 != 0
}

/// Conta peças atacando uma casa
fn count_pieces_attacking_square(
    board: &Board,
    color: Color,
    target_square: usize,
    piece_bb: Bitboard
) -> i32 {
    let mut count = 0;
    let mut bb = piece_bb;
    
    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        
        if piece_attacks_square(board, sq, target_square as u8, color) {
            count += 1;
        }
    }
    
    count
}

/// Obtém número de peças atacando uma zona
fn get_pieces_attacking_zone(board: &Board, color: Color, zone: Bitboard) -> i32 {
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let mut attacking_pieces = 0;
    let all_pieces = board.white_pieces | board.black_pieces;
    
    let mut piece_bb = pieces;
    while piece_bb != 0 {
        let sq = piece_bb.trailing_zeros() as u8;
        piece_bb &= piece_bb - 1;
        
        let attacks = get_piece_attacks(board, sq, color);
        if (attacks & zone) != 0 {
            attacking_pieces += 1;
        }
    }
    
    attacking_pieces
}

/// Verifica sacrifício clássico em h7/h2
fn can_sacrifice_on_h7_h2(board: &Board, color: Color, king_square: usize) -> bool {
    let target_square = if color == Color::White { 15 } else { 48 }; // h2 ou h7
    let enemy_color = !color;
    let enemy_pieces = if enemy_color == Color::White { board.white_pieces } else { board.black_pieces };
    
    // Verifica se há bispo inimigo que pode sacrificar
    let enemy_bishops = board.bishops & enemy_pieces;
    
    if (enemy_bishops & (1u64 << target_square)) == 0 {
        return false; // Não há bispo na casa de sacrifício
    }
    
    // Verifica se o rei está na posição vulnerável
    let king_file = king_square % 8;
    let king_rank = king_square / 8;
    
    if color == Color::White {
        king_rank == 0 && king_file >= 6 // Rei no lado do rei
    } else {
        king_rank == 7 && king_file >= 6
    }
}

/// Verifica sacrifício de cavalo para garfo
fn can_sacrifice_knight_fork(board: &Board, color: Color, king_square: usize) -> bool {
    // Implementação simplificada
    let enemy_color = !color;
    let enemy_pieces = if enemy_color == Color::White { board.white_pieces } else { board.black_pieces };
    let enemy_knights = board.knights & enemy_pieces;
    
    enemy_knights != 0 // Se há cavalos inimigos, há potencial
}

/// Verifica sacrifício de qualidade
fn can_sacrifice_exchange(board: &Board, color: Color, king_square: usize) -> bool {
    let enemy_color = !color;
    let enemy_pieces = if enemy_color == Color::White { board.white_pieces } else { board.black_pieces };
    let enemy_rooks = board.rooks & enemy_pieces;
    
    enemy_rooks != 0 && king_square < 64 // Condição básica
}

/// Verifica padrão do "sacrifício grego"
fn has_greek_gift_pattern(board: &Board, color: Color, king_square: usize) -> bool {
    let enemy_color = !color;
    let enemy_pieces = if enemy_color == Color::White { board.white_pieces } else { board.black_pieces };
    
    // Deve haver bispo E cavalo inimigos próximos ao rei
    let enemy_bishops = board.bishops & enemy_pieces;
    let enemy_knights = board.knights & enemy_pieces;
    let king_zone = get_king_zone(king_square, color);
    
    (enemy_bishops & king_zone) != 0 && (enemy_knights & king_zone) != 0
}

// ============================
// SISTEMA DE AVALIAÇÃO DINÂMICA
// ============================

/// Avaliação dinâmica baseada em contexto temporal
pub struct DynamicKingSafety {
    pub time_pressure_factor: f32,
    pub material_imbalance_factor: f32,
    pub tactical_complexity: f32,
    pub endgame_transition: f32,
}

impl Default for DynamicKingSafety {
    fn default() -> Self {
        Self {
            time_pressure_factor: 1.0,
            material_imbalance_factor: 1.0,
            tactical_complexity: 1.0,
            endgame_transition: 1.0,
        }
    }
}

/// Avalia segurança do rei com fatores dinâmicos
pub fn evaluate_dynamic_king_safety(
    board: &Board,
    color: Color,
    game_phase: &GamePhase,
    dynamic_factors: &DynamicKingSafety
) -> i32 {
    let base_safety = evaluate_king_safety(board, color, game_phase);
    
    // Aplica fatores dinâmicos
    let mut adjusted_safety = base_safety as f32;
    
    adjusted_safety *= dynamic_factors.time_pressure_factor;
    adjusted_safety *= dynamic_factors.material_imbalance_factor;
    adjusted_safety *= dynamic_factors.tactical_complexity;
    adjusted_safety *= dynamic_factors.endgame_transition;
    
    adjusted_safety as i32
}

/// Calcula fatores dinâmicos baseados na posição
pub fn calculate_dynamic_factors(
    board: &Board,
    color: Color,
    game_phase: &GamePhase
) -> DynamicKingSafety {
    let mut factors = DynamicKingSafety::default();
    
    // Fator de pressão temporal
    factors.time_pressure_factor = calculate_time_pressure_factor(board, color);
    
    // Fator de desequilíbrio material
    factors.material_imbalance_factor = calculate_material_imbalance_factor(board, color);
    
    // Complexidade tática
    factors.tactical_complexity = calculate_tactical_complexity(board, color);
    
    // Transição para endgame
    factors.endgame_transition = calculate_endgame_transition_factor(board, game_phase);
    
    factors
}

/// Calcula fator de pressão temporal
fn calculate_time_pressure_factor(board: &Board, color: Color) -> f32 {
    // Em situações de pressão (muitos atacantes), segurança é mais crítica
    let attackers = count_total_attackers(board, color);
    
    if attackers >= 4 {
        1.5 // Aumenta importância da segurança
    } else if attackers >= 2 {
        1.2
    } else {
        1.0
    }
}

/// Calcula fator de desequilíbrio material
fn calculate_material_imbalance_factor(board: &Board, color: Color) -> f32 {
    let our_material = calculate_material_value(board, color);
    let enemy_material = calculate_material_value(board, !color);
    
    let imbalance = (our_material as f32 - enemy_material as f32) / our_material.max(1) as f32;
    
    if imbalance < -0.2 {
        // Estamos em desvantagem material - segurança é crucial
        1.3
    } else if imbalance > 0.2 {
        // Vantagem material - podemos relaxar um pouco
        0.8
    } else {
        1.0
    }
}

/// Calcula complexidade tática
fn calculate_tactical_complexity(board: &Board, color: Color) -> f32 {
    let mut complexity = 0;
    
    // Mais peças = mais complexidade
    let total_pieces = (board.white_pieces | board.black_pieces).count_ones();
    complexity += total_pieces as i32;
    
    // Peças centralizadas aumentam complexidade
    let central_pieces = ((board.white_pieces | board.black_pieces) & EXTENDED_CENTER).count_ones();
    complexity += (central_pieces * 2) as i32;
    
    // Ataques múltiplos aumentam complexidade
    let attackers = count_total_attackers(board, color);
    complexity += attackers * 3;
    
    // Normaliza para fator 0.8 - 1.5
    let normalized = (complexity as f32 / 50.0).min(1.5).max(0.8);
    normalized
}

/// Calcula fator de transição para endgame
fn calculate_endgame_transition_factor(board: &Board, game_phase: &GamePhase) -> f32 {
    match game_phase {
        GamePhase::Opening => 1.2, // Segurança muito importante
        GamePhase::EarlyMiddlegame => 1.1,
        GamePhase::Middlegame => 1.0,
        GamePhase::LateMiddlegame => 0.9,
        GamePhase::EarlyEndgame => 0.7, // Segurança menos importante
        GamePhase::Endgame => 0.5,
        GamePhase::LateEndgame => 0.3,
        GamePhase::PureEndgame => 0.2,
        GamePhase::TheoreticalEndgame => 0.1,
    }
}

/// Conta total de atacantes
fn count_total_attackers(board: &Board, color: Color) -> i32 {
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let king_bb = board.kings & pieces;
    
    if king_bb == 0 {
        return 0;
    }
    
    let king_square = king_bb.trailing_zeros() as usize;
    let (_, attackers, _) = count_enemy_attackers_detailed(board, color, king_square);
    
    attackers
}

/// Calcula valor material de uma cor
fn calculate_material_value(board: &Board, color: Color) -> i32 {
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    
    let pawns = (board.pawns & pieces).count_ones() as i32 * 100;
    let knights = (board.knights & pieces).count_ones() as i32 * 320;
    let bishops = (board.bishops & pieces).count_ones() as i32 * 330;
    let rooks = (board.rooks & pieces).count_ones() as i32 * 500;
    let queens = (board.queens & pieces).count_ones() as i32 * 900;
    
    pawns + knights + bishops + rooks + queens
}

// ============================
// SISTEMA DE ALERTAS DE PERIGO
// ============================

/// Níveis de perigo para o rei
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DangerLevel {
    Safe,        // Seguro
    Cautious,    // Cautela
    Dangerous,   // Perigoso
    Critical,    // Crítico
    Desperate,   // Desesperador
}

/// Análise completa de perigo
pub struct DangerAnalysis {
    pub level: DangerLevel,
    pub immediate_threats: Vec<String>,
    pub defensive_moves: Vec<String>,
    pub escape_squares: i32,
    pub attacker_count: i32,
    pub safety_score: i32,
}

/// Analisa nível de perigo do rei
pub fn analyze_king_danger(
    board: &Board,
    color: Color,
    game_phase: &GamePhase
) -> DangerAnalysis {
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let king_bb = board.kings & pieces;
    
    if king_bb == 0 {
        return DangerAnalysis {
            level: DangerLevel::Desperate,
            immediate_threats: vec!["King not found!".to_string()],
            defensive_moves: vec![],
            escape_squares: 0,
            attacker_count: 0,
            safety_score: -1000,
        };
    }
    
    let king_square = king_bb.trailing_zeros() as usize;
    let safety_analysis = analyze_king_safety_detailed(board, color, king_square, game_phase);
    let attack_patterns = evaluate_king_attack_patterns(board, color, king_square);
    
    let mut analysis = DangerAnalysis {
        level: DangerLevel::Safe,
        immediate_threats: Vec::new(),
        defensive_moves: Vec::new(),
        escape_squares: count_escape_squares(board, color, king_square as u8),
        attacker_count: safety_analysis.total_attackers,
        safety_score: safety_analysis.total_score(),
    };
    
    // Determina nível de perigo
    if safety_analysis.total_score() <= -200 || safety_analysis.total_attackers >= 4 {
        analysis.level = DangerLevel::Desperate;
    } else if safety_analysis.total_score() <= -150 || safety_analysis.total_attackers >= 3 {
        analysis.level = DangerLevel::Critical;
    } else if safety_analysis.total_score() <= -100 || safety_analysis.total_attackers >= 2 {
        analysis.level = DangerLevel::Dangerous;
    } else if safety_analysis.total_score() <= -50 || safety_analysis.total_attackers >= 1 {
        analysis.level = DangerLevel::Cautious;
    }
    
    // Identifica ameaças específicas
    if attack_patterns.mating_net_pressure > 50 {
        analysis.immediate_threats.push("Mating net detected!".to_string());
    }
    if attack_patterns.back_rank_weakness > 60 {
        analysis.immediate_threats.push("Back rank mate threat!".to_string());
    }
    if attack_patterns.sacrificial_threats > 40 {
        analysis.immediate_threats.push("Sacrificial attack possible!".to_string());
    }
    if attack_patterns.knight_fork_threats > 35 {
        analysis.immediate_threats.push("Knight fork threat!".to_string());
    }
    
    // Sugere movimentos defensivos
    if analysis.escape_squares == 0 {
        analysis.defensive_moves.push("Create escape squares for king".to_string());
    }
    if safety_analysis.pawn_shield_score < -30 {
        analysis.defensive_moves.push("Reinforce pawn shield".to_string());
    }
    if safety_analysis.defense_weight < 20 {
        analysis.defensive_moves.push("Bring defenders near king".to_string());
    }
    
    analysis
}

/// Sistema de cache para avaliações de segurança
pub struct KingSafetyCache {
    position_hash: u64,
    white_analysis: Option<KingSafetyAnalysis>,
    black_analysis: Option<KingSafetyAnalysis>,
}

impl KingSafetyCache {
    pub fn new() -> Self {
        Self {
            position_hash: 0,
            white_analysis: None,
            black_analysis: None,
        }
    }
    
    pub fn get_or_compute(
        &mut self,
        board: &Board,
        color: Color,
        game_phase: &GamePhase
    ) -> KingSafetyAnalysis {
        let current_hash = self.compute_position_hash(board);
        
        if self.position_hash != current_hash {
            // Cache miss - recomputa tudo
            self.position_hash = current_hash;
            self.white_analysis = None;
            self.black_analysis = None;
        }
        
        // Calcula king_square antes de pegar referência mutável
        let king_square = self.get_king_square(board, color);
        
        let cached_analysis = match color {
            Color::White => &mut self.white_analysis,
            Color::Black => &mut self.black_analysis,
        };
        
        if let Some(analysis) = cached_analysis {
            *analysis
        } else {
            let new_analysis = analyze_king_safety_detailed(
                board, color, king_square, game_phase
            );
            *cached_analysis = Some(new_analysis);
            new_analysis
        }
    }
    
    fn compute_position_hash(&self, board: &Board) -> u64 {
        // Hash simplificado - na prática usaria Zobrist
        board.white_pieces.wrapping_mul(31) ^ board.black_pieces.wrapping_mul(37)
    }
    
    fn get_king_square(&self, board: &Board, color: Color) -> usize {
        let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
        let king_bb = board.kings & pieces;
        
        if king_bb == 0 {
            return 0;
        }
        
        king_bb.trailing_zeros() as usize
    }
}

/// Função principal para avaliação completa de king safety
pub fn comprehensive_king_safety_evaluation(
    board: &Board,
    color: Color,
    game_phase: &GamePhase,
    cache: &mut KingSafetyCache
) -> (i32, DangerAnalysis, KingAttackPatterns) {
    // Análise básica com cache
    let _basic_analysis = cache.get_or_compute(board, color, game_phase);
    
    // Análise de padrões de ataque
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let king_bb = board.kings & pieces;
    let king_square = if king_bb == 0 { 0 } else { king_bb.trailing_zeros() as usize };
    
    let attack_patterns = evaluate_king_attack_patterns(board, color, king_square);
    
    // Análise de perigo
    let danger_analysis = analyze_king_danger(board, color, game_phase);
    
    // Calcula score final considerando fatores dinâmicos
    let dynamic_factors = calculate_dynamic_factors(board, color, game_phase);
    let final_score = evaluate_dynamic_king_safety(board, color, game_phase, &dynamic_factors);
    
    (final_score, danger_analysis, attack_patterns)
}