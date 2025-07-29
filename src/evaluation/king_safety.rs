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
        GamePhase::Middlegame => analyze_middlegame_king_safety(board, color, king_square),
        GamePhase::Endgame => KingSafetyAnalysis::default(), // Segurança não importa no endgame
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
