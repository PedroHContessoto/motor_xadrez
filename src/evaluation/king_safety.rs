// Sistema avançado de segurança do rei com Attack Units
use crate::{board::Board, types::{Color, Bitboard}};
use super::game_phase::GamePhase;
use std::collections::HashMap;
use std::sync::Mutex;

// Cache para king safety evaluation
lazy_static::lazy_static! {
    static ref KING_SAFETY_CACHE: Mutex<HashMap<(u64, Color, u8), i32>> = 
        Mutex::new(HashMap::with_capacity(6000));
}

// === CONSTANTES PARA ATTACK UNITS SYSTEM ===
const CENTRAL_SQUARES: Bitboard = (1u64 << 27) | (1u64 << 28) | (1u64 << 35) | (1u64 << 36);

// Valores de Attack Units por tipo de peça
const ATTACK_UNITS: [i32; 6] = [
    0,   // Pawn (não conta como atacante direto)
    2,   // Knight
    2,   // Bishop  
    3,   // Rook
    5,   // Queen
    0,   // King (não conta)
];

// Pesos para casas ao redor do rei (zona de perigo)
const KING_DANGER_ZONE: [i32; 3] = [
    100, // Casas adjacentes ao rei
    50,  // Casas a 2 casas do rei
    25,  // Casas a 3 casas do rei
];

// Thresholds para conversão de Attack Units em penalidades
const DANGER_THRESHOLDS: [(i32, i32); 6] = [
    (0,   0),    // Sem perigo
    (5,   -50),  // Perigo leve
    (10,  -150), // Perigo moderado
    (20,  -300), // Perigo alto
    (35,  -500), // Perigo muito alto
    (50,  -800), // Perigo extremo
];

/// Estrutura para análise completa de segurança do rei
#[derive(Debug, Clone)]
pub struct KingSafetyAnalysis {
    pub attack_units: i32,
    pub weak_squares: Bitboard,
    pub pawn_shield_score: i32,
    pub storm_danger: i32,
    pub tropism_penalty: i32,
    pub total_danger: i32,
}

/// Avalia a segurança do rei com sistema avançado de Attack Units COM CACHE
pub fn evaluate_king_safety(board: &Board, color: Color, game_phase: &GamePhase) -> i32 {
    let game_phase_id = match game_phase {
        GamePhase::Opening => 1,
        GamePhase::EarlyMiddlegame => 2,
        GamePhase::Middlegame => 3,
        GamePhase::LateMiddlegame => 4,
        GamePhase::EarlyEndgame => 5,
        GamePhase::Endgame => 6,
        GamePhase::PureEndgame => 7,
    };
    
    let cache_key = (board.zobrist_hash, color, game_phase_id);
    
    // Verifica cache primeiro
    if let Ok(cache) = KING_SAFETY_CACHE.try_lock() {
        if let Some(&cached_result) = (*cache).get(&cache_key) {
            return cached_result;
        }
    }
    
    // Cálculo original completo
    let safety_score = if matches!(game_phase, GamePhase::Endgame | GamePhase::PureEndgame) {
        evaluate_endgame_king_safety(board, color)
    } else {
        let analysis = analyze_king_safety_comprehensive(board, color, game_phase);
        
        // Conversão de attack units para penalidade usando curva não-linear
        let danger_penalty = convert_attack_units_to_penalty(analysis.attack_units);
        
        // Score final combinando todos os fatores
        let total_score = analysis.pawn_shield_score 
                         - danger_penalty 
                         - analysis.storm_danger 
                         - analysis.tropism_penalty;
        
        // Aplica fator de escala baseado na fase do jogo
        let phase_factor = match game_phase {
            GamePhase::Opening => 0.7,           // Menos crítico na abertura
            GamePhase::EarlyMiddlegame => 1.0,   // Muito crítico
            GamePhase::Middlegame => 1.2,        // Máxima criticidade
            GamePhase::LateMiddlegame => 1.0,    // Ainda crítico
            GamePhase::EarlyEndgame => 0.5,      // Menos crítico
            _ => 0.2,                            // Mínimo no endgame
        };
        
        (total_score as f32 * phase_factor) as i32
    };
    
    // Armazena no cache
    if let Ok(mut cache) = KING_SAFETY_CACHE.try_lock() {
        if (*cache).len() >= 6000 {
            (*cache).clear(); // LRU simples: limpa quando cheio
        }
        (*cache).insert(cache_key, safety_score);
    }
    
    safety_score
}

/// Análise comprehensive de segurança do rei
fn analyze_king_safety_comprehensive(board: &Board, color: Color, game_phase: &GamePhase) -> KingSafetyAnalysis {
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let king_bb = board.kings & pieces;
    
    if king_bb == 0 {
        return KingSafetyAnalysis {
            attack_units: 0,
            weak_squares: 0,
            pawn_shield_score: 0,
            storm_danger: 0,
            tropism_penalty: 0,
            total_danger: 0,
        };
    }
    
    let king_square = king_bb.trailing_zeros() as u8;
    let rank = king_square / 8;
    let file = king_square % 8;
    
    // === ANÁLISE COMPLETA DE ATTACK UNITS ===
    let attack_units = calculate_attack_units(board, color, king_square);
    let weak_squares = identify_weak_squares_around_king(board, color, king_square);
    
    // === ANÁLISE DE PAWN SHIELD ===
    let pawn_shield_score = evaluate_pawn_shield_advanced(board, color, king_square);
    
    // === ANÁLISE DE PAWN STORM ===
    let storm_danger = evaluate_pawn_storm_danger(board, color, king_square);
    
    // === ANÁLISE DE TROPISMO ===
    let tropism_penalty = calculate_piece_tropism(board, color, king_square);
    
    // === PENALIDADES ESPECIAIS ===
    let mut total_danger = attack_units;
    
    // Rei no centro durante meio-jogo
    if (CENTRAL_SQUARES & king_bb) != 0 {
        total_danger += 15; // Adiciona perigo por exposição central
    }
    
    // Rei muito avançado
    let advanced_penalty = match color {
        Color::White => if rank > 2 { (rank - 2) as i32 * 8 } else { 0 },
        Color::Black => if rank < 5 { (5 - rank) as i32 * 8 } else { 0 },
    };
    total_danger += advanced_penalty;
    
    KingSafetyAnalysis {
        attack_units,
        weak_squares,
        pawn_shield_score,
        storm_danger,
        tropism_penalty,
        total_danger,
    }
}

/// Calcula Attack Units baseado em peças inimigas que atacam a zona do rei
fn calculate_attack_units(board: &Board, color: Color, king_square: u8) -> i32 {
    let enemy_pieces = if color == Color::White { board.black_pieces } else { board.white_pieces };
    let mut attack_units = 0;
    
    // Zona de perigo ao redor do rei (3x3 centrado no rei)
    let danger_zone = get_king_danger_zone(king_square);
    
    // === ANÁLISE DE CAVALOS ===
    let enemy_knights = board.knights & enemy_pieces;
    let mut knights = enemy_knights;
    while knights != 0 {
        let knight_sq = knights.trailing_zeros() as u8;
        knights &= knights - 1;
        
        let knight_attacks = crate::moves::knight::get_knight_attacks_lookup(knight_sq);
        if (knight_attacks & danger_zone) != 0 {
            attack_units += ATTACK_UNITS[1]; // Knight = 2 units
            
            // Bônus por atacar casas críticas adjacentes ao rei
            let king_adjacent = crate::moves::king::get_king_attacks_lookup(king_square);
            if (knight_attacks & king_adjacent) != 0 {
                attack_units += 1; // Bônus por atacar casa adjacente
            }
        }
    }
    
    // === ANÁLISE DE BISPOS ===
    let enemy_bishops = board.bishops & enemy_pieces;
    let mut bishops = enemy_bishops;
    while bishops != 0 {
        let bishop_sq = bishops.trailing_zeros() as u8;
        bishops &= bishops - 1;
        
        let bishop_attacks = crate::moves::sliding::get_bishop_attacks(bishop_sq, board.white_pieces | board.black_pieces);
        if (bishop_attacks & danger_zone) != 0 {
            attack_units += ATTACK_UNITS[2]; // Bishop = 2 units
            
            // Bônus por diagonal longa apontando para o rei
            if (bishop_attacks & (1u64 << king_square)) != 0 {
                attack_units += 2; // Ataque direto ao rei
            }
        }
    }
    
    // === ANÁLISE DE TORRES ===
    let enemy_rooks = board.rooks & enemy_pieces;
    let mut rooks = enemy_rooks;
    while rooks != 0 {
        let rook_sq = rooks.trailing_zeros() as u8;
        rooks &= rooks - 1;
        
        let rook_attacks = crate::moves::sliding::get_rook_attacks(rook_sq, board.white_pieces | board.black_pieces);
        if (rook_attacks & danger_zone) != 0 {
            attack_units += ATTACK_UNITS[3]; // Rook = 3 units
            
            // Bônus por ataque direto na mesma fileira/coluna
            if (rook_attacks & (1u64 << king_square)) != 0 {
                attack_units += 3; // Ataque direto poderoso
            }
        }
    }
    
    // === ANÁLISE DE RAINHAS ===
    let enemy_queens = board.queens & enemy_pieces;
    let mut queens = enemy_queens;
    while queens != 0 {
        let queen_sq = queens.trailing_zeros() as u8;
        queens &= queens - 1;
        
        // Rainha ataca como torre + bispo
        let rook_attacks = crate::moves::sliding::get_rook_attacks(queen_sq, board.white_pieces | board.black_pieces);
        let bishop_attacks = crate::moves::sliding::get_bishop_attacks(queen_sq, board.white_pieces | board.black_pieces);
        let queen_attacks = rook_attacks | bishop_attacks;
        if (queen_attacks & danger_zone) != 0 {
            attack_units += ATTACK_UNITS[4]; // Queen = 5 units
            
            // Bônus massivo por ataque direto da rainha
            if (queen_attacks & (1u64 << king_square)) != 0 {
                attack_units += 5; // Ataque direto extremamente perigoso
            }
        }
    }
    
    attack_units
}

/// Obtém zona de perigo 3x3 ao redor do rei
fn get_king_danger_zone(king_square: u8) -> Bitboard {
    let king_attacks = crate::moves::king::get_king_attacks_lookup(king_square);
    let king_bit = 1u64 << king_square;
    
    // Zona inclui o rei e todas as casas adjacentes
    king_attacks | king_bit
}

/// Identifica casas fracas ao redor do rei
fn identify_weak_squares_around_king(board: &Board, color: Color, king_square: u8) -> Bitboard {
    let our_pawns = board.pawns & if color == Color::White { board.white_pieces } else { board.black_pieces };
    let king_zone = get_king_danger_zone(king_square);
    
    // Casas fracas são aquelas na zona do rei não defendidas por peões
    let pawn_defended = get_pawn_defended_squares(our_pawns, color);
    king_zone & !pawn_defended
}

/// Calcula casas defendidas por peões
fn get_pawn_defended_squares(pawns: Bitboard, color: Color) -> Bitboard {
    match color {
        Color::White => {
            ((pawns & 0xFEFEFEFEFEFEFEFE) << 9) | // Diagonal direita
            ((pawns & 0x7F7F7F7F7F7F7F7F) << 7)   // Diagonal esquerda
        },
        Color::Black => {
            ((pawns & 0xFEFEFEFEFEFEFEFE) >> 7) | // Diagonal direita
            ((pawns & 0x7F7F7F7F7F7F7F7F) >> 9)   // Diagonal esquerda
        }
    }
}

/// Avaliação avançada de pawn shield
fn evaluate_pawn_shield_advanced(board: &Board, color: Color, king_square: u8) -> i32 {
    let our_pawns = board.pawns & if color == Color::White { board.white_pieces } else { board.black_pieces };
    let king_file = king_square % 8;
    let mut shield_score = 0;
    
    // Avalia peões nas 3 colunas próximas ao rei
    for file_offset in -1i8..=1i8 {
        let check_file = (king_file as i8 + file_offset) as u8;
        if check_file < 8 {
            let file_mask = 0x0101010101010101u64 << check_file;
            let pawns_on_file = our_pawns & file_mask;
            
            if pawns_on_file != 0 {
                let closest_pawn = if color == Color::White {
                    pawns_on_file.leading_zeros() as u8 // Peão mais próximo do rei
                } else {
                    pawns_on_file.trailing_zeros() as u8
                };
                
                let pawn_rank = closest_pawn / 8;
                let king_rank = king_square / 8;
                
                // Bônus baseado na distância do peão ao rei
                let distance = (pawn_rank as i32 - king_rank as i32).abs();
                let shield_bonus = match distance {
                    0 => 0,  // Peão na mesma fileira (ruim)
                    1 => 25, // Peão uma fileira à frente (ótimo)
                    2 => 15, // Peão duas fileiras à frente (bom)
                    _ => 5,  // Peão muito distante (fraco)
                };
                
                shield_score += shield_bonus;
                
                // Bônus extra para peão na coluna do rei
                if file_offset == 0 {
                    shield_score += 5;
                }
            } else {
                // Penalidade por ausência de peão na coluna
                shield_score -= 20;
            }
        }
    }
    
    shield_score
}

/// Avalia perigo de pawn storm inimigo
fn evaluate_pawn_storm_danger(board: &Board, color: Color, king_square: u8) -> i32 {
    let enemy_pawns = board.pawns & if color == Color::White { board.black_pieces } else { board.white_pieces };
    let king_file = king_square % 8;
    let king_rank = king_square / 8;
    let mut storm_danger = 0;
    
    // Verifica peões inimigos avançados nas colunas próximas
    for file_offset in -2i8..=2i8 {
        let check_file = (king_file as i8 + file_offset) as u8;
        if check_file < 8 {
            let file_mask = 0x0101010101010101u64 << check_file;
            let pawns_on_file = enemy_pawns & file_mask;
            
            if pawns_on_file != 0 {
                let closest_pawn = if color == Color::White {
                    pawns_on_file.trailing_zeros() as u8 // Peão inimigo mais avançado
                } else {
                    pawns_on_file.leading_zeros() as u8
                };
                
                let pawn_rank = closest_pawn / 8;
                let advance_distance = (pawn_rank as i32 - king_rank as i32).abs();
                
                // Perigo aumenta quanto mais próximo o peão
                let storm_penalty = match advance_distance {
                    0..=1 => 30, // Muito perigoso
                    2 => 20,     // Perigoso
                    3 => 10,     // Moderado
                    _ => 5,      // Leve
                };
                
                storm_danger += storm_penalty;
                
                // Penalty extra para peões na coluna do rei
                if file_offset.abs() <= 1 {
                    storm_danger += 10;
                }
            }
        }
    }
    
    storm_danger
}

/// Calcula tropismo (proximidade de peças inimigas)
fn calculate_piece_tropism(board: &Board, color: Color, king_square: u8) -> i32 {
    let enemy_pieces = if color == Color::White { board.black_pieces } else { board.white_pieces };
    let mut tropism = 0;
    
    // Analisa todas as peças inimigas
    let mut pieces = enemy_pieces;
    while pieces != 0 {
        let piece_sq = pieces.trailing_zeros() as u8;
        pieces &= pieces - 1;
        
        let distance = calculate_square_distance(king_square, piece_sq);
        
        // Penalidade baseada na proximidade e tipo da peça
        let piece_penalty = if (board.queens & (1u64 << piece_sq)) != 0 {
            // Rainha próxima é muito perigosa
            match distance {
                1..=2 => 25,
                3..=4 => 15,
                5..=6 => 8,
                _ => 0,
            }
        } else if (board.rooks & (1u64 << piece_sq)) != 0 {
            // Torre próxima
            match distance {
                1..=2 => 15,
                3..=4 => 8,
                _ => 0,
            }
        } else if (board.knights & (1u64 << piece_sq)) != 0 {
            // Cavalo próximo (especialmente perigoso a distância 2-3)
            match distance {
                1..=3 => 12,
                4..=5 => 6,
                _ => 0,
            }
        } else if (board.bishops & (1u64 << piece_sq)) != 0 {
            // Bispo próximo
            match distance {
                1..=3 => 8,
                4..=5 => 4,
                _ => 0,
            }
        } else {
            0
        };
        
        tropism += piece_penalty;
    }
    
    tropism
}

/// Converte attack units em penalidade usando curva não-linear
fn convert_attack_units_to_penalty(attack_units: i32) -> i32 {
    // Encontra o threshold apropriado
    for &(threshold, penalty) in &DANGER_THRESHOLDS {
        if attack_units >= threshold {
            continue;
        } else {
            // Interpola entre thresholds para curva suave
            if let Some(&(prev_threshold, prev_penalty)) = DANGER_THRESHOLDS.iter()
                .rev()
                .find(|&&(t, _)| t <= attack_units) {
                
                if threshold == prev_threshold {
                    return -prev_penalty;
                }
                
                let ratio = (attack_units - prev_threshold) as f32 / (threshold - prev_threshold) as f32;
                let interpolated = prev_penalty as f32 + (penalty - prev_penalty) as f32 * ratio;
                return -interpolated as i32;
            }
            break;
        }
    }
    
    // Para valores extremos acima do último threshold
    -DANGER_THRESHOLDS.last().unwrap().1
}

/// Segurança do rei no endgame (simplificada)
fn evaluate_endgame_king_safety(board: &Board, color: Color) -> i32 {
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let king_bb = board.kings & pieces;
    
    if king_bb == 0 {
        return 0;
    }
    
    let king_square = king_bb.trailing_zeros() as u8;
    let mut score = 0;
    
    // No endgame, rei ativo no centro é bom
    let file = king_square % 8;
    let rank = king_square / 8;
    
    // Centralização
    let center_distance = ((file as i32 - 3).abs() + (rank as i32 - 3).abs()) / 2;
    score += (4 - center_distance) * 10;
    
    // Mobilidade do rei
    let king_attacks = crate::moves::king::get_king_attacks_lookup(king_square);
    let mobility = king_attacks.count_ones() as i32;
    score += mobility * 5;
    
    score
}

/// Calcula distância entre duas casas
fn calculate_square_distance(sq1: u8, sq2: u8) -> u8 {
    let file1 = sq1 % 8;
    let rank1 = sq1 / 8;
    let file2 = sq2 % 8;
    let rank2 = sq2 / 8;
    
    let file_diff = (file1 as i8 - file2 as i8).abs() as u8;
    let rank_diff = (rank1 as i8 - rank2 as i8).abs() as u8;
    
    file_diff.max(rank_diff)
}

// === FUNÇÕES AUXILIARES DE COMPATIBILIDADE ===

fn evaluate_pawn_shield(board: &Board, color: Color, _king_square: usize, rank: usize, file: usize) -> i32 {
    // Implementação simplificada para compatibilidade
    evaluate_pawn_shield_advanced(board, color, (rank * 8 + file) as u8)
}

fn evaluate_enemy_attackers(board: &Board, color: Color, king_square: usize) -> i32 {
    calculate_attack_units(board, color, king_square as u8) * 5
}

fn evaluate_tropism(board: &Board, color: Color, king_square: usize) -> i32 {
    calculate_piece_tropism(board, color, king_square as u8)
}

fn has_castled(board: &Board, color: Color) -> bool {
    let king_bb = board.kings & if color == Color::White { board.white_pieces } else { board.black_pieces };
    
    if king_bb == 0 {
        return false;
    }
    
    let king_square = king_bb.trailing_zeros() as u8;
    
    match color {
        Color::White => king_square == 2 || king_square == 6, // c1 ou g1
        Color::Black => king_square == 58 || king_square == 62, // c8 ou g8
    }
}

fn can_castle(board: &Board, color: Color) -> bool {
    match color {
        Color::White => (board.castling_rights & 0x03) != 0,
        Color::Black => (board.castling_rights & 0x0C) != 0,
    }
}