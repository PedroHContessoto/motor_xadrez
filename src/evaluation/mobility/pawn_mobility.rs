// Mobilidade e estratégia avançada de peões
use crate::types::{Color, Bitboard};
use super::{MobilityContext, GamePhase, utils::*};
use super::super::game_phase::{interpolate_phase_i32, GamePhaseInfo};
use super::super::pawn_structure;

/// Pesos para diferentes aspectos da estrutura de peões por fase do jogo
#[derive(Debug, Clone, Copy)]
pub struct PawnWeights {
    pub passed_pawn: [i32; 3],      // [opening, middlegame, endgame]
    pub doubled_pawn: [i32; 3],     // Penalidade por peões dobrados
    pub isolated_pawn: [i32; 3],    // Penalidade por peões isolados
    pub backward_pawn: [i32; 3],    // Penalidade por peões atrasados
    pub pawn_chain: [i32; 3],       // Bônus por cadeia de peões
    pub pawn_storm: [i32; 3],       // Bônus por tempestade de peões
    pub mobility_per_square: [i32; 3], // Valor por casa de avanço
}

impl Default for PawnWeights {
    fn default() -> Self {
        PawnWeights {
            passed_pawn: [10, 30, 60],      // Peões passados mais valiosos no final
            doubled_pawn: [-15, -20, -10],  // Menos prejudicial no final
            isolated_pawn: [-10, -15, -20], // Mais prejudicial no final
            backward_pawn: [-8, -12, -15],  // Mais prejudicial no final
            pawn_chain: [8, 12, 6],         // Menos importante no final
            pawn_storm: [15, 20, 5],        // Principalmente meio-jogo
            mobility_per_square: [2, 3, 4], // Avanços mais valiosos no final
        }
    }
}

/// Resultado específico da análise de peões
#[derive(Debug, Clone, Copy)]
pub struct PawnAnalysis {
    pub passed_pawns: i32,
    pub doubled_pawns: i32,
    pub isolated_pawns: i32,
    pub backward_pawns: i32,
    pub pawn_chains: i32,
    pub pawn_storms: i32,
    pub pawn_mobility: i32,
    pub king_safety_contribution: i32,
}

impl PawnAnalysis {
    pub fn new() -> Self {
        PawnAnalysis {
            passed_pawns: 0,
            doubled_pawns: 0,
            isolated_pawns: 0,
            backward_pawns: 0,
            pawn_chains: 0,
            pawn_storms: 0,
            pawn_mobility: 0,
            king_safety_contribution: 0,
        }
    }
}

/// Avaliação avançada de mobilidade de peões (usa tapered evaluation)
pub fn evaluate_pawn_mobility_advanced(context: &MobilityContext) -> i32 {
    let weights = PawnWeights::default();
    
    // Usa o novo sistema de pawn_structure.rs para análise unificada
    let main_score = pawn_structure::evaluate_pawn_structure(&context.board, context.color);
    
    // Adiciona análise específica de mobilidade
    let mobility_analysis = analyze_pawn_mobility_specific(context);
    
    // Aplica pesos baseados na fase do jogo usando tapered evaluation
    let mut score = main_score; // Estrutura base já calculada
    
    // Adiciona análise específica de mobilidade interpolada
    score += interpolate_phase_i32(
        mobility_analysis.pawn_mobility * weights.mobility_per_square[0],
        mobility_analysis.pawn_mobility * weights.mobility_per_square[2],
        &context.phase_info
    );
    
    score += interpolate_phase_i32(
        mobility_analysis.pawn_storms * weights.pawn_storm[0],
        mobility_analysis.pawn_storms * weights.pawn_storm[2],
        &context.phase_info
    );
    
    score += mobility_analysis.king_safety_contribution;
    
    score
}

/// Análise específica de mobilidade de peões (complementa pawn_structure.rs)
pub fn analyze_pawn_mobility_specific(context: &MobilityContext) -> PawnAnalysis {
    let mut analysis = PawnAnalysis::new();
    let our_pawns = context.board.pawns & context.our_pieces;
    let enemy_pawns = context.board.pawns & context.enemy_pieces;
    
    if our_pawns == 0 {
        return analysis; // Não há peões para analisar
    }
    
    // Foca apenas em aspectos específicos de mobilidade, evitando duplicação
    let pawn_squares = pawn_structure::get_set_bits_simple(our_pawns);
    for &pawn_sq in &pawn_squares {
        // Mobilidade específica do peão (movimento, capturas)
        analysis.pawn_mobility += calculate_pawn_mobility(pawn_sq as u8, context);
    }
    
    // Análise de tempestades de peões (específica para mobilidade tática)
    analysis.pawn_storms = evaluate_pawn_storms_enhanced(our_pawns, context);
    
    // Contribuição para segurança do rei (específica para mobilidade)
    analysis.king_safety_contribution = evaluate_pawn_king_safety_mobility(our_pawns, context);
    
    // Análise dinâmica de cadeias (nova funcionalidade)
    analysis.pawn_chains = evaluate_dynamic_pawn_chains(our_pawns, enemy_pawns, context);
    
    analysis
}

/// Verifica se peão é passado
fn is_passed_pawn(pawn_sq: u8, color: Color, _our_pawns: Bitboard, enemy_pawns: Bitboard) -> bool {
    let file = pawn_sq % 8;
    let rank = pawn_sq / 8;
    
    // Verifica arquivos adjacentes e o próprio arquivo
    for check_file in (file.saturating_sub(1))..=(file.saturating_add(1)).min(7) {
        let file_mask = get_file_mask(check_file);
        let file_pawns = enemy_pawns & file_mask;
        
        if file_pawns != 0 {
            // Verifica se há peão inimigo que pode bloquear
            let enemy_pawn_squares = get_set_bits(file_pawns);
            for enemy_sq in enemy_pawn_squares {
                let enemy_rank = enemy_sq / 8;
                
                let blocks_advancement = if color == Color::White {
                    enemy_rank > rank // Peão inimigo está à frente
                } else {
                    enemy_rank < rank // Peão inimigo está à frente
                };
                
                if blocks_advancement {
                    return false;
                }
            }
        }
    }
    
    true
}

/// Calcula bônus por peão passado baseado na proximidade de promoção
fn calculate_passed_pawn_bonus(pawn_sq: u8, color: Color) -> i32 {
    let rank = pawn_sq / 8;
    let distance_to_promotion = if color == Color::White {
        7 - rank
    } else {
        rank
    };
    
    match distance_to_promotion {
        1 => 30,  // Prestes a promover
        2 => 20,  // 2 casas para promover
        3 => 12,  // 3 casas para promover
        4 => 8,   // 4 casas para promover
        5 => 5,   // 5 casas para promover
        6 => 3,   // 6 casas para promover
        _ => 1,   // Muito longe
    }
}

/// Verifica se peão está dobrado
fn is_doubled_pawn(file: u8, pawn_sq: u8, our_pawns: Bitboard) -> bool {
    let file_mask = get_file_mask(file);
    let file_pawns = our_pawns & file_mask;
    
    // Remove o próprio peão e verifica se ainda há peões no arquivo
    let file_pawns_without_self = file_pawns & !(1u64 << pawn_sq);
    file_pawns_without_self != 0
}

/// Verifica se peão está isolado
fn is_isolated_pawn(file: u8, our_pawns: Bitboard) -> bool {
    // Verifica arquivos adjacentes
    let left_file = if file > 0 { get_file_mask(file - 1) } else { 0 };
    let right_file = if file < 7 { get_file_mask(file + 1) } else { 0 };
    
    let adjacent_pawns = our_pawns & (left_file | right_file);
    adjacent_pawns == 0
}

/// Verifica se peão está atrasado
fn is_backward_pawn(pawn_sq: u8, color: Color, our_pawns: Bitboard, enemy_pawns: Bitboard) -> bool {
    let file = pawn_sq % 8;
    let rank = pawn_sq / 8;
    
    // Verifica se peões aliados nos arquivos adjacentes estão mais avançados
    let left_file = if file > 0 { file - 1 } else { return false; };
    let right_file = if file < 7 { file + 1 } else { return false; };
    
    for adj_file in [left_file, right_file] {
        let file_mask = get_file_mask(adj_file);
        let file_pawns = our_pawns & file_mask;
        
        if file_pawns != 0 {
            let allied_squares = get_set_bits(file_pawns);
            for allied_sq in allied_squares {
                let allied_rank = allied_sq / 8;
                
                let is_behind = if color == Color::White {
                    rank < allied_rank // Peão está atrás do aliado
                } else {
                    rank > allied_rank
                };
                
                if !is_behind {
                    return false; // Não está atrasado
                }
            }
        }
    }
    
    // Verifica se casa à frente está controlada por inimigo
    let advance_sq = if color == Color::White {
        if rank >= 7 { return false; } // Não pode avançar
        pawn_sq + 8
    } else {
        if rank <= 0 { return false; } // Não pode avançar
        pawn_sq - 8
    };
    
    let advance_bb = 1u64 << advance_sq;
    let enemy_controlled = super::compute_pawn_attacks(enemy_pawns, !color);
    
    (enemy_controlled & advance_bb) != 0
}

/// Calcula mobilidade específica do peão
fn calculate_pawn_mobility(pawn_sq: u8, context: &MobilityContext) -> i32 {
    let rank = pawn_sq / 8;
    let mut mobility = 0;
    
    // Casa à frente
    let one_step = if context.color == Color::White {
        if rank >= 7 { return mobility; }
        pawn_sq + 8
    } else {
        if rank <= 0 { return mobility; }
        pawn_sq - 8
    };
    
    // Verifica se pode avançar uma casa
    if (context.all_pieces & (1u64 << one_step)) == 0 {
        mobility += 1;
        
        // Se pode avançar uma casa, verifica duplo avanço (peões iniciais)
        let can_double_move = if context.color == Color::White {
            rank == 1 && rank < 6
        } else {
            rank == 6 && rank > 1
        };
        
        if can_double_move {
            let two_steps = if context.color == Color::White {
                pawn_sq + 16
            } else {
                pawn_sq - 16
            };
            
            if (context.all_pieces & (1u64 << two_steps)) == 0 {
                mobility += 1;
            }
        }
    }
    
    // Considera capturas como "mobilidade tática"
    let pawn_attacks = if context.color == Color::White {
        let left = if pawn_sq % 8 > 0 && rank < 7 { 1u64 << (pawn_sq + 7) } else { 0 };
        let right = if pawn_sq % 8 < 7 && rank < 7 { 1u64 << (pawn_sq + 9) } else { 0 };
        left | right
    } else {
        let left = if pawn_sq % 8 < 7 && rank > 0 { 1u64 << (pawn_sq - 7) } else { 0 };
        let right = if pawn_sq % 8 > 0 && rank > 0 { 1u64 << (pawn_sq - 9) } else { 0 };
        left | right
    };
    
    let possible_captures = pawn_attacks & context.enemy_pieces;
    mobility += possible_captures.count_ones() as i32;
    
    mobility
}

/// Conta cadeias de peões
fn count_pawn_chains(our_pawns: Bitboard, color: Color) -> i32 {
    let mut chains = 0;
    let pawn_squares = get_set_bits(our_pawns);
    
    for &pawn_sq in &pawn_squares {
        // Verifica se este peão está sendo apoiado por outro peão
        let supported_by = if color == Color::White {
            // Peões que apoiam de trás
            let left_support = if pawn_sq % 8 > 0 && pawn_sq >= 9 { 1u64 << (pawn_sq - 9) } else { 0 };
            let right_support = if pawn_sq % 8 < 7 && pawn_sq >= 7 { 1u64 << (pawn_sq - 7) } else { 0 };
            left_support | right_support
        } else {
            let left_support = if pawn_sq % 8 < 7 && pawn_sq + 9 < 64 { 1u64 << (pawn_sq + 9) } else { 0 };
            let right_support = if pawn_sq % 8 > 0 && pawn_sq + 7 < 64 { 1u64 << (pawn_sq + 7) } else { 0 };
            left_support | right_support
        };
        
        if (supported_by & our_pawns) != 0 {
            chains += 1;
        }
    }
    
    chains
}

/// Avalia tempestades de peões aprimoradas (específica para mobilidade tática)
fn evaluate_pawn_storms_enhanced(our_pawns: Bitboard, context: &MobilityContext) -> i32 {
    let enemy_king = context.board.kings & context.enemy_pieces;
    if enemy_king == 0 { return 0; }
    
    let enemy_king_sq = enemy_king.trailing_zeros() as u8;
    let enemy_king_file = enemy_king_sq % 8;
    
    let mut storm_value = 0;
    let pawn_squares = get_set_bits(our_pawns);
    
    for &pawn_sq in &pawn_squares {
        let pawn_file = pawn_sq % 8;
        let pawn_rank = pawn_sq / 8;
        
        // Peões próximos ao rei inimigo são mais valiosos
        let file_distance = (pawn_file as i8 - enemy_king_file as i8).abs();
        if file_distance <= 2 {
            let advancement_bonus = if context.color == Color::White {
                pawn_rank
            } else {
                7 - pawn_rank
            };
            
            storm_value += (8 - file_distance as i32) * (advancement_bonus as i32 + 1);
        }
    }
    
    storm_value / 4 // Normaliza o valor
}

/// Avalia contribuição dos peões para segurança do rei
fn evaluate_pawn_king_safety(our_pawns: Bitboard, context: &MobilityContext) -> i32 {
    let our_king = context.board.kings & context.our_pieces;
    if our_king == 0 { return 0; }
    
    let king_sq = our_king.trailing_zeros() as u8;
    let king_file = king_sq % 8;
    let king_rank = king_sq / 8;
    
    let mut safety_bonus = 0;
    
    // Verifica escudo de peões na frente do rei
    let shield_ranks = if context.color == Color::White {
        if king_rank < 6 { king_rank + 1..=king_rank + 2 } else { return 0; }
    } else {
        if king_rank > 1 { king_rank - 2..=king_rank - 1 } else { return 0; }
    };
    
    for shield_rank in shield_ranks {
        for shield_file in (king_file.saturating_sub(1))..=(king_file.saturating_add(1)).min(7) {
            let shield_sq = shield_rank * 8 + shield_file;
            let shield_bb = 1u64 << shield_sq;
            
            if (our_pawns & shield_bb) != 0 {
                safety_bonus += 5; // Bônus por peão protetor
            } else {
                safety_bonus -= 3; // Penalidade por buraco no escudo
            }
        }
    }
    
    safety_bonus
}

/// Versão específica de mobilidade para segurança do rei
fn evaluate_pawn_king_safety_mobility(our_pawns: Bitboard, context: &MobilityContext) -> i32 {
    let our_king = context.board.kings & context.our_pieces;
    if our_king == 0 { return 0; }
    
    let king_sq = our_king.trailing_zeros() as u8;
    let king_file = king_sq % 8;
    let king_rank = king_sq / 8;
    
    let mut mobility_safety_bonus = 0;
    
    // Analisa mobilidade dos peões do escudo (podem avançar para defender?)
    let shield_files = [
        king_file.saturating_sub(1),
        king_file,
        king_file.saturating_add(1).min(7)
    ];
    
    for &file in &shield_files {
        let file_mask = 0x0101010101010101u64 << file;
        let file_pawns = our_pawns & file_mask;
        
        if file_pawns != 0 {
            let file_pawn_squares = pawn_structure::get_set_bits_simple(file_pawns);
            for pawn_sq in file_pawn_squares {
                let pawn_rank = pawn_sq / 8;
                let pawn_sq_u8 = pawn_sq as u8;
                
                // Peão pode avançar para melhorar defesa?
                let mobility = calculate_pawn_mobility(pawn_sq_u8, context);
                if mobility > 0 {
                    // Bônus se peão pode avançar para melhorar escudo
                    if context.color == Color::White && (pawn_rank as usize) < (king_rank as usize + 2) {
                        mobility_safety_bonus += 3;
                    } else if context.color == Color::Black && (pawn_rank as usize) > (king_rank as usize).saturating_sub(2) {
                        mobility_safety_bonus += 3;
                    }
                }
            }
        }
    }
    
    mobility_safety_bonus
}

/// Avaliação dinâmica de cadeias de peões considerando alavancas e quebras
fn evaluate_dynamic_pawn_chains(our_pawns: Bitboard, enemy_pawns: Bitboard, context: &MobilityContext) -> i32 {
    let mut dynamic_chains = 0;
    let pawn_squares = pawn_structure::get_set_bits_simple(our_pawns);
    
    for &pawn_sq in &pawn_squares {
        let file = pawn_sq % 8;
        let rank = pawn_sq / 8;
        let pawn_sq_u8 = pawn_sq as u8;
        
        // Verifica se este peão está sendo apoiado por outro peão
        let supported_by = if context.color == Color::White {
            let left_support = if file > 0 && rank > 0 { 1u64 << ((rank - 1) * 8 + file - 1) } else { 0 };
            let right_support = if file < 7 && rank > 0 { 1u64 << ((rank - 1) * 8 + file + 1) } else { 0 };
            left_support | right_support
        } else {
            let left_support = if file < 7 && rank < 7 { 1u64 << ((rank + 1) * 8 + file + 1) } else { 0 };
            let right_support = if file > 0 && rank < 7 { 1u64 << ((rank + 1) * 8 + file - 1) } else { 0 };
            left_support | right_support
        };
        
        if (supported_by & our_pawns) != 0 {
            let mut chain_strength = 1;
            
            // Análise de vulnerabilidade da cadeia
            
            // 1. Verifica alavanca (peão inimigo atacando base da cadeia)
            let lever_threats = evaluate_lever_threats(pawn_sq_u8, enemy_pawns, context);
            chain_strength -= lever_threats;
            
            // 2. Verifica se a cadeia pode ser quebrada facilmente
            let breakability = evaluate_chain_breakability(pawn_sq_u8, enemy_pawns, context);
            chain_strength -= breakability;
            
            // 3. Bônus se a cadeia pode avançar coordenadamente
            let advance_potential = evaluate_chain_advance_potential(pawn_sq_u8, context);
            chain_strength += advance_potential;
            
            // 4. Verifica controle de casas importantes
            let square_control = evaluate_chain_square_control(pawn_sq_u8, context);
            chain_strength += square_control;
            
            dynamic_chains += chain_strength.max(0);
        }
    }
    
    dynamic_chains
}

/// Avalia ameaças de alavanca a uma cadeia de peões
fn evaluate_lever_threats(pawn_sq: u8, enemy_pawns: Bitboard, context: &MobilityContext) -> i32 {
    let file = pawn_sq % 8;
    let rank = pawn_sq / 8;
    let mut lever_threats = 0;
    
    // Verifica peões inimigos que podem atacar a base da cadeia
    let threat_squares = if context.color == Color::White {
        // Para peões brancos, ameaças vêm de baixo
        let mut threats = Vec::new();
        if rank > 0 {
            if file > 0 { threats.push((rank - 1) * 8 + file - 1); }
            if file < 7 { threats.push((rank - 1) * 8 + file + 1); }
        }
        threats
    } else {
        // Para peões pretos, ameaças vêm de cima
        let mut threats = Vec::new();
        if rank < 7 {
            if file > 0 { threats.push((rank + 1) * 8 + file - 1); }
            if file < 7 { threats.push((rank + 1) * 8 + file + 1); }
        }
        threats
    };
    
    for &threat_sq in &threat_squares {
        let threat_bb = 1u64 << threat_sq;
        if (enemy_pawns & threat_bb) != 0 {
            lever_threats += 1;
        }
    }
    
    lever_threats
}

/// Avalia quão facilmente uma cadeia pode ser quebrada
fn evaluate_chain_breakability(pawn_sq: u8, enemy_pawns: Bitboard, context: &MobilityContext) -> i32 {
    let file = pawn_sq % 8;
    let rank = pawn_sq / 8;
    let mut breakability = 0;
    
    // Verifica se há peões inimigos avançados nos arquivos adjacentes
    let adjacent_files = [
        if file > 0 { Some(file - 1) } else { None },
        if file < 7 { Some(file + 1) } else { None },
    ];
    
    for adj_file_opt in &adjacent_files {
        if let Some(adj_file) = adj_file_opt {
            let file_mask = 0x0101010101010101u64 << adj_file;
            let adj_enemy_pawns = enemy_pawns & file_mask;
            
            if adj_enemy_pawns != 0 {
                let enemy_pawn_squares = pawn_structure::get_set_bits_simple(adj_enemy_pawns);
                for enemy_sq in enemy_pawn_squares {
                    let enemy_rank = enemy_sq / 8;
                    
                    // Peão inimigo avançado representa ameaça de quebra
                    let threat_level = if context.color == Color::White {
                        if (enemy_rank as usize) >= rank as usize { 1 } else { 0 }
                    } else {
                        if (enemy_rank as usize) <= rank as usize { 1 } else { 0 }
                    };
                    
                    breakability += threat_level;
                }
            }
        }
    }
    
    breakability
}

/// Avalia potencial de avanço coordenado da cadeia
fn evaluate_chain_advance_potential(pawn_sq: u8, context: &MobilityContext) -> i32 {
    let mobility = calculate_pawn_mobility(pawn_sq, context);
    let file = pawn_sq % 8;
    let our_pawns = context.board.pawns & context.our_pieces;
    
    // Verifica se peões adjacentes também podem avançar (avanço coordenado)
    let adjacent_files = [
        if file > 0 { Some(file - 1) } else { None },
        if file < 7 { Some(file + 1) } else { None },
    ];
    
    let mut coordinated_advance = 0;
    for adj_file_opt in &adjacent_files {
        if let Some(adj_file) = adj_file_opt {
            let file_mask = 0x0101010101010101u64 << adj_file;
            let adj_our_pawns = our_pawns & file_mask;
            
            if adj_our_pawns != 0 {
                let adj_pawn_squares = pawn_structure::get_set_bits_simple(adj_our_pawns);
                for adj_pawn_sq in adj_pawn_squares {
                    let adj_mobility = calculate_pawn_mobility(adj_pawn_sq as u8, context);
                    if adj_mobility > 0 {
                        coordinated_advance += 1;
                    }
                }
            }
        }
    }
    
    if mobility > 0 && coordinated_advance > 0 {
        return coordinated_advance;
    }
    
    0
}

/// Avalia controle de casas importantes por cadeia de peões
fn evaluate_chain_square_control(pawn_sq: u8, context: &MobilityContext) -> i32 {
    let file = pawn_sq % 8;
    let rank = pawn_sq / 8;
    let mut square_control = 0;
    
    // Verifica controle de casas importantes (centro, casas de passagem)
    let controlled_squares = if context.color == Color::White {
        let mut squares = Vec::new();
        if rank < 7 {
            if file > 0 { squares.push((rank + 1) * 8 + file - 1); }
            if file < 7 { squares.push((rank + 1) * 8 + file + 1); }
        }
        squares
    } else {
        let mut squares = Vec::new();
        if rank > 0 {
            if file > 0 { squares.push((rank - 1) * 8 + file - 1); }
            if file < 7 { squares.push((rank - 1) * 8 + file + 1); }
        }
        squares
    };
    
    // Centro do tabuleiro
    let center_squares = [27, 28, 35, 36]; // d4, e4, d5, e5
    let extended_center = [18, 19, 20, 21, 26, 29, 34, 37, 42, 43, 44, 45]; // c3-f6 área
    
    for &controlled_sq in &controlled_squares {
        if center_squares.contains(&(controlled_sq as u8)) {
            square_control += 2; // Controle do centro é muito valioso
        } else if extended_center.contains(&(controlled_sq as u8)) {
            square_control += 1; // Controle do centro expandido
        }
    }
    
    square_control
}