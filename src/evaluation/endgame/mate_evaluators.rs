// Avaliadores específicos para finais de mate teóricos
use crate::{board::Board, types::{Color, PieceKind}};

const MATE_VALUE: i32 = 99999;

/// Resultado da avaliação de um final teórico
#[derive(Debug, Clone)]
pub struct EndgameEvalResult {
    pub score: i32,
    pub is_theoretical: bool,
    pub best_technique: String,
    pub estimated_moves_to_mate: Option<u8>,
}

/// Avaliador para KQ vs K (Rei + Dama vs Rei) - Consolidado e melhorado
pub fn evaluate_kq_vs_k(board: &Board, winning_color: Color) -> EndgameEvalResult {
    let (our_pieces, enemy_pieces) = if winning_color == Color::White {
        (board.white_pieces, board.black_pieces)
    } else {
        (board.black_pieces, board.white_pieces)
    };

    // Localiza reis e dama
    let our_king = (board.kings & our_pieces).trailing_zeros() as u8;
    let enemy_king = (board.kings & enemy_pieces).trailing_zeros() as u8;
    let our_queen = (board.queens & our_pieces).trailing_zeros() as u8;

    let mut score = 9000; // Base winning score melhorado (era 900)
    
    // 1. FORÇA REI INIMIGO PARA A BORDA (técnica fundamental)
    let enemy_king_file = enemy_king % 8;
    let enemy_king_rank = enemy_king / 8;
    
    // Distância da borda (quanto mais longe da borda, pior)
    let distance_to_edge = enemy_king_file.min(7 - enemy_king_file).min(enemy_king_rank.min(7 - enemy_king_rank));
    
    score += (7 - distance_to_edge as i32) * 150; // Bônus por rei próximo da borda (melhorado)
    
    // Bônus especial se rei está na borda
    if distance_to_edge == 0 {
        score += 200;
        // Se na corner, ainda melhor
        if matches!(enemy_king, 0 | 7 | 56 | 63) {
            score += 300;
        }
    }

    // 2. APROXIMAÇÃO DOS REIS (técnica fundamental)
    let king_distance = calculate_king_distance(our_king, enemy_king);
    score += (7 - king_distance as i32) * 20; // Bônus por aproximação
    
    // 3. CONTROLE DE CASAS CENTRAIS PELA DAMA (melhorado)
    let queen_central_control = evaluate_queen_central_control(our_queen, enemy_king);
    score += queen_central_control * 30;
    
    // 4. DAMA DEVE ESTAR A DISTÂNCIA IDEAL
    let queen_king_distance = calculate_distance(our_queen, enemy_king);
    if queen_king_distance == 2 || queen_king_distance == 3 {
        score += 50; // Distância ideal
    } else if queen_king_distance > 5 {
        score -= 30; // Muito longe
    }
    
    // 4. VERIFICA SE ESTÁ PRÓXIMO DO MATE
    let moves_to_mate = estimate_kq_vs_k_mate_distance(our_king, our_queen, enemy_king);
    
    // Se mate está próximo, aumenta drasticamente a avaliação
    if moves_to_mate <= 5 {
        score += (MATE_VALUE / 2) - (moves_to_mate as i32 * 1000);
    }

    EndgameEvalResult {
        score,
        is_theoretical: true,
        best_technique: "Force enemy king to edge, approach with own king, deliver mate".to_string(),
        estimated_moves_to_mate: if moves_to_mate <= 10 { Some(moves_to_mate) } else { None },
    }
}

/// Avaliador para KR vs K (Rei + Torre vs Rei)
pub fn evaluate_kr_vs_k(board: &Board, winning_color: Color) -> EndgameEvalResult {
    let (our_pieces, enemy_pieces) = if winning_color == Color::White {
        (board.white_pieces, board.black_pieces)
    } else {
        (board.black_pieces, board.white_pieces)
    };

    let our_king = (board.kings & our_pieces).trailing_zeros() as u8;
    let enemy_king = (board.kings & enemy_pieces).trailing_zeros() as u8;
    let our_rook = (board.rooks & our_pieces).trailing_zeros() as u8;

    let mut score = 500; // Vantagem material da torre
    
    // 1. TÉCNICA DE CORTE - Torre corta rei da fuga
    let rook_cuts_king = evaluates_cutting_technique(our_rook, enemy_king);
    if rook_cuts_king {
        score += 50;
    }
    
    // 2. REI NA 6ª FILEIRA (técnica clássica)
    let our_king_rank = our_king / 8;
    let enemy_king_rank = enemy_king / 8;
    
    if winning_color == Color::White && our_king_rank >= 5 {
        score += 30; // Rei branco na 6ª+ fileira
    } else if winning_color == Color::Black && our_king_rank <= 2 {
        score += 30; // Rei preto na 3ª- fileira
    }
    
    // 3. OPOSIÇÃO COM APOIO DA TORRE
    if has_supported_opposition(our_king, enemy_king, our_rook) {
        score += 40;
    }
    
    // 4. FORÇA REI INIMIGO PARA A BORDA
    let enemy_king_file = enemy_king % 8;
    let enemy_king_rank = enemy_king / 8;
    let distance_to_edge = enemy_king_file.min(7 - enemy_king_file).min(enemy_king_rank.min(7 - enemy_king_rank));
    
    score -= (distance_to_edge as i32) * 12;
    
    let moves_to_mate = estimate_kr_vs_k_mate_distance(our_king, our_rook, enemy_king);
    
    if moves_to_mate <= 8 {
        score += (MATE_VALUE / 3) - (moves_to_mate as i32 * 800);
    }

    EndgameEvalResult {
        score,
        is_theoretical: true,
        best_technique: "Cut with rook, advance king to 6th rank, deliver mate".to_string(),
        estimated_moves_to_mate: if moves_to_mate <= 15 { Some(moves_to_mate) } else { None },
    }
}

/// Avaliador para KP vs K (Rei + Peão vs Rei)
pub fn evaluate_kp_vs_k(board: &Board, winning_color: Color) -> EndgameEvalResult {
    let (our_pieces, enemy_pieces) = if winning_color == Color::White {
        (board.white_pieces, board.black_pieces)
    } else {
        (board.black_pieces, board.white_pieces)
    };

    let our_king = (board.kings & our_pieces).trailing_zeros() as u8;
    let enemy_king = (board.kings & enemy_pieces).trailing_zeros() as u8;
    
    // Encontra o peão mais avançado
    let our_pawns = board.pawns & our_pieces;
    let mut most_advanced_pawn = 0u8;
    let mut best_pawn_rank = if winning_color == Color::White { 0 } else { 7 };
    
    let mut pawn_bb = our_pawns;
    while pawn_bb != 0 {
        let pawn_sq = pawn_bb.trailing_zeros() as u8;
        pawn_bb &= pawn_bb - 1;
        
        let pawn_rank = pawn_sq / 8;
        if (winning_color == Color::White && pawn_rank > best_pawn_rank) ||
           (winning_color == Color::Black && pawn_rank < best_pawn_rank) {
            best_pawn_rank = pawn_rank;
            most_advanced_pawn = pawn_sq;
        }
    }

    let mut score = 100; // Valor base do peão
    
    // 1. REGRA DO QUADRADO
    let square_rule_result = evaluate_square_rule(most_advanced_pawn, enemy_king, winning_color);
    match square_rule_result {
        SquareRuleResult::PawnWins => score += 400,
        SquareRuleResult::KingCatches => score -= 200,
        SquareRuleResult::Unclear => score += 0,
    }
    
    // 2. OPOSIÇÃO (fundamental em finais de peões)
    let opposition_value = evaluate_pawn_endgame_opposition(our_king, enemy_king, most_advanced_pawn);
    score += opposition_value;
    
    // 3. APOIO DO REI AO PEÃO
    let king_support = evaluate_king_pawn_support(our_king, most_advanced_pawn, winning_color);
    score += king_support;
    
    // 4. PROXIMIDADE DA PROMOÇÃO
    let promotion_distance = if winning_color == Color::White {
        7 - best_pawn_rank
    } else {
        best_pawn_rank
    };
    
    score += (7 - promotion_distance as i32) * 30;
    
    // Se peão está muito próximo da promoção
    if promotion_distance <= 2 {
        score += 200 + (2 - promotion_distance as i32) * 100;
    }

    EndgameEvalResult {
        score,
        is_theoretical: true,
        best_technique: "Support pawn with king, gain opposition, promote".to_string(),
        estimated_moves_to_mate: if promotion_distance <= 3 { Some(promotion_distance + 2) } else { None },
    }
}

/// Avaliador para KBB vs K (Rei + Dois Bispos vs Rei) - Consolidado de theoretical.rs
pub fn evaluate_kbb_vs_k(board: &Board, winning_color: Color) -> EndgameEvalResult {
    let (our_pieces, enemy_pieces) = if winning_color == Color::White {
        (board.white_pieces, board.black_pieces)
    } else {
        (board.black_pieces, board.white_pieces)
    };

    let enemy_king = (board.kings & enemy_pieces).trailing_zeros() as u8;
    let mut score = 3000; // Base winning score
    
    // Força para qualquer canto primeiro
    let distance_to_corner = calculate_distance_to_nearest_corner(enemy_king);
    score += (7 - distance_to_corner as i32) * 150;
    
    // Bônus se rei está em canto correto (simplificado)
    if distance_to_corner == 0 {
        score += 500;
    }
    
    EndgameEvalResult {
        score,
        is_theoretical: true,
        best_technique: "Force king to correct corner, coordinate bishops".to_string(),
        estimated_moves_to_mate: Some((distance_to_corner + 5).min(19)),
    }
}

/// Avaliador para KBN vs K (Rei + Bispo + Cavalo vs Rei) - Consolidado de theoretical.rs
pub fn evaluate_kbn_vs_k(board: &Board, winning_color: Color) -> EndgameEvalResult {
    let (our_pieces, enemy_pieces) = if winning_color == Color::White {
        (board.white_pieces, board.black_pieces)
    } else {
        (board.black_pieces, board.white_pieces)
    };

    let enemy_king = (board.kings & enemy_pieces).trailing_zeros() as u8;
    let mut score = 2000; // Base winning score
    
    // Determina cor do bispo e canto correto (simplificado - força para canto mais próximo)
    let distance_to_corner = calculate_distance_to_nearest_corner(enemy_king);
    score += (7 - distance_to_corner as i32) * 100;
    
    // Bônus especial se em canto correto
    if distance_to_corner == 0 {
        score += 400;
    }
    
    EndgameEvalResult {
        score,
        is_theoretical: true,
        best_technique: "Deletang W pattern, force to correct corner".to_string(),
        estimated_moves_to_mate: Some((distance_to_corner * 3 + 8).min(33)),
    }
}

/// Avalia a regra do quadrado para peões passados
#[derive(Debug)]
enum SquareRuleResult {
    PawnWins,
    KingCatches,
    Unclear,
}

fn evaluate_square_rule(pawn_sq: u8, enemy_king: u8, winning_color: Color) -> SquareRuleResult {
    let pawn_file = pawn_sq % 8;
    let pawn_rank = pawn_sq / 8;
    
    let promotion_rank = if winning_color == Color::White { 7 } else { 0 };
    let moves_to_promote = (promotion_rank as i32 - pawn_rank as i32).abs();
    
    let king_file = enemy_king % 8;
    let king_rank = enemy_king / 8;
    
    // Distância do rei até a casa de promoção
    let king_distance_to_promotion = std::cmp::max(
        (king_file as i32 - pawn_file as i32).abs(),
        (king_rank as i32 - promotion_rank as i32).abs()
    );
    
    if moves_to_promote < king_distance_to_promotion {
        SquareRuleResult::PawnWins
    } else if moves_to_promote > king_distance_to_promotion {
        SquareRuleResult::KingCatches
    } else {
        SquareRuleResult::Unclear
    }
}

/// Avalia oposição em finais de peões
fn evaluate_pawn_endgame_opposition(our_king: u8, enemy_king: u8, pawn_sq: u8) -> i32 {
    let our_file = our_king % 8;
    let our_rank = our_king / 8;
    let enemy_file = enemy_king % 8;
    let enemy_rank = enemy_king / 8;
    
    let file_diff = (our_file as i32 - enemy_file as i32).abs();
    let rank_diff = (our_rank as i32 - enemy_rank as i32).abs();
    
    // Oposição direta (fundamental)
    if (file_diff == 0 && rank_diff == 2) || (rank_diff == 0 && file_diff == 2) {
        return 150; // Aumentado de 25 para 150!
    }
    
    // Oposição distante
    if file_diff == 0 && rank_diff % 2 == 0 && rank_diff > 2 {
        return 100;
    }
    
    // Oposição diagonal
    if file_diff == 2 && rank_diff == 2 {
        return 80;
    }
    
    0
}

/// Avalia apoio do rei ao peão
fn evaluate_king_pawn_support(king_sq: u8, pawn_sq: u8, winning_color: Color) -> i32 {
    let king_distance = calculate_distance(king_sq, pawn_sq);
    let base_support = (4 - king_distance as i32).max(0) * 25;
    
    let king_file = king_sq % 8;
    let king_rank = king_sq / 8;
    let pawn_file = pawn_sq % 8;
    let pawn_rank = pawn_sq / 8;
    
    // Bônus se rei está na frente do peão
    let mut bonus = 0;
    if winning_color == Color::White && king_rank > pawn_rank {
        bonus += 30;
    } else if winning_color == Color::Black && king_rank < pawn_rank {
        bonus += 30;
    }
    
    // Bônus se rei está lateralmente protegendo o peão
    if (king_file as i32 - pawn_file as i32).abs() == 1 {
        bonus += 20;
    }
    
    base_support + bonus
}

// === FUNÇÕES AUXILIARES ===

fn calculate_king_distance(king1: u8, king2: u8) -> u8 {
    let file1 = king1 % 8;
    let rank1 = king1 / 8;
    let file2 = king2 % 8;
    let rank2 = king2 / 8;
    
    std::cmp::max(
        (file1 as i32 - file2 as i32).abs(),
        (rank1 as i32 - rank2 as i32).abs()
    ) as u8
}

fn calculate_distance(sq1: u8, sq2: u8) -> u8 {
    let file1 = sq1 % 8;
    let rank1 = sq1 / 8;
    let file2 = sq2 % 8;
    let rank2 = sq2 / 8;
    
    std::cmp::max(
        (file1 as i32 - file2 as i32).abs(),
        (rank1 as i32 - rank2 as i32).abs()
    ) as u8
}

fn evaluate_queen_central_control(queen_sq: u8, enemy_king_sq: u8) -> i32 {
    // Dama controlando centro e próxima do rei inimigo (consolidado de theoretical.rs)
    let distance = calculate_distance(queen_sq, enemy_king_sq);
    (8 - distance as i32).max(0)
}

fn evaluates_cutting_technique(rook_sq: u8, enemy_king: u8) -> bool {
    let rook_file = rook_sq % 8;
    let rook_rank = rook_sq / 8;
    let king_file = enemy_king % 8;
    let king_rank = enemy_king / 8;
    
    // Torre corta se está na mesma fileira/coluna e limita movimentação (melhorado)
    (rook_file == king_file && calculate_distance_to_edge(enemy_king) <= 2 && (rook_rank as i32 - king_rank as i32).abs() >= 2) ||
    (rook_rank == king_rank && calculate_distance_to_edge(enemy_king) <= 2 && (rook_file as i32 - king_file as i32).abs() >= 2)
}

fn has_supported_opposition(our_king: u8, enemy_king: u8, rook_sq: u8) -> bool {
    let king_distance = calculate_king_distance(our_king, enemy_king);
    let rook_supports = rook_supports_king(our_king, rook_sq);
    
    king_distance == 2 && rook_supports
}

fn rook_supports_king(king_sq: u8, rook_sq: u8) -> bool {
    let king_file = king_sq % 8;
    let king_rank = king_sq / 8;
    let rook_file = rook_sq % 8;
    let rook_rank = rook_sq / 8;
    
    // Torre na mesma fileira ou coluna que o rei
    (king_file == rook_file) || (king_rank == rook_rank)
}

fn estimate_kq_vs_k_mate_distance(our_king: u8, our_queen: u8, enemy_king: u8) -> u8 {
    // Estimativa simples baseada na posição
    let king_distance = calculate_king_distance(our_king, enemy_king);
    let edge_distance = calculate_distance_to_edge(enemy_king);
    
    // Fórmula heurística: distância até a borda + necessidade de aproximação
    (edge_distance + king_distance).max(4).min(20)
}

fn estimate_kr_vs_k_mate_distance(our_king: u8, our_rook: u8, enemy_king: u8) -> u8 {
    let king_distance = calculate_king_distance(our_king, enemy_king);
    let edge_distance = calculate_distance_to_edge(enemy_king);
    
    // Torre precisa de mais tempo para cortar e mate
    (edge_distance * 2 + king_distance).max(6).min(25)
}

fn calculate_distance_to_edge(sq: u8) -> u8 {
    let file = sq % 8;
    let rank = sq / 8;
    
    file.min(7 - file).min(rank.min(7 - rank))
}

fn calculate_distance_to_nearest_corner(square: u8) -> u8 {
    let file = square % 8;
    let rank = square / 8;
    
    let corners = [(0, 0), (0, 7), (7, 0), (7, 7)];
    corners.iter()
        .map(|(cf, cr)| {
            let file_diff = (file as i8 - *cf as i8).abs() as u8;
            let rank_diff = (rank as i8 - *cr as i8).abs() as u8;
            file_diff.max(rank_diff)
        })
        .min()
        .unwrap_or(7)
}

/// Interface principal para detectar e avaliar finais teóricos (expandida)
pub fn evaluate_theoretical_endgame(board: &Board) -> Option<EndgameEvalResult> {
    let white_pieces = board.white_pieces.count_ones();
    let black_pieces = board.black_pieces.count_ones();
    
    // Só avalia se é realmente um final com poucas peças (expandido para incluir KBB, KBN)
    if white_pieces + black_pieces > 6 {
        return None;
    }
    
    // Determina qual lado tem vantagem material
    let white_material = count_material_value(board, Color::White);
    let black_material = count_material_value(board, Color::Black);
    
    if white_material > black_material + 200 { // Threshold reduzido para incluir mais finais
        // Brancas têm vantagem
        detect_endgame_type(board, Color::White)
    } else if black_material > white_material + 200 {
        // Pretas têm vantagem  
        detect_endgame_type(board, Color::Black)
    } else {
        None // Material equilibrado, não é final teórico ganho
    }
}

fn count_material_value(board: &Board, color: Color) -> i32 {
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    
    (board.pawns & pieces).count_ones() as i32 * 100 +
    (board.knights & pieces).count_ones() as i32 * 320 +
    (board.bishops & pieces).count_ones() as i32 * 330 +
    (board.rooks & pieces).count_ones() as i32 * 500 +
    (board.queens & pieces).count_ones() as i32 * 900
}

fn detect_endgame_type(board: &Board, winning_color: Color) -> Option<EndgameEvalResult> {
    let our_pieces = if winning_color == Color::White { board.white_pieces } else { board.black_pieces };
    let enemy_pieces = if winning_color == Color::White { board.black_pieces } else { board.white_pieces };
    
    let our_queens = (board.queens & our_pieces).count_ones();
    let our_rooks = (board.rooks & our_pieces).count_ones();
    let our_bishops = (board.bishops & our_pieces).count_ones();
    let our_knights = (board.knights & our_pieces).count_ones();
    let our_pawns = (board.pawns & our_pieces).count_ones();
    let enemy_total = enemy_pieces.count_ones();
    
    // KQ vs K
    if our_queens == 1 && enemy_total == 1 {
        return Some(evaluate_kq_vs_k(board, winning_color));
    }
    
    // KR vs K
    if our_rooks == 1 && our_queens == 0 && enemy_total == 1 {
        return Some(evaluate_kr_vs_k(board, winning_color));
    }
    
    // KBB vs K (dois bispos vs rei nu)
    if our_bishops == 2 && our_queens == 0 && our_rooks == 0 && our_knights == 0 && our_pawns == 0 && enemy_total == 1 {
        return Some(evaluate_kbb_vs_k(board, winning_color));
    }
    
    // KBN vs K (bispo + cavalo vs rei nu)
    if our_bishops == 1 && our_knights == 1 && our_queens == 0 && our_rooks == 0 && our_pawns == 0 && enemy_total == 1 {
        return Some(evaluate_kbn_vs_k(board, winning_color));
    }
    
    // KP vs K
    if our_pawns >= 1 && our_queens == 0 && our_rooks == 0 && enemy_total == 1 {
        return Some(evaluate_kp_vs_k(board, winning_color));
    }
    
    None
}