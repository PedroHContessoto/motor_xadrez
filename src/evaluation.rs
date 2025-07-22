// Ficheiro: src/evaluation.rs
// Descrição: Lógica de avaliação, agora com deteção de peões passados.

use crate::{board::Board, types::*};

// --- As tabelas de peças e valores materiais não mudam ---
const PAWN_TABLE: [i32; 64] = [0,0,0,0,0,0,0,0,50,50,50,50,50,50,50,50,10,10,20,30,30,20,10,10,5,5,10,25,25,10,5,5,0,0,0,20,20,0,0,0,5,-5,-10,0,0,-10,-5,5,5,10,10,-20,-20,10,10,5,0,0,0,0,0,0,0,0];
const KNIGHT_TABLE: [i32; 64] = [-50,-40,-30,-30,-30,-30,-40,-50,-40,-20,0,0,0,0,-20,-40,-30,0,10,15,15,10,0,-30,-30,5,15,20,20,15,5,-30,-30,0,15,20,20,15,0,-30,-30,5,10,15,15,10,5,-30,-40,-20,0,5,5,0,-20,-40,-50,-40,-30,-30,-30,-30,-40,-50];
const BISHOP_TABLE: [i32; 64] = [-20,-10,-10,-10,-10,-10,-10,-20,-10,0,0,0,0,0,0,-10,-10,0,5,10,10,5,0,-10,-10,5,5,10,10,5,5,-10,-10,0,10,10,10,10,0,-10,-10,10,10,10,10,10,10,-10,-10,5,0,0,0,0,5,-10,-20,-10,-10,-10,-10,-10,-10,-20];
const ROOK_TABLE: [i32; 64] = [0,0,0,0,0,0,0,0,5,10,10,10,10,10,10,5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,0,0,0,5,5,0,0,0];
const QUEEN_TABLE: [i32; 64] = [-20,-10,-10,-5,-5,-10,-10,-20,-10,0,0,0,0,0,0,-10,-10,0,5,5,5,5,0,-10,-5,0,5,5,5,5,0,-5,0,0,5,5,5,5,0,-5,-10,5,5,5,5,5,0,-10,-10,0,5,0,0,0,0,-10,-20,-10,-10,-5,-5,-10,-10,-20];
const KING_TABLE_MIDGAME: [i32; 64] = [-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-20,-30,-30,-40,-40,-30,-30,-20,-10,-20,-20,-20,-20,-20,-20,-10,20,20,0,0,0,0,20,20,20,30,10,0,0,10,30,20];
const KING_TABLE_ENDGAME: [i32; 64] = [-50,-40,-30,-20,-20,-30,-40,-50,-30,-20,-10,0,0,-10,-20,-30,-30,-10,20,30,30,20,-10,-30,-30,-10,30,40,40,30,-10,-30,-30,-10,30,40,40,30,-10,-30,-30,-10,20,30,30,20,-10,-30,-30,-30,0,0,0,0,-30,-30,-50,-30,-30,-30,-30,-30,-30,-50];
const MATERIAL_VALUES: [i32; 6] = [100, 320, 330, 500, 900, 20000];

// Constantes para avaliação estratégica
const CENTRAL_SQUARES: Bitboard = (1u64 << 27) | (1u64 << 28) | (1u64 << 35) | (1u64 << 36); // d4, e4, d5, e5
const EXTENDED_CENTER: Bitboard = 0x00003C3C3C3C0000; // c3-f3 to c6-f6

// Pesos para mobilidade por tipo de peça
const MOBILITY_WEIGHTS: [i32; 6] = [0, 4, 4, 2, 1, 0]; // [pawn, knight, bishop, rook, queen, king]

// =======================================================
// PASSO 8: ADICIONANDO BÓNUS PARA PEÕES PASSADOS
// =======================================================

// Bónus para peões passados com base na sua fileira (rank)
// Quanto mais perto da promoção, maior o bónus
const PASSED_PAWN_BONUS: [i32; 8] = [0, 10, 20, 30, 50, 75, 100, 0];

// Máscaras de bits para cada coluna, para ajudar a detetar peões adjacentes
const FILE_MASKS: [Bitboard; 8] = [
    0x0101010101010101, 0x0202020202020202, 0x0404040404040404, 0x0808080808080808,
    0x1010101010101010, 0x2020202020202020, 0x4040404040404040, 0x8080808080808080,
];

// Máscaras que cobrem a frente e as colunas adjacentes para um peão.
// Usado para verificar se há peões inimigos a bloquear o caminho.
static mut PASSED_PAWN_MASKS: [[Bitboard; 64]; 2] = [[0; 64]; 2];

// Função para inicializar as máscaras uma única vez (chamada a partir do main)
pub fn init_evaluation_masks() {
    unsafe {
        for sq in 0..64 {
            let rank = sq / 8;
            let file = sq % 8;

            // Máscara para Brancas
            let mut white_mask: Bitboard = 0;
            for r in (rank + 1)..8 {
                white_mask |= FILE_MASKS[file] & (0xFF << (r * 8)); // Coluna da frente
                if file > 0 { white_mask |= FILE_MASKS[file - 1] & (0xFF << (r * 8)); } // Coluna adjacente esquerda
                if file < 7 { white_mask |= FILE_MASKS[file + 1] & (0xFF << (r * 8)); } // Coluna adjacente direita
            }
            PASSED_PAWN_MASKS[Color::White as usize][sq] = white_mask;

            // Máscara para Pretas
            let mut black_mask: Bitboard = 0;
            for r in 0..rank {
                black_mask |= FILE_MASKS[file] & (0xFF << (r * 8));
                if file > 0 { black_mask |= FILE_MASKS[file - 1] & (0xFF << (r * 8)); }
                if file < 7 { black_mask |= FILE_MASKS[file + 1] & (0xFF << (r * 8)); }
            }
            PASSED_PAWN_MASKS[Color::Black as usize][sq] = black_mask;
        }
    }
}


/// Avaliação dinâmica melhorada baseada na fase do jogo
pub fn evaluate(board: &Board) -> i32 {
    let mut game_phase_score = 0;
    game_phase_score += (board.knights.count_ones() as i32) * 1;
    game_phase_score += (board.bishops.count_ones() as i32) * 1;
    game_phase_score += (board.rooks.count_ones() as i32) * 2;
    game_phase_score += (board.queens.count_ones() as i32) * 4;

    // Fases do jogo mais precisas
    let is_opening = game_phase_score > 20;
    let is_middlegame = game_phase_score > 10 && game_phase_score <= 20;
    let is_endgame = game_phase_score <= 10;

    let white_score = evaluate_color(board, Color::White, is_endgame, is_opening);
    let black_score = evaluate_color(board, Color::Black, is_endgame, is_opening);

    let mut final_score = white_score - black_score;
    
    // Bônus por tempo (encoraja jogadas mais rápidas quando vantajoso)
    if final_score.abs() < 50 {
        // Posição equilibrada - valoriza mobilidade
        final_score += evaluate_tempo(board);
    }
    
    // Ajuste dinâmico baseado na fase
    let phase_multiplier = if is_opening {
        0.8  // Abertura: menos agressivo na avaliação
    } else if is_middlegame {
        1.2  // Meio-jogo: mais agressivo
    } else {
        1.0  // Endgame: avaliação padrão
    };
    
    let adjusted_score = (final_score as f32 * phase_multiplier) as i32;
    
    // Clamp score para evitar valores absurdos
    let absolute_score = adjusted_score.clamp(-10000, 10000);
    
    // Torna relativa ao jogador atual (positivo = jogador atual melhor)
    if board.to_move == Color::White {
        absolute_score
    } else {
        -absolute_score
    }
}
fn evaluate_color(board: &Board, color: Color, is_endgame: bool, is_opening: bool) -> i32 {
    let mut score = 0;
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let king_table = if is_endgame { &KING_TABLE_ENDGAME } else { &KING_TABLE_MIDGAME };

    score += evaluate_pawns(board, board.pawns & pieces, color);
    score += evaluate_piece_type(board.knights & pieces, &KNIGHT_TABLE, PieceKind::Knight, color);
    score += evaluate_piece_type(board.bishops & pieces, &BISHOP_TABLE, PieceKind::Bishop, color);
    score += evaluate_piece_type(board.rooks & pieces, &ROOK_TABLE, PieceKind::Rook, color);
    score += evaluate_piece_type(board.queens & pieces, &QUEEN_TABLE, PieceKind::Queen, color);
    score += evaluate_piece_type(board.kings & pieces, king_table, PieceKind::King, color);
    
    // Avaliações estratégicas adicionais ajustadas por fase
    score += evaluate_mobility(board, color);
    score += evaluate_center_control(board, color, is_opening);
    score += evaluate_king_safety(board, color, is_endgame);
    score += evaluate_connected_passed_pawns(board, board.pawns & pieces, color);
    
    // Avaliações específicas por fase
    if is_opening {
        score += evaluate_development(board, color);
    } else if is_endgame {
        score += evaluate_king_activity(board, color);
    }

    score
}

/// Função específica para avaliar peões, incluindo bónus de peão passado.
fn evaluate_pawns(board: &Board, mut pawn_bb: Bitboard, color: Color) -> i32 {
    let mut score = 0;
    let enemy_pawns = if color == Color::White { board.pawns & board.black_pieces } else { board.pawns & board.white_pieces };
    let color_idx = color as usize;

    while pawn_bb != 0 {
        let sq = pawn_bb.trailing_zeros() as usize;
        pawn_bb &= pawn_bb - 1;

        // Avaliação normal de material + PST
        score += MATERIAL_VALUES[PieceKind::Pawn as usize];
        score += if color == Color::White { PAWN_TABLE[sq] } else { PAWN_TABLE[sq ^ 56] };

        // Verificação de Peão Passado
        unsafe {
            if (PASSED_PAWN_MASKS[color_idx][sq] & enemy_pawns) == 0 {
                // É um peão passado! Adiciona bónus com base na fileira.
                let rank = sq / 8;
                score += if color == Color::White {
                    PASSED_PAWN_BONUS[rank]
                } else {
                    PASSED_PAWN_BONUS[7 - rank]
                };
            }
        }
    }
    score
}

fn evaluate_piece_type(mut piece_bb: Bitboard, pst: &'static [i32; 64], kind: PieceKind, color: Color) -> i32 {
    let mut score = 0;
    while piece_bb != 0 {
        let sq = piece_bb.trailing_zeros() as usize;
        piece_bb &= piece_bb - 1;
        let positional_score = if color == Color::White { pst[sq] } else { pst[sq ^ 56] };
        score += MATERIAL_VALUES[kind as usize] + positional_score;
    }
    score
}

/// Avalia a mobilidade das peças (número de movimentos possíveis)
fn evaluate_mobility(board: &Board, color: Color) -> i32 {
    let mut score = 0;
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let all_pieces = board.white_pieces | board.black_pieces;
    
    // Mobilidade dos cavalos
    let mut knights = board.knights & pieces;
    while knights != 0 {
        let sq = knights.trailing_zeros() as usize;
        knights &= knights - 1;
        let attacks = crate::moves::knight::get_knight_attacks_lookup(sq as u8);
        let legal_squares = attacks & !pieces;
        score += (legal_squares.count_ones() as i32) * MOBILITY_WEIGHTS[PieceKind::Knight as usize];
    }
    
    // Mobilidade dos bispos
    let mut bishops = board.bishops & pieces;
    while bishops != 0 {
        let sq = bishops.trailing_zeros() as usize;
        bishops &= bishops - 1;
        let attacks = crate::moves::sliding::get_bishop_attacks(sq as u8, all_pieces);
        let legal_squares = attacks & !pieces;
        score += (legal_squares.count_ones() as i32) * MOBILITY_WEIGHTS[PieceKind::Bishop as usize];
    }
    
    // Mobilidade das torres
    let mut rooks = board.rooks & pieces;
    while rooks != 0 {
        let sq = rooks.trailing_zeros() as usize;
        rooks &= rooks - 1;
        let attacks = crate::moves::sliding::get_rook_attacks(sq as u8, all_pieces);
        let legal_squares = attacks & !pieces;
        score += (legal_squares.count_ones() as i32) * MOBILITY_WEIGHTS[PieceKind::Rook as usize];
    }
    
    // Mobilidade das rainhas
    let mut queens = board.queens & pieces;
    while queens != 0 {
        let sq = queens.trailing_zeros() as usize;
        queens &= queens - 1;
        let bishop_attacks = crate::moves::sliding::get_bishop_attacks(sq as u8, all_pieces);
        let rook_attacks = crate::moves::sliding::get_rook_attacks(sq as u8, all_pieces);
        let attacks = bishop_attacks | rook_attacks;
        let legal_squares = attacks & !pieces;
        score += (legal_squares.count_ones() as i32) * MOBILITY_WEIGHTS[PieceKind::Queen as usize];
    }
    
    score
}

/// Avalia o controle do centro com ajuste por fase
fn evaluate_center_control(board: &Board, color: Color, is_opening: bool) -> i32 {
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let mut score = 0;
    
    // Bônus ajustado por fase do jogo
    let center_multiplier = if is_opening { 1.5 } else { 1.0 };
    
    // Bônus para peões no centro
    let pawns_in_center = (board.pawns & pieces & CENTRAL_SQUARES).count_ones() as i32 * (20.0 * center_multiplier) as i32;
    let pawns_in_extended_center = (board.pawns & pieces & EXTENDED_CENTER).count_ones() as i32 * (10.0 * center_multiplier) as i32;
    
    // Bônus para cavalos no centro (mais importante na abertura)
    let knights_in_center = (board.knights & pieces & CENTRAL_SQUARES).count_ones() as i32 * (15.0 * center_multiplier) as i32;
    let knights_in_extended_center = (board.knights & pieces & EXTENDED_CENTER).count_ones() as i32 * (8.0 * center_multiplier) as i32;
    
    // Bônus para bispos no centro
    let bishops_in_center = (board.bishops & pieces & CENTRAL_SQUARES).count_ones() as i32 * (12.0 * center_multiplier) as i32;
    
    score += pawns_in_center + pawns_in_extended_center;
    score += knights_in_center + knights_in_extended_center;
    score += bishops_in_center;
    
    score
}

/// Avalia a segurança do rei
fn evaluate_king_safety(board: &Board, color: Color, is_endgame: bool) -> i32 {
    if is_endgame {
        return 0; // Segurança do rei menos importante no final
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
    
    // Penaliza peões ausentes na frente do rei
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
    
    // Penaliza rei no centro durante meio-jogo
    if (CENTRAL_SQUARES & king_bb) != 0 {
        score -= 50;
    }
    
    score
}

/// Avalia peões passados conectados
fn evaluate_connected_passed_pawns(board: &Board, pawn_bb: Bitboard, color: Color) -> i32 {
    let mut score = 0;
    let enemy_pawns = if color == Color::White { 
        board.pawns & board.black_pieces 
    } else { 
        board.pawns & board.white_pieces 
    };
    let color_idx = color as usize;
    
    let mut passed_pawns: Vec<usize> = Vec::new();
    let mut temp_bb = pawn_bb;
    
    // Identifica peões passados
    while temp_bb != 0 {
        let sq = temp_bb.trailing_zeros() as usize;
        temp_bb &= temp_bb - 1;
        
        unsafe {
            if (PASSED_PAWN_MASKS[color_idx][sq] & enemy_pawns) == 0 {
                passed_pawns.push(sq);
            }
        }
    }
    
    // Avalia conexões entre peões passados
    for &sq1 in &passed_pawns {
        let file1 = sq1 % 8;
        for &sq2 in &passed_pawns {
            let file2 = sq2 % 8;
            if file1.abs_diff(file2) == 1 { // Peões adjacentes
                score += 25; // Bônus para peões passados conectados
            }
        }
    }
    
    score
}

/// Avalia o tempo (iniciativa) - favorece quem tem mais opções
fn evaluate_tempo(board: &Board) -> i32 {
    let current_moves = board.generate_legal_moves().len() as i32;
    
    // Simula jogada do oponente para ver suas opções
    let mut temp_board = *board;
    temp_board.to_move = !temp_board.to_move;
    let opponent_moves = temp_board.generate_legal_moves().len() as i32;
    
    // Bônus por ter mais opções (tempo/iniciativa) - limitado
    ((current_moves - opponent_moves) * 2).clamp(-50, 50)
}

/// Avalia o desenvolvimento na abertura
fn evaluate_development(board: &Board, color: Color) -> i32 {
    let mut score = 0;
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let back_rank = if color == Color::White { 0xFF } else { 0xFF00000000000000 };
    
    // Penaliza peças ainda no rank inicial
    let knights_undeveloped = (board.knights & pieces & back_rank).count_ones() as i32 * -15;
    let bishops_undeveloped = (board.bishops & pieces & back_rank).count_ones() as i32 * -15;
    
    // Avaliação melhorada de castling
    if color == Color::White {
        // Verifica se o rei branco já fez castling (não está mais em e1)
        let white_king = board.kings & board.white_pieces;
        if white_king != 0 {
            let king_square = white_king.trailing_zeros();
            if king_square == 6 || king_square == 2 { // g1 ou c1 (castling feito)
                score += 50; // Grande bônus por ter feito castling
            } else if king_square == 4 { // Ainda em e1
                // Penaliza se perdeu os direitos de castling
                if board.castling_rights & 0x03 == 0 {
                    score -= 30; // Penalidade por perder castling sem fazer
                } else {
                    // Pequeno bônus por ainda poder fazer castling
                    score += 10;
                }
            }
        }
    } else {
        // Mesma lógica para as pretas
        let black_king = board.kings & board.black_pieces;
        if black_king != 0 {
            let king_square = black_king.trailing_zeros();
            if king_square == 62 || king_square == 58 { // g8 ou c8 (castling feito)
                score += 50; // Grande bônus por ter feito castling
            } else if king_square == 60 { // Ainda em e8
                // Penaliza se perdeu os direitos de castling
                if board.castling_rights & 0x0C == 0 {
                    score -= 30; // Penalidade por perder castling sem fazer
                } else {
                    // Pequeno bônus por ainda poder fazer castling
                    score += 10;
                }
            }
        }
    }
    
    score + knights_undeveloped + bishops_undeveloped
}

/// Avalia a atividade do rei no endgame
fn evaluate_king_activity(board: &Board, color: Color) -> i32 {
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let king_bb = board.kings & pieces;
    
    if king_bb == 0 {
        return 0;
    }
    
    let king_square = king_bb.trailing_zeros() as usize;
    let rank = king_square / 8;
    let file = king_square % 8;
    
    // Bônus por rei centralizado no endgame
    let centralization_bonus = match (rank, file) {
        (3, 3) | (3, 4) | (4, 3) | (4, 4) => 40, // Centro
        (2, 2) | (2, 3) | (2, 4) | (2, 5) |
        (3, 2) | (3, 5) | (4, 2) | (4, 5) |
        (5, 2) | (5, 3) | (5, 4) | (5, 5) => 25, // Próximo ao centro
        _ => 0
    };
    
    // Bônus por mobilidade do rei
    let king_mobility = crate::moves::king::get_king_attacks_lookup(king_square as u8)
        .count_ones() as i32 * 5;
    
    centralization_bonus + king_mobility
}