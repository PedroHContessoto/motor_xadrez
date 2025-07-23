// Módulo principal de avaliação modular
// Descrição: Coordena todas as subfunções de avaliação

pub mod material;
pub mod king_safety;
pub mod threats;
pub mod mobility;
pub mod pawn_structure;
pub mod game_phase;

use crate::{board::Board, types::Color};

/// Função principal de avaliação (interface pública)
pub fn evaluate(board: &Board) -> i32 {
    // Versão simplificada apenas com material para estabilidade
    let mut white_material = 0;
    let mut black_material = 0;
    
    // Material básico
    white_material += (board.pawns & board.white_pieces).count_ones() as i32 * 100;
    white_material += (board.knights & board.white_pieces).count_ones() as i32 * 320;
    white_material += (board.bishops & board.white_pieces).count_ones() as i32 * 330;
    white_material += (board.rooks & board.white_pieces).count_ones() as i32 * 500;
    white_material += (board.queens & board.white_pieces).count_ones() as i32 * 900;
    
    black_material += (board.pawns & board.black_pieces).count_ones() as i32 * 100;
    black_material += (board.knights & board.black_pieces).count_ones() as i32 * 320;
    black_material += (board.bishops & board.black_pieces).count_ones() as i32 * 330;
    black_material += (board.rooks & board.black_pieces).count_ones() as i32 * 500;
    black_material += (board.queens & board.black_pieces).count_ones() as i32 * 900;
    
    let final_score = white_material - black_material;
    
    // Retorna relativo ao jogador atual
    if board.to_move == Color::White {
        final_score
    } else {
        -final_score
    }
}

/// Avalia cor específica
fn evaluate_color(board: &Board, color: Color, game_phase: &game_phase::GamePhase) -> i32 {
    let mut score = 0;

    // Material + PST
    score += material::evaluate_material_and_pst(board, color, game_phase);

    // Estrutura de peões (incluindo passados)
    score += pawn_structure::evaluate_pawn_structure(board, color);

    // Mobilidade segura
    score += mobility::evaluate_mobility(board, color);

    // Segurança do rei (aprimorada)
    score += king_safety::evaluate_king_safety(board, color, game_phase);

    // NOVO: Avaliação de ameaças (peças penduradas, ataques)
    score += threats::evaluate_threats(board, color);

    // Avaliações específicas por fase
    match game_phase {
        game_phase::GamePhase::Opening => {
            score += evaluate_development(board, color);
        },
        game_phase::GamePhase::Endgame => {
            score += evaluate_king_activity(board, color);
        },
        _ => {}
    }

    score
}

/// Avalia o tempo (iniciativa)
fn evaluate_tempo(board: &Board) -> i32 {
    let current_moves = board.generate_legal_moves().len() as i32;

    let mut temp_board = *board;
    temp_board.to_move = !temp_board.to_move;
    let opponent_moves = temp_board.generate_legal_moves().len() as i32;

    ((current_moves - opponent_moves) * 2).clamp(-50, 50)
}

/// Avalia desenvolvimento na abertura
fn evaluate_development(board: &Board, color: Color) -> i32 {
    let mut score = 0;
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let back_rank = if color == Color::White { 0xFF } else { 0xFF00000000000000 };

    // Penaliza peças ainda no rank inicial
    let knights_undeveloped = (board.knights & pieces & back_rank).count_ones() as i32 * -15;
    let bishops_undeveloped = (board.bishops & pieces & back_rank).count_ones() as i32 * -15;

    // Avaliação de castling melhorada (do código original)
    score += evaluate_castling(board, color);

    score + knights_undeveloped + bishops_undeveloped
}

/// Avalia castling
fn evaluate_castling(board: &Board, color: Color) -> i32 {
    let mut score = 0;

    if color == Color::White {
        let white_king = board.kings & board.white_pieces;
        if white_king != 0 {
            let king_square = white_king.trailing_zeros();
            if king_square == 6 || king_square == 2 { // g1 ou c1 (castling feito)
                score += 50;
            } else if king_square == 4 { // Ainda em e1
                if board.castling_rights & 0x03 == 0 {
                    score -= 30; // Perdeu castling sem fazer
                } else {
                    score += 10; // Ainda pode fazer
                }
            }
        }
    } else {
        let black_king = board.kings & board.black_pieces;
        if black_king != 0 {
            let king_square = black_king.trailing_zeros();
            if king_square == 62 || king_square == 58 { // g8 ou c8
                score += 50;
            } else if king_square == 60 { // Ainda em e8
                if board.castling_rights & 0x0C == 0 {
                    score -= 30;
                } else {
                    score += 10;
                }
            }
        }
    }

    score
}

/// Avalia atividade do rei no endgame
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

    // Mobilidade do rei
    let king_mobility = crate::moves::king::get_king_attacks_lookup(king_square as u8)
        .count_ones() as i32 * 5;

    centralization_bonus + king_mobility
}