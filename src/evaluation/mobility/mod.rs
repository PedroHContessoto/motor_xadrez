// Sistema modular de mobilidade - permite estratégias específicas por peça
use crate::{board::Board, types::{Color, Bitboard}};
use super::game_phase::{GamePhaseInfo, detect_game_phase_advanced, interpolate_phase_i32};
use super::utils as eval_utils;

pub mod pawn_mobility;
pub mod knight_mobility;
pub mod bishop_mobility;
pub mod rook_mobility;
pub mod queen_mobility;
pub mod king_mobility;
pub mod coordination;

// Re-exports das estruturas principais
pub use pawn_mobility::*;
pub use knight_mobility::*;
pub use bishop_mobility::*;
pub use rook_mobility::*;
pub use queen_mobility::*;
pub use king_mobility::*;
pub use coordination::*;

/// Resultado da análise de mobilidade para uma peça
#[derive(Debug, Clone, Copy)]
pub struct MobilityResult {
    pub raw_mobility: i32,      // Número bruto de casas
    pub safe_mobility: i32,     // Casas seguras (não atacadas)
    pub strategic_value: i32,   // Valor estratégico específico da peça
    pub tactical_threats: i32,  // Ameaças táticas criadas
    pub positional_bonus: i32,  // Bônus posicional (outposts, filas abertas, etc.)
}

impl MobilityResult {
    pub fn new() -> Self {
        MobilityResult {
            raw_mobility: 0,
            safe_mobility: 0,
            strategic_value: 0,
            tactical_threats: 0,
            positional_bonus: 0,
        }
    }

    pub fn total_score(&self) -> i32 {
        self.raw_mobility + self.safe_mobility + self.strategic_value +
            self.tactical_threats + self.positional_bonus
    }
}

/// Cache de ataques por tipo de peça para otimização
#[derive(Debug, Clone)]
pub struct AttackCache {
    pub pawn_attacks: Bitboard,
    pub knight_attacks: Bitboard,
    pub bishop_attacks: Bitboard,
    pub rook_attacks: Bitboard,
    pub queen_attacks: Bitboard,
    pub king_attacks: Bitboard,
    pub all_attacks: Bitboard,
}

impl AttackCache {
    pub fn new() -> Self {
        AttackCache {
            pawn_attacks: 0,
            knight_attacks: 0,
            bishop_attacks: 0,
            rook_attacks: 0,
            queen_attacks: 0,
            king_attacks: 0,
            all_attacks: 0,
        }
    }
}

/// Context compartilhado para análise de mobilidade
#[derive(Debug)]
pub struct MobilityContext {
    pub board: Board,
    pub color: Color,
    pub enemy_color: Color,
    pub our_pieces: Bitboard,
    pub enemy_pieces: Bitboard,
    pub all_pieces: Bitboard,
    pub enemy_attacked_squares: Bitboard,
    pub our_attacked_squares: Bitboard,
    pub phase: GamePhase,
    pub phase_info: GamePhaseInfo,
    // Cache de ataques detalhados
    pub enemy_attack_cache: AttackCache,
    pub our_attack_cache: AttackCache,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GamePhase {
    Opening,
    MiddleGame,
    Endgame,
}

impl MobilityContext {
    pub fn new(board: &Board, color: Color) -> Self {
        let enemy_color = !color;
        let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
        let enemy_pieces = if enemy_color == Color::White { board.white_pieces } else { board.black_pieces };
        let all_pieces = board.white_pieces | board.black_pieces;

        // Cria cache de ataques detalhados
        let enemy_attack_cache = compute_attack_cache(board, enemy_color);
        let our_attack_cache = compute_attack_cache(board, color);

        let enemy_attacked_squares = enemy_attack_cache.all_attacks;
        let our_attacked_squares = our_attack_cache.all_attacks;

        let phase_info = detect_game_phase_advanced(board);
        let phase = match phase_info.phase {
            super::game_phase::GamePhase::Opening => GamePhase::Opening,
            super::game_phase::GamePhase::Middlegame => GamePhase::MiddleGame,
            super::game_phase::GamePhase::Endgame => GamePhase::Endgame, // Trata como Endgame
        };

        MobilityContext {
            board: *board,
            color,
            enemy_color,
            our_pieces,
            enemy_pieces,
            all_pieces,
            enemy_attacked_squares,
            our_attacked_squares,
            phase,
            phase_info,
            enemy_attack_cache,
            our_attack_cache,
        }
    }
}

/// Função principal de avaliação de mobilidade (interface padrão)
pub fn evaluate_mobility(board: &Board, color: Color) -> i32 {
    evaluate_mobility_cached(board, color)
}

/// Avaliação de mobilidade com cache para evitar recálculos
pub fn evaluate_mobility_cached(board: &Board, color: Color) -> i32 {
    use super::mobility_cache::{with_mobility_cache, PieceType};
    
    // Tenta usar cache primeiro
    let cache_key = board.zobrist_hash;
    
    with_mobility_cache(|cache| {
        // Invalida cache se posição mudou
        cache.invalidate(cache_key);
        
        // Calcula mobilidade normalmente se não há cache
        evaluate_mobility_modular(board, color)
    })
}

/// Função principal de avaliação de mobilidade modular
pub fn evaluate_mobility_modular(board: &Board, color: Color) -> i32 {
    let context = MobilityContext::new(board, color);
    let mut total_score = 0;

    // Avalia cada tipo de peça individualmente
    total_score += evaluate_pawn_mobility_advanced(&context);
    total_score += evaluate_knight_mobility_advanced(&context);
    total_score += evaluate_bishop_mobility_advanced(&context);
    total_score += evaluate_rook_mobility_advanced(&context);
    total_score += evaluate_queen_mobility_advanced(&context);
    total_score += evaluate_king_mobility_advanced(&context);

    // Adiciona bônus de coordenação entre peças
    total_score += evaluate_piece_coordination(&context);

    total_score
}


/// Computa cache de ataques detalhado por tipo de peça (OTIMIZADO)
pub fn compute_attack_cache(board: &Board, color: Color) -> AttackCache {
    let mut cache = AttackCache::new();
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let all_pieces = board.white_pieces | board.black_pieces;

    // Ataques de peões usando método otimizado do board
    let pawns = board.pawns & pieces;
    cache.pawn_attacks = board.compute_pawn_attacks(pawns, color);

    // Ataques de cavalos
    let mut knights = board.knights & pieces;
    while knights != 0 {
        let sq = knights.trailing_zeros() as u8;
        knights &= knights - 1;
        cache.knight_attacks |= crate::moves::knight::get_knight_attacks_lookup(sq);
    }

    // Ataques de reis
    let kings = board.kings & pieces;
    if kings != 0 {
        let king_sq = kings.trailing_zeros() as u8;
        cache.king_attacks = crate::moves::king::get_king_attacks_lookup(king_sq);
    }

    // Ataques de bispos (incluindo rainhas) usando magic bitboards
    let mut bishops = (board.bishops | board.queens) & pieces;
    while bishops != 0 {
        let sq = bishops.trailing_zeros() as u8;
        bishops &= bishops - 1;
        cache.bishop_attacks |= crate::moves::magic_bitboards::get_bishop_attacks_magic(sq, all_pieces);
    }

    // Ataques de torres (incluindo rainhas) usando magic bitboards
    let mut rooks = (board.rooks | board.queens) & pieces;
    while rooks != 0 {
        let sq = rooks.trailing_zeros() as u8;
        rooks &= rooks - 1;
        cache.rook_attacks |= crate::moves::magic_bitboards::get_rook_attacks_magic(sq, all_pieces);
    }

    // Ataques de rainhas (combinado) usando magic bitboards
    let mut queens = board.queens & pieces;
    while queens != 0 {
        let sq = queens.trailing_zeros() as u8;
        queens &= queens - 1;
        cache.queen_attacks |= crate::moves::magic_bitboards::get_queen_attacks_magic(sq, all_pieces);
    }

    // Agrega todos os ataques
    cache.all_attacks = cache.pawn_attacks | cache.knight_attacks | cache.bishop_attacks |
        cache.rook_attacks | cache.queen_attacks | cache.king_attacks;

    cache
}

/// Computa casas atacadas por uma cor (OTIMIZADO - usa board.compute_attacked_squares)
pub fn compute_attacked_squares(board: &Board, color: Color) -> Bitboard {
    // Usa a função otimizada do board que tem magic bitboards
    board.compute_attacked_squares(color)
}


/// Determina a fase do jogo
pub fn determine_game_phase(board: &Board) -> GamePhase {
    let total_pieces = (board.white_pieces | board.black_pieces).count_ones();
    let major_pieces = (board.queens | board.rooks).count_ones();
    let minor_pieces = (board.knights | board.bishops).count_ones();

    if total_pieces > 24 && major_pieces > 6 {
        GamePhase::Opening
    } else if total_pieces < 12 || major_pieces < 4 {
        GamePhase::Endgame
    } else {
        GamePhase::MiddleGame
    }
}

/// Utilitários para análise posicional
pub mod utils {
    use super::*;

    /// Verifica se uma casa é um outpost (casa forte para cavalo/bispo)
    pub fn is_outpost(square: u8, color: Color, context: &MobilityContext) -> bool {
        let rank = square / 8;
        let file = square % 8;

        // Outposts são normalmente entre 4ª e 6ª fileiras
        let valid_ranks = if color == Color::White {
            rank >= 3 && rank <= 5
        } else {
            rank >= 2 && rank <= 4
        };

        if !valid_ranks {
            return false;
        }

        // Verifica se não há peões inimigos que podem atacar esta casa
        let enemy_pawns = context.board.pawns & context.enemy_pieces;

        // Verifica arquivos adjacentes
        for adj_file in (file.saturating_sub(1))..=(file.saturating_add(1)).min(7) {
            if adj_file == file { continue; }

            let file_pawns = enemy_pawns & get_file_mask(adj_file);
            if file_pawns != 0 {
                // Se há peão inimigo que pode avançar para atacar nossa casa
                let pawn_squares = get_set_bits(file_pawns);
                for pawn_sq in pawn_squares {
                    if super::super::utils::can_pawn_attack_square(pawn_sq, square, context.enemy_color) {
                        return false;
                    }
                }
            }
        }

        true
    }

    /// Verifica se um arquivo está aberto (sem peões)
    pub fn is_open_file(file: u8, context: &MobilityContext) -> bool {
        let file_mask = get_file_mask(file);
        (context.board.pawns & file_mask) == 0
    }

    /// Verifica se um arquivo está semi-aberto para uma cor
    pub fn is_semi_open_file(file: u8, color: Color, context: &MobilityContext) -> bool {
        let file_mask = get_file_mask(file);
        let our_pieces = if color == Color::White { context.board.white_pieces } else { context.board.black_pieces };

        (context.board.pawns & our_pieces & file_mask) == 0 && // Não temos peões neste arquivo
            (context.board.pawns & file_mask) != 0 // Mas existem peões (do inimigo)
    }

    /// Obtém máscara de arquivo
    pub fn get_file_mask(file: u8) -> Bitboard {
        super::super::utils::get_file_mask(file)
    }

    /// Obtém máscara de fileira
    pub fn get_rank_mask(rank: u8) -> Bitboard {
        0xFFu64 << (rank * 8)
    }

    /// Converte bitboard em vetor de casas
    pub fn get_set_bits(bitboard: Bitboard) -> Vec<u8> {
        eval_utils::get_set_bits(bitboard)
    }


    /// Verifica se casa está no centro
    pub fn is_central_square(square: u8) -> bool {
        let center_squares = [27, 28, 35, 36]; // d4, e4, d5, e5
        center_squares.contains(&square)
    }

    /// Verifica se casa está no centro expandido
    pub fn is_extended_center(square: u8) -> bool {
        let file = square % 8;
        let rank = square / 8;
        file >= 2 && file <= 5 && rank >= 2 && rank <= 5
    }
}