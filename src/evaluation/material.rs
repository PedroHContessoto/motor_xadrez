// Avaliação de material e tabelas de posição
use crate::{board::Board, types::{Color, PieceKind, Bitboard}};
use super::game_phase::{GamePhase, GamePhaseInfo};

// Extension trait para funções auxiliares do Board
trait BoardExt {
    fn has_many_attacks(&self) -> bool;
    fn has_hanging_pieces(&self, color: Color) -> bool;
    fn has_tactical_threats(&self, color: Color) -> bool;
    fn get_valuable_pieces(&self, color: Color) -> Bitboard;
    fn count_piece_type(&self, piece_type: Bitboard, color: Color) -> u32;
    fn has_bishop_pair(&self, color: Color) -> bool;
    fn get_material_signature(&self) -> MaterialSignature;
}

impl BoardExt for Board {
    fn has_many_attacks(&self) -> bool {
        // Detecta posições táticas pela densidade de peças
        let total_pieces = (self.white_pieces | self.black_pieces).count_ones();
        let non_pawn_pieces = ((self.knights | self.bishops | self.rooks | self.queens) &
            (self.white_pieces | self.black_pieces)).count_ones();

        total_pieces > 20 && non_pawn_pieces > 8
    }

    fn has_hanging_pieces(&self, color: Color) -> bool {
        let our_valuables = self.get_valuable_pieces(color);

        let mut bb = our_valuables;
        while bb != 0 {
            let sq = bb.trailing_zeros() as u8;
            bb &= bb - 1;

            if self.is_square_attacked_by(sq, !color) && !self.is_square_attacked_by(sq, color) {
                return true;
            }
        }
        false
    }

    fn has_tactical_threats(&self, color: Color) -> bool {
        let valuable_pieces = self.get_valuable_pieces(color);
        let mut threatened_count = 0;

        let mut bb = valuable_pieces;
        while bb != 0 {
            let sq = bb.trailing_zeros() as u8;
            bb &= bb - 1;

            if self.is_square_attacked_by(sq, !color) {
                threatened_count += 1;
                if threatened_count > 1 {
                    return true;
                }
            }
        }

        false
    }

    fn get_valuable_pieces(&self, color: Color) -> Bitboard {
        let pieces = if color == Color::White { self.white_pieces } else { self.black_pieces };
        (self.queens | self.rooks | self.bishops | self.knights) & pieces
    }

    fn count_piece_type(&self, piece_type: Bitboard, color: Color) -> u32 {
        let pieces = if color == Color::White { self.white_pieces } else { self.black_pieces };
        (piece_type & pieces).count_ones()
    }

    fn has_bishop_pair(&self, color: Color) -> bool {
        self.count_piece_type(self.bishops, color) >= 2
    }

    fn get_material_signature(&self) -> MaterialSignature {
        MaterialSignature::from_board(self)
    }
}

// Valores de material base
pub const MATERIAL_VALUES: [i32; 6] = [100, 320, 330, 500, 900, 20000];

// Valores refinados por fase
const PAWN_VALUE: [i32; 3] = [90, 100, 110];      // Opening, Middlegame, Endgame
const KNIGHT_VALUE: [i32; 3] = [305, 320, 315];
const BISHOP_VALUE: [i32; 3] = [315, 330, 335];
const ROOK_VALUE: [i32; 3] = [480, 500, 510];
const QUEEN_VALUE: [i32; 3] = [880, 900, 920];

// Bônus por par de bispos
const BISHOP_PAIR_BONUS: [i32; 3] = [30, 50, 55];

// Tabelas de posição (PST)
const PAWN_TABLE: [i32; 64] = [
    0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5,  5, 10, 25, 25, 10,  5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5, -5,-10,  0,  0,-10, -5,  5,
    5, 10, 10,-20,-20, 10, 10,  5,
    0,  0,  0,  0,  0,  0,  0,  0
];

const KNIGHT_TABLE: [i32; 64] = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
];

const BISHOP_TABLE: [i32; 64] = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
];

const ROOK_TABLE: [i32; 64] = [
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    0,  0,  0,  5,  5,  0,  0,  0
];

const QUEEN_TABLE: [i32; 64] = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
    0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
];

const KING_TABLE_MIDGAME: [i32; 64] = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20
];

const KING_TABLE_ENDGAME: [i32; 64] = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50
];

// Máscaras para estruturas
const CENTRAL_SQUARES: Bitboard = (1u64 << 27) | (1u64 << 28) | (1u64 << 35) | (1u64 << 36);
const EXTENDED_CENTER: Bitboard = 0x00003C3C3C3C0000;

/// Estrutura para análise detalhada de material
#[derive(Debug, Clone, Copy)]
pub struct MaterialAnalysis {
    pub white_material: i32,
    pub black_material: i32,
    pub material_balance: i32,
    pub phase_adjusted_balance: i32,
    pub has_bishop_pair: [bool; 2],
    pub pawn_count: [u32; 2],
    pub piece_activity_bonus: i32,
    pub imbalance_penalty: i32,
}

impl MaterialAnalysis {
    pub fn new() -> Self {
        MaterialAnalysis {
            white_material: 0,
            black_material: 0,
            material_balance: 0,
            phase_adjusted_balance: 0,
            has_bishop_pair: [false; 2],
            pawn_count: [0; 2],
            piece_activity_bonus: 0,
            imbalance_penalty: 0,
        }
    }

    /// Retorna diferença material do ponto de vista da cor
    pub fn score_for_color(&self, color: Color) -> i32 {
        if color == Color::White {
            self.phase_adjusted_balance
        } else {
            -self.phase_adjusted_balance
        }
    }
}

/// Assinatura material para detecção de finais específicos
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MaterialSignature {
    pub white_pawns: u8,
    pub white_knights: u8,
    pub white_bishops: u8,
    pub white_rooks: u8,
    pub white_queens: u8,
    pub black_pawns: u8,
    pub black_knights: u8,
    pub black_bishops: u8,
    pub black_rooks: u8,
    pub black_queens: u8,
}

impl MaterialSignature {
    pub fn from_board(board: &Board) -> Self {
        MaterialSignature {
            white_pawns: board.count_piece_type(board.pawns, Color::White) as u8,
            white_knights: board.count_piece_type(board.knights, Color::White) as u8,
            white_bishops: board.count_piece_type(board.bishops, Color::White) as u8,
            white_rooks: board.count_piece_type(board.rooks, Color::White) as u8,
            white_queens: board.count_piece_type(board.queens, Color::White) as u8,
            black_pawns: board.count_piece_type(board.pawns, Color::Black) as u8,
            black_knights: board.count_piece_type(board.knights, Color::Black) as u8,
            black_bishops: board.count_piece_type(board.bishops, Color::Black) as u8,
            black_rooks: board.count_piece_type(board.rooks, Color::Black) as u8,
            black_queens: board.count_piece_type(board.queens, Color::Black) as u8,
        }
    }

    /// Verifica se é um final específico conhecido
    pub fn is_special_endgame(&self) -> Option<SpecialEndgame> {
        let total_pieces = self.total_pieces();

        if total_pieces <= 6 {
            if self.is_kpk() {
                return Some(SpecialEndgame::KPK);
            }
            if self.is_krk() {
                return Some(SpecialEndgame::KRK);
            }
            if self.is_kqk() {
                return Some(SpecialEndgame::KQK);
            }
            if self.is_kbbk() {
                return Some(SpecialEndgame::KBBK);
            }
            if self.is_kbnk() {
                return Some(SpecialEndgame::KBNK);
            }
        }

        None
    }

    fn total_pieces(&self) -> u8 {
        self.white_pawns + self.white_knights + self.white_bishops +
            self.white_rooks + self.white_queens + self.black_pawns +
            self.black_knights + self.black_bishops + self.black_rooks +
            self.black_queens + 2 // +2 pelos reis
    }

    fn is_kpk(&self) -> bool {
        self.total_pieces() == 3 && (self.white_pawns + self.black_pawns) == 1
    }

    fn is_krk(&self) -> bool {
        self.total_pieces() == 3 && (self.white_rooks + self.black_rooks) == 1
    }

    fn is_kqk(&self) -> bool {
        self.total_pieces() == 3 && (self.white_queens + self.black_queens) == 1
    }

    fn is_kbbk(&self) -> bool {
        self.total_pieces() == 4 &&
            ((self.white_bishops == 2 && self.black_bishops == 0) ||
                (self.white_bishops == 0 && self.black_bishops == 2))
    }

    fn is_kbnk(&self) -> bool {
        self.total_pieces() == 4 &&
            ((self.white_bishops == 1 && self.white_knights == 1) ||
                (self.black_bishops == 1 && self.black_knights == 1))
    }
}

/// Tipos especiais de endgame
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecialEndgame {
    KPK,   // King and Pawn vs King
    KRK,   // King and Rook vs King
    KQK,   // King and Queen vs King
    KBBK,  // King and Two Bishops vs King
    KBNK,  // King, Bishop and Knight vs King
}

/// Avaliação rápida apenas de material (para lazy eval)
pub fn evaluate_material_only(board: &Board) -> i32 {
    let white_material = calculate_material_value(board, Color::White);
    let black_material = calculate_material_value(board, Color::Black);

    let material_diff = white_material - black_material;

    if board.to_move == Color::White {
        material_diff
    } else {
        -material_diff
    }
}

/// Análise completa de material
pub fn analyze_material(board: &Board, game_phase: &GamePhase) -> MaterialAnalysis {
    let mut analysis = MaterialAnalysis::new();

    // Calcula material bruto
    analysis.white_material = calculate_material_value(board, Color::White);
    analysis.black_material = calculate_material_value(board, Color::Black);
    analysis.material_balance = analysis.white_material - analysis.black_material;

    // Detecta par de bispos
    analysis.has_bishop_pair[0] = board.has_bishop_pair(Color::White);
    analysis.has_bishop_pair[1] = board.has_bishop_pair(Color::Black);

    // Conta peões
    analysis.pawn_count[0] = board.count_piece_type(board.pawns, Color::White);
    analysis.pawn_count[1] = board.count_piece_type(board.pawns, Color::Black);

    // Ajusta por fase
    let phase_value = match game_phase {
        GamePhase::Opening => 0,
        GamePhase::Middlegame => 1,
        GamePhase::Endgame => 2,
    };

    // Recalcula com valores ajustados por fase
    let white_adjusted = calculate_phase_adjusted_material(board, Color::White, phase_value);
    let black_adjusted = calculate_phase_adjusted_material(board, Color::Black, phase_value);

    analysis.phase_adjusted_balance = white_adjusted - black_adjusted;

    // Avalia desequilíbrios
    analysis.imbalance_penalty = evaluate_material_imbalance(board);

    analysis
}

/// Calcula material com ajuste de fase
fn calculate_phase_adjusted_material(board: &Board, color: Color, phase: usize) -> i32 {
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let mut total = 0;

    total += (board.pawns & pieces).count_ones() as i32 * PAWN_VALUE[phase];
    total += (board.knights & pieces).count_ones() as i32 * KNIGHT_VALUE[phase];
    total += (board.bishops & pieces).count_ones() as i32 * BISHOP_VALUE[phase];
    total += (board.rooks & pieces).count_ones() as i32 * ROOK_VALUE[phase];
    total += (board.queens & pieces).count_ones() as i32 * QUEEN_VALUE[phase];

    // Bônus por par de bispos
    if board.has_bishop_pair(color) {
        total += BISHOP_PAIR_BONUS[phase];
    }

    total
}

/// Calcula valor material básico
fn calculate_material_value(board: &Board, color: Color) -> i32 {
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };

    let pawns = (board.pawns & pieces).count_ones() as i32 * MATERIAL_VALUES[0];
    let knights = (board.knights & pieces).count_ones() as i32 * MATERIAL_VALUES[1];
    let bishops = (board.bishops & pieces).count_ones() as i32 * MATERIAL_VALUES[2];
    let rooks = (board.rooks & pieces).count_ones() as i32 * MATERIAL_VALUES[3];
    let queens = (board.queens & pieces).count_ones() as i32 * MATERIAL_VALUES[4];

    pawns + knights + bishops + rooks + queens
}

/// Avalia material e PST combinados
pub fn evaluate_material_and_pst(board: &Board, color: Color, game_phase: &GamePhase) -> i32 {
    let mut score = 0;
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };

    // Escolhe tabela do rei baseada na fase
    let king_table = match game_phase {
        GamePhase::Endgame => &KING_TABLE_ENDGAME,
        _ => &KING_TABLE_MIDGAME,
    };

    // Avalia cada tipo de peça
    score += evaluate_piece_type(board.pawns & pieces, &PAWN_TABLE, PieceKind::Pawn, color);
    score += evaluate_knights_enhanced(board, board.knights & pieces, color, game_phase);
    score += evaluate_bishops_enhanced(board, board.bishops & pieces, color, game_phase);
    score += evaluate_rooks_enhanced(board, board.rooks & pieces, color, game_phase);
    score += evaluate_queens_enhanced(board, board.queens & pieces, color, game_phase);
    score += evaluate_piece_type(board.kings & pieces, king_table, PieceKind::King, color);

    // Bônus por controle do centro
    score += evaluate_center_control(board, color, game_phase);

    // Avaliações especiais
    score += evaluate_special_patterns(board, color, game_phase);

    score
}

/// Avalia tipo de peça básico
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

/// Avalia cavalos com melhorias
fn evaluate_knights_enhanced(board: &Board, mut knight_bb: Bitboard, color: Color, game_phase: &GamePhase) -> i32 {
    let mut score = 0;

    while knight_bb != 0 {
        let sq = knight_bb.trailing_zeros() as usize;
        knight_bb &= knight_bb - 1;

        // Material + PST base
        let positional_score = if color == Color::White { KNIGHT_TABLE[sq] } else { KNIGHT_TABLE[sq ^ 56] };
        score += MATERIAL_VALUES[PieceKind::Knight as usize] + positional_score;

        // Avaliações específicas do cavalo
        score += evaluate_knight_specific(board, sq as u8, color, game_phase);
    }

    score
}

/// Avaliações específicas para cavalos
fn evaluate_knight_specific(board: &Board, square: u8, color: Color, game_phase: &GamePhase) -> i32 {
    let mut bonus = 0;

    // Outpost
    let enemy_pawns = board.pawns & if color == Color::White { board.black_pieces } else { board.white_pieces };
    if super::utils::is_outpost(square, color, enemy_pawns) {
        bonus += match game_phase {
            GamePhase::Opening => 20,
            GamePhase::Middlegame => 30,
            GamePhase::Endgame => 35,
        };

        // Extra se defendido por peão
        if is_defended_by_pawn(board, square, color) {
            bonus += 15;
        }
    }

    // Mobilidade segura
    bonus += evaluate_knight_mobility_safe(board, square, color);

    // Penalidade por cavalo na borda
    if is_knight_on_rim(square) {
        bonus -= 15;
    }

    // Bônus por cavalo bloqueando peões passados inimigos
    if is_blocking_passed_pawn(board, square, color) {
        bonus += 20;
    }

    bonus
}

/// Verifica se cavalo está na borda
fn is_knight_on_rim(square: u8) -> bool {
    let file = square % 8;
    let rank = square / 8;
    file == 0 || file == 7 || rank == 0 || rank == 7
}

/// Verifica se está bloqueando peão passado
fn is_blocking_passed_pawn(board: &Board, square: u8, color: Color) -> bool {
    let enemy_pawns = board.pawns & if color == Color::White { board.black_pieces } else { board.white_pieces };
    let our_pawns = board.pawns & if color == Color::White { board.white_pieces } else { board.black_pieces };

    // Verifica se há peão inimigo atrás (que seria passado)
    let behind_sq = if color == Color::White {
        if square >= 8 { square - 8 } else { return false; }
    } else {
        if square < 56 { square + 8 } else { return false; }
    };

    if (enemy_pawns & (1u64 << behind_sq)) != 0 {
        // Verifica se é passado
        super::utils::is_passed_pawn(behind_sq, !color, enemy_pawns, our_pawns)
    } else {
        false
    }
}

/// Avalia mobilidade segura do cavalo
fn evaluate_knight_mobility_safe(board: &Board, square: u8, color: Color) -> i32 {
    let enemy_color = !color;
    let knight_moves = crate::moves::knight::get_knight_attacks_lookup(square);
    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };

    let mut safe_squares = 0;
    let mut moves_bb = knight_moves & !our_pieces;

    while moves_bb != 0 {
        let target_sq = moves_bb.trailing_zeros() as u8;
        moves_bb &= moves_bb - 1;

        // Conta apenas casas não atacadas por peões inimigos
        if !is_attacked_by_enemy_pawns(board, target_sq, enemy_color) {
            safe_squares += 1;
        }
    }

    // +3 por cada casa segura
    safe_squares * 3
}

/// Verifica se casa é atacada por peões inimigos
fn is_attacked_by_enemy_pawns(board: &Board, square: u8, enemy_color: Color) -> bool {
    let enemy_pawns = board.pawns & if enemy_color == Color::White { board.white_pieces } else { board.black_pieces };
    let pawn_attacks = super::utils::compute_pawn_attacks(enemy_pawns, enemy_color);
    (pawn_attacks & (1u64 << square)) != 0
}

/// Verifica se peça está defendida por peão
fn is_defended_by_pawn(board: &Board, square: u8, color: Color) -> bool {
    let our_pawns = board.pawns & if color == Color::White { board.white_pieces } else { board.black_pieces };
    let pawn_attacks = super::utils::compute_pawn_attacks(our_pawns, color);
    (pawn_attacks & (1u64 << square)) != 0
}

/// Avalia bispos com melhorias
fn evaluate_bishops_enhanced(board: &Board, mut bishop_bb: Bitboard, color: Color, game_phase: &GamePhase) -> i32 {
    let mut score = 0;

    while bishop_bb != 0 {
        let sq = bishop_bb.trailing_zeros() as usize;
        bishop_bb &= bishop_bb - 1;

        // Material + PST base
        let positional_score = if color == Color::White { BISHOP_TABLE[sq] } else { BISHOP_TABLE[sq ^ 56] };
        score += MATERIAL_VALUES[PieceKind::Bishop as usize] + positional_score;

        // Avaliações específicas do bispo
        score += evaluate_bishop_specific(board, sq as u8, color, game_phase);
    }

    // Bônus por par de bispos
    if board.has_bishop_pair(color) {
        score += match game_phase {
            GamePhase::Opening => BISHOP_PAIR_BONUS[0],
            GamePhase::Middlegame => BISHOP_PAIR_BONUS[1],
            GamePhase::Endgame => BISHOP_PAIR_BONUS[2],
        };
    }

    score
}

/// Avaliações específicas para bispos
fn evaluate_bishop_specific(board: &Board, square: u8, color: Color, game_phase: &GamePhase) -> i32 {
    let mut bonus = 0;

    // Mobilidade de longo alcance
    let mobility = calculate_bishop_mobility(board, square);
    bonus += mobility * 2;

    // Penalidade por bispo bloqueado por peões próprios
    let blocked_penalty = calculate_bishop_blocked_penalty(board, square, color);
    bonus -= blocked_penalty;

    // Bônus por diagonais longas
    if is_on_long_diagonal(square) {
        bonus += 10;
    }

    // Fianchetto
    if is_fianchetto_bishop(square, color) {
        bonus += 15;
    }

    bonus
}

/// Calcula mobilidade do bispo
fn calculate_bishop_mobility(board: &Board, square: u8) -> i32 {
    let all_pieces = board.white_pieces | board.black_pieces;
    let attacks = crate::moves::magic_bitboards::get_bishop_attacks_magic(square, all_pieces);
    attacks.count_ones() as i32
}

/// Calcula penalidade por bispo bloqueado
fn calculate_bishop_blocked_penalty(board: &Board, square: u8, color: Color) -> i32 {
    let our_pawns = board.pawns & if color == Color::White { board.white_pieces } else { board.black_pieces };
    let bishop_on_light = (square / 8 + square % 8) % 2 == 0;

    let mut blocked_count = 0;
    let mut pawn_bb = our_pawns;

    while pawn_bb != 0 {
        let pawn_sq = pawn_bb.trailing_zeros() as u8;
        pawn_bb &= pawn_bb - 1;

        let pawn_on_light = (pawn_sq / 8 + pawn_sq % 8) % 2 == 0;
        if pawn_on_light == bishop_on_light {
            blocked_count += 1;
        }
    }

    blocked_count * 5
}

/// Verifica se está em diagonal longa
fn is_on_long_diagonal(square: u8) -> bool {
    let diag1 = 0x8040201008040201u64; // a1-h8
    let diag2 = 0x0102040810204080u64; // h1-a8
    ((1u64 << square) & (diag1 | diag2)) != 0
}

/// Verifica se é bispo de fianchetto
fn is_fianchetto_bishop(square: u8, color: Color) -> bool {
    match color {
        Color::White => square == 9 || square == 14,  // b2 ou g2
        Color::Black => square == 49 || square == 54, // b7 ou g7
    }
}

/// Avalia torres com melhorias
fn evaluate_rooks_enhanced(board: &Board, mut rook_bb: Bitboard, color: Color, game_phase: &GamePhase) -> i32 {
    let mut score = 0;

    while rook_bb != 0 {
        let sq = rook_bb.trailing_zeros() as usize;
        rook_bb &= rook_bb - 1;

        // Material + PST base
        let positional_score = if color == Color::White { ROOK_TABLE[sq] } else { ROOK_TABLE[sq ^ 56] };
        score += MATERIAL_VALUES[PieceKind::Rook as usize] + positional_score;

        // Avaliações específicas da torre
        score += evaluate_rook_specific(board, sq as u8, color, game_phase);
    }

    // Bônus por torres conectadas
    if are_rooks_connected(board, color) {
        score += 15;
    }

    score
}

/// Avaliações específicas para torres
fn evaluate_rook_specific(board: &Board, square: u8, color: Color, game_phase: &GamePhase) -> i32 {
    let mut bonus = 0;
    let rank = square / 8;
    let file = square % 8;

    // Torre na 7ª/2ª fileira
    if (color == Color::White && rank == 6) || (color == Color::Black && rank == 1) {
        bonus += 20;

        // Extra se rei inimigo na 8ª/1ª
        let enemy_king = board.kings & if color == Color::White { board.black_pieces } else { board.white_pieces };
        if enemy_king != 0 {
            let king_rank = enemy_king.trailing_zeros() / 8;
            if (color == Color::White && king_rank == 7) || (color == Color::Black && king_rank == 0) {
                bonus += 15;
            }
        }
    }

    // Avalia tipo de arquivo
    let file_type = evaluate_file_type(board, file);
    bonus += match file_type {
        FileType::Open => 25,
        FileType::HalfOpenOur => 15,
        FileType::HalfOpenTheir => 10,
        FileType::Closed => 0,
    };

    // Mobilidade
    let mobility = calculate_rook_mobility(board, square);
    bonus += mobility;

    bonus
}

/// Tipos de arquivo para torres
#[derive(Debug, Clone, Copy)]
enum FileType {
    Open,
    HalfOpenOur,
    HalfOpenTheir,
    Closed,
}

/// Avalia tipo de arquivo
fn evaluate_file_type(board: &Board, file: u8) -> FileType {
    let file_mask = super::utils::get_file_mask_from_file(file);
    let white_pawns = board.pawns & board.white_pieces & file_mask;
    let black_pawns = board.pawns & board.black_pieces & file_mask;

    match (white_pawns == 0, black_pawns == 0) {
        (true, true) => FileType::Open,
        (true, false) => FileType::HalfOpenTheir,
        (false, true) => FileType::HalfOpenOur,
        (false, false) => FileType::Closed,
    }
}

/// Calcula mobilidade da torre
fn calculate_rook_mobility(board: &Board, square: u8) -> i32 {
    let all_pieces = board.white_pieces | board.black_pieces;
    let attacks = crate::moves::magic_bitboards::get_rook_attacks_magic(square, all_pieces);
    (attacks.count_ones() as i32) / 2 // Divide por 2 para não supervalorizar
}

/// Verifica se torres estão conectadas
fn are_rooks_connected(board: &Board, color: Color) -> bool {
    let our_rooks = board.rooks & if color == Color::White { board.white_pieces } else { board.black_pieces };

    if our_rooks.count_ones() != 2 {
        return false;
    }

    let mut rook_squares = Vec::new();
    let mut rook_bb = our_rooks;
    while rook_bb != 0 {
        rook_squares.push(rook_bb.trailing_zeros() as u8);
        rook_bb &= rook_bb - 1;
    }

    let r1_rank = rook_squares[0] / 8;
    let r1_file = rook_squares[0] % 8;
    let r2_rank = rook_squares[1] / 8;
    let r2_file = rook_squares[1] % 8;

    // Conectadas se na mesma linha ou coluna
    r1_rank == r2_rank || r1_file == r2_file
}

/// Avalia rainhas com melhorias
fn evaluate_queens_enhanced(board: &Board, mut queen_bb: Bitboard, color: Color, game_phase: &GamePhase) -> i32 {
    let mut score = 0;

    while queen_bb != 0 {
        let sq = queen_bb.trailing_zeros() as usize;
        queen_bb &= queen_bb - 1;

        // Material + PST base
        let positional_score = if color == Color::White { QUEEN_TABLE[sq] } else { QUEEN_TABLE[sq ^ 56] };
        score += MATERIAL_VALUES[PieceKind::Queen as usize] + positional_score;

        // Avaliações específicas da rainha
        score += evaluate_queen_specific(board, sq as u8, color, game_phase);
    }

    score
}

/// Avaliações específicas para rainhas
fn evaluate_queen_specific(board: &Board, square: u8, color: Color, game_phase: &GamePhase) -> i32 {
    let mut bonus = 0;

    // Penalidade por desenvolvimento precoce
    if matches!(game_phase, GamePhase::Opening) {
        let rank = square / 8;
        let starting_rank = if color == Color::White { 0 } else { 7 };

        if rank != starting_rank {
            let minors_developed = count_developed_minors(board, color);
            if minors_developed < 2 {
                bonus -= 30;
            } else if minors_developed < 3 {
                bonus -= 15;
            }
        }
    }

    // Mobilidade (reduzida para não supervalorizar)
    let mobility = calculate_queen_mobility(board, square);
    bonus += mobility / 2;

    bonus
}

/// Conta peças menores desenvolvidas
fn count_developed_minors(board: &Board, color: Color) -> i32 {
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let back_rank = if color == Color::White { 0xFF } else { 0xFF00000000000000 };

    let minors = (board.knights | board.bishops) & pieces;
    let developed = minors & !back_rank;

    developed.count_ones() as i32
}

/// Calcula mobilidade da rainha
fn calculate_queen_mobility(board: &Board, square: u8) -> i32 {
    let all_pieces = board.white_pieces | board.black_pieces;
    let attacks = crate::moves::magic_bitboards::get_queen_attacks_magic(square, all_pieces);
    (attacks.count_ones() as i32) / 3 // Divide por 3 para balancear
}

/// Avalia controle do centro
fn evaluate_center_control(board: &Board, color: Color, game_phase: &GamePhase) -> i32 {
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let mut score = 0;

    // Ajusta importância por fase e situação
    let center_multiplier = match game_phase {
        GamePhase::Opening => {
            if board.has_many_attacks() {
                1.0
            } else {
                1.2
            }
        },
        GamePhase::Middlegame => 1.0,
        GamePhase::Endgame => 0.5,
    };

    // Peões no centro
    let pawns_in_center = (board.pawns & pieces & CENTRAL_SQUARES).count_ones() as i32;
    let pawns_in_extended_center = (board.pawns & pieces & EXTENDED_CENTER).count_ones() as i32;
    score += ((pawns_in_center as f64) * 20.0 * center_multiplier) as i32;
    score += ((pawns_in_extended_center as f64) * 10.0 * center_multiplier) as i32;

    // Peças no centro
    let knights_in_center = (board.knights & pieces & CENTRAL_SQUARES).count_ones() as i32;
    let bishops_in_center = (board.bishops & pieces & CENTRAL_SQUARES).count_ones() as i32;
    score += ((knights_in_center as f64) * 15.0 * center_multiplier) as i32;
    score += ((bishops_in_center as f64) * 12.0 * center_multiplier) as i32;

    // Controle por ataques (mais sutil)
    let center_control = evaluate_center_control_by_attacks(board, color);
    score += (center_control as f64 * center_multiplier * 0.5) as i32;

    score
}

/// Avalia controle do centro por ataques
fn evaluate_center_control_by_attacks(board: &Board, color: Color) -> i32 {
    let mut control = 0;
    let central_squares = [27, 28, 35, 36]; // d4, e4, d5, e5

    for &sq in &central_squares {
        if board.is_square_attacked_by(sq, color) {
            control += 3;

            // Bônus se atacado múltiplas vezes
            let attackers = count_attackers(board, sq, color);
            if attackers > 1 {
                control += attackers - 1;
            }
        }
    }

    control
}

/// Conta número de atacantes a uma casa
fn count_attackers(board: &Board, square: u8, color: Color) -> i32 {
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let mut count = 0;

    // Simplificado - conta apenas se há atacantes
    if board.is_square_attacked_by(square, color) {
        count = 1; // Versão simplificada
    }

    count
}

/// Avalia padrões especiais
fn evaluate_special_patterns(board: &Board, color: Color, game_phase: &GamePhase) -> i32 {
    let mut score = 0;

    // Desequilíbrios de material
    score += evaluate_material_imbalance_for_color(board, color);

    // Padrões de peças específicos
    score += evaluate_piece_coordination(board, color);

    // Ajustes por fase
    if matches!(game_phase, GamePhase::Endgame) {
        score += evaluate_endgame_material_patterns(board, color);
    }

    score
}

/// Avalia desequilíbrio material
fn evaluate_material_imbalance(board: &Board) -> i32 {
    let mut penalty = 0;

    // Torre vs peças menores
    let white_rooks = board.count_piece_type(board.rooks, Color::White);
    let black_rooks = board.count_piece_type(board.rooks, Color::Black);
    let white_minors = board.count_piece_type(board.knights | board.bishops, Color::White);
    let black_minors = board.count_piece_type(board.knights | board.bishops, Color::Black);

    // Penaliza desequilíbrios extremos
    if white_rooks > black_rooks + 1 && black_minors > white_minors + 2 {
        penalty += 20;
    }
    if black_rooks > white_rooks + 1 && white_minors > black_minors + 2 {
        penalty += 20;
    }

    penalty
}

/// Avalia desequilíbrio para uma cor
fn evaluate_material_imbalance_for_color(board: &Board, color: Color) -> i32 {
    let our_rooks = board.count_piece_type(board.rooks, color);
    let enemy_rooks = board.count_piece_type(board.rooks, !color);
    let our_minors = board.count_piece_type(board.knights | board.bishops, color);
    let enemy_minors = board.count_piece_type(board.knights | board.bishops, !color);

    let mut score = 0;

    // Troca favorável: trocar torre por 2 menores quando temos mais torres
    if our_rooks > enemy_rooks && enemy_minors >= our_minors + 2 {
        score -= 15;
    }

    // Cavalos vs Bispos baseado na estrutura
    let pawn_count = board.pawns.count_ones();
    let our_knights = board.count_piece_type(board.knights, color);
    let our_bishops = board.count_piece_type(board.bishops, color);

    if pawn_count > 12 { // Posição fechada
        score += (our_knights as i32 - our_bishops as i32) * 5;
    } else { // Posição aberta
        score += (our_bishops as i32 - our_knights as i32) * 5;
    }

    score
}

/// Avalia coordenação entre peças
fn evaluate_piece_coordination(board: &Board, color: Color) -> i32 {
    let mut score = 0;

    // Baterias (torre + rainha, bispo + rainha)
    score += count_batteries(board, color) * 8;

    // Peças defendendo-se mutuamente
    score += count_mutual_defense(board, color) * 3;

    score
}

/// Conta baterias (peças alinhadas)
fn count_batteries(board: &Board, color: Color) -> i32 {
    let mut batteries = 0;
    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let our_queens = board.queens & our_pieces;
    let our_rooks = board.rooks & our_pieces;
    let our_bishops = board.bishops & our_pieces;

    // Simplificado - conta apenas se há rainha + torre/bispo
    if our_queens != 0 && (our_rooks != 0 || our_bishops != 0) {
        batteries += 1;
    }

    batteries
}

/// Conta defesas mútuas
fn count_mutual_defense(board: &Board, color: Color) -> i32 {
    // Implementação simplificada
    0
}

/// Avalia padrões de material em endgame
fn evaluate_endgame_material_patterns(board: &Board, color: Color) -> i32 {
    let mut score = 0;
    let signature = board.get_material_signature();

    // Verifica finais específicos
    if let Some(endgame_type) = signature.is_special_endgame() {
        score += evaluate_special_endgame(board, color, endgame_type);
    }

    score
}

/// Avalia finais específicos
fn evaluate_special_endgame(board: &Board, color: Color, endgame_type: SpecialEndgame) -> i32 {
    match endgame_type {
        SpecialEndgame::KPK => evaluate_kpk(board, color),
        SpecialEndgame::KRK => evaluate_krk(board, color),
        SpecialEndgame::KQK => evaluate_kqk(board, color),
        SpecialEndgame::KBBK => evaluate_kbbk(board, color),
        SpecialEndgame::KBNK => evaluate_kbnk(board, color),
    }
}

/// Avalia KPK
fn evaluate_kpk(board: &Board, color: Color) -> i32 {
    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let has_pawn = (board.pawns & our_pieces) != 0;

    if has_pawn { 50 } else { -50 }
}

/// Avalia KRK
fn evaluate_krk(board: &Board, color: Color) -> i32 {
    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let has_rook = (board.rooks & our_pieces) != 0;

    if has_rook { 400 } else { -400 }
}

/// Avalia KQK
fn evaluate_kqk(board: &Board, color: Color) -> i32 {
    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let has_queen = (board.queens & our_pieces) != 0;

    if has_queen { 800 } else { -800 }
}

/// Avalia KBBK
fn evaluate_kbbk(board: &Board, color: Color) -> i32 {
    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let our_bishops = (board.bishops & our_pieces).count_ones();

    if our_bishops == 2 { 100 } else { -100 }
}

/// Avalia KBNK
fn evaluate_kbnk(board: &Board, color: Color) -> i32 {
    let our_pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let our_bishops = (board.bishops & our_pieces).count_ones();
    let our_knights = (board.knights & our_pieces).count_ones();

    if our_bishops == 1 && our_knights == 1 { 50 } else { -50 }
}

/// Avalia material básico para mobilidade
pub fn evaluate_mobility_basic(board: &Board, color: Color) -> i32 {
    // Implementação básica para o módulo de mobilidade
    let pieces = if color == Color::White { board.white_pieces } else { board.black_pieces };
    let piece_count = pieces.count_ones() as i32;
    piece_count * 2
}
