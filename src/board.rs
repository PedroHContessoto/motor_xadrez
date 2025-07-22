// Ficheiro: src/board.rs
// Descrição: Módulo que contém a struct Board e os seus métodos principais.

use super::types::*;
use crate::moves;
use crate::zobrist::{ZOBRIST_KEYS, piece_to_index, color_to_index};

// A struct principal do tabuleiro, usando Bitboards.
#[derive(Debug, Clone, Copy)]
pub struct Board {
    // Bitboards para cada tipo de peça.
    pub pawns: Bitboard,
    pub knights: Bitboard,
    pub bishops: Bitboard,
    pub rooks: Bitboard,
    pub queens: Bitboard,
    pub kings: Bitboard,

    // Bitboards para as peças de cada cor.
    pub white_pieces: Bitboard,
    pub black_pieces: Bitboard,

    // De quem é a vez de jogar.
    pub to_move: Color,

    pub en_passant_target: Option<u8>,
    
    // Direitos de roque (pode_rocar_pequeno_brancas, pode_rocar_grande_brancas, pode_rocar_pequeno_pretas, pode_rocar_grande_pretas)
    pub castling_rights: u8, // Bits: 0=K, 1=Q, 2=k, 3=q
    
    // Cache do estado de xeque para otimização
    pub white_king_in_check: bool,
    pub black_king_in_check: bool,
    
    // Para detecção de draws
    pub halfmove_clock: u16,   // Contador para regra dos 50 movimentos
    pub zobrist_hash: u64,     // Hash Zobrist para detecção de repetição
}

impl Board {
    /// Cria um novo tabuleiro a partir de uma string FEN.
    pub fn from_fen(fen: &str) -> Result<Self, String> {
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.len() != 6 {
            return Err("Invalid FEN: Wrong number of parts".to_string());
        }

        let mut board = Board {
            pawns: 0, knights: 0, bishops: 0, rooks: 0, queens: 0, kings: 0,
            white_pieces: 0, black_pieces: 0,
            to_move: Color::White, en_passant_target: None, castling_rights: 0,
            white_king_in_check: false, black_king_in_check: false,
            halfmove_clock: 0, zobrist_hash: 0,
        };

        // Parse board (parts[0])
        let rows: Vec<&str> = parts[0].split('/').collect();
        if rows.len() != 8 {
            return Err("Invalid FEN: Wrong number of rows".to_string());
        }

        let mut sq = 56; // Começa em a8 (rank 8)
        for (i, row) in rows.iter().enumerate() {
            for ch in row.chars() {
                if let Some(digit) = ch.to_digit(10) {
                    sq += digit as u8; // Pula casas vazias
                } else {
                    let bb = 1u64 << sq;
                    let is_white = ch.is_uppercase();
                    let piece = ch.to_ascii_lowercase();
                    match piece {
                        'p' => board.pawns |= bb,
                        'n' => board.knights |= bb,
                        'b' => board.bishops |= bb,
                        'r' => board.rooks |= bb,
                        'q' => board.queens |= bb,
                        'k' => board.kings |= bb,
                        _ => return Err(format!("Invalid piece: {}", ch)),
                    }
                    if is_white {
                        board.white_pieces |= bb;
                    } else {
                        board.black_pieces |= bb;
                    }
                    sq += 1;
                }
            }

            // Só subtrai para ir para a próxima fileira se NÃO for a última
            if i < 7 {
                sq -= 16;
            }
        }

        // To move (parts[1])
        board.to_move = match parts[1] {
            "w" => Color::White,
            "b" => Color::Black,
            _ => return Err("Invalid turn".to_string()),
        };

        // Castling (parts[2])
        for ch in parts[2].chars() {
            match ch {
                'K' => board.castling_rights |= 0b0001,
                'Q' => board.castling_rights |= 0b0010,
                'k' => board.castling_rights |= 0b0100,
                'q' => board.castling_rights |= 0b1000,
                '-' => {}, 
                _ => return Err("Invalid castling".to_string()),
            }
        }

        // En passant (parts[3])
        if parts[3] != "-" {
            let file = (parts[3].as_bytes()[0] - b'a') as u8;
            let rank = (parts[3].as_bytes()[1] - b'1') as u8;
            board.en_passant_target = Some(rank * 8 + file);
        }

        // Halfmove clock (parts[4])
        board.halfmove_clock = parts[4].parse().unwrap_or(0);

        board.update_check_cache();
        board.zobrist_hash = board.compute_zobrist_hash();
        Ok(board)
    }

    /// Cria um novo tabuleiro na posição inicial padrão usando bitboards.
    pub fn new() -> Self {
        const WHITE_PAWNS: Bitboard = 0b00000000_00000000_00000000_00000000_00000000_00000000_11111111_00000000;
        const WHITE_ROOKS: Bitboard = 0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_10000001;
        const WHITE_KNIGHTS: Bitboard = 0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_01000010;
        const WHITE_BISHOPS: Bitboard = 0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00100100;
        const WHITE_QUEEN: Bitboard = 0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00001000;
        const WHITE_KING: Bitboard = 0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00010000;

        const BLACK_PAWNS: Bitboard = 0b00000000_11111111_00000000_00000000_00000000_00000000_00000000_00000000;
        const BLACK_ROOKS: Bitboard = 0b10000001_00000000_00000000_00000000_00000000_00000000_00000000_00000000;
        const BLACK_KNIGHTS: Bitboard = 0b01000010_00000000_00000000_00000000_00000000_00000000_00000000_00000000;
        const BLACK_BISHOPS: Bitboard = 0b00100100_00000000_00000000_00000000_00000000_00000000_00000000_00000000;
        const BLACK_QUEEN: Bitboard = 0b00001000_00000000_00000000_00000000_00000000_00000000_00000000_00000000;
        const BLACK_KING: Bitboard = 0b00010000_00000000_00000000_00000000_00000000_00000000_00000000_00000000;

        let mut board = Board {
            pawns: WHITE_PAWNS | BLACK_PAWNS,
            knights: WHITE_KNIGHTS | BLACK_KNIGHTS,
            bishops: WHITE_BISHOPS | BLACK_BISHOPS,
            rooks: WHITE_ROOKS | BLACK_ROOKS,
            queens: WHITE_QUEEN | BLACK_QUEEN,
            kings: WHITE_KING | BLACK_KING,
            white_pieces: WHITE_PAWNS | WHITE_ROOKS | WHITE_KNIGHTS | WHITE_BISHOPS | WHITE_QUEEN | WHITE_KING,
            black_pieces: BLACK_PAWNS | BLACK_ROOKS | BLACK_KNIGHTS | BLACK_BISHOPS | BLACK_QUEEN | BLACK_KING,
            to_move: Color::White,
            en_passant_target: None,
            castling_rights: 0b1111, // Todos os roques inicialmente permitidos
            white_king_in_check: false,
            black_king_in_check: false,
            halfmove_clock: 0,
            zobrist_hash: 0,
        };
        
        board.zobrist_hash = board.compute_zobrist_hash();
        board
    }
    
    /// Gera todos os lances pseudo-legais para todas as peças do jogador atual.
    pub fn generate_all_moves(&self) -> Vec<Move> {
        // Pre-aloca com capacidade estimada para reduzir realocações
        let mut moves = Vec::with_capacity(64);

        moves.extend(moves::pawn::generate_pawn_moves(self));
        moves.extend(moves::knight::generate_knight_moves(self));
        moves.extend(moves::sliding::generate_sliding_moves(self, PieceKind::Bishop));
        moves.extend(moves::sliding::generate_sliding_moves(self, PieceKind::Rook));
        moves.extend(moves::queen::generate_queen_moves(self));
        moves.extend(moves::king::generate_king_moves(self));

        moves
    }

    pub fn is_capture(&self, mv: Move) -> bool {
        // Um lance é uma captura se a casa de destino está ocupada por uma peça inimiga.
        let to_bb = 1u64 << mv.to;
        let enemy_pieces = if self.to_move == Color::White {
            self.black_pieces
        } else {
            self.white_pieces
        };

        // Também considera o caso especial de en passant, que é uma captura.
        mv.is_en_passant || (to_bb & enemy_pieces) != 0
    }

    pub fn get_piece_on_square(&self, sq: u8) -> Option<PieceKind> {
        let bb = 1u64 << sq;
        if (self.pawns & bb) != 0 { Some(PieceKind::Pawn) }
        else if (self.knights & bb) != 0 { Some(PieceKind::Knight) }
        else if (self.bishops & bb) != 0 { Some(PieceKind::Bishop) }
        else if (self.rooks & bb) != 0 { Some(PieceKind::Rook) }
        else if (self.queens & bb) != 0 { Some(PieceKind::Queen) }
        else if (self.kings & bb) != 0 { Some(PieceKind::King) }
        else { None }
    }



    /// Executa um lance, atualizando o estado do tabuleiro.
    pub fn make_move(&mut self, mv: Move) {
        let from_bb = 1u64 << mv.from;
        let to_bb = 1u64 << mv.to;
        let moving_color = self.to_move;
        let captured_color = !moving_color;

        // --- ATUALIZAÇÃO ZOBRIST - REMOVER ESTADO ANTIGO ---
        self.zobrist_hash ^= ZOBRIST_KEYS.side_to_move;
        if let Some(ep_square) = self.en_passant_target {
            self.zobrist_hash ^= ZOBRIST_KEYS.en_passant[(ep_square % 8) as usize];
        }
        self.zobrist_hash ^= ZOBRIST_KEYS.castling[self.castling_rights as usize];

        // Reset halfmove_clock para capturas ou movimentos de peão
        let is_pawn_move = (self.pawns & from_bb) != 0;
        let is_capture = self.is_capture(mv);
        if is_pawn_move || is_capture {
            self.halfmove_clock = 0;
        } else {
            self.halfmove_clock += 1;
        }

        // Reset do alvo de en passant
        self.en_passant_target = None;

        // Obtém a peça que se está a mover ANTES de alterar os bitboards
        let moving_piece_kind = self.get_piece_on_square(mv.from).unwrap();

        // --- LÓGICA DE MOVIMENTO ---
        // Remove a peça da casa de origem (usando XOR para eficiência)
        let move_bb = from_bb ^ to_bb; // Note: use ^ para toggle se não houver promoção/captura, mas ajuste para casos especiais
        if moving_color == Color::White {
            self.white_pieces &= !from_bb;
        } else {
            self.black_pieces &= !from_bb;
        }
        match moving_piece_kind {
            PieceKind::Pawn => self.pawns &= !from_bb,
            PieceKind::Knight => self.knights &= !from_bb,
            PieceKind::Bishop => self.bishops &= !from_bb,
            PieceKind::Rook => self.rooks &= !from_bb,
            PieceKind::Queen => self.queens &= !from_bb,
            PieceKind::King => self.kings &= !from_bb,
        }
        // Atualiza o hash Zobrist para a peça removida
        self.zobrist_hash ^= ZOBRIST_KEYS.pieces[color_to_index(moving_color)][piece_to_index(moving_piece_kind)][mv.from as usize];

        // Trata capturas
        let mut captured_piece_kind: Option<PieceKind> = None;
        let mut captured_sq = mv.to;
        if mv.is_en_passant {
            captured_sq = if moving_color == Color::White { mv.to - 8 } else { mv.to + 8 };
        }
        let captured_bb = 1u64 << captured_sq;
        if is_capture {
            captured_piece_kind = self.get_piece_on_square(captured_sq);
            if let Some(kind) = captured_piece_kind {
                // Remove a peça capturada
                if captured_color == Color::White {
                    self.white_pieces &= !captured_bb;
                } else {
                    self.black_pieces &= !captured_bb;
                }
                match kind {
                    PieceKind::Pawn => self.pawns &= !captured_bb,
                    PieceKind::Knight => self.knights &= !captured_bb,
                    PieceKind::Bishop => self.bishops &= !captured_bb,
                    PieceKind::Rook => self.rooks &= !captured_bb,
                    PieceKind::Queen => self.queens &= !captured_bb,
                    _ => {}, // Rei não pode ser capturado
                }
                // Atualiza o hash para a peça capturada
                self.zobrist_hash ^= ZOBRIST_KEYS.pieces[color_to_index(captured_color)][piece_to_index(kind)][captured_sq as usize];
            }
        }

        // Adiciona a peça na casa de destino
        let mut piece_on_to_square = moving_piece_kind;
        if let Some(promotion) = mv.promotion {
            piece_on_to_square = promotion;
        }

        if moving_color == Color::White {
            self.white_pieces |= to_bb;
        } else {
            self.black_pieces |= to_bb;
        }
        match piece_on_to_square {
            PieceKind::Pawn => {
                self.pawns |= to_bb;
                if (mv.to as i8 - mv.from as i8).abs() == 16 {
                    self.en_passant_target = Some((mv.from + mv.to) / 2);
                }
            },
            PieceKind::Knight => self.knights |= to_bb,
            PieceKind::Bishop => self.bishops |= to_bb,
            PieceKind::Rook => self.rooks |= to_bb,
            PieceKind::Queen => self.queens |= to_bb,
            PieceKind::King => self.kings |= to_bb,
        }
        // Atualiza o hash para a peça adicionada
        self.zobrist_hash ^= ZOBRIST_KEYS.pieces[color_to_index(moving_color)][piece_to_index(piece_on_to_square)][mv.to as usize];

        // Atualiza direitos de roque
        // 1. Se rei se move (incluindo roque)
        if moving_piece_kind == PieceKind::King {
            if moving_color == Color::White {
                self.castling_rights &= 0b1100; // Limpa K e Q
            } else {
                self.castling_rights &= 0b0011; // Limpa k e q
            }
        }
        // 2. Se torre se move da casa inicial
        if moving_piece_kind == PieceKind::Rook {
            match mv.from {
                0 if moving_color == Color::White => self.castling_rights &= !0b0010, // Q branco (a1)
                7 if moving_color == Color::White => self.castling_rights &= !0b0001, // K branco (h1)
                56 if moving_color == Color::Black => self.castling_rights &= !0b1000, // q preto (a8)
                63 if moving_color == Color::Black => self.castling_rights &= !0b0100, // k preto (h8)
                _ => {},
            }
        }
        // 3. Se torre é capturada na casa inicial
        if let Some(PieceKind::Rook) = captured_piece_kind {
            match captured_sq {
                0 if captured_color == Color::White => self.castling_rights &= !0b0010, // Q branco
                7 if captured_color == Color::White => self.castling_rights &= !0b0001, // K branco
                56 if captured_color == Color::Black => self.castling_rights &= !0b1000, // q preto
                63 if captured_color == Color::Black => self.castling_rights &= !0b0100, // k preto
                _ => {},
            }
        }

        // Trata o movimento da torre no roque
        if mv.is_castling {
            let (rook_from, rook_to) = match mv.to {
                6 => (7, 5),   // Roque pequeno branco (g1)
                2 => (0, 3),   // Roque grande branco (c1)
                62 => (63, 61), // Roque pequeno preto (g8)
                58 => (56, 59), // Roque grande preto (c8)
                _ => unreachable!(),
            };
            let rook_from_bb = 1u64 << rook_from;
            let rook_to_bb = 1u64 << rook_to;
            let rook_move_bb = rook_from_bb ^ rook_to_bb;
            self.rooks ^= rook_move_bb;
            if moving_color == Color::White {
                self.white_pieces ^= rook_move_bb;
            } else {
                self.black_pieces ^= rook_move_bb;
            }

            // Atualiza o hash para o movimento da torre no roque
            self.zobrist_hash ^= ZOBRIST_KEYS.pieces[color_to_index(moving_color)][piece_to_index(PieceKind::Rook)][rook_from as usize];
            self.zobrist_hash ^= ZOBRIST_KEYS.pieces[color_to_index(moving_color)][piece_to_index(PieceKind::Rook)][rook_to as usize];
        }

        // Inverte a vez de jogar
        self.to_move = !self.to_move;

        // Atualiza o cache de xeque
        self.update_check_cache();

        // --- ATUALIZAÇÃO ZOBRIST - ADICIONAR NOVO ESTADO ---
        if let Some(ep_square) = self.en_passant_target {
            self.zobrist_hash ^= ZOBRIST_KEYS.en_passant[(ep_square % 8) as usize];
        }
        self.zobrist_hash ^= ZOBRIST_KEYS.castling[self.castling_rights as usize];
    }

    /// Verifica se o rei da cor especificada está em xeque (usa cache)
    pub fn is_king_in_check(&self, color: Color) -> bool {
        if color == Color::White {
            self.white_king_in_check
        } else {
            self.black_king_in_check
        }
    }
    
    /// Atualiza o cache de estado de xeque para ambos os reis
    fn update_check_cache(&mut self) {
        self.white_king_in_check = self.compute_king_in_check(Color::White);
        self.black_king_in_check = self.compute_king_in_check(Color::Black);
    }
    
    /// Calcula se o rei da cor especificada está em xeque (sem usar cache)
    fn compute_king_in_check(&self, color: Color) -> bool {
        // Encontra a posição do rei
        let king_bb = self.kings & if color == Color::White { self.white_pieces } else { self.black_pieces };
        if king_bb == 0 { return false; } // Não há rei (situação anormal)
        
        let king_square = king_bb.trailing_zeros() as u8;
        
        // Verifica se alguma peça inimiga pode atacar o rei
        self.is_square_attacked_by(king_square, !color)
    }

    /// Verifica se uma casa é atacada por peças da cor especificada
    pub fn is_square_attacked_by(&self, square: u8, attacking_color: Color) -> bool {
        let square_bb = 1u64 << square;
        let attacking_pieces = if attacking_color == Color::White { self.white_pieces } else { self.black_pieces };
        
        // Early exit: se não há peças atacantes, não há ataques
        if attacking_pieces == 0 { return false; }
        
        // Verifica ataques de peões (mais comuns, verificar primeiro)
        if attacking_color == Color::White {
            // Peões brancos atacam diagonalmente para cima
            let pawn_attacks = ((square_bb >> 7) & 0xfefefefefefefefe) | ((square_bb >> 9) & 0x7f7f7f7f7f7f7f7f);
            if (pawn_attacks & self.pawns & attacking_pieces) != 0 { return true; }
        } else {
            // Peões pretos atacam diagonalmente para baixo  
            let pawn_attacks = ((square_bb << 7) & 0x7f7f7f7f7f7f7f7f) | ((square_bb << 9) & 0xfefefefefefefefe);
            if (pawn_attacks & self.pawns & attacking_pieces) != 0 { return true; }
        }
        
        // Verifica ataques de cavalos (rápido)
        if (self.knights & attacking_pieces) != 0 {
            let knight_attacks = self.get_knight_attacks(square);
            if (knight_attacks & self.knights & attacking_pieces) != 0 { return true; }
        }
        
        // Verifica ataques do rei (rápido)
        if (self.kings & attacking_pieces) != 0 {
            let king_attacks = self.get_king_attacks(square);
            if (king_attacks & self.kings & attacking_pieces) != 0 { return true; }
        }
        
        // Verifica ataques de peças deslizantes (mais lento, verificar por último)
        if (self.bishops & attacking_pieces) != 0 || (self.queens & attacking_pieces) != 0 {
            if self.is_attacked_by_sliding_piece(square, attacking_color, true) { return true; }
        }
        
        if (self.rooks & attacking_pieces) != 0 || (self.queens & attacking_pieces) != 0 {
            if self.is_attacked_by_sliding_piece(square, attacking_color, false) { return true; }
        }
        
        false
    }

    fn get_knight_attacks(&self, square: u8) -> u64 {
        crate::moves::knight::get_knight_attacks_lookup(square)
    }

    fn get_king_attacks(&self, square: u8) -> u64 {
        crate::moves::king::get_king_attacks_lookup(square)
    }

    fn is_attacked_by_sliding_piece(&self, square: u8, attacking_color: Color, is_diagonal: bool) -> bool {
        let attacking_pieces = if attacking_color == Color::White { self.white_pieces } else { self.black_pieces };
        let all_pieces = self.white_pieces | self.black_pieces;
        
        let directions = if is_diagonal { &[7i8, 9, -7, -9] } else { &[1i8, -1, 8, -8] };
        let piece_types = if is_diagonal { 
            (self.bishops | self.queens) & attacking_pieces 
        } else { 
            (self.rooks | self.queens) & attacking_pieces 
        };
        
        for &direction in directions {
            let mut current = square as i8;
            loop {
                let prev = current;
                current += direction;
                
                if current < 0 || current >= 64 { break; }
                
                // Verifica wrap-around
                let prev_file = prev % 8;
                let curr_file = current % 8;
                if (curr_file - prev_file).abs() > 1 { break; }
                
                let current_bb = 1u64 << current;
                
                // Se encontrou uma peça atacante do tipo correto
                if (current_bb & piece_types) != 0 { return true; }
                
                // Se encontrou qualquer peça, para a busca nesta direção
                if (current_bb & all_pieces) != 0 { break; }
            }
        }
        false
    }

    /// Verifica se a posição atual é xeque-mate
    pub fn is_checkmate(&self) -> bool {
        if !self.is_king_in_check(self.to_move) {
            return false;
        }
        
        let moves = self.generate_all_moves();
        moves.iter().all(|&mv| {
            let mut temp = *self;
            temp.make_move(mv);
            temp.is_king_in_check(self.to_move)
        })
    }

    /// Verifica se a posição atual é empate por afogamento
    pub fn is_stalemate(&self) -> bool {
        if self.is_king_in_check(self.to_move) {
            return false;
        }
        
        let moves = self.generate_all_moves();
        moves.iter().all(|&mv| {
            let mut temp = *self;
            temp.make_move(mv);
            temp.is_king_in_check(self.to_move)
        })
    }

    /// Verifica se há empate por material insuficiente
    pub fn is_draw_by_insufficient_material(&self) -> bool {
        let total_pieces = self.white_pieces | self.black_pieces;
        let piece_count = total_pieces.count_ones();
        
        // King vs King
        if piece_count == 2 {
            return true;
        }
        
        // King + minor piece vs King
        if piece_count == 3 {
            let has_major_pieces = (self.pawns | self.rooks | self.queens) != 0;
            if !has_major_pieces {
                let minors = self.knights | self.bishops;
                return minors.count_ones() == 1;
            }
        }
        
        // King + Bishop vs King + Bishop (same color squares)
        if piece_count == 4 && (self.pawns | self.rooks | self.queens | self.knights) == 0 {
            let white_bishops = self.bishops & self.white_pieces;
            let black_bishops = self.bishops & self.black_pieces;
            
            if white_bishops.count_ones() == 1 && black_bishops.count_ones() == 1 {
                let light_squares = 0x55AA55AA55AA55AA;
                let white_on_light = (white_bishops & light_squares) != 0;
                let black_on_light = (black_bishops & light_squares) != 0;
                return white_on_light == black_on_light;
            }
        }
        
        false
    }

    /// Verifica se há empate pela regra dos 50 movimentos
    pub fn is_draw_by_50_moves(&self) -> bool {
        self.halfmove_clock >= 100 // 50 movimentos = 100 half-moves
    }

    /// Calcula o hash Zobrist da posição atual
    pub fn compute_zobrist_hash(&self) -> u64 {
        let mut hash = 0u64;
        
        // Hash das peças
        for square in 0..64 {
            let bb = 1u64 << square;
            
            if (self.white_pieces & bb) != 0 {
                let color_idx = color_to_index(Color::White);
                if (self.pawns & bb) != 0 {
                    hash ^= ZOBRIST_KEYS.pieces[color_idx][piece_to_index(PieceKind::Pawn)][square];
                } else if (self.knights & bb) != 0 {
                    hash ^= ZOBRIST_KEYS.pieces[color_idx][piece_to_index(PieceKind::Knight)][square];
                } else if (self.bishops & bb) != 0 {
                    hash ^= ZOBRIST_KEYS.pieces[color_idx][piece_to_index(PieceKind::Bishop)][square];
                } else if (self.rooks & bb) != 0 {
                    hash ^= ZOBRIST_KEYS.pieces[color_idx][piece_to_index(PieceKind::Rook)][square];
                } else if (self.queens & bb) != 0 {
                    hash ^= ZOBRIST_KEYS.pieces[color_idx][piece_to_index(PieceKind::Queen)][square];
                } else if (self.kings & bb) != 0 {
                    hash ^= ZOBRIST_KEYS.pieces[color_idx][piece_to_index(PieceKind::King)][square];
                }
            } else if (self.black_pieces & bb) != 0 {
                let color_idx = color_to_index(Color::Black);
                if (self.pawns & bb) != 0 {
                    hash ^= ZOBRIST_KEYS.pieces[color_idx][piece_to_index(PieceKind::Pawn)][square];
                } else if (self.knights & bb) != 0 {
                    hash ^= ZOBRIST_KEYS.pieces[color_idx][piece_to_index(PieceKind::Knight)][square];
                } else if (self.bishops & bb) != 0 {
                    hash ^= ZOBRIST_KEYS.pieces[color_idx][piece_to_index(PieceKind::Bishop)][square];
                } else if (self.rooks & bb) != 0 {
                    hash ^= ZOBRIST_KEYS.pieces[color_idx][piece_to_index(PieceKind::Rook)][square];
                } else if (self.queens & bb) != 0 {
                    hash ^= ZOBRIST_KEYS.pieces[color_idx][piece_to_index(PieceKind::Queen)][square];
                } else if (self.kings & bb) != 0 {
                    hash ^= ZOBRIST_KEYS.pieces[color_idx][piece_to_index(PieceKind::King)][square];
                }
            }
        }
        
        // Hash dos direitos de roque
        hash ^= ZOBRIST_KEYS.castling[self.castling_rights as usize];
        
        // Hash do en passant
        if let Some(ep_square) = self.en_passant_target {
            hash ^= ZOBRIST_KEYS.en_passant[(ep_square % 8) as usize];
        }
        
        // Hash de quem joga
        if self.to_move == Color::Black {
            hash ^= ZOBRIST_KEYS.side_to_move;
        }
        
        hash
    }

    /// Verifica se o jogo acabou (xeque-mate ou empate)
    pub fn is_game_over(&self) -> bool {
        self.is_checkmate() || self.is_stalemate() || self.is_draw_by_insufficient_material() || self.is_draw_by_50_moves()
    }

    /// Gera apenas movimentos legais (filtra movimentos que deixam o rei em xeque)
    pub fn generate_legal_moves(&self) -> Vec<Move> {
        let pseudo_legal = self.generate_all_moves();
        pseudo_legal.into_iter()
            .filter(|&mv| {
                let mut temp = *self;
                temp.make_move(mv);
                !temp.is_king_in_check(self.to_move)
            })
            .collect()
    }

    /// Verifica se um movimento é legal
    pub fn is_legal_move(&self, mv: Move) -> bool {
        let mut temp = *self;
        temp.make_move(mv);
        !temp.is_king_in_check(self.to_move)
    }

    /// Retorna o número de peças de cada tipo para avaliação
    pub fn piece_count(&self, color: Color, piece_kind: PieceKind) -> u32 {
        let color_pieces = if color == Color::White { self.white_pieces } else { self.black_pieces };
        let piece_bb = match piece_kind {
            PieceKind::Pawn => self.pawns,
            PieceKind::Knight => self.knights,
            PieceKind::Bishop => self.bishops,
            PieceKind::Rook => self.rooks,
            PieceKind::Queen => self.queens,
            PieceKind::King => self.kings,
        };
        (color_pieces & piece_bb).count_ones()
    }

    /// Verifica se há peões passados (útil para avaliação)
    pub fn has_passed_pawn(&self, color: Color) -> bool {
        let my_pawns = if color == Color::White { self.white_pieces } else { self.black_pieces } & self.pawns;
        let enemy_pawns = if color == Color::White { self.black_pieces } else { self.white_pieces } & self.pawns;
        
        let mut bb = my_pawns;
        while bb != 0 {
            let square = bb.trailing_zeros() as u8;
            bb &= bb - 1;
            
            let file = square % 8;
            let rank = square / 8;
            
            let front_span = if color == Color::White {
                let mask = !((1u64 << (rank + 1) * 8) - 1);
                mask & (0x0101010101010101u64 << file)
            } else {
                let mask = (1u64 << (rank * 8)) - 1;
                mask & (0x0101010101010101u64 << file)
            };
            
            // Verifica se há peões inimigos à frente
            if (enemy_pawns & front_span) == 0 {
                return true;
            }
        }
        false
    }
}
