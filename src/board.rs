// Ficheiro: src/board.rs
// Descrição: Módulo que contém a struct Board e os seus métodos principais.

use super::types::*;
use crate::moves;

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
}

impl Board {
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

        Board {
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
        }
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

    /// Executa um lance, atualizando o estado do tabuleiro.
    pub fn make_move(&mut self, mv: Move) {
        let from_bb = 1u64 << mv.from;
        let to_bb = 1u64 << mv.to;
        let moving_color = self.to_move;

        // Reset en passant target
        self.en_passant_target = None;

        // Trata roque
        if mv.is_castling {
            // Move o rei
            if moving_color == Color::White {
                self.white_pieces ^= from_bb | to_bb;
                self.kings ^= from_bb | to_bb;
                
                // Move a torre correspondente
                if mv.to == 6 { // Roque pequeno
                    self.white_pieces ^= 0b10000000 | 0b00100000; // h1 -> f1
                    self.rooks ^= 0b10000000 | 0b00100000;
                } else { // Roque grande
                    self.white_pieces ^= 0b00000001 | 0b00001000; // a1 -> d1
                    self.rooks ^= 0b00000001 | 0b00001000;
                }
                // Remove direitos de roque das brancas
                self.castling_rights &= 0b1100;
            } else {
                self.black_pieces ^= from_bb | to_bb;
                self.kings ^= from_bb | to_bb;
                
                // Move a torre correspondente
                if mv.to == 62 { // Roque pequeno
                    self.black_pieces ^= 0x8000000000000000 | 0x2000000000000000; // h8 -> f8
                    self.rooks ^= 0x8000000000000000 | 0x2000000000000000;
                } else { // Roque grande
                    self.black_pieces ^= 0x0100000000000000 | 0x0800000000000000; // a8 -> d8
                    self.rooks ^= 0x0100000000000000 | 0x0800000000000000;
                }
                // Remove direitos de roque das pretas
                self.castling_rights &= 0b0011;
            }
        } else if mv.is_en_passant {
            // En passant: remove o peão capturado
            let captured_pawn_square = if moving_color == Color::White { mv.to - 8 } else { mv.to + 8 };
            let captured_pawn_bb = 1u64 << captured_pawn_square;
            
            // Remove o peão capturado
            self.pawns &= !captured_pawn_bb;
            if moving_color == Color::White {
                self.black_pieces &= !captured_pawn_bb;
                self.white_pieces ^= from_bb | to_bb;
            } else {
                self.white_pieces &= !captured_pawn_bb;
                self.black_pieces ^= from_bb | to_bb;
            }
            self.pawns ^= from_bb | to_bb;
        } else {
            let move_bb = from_bb | to_bb;
            let enemy_pieces = if moving_color == Color::White { self.black_pieces } else { self.white_pieces };
            let is_capture = (enemy_pieces & to_bb) != 0;

            // Trata capturas normais
            if is_capture {
                if moving_color == Color::White {
                    self.black_pieces &= !to_bb;
                } else {
                    self.white_pieces &= !to_bb;
                }
                if (self.pawns & to_bb) != 0 { self.pawns &= !to_bb; }
                else if (self.knights & to_bb) != 0 { self.knights &= !to_bb; }
                else if (self.bishops & to_bb) != 0 { self.bishops &= !to_bb; }
                else if (self.rooks & to_bb) != 0 { self.rooks &= !to_bb; }
                else if (self.queens & to_bb) != 0 { self.queens &= !to_bb; }
            }

            if let Some(promotion) = mv.promotion {
                // Promoção: remove o peão e adiciona a peça promovida
                self.pawns &= !from_bb;
                match promotion {
                    PieceKind::Queen => self.queens |= to_bb,
                    PieceKind::Rook => self.rooks |= to_bb,
                    PieceKind::Bishop => self.bishops |= to_bb,
                    PieceKind::Knight => self.knights |= to_bb,
                    _ => unreachable!(),
                }
                if moving_color == Color::White {
                    self.white_pieces &= !from_bb;
                    self.white_pieces |= to_bb;
                } else {
                    self.black_pieces &= !from_bb;
                    self.black_pieces |= to_bb;
                }
            } else {
                // Movimento normal
                if moving_color == Color::White {
                    self.white_pieces ^= move_bb;
                } else {
                    self.black_pieces ^= move_bb;
                }
                
                if (self.pawns & from_bb) != 0 { 
                    self.pawns ^= move_bb;
                    // Verifica movimento duplo de peão para en passant
                    if (mv.to as i8 - mv.from as i8).abs() == 16 {
                        self.en_passant_target = Some((mv.from + mv.to) / 2);
                    }
                }
                else if (self.knights & from_bb) != 0 { self.knights ^= move_bb; }
                else if (self.bishops & from_bb) != 0 { self.bishops ^= move_bb; }
                else if (self.rooks & from_bb) != 0 { self.rooks ^= move_bb; }
                else if (self.queens & from_bb) != 0 { self.queens ^= move_bb; }
                else if (self.kings & from_bb) != 0 { 
                    self.kings ^= move_bb;
                    // Remove direitos de roque quando o rei se move
                    if moving_color == Color::White {
                        self.castling_rights &= 0b1100;
                    } else {
                        self.castling_rights &= 0b0011;
                    }
                }
            }
        }

        // Atualiza direitos de roque quando torres se movem
        if mv.from == 0 || mv.to == 0 { self.castling_rights &= 0b1101; } // a1
        if mv.from == 7 || mv.to == 7 { self.castling_rights &= 0b1110; } // h1
        if mv.from == 56 || mv.to == 56 { self.castling_rights &= 0b0111; } // a8
        if mv.from == 63 || mv.to == 63 { self.castling_rights &= 0b1011; } // h8

        self.to_move = if moving_color == Color::White { Color::Black } else { Color::White };
        
        // Atualiza cache de xeque
        self.update_check_cache();
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
}
