// Ficheiro: src/types.rs
// Descrição: Módulo para as definições de tipos de dados fundamentais do jogo.

// Um Bitboard é um inteiro de 64 bits sem sinal. Cada bit representa uma casa.
// Bit 0 = a1, Bit 1 = b1, ..., Bit 63 = h8.
pub type Bitboard = u64;

// Enum para representar a cor de uma peça ou de um jogador.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Color {
    White,
    Black,
}

impl std::ops::Not for Color {
    type Output = Color;
    
    fn not(self) -> Self::Output {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

// Enum para representar o tipo de uma peça de xadrez.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PieceKind {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

// Struct para representar uma peça no tabuleiro, combinando o tipo e a cor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Piece {
    pub kind: PieceKind,
    pub color: Color,
}

// Struct para representar um lance no jogo.
// Guarda a casa de origem e a de destino.
#[derive(Debug, Clone, Copy)]
pub struct Move {
    pub from: u8,
    pub to: u8,
    pub promotion: Option<PieceKind>,
    pub is_castling: bool,
    pub is_en_passant: bool,
}