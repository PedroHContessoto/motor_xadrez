# Motor Xadrez 🏆

High-Performance Chess Engine written in Rust

## Features ✨

- **Bitboard representation** for maximum performance
- **Zobrist hashing** for position repetition detection
- **Complete move generation** including castling, en passant, and promotion
- **Draw detection** (50-move rule, insufficient material, stalemate)
- **FEN parsing** for position setup
- **60M+ NPS** performance in perft tests
- **Memory efficient** with optimized data structures

## Performance 🚀

- **Perft(7)**: 3.2 billion nodes in ~50 seconds
- **NPS**: 60-70 million nodes per second
- **Parallelized** with Rayon for multi-core performance
- **Validated** against standard chess programming test suites

## Usage 📖

### As a Library

```rust
use motor_xadrez::{Board, Color, PieceKind};

fn main() {
    // Create new board in starting position
    let board = Board::new();
    
    // Generate legal moves
    let moves = board.generate_legal_moves();
    println!("Available moves: {}", moves.len());
    
    // Check game state
    if board.is_game_over() {
        println!("Game over!");
    }
    
    // Get piece counts
    let white_pawns = board.piece_count(Color::White, PieceKind::Pawn);
    
    // Parse FEN
    let fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1";
    let board = Board::from_fen(fen).unwrap();
}
```

### Run Demo

```bash
cargo run --release
```

## API Reference 📚

### Board Methods

- `Board::new()` - Create starting position
- `Board::from_fen(fen)` - Parse FEN string
- `generate_legal_moves()` - Get all legal moves
- `make_move(mv)` - Execute a move
- `is_game_over()` - Check if game ended
- `is_checkmate()` - Check for checkmate
- `is_stalemate()` - Check for stalemate
- `is_king_in_check(color)` - Check if king is in check
- `piece_count(color, piece)` - Count pieces of type

### Core Types

- `Color` - White or Black
- `PieceKind` - Pawn, Knight, Bishop, Rook, Queen, King
- `Move` - Represents a chess move
- `Bitboard` - 64-bit integer for board representation

## Architecture 🔧

- **`board.rs`** - Main board representation and game logic
- **`moves/`** - Move generation for each piece type
- **`types.rs`** - Core data structures
- **`zobrist.rs`** - Hashing system for position repetition

## Validation ✅

The engine has been extensively tested with:
- Standard perft test suites
- Tactical positions
- Endgame scenarios
- Edge cases (castling, en passant, promotion)

## Next Steps 🎯

This engine provides a solid foundation for:
- **Position evaluation** functions
- **Search algorithms** (minimax, alpha-beta)
- **UCI protocol** implementation
- **Opening book** integration
- **Endgame tablebase** support

## License 📄

This project is open source and available under the MIT License.