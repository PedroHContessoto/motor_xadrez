# Motor Xadrez - High-Performance Chess Engine

A bitboard-based chess engine written in Rust, optimized for maximum performance while maintaining 100% correctness.

## Performance

- **65M NPS** (nodes per second) at depth 6 in release mode
- **4.2M move generations** per second  
- All perft results match reference values exactly

## Results Validation

| Depth | Expected | Time (release) | NPS |
|-------|----------|---------------|-----|
| 1 | 20 | 0.000s | 1.7M |
| 2 | 400 | 0.000s | 5.9M |
| 3 | 8,902 | 0.002s | 5.5M |
| 4 | 197,281 | 0.010s | 20.7M |
| 5 | 4,865,609 | 0.096s | 50.6M |
| 6 | 119,060,324 | 1.832s | **65.0M** |

## Usage

```bash
# Quick validation (depths 1-5)
cargo run --release

# Full test (depths 1-6) 
cargo run --release -- --full

# Performance benchmark
cargo run --release -- benchmark
```

## Features

- ✅ Complete chess move generation
- ✅ Bitboard representation for efficiency  
- ✅ Parallel processing with Rayon
- ✅ Optimized release profile with LTO
- ✅ Perft validation for correctness
- ✅ Clean, minimal codebase

## Architecture

- `board.rs` - Bitboard representation and move execution
- `moves/` - Move generation for each piece type
- `perft.rs` - Performance testing and validation
- `types.rs` - Core data structures

Built with Rust 2021 edition for maximum performance and safety.