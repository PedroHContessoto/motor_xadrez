// Motor Xadrez - High-Performance Chess Engine Library

pub mod types;
pub mod board;
pub mod moves;
pub mod zobrist;
pub mod nnue;
pub mod evaluation;
pub mod selfplay;

pub use types::*;
pub use board::Board;
pub use evaluation::Evaluator;
pub use nnue::{NNUEEvaluator, NNUEConfig};
pub use selfplay::{run_selfplay_training, SelfPlayConfig};