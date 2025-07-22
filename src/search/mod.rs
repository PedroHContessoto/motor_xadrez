// Consts movidas para aqui (pub) para visibilidade global no módulo search
pub const PIECE_VALUES: [i32; 6] = [100, 320, 330, 500, 900, 20000];
pub const MATE_VALUE: i32 = 99999;
pub const LMR_MIN_DEPTH: u8 = 3;
pub const LMR_MIN_MOVES: usize = 4;
pub const LMR_REDUCTION: u8 = 2;
pub const FUTILITY_MARGIN: [i32; 4] = [0, 300, 500, 800];

pub use context::SearchContext;
pub use ordering::order_moves;
pub use see::see;
pub use find_best::{find_best_move, find_best_move_with_time};
pub use aspiration::aspiration_search;
pub use pvs::pvs_search;
pub use quiescence::quiescence_search;

mod context;
mod ordering;
mod see;
mod find_best;
mod aspiration;
mod pvs;
mod quiescence;