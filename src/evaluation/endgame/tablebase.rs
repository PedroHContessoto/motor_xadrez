// Interface para tablebases futuras (Syzygy, etc.)
use crate::{board::Board, types::Move};

/// Resultado de consulta a tablebase
#[derive(Debug, Clone)]
pub struct TablebaseResult {
    pub is_win: bool,
    pub is_draw: bool,
    pub is_loss: bool,
    pub distance_to_mate: Option<u8>,
    pub best_move: Option<Move>,
}

/// Interface para tablebases
pub trait TablebaseInterface {
    fn probe(&self, board: &Board) -> Option<TablebaseResult>;
    fn is_available(&self) -> bool;
    fn max_pieces(&self) -> u8;
}

/// Implementação placeholder para Syzygy
pub struct SyzygyTablebase {
    available: bool,
    max_pieces: u8,
}

impl SyzygyTablebase {
    pub fn new() -> Self {
        Self {
            available: false, // Por enquanto não implementado
            max_pieces: 7,    // Syzygy padrão suporta até 7 peças
        }
    }
    
    pub fn load_from_path(&mut self, _path: &str) -> Result<(), String> {
        // Implementação futura
        Err("Not implemented yet".to_string())
    }
}

impl TablebaseInterface for SyzygyTablebase {
    fn probe(&self, _board: &Board) -> Option<TablebaseResult> {
        if !self.available {
            return None;
        }
        
        // Implementação futura - consulta real à tablebase
        None
    }
    
    fn is_available(&self) -> bool {
        self.available
    }
    
    fn max_pieces(&self) -> u8 {
        self.max_pieces
    }
}

impl Default for SyzygyTablebase {
    fn default() -> Self {
        Self::new()
    }
}

/// Coordenador de tablebases
pub struct TablebaseCoordinator {
    syzygy: SyzygyTablebase,
}

impl TablebaseCoordinator {
    pub fn new() -> Self {
        Self {
            syzygy: SyzygyTablebase::new(),
        }
    }
    
    /// Consulta tablebases disponíveis
    pub fn probe(&self, board: &Board) -> Option<TablebaseResult> {
        let total_pieces = (board.white_pieces | board.black_pieces).count_ones() as u8;
        
        // Verifica se está dentro do limite das tablebases
        if self.syzygy.is_available() && total_pieces <= self.syzygy.max_pieces() {
            return self.syzygy.probe(board);
        }
        
        None
    }
    
    /// Carrega tablebases de um diretório
    pub fn load_syzygy(&mut self, path: &str) -> Result<(), String> {
        self.syzygy.load_from_path(path)
    }
}

impl Default for TablebaseCoordinator {
    fn default() -> Self {
        Self::new()
    }
}