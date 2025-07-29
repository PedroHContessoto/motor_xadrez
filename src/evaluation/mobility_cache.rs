// Cache de mobilidade para evitar recálculos dentro da mesma avaliação
use crate::types::{Color, Bitboard};
use std::collections::HashMap;

/// Cache de mobilidade por posição de peça
#[derive(Debug, Clone, Copy, Default)]
pub struct PieceMobilityCache {
    pub knight_mobility: u8,
    pub bishop_mobility: u8,
    pub rook_mobility: u8,
    pub queen_mobility: u8,
    pub king_mobility: u8,
    pub valid: bool,
}

/// Cache principal de mobilidade para ambas as cores
#[derive(Debug)]
pub struct MobilityCache {
    // Cache por quadrado [square][color] 
    piece_cache: [[PieceMobilityCache; 2]; 64],
    // Hash da posição para invalidação
    position_hash: u64,
    // Flag para indicar se cache é válido
    valid: bool,
}

impl Default for MobilityCache {
    fn default() -> Self {
        MobilityCache {
            piece_cache: [[PieceMobilityCache::default(); 2]; 64],
            position_hash: 0,
            valid: false,
        }
    }
}

impl MobilityCache {
    pub fn new() -> Self {
        MobilityCache::default()
    }

    /// Invalida o cache quando a posição muda
    pub fn invalidate(&mut self, new_position_hash: u64) {
        if self.position_hash != new_position_hash {
            self.clear();
            self.position_hash = new_position_hash;
        }
    }

    /// Limpa todo o cache
    pub fn clear(&mut self) {
        for square_cache in &mut self.piece_cache {
            for color_cache in square_cache {
                *color_cache = PieceMobilityCache::default();
            }
        }
        self.valid = false;
    }

    /// Obtém mobilidade de peça específica no cache
    pub fn get_piece_mobility(&self, square: u8, color: Color, piece_type: PieceType) -> Option<u8> {
        if !self.valid || square >= 64 {
            return None;
        }

        let color_idx = color as usize;
        let cache = self.piece_cache[square as usize][color_idx];
        
        if !cache.valid {
            return None;
        }

        match piece_type {
            PieceType::Knight => Some(cache.knight_mobility),
            PieceType::Bishop => Some(cache.bishop_mobility),
            PieceType::Rook => Some(cache.rook_mobility),
            PieceType::Queen => Some(cache.queen_mobility),
            PieceType::King => Some(cache.king_mobility),
        }
    }

    /// Armazena mobilidade de peça específica no cache
    pub fn store_piece_mobility(&mut self, square: u8, color: Color, piece_type: PieceType, mobility: u8) {
        if square >= 64 {
            return;
        }

        let color_idx = color as usize;
        let cache = &mut self.piece_cache[square as usize][color_idx];
        
        match piece_type {
            PieceType::Knight => cache.knight_mobility = mobility,
            PieceType::Bishop => cache.bishop_mobility = mobility,
            PieceType::Rook => cache.rook_mobility = mobility,
            PieceType::Queen => cache.queen_mobility = mobility,
            PieceType::King => cache.king_mobility = mobility,
        }
        
        cache.valid = true;
        self.valid = true;
    }

    /// Verifica se há cache válido para uma cor
    pub fn has_valid_cache(&self, square: u8, color: Color) -> bool {
        if !self.valid || square >= 64 {
            return false;
        }
        
        self.piece_cache[square as usize][color as usize].valid
    }
}

#[derive(Debug, Clone, Copy)]
pub enum PieceType {
    Knight,
    Bishop, 
    Rook,
    Queen,
    King,
}

// Cache global thread-local
thread_local! {
    static MOBILITY_CACHE: std::cell::RefCell<MobilityCache> = std::cell::RefCell::new(MobilityCache::new());
}

/// Função auxiliar para usar o cache global
pub fn with_mobility_cache<F, R>(f: F) -> R 
where
    F: FnOnce(&mut MobilityCache) -> R,
{
    MOBILITY_CACHE.with(|cache| {
        if let Ok(mut cache_ref) = cache.try_borrow_mut() {
            f(&mut cache_ref)
        } else {
            // Se falhar, cria cache temporário
            let mut temp_cache = MobilityCache::new();
            f(&mut temp_cache)
        }
    })
}

/// Função auxiliar para leitura do cache global
pub fn read_mobility_cache<F, R>(f: F) -> R 
where
    F: FnOnce(&MobilityCache) -> R,
{
    MOBILITY_CACHE.with(|cache| {
        if let Ok(cache_ref) = cache.try_borrow() {
            f(&cache_ref)
        } else {
            // Se falhar, cria cache temporário
            let temp_cache = MobilityCache::new();
            f(&temp_cache)
        }
    })
}

/// Limpa o cache de mobilidade (para usar entre jogadas)
pub fn clear_mobility_cache() {
    MOBILITY_CACHE.with(|cache| {
        if let Ok(mut cache_ref) = cache.try_borrow_mut() {
            cache_ref.clear();
        }
    });
}