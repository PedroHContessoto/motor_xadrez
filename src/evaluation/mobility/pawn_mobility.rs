// Mobilidade de peões - Integração com sistema avançado de pawn_structure
use crate::types::Color;
use super::MobilityContext;
use super::super::pawn_structure;

/// Avaliação de mobilidade de peões (integrada com pawn_structure.rs)
pub fn evaluate_pawn_mobility_advanced(context: &MobilityContext) -> i32 {
    // Delega para o sistema integrado avançado em pawn_structure.rs
    // que agora inclui toda a funcionalidade de mobilidade
    let analysis = pawn_structure::evaluate_pawn_structure(&context.board, context.color);
    
    // Retorna apenas os componentes de mobilidade/dinâmica
    analysis
}

// Mantém funções auxiliares para compatibilidade se necessário
pub use super::super::pawn_structure::{get_set_bits_simple, evaluate_pawn_structure};

/// Análise rápida de mobilidade para uso em outras avaliações
pub fn quick_pawn_mobility_analysis(context: &MobilityContext) -> i32 {
    let our_pawns = context.board.pawns & context.our_pieces;
    if our_pawns == 0 { return 0; }
    
    let mut mobility = 0;
    let all_pieces = context.all_pieces;
    let mut pawns = our_pawns;
    
    while pawns != 0 {
        let sq = pawns.trailing_zeros() as u8;
        pawns &= pawns - 1;
        
        let rank = sq / 8;
        
        // Avanço simples
        let one_step = match context.color {
            Color::White => if rank >= 7 { continue; } else { sq + 8 },
            Color::Black => if rank <= 0 { continue; } else { sq - 8 },
        };
        
        if (all_pieces & (1u64 << one_step)) == 0 {
            mobility += 1;
            
            // Avanço duplo
            let can_double = match context.color {
                Color::White => rank == 1,
                Color::Black => rank == 6,
            };
            
            if can_double {
                let two_steps = match context.color {
                    Color::White => sq + 16,
                    Color::Black => sq - 16,
                };
                
                if (all_pieces & (1u64 << two_steps)) == 0 {
                    mobility += 1;
                }
            }
        }
    }
    
    mobility * 2 // Valor base por movimento possível
}