// Sistema de livro de aberturas com principais aberturas do xadrez
use crate::{board::Board, types::Move};
use std::collections::HashMap;

/// Estrutura principal do livro de aberturas
pub struct OpeningBook {
    /// Mapa de hash da posição para lista de movimentos com pesos
    positions: HashMap<u64, Vec<WeightedMove>>,
}

/// Movimento com peso (probabilidade de ser jogado)
#[derive(Debug, Clone)]
struct WeightedMove {
    mv: String,      // Movimento em notação algebraica (ex: "e2e4")
    weight: u32,     // Peso/frequência do movimento (maior = mais provável)
    name: String,    // Nome da abertura/variação
}

impl OpeningBook {
    /// Cria um novo livro de aberturas com todas as principais aberturas
    pub fn new() -> Self {
        let mut book = OpeningBook {
            positions: HashMap::new(),
        };

        book.load_openings();
        book
    }

    /// Busca o melhor movimento para a posição atual
    pub fn get_move(&self, board: &Board) -> Option<(Move, String)> {
        let hash = board.zobrist_hash;

        if let Some(moves) = self.positions.get(&hash) {
            // Escolhe movimento com maior peso (mais comum/forte)
            let best_weighted = moves.iter()
                .max_by_key(|wm| wm.weight)?;

            // Converte string para Move
            if let Some(mv) = self.parse_move(board, &best_weighted.mv) {
                Some((mv, best_weighted.name.clone()))
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Verifica se a posição está no livro de aberturas
    pub fn has_position(&self, board: &Board) -> bool {
        self.positions.contains_key(&board.zobrist_hash)
    }

    /// Carrega todas as aberturas principais
    fn load_openings(&mut self) {
        // Cria um tabuleiro temporário para calcular posições
        let mut board = Board::new();

        // === ABERTURAS DE PEÃO DO REI (1.e4) ===
        self.add_opening_line(&mut board, vec![
            ("e2e4", 100, "Abertura do Peão do Rei"),
        ]);

        // Defesa Siciliana
        self.add_opening_line(&mut board, vec![
            ("e2e4", 100, "Abertura do Peão do Rei"),
            ("c7c5", 95, "Defesa Siciliana"),
        ]);

        self.add_opening_line(&mut board, vec![
            ("e2e4", 100, "Abertura do Peão do Rei"),
            ("c7c5", 95, "Defesa Siciliana"),
            ("g1f3", 90, "Siciliana - Ataque Aberto"),
        ]);

        self.add_opening_line(&mut board, vec![
            ("e2e4", 100, "Abertura do Peão do Rei"),
            ("c7c5", 95, "Defesa Siciliana"),
            ("g1f3", 90, "Siciliana - Ataque Aberto"),
            ("d7d6", 85, "Siciliana - Dragão Acelerado"),
        ]);

        self.add_opening_line(&mut board, vec![
            ("e2e4", 100, "Abertura do Peão do Rei"),
            ("c7c5", 95, "Defesa Siciliana"),
            ("g1f3", 90, "Siciliana - Ataque Aberto"),
            ("b8c6", 85, "Siciliana - Variação Acelerada"),
        ]);

        // Defesa Francesa
        self.add_opening_line(&mut board, vec![
            ("e2e4", 100, "Abertura do Peão do Rei"),
            ("e7e6", 90, "Defesa Francesa"),
        ]);

        self.add_opening_line(&mut board, vec![
            ("e2e4", 100, "Abertura do Peão do Rei"),
            ("e7e6", 90, "Defesa Francesa"),
            ("d2d4", 88, "Francesa - Ataque Principal"),
        ]);

        self.add_opening_line(&mut board, vec![
            ("e2e4", 100, "Abertura do Peão do Rei"),
            ("e7e6", 90, "Defesa Francesa"),
            ("d2d4", 88, "Francesa - Ataque Principal"),
            ("d7d5", 85, "Francesa - Variação Principal"),
        ]);

        // Defesa Caro-Kann
        self.add_opening_line(&mut board, vec![
            ("e2e4", 100, "Abertura do Peão do Rei"),
            ("c7c6", 85, "Defesa Caro-Kann"),
        ]);

        self.add_opening_line(&mut board, vec![
            ("e2e4", 100, "Abertura do Peão do Rei"),
            ("c7c6", 85, "Defesa Caro-Kann"),
            ("d2d4", 83, "Caro-Kann - Variação Principal"),
        ]);

        // Ruy Lopez
        self.add_opening_line(&mut board, vec![
            ("e2e4", 100, "Abertura do Peão do Rei"),
            ("e7e5", 95, "Defesa do Peão do Rei"),
        ]);

        self.add_opening_line(&mut board, vec![
            ("e2e4", 100, "Abertura do Peão do Rei"),
            ("e7e5", 95, "Defesa do Peão do Rei"),
            ("g1f3", 93, "Cavaleiro do Rei"),
        ]);

        self.add_opening_line(&mut board, vec![
            ("e2e4", 100, "Abertura do Peão do Rei"),
            ("e7e5", 95, "Defesa do Peão do Rei"),
            ("g1f3", 93, "Cavaleiro do Rei"),
            ("b8c6", 90, "Defesa do Cavaleiro"),
        ]);

        self.add_opening_line(&mut board, vec![
            ("e2e4", 100, "Abertura do Peão do Rei"),
            ("e7e5", 95, "Defesa do Peão do Rei"),
            ("g1f3", 93, "Cavaleiro do Rei"),
            ("b8c6", 90, "Defesa do Cavaleiro"),
            ("f1b5", 88, "Ruy Lopez"),
        ]);

        self.add_opening_line(&mut board, vec![
            ("e2e4", 100, "Abertura do Peão do Rei"),
            ("e7e5", 95, "Defesa do Peão do Rei"),
            ("g1f3", 93, "Cavaleiro do Rei"),
            ("b8c6", 90, "Defesa do Cavaleiro"),
            ("f1b5", 88, "Ruy Lopez"),
            ("a7a6", 85, "Ruy Lopez - Defesa Morphy"),
        ]);

        // Gambito do Rei
        self.add_opening_line(&mut board, vec![
            ("e2e4", 100, "Abertura do Peão do Rei"),
            ("e7e5", 95, "Defesa do Peão do Rei"),
            ("f2f4", 75, "Gambito do Rei"),
        ]);

        // Italiana
        self.add_opening_line(&mut board, vec![
            ("e2e4", 100, "Abertura do Peão do Rei"),
            ("e7e5", 95, "Defesa do Peão do Rei"),
            ("g1f3", 93, "Cavaleiro do Rei"),
            ("b8c6", 90, "Defesa do Cavaleiro"),
            ("f1c4", 85, "Abertura Italiana"),
        ]);

        self.add_opening_line(&mut board, vec![
            ("e2e4", 100, "Abertura do Peão do Rei"),
            ("e7e5", 95, "Defesa do Peão do Rei"),
            ("g1f3", 93, "Cavaleiro do Rei"),
            ("b8c6", 90, "Defesa do Cavaleiro"),
            ("f1c4", 85, "Abertura Italiana"),
            ("f8c5", 80, "Italiana - Variação Simétrica"),
        ]);

        // === ABERTURAS DE PEÃO DA DAMA (1.d4) ===
        self.add_opening_line(&mut board, vec![
            ("d2d4", 95, "Abertura do Peão da Dama"),
        ]);

        // Defesa Índia do Rei
        self.add_opening_line(&mut board, vec![
            ("d2d4", 95, "Abertura do Peão da Dama"),
            ("g8f6", 90, "Defesa Índia do Rei"),
        ]);

        self.add_opening_line(&mut board, vec![
            ("d2d4", 95, "Abertura do Peão da Dama"),
            ("g8f6", 90, "Defesa Índia do Rei"),
            ("c2c4", 88, "Sistema Inglês"),
        ]);

        self.add_opening_line(&mut board, vec![
            ("d2d4", 95, "Abertura do Peão da Dama"),
            ("g8f6", 90, "Defesa Índia do Rei"),
            ("c2c4", 88, "Sistema Inglês"),
            ("g7g6", 85, "Índia do Rei - Fianchetto"),
        ]);

        // Gambito da Dama
        self.add_opening_line(&mut board, vec![
            ("d2d4", 95, "Abertura do Peão da Dama"),
            ("d7d5", 93, "Defesa Simétrica"),
        ]);

        self.add_opening_line(&mut board, vec![
            ("d2d4", 95, "Abertura do Peão da Dama"),
            ("d7d5", 93, "Defesa Simétrica"),
            ("c2c4", 90, "Gambito da Dama"),
        ]);

        self.add_opening_line(&mut board, vec![
            ("d2d4", 95, "Abertura do Peão da Dama"),
            ("d7d5", 93, "Defesa Simétrica"),
            ("c2c4", 90, "Gambito da Dama"),
            ("e7e6", 85, "Gambito da Dama Recusado"),
        ]);

        self.add_opening_line(&mut board, vec![
            ("d2d4", 95, "Abertura do Peão da Dama"),
            ("d7d5", 93, "Defesa Simétrica"),
            ("c2c4", 90, "Gambito da Dama"),
            ("d5c4", 75, "Gambito da Dama Aceito"),
        ]);

        // Defesa Eslava
        self.add_opening_line(&mut board, vec![
            ("d2d4", 95, "Abertura do Peão da Dama"),
            ("d7d5", 93, "Defesa Simétrica"),
            ("c2c4", 90, "Gambito da Dama"),
            ("c7c6", 80, "Defesa Eslava"),
        ]);

        // Catalan
        self.add_opening_line(&mut board, vec![
            ("d2d4", 95, "Abertura do Peão da Dama"),
            ("g8f6", 90, "Defesa Índia do Rei"),
            ("c2c4", 88, "Sistema Inglês"),
            ("e7e6", 85, "Sistema Clássico"),
            ("g2g3", 80, "Sistema Catalão"),
        ]);

        // === ABERTURAS DE FLANCO ===

        // Abertura Inglesa
        self.add_opening_line(&mut board, vec![
            ("c2c4", 85, "Abertura Inglesa"),
        ]);

        self.add_opening_line(&mut board, vec![
            ("c2c4", 85, "Abertura Inglesa"),
            ("e7e5", 80, "Inglesa - Variação Reversa"),
        ]);

        self.add_opening_line(&mut board, vec![
            ("c2c4", 85, "Abertura Inglesa"),
            ("g8f6", 78, "Inglesa - Sistema Índio"),
        ]);

        // Reti
        self.add_opening_line(&mut board, vec![
            ("g1f3", 80, "Abertura Reti"),
        ]);

        self.add_opening_line(&mut board, vec![
            ("g1f3", 80, "Abertura Reti"),
            ("d7d5", 75, "Reti - Sistema Clássico"),
        ]);

        self.add_opening_line(&mut board, vec![
            ("g1f3", 80, "Abertura Reti"),
            ("g8f6", 75, "Reti - Variação Índia"),
        ]);

        // Abertura dos Pássaros
        self.add_opening_line(&mut board, vec![
            ("f2f4", 60, "Abertura dos Pássaros"),
        ]);

        // === SISTEMAS ESPECIAIS ===

        // Sistema Londres
        self.add_opening_line(&mut board, vec![
            ("d2d4", 95, "Abertura do Peão da Dama"),
            ("g8f6", 90, "Defesa Índia do Rei"),
            ("g1f3", 85, "Cavaleiro do Rei"),
            ("e7e6", 80, "Sistema Clássico"),
            ("c1f4", 75, "Sistema Londres"),
        ]);

        // Sistema Colle
        self.add_opening_line(&mut board, vec![
            ("d2d4", 95, "Abertura do Peão da Dama"),
            ("g8f6", 90, "Defesa Índia do Rei"),
            ("g1f3", 85, "Cavaleiro do Rei"),
            ("e7e6", 80, "Sistema Clássico"),
            ("e2e3", 70, "Sistema Colle"),
        ]);

        // === MAIS VARIAÇÕES DAS ABERTURAS PRINCIPAIS ===
        
        // Ruy Lopez - Mais variações
        self.add_opening_line(&mut board, vec![
            ("e2e4", 100, "Abertura do Peão do Rei"),
            ("e7e5", 95, "Defesa do Peão do Rei"),
            ("g1f3", 93, "Cavaleiro do Rei"),
            ("b8c6", 90, "Defesa do Cavaleiro"),
            ("f1b5", 88, "Ruy Lopez"),
            ("a7a6", 85, "Ruy Lopez - Defesa Morphy"),
            ("b5a4", 83, "Ruy Lopez - Variação Principal"),
            ("g8f6", 80, "Ruy Lopez - Berlin Defense"),
        ]);

        self.add_opening_line(&mut board, vec![
            ("e2e4", 100, "Abertura do Peão do Rei"),
            ("e7e5", 95, "Defesa do Peão do Rei"),
            ("g1f3", 93, "Cavaleiro do Rei"),
            ("b8c6", 90, "Defesa do Cavaleiro"),
            ("f1b5", 88, "Ruy Lopez"),
            ("a7a6", 85, "Ruy Lopez - Defesa Morphy"),
            ("b5a4", 83, "Ruy Lopez - Variação Principal"),
            ("b7b5", 78, "Ruy Lopez - Breyer Defense"),
        ]);

        self.add_opening_line(&mut board, vec![
            ("e2e4", 100, "Abertura do Peão do Rei"),
            ("e7e5", 95, "Defesa do Peão do Rei"),
            ("g1f3", 93, "Cavaleiro do Rei"),
            ("b8c6", 90, "Defesa do Cavaleiro"),
            ("f1b5", 88, "Ruy Lopez"),
            ("f8c5", 82, "Ruy Lopez - Classical Defense"),
        ]);

        // Siciliana - Mais variações
        self.add_opening_line(&mut board, vec![
            ("e2e4", 100, "Abertura do Peão do Rei"),
            ("c7c5", 95, "Defesa Siciliana"),
            ("g1f3", 90, "Siciliana - Ataque Aberto"),
            ("d7d6", 85, "Siciliana - Dragon Acelerado"),
            ("d2d4", 82, "Siciliana - Variação Principal"),
            ("c5d4", 80, "Siciliana - Captura Central"),
            ("f3d4", 78, "Siciliana - Cavalo Central"),
            ("g8f6", 75, "Siciliana - Najdorf Setup"),
        ]);

        self.add_opening_line(&mut board, vec![
            ("e2e4", 100, "Abertura do Peão do Rei"),
            ("c7c5", 95, "Defesa Siciliana"),
            ("g1f3", 90, "Siciliana - Ataque Aberto"),
            ("b8c6", 85, "Siciliana - Variação Acelerada"),
            ("d2d4", 82, "Siciliana - Variação Principal"),
            ("c5d4", 80, "Siciliana - Captura Central"),
            ("f3d4", 78, "Siciliana - Cavalo Central"),
            ("g7g6", 75, "Siciliana - Dragon Variation"),
        ]);

        self.add_opening_line(&mut board, vec![
            ("e2e4", 100, "Abertura do Peão do Rei"),
            ("c7c5", 95, "Defesa Siciliana"),
            ("g1f3", 90, "Siciliana - Ataque Aberto"),
            ("e7e6", 85, "Siciliana - Paulsen Variation"),
            ("d2d4", 82, "Siciliana - Variação Principal"),
            ("c5d4", 80, "Siciliana - Captura Central"),
        ]);

        // Francesa - Mais variações
        self.add_opening_line(&mut board, vec![
            ("e2e4", 100, "Abertura do Peão do Rei"),
            ("e7e6", 90, "Defesa Francesa"),
            ("d2d4", 88, "Francesa - Ataque Principal"),
            ("d7d5", 85, "Francesa - Variação Principal"),
            ("b1c3", 82, "Francesa - Winawer Variation"),
            ("f8b4", 78, "Francesa - Winawer Main Line"),
        ]);

        self.add_opening_line(&mut board, vec![
            ("e2e4", 100, "Abertura do Peão do Rei"),
            ("e7e6", 90, "Defesa Francesa"),
            ("d2d4", 88, "Francesa - Ataque Principal"),
            ("d7d5", 85, "Francesa - Variação Principal"),
            ("e4e5", 80, "Francesa - Advance Variation"),
            ("c7c5", 75, "Francesa - Advance Main Line"),
        ]);

        // Caro-Kann - Mais variações
        self.add_opening_line(&mut board, vec![
            ("e2e4", 100, "Abertura do Peão do Rei"),
            ("c7c6", 85, "Defesa Caro-Kann"),
            ("d2d4", 83, "Caro-Kann - Variação Principal"),
            ("d7d5", 80, "Caro-Kann - Main Line"),
            ("b1c3", 78, "Caro-Kann - Classical"),
            ("d5e4", 75, "Caro-Kann - Exchange"),
        ]);

        self.add_opening_line(&mut board, vec![
            ("e2e4", 100, "Abertura do Peão do Rei"),
            ("c7c6", 85, "Defesa Caro-Kann"),
            ("d2d4", 83, "Caro-Kann - Variação Principal"),
            ("d7d5", 80, "Caro-Kann - Main Line"),
            ("e4d5", 75, "Caro-Kann - Exchange Variation"),
            ("c6d5", 72, "Caro-Kann - Exchange Main"),
        ]);

        // Italiana - Mais desenvolvimento
        self.add_opening_line(&mut board, vec![
            ("e2e4", 100, "Abertura do Peão do Rei"),
            ("e7e5", 95, "Defesa do Peão do Rei"),
            ("g1f3", 93, "Cavaleiro do Rei"),
            ("b8c6", 90, "Defesa do Cavaleiro"),
            ("f1c4", 85, "Abertura Italiana"),
            ("f8c5", 80, "Italiana - Variação Simétrica"),
            ("c2c3", 75, "Italiana - Classical Setup"),
            ("f7f5", 70, "Italiana - Rousseau Gambit"),
        ]);

        self.add_opening_line(&mut board, vec![
            ("e2e4", 100, "Abertura do Peão do Rei"),
            ("e7e5", 95, "Defesa do Peão do Rei"),
            ("g1f3", 93, "Cavaleiro do Rei"),
            ("b8c6", 90, "Defesa do Cavaleiro"),
            ("f1c4", 85, "Abertura Italiana"),
            ("f8e7", 80, "Italiana - Hungarian Defense"),
        ]);

        // Gambito da Dama - Mais variações
        self.add_opening_line(&mut board, vec![
            ("d2d4", 95, "Abertura do Peão da Dama"),
            ("d7d5", 93, "Defesa Simétrica"),
            ("c2c4", 90, "Gambito da Dama"),
            ("e7e6", 85, "Gambito da Dama Recusado"),
            ("b1c3", 82, "QGD - Orthodox Defense"),
            ("g8f6", 80, "QGD - Main Line"),
            ("c1g5", 75, "QGD - Exchange Variation"),
        ]);

        self.add_opening_line(&mut board, vec![
            ("d2d4", 95, "Abertura do Peão da Dama"),
            ("d7d5", 93, "Defesa Simétrica"),
            ("c2c4", 90, "Gambito da Dama"),
            ("d5c4", 75, "Gambito da Dama Aceito"),
            ("g1f3", 72, "QGA - Main Line"),
            ("g8f6", 70, "QGA - Classical"),
        ]);

        // Eslava - Mais variações
        self.add_opening_line(&mut board, vec![
            ("d2d4", 95, "Abertura do Peão da Dama"),
            ("d7d5", 93, "Defesa Simétrica"),
            ("c2c4", 90, "Gambito da Dama"),
            ("c7c6", 80, "Defesa Eslava"),
            ("g1f3", 78, "Eslava - Main Line"),
            ("g8f6", 75, "Eslava - Classical"),
            ("b1c3", 72, "Eslava - Orthodox"),
        ]);

        // King's Indian - Mais desenvolvimento
        self.add_opening_line(&mut board, vec![
            ("d2d4", 95, "Abertura do Peão da Dama"),
            ("g8f6", 90, "Defesa Índia do Rei"),
            ("c2c4", 88, "Sistema Inglês"),
            ("g7g6", 85, "Índia do Rei - Fianchetto"),
            ("b1c3", 82, "KID - Classical Setup"),
            ("f8g7", 80, "KID - Main Line"),
            ("e2e4", 75, "KID - Four Pawns Attack"),
        ]);

        self.add_opening_line(&mut board, vec![
            ("d2d4", 95, "Abertura do Peão da Dama"),
            ("g8f6", 90, "Defesa Índia do Rei"),
            ("c2c4", 88, "Sistema Inglês"),
            ("g7g6", 85, "Índia do Rei - Fianchetto"),
            ("g2g3", 78, "KID - Fianchetto Variation"),
            ("f8g7", 75, "KID - Fianchetto Main"),
        ]);

        // Nimzo-Indian
        self.add_opening_line(&mut board, vec![
            ("d2d4", 95, "Abertura do Peão da Dama"),
            ("g8f6", 90, "Defesa Índia do Rei"),
            ("c2c4", 88, "Sistema Inglês"),
            ("e7e6", 85, "Sistema Clássico"),
            ("b1c3", 82, "Nimzo-Indian Setup"),
            ("f8b4", 80, "Nimzo-Indian Defense"),
        ]);

        // Abertura Inglesa - Mais variações
        self.add_opening_line(&mut board, vec![
            ("c2c4", 85, "Abertura Inglesa"),
            ("e7e5", 80, "Inglesa - Variação Reversa"),
            ("b1c3", 75, "Inglesa - Closed System"),
            ("b8c6", 70, "Inglesa - Closed Main"),
        ]);

        self.add_opening_line(&mut board, vec![
            ("c2c4", 85, "Abertura Inglesa"),
            ("g8f6", 78, "Inglesa - Sistema Índio"),
            ("b1c3", 75, "Inglesa - Three Knights"),
            ("e7e6", 70, "Inglesa - Symmetrical"),
        ]);

        self.add_opening_line(&mut board, vec![
            ("c2c4", 85, "Abertura Inglesa"),
            ("c7c5", 75, "Inglesa - Symmetrical"),
            ("b1c3", 70, "Inglesa - Symmetrical Main"),
            ("b8c6", 65, "Inglesa - Four Knights"),
        ]);

        // Reti - Mais desenvolvimento
        self.add_opening_line(&mut board, vec![
            ("g1f3", 80, "Abertura Reti"),
            ("d7d5", 75, "Reti - Sistema Clássico"),
            ("c2c4", 70, "Reti - Queen's Indian Setup"),
            ("e7e6", 65, "Reti - Transpose to QID"),
        ]);

        self.add_opening_line(&mut board, vec![
            ("g1f3", 80, "Abertura Reti"),
            ("g8f6", 75, "Reti - Variação Índia"),
            ("c2c4", 70, "Reti - English Setup"),
            ("c7c5", 65, "Reti - Symmetrical"),
        ]);

        // Scandinavian Defense
        self.add_opening_line(&mut board, vec![
            ("e2e4", 100, "Abertura do Peão do Rei"),
            ("d7d5", 70, "Defesa Escandinava"),
            ("e4d5", 68, "Scandinavian - Main Line"),
            ("d8d5", 65, "Scandinavian - Modern"),
        ]);

        // Alekhine Defense
        self.add_opening_line(&mut board, vec![
            ("e2e4", 100, "Abertura do Peão do Rei"),
            ("g8f6", 75, "Defesa Alekhine"),
            ("e4e5", 72, "Alekhine - Advance Variation"),
            ("f6d5", 70, "Alekhine - Main Line"),
        ]);

        // Pirc Defense
        self.add_opening_line(&mut board, vec![
            ("e2e4", 100, "Abertura do Peão do Rei"),
            ("d7d6", 78, "Defesa Pirc"),
            ("d2d4", 75, "Pirc - Main Line"),
            ("g8f6", 72, "Pirc - Classical"),
            ("b1c3", 70, "Pirc - Austrian Attack Setup"),
        ]);

        // Modern Defense
        self.add_opening_line(&mut board, vec![
            ("e2e4", 100, "Abertura do Peão do Rei"),
            ("g7g6", 75, "Defesa Moderna"),
            ("d2d4", 72, "Modern - Main Line"),
            ("f8g7", 70, "Modern - Fianchetto"),
        ]);

        // Dutch Defense
        self.add_opening_line(&mut board, vec![
            ("d2d4", 95, "Abertura do Peão da Dama"),
            ("f7f5", 72, "Defesa Holandesa"),
            ("g2g3", 70, "Dutch - Fianchetto"),
            ("g8f6", 68, "Dutch - Leningrad"),
        ]);

        // Benoni Defense
        self.add_opening_line(&mut board, vec![
            ("d2d4", 95, "Abertura do Peão da Dama"),
            ("g8f6", 90, "Defesa Índia do Rei"),
            ("c2c4", 88, "Sistema Inglês"),
            ("c7c5", 75, "Benoni Defense"),
            ("d4d5", 72, "Benoni - Advance"),
        ]);

        // Budapest Gambit
        self.add_opening_line(&mut board, vec![
            ("d2d4", 95, "Abertura do Peão da Dama"),
            ("g8f6", 90, "Defesa Índia do Rei"),
            ("c2c4", 88, "Sistema Inglês"),
            ("e7e5", 65, "Gambito de Budapeste"),
        ]);

        // Livro carregado silenciosamente para compatibilidade UCI
    }

    /// Adiciona uma linha de abertura ao livro
    fn add_opening_line(&mut self, board: &mut Board, moves: Vec<(&str, u32, &str)>) {
        *board = Board::new(); // Reset para posição inicial

        for (i, (move_str, weight, name)) in moves.iter().enumerate() {
            let hash = board.zobrist_hash;

            // Adiciona o movimento para esta posição
            if let Some(mv) = self.parse_move(board, move_str) {
                let weighted_move = WeightedMove {
                    mv: move_str.to_string(),
                    weight: *weight,
                    name: name.to_string(),
                };

                self.positions
                    .entry(hash)
                    .or_insert_with(Vec::new)
                    .push(weighted_move);

                // Faz o movimento para continuar a linha
                board.make_move(mv);
            } else {
                println!("⚠️  Erro ao parsear movimento: {} na posição {}", move_str, i);
                break;
            }
        }
    }

    /// Converte string de movimento para Move (similar ao main.rs)
    fn parse_move(&self, board: &Board, move_str: &str) -> Option<Move> {
        let legal_moves = board.generate_legal_moves();
        for mv in legal_moves {
            let mv_str = mv.to_string();
            // Tenta match exato primeiro
            if mv_str == *move_str {
                return Some(mv);
            }
            // Tenta sem a notação de captura 'x'
            let clean_move_str = move_str.replace("x", "").replace("+", "").replace("#", "");
            if mv_str == clean_move_str {
                return Some(mv);
            }
        }
        None
    }

    /// Retorna estatísticas do livro de aberturas
    pub fn stats(&self) -> (usize, usize) {
        let positions = self.positions.len();
        let total_moves: usize = self.positions.values().map(|moves| moves.len()).sum();
        (positions, total_moves)
    }

    /// Lista todas as aberturas disponíveis para uma posição
    pub fn get_all_moves(&self, board: &Board) -> Vec<(String, u32, String)> {
        if let Some(moves) = self.positions.get(&board.zobrist_hash) {
            moves.iter()
                .map(|wm| (wm.mv.clone(), wm.weight, wm.name.clone()))
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Adiciona um movimento ao livro de aberturas
    pub fn add_move(&mut self, hash: u64, move_str: String, weight: u32, name: String) {
        let weighted_move = WeightedMove {
            mv: move_str,
            weight,
            name,
        };

        self.positions
            .entry(hash)
            .or_insert_with(Vec::new)
            .push(weighted_move);
    }

    /// Retorna todos os hashes de posição no livro
    pub fn get_all_position_hashes(&self) -> Vec<u64> {
        self.positions.keys().cloned().collect()
    }
}

/// Função de conveniência para verificar se estamos ainda na abertura
pub fn is_in_opening_phase(board: &Board) -> bool {
    // Considera fase de abertura se:
    // 1. Menos de 16 lances foram jogados (aprox. 8 por lado)
    // 2. Muitas peças ainda nas posições iniciais
    let total_pieces = (board.white_pieces | board.black_pieces).count_ones();
    let back_rank_pieces = (board.white_pieces | board.black_pieces) &
        (0xFF | 0xFF00000000000000); // Ranks 1 e 8

    total_pieces >= 28 && back_rank_pieces.count_ones() >= 12
}