// Ficheiro: src/main.rs
// Descrição: Ponto de entrada com o loop principal do protocolo UCI.

use motor_xadrez::{Board, Move};
use motor_xadrez::evaluation;
use motor_xadrez::search;
use motor_xadrez::transposition::TranspositionTable;
use motor_xadrez::opening_book::{OpeningBook, is_in_opening_phase};
use std::io;
use std::time::Instant;
use std::io::Write; // Importa o trait Write para usar flush()

// Estrutura para gerenciar tempos de forma inteligente
#[derive(Debug)]
struct TimeManager {
    wtime: Option<u64>,    // Tempo restante das brancas em ms
    btime: Option<u64>,    // Tempo restante das pretas em ms
    winc: Option<u64>,     // Incremento por lance das brancas em ms
    binc: Option<u64>,     // Incremento por lance das pretas em ms
    movetime: Option<u64>, // Tempo fixo por lance em ms
    depth: Option<u8>,     // Profundidade fixa
    nodes: Option<u64>,    // Número máximo de nós
    infinite: bool,        // Busca infinita
}

#[derive(Debug, Default)]
struct PositionComplexity {
    is_tactical: bool,           // Posição tem características táticas
    is_critical: bool,           // Posição crítica que precisa de muito tempo
    in_check: bool,              // Estamos em xeque
    has_hanging_pieces: bool,    // Temos peças penduradas
    attacked_pieces_count: u32,  // Número de nossas peças atacadas
    piece_density: f32,          // Densidade de peças no tabuleiro (0.0-1.0)
    king_proximity: u8,          // Proximidade dos reis (0-8, maior = mais próximos)
}

impl TimeManager {
    fn new() -> Self {
        TimeManager {
            wtime: None,
            btime: None,
            winc: None,
            binc: None,
            movetime: None,
            depth: None,
            nodes: None,
            infinite: false,
        }
    }

    // Calcula o tempo ótimo para este lance baseado na situação
    fn calculate_time_for_move(&self, board: &Board, moves_played: u16) -> u64 {
        if let Some(movetime) = self.movetime {
            return movetime;
        }

        if self.infinite {
            return u64::MAX; // Retorna tempo máximo se a busca for infinita
        }

        let (my_time, my_inc) = if board.to_move == motor_xadrez::types::Color::White {
            (self.wtime, self.winc)
        } else {
            (self.btime, self.binc)
        };

        if let Some(time_left) = my_time {
            // Análise de complexidade da posição
            let complexity = self.analyze_position_complexity(board);

            // Base mais generosa para o divisor
            let mut base_divisor = if moves_played < 10 {
                25  // Abertura: mais tempo
            } else if moves_played < 25 {
                15  // Meio-jogo: muito mais tempo
            } else if moves_played < 50 {
                20  // Final médio
            } else {
                15  // Final: mais tempo para precisão
            };

            // Ajustes menos agressivos baseados na complexidade
            if complexity.is_tactical {
                // Posição tática: 50% mais tempo
                base_divisor = (base_divisor as f32 * 0.67).round() as u64; // Usar round para evitar truncamento
                if complexity.has_hanging_pieces {
                    base_divisor = base_divisor.saturating_sub(3);
                }
                if complexity.in_check {
                    base_divisor = base_divisor.saturating_sub(5);
                }
                if complexity.is_critical {
                    base_divisor = base_divisor.saturating_sub(4);
                }
            }

            // Garante que o divisor nunca seja zero ou muito pequeno
            base_divisor = base_divisor.max(10);

            // Calcula o tempo base
            let base_time = time_left / base_divisor;

            // Usa incremento de forma mais inteligente
            let increment_bonus = if let Some(inc) = my_inc {
                if time_left < 30000 { // Menos de 30 segundos
                    inc * 4 / 5  // Usa 80% do incremento
                } else {
                    inc * 3 / 4  // Usa 75% do incremento
                }
            } else {
                0
            };

            let mut time_with_increment = base_time + increment_bonus;

            // Bônus maior para posições críticas
            if complexity.is_critical {
                time_with_increment = (time_with_increment as f32 * 1.5).round() as u64;
            } else if complexity.is_tactical {
                time_with_increment = (time_with_increment as f32 * 1.3).round() as u64;
            }

            // Limites ajustados para o tempo final
            let min_time = if time_left > 60000 {
                1000  // 1 segundo mínimo com muito tempo
            } else if time_left > 10000 {
                500   // 0.5 segundos com tempo moderado
            } else {
                200   // 0.2 segundos com pouco tempo
            };

            // Garante que max_time não seja zero se time_left for zero, para evitar divisão por zero ou tempo 0
            let max_time_divisor = if complexity.is_critical {
                3
            } else if complexity.is_tactical {
                4
            } else {
                5
            };
            let max_time = time_left / max_time_divisor.max(1); // Garante divisor mínimo de 1

            // Garante tempo mínimo absoluto e respeita o tempo máximo
            let final_time = time_with_increment.max(min_time).min(max_time);

            println!("info string Time allocation: base={}, inc_bonus={}, final={}, complexity={:?}",
                     base_time, increment_bonus, final_time, complexity);

            final_time
        } else {
            // Se não há tempo definido (ex: no `go` não veio `wtime`/`btime`), usa um padrão razoável
            10000 // 10 segundos padrão
        }
    }

    /// Analisa a complexidade da posição para determinar gestão de tempo
    fn analyze_position_complexity(&self, board: &Board) -> PositionComplexity {
        let mut complexity = PositionComplexity::default();

        // 1. Xeque é sempre crítico
        complexity.in_check = board.is_king_in_check(board.to_move);
        if complexity.in_check {
            complexity.is_critical = true;
            complexity.is_tactical = true;
            return complexity; // Retorna imediatamente
        }

        // 2. Análise rápida de peças penduradas
        let our_pieces = if board.to_move == motor_xadrez::types::Color::White {
            board.white_pieces
        } else {
            board.black_pieces
        };

        // Verifica apenas peças valiosas (não peões)
        let our_valuables = (board.knights | board.bishops | board.rooks | board.queens) & our_pieces;
        let mut hanging_count = 0;
        let mut attacked_count = 0;

        // Amostragem: verifica apenas as primeiras peças encontradas
        let mut bb = our_valuables;
        let mut checked = 0;
        while bb != 0 && checked < 4 { // Limita verificação para ser rápido
            let sq = bb.trailing_zeros() as u8;
            bb &= bb - 1;
            checked += 1;

            if board.is_square_attacked_by(sq, !board.to_move) {
                attacked_count += 1;
                if !board.is_square_attacked_by(sq, board.to_move) {
                    hanging_count += 1;
                }
            }
        }

        complexity.has_hanging_pieces = hanging_count > 0;
        complexity.attacked_pieces_count = attacked_count;

        // 3. Densidade de peças
        let total_pieces = (board.white_pieces | board.black_pieces).count_ones();
        complexity.piece_density = total_pieces as f32 / 64.0;

        // 4. Determina táticas
        complexity.is_tactical = hanging_count > 0 ||
            attacked_count >= 2 ||
            total_pieces <= 16;

        // 5. Posições críticas
        complexity.is_critical = hanging_count >= 2 ||
            (hanging_count > 0 && attacked_count > 1) ||
            (total_pieces <= 10 && complexity.is_tactical);

        complexity
    }

    /// Calcula distância entre duas casas do tabuleiro (não usada no código fornecido, mas mantida)
    fn square_distance(&self, sq1: u8, sq2: u8) -> u8 {
        let file1 = sq1 % 8;
        let rank1 = sq1 / 8;
        let file2 = sq2 % 8;
        let rank2 = sq2 / 8;

        let file_diff = (file1 as i8 - file2 as i8).abs() as u8;
        let rank_diff = (rank1 as i8 - rank2 as i8).abs() as u8;

        file_diff.max(rank_diff)
    }

    fn get_max_depth(&self) -> u8 {
        self.depth.unwrap_or(50) // Profundidade máxima de 50
    }

    /// Calcula profundidade máxima adaptativa baseada na complexidade
    fn get_adaptive_depth(&self, board: &Board) -> u8 {
        if let Some(fixed_depth) = self.depth {
            return fixed_depth;
        }

        let tactical_factors = self.analyze_position_complexity(board);
        let mut max_depth = 40u8; // Profundidade base mais realista

        // Ajusta profundidade baseada na complexidade tática
        if tactical_factors.is_critical {
            max_depth = 45;
        } else if tactical_factors.is_tactical {
            max_depth = 42;
        }

        // Em finais simples, limita a profundidade para evitar perder tempo
        if tactical_factors.piece_density < 0.25 && !tactical_factors.is_tactical {
            max_depth = 35;
        }

        max_depth
    }
}

fn main() {
    // Inicializa as dependências do motor
    motor_xadrez::evaluation::pawn_structure::init_pawn_masks();
    let mut board = Board::new();
    let mut tt = TranspositionTable::new(128); // 128 MB (corrigido de 16 para 128 para ser mais comum)
    let opening_book = OpeningBook::new(); // Carrega livro de aberturas
    let mut use_opening_book = true; // Configurável via UCI

    // O número de lances jogados deve ser persistido entre os comandos 'go'
    // e atualizado por 'position' e 'make_move'.
    let mut moves_played = 0u16;

    // Loop principal que espera por comandos da GUI
    loop {
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let commands: Vec<&str> = input.trim().split_whitespace().collect();

        if let Some(&command) = commands.get(0) {
            match command {
                "uci" => {
                    println!("id name MotorXadrez 2.0");
                    println!("id author Pedro Contessoto");

                    // Opções UCI configuráveis
                    println!("option name Hash type spin default 128 min 16 max 2048");
                    println!("option name Threads type spin default 1 min 1 max 1");
                    println!("option name Ponder type check default false");
                    println!("option name MultiPV type spin default 1 min 1 max 5");
                    println!("option name OwnBook type check default true");

                    println!("uciok");
                    io::stdout().flush().ok(); // Garante que a saída é enviada imediatamente
                }
                "isready" => {
                    println!("readyok");
                    io::stdout().flush().ok();
                }
                "position" => {
                    // Atualiza moves_played com base no comando position
                    moves_played = handle_position_command(&mut board, &commands);
                }
                "go" => {
                    handle_go_command(&board, &mut tt, &opening_book, use_opening_book, &commands, moves_played);
                }
                "setoption" => {
                    handle_setoption_command(&commands, &mut use_opening_book);
                }
                "stop" => {
                    // Para a busca atual (para implementar futuramente com threading)
                    // Se não houver threading, este comando pode não ter efeito imediato
                    // e a engine continuará a busca até o fim do tempo alocado ou profundidade.
                    // Em um motor real, 'stop' sinalizaria para a thread de busca parar.
                    // Por enquanto, não há uma ação direta aqui para parar a busca em andamento.
                    // A linha `println!("bestmove (none)");` aqui é um placeholder e pode ser removida
                    // ou usada apenas se o motor realmente não tiver um lance para retornar.
                    // O motor deve enviar bestmove *após* parar a busca, não no comando stop.
                }
                "quit" => {
                    break; // Sai do loop e termina o programa
                }
                _ => {
                    // Ignora comandos desconhecidos
                }
            }
        }
    }
}

/// Processa o comando "position" e retorna o número de lances jogados
fn handle_position_command(board: &mut Board, commands: &[&str]) -> u16 {
    let mut move_start_index = 0;
    let mut moves_count = 0u16; // Reinicia a contagem de lances para a nova posição

    if commands.get(1) == Some(&"startpos") {
        *board = Board::new();
        move_start_index = 2;
    } else if commands.get(1) == Some(&"fen") {
        // Encontra o início da string FEN
        let fen_parts: Vec<&str> = commands.iter().skip(2).take_while(|&&c| c != "moves").cloned().collect();
        let fen = fen_parts.join(" ");
        if let Ok(new_board) = Board::from_fen(&fen) {
            *board = new_board;
            // Extrai o número de lances do FEN se possível
            // O FEN tem o número de lances completos na 6ª parte (índice 5)
            if let Some(fullmove_str) = fen_parts.get(5) { // Corrigido para pegar o fullmove do FEN
                if let Ok(fullmove) = fullmove_str.parse::<u16>() {
                    // O fullmove no FEN é o número do lance completo.
                    // moves_played é o número de meios-lances.
                    moves_count = (fullmove - 1) * 2;
                    if board.to_move == motor_xadrez::types::Color::Black {
                        moves_count += 1;
                    }
                }
            }
        }
        move_start_index = 2 + fen_parts.len();
    }

    // Se houver lances após a posição, aplica-os
    if commands.get(move_start_index) == Some(&"moves") {
        for move_str in commands.iter().skip(move_start_index + 1) {
            if let Some(mv) = parse_move(board, move_str) {
                board.make_move(mv);
                moves_count += 1; // Incrementa para cada lance aplicado
            } else {
                eprintln!("Erro: Não foi possível parsear o lance: {}", move_str);
                // Pode ser útil adicionar um tratamento de erro mais robusto aqui
            }
        }
    }

    moves_count
}

/// Processa o comando "go" com gestão inteligente de tempo e livro de aberturas
fn handle_go_command(board: &Board, tt: &mut TranspositionTable, opening_book: &OpeningBook, use_book: bool, commands: &[&str], moves_played: u16) {
    let mut time_manager = TimeManager::new();

    // Processa os parâmetros do comando go
    let mut i = 1;
    while i < commands.len() {
        match commands[i] {
            "wtime" => {
                if i + 1 < commands.len() {
                    time_manager.wtime = commands[i + 1].parse().ok();
                    i += 1;
                }
            },
            "btime" => {
                if i + 1 < commands.len() {
                    time_manager.btime = commands[i + 1].parse().ok();
                    i += 1;
                }
            },
            "winc" => {
                if i + 1 < commands.len() {
                    time_manager.winc = commands[i + 1].parse().ok();
                    i += 1;
                }
            },
            "binc" => {
                if i + 1 < commands.len() {
                    time_manager.binc = commands[i + 1].parse().ok();
                    i += 1;
                }
            },
            "movetime" => {
                if i + 1 < commands.len() {
                    time_manager.movetime = commands[i + 1].parse().ok();
                    i += 1;
                }
            },
            "depth" => {
                if i + 1 < commands.len() {
                    time_manager.depth = commands[i + 1].parse().ok();
                    i += 1;
                }
            },
            "nodes" => {
                if i + 1 < commands.len() {
                    time_manager.nodes = commands[i + 1].parse().ok();
                    i += 1;
                }
            },
            "infinite" => {
                time_manager.infinite = true;
            },
            _ => {} // Ignora outros tokens que não são parâmetros de tempo/profundidade
        }
        i += 1;
    }

    // 📚 VERIFICA PRIMEIRO O LIVRO DE ABERTURAS
    if use_book && is_in_opening_phase(board) {
        if let Some((book_move, opening_name)) = opening_book.get_move(board) {
            // Usa movimento do livro de aberturas
            println!("info string Usando livro: {}", opening_name);
            println!("bestmove {}", book_move);
            io::stdout().flush().ok(); // Garante que a saída é enviada imediatamente
            return; // Retorna após encontrar um lance no livro
        } else {
            println!("info string Saindo do livro de aberturas - calculando...");
        }
    }

    // Se não estiver no livro, calcula normalmente
    let time_for_move = time_manager.calculate_time_for_move(board, moves_played);
    let max_depth = time_manager.get_adaptive_depth(board);

    println!("info string Allocated time: {}ms, max depth: {}", time_for_move, max_depth);

    // Busca o melhor movimento
    if let Some((best_move, final_score)) = search::find_best_move_with_time(board, max_depth, time_for_move, tt) {
        println!("bestmove {}", best_move);
        io::stdout().flush().ok();
    } else {
        // Fallback: Se não encontrou um best_move, tenta gerar um lance legal ou reporta (none)
        let legal_moves = board.generate_legal_moves();
        if let Some(fallback_move) = legal_moves.first() {
            println!("bestmove {}", fallback_move);
        } else {
            // Se não há lances legais, é xeque-mate ou stalemate
            println!("bestmove (none)");
        }
        io::stdout().flush().ok();
    }
}

/// Processa o comando "setoption"
fn handle_setoption_command(commands: &[&str], use_opening_book: &mut bool) {
    if commands.len() >= 5 && commands[1] == "name" && commands[3] == "value" {
        let option_name = commands[2];
        let option_value = commands[4];

        match option_name {
            "Hash" => {
                if let Ok(size) = option_value.parse::<usize>() {
                    println!("info string Hash table size set to {} MB", size);
                    // TODO: Redimensionar a tabela de transposição (necessita passar tt como mut)
                }
            },
            "Threads" => {
                if let Ok(threads) = option_value.parse::<usize>() {
                    println!("info string Thread count set to {}", threads);
                    // TODO: Implementar busca multi-threaded
                }
            },
            "Ponder" => {
                let ponder_enabled = option_value == "true";
                println!("info string Pondering {}", if ponder_enabled { "enabled" } else { "disabled" });
            },
            "OwnBook" => {
                *use_opening_book = option_value == "true";
                println!("info string Opening book {}", if *use_opening_book { "enabled" } else { "disabled" });
            },
            _ => {
                println!("info string Unknown option: {}", option_name);
            }
        }
    }
    io::stdout().flush().ok(); // Garante que a saída é enviada imediatamente
}

/// Função auxiliar para converter uma string (ex: "e2e4") num objeto Move
fn parse_move(board: &Board, move_str: &str) -> Option<Move> {
    // É mais eficiente gerar lances legais uma vez e depois procurar.
    // No entanto, se o `board.generate_legal_moves()` for muito lento,
    // pode-se otimizar esta função para tentar parsear diretamente e validar depois.
    let legal_moves = board.generate_legal_moves();
    for mv in legal_moves {
        let mv_str = mv.to_string();
        // Tenta match exato primeiro
        if mv_str == *move_str {
            return Some(mv);
        }
        // Tenta sem a notação de captura 'x' ou xeque/xeque-mate
        let clean_move_str = move_str.replace("x", "").replace("+", "").replace("#", "");
        if mv_str == clean_move_str {
            return Some(mv);
        }
    }
    None
}
