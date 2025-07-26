// Sistema avançado de X-ray attacks para mobilidade tática
use crate::{board::Board, types::{Color, Bitboard, PieceKind}};
use super::{MobilityContext, MobilityResult};

// ============================================================================
// X-RAY ATTACK SYSTEM - AVALIAÇÃO AVANÇADA DE MOBILIDADE
// ============================================================================

/// Configuração para diferentes tipos de X-ray
#[derive(Debug, Clone)]
pub struct XRayConfig {
    pub evaluate_pinned_pieces: bool,      // Avalia peças pregadas
    pub evaluate_discovered_attacks: bool, // Avalia ataques descobertos
    pub evaluate_battery_potential: bool,  // Avalia potencial de bateria
    pub evaluate_removal_tactics: bool,    // Avalia táticas de remoção
    pub xray_mobility_bonus: i32,          // Bônus por mobilidade X-ray
    pub pinned_piece_penalty: i32,         // Penalidade por estar pregado
    pub pin_creation_bonus: i32,           // Bônus por criar prego
    pub battery_bonus: i32,                // Bônus por formação de bateria
}

impl Default for XRayConfig {
    fn default() -> Self {
        XRayConfig {
            evaluate_pinned_pieces: true,
            evaluate_discovered_attacks: true,
            evaluate_battery_potential: true,
            evaluate_removal_tactics: true,
            xray_mobility_bonus: 3,
            pinned_piece_penalty: 15,
            pin_creation_bonus: 20,
            battery_bonus: 25,
        }
    }
}

/// Resultado da análise de X-ray para uma peça
#[derive(Debug, Clone)]
pub struct XRayAnalysis {
    pub direct_attacks: Bitboard,          // Ataques diretos normais
    pub xray_attacks: Bitboard,            // Ataques X-ray (através de peças)
    pub pinned_pieces: Vec<u8>,            // Peças inimigas pregadas por esta peça
    pub discovered_attacks: Vec<u8>,       // Ataques descobertos possíveis
    pub battery_partners: Vec<u8>,         // Peças que podem formar bateria
    pub removal_targets: Vec<u8>,          // Peças cuja remoção criaria ataques
    pub tactical_value: i32,               // Valor tático total
    pub positional_value: i32,             // Valor posicional
}

impl XRayAnalysis {
    pub fn new() -> Self {
        XRayAnalysis {
            direct_attacks: 0,
            xray_attacks: 0,
            pinned_pieces: Vec::new(),
            discovered_attacks: Vec::new(),
            battery_partners: Vec::new(),
            removal_targets: Vec::new(),
            tactical_value: 0,
            positional_value: 0,
        }
    }

    pub fn total_attacks(&self) -> Bitboard {
        self.direct_attacks | self.xray_attacks
    }

    pub fn total_value(&self) -> i32 {
        self.tactical_value + self.positional_value
    }
}

/// Tipos de X-ray táticos
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum XRayType {
    Pin,                    // Prego absoluto
    Skewer,                 // Espeto
    DiscoveredAttack,       // Ataque descoberto
    Battery,                // Bateria (duas peças na mesma linha)
    XRayDefense,            // Defesa X-ray
    RemovalTactic,          // Tática de remoção
}

/// Análise completa de X-ray para mobilidade avançada
pub fn evaluate_xray_mobility(
    piece_sq: u8, 
    piece_type: PieceKind,
    context: &MobilityContext,
    config: &XRayConfig
) -> XRayAnalysis {
    let mut analysis = XRayAnalysis::new();
    
    // Calcula ataques diretos primeiro
    analysis.direct_attacks = calculate_direct_attacks(piece_sq, piece_type, context);
    
    // Calcula ataques X-ray baseado no tipo da peça
    match piece_type {
        PieceKind::Bishop | PieceKind::Rook | PieceKind::Queen => {
            analysis.xray_attacks = calculate_sliding_xray_attacks(piece_sq, piece_type, context);
            
            if config.evaluate_pinned_pieces {
                analysis.pinned_pieces = find_pinned_pieces(piece_sq, piece_type, context);
            }
            
            if config.evaluate_discovered_attacks {
                analysis.discovered_attacks = find_discovered_attacks(piece_sq, piece_type, context);
            }
            
            if config.evaluate_battery_potential {
                analysis.battery_partners = find_battery_partners(piece_sq, piece_type, context);
            }
            
            if config.evaluate_removal_tactics {
                analysis.removal_targets = find_removal_tactics(piece_sq, piece_type, context);
            }
        },
        PieceKind::Knight => {
            // Cavalos não fazem X-ray, mas podem participar de ataques descobertos
            if config.evaluate_discovered_attacks {
                analysis.discovered_attacks = find_knight_discovered_attacks(piece_sq, context);
            }
        },
        _ => {
            // Peões e reis têm análise X-ray limitada
        }
    }
    
    // Calcula valores táticos e posicionais
    analysis.tactical_value = calculate_xray_tactical_value(&analysis, config);
    analysis.positional_value = calculate_xray_positional_value(&analysis, context, config);
    
    analysis
}

/// Calcula ataques diretos normais
fn calculate_direct_attacks(piece_sq: u8, piece_type: PieceKind, context: &MobilityContext) -> Bitboard {
    match piece_type {
        PieceKind::Pawn => {
            super::compute_pawn_attacks(1u64 << piece_sq, context.color)
        },
        PieceKind::Knight => {
            crate::moves::knight::get_knight_attacks_lookup(piece_sq)
        },
        PieceKind::Bishop => {
            crate::moves::sliding::get_bishop_attacks(piece_sq, context.all_pieces)
        },
        PieceKind::Rook => {
            crate::moves::sliding::get_rook_attacks(piece_sq, context.all_pieces)
        },
        PieceKind::Queen => {
            let bishop_attacks = crate::moves::sliding::get_bishop_attacks(piece_sq, context.all_pieces);
            let rook_attacks = crate::moves::sliding::get_rook_attacks(piece_sq, context.all_pieces);
            bishop_attacks | rook_attacks
        },
        PieceKind::King => {
            crate::moves::king::get_king_attacks_lookup(piece_sq)
        },
    }
}

/// Calcula ataques X-ray para peças deslizantes
fn calculate_sliding_xray_attacks(piece_sq: u8, piece_type: PieceKind, context: &MobilityContext) -> Bitboard {
    let mut xray_attacks = 0u64;
    
    // Para cada direção possível, calcula o X-ray
    let directions = get_piece_directions(piece_type);
    
    for direction in directions {
        let xray = calculate_xray_in_direction(piece_sq, direction, context);
        xray_attacks |= xray;
    }
    
    xray_attacks
}

/// Obtém direções de movimento para um tipo de peça
fn get_piece_directions(piece_type: PieceKind) -> Vec<i8> {
    match piece_type {
        PieceKind::Bishop => vec![-9, -7, 7, 9], // Diagonais
        PieceKind::Rook => vec![-8, -1, 1, 8],   // Horizontais/verticais
        PieceKind::Queen => vec![-9, -8, -7, -1, 1, 7, 8, 9], // Todas as direções
        _ => vec![],
    }
}

/// Calcula X-ray em uma direção específica
fn calculate_xray_in_direction(piece_sq: u8, direction: i8, context: &MobilityContext) -> Bitboard {
    let mut xray_attacks = 0u64;
    let mut current_sq = piece_sq as i8;
    let mut pieces_encountered = 0;
    let max_distance = 8; // Máximo em um tabuleiro 8x8
    
    for _ in 0..max_distance {
        current_sq += direction;
        
        // Verifica limites do tabuleiro
        if current_sq < 0 || current_sq >= 64 {
            break;
        }
        
        // Verifica wrapping nas bordas
        if !is_valid_move(piece_sq, current_sq as u8, direction) {
            break;
        }
        
        let current_bit = 1u64 << current_sq;
        
        if (context.all_pieces & current_bit) != 0 {
            pieces_encountered += 1;
            
            if pieces_encountered == 1 {
                // Primeira peça encontrada - continua para ver o que está atrás
                continue;
            } else if pieces_encountered == 2 {
                // Segunda peça - esta é um alvo X-ray potencial
                if (context.enemy_pieces & current_bit) != 0 {
                    // É uma peça inimiga - X-ray attack válido
                    xray_attacks |= current_bit;
                }
                break; // Para aqui - não podemos ver além da segunda peça
            }
        } else if pieces_encountered == 1 {
            // Casa vazia atrás da primeira peça - X-ray mobility
            xray_attacks |= current_bit;
        }
    }
    
    xray_attacks
}

/// Verifica se movimento é válido (não wrapping)
fn is_valid_move(from: u8, to: u8, direction: i8) -> bool {
    let from_file = from % 8;
    let from_rank = from / 8;
    let to_file = to % 8;
    let to_rank = to / 8;
    
    // Verifica wrapping horizontal
    match direction {
        -1 | 7 | -9 => from_file > 0,  // Movendo para esquerda
        1 | 9 | -7 => from_file < 7,   // Movendo para direita
        _ => true,
    }
}

/// Encontra peças inimigas pregadas por esta peça
fn find_pinned_pieces(piece_sq: u8, piece_type: PieceKind, context: &MobilityContext) -> Vec<u8> {
    let mut pinned_pieces = Vec::new();
    let directions = get_piece_directions(piece_type);
    
    // Localiza o rei inimigo
    let enemy_king = context.board.kings & context.enemy_pieces;
    if enemy_king == 0 { return pinned_pieces; }
    let enemy_king_sq = enemy_king.trailing_zeros() as u8;
    
    for direction in directions {
        let pinned = find_pin_in_direction(piece_sq, direction, enemy_king_sq, context);
        if let Some(pinned_sq) = pinned {
            pinned_pieces.push(pinned_sq);
        }
    }
    
    pinned_pieces
}

/// Encontra prego em uma direção específica
fn find_pin_in_direction(piece_sq: u8, direction: i8, enemy_king_sq: u8, context: &MobilityContext) -> Option<u8> {
    let mut current_sq = piece_sq as i8;
    let mut potential_pinned = None;
    let max_distance = 8;
    
    for _ in 0..max_distance {
        current_sq += direction;
        
        if current_sq < 0 || current_sq >= 64 {
            break;
        }
        
        if !is_valid_move(piece_sq, current_sq as u8, direction) {
            break;
        }
        
        let current_bit = 1u64 << current_sq;
        
        if current_sq as u8 == enemy_king_sq {
            // Encontrou o rei inimigo - se há uma peça no meio, está pregada
            return potential_pinned;
        }
        
        if (context.all_pieces & current_bit) != 0 {
            if potential_pinned.is_some() {
                // Já temos uma peça no meio - não é prego válido
                break;
            }
            
            if (context.enemy_pieces & current_bit) != 0 {
                // Peça inimiga - potencial prego
                potential_pinned = Some(current_sq as u8);
            } else {
                // Nossa peça - bloqueia o prego
                break;
            }
        }
    }
    
    None
}

/// Encontra ataques descobertos possíveis
fn find_discovered_attacks(piece_sq: u8, piece_type: PieceKind, context: &MobilityContext) -> Vec<u8> {
    let mut discovered_attacks = Vec::new();
    
    // Procura por peças nossas que podem criar ataques descobertos movendo esta peça
    let our_sliding_pieces = get_our_sliding_pieces(context);
    
    for ally_sq in our_sliding_pieces {
        if ally_sq == piece_sq { continue; }
        
        if can_create_discovered_attack(piece_sq, ally_sq, context) {
            discovered_attacks.push(ally_sq);
        }
    }
    
    discovered_attacks
}

/// Encontra ataques descobertos para cavalos
fn find_knight_discovered_attacks(piece_sq: u8, context: &MobilityContext) -> Vec<u8> {
    let mut discovered_attacks = Vec::new();
    
    // Cavalos podem se mover para revelar ataques de peças deslizantes atrás deles
    let our_sliding_pieces = get_our_sliding_pieces(context);
    
    for ally_sq in our_sliding_pieces {
        if can_create_discovered_attack(piece_sq, ally_sq, context) {
            discovered_attacks.push(ally_sq);
        }
    }
    
    discovered_attacks
}

/// Obtém nossas peças deslizantes
fn get_our_sliding_pieces(context: &MobilityContext) -> Vec<u8> {
    let mut pieces = Vec::new();
    let our_sliding = (context.board.bishops | context.board.rooks | context.board.queens) & context.our_pieces;
    
    let mut bb = our_sliding;
    while bb != 0 {
        let sq = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        pieces.push(sq);
    }
    
    pieces
}

/// Verifica se pode criar ataque descoberto
fn can_create_discovered_attack(moving_piece_sq: u8, attacking_piece_sq: u8, context: &MobilityContext) -> bool {
    // Verifica se a peça que move está na linha de ataque da peça atacante
    let enemy_king = context.board.kings & context.enemy_pieces;
    if enemy_king == 0 { return false; }
    let enemy_king_sq = enemy_king.trailing_zeros() as u8;
    
    // Determina tipo da peça atacante
    let attacking_bit = 1u64 << attacking_piece_sq;
    let piece_type = if (context.board.bishops & attacking_bit) != 0 {
        PieceKind::Bishop
    } else if (context.board.rooks & attacking_bit) != 0 {
        PieceKind::Rook
    } else if (context.board.queens & attacking_bit) != 0 {
        PieceKind::Queen
    } else {
        return false;
    };
    
    let directions = get_piece_directions(piece_type);
    
    for direction in directions {
        if is_piece_in_attack_line(attacking_piece_sq, moving_piece_sq, enemy_king_sq, direction) {
            return true;
        }
    }
    
    false
}

/// Verifica se peça está na linha de ataque
fn is_piece_in_attack_line(attacker_sq: u8, blocker_sq: u8, target_sq: u8, direction: i8) -> bool {
    let mut current_sq = attacker_sq as i8;
    
    loop {
        current_sq += direction;
        
        if current_sq < 0 || current_sq >= 64 {
            break;
        }
        
        if !is_valid_move(attacker_sq, current_sq as u8, direction) {
            break;
        }
        
        if current_sq as u8 == blocker_sq {
            // Encontrou a peça que bloqueia - continua para ver se o rei está atrás
            continue;
        }
        
        if current_sq as u8 == target_sq {
            // Encontrou o alvo - ataque descoberto possível
            return true;
        }
    }
    
    false
}

/// Encontra parceiros para formação de bateria
fn find_battery_partners(piece_sq: u8, piece_type: PieceKind, context: &MobilityContext) -> Vec<u8> {
    let mut partners = Vec::new();
    let directions = get_piece_directions(piece_type);
    
    for direction in directions {
        if let Some(partner) = find_battery_partner_in_direction(piece_sq, direction, piece_type, context) {
            partners.push(partner);
        }
    }
    
    partners
}

/// Encontra parceiro de bateria em uma direção
fn find_battery_partner_in_direction(piece_sq: u8, direction: i8, piece_type: PieceKind, context: &MobilityContext) -> Option<u8> {
    let mut current_sq = piece_sq as i8;
    let max_distance = 8;
    
    for _ in 0..max_distance {
        current_sq += direction;
        
        if current_sq < 0 || current_sq >= 64 {
            break;
        }
        
        if !is_valid_move(piece_sq, current_sq as u8, direction) {
            break;
        }
        
        let current_bit = 1u64 << current_sq;
        
        if (context.our_pieces & current_bit) != 0 {
            // Encontrou nossa peça - verifica se pode formar bateria
            if can_form_battery(piece_type, current_sq as u8, context) {
                return Some(current_sq as u8);
            } else {
                break; // Peça incompatível bloqueia
            }
        } else if (context.enemy_pieces & current_bit) != 0 {
            break; // Peça inimiga bloqueia
        }
    }
    
    None
}

/// Verifica se pode formar bateria
fn can_form_battery(piece_type: PieceKind, partner_sq: u8, context: &MobilityContext) -> bool {
    let partner_bit = 1u64 << partner_sq;
    
    match piece_type {
        PieceKind::Bishop => {
            // Bispo pode formar bateria com outro bispo ou rainha
            (context.board.bishops & partner_bit) != 0 || (context.board.queens & partner_bit) != 0
        },
        PieceKind::Rook => {
            // Torre pode formar bateria com outra torre ou rainha
            (context.board.rooks & partner_bit) != 0 || (context.board.queens & partner_bit) != 0
        },
        PieceKind::Queen => {
            // Rainha pode formar bateria com qualquer peça deslizante
            (context.board.bishops & partner_bit) != 0 || 
            (context.board.rooks & partner_bit) != 0 || 
            (context.board.queens & partner_bit) != 0
        },
        _ => false,
    }
}

/// Encontra alvos para táticas de remoção
fn find_removal_tactics(piece_sq: u8, piece_type: PieceKind, context: &MobilityContext) -> Vec<u8> {
    let mut targets = Vec::new();
    let directions = get_piece_directions(piece_type);
    
    for direction in directions {
        let removal_targets = find_removal_targets_in_direction(piece_sq, direction, context);
        targets.extend(removal_targets);
    }
    
    targets
}

/// Encontra alvos de remoção em uma direção
fn find_removal_targets_in_direction(piece_sq: u8, direction: i8, context: &MobilityContext) -> Vec<u8> {
    let mut targets = Vec::new();
    let mut current_sq = piece_sq as i8;
    let max_distance = 8;
    
    for _ in 0..max_distance {
        current_sq += direction;
        
        if current_sq < 0 || current_sq >= 64 {
            break;
        }
        
        if !is_valid_move(piece_sq, current_sq as u8, direction) {
            break;
        }
        
        let current_bit = 1u64 << current_sq;
        
        if (context.enemy_pieces & current_bit) != 0 {
            // Peça inimiga - verifica se sua remoção criaria táticas
            if removal_creates_tactical_opportunity(current_sq as u8, piece_sq, direction, context) {
                targets.push(current_sq as u8);
            }
            break; // Para na primeira peça inimiga
        } else if (context.our_pieces & current_bit) != 0 {
            break; // Nossa peça bloqueia
        }
    }
    
    targets
}

/// Verifica se remoção de peça cria oportunidade tática
fn removal_creates_tactical_opportunity(target_sq: u8, attacker_sq: u8, direction: i8, context: &MobilityContext) -> bool {
    // Simula remoção da peça e verifica se isso cria ataques valiosos
    let mut temp_pieces = context.all_pieces;
    temp_pieces &= !(1u64 << target_sq); // Remove a peça
    
    // Verifica se agora podemos atacar algo valioso atrás da peça removida
    let mut current_sq = target_sq as i8;
    
    loop {
        current_sq += direction;
        
        if current_sq < 0 || current_sq >= 64 {
            break;
        }
        
        if !is_valid_move(attacker_sq, current_sq as u8, direction) {
            break;
        }
        
        let current_bit = 1u64 << current_sq;
        
        if (temp_pieces & current_bit) != 0 {
            if (context.enemy_pieces & current_bit) != 0 {
                // Encontrou peça inimiga valiosa atrás
                return is_valuable_target(current_sq as u8, context);
            } else {
                break; // Nossa peça
            }
        }
    }
    
    false
}

/// Verifica se alvo é valioso
fn is_valuable_target(target_sq: u8, context: &MobilityContext) -> bool {
    let target_bit = 1u64 << target_sq;
    
    // Rei é sempre valioso
    if (context.board.kings & target_bit) != 0 {
        return true;
    }
    
    // Rainha é muito valiosa
    if (context.board.queens & target_bit) != 0 {
        return true;
    }
    
    // Torres e peças menores têm valor moderado
    (context.board.rooks & target_bit) != 0 ||
    (context.board.bishops & target_bit) != 0 ||
    (context.board.knights & target_bit) != 0
}

/// Calcula valor tático dos X-rays
fn calculate_xray_tactical_value(analysis: &XRayAnalysis, config: &XRayConfig) -> i32 {
    let mut value = 0;
    
    // Bônus por mobilidade X-ray
    value += analysis.xray_attacks.count_ones() as i32 * config.xray_mobility_bonus;
    
    // Bônus por criar pregos
    value += analysis.pinned_pieces.len() as i32 * config.pin_creation_bonus;
    
    // Bônus por ataques descobertos
    value += analysis.discovered_attacks.len() as i32 * 15;
    
    // Bônus por formação de bateria
    value += analysis.battery_partners.len() as i32 * config.battery_bonus;
    
    // Bônus por táticas de remoção
    value += analysis.removal_targets.len() as i32 * 10;
    
    value
}

/// Calcula valor posicional dos X-rays
fn calculate_xray_positional_value(analysis: &XRayAnalysis, context: &MobilityContext, config: &XRayConfig) -> i32 {
    let mut value = 0;
    
    // Valor posicional baseado no controle de casas importantes
    let important_squares = get_important_squares(context);
    let controlled_important = analysis.total_attacks() & important_squares;
    value += controlled_important.count_ones() as i32 * 5;
    
    // Penalidade se nossa peça está pregada
    // (isso seria calculado externamente, mas mantemos a interface)
    
    value
}

/// Obtém casas importantes no tabuleiro
fn get_important_squares(context: &MobilityContext) -> Bitboard {
    let mut important = 0u64;
    
    // Centro e centro expandido
    important |= 0x0000001818000000u64; // d4, e4, d5, e5
    important |= 0x00003C3C3C3C0000u64; // Centro expandido
    
    // Casas próximas aos reis
    let our_king = context.board.kings & context.our_pieces;
    let enemy_king = context.board.kings & context.enemy_pieces;
    
    if our_king != 0 {
        let king_sq = our_king.trailing_zeros() as u8;
        important |= crate::moves::king::get_king_attacks_lookup(king_sq);
    }
    
    if enemy_king != 0 {
        let king_sq = enemy_king.trailing_zeros() as u8;
        important |= crate::moves::king::get_king_attacks_lookup(king_sq);
    }
    
    important
}

/// Interface principal para integração com o sistema de mobilidade
pub fn enhance_mobility_with_xray(
    piece_sq: u8,
    piece_type: PieceKind,
    base_mobility: MobilityResult,
    context: &MobilityContext
) -> MobilityResult {
    let config = XRayConfig::default();
    let xray_analysis = evaluate_xray_mobility(piece_sq, piece_type, context, &config);
    
    let mut enhanced_mobility = base_mobility;
    
    // Adiciona valor X-ray ao resultado base
    enhanced_mobility.tactical_threats += xray_analysis.tactical_value;
    enhanced_mobility.positional_bonus += xray_analysis.positional_value;
    
    // Adiciona mobilidade X-ray como "safe mobility" se as casas não estão atacadas
    let safe_xray_mobility = xray_analysis.xray_attacks & !context.enemy_attacked_squares;
    enhanced_mobility.safe_mobility += safe_xray_mobility.count_ones() as i32 * 2;
    
    enhanced_mobility
}