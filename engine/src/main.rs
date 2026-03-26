// HiveChess — UCI chess engine
// Ported from github.com/deedy/chess (Rust search engine)
// Added UCI protocol for standalone operation

use std::io::{self, BufRead, Write};

fn send(msg: &str) {
    println!("{}", msg);
    io::stdout().flush().ok();
}

fn main() {
    let stdin = io::stdin();
    let mut engine = RustAlphaBetaEngine::new(64);
    let mut root_fen = String::from("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    let mut move_history: Vec<String> = Vec::new();
    let mut current_board = Board::default();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.is_empty() { continue; }

        match tokens[0] {
            "uci" => {
                send("id name HiveChess 0.3");
                send("id author HiveAgent");
                send("uciok");
            }
            "isready" => send("readyok"),
            "ucinewgame" => {
                engine = RustAlphaBetaEngine::new(64);
            }
            "position" => {
                let mut idx = 1;
                if tokens.get(idx) == Some(&"startpos") {
                    current_board = Board::default();
                    root_fen = current_board.to_string();
                    idx = 2;
                } else if tokens.get(idx) == Some(&"fen") {
                    idx += 1;
                    let mut fen_parts = Vec::new();
                    while idx < tokens.len() && tokens[idx] != "moves" {
                        fen_parts.push(tokens[idx]);
                        idx += 1;
                    }
                    root_fen = fen_parts.join(" ");
                    current_board = Board::from_str(&root_fen).unwrap_or_default();
                }
                move_history.clear();
                if tokens.get(idx) == Some(&"moves") {
                    idx += 1;
                    let mut b = Board::from_str(&root_fen).unwrap_or_default();
                    while idx < tokens.len() {
                        move_history.push(tokens[idx].to_string());
                        if let Ok(mv) = ChessMove::from_str(tokens[idx]) {
                            b = b.make_move_new(mv);
                        }
                        idx += 1;
                    }
                    current_board = b;
                }
            }
            "go" => {
                let side = current_board.side_to_move();
                let mut time_ms: u64 = 10000;
                let mut inc_ms: u64 = 0;
                let mut movestogo: u32 = 0;
                let mut max_depth: Option<i32> = None;
                let mut movetime: Option<u64> = None;

                let mut i = 1;
                while i < tokens.len() {
                    match tokens[i] {
                        "wtime" => { if let Some(t) = tokens.get(i+1).and_then(|t| t.parse().ok()) { if side == Color::White { time_ms = t; } } i += 2; }
                        "btime" => { if let Some(t) = tokens.get(i+1).and_then(|t| t.parse().ok()) { if side == Color::Black { time_ms = t; } } i += 2; }
                        "winc"  => { if let Some(t) = tokens.get(i+1).and_then(|t| t.parse().ok()) { if side == Color::White { inc_ms = t; } } i += 2; }
                        "binc"  => { if let Some(t) = tokens.get(i+1).and_then(|t| t.parse().ok()) { if side == Color::Black { inc_ms = t; } } i += 2; }
                        "movestogo" => { movestogo = tokens.get(i+1).and_then(|t| t.parse().ok()).unwrap_or(0); i += 2; }
                        "depth" => { max_depth = tokens.get(i+1).and_then(|t| t.parse().ok()); i += 2; }
                        "movetime" => { movetime = tokens.get(i+1).and_then(|t| t.parse().ok()); i += 2; }
                        "infinite" => { time_ms = 999_999_999; i += 1; }
                        _ => { i += 1; }
                    }
                }

                let alloc_ms = if let Some(mt) = movetime {
                    mt
                } else if movestogo > 0 {
                    // Moves-to-go: allocate time/moves_remaining, keep safety margin
                    let base = time_ms / (movestogo as u64 + 2);
                    let with_inc = base + inc_ms * 3 / 4;
                    with_inc.min(time_ms * 2 / 5).max(50)
                } else {
                    // Sudden death: use 1/20 of remaining
                    let base = time_ms / 20;
                    let with_inc = base + inc_ms;
                    with_inc.min(time_ms / 3).max(100)
                };

                let history = if move_history.is_empty() { None } else { Some(move_history.clone()) };
                match engine.choose_move(&root_fen, max_depth, Some(alloc_ms), history) {
                    Ok(res) => {
                        let pv = res.pv_uci.join(" ");
                        send(&format!("info depth {} score cp {} nodes {} pv {}", res.depth, res.score, res.nodes, pv));
                        send(&format!("bestmove {}", res.move_uci));
                    }
                    Err(_) => {
                        if let Some(mv) = MoveGen::new_legal(&current_board).next() {
                            send(&format!("bestmove {}", mv));
                        } else {
                            send("bestmove 0000");
                        }
                    }
                }
            }
            "quit" => break,
            _ => {}
        }
    }
}

// =============================================================================
// Engine code below
// =============================================================================

use std::collections::HashMap;
use std::str::FromStr;
use std::thread;
use std::time::{Duration, Instant};

use chess::{
    get_bishop_moves, get_king_moves, get_knight_moves, get_pawn_attacks, get_rook_moves,
    BitBoard, Board, BoardStatus, ChessMove, Color, File, MoveGen, Piece, Rank, Square,
};

const INFINITY: i32 = 1_000_000;
const MATE_SCORE: i32 = 100_000;
const DRAW_SCORE: i32 = 0;
const ASPIRATION_WINDOW: i32 = 40;
const HISTORY_SIZE: usize = 1 << 16;
const KILLER_PLY_CAPACITY: usize = 128;
const MAX_ROOT_THREADS: usize = 8;

// Fixed-size hash tables (power-of-2 for fast masking)
const TT_SIZE: usize = 1 << 21; // 2M entries
const TT_MASK: usize = TT_SIZE - 1;
const EVAL_CACHE_SIZE: usize = 1 << 19; // 512K entries
const EVAL_CACHE_MASK: usize = EVAL_CACHE_SIZE - 1;
const PAWN_CACHE_SIZE: usize = 1 << 18; // 256K entries
const PAWN_CACHE_MASK: usize = PAWN_CACHE_SIZE - 1;

const HISTORY_LIMIT: i32 = 32_000;
const CAPTURE_HISTORY_PIECES: usize = 6;
const CAPTURE_HISTORY_LIMIT: i32 = 16_000;
const MAX_TIME_SEARCH_DEPTH: i32 = 64;
const PHASE_TOTAL: i32 = 24;
const SEE_PRUNE_MARGIN: i32 = 80;

const EXACT: u8 = 0;
const LOWER_BOUND: u8 = 1;
const UPPER_BOUND: u8 = 2;

const PAWN: i32 = 100;
const KNIGHT: i32 = 310;
const BISHOP: i32 = 333;
const ROOK: i32 = 550;
const QUEEN: i32 = 950;

const BISHOP_PAIR_BONUS: i32 = 45;
const CASTLED_KING_BONUS: i32 = 18;
const DOUBLED_PAWN_PENALTY: i32 = 14;
const ISOLATED_PAWN_PENALTY: i32 = 11;
const TEMPO_BONUS: i32 = 10;
const ROOK_OPEN_FILE_BONUS: i32 = 18;
const ROOK_SEMI_OPEN_FILE_BONUS: i32 = 10;
const ROOK_SEVENTH_RANK_BONUS: i32 = 14;
const KNIGHT_OUTPOST_BONUS: i32 = 18;
const CENTER_PAWN_BONUS: i32 = 10;
const UNDEVELOPED_MINOR_PENALTY: i32 = 8;

const MOBILITY_KNIGHT: i32 = 4;
const MOBILITY_BISHOP: i32 = 5;
const MOBILITY_ROOK: i32 = 2;
const MOBILITY_QUEEN: i32 = 1;

const CONNECTED_ROOKS_BONUS: i32 = 10;
const MINOR_BEHIND_PAWN_BONUS: i32 = 5;
const THREAT_MINOR_BY_PAWN: i32 = 25;
const THREAT_ROOK_BY_MINOR: i32 = 20;
const THREAT_QUEEN_BY_MINOR: i32 = 30;
const THREAT_QUEEN_BY_ROOK: i32 = 25;

const PASSED_PAWN_BONUS: [i32; 8] = [0, 8, 12, 20, 35, 60, 90, 0];
const ENDGAME_PASSED_PAWN_BONUS: [i32; 8] = [0, 0, 4, 8, 16, 32, 56, 0];
const SUPPORTED_PASSED_PAWN_BONUS: [i32; 8] = [0, 0, 3, 6, 12, 20, 32, 0];
const REVERSE_FUTILITY_MARGIN: [i32; 5] = [0, 75, 140, 225, 310];
const FUTILITY_MARGIN: [i32; 5] = [0, 90, 155, 245, 340];
const RAZOR_MARGIN: [i32; 4] = [0, 230, 360, 500];

// Contempt: slight penalty for draws when we likely have advantage
const CONTEMPT: i32 = 12;

/// Build the opening book: maps position hash -> best move UCI string.
/// These are strong opening moves from theory, covering common openings.
fn build_opening_book() -> HashMap<u64, &'static str> {
    let mut book = HashMap::new();
    // We build the book by replaying move sequences and recording hash -> next_move
    // Each line is a sequence of moves; we record each intermediate position hash -> next move
    let lines: &[&[&str]] = &[
        // Italian Game / Giuoco Piano
        &["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "c2c3", "g8f6", "d2d4"],
        // Ruy Lopez main line
        &["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1"],
        // Scotch Game
        &["e2e4", "e7e5", "g1f3", "b8c6", "d2d4", "e5d4", "f3d4"],
        // Queen's Gambit
        &["d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6", "c1g5"],
        // Queen's Gambit Declined
        &["d2d4", "d7d5", "c2c4", "c7c6", "g1f3", "g8f6", "b1c3"],
        // Sicilian Defense (Open)
        &["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4"],
        // Sicilian Najdorf
        &["e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6", "c1e3"],
        // French Defense
        &["e2e4", "e7e6", "d2d4", "d7d5", "b1c3", "g8f6", "c1g5"],
        // Caro-Kann
        &["e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4", "c3e4"],
        // King's Indian
        &["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "f8g7", "e2e4", "d7d6", "g1f3"],
        // Nimzo-Indian
        &["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "e2e3"],
        // English Opening
        &["c2c4", "e7e5", "b1c3", "g8f6", "g1f3"],
        // London System
        &["d2d4", "d7d5", "g1f3", "g8f6", "c1f4", "e7e6", "e2e3"],
        // Catalan
        &["d2d4", "g8f6", "c2c4", "e7e6", "g2g3", "d7d5", "f1g2"],
        // Pirc Defense
        &["e2e4", "d7d6", "d2d4", "g8f6", "b1c3", "g7g6", "g1f3"],
        // Scandinavian
        &["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5", "d2d4"],
        // Slav Defense
        &["d2d4", "d7d5", "c2c4", "c7c6", "g1f3", "g8f6", "b1c3", "d5c4", "a2a4"],
        // Grunfeld
        &["d2d4", "g8f6", "c2c4", "g7g6", "b1c3", "d7d5", "c4d5", "f6d5", "e2e4"],
        // Default opening moves
        &["e2e4"],
        &["e2e4", "e7e5", "g1f3"],
        &["d2d4", "d7d5", "c2c4"],
        &["d2d4", "g8f6", "c2c4"],
    ];

    for line in lines {
        let mut board = Board::default();
        for (i, move_uci) in line.iter().enumerate() {
            if let Ok(mv) = ChessMove::from_str(move_uci) {
                if move_is_legal(&board, mv) {
                    // Record this position -> this move (only if not already recorded)
                    let hash = board.get_hash();
                    book.entry(hash).or_insert(line[i]);
                    board = board.make_move_new(mv);
                } else {
                    break; // Illegal move in sequence, stop this line
                }
            } else {
                break;
            }
        }
    }
    book
}

const PAWN_TABLE: [i32; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0, 5, 10, 10, -20, -20, 10, 10, 5, 5, -5, -10, 0, 0, -10, -5, 5, 0, 0, 0,
    20, 20, 0, 0, 0, 5, 5, 10, 25, 25, 10, 5, 5, 10, 10, 20, 30, 30, 20, 10, 10, 50, 50, 50, 50,
    50, 50, 50, 50, 0, 0, 0, 0, 0, 0, 0, 0,
];

const KNIGHT_TABLE: [i32; 64] = [
    -50, -40, -30, -30, -30, -30, -40, -50, -40, -20, 0, 0, 0, 0, -20, -40, -30, 0, 10, 15, 15, 10,
    0, -30, -30, 5, 15, 20, 20, 15, 5, -30, -30, 0, 15, 20, 20, 15, 0, -30, -30, 5, 10, 15, 15, 10,
    5, -30, -40, -20, 0, 5, 5, 0, -20, -40, -50, -40, -30, -30, -30, -30, -40, -50,
];

const BISHOP_TABLE: [i32; 64] = [
    -20, -10, -10, -10, -10, -10, -10, -20, -10, 0, 0, 0, 0, 0, 0, -10, -10, 0, 5, 10, 10, 5, 0,
    -10, -10, 5, 5, 10, 10, 5, 5, -10, -10, 0, 10, 10, 10, 10, 0, -10, -10, 10, 10, 10, 10, 10, 10,
    -10, -10, 5, 0, 0, 0, 0, 5, -10, -20, -10, -10, -10, -10, -10, -10, -20,
];

const ROOK_TABLE: [i32; 64] = [
    0, 0, 0, 5, 5, 0, 0, 0, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0,
    0, -5, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 0, 0, 0, 0, 0, -5, 5, 10, 10, 10, 10, 10, 10, 5, 0, 0,
    0, 0, 0, 0, 0, 0,
];

const QUEEN_TABLE: [i32; 64] = [
    -20, -10, -10, -5, -5, -10, -10, -20, -10, 0, 0, 0, 0, 0, 0, -10, -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5, 0, 0, 5, 5, 5, 5, 0, -5, -10, 5, 5, 5, 5, 5, 0, -10, -10, 0, 5, 0, 0,
    0, 0, -10, -20, -10, -10, -5, -5, -10, -10, -20,
];

const KING_MIDGAME_TABLE: [i32; 64] = [
    -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -30, -40, -40,
    -50, -50, -40, -40, -30, -30, -40, -40, -50, -50, -40, -40, -30, -20, -30, -30, -40, -40, -30,
    -30, -20, -10, -20, -20, -20, -20, -20, -20, -10, 20, 20, 0, 0, 0, 0, 20, 20, 20, 30, 10, 0, 0,
    10, 30, 20,
];

const KING_ENDGAME_TABLE: [i32; 64] = [
    -50, -40, -30, -20, -20, -30, -40, -50, -30, -20, -10, 0, 0, -10, -20, -30, -30, -10, 20, 30,
    30, 20, -10, -30, -30, -10, 30, 40, 40, 30, -10, -30, -30, -10, 30, 40, 40, 30, -10, -30, -30,
    -10, 20, 30, 30, 20, -10, -30, -30, -30, 0, 0, 0, 0, -30, -30, -50, -30, -30, -30, -30, -30,
    -30, -50,
];

// Endgame piece-square tables (for proper tapered eval)
const PAWN_EG_TABLE: [i32; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    20, 20, 20, 20, 20, 20, 20, 20, 30, 30, 30, 30, 30, 30, 30, 30, 50, 50, 50, 50, 50, 50, 50, 50,
    80, 80, 80, 80, 80, 80, 80, 80, 0, 0, 0, 0, 0, 0, 0, 0,
];

const KNIGHT_EG_TABLE: [i32; 64] = [
    -30, -20, -10, -10, -10, -10, -20, -30, -20, -10, 0, 5, 5, 0, -10, -20, -10, 0, 10, 15, 15, 10,
    0, -10, -10, 5, 15, 20, 20, 15, 5, -10, -10, 5, 15, 20, 20, 15, 5, -10, -10, 0, 10, 15, 15, 10,
    0, -10, -20, -10, 0, 5, 5, 0, -10, -20, -30, -20, -10, -10, -10, -10, -20, -30,
];

const BISHOP_EG_TABLE: [i32; 64] = [
    -15, -10, -10, -10, -10, -10, -10, -15, -10, 0, 0, 0, 0, 0, 0, -10, -10, 0, 5, 10, 10, 5, 0,
    -10, -10, 0, 10, 15, 15, 10, 0, -10, -10, 0, 10, 15, 15, 10, 0, -10, -10, 0, 5, 10, 10, 5, 0,
    -10, -10, 0, 0, 0, 0, 0, 0, -10, -15, -10, -10, -10, -10, -10, -10, -15,
];

const ROOK_EG_TABLE: [i32; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0,
];

const QUEEN_EG_TABLE: [i32; 64] = [
    -10, -5, -5, -5, -5, -5, -5, -10, -5, 0, 0, 0, 0, 0, 0, -5, -5, 0, 5, 5, 5, 5, 0, -5,
    -5, 0, 5, 10, 10, 5, 0, -5, -5, 0, 5, 10, 10, 5, 0, -5, -5, 0, 5, 5, 5, 5, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5, -10, -5, -5, -5, -5, -5, -5, -10,
];

// Precomputed LMR reduction table
static LMR_TABLE: [[i32; 64]; 64] = {
    let mut table = [[0i32; 64]; 64];
    let mut depth = 1;
    while depth < 64 {
        let mut moves = 1;
        while moves < 64 {
            let d = depth as f64;
            let m = moves as f64;
            // Use ln approximation via log2 * ln(2)
            let ln2 = 0.6931471805599453;
            let log2_d = {
                let mut v = d;
                let mut r = 0.0;
                while v >= 2.0 { v /= 2.0; r += 1.0; }
                r + (v - 1.0) * 0.5 // rough ln via log2
            };
            let log2_m = {
                let mut v = m;
                let mut r = 0.0;
                while v >= 2.0 { v /= 2.0; r += 1.0; }
                r + (v - 1.0) * 0.5
            };
            let ln_d = log2_d * ln2;
            let ln_m = log2_m * ln2;
            let r = (0.75 + ln_d * ln_m / 2.25) as i32;
            table[depth][moves] = if r < 1 { 1 } else { r };
            moves += 1;
        }
        depth += 1;
    }
    table
};

#[derive(Clone, Copy)]
struct TTEntry {
    key: u64,
    depth: i32,
    score: i32,
    flag: u8,
    best_move: Option<ChessMove>,
}

impl Default for TTEntry {
    fn default() -> Self {
        Self { key: 0, depth: 0, score: 0, flag: 0, best_move: None }
    }
}

#[derive(Clone, Copy)]
struct EvalCacheEntry {
    key: u64,
    score: i32,
}

impl Default for EvalCacheEntry {
    fn default() -> Self {
        Self { key: 0, score: 0 }
    }
}

#[derive(Clone, Copy)]
struct PawnCacheEntry {
    key: u64,
    white_structure_mg: i32,
    black_structure_mg: i32,
    white_structure_eg: i32,
    black_structure_eg: i32,
    white_center_mg: i32,
    black_center_mg: i32,
    white_files: [u8; 8],
    black_files: [u8; 8],
}

impl Default for PawnCacheEntry {
    fn default() -> Self {
        Self {
            key: 0,
            white_structure_mg: 0, black_structure_mg: 0,
            white_structure_eg: 0, black_structure_eg: 0,
            white_center_mg: 0, black_center_mg: 0,
            white_files: [0; 8], black_files: [0; 8],
        }
    }
}

#[derive(Clone, Copy)]
struct ScoredMove {
    chess_move: ChessMove,
    score: i32,
}

#[derive(Clone)]
struct RepetitionTracker {
    hashes: Vec<u64>,
}

impl RepetitionTracker {
    fn new(root_hash: u64) -> Self {
        let mut hashes = Vec::with_capacity(128);
        hashes.push(root_hash);
        Self { hashes }
    }

    fn count(&self, hash: u64) -> u8 {
        let mut c = 0u8;
        for &h in &self.hashes {
            if h == hash {
                c += 1;
                if c >= 3 { return c; }
            }
        }
        c
    }

    fn push(&mut self, hash: u64) {
        self.hashes.push(hash);
    }

    fn pop(&mut self, _hash: u64) {
        self.hashes.pop();
    }
}

struct RawSearchResult {
    move_uci: String,
    score: i32,
    depth: i32,
    nodes: u64,
    pv_uci: Vec<String>,
}

struct RootChunkResult {
    best_move: Option<ChessMove>,
    best_score: i32,
    nodes: u64,
    tt: Vec<TTEntry>,
}

struct RustAlphaBetaEngine {
    depth: i32,
    threads: usize,
    nodes: u64,
    tt: Vec<TTEntry>,
    killer_moves: Vec<[Option<ChessMove>; 2]>,
    history_heuristic: Vec<i32>,
    capture_history: Vec<i16>,
    countermove: Vec<Option<ChessMove>>,  // indexed by previous move's move_key
    move_stack: Vec<Option<ChessMove>>,  // move played at each ply for countermove tracking
    eval_stack: Vec<i32>,  // static eval at each ply for improving detection
    eval_cache: Vec<EvalCacheEntry>,
    pawn_cache: Vec<PawnCacheEntry>,
    deadline: Option<Instant>,
    stopped: bool,
    opening_book: HashMap<u64, &'static str>,
}

impl RustAlphaBetaEngine {
    fn new(depth: i32) -> Self {
        Self {
            depth,
            threads: available_root_threads(),
            nodes: 0,
            tt: vec![TTEntry::default(); TT_SIZE],
            killer_moves: vec![[None, None]; KILLER_PLY_CAPACITY],
            history_heuristic: vec![0; HISTORY_SIZE],
            capture_history: vec![0; HISTORY_SIZE * CAPTURE_HISTORY_PIECES],
            countermove: vec![None; HISTORY_SIZE],
            move_stack: vec![None; KILLER_PLY_CAPACITY],
            eval_stack: vec![0; KILLER_PLY_CAPACITY],
            eval_cache: vec![EvalCacheEntry::default(); EVAL_CACHE_SIZE],
            pawn_cache: vec![PawnCacheEntry::default(); PAWN_CACHE_SIZE],
            deadline: None,
            stopped: false,
            opening_book: build_opening_book(),
        }
    }

    fn choose_move(
        &mut self,
        fen: &str,
        depth: Option<i32>,
        movetime_ms: Option<u64>,
        moves_uci: Option<Vec<String>>,
    ) -> Result<RawSearchResult, String> {
        let mut board = Board::from_str(fen).map_err(|error| {
            format!("invalid FEN for Rust search backend: {error}")
        })?;
        let mut repetition = RepetitionTracker::new(board_hash(&board));

        if let Some(history) = moves_uci {
            for move_uci in history {
                let chess_move = ChessMove::from_str(&move_uci).map_err(|error| {
                    format!("invalid move history for Rust search backend: {error}")
                })?;
                if !move_is_legal(&board, chess_move) {
                    return Err(format!("illegal move history for Rust search backend: {move_uci}"));
                }
                board = board.make_move_new(chess_move);
                repetition.push(board_hash(&board));
            }
        }

        if !matches!(board.status(), BoardStatus::Ongoing) {
            return Err(String::from("Cannot search a finished game."));
        }

        // Opening book lookup: if this position is in the book, play instantly
        let board_key = board_hash(&board);
        if let Some(book_move_str) = self.opening_book.get(&board_key) {
            if let Ok(book_move) = ChessMove::from_str(book_move_str) {
                if move_is_legal(&board, book_move) {
                    return Ok(RawSearchResult {
                        move_uci: book_move.to_string(),
                        score: 0,
                        depth: 1,
                        nodes: 1,
                        pv_uci: vec![book_move.to_string()],
                    });
                }
            }
        }

        let time_budget = movetime_ms.filter(|ms| *ms > 0).map(Duration::from_millis);
        let requested_depth = depth.unwrap_or(self.depth);
        let target_depth = if requested_depth <= 0 && time_budget.is_some() {
            MAX_TIME_SEARCH_DEPTH
        } else {
            requested_depth.max(1)
        };
        self.nodes = 0;
        self.stopped = false;
        let search_start = Instant::now();
        self.deadline = time_budget.map(|budget| search_start + budget);

        let mut best_move: Option<ChessMove> = None;
        let mut best_score = 0;
        let mut completed_depth = 0;
        let mut last_iteration_nodes: Option<u64> = None;
        let mut previous_iteration_nodes: Option<u64> = None;
        let mut last_iteration_time: Option<Duration> = None;
        let mut previous_iteration_time: Option<Duration> = None;

        for current_depth in 1..=target_depth {
            if completed_depth > 0
                && time_budget.is_some()
                && should_skip_next_iteration(
                    search_start,
                    time_budget.unwrap(),
                    current_depth,
                    last_iteration_nodes,
                    previous_iteration_nodes,
                    last_iteration_time,
                    previous_iteration_time,
                    best_score,
                )
            {
                break;
            }

            self.ensure_ply_capacity(current_depth as usize + 16);
            let nodes_before = self.nodes;
            let iteration_start = Instant::now();
            let search = self.search_root(
                &board,
                current_depth,
                (completed_depth > 0).then_some(best_score),
                &mut repetition,
            );
            let Some((score, candidate)) = search else {
                break;
            };
            let iteration_nodes = self.nodes.saturating_sub(nodes_before);
            let iteration_time = iteration_start.elapsed();
            previous_iteration_nodes = last_iteration_nodes;
            previous_iteration_time = last_iteration_time;
            last_iteration_nodes = Some(iteration_nodes);
            last_iteration_time = Some(iteration_time);
            best_score = score;
            best_move = Some(candidate);
            completed_depth = current_depth;

            if best_score.abs() >= MATE_SCORE - 512 {
                break;
            }
        }

        self.deadline = None;

        let chosen = best_move
            .or_else(|| MoveGen::new_legal(&board).next())
            .ok_or_else(|| {
                String::from("No legal moves available for an ongoing position.")
            })?;

        Ok(RawSearchResult {
            move_uci: chosen.to_string(),
            score: best_score,
            depth: completed_depth.max(1),
            nodes: self.nodes,
            pv_uci: self.extract_principal_variation(board, completed_depth.max(1)),
        })
    }
}

impl RustAlphaBetaEngine {
    fn search_root(
        &mut self,
        board: &Board,
        depth: i32,
        previous_score: Option<i32>,
        repetition: &mut RepetitionTracker,
    ) -> Option<(i32, ChessMove)> {
        let mut alpha = -INFINITY;
        let mut beta = INFINITY;
        let mut delta = ASPIRATION_WINDOW;
        if let Some(score) = previous_score.filter(|_| depth >= 3) {
            alpha = (score - delta).max(-INFINITY);
            beta = (score + delta).min(INFINITY);
        }

        loop {
            let Some((score, best_move)) =
                self.search_root_window(board, depth, alpha, beta, repetition)
            else {
                break;
            };

            if alpha != -INFINITY && score <= alpha {
                delta = delta * 3 / 2;
                alpha = (score - delta).max(-INFINITY);
                continue;
            }
            if beta != INFINITY && score >= beta {
                delta = delta * 3 / 2;
                beta = (score + delta).min(INFINITY);
                continue;
            }
            return Some((score, best_move));
        }

        None
    }

    fn search_root_window(
        &mut self,
        board: &Board,
        depth: i32,
        alpha: i32,
        beta: i32,
        repetition: &mut RepetitionTracker,
    ) -> Option<(i32, ChessMove)> {
        let tt_key = board_hash(board);
        let tt_idx = tt_key as usize & TT_MASK;
        let tt_move = {
            let entry = &self.tt[tt_idx];
            if entry.key == tt_key { entry.best_move } else { None }
        };
        let mut root_moves = self.scored_moves(board, tt_move, 0, false);
        if root_moves.is_empty() {
            return None;
        }
        root_moves.sort_unstable_by(|left, right| right.score.cmp(&left.score));

        if !self.should_parallelize_root(depth, root_moves.len()) {
            let score = self.negamax(board, depth, alpha, beta, 0, repetition)?;
            let tt_idx2 = board_hash(board) as usize & TT_MASK;
            let best_move = {
                let entry = &self.tt[tt_idx2];
                if entry.key == board_hash(board) { entry.best_move } else { None }
            }.or_else(|| MoveGen::new_legal(board).next())?;
            return Some((score, best_move));
        }

        self.search_root_parallel(board, depth, alpha, beta, repetition, root_moves)
    }

    fn search_root_parallel(
        &mut self,
        board: &Board,
        depth: i32,
        alpha: i32,
        beta: i32,
        repetition: &mut RepetitionTracker,
        root_moves: Vec<ScoredMove>,
    ) -> Option<(i32, ChessMove)> {
        let first_move = root_moves[0].chess_move;
        let first_child = board.make_move_new(first_move);
        let first_hash = board_hash(&first_child);
        repetition.push(first_hash);
        let mut best_score =
            -self.negamax(&first_child, depth - 1, -beta, -alpha, 1, repetition)?;
        repetition.pop(first_hash);
        let mut best_move = first_move;
        let current_alpha = alpha.max(best_score);

        if current_alpha >= beta || root_moves.len() == 1 {
            self.store_root_result(board, depth, best_score, alpha, beta, best_move);
            return Some((best_score, best_move));
        }

        let remaining: Vec<ChessMove> = root_moves
            .into_iter()
            .skip(1)
            .map(|entry| entry.chess_move)
            .collect();
        let worker_count = self.threads.min(remaining.len()).max(1);
        let mut move_groups = vec![Vec::new(); worker_count];
        for (index, chess_move) in remaining.into_iter().enumerate() {
            move_groups[index % worker_count].push(chess_move);
        }

        let board_copy = *board;
        let repetition_base = repetition.clone();
        let chunk_results = thread::scope(|scope| {
            let mut handles = Vec::with_capacity(move_groups.len());
            for group in move_groups {
                if group.is_empty() {
                    continue;
                }
                let board = board_copy;
                let repetition = repetition_base.clone();
                let mut worker = self.worker_clone();
                handles.push(scope.spawn(move || {
                    worker.search_root_chunk(board, depth, current_alpha, beta, repetition, group)
                }));
            }

            let mut results = Vec::with_capacity(handles.len());
            for handle in handles {
                let chunk = handle.join().ok()??;
                results.push(chunk);
            }
            Some(results)
        })?;

        for chunk in chunk_results {
            self.nodes = self.nodes.saturating_add(chunk.nodes);
            self.merge_tt(chunk.tt);
            if let Some(chess_move) = chunk.best_move {
                if chunk.best_score > best_score {
                    best_score = chunk.best_score;
                    best_move = chess_move;
                }
            }
        }

        self.store_root_result(board, depth, best_score, alpha, beta, best_move);
        Some((best_score, best_move))
    }

    fn search_root_chunk(
        &mut self,
        board: Board,
        depth: i32,
        alpha: i32,
        beta: i32,
        repetition: RepetitionTracker,
        moves: Vec<ChessMove>,
    ) -> Option<RootChunkResult> {
        let mut best_move = None;
        let mut best_score = -INFINITY;
        let mut local_alpha = alpha;

        for chess_move in moves {
            if self.should_stop() {
                break;
            }

            let child = board.make_move_new(chess_move);
            let child_hash = board_hash(&child);
            let mut local_repetition = repetition.clone();
            local_repetition.push(child_hash);

            let mut score = -self.negamax(
                &child,
                depth - 1,
                -local_alpha - 1,
                -local_alpha,
                1,
                &mut local_repetition,
            )?;
            if score > local_alpha {
                score = -self.negamax(
                    &child,
                    depth - 1,
                    -beta,
                    -local_alpha,
                    1,
                    &mut local_repetition,
                )?;
            }

            if score > best_score {
                best_score = score;
                best_move = Some(chess_move);
            }
            if score > local_alpha {
                local_alpha = score;
            }
        }

        Some(RootChunkResult {
            best_move,
            best_score,
            nodes: self.nodes,
            tt: std::mem::take(&mut self.tt),
        })
    }

    fn should_parallelize_root(&self, _depth: i32, _move_count: usize) -> bool {
        // Disabled: TT clone overhead (48MB per worker) outweighs parallelism benefit
        false
    }

    fn worker_clone(&self) -> Self {
        Self {
            depth: self.depth,
            threads: 1,
            nodes: 0,
            tt: self.tt.clone(),
            killer_moves: self.killer_moves.clone(),
            history_heuristic: self.history_heuristic.clone(),
            capture_history: self.capture_history.clone(),
            countermove: self.countermove.clone(),
            move_stack: self.move_stack.clone(),
            eval_stack: self.eval_stack.clone(),
            eval_cache: self.eval_cache.clone(),
            pawn_cache: self.pawn_cache.clone(),
            deadline: self.deadline,
            stopped: false,
            opening_book: HashMap::new(), // Workers don't need the opening book
        }
    }

    fn merge_tt(&mut self, other: Vec<TTEntry>) {
        for (i, entry) in other.iter().enumerate() {
            if entry.key != 0 {
                let existing = &self.tt[i];
                if existing.key == 0 || entry.depth >= existing.depth {
                    self.tt[i] = *entry;
                }
            }
        }
    }

    fn store_root_result(
        &mut self,
        board: &Board,
        depth: i32,
        score: i32,
        alpha_original: i32,
        beta_original: i32,
        best_move: ChessMove,
    ) {
        let flag = if score <= alpha_original {
            UPPER_BOUND
        } else if score >= beta_original {
            LOWER_BOUND
        } else {
            EXACT
        };
        let key = board_hash(board);
        let idx = key as usize & TT_MASK;
        let existing = &self.tt[idx];
        if existing.key == 0 || depth >= existing.depth || existing.key == key {
            self.tt[idx] = TTEntry {
                key,
                depth,
                score,
                flag,
                best_move: Some(best_move),
            };
        }
    }

    fn negamax(
        &mut self,
        board: &Board,
        depth: i32,
        mut alpha: i32,
        mut beta: i32,
        ply: usize,
        repetition: &mut RepetitionTracker,
    ) -> Option<i32> {
        if self.should_stop() {
            return None;
        }
        self.nodes += 1;

        if let Some(terminal_score) = self.terminal_score(board, ply, repetition) {
            return Some(terminal_score);
        }

        // Max ply limit to prevent search explosions
        if ply >= 96 {
            return Some(self.evaluate(board));
        }

        // Mate distance pruning
        {
            let mating_value = MATE_SCORE - ply as i32;
            if mating_value < beta {
                beta = mating_value;
                if alpha >= beta { return Some(beta); }
            }
            let mated_value = -MATE_SCORE + ply as i32;
            if mated_value > alpha {
                alpha = mated_value;
                if alpha >= beta { return Some(alpha); }
            }
        }

        let in_check_now = in_check(board);
        let mut effective_depth = depth;
        // Cap check extension to prevent infinite check sequences
        if in_check_now && ply < 80 {
            effective_depth += 1;
        }

        if effective_depth <= 0 {
            return self.quiescence(board, alpha, beta, ply, repetition);
        }

        let alpha_original = alpha;
        let beta_original = beta;
        let tt_key = board_hash(board);
        let tt_idx = tt_key as usize & TT_MASK;
        let tt_entry = {
            let entry = self.tt[tt_idx];
            if entry.key == tt_key { Some(entry) } else { None }
        };
        let mut tt_move = tt_entry.and_then(|entry| entry.best_move);

        if let Some(entry) = tt_entry {
            if entry.depth >= effective_depth {
                match entry.flag {
                    EXACT => return Some(entry.score),
                    LOWER_BOUND => alpha = alpha.max(entry.score),
                    UPPER_BOUND => beta = beta.min(entry.score),
                    _ => {}
                }
                if alpha >= beta {
                    return Some(entry.score);
                }
            }
        }

        if tt_move.is_none() && effective_depth >= 6 && !in_check_now {
            let iid_depth = if effective_depth >= 8 {
                effective_depth - 3
            } else {
                effective_depth - 2
            };
            if iid_depth > 0 {
                let _ = self.negamax(board, iid_depth, alpha, beta, ply, repetition);
                let entry = &self.tt[tt_idx];
                tt_move = if entry.key == tt_key { entry.best_move } else { None };
            }
        }

        let static_eval = if !in_check_now {
            Some(self.evaluate(board))
        } else {
            None
        };

        // Store static eval for improving detection
        self.ensure_ply_capacity(ply + 2);
        self.eval_stack[ply] = static_eval.unwrap_or(0);
        let improving = !in_check_now && ply >= 2
            && static_eval.map_or(false, |e| e > self.eval_stack[ply - 2]);

        if let Some(eval) = static_eval {
            if effective_depth <= 4
                && eval >= beta + REVERSE_FUTILITY_MARGIN[effective_depth as usize]
                && beta < MATE_SCORE - 1_000
            {
                return Some(eval);
            }
            if effective_depth <= 3
                && eval + RAZOR_MARGIN[effective_depth as usize] <= alpha
                && alpha > -MATE_SCORE + 1_000
            {
                return self.quiescence(board, alpha, beta, ply, repetition);
            }
        }

        if effective_depth >= 3
            && !in_check_now
            && has_non_pawn_material(board, board.side_to_move())
            && beta < MATE_SCORE - 1_000
        {
            if let Some(null_board) = board.null_move() {
                let reduction = 2 + effective_depth / 3;
                let null_hash = board_hash(&null_board);
                repetition.push(null_hash);
                let search = self.negamax(
                    &null_board,
                    effective_depth - 1 - reduction,
                    -beta,
                    -beta + 1,
                    ply + 1,
                    repetition,
                );
                repetition.pop(null_hash);
                let score = -search?;
                if score >= beta {
                    return Some(score);
                }
            }
        }

        // Singular extension: if TT move is much better than alternatives, extend it
        let mut singular_move: Option<ChessMove> = None;
        if let (Some(entry), Some(tt_mv)) = (tt_entry, tt_move) {
            if effective_depth >= 8
                && entry.depth >= effective_depth - 3
                && entry.flag != UPPER_BOUND
                && entry.score.abs() < MATE_SCORE - 512
            {
                singular_move = Some(tt_mv);
            }
        }

        let mut best_move: Option<ChessMove> = None;
        let mut best_score = -INFINITY;
        let mut move_picker = self.scored_moves(board, tt_move, ply, false);
        let mut searched_quiets: Vec<ChessMove> = Vec::with_capacity(16);
        let mut searched_captures: Vec<(ChessMove, Piece)> = Vec::with_capacity(8);

        for index in 0..move_picker.len() {
            if self.should_stop() {
                return None;
            }

            let chess_move = pick_next_move(&mut move_picker, index)?;
            let move_count = index + 1;
            let child = board.make_move_new(chess_move);
            let child_hash = board_hash(&child);
            let is_capture_move = is_capture(board, chess_move);
            let capture_victim = if is_capture_move {
                Some(victim_piece(board, chess_move).unwrap_or(Piece::Pawn))
            } else {
                None
            };
            let is_quiet = !is_capture_move && chess_move.get_promotion().is_none();
            let gives_check_move = in_check(&child);

            if is_quiet && !in_check_now && !gives_check_move {
                if let Some(eval) = static_eval {
                    if effective_depth <= 4
                        && move_count > 1
                        && eval + FUTILITY_MARGIN[effective_depth as usize] <= alpha
                    {
                        continue;
                    }
                }
                if move_count > late_move_pruning_limit(effective_depth) {
                    continue;
                }
                // SEE pruning for quiet moves at low depth
                if effective_depth <= 4 && move_count > 3
                    && static_exchange_eval(board, chess_move) < -50 * effective_depth
                {
                    continue;
                }
                searched_quiets.push(chess_move);
            }
            if let Some(victim) = capture_victim {
                searched_captures.push((chess_move, victim));
            }

            // Singular extension: extend TT move if it's singularly better
            let mut extension = 0;
            if singular_move == Some(chess_move) {
                let se_beta = tt_entry.unwrap().score - 2 * effective_depth;
                let se_depth = (effective_depth - 1) / 2;
                // Search excluding TT move at reduced depth/window
                let excluded_score = self.negamax_excluding(
                    board, se_depth, se_beta - 1, se_beta,
                    ply, repetition, chess_move,
                );
                if let Some(se_score) = excluded_score {
                    if se_score < se_beta {
                        extension = 1; // TT move is singular, extend it
                    }
                }
            }

            // Track move at this ply for countermove recording
            self.ensure_ply_capacity(ply + 2);
            self.move_stack[ply] = Some(chess_move);

            repetition.push(child_hash);

            let score = (|| -> Option<i32> {
                if move_count == 1 {
                    return Some(-self.negamax(
                        &child,
                        effective_depth - 1 + extension,
                        -beta,
                        -alpha,
                        ply + 1,
                        repetition,
                    )?);
                }

                let mut search_depth = effective_depth - 1;
                if effective_depth >= 4
                    && move_count > 3
                    && is_quiet
                    && !in_check_now
                    && !gives_check_move
                {
                    let mut reduction = late_move_reduction(effective_depth, move_count);
                    // Reduce less for countermoves and killers
                    let mk = move_key(chess_move) as usize;
                    if self.killer_moves.get(ply).map_or(false, |k| k[0] == Some(chess_move) || k[1] == Some(chess_move)) {
                        reduction = (reduction - 1).max(0);
                    }
                    // Reduce more if not improving
                    if !improving {
                        reduction += 1;
                    }
                    // Reduce more for moves with bad history
                    let hist = self.history_heuristic[mk];
                    if hist < -2000 {
                        reduction += 1;
                    } else if hist > 8000 {
                        reduction = (reduction - 1).max(0);
                    }
                    search_depth = (search_depth - reduction).max(0);
                }

                let mut score = -self.negamax(
                    &child,
                    search_depth,
                    -alpha - 1,
                    -alpha,
                    ply + 1,
                    repetition,
                )?;
                if score > alpha && search_depth != effective_depth - 1 {
                    score = -self.negamax(
                        &child,
                        effective_depth - 1,
                        -alpha - 1,
                        -alpha,
                        ply + 1,
                        repetition,
                    )?;
                }
                if score > alpha && score < beta {
                    score = -self.negamax(
                        &child,
                        effective_depth - 1,
                        -beta,
                        -alpha,
                        ply + 1,
                        repetition,
                    )?;
                }
                Some(score)
            })();
            repetition.pop(child_hash);
            let score = score?;

            if score > best_score {
                best_score = score;
                best_move = Some(chess_move);
            }
            if score > alpha {
                alpha = score;
            }
            if alpha >= beta {
                let bonus = effective_depth * effective_depth;
                if is_quiet {
                    self.record_killer(chess_move, ply);
                    self.update_history(chess_move, bonus);
                    // Record countermove: this move refutes the previous move
                    if ply > 0 {
                        if let Some(prev_move) = self.move_stack[ply - 1] {
                            self.countermove[move_key(prev_move) as usize] = Some(chess_move);
                        }
                    }
                    for previous in searched_quiets.iter().copied() {
                        if previous != chess_move {
                            self.update_history(previous, -bonus);
                        }
                    }
                } else if let Some(victim) = capture_victim {
                    self.update_capture_history(chess_move, victim, bonus);
                    for (previous, previous_victim) in searched_captures.iter().copied() {
                        if previous != chess_move {
                            self.update_capture_history(previous, previous_victim, -bonus);
                        }
                    }
                }
                break;
            }
        }

        if best_move.is_none() {
            return Some(
                self.terminal_score(board, ply, repetition)
                    .unwrap_or(DRAW_SCORE),
            );
        }

        let flag = if best_score <= alpha_original {
            UPPER_BOUND
        } else if best_score >= beta_original {
            LOWER_BOUND
        } else {
            EXACT
        };
        let new_entry = TTEntry {
            key: tt_key,
            depth: effective_depth,
            score: best_score,
            flag,
            best_move,
        };
        let existing = &self.tt[tt_idx];
        if existing.key == 0 || new_entry.depth >= existing.depth || existing.key == tt_key {
            self.tt[tt_idx] = new_entry;
        }
        Some(best_score)
    }

    fn quiescence(
        &mut self,
        board: &Board,
        mut alpha: i32,
        beta: i32,
        ply: usize,
        repetition: &mut RepetitionTracker,
    ) -> Option<i32> {
        if self.should_stop() {
            return None;
        }
        self.nodes += 1;

        if let Some(terminal_score) = self.terminal_score(board, ply, repetition) {
            return Some(terminal_score);
        }

        let in_check_now = in_check(board);
        let stand_pat = self.evaluate(board);
        if !in_check_now {
            if stand_pat >= beta {
                return Some(stand_pat);
            }
            alpha = alpha.max(stand_pat);
        }

        let mut move_picker = self.scored_moves(board, None, ply, !in_check_now);
        for index in 0..move_picker.len() {
            let chess_move = pick_next_move(&mut move_picker, index)?;
            if !in_check_now {
                if let Some(victim) = victim_piece(board, chess_move) {
                    if stand_pat + piece_value(victim) + 200 < alpha {
                        continue;
                    }
                }
                if is_capture(board, chess_move)
                    && static_exchange_eval(board, chess_move) < -SEE_PRUNE_MARGIN
                {
                    continue;
                }
            }

            let child = board.make_move_new(chess_move);
            let child_hash = board_hash(&child);
            repetition.push(child_hash);
            let search = self.quiescence(&child, -beta, -alpha, ply + 1, repetition);
            repetition.pop(child_hash);
            let score = -search?;
            if score >= beta {
                return Some(score);
            }
            alpha = alpha.max(score);
        }

        Some(alpha)
    }

    /// Simplified negamax that excludes a specific move — used for singular extension verification
    fn negamax_excluding(
        &mut self,
        board: &Board,
        depth: i32,
        alpha: i32,
        beta: i32,
        ply: usize,
        repetition: &mut RepetitionTracker,
        excluded_move: ChessMove,
    ) -> Option<i32> {
        if self.should_stop() || depth <= 0 {
            return Some(self.evaluate(board));
        }

        let mut best_score = -INFINITY;
        let mut moves = self.scored_moves(board, None, ply, false);

        for index in 0..moves.len() {
            let chess_move = pick_next_move(&mut moves, index)?;
            if chess_move == excluded_move {
                continue;
            }

            let child = board.make_move_new(chess_move);
            let child_hash = board_hash(&child);
            repetition.push(child_hash);
            let score = -self.negamax(&child, depth - 1, -beta, -alpha, ply + 1, repetition)?;
            repetition.pop(child_hash);

            if score > best_score {
                best_score = score;
            }
            if score >= beta {
                return Some(score);
            }
        }

        Some(best_score)
    }

    fn should_stop(&mut self) -> bool {
        if self.stopped {
            return true;
        }
        if self.nodes & 1023 != 0 {
            return false;
        }
        if let Some(deadline) = self.deadline {
            if Instant::now() >= deadline {
                self.stopped = true;
                return true;
            }
        }
        false
    }

    fn evaluate(&mut self, board: &Board) -> i32 {
        let board_key = board_hash(board);
        let eval_idx = board_key as usize & EVAL_CACHE_MASK;
        let cached = &self.eval_cache[eval_idx];
        if cached.key == board_key {
            return cached.score;
        }

        let phase = game_phase(board);
        let endgame = phase <= 6;
        let mut white_mg = 0;
        let mut black_mg = 0;
        let mut white_eg = 0;
        let mut black_eg = 0;

        for square in piece_bb(board, Color::White, Piece::Pawn) {
            let idx = square_index(square);
            white_mg += PAWN + PAWN_TABLE[idx];
            white_eg += PAWN + PAWN_EG_TABLE[idx];
        }
        for square in piece_bb(board, Color::Black, Piece::Pawn) {
            let idx = mirror_index(square_index(square));
            black_mg += PAWN + PAWN_TABLE[idx];
            black_eg += PAWN + PAWN_EG_TABLE[idx];
        }

        for square in piece_bb(board, Color::White, Piece::Knight) {
            let idx = square_index(square);
            white_mg += KNIGHT + KNIGHT_TABLE[idx];
            white_eg += KNIGHT + KNIGHT_EG_TABLE[idx];
        }
        for square in piece_bb(board, Color::Black, Piece::Knight) {
            let idx = mirror_index(square_index(square));
            black_mg += KNIGHT + KNIGHT_TABLE[idx];
            black_eg += KNIGHT + KNIGHT_EG_TABLE[idx];
        }

        for square in piece_bb(board, Color::White, Piece::Bishop) {
            let idx = square_index(square);
            white_mg += BISHOP + BISHOP_TABLE[idx];
            white_eg += BISHOP + BISHOP_EG_TABLE[idx];
        }
        for square in piece_bb(board, Color::Black, Piece::Bishop) {
            let idx = mirror_index(square_index(square));
            black_mg += BISHOP + BISHOP_TABLE[idx];
            black_eg += BISHOP + BISHOP_EG_TABLE[idx];
        }

        for square in piece_bb(board, Color::White, Piece::Rook) {
            let idx = square_index(square);
            white_mg += ROOK + ROOK_TABLE[idx];
            white_eg += ROOK + ROOK_EG_TABLE[idx];
        }
        for square in piece_bb(board, Color::Black, Piece::Rook) {
            let idx = mirror_index(square_index(square));
            black_mg += ROOK + ROOK_TABLE[idx];
            black_eg += ROOK + ROOK_EG_TABLE[idx];
        }

        for square in piece_bb(board, Color::White, Piece::Queen) {
            let idx = square_index(square);
            white_mg += QUEEN + QUEEN_TABLE[idx];
            white_eg += QUEEN + QUEEN_EG_TABLE[idx];
        }
        for square in piece_bb(board, Color::Black, Piece::Queen) {
            let idx = mirror_index(square_index(square));
            black_mg += QUEEN + QUEEN_TABLE[idx];
            black_eg += QUEEN + QUEEN_EG_TABLE[idx];
        }

        white_mg += king_position_score(board, Color::White, false);
        black_mg += king_position_score(board, Color::Black, false);
        white_eg += king_position_score(board, Color::White, true);
        black_eg += king_position_score(board, Color::Black, true);

        let pawn_entry = self.pawn_entry(board);
        white_mg += pawn_entry.white_structure_mg + pawn_entry.white_center_mg;
        black_mg += pawn_entry.black_structure_mg + pawn_entry.black_center_mg;
        white_eg += pawn_entry.white_structure_eg;
        black_eg += pawn_entry.black_structure_eg;

        let white_mobility = mobility_score(board, Color::White);
        let black_mobility = mobility_score(board, Color::Black);
        white_mg += white_mobility;
        black_mg += black_mobility;
        white_eg += white_mobility / 2;
        black_eg += black_mobility / 2;

        let white_rook_activity = rook_activity_score(
            board,
            Color::White,
            &pawn_entry.white_files,
            &pawn_entry.black_files,
        );
        let black_rook_activity = rook_activity_score(
            board,
            Color::Black,
            &pawn_entry.black_files,
            &pawn_entry.white_files,
        );
        white_mg += white_rook_activity;
        black_mg += black_rook_activity;
        white_eg += white_rook_activity;
        black_eg += black_rook_activity;

        let white_outposts = knight_outpost_score(board, Color::White);
        let black_outposts = knight_outpost_score(board, Color::Black);
        white_mg += white_outposts;
        black_mg += black_outposts;
        white_eg += white_outposts / 3;
        black_eg += black_outposts / 3;

        white_mg += development_score(board, Color::White, endgame);
        black_mg += development_score(board, Color::Black, endgame);

        if piece_bb(board, Color::White, Piece::Bishop).popcnt() >= 2 {
            white_mg += BISHOP_PAIR_BONUS;
            white_eg += BISHOP_PAIR_BONUS + 6;
        }
        if piece_bb(board, Color::Black, Piece::Bishop).popcnt() >= 2 {
            black_mg += BISHOP_PAIR_BONUS;
            black_eg += BISHOP_PAIR_BONUS + 6;
        }

        white_mg += king_safety_score(
            board,
            Color::White,
            phase,
            &pawn_entry.white_files,
            &pawn_entry.black_files,
        );
        black_mg += king_safety_score(
            board,
            Color::Black,
            phase,
            &pawn_entry.black_files,
            &pawn_entry.white_files,
        );

        // Threat evaluation
        let white_threats = threat_score(board, Color::White);
        let black_threats = threat_score(board, Color::Black);
        white_mg += white_threats;
        black_mg += black_threats;
        white_eg += white_threats / 2;
        black_eg += black_threats / 2;

        // Connected rooks bonus
        let white_connected = connected_rooks_score(board, Color::White);
        let black_connected = connected_rooks_score(board, Color::Black);
        white_mg += white_connected;
        black_mg += black_connected;

        // Passed pawn king distance (endgame only)
        white_eg += passed_pawn_king_bonus(board, Color::White);
        black_eg += passed_pawn_king_bonus(board, Color::Black);

        // Mop-up eval: drive enemy king to corner in won endgames
        let white_mopup = mopup_score(board, Color::White);
        let black_mopup = mopup_score(board, Color::Black);
        white_eg += white_mopup;
        black_eg += black_mopup;

        let white_score = tapered_score(white_mg, white_eg, phase);
        let black_score = tapered_score(black_mg, black_eg, phase);

        let mut score = if board.side_to_move() == Color::White {
            white_score - black_score + TEMPO_BONUS
        } else {
            black_score - white_score + TEMPO_BONUS
        };

        // Contempt: penalize draws slightly when we have material advantage
        // This makes the engine play for wins against weaker opponents
        if score.abs() < 50 {
            let our_material = if board.side_to_move() == Color::White {
                white_mg
            } else {
                black_mg
            };
            let their_material = if board.side_to_move() == Color::White {
                black_mg
            } else {
                white_mg
            };
            if our_material > their_material + 100 {
                score += CONTEMPT;
            }
        }

        self.eval_cache[eval_idx] = EvalCacheEntry { key: board_key, score };
        score
    }

    fn terminal_score(
        &self,
        board: &Board,
        ply: usize,
        repetition: &RepetitionTracker,
    ) -> Option<i32> {
        match board.status() {
            BoardStatus::Checkmate => Some(-MATE_SCORE + ply as i32),
            BoardStatus::Stalemate => Some(-CONTEMPT), // Slight contempt: avoid stalemate
            BoardStatus::Ongoing => {
                let rep_count = repetition.count(board_hash(board));
                if rep_count >= 3 {
                    Some(-CONTEMPT) // Slight contempt: avoid repetition draws
                } else if rep_count >= 2 && ply > 0 {
                    Some(-CONTEMPT)
                } else {
                    None
                }
            }
        }
    }

    fn scored_moves(
        &self,
        board: &Board,
        tt_move: Option<ChessMove>,
        ply: usize,
        captures_only: bool,
    ) -> Vec<ScoredMove> {
        MoveGen::new_legal(board)
            .filter(|candidate| {
                !captures_only
                    || is_capture(board, *candidate)
                    || candidate.get_promotion().is_some()
            })
            .map(|candidate| ScoredMove {
                chess_move: candidate,
                score: self.move_order_score(board, candidate, tt_move, ply),
            })
            .collect()
    }

    fn move_order_score(
        &self,
        board: &Board,
        chess_move: ChessMove,
        tt_move: Option<ChessMove>,
        ply: usize,
    ) -> i32 {
        if tt_move == Some(chess_move) {
            return 10_000_000;
        }

        let mut score = self.history_heuristic[move_key(chess_move) as usize];
        let is_quiet = !is_capture(board, chess_move) && chess_move.get_promotion().is_none();

        if let Some(promotion) = chess_move.get_promotion() {
            score += 800_000 + piece_value(promotion);
        }

        if !is_quiet {
            let victim = victim_piece(board, chess_move).unwrap_or(Piece::Pawn);
            let attacker = board
                .piece_on(chess_move.get_source())
                .unwrap_or(Piece::Pawn);
            score += 500_000 + 16 * piece_value(victim) - piece_value(attacker);
            score += i32::from(self.capture_history[capture_history_key(chess_move, victim)]);
        }

        if let Some(killers) = self.killer_moves.get(ply) {
            if killers[0] == Some(chess_move) {
                score += 300_000;
            } else if killers[1] == Some(chess_move) {
                score += 290_000;
            }
        }

        // Countermove bonus: if this move refutes the previous move
        if ply > 0 {
            if let Some(prev_move) = self.move_stack.get(ply - 1).copied().flatten() {
                if self.countermove.get(move_key(prev_move) as usize).copied().flatten() == Some(chess_move) {
                    score += 200_000;
                }
            }
        }

        if is_castling(board, chess_move) {
            score += 10_000;
        }

        score
    }

    fn record_killer(&mut self, chess_move: ChessMove, ply: usize) {
        self.ensure_ply_capacity(ply + 1);
        let killers = &mut self.killer_moves[ply];
        if killers[0] == Some(chess_move) {
            return;
        }
        killers[1] = killers[0];
        killers[0] = Some(chess_move);
    }

    /// Stockfish-style history gravity: bonus is scaled down as the value approaches the limit,
    /// preventing saturation and allowing recent information to have more weight.
    fn update_history(&mut self, chess_move: ChessMove, bonus: i32) {
        let history = &mut self.history_heuristic[move_key(chess_move) as usize];
        let clamped = bonus.clamp(-HISTORY_LIMIT, HISTORY_LIMIT);
        *history += clamped - *history * clamped.abs() / HISTORY_LIMIT;
    }

    fn update_capture_history(&mut self, chess_move: ChessMove, victim: Piece, bonus: i32) {
        let history = &mut self.capture_history[capture_history_key(chess_move, victim)];
        let h = i32::from(*history);
        let clamped = bonus.clamp(-CAPTURE_HISTORY_LIMIT, CAPTURE_HISTORY_LIMIT);
        let updated = h + clamped - h * clamped.abs() / CAPTURE_HISTORY_LIMIT;
        *history = updated.clamp(-CAPTURE_HISTORY_LIMIT, CAPTURE_HISTORY_LIMIT) as i16;
    }

    fn ensure_ply_capacity(&mut self, size: usize) {
        if self.killer_moves.len() < size {
            self.killer_moves.resize(size, [None, None]);
        }
        if self.move_stack.len() < size {
            self.move_stack.resize(size, None);
        }
        if self.eval_stack.len() < size {
            self.eval_stack.resize(size, 0);
        }
    }

    fn pawn_entry(&mut self, board: &Board) -> PawnCacheEntry {
        let pawn_hash = board.get_pawn_hash();
        let pawn_idx = pawn_hash as usize & PAWN_CACHE_MASK;
        let cached = &self.pawn_cache[pawn_idx];
        if cached.key == pawn_hash {
            return *cached;
        }

        let mut entry = analyze_pawns(board);
        entry.key = pawn_hash;
        self.pawn_cache[pawn_idx] = entry;
        entry
    }

    fn extract_principal_variation(&self, mut board: Board, depth: i32) -> Vec<String> {
        let mut line = Vec::new();

        for _ in 0..depth {
            let key = board_hash(&board);
            let idx = key as usize & TT_MASK;
            let entry = &self.tt[idx];
            if entry.key != key {
                break;
            }
            let Some(best_move) = entry.best_move else {
                break;
            };
            if !move_is_legal(&board, best_move) {
                break;
            }
            line.push(best_move.to_string());
            board = board.make_move_new(best_move);
        }

        line
    }
}

// =============================================================================
// Helper functions
// =============================================================================

fn board_hash(board: &Board) -> u64 {
    board.get_hash()
}

fn available_root_threads() -> usize {
    thread::available_parallelism()
        .map(|parallelism| parallelism.get().min(MAX_ROOT_THREADS))
        .unwrap_or(1)
}

#[inline(always)]
fn square_index(square: Square) -> usize {
    square.to_index()
}

#[inline(always)]
fn mirror_index(index: usize) -> usize {
    index ^ 56
}

fn piece_value(piece: Piece) -> i32 {
    match piece {
        Piece::Pawn => PAWN,
        Piece::Knight => KNIGHT,
        Piece::Bishop => BISHOP,
        Piece::Rook => ROOK,
        Piece::Queen => QUEEN,
        Piece::King => 0,
    }
}

fn move_key(chess_move: ChessMove) -> u16 {
    let source = square_index(chess_move.get_source()) as u16;
    let dest = square_index(chess_move.get_dest()) as u16;
    let promotion = match chess_move.get_promotion() {
        Some(Piece::Knight) => 1,
        Some(Piece::Bishop) => 2,
        Some(Piece::Rook) => 3,
        Some(Piece::Queen) => 4,
        _ => 0,
    };
    (source << 10) | (dest << 4) | promotion
}

fn capture_piece_index(piece: Piece) -> usize {
    match piece {
        Piece::Pawn => 0,
        Piece::Knight => 1,
        Piece::Bishop => 2,
        Piece::Rook => 3,
        Piece::Queen => 4,
        Piece::King => 5,
    }
}

fn capture_history_key(chess_move: ChessMove, victim: Piece) -> usize {
    move_key(chess_move) as usize * CAPTURE_HISTORY_PIECES + capture_piece_index(victim)
}

fn color_index(color: Color) -> usize {
    match color {
        Color::White => 0,
        Color::Black => 1,
    }
}

fn piece_index(piece: Piece) -> usize {
    match piece {
        Piece::Pawn => 0,
        Piece::Knight => 1,
        Piece::Bishop => 2,
        Piece::Rook => 3,
        Piece::Queen => 4,
        Piece::King => 5,
    }
}

/// Fast bitboard accessor — no allocation, returns BitBoard for direct iteration
#[inline(always)]
fn piece_bb(board: &Board, color: Color, piece: Piece) -> BitBoard {
    *board.pieces(piece) & *board.color_combined(color)
}

/// Fast king square lookup — no Vec allocation
#[inline(always)]
fn king_square(board: &Board, color: Color) -> Option<Square> {
    let bb = piece_bb(board, color, Piece::King);
    if bb == BitBoard(0) { None } else { Some(bb.to_square()) }
}

fn remove_piece(
    piece_occ: &mut [BitBoard; 6],
    color_occ: &mut [BitBoard; 2],
    color: Color,
    piece: Piece,
    square: Square,
) {
    let bit = BitBoard::from_square(square);
    piece_occ[piece_index(piece)] ^= bit;
    color_occ[color_index(color)] ^= bit;
}

fn add_piece(
    piece_occ: &mut [BitBoard; 6],
    color_occ: &mut [BitBoard; 2],
    color: Color,
    piece: Piece,
    square: Square,
) {
    let bit = BitBoard::from_square(square);
    piece_occ[piece_index(piece)] |= bit;
    color_occ[color_index(color)] |= bit;
}

fn attackers_to_square(
    square: Square,
    occupied: BitBoard,
    piece_occ: &[BitBoard; 6],
    color_occ: &[BitBoard; 2],
) -> BitBoard {
    let pawns = piece_occ[piece_index(Piece::Pawn)];
    let rooks = piece_occ[piece_index(Piece::Rook)] | piece_occ[piece_index(Piece::Queen)];
    let bishops = piece_occ[piece_index(Piece::Bishop)] | piece_occ[piece_index(Piece::Queen)];
    let knights = piece_occ[piece_index(Piece::Knight)];
    let kings = piece_occ[piece_index(Piece::King)];
    let white_pawns = pawns & color_occ[color_index(Color::White)];
    let black_pawns = pawns & color_occ[color_index(Color::Black)];

    get_rook_moves(square, occupied) & rooks
        | get_bishop_moves(square, occupied) & bishops
        | get_knight_moves(square) & knights
        | get_king_moves(square) & kings
        | get_pawn_attacks(square, Color::Black, white_pawns)
        | get_pawn_attacks(square, Color::White, black_pawns)
}

fn least_valuable_attacker(attackers: BitBoard, piece_occ: &[BitBoard; 6]) -> Option<(Square, Piece)> {
    for piece in [
        Piece::Pawn,
        Piece::Knight,
        Piece::Bishop,
        Piece::Rook,
        Piece::Queen,
        Piece::King,
    ] {
        let matches = attackers & piece_occ[piece_index(piece)];
        if matches != BitBoard(0) {
            return Some((matches.to_square(), piece));
        }
    }
    None
}

fn static_exchange_eval(board: &Board, chess_move: ChessMove) -> i32 {
    let source = chess_move.get_source();
    let dest = chess_move.get_dest();
    let side = board.side_to_move();
    let moving_piece = match board.piece_on(source) {
        Some(piece) => piece,
        None => return 0,
    };
    let captured_piece = match victim_piece(board, chess_move) {
        Some(piece) => piece,
        None => return 0,
    };

    let mut piece_occ = [
        *board.pieces(Piece::Pawn),
        *board.pieces(Piece::Knight),
        *board.pieces(Piece::Bishop),
        *board.pieces(Piece::Rook),
        *board.pieces(Piece::Queen),
        *board.pieces(Piece::King),
    ];
    let mut color_occ = [*board.color_combined(Color::White), *board.color_combined(Color::Black)];
    let mut gains = [0; 32];
    gains[0] = piece_value(captured_piece);

    remove_piece(&mut piece_occ, &mut color_occ, side, moving_piece, source);
    if is_en_passant_capture(board, chess_move) {
        let captured_square = Square::make_square(source.get_rank(), dest.get_file());
        remove_piece(
            &mut piece_occ,
            &mut color_occ,
            !side,
            Piece::Pawn,
            captured_square,
        );
    } else {
        remove_piece(&mut piece_occ, &mut color_occ, !side, captured_piece, dest);
    }

    let placed_piece = chess_move.get_promotion().unwrap_or(moving_piece);
    if let Some(promotion) = chess_move.get_promotion() {
        gains[0] += piece_value(promotion) - piece_value(Piece::Pawn);
    }
    add_piece(&mut piece_occ, &mut color_occ, side, placed_piece, dest);

    let mut occupied = color_occ[0] | color_occ[1];
    let mut occupant_color = side;
    let mut occupant_piece = placed_piece;
    let mut current_side = !side;
    let mut depth = 0usize;

    loop {
        let attackers =
            attackers_to_square(dest, occupied, &piece_occ, &color_occ) & color_occ[color_index(current_side)];
        if attackers == BitBoard(0) {
            break;
        }
        let Some((attacker_square, attacker_piece)) = least_valuable_attacker(attackers, &piece_occ) else {
            break;
        };

        depth += 1;
        gains[depth] = piece_value(attacker_piece) - gains[depth - 1];
        if gains[depth].max(-gains[depth - 1]) < 0 {
            break;
        }

        remove_piece(&mut piece_occ, &mut color_occ, occupant_color, occupant_piece, dest);
        remove_piece(
            &mut piece_occ,
            &mut color_occ,
            current_side,
            attacker_piece,
            attacker_square,
        );
        add_piece(
            &mut piece_occ,
            &mut color_occ,
            current_side,
            attacker_piece,
            dest,
        );
        occupied = color_occ[0] | color_occ[1];
        occupant_color = current_side;
        occupant_piece = attacker_piece;
        current_side = !current_side;
    }

    while depth > 0 {
        depth -= 1;
        gains[depth] = -gains[depth + 1].max(-gains[depth]);
    }

    gains[0]
}

fn pick_next_move(moves: &mut [ScoredMove], start: usize) -> Option<ChessMove> {
    if start >= moves.len() {
        return None;
    }

    let mut best_index = start;
    for index in (start + 1)..moves.len() {
        if moves[index].score > moves[best_index].score {
            best_index = index;
        }
    }
    moves.swap(start, best_index);
    Some(moves[start].chess_move)
}

/// Logarithmic LMR reduction using precomputed table
fn late_move_reduction(depth: i32, move_count: usize) -> i32 {
    let d = (depth as usize).min(63);
    let m = move_count.min(63);
    LMR_TABLE[d][m]
}

fn late_move_pruning_limit(depth: i32) -> usize {
    match depth {
        d if d <= 1 => 4,
        2 => 9,
        3 => 14,
        4 => 20,
        5 => 28,
        6 => 38,
        7 => 50,
        8 => 65,
        _ => usize::MAX,
    }
}

fn projected_next_iteration_time(
    depth: i32,
    last_time: Duration,
    previous_time: Option<Duration>,
    last_nodes: u64,
    previous_nodes: Option<u64>,
) -> Duration {
    let mut factor: f64 = match depth {
        0..=2 => 1.3,
        3 => 1.45,
        4 => 1.65,
        5 => 1.82,
        _ => 1.95,
    };

    if let Some(previous) = previous_time {
        if previous.as_secs_f64() > 0.0 {
            let ratio = (last_time.as_secs_f64() / previous.as_secs_f64()).clamp(1.15, 3.2);
            factor = factor.max((0.55 * factor + 0.45 * ratio).clamp(1.15, 3.2));
        }
    }
    if let Some(previous) = previous_nodes {
        if previous > 0 {
            let ratio = (last_nodes as f64 / previous as f64).clamp(1.15, 3.2);
            factor = factor.max((0.55 * factor + 0.45 * ratio).clamp(1.15, 3.2));
        }
    }

    Duration::from_secs_f64((last_time.as_secs_f64() * factor.clamp(1.15, 3.2)).max(0.001))
}

fn should_skip_next_iteration(
    search_start: Instant,
    budget: Duration,
    next_depth: i32,
    last_nodes: Option<u64>,
    previous_nodes: Option<u64>,
    last_time: Option<Duration>,
    previous_time: Option<Duration>,
    best_score: i32,
) -> bool {
    if best_score.abs() >= MATE_SCORE - 512 {
        return true;
    }

    let Some(last_time) = last_time else {
        return false;
    };
    let Some(last_nodes) = last_nodes else {
        return false;
    };

    let elapsed = search_start.elapsed();
    if elapsed >= budget {
        return true;
    }
    let remaining = budget.saturating_sub(elapsed);
    if remaining.as_secs_f64() <= 0.0 {
        return true;
    }

    if next_depth <= 4 {
        return false;
    }

    if elapsed.mul_f64(2.0) <= budget {
        return false;
    }

    if next_depth <= 6 && last_time <= remaining {
        return false;
    }

    let projected = projected_next_iteration_time(
        next_depth,
        last_time,
        previous_time,
        last_nodes,
        previous_nodes,
    );
    let threshold = if next_depth >= 7 {
        1.02
    } else if next_depth >= 6 {
        1.12
    } else {
        1.2
    };
    projected.as_secs_f64() > remaining.as_secs_f64() * threshold
}

fn move_is_legal(board: &Board, candidate: ChessMove) -> bool {
    MoveGen::new_legal(board).any(|legal_move| legal_move == candidate)
}

fn in_check(board: &Board) -> bool {
    board.checkers().popcnt() > 0
}


fn is_castling(board: &Board, chess_move: ChessMove) -> bool {
    if board.piece_on(chess_move.get_source()) != Some(Piece::King) {
        return false;
    }
    let source_file = file_index(chess_move.get_source());
    let dest_file = file_index(chess_move.get_dest());
    (source_file - dest_file).abs() > 1
}

fn is_capture(board: &Board, chess_move: ChessMove) -> bool {
    victim_piece(board, chess_move).is_some()
}

fn is_en_passant_capture(board: &Board, chess_move: ChessMove) -> bool {
    board.piece_on(chess_move.get_source()) == Some(Piece::Pawn)
        && board.en_passant() == Some(chess_move.get_dest())
        && board.piece_on(chess_move.get_dest()).is_none()
        && file_index(chess_move.get_source()) != file_index(chess_move.get_dest())
}

fn victim_piece(board: &Board, chess_move: ChessMove) -> Option<Piece> {
    if let Some(piece) = board.piece_on(chess_move.get_dest()) {
        return Some(piece);
    }

    if board.piece_on(chess_move.get_source()) == Some(Piece::Pawn) {
        if let Some(ep_square) = board.en_passant() {
            let source_file = file_index(chess_move.get_source());
            let dest_file = file_index(chess_move.get_dest());
            if ep_square == chess_move.get_dest() && source_file != dest_file {
                return Some(Piece::Pawn);
            }
        }
    }

    None
}

fn has_non_pawn_material(board: &Board, color: Color) -> bool {
    for piece in [Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen] {
        if (*board.pieces(piece) & *board.color_combined(color)).popcnt() > 0 {
            return true;
        }
    }
    false
}

fn file_index(square: Square) -> i32 {
    square.get_file().to_index() as i32
}

fn rank_index(square: Square) -> i32 {
    square.get_rank().to_index() as i32
}

fn square_from_coords(file: i32, rank: i32) -> Option<Square> {
    if !(0..=7).contains(&file) || !(0..=7).contains(&rank) {
        return None;
    }
    Some(Square::make_square(
        Rank::from_index(rank as usize),
        File::from_index(file as usize),
    ))
}

fn king_position_score(board: &Board, color: Color, endgame: bool) -> i32 {
    let Some(square) = king_square(board, color) else {
        return 0;
    };
    let table = if endgame {
        &KING_ENDGAME_TABLE
    } else {
        &KING_MIDGAME_TABLE
    };
    let index = square_index(square);
    if color == Color::White {
        table[index]
    } else {
        table[mirror_index(index)]
    }
}

fn game_phase(board: &Board) -> i32 {
    let knights = (*board.pieces(Piece::Knight)).popcnt() as i32;
    let bishops = (*board.pieces(Piece::Bishop)).popcnt() as i32;
    let rooks = (*board.pieces(Piece::Rook)).popcnt() as i32;
    let queens = (*board.pieces(Piece::Queen)).popcnt() as i32;
    (knights + bishops + rooks * 2 + queens * 4).min(PHASE_TOTAL)
}

fn tapered_score(midgame: i32, endgame: i32, phase: i32) -> i32 {
    (midgame * phase + endgame * (PHASE_TOTAL - phase)) / PHASE_TOTAL
}

fn analyze_pawns(board: &Board) -> PawnCacheEntry {
    let mut white_rank_bits: [u8; 8] = [0; 8];
    let mut black_rank_bits: [u8; 8] = [0; 8];
    let mut white_files = [0_u8; 8];
    let mut black_files = [0_u8; 8];
    let mut white_center_mg = 0;
    let mut black_center_mg = 0;

    for square in piece_bb(board, Color::White, Piece::Pawn) {
        let file = file_index(square) as usize;
        let rank = rank_index(square);
        white_rank_bits[file] |= 1u8 << rank;
        white_files[file] += 1;
        if file == 3 || file == 4 {
            if (3..=4).contains(&rank) {
                white_center_mg += CENTER_PAWN_BONUS;
            }
            if rank >= 4 {
                white_center_mg += 4;
            }
        }
    }

    for square in piece_bb(board, Color::Black, Piece::Pawn) {
        let file = file_index(square) as usize;
        let rank = rank_index(square);
        black_rank_bits[file] |= 1u8 << rank;
        black_files[file] += 1;
        if file == 3 || file == 4 {
            if (3..=4).contains(&rank) {
                black_center_mg += CENTER_PAWN_BONUS;
            }
            if rank <= 3 {
                black_center_mg += 4;
            }
        }
    }

    let (mut white_structure_mg, mut white_structure_eg) = pawn_structure_from_bits(&white_rank_bits);
    let (mut black_structure_mg, mut black_structure_eg) = pawn_structure_from_bits(&black_rank_bits);

    for square in piece_bb(board, Color::White, Piece::Pawn) {
        let file = file_index(square);
        let rank = rank_index(square);
        if is_passed_pawn_bits(Color::White, file, rank, &black_rank_bits) {
            let progress = rank as usize;
            white_structure_mg += PASSED_PAWN_BONUS[progress];
            white_structure_eg += PASSED_PAWN_BONUS[progress] + ENDGAME_PASSED_PAWN_BONUS[progress];
            if supported_by_pawn(board, Color::White, square) {
                white_structure_mg += SUPPORTED_PASSED_PAWN_BONUS[progress] / 2;
                white_structure_eg += SUPPORTED_PASSED_PAWN_BONUS[progress];
            }
        }
    }
    for square in piece_bb(board, Color::Black, Piece::Pawn) {
        let file = file_index(square);
        let rank = rank_index(square);
        if is_passed_pawn_bits(Color::Black, file, rank, &white_rank_bits) {
            let progress = (7 - rank) as usize;
            black_structure_mg += PASSED_PAWN_BONUS[progress];
            black_structure_eg += PASSED_PAWN_BONUS[progress] + ENDGAME_PASSED_PAWN_BONUS[progress];
            if supported_by_pawn(board, Color::Black, square) {
                black_structure_mg += SUPPORTED_PASSED_PAWN_BONUS[progress] / 2;
                black_structure_eg += SUPPORTED_PASSED_PAWN_BONUS[progress];
            }
        }
    }

    PawnCacheEntry {
        key: 0,
        white_structure_mg,
        black_structure_mg,
        white_structure_eg,
        black_structure_eg,
        white_center_mg,
        black_center_mg,
        white_files,
        black_files,
    }
}

fn pawn_structure_from_bits(file_bits: &[u8; 8]) -> (i32, i32) {
    let mut midgame = 0;
    let mut endgame = 0;
    for file_idx in 0..8 {
        let count = file_bits[file_idx].count_ones() as i32;
        if count > 1 {
            let penalty = DOUBLED_PAWN_PENALTY * (count - 1);
            midgame -= penalty;
            endgame -= penalty * 3 / 4;
        }
        if count > 0 {
            let left = if file_idx > 0 { file_bits[file_idx - 1] } else { 0 };
            let right = if file_idx < 7 { file_bits[file_idx + 1] } else { 0 };
            if left == 0 && right == 0 {
                midgame -= ISOLATED_PAWN_PENALTY * count;
                endgame -= ISOLATED_PAWN_PENALTY * count * 3 / 4;
            }
        }
    }
    (midgame, endgame)
}

fn is_passed_pawn_bits(color: Color, file_idx: i32, rank: i32, enemy_bits: &[u8; 8]) -> bool {
    for delta in -1..=1 {
        let ef = file_idx + delta;
        if !(0..=7).contains(&ef) { continue; }
        let bits = enemy_bits[ef as usize];
        if bits == 0 { continue; }
        if color == Color::White {
            if rank < 7 {
                let mask = 0xFFu8 << (rank + 1);
                if bits & mask != 0 { return false; }
            }
        } else {
            if rank > 0 {
                let mask = (1u8 << rank) - 1;
                if bits & mask != 0 { return false; }
            }
        }
    }
    true
}

/// Bitboard-based mobility scoring — uses magic bitboard lookups instead of manual stepping
fn mobility_score(board: &Board, color: Color) -> i32 {
    let own = *board.color_combined(color);
    let occupied = *board.combined();
    let mut score = 0;

    for square in piece_bb(board, color, Piece::Knight) {
        let attacks = get_knight_moves(square) & !own;
        score += attacks.popcnt() as i32 * MOBILITY_KNIGHT;
    }
    for square in piece_bb(board, color, Piece::Bishop) {
        let attacks = get_bishop_moves(square, occupied) & !own;
        score += attacks.popcnt() as i32 * MOBILITY_BISHOP;
    }
    for square in piece_bb(board, color, Piece::Rook) {
        let attacks = get_rook_moves(square, occupied) & !own;
        score += attacks.popcnt() as i32 * MOBILITY_ROOK;
    }
    for square in piece_bb(board, color, Piece::Queen) {
        let attacks = (get_rook_moves(square, occupied) | get_bishop_moves(square, occupied)) & !own;
        score += attacks.popcnt() as i32 * MOBILITY_QUEEN;
    }

    score
}

/// Threat evaluation: bonus for attacking enemy pieces with lower-value pieces
fn threat_score(board: &Board, color: Color) -> i32 {
    let enemy = !color;
    let occupied = *board.combined();
    let mut score = 0;

    // Pawn attacks on enemy minors (knights/bishops)
    let enemy_minors = piece_bb(board, enemy, Piece::Knight) | piece_bb(board, enemy, Piece::Bishop);
    for square in piece_bb(board, color, Piece::Pawn) {
        let attacks = get_pawn_attacks(square, color, enemy_minors);
        score += attacks.popcnt() as i32 * THREAT_MINOR_BY_PAWN;
    }

    // Minor attacks on enemy rooks
    let enemy_rooks = piece_bb(board, enemy, Piece::Rook);
    for square in piece_bb(board, color, Piece::Knight) {
        let attacks = get_knight_moves(square) & enemy_rooks;
        score += attacks.popcnt() as i32 * THREAT_ROOK_BY_MINOR;
    }
    for square in piece_bb(board, color, Piece::Bishop) {
        let attacks = get_bishop_moves(square, occupied) & enemy_rooks;
        score += attacks.popcnt() as i32 * THREAT_ROOK_BY_MINOR;
    }

    // Minor/rook attacks on enemy queens
    let enemy_queens = piece_bb(board, enemy, Piece::Queen);
    for square in piece_bb(board, color, Piece::Knight) {
        let attacks = get_knight_moves(square) & enemy_queens;
        score += attacks.popcnt() as i32 * THREAT_QUEEN_BY_MINOR;
    }
    for square in piece_bb(board, color, Piece::Bishop) {
        let attacks = get_bishop_moves(square, occupied) & enemy_queens;
        score += attacks.popcnt() as i32 * THREAT_QUEEN_BY_MINOR;
    }
    for square in piece_bb(board, color, Piece::Rook) {
        let attacks = get_rook_moves(square, occupied) & enemy_queens;
        score += attacks.popcnt() as i32 * THREAT_QUEEN_BY_ROOK;
    }

    score
}

/// Passed pawn king distance bonus (endgame)
/// Bonus when our king is close to our passed pawns, penalty when enemy king is close
fn passed_pawn_king_bonus(board: &Board, color: Color) -> i32 {
    let enemy = !color;
    let our_king = king_square(board, color);
    let enemy_king = king_square(board, enemy);
    let (our_king, enemy_king) = match (our_king, enemy_king) {
        (Some(ok), Some(ek)) => (ok, ek),
        _ => return 0,
    };

    let mut bonus = 0;
    for square in piece_bb(board, color, Piece::Pawn) {
        let rank = rank_index(square);
        let file = file_index(square);
        // Quick passed pawn check: no enemy pawns on same or adjacent files ahead
        let progress = if color == Color::White { rank } else { 7 - rank };
        if progress < 3 { continue; } // Only care about advanced pawns

        let mut is_passed = true;
        for df in -1..=1i32 {
            let ef = file + df;
            if !(0..=7).contains(&ef) { continue; }
            for ep_sq in piece_bb(board, enemy, Piece::Pawn) {
                let er = rank_index(ep_sq);
                let ef2 = file_index(ep_sq);
                if ef2 == ef {
                    if color == Color::White && er > rank { is_passed = false; break; }
                    if color == Color::Black && er < rank { is_passed = false; break; }
                }
            }
            if !is_passed { break; }
        }

        if is_passed {
            let our_dist = chebyshev_distance(our_king, square);
            let enemy_dist = chebyshev_distance(enemy_king, square);
            // Bonus scales with advancement
            let weight = progress as i32;
            bonus += (enemy_dist - our_dist) * weight * 2;
        }
    }
    bonus
}

fn chebyshev_distance(a: Square, b: Square) -> i32 {
    let df = (file_index(a) - file_index(b)).abs();
    let dr = (rank_index(a) - rank_index(b)).abs();
    df.max(dr)
}

/// Center distance for mop-up eval: how far is a square from the center?
fn center_distance(square: Square) -> i32 {
    let file = file_index(square);
    let rank = rank_index(square);
    let file_dist = (file * 2 - 7).abs();  // 0-7 -> distance from center
    let rank_dist = (rank * 2 - 7).abs();
    file_dist + rank_dist  // Manhattan distance from center (0-14 range)
}

/// Mop-up evaluation: bonus for driving enemy king to corner in won endgames.
/// Only applies when we have a major material advantage (Q or R+minor vs nothing).
fn mopup_score(board: &Board, color: Color) -> i32 {
    let enemy = !color;
    // Check if we have overwhelming material advantage
    let our_queens = piece_bb(board, color, Piece::Queen).popcnt();
    let our_rooks = piece_bb(board, color, Piece::Rook).popcnt();
    let enemy_queens = piece_bb(board, enemy, Piece::Queen).popcnt();
    let enemy_rooks = piece_bb(board, enemy, Piece::Rook).popcnt();
    let enemy_minors = piece_bb(board, enemy, Piece::Knight).popcnt()
        + piece_bb(board, enemy, Piece::Bishop).popcnt();

    // Only apply mop-up when we have a queen or rook and enemy has no major pieces
    if (our_queens == 0 && our_rooks == 0) || enemy_queens > 0 || enemy_rooks > 0 {
        return 0;
    }
    // Enemy should have very little material
    if enemy_minors > 1 {
        return 0;
    }

    let Some(our_king) = king_square(board, color) else { return 0 };
    let Some(enemy_king) = king_square(board, enemy) else { return 0 };

    let mut score = 0;
    // Bonus for enemy king being far from center (pushed to edge/corner)
    score += center_distance(enemy_king) * 8;
    // Bonus for our king being close to enemy king (to help checkmate)
    score += (14 - chebyshev_distance(our_king, enemy_king)) * 4;

    score
}

/// Connected rooks: bonus when rooks can see each other (same rank/file, no pieces between)
fn connected_rooks_score(board: &Board, color: Color) -> i32 {
    let rook_bb = piece_bb(board, color, Piece::Rook);
    if rook_bb.popcnt() < 2 {
        return 0;
    }
    let first_rook = rook_bb.to_square();
    let occupied = *board.combined();
    let attacks = get_rook_moves(first_rook, occupied);
    if (attacks & rook_bb & !BitBoard::from_square(first_rook)) != BitBoard(0) {
        CONNECTED_ROOKS_BONUS
    } else {
        0
    }
}

fn rook_activity_score(
    board: &Board,
    color: Color,
    own_files: &[u8; 8],
    enemy_files: &[u8; 8],
) -> i32 {
    let mut score = 0;
    for rook in piece_bb(board, color, Piece::Rook) {
        let file = file_index(rook) as usize;
        if own_files[file] == 0 && enemy_files[file] == 0 {
            score += ROOK_OPEN_FILE_BONUS;
        } else if own_files[file] == 0 {
            score += ROOK_SEMI_OPEN_FILE_BONUS;
        }

        let rank = rank_index(rook);
        if (color == Color::White && rank == 6) || (color == Color::Black && rank == 1) {
            score += ROOK_SEVENTH_RANK_BONUS;
        }
    }

    score
}

fn knight_outpost_score(board: &Board, color: Color) -> i32 {
    let mut score = 0;
    for square in piece_bb(board, color, Piece::Knight) {
        let rank = rank_index(square);
        let file = file_index(square);
        let advanced = if color == Color::White {
            rank >= 3
        } else {
            rank <= 4
        };
        if !advanced {
            continue;
        }
        if !supported_by_pawn(board, color, square) {
            continue;
        }
        if attacked_by_enemy_pawn(board, color, square) {
            continue;
        }

        score += KNIGHT_OUTPOST_BONUS;
        if (2..=5).contains(&file) {
            score += 4;
        }
    }
    score
}

fn development_score(board: &Board, color: Color, endgame: bool) -> i32 {
    if endgame {
        return 0;
    }

    let home_minors = match color {
        Color::White => [Square::B1, Square::G1, Square::C1, Square::F1],
        Color::Black => [Square::B8, Square::G8, Square::C8, Square::F8],
    };
    let home_pieces = [Piece::Knight, Piece::Knight, Piece::Bishop, Piece::Bishop];
    let mut score = 0;
    for (square, piece) in home_minors.into_iter().zip(home_pieces) {
        if board.piece_on(square) == Some(piece) && board.color_on(square) == Some(color) {
            score -= UNDEVELOPED_MINOR_PENALTY;
        }
    }
    score
}

fn supported_by_pawn(board: &Board, color: Color, square: Square) -> bool {
    let file = file_index(square);
    let rank = rank_index(square);
    let support_rank = if color == Color::White {
        rank - 1
    } else {
        rank + 1
    };

    for delta in [-1, 1] {
        let Some(candidate) = square_from_coords(file + delta, support_rank) else {
            continue;
        };
        if board.piece_on(candidate) == Some(Piece::Pawn)
            && board.color_on(candidate) == Some(color)
        {
            return true;
        }
    }

    false
}

fn attacked_by_enemy_pawn(board: &Board, color: Color, square: Square) -> bool {
    let file = file_index(square);
    let rank = rank_index(square);
    let attack_rank = if color == Color::White {
        rank + 1
    } else {
        rank - 1
    };

    for delta in [-1, 1] {
        let Some(candidate) = square_from_coords(file + delta, attack_rank) else {
            continue;
        };
        if board.piece_on(candidate) == Some(Piece::Pawn)
            && board.color_on(candidate) == Some(!color)
        {
            return true;
        }
    }

    false
}

fn king_safety_score(
    board: &Board,
    color: Color,
    phase: i32,
    own_files: &[u8; 8],
    enemy_files: &[u8; 8],
) -> i32 {
    if phase <= 4 {
        return 0;
    }

    let Some(king_sq) = king_square(board, color) else {
        return 0;
    };

    let mut score = 0;
    let rank = rank_index(king_sq);
    let file = file_index(king_sq);

    if (color == Color::White && (king_sq == Square::G1 || king_sq == Square::C1))
        || (color == Color::Black && (king_sq == Square::G8 || king_sq == Square::C8))
    {
        score += CASTLED_KING_BONUS;
    }

    let shield_rank = if color == Color::White {
        rank + 1
    } else {
        rank - 1
    };
    if (0..=7).contains(&shield_rank) {
        for delta in -1..=1 {
            let shield_file = file + delta;
            let Some(shield_square) = square_from_coords(shield_file, shield_rank) else {
                continue;
            };
            let piece = board.piece_on(shield_square);
            let piece_color = board.color_on(shield_square);
            if piece == Some(Piece::Pawn) && piece_color == Some(color) {
                score += 8;
            } else {
                score -= 6;
            }
        }
    }

    for delta in -1..=1 {
        let king_file = file + delta;
        if !(0..=7).contains(&king_file) {
            continue;
        }

        let idx = king_file as usize;
        let mut penalty = if delta == 0 { 12 } else { 8 };
        if enemy_files[idx] == 0 {
            penalty -= 3;
        }
        if own_files[idx] == 0 {
            score -= penalty.max(4);
        }
    }

    // Bitboard-based king ring attack pressure
    let pressure = king_ring_attack_pressure(board, color, king_sq);
    if pressure > 0 {
        let enemy_has_queen = piece_bb(board, !color, Piece::Queen) != BitBoard(0);
        let multiplier = if enemy_has_queen {
            phase + 6
        } else {
            phase + 1
        };
        score -= pressure * multiplier / 12;
    }

    score
}

/// Bitboard-based king ring attack pressure — uses magic bitboard lookups
fn king_ring_attack_pressure(board: &Board, color: Color, king_sq: Square) -> i32 {
    let enemy = !color;
    let occupied = *board.combined();
    let king_ring = get_king_moves(king_sq) | BitBoard::from_square(king_sq);
    let mut pressure = 0;

    // Pawn attacks on king ring
    for square in piece_bb(board, enemy, Piece::Pawn) {
        let attacks = get_pawn_attacks(square, enemy, king_ring);
        pressure += attacks.popcnt() as i32 * 2;
    }
    // Knight attacks on king ring
    for square in piece_bb(board, enemy, Piece::Knight) {
        let attacks = get_knight_moves(square) & king_ring;
        pressure += attacks.popcnt() as i32 * 2;
    }
    // Bishop attacks on king ring
    for square in piece_bb(board, enemy, Piece::Bishop) {
        let attacks = get_bishop_moves(square, occupied) & king_ring;
        pressure += attacks.popcnt() as i32 * 2;
    }
    // Rook attacks on king ring
    for square in piece_bb(board, enemy, Piece::Rook) {
        let attacks = get_rook_moves(square, occupied) & king_ring;
        pressure += attacks.popcnt() as i32 * 3;
    }
    // Queen attacks on king ring
    for square in piece_bb(board, enemy, Piece::Queen) {
        let attacks = (get_rook_moves(square, occupied) | get_bishop_moves(square, occupied)) & king_ring;
        pressure += attacks.popcnt() as i32 * 4;
    }

    pressure
}
