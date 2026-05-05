use chess_resurected_engine::{Board, ChessMove, Color, Piece};
use std::env;
use std::io::{self, BufRead, Write};
use std::process;

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if !args.is_empty() {
        if let Err(err) = run_cli(&args) {
            eprintln!("{err}");
            process::exit(2);
        }
        return;
    }

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut board = Board::startpos();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let mut tokens = line.splitn(2, ' ');
        let cmd = tokens.next().unwrap_or("");
        let rest = tokens.next().unwrap_or("").trim();

        match cmd {
            "uci" => {
                let mut out = stdout.lock();
                writeln!(out, "id name ChessResurected").unwrap();
                writeln!(out, "id author CooperJRG").unwrap();
                writeln!(out, "uciok").unwrap();
            }
            "isready" => {
                let mut out = stdout.lock();
                writeln!(out, "readyok").unwrap();
            }
            "ucinewgame" => {
                board = Board::startpos();
            }
            "position" => {
                board = parse_position(rest);
            }
            "go" => {
                let best = search(&mut board);
                let mut out = stdout.lock();
                if let Some(m) = best {
                    writeln!(out, "bestmove {}", m.to_uci()).unwrap();
                } else {
                    writeln!(out, "bestmove 0000").unwrap();
                }
            }
            "quit" | "exit" => break,
            _ => {}
        }
    }
}

fn run_cli(args: &[String]) -> Result<(), String> {
    match args.first().map(String::as_str) {
        Some("legal-moves") => {
            let fen = option_value(args, "--fen")?;
            let mut board = Board::from_fen(fen).map_err(|e| format!("invalid --fen: {e}"))?;
            let mut moves: Vec<String> =
                board.legal_moves().iter().map(ChessMove::to_uci).collect();
            moves.sort();
            println!("{}", moves.join(" "));
            Ok(())
        }
        Some("perft") => {
            let fen = option_value(args, "--fen")?;
            let depth: u32 = option_value(args, "--depth")?
                .parse()
                .map_err(|_| "--depth must be a non-negative integer".to_owned())?;
            let mut board = Board::from_fen(fen).map_err(|e| format!("invalid --fen: {e}"))?;
            println!("{}", board.perft(depth));
            Ok(())
        }
        Some("help") | Some("--help") | Some("-h") => {
            print_help();
            Ok(())
        }
        Some(cmd) => Err(format!("unknown command '{cmd}'\n\n{}", help_text())),
        None => Ok(()),
    }
}

fn option_value<'a>(args: &'a [String], name: &str) -> Result<&'a str, String> {
    args.windows(2)
        .find(|pair| pair[0] == name)
        .map(|pair| pair[1].as_str())
        .ok_or_else(|| format!("missing required option {name}\n\n{}", help_text()))
}

fn print_help() {
    println!("{}", help_text());
}

fn help_text() -> &'static str {
    "Usage:
  chess-resurected legal-moves --fen <FEN>
  chess-resurected perft --fen <FEN> --depth <N>

With no arguments, starts the UCI-compatible stdin loop."
}

fn parse_position(rest: &str) -> Board {
    // Split on " moves " first so FEN parsing is unambiguous
    let (pos_part, moves_part) = match rest.find(" moves ") {
        Some(idx) => (&rest[..idx], Some(&rest[idx + 7..])),
        None => (rest, None),
    };

    let mut board = if pos_part == "startpos" {
        Board::startpos()
    } else if let Some(fen) = pos_part.strip_prefix("fen ") {
        Board::from_fen(fen).unwrap_or_else(|_| Board::startpos())
    } else {
        Board::startpos()
    };

    if let Some(moves) = moves_part {
        for uci in moves.split_whitespace() {
            if let Some(m) = find_uci_move(&mut board, uci) {
                board.make_move(m);
            }
        }
    }

    board
}

fn find_uci_move(board: &mut Board, uci: &str) -> Option<ChessMove> {
    if uci.len() < 4 {
        return None;
    }
    let from = parse_sq(&uci[0..2])?;
    let to = parse_sq(&uci[2..4])?;
    let promo = uci.chars().nth(4).and_then(|c| match c {
        'q' => Some(Piece::Queen),
        'r' => Some(Piece::Rook),
        'b' => Some(Piece::Bishop),
        'n' => Some(Piece::Knight),
        _ => None,
    });

    board
        .legal_moves()
        .into_iter()
        .find(|m| m.from == from && m.to == to && m.promotion == promo)
}

fn parse_sq(s: &str) -> Option<u8> {
    let b = s.as_bytes();
    if b.len() < 2 {
        return None;
    }
    let file = b[0].checked_sub(b'a')?;
    let rank = b[1].checked_sub(b'1')?;
    if file > 7 || rank > 7 {
        return None;
    }
    Some(rank * 8 + file)
}

// ── search ────────────────────────────────────────────────────────────────────

const MATE_SCORE: i32 = 100_000;
const MAX_DEPTH: u32 = 3;

fn search(board: &mut Board) -> Option<ChessMove> {
    let moves = board.legal_moves();
    if moves.is_empty() {
        return None;
    }

    let color = board.side_to_move();
    let maximizing = color == Color::White;
    let mut best_move = moves[0];
    let mut best_score = if maximizing { i32::MIN } else { i32::MAX };

    for m in moves {
        let state = board.make_move(m);
        let score = alpha_beta(board, MAX_DEPTH - 1, i32::MIN, i32::MAX, !maximizing);
        board.unmake_move(state);

        if maximizing && score > best_score || !maximizing && score < best_score {
            best_score = score;
            best_move = m;
        }
    }

    Some(best_move)
}

fn alpha_beta(
    board: &mut Board,
    depth: u32,
    mut alpha: i32,
    mut beta: i32,
    maximizing: bool,
) -> i32 {
    if depth == 0 {
        return evaluate(board);
    }

    let moves = board.legal_moves();
    if moves.is_empty() {
        let color = board.side_to_move();
        return if board.is_in_check(color) {
            if maximizing {
                -MATE_SCORE
            } else {
                MATE_SCORE
            }
        } else {
            0 // stalemate
        };
    }

    if maximizing {
        let mut value = i32::MIN;
        for m in moves {
            let state = board.make_move(m);
            value = value.max(alpha_beta(board, depth - 1, alpha, beta, false));
            board.unmake_move(state);
            alpha = alpha.max(value);
            if value >= beta {
                break;
            }
        }
        value
    } else {
        let mut value = i32::MAX;
        for m in moves {
            let state = board.make_move(m);
            value = value.min(alpha_beta(board, depth - 1, alpha, beta, true));
            board.unmake_move(state);
            beta = beta.min(value);
            if value <= alpha {
                break;
            }
        }
        value
    }
}

fn evaluate(board: &Board) -> i32 {
    let mut score = 0i32;
    for color in [Color::White, Color::Black] {
        let sign = if color == Color::White { 1 } else { -1 };
        score += sign * piece_value(board, color, Piece::Pawn, 100);
        score += sign * piece_value(board, color, Piece::Knight, 320);
        score += sign * piece_value(board, color, Piece::Bishop, 330);
        score += sign * piece_value(board, color, Piece::Rook, 500);
        score += sign * piece_value(board, color, Piece::Queen, 900);
    }
    score
}

fn piece_value(board: &Board, color: Color, piece: Piece, value: i32) -> i32 {
    board.bitboard(color, piece).count_ones() as i32 * value
}
