use chess_resurected_engine::{Board, ChessMove, Color, Piece};
use std::io::{self, BufRead, Write};

fn main() {
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

    board.legal_moves().into_iter().find(|m| {
        m.from == from && m.to == to && m.promotion == promo
    })
}

fn parse_sq(s: &str) -> Option<u8> {
    let b = s.as_bytes();
    if b.len() < 2 { return None; }
    let file = b[0].checked_sub(b'a')?;
    let rank = b[1].checked_sub(b'1')?;
    if file > 7 || rank > 7 { return None; }
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

fn alpha_beta(board: &mut Board, depth: u32, mut alpha: i32, mut beta: i32, maximizing: bool) -> i32 {
    if depth == 0 {
        return evaluate(board);
    }

    let moves = board.legal_moves();
    if moves.is_empty() {
        let color = board.side_to_move();
        return if board.is_in_check(color) {
            if maximizing { -MATE_SCORE } else { MATE_SCORE }
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
            if value >= beta { break; }
        }
        value
    } else {
        let mut value = i32::MAX;
        for m in moves {
            let state = board.make_move(m);
            value = value.min(alpha_beta(board, depth - 1, alpha, beta, true));
            board.unmake_move(state);
            beta = beta.min(value);
            if value <= alpha { break; }
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
