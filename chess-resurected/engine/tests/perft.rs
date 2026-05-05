use chess_resurected_engine::Board;

fn perft(fen: &str, depth: u32) -> u64 {
    Board::from_fen(fen).unwrap().perft(depth)
}

// Position 1: starting position
// https://www.chessprogramming.org/Perft_Results
#[test]
fn perft_startpos_d1() { assert_eq!(perft(chess_resurected_engine::STARTPOS_FEN, 1), 20); }

#[test]
fn perft_startpos_d2() { assert_eq!(perft(chess_resurected_engine::STARTPOS_FEN, 2), 400); }

#[test]
fn perft_startpos_d3() { assert_eq!(perft(chess_resurected_engine::STARTPOS_FEN, 3), 8902); }

#[test]
fn perft_startpos_d4() { assert_eq!(perft(chess_resurected_engine::STARTPOS_FEN, 4), 197281); }

// Position 2: kiwipete — exercises castling, en passant, promotions
const KIWIPETE: &str =
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

#[test]
fn perft_kiwipete_d1() { assert_eq!(perft(KIWIPETE, 1), 48); }

#[test]
fn perft_kiwipete_d2() { assert_eq!(perft(KIWIPETE, 2), 2039); }

#[test]
fn perft_kiwipete_d3() { assert_eq!(perft(KIWIPETE, 3), 97862); }

#[test]
fn perft_kiwipete_d4() { assert_eq!(perft(KIWIPETE, 4), 4085603); }

// Position 3: exercises en passant and discovered check
// https://www.chessprogramming.org/Perft_Results
const POS3: &str = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";

#[test]
fn perft_pos3_d1() { assert_eq!(perft(POS3, 1), 14); }

#[test]
fn perft_pos3_d2() { assert_eq!(perft(POS3, 2), 191); }

#[test]
fn perft_pos3_d3() { assert_eq!(perft(POS3, 3), 2812); }

#[test]
fn perft_pos3_d4() { assert_eq!(perft(POS3, 4), 43238); }
