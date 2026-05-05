use std::process::Command;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_chess-resurected")
}

fn run(args: &[&str]) -> String {
    let output = Command::new(bin())
        .args(args)
        .output()
        .expect("engine CLI should run");
    assert!(
        output.status.success(),
        "CLI failed with status {:?}\nstderr: {}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("stdout should be utf-8")
}

#[test]
fn legal_moves_startpos_lists_twenty_moves() {
    let out = run(&[
        "legal-moves",
        "--fen",
        chess_resurected_engine::STARTPOS_FEN,
    ]);
    let moves: Vec<&str> = out.split_whitespace().collect();

    assert_eq!(moves.len(), 20);
    assert!(moves.contains(&"e2e4"));
    assert!(moves.contains(&"g1f3"));
}

#[test]
fn legal_moves_include_castling() {
    let out = run(&[
        "legal-moves",
        "--fen",
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    ]);

    assert!(out.split_whitespace().any(|m| m == "e1g1"));
    assert!(out.split_whitespace().any(|m| m == "e1c1"));
}

#[test]
fn legal_moves_include_promotions() {
    let out = run(&["legal-moves", "--fen", "4k3/P7/8/8/8/8/8/4K3 w - - 0 1"]);
    let moves: Vec<&str> = out.split_whitespace().collect();

    assert!(moves.contains(&"a7a8q"));
    assert!(moves.contains(&"a7a8r"));
    assert!(moves.contains(&"a7a8b"));
    assert!(moves.contains(&"a7a8n"));
}

#[test]
fn legal_moves_include_en_passant() {
    let out = run(&["legal-moves", "--fen", "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2"]);

    assert!(out.split_whitespace().any(|m| m == "e5d6"));
}

#[test]
fn perft_cli_reports_nodes() {
    let out = run(&[
        "perft",
        "--fen",
        chess_resurected_engine::STARTPOS_FEN,
        "--depth",
        "2",
    ]);

    assert_eq!(out.trim(), "400");
}
