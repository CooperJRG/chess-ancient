use chess_resurected_engine::{Board, Color, Piece, STARTPOS_FEN};

#[test]
fn start_position_has_expected_piece_counts() {
    let board = Board::from_fen(STARTPOS_FEN).unwrap();

    assert_eq!(board.bitboard(Color::White, Piece::Pawn).count_ones(), 8);
    assert_eq!(board.bitboard(Color::Black, Piece::Pawn).count_ones(), 8);
    assert_eq!(board.occupancy(Color::White).count_ones(), 16);
    assert_eq!(board.occupancy(Color::Black).count_ones(), 16);
    assert_eq!(board.all_occupancy().count_ones(), 32);
}

#[test]
fn first_slice_generates_initial_pawn_and_knight_moves() {
    let board = Board::from_fen(STARTPOS_FEN).unwrap();
    let moves = board.pseudo_legal_pawn_and_knight_moves();

    assert_eq!(moves.len(), 20);
}

#[test]
fn black_to_move_generates_initial_pawn_and_knight_moves() {
    let board =
        Board::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1").unwrap();
    let moves = board.pseudo_legal_pawn_and_knight_moves();

    assert_eq!(moves.len(), 20);
}
