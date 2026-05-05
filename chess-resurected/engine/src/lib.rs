//! Chess Resurected engine core.

use std::error::Error;
use std::fmt::{Display, Formatter};

pub const STARTPOS_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

const BISHOP_DIRS: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
const ROOK_DIRS: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
const QUEEN_DIRS: [(i8, i8); 8] = [
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Color {
    White,
    Black,
}

impl Color {
    pub fn opponent(self) -> Self {
        match self {
            Self::White => Self::Black,
            Self::Black => Self::White,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Piece {
    Pawn,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CastlingRights {
    pub white_king_side: bool,
    pub white_queen_side: bool,
    pub black_king_side: bool,
    pub black_queen_side: bool,
}

impl CastlingRights {
    pub const NONE: Self = Self {
        white_king_side: false,
        white_queen_side: false,
        black_king_side: false,
        black_queen_side: false,
    };

    fn from_fen(value: &str) -> Result<Self, FenError> {
        if value == "-" {
            return Ok(Self::NONE);
        }
        let mut rights = Self::NONE;
        for ch in value.chars() {
            match ch {
                'K' if !rights.white_king_side => rights.white_king_side = true,
                'Q' if !rights.white_queen_side => rights.white_queen_side = true,
                'k' if !rights.black_king_side => rights.black_king_side = true,
                'q' if !rights.black_queen_side => rights.black_queen_side = true,
                _ => return Err(FenError::InvalidCastlingRights(value.to_owned())),
            }
        }
        Ok(rights)
    }

    fn to_fen(self) -> String {
        let mut value = String::new();
        if self.white_king_side {
            value.push('K');
        }
        if self.white_queen_side {
            value.push('Q');
        }
        if self.black_king_side {
            value.push('k');
        }
        if self.black_queen_side {
            value.push('q');
        }
        if value.is_empty() {
            value.push('-');
        }
        value
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoveKind {
    Normal,
    DoublePawnPush,
    EnPassant,
    CastleKingSide,
    CastleQueenSide,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChessMove {
    pub from: u8,
    pub to: u8,
    pub promotion: Option<Piece>,
    pub kind: MoveKind,
}

impl ChessMove {
    pub fn to_uci(&self) -> String {
        let mut s = square_name(self.from);
        s.push_str(&square_name(self.to));
        if let Some(p) = self.promotion {
            s.push(match p {
                Piece::Queen => 'q',
                Piece::Rook => 'r',
                Piece::Bishop => 'b',
                Piece::Knight => 'n',
                _ => unreachable!(),
            });
        }
        s
    }
}

/// Saved board state for unmake_move.
#[derive(Debug, Clone)]
pub struct BoardState {
    bitboards: [u64; 12],
    side_to_move: Color,
    castling_rights: CastlingRights,
    en_passant: Option<u8>,
    halfmove_clock: u32,
    fullmove_number: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Board {
    bitboards: [u64; 12],
    side_to_move: Color,
    castling_rights: CastlingRights,
    en_passant: Option<u8>,
    halfmove_clock: u32,
    fullmove_number: u32,
}

impl Board {
    pub fn startpos() -> Self {
        Self::from_fen(STARTPOS_FEN).expect("built-in start position FEN must be valid")
    }

    pub fn from_fen(fen: &str) -> Result<Self, FenError> {
        let mut parts = fen.split_whitespace();
        let placement = parts
            .next()
            .ok_or(FenError::MissingField("piece placement"))?;
        let side = parts.next().ok_or(FenError::MissingField("side to move"))?;
        let castling = parts
            .next()
            .ok_or(FenError::MissingField("castling rights"))?;
        let en_passant = parts
            .next()
            .ok_or(FenError::MissingField("en-passant square"))?;
        let halfmove = parts
            .next()
            .ok_or(FenError::MissingField("halfmove clock"))?;
        let fullmove = parts
            .next()
            .ok_or(FenError::MissingField("fullmove number"))?;
        if parts.next().is_some() {
            return Err(FenError::TooManyFields);
        }

        let mut board = Self {
            bitboards: [0; 12],
            side_to_move: match side {
                "w" => Color::White,
                "b" => Color::Black,
                _ => return Err(FenError::InvalidSideToMove(side.to_owned())),
            },
            castling_rights: CastlingRights::from_fen(castling)?,
            en_passant: parse_square(en_passant)?,
            halfmove_clock: halfmove
                .parse()
                .map_err(|_| FenError::InvalidNumber("halfmove clock", halfmove.to_owned()))?,
            fullmove_number: fullmove
                .parse()
                .map_err(|_| FenError::InvalidNumber("fullmove number", fullmove.to_owned()))?,
        };
        board.parse_placement(placement)?;
        Ok(board)
    }

    pub fn to_fen(&self) -> String {
        format!(
            "{} {} {} {} {} {}",
            self.placement_to_fen(),
            match self.side_to_move {
                Color::White => "w",
                Color::Black => "b",
            },
            self.castling_rights.to_fen(),
            self.en_passant
                .map(square_name)
                .unwrap_or_else(|| "-".to_owned()),
            self.halfmove_clock,
            self.fullmove_number
        )
    }

    pub fn side_to_move(&self) -> Color {
        self.side_to_move
    }
    pub fn castling_rights(&self) -> CastlingRights {
        self.castling_rights
    }
    pub fn en_passant(&self) -> Option<u8> {
        self.en_passant
    }
    pub fn halfmove_clock(&self) -> u32 {
        self.halfmove_clock
    }
    pub fn fullmove_number(&self) -> u32 {
        self.fullmove_number
    }

    pub fn bitboard(&self, color: Color, piece: Piece) -> u64 {
        self.bitboards[piece_index(color, piece)]
    }

    pub fn occupancy(&self, color: Color) -> u64 {
        self.bitboards[color_offset(color)..color_offset(color) + 6]
            .iter()
            .fold(0, |acc, bb| acc | bb)
    }

    pub fn all_occupancy(&self) -> u64 {
        self.occupancy(Color::White) | self.occupancy(Color::Black)
    }

    pub fn piece_at(&self, square: u8) -> Option<(Color, Piece)> {
        let mask = 1u64 << square;
        for color in [Color::White, Color::Black] {
            for piece in ALL_PIECES {
                if self.bitboard(color, piece) & mask != 0 {
                    return Some((color, piece));
                }
            }
        }
        None
    }

    pub fn is_square_attacked(&self, sq: u8, by: Color) -> bool {
        let sq_r = sq / 8;
        let sq_f = sq % 8;
        let occupied = self.all_occupancy();

        // Pawn attacks
        match by {
            Color::White => {
                if sq_r > 0 {
                    if sq_f < 7
                        && self.bitboard(Color::White, Piece::Pawn) & (1u64 << (sq - 7)) != 0
                    {
                        return true;
                    }
                    if sq_f > 0
                        && self.bitboard(Color::White, Piece::Pawn) & (1u64 << (sq - 9)) != 0
                    {
                        return true;
                    }
                }
            }
            Color::Black => {
                if sq_r < 7 {
                    if sq_f < 7
                        && self.bitboard(Color::Black, Piece::Pawn) & (1u64 << (sq + 9)) != 0
                    {
                        return true;
                    }
                    if sq_f > 0
                        && self.bitboard(Color::Black, Piece::Pawn) & (1u64 << (sq + 7)) != 0
                    {
                        return true;
                    }
                }
            }
        }

        // Knight attacks
        if self.bitboard(by, Piece::Knight) & knight_attack_mask(sq) != 0 {
            return true;
        }

        // Bishop / Queen diagonals
        let diag = sliding_attacks(sq, occupied, &BISHOP_DIRS);
        if self.bitboard(by, Piece::Bishop) & diag != 0
            || self.bitboard(by, Piece::Queen) & diag != 0
        {
            return true;
        }

        // Rook / Queen straights
        let straight = sliding_attacks(sq, occupied, &ROOK_DIRS);
        if self.bitboard(by, Piece::Rook) & straight != 0
            || self.bitboard(by, Piece::Queen) & straight != 0
        {
            return true;
        }

        // King
        if self.bitboard(by, Piece::King) & king_attack_mask(sq) != 0 {
            return true;
        }

        false
    }

    pub fn is_in_check(&self, color: Color) -> bool {
        let king_bb = self.bitboard(color, Piece::King);
        if king_bb == 0 {
            return false;
        }
        let king_sq = king_bb.trailing_zeros() as u8;
        self.is_square_attacked(king_sq, color.opponent())
    }

    /// All pseudo-legal moves for the side to move.
    pub fn pseudo_legal_moves(&self) -> Vec<ChessMove> {
        let mut moves = Vec::new();
        self.generate_pawn_moves(&mut moves);
        self.generate_knight_moves(&mut moves);
        self.generate_sliding_moves(&mut moves, Piece::Bishop, &BISHOP_DIRS);
        self.generate_sliding_moves(&mut moves, Piece::Rook, &ROOK_DIRS);
        self.generate_sliding_moves(&mut moves, Piece::Queen, &QUEEN_DIRS);
        self.generate_king_moves(&mut moves);
        self.generate_castling_moves(&mut moves);
        moves
    }

    /// All legal moves for the side to move.
    pub fn legal_moves(&mut self) -> Vec<ChessMove> {
        let pseudo = self.pseudo_legal_moves();
        let color = self.side_to_move;
        let mut legal = Vec::new();
        for m in pseudo {
            let state = self.make_move(m);
            if !self.is_in_check(color) {
                legal.push(m);
            }
            self.unmake_move(state);
        }
        legal
    }

    /// Returns true when the side to move has no legal moves and is in check.
    pub fn is_checkmate(&mut self) -> bool {
        self.is_in_check(self.side_to_move) && self.legal_moves().is_empty()
    }

    /// Returns true when the side to move has no legal moves and is NOT in check.
    pub fn is_stalemate(&mut self) -> bool {
        !self.is_in_check(self.side_to_move) && self.legal_moves().is_empty()
    }

    /// Counts leaf nodes at the given depth (perft).
    pub fn perft(&mut self, depth: u32) -> u64 {
        if depth == 0 {
            return 1;
        }
        let moves = self.legal_moves();
        if depth == 1 {
            return moves.len() as u64;
        }
        let mut nodes = 0u64;
        for m in moves {
            let state = self.make_move(m);
            nodes += self.perft(depth - 1);
            self.unmake_move(state);
        }
        nodes
    }

    /// Make a move, returning the saved state needed to unmake it.
    pub fn make_move(&mut self, m: ChessMove) -> BoardState {
        let saved = BoardState {
            bitboards: self.bitboards,
            side_to_move: self.side_to_move,
            castling_rights: self.castling_rights,
            en_passant: self.en_passant,
            halfmove_clock: self.halfmove_clock,
            fullmove_number: self.fullmove_number,
        };

        let color = self.side_to_move;
        let opp = color.opponent();
        let from_mask = 1u64 << m.from;
        let to_mask = 1u64 << m.to;

        let moving_piece = self
            .piece_at(m.from)
            .expect("make_move: no piece at from square")
            .1;

        let is_capture = matches!(m.kind, MoveKind::EnPassant)
            || (matches!(m.kind, MoveKind::Normal) && self.piece_at(m.to).is_some());

        if moving_piece == Piece::Pawn || is_capture {
            self.halfmove_clock = 0;
        } else {
            self.halfmove_clock += 1;
        }

        self.en_passant = None;

        match m.kind {
            MoveKind::Normal => {
                self.bitboards[piece_index(color, moving_piece)] &= !from_mask;
                for p in ALL_PIECES {
                    self.bitboards[piece_index(opp, p)] &= !to_mask;
                }
                let placed = m.promotion.unwrap_or(moving_piece);
                self.bitboards[piece_index(color, placed)] |= to_mask;
            }
            MoveKind::DoublePawnPush => {
                self.bitboards[piece_index(color, Piece::Pawn)] ^= from_mask | to_mask;
                self.en_passant = Some(match color {
                    Color::White => m.from + 8,
                    Color::Black => m.from - 8,
                });
            }
            MoveKind::EnPassant => {
                self.bitboards[piece_index(color, Piece::Pawn)] ^= from_mask | to_mask;
                let captured_sq = match color {
                    Color::White => m.to - 8,
                    Color::Black => m.to + 8,
                };
                self.bitboards[piece_index(opp, Piece::Pawn)] &= !(1u64 << captured_sq);
            }
            MoveKind::CastleKingSide => {
                self.bitboards[piece_index(color, Piece::King)] ^= from_mask | to_mask;
                let (rf, rt) = match color {
                    Color::White => (7u8, 5u8),
                    Color::Black => (63u8, 61u8),
                };
                self.bitboards[piece_index(color, Piece::Rook)] ^= (1u64 << rf) | (1u64 << rt);
            }
            MoveKind::CastleQueenSide => {
                self.bitboards[piece_index(color, Piece::King)] ^= from_mask | to_mask;
                let (rf, rt) = match color {
                    Color::White => (0u8, 3u8),
                    Color::Black => (56u8, 59u8),
                };
                self.bitboards[piece_index(color, Piece::Rook)] ^= (1u64 << rf) | (1u64 << rt);
            }
        }

        self.update_castling_rights(m.from, m.to);

        if color == Color::Black {
            self.fullmove_number += 1;
        }
        self.side_to_move = opp;

        saved
    }

    /// Restore the board to the state before a move was made.
    pub fn unmake_move(&mut self, state: BoardState) {
        self.bitboards = state.bitboards;
        self.side_to_move = state.side_to_move;
        self.castling_rights = state.castling_rights;
        self.en_passant = state.en_passant;
        self.halfmove_clock = state.halfmove_clock;
        self.fullmove_number = state.fullmove_number;
    }

    /// Backward-compatible alias retained for the first-slice smoke tests.
    pub fn pseudo_legal_pawn_and_knight_moves(&self) -> Vec<ChessMove> {
        let mut moves = Vec::new();
        self.generate_pawn_moves(&mut moves);
        self.generate_knight_moves(&mut moves);
        moves
    }

    // ── private move generators ────────────────────────────────────────────

    fn generate_pawn_moves(&self, moves: &mut Vec<ChessMove>) {
        let color = self.side_to_move;
        let own = self.occupancy(color);
        let enemy = self.occupancy(color.opponent());
        let occupied = own | enemy;
        let pawns = self.bitboard(color, Piece::Pawn);

        for from in squares(pawns) {
            let rank = from / 8;
            let file = from % 8;
            match color {
                Color::White => {
                    let one = from + 8;
                    if one < 64 && occupied & (1u64 << one) == 0 {
                        push_pawn_move(moves, from, one, Color::White);
                        if rank == 1 {
                            let two = from + 16;
                            if two < 64 && occupied & (1u64 << two) == 0 {
                                moves.push(ChessMove {
                                    from,
                                    to: two,
                                    promotion: None,
                                    kind: MoveKind::DoublePawnPush,
                                });
                            }
                        }
                    }
                    if file > 0 {
                        let cap = from + 7;
                        if enemy & (1u64 << cap) != 0 {
                            push_pawn_move(moves, from, cap, Color::White);
                        }
                        if self.en_passant == Some(cap) && rank == 4 {
                            moves.push(ChessMove {
                                from,
                                to: cap,
                                promotion: None,
                                kind: MoveKind::EnPassant,
                            });
                        }
                    }
                    if file < 7 {
                        let cap = from + 9;
                        if enemy & (1u64 << cap) != 0 {
                            push_pawn_move(moves, from, cap, Color::White);
                        }
                        if self.en_passant == Some(cap) && rank == 4 {
                            moves.push(ChessMove {
                                from,
                                to: cap,
                                promotion: None,
                                kind: MoveKind::EnPassant,
                            });
                        }
                    }
                }
                Color::Black => {
                    if from >= 8 {
                        let one = from - 8;
                        if occupied & (1u64 << one) == 0 {
                            push_pawn_move(moves, from, one, Color::Black);
                            if rank == 6 {
                                let two = from - 16;
                                if occupied & (1u64 << two) == 0 {
                                    moves.push(ChessMove {
                                        from,
                                        to: two,
                                        promotion: None,
                                        kind: MoveKind::DoublePawnPush,
                                    });
                                }
                            }
                        }
                    }
                    if file > 0 && from >= 9 {
                        let cap = from - 9;
                        if enemy & (1u64 << cap) != 0 {
                            push_pawn_move(moves, from, cap, Color::Black);
                        }
                        if self.en_passant == Some(cap) && rank == 3 {
                            moves.push(ChessMove {
                                from,
                                to: cap,
                                promotion: None,
                                kind: MoveKind::EnPassant,
                            });
                        }
                    }
                    if file < 7 && from >= 7 {
                        let cap = from - 7;
                        if enemy & (1u64 << cap) != 0 {
                            push_pawn_move(moves, from, cap, Color::Black);
                        }
                        if self.en_passant == Some(cap) && rank == 3 {
                            moves.push(ChessMove {
                                from,
                                to: cap,
                                promotion: None,
                                kind: MoveKind::EnPassant,
                            });
                        }
                    }
                }
            }
        }
    }

    fn generate_knight_moves(&self, moves: &mut Vec<ChessMove>) {
        let color = self.side_to_move;
        let own = self.occupancy(color);
        for from in squares(self.bitboard(color, Piece::Knight)) {
            for to in squares(knight_attack_mask(from) & !own) {
                moves.push(ChessMove {
                    from,
                    to,
                    promotion: None,
                    kind: MoveKind::Normal,
                });
            }
        }
    }

    fn generate_sliding_moves(&self, moves: &mut Vec<ChessMove>, piece: Piece, dirs: &[(i8, i8)]) {
        let color = self.side_to_move;
        let own = self.occupancy(color);
        let occupied = self.all_occupancy();
        for from in squares(self.bitboard(color, piece)) {
            for to in squares(sliding_attacks(from, occupied, dirs) & !own) {
                moves.push(ChessMove {
                    from,
                    to,
                    promotion: None,
                    kind: MoveKind::Normal,
                });
            }
        }
    }

    fn generate_king_moves(&self, moves: &mut Vec<ChessMove>) {
        let color = self.side_to_move;
        let own = self.occupancy(color);
        for from in squares(self.bitboard(color, Piece::King)) {
            for to in squares(king_attack_mask(from) & !own) {
                moves.push(ChessMove {
                    from,
                    to,
                    promotion: None,
                    kind: MoveKind::Normal,
                });
            }
        }
    }

    fn generate_castling_moves(&self, moves: &mut Vec<ChessMove>) {
        let color = self.side_to_move;
        let opp = color.opponent();
        let occupied = self.all_occupancy();

        match color {
            Color::White => {
                // King-side: f1 and g1 must be empty, e1/f1/g1 not attacked
                if self.castling_rights.white_king_side
                    && occupied & 0x60 == 0
                    && !self.is_square_attacked(4, opp)
                    && !self.is_square_attacked(5, opp)
                    && !self.is_square_attacked(6, opp)
                {
                    moves.push(ChessMove {
                        from: 4,
                        to: 6,
                        promotion: None,
                        kind: MoveKind::CastleKingSide,
                    });
                }
                // Queen-side: b1/c1/d1 empty, e1/d1/c1 not attacked
                if self.castling_rights.white_queen_side
                    && occupied & 0x0E == 0
                    && !self.is_square_attacked(4, opp)
                    && !self.is_square_attacked(3, opp)
                    && !self.is_square_attacked(2, opp)
                {
                    moves.push(ChessMove {
                        from: 4,
                        to: 2,
                        promotion: None,
                        kind: MoveKind::CastleQueenSide,
                    });
                }
            }
            Color::Black => {
                // King-side: f8 and g8 must be empty
                if self.castling_rights.black_king_side
                    && occupied & (0x60u64 << 56) == 0
                    && !self.is_square_attacked(60, opp)
                    && !self.is_square_attacked(61, opp)
                    && !self.is_square_attacked(62, opp)
                {
                    moves.push(ChessMove {
                        from: 60,
                        to: 62,
                        promotion: None,
                        kind: MoveKind::CastleKingSide,
                    });
                }
                // Queen-side: b8/c8/d8 empty
                if self.castling_rights.black_queen_side
                    && occupied & (0x0Eu64 << 56) == 0
                    && !self.is_square_attacked(60, opp)
                    && !self.is_square_attacked(59, opp)
                    && !self.is_square_attacked(58, opp)
                {
                    moves.push(ChessMove {
                        from: 60,
                        to: 58,
                        promotion: None,
                        kind: MoveKind::CastleQueenSide,
                    });
                }
            }
        }
    }

    fn update_castling_rights(&mut self, from: u8, to: u8) {
        if from == 4 {
            // white king start
            self.castling_rights.white_king_side = false;
            self.castling_rights.white_queen_side = false;
        }
        if from == 60 {
            // black king start
            self.castling_rights.black_king_side = false;
            self.castling_rights.black_queen_side = false;
        }
        if from == 7 || to == 7 {
            self.castling_rights.white_king_side = false;
        }
        if from == 0 || to == 0 {
            self.castling_rights.white_queen_side = false;
        }
        if from == 63 || to == 63 {
            self.castling_rights.black_king_side = false;
        }
        if from == 56 || to == 56 {
            self.castling_rights.black_queen_side = false;
        }
    }

    fn parse_placement(&mut self, placement: &str) -> Result<(), FenError> {
        let ranks: Vec<&str> = placement.split('/').collect();
        if ranks.len() != 8 {
            return Err(FenError::InvalidPlacement(placement.to_owned()));
        }
        for (rank_from_top, rank_text) in ranks.iter().enumerate() {
            let rank = 7u8 - rank_from_top as u8;
            let mut file = 0u8;
            for ch in rank_text.chars() {
                if let Some(empty) = ch.to_digit(10) {
                    if empty == 0 || empty > 8 {
                        return Err(FenError::InvalidPlacement(placement.to_owned()));
                    }
                    file += empty as u8;
                    continue;
                }
                let Some((color, piece)) = piece_from_fen(ch) else {
                    return Err(FenError::InvalidPiece(ch));
                };
                if file >= 8 {
                    return Err(FenError::InvalidPlacement(placement.to_owned()));
                }
                let square = rank * 8 + file;
                self.bitboards[piece_index(color, piece)] |= 1u64 << square;
                file += 1;
            }
            if file != 8 {
                return Err(FenError::InvalidPlacement(placement.to_owned()));
            }
        }
        Ok(())
    }

    fn placement_to_fen(&self) -> String {
        let mut ranks = Vec::with_capacity(8);
        for rank in (0u8..8).rev() {
            let mut text = String::new();
            let mut empty = 0u8;
            for file in 0u8..8 {
                let square = rank * 8 + file;
                if let Some((color, piece)) = self.piece_at(square) {
                    if empty > 0 {
                        text.push(char::from_digit(empty.into(), 10).unwrap());
                        empty = 0;
                    }
                    text.push(piece_to_fen(color, piece));
                } else {
                    empty += 1;
                }
            }
            if empty > 0 {
                text.push(char::from_digit(empty.into(), 10).unwrap());
            }
            ranks.push(text);
        }
        ranks.join("/")
    }
}

// ── errors ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FenError {
    MissingField(&'static str),
    TooManyFields,
    InvalidPlacement(String),
    InvalidPiece(char),
    InvalidSideToMove(String),
    InvalidCastlingRights(String),
    InvalidSquare(String),
    InvalidNumber(&'static str, String),
}

impl Display for FenError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingField(field) => write!(f, "missing FEN field: {field}"),
            Self::TooManyFields => write!(f, "FEN has too many fields"),
            Self::InvalidPlacement(v) => write!(f, "invalid FEN placement: {v}"),
            Self::InvalidPiece(v) => write!(f, "invalid FEN piece: {v}"),
            Self::InvalidSideToMove(v) => write!(f, "invalid side to move: {v}"),
            Self::InvalidCastlingRights(v) => write!(f, "invalid castling rights: {v}"),
            Self::InvalidSquare(v) => write!(f, "invalid square: {v}"),
            Self::InvalidNumber(field, v) => write!(f, "invalid {field}: {v}"),
        }
    }
}

impl Error for FenError {}

// ── helpers ───────────────────────────────────────────────────────────────────

const ALL_PIECES: [Piece; 6] = [
    Piece::Pawn,
    Piece::Knight,
    Piece::Bishop,
    Piece::Rook,
    Piece::Queen,
    Piece::King,
];

fn color_offset(color: Color) -> usize {
    match color {
        Color::White => 0,
        Color::Black => 6,
    }
}

fn piece_index(color: Color, piece: Piece) -> usize {
    color_offset(color)
        + match piece {
            Piece::Pawn => 0,
            Piece::Knight => 1,
            Piece::Bishop => 2,
            Piece::Rook => 3,
            Piece::Queen => 4,
            Piece::King => 5,
        }
}

fn piece_from_fen(ch: char) -> Option<(Color, Piece)> {
    let color = if ch.is_ascii_uppercase() {
        Color::White
    } else {
        Color::Black
    };
    let piece = match ch.to_ascii_lowercase() {
        'p' => Piece::Pawn,
        'n' => Piece::Knight,
        'b' => Piece::Bishop,
        'r' => Piece::Rook,
        'q' => Piece::Queen,
        'k' => Piece::King,
        _ => return None,
    };
    Some((color, piece))
}

fn piece_to_fen(color: Color, piece: Piece) -> char {
    let ch = match piece {
        Piece::Pawn => 'p',
        Piece::Knight => 'n',
        Piece::Bishop => 'b',
        Piece::Rook => 'r',
        Piece::Queen => 'q',
        Piece::King => 'k',
    };
    match color {
        Color::White => ch.to_ascii_uppercase(),
        Color::Black => ch,
    }
}

fn parse_square(value: &str) -> Result<Option<u8>, FenError> {
    if value == "-" {
        return Ok(None);
    }
    let bytes = value.as_bytes();
    if bytes.len() != 2 || !(b'a'..=b'h').contains(&bytes[0]) || !(b'1'..=b'8').contains(&bytes[1])
    {
        return Err(FenError::InvalidSquare(value.to_owned()));
    }
    Ok(Some((bytes[1] - b'1') * 8 + (bytes[0] - b'a')))
}

pub fn square_name(square: u8) -> String {
    let file = square % 8;
    let rank = square / 8;
    format!("{}{}", (b'a' + file) as char, (b'1' + rank) as char)
}

fn squares(mut bb: u64) -> impl Iterator<Item = u8> {
    std::iter::from_fn(move || {
        if bb == 0 {
            return None;
        }
        let sq = bb.trailing_zeros() as u8;
        bb &= bb - 1;
        Some(sq)
    })
}

fn push_pawn_move(moves: &mut Vec<ChessMove>, from: u8, to: u8, color: Color) {
    let promo_rank = match color {
        Color::White => 7,
        Color::Black => 0,
    };
    if to / 8 == promo_rank {
        for p in [Piece::Queen, Piece::Rook, Piece::Bishop, Piece::Knight] {
            moves.push(ChessMove {
                from,
                to,
                promotion: Some(p),
                kind: MoveKind::Normal,
            });
        }
    } else {
        moves.push(ChessMove {
            from,
            to,
            promotion: None,
            kind: MoveKind::Normal,
        });
    }
}

fn knight_attack_mask(sq: u8) -> u64 {
    let rank = (sq / 8) as i8;
    let file = (sq % 8) as i8;
    let mut mask = 0u64;
    for (dr, df) in [
        (-2, -1),
        (-2, 1),
        (-1, -2),
        (-1, 2),
        (1, -2),
        (1, 2),
        (2, -1),
        (2, 1),
    ] {
        let nr = rank + dr;
        let nf = file + df;
        if (0..8).contains(&nr) && (0..8).contains(&nf) {
            mask |= 1u64 << (nr * 8 + nf);
        }
    }
    mask
}

fn king_attack_mask(sq: u8) -> u64 {
    let rank = (sq / 8) as i8;
    let file = (sq % 8) as i8;
    let mut mask = 0u64;
    for dr in -1i8..=1 {
        for df in -1i8..=1 {
            if dr == 0 && df == 0 {
                continue;
            }
            let nr = rank + dr;
            let nf = file + df;
            if (0..8).contains(&nr) && (0..8).contains(&nf) {
                mask |= 1u64 << (nr * 8 + nf);
            }
        }
    }
    mask
}

fn sliding_attacks(sq: u8, occupied: u64, dirs: &[(i8, i8)]) -> u64 {
    let mut attacks = 0u64;
    let rank = (sq / 8) as i8;
    let file = (sq % 8) as i8;
    for &(dr, df) in dirs {
        let mut r = rank + dr;
        let mut f = file + df;
        while (0..8).contains(&r) && (0..8).contains(&f) {
            let target = (r * 8 + f) as u8;
            attacks |= 1u64 << target;
            if occupied & (1u64 << target) != 0 {
                break;
            }
            r += dr;
            f += df;
        }
    }
    attacks
}

// ── unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_position_round_trips() {
        let board = Board::startpos();
        assert_eq!(board.to_fen(), STARTPOS_FEN);
    }

    #[test]
    fn parses_en_passant_square() {
        let board = Board::from_fen("8/8/8/3pP3/8/8/8/8 w - d6 0 2").unwrap();
        assert_eq!(board.en_passant(), Some(43));
        assert_eq!(board.to_fen(), "8/8/8/3pP3/8/8/8/8 w - d6 0 2");
    }

    #[test]
    fn rejects_bad_rank_width() {
        assert!(matches!(
            Board::from_fen("9/8/8/8/8/8/8/8 w - - 0 1"),
            Err(FenError::InvalidPlacement(_))
        ));
    }

    #[test]
    fn start_position_has_twenty_legal_moves() {
        let mut board = Board::startpos();
        assert_eq!(board.legal_moves().len(), 20);
    }

    #[test]
    fn make_unmake_preserves_fen() {
        let mut board = Board::startpos();
        let original = board.to_fen();
        let moves = board.pseudo_legal_moves();
        let m = moves[0];
        let state = board.make_move(m);
        board.unmake_move(state);
        assert_eq!(board.to_fen(), original);
    }
}
