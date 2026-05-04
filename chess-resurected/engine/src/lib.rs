//! Chess Resurected engine core.
//!
//! This crate is intentionally small for the first resurrection slice: it
//! defines a bitboard-backed board, full FEN parsing/serialization, and a
//! starter pseudo-legal move generator for pawns and knights. Legal move
//! generation, make/unmake, perft, and search should grow from this foundation.

use std::error::Error;
use std::fmt::{Display, Formatter};

pub const STARTPOS_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

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
pub struct ChessMove {
    pub from: u8,
    pub to: u8,
    pub promotion: Option<Piece>,
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

    pub fn bitboard(&self, color: Color, piece: Piece) -> u64 {
        self.bitboards[piece_index(color, piece)]
    }

    pub fn occupancy(&self, color: Color) -> u64 {
        self.bitboards[color_offset(color)..color_offset(color) + 6]
            .iter()
            .fold(0, |acc, bitboard| acc | bitboard)
    }

    pub fn all_occupancy(&self) -> u64 {
        self.occupancy(Color::White) | self.occupancy(Color::Black)
    }

    pub fn piece_at(&self, square: u8) -> Option<(Color, Piece)> {
        let mask = 1u64 << square;
        for color in [Color::White, Color::Black] {
            for piece in [
                Piece::Pawn,
                Piece::Knight,
                Piece::Bishop,
                Piece::Rook,
                Piece::Queen,
                Piece::King,
            ] {
                if self.bitboard(color, piece) & mask != 0 {
                    return Some((color, piece));
                }
            }
        }
        None
    }

    /// Returns starter pseudo-legal moves for pawns and knights only.
    ///
    /// This is deliberately named pseudo-legal because it does not yet filter
    /// self-check and does not generate sliding, king, castling, or en-passant
    /// moves. It exists to begin executable migration work while keeping future
    /// correctness gates explicit.
    pub fn pseudo_legal_pawn_and_knight_moves(&self) -> Vec<ChessMove> {
        let mut moves = Vec::new();
        self.generate_pawn_moves(&mut moves);
        self.generate_knight_moves(&mut moves);
        moves
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
                        text.push(char::from_digit(empty.into(), 10).expect("empty run is 1..8"));
                        empty = 0;
                    }
                    text.push(piece_to_fen(color, piece));
                } else {
                    empty += 1;
                }
            }
            if empty > 0 {
                text.push(char::from_digit(empty.into(), 10).expect("empty run is 1..8"));
            }
            ranks.push(text);
        }
        ranks.join("/")
    }

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
                            if occupied & (1u64 << two) == 0 {
                                moves.push(ChessMove {
                                    from,
                                    to: two,
                                    promotion: None,
                                });
                            }
                        }
                    }
                    if file > 0 {
                        let to = from + 7;
                        if enemy & (1u64 << to) != 0 {
                            push_pawn_move(moves, from, to, Color::White);
                        }
                    }
                    if file < 7 {
                        let to = from + 9;
                        if enemy & (1u64 << to) != 0 {
                            push_pawn_move(moves, from, to, Color::White);
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
                                    });
                                }
                            }
                        }
                    }
                    if file > 0 && from >= 9 {
                        let to = from - 9;
                        if enemy & (1u64 << to) != 0 {
                            push_pawn_move(moves, from, to, Color::Black);
                        }
                    }
                    if file < 7 && from >= 7 {
                        let to = from - 7;
                        if enemy & (1u64 << to) != 0 {
                            push_pawn_move(moves, from, to, Color::Black);
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
            let rank = (from / 8) as i8;
            let file = (from % 8) as i8;
            for (df, dr) in [
                (-2, -1),
                (-2, 1),
                (-1, -2),
                (-1, 2),
                (1, -2),
                (1, 2),
                (2, -1),
                (2, 1),
            ] {
                let next_file = file + df;
                let next_rank = rank + dr;
                if (0..8).contains(&next_file) && (0..8).contains(&next_rank) {
                    let to = (next_rank * 8 + next_file) as u8;
                    if own & (1u64 << to) == 0 {
                        moves.push(ChessMove {
                            from,
                            to,
                            promotion: None,
                        });
                    }
                }
            }
        }
    }
}

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
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingField(field) => write!(formatter, "missing FEN field: {field}"),
            Self::TooManyFields => write!(formatter, "FEN has too many fields"),
            Self::InvalidPlacement(value) => write!(formatter, "invalid FEN placement: {value}"),
            Self::InvalidPiece(value) => write!(formatter, "invalid FEN piece: {value}"),
            Self::InvalidSideToMove(value) => write!(formatter, "invalid side to move: {value}"),
            Self::InvalidCastlingRights(value) => {
                write!(formatter, "invalid castling rights: {value}")
            }
            Self::InvalidSquare(value) => write!(formatter, "invalid square: {value}"),
            Self::InvalidNumber(field, value) => write!(formatter, "invalid {field}: {value}"),
        }
    }
}

impl Error for FenError {}

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
    let file = bytes[0] - b'a';
    let rank = bytes[1] - b'1';
    Ok(Some(rank * 8 + file))
}

fn square_name(square: u8) -> String {
    let file = square % 8;
    let rank = square / 8;
    format!("{}{}", (b'a' + file) as char, (b'1' + rank) as char)
}

fn squares(mut bitboard: u64) -> impl Iterator<Item = u8> {
    std::iter::from_fn(move || {
        if bitboard == 0 {
            return None;
        }
        let square = bitboard.trailing_zeros() as u8;
        bitboard &= bitboard - 1;
        Some(square)
    })
}

fn push_pawn_move(moves: &mut Vec<ChessMove>, from: u8, to: u8, color: Color) {
    let promotion_rank = match color {
        Color::White => 7,
        Color::Black => 0,
    };
    if to / 8 == promotion_rank {
        for promotion in [Piece::Queen, Piece::Rook, Piece::Bishop, Piece::Knight] {
            moves.push(ChessMove {
                from,
                to,
                promotion: Some(promotion),
            });
        }
    } else {
        moves.push(ChessMove {
            from,
            to,
            promotion: None,
        });
    }
}

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
}
