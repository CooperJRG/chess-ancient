# Chess Resurected Architecture

This document translates the root resurrection plan into the first implementation slice.

## Preserved ideas

- **Bitboards remain the engine primitive.** The Rust engine stores one 64-bit bitboard per piece/color pair.
- **FEN and UCI are the engine boundary formats.** FEN is used for position state, and UCI move strings will be used for machine-facing move exchange.
- **PGN becomes an ingestion input, not a core engine dependency.** Python will parse games and emit normalized position records.
- **Evaluation is modular.** Classical and learned evaluators should implement the same conceptual contract: position in, score/policy out.

## New module boundaries

### `engine/` Rust crate

Responsibilities:

- FEN parsing and serialization.
- Board state, bitboards, side to move, castling rights, en-passant square, and move counters.
- Legal move generation and make/unmake operations.
- Perft, attack detection, search, and UCI loop in later milestones.

Non-responsibilities:

- PGN file parsing.
- Model training.
- Dataset storage.
- UI rendering.

### `python/` package

Responsibilities:

- Position record schema and validation helpers.
- PGN ingestion and dataset export in later milestones.
- ML feature extraction, training, and model artifact metadata in later milestones.
- Optional subprocess/FFI bridge to the Rust engine when high-throughput validation is needed.

Non-responsibilities:

- Owning chess rules correctness.
- Reimplementing the production move generator.

## Interchange contracts

### FEN

Rust engine APIs accept and emit full FEN strings with:

1. piece placement,
2. side to move,
3. castling rights,
4. en-passant target,
5. halfmove clock,
6. fullmove number.

### UCI moves

Move strings should use long UCI notation such as `e2e4`, `g1f3`, or `e7e8q`. The first Rust slice stores moves as source square, destination square, and optional promotion piece; string parsing/formatting is the next step.

### Position records

Python position records are JSON-serializable objects with these stable fields:

- `position_id`
- `fen`
- `move_uci`
- `legal_moves_uci`
- `result`
- `score_cp`
- `mate_in`
- `source_game_id`
- `ply`
- `split`
- `metadata`

## Correctness policy

The engine must pass correctness gates before speed, search strength, or ML work receives priority. Minimum gates:

1. FEN parse/serialize round-trips.
2. Perft counts for standard positions.
3. Special moves: castling, en passant, promotion.
4. Check, checkmate, stalemate, and draw state.
5. Deterministic hashes and reproducible test fixtures.

## Language policy

Rust and Python are currently used because they fit different parts of the revival:

- Rust is a strong fit for a fast, safe, testable engine.
- Python is a strong fit for data ingestion and ML iteration.

The language boundary should not leak into data formats. FEN, UCI, and documented records are the long-lived contracts.
