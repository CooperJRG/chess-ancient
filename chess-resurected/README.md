# Chess Resurected

Chess Resurected is the clean-room revival of the original Chess Ancient project. It preserves the original project's broad architecture—bitboard chess mechanics, PGN-to-position data generation, evaluators, and eventual search—while allowing the implementation language and tools to change.

> Note: the directory name intentionally follows the requested spelling: `chess-resurected`.

## Goals

1. Build a deterministic, heavily tested chess engine core.
2. Keep data ingestion and model training separate from the engine.
3. Use stable interchange formats: FEN for positions, UCI for moves, JSON Lines/CSV/Parquet for datasets, and portable model artifacts when possible.
4. Make every module runnable from a simple CLI before adding UI complexity.

## Initial language split

- **Rust (`engine/`)**: performance-critical chess rules, bitboards, FEN parsing/serialization, move generation, perft, and eventually search/UCI.
- **Python (`python/`)**: PGN ingestion, dataset preparation, ML experiments, analysis scripts, and bridges to the engine.

This is a starting point, not a permanent constraint. Interfaces should remain stable enough that modules can be replaced later.

## Current status

The first milestone is a foundation rather than a complete engine:

- Rust engine crate with a bitboard-backed `Board` type.
- FEN parsing and serialization for complete position state.
- Basic pseudo-legal pawn and knight move generation.
- Python package skeleton with a documented position-record schema.
- Tests for FEN round-tripping, starting-position occupancy, initial pawn/knight pseudo-legal move counts, and Python record serialization.

## Repository layout

```text
chess-resurected/
├── README.md
├── docs/
│   └── architecture.md
├── engine/
│   ├── Cargo.toml
│   ├── src/
│   │   └── lib.rs
│   └── tests/
│       └── engine_smoke.rs
└── python/
    ├── pyproject.toml
    ├── chess_resurected/
    │   ├── __init__.py
    │   └── records.py
    └── tests/
        └── test_records.py
```

## Quick start

### Rust engine checks

```bash
cd chess-resurected/engine
cargo test
```

### Python data package checks

```bash
cd chess-resurected/python
python -m pytest
```

## Near-term roadmap

1. Complete legal move generation and make/unmake moves in Rust.
2. Add perft fixtures from standard chess test positions.
3. Add a CLI command for `fen`, `legal-moves`, and `perft`.
4. Add PGN ingestion in Python that emits documented position records.
5. Add a simple classical evaluator and alpha-beta search.
6. Add ML evaluator experiments only after engine correctness gates are reliable.
