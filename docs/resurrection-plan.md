# Chess Ancient Resurrection Plan

This document defines the modernization direction for Chess Ancient. It should guide implementation decisions instead of encouraging a direct translation of the current package tree. Preserve the durable ideas, replace brittle implementation details, and design new module boundaries around stable interfaces.

## 1. What Stays from the Original Project

The original project contains several ideas worth carrying forward:

- **Bitboards:** Keep the bitboard-oriented board representation for efficient move generation, attack maps, occupancy checks, and evaluation features. New code may reorganize the implementation, but bitboards should remain a core engine primitive.
- **PGN-to-position data idea:** Preserve the concept of converting game records into position-level training and evaluation data. The exact parser and storage format can change, but the pipeline from chess games to labeled positions remains valuable.
- **Board evaluator:** Keep a board evaluation layer as a first-class component. It may start as a classical heuristic evaluator, a learned model, or a hybrid, but it should be independently testable from move generation and search.
- **Possible search engine:** Retain the option to build a search engine around the evaluator. The design should leave room for alpha-beta, iterative deepening, transposition tables, move ordering, quiescence search, and later neural-guided search.

## 2. What Can Be Replaced

The resurrection should not preserve implementation choices that make the project harder to evolve:

- **UI stack:** Replace the existing UI approach if it slows development. A simple CLI, web UI, or lightweight desktop UI is acceptable as long as it consumes stable engine interfaces.
- **DL4J experiments:** Treat DL4J-based experiments as historical prototypes, not as mandatory architecture. Future ML work can use whatever ecosystem best supports iteration, reproducibility, and deployment.
- **Custom neural layers:** Avoid carrying forward custom neural-network layers unless a current experiment proves they are necessary. Prefer standard model components and documented preprocessing.
- **Ad hoc PGN parser:** Replace one-off parsing logic with a tested parser or a clearly specified ingestion module. Parsing should handle malformed input gracefully and report useful diagnostics.
- **Hard-coded paths:** Remove hard-coded local paths. All datasets, model artifacts, logs, and configuration values should come from command-line flags, config files, or environment variables.

## 3. Module Boundaries

Use these boundaries when rebuilding the project. They are conceptual boundaries first; they do not require one package, repository, or language per module.

### Engine

Responsible for chess rules and position mechanics:

- Board representation, preferably bitboard-backed.
- Legal move generation and make/unmake move operations.
- Check, checkmate, stalemate, castling, en passant, promotion, repetition, and draw-rule state.
- FEN import/export.
- UCI move parsing and formatting.

The engine must be deterministic, heavily tested, and independent of UI, training, and dataset storage concerns.

### Search

Responsible for selecting moves from engine positions:

- Search algorithms such as alpha-beta, minimax, iterative deepening, quiescence search, and later MCTS if useful.
- Move ordering, transposition tables, time management, and search diagnostics.
- Integration with one or more board evaluators.
- Optional UCI-compatible command loop.

Search should depend on the engine and evaluator interfaces, but the engine should not depend on search.

### Data Ingestion

Responsible for turning source chess data into normalized records:

- PGN ingestion and validation.
- Position extraction from complete games.
- Label generation from game result, engine analysis, human move choice, or other supervision signals.
- Dataset splitting, deduplication, filtering, and metadata capture.
- Export to stable tabular or record-oriented datasets.

Data ingestion should produce interchange formats that training and evaluation can consume without knowing about the original PGN files.

### Training

Responsible for model development and reproducibility:

- Feature extraction from normalized position records.
- Model training, validation, and evaluation.
- Experiment configuration, metrics, checkpoints, and artifact export.
- Compatibility adapters for engine/search evaluation.

Training code should be free to use the most productive ML ecosystem, provided it exports documented artifacts and preprocessing metadata.

### UI / CLI

Responsible for human and automation interfaces:

- CLI tools for perft, FEN evaluation, PGN ingestion, dataset generation, model evaluation, and self-play.
- Optional UI for playing, inspecting positions, or visualizing search.
- Configuration loading and user-facing diagnostics.

The UI/CLI layer should be thin and should call engine, search, data, and training APIs rather than embedding chess logic.

## 4. Interchange Formats

Prefer stable, documented formats between modules so each part can be replaced independently.

### FEN

Use Forsyth-Edwards Notation for complete position exchange, including side to move, castling rights, en passant target square, halfmove clock, and fullmove number. FEN is the default format for engine tests, position evaluation commands, and dataset position identity.

### UCI Move Notation

Use long algebraic UCI move notation, such as `e2e4`, `g1f3`, or `e7e8q`, for machine-facing move exchange. UCI notation should be accepted by the engine, search, CLI tools, and any bridge layer.

### PGN

Use Portable Game Notation as the source format for historical games and external game collections. PGN should remain an ingestion input rather than an internal dependency for training or search.

### Position Dataset Schema

Use a documented schema for extracted positions. A minimal row-oriented schema should include:

| Field | Description |
| --- | --- |
| `position_id` | Stable identifier, such as a hash of normalized FEN plus optional context. |
| `fen` | Full FEN string for the position. |
| `move_uci` | Optional move played from the position in UCI notation. |
| `legal_moves_uci` | Optional list of legal UCI moves, encoded as JSON or a repeated field. |
| `result` | Game result from the side-to-move perspective or another documented convention. |
| `score_cp` | Optional centipawn score from an engine or evaluator. |
| `mate_in` | Optional mate distance when known. |
| `source_game_id` | Identifier for the source PGN game or generated game. |
| `ply` | Ply index within the source game. |
| `split` | Dataset split, such as `train`, `validation`, or `test`. |
| `metadata` | Optional JSON object for provenance, parser warnings, time control, ratings, or experiment tags. |

Candidate physical formats include CSV for inspection, JSON Lines for flexible records, Parquet for larger datasets, and SQLite/DuckDB for local analysis. The selected format must preserve the schema and be documented.

### Model Artifact Format

Model artifacts should include both weights and preprocessing metadata. At minimum, an artifact should define:

- Model architecture or runtime format.
- Weights/checkpoint data.
- Feature encoding version.
- Input tensor shapes or feature columns.
- Output semantics, such as centipawn score, win/draw/loss probabilities, policy logits, or move ranking.
- Training dataset version and evaluation metrics.

Prefer portable formats such as ONNX when cross-language inference matters. Native framework formats are acceptable for prototypes if an export path is documented.

## 5. Language Options

The project can be revived through several viable language strategies. Choose based on the next milestone rather than the historical package layout.

### Java Continuation

Continue in Java if the priority is preserving the existing build, reusing current engine code, or maintaining JVM deployment.

- Best for incremental cleanup and retaining familiar project structure.
- Can use modern Java, JUnit, Maven/Gradle, and a cleaner module layout.
- ML should be isolated behind evaluator interfaces so the project is not locked to a single Java ML library.

### Python-First ML/Data Prototype

Use Python first if the priority is data ingestion, experiments, model training, and rapid iteration.

- Best for PGN processing, dataset generation, notebooks, PyTorch/JAX/TensorFlow experiments, and analysis.
- Engine functionality can begin with a trusted library for data work, then later connect to a custom engine.
- Production engine code should not be blocked by prototype training code.

### C/Rust Engine with Python Training Bridge

Use C or Rust for the engine if the priority is performance, correctness, and long-term engine quality, while keeping Python for ML.

- Best for fast move generation, perft validation, search, and embeddable engine components.
- Python can call the engine through FFI, bindings, subprocess UCI, or generated datasets.
- Requires stricter interface contracts but gives strong separation between performance-critical code and experimentation.

### Hybrid Engine Exposed Through UCI or FFI

Use a hybrid architecture when different modules benefit from different ecosystems.

- Expose the engine/search through UCI for maximum interoperability with chess tools and simple process boundaries.
- Expose lower-level APIs through FFI when training or evaluation needs high-throughput position generation.
- Keep FEN, UCI moves, and the position dataset schema as the contract between languages.
- Avoid allowing language boundaries to leak implementation details into the dataset, model, or UI layers.

## Implementation Guidance

When adding new code, start from these interfaces and formats rather than mirroring the old directories. A good resurrection milestone should state:

1. Which module it belongs to.
2. Which interchange formats it consumes and produces.
3. Which legacy ideas it preserves.
4. Which legacy implementation details it intentionally replaces.
5. How it can be tested independently.
