# Chess-Resurected Model Learning Plan (Effective + Fast)

## What exists today

From reviewing `chess-resurected/`, the project already has a healthy split:

- Rust engine core with deterministic board state + FEN and move primitives.
- Python package with a stable position record schema.
- Architecture docs that prioritize correctness gates before ML.

This is a strong foundation for *fast learning* because model quality in chess is mostly bottlenecked by data correctness and fast labeling loops.

## Main gaps blocking fast model learning

1. **No ingestion pipeline yet**: there is no PGN -> `PositionRecord` builder.
2. **No legal-move oracle in Python flow**: records can carry `legal_moves_uci`, but generation/validation is not yet wired.
3. **No training-ready dataset lifecycle**: no split strategy, dedup, filtering, or artifact versioning process.
4. **No baseline trainer/evaluator loop**: no fast benchmark loop to quantify improvements.
5. **No engine-assisted labeling loop**: no cp/mate annotations pipeline for supervised value learning.

## High-level strategy

Use a staged plan where each stage is independently useful and testable:

1. **Correctness first, then throughput** (matches current architecture policy).
2. **Start with supervised policy/value from real games** for fastest initial signal.
3. **Add engine-labeled targets incrementally** to improve quality.
4. **Only then add self-play** once the engine and evaluator are stable.

---

## 30-day implementation plan

## Phase 1 (Days 1-5): Data ingestion and schema hardening

### Deliverables

- `python/chess_resurected/pgn_ingest.py`
  - Parse PGN games.
  - Emit one `PositionRecord` per ply.
  - Populate: `position_id`, `fen`, `move_uci`, `result`, `source_game_id`, `ply`, `split`, `metadata`.
- `python/chess_resurected/validation.py`
  - Validate FEN parsability and UCI move format.
- CLI entrypoint for batch ingestion.

### Fast-learning impact

You cannot train effectively without high-volume, normalized records. This unlocks the first training set quickly.

### Quality gates

- Unit tests for edge PGNs (promotions, castling, en passant, resignations).
- Deterministic `position_id` from `(source_game_id, ply, fen, move_uci)` hash.

## Phase 2 (Days 6-10): Legal move and dedup filters

### Deliverables

- Add a bridge command from Python to Rust engine executable for:
  - `legal-moves --fen <FEN>`
- Enrich records with `legal_moves_uci`.
- Add dataset filters:
  - remove illegal moves
  - remove duplicated `(fen, move_uci)` pairs
  - optional rating/time-control filtering

### Fast-learning impact

Bad labels poison training. Legality filtering gives a large quality jump with modest effort.

### Quality gates

- Sample audit: illegal move rate before/after filter.
- Stratified dedup report by opening phase and result.

## Phase 3 (Days 11-18): Baseline training loop (small + fast)

### Model choice

- Start with **small residual CNN** on 8x8 planes (piece planes + side/castling/en-passant features).
- Two heads:
  - **Policy** over legal moves
  - **Value** in [-1, 1]

### Deliverables

- `python/chess_resurected/features.py` (FEN -> tensor)
- `python/chess_resurected/train_baseline.py`
- `python/chess_resurected/eval_baseline.py`
- Training artifacts: metrics JSON + checkpoints.

### Training recipe (speed-first)

- Mixed precision enabled.
- Batch size tuned for full GPU utilization.
- Cosine LR schedule with warmup.
- Early stopping on validation value loss + policy accuracy.

### Success metrics

- Top-1 policy accuracy on held-out games.
- Value calibration (Brier / MSE vs game outcome).
- Inference latency target per position.

## Phase 4 (Days 19-24): Engine-labeled value enhancement

### Deliverables

- Position subsampling pipeline for analysis (e.g., 10-20% of records).
- Engine annotation pass producing `score_cp` / `mate_in`.
- Multi-target training:
  - game-result value target
  - engine-eval regression target

### Fast-learning impact

Engine labels sharpen tactical/value signal much faster than outcome-only labels.

### Quality gates

- Correlation between model value and engine cp.
- Tactical slice benchmark (mates, hanging pieces, checks).

## Phase 5 (Days 25-30): Closed-loop playing strength checks

### Deliverables

- Integrate model inference into engine evaluator interface.
- A/B self-play harness:
  - classical evaluator vs model-assisted evaluator
  - fixed opening suite
- Elo estimate with confidence bounds.

### Success metrics

- Positive Elo delta vs baseline at fixed time control.
- No regression on perft/correctness tests.

---

## Practical defaults for “effective and fast”

- Start with 1-3 million positions, not full corpus.
- Keep model under ~5M parameters for quick iteration.
- Re-train daily with immutable dataset versions.
- Track experiments with a minimal run registry (YAML/JSON + git SHA).
- Prefer fewer, cleaner features over many noisy handcrafted features.

## Recommended immediate next tasks (this week)

1. Implement PGN ingestion to `PositionRecord` with tests.
2. Add engine legal-move CLI and Python bridge validation.
3. Build a first 100k-position dataset and run a tiny baseline training job.
4. Publish a short benchmark report (accuracy + throughput + failure modes).

These four tasks provide the fastest path from architecture to measurable model learning.
