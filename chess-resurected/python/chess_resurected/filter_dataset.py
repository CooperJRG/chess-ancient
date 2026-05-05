"""Filter and report on PositionRecord JSONL datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import dataset_report, git_sha, read_jsonl, write_jsonl, write_report
from .engine_bridge import LegalMoveOracle
from .records import PositionRecord
from .validation import validate_record


def filter_records(
    records: list[PositionRecord],
    *,
    oracle: LegalMoveOracle | None,
    dedup: bool = True,
    drop_illegal: bool = True,
) -> tuple[list[PositionRecord], dict]:
    kept: list[PositionRecord] = []
    seen: set[tuple[str, str | None]] = set()
    duplicate_count = 0
    illegal_count = 0

    for record in records:
        key = (record.fen, record.move_uci)
        if dedup and key in seen:
            duplicate_count += 1
            continue
        issues = validate_record(record, oracle=oracle, require_legal=drop_illegal)
        is_illegal = any(issue.field == "move_uci" for issue in issues)
        if drop_illegal and issues:
            if is_illegal:
                illegal_count += 1
            continue
        seen.add(key)
        kept.append(record)

    report = dataset_report(kept, input_count=len(records))
    report.update(
        {
            "dedup_removed": duplicate_count,
            "illegal_removed": illegal_count,
            "git_sha": git_sha(Path(__file__).resolve().parents[2]),
        }
    )
    return kept, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input PositionRecord JSONL")
    parser.add_argument("--output", required=True, help="Filtered output JSONL")
    parser.add_argument("--report", required=True, help="Report JSON path")
    parser.add_argument("--engine", help="Path to chess-resurected engine binary")
    parser.add_argument("--keep-duplicates", action="store_true")
    parser.add_argument("--keep-illegal", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = list(read_jsonl(args.input))
    oracle = None if args.keep_illegal else LegalMoveOracle(args.engine)
    kept, report = filter_records(
        records,
        oracle=oracle,
        dedup=not args.keep_duplicates,
        drop_illegal=not args.keep_illegal,
    )
    report["input_path"] = str(args.input)
    report["output_path"] = str(args.output)
    write_jsonl(args.output, kept)
    write_report(args.report, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
