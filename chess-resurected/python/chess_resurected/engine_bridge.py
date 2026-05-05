"""Subprocess bridges for the Rust engine and UCI opponents."""

from __future__ import annotations

import os
import queue
import subprocess
import threading
from functools import lru_cache
from pathlib import Path


class EngineBridgeError(RuntimeError):
    """Raised when an engine subprocess cannot satisfy a request."""


def default_engine_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    exe = "chess-resurected.exe" if os.name == "nt" else "chess-resurected"
    candidates = [
        root / "engine" / "target" / "release" / exe,
        root / "engine" / "target" / "debug" / exe,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise EngineBridgeError(
        "Rust engine binary not found. Run `cargo build` in chess-resurected/engine "
        "or pass an explicit engine path."
    )


class LegalMoveOracle:
    """Calls `chess-resurected legal-moves --fen ...` and caches responses."""

    def __init__(self, engine_path: str | os.PathLike[str] | None = None, timeout: float = 10.0):
        self.engine_path = Path(engine_path) if engine_path else default_engine_path()
        self.timeout = timeout

    @lru_cache(maxsize=200_000)
    def legal_moves(self, fen: str) -> tuple[str, ...]:
        try:
            result = subprocess.run(
                [str(self.engine_path), "legal-moves", "--fen", fen],
                check=True,
                text=True,
                capture_output=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise EngineBridgeError(f"legal-moves timed out for FEN: {fen}") from exc
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if exc.stderr else str(exc)
            raise EngineBridgeError(f"legal-moves failed: {stderr}") from exc
        return tuple(result.stdout.split())

    def is_legal(self, fen: str, move_uci: str) -> bool:
        return move_uci in self.legal_moves(fen)


class UciEngine:
    """Small UCI subprocess wrapper for match play."""

    def __init__(self, path: str | os.PathLike[str], timeout: float = 10.0):
        self.path = Path(path)
        if not self.path.exists():
            raise EngineBridgeError(f"UCI engine path does not exist: {self.path}")
        self.timeout = timeout
        self._proc = subprocess.Popen(
            [str(self.path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._q: queue.Queue[str] = queue.Queue()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self.send("uci")
        self.wait_for("uciok")
        self.send("isready")
        self.wait_for("readyok")

    def _read_loop(self) -> None:
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            self._q.put(line.rstrip())

    def send(self, command: str) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.write(command + "\n")
        self._proc.stdin.flush()

    def wait_for(self, prefix: str, timeout: float | None = None) -> str:
        deadline = self.timeout if timeout is None else timeout
        while True:
            try:
                line = self._q.get(timeout=deadline)
            except queue.Empty as exc:
                raise EngineBridgeError(f"{self.path} did not respond with {prefix!r}") from exc
            if line.startswith(prefix):
                return line

    def bestmove(
        self,
        fen: str,
        *,
        movetime_ms: int | None = None,
        nodes: int | None = None,
    ) -> str | None:
        self.send(f"position fen {fen}")
        if nodes is not None:
            self.send(f"go nodes {nodes}")
            timeout = self.timeout
        else:
            movetime_ms = 100 if movetime_ms is None else movetime_ms
            self.send(f"go movetime {movetime_ms}")
            timeout = self.timeout + movetime_ms / 1000
        line = self.wait_for("bestmove", timeout=timeout)
        parts = line.split()
        if len(parts) < 2 or parts[1] == "0000":
            return None
        return parts[1]

    def close(self) -> None:
        try:
            self.send("quit")
            self._proc.wait(timeout=2)
        except Exception:
            self._proc.kill()

    def __enter__(self) -> "UciEngine":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
