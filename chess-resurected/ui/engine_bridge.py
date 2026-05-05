"""Thin wrapper around the UCI engine subprocess."""
import subprocess
import threading
import queue
import os
import sys

def _default_engine_path() -> str:
    here = os.path.dirname(__file__)
    exe = "chess-resurected.exe" if os.name == "nt" else "chess-resurected"
    release = os.path.join(here, "..", "engine", "target", "release", exe)
    debug = os.path.join(here, "..", "engine", "target", "debug", exe)
    return release if os.path.exists(release) else debug


ENGINE_PATH = _default_engine_path()


class Engine:
    def __init__(self, path: str = ENGINE_PATH):
        self._proc = subprocess.Popen(
            [path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._q: queue.Queue[str] = queue.Queue()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self._send("uci")
        self._wait_for("uciok")
        self._send("isready")
        self._wait_for("readyok")

    def _read_loop(self):
        assert self._proc.stdout
        for line in self._proc.stdout:
            self._q.put(line.rstrip())

    def _send(self, cmd: str):
        assert self._proc.stdin
        self._proc.stdin.write(cmd + "\n")
        self._proc.stdin.flush()

    def _wait_for(self, prefix: str, timeout: float = 5.0) -> str:
        while True:
            try:
                line = self._q.get(timeout=timeout)
                if line.startswith(prefix):
                    return line
            except queue.Empty:
                raise TimeoutError(f"engine did not respond with '{prefix}'")

    def get_best_move(self, fen: str, movetime_ms: int = 2000) -> str | None:
        self._send(f"position fen {fen}")
        self._send(f"go movetime {movetime_ms}")
        line = self._wait_for("bestmove", timeout=movetime_ms / 1000 + 5)
        parts = line.split()
        move = parts[1] if len(parts) >= 2 else None
        return None if move == "0000" else move

    def set_option(self, name: str, value: str | int | float) -> None:
        self._send(f"setoption name {name} value {value}")
        self._send("isready")
        self._wait_for("readyok")

    def close(self):
        try:
            self._send("quit")
            self._proc.wait(timeout=2)
        except Exception:
            self._proc.kill()
