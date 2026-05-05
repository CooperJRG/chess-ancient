import torch

from chess_resurected.mcts_pipeline import (
    generate_mcts_cycle,
    load_replay,
    save_replay,
    train_replay,
)
from chess_resurected.model import AlphaZeroNet


def _settings(tmp_path):
    return {
        "seed": 3,
        "device": "cpu",
        "amp": False,
        "cycle_games": 1,
        "warmup_cycles": 0,
        "max_ply": 2,
        "mcts_simulations": 2,
        "random_opening_moves": 0,
        "temperature": 1.0,
        "temp_threshold": 2,
        "train_epochs": 1,
        "batch_size": 2,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "num_blocks": 1,
        "channels": 8,
        "dropout_p": 0.0,
        "decisive_sample_weight": 1.0,
        "replay_buffer_size": 100,
        "run_name": str(tmp_path.name),
    }


def test_mcts_cycle_generates_sparse_policy_distribution(tmp_path):
    settings = _settings(tmp_path)
    net = AlphaZeroNet(1, 8, 0.0)

    samples, report = generate_mcts_cycle(
        cycle=1,
        engine=None,
        net=net,
        device=torch.device("cpu"),
        settings=settings,
    )

    assert report["games"] == 1
    assert len(samples) > 0
    first = samples[0]
    assert first["tensor"].shape == (18, 8, 8)
    assert abs(first["policy_values"].sum().item() - 1.0) < 1e-6
    assert set(first["policy_indices"].tolist()).issubset(set(first["legal_indices"].tolist()))


def test_replay_training_accepts_distribution_targets(tmp_path):
    settings = _settings(tmp_path)
    net = AlphaZeroNet(1, 8, 0.0)
    samples, _report = generate_mcts_cycle(
        cycle=1,
        engine=None,
        net=net,
        device=torch.device("cpu"),
        settings=settings,
    )
    replay = tmp_path / "replay.pt"
    checkpoint = tmp_path / "latest.pt"
    metrics = tmp_path / "metrics.json"
    config = tmp_path / "config.json"
    save_replay(replay, samples, {"cycle": 1})

    report = train_replay(
        replay_path=replay,
        checkpoint_path=checkpoint,
        metrics_path=metrics,
        config_path=config,
        settings=settings,
    )

    assert checkpoint.exists()
    assert metrics.exists()
    assert config.exists()
    assert report["train_records"] == len(samples)
    assert report["history"][-1]["total_loss"] > 0


def test_stop_callback_finishes_current_cycle_without_corrupting_replay(tmp_path):
    settings = _settings(tmp_path)
    settings["cycle_games"] = 2
    calls = {"count": 0}

    def stop_after_one_progress(_report):
        calls["count"] += 1

    def stop_cb():
        return calls["count"] >= 1

    samples, report = generate_mcts_cycle(
        cycle=1,
        engine=None,
        net=AlphaZeroNet(1, 8, 0.0),
        device=torch.device("cpu"),
        settings=settings,
        stop_cb=stop_cb,
        progress_cb=stop_after_one_progress,
    )
    replay = tmp_path / "stopped.pt"
    save_replay(replay, samples, report)
    loaded, meta = load_replay(replay)

    assert report["games"] == 1
    assert meta["stopped"] is True
    assert len(loaded) == len(samples)
