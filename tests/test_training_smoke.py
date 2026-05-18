from qecml.data.stim_dataset import save_npz_dataset
from qecml.sim.stim_surface_code import (
    build_rotated_surface_code_circuit,
    circuit_metadata,
    get_detector_coordinates,
    sample_detector_events,
)
from qecml.training.trainer import train_from_config


def test_tiny_training_saves_checkpoint(tmp_path):
    circuit = build_rotated_surface_code_circuit(basis="z", distance=3, rounds=3, p=0.01)
    coords = get_detector_coordinates(circuit)
    metadata = circuit_metadata(basis="z", distance=3, rounds=3, p=0.01)
    train_events, train_labels = sample_detector_events(circuit, shots=96, seed=5)
    val_events, val_labels = sample_detector_events(circuit, shots=32, seed=6)
    train_path = tmp_path / "train.npz"
    val_path = tmp_path / "val.npz"
    save_npz_dataset(
        train_path,
        events=train_events,
        labels=train_labels,
        detector_coords=coords,
        metadata=metadata,
    )
    save_npz_dataset(
        val_path,
        events=val_events,
        labels=val_labels,
        detector_coords=coords,
        metadata=metadata,
    )
    run_dir = tmp_path / "run"
    result = train_from_config(
        {
            "data": {"train": str(train_path), "val": str(val_path)},
            "model": {"name": "mlp", "hidden_dim": 16, "dropout": 0.0},
            "training": {
                "batch_size": 32,
                "epochs": 2,
                "lr": 0.001,
                "weight_decay": 0.0,
                "grad_clip_norm": 1.0,
                "pos_weight": None,
                "seed": 0,
            },
            "output": {"run_dir": str(run_dir)},
            "device": "cpu",
        }
    )
    assert (run_dir / "checkpoint.pt").exists()
    assert (run_dir / "metrics.jsonl").exists()
    assert result["best_val_ler"] >= 0.0
