import json
from pathlib import Path

from seqformer_lab.cli import main


def test_demo_and_evaluate_workflow(tmp_path: Path) -> None:
    output = tmp_path / "run"
    assert main(["demo", "--output", str(output), "--epochs", "1", "--seed", "4"]) == 0
    assert (output / "model.pt").is_file()
    assert (output / "manifest.json").is_file()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["train_periods"] == [0, 1, 2, 3]
    assert manifest["test_periods"] == [4]

    assert (
        main(
            [
                "evaluate",
                "--data",
                str(output / "dataset.jsonl"),
                "--checkpoint",
                str(output),
            ]
        )
        == 0
    )
