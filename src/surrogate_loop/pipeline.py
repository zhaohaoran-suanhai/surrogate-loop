from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from surrogate_loop.artifacts import (
    create_run_directory,
    save_successful_run,
    write_failed_run,
)
from surrogate_loop.config import load_spec
from surrogate_loop.data import generate_dataset
from surrogate_loop.evaluation import train_select_and_test
from surrogate_loop.models import build_candidates
from surrogate_loop.split import split_dataset


@dataclass(frozen=True)
class RunResult:
    run_dir: Path
    status: str
    selected_model: str
    test_metrics: dict[str, float]


def run_pipeline(spec_path: Path, runs_dir: Path, request_text: str) -> RunResult:
    run_dir = create_run_directory(runs_dir)
    try:
        spec = load_spec(spec_path)
        dataset = generate_dataset(spec)
        split = split_dataset(dataset, spec.sampling)
        candidates = build_candidates(spec.sampling.seed, spec.models.candidates)
        selection = train_select_and_test(split, candidates, spec.acceptance)
        save_successful_run(run_dir, spec, request_text, dataset, split, selection)
    except Exception as error:
        write_failed_run(run_dir, spec_path, error)
        raise
    return RunResult(
        run_dir=run_dir,
        status="accepted" if selection.accepted else "rejected",
        selected_model=selection.selected_name,
        test_metrics=selection.test_metrics.to_dict(),
    )
