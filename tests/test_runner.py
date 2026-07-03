from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.runner import Job, run_jobs


def _smoke_jobs() -> list[Job]:
    return [
        Job(method=method, pool="tuning", size_class="S", flow_pattern="uniform", index=index)
        for method in ("VAA", "VAA-QRL-50")
        for index in (0, 1)
    ]


def test_runner_executes_and_appends_records(tmp_path: Path) -> None:
    output = tmp_path / "results.jsonl"
    executed = run_jobs(_smoke_jobs(), output)

    assert executed == 4
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(records) == 4
    assert all(record["makespan"] > 0 for record in records)
    assert all(record["runtime_sec"] >= 0 for record in records)
    assert len({record["key"] for record in records}) == 4


def test_runner_resumes_without_duplicates(tmp_path: Path) -> None:
    output = tmp_path / "results.jsonl"
    jobs = _smoke_jobs()

    assert run_jobs(jobs[:2], output) == 2
    assert run_jobs(jobs, output) == 2
    assert run_jobs(jobs, output) == 0

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(records) == 4


def test_runner_rejects_unknown_method(tmp_path: Path) -> None:
    job = Job(method="NoSuchMethod", pool="tuning", size_class="S", flow_pattern="uniform", index=0)
    with pytest.raises(ValueError):
        run_jobs([job], tmp_path / "results.jsonl")


def test_runner_parallel_workers(tmp_path: Path) -> None:
    output = tmp_path / "results.jsonl"
    jobs = [
        Job(method="VAA", pool="tuning", size_class="S", flow_pattern="uniform", index=index)
        for index in range(4)
    ]
    assert run_jobs(jobs, output, workers=2) == 4
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert len({record["key"] for record in records}) == 4
