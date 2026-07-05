"""Append-only, resumable experiment runner.

Each job is identified by a unique key. Results are appended to a JSONL file;
jobs whose keys already appear in the file are skipped, so an interrupted
sweep can be re-run with the same job list and only missing work executes.
Workers return records to the parent process, which is the only writer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import multiprocessing
from pathlib import Path
import time

from experiments.methods import METHOD_REGISTRY
from experiments.protocol import BenchmarkCell, cell_instance, cell_seed


@dataclass(frozen=True)
class Job:
    method: str
    pool: str
    size_class: str
    flow_pattern: str
    index: int
    rep: int = 0
    budget_sec: float | None = None
    tw_tightness: str | None = None

    @property
    def key(self) -> str:
        tightness = self.tw_tightness or "none"
        return (
            f"{self.method}|{self.pool}|{self.size_class}|{self.flow_pattern}"
            f"|{tightness}|{self.index}|{self.rep}"
        )


def run_jobs(
    jobs: list[Job],
    output_path: Path,
    *,
    workers: int = 1,
) -> int:
    """Execute all jobs not yet present in `output_path`. Returns the number executed."""

    unknown = sorted({job.method for job in jobs} - set(METHOD_REGISTRY))
    if unknown:
        raise ValueError(f"unknown methods (register in experiments.methods): {unknown}")

    done = _load_done_keys(output_path)
    pending = [job for job in jobs if job.key not in done]
    if not pending:
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as sink:
        if workers <= 1:
            for job in pending:
                _append(sink, _execute(job))
        else:
            with multiprocessing.Pool(workers) as pool:
                for record in pool.imap_unordered(_execute, pending):
                    _append(sink, record)
    return len(pending)


def _execute(job: Job) -> dict:
    cell = BenchmarkCell(
        size_class=job.size_class,
        flow_pattern=job.flow_pattern,
        tw_tightness=job.tw_tightness,
    )
    instance = cell_instance(job.pool, cell, job.index)
    instance_seed = cell_seed(job.pool, cell, job.index)
    method_seed = instance_seed * 100 + job.rep

    method = METHOD_REGISTRY[job.method]
    result = method(instance, method_seed, job.budget_sec)

    return {
        "key": job.key,
        **asdict(job),
        "instance_seed": instance_seed,
        "method_seed": method_seed,
        "finished_at": time.time(),
        **result,
    }


def _append(sink, record: dict) -> None:
    sink.write(json.dumps(record) + "\n")
    sink.flush()


def _load_done_keys(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    keys: set[str] = set()
    with output_path.open("r", encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if not line:
                continue
            keys.add(json.loads(line)["key"])
    return keys
