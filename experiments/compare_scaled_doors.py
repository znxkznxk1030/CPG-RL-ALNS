from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.compare_baselines import InstanceSpec, run_suite


SCALED_DOOR_SPECS = (
    # Larger than Medium-lite, with more common dock doors than compounds.
    InstanceSpec("Medium", 6, 9, 8, 4, (401, 402)),
    InstanceSpec("Big", 8, 12, 10, 4, (501, 502)),
)


def main() -> None:
    observations = run_suite(
        specs=SCALED_DOOR_SPECS,
        random_samples=30,
        paper_iterations=300,
        destination_agent_episodes=150,
        cargo_matrix_episodes=150,
        alns_iterations=300,
        raw_filename="scaled_door_results.csv",
        summary_filename="scaled_door_summary.md",
        summary_title="Scaled Door Experiment Summary",
    )
    summary_path = ROOT / "outputs" / "scaled_door_summary.md"
    print(summary_path.read_text(encoding="utf-8"))
    print(f"Wrote {len(observations)} raw observations to outputs/scaled_door_results.csv")


if __name__ == "__main__":
    main()
