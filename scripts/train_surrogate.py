"""Trains the surrogate models and records how well they actually do."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from rocketopt.design import DesignSpace
from rocketopt.ric import motor_path, load_ric
from rocketopt.surrogate import TARGETS, Surrogate, compare_models

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def main() -> None:
    space = DesignSpace(load_ric(motor_path(ROOT)))
    frame = pd.read_parquet(OUT / "data" / "designs.parquet")
    print("dataset: {} designs, {} simulated cleanly, {} feasible".format(
        len(frame), int(frame.ok.sum()), int(frame.feasible.sum())))

    comparison = compare_models(space, frame)
    comparison.to_csv(OUT / "data" / "model_comparison.csv")
    print("\nheld-out R² by model family:")
    print(comparison.round(4).to_string())

    best_kind = comparison.mean(axis=0).idxmax()
    print("\nbest average R²: {}".format(best_kind))

    surrogate = Surrogate(space, kind=best_kind)
    scores = surrogate.fit(frame)
    surrogate.save(OUT / "models")
    print("\nfinal surrogate ({}):".format(best_kind))
    for score in scores:
        print("  {:18s} R²={:7.4f}  MAE={:12.3f}  MAPE={:6.2f}%".format(
            score.target, score.r2, score.mae, score.mape))

    importances = {}
    for target in ("initial_thrust", "total_impulse", "peak_mass_flux"):
        table = surrogate.importances(frame, target)
        table.to_csv(OUT / "data" / "importance_{}.csv".format(target), index=False)
        importances[target] = table.head(6).to_dict("records")
        print("\ntop drivers of {}:".format(target))
        print(table.head(6).to_string(index=False))

    (OUT / "data" / "surrogate_summary.json").write_text(json.dumps(
        {"kind": best_kind, "scores": [s.as_row() for s in scores],
         "importances": importances}, indent=2))


if __name__ == "__main__":
    main()
