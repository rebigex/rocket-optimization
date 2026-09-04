"""Builds the training dataset for the surrogate models."""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rocketopt.design import DesignSpace
from rocketopt.ric import load_ric
from rocketopt.sampling import generate_mixed_dataset

ROOT = Path(__file__).resolve().parents[1]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 16384



def main() -> None:
    space = DesignSpace(load_ric(ROOT / "Data" / "Open Motor Data" / "Current.ric"))
    started = time.time()
    frame = generate_mixed_dataset(space, N, timestep=0.02, seed=7, workers=12)
    elapsed = time.time() - started

    out = ROOT / "outputs" / "data" / "designs.parquet"
    frame.to_parquet(out, index=False)
    print("{} designs in {:.1f}s ({:.1f}/s) -> {}".format(N, elapsed, N / elapsed, out))
    print("ok {:.1f}%  feasible {:.1f}%".format(
        100 * frame.ok.mean(), 100 * frame.feasible.mean()))


# Required: macOS spawns workers by re-importing this file.
if __name__ == "__main__":
    main()
