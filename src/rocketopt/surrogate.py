"""Surrogate models that predict burn outcomes without running openMotor.

Not every quantity needs a model. ``port/throat``, initial ``Kn`` and
propellant mass are closed-form BATES results that :mod:`rocketopt.design`
already computes exactly, so learning them would only add error. The models
here cover the quantities that genuinely depend on integrating the whole burn:
initial thrust, total impulse, peak pressure, peak mass flux and so on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

#: Quantities the optimiser needs that are not available in closed form.
TARGETS: List[str] = [
    "initial_thrust",
    "total_impulse",
    "max_pressure",
    "peak_mass_flux",
    "peak_thrust",
    "isp",
    "burn_time",
    "thrust_variation",
    "peak_kn",
    # Added so the app can optimise or constrain any offered metric without
    # falling back to the simulator mid-search.
    "avg_thrust",
    "avg_pressure",
    "volume_loading",
]

#: Quantities computed exactly by DesignSpace.features, so never learned.
ANALYTIC = {
    "port_throat": "port_throat_0",
    "initial_kn": "kn_0",
    "prop_mass": "prop_mass",
}


def build_model(kind: str = "gbt", seed: int = 0):
    if kind == "gbt":
        return HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.06, max_leaf_nodes=63,
            min_samples_leaf=10, l2_regularization=1e-3,
            early_stopping=True, validation_fraction=0.12, random_state=seed,
        )
    if kind == "rf":
        return RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed
        )
    if kind == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(128, 128, 64), activation="relu",
                learning_rate_init=2e-3, max_iter=800, early_stopping=True,
                n_iter_no_change=25, random_state=seed,
            ),
        )
    raise ValueError("unknown model kind {!r}".format(kind))


@dataclass
class TargetScore:
    target: str
    r2: float
    mae: float
    mape: float

    def as_row(self) -> Dict:
        return {"target": self.target, "r2": self.r2, "mae": self.mae, "mape": self.mape}


class Surrogate:
    """One regressor per target, sharing the physics feature vector as input."""

    def __init__(self, space, kind: str = "gbt", seed: int = 0) -> None:
        self.space = space
        self.kind = kind
        self.seed = seed
        self.models: Dict[str, object] = {}
        self.scores: List[TargetScore] = []
        self.feature_names = list(space.feature_names)

    # ------------------------------------------------------------- training

    def fit(self, frame: pd.DataFrame, test_size: float = 0.2) -> List[TargetScore]:
        """Trains on the feasible-or-not but successfully simulated designs.

        Failed simulations carry no usable target values, so they are dropped;
        infeasible ones are kept, because the optimiser has to be able to see
        the constraint boundary from the inside and the outside.
        """
        usable = frame[frame["ok"]].reset_index(drop=True)
        X = usable[self.feature_names].to_numpy(dtype=float)
        idx_train, idx_test = train_test_split(
            np.arange(len(usable)), test_size=test_size, random_state=self.seed
        )
        self.scores = []
        for target in TARGETS:
            y = usable[target].to_numpy(dtype=float)
            model = build_model(self.kind, self.seed)
            model.fit(X[idx_train], y[idx_train])
            pred = model.predict(X[idx_test])
            actual = y[idx_test]
            denom = np.maximum(np.abs(actual), 1e-9)
            self.models[target] = model
            self.scores.append(
                TargetScore(
                    target=target,
                    r2=float(r2_score(actual, pred)),
                    mae=float(mean_absolute_error(actual, pred)),
                    mape=float(np.mean(np.abs(pred - actual) / denom) * 100.0),
                )
            )
        return self.scores

    # ----------------------------------------------------------- prediction

    def predict(self, X_design: np.ndarray) -> pd.DataFrame:
        """Predicts every target for a batch of design vectors."""
        features = self.space.features(np.atleast_2d(X_design))
        out = pd.DataFrame(
            {target: model.predict(features) for target, model in self.models.items()}
        )
        frame = pd.DataFrame(features, columns=self.feature_names)
        for name, source in ANALYTIC.items():
            out[name] = frame[source].to_numpy()
        return out

    # ------------------------------------------------------------ diagnostics

    def importances(self, frame: pd.DataFrame, target: str, n_repeats: int = 8,
                    n_samples: int = 2000) -> pd.DataFrame:
        usable = frame[frame["ok"]]
        usable = usable.sample(min(n_samples, len(usable)), random_state=self.seed)
        X = usable[self.feature_names].to_numpy(dtype=float)
        y = usable[target].to_numpy(dtype=float)
        result = permutation_importance(
            self.models[target], X, y, n_repeats=n_repeats,
            random_state=self.seed, n_jobs=-1,
        )
        return (
            pd.DataFrame(
                {"feature": self.feature_names,
                 "importance": result.importances_mean,
                 "std": result.importances_std}
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # ---------------------------------------------------------- persistence

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump({"models": self.models, "kind": self.kind,
                     "feature_names": self.feature_names}, directory / "surrogate.joblib")
        (directory / "scores.json").write_text(
            json.dumps([s.as_row() for s in self.scores], indent=2)
        )
        return directory

    @classmethod
    def load(cls, directory: str | Path, space) -> "Surrogate":
        directory = Path(directory)
        blob = joblib.load(directory / "surrogate.joblib")
        surrogate = cls(space, kind=blob["kind"])
        surrogate.models = blob["models"]
        surrogate.feature_names = blob["feature_names"]
        scores = json.loads((directory / "scores.json").read_text())
        surrogate.scores = [TargetScore(**s) for s in scores]
        return surrogate


def compare_models(space, frame: pd.DataFrame, kinds: Sequence[str] = ("gbt", "rf", "mlp"),
                   test_size: float = 0.2) -> pd.DataFrame:
    """Trains each model family so the choice of surrogate is evidence-based."""
    rows = []
    for kind in kinds:
        surrogate = Surrogate(space, kind=kind)
        for score in surrogate.fit(frame, test_size=test_size):
            row = score.as_row()
            row["model"] = kind
            rows.append(row)
    return pd.DataFrame(rows).pivot(index="target", columns="model", values="r2")
