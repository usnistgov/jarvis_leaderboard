from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

from .featurize import composition_features


@dataclass
class ClosureGapModel:
    weights: list[float]
    centroid: list[float]
    scale: list[float]

    @classmethod
    def train_csv(cls, path: Path, epochs: int = 900, lr: float = 0.00002, ridge: float = 0.0001) -> "ClosureGapModel":
        rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
        xs = [composition_features(r["formula"]) for r in rows]
        ys = [float(r["bandgap"]) for r in rows]
        dim = len(xs[0])
        centroid = [sum(x[i] for x in xs) / len(xs) for i in range(dim)]
        scale = []
        for i in range(dim):
            var = sum((x[i] - centroid[i]) ** 2 for x in xs) / max(len(xs) - 1, 1)
            scale.append(math.sqrt(var) or 1.0)
        xsn = [[(x[i] - centroid[i]) / scale[i] for i in range(dim)] for x in xs]
        weights = [0.0] * dim
        for _ in range(epochs):
            for x, y in zip(xsn, ys):
                pred = sum(w * v for w, v in zip(weights, x))
                err = pred - y
                for i in range(dim):
                    weights[i] -= lr * (err * x[i] + ridge * weights[i])
        return cls(weights=weights, centroid=centroid, scale=scale)

    def predict(self, formula: str) -> dict[str, float | str]:
        x = composition_features(formula)
        xn = [(x[i] - self.centroid[i]) / self.scale[i] for i in range(len(x))]
        pred = sum(w * v for w, v in zip(self.weights, xn))
        uncertainty = math.sqrt(sum(v * v for v in xn) / len(xn))
        closure_priority = 1.0 / (1.0 + abs(pred)) * (1.0 + min(uncertainty, 5.0) / 5.0)
        return {
            "formula": formula,
            "predicted_bandgap": max(pred, 0.0),
            "uncertainty_proxy": uncertainty,
            "closure_priority": closure_priority,
        }

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.__dict__, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "ClosureGapModel":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)
