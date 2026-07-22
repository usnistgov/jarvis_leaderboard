# -*- coding: UTF-8 -*-
""" -----------------------------------------
@Project : jarvis_closure_gap
@Core Libraries : jarvis-tools, closuregap_jarvis (stdlib-only algorithm)
@Author : Ares (Mingze) ares
@Date : 22/07/2026 - 09:45
@Description: JARVIS-Leaderboard adapter for ClosureGap on the
             AI-SinglePropertyPrediction-optb88vdw_bandgap-dft_3d benchmark.
@Annotation: Pipeline: load official benchmark split -> map JVASP id to formula
             via jarvis dft_3d -> train ClosureGap on the train split ->
             predict on the test split -> write the official
             `id,target,prediction` CSV, zip it, and report local test MAE.
             The ClosureGap algorithm is used unmodified; we only extend its
             element property table so real formulas can be featurized.
----------------------------------------- """
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import zipfile
from pathlib import Path

# ClosureGap algorithm (unmodified) and its featurizer module.
from closuregap_jarvis import ClosureGapModel
from closuregap_jarvis import featurize as cg_featurize

from periodic_table import FULL_ELEMENTS

# Official benchmark identity. Keep this exact string: the leaderboard scorer
# derives category/property/dataset/split/metric from the file name.
BENCH_STEM = "AI-SinglePropertyPrediction-optb88vdw_bandgap-dft_3d-test-mae"
BENCH_JSON_NAME = "dft_3d_optb88vdw_bandgap.json"


def extend_element_table() -> None:
    """Extend ClosureGap's element table to the full periodic table.

    ClosureGap ships only ~29 elements, so real dft_3d formulas raise KeyError.
    We add missing elements without overwriting the original shipped values,
    keeping the descriptor math identical to the published algorithm.
    """
    added = 0
    for symbol, props in FULL_ELEMENTS.items():
        if symbol not in cg_featurize.ELEMENTS:
            cg_featurize.ELEMENTS[symbol] = props
            added += 1
    print(f"[elements] table size {len(cg_featurize.ELEMENTS)} (added {added})")


def load_benchmark_split(bench_json: Path) -> dict[str, dict[str, float]]:
    """Load the official train/val/test split (id -> target bandgap)."""
    data = json.loads(bench_json.read_text(encoding="utf-8"))
    for key in ("train", "val", "test"):
        if key not in data:
            raise KeyError(f"benchmark json missing '{key}' split")
    print(
        f"[split] train={len(data['train'])} "
        f"val={len(data['val'])} test={len(data['test'])}"
    )
    return data


def build_id_to_formula(cache_dir: Path) -> dict[str, str]:
    """Map every JVASP id in dft_3d to its chemical formula.

    Uses jarvis-tools figshare data. The formula string feeds ClosureGap's
    own parser, so the composition descriptors stay faithful to the algorithm.
    """
    from jarvis.db.figshare import data as jdata  # heavy import, keep local

    print("[data] loading dft_3d from jarvis-tools (first run downloads it)...")
    records = jdata("dft_3d")
    id2formula: dict[str, str] = {}
    for rec in records:
        jid = rec.get("jid")
        formula = rec.get("formula")
        if jid and formula:
            id2formula[jid] = str(formula)
    print(f"[data] dft_3d records={len(records)} mapped_formulae={len(id2formula)}")
    return id2formula


def make_training_csv(
    train_split: dict[str, float], id2formula: dict[str, str], out_csv: Path
) -> int:
    """Write a ClosureGap training CSV (formula,bandgap) from the train split."""
    n = 0
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["formula", "bandgap"])
        for jid, target in train_split.items():
            formula = id2formula.get(jid)
            if formula is None:
                continue
            writer.writerow([formula, target])
            n += 1
    print(f"[train] usable training rows={n}/{len(train_split)}")
    if n == 0:
        raise RuntimeError("no training rows mapped; check dft_3d/id alignment")
    return n


def safe_predict(model: ClosureGapModel, formula: str, fallback: float) -> float:
    """Predict a bandgap; fall back to the train mean on any parse failure.

    Every test id must get a row, so unparseable/rare formulas use a fallback
    rather than being dropped.
    """
    try:
        return float(model.predict(formula)["predicted_bandgap"])
    except (KeyError, ValueError, ZeroDivisionError):
        return fallback


def main() -> None:
    parser = argparse.ArgumentParser(description="ClosureGap JARVIS adapter")
    parser.add_argument(
        "--bench-json",
        type=Path,
        required=True,
        help=f"path to {BENCH_JSON_NAME} (decoded from the benchmark json.zip)",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("./data"))
    parser.add_argument("--out-dir", type=Path, default=Path("./output"))
    parser.add_argument("--epochs", type=int, default=900)
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    extend_element_table()
    split = load_benchmark_split(args.bench_json)
    id2formula = build_id_to_formula(args.cache_dir)

    train_csv = args.cache_dir / "closuregap_train.csv"
    make_training_csv(split["train"], id2formula, train_csv)

    train_mean = statistics.fmean(split["train"].values())
    print(f"[train] training ClosureGap (epochs={args.epochs})...")
    model = ClosureGapModel.train_csv(train_csv, epochs=args.epochs)

    # Predict on the official test split and write id,target,prediction.
    out_csv = args.out_dir / f"{BENCH_STEM}.csv"
    abs_errors: list[float] = []
    missing_formula = 0
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "target", "prediction"])
        for jid, target in split["test"].items():
            formula = id2formula.get(jid)
            if formula is None:
                missing_formula += 1
                pred = train_mean
            else:
                pred = safe_predict(model, formula, train_mean)
            writer.writerow([jid, target, round(pred, 6)])
            abs_errors.append(abs(pred - float(target)))

    mae = statistics.fmean(abs_errors)
    baseline_mae = statistics.fmean(
        abs(float(t) - train_mean) for t in split["test"].values()
    )

    # Zip exactly one CSV named identically to the zip stem.
    out_zip = args.out_dir / f"{BENCH_STEM}.csv.zip"
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_csv, arcname=out_csv.name)

    print("-" * 60)
    print(f"[result] test rows          : {len(abs_errors)}")
    print(f"[result] test ids w/o formula: {missing_formula} (used train mean)")
    print(f"[result] ClosureGap test MAE : {mae:.4f} eV")
    print(f"[result] mean-baseline MAE   : {baseline_mae:.4f} eV")
    print(f"[result] submission csv.zip  : {out_zip}")
    print("-" * 60)
    if missing_formula > 0:
        print(
            "[warn] some test ids lacked a dft_3d formula; if this count is "
            "large the id/dataset versions may be misaligned.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
