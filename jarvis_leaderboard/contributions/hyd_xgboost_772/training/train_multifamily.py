"""Prespecified grouped search on enriched JARVIS structure features.

Only train.npz is opened during selection. Candidate choice, refit rounds,
ensemble weights and threshold are frozen before any access to val.npz.
The validation split is an already observed development benchmark, not a test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def progress(phase, step, values):
    print("TRISOL_PROGRESS " + json.dumps({"v": 1, "step": step,
        "metrics": {f"custom/{phase}_{key}": value for key, value in values.items()}}), flush=True)


def union_groups(ids, duplicates):
    index = {str(jid): i for i, jid in enumerate(ids)}
    parent = np.arange(len(ids))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    if isinstance(duplicates, dict):
        duplicates = duplicates.get("duplicate_groups", duplicates.get("groups"))
    if not isinstance(duplicates, list):
        raise ValueError("Duplicate groups must be a list")
    for group in duplicates:
        members = [index[str(jid)] for jid in group if str(jid) in index]
        for i in members[1:]:
            parent[find(i)] = find(members[0])
    return np.asarray([find(i) for i in range(len(ids))])


def calibrate(y, probability, thresholds):
    correct = np.asarray([np.count_nonzero((probability >= t) == y) for t in thresholds])
    best = min(np.flatnonzero(correct == correct.max()), key=lambda i: (abs(thresholds[i] - .5), thresholds[i]))
    return {"threshold": float(thresholds[best]), "correct_count": int(correct[best]),
            "sample_count": len(y), "accuracy": float(correct[best] / len(y)),
            "log_loss": float(log_loss(y, probability, labels=[0, 1])),
            "fixed_half_accuracy": float(accuracy_score(y, probability >= .5))}


def score(y, p, threshold, novel=None):
    pred = p >= threshold
    result = {"accuracy": float(accuracy_score(y, pred)), "correct_count": int(np.sum(pred == y)),
        "sample_count": len(y), "threshold": float(threshold),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "confusion_matrix_labels_0_1": confusion_matrix(y, pred, labels=[0, 1]).tolist()}
    if novel is not None:
        result.update(novel_structure_count=int(np.sum(novel)),
                      novel_structure_accuracy=float(accuracy_score(y[novel], pred[novel])) if np.any(novel) else None)
    return result


def fit_model(candidate, x, y, fit, tune, seed, config, cpu_test=False, rounds=None):
    """Fit one family; return model and the exact retained boosting length."""
    family = candidate["family"]
    early_stop = min(5, config["early_stopping_rounds"]) if cpu_test else config["early_stopping_rounds"]
    max_rounds = int(rounds or candidate.get("max_rounds", config["max_rounds"]))
    if cpu_test and rounds is None:
        max_rounds = 16
    full = tune is None
    threads = min(2, config["threads"]) if cpu_test else config["threads"]
    if family == "xgboost":
        import xgboost as xgb
        params = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist",
                  "learning_rate": .04, "n_jobs": threads, **candidate["params"],
                  "n_estimators": max_rounds, "device": "cpu" if cpu_test else "cuda:0", "random_state": seed}
        if not full:
            params["early_stopping_rounds"] = early_stop
        model = xgb.XGBClassifier(**params)
        model.fit(x[fit], y[fit], verbose=False, **({} if full else {"eval_set": [(x[tune], y[tune])]}))
        actual = json.loads(model.get_booster().save_config())["learner"]["generic_param"]["device"]
        if not cpu_test and not actual.startswith("cuda"):
            raise RuntimeError("XGBoost training silently fell back to CPU")
        retained = max_rounds if full else int(model.best_iteration) + 1
        version = xgb.__version__
    elif family == "lightgbm":
        import lightgbm as lgb
        params = {"objective": "binary", "learning_rate": .04, "verbosity": -1,
                  "n_jobs": threads, "deterministic": True, "force_col_wise": True,
                  **candidate["params"], "n_estimators": max_rounds, "random_state": seed, "device_type": "cpu"}
        model = lgb.LGBMClassifier(**params)
        extra = {} if full else {"eval_set": [(x[tune], y[tune])], "eval_metric": "binary_logloss",
                                 "callbacks": [lgb.early_stopping(early_stop, first_metric_only=True, verbose=False)]}
        model.fit(x[fit], y[fit], **extra)
        retained = max_rounds if full else int(model.best_iteration_)
        if retained <= 0:
            retained = max_rounds
        actual, version = "cpu", lgb.__version__
    elif family == "catboost":
        import catboost as cat
        params = {"loss_function": "Logloss", "eval_metric": "Logloss", "learning_rate": .08,
                  "verbose": False, "allow_writing_files": False, "thread_count": threads,
                  "border_count": 128, **candidate["params"], "iterations": max_rounds,
                  "task_type": "CPU" if cpu_test else "GPU", "random_seed": seed}
        if not cpu_test:
            params.update(devices="0", gpu_ram_part=.85)
        model = cat.CatBoostClassifier(**params)
        extra = {} if full else {"eval_set": (x[tune], y[tune]), "early_stopping_rounds": early_stop, "use_best_model": True}
        model.fit(x[fit], y[fit], **extra)
        retained = int(model.tree_count_)
        actual = model.get_all_params()["task_type"].lower()
        if not cpu_test and actual != "gpu":
            raise RuntimeError("CatBoost training silently fell back to CPU")
        version = cat.__version__
    else:
        raise ValueError(f"Unknown model family {family}")
    return model, retained, {"family": family, "params": params, "actual_device": actual, "version": version}


def predict(model, family, x, rounds):
    if family == "xgboost":
        return np.asarray(model.predict_proba(x, iteration_range=(0, rounds))[:, 1], dtype=np.float64)
    if family == "lightgbm":
        return np.asarray(model.booster_.predict(x, num_iteration=rounds, num_threads=model.get_params()["n_jobs"]), dtype=np.float64)
    return np.asarray(model.predict_proba(x)[:, 1], dtype=np.float64)


def save_model(model, family, path):
    if family == "lightgbm":
        # LightGBM's native file APIs do not support every Windows Unicode path.
        Path(path).write_text(model.booster_.model_to_string(), encoding="utf-8")
    else:
        model.save_model(str(path))


def load_predict(trained, out, x):
    family, path = trained["family"], out / trained["file"]
    if family == "xgboost":
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(path)
        return np.asarray(model.predict_proba(x)[:, 1], dtype=np.float64)
    if family == "lightgbm":
        import lightgbm as lgb
        model = lgb.Booster(model_str=path.read_text(encoding="utf-8"))
        return np.asarray(model.predict(x, num_threads=trained["params"]["n_jobs"]), dtype=np.float64)
    from catboost import CatBoostClassifier
    model = CatBoostClassifier()
    model.load_model(str(path))
    return np.asarray(model.predict_proba(x)[:, 1], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-test", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    data, out = args.data.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    internal = out / "internal"
    internal.mkdir(exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    manifest = json.loads((data / "data_manifest.json").read_text(encoding="utf-8-sig"))
    if config.get("test_included") is not False or manifest.get("test_included") is not False:
        raise ValueError("Both config and manifest must explicitly exclude the test set")
    if not args.cpu_test and manifest.get("partial_smoke_test"):
        raise ValueError("Production run cannot use smoke-test data")
    if config.get("data_manifest_sha256") and digest(data / "data_manifest.json") != config["data_manifest_sha256"]:
        raise ValueError("Feature manifest differs from frozen configuration")
    if digest(data / "train.npz") != manifest["file_sha256"]["train.npz"]:
        raise ValueError("Training checksum mismatch")
    with np.load(data / "train.npz", allow_pickle=False) as z:
        ids, x, y = z["ids"], z["X"], z["y"]
    if x.ndim != 2 or y.ndim != 1 or ids.ndim != 1 or len(x) != len(y) or len(ids) != len(y):
        raise ValueError("Invalid training array shapes")
    if not args.cpu_test and (len(y) != 41766 or x.shape[1] < 434):
        raise ValueError(f"Unexpected full training shape {x.shape}")
    if len(set(ids)) != len(ids) or not np.isfinite(x).all() or set(y.tolist()) != {0, 1}:
        raise ValueError("Invalid training IDs, features or labels")
    duplicate_path = data / "duplicate_groups.json"
    groups = union_groups(ids, json.loads(duplicate_path.read_text(encoding="utf-8-sig")))
    candidates = config["candidates"]
    if args.cpu_test:
        # Exercise each family and each feature view with a small synthetic fixture.
        seen = set()
        smoke_candidates = []
        for candidate in candidates:
            stratum = (candidate["family"], candidate["feature_view"])
            if stratum not in seen:
                seen.add(stratum)
                smoke_candidates.append(candidate)
        candidates = smoke_candidates
    names = [c["name"] for c in candidates]
    if len(set(names)) != len(names) or any(re.fullmatch(r"[A-Za-z0-9_-]+", name) is None for name in names):
        raise ValueError("Candidate names must be unique, safe filenames")
    by_name = {c["name"]: c for c in candidates}
    views = {}
    for candidate in candidates:
        view = candidate["feature_view"]
        if view not in config["feature_views"]:
            raise ValueError(f"Unknown feature view {view}")
        end = config["feature_views"][view]
        if end is not None and (not isinstance(end, int) or not 0 < end <= x.shape[1]):
            raise ValueError(f"Invalid feature view endpoint {end}")
        views[view] = x if end is None else x[:, :end]
    thresholds = np.asarray(config["threshold_grid"], dtype=float)
    if thresholds.ndim != 1 or not len(thresholds) or not np.all((thresholds > 0) & (thresholds < 1)):
        raise ValueError("Invalid threshold grid")
    seeds = config["split_seeds"]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("At least two distinct internal splits required")
    splits = {}
    for seed in seeds:
        fit, tune = next(GroupShuffleSplit(n_splits=1, test_size=config["internal_fraction"], random_state=seed).split(x, y, groups))
        if set(groups[fit]) & set(groups[tune]) or len(np.unique(y[tune])) != 2:
            raise ValueError("Invalid grouped internal holdout")
        splits[seed] = (fit, tune)
        np.savez_compressed(internal / f"split-{seed}.npz", fit=fit, tune=tune, ids=ids[tune])
    gpu = None
    if not args.cpu_test and any(c["family"] in ("xgboost", "catboost") for c in candidates):
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,uuid", "--format=csv,noheader"], text=True, timeout=20).strip()
        if len(gpu.splitlines()) != 1 or "A100" not in gpu:
            raise RuntimeError("Expected exactly one A100 GPU")
    protocol = {"config": config, "config_sha256": digest(args.config), "script_sha256": digest(__file__),
        "data_manifest_sha256": digest(data / "data_manifest.json"), "duplicate_groups_sha256": digest(duplicate_path),
        "test_accessed": False, "official_validation_loaded": False, "selection_source": "training-internal grouped holdouts",
        "train_shape": list(x.shape), "gpu": gpu, "cpu_test": args.cpu_test,
        "limitations": ["Internal holdouts are tuning data and may overlap across seeds.",
                        "Official validation is an already observed development benchmark.",
                        "Reconstructed features are not byte-identical to the reference member's inaccessible package."]}
    write_json(out / "search_protocol.json", protocol)
    probabilities, records = {}, []

    def trial(candidate, seed):
        if time.monotonic() - started > config["max_search_seconds"]:
            raise TimeoutError("Prespecified search time budget exhausted; no validation was opened")
        begin = time.monotonic()
        fit, tune = splits[seed]
        model, rounds, details = fit_model(candidate, views[candidate["feature_view"]], y, fit, tune, seed, config, args.cpu_test)
        p = predict(model, candidate["family"], views[candidate["feature_view"]][tune], rounds)
        if p.shape != (len(tune),) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("Invalid internal probabilities")
        probabilities[(candidate["name"], seed)] = p
        np.savez_compressed(internal / f"{candidate['name']}-s{seed}.npz", tune_indices=tune, ids=ids[tune], probability=p, target=y[tune])
        row = {"candidate": candidate["name"], "family": candidate["family"], "feature_view": candidate["feature_view"],
            "seed": seed, "rounds": rounds, "seconds": time.monotonic() - begin,
            **calibrate(y[tune], p, thresholds), **details}
        records.append(row)
        write_json(out / "internal_search.json", records)
        progress("search", len(records), {"accuracy": row["accuracy"], "rounds": rounds, "seconds": row["seconds"]})
        print(json.dumps(row), flush=True)
        del model
        return row

    first = [trial(candidate, seeds[0]) for candidate in candidates]
    first.sort(key=lambda row: (-row["accuracy"], row["log_loss"], row["candidate"]))
    # Retain the best of each family/view so the baseline and model diversity
    # survive first-split screening; fill remaining slots by global rank.
    shortlist, strata = [], set()
    for row in first:
        stratum = (row["family"], row["feature_view"])
        if stratum not in strata:
            strata.add(stratum)
            shortlist.append(row["candidate"])
    if len(shortlist) > config["shortlist_count"]:
        raise ValueError("Shortlist too small to cover every family and feature view")
    for row in first:
        if len(shortlist) >= config["shortlist_count"]:
            break
        if row["candidate"] not in shortlist:
            shortlist.append(row["candidate"])
    for name in shortlist:
        for seed in seeds[1:]:
            trial(by_name[name], seed)
    pooled_y = np.concatenate([y[splits[seed][1]] for seed in seeds])
    pooled = {name: np.concatenate([probabilities[(name, seed)] for seed in seeds]) for name in shortlist}
    ranked = []
    for name in shortlist:
        rows = [row for row in records if row["candidate"] == name]
        ranked.append({"name": name, "family": by_name[name]["family"], "feature_view": by_name[name]["feature_view"],
            "rounds": int(np.median([row["rounds"] for row in rows])),
            **calibrate(pooled_y, pooled[name], thresholds)})
    ranked.sort(key=lambda row: (-row["accuracy"], row["log_loss"], row["name"]))
    # A fixed finite ensemble grid is searched only on aligned internal outputs.
    # Singles, weighted pairs, and equal-weight triples/quads are all explicit.
    mixtures = [{name: 1.0} for name in shortlist]
    for a, b in itertools.combinations(shortlist, 2):
        for weight in config["pair_weight_grid"]:
            mixtures.append({a: float(weight), b: float(1 - weight)})
    for count in range(3, min(config["max_ensemble_configs"], len(shortlist)) + 1):
        mixtures.extend({name: 1 / count for name in combo} for combo in itertools.combinations(shortlist, count))
    blend_scores = []
    for weights in mixtures:
        p = sum(weight * pooled[name] for name, weight in weights.items())
        blend_scores.append({"weights": weights, **calibrate(pooled_y, p, thresholds)})
    blend_scores.sort(key=lambda row: (-row["correct_count"], row["log_loss"], len(row["weights"]), sorted(row["weights"])))
    choice = blend_scores[0]
    chosen_by_name = {row["name"]: row for row in ranked}
    selected = []
    for name, weight in choice["weights"].items():
        chosen = chosen_by_name[name]
        selected.append({**by_name[name], "weight": weight,
                         "rounds": max(1, int(round(chosen["rounds"] * config.get("refit_round_multiplier", 1)))),
                         "internal_metrics": chosen})
    frozen = {"selected": selected, "ensemble_threshold": choice["threshold"], "internal_ensemble_metrics": choice,
        "final_seeds": config["final_seeds"], "ranked_candidates": ranked,
        "selection_source": "training-internal grouped holdouts only", "official_validation_loaded": False,
        "test_accessed": False, "config_sha256": digest(args.config), "script_sha256": digest(__file__),
        "data_manifest_sha256": digest(data / "data_manifest.json"), "search_seconds": time.monotonic() - started}
    write_json(out / "internal_blend_grid.json", blend_scores)
    write_json(out / "frozen_selection.json", frozen)
    np.savez_compressed(out / "internal_ensemble_probabilities.npz", target=pooled_y,
        indices=np.concatenate([splits[seed][1] for seed in seeds]),
        probability=sum(weight * pooled[name] for name, weight in choice["weights"].items()))
    # Every final model is fully trained and saved before validation is opened.
    per_model = []
    extensions = {"xgboost": "ubj", "lightgbm": "txt", "catboost": "cbm"}
    for chosen in selected:
        for seed in config["final_seeds"]:
            if time.monotonic() - started > config["max_total_seconds"]:
                raise TimeoutError("Prespecified total time budget exhausted before validation")
            begin = time.monotonic()
            model, rounds, details = fit_model(chosen, views[chosen["feature_view"]], y, np.arange(len(y)), None,
                                               seed, config, args.cpu_test, rounds=chosen["rounds"])
            file = f"{chosen['name']}-s{seed}.{extensions[chosen['family']]}"
            save_model(model, chosen["family"], out / file)
            per_model.append({"candidate": chosen["name"], "feature_view": chosen["feature_view"], "seed": seed,
                "file": file, "sha256": digest(out / file), "rounds": rounds,
                "ensemble_weight": chosen["weight"] / len(config["final_seeds"]), "seconds": time.monotonic() - begin, **details})
            write_json(out / "refit_manifest.json", per_model)
            progress("refit", len(per_model), {"models": len(per_model), "seconds": time.monotonic() - started})
            del model
    if digest(data / "val.npz") != manifest["file_sha256"]["val.npz"]:
        raise ValueError("Validation checksum mismatch")
    with np.load(data / "val.npz", allow_pickle=False) as z:
        val_ids, xv, yv, novel = z["ids"], z["X"], z["y"], z["novel_structure"].astype(bool)
    if xv.ndim != 2 or xv.shape[1] != x.shape[1] or len(xv) != len(yv) or len(val_ids) != len(yv) or len(novel) != len(yv):
        raise ValueError("Invalid validation shapes")
    if not args.cpu_test and len(yv) != 5221:
        raise ValueError("Expected the same 5,221-row validation benchmark")
    if set(ids) & set(val_ids) or len(set(val_ids)) != len(val_ids) or not np.isfinite(xv).all() or set(yv.tolist()) != {0, 1}:
        raise ValueError("Invalid validation data or IDs")
    prediction_list = []
    for trained in per_model:
        end = config["feature_views"][trained["feature_view"]]
        p = load_predict(trained, out, xv if end is None else xv[:, :end])
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("Invalid validation probabilities")
        prediction_list.append(p)
    probability = sum(row["ensemble_weight"] * p for row, p in zip(per_model, prediction_list))
    threshold = frozen["ensemble_threshold"]
    metrics = {**score(yv, probability, threshold, novel), "train_count": len(y), "feature_count": x.shape[1],
        "model_count": len(per_model), "frozen_selection_sha256": digest(out / "frozen_selection.json"),
        "test_accessed": False, "official_validation_used_for_model_choice": False,
        "validation_opened_after_all_refits": True, "total_seconds": time.monotonic() - started,
        "reference_accuracy": 4969 / 5221, "net_correct_gain_vs_reference": int(np.sum((probability >= threshold) == yv) - 4969)}
    if args.cpu_test:
        metrics.pop("net_correct_gain_vs_reference")
    with (out / "validation_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "target", "probability", "prediction", "novel_structure"])
        writer.writerows(zip(val_ids.tolist(), yv.tolist(), probability.tolist(), (probability >= threshold).astype(int).tolist(), novel.tolist()))
    np.savez_compressed(out / "per_model_probabilities.npz", ids=val_ids, probabilities=np.asarray(prediction_list),
                        model_names=np.asarray([row["file"] for row in per_model]), weights=np.asarray([row["ensemble_weight"] for row in per_model]))
    write_json(out / "metrics.json", metrics)
    write_json(out / "experiment.json", {"config": config, "per_model": per_model, "data_manifest": manifest,
        "python": platform.python_version(), "numpy": np.__version__, "script_sha256": digest(__file__), "limitations": protocol["limitations"]})
    write_json(out / "run_status.json", {"status": "completed", "search_fits": len(records), "refit_models": len(per_model),
        "test_accessed": False, "validation_opened_after_all_refits": True, "total_seconds": time.monotonic() - started})
    progress("final", len(records), {"validation_accuracy": metrics["accuracy"], "correct_count": metrics["correct_count"]})
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if "--output" in sys.argv:
            out = Path(sys.argv[sys.argv.index("--output") + 1])
            write_json(out / "run_status.json", {"status": "failed", "error": str(error)})
        raise
