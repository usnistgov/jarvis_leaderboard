"""Generate JARVIS exfoliation predictions from the pinned public structures."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import sys
import urllib.request
import zipfile
from importlib.metadata import version
from pathlib import Path


HERE = Path(__file__).resolve().parent
CSV_NAME = "AI-SinglePropertyPrediction-exfoliation_energy-dft_3d-test-mae.csv"
DATA_URL = "https://figshare.com/ndownloader/files/38521619"
DATA_MEMBER = "jdft_3d-12-12-2022.json"
DATA_SHA256 = "d4c64660e9e1fa45c82bd8868a96ec10162195eed69636445972c05550d8d0d6"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_versions() -> None:
    expected = {
        "numpy": "1.26.4", "pandas": "2.3.3", "torch": "2.5.1",
        "pymatgen": "2025.10.7", "scipy": "1.15.2", "monty": "2025.3.3",
    }
    for package, wanted in expected.items():
        actual = version(package).split("+", 1)[0]
        if actual != wanted:
            raise RuntimeError(f"{package}: expected {wanted}, found {actual}; use requirements.txt")


def load_ids() -> list[str]:
    ids = json.loads((HERE / "test_ids.json").read_text(encoding="utf-8"))
    if (not isinstance(ids, list) or len(ids) != 81
            or any(not isinstance(jid, str) or not jid.strip() for jid in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("Expected 81 unique test IDs")
    return ids


def dataset_zip(provided: Path | None, cache_dir: Path) -> Path:
    path = provided.resolve() if provided else cache_dir.resolve() / (DATA_MEMBER + ".zip")
    if not path.is_file():
        if provided:
            raise FileNotFoundError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_suffix(path.suffix + ".download")
        print("Downloading the pinned JARVIS dft_3d structure archive...", flush=True)
        request = urllib.request.Request(DATA_URL, headers={"User-Agent": "JARVIS-V7-inference/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, pending.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        if sha256(pending) != DATA_SHA256:
            raise ValueError("Downloaded dataset checksum mismatch")
        pending.replace(path)
    if sha256(path) != DATA_SHA256:
        raise ValueError("Dataset checksum mismatch")
    return path


def read_structures(path: Path, ids: list[str]) -> list[dict]:
    wanted, selected = set(ids), {}
    with zipfile.ZipFile(path) as archive, archive.open(DATA_MEMBER) as stream:
        # The pinned official archive contains NaN tokens. Python's decoder
        # preserves the behavior of the original structure preparation script.
        records = json.load(stream)
        if not isinstance(records, list):
            raise ValueError("Expected a JARVIS record list")
        for record in records:
            jid = record.get("jid")
            if jid not in wanted:
                continue
            if jid in selected:
                raise ValueError(f"Duplicate structure ID: {jid}")
            formula = (record.get("formula") or record.get("formula_pretty")
                       or record.get("reduced_formula"))
            atoms = record.get("atoms")
            if not formula or not isinstance(atoms, dict):
                raise ValueError(f"Missing formula or structure: {jid}")
            # Only these three fields enter the feature extractor.
            selected[jid] = {"jid": jid, "formula": formula, "atoms": atoms}
    missing = wanted.difference(selected)
    if missing:
        raise ValueError(f"Missing structure IDs: {sorted(missing)}")
    return [selected[jid] for jid in ids]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-zip", type=Path, help="Optional local copy of the pinned JARVIS archive")
    parser.add_argument("--cache-dir", type=Path, default=HERE / ".cache")
    parser.add_argument("--out-dir", type=Path, default=HERE / "reproduced",
                        help="New output directory (must not already exist)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--profile", choices=["portable_tanh", "historical_cuda"],
                        default="portable_tanh",
                        help="CPU-compatible tanh GELU, or original CUDA activation path")
    args = parser.parse_args()
    output = args.out_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new --out-dir: {output}")
    if args.profile == "historical_cuda" and args.device != "cuda":
        raise ValueError("Use --device cuda for --profile historical_cuda")
    check_versions()

    import pandas as pd
    from featurize import featurize_records
    from model_inference import predict_feature_frame

    ids = load_ids()
    records = read_structures(dataset_zip(args.data_zip, args.cache_dir), ids)
    print(f"Generating descriptors for {len(ids)} structures...", flush=True)
    frame = featurize_records(records)
    # Preserve the historical feature CSV write/read preprocessing boundary.
    feature_csv = frame.to_csv(index=False, lineterminator="\n")
    frame = pd.read_csv(io.StringIO(feature_csv), dtype={"sample_id": str})
    predictions = predict_feature_frame(frame, HERE / "model", device=args.device, profile=args.profile)
    result = pd.DataFrame({"jid": ids, "prediction": predictions})
    # Preserve Stage38's prediction CSV write/read/write serialization.
    first_csv = result.to_csv(index=False, lineterminator="\n")
    csv_bytes = pd.read_csv(io.StringIO(first_csv)).to_csv(index=False, lineterminator="\n").encode("utf-8")
    output.mkdir(parents=True, exist_ok=False)
    with (output / CSV_NAME).open("xb") as handle:
        handle.write(csv_bytes)
    member = zipfile.ZipInfo(CSV_NAME, date_time=(1980, 1, 1, 0, 0, 0))
    member.compress_type = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(output / (CSV_NAME + ".zip"), "x") as archive:
        archive.writestr(member, csv_bytes)
    print(f"Predictions written to {output / (CSV_NAME + '.zip')}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
