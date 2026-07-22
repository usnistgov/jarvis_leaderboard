from __future__ import annotations

import argparse
import json
from pathlib import Path

from .model import ClosureGapModel


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_train = sub.add_parser("train")
    p_train.add_argument("csv", type=Path)
    p_train.add_argument("model", type=Path)
    p_predict = sub.add_parser("predict")
    p_predict.add_argument("model", type=Path)
    p_predict.add_argument("formula")
    args = parser.parse_args()
    if args.cmd == "train":
        model = ClosureGapModel.train_csv(args.csv)
        model.save(args.model)
        print(json.dumps({"saved": str(args.model)}, sort_keys=True))
    else:
        model = ClosureGapModel.load(args.model)
        print(json.dumps(model.predict(args.formula), sort_keys=True))


if __name__ == "__main__":
    main()
