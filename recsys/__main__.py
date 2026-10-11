"""Run with python -m recsys download|evaluate|demo."""

import argparse
import csv
import json
from pathlib import Path

from recsys.data import DEFAULT_DATA_DIR, chronological_split, download_dataset, load_dataset
from recsys.evaluate import run_experiment
from recsys.models import METHODS, Recommender
from recsys.robustness import run_shilling_experiment


def main():
    parser = argparse.ArgumentParser(description="Block 1, Part 3: movie recommendations")
    subparsers = parser.add_subparsers(dest="command", required=True)
    download = subparsers.add_parser("download", help="Download and verify MovieLens 100K")
    download.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    evaluate = subparsers.add_parser("evaluate", help="Run all four methods on held-out data")
    evaluate.add_argument("--output", type=Path, default=Path("artifacts/recsys"))
    audit = subparsers.add_parser("audit", help="Offline sensitivity to synthetic rating profiles")
    audit.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    audit.add_argument("--output", type=Path, default=Path("artifacts/recsys/shilling.json"))
    demo = subparsers.add_parser("demo", help="Print real recommendations and explanations")
    demo.add_argument("--user-id", type=int, default=1)
    demo.add_argument("--method", choices=METHODS, default="item_cf")
    demo.add_argument("--genre")
    for command in (evaluate, demo):
        command.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
        command.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    try:
        if args.command == "download":
            print(f"Verified MovieLens downloaded to {download_dataset(args.data_dir)}")
            return
        dataset = load_dataset(args.data_dir)
        if args.command == "audit":
            result = run_shilling_experiment(dataset)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        if args.command == "demo":
            train, _, _ = chronological_split(dataset.ratings)
            result = Recommender(dataset, train).recommend(args.user_id, args.method, args.k, args.genre)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        result = run_experiment(dataset, args.k)
        args.output.mkdir(parents=True, exist_ok=True)
        per_user = result.pop("per_user")
        result["per_user_file"] = "per_user.csv"
        with (args.output / "per_user.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, lineterminator="\n", fieldnames=["method", "user_id", "relevant_count",
                                    "recommendations", "precision", "recall", "ndcg", "hit_rate"])
            writer.writeheader()
            for method, rows in per_user.items():
                for row in rows:
                    writer.writerow({"method": method, **row,
                                     "recommendations": " ".join(map(str, row["recommendations"]))})
        (args.output / "results.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        fieldnames = ["method", *next(iter(result["metrics"].values())).keys()]
        with (args.output / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for method, metrics in result["metrics"].items():
                writer.writerow({"method": method, **metrics})
        print(json.dumps({"protocol": result["protocol"], "metrics": result["metrics"]},
                         ensure_ascii=False, indent=2))
        print(f"Full results saved to {args.output}")
    except (FileNotFoundError, ValueError, OSError) as error:
        parser.exit(1, f"RecSys: {error}\n")


if __name__ == "__main__":
    main()
