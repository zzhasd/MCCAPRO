from __future__ import annotations

import argparse

from mcpp_compare import run_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MCPP baseline comparisons and write an objective report.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for CSV metrics and markdown report.")
    parser.add_argument(
        "--skip-visualizations",
        action="store_true",
        help="Write metrics and report without regenerating visualization PNG files.",
    )
    args = parser.parse_args()
    paths = run_all(args.output_dir, include_visualizations=not args.skip_visualizations)
    print("Generated comparison artifacts:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
