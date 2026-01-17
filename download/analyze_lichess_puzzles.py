from collections import Counter


def main() -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install with: python -m pip install datasets"
        ) from exc

    ds = load_dataset("Lichess/chess-puzzles")
    split_names = list(ds.keys())

    mate_total = 0
    mate_in_counts = Counter()

    for split in split_names:
        for row in ds[split]:
            themes = row.get("Themes") or []
            if "mate" in themes:
                mate_total += 1
            for theme in themes:
                if theme.startswith("mateIn"):
                    mate_in_counts[theme] += 1

    print("splits:", split_names)
    print("mate_total:", mate_total)
    print("mate_in_counts:")
    for key in sorted(mate_in_counts, key=lambda x: int(x.replace("mateIn", ""))):
        print(f"{key}: {mate_in_counts[key]}")


if __name__ == "__main__":
    main()
