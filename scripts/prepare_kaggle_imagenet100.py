import argparse
import json
from pathlib import Path
import random


EXTENSIONS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Create a fixed ImageFolder subset using symlinks")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--classes", type=int, default=20)
    parser.add_argument("--train-per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.classes < 1 or args.train_per_class < 1:
        parser.error("classes and train-per-class must be positive")
    root = args.input.resolve()
    if not root.is_dir():
        parser.error(f"dataset directory does not exist: {root}")

    roots = {}
    for split in ("train", "val"):
        candidates = [root / split, *sorted(root.glob(f"{split}.X*"))]
        roots[split] = [path for path in candidates if path.is_dir()]
    if not all(roots.values()):
        children = [path for path in root.iterdir() if path.is_dir()]
        if len(children) == 1:
            root = children[0]
            for split in ("train", "val"):
                candidates = [root / split, *sorted(root.glob(f"{split}.X*"))]
                roots[split] = [path for path in candidates if path.is_dir()]
    if not all(roots.values()):
        parser.error("expected train or train.X* and val or val.X* directories")

    class_dirs = {}
    for split in ("train", "val"):
        class_dirs[split] = {}
        for split_root in roots[split]:
            for path in split_root.iterdir():
                if path.is_dir():
                    class_dirs[split].setdefault(path.name, []).append(path)
    available = sorted(set(class_dirs["train"]) & set(class_dirs["val"]))
    if len(available) < args.classes:
        parser.error(f"only {len(available)} classes occur in both train and val")
    rng = random.Random(args.seed)
    selected = sorted(rng.sample(available, args.classes))
    selection = {"source": str(root), "seed": args.seed, "classes": selected,
                 "train_per_class": args.train_per_class, "train": {}, "val": {}}
    for name in selected:
        for split in ("train", "val"):
            files = sorted(path for directory in class_dirs[split][name]
                           for path in directory.rglob("*")
                           if path.is_file() and path.suffix.lower() in EXTENSIONS)
            if split == "train" and len(files) < args.train_per_class:
                parser.error(f"{name} has only {len(files)} train images")
            if split == "val" and not files:
                parser.error(f"{name} has no validation images")
            if split == "train":
                files = sorted(rng.sample(files, args.train_per_class))
            selection[split][name] = [[str(path.relative_to(root)), path.stat().st_size]
                                      for path in files]

    train_files = {row[0] for rows in selection["train"].values() for row in rows}
    val_files = {row[0] for rows in selection["val"].values() for row in rows}
    if train_files & val_files:
        parser.error("train and val contain overlapping files")
    if args.manifest.exists():
        if json.loads(args.manifest.read_text(encoding="utf-8")) != selection:
            parser.error("selected data differ from the existing manifest")
    if args.output.exists():
        parser.error(f"output directory already exists: {args.output}")

    for split in ("train", "val"):
        for name, rows in selection[split].items():
            destination = args.output / split / name
            destination.mkdir(parents=True)
            for index, (relative, _) in enumerate(rows):
                source = root / relative
                (destination / f"{index:04d}{source.suffix.lower()}").symlink_to(source)
    if not args.manifest.exists():
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(f"Selected {len(selected)} classes, {len(train_files)} train and {len(val_files)} val images")
    print(f"Data: {args.output.resolve()}; manifest: {args.manifest.resolve()}")


if __name__ == "__main__":
    main()
