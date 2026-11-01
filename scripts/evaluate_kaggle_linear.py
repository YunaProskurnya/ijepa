import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.encoder import VisionTransformer
from src.transforms import make_evaluation_transform


@torch.inference_mode()
def extract(encoder, loader, device):
    features, labels = [], []
    for images, targets in loader:
        values = encoder(images.to(device, non_blocking=True)).float().mean(dim=1).cpu()
        if not torch.isfinite(values).all():
            raise RuntimeError("non-finite encoder features")
        features.append(values)
        labels.append(targets)
    return torch.cat(features), torch.cat(labels)


@torch.inference_mode()
def score(head, features, labels, device, batch_size):
    total_loss, correct = 0.0, 0
    head.eval()
    for start in range(0, len(features), batch_size):
        values = features[start:start + batch_size].to(device)
        targets = labels[start:start + batch_size].to(device)
        logits = head(values)
        total_loss += F.cross_entropy(logits, targets, reduction="sum").item()
        correct += (logits.argmax(dim=1) == targets).sum().item()
    return {"loss": total_loss / len(features), "top1": 100 * correct / len(features)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fixed linear probe of a frozen joint encoder")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--probe-epochs", type=int, default=30)
    parser.add_argument("--probe-lr", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if min(args.batch_size, args.probe_batch_size, args.probe_epochs) < 1 or args.num_workers < 0:
        parser.error("batch sizes and probe epochs must be positive; workers nonnegative")
    if not 0 < args.probe_lr < float("inf"):
        parser.error("probe-lr must be positive and finite")
    if args.output.exists():
        parser.error("output already exists")
    if args.train.resolve() == args.val.resolve():
        parser.error("train and val must differ")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["config"].get("mode") != "joint":
        parser.error("this probe expects a joint encoder checkpoint")
    model_options = checkpoint["config"]["encoder"]
    transform = make_evaluation_transform(model_options["image_size"])
    train = datasets.ImageFolder(args.train, transform=transform)
    val = datasets.ImageFolder(args.val, transform=transform)
    if not train.samples or not val.samples or train.class_to_idx != val.class_to_idx:
        parser.error("train and val need the same nonempty ImageFolder classes")
    if checkpoint["config"].get("selected_classes") != train.classes:
        parser.error("probe classes differ from checkpoint training classes")

    encoder = VisionTransformer(**model_options)
    encoder.load_state_dict(checkpoint["encoder"])
    encoder.to(device).eval().requires_grad_(False)
    del checkpoint
    options = {"batch_size": args.batch_size, "num_workers": args.num_workers,
               "pin_memory": device.type == "cuda"}
    train_features, train_labels = extract(encoder, DataLoader(train, **options), device)
    val_features, val_labels = extract(encoder, DataLoader(val, **options), device)
    del encoder

    torch.manual_seed(args.seed)
    head = nn.Linear(train_features.shape[1], len(train.classes)).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=args.probe_lr)
    generator = torch.Generator().manual_seed(args.seed)
    for _ in range(args.probe_epochs):
        head.train()
        for indices in torch.randperm(len(train_features), generator=generator).split(args.probe_batch_size):
            values = train_features[indices].to(device)
            targets = train_labels[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(values), targets)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite probe loss")
            loss.backward()
            optimizer.step()

    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "train": str(args.train.resolve()), "val": str(args.val.resolve()),
        "classes": train.classes, "train_images": len(train), "val_images": len(val),
        "protocol": {"feature": "mean final encoder patch tokens", "transform": "resize and center crop",
                     "head": "linear with bias", "optimizer": "Adam", "epochs": args.probe_epochs,
                     "lr": args.probe_lr, "batch_size": args.probe_batch_size, "seed": args.seed},
        "train_metrics": score(head, train_features, train_labels, device, args.probe_batch_size),
        "val_metrics": score(head, val_features, val_labels, device, args.probe_batch_size),
    }
    if args.manifest:
        result["manifest_sha256"] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"train_top1={result['train_metrics']['top1']:.2f}% "
          f"val_top1={result['val_metrics']['top1']:.2f}%")
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
