"""Evaluate head-pose MAE on 300W hold-out."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hpe_rcr.data.dataset300w import Face300W, build_index, resolve_300w_root, split_source_target
from hpe_rcr.models import build_pose_resnet18


def collate_source(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "pose": torch.stack([b["pose"] for b in batch]),
        "path": [b["path"] for b in batch],
    }


@torch.no_grad()
def eval_loader(model, loader, device) -> dict:
    model.eval()
    errs = []
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["pose"].to(device)
        pred = model(x)
        errs.append((pred - y).abs().cpu())
    e = torch.cat(errs, 0)
    mae = e.mean(dim=0)
    return {
        "MAE": float(e.mean()),
        "pitch": float(mae[0]),
        "yaw": float(mae[1]),
        "roll": float(mae[2]),
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    cfg = yaml.safe_load(open(args.config))
    device = torch.device(args.device or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    root = resolve_300w_root(cfg)
    rows = build_index(root, Path(cfg["paths"]["log_dir"]) / "pose_index.json")
    if cfg["data"].get("max_samples"):
        rows = rows[: int(cfg["data"]["max_samples"])]
    _, _, test_rows = split_source_target(
        rows, float(cfg["data"].get("source_ratio", 0.6)), int(cfg["train"].get("seed", 42))
    )
    tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    loader = DataLoader(
        Face300W(test_rows, transform=tf),
        batch_size=int(cfg["eval"].get("batch_size", 32)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_source,
    )
    model = build_pose_resnet18(pretrained=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    metrics = eval_loader(model, loader, device)
    metrics["checkpoint"] = args.checkpoint
    metrics["n_test"] = len(test_rows)
    Path(cfg["paths"]["log_dir"]).mkdir(parents=True, exist_ok=True)
    (Path(cfg["paths"]["log_dir"]) / "eval_results.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
