"""Predict pitch/yaw/roll for images in a folder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hpe_rcr.models import build_pose_resnet18


class Folder(Dataset):
    def __init__(self, root, tf):
        self.paths = sorted(list(Path(root).glob("*.jpg")) + list(Path(root).glob("*.png")))
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return {"image": self.tf(Image.open(self.paths[i]).convert("RGB")), "path": str(self.paths[i])}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    p.add_argument("--input", required=True)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    import yaml

    cfg = yaml.safe_load(open(args.config))
    device = torch.device(args.device or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    model = build_pose_resnet18(pretrained=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in DataLoader(Folder(args.input, tf), batch_size=16):
            pred = model(batch["image"].to(device)).cpu()
            for path, pose in zip(batch["path"], pred):
                rows.append({"path": path, "pitch": float(pose[0]), "yaw": float(pose[1]), "roll": float(pose[2])})
    out = Path(cfg["paths"]["pred_dir"]) / "predictions.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {len(rows)} → {out}")


if __name__ == "__main__":
    main()
