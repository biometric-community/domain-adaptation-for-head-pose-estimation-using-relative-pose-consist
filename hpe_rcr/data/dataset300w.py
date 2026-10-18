"""300W images + landmark-derived Euler labels; source/target splits for DA."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .pose_from_landmarks import euler_from_landmarks, parse_pts


def resolve_300w_root(cfg: dict) -> Path:
    root = Path(cfg["data"]["root"])
    if not root.is_absolute():
        proj = Path(__file__).resolve().parents[2]
        cand = (proj / root).resolve()
        root = cand if cand.exists() else root
    for sub in (
        root / "extracted" / "300W",
        root / "300W",
        root,
    ):
        if (sub / "01_Indoor").is_dir() or (sub / "02_Outdoor").is_dir():
            return sub
    raise FileNotFoundError(f"300W not found under {cfg['data']['root']}")


def list_samples(root: Path) -> List[Tuple[Path, Path]]:
    pairs = []
    for split in ("01_Indoor", "02_Outdoor"):
        d = root / split
        if not d.is_dir():
            continue
        for img in sorted(list(d.glob("*.png")) + list(d.glob("*.jpg"))):
            pts = img.with_suffix(".pts")
            if pts.exists():
                pairs.append((img, pts))
    return pairs


def build_index(root: Path, cache: Path) -> List[dict]:
    if cache.exists():
        return json.loads(cache.read_text())
    rows = []
    for img, pts in list_samples(root):
        try:
            lm = parse_pts(pts)
            with Image.open(img) as im:
                w, h = im.size
            pose = euler_from_landmarks(lm, (h, w)).tolist()
            rows.append({"image": str(img), "pts": str(pts), "pose": pose})
        except Exception:
            continue
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows))
    return rows


class Face300W(Dataset):
    def __init__(
        self,
        rows: List[dict],
        transform=None,
        return_pose: bool = True,
    ):
        self.rows = rows
        self.return_pose = return_pose
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(r["image"]).convert("RGB")
        x = self.transform(img)
        out = {"image": x, "path": r["image"]}
        if self.return_pose:
            out["pose"] = torch.tensor(r["pose"], dtype=torch.float32)
        return out


def split_source_target(
    rows: List[dict],
    source_ratio: float = 0.6,
    seed: int = 42,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """source (labeled), target (unlabeled for CR), test (labeled eval)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(rows))
    rng.shuffle(idx)
    n = len(idx)
    n_src = int(n * source_ratio)
    n_tgt = int(n * 0.2)
    src = [rows[i] for i in idx[:n_src]]
    tgt = [rows[i] for i in idx[n_src : n_src + n_tgt]]
    test = [rows[i] for i in idx[n_src + n_tgt :]]
    return src, tgt, test
