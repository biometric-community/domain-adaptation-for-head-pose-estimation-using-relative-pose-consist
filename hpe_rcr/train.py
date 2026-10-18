"""Train ResNet18 HPE with relative pose consistency on 300W."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hpe_rcr.data.augment import apply_pose_transform, augment_pair
from hpe_rcr.data.dataset300w import build_index, resolve_300w_root, split_source_target
from hpe_rcr.losses import pose_mse, relative_consistency_loss
from hpe_rcr.models import build_pose_resnet18

_NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def to_tensor(pil: Image.Image) -> torch.Tensor:
    return _NORM(transforms.ToTensor()(pil))


@torch.no_grad()
def set_bn_eval(model: torch.nn.Module):
    """Force BN to use running stats (source-only BN on target)."""
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()


class PathPoseDS(Dataset):
    """Lightweight index over preloaded PIL images."""

    def __init__(self, rows, images, with_pose: bool = True):
        self.rows = rows
        self.images = images  # list[PIL] aligned with rows
        self.with_pose = with_pose

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        out = {"idx": i, "path": self.rows[i]["image"]}
        if self.with_pose:
            out["pose"] = torch.tensor(self.rows[i]["pose"], dtype=torch.float32)
        return out


def preload_pils(rows, short_side: int = 256):
    """Load RGB and downscale so pose augs stay cheap on large 300W photos."""
    out = []
    for r in rows:
        im = Image.open(r["image"]).convert("RGB")
        w, h = im.size
        if min(w, h) > short_side:
            if w < h:
                im = im.resize((short_side, int(h * short_side / w)), Image.BILINEAR)
            else:
                im = im.resize((int(w * short_side / h), short_side), Image.BILINEAR)
        out.append(im)
    return out


def collate_idx(batch):
    out = {"idx": [b["idx"] for b in batch], "path": [b["path"] for b in batch]}
    if "pose" in batch[0]:
        out["pose"] = torch.stack([b["pose"] for b in batch])
    return out


def run_eval(model, loader, images, device) -> dict:
    model.eval()
    errs = []
    tf = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), _NORM]
    )
    with torch.no_grad():
        for batch in loader:
            xs, ys = [], []
            for i, pose in zip(batch["idx"], batch["pose"]):
                xs.append(tf(images[i]))
                ys.append(pose)
            x = torch.stack(xs).to(device)
            y = torch.stack(ys).to(device)
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
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    for k in ("checkpoint_dir", "log_dir", "pred_dir", "figure_dir"):
        Path(cfg["paths"][k]).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    root = resolve_300w_root(cfg)
    cache = Path(cfg["paths"]["log_dir"]) / "pose_index.json"
    rows = build_index(root, cache)
    if cfg["data"].get("max_samples"):
        rows = rows[: int(cfg["data"]["max_samples"])]
    src_rows, tgt_rows, test_rows = split_source_target(
        rows, source_ratio=float(cfg["data"].get("source_ratio", 0.6)), seed=int(cfg["train"].get("seed", 42))
    )
    print(f"[data] total={len(rows)} source={len(src_rows)} target={len(tgt_rows)} test={len(test_rows)}", flush=True)
    print(f"[device] {device}", flush=True)

    print("[data] preloading PIL…", flush=True)
    src_imgs = preload_pils(src_rows)
    tgt_imgs = preload_pils(tgt_rows)
    test_imgs = preload_pils(test_rows)

    bs = int(cfg["train"]["batch_size"])
    src_loader = DataLoader(
        PathPoseDS(src_rows, src_imgs, True),
        batch_size=bs, shuffle=True, num_workers=0, drop_last=True, collate_fn=collate_idx,
    )
    tgt_loader = DataLoader(
        PathPoseDS(tgt_rows, tgt_imgs, False),
        batch_size=bs, shuffle=True, num_workers=0, drop_last=True, collate_fn=collate_idx,
    )
    test_loader = DataLoader(
        PathPoseDS(test_rows, test_imgs, True),
        batch_size=int(cfg["eval"].get("batch_size", 32)), shuffle=False, num_workers=0, collate_fn=collate_idx,
    )

    model = build_pose_resnet18(pretrained=bool(cfg["model"].get("pretrained", True))).to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
    )
    epochs = int(cfg["train"]["epochs"])
    lambda_max = float(cfg["train"].get("lambda_max", 0.2))
    max_steps = cfg["train"].get("max_steps")
    history, best, global_step = [], {"MAE": 1e9}, 0
    tgt_iter = iter(tgt_loader)

    for epoch in range(epochs):
        t0 = time.time()
        lam = lambda_max * min(1.0, (epoch + 1) / max(1, epochs // 2))
        model.train()
        running, n = 0.0, 0
        for sbatch in src_loader:
            imgs, poses = [], []
            for i, pose in zip(sbatch["idx"], sbatch["pose"]):
                a, b, sa, da, sb, db = augment_pair(src_imgs[i], size=224)
                imgs.append(to_tensor(a))
                poses.append(apply_pose_transform(pose, sa, da))
            x = torch.stack(imgs).to(device, non_blocking=True)
            y = torch.stack(poses).to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss_sup = pose_mse(pred, y)

            try:
                tbatch = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                tbatch = next(tgt_iter)
            ta, tb, signs_a, deltas_a, signs_b, deltas_b = [], [], [], [], [], []
            for i in tbatch["idx"]:
                a, b, sa, da, sb, db = augment_pair(tgt_imgs[i], size=224)
                ta.append(to_tensor(a))
                tb.append(to_tensor(b))
                signs_a.append(sa)
                signs_b.append(sb)
                deltas_a.append(da)
                deltas_b.append(db)
            xa = torch.stack(ta).to(device, non_blocking=True)
            xb = torch.stack(tb).to(device, non_blocking=True)
            sa = torch.stack(signs_a).to(device)
            sb = torch.stack(signs_b).to(device)
            da = torch.stack(deltas_a).to(device)
            db = torch.stack(deltas_b).to(device)
            set_bn_eval(model)
            pred_a = model(xa)
            pred_b = model(xb)
            loss_cr = relative_consistency_loss(
                pred_a, pred_b, sa, sb, da, db,
                weighted=bool(cfg["train"].get("weighted_rcr", True)),
            )
            loss = loss_sup + lam * loss_cr
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"].get("grad_clip", 5.0)))
            opt.step()
            model.train()
            running += float(loss.detach())
            n += 1
            global_step += 1
            if max_steps is not None and global_step >= int(max_steps):
                break

        metrics = run_eval(model, test_loader, test_imgs, device)
        row = {"epoch": epoch + 1, "loss": running / max(n, 1), "lambda": lam, "sec": time.time() - t0, **metrics}
        history.append(row)
        print(
            f"[epoch] {epoch+1}/{epochs} loss={row['loss']:.4f} λ={lam:.3f} "
            f"MAE={metrics['MAE']:.2f} (p={metrics['pitch']:.2f} y={metrics['yaw']:.2f} r={metrics['roll']:.2f}) "
            f"[{row['sec']:.1f}s]",
            flush=True,
        )
        if metrics["MAE"] < best["MAE"]:
            best = {**metrics, "epoch": epoch + 1}
            torch.save({"model": model.state_dict(), "best": best, "cfg": cfg}, Path(cfg["paths"]["checkpoint_dir"]) / "best.pt")
        (Path(cfg["paths"]["log_dir"]) / "train_history.json").write_text(json.dumps(history, indent=2))
        torch.save({"model": model.state_dict(), "history": history, "best": best}, Path(cfg["paths"]["checkpoint_dir"]) / "last.pt")
        if max_steps is not None and global_step >= int(max_steps):
            break

    summary = {
        "best": best,
        "n_source": len(src_rows),
        "n_target": len(tgt_rows),
        "n_test": len(test_rows),
        "device": str(device),
        "note": "300W landmark-derived poses; Biwi+/SynHead++ blocked — see DEVIATIONS.md",
    }
    (Path(cfg["paths"]["log_dir"]) / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[done] best={best}", flush=True)


if __name__ == "__main__":
    main()
