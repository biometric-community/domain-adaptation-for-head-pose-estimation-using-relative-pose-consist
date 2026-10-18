"""REPORT.md + SVG figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hpe_rcr.plot_style import (
    BAR_COLOR, BAR_EDGE_COLOR, BAR_EDGE_WIDTH, FIGSIZE, LINEWIDTH_MAIN,
    MULTI_SERIES_COLORS, apply_rcparams, apply_style,
)


def load_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args(argv)
    cfg = yaml.safe_load(open(args.config))
    logs = Path(cfg["paths"]["log_dir"])
    figdir = Path(cfg["paths"]["figure_dir"])
    if not logs.is_absolute():
        logs = _ROOT / logs
    if not figdir.is_absolute():
        figdir = _ROOT / figdir
    figdir.mkdir(parents=True, exist_ok=True)
    hist = load_json(logs / "train_history.json") or []
    summary = load_json(logs / "train_summary.json") or {}
    ev = load_json(logs / "eval_results.json") or {}
    size = load_json(logs / "dataset_size.json") or {}
    metrics = ev if ev.get("MAE") is not None else (summary.get("best") or {})
    figs = []
    if metrics:
        apply_rcparams()
        fig, ax = plt.subplots(figsize=FIGSIZE)
        labs = ["MAE", "pitch", "yaw", "roll"]
        vals = [float(metrics.get(k, 0)) for k in labs]
        ax.bar(np.arange(4), vals, 0.55, color=BAR_COLOR, edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH)
        ax.set_xticks(np.arange(4)); ax.set_xticklabels(labs)
        ax.set_ylabel("Error (deg)"); ax.set_title("300W hold-out HPE (Ours)")
        apply_style(ax, grid_axis="y"); fig.tight_layout()
        out = figdir / "mae_angles.svg"; fig.savefig(out, format="svg"); plt.close(fig)
        figs.append(out)
    if hist:
        apply_rcparams()
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.plot([h["epoch"] for h in hist], [h.get("MAE", 0) for h in hist],
                color=BAR_COLOR, linewidth=LINEWIDTH_MAIN, label="MAE")
        ax.plot([h["epoch"] for h in hist], [h.get("loss", 0) for h in hist],
                color=MULTI_SERIES_COLORS[1], linewidth=LINEWIDTH_MAIN, label="Train loss")
        ax.set_xlabel("Epoch"); ax.set_title("Training curves"); ax.legend()
        apply_style(ax); fig.tight_layout()
        out = figdir / "train_curves.svg"; fig.savefig(out, format="svg"); plt.close(fig)
        figs.append(out)

    lines = [
        "# Report: Domain Adaptation for Head Pose Estimation (RCRw)",
        "",
        "Rebuild of Kuhnke & Ostermann, TBIOM 2023 "
        "(relative pose consistency). DOI from paper folder analysis.",
        "",
        "## Setup",
        "",
        f"- ResNet-18 → 3 Euler angles; Adam lr={cfg['train']['lr']}; λ→{cfg['train'].get('lambda_max', 0.2)}",
        f"- Data: 300W (landmark→solvePnP poses); size gate {size.get('decision','n/a')} ({size.get('total_gib','?')} GiB)",
        "- Biwi+ / SynHead++ / 300W-LP not registered — see `DEVIATIONS.md`",
        "",
        "## Hold-out results (degrees)",
        "",
        "| | MAE | pitch | yaw | roll |",
        "|--|--:|--:|--:|--:|",
        f"| **Ours** | {float(metrics.get('MAE', float('nan'))):.2f} | "
        f"{float(metrics.get('pitch', float('nan'))):.2f} | "
        f"{float(metrics.get('yaw', float('nan'))):.2f} | "
        f"{float(metrics.get('roll', float('nan'))):.2f} |",
        "",
        "## Figures",
        "",
    ]
    root = (_ROOT / "REPORT.md").resolve().parent
    for fp in figs:
        rel = fp.resolve().relative_to(root)
        lines += [f"### `{rel}`", "", f"![]({rel})", ""]
    lines += ["## Regenerate", "", "```bash", "bash scripts/report.sh", "```", ""]
    (_ROOT / "REPORT.md").write_text("\n".join(lines))
    print(f"Wrote {_ROOT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
