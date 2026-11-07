"""MSE + weighted relative pose consistency (RCRw)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pose_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def rcr_weights(pose: torch.Tensor) -> torch.Tensor:
    """Eqs. 5-6: weight each angle by how rare it is in the batch, rescaled to a mean of one."""
    z = (pose - pose.mean(dim=0)) / pose.std(dim=0).clamp_min(1e-6)
    w = 1.0 - torch.exp(-(z**2))
    return w / w.sum(dim=0).clamp_min(1e-6) * pose.size(0)


def relative_consistency_loss(
    pred_a: torch.Tensor,
    pred_b: torch.Tensor,
    sign_a: torch.Tensor,
    sign_b: torch.Tensor,
    delta_a: torch.Tensor,
    delta_b: torch.Tensor,
    weighted: bool = True,
) -> torch.Tensor:
    """
    Map both predictions back to a common frame and MSE.
    Inverse of pose_aug = pose * sign + delta  →  pose = (pose_aug - delta) * sign
    (since sign is ±1).
    """
    pose_a = (pred_a - delta_a) * sign_a
    pose_b = (pred_b - delta_b) * sign_b
    err = (pose_a - pose_b).pow(2)
    if weighted:
        # weights come from the mean of the mapped pose predictions
        mid = 0.5 * (pose_a + pose_b)
        w = rcr_weights(mid)
        return (err * w).mean()
    return err.mean()
