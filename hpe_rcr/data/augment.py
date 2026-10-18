"""Pose-preserving and pose-altering augmentations with relative Euler offsets."""

from __future__ import annotations

import math
import random
from typing import Tuple

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance, ImageFilter


def _color_jitter(img: Image.Image) -> Image.Image:
    if random.random() < 0.8:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.6, 1.4))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.6, 1.4))
        img = ImageEnhance.Color(img).enhance(random.uniform(0.6, 1.4))  # saturation
    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.5)))
    return img


def augment_pair(
    img: Image.Image,
    size: int = 224,
) -> Tuple[Image.Image, Image.Image, torch.Tensor, torch.Tensor]:
    """
    Two augmented views + relative pose deltas (pitch,yaw,roll) applied to labels.
    Convention: pose' = pose * sign + delta, with sign=(-1 on yaw/roll if flipped).
    Returns (view_a, view_b, delta_a, delta_b) where delta is additive in degrees
    after applying sign flips to the base pose.
    Also returns sign tensors sa, sb for (pitch,yaw,roll).
    """
    # We return views as PIL; caller converts. Relative: flip negates yaw&roll; rotation adds to roll.
    def one_view():
        im = img.copy()
        # scale / crop-ish via resize then random crop
        scale = random.uniform(0.8, 1.2)
        w, h = im.size
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        im = im.resize((nw, nh), Image.BILINEAR)
        # pad if needed
        if nw < size or nh < size:
            pad = Image.new("RGB", (max(nw, size), max(nh, size)), (0, 0, 0))
            pad.paste(im, ((pad.size[0] - nw) // 2, (pad.size[1] - nh) // 2))
            im = pad
            nw, nh = im.size
        left = random.randint(0, max(0, nw - size))
        top = random.randint(0, max(0, nh - size))
        im = im.crop((left, top, left + size, top + size))
        im = _color_jitter(im)
        flip = random.random() < 0.5
        if flip:
            im = TF.hflip(im)
        rot = random.uniform(-30.0, 30.0)
        if abs(rot) > 89:
            rot = -rot
        im = TF.rotate(im, rot, fill=0)
        # signs and delta relative to original pose label
        sign = torch.tensor([1.0, -1.0 if flip else 1.0, -1.0 if flip else 1.0])
        delta = torch.tensor([0.0, 0.0, rot])  # rotation adds to roll
        return im, sign, delta

    a, sa, da = one_view()
    b, sb, db = one_view()
    return a, b, sa, da, sb, db


def apply_pose_transform(pose: torch.Tensor, sign: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """pose_aug = pose * sign + delta (degrees)."""
    return pose * sign + delta
