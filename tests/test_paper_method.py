"""Relative pose consistency, augmentations and labels, checked for mutual agreement.

Abstract: the relative pose created by pose-altering augmentations is the training
signal, so a label change must match what an augmentation does to the image.
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import pytest
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from hpe_rcr.data import augment
from hpe_rcr.data.augment import apply_pose_transform, augment_pair
from hpe_rcr.data.dataset300w import split_source_target
from hpe_rcr.data.pose_from_landmarks import _LM_IDX, _MODEL_3D, euler_from_landmarks
from hpe_rcr.losses import rcr_weights, relative_consistency_loss
from hpe_rcr.train import set_bn_eval

ROOT = Path(__file__).resolve().parents[1]
SIZE = 1000


def landmarks(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """Project the canonical 3D face at a known pose into a 68-point array."""
    rot = lambda axis, deg: cv2.Rodrigues(np.radians(deg) * np.eye(3)[axis])[0]
    camera = np.array([[SIZE, 0, SIZE / 2], [0, SIZE, SIZE / 2], [0, 0, 1.0]])
    model_to_camera = rot(2, roll) @ rot(1, yaw) @ rot(0, pitch) @ np.diag([1.0, -1.0, -1.0])
    points = (model_to_camera @ _MODEL_3D.T).T + np.array([0.0, 0.0, 2500.0])
    projected = (camera @ points.T).T
    pts = np.zeros((68, 2))
    pts[_LM_IDX] = projected[:, :2] / projected[:, 2:]
    return pts


def label(pts: np.ndarray) -> np.ndarray:
    return euler_from_landmarks(pts, (SIZE, SIZE))


def test_labels_recover_a_known_pose():
    assert label(landmarks(5, 12, 3)) == pytest.approx([5, 12, 3], abs=0.5)


def test_frontal_face_is_near_zero():
    assert label(landmarks(0, 0, 0)) == pytest.approx([0, 0, 0], abs=0.5)


def test_tf_rotate_turns_images_counter_clockwise():
    image = Image.new("L", (101, 101))
    image.putpixel((90, 50), 255)  # right of centre
    rotated = np.asarray(TF.rotate(image, 90.0))
    y, x = np.unravel_index(rotated.argmax(), rotated.shape)
    assert y < 50 and abs(x - 50) <= 1  # now above centre


def rotate_counter_clockwise(pts: np.ndarray, degrees: float) -> np.ndarray:
    t, c = math.radians(degrees), np.array([SIZE / 2, SIZE / 2])
    d = pts - c
    return np.stack(
        [d[:, 0] * math.cos(t) + d[:, 1] * math.sin(t), -d[:, 0] * math.sin(t) + d[:, 1] * math.cos(t)], axis=1
    ) + c


def augmentation_with(rotation: float, flip: bool):
    """Run augment_pair with a fixed rotation and flip; return view A's sign and delta."""
    uniform = lambda a, b: rotation if (a, b) == (-30.0, 30.0) else 1.0
    with mock.patch.object(augment.random, "uniform", uniform), mock.patch.object(
        augment.random, "random", lambda: 0.0 if flip else 0.99
    ):
        _, _, sign, delta, _, _ = augment_pair(Image.new("RGB", (240, 240)), size=224)
    return sign, delta


def test_rotation_delta_matches_the_label_of_a_rotated_face():
    """Rotating the image must change the roll label by the recorded delta."""
    pts = landmarks(5, 12, 3)
    sign, delta = augmentation_with(10.0, flip=False)
    expected = label(rotate_counter_clockwise(pts, 10.0))
    predicted = apply_pose_transform(torch.tensor(label(pts)), sign, delta).numpy()
    assert predicted == pytest.approx(expected, abs=0.5)


def test_flip_sign_matches_the_label_of_a_mirrored_face():
    pts = landmarks(5, 12, 3)
    mirrored = pts.copy()
    mirrored[:, 0] = SIZE - 1 - mirrored[:, 0]
    for a, b in ((36, 45), (48, 54)):
        mirrored[[a, b]] = mirrored[[b, a]]
    sign, delta = augmentation_with(0.0, flip=True)
    predicted = apply_pose_transform(torch.tensor(label(pts)), sign, delta).numpy()
    assert predicted == pytest.approx(label(mirrored), abs=0.5)


def test_consistency_is_zero_for_a_consistent_prediction():
    torch.manual_seed(0)
    pose = torch.randn(8, 3) * 20
    sign_a = torch.tensor([1.0, -1.0, -1.0]).repeat(8, 1)
    sign_b = torch.ones(8, 3)
    delta_a = torch.tensor([0.0, 0.0, 12.0]).repeat(8, 1)
    delta_b = torch.tensor([0.0, 0.0, -7.0]).repeat(8, 1)
    pred_a, pred_b = apply_pose_transform(pose, sign_a, delta_a), apply_pose_transform(pose, sign_b, delta_b)
    loss = relative_consistency_loss(pred_a, pred_b, sign_a, sign_b, delta_a, delta_b)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_consistency_penalises_an_inconsistent_prediction():
    ones, zeros = torch.ones(4, 3), torch.zeros(4, 3)
    loss = relative_consistency_loss(zeros + 5.0, zeros, ones, ones, zeros, zeros, weighted=False)
    assert loss.item() == pytest.approx(25.0)


def test_rcr_weights_favour_non_frontal_poses():
    w = rcr_weights(torch.tensor([[0.0, 1.0, 0.0], [0.0, 60.0, 0.0]]))
    assert w[1] > w[0]
    assert w.mean().item() == pytest.approx(1.0)


def test_target_batches_do_not_change_source_batch_norm():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4))
    model.train()
    before = model[1].running_mean.clone()
    set_bn_eval(model)
    model(torch.randn(16, 4) + 5.0)
    assert torch.equal(model[1].running_mean, before)


def test_splits_are_disjoint():
    rows = [{"image": str(i)} for i in range(100)]
    source, target, test = split_source_target(rows)
    names = [{r["image"] for r in part} for part in (source, target, test)]
    assert not (names[0] & names[1] or names[0] & names[2] or names[1] & names[2])


def test_checkpoints_are_not_selected_on_the_test_split():
    """Picking the best epoch by test MAE makes the reported test MAE optimistic."""
    assert "run_eval(model, test_loader" not in (ROOT / "hpe_rcr" / "train.py").read_text(encoding="utf-8")
