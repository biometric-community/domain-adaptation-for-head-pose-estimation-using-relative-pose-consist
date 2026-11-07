"""REPORT.md, the committed output of the full run."""

from __future__ import annotations

import re

import pytest


def ours(report: str) -> list[float]:
    match = re.search(r"\| \*\*Ours\*\* \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|", report)
    assert match, "results row missing from REPORT.md"
    return [float(g) for g in match.groups()]


def test_mae_is_the_mean_of_the_three_angles(report):
    mae, pitch, yaw, roll = ours(report)
    assert mae == pytest.approx((pitch + yaw + roll) / 3.0, abs=0.02)


def test_published_values(report, targets):
    if targets["published_values"] is None:
        pytest.skip("paper tables not recorded in tests/paper_targets.yaml")
    expected = targets["published_values"]
    assert ours(report) == pytest.approx(
        [expected["mae"], expected["pitch"], expected["yaw"], expected["roll"]], abs=1.0
    )
