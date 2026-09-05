import cv2
import numpy as np
import pytest

from src.feature_extraction.morphological_features import extract_features


@pytest.fixture
def rectangle_mask():
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[50:100, 60:180] = 255
    return mask


@pytest.fixture
def diagonal_scratch_mask():
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.line(mask, (30, 30), (170, 150), color=255, thickness=4)
    return mask


@pytest.fixture
def horizontal_scratch_mask():
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.line(mask, (30, 100), (170, 100), color=255, thickness=4)
    return mask


@pytest.fixture
def round_patch_mask():
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (100, 100), radius=40, color=255, thickness=-1)
    return mask


@pytest.fixture
def l_shape_mask():
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[40:160, 40:80] = 255
    mask[120:160, 40:160] = 255
    return mask


@pytest.fixture
def empty_mask():
    return np.zeros((200, 200), dtype=np.uint8)


@pytest.fixture
def single_pixel_mask():
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[100, 100] = 255
    return mask


def test_rectangle_fills_its_bounding_box(rectangle_mask):
    features = extract_features(rectangle_mask)
    assert len(features) == 1
    f = features[0]
    assert f["extent"] > 0.9
    assert f["solidity"] == pytest.approx(1.0)


def test_diagonal_scratch_has_low_extent(diagonal_scratch_mask):
    # A diagonal line's bounding box is mostly empty space, even though
    # the shape itself is thin and elongated. Aspect ratio misses this
    # (bounding box is roughly square) — extent is what catches it.
    features = extract_features(diagonal_scratch_mask)
    assert len(features) == 1
    f = features[0]
    assert f["extent"] < 0.1
    assert f["aspect_ratio"] < 2.0  # confirms aspect ratio is NOT the signal here


def test_horizontal_scratch_has_high_aspect_ratio(horizontal_scratch_mask):
    # Once the scratch is axis-aligned, its bounding box hugs it tightly,
    # so aspect ratio becomes the discriminating feature instead of extent.
    features = extract_features(horizontal_scratch_mask)
    assert len(features) == 1
    f = features[0]
    assert f["aspect_ratio"] > 10.0


def test_round_patch_has_moderate_extent_and_low_aspect_ratio(round_patch_mask):
    features = extract_features(round_patch_mask)
    assert len(features) == 1
    f = features[0]
    assert f["aspect_ratio"] == pytest.approx(1.0, abs=0.1)
    assert f["extent"] > 0.6
    assert f["solidity"] > 0.95


def test_l_shape_has_low_solidity(l_shape_mask):
    # Solidity is the feature that catches genuine concavity — an L-shape's
    # convex hull fills in the missing notch, making hull_area notably
    # bigger than the actual shape area. None of the other shapes we test
    # (all convex) get below ~0.88 solidity, so this is the discriminator.
    features = extract_features(l_shape_mask)
    assert len(features) == 1
    f = features[0]
    assert f["solidity"] < 0.8


def test_empty_mask_returns_no_features(empty_mask):
    assert extract_features(empty_mask) == []


def test_single_pixel_does_not_crash(single_pixel_mask):
    # Regression test for the hull_area == 0 guard. Real background
    # subtraction produces stray single-pixel noise regularly — this
    # must return a degenerate-but-valid result, not raise ZeroDivisionError.
    features = extract_features(single_pixel_mask)
    assert len(features) == 1
    assert features[0]["solidity"] == 0