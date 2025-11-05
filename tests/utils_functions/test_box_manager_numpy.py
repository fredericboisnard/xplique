"""
Tests for NumPy BoxManager and BoxCoordinatesTranslator.
"""

import numpy as np

from xplique.utils_functions.object_detection.base.box_manager import (
    BoxFormat,
    BoxType,
    NumpyBoxCoordinatesTranslator,
    NumpyBoxManager,
)

# Tests for methods used by NumpyBoxCoordinatesTranslator


def test_normalize_boxes():
    """Test normalizing boxes from pixel coordinates to [0, 1] range."""
    raw_boxes = np.array([[50, 50, 100, 100], [30, 30, 60, 60]], dtype=np.float32)
    image_source_size = (200, 200)
    normalized_boxes = NumpyBoxManager.normalize_boxes(raw_boxes.copy(), image_source_size)
    expected_boxes = np.array([[0.25, 0.25, 0.5, 0.5], [0.15, 0.15, 0.3, 0.3]], dtype=np.float32)
    assert np.allclose(normalized_boxes, expected_boxes)


def test_box_cxcywh_to_xyxy():
    """Test converting boxes from CXCYWH format to XYXY format."""
    cxcywh_boxes = np.array([[50, 50, 20, 20], [0.3, 0.3, 0.1, 0.1]], dtype=np.float32)
    xyxy_boxes = NumpyBoxManager.box_cxcywh_to_xyxy(cxcywh_boxes)
    expected_boxes = np.array([[40, 40, 60, 60], [0.25, 0.25, 0.35, 0.35]], dtype=np.float32)
    assert np.allclose(xyxy_boxes, expected_boxes)


def test_box_xyxy_to_cxcywh():
    """Test converting boxes from XYXY format to CXCYWH format."""
    xyxy_boxes = np.array([[40, 40, 60, 60], [0.25, 0.25, 0.35, 0.35]], dtype=np.float32)
    cxcywh_boxes = NumpyBoxManager.box_xyxy_to_cxcywh(xyxy_boxes)
    expected_boxes = np.array([[50, 50, 20, 20], [0.3, 0.3, 0.1, 0.1]], dtype=np.float32)
    assert np.allclose(cxcywh_boxes, expected_boxes)


def test_box_xywh_to_xyxy():
    """Test converting boxes from XYWH format to XYXY format."""
    xywh_boxes = np.array([[40, 40, 20, 20], [0.25, 0.25, 0.1, 0.1]], dtype=np.float32)
    xyxy_boxes = NumpyBoxManager.box_xywh_to_xyxy(xywh_boxes)
    expected_boxes = np.array([[40, 40, 60, 60], [0.25, 0.25, 0.35, 0.35]], dtype=np.float32)
    assert np.allclose(xyxy_boxes, expected_boxes)


def test_box_xyxy_to_xywh():
    """Test converting boxes from XYXY format to XYWH format."""
    xyxy_boxes = np.array([[40, 40, 60, 60], [0.25, 0.25, 0.35, 0.35]], dtype=np.float32)
    xywh_boxes = NumpyBoxManager.box_xyxy_to_xywh(xyxy_boxes)
    expected_boxes = np.array([[40, 40, 20, 20], [0.25, 0.25, 0.1, 0.1]], dtype=np.float32)
    assert np.allclose(xywh_boxes, expected_boxes)


def test_denormalize_boxes():
    """Test denormalizing boxes from [0, 1] range to pixel coordinates."""
    normalized_boxes = np.array(
        [[0.25, 0.25, 0.5, 0.5], [0.15, 0.15, 0.3, 0.3]], dtype=np.float32
    )
    image_target_size = (400, 400)
    denormalized_boxes = NumpyBoxManager.denormalize_boxes(normalized_boxes, image_target_size)
    expected_boxes = np.array([[100, 100, 200, 200], [60, 60, 120, 120]], dtype=np.float32)
    assert np.allclose(denormalized_boxes, expected_boxes)


# Integration tests for NumpyBoxCoordinatesTranslator


def test_translator_detr_normalized_cxcywh_to_normalized_xyxy():
    """Test DETR typical case: normalized CXCYWH -> normalized XYXY."""
    translator = NumpyBoxCoordinatesTranslator(
        input_box_type=BoxType(BoxFormat.CXCYWH, is_normalized=True),
        output_box_type=BoxType(BoxFormat.XYXY, is_normalized=True),
    )
    input_boxes = np.array([[0.3, 0.4, 0.1, 0.2]], dtype=np.float32)
    output_boxes = translator.translate(input_boxes)
    # CXCYWH [0.3, 0.4, 0.1, 0.2] -> XYXY [0.25, 0.3, 0.35, 0.5]
    expected_boxes = np.array([[0.25, 0.3, 0.35, 0.5]], dtype=np.float32)
    assert np.allclose(output_boxes, expected_boxes)


def test_translator_fcos_pixel_xyxy_to_normalized_xyxy():
    """Test FCOS typical case: pixel XYXY -> normalized XYXY."""
    translator = NumpyBoxCoordinatesTranslator(
        input_box_type=BoxType(BoxFormat.XYXY, is_normalized=False),
        output_box_type=BoxType(BoxFormat.XYXY, is_normalized=True),
    )
    input_boxes = np.array([[50, 50, 100, 100]], dtype=np.float32)
    image_size = (200, 200)
    output_boxes = translator.translate(input_boxes, image_size)
    # Pixel XYXY [50, 50, 100, 100] with image 200x200 -> normalized [0.25, 0.25, 0.5, 0.5]
    expected_boxes = np.array([[0.25, 0.25, 0.5, 0.5]], dtype=np.float32)
    assert np.allclose(output_boxes, expected_boxes)


def test_translator_normalized_xyxy_to_normalized_cxcywh():
    """Test converting normalized XYXY to normalized CXCYWH."""
    translator = NumpyBoxCoordinatesTranslator(
        input_box_type=BoxType(BoxFormat.XYXY, is_normalized=True),
        output_box_type=BoxType(BoxFormat.CXCYWH, is_normalized=True),
    )
    input_boxes = np.array([[0.25, 0.3, 0.35, 0.5]], dtype=np.float32)
    output_boxes = translator.translate(input_boxes)
    # XYXY [0.25, 0.3, 0.35, 0.5] -> CXCYWH [0.3, 0.4, 0.1, 0.2]
    expected_boxes = np.array([[0.3, 0.4, 0.1, 0.2]], dtype=np.float32)
    assert np.allclose(output_boxes, expected_boxes)


def test_translator_normalized_xyxy_to_pixel_xyxy():
    """Test converting normalized XYXY to pixel XYXY."""
    translator = NumpyBoxCoordinatesTranslator(
        input_box_type=BoxType(BoxFormat.XYXY, is_normalized=True),
        output_box_type=BoxType(BoxFormat.XYXY, is_normalized=False),
    )
    input_boxes = np.array([[0.25, 0.25, 0.5, 0.5]], dtype=np.float32)
    image_size = (400, 400)
    output_boxes = translator.translate(input_boxes, image_size)
    # Normalized [0.25, 0.25, 0.5, 0.5] with image 400x400 -> pixel [100, 100, 200, 200]
    expected_boxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
    assert np.allclose(output_boxes, expected_boxes)
