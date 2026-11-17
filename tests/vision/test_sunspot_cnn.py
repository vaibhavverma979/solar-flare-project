"""Tests for the sunspot CNN classifier and heuristic fallback."""

import numpy as np
import pytest

from src.vision.suit_reader import SyntheticSuitGenerator
from src.vision.sunspot_cnn import (
    SunspotClassifier,
    SunspotClassification,
    classify_sunspots,
    MAGNETIC_CLASSES,
)


@pytest.fixture
def generator():
    return SyntheticSuitGenerator(size=128, seed=42)


class TestSunspotClassifier:
    """Test the SunspotClassifier (may use CNN or heuristic fallback)."""

    def test_classify_returns_correct_type(self, generator):
        classifier = SunspotClassifier()
        img = generator.generate(n_sunspots=3, complexity="beta-gamma")
        result = classifier.classify(img)
        assert isinstance(result, SunspotClassification)

    def test_classify_output_shape(self, generator):
        classifier = SunspotClassifier()
        img = generator.generate(n_sunspots=2, complexity="beta")
        result = classifier.classify(img)
        assert result.class_probabilities.shape == (4,)
        assert 0.0 <= result.complexity_score <= 1.0
        assert result.class_index in (0, 1, 2, 3)
        assert result.magnetic_class in MAGNETIC_CLASSES

    def test_classify_probabilities_sum_to_one(self, generator):
        classifier = SunspotClassifier()
        img = generator.generate(n_sunspots=3, complexity="beta-gamma-delta")
        result = classifier.classify(img)
        np.testing.assert_almost_equal(result.class_probabilities.sum(), 1.0, decimal=3)

    def test_classify_all_complexity_levels(self, generator):
        classifier = SunspotClassifier()
        for complexity in MAGNETIC_CLASSES:
            img = generator.generate(n_sunspots=3, complexity=complexity)
            result = classifier.classify(img)
            assert isinstance(result, SunspotClassification)
            assert result.method in ("cnn", "heuristic")

    def test_convenience_function(self, generator):
        img = generator.generate(n_sunspots=2, complexity="alpha")
        result = classify_sunspots(img)
        assert isinstance(result, SunspotClassification)

    def test_preprocess_resizes_image(self):
        img_large = np.random.rand(512, 512).astype(np.float32)
        resized = SunspotClassifier._preprocess(img_large, target_size=224)
        assert resized.shape == (224, 224)

    def test_preprocess_preserves_correct_size(self):
        img = np.random.rand(224, 224).astype(np.float32)
        resized = SunspotClassifier._preprocess(img, target_size=224)
        assert resized.shape == (224, 224)
