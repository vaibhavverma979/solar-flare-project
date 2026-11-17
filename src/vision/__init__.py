"""
vision — SUIT (Solar Ultraviolet Imaging Telescope) computer-vision pipeline.

Provides CNN-based sunspot magnetic classification and physics-motivated
visual feature extraction from solar UV images.
"""

from src.vision.suit_reader import read_suit, SyntheticSuitGenerator
from src.vision.sunspot_cnn import SunspotClassifier, classify_sunspots
from src.vision.visual_features import extract_visual_features
