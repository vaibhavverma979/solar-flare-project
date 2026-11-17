"""
vectordb — Vector database for historical solar flare pattern matching.

Encodes X-ray light curves into dense embeddings and performs similarity
search against decades of historical flare profiles using ChromaDB.
"""

from src.vectordb.embedder import LightCurveEmbedder, StatisticalEmbedder
from src.vectordb.historical_db import HistoricalFlareDB
from src.vectordb.pattern_matcher import PatternMatcher, PatternMatchResult
