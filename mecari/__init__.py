"""Mecari - Japanese Morphological Analysis with Graph Neural Networks"""

__version__ = "0.1.0"

# Export minimal API (avoid heavy imports at package import time)
from mecari.config.config import get_model_config, load_config, override_config, save_config  # noqa: F401
from mecari.data.data_module import DataModule  # noqa: F401

__all__ = ["DataModule", "get_model_config", "override_config", "save_config", "load_config"]
