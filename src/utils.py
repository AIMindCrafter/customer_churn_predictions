"""Utility functions"""

import os
import pickle

from src.logger import get_logger

logger = get_logger(__name__)


def create_dirs_if_not_exist(dirs):
    """Create directories if they don't exist"""
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        logger.debug("Created directory: %s", dir_path)


def print_section(title):
    """Print formatted section header"""
    logger.info("=" * 60)
    logger.info("  %s", title)
    logger.info("=" * 60)


def save_pickle(obj, filepath):
    """Save object as pickle"""
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    logger.debug("Saved pickle: %s", filepath)


def load_pickle(filepath):
    """Load object from pickle"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)
