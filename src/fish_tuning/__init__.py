# src/fish_tuning/__init__.py

# Expose the main classes and functions for easy importing from the package.
# This allows other parts of the project to do:
# from fish_tuning import FishTuningTrainer, calculate_fisher_information, create_mask_from_fisher

from .trainer import FishTuningTrainer
from .fisher_utils import calculate_fisher_information, create_mask_from_fisher