from lib.utils.config import CN

from ..utils.builder import build_dataset
from . import lewm_data, libero, libero_seer  # side-effect: registers the datasets

__all__ = ["create_dataset", "lewm_data", "libero", "libero_seer"]


def create_dataset(cfg: CN, data_preset: CN):
    """Create a dataset instance."""
    return build_dataset(cfg, data_preset=data_preset)
