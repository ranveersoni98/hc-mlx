"""hc-mlx: a small, unserious house-type classifier built on Apple MLX."""

from hc_mlx.config import TARGET_CLASSES
from hc_mlx.data import build_combined, download_places_val, load_images, summarize_combined
from hc_mlx.models import HouseCNN

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "TARGET_CLASSES",
    "HouseCNN",
    "download_places_val",
    "build_combined",
    "summarize_combined",
    "load_images",
]


def main() -> None:
    print("hc-mlx: classify houses into", TARGET_CLASSES)
    print("Try `hc-mlx info`, `hc-mlx download`, `hc-mlx train`, `hc-mlx predict photo.jpg`")
