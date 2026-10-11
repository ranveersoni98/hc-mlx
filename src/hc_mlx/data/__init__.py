"""Dataset helpers, re-exported for `from hc_mlx.data import ...`."""

from hc_mlx.data.build import (
    build_combined,
    load_images,
    stratified_split,
    summarize_combined,
)
from hc_mlx.data.sources import (
    download_indian_pool,
    download_places_val,
    fetch_places256_duckdb,
    stream_places256,
)

__all__ = [
    "build_combined",
    "summarize_combined",
    "load_images",
    "stratified_split",
    "download_places_val",
    "stream_places256",
    "fetch_places256_duckdb",
    "download_indian_pool",
]
