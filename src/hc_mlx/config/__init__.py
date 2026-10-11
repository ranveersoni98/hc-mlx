"""Shared constants for hc-mlx: classes, sources, and paths."""

from pathlib import Path

# The five house types we classify.
TARGET_CLASSES = ["apartment", "bungalow", "villa", "mansion", "haveli"]


def class_to_id(name: str) -> int:
    return TARGET_CLASSES.index(name)


def id_to_class(i: int) -> str:
    return TARGET_CLASSES[i]


# Primary labeled source: small imagefolder set, 100 images per class.
PLACES_VAL_REPO = "rumenguin/places365-val"
PLACES_VAL_MAP: dict[str, list[str]] = {
    "apartment": ["apartment_building-outdoor"],
    "bungalow": ["house", "cottage"],
    "villa": ["beach_house"],
    "mansion": ["mansion"],
    "haveli": ["palace"],
}

# Scale-up source: same 365 label space, 1.8M images in 39 parquet shards.
# Places365 has no exact bungalow/villa/haveli labels, so each target maps
# to the closest real labels (see README for the reasoning).
PLACES_256_REPO = "ljnlonoljpiljm/places365-256px"
PLACES_256_LABEL_IDS: dict[str, list[int]] = {
    "apartment": [8],  # apartment building/outdoor
    "bungalow": [183, 107],  # house, cottage
    "villa": [49],  # beach house
    "mansion": [220],  # mansion
    "haveli": [252],  # palace
}
PLACES_256_N_SHARDS = 39
PLACES_256_URL = (
    "https://huggingface.co/datasets/ljnlonoljpiljm/places365-256px"
    "/resolve/main/data/train-{i:05d}-of-00039.parquet"
)

# Unlabeled Indian-context pools, useful for haveli/bungalow examples.
INDIAN_REPOS = [
    "scikit-harry/Global_Residential_Image_Dataset",
    "scikit-harry/Rural_And_Suburban_Indian_Residential_Images",
    "scikit-harry/Indian_Suburban_House_Image_Dataset",
]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Local layout (all git-ignored, reproducible from the Hub).
DATA_DIR = Path("data/combined")
RAW_DIR = Path("data/raw")
CACHE_DIR = Path("data/processed")
CKPT_DIR = Path("checkpoints")
BEST_CKPT = CKPT_DIR / "house_cnn_best.safetensors"
LAST_CKPT = CKPT_DIR / "house_cnn.safetensors"

# ImageNet normalization used for every image fed to the model.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
