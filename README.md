# hc-mlx

A small classifier that looks at a photo of a house and guesses what kind it
is: apartment, bungalow, villa, mansion, or haveli. It runs on Apple MLX, trains
in minutes on a Mac, and is exactly as serious as it sounds.

Best validation accuracy so far is around 46 percent on 600 images, where
guessing at random would get 20. This is a toy. Enjoy it as one.

## Quickstart

You need Python 3.14, `uv`, and a Mac with Apple silicon.

```bash
uv sync
hc-mlx download          # fetch the datasets and build data/combined
hc-mlx train             # train, saves weights to checkpoints/
hc-mlx predict photo.jpg # top 3 guesses for your photo
```

Training flags you may want: `hc-mlx train --epochs 50 --batch 32 --size 128`.

## How it works

No single public dataset labels houses the way we wanted, so we combined a few
from the Hugging Face Hub and mapped their labels onto our five classes:

| Our class | Source | Original label |
|---|---|---|
| apartment | Places365 val | apartment building, outdoor |
| bungalow | Places365 val | house, cottage |
| villa | Places365 val | beach house |
| mansion | Places365 val | mansion |
| haveli | Places365 val | palace |

Places365 has 365 scene types and no exact bungalow, villa, or haveli labels,
so each of those maps to the closest thing available. A haveli is a courtyard
mansion from India, so palace is the nearest stand in. A villa maps to beach
house. A bungalow maps to house and cottage. It is approximate on purpose.

For scale, the same five mappings point at matching labels in the 1.8M image
Places365 256px set. `hc-mlx download --per-class 1000` pulls those with
DuckDB straight from the parquet files, which beats streaming by a lot. There
is also support for three small Indian residential sets from `scikit-harry` as
extra haveli and bungalow material.

The model itself is a plain four block CNN with batch norm, about 390K
parameters, trained with AdamW, flip augmentation, and label smoothing. Images
are resized to 128px and normalized like ImageNet. Nothing fancy.

## Project layout

```text
src/hc_mlx/
  __init__.py      # package intro, public names
  config/          # classes, dataset sources, paths
  data/            # Hub fetchers, local build, caching, splits
  models/          # the HouseCNN model
  training/        # training loop (python -m hc_mlx.training)
  inference/       # classify one photo (python -m hc_mlx.inference)
  cli/             # the hc-mlx command
```

`data/` and `checkpoints/` are git ignored. Everything in them can be rebuilt
with `hc-mlx download` and `hc-mlx train`.

## Results

| Setup | Val accuracy |
|---|---|
| 600 images, 20 epochs | 0.325 |
| 600 images, 30 epochs | 0.442 |
| 600 images + flips, 50 epochs | 0.458 |

Typical confusions are sensible ones: mansion vs haveli, apartment vs villa.
More data helps more than more epochs. The DuckDB scale up path exists for
exactly that reason.

## License

MIT. The source datasets keep their own licenses (MIT for the Places365
mirrors used here, CC-BY-4.0 for the scikit-harry sets).
