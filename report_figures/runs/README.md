# A/B/C/D run results (Zelda)

Qualitative eval filmstrips for the four compared runs. In each `clip_*.png`:
top row = ground truth, middle = prediction with the true action, bottom = prediction with a random action.

| Run | Folder | Tokenizer | Motion | Delta control | Quality |
|-----|--------|-----------|--------|---------------|---------|
| A | [`A/`](A) | tiny, no perceptual (0.14M) | 0.78 | +0.47 | blurry |
| B | [`B/`](B) | tiny + perceptual (0.14M) | 1.21 | +0.88 | modest (best control) |
| C | [`C/`](C) | big (67.3M) | 0.96 | -0.05 | sharp |
| D | [`D/`](D) | big + strong control (67.3M) | 0.44 | +2.24 | sharpest (showcase) |

Checkpoints are not in the repo (13–149 GB each); see the run/checkpoint table in the top-level [README](../../README.md).
