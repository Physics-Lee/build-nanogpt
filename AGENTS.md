# Repository Guidelines

## Project Structure & Module Organization
This repository is a script-first reproduction of nanoGPT, with most code at the repo root.

- `train_gpt2.py`: main training script (DDP-aware, logging, eval hooks).
- `train_gpt2_original.py`: close-to-upstream baseline variant.
- `train_gpt2_myself.py`: experimental local variant.
- `fineweb.py`: downloads/tokenizes FineWeb-Edu shards into `edu_fineweb10B/`.
- `hellaswag.py`: downloads and evaluates HellaSwag into `hellaswag/`.
- `log/`: training logs and checkpoints (runtime output).
- `play.ipynb`: exploratory notebook; keep non-essential experiments here.

## Build, Test, and Development Commands
Use Python 3.10+ in a virtual environment.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch numpy tiktoken datasets tqdm transformers
```

Core workflows:

```bash
python fineweb.py                      # build local token shards in edu_fineweb10B/
python train_gpt2.py                   # single-process training (auto device detection)
torchrun --standalone --nproc_per_node=8 train_gpt2.py  # multi-GPU DDP training
python hellaswag.py -m gpt2 -d cuda    # standalone HellaSwag eval
```

## Coding Style & Naming Conventions
- Follow existing Python style: 4-space indentation, snake_case for functions/vars, PascalCase for classes.
- Keep scripts runnable directly; avoid introducing hidden framework bootstrapping.
- Prefer small, explicit changes and preserve current inline-comment style where it clarifies tensor shapes or training logic.

## Testing Guidelines
There is no dedicated unit-test suite in this repo today. Validate changes with:
- quick smoke run of `python train_gpt2.py` (startup, dataloader, first steps),
- `python hellaswag.py -m gpt2 -d cuda|cpu` for evaluation-path sanity,
- optional notebook checks in `play.ipynb` for analysis-related changes.

When changing training math, include before/after loss or accuracy snippets from `log/log.txt`.

## Commit & Pull Request Guidelines
- Match existing commit style: short, imperative summaries (e.g., `fix device_type`, `plot QKV`, `Start Training!`).
- Keep commits focused to one concern.
- PRs should include: what changed, why, exact commands run, and key output/metrics.
- Link related issues/discussions when relevant; include screenshots only for notebook/plot changes.
