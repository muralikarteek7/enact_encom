# Fine-Tuning Qwen2.5-VL-7B on ENACT

<!-- Badges -->
<div align="center">

[![Homepage](https://img.shields.io/badge/🏠-Homepage-blue.svg)](https://enact-embodied-cognition.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2511.20937-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2511.20937)
[![Dataset](https://img.shields.io/badge/🤗-Dataset-yellow.svg)](https://huggingface.co/datasets/MLL-Lab/ENACT)

</div>

This repo fine-tunes [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) on the **ENACT ordering task** (predict the correct order of egocentric future states) using QLoRA, and runs inference on Linux/CUDA. The model outputs an ordering for each QA instance, which the `enact eval` command scores for task and pairwise accuracy.

## Results

Our QLoRA fine-tuned Qwen2.5-VL-7B (team **encom**) ranked **#1** on both the dev and test leaderboards, a large jump over the un-tuned baseline:

| Model | Split | Task Accuracy ↑ | Pairwise Accuracy ↑ |
|-------|-------|-----------------|---------------------|
| Qwen2.5-VL-7B (baseline, no fine-tuning) | dev | 9.93% | 33.45% |
| **Qwen2.5-VL-7B + LoRA (ours)** | **dev** | **97.57%** | **99.06%** |
| **Qwen2.5-VL-7B + LoRA (ours)** | **test** | **94.61%** | **98.39%** |

<details>
<summary>Leaderboard standings (team encom, rank #1)</summary>

| Split | Team | Rank | Task Accuracy ↑ | Pairwise Accuracy ↑ |
|-------|------|------|-----------------|---------------------|
| Dev | encom (Qwen2.5-VL) | 1 | 0.9757 | 0.9906 |
| Test | encom (Qwen2.5-VL-7B LoRA Fine-tuned) | 1 | 0.9461 | 0.9839 |

</details>

---

## Table of Contents

- [Results](#results)
- [Setup](#setup-linux-gpu-cuda-124)
- [Get the Data](#get-the-data)
- [Fine-Tuning](#fine-tuning-linuxgpu)
- [Use the Pre-Trained Adapter](#use-the-pre-trained-adapter-skip-fine-tuning)
- [Inference](#inference-linuxgpu)
- [Evaluating Splits](#evaluating-splits)
- [Citation](#citation)

---

## Setup: Linux GPU (CUDA 12.4)

All pinned versions live in [requirements.txt](requirements.txt) (verified on A100, CUDA 12.4, driver 565.57). Use a fresh conda env.

```bash
# 1. Create env — Python 3.11 is required because numpy 2.4.x (pinned in
#    requirements.txt) drops support for 3.10.
conda create -n enact-qwen python=3.11 -y
conda activate enact-qwen

# 2. PyTorch 2.5.1 + CUDA 12.4 (must be installed before flash-attn)
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 triton==3.1.0 \
    --index-url https://download.pytorch.org/whl/cu124

# 3. Flash Attention 2 — builds from source, requires torch + the build tools
#    below already installed (--no-build-isolation skips pip's sandbox).
pip install psutil packaging ninja wheel setuptools
pip install flash-attn==2.8.3 --no-build-isolation

# 4. Remaining deps (Unsloth, HF stack, qwen-vl-utils, etc.)
#    PYTHONNOUSERSITE=1 prevents pip from treating packages already in
#    ~/.local/lib/python3.10/site-packages as "already satisfied" and silently
#    skipping the env install. Skip the prefix if you have a clean ~/.local.
PYTHONNOUSERSITE=1 pip install -r requirements.txt

# 5. Install the ENACT package itself (provides the `enact` CLI used for scoring)
pip install -e .
enact --help

# 6. Remove torchao if it was pulled in transitively (see note below)
pip uninstall -y torchao
```

> **GPU note:** the pinned wheels target CUDA 12.4. If your driver supports a different CUDA toolchain, swap the `+cu124` tags and the `--index-url` accordingly — but you may then need to rebuild `flash-attn` and verify `unsloth` compatibility.

> **⚠️ Do not `pip install torchao` in this env.** `unsloth-zoo` lists `torchao>=0.13.0` as a dependency, but those builds target torch ≥ 2.6. Installing torchao here pulls a cascade of upgrades (torch → 2.12, transformers → 5.10, triton, sympy) that breaks the pinned stack. `torchao` is **not needed** for LoRA training or inference — leave it uninstalled and ignore the `unsloth-zoo requires torchao>=0.13.0` resolver warning. Two failure modes you may hit if it slips in:
>
> - `AttributeError: module 'torch' has no attribute 'int1'` — torchao itself loading against torch < 2.6.
> - `RuntimeError: operator torchvision::nms does not exist` — torch got upgraded but torchvision didn't (ABI mismatch).
>
> Both surface as a misleading `Could not import module 'Qwen2_5_VLForConditionalGeneration'` — read the real error higher in the traceback. Recovery: restore the pinned trio (matches torchvision) and remove torchao:
> ```bash
> pip uninstall -y torchao
> pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 triton==3.1.0 \
>     --index-url https://download.pytorch.org/whl/cu124
> pip install transformers==5.5.0
> ```

---

## Get the Data

The QA dataset (questions + images) must be downloaded before training or inference.

```bash
# Downloads the ENACT QA dataset (~17 GB) to data/QA/ by default
python scripts/helpers/download_dataset.py
```

After this, your `data/` directory contains:

```
data/
├── QA/                              # default training QA
│   ├── enact_ordering.jsonl         # QA pairs (id, images, question, gt_answer)
│   └── images/
├── QA_dev/                          # dev split (if provided)
│   └── enact_ordering_dev.jsonl
└── QA_test/                         # held-out test split (if provided)
    └── enact_ordering_test.jsonl
```

Each line in an `enact_ordering*.jsonl` is one question: an `id`, a list of `images` (first is the current state, the rest are shuffled future states), a `question` prompt, and the `gt_answer` ordering.

---

## Fine-Tuning (Linux/GPU)

Tuned for a 32GB GPU (e.g. RTX 5090).

```bash
# Full training run (3 epochs, batch=8, LoRA rank=64)
python scripts/finetune_qwen25vl.py \
    --output ./lora_enact_ordering

# Quick test (100 samples, 1 epoch)
python scripts/finetune_qwen25vl.py \
    --limit 100 --epochs 1 --output ./test_adapter
```

Key hyperparameters:
- Base model: `unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit`
- LoRA rank: 64, alpha: 64
- Batch size: 8 (effective 16 with grad accumulation)
- Learning rate: 2e-4, cosine decay, 3 epochs

The adapter is written to `--output` (e.g. `./lora_enact_ordering/`), along with a `split_ids.json` recording the train/val split.

---

## Use the Pre-Trained Adapter (skip fine-tuning)

A ready-to-use LoRA adapter is hosted at [gkarteek99/enact-qwen25vl-lora](https://huggingface.co/gkarteek99/enact-qwen25vl-lora). Download it into `./lora_enact_ordering` so every command below works unchanged (this also pulls `train_config.json` and `split_ids.json`, which the script reads locally):

```bash
huggingface-cli download gkarteek99/enact-qwen25vl-lora \
    --local-dir ./lora_enact_ordering \
    --exclude "checkpoint-*/*"   # skip intermediate checkpoints; keep only the final adapter
```

Then run inference with `--adapter ./lora_enact_ordering` exactly as shown below.

> You can also pass the repo ID straight to `--adapter` (`--adapter gkarteek99/enact-qwen25vl-lora`) — PEFT fetches the weights directly. In that case the script can't read `train_config.json` locally and defaults to `--image-size 336` (which matches this adapter's training), and `--split` evaluation won't have `split_ids.json`. Downloading locally avoids both caveats.

---

## Inference (Linux/GPU)

`scripts/inference_hf.py` runs inference using HuggingFace transformers + PEFT. It loads the base Qwen2.5-VL model, optionally applies a LoRA adapter, generates an ordering for each QA instance, and writes an `enact eval`-ready JSONL.

### How the paths work

The script reads **two** path inputs and produces **one** output:

| You provide | Flag | What it points to |
|-------------|------|-------------------|
| **QA file** | `--input` | The `.jsonl` of questions to answer (e.g. `data/QA_test/enact_ordering_test.jsonl`). Defaults to `data/QA/enact_ordering.jsonl`. |
| **Data root** | `--data-root` | The folder the image paths inside the QA file are resolved against. Use `data` whenever your images live under `data/…`. |
| **Output** | `--output` | The predictions file to write. **Relative paths are written to your current directory** (the repo root if you run from there). If omitted, it auto-names to `enact_ordering_<model>.jsonl`. |

> ⚠️ `--output` opens in **overwrite** (`w`) mode by default — re-running replaces the file. Pass `--resume` to keep existing predictions and continue an interrupted run (it skips IDs already present). On completion the script prints the final `Output:` path and the exact `enact eval …` command to score it.

### Step-by-step

```bash
# 1. Smoke test first (first 50 samples) — confirms the env loads before a full run.
#    Writes ./test.jsonl in the current directory.
python scripts/inference_hf.py \
    --adapter ./lora_enact_ordering \
    --input data/QA_test/enact_ordering_test.jsonl \
    --data-root data \
    --limit 50 \
    --output test.jsonl

# 2. Full run on the test set. Writes ./enact_test.jsonl (takes a while).
python scripts/inference_hf.py \
    --adapter ./lora_enact_ordering \
    --input data/QA_test/enact_ordering_test.jsonl \
    --data-root data \
    --output enact_test.jsonl

# 3. Score the predictions. Writes results under data/evaluation/.
enact eval enact_test.jsonl
```

After step 3, the accuracy summary is at `data/evaluation/meta_performance/enact_ordering_test.json` and per-sample results at `data/evaluation/detailed_eval/enact_ordering_test.jsonl`.

### Other run modes

```bash
# Resume an interrupted run (keeps predictions already written, fills in the rest)
python scripts/inference_hf.py \
    --adapter ./lora_enact_ordering \
    --input data/QA_test/enact_ordering_test.jsonl --data-root data \
    --output enact_test.jsonl --resume

# Base model (no --adapter) — for baseline comparison
python scripts/inference_hf.py \
    --input data/QA_test/enact_ordering_test.jsonl --data-root data \
    --output enact_base.jsonl

# Default input (the full training QA file, data/QA/enact_ordering.jsonl)
python scripts/inference_hf.py \
    --adapter ./lora_enact_ordering \
    --output enact_finetuned.jsonl

# Multi-GPU sharding (2 GPUs → ~2x faster)
python scripts/inference_hf.py \
    --adapter ./lora_enact_ordering \
    --input data/QA_test/enact_ordering_test.jsonl --data-root data \
    --run-shards 2 --output enact_test.jsonl
```

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `Qwen/Qwen2.5-VL-7B-Instruct` | Base model ID |
| `--adapter` | None | Path to LoRA adapter directory (omit for base model) |
| `--input` | `data/QA/enact_ordering.jsonl` | Input QA `.jsonl` file |
| `--data-root` | `data/` | Root the QA file's image paths resolve against |
| `--output` | auto-named | Output predictions JSONL (relative → current dir) |
| `--resume` | off | Continue an existing output file instead of overwriting |
| `--limit` | None | Cap number of samples (for testing) |
| `--id-file` | None | JSON with `train`/`val`/`test` split IDs |
| `--split` | None | `train`, `val`, or `test` — filter by split |
| `--run-shards` | None | Launch N parallel GPU processes (one per GPU) |
| `--prefetch` | 4 | CPU workers for image prefetch |

---

## Evaluating Splits

After fine-tuning, `split_ids.json` is saved in the adapter directory. Use it to evaluate the train and validation splits separately and measure generalization:

```bash
# Validation split only (unseen during training)
python scripts/inference_hf.py \
    --adapter ./lora_enact_ordering \
    --id-file ./lora_enact_ordering/split_ids.json \
    --split val \
    --output enact_val.jsonl

# Train split (expect high accuracy = model learned the task)
python scripts/inference_hf.py \
    --adapter ./lora_enact_ordering \
    --id-file ./lora_enact_ordering/split_ids.json \
    --split train \
    --output enact_train.jsonl

enact eval enact_val.jsonl
enact eval enact_train.jsonl
```

Val accuracy = true generalization. A large train/val gap = overfitting.

> **Note:** The dev set (`QA_dev`) shares IDs with the training QA file (`QA`), so dev accuracy reflects memorization, not generalization. Use the val split from `split_ids.json` or the held-out test set for true evaluation.

---

## Citation

```bibtex
@article{wang2025enact,
  title={ENACT: Evaluating Embodied Cognition with World Modeling of Egocentric Interaction},
  author={Wang, Qineng and Huang, Wenlong and Zhou, Yu and Yin, Hang
          and Bao, Tianwei and Lyu, Jianwen and Liu, Weiyu and Zhang, Ruohan
          and Wu, Jiajun and Li, Fei-Fei and Li, Manling},
  journal={arXiv preprint arXiv:2511.20937},
  year={2025}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. Built upon the [ENACT benchmark](https://enact-embodied-cognition.github.io/).
