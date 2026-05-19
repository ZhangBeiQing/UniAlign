# DPO LoRA Training

This repository's seed file contains `instruction/chosen/rejected` preference
pairs. 

## Environment

Use the existing virtual environment only:

```bash
/root/venv/refuse/bin/python -m pip install --no-cache-dir -r requirements-refuse.txt
```

Optional WandB configuration:

```bash
cp .env.example .env
```

Then fill `WANDB_API_KEY` and related fields in `.env`.

## Configuration

Training defaults are centralized in:

```text
configs/train_dpo_lora.yaml    # Standalone DPO
configs/train_sft_lora.yaml    # Standalone SFT
configs/train_pipeline.yaml    # SFT → DPO pipeline
```

This follows a lightweight `verl`-style pattern: keep grouped training config in
YAML, while allowing CLI flags to override specific fields for one-off runs.

## Pipeline: SFT → DPO (recommended)

Run supervised fine-tuning first, then preference optimisation on top:

```bash
CUDA_VISIBLE_DEVICES=0 /root/venv/refuse/bin/python scripts/train_pipeline.py \
  --config configs/train_pipeline.yaml \
  --model-path /root/autodl-tmp/Qwen3.5-4B-ortho \
  --data-path reward_seed.json \
  --sft-output-dir /root/autodl-tmp/UniAlign-sft-ortho-lora \
  --dpo-output-dir /root/autodl-tmp/UniAlign-dpo-ortho-lora
```

Key pipeline flags:

| Flag | Default | Description |
| --- | --- | --- |
| `--sft-target` | `chosen` | Which answer to SFT on: `chosen` or `rejected` |
| `--sft-epochs` | `5.0` | SFT phase epochs |
| `--sft-lora-r` | `32` | SFT LoRA rank |
| `--dpo-epochs` | `3.0` | DPO phase epochs |
| `--dpo-beta` | `0.1` | DPO temperature |
| `--skip-sft` | - | Skip SFT, run DPO from existing adapter |
| `--skip-dpo` | - | Run SFT only, stop before DPO |

## Standalone SFT

```bash
CUDA_VISIBLE_DEVICES=0 /root/venv/refuse/bin/python scripts/train_sft_lora.py \
  --config configs/train_sft_lora.yaml
```

The SFT loader uses `prompt` + `completion` formatting with
`completion_only_loss=True`, so only assistant answers contribute to the SFT
loss. For Qwen thinking models, training configs set `enable_thinking: false`
and render the prompt with the empty `<think></think>` block before the answer,
matching this repository's non-thinking SFT answers. `configs/train_sft_lora.yaml`
and `configs/train_pipeline.yaml` also exclude row index `28` by default because
that row is an 80k+ character diary entry; train it only after converting it
into shorter question-answer examples or a compact persona summary.

## Standalone DPO

```bash
CUDA_VISIBLE_DEVICES=0 /root/venv/refuse/bin/python scripts/train_dpo_lora.py \
  --config configs/train_dpo_lora.yaml \
  --model-path /root/autodl-tmp/Qwen3.5-4B \
  --data-path reward_seed.json \
  --output-dir /root/autodl-tmp/UniAlign-dpo-safe-lora \
  --epochs 3 \
  --batch-size 1 \
  --grad-accum 8 \
  --max-length 2048
```

The default path saves a PEFT LoRA adapter, not a merged full model. To reduce
VRAM use, the base model is loaded with 4-bit NF4 quantization by default. Add
`--no-4bit` only if you want bf16 base-model loading and have enough memory.

On RTX 5090 in this environment, Qwen3.5 training is stable when FLA runs on
the Triton backend instead of TileLang, with `TRITON_F32_DEFAULT=ieee`. The
training script sets these automatically:

```text
FLA_TILELANG=0
TRITON_F32_DEFAULT=ieee
```

## Logging and WandB

- Unified file and console logging is provided by `core.logging`
- Script logs are written under `logs/scripts/<ScriptName>/`
- WandB is enabled automatically when `WANDB_API_KEY` is available in the
  environment or `.env`
- The pipeline creates separate WandB runs for SFT and DPO phases, linked by group
- To disable WandB for a single run, add:

```bash
--disable-wandb
```

## Output

**Pipeline:**
- SFT adapter → `--sft-output-dir` (default: `/root/autodl-tmp/UniAlign-sft-ortho-lora`)
- DPO adapter → `--dpo-output-dir` (default: `/root/autodl-tmp/UniAlign-dpo-ortho-lora`)

The DPO adapter is trained on top of the SFT adapter: load the base model, apply
the SFT adapter, then continue training with DPO loss. The final DPO adapter
contains both SFT and DPO training in a single LoRA checkpoint.

**Standalone:**
- SFT adapter → `/root/autodl-tmp/UniAlign-sft-ortho-lora`
- DPO adapter → `/root/autodl-tmp/UniAlign-dpo-safe-lora`
