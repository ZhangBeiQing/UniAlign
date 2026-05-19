#!/usr/bin/env python3
"""SFT → DPO pipeline entrypoint for Qwen-style chat models.

Runs supervised fine-tuning first, then preference optimisation on top,
loading the SFT adapter as the starting point for DPO.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
import yaml

import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

from core.logging import get_logger, init_script_logger
from core.training import (
    CheckpointLoggingCallback,
    apply_chat_template_text,
    build_model,
    configure_wandb,
    create_tokenizer,
    get_world_size,
    load_config,
    log_cuda_memory,
    log_effective_batch,
    set_runtime_env,
)

DEFAULT_MODEL_PATH = "/root/autodl-tmp/Qwen3.5-4B-ortho"
DEFAULT_DATA_PATH = "reward_seed.json"
DEFAULT_SFT_DIR = "/root/autodl-tmp/UniAlign-sft-ortho-lora"
DEFAULT_DPO_DIR = "/root/autodl-tmp/UniAlign-dpo-ortho-lora"
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "train_pipeline.yaml"
ENV_PATH = REPO_ROOT / ".env"

SCRIPT_LOGGER = init_script_logger("SftDpoPipeline")
DATA_LOGGER = get_logger("PipelineData")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SFT → DPO LoRA pipeline on reward_seed.json."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--model-path")
    parser.add_argument("--data-path")
    parser.add_argument("--sft-output-dir")
    parser.add_argument("--dpo-output-dir")
    parser.add_argument("--sft-target", choices=["chosen", "rejected"])
    parser.add_argument("--sft-epochs", type=float)
    parser.add_argument("--sft-max-steps", type=int)
    parser.add_argument("--sft-learning-rate", type=float)
    parser.add_argument("--sft-max-length", type=int)
    parser.add_argument("--sft-batch-size", type=int)
    parser.add_argument("--sft-grad-accum", type=int)
    parser.add_argument("--sft-lora-r", type=int)
    parser.add_argument("--sft-lora-alpha", type=int)
    parser.add_argument("--sft-seed", type=int)
    parser.add_argument("--dpo-epochs", type=float)
    parser.add_argument("--dpo-max-steps", type=int)
    parser.add_argument("--dpo-learning-rate", type=float)
    parser.add_argument("--dpo-beta", type=float)
    parser.add_argument("--dpo-max-length", type=int)
    parser.add_argument("--dpo-batch-size", type=int)
    parser.add_argument("--dpo-grad-accum", type=int)
    parser.add_argument("--dpo-lora-r", type=int)
    parser.add_argument("--dpo-lora-alpha", type=int)
    parser.add_argument("--dpo-seed", type=int)
    parser.add_argument("--logging-steps", type=int)
    parser.add_argument("--save-steps", type=int)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-mode")
    parser.add_argument("--wandb-group")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--wandb-report-model", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--skip-sft", action="store_true", help="Skip SFT phase and run DPO only (--sft-output-dir must point to an existing adapter).")
    parser.add_argument("--skip-dpo", action="store_true", help="Run SFT phase only without DPO.")
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    output_cfg = config.get("output", {})
    sft_cfg = config.get("sft", {})
    dpo_cfg = config.get("dpo", {})
    lora_cfg = config.get("lora", {})
    wandb_cfg = config.get("wandb", {})
    runtime_cfg = config.get("runtime", {})

    set_runtime_env(runtime_cfg)

    args.model_path = args.model_path or model_cfg.get("path", DEFAULT_MODEL_PATH)
    args.data_path = args.data_path or data_cfg.get("path", DEFAULT_DATA_PATH)
    args.sft_output_dir = args.sft_output_dir or output_cfg.get("sft_dir", DEFAULT_SFT_DIR)
    args.dpo_output_dir = args.dpo_output_dir or output_cfg.get("dpo_dir", DEFAULT_DPO_DIR)

    sft_lora = sft_cfg.get("lora", {})
    dpo_lora_defaults = dpo_cfg.get("lora", sft_lora)

    args.sft_target = args.sft_target or sft_cfg.get("target", "chosen")
    args.sft_exclude_indices = set(sft_cfg.get("exclude_indices", []))
    args.sft_max_target_chars = sft_cfg.get("max_target_chars")
    args.sft_enable_thinking = bool(sft_cfg.get("enable_thinking", False))
    args.sft_max_length = args.sft_max_length if args.sft_max_length is not None else sft_cfg.get("max_length", 1024)
    args.sft_epochs = args.sft_epochs if args.sft_epochs is not None else sft_cfg.get("epochs", 5.0)
    args.sft_max_steps = args.sft_max_steps if args.sft_max_steps is not None else sft_cfg.get("max_steps", -1)
    args.sft_learning_rate = args.sft_learning_rate if args.sft_learning_rate is not None else sft_cfg.get("learning_rate", 1e-5)
    args.sft_batch_size = args.sft_batch_size if args.sft_batch_size is not None else sft_cfg.get("batch_size", 1)
    args.sft_grad_accum = args.sft_grad_accum if args.sft_grad_accum is not None else sft_cfg.get("grad_accum", 8)
    args.sft_lora_r = args.sft_lora_r if args.sft_lora_r is not None else sft_lora.get("r", 32)
    args.sft_lora_alpha = args.sft_lora_alpha if args.sft_lora_alpha is not None else sft_lora.get("alpha", 64)
    args.sft_lora_dropout = sft_lora.get("dropout", 0.05)
    args.sft_seed = args.sft_seed if args.sft_seed is not None else sft_cfg.get("seed", 42)
    args.sft_packing = sft_cfg.get("packing", False)

    args.dpo_max_length = args.dpo_max_length if args.dpo_max_length is not None else dpo_cfg.get("max_length", 2048)
    args.dpo_epochs = args.dpo_epochs if args.dpo_epochs is not None else dpo_cfg.get("epochs", 3.0)
    args.dpo_max_steps = args.dpo_max_steps if args.dpo_max_steps is not None else dpo_cfg.get("max_steps", -1)
    args.dpo_learning_rate = args.dpo_learning_rate if args.dpo_learning_rate is not None else dpo_cfg.get("learning_rate", 5e-6)
    args.dpo_beta = args.dpo_beta if args.dpo_beta is not None else dpo_cfg.get("beta", 0.1)
    args.dpo_batch_size = args.dpo_batch_size if args.dpo_batch_size is not None else dpo_cfg.get("batch_size", 1)
    args.dpo_grad_accum = args.dpo_grad_accum if args.dpo_grad_accum is not None else dpo_cfg.get("grad_accum", 8)
    args.dpo_lora_r = args.dpo_lora_r if args.dpo_lora_r is not None else dpo_lora_defaults.get("r", 16)
    args.dpo_lora_alpha = args.dpo_lora_alpha if args.dpo_lora_alpha is not None else dpo_lora_defaults.get("alpha", 32)
    args.dpo_lora_dropout = dpo_lora_defaults.get("dropout", 0.05)
    args.dpo_seed = args.dpo_seed if args.dpo_seed is not None else dpo_cfg.get("seed", 42)
    args.dpo_enable_thinking = bool(dpo_cfg.get("enable_thinking", False))

    common_logging = sft_cfg if "logging_steps" in sft_cfg else dpo_cfg
    args.logging_steps = args.logging_steps if args.logging_steps is not None else common_logging.get("logging_steps", 1)
    args.save_steps = args.save_steps if args.save_steps is not None else common_logging.get("save_steps", 20)

    args.wandb_project = args.wandb_project or os.environ.get("WANDB_PROJECT") or wandb_cfg.get("project", "unialign")
    args.wandb_entity = args.wandb_entity or os.environ.get("WANDB_ENTITY") or wandb_cfg.get("entity")
    args.wandb_mode = args.wandb_mode or os.environ.get("WANDB_MODE") or wandb_cfg.get("mode", "online")
    args.wandb_group = args.wandb_group or os.environ.get("WANDB_GROUP") or wandb_cfg.get("group", "qwen35-sft-dpo-pipeline")
    if not args.wandb_report_model:
        args.wandb_report_model = bool(wandb_cfg.get("report_model", False))
    if wandb_cfg.get("enabled") is False:
        args.disable_wandb = True

    args.use_4bit = bool(model_cfg.get("use_4bit", True)) and not args.no_4bit
    args.lora_target_modules = lora_cfg.get(
        "target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    args.attn_implementation = model_cfg.get("attn_implementation", "sdpa")
    args.config_data = config

    return args


def load_sft_dataset(
    data_path: str | Path,
    tokenizer: AutoTokenizer,
    target: str,
    exclude_indices: set[int] | None = None,
    max_target_chars: int | None = None,
    enable_thinking: bool = False,
) -> Dataset:
    path = Path(data_path)
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    skipped = 0
    examples: list[dict[str, str]] = []
    exclude_indices = exclude_indices or set()
    for idx, row in enumerate(rows):
        if idx in exclude_indices:
            skipped += 1
            DATA_LOGGER.info("Skipped SFT row index=%d because it is configured in exclude_indices", idx)
            continue
        if not row.get("instruction") or not row.get(target):
            skipped += 1
            continue
        if max_target_chars is not None and len(row[target]) > max_target_chars:
            skipped += 1
            DATA_LOGGER.info(
                "Skipped SFT row index=%d because %s length=%d exceeds max_target_chars=%d",
                idx,
                target,
                len(row[target]),
                max_target_chars,
            )
            continue

        messages = [{"role": "user", "content": row["instruction"].strip()}]
        prompt = apply_chat_template_text(
            tokenizer,
            messages,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        completion = row[target].strip()
        if not completion:
            skipped += 1
            continue

        examples.append({"prompt": prompt, "completion": completion})

    if not examples:
        raise ValueError(f"{path} does not contain any training examples")
    DATA_LOGGER.info("Loaded %d SFT examples (target=%s, skipped=%d) from %s", len(examples), target, skipped, path)
    return Dataset.from_list(examples)


def load_dpo_dataset(
    data_path: str | Path,
    tokenizer: AutoTokenizer,
    enable_thinking: bool = False,
) -> Dataset:
    path = Path(data_path)
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    skipped = 0
    examples: list[dict[str, str]] = []
    for idx, row in enumerate(rows):
        if not (row.get("instruction") and row.get("chosen") and row.get("rejected")):
            skipped += 1
            continue

        messages = [{"role": "user", "content": row["instruction"].strip()}]
        prompt = apply_chat_template_text(
            tokenizer,
            messages,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        examples.append(
            {
                "prompt": prompt,
                "chosen": row["chosen"].strip(),
                "rejected": row["rejected"].strip(),
            }
        )

    if not examples:
        raise ValueError(f"{path} does not contain any training examples with chosen+rejected")
    DATA_LOGGER.info("Loaded %d DPO preference pairs (skipped=%d) from %s", len(examples), skipped, path)
    return Dataset.from_list(examples)


def _chunked_cross_entropy(
    hidden: torch.Tensor,
    lm_weight: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = 16384,
    seq_chunk: int = 2048,
) -> torch.Tensor:
    total_loss = 0.0
    total_n = 0
    h_f32 = hidden.float()
    V = lm_weight.size(0)
    device = hidden.device

    for t_start in range(0, hidden.size(0), seq_chunk):
        t_end = min(t_start + seq_chunk, hidden.size(0))
        h = h_f32[t_start:t_end]
        l = labels[t_start:t_end]
        n = h.size(0)

        max_logits = torch.full((n,), -float("inf"), device=device, dtype=torch.float)
        sum_exp = torch.zeros(n, device=device, dtype=torch.float)
        correct_logits = torch.zeros(n, device=device, dtype=torch.float)

        for s in range(0, V, chunk_size):
            e = min(s + chunk_size, V)
            w = lm_weight[s:e].float()
            logits = F.linear(h, w)
            chunk_max = logits.max(dim=-1).values
            new_max = torch.maximum(max_logits, chunk_max)

            sum_exp = sum_exp * torch.exp(max_logits - new_max) + \
                      torch.exp(logits - new_max.unsqueeze(-1)).sum(dim=-1)
            max_logits = new_max

            local_labels = l - s
            in_chunk = (local_labels >= 0) & (local_labels < (e - s))
            if in_chunk.any():
                correct_logits[in_chunk] = logits[in_chunk, local_labels[in_chunk]]

            del logits, w

        log_probs = correct_logits - max_logits - torch.log(sum_exp + 1e-10)
        total_loss += -log_probs.sum()
        total_n += n
        del max_logits, sum_exp, correct_logits, h

    return total_loss / max(total_n, 1)


class ChunkedSFTTrainer(SFTTrainer):

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels", None)
        if labels is None:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

        transformer = model.model.model
        outputs = transformer(**inputs, use_cache=False)
        hidden = outputs[0].contiguous()

        shift_hidden = hidden[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        shift_hidden = shift_hidden.view(-1, shift_hidden.size(-1))
        shift_labels = shift_labels.view(-1)

        active = shift_labels != -100
        if active.sum() == 0:
            return (torch.tensor(0.0, device=hidden.device, requires_grad=True), outputs) \
                if return_outputs else torch.tensor(0.0, device=hidden.device, requires_grad=True)

        active_hidden = shift_hidden[active]
        active_labels = shift_labels[active]

        lm_weight = model.get_output_embeddings().weight
        loss = _chunked_cross_entropy(active_hidden, lm_weight, active_labels)

        return (loss, outputs) if return_outputs else loss


def run_sft_phase(args: argparse.Namespace, tokenizer: AutoTokenizer) -> None:
    SCRIPT_LOGGER.info("=" * 60)
    SCRIPT_LOGGER.info("Phase 1/2: SFT Training")
    SCRIPT_LOGGER.info("=" * 60)

    sft_run_name = f"{Path(args.sft_output_dir).name}-sft"
    wandb_sft_group = f"{args.wandb_group}-sft"

    report_to, run_name = configure_wandb(
        disable_wandb=args.disable_wandb,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=sft_run_name,
        wandb_group=wandb_sft_group,
        wandb_report_model=args.wandb_report_model,
        output_dir=args.sft_output_dir,
    )

    SCRIPT_LOGGER.info(
        "SFT config: model=%s data=%s output=%s target=%s epochs=%s max_steps=%s "
        "lr=%s batch=%s grad_accum=%s max_length=%s lora_r=%s lora_alpha=%s use_4bit=%s",
        args.model_path,
        args.data_path,
        args.sft_output_dir,
        args.sft_target,
        args.sft_epochs,
        args.sft_max_steps,
        args.sft_learning_rate,
        args.sft_batch_size,
        args.sft_grad_accum,
        args.sft_max_length,
        args.sft_lora_r,
        args.sft_lora_alpha,
        args.use_4bit,
    )
    log_effective_batch(args.sft_batch_size, args.sft_grad_accum)

    train_dataset = load_sft_dataset(
        args.data_path,
        tokenizer,
        args.sft_target,
        exclude_indices=args.sft_exclude_indices,
        max_target_chars=args.sft_max_target_chars,
        enable_thinking=args.sft_enable_thinking,
    )
    model = build_model(args.model_path, use_4bit=args.use_4bit)
    log_cuda_memory("after-model-load")

    peft_config = LoraConfig(
        r=args.sft_lora_r,
        lora_alpha=args.sft_lora_alpha,
        lora_dropout=args.sft_lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.lora_target_modules,
    )

    sft_args = SFTConfig(
        output_dir=args.sft_output_dir,
        per_device_train_batch_size=args.sft_batch_size,
        gradient_accumulation_steps=args.sft_grad_accum,
        num_train_epochs=args.sft_epochs,
        max_steps=args.sft_max_steps,
        learning_rate=args.sft_learning_rate,
        max_length=args.sft_max_length,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_steps=args.save_steps,
        save_strategy="steps",
        save_total_limit=2,
        report_to=report_to,
        run_name=run_name,
        save_only_model=not args.wandb_report_model,
        remove_unused_columns=False,
        optim="adamw_torch_fused",
        seed=args.sft_seed,
        completion_only_loss=True,
        packing=args.sft_packing,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[CheckpointLoggingCallback()],
    )
    SCRIPT_LOGGER.info("SFT trainer constructed; starting train()")
    trainer.train()
    log_cuda_memory("after-sft-train")

    SCRIPT_LOGGER.info("Saving SFT adapter to %s", args.sft_output_dir)
    trainer.save_model(args.sft_output_dir)
    tokenizer.save_pretrained(args.sft_output_dir)
    SCRIPT_LOGGER.info("SFT adapter saved to %s", args.sft_output_dir)

    del trainer
    del model
    torch.cuda.empty_cache()
    log_cuda_memory("after-sft-cleanup")

    import wandb as _wandb
    _wandb.finish()


def run_dpo_phase(args: argparse.Namespace, tokenizer: AutoTokenizer) -> None:
    SCRIPT_LOGGER.info("=" * 60)
    SCRIPT_LOGGER.info("Phase 2/2: DPO Training")
    SCRIPT_LOGGER.info("=" * 60)

    dpo_run_name = f"{Path(args.dpo_output_dir).name}-dpo"
    wandb_dpo_group = f"{args.wandb_group}-dpo"

    report_to, run_name = configure_wandb(
        disable_wandb=args.disable_wandb,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=dpo_run_name,
        wandb_group=wandb_dpo_group,
        wandb_report_model=args.wandb_report_model,
        output_dir=args.dpo_output_dir,
    )

    SCRIPT_LOGGER.info(
        "DPO config: model=%s adapter=%s data=%s output=%s epochs=%s max_steps=%s "
        "lr=%s beta=%s batch=%s grad_accum=%s max_length=%s use_4bit=%s",
        args.model_path,
        args.sft_output_dir,
        args.data_path,
        args.dpo_output_dir,
        args.dpo_epochs,
        args.dpo_max_steps,
        args.dpo_learning_rate,
        args.dpo_beta,
        args.dpo_batch_size,
        args.dpo_grad_accum,
        args.dpo_max_length,
        args.use_4bit,
    )
    log_effective_batch(args.dpo_batch_size, args.dpo_grad_accum)

    train_dataset = load_dpo_dataset(
        args.data_path,
        tokenizer,
        enable_thinking=args.dpo_enable_thinking,
    )

    base_model = build_model(args.model_path, use_4bit=args.use_4bit)
    SCRIPT_LOGGER.info("Loading SFT adapter from %s", args.sft_output_dir)
    model = PeftModel.from_pretrained(
        base_model,
        args.sft_output_dir,
        is_trainable=True,
    )
    log_cuda_memory("after-model-load-with-adapter")

    dpo_args = DPOConfig(
        output_dir=args.dpo_output_dir,
        per_device_train_batch_size=args.dpo_batch_size,
        gradient_accumulation_steps=args.dpo_grad_accum,
        num_train_epochs=args.dpo_epochs,
        max_steps=args.dpo_max_steps,
        learning_rate=args.dpo_learning_rate,
        beta=args.dpo_beta,
        max_length=args.dpo_max_length,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_steps=args.save_steps,
        save_strategy="steps",
        save_total_limit=2,
        report_to=report_to,
        run_name=run_name,
        save_only_model=not args.wandb_report_model,
        remove_unused_columns=False,
        optim="adamw_torch_fused",
        seed=args.dpo_seed,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[CheckpointLoggingCallback()],
    )
    SCRIPT_LOGGER.info("DPO trainer constructed; starting train()")
    trainer.train()
    log_cuda_memory("after-dpo-train")

    SCRIPT_LOGGER.info("Saving DPO adapter to %s", args.dpo_output_dir)
    trainer.save_model(args.dpo_output_dir)
    tokenizer.save_pretrained(args.dpo_output_dir)
    SCRIPT_LOGGER.info("DPO adapter saved to %s", args.dpo_output_dir)

    del trainer
    del model
    torch.cuda.empty_cache()

    import wandb as _wandb
    _wandb.finish()


def main() -> None:
    load_dotenv(dotenv_path=ENV_PATH, override=False)
    args = parse_args()

    if args.skip_sft and args.skip_dpo:
        SCRIPT_LOGGER.error("Both --skip-sft and --skip-dpo are set; nothing to do.")
        sys.exit(1)

    tokenizer = create_tokenizer(args.model_path)

    if not args.skip_sft:
        run_sft_phase(args, tokenizer)
    else:
        SCRIPT_LOGGER.info("Skipping SFT phase (--skip-sft); will load adapter from %s", args.sft_output_dir)
        if not Path(args.sft_output_dir).is_dir():
            SCRIPT_LOGGER.error("SFT adapter directory not found: %s", args.sft_output_dir)
            sys.exit(1)

    if not args.skip_dpo:
        run_dpo_phase(args, tokenizer)
    else:
        SCRIPT_LOGGER.info("Skipping DPO phase (--skip-dpo)")

    SCRIPT_LOGGER.info(
        "Pipeline complete. SFT adapter: %s  DPO adapter: %s",
        args.sft_output_dir,
        args.dpo_output_dir,
    )
    print(f"SFT adapter: {args.sft_output_dir}")
    print(f"DPO adapter: {args.dpo_output_dir}")


if __name__ == "__main__":
    main()
