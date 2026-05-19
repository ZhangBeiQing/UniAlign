#!/usr/bin/env python3
"""SFT LoRA fine-tuning entrypoint for Qwen-style chat models."""

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
from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from core.logging import get_logger, init_script_logger
from core.training import (
    CheckpointLoggingCallback,
    apply_chat_template_text,
    build_model,
    configure_wandb,
    create_tokenizer,
    load_config,
    log_cuda_memory,
    log_effective_batch,
    set_runtime_env,
)


DEFAULT_MODEL_PATH = "/root/autodl-tmp/Qwen3.5-4B-ortho"
DEFAULT_DATA_PATH = "reward_seed.json"
DEFAULT_OUTPUT_DIR = "/root/autodl-tmp/UniAlign-sft-ortho-lora"
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "train_sft_lora.yaml"
ENV_PATH = REPO_ROOT / ".env"

SCRIPT_LOGGER = init_script_logger("TrainSftLora")
DATA_LOGGER = get_logger("SFTData")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run safety-oriented SFT LoRA training on reward_seed.json."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--model-path")
    parser.add_argument("--data-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--target", choices=["chosen", "rejected"])
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--epochs", type=float)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accum", type=int)
    parser.add_argument("--logging-steps", type=int)
    parser.add_argument("--save-steps", type=int)
    parser.add_argument("--lora-r", type=int)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--lora-dropout", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-mode")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-group")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--wandb-report-model", action="store_true")
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    output_cfg = config.get("output", {})
    training_cfg = config.get("training", {})
    lora_cfg = config.get("lora", {})
    wandb_cfg = config.get("wandb", {})
    runtime_cfg = config.get("runtime", {})

    set_runtime_env(runtime_cfg)

    args.model_path = args.model_path or model_cfg.get("path", DEFAULT_MODEL_PATH)
    args.data_path = args.data_path or data_cfg.get("path", DEFAULT_DATA_PATH)
    args.output_dir = args.output_dir or output_cfg.get("dir", DEFAULT_OUTPUT_DIR)
    args.target = args.target or data_cfg.get("target", "chosen")
    args.exclude_indices = set(data_cfg.get("exclude_indices", []))
    args.max_target_chars = data_cfg.get("max_target_chars")
    args.enable_thinking = bool(data_cfg.get("enable_thinking", False))
    args.max_length = args.max_length if args.max_length is not None else data_cfg.get("max_length", 1024)
    args.epochs = args.epochs if args.epochs is not None else training_cfg.get("epochs", 5.0)
    args.max_steps = args.max_steps if args.max_steps is not None else training_cfg.get("max_steps", -1)
    args.learning_rate = (
        args.learning_rate if args.learning_rate is not None else training_cfg.get("learning_rate", 1e-5)
    )
    args.batch_size = args.batch_size if args.batch_size is not None else training_cfg.get("batch_size", 1)
    args.grad_accum = args.grad_accum if args.grad_accum is not None else training_cfg.get("grad_accum", 8)
    args.logging_steps = (
        args.logging_steps if args.logging_steps is not None else training_cfg.get("logging_steps", 1)
    )
    args.save_steps = args.save_steps if args.save_steps is not None else training_cfg.get("save_steps", 20)
    args.lora_r = args.lora_r if args.lora_r is not None else lora_cfg.get("r", 32)
    args.lora_alpha = args.lora_alpha if args.lora_alpha is not None else lora_cfg.get("alpha", 64)
    args.lora_dropout = (
        args.lora_dropout if args.lora_dropout is not None else lora_cfg.get("dropout", 0.05)
    )
    args.seed = args.seed if args.seed is not None else training_cfg.get("seed", 42)
    args.wandb_project = args.wandb_project or os.environ.get("WANDB_PROJECT") or wandb_cfg.get("project", "unialign")
    args.wandb_entity = args.wandb_entity or os.environ.get("WANDB_ENTITY") or wandb_cfg.get("entity")
    args.wandb_mode = args.wandb_mode or os.environ.get("WANDB_MODE") or wandb_cfg.get("mode", "online")
    args.wandb_run_name = args.wandb_run_name or os.environ.get("WANDB_RUN_NAME") or wandb_cfg.get("run_name")
    args.wandb_group = args.wandb_group or os.environ.get("WANDB_GROUP") or wandb_cfg.get("group", "qwen35-sft-lora")
    args.config_data = config
    args.use_4bit = bool(model_cfg.get("use_4bit", True)) and not args.no_4bit
    args.lora_target_modules = lora_cfg.get(
        "target_modules",
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    if not args.wandb_report_model:
        args.wandb_report_model = bool(wandb_cfg.get("report_model", False))
    if wandb_cfg.get("enabled") is False:
        args.disable_wandb = True
    return args


def load_sft_dataset(
    data_path: str | Path,
    tokenizer,
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
    DATA_LOGGER.info(
        "Loaded %d SFT examples (target=%s, skipped=%d) from %s",
        len(examples),
        target,
        skipped,
        path,
    )
    return Dataset.from_list(examples)


def main() -> None:
    load_dotenv(dotenv_path=ENV_PATH, override=False)
    args = parse_args()

    report_to, run_name = configure_wandb(
        disable_wandb=args.disable_wandb,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_group=args.wandb_group,
        wandb_report_model=args.wandb_report_model,
        output_dir=args.output_dir,
    )

    SCRIPT_LOGGER.info(
        "Starting SFT LoRA training: config=%s model=%s data=%s output=%s target=%s epochs=%s max_steps=%s batch=%s grad_accum=%s max_length=%s use_4bit=%s",
        args.config,
        args.model_path,
        args.data_path,
        args.output_dir,
        args.target,
        args.epochs,
        args.max_steps,
        args.batch_size,
        args.grad_accum,
        args.max_length,
        args.use_4bit,
    )

    log_effective_batch(args.batch_size, args.grad_accum)

    if report_to == "wandb":
        get_logger("WandbSetup").info(
            "WandB config summary: project=%s entity=%s mode=%s run_name=%s group=%s",
            args.wandb_project,
            args.wandb_entity or "",
            args.wandb_mode,
            run_name,
            args.wandb_group,
        )

    tokenizer = create_tokenizer(args.model_path)
    train_dataset = load_sft_dataset(
        args.data_path,
        tokenizer,
        args.target,
        exclude_indices=args.exclude_indices,
        max_target_chars=args.max_target_chars,
        enable_thinking=args.enable_thinking,
    )
    model = build_model(args.model_path, use_4bit=args.use_4bit)
    log_cuda_memory("after-model-load")

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.lora_target_modules,
    )

    sft_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
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
        seed=args.seed,
        completion_only_loss=True,
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
    log_cuda_memory("after-train")
    SCRIPT_LOGGER.info("Training finished; saving adapter to %s", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    SCRIPT_LOGGER.info("Saved LoRA adapter and tokenizer to %s", args.output_dir)
    print(f"Saved LoRA adapter and tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
