"""Shared training utilities for UniAlign training scripts."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback

from .logging import get_logger


def load_config(config_path: str | Path) -> dict:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_model(model_path: str, use_4bit: bool) -> AutoModelForCausalLM:
    kwargs = {
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    if use_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model_logger = get_logger("ModelLoader")
    model_logger.info(
        "Loading model from %s with use_4bit=%s, attn_implementation=%s",
        model_path,
        use_4bit,
        kwargs["attn_implementation"],
    )
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.config.use_cache = False
    return model


def load_model_with_adapter(
    model_path: str,
    adapter_path: str | Path,
    use_4bit: bool = True,
    is_trainable: bool = True,
) -> PeftModel:
    model_logger = get_logger("ModelLoader")
    model_logger.info("Loading base model from %s", model_path)
    base_model = build_model(model_path, use_4bit=use_4bit)
    model_logger.info("Loading adapter from %s", adapter_path)
    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        is_trainable=is_trainable,
    )
    return model


def get_world_size() -> int:
    value = os.environ.get("WORLD_SIZE", "1").strip()
    try:
        return max(1, int(value))
    except ValueError:
        return 1


def log_cuda_memory(prefix: str) -> None:
    metrics_logger = get_logger("TrainingMetrics")
    if not torch.cuda.is_available():
        metrics_logger.info("%s cuda unavailable", prefix)
        return

    device = torch.cuda.current_device()
    allocated_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
    reserved_gb = torch.cuda.memory_reserved(device) / (1024 ** 3)
    max_allocated_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    max_reserved_gb = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
    total_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    metrics_logger.info(
        "%s gpu=%s allocated=%.2fGB reserved=%.2fGB max_allocated=%.2fGB max_reserved=%.2fGB total=%.2fGB",
        prefix,
        torch.cuda.get_device_name(device),
        allocated_gb,
        reserved_gb,
        max_allocated_gb,
        max_reserved_gb,
        total_gb,
    )


def create_tokenizer(model_path: str) -> AutoTokenizer:
    script_logger = get_logger("TokenizerSetup")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    script_logger.info(
        "Tokenizer ready: pad_token_id=%s eos_token_id=%s padding_side=%s",
        tokenizer.pad_token_id,
        tokenizer.eos_token_id,
        tokenizer.padding_side,
    )
    return tokenizer


def apply_chat_template_text(
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
    add_generation_prompt: bool,
    enable_thinking: bool = False,
) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            chat_template_kwargs={"enable_thinking": enable_thinking},
        )


class CheckpointLoggingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_logger = get_logger("Checkpointing")
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        checkpoint_logger.info(
            "Checkpoint saved at global_step=%s path=%s",
            state.global_step,
            checkpoint_dir,
        )
        log_cuda_memory("after-checkpoint")
        return control


def configure_wandb(
    disable_wandb: bool,
    wandb_mode: str,
    wandb_project: str,
    wandb_entity: str | None = None,
    wandb_run_name: str | None = None,
    wandb_group: str | None = None,
    wandb_report_model: bool = False,
    output_dir: str = "",
) -> tuple[str, str | None]:
    wandb_logger = get_logger("WandbSetup")
    api_key = os.environ.get("WANDB_API_KEY", "").strip()

    if disable_wandb:
        os.environ["WANDB_DISABLED"] = "true"
        wandb_logger.info("WandB disabled by --disable-wandb")
        return "none", None

    if not api_key:
        os.environ["WANDB_DISABLED"] = "true"
        wandb_logger.warning("WANDB_API_KEY not found; training will continue without WandB")
        return "none", None

    os.environ["WANDB_MODE"] = wandb_mode
    os.environ["WANDB_PROJECT"] = wandb_project
    if wandb_entity:
        os.environ["WANDB_ENTITY"] = wandb_entity
    if wandb_run_name:
        os.environ["WANDB_NAME"] = wandb_run_name
    if wandb_group:
        os.environ["WANDB_RUN_GROUP"] = wandb_group

    run_name = wandb_run_name or Path(output_dir).name if output_dir else wandb_run_name
    wandb_logger.info(
        "WandB enabled: project=%s, entity=%s, mode=%s, run_name=%s, group=%s, report_model=%s",
        wandb_project,
        wandb_entity or "",
        wandb_mode,
        run_name,
        wandb_group,
        wandb_report_model,
    )
    return "wandb", run_name


def set_runtime_env(runtime_cfg: dict) -> None:
    os.environ.setdefault("FLA_TILELANG", str(runtime_cfg.get("fla_tilelang", "0")))
    os.environ.setdefault("TRITON_F32_DEFAULT", str(runtime_cfg.get("triton_f32_default", "ieee")))


def log_effective_batch(batch_size: int, grad_accum: int) -> None:
    metrics_logger = get_logger("TrainingMetrics")
    world_size = get_world_size()
    effective = batch_size * grad_accum * world_size
    metrics_logger.info(
        "Effective batch size: per_device=%s grad_accum=%s world_size=%s effective=%s",
        batch_size,
        grad_accum,
        world_size,
        effective,
    )
