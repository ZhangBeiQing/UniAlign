"""Batch inference for Qwen3.5 over ``sft_question.json``.

The script can run the base model as-is or apply refusal-direction weight
orthogonalization in memory before generation. Output format matches
sft_gemma4_4b_answers.json: [{"question": ..., "ai": ...}, ...].
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.logging import get_logger


LOGGER = get_logger("BatchInferQwen")

DEFAULT_MODEL_PATH = "/root/autodl-tmp/Qwen3.5-4B"
DEFAULT_DATA_PATH = "sft_question.json"
DEFAULT_DIRECTION_PATH = "/root/program/geometry-of-refusal/results/dim/Qwen3.5-4B/direction.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--direction-path", default=None)
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--model-class", choices=("image-text", "causal-lm"), default="image-text")
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def apply_chat_template(tokenizer, instruction: str) -> str:
    messages = [{"role": "user", "content": instruction}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template_kwargs={"enable_thinking": False},
        )


def get_qwen_transformer(model):
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    if hasattr(model, "transformer"):
        return model.transformer
    raise NotImplementedError(f"Unsupported Qwen architecture: {type(model)}")


def orthogonalize_matrix(matrix: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    unit_direction = direction / (direction.norm() + 1e-8)
    unit_direction = unit_direction.to(device=matrix.device, dtype=matrix.dtype)
    return matrix - (matrix @ unit_direction.unsqueeze(-1)) * unit_direction


def orthogonalize_qwen35_weights(model, direction: torch.Tensor) -> None:
    transformer = get_qwen_transformer(model)
    direction = direction.detach().flatten().float()

    transformer.embed_tokens.weight.data = orthogonalize_matrix(
        transformer.embed_tokens.weight.data,
        direction,
    )

    for layer_idx, block in enumerate(transformer.layers):
        if layer_idx % 4 == 0:
            LOGGER.info("orthogonalizing layer %s/%s", layer_idx, len(transformer.layers))

        if hasattr(block, "linear_attn"):
            out_proj = block.linear_attn.out_proj
        elif hasattr(block, "self_attn"):
            out_proj = block.self_attn.o_proj
        else:
            raise NotImplementedError(f"Cannot resolve residual writer at layer {layer_idx}")

        out_proj.weight.data = orthogonalize_matrix(
            out_proj.weight.data.T,
            direction,
        ).T
        block.mlp.down_proj.weight.data = orthogonalize_matrix(
            block.mlp.down_proj.weight.data.T,
            direction,
        ).T

    LOGGER.info("orthogonalizing layer %s/%s done", len(transformer.layers), len(transformer.layers))


def load_model_and_tokenizer(model_path: str, dtype: torch.dtype, model_class: str):
    LOGGER.info("loading tokenizer: %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    LOGGER.info("loading model: %s", model_path)
    model_cls = AutoModelForCausalLM if model_class == "causal-lm" else AutoModelForImageTextToText
    model = model_cls.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=dtype,
        device_map="auto",
    ).eval()
    model.requires_grad_(False)
    return model, tokenizer


def load_lora_adapter(model, lora_path: str):
    LOGGER.info("loading LoRA adapter: %s", lora_path)
    model = PeftModel.from_pretrained(model, lora_path).eval()
    model.requires_grad_(False)
    return model


def generate_batch(model, tokenizer, instructions: list[str], max_new_tokens: int) -> list[str]:
    prompts = [apply_chat_template(tokenizer, instruction) for instruction in instructions]
    inputs = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
        add_special_tokens=False,
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output[:, inputs["input_ids"].shape[-1] :]
    return [text.strip() for text in tokenizer.batch_decode(generated, skip_special_tokens=True)]


def main() -> None:
    args = parse_args()
    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    instructions = [item["instruction"] for item in data]

    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        resolve_dtype(args.dtype),
        args.model_class,
    )
    if args.direction_path:
        LOGGER.info("loading refusal direction: %s", args.direction_path)
        direction = torch.load(args.direction_path, map_location="cpu", weights_only=True)
        LOGGER.info("direction shape=%s norm=%.6f", tuple(direction.shape), direction.float().norm().item())
        orthogonalize_qwen35_weights(model, direction)
    if args.lora_path:
        model = load_lora_adapter(model, args.lora_path)

    results = []
    total = len(instructions)
    for start in range(0, total, args.batch_size):
        end = min(start + args.batch_size, total)
        LOGGER.info("generating %s-%s/%s", start, end, total)
        answers = generate_batch(
            model,
            tokenizer,
            instructions[start:end],
            max_new_tokens=args.max_new_tokens,
        )
        for question, answer in zip(instructions[start:end], answers):
            results.append({"question": question, "ai": answer})

    out_path = Path(args.out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    LOGGER.info("wrote %s answers to %s", len(results), out_path)


if __name__ == "__main__":
    main()
