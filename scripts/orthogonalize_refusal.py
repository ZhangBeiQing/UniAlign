#!/usr/bin/env python3
"""Permanently bake refusal-direction removal into Qwen3.5-4B weights.

Applies weight orthogonalization on CPU to avoid GPU OOM. The output model
can be loaded with 4-bit quantization by the training pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DIRECTION_DIR = Path("/root/program/refusal_direction/pipeline/runs/Qwen3.5-4B")

DEFAULT_MODEL_PATH = "/root/autodl-tmp/Qwen3.5-4B"
DEFAULT_OUTPUT_PATH = "/root/autodl-tmp/Qwen3.5-4B-ortho"


def orthogonalize_matrix(matrix: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    vec = vec / torch.norm(vec)
    vec = vec.to(device=matrix.device, dtype=matrix.dtype)
    proj = (matrix @ vec.unsqueeze(-1)) * vec
    return matrix - proj


def orthogonalize_qwen_weights(model, direction: torch.Tensor):
    direction = direction.cpu().to(torch.float32)

    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        text_model = model.model.language_model
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        text_model = model.model
    elif hasattr(model, "transformer"):
        text_model = model.transformer
    else:
        raise NotImplementedError("Unsupported Qwen architecture.")

    text_model.embed_tokens.weight.data = orthogonalize_matrix(
        text_model.embed_tokens.weight.data.float(), direction
    ).to(text_model.embed_tokens.weight.dtype)

    total = len(text_model.layers)
    for i, block in enumerate(text_model.layers):
        if i % 4 == 0:
            print(f"  layer {i}/{total}")

        if hasattr(block, "self_attn") and hasattr(block.self_attn, "o_proj"):
            attn_out = block.self_attn.o_proj
        elif hasattr(block, "linear_attn") and hasattr(block.linear_attn, "out_proj"):
            attn_out = block.linear_attn.out_proj
        else:
            raise NotImplementedError("Cannot resolve attention output projection.")

        attn_out.weight.data = orthogonalize_matrix(
            attn_out.weight.data.T.float(), direction
        ).T.to(attn_out.weight.dtype)

        block.mlp.down_proj.weight.data = orthogonalize_matrix(
            block.mlp.down_proj.weight.data.T.float(), direction
        ).T.to(block.mlp.down_proj.weight.dtype)

    print(f"  layer {total}/{total} done")


def main():
    direction_path = DIRECTION_DIR / "direction.pt"
    metadata_path = DIRECTION_DIR / "direction_metadata.json"

    if not direction_path.exists():
        raise FileNotFoundError(f"Refusal direction not found at {direction_path}")

    print(f"Loading refusal direction from {direction_path}")
    direction = torch.load(direction_path, map_location="cpu", weights_only=True)
    with open(metadata_path) as f:
        metadata = json.load(f)
    print(f"  shape={list(direction.shape)} pos={metadata['pos']} layer={metadata['layer']}")

    print(f"\nLoading model from {DEFAULT_MODEL_PATH} (CPU)")
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL_PATH,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
    )
    print(f"  model loaded, dtype={next(model.parameters()).dtype}")

    print("Applying weight orthogonalization (CPU, removing refusal direction)...")
    orthogonalize_qwen_weights(model, direction)

    print(f"\nSaving orthogonalized model to {DEFAULT_OUTPUT_PATH}")
    model.save_pretrained(
        DEFAULT_OUTPUT_PATH,
        safe_serialization=True,
        max_shard_size="5GB",
    )

    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_PATH, trust_remote_code=True)
    tokenizer.save_pretrained(DEFAULT_OUTPUT_PATH)

    print(f"\nDone! Model saved to {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
