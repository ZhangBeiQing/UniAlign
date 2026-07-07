"""批量推理：用 gemma-4-E4B-it 对 sft_question.json 中每个问题生成回复。

- 8 GPU 各起一个进程（mp.spawn），每个进程把模型加载到自己的 GPU
- 数据按 round-robin 分片，每个进程把分到的数据一次性 batch 推理（batch 拉满）
- max_new_tokens=2048 防止输出截断
- 每个 worker 写分片结果到临时文件，主进程合并并按原始顺序输出 sft_answers.json
"""
import argparse
import json
import logging
import os
import tempfile
import time

import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/mnt/d_7T/model/gemma-4-E4B-it"
DATA_PATH = "sft_question.json"
OUT_PATH = "sft_answers.json"
MAX_NEW_TOKENS = 2048

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("BatchInferGemma")


def _log(msg, rank=0):
    LOGGER.info(f"[rank {rank}] {msg}")


def worker(rank, world_size, shards, model_path, out_dir, max_new_tokens):
    """单个 GPU worker：加载模型、batch 推理、写分片结果。"""
    shard = shards[rank]
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)
    _log(f"绑定 {device}，加载 tokenizer/model", rank)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # 左侧 padding 适合批量生成
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    _log(f"模型就绪，开始处理 {len(shard)} 条", rank)

    # 构造 prompt
    prompts = [item["instruction"] for item in shard]
    input_ids_list = []
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        enc = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True, return_dict=False
        )
        input_ids_list.append(enc)

    pad_id = tokenizer.pad_token_id
    max_in = max(len(x) for x in input_ids_list)
    padded, attn = [], []
    for ids in input_ids_list:
        pad_len = max_in - len(ids)
        padded.append([pad_id] * pad_len + ids)  # left padding
        attn.append([0] * pad_len + [1] * len(ids))

    input_ids = torch.tensor(padded, dtype=torch.long, device=device)
    attention_mask = torch.tensor(attn, dtype=torch.long, device=device)
    _log(f"batch shape={tuple(input_ids.shape)}，开始 generate", rank)

    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
        )
    dt = time.time() - t0
    gen = out[:, input_ids.shape[1]:]
    texts = tokenizer.batch_decode(gen, skip_special_tokens=True)

    results = []
    for idx_in_shard, (item, text) in enumerate(zip(shard, texts)):
        # 还原原始索引：shard 是 data[rank::world_size]
        orig_idx = rank + idx_in_shard * world_size
        results.append(
            {"_idx": orig_idx, "question": item["instruction"], "ai": text.strip()}
        )
    _log(f"完成 {len(results)} 条，耗时 {dt:.1f}s", rank)

    part_path = os.path.join(out_dir, f"part{rank}.json")
    with open(part_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--data-path", default=DATA_PATH)
    parser.add_argument("--out-path", default=OUT_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--world-size", type=int, default=torch.cuda.device_count())
    args = parser.parse_args()

    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    _log(f"共 {len(data)} 条问题，使用 {args.world_size} 个 GPU")

    # round-robin 分片
    shards = [data[i::args.world_size] for i in range(args.world_size)]

    out_dir = tempfile.mkdtemp(prefix="gemma_infer_")
    _log(f"临时分片目录: {out_dir}")

    t0 = time.time()
    mp.spawn(
        worker,
        args=(
            args.world_size,
            shards,
            args.model_path,
            out_dir,
            args.max_new_tokens,
        ),
        nprocs=args.world_size,
        join=True,
    )
    _log(f"所有 worker 完成，总耗时 {time.time()-t0:.1f}s，开始合并")

    all_results = []
    for r in range(args.world_size):
        part_path = os.path.join(out_dir, f"part{r}.json")
        with open(part_path, "r", encoding="utf-8") as f:
            all_results.extend(json.load(f))

    all_results.sort(key=lambda x: x["_idx"])
    final = [{"question": r["question"], "ai": r["ai"]} for r in all_results]

    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    _log(f"已写出 {len(final)} 条结果到 {args.out_path}")


if __name__ == "__main__":
    main()
