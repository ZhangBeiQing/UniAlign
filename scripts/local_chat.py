import argparse
import json
import os
import select
import sys
import termios
import threading
import tty
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer


DEFAULT_MODEL_PATH = "/root/autodl-tmp/Qwen3.5-4B-Base"


class StopOnEvent(StoppingCriteria):
    def __init__(self, stop_event: threading.Event):
        self.stop_event = stop_event

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self.stop_event.is_set()


def parse_args():
    parser = argparse.ArgumentParser(description="Chat with a local Hugging Face causal LM.")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--prompt", type=str, default=None, help="Run one prompt and exit.")
    parser.add_argument("--interactive", action="store_true", help="Start a multi-turn chat loop.")
    parser.add_argument("--system", type=str, default=None, help="Optional system prompt.")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--no_sample", action="store_true", help="Use greedy decoding.")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--history_path", type=str, default=None, help="Load/save chat history as JSON.")
    parser.add_argument("--no_stream", action="store_true", help="Disable streaming output.")
    return parser.parse_args()


def resolve_dtype(dtype: str):
    if dtype == "auto":
        return "auto"
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]


def load_history(path: str | None) -> list[dict[str, str]]:
    if path is None or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        history = json.load(f)
    if not isinstance(history, list):
        raise ValueError(f"History file must contain a JSON list: {path}")
    return history


def save_history(path: str | None, history: list[dict[str, str]]) -> None:
    if path is None:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def apply_chat_template(tokenizer, messages: list[dict[str, str]]) -> str:
    if not getattr(tokenizer, "chat_template", None):
        parts = []
        for message in messages:
            role = message["role"].capitalize()
            parts.append(f"{role}: {message['content']}")
        parts.append("Assistant:")
        return "\n\n".join(parts)

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


def build_messages(system_prompt: str | None, history: list[dict[str, str]]) -> list[dict[str, str]]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history)
    return messages


def generation_kwargs(args, tokenizer) -> dict[str, Any]:
    do_sample = not args.no_sample and args.temperature > 0
    eos_token_ids = []
    for token_id in (tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")):
        if isinstance(token_id, int) and token_id >= 0 and token_id not in eos_token_ids:
            eos_token_ids.append(token_id)

    kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": eos_token_ids or tokenizer.eos_token_id,
        "repetition_penalty": args.repetition_penalty,
    }
    if do_sample:
        kwargs.update(
            {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
            }
        )
    return kwargs


def listen_for_escape(stop_event: threading.Event, done_event: threading.Event) -> None:
    if not sys.stdin.isatty():
        return
    while not done_event.is_set() and not stop_event.is_set():
        readable, _, _ = select.select([sys.stdin], [], [], 0.05)
        if not readable:
            continue
        if sys.stdin.read(1) == "\x1b":
            stop_event.set()
            print("\n[ESC pressed: stopping generation]", flush=True)
            return


def generate_response(model, tokenizer, messages: list[dict[str, str]], args) -> str:
    prompt = apply_chat_template(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    kwargs = generation_kwargs(args, tokenizer)

    if args.no_stream:
        with torch.inference_mode():
            output = model.generate(**inputs, **kwargs)
        generated = output[0, inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    stop_event = threading.Event()
    done_event = threading.Event()
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    thread = threading.Thread(
        target=model.generate,
        kwargs={
            **inputs,
            **kwargs,
            "streamer": streamer,
            "stopping_criteria": StoppingCriteriaList([StopOnEvent(stop_event)]),
        },
    )
    thread.start()

    old_terminal_attrs = None
    escape_thread = None
    chunks = []
    try:
        if sys.stdin.isatty():
            old_terminal_attrs = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())
            escape_thread = threading.Thread(
                target=listen_for_escape,
                args=(stop_event, done_event),
                daemon=True,
            )
            escape_thread.start()

        print("\nAssistant> ", end="", flush=True)
        for text in streamer:
            chunks.append(text)
            print(text, end="", flush=True)
    finally:
        done_event.set()
        thread.join()
        if escape_thread is not None:
            escape_thread.join(timeout=0.2)
        if old_terminal_attrs is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_terminal_attrs)
        print()
    return "".join(chunks).strip()


def read_user_message() -> str | None:
    print("\nUser> ", end="", flush=True)
    try:
        first_line = input()
    except EOFError:
        return None

    command = first_line.strip()
    if command.lower() in {"exit", "quit", ":q"}:
        return "exit"
    if command in {":reset", ":history"}:
        return command
    if command != ":paste":
        return first_line.strip()

    print("Paste mode. End with :send on its own line.")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            return None
        if line.strip() == ":send":
            break
        lines.append(line)

    return "\n".join(lines).strip()


def interactive_loop(model, tokenizer, args, history: list[dict[str, str]]) -> None:
    print("Commands: exit/quit/:q to leave, :reset to clear history, :history to show history.")
    print("For multi-line input, type :paste first, then end with :send on its own line.")

    while True:
        user_message = read_user_message()
        if user_message is None or user_message == "exit":
            break
        if user_message == ":reset":
            history.clear()
            save_history(args.history_path, history)
            print("History cleared.")
            continue
        if user_message == ":history":
            print(json.dumps(history, ensure_ascii=False, indent=2))
            continue
        if not user_message:
            continue

        history.append({"role": "user", "content": user_message})
        messages = build_messages(args.system, history)
        response = generate_response(model, tokenizer, messages, args)
        if args.no_stream:
            print(f"\nAssistant> {response}")
        history.append({"role": "assistant", "content": response})
        save_history(args.history_path, history)


def main():
    args = parse_args()
    if args.prompt is None and not args.interactive:
        args.interactive = True

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        dtype=resolve_dtype(args.dtype),
        device_map=args.device_map,
    ).eval()
    model.requires_grad_(False)

    history = load_history(args.history_path)

    if args.prompt is not None:
        history.append({"role": "user", "content": args.prompt})
        response = generate_response(model, tokenizer, build_messages(args.system, history), args)
        if args.no_stream:
            print(response)
        history.append({"role": "assistant", "content": response})
        save_history(args.history_path, history)

    if args.interactive:
        interactive_loop(model, tokenizer, args, history)


if __name__ == "__main__":
    main()
