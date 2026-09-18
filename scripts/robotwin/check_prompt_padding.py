#!/usr/bin/env python3
"""Measure how LingBot-VA prompt embeddings differ between tokenizer paddings.

Loads the upstream tokenizer and T5 text encoder exactly as ``VA_Server`` does
(CPU, config ``param_dtype``) and runs upstream ``_get_t5_prompt_embeds`` logic
twice per instruction: ``padding="max_length"`` (upstream) and ``"longest"``
(SimuGuard shim).  Reports max/mean absolute and relative differences of the
final 512-token embeddings (real tokens + zero padding) and wall time.

Usage (policy-server Python env)::

    CUDA_VISIBLE_DEVICES= python check_prompt_padding.py --server <wan_va_server.py> \\
        --model-dir <.merged_ckpt> --config robotwin30_train --instructions-json <task>.json \\
        --count 2 --report out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "simuguard" / "integrations"))

from lingbot_va_server_shim import load_upstream  # noqa: E402


def embeds(module, tokenizer, text_encoder, prompt: str, padding: str, dtype, max_len: int = 512):
    import torch

    cleaned = [module.prompt_clean(prompt)]
    inputs = tokenizer(
        cleaned, padding=padding, max_length=max_len, truncation=True,
        add_special_tokens=True, return_attention_mask=True, return_tensors="pt",
    )
    ids, mask = inputs.input_ids, inputs.attention_mask
    seq_len = int(mask.gt(0).sum(dim=1)[0])
    started = time.perf_counter()
    with torch.no_grad():
        hidden = text_encoder(ids, mask).last_hidden_state
    elapsed = time.perf_counter() - started
    hidden = hidden.to(dtype=dtype)[0, :seq_len]
    full = torch.cat([hidden, hidden.new_zeros(max_len - seq_len, hidden.size(1))])
    return full.float(), seq_len, tuple(ids.shape), elapsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--config", default="robotwin30_train")
    parser.add_argument("--instructions-json", required=True)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    import torch

    module = load_upstream(args.server)
    config = module.VA_CONFIGS[args.config]
    dtype = config.param_dtype
    started = time.perf_counter()
    tokenizer = module.load_tokenizer(os.path.join(args.model_dir, "tokenizer"))
    text_encoder = module.load_text_encoder(os.path.join(args.model_dir, "text_encoder"), torch_dtype=dtype, torch_device="cpu")
    load_s = time.perf_counter() - started

    data = json.loads(Path(args.instructions_json).read_text())
    prompts = (data.get("seen") or data.get("unseen") or [])[: args.count]
    results = []
    for prompt in prompts:
        a, len_a, shape_a, t_a = embeds(module, tokenizer, text_encoder, prompt, "max_length", dtype)
        b, len_b, shape_b, t_b = embeds(module, tokenizer, text_encoder, prompt, "longest", dtype)
        diff = (a - b).abs()
        real = diff[:len_a]
        scale = a[:len_a].abs().mean().item()
        results.append(
            {
                "prompt": prompt,
                "real_tokens": [len_a, len_b],
                "input_shapes": [list(shape_a), list(shape_b)],
                "encoder_seconds": {"max_length": round(t_a, 2), "longest": round(t_b, 2)},
                "bit_identical": bool(torch.equal(a, b)),
                "max_abs_diff": diff.max().item(),
                "mean_abs_diff_real_tokens": real.mean().item(),
                "mean_abs_embedding_real_tokens": scale,
                "max_rel_diff_real_tokens": (real.max().item() / scale) if scale else None,
                "cosine_similarity_real_tokens": torch.nn.functional.cosine_similarity(
                    a[:len_a].flatten(), b[:len_a].flatten(), dim=0
                ).item(),
                "padding_region_all_zero": bool((a[len_a:] == 0).all() and (b[len_b:] == 0).all()),
            }
        )
        print(json.dumps(results[-1]), flush=True)
    report = {
        "config": args.config,
        "param_dtype": str(dtype),
        "torch_threads": torch.get_num_threads(),
        "text_encoder_load_seconds": round(load_s, 1),
        "results": results,
    }
    Path(args.report).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
