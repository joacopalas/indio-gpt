import os
import json
from collections import OrderedDict
from dataclasses import fields
from typing import Iterator

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from .model import (
    Config,
    TinyTransformer,
    SONG_TOK,
    END_TOK,
)

MODELS_DIR = os.environ.get("MODELS_DIR", "models")
MAX_LOADED = int(os.environ.get("MAX_LOADED", "1"))

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))


def load_index():
    index_path = os.path.join(MODELS_DIR, "index.json")
    with open(index_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    return {item["id"]: item for item in items}


MODEL_INDEX = load_index()
MODEL_CACHE: OrderedDict[str, dict] = OrderedDict()


def clean_state(state_dict):
    return {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}


def get_bundle(candidate: str):
    if candidate not in MODEL_INDEX:
        raise KeyError(f"Unknown candidate: {candidate}")

    if candidate in MODEL_CACHE:
        MODEL_CACHE.move_to_end(candidate)
        return MODEL_CACHE[candidate]

    meta = MODEL_INDEX[candidate]
    ckpt_path = os.path.join(MODELS_DIR, meta["file"])

    ckpt = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    tokenizer = Tokenizer.from_str(ckpt["tokenizer_json"])

    valid_keys = {f.name for f in fields(Config)}
    cfg = Config(**{
        k: v
        for k, v in ckpt["config"].items()
        if k in valid_keys
    })

    cfg.compile_model = False
    cfg.dropout = 0.0  # inference only

    model = TinyTransformer(cfg, ckpt["vocab_size"])
    model.load_state_dict(clean_state(ckpt["model"]))
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    bundle = {
        "model": model,
        "tokenizer": tokenizer,
        "cfg": cfg,
        "meta": meta,
        "ckpt": ckpt,
    }

    MODEL_CACHE[candidate] = bundle

    while len(MODEL_CACHE) > MAX_LOADED:
        MODEL_CACHE.popitem(last=False)

    return bundle


def sample_next(logits, temperature=1.0, top_k=40, top_p=0.95):
    logits = logits / max(temperature, 1e-6)

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        v, _ = torch.topk(logits, k)
        logits = torch.where(
            logits < v[-1],
            torch.full_like(logits, float("-inf")),
            logits,
        )

    if top_p is not None and 0 < top_p < 1:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1),
            dim=-1,
        )

        remove = cumulative_probs > top_p
        remove[0] = False

        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter_(
            0,
            sorted_indices,
            sorted_logits,
        )

    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())


def generate_events(bundle, req) -> Iterator[dict]:
    model = bundle["model"]
    tok = bundle["tokenizer"]
    cfg = bundle["cfg"]

    if req.seed is not None:
        torch.manual_seed(req.seed)

    prompt = req.prompt if req.prompt else f"{SONG_TOK}\n"

    with torch.inference_mode():
        ids = tok.encode(prompt).ids
        if not ids:
            ids = tok.encode("\n").ids

        idx = torch.tensor([ids], dtype=torch.long)

        if idx.size(1) > cfg.block_size:
            idx = idx[:, -cfg.block_size:]

        end_id = tok.token_to_id(END_TOK)

        keep = req.slide_keep or cfg.block_size // 2
        keep = max(1, min(keep, cfg.block_size - 1))

        caches = [None] * cfg.n_layer

        logits, _ = model(idx, caches=caches, start_pos=0)
        pos = idx.size(1)

        def decode_current():
            return tok.decode(
                idx[0].tolist(),
                skip_special_tokens=not req.include_specials,
            )

        prev_full = decode_current()
        initial_len = len(prev_full)

        yield {
            "type": "start",
            "candidate": req.candidate,
        }

        hit_end = False

        for _ in range(req.max_new_tokens):
            lg = logits[0, -1, :].float()

            if req.rep_penalty != 1.0 and req.rep_window > 0:
                recent = idx[0, -req.rep_window:].tolist()
                for t in set(recent):
                    lg[t] = (
                        lg[t] / req.rep_penalty
                        if lg[t] > 0
                        else lg[t] * req.rep_penalty
                    )

            nxt = sample_next(
                lg,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            )

            if req.stop_at_end and end_id is not None and nxt == end_id:
                hit_end = True
                break

            idx = torch.cat(
                (idx, torch.tensor([[nxt]], dtype=torch.long)),
                dim=1,
            )

            if pos >= cfg.block_size:
                caches = [None] * cfg.n_layer
                win = idx[:, -keep:]
                logits, _ = model(win, caches=caches, start_pos=0)
                pos = win.size(1)
            else:
                logits, _ = model(idx[:, -1:], caches=caches, start_pos=pos)
                pos += 1

            cur_full = decode_current()

            if cur_full != prev_full:
                if cur_full.startswith(prev_full):
                    delta = cur_full[len(prev_full):]
                    if delta:
                        yield {
                            "type": "delta",
                            "text": delta,
                        }
                else:
                    yield {
                        "type": "replace",
                        "text": cur_full[initial_len:],
                    }

                prev_full = cur_full

        text = prev_full[initial_len:]

        yield {
            "type": "done",
            "text": text,
            "hit_end": hit_end,
        }


def generate_full(bundle, req):
    text = ""
    hit_end = False

    for event in generate_events(bundle, req):
        if event["type"] == "done":
            text = event["text"]
            hit_end = event["hit_end"]

    return text, hit_end