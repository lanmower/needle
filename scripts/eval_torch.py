"""Score the held-out preference set with the torch port, on the GPU.

`scripts/eval_pref.py` scores the same rows with JAX on the CPU and needs about
90 minutes for 147 of them; this does it in seconds. Both score identically, so
what licenses comparing a number from one against a number from the other is the
`gate` mode of `torch_finetune.py`, which proves the two forward passes agree.

Usage:
  python scripts/eval_torch.py                                   # untuned base
  python scripts/eval_torch.py --adapter checkpoints/pref_lora_gpu.safetensors
"""

import argparse
import json
import os
import sys
import time
from collections import Counter

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from needle.model.checkpoints import read_adapter  # noqa: E402
from needle.model.finetune import (  # noqa: E402
    BOS_ID, EOS_ID, _calls_of, render_example,
)
from needle.model.tokenizer import get_tokenizer  # noqa: E402

import torch_finetune as T  # noqa: E402


def predicted(text, tool_name):
    """The resolution the model chose, or None when it made no usable call."""
    calls = _calls_of(text)
    if not calls:
        return None, "no_call"
    if len(calls) != 1:
        return None, "multi_call"
    name, args = calls[0]
    if name != tool_name:
        return None, "wrong_tool"
    try:
        return json.loads(args).get("resolution"), "ok"
    except ValueError:
        return None, "bad_json"


def freeze_kernels(model):
    """Hold one set of quantised weights for the whole scoring pass.

    `NeedleTorch.kernels()` rebuilds every weight and re-runs CQ W4 on each
    forward. During inference the LoRA does not change, so that is the same
    answer recomputed once per decoded token instead of once per pass -- it
    dominated the cost of scoring.
    """
    with torch.no_grad():
        frozen = model.kernels()
    model.kernels = lambda: frozen
    return len(frozen)


_ORIGIN_GROUPS = {"rows_labeled": "labeled", "rows_neutral": "neutral",
                  "rows_neutral_extra": "neutral", "rows_refusal": "refusal"}


def load_origins(data_dir):
    """The authored file each held-out row came from, matched on query text.

    Rows carry no provenance field, and the held-out set mixes real decisions
    with rehearsal rows. A headline number that blends them cannot say whether
    the preference was learned, so every score is also split by origin.
    """
    sources = {}
    for filename, origin in _ORIGIN_GROUPS.items():
        path = os.path.join(data_dir, filename + ".jsonl")
        if not os.path.exists(path):
            continue
        queries = {(json.loads(line).get("query") or "").strip()
                   for line in open(path, encoding="utf-8") if line.strip()}
        sources.setdefault(origin, set()).update(queries)
    return sources


def origin_of(row, sources):
    query = (row.get("query") or "").strip()
    for origin, queries in sources.items():
        if query in queries:
            return origin
    return "unknown"


def generate(model, prompt_ids, max_new_tokens, device):
    """Greedy decode. No sampler and no temperature: the deployment does greedy."""
    ids = torch.tensor([[BOS_ID] + list(prompt_ids)], device=device)
    out = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            nxt = int(model(ids)[0, -1].argmax())
            if nxt == EOS_ID:
                break
            out.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
    return out


def load_adapter(model, path, device):
    """Inject a saved LoRA into the live model the way `kernels()` reads it.

    The adapter is stored per target path; `model.names` is the same ordered
    list `lora_target_paths` produced at setup, so the index lookup is exact.
    """
    adapter = read_adapter(path)
    lora = adapter["lora"]
    scale = float(adapter["scale"])
    for i, name in enumerate(model.names):
        entry = lora["/".join(name)] if isinstance(name, (list, tuple)) else lora[name]
        model.A[i] = torch.nn.Parameter(
            torch.as_tensor(entry["A"], device=device).float(), requires_grad=False)
        model.B[i] = torch.nn.Parameter(
            torch.as_tensor(entry["B"], device=device).float(), requires_grad=False)
    model.scale = scale
    return scale


def report(outcomes, label):
    answered = [o for o in outcomes if o["gold"] is not None]
    refusals = [o for o in outcomes if o["gold"] is None]
    exact = sum(1 for o in outcomes if o["exact"])
    print(f"--- {label} ---")
    print(f"  rows         {len(outcomes)}  ({len(answered)} with a gold call, "
          f"{len(refusals)} refusal)")
    print(f"  exact        {exact}/{len(outcomes)}  {exact / len(outcomes):.1%}")

    if refusals:
        false_calls = sum(1 for o in refusals if o["got"] is not None)
        print(f"  false calls  {false_calls}/{len(refusals)} refused rows drew a call "
              f"({false_calls / len(refusals):.1%})")

    print("  per-class recall (of rows whose gold is that class):")
    by_gold = Counter(o["gold"] for o in answered)
    for klass, n in sorted(by_gold.items(), key=lambda kv: -kv[1]):
        hit = sum(1 for o in answered if o["gold"] == klass and o["got"] == klass)
        print(f"    {klass:<18} {hit}/{n}  {hit / n:.1%}")

    by_origin = Counter(o["origin"] for o in outcomes)
    if len(by_origin) > 1:
        print("  by origin (rehearsal rows are far easier than real decisions):")
        for origin, n in sorted(by_origin.items(), key=lambda kv: -kv[1]):
            hit = sum(1 for o in outcomes if o["origin"] == origin and o["exact"])
            print(f"    {origin:<18} {hit}/{n}  {hit / n:.1%}")
        from_rehearsal = sum(1 for o in outcomes
                             if o["origin"] == "neutral" and o["exact"])
        if exact:
            print(f"    {from_rehearsal}/{exact} correct answers came from "
                  f"rehearsal rows")

    got_hist = Counter(o["got"] for o in outcomes if o["got"] is not None)
    print("  predicted histogram:")
    for klass, n in got_hist.most_common():
        print(f"    {str(klass):<18} {n}")
    if len(outcomes) - sum(got_hist.values()):
        print(f"    {'<no usable call>':<18} {len(outcomes) - sum(got_hist.values())}")

    kinds = Counter(o["kind"] for o in outcomes)
    bad = {k: v for k, v in kinds.items() if k != "ok"}
    if bad:
        print(f"  unusable     {bad}")

    top, top_n = (got_hist.most_common(1)[0] if got_hist else (None, 0))
    if top is not None and top_n / len(outcomes) > 0.8:
        print(f"  COLLAPSE     {top_n}/{len(outcomes)} rows predicted {top!r} -- "
              f"the adapter learned one class, not the preference")

    majority = by_gold.most_common(1)[0][0] if by_gold else None
    if majority:
        base = sum(1 for o in answered if o["gold"] == majority)
        print(f"  baseline     constant-{majority} scores {base}/{len(answered)} "
              f"{base / len(answered):.1%} on the answered rows")
    return exact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pref_heldout.jsonl")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--show", type=int, default=0, help="print the first N generations")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    tool_name = json.loads(rows[0]["tools"])[0]["name"] if isinstance(
        rows[0]["tools"], str) else rows[0]["tools"][0]["name"]

    sources = load_origins("data")
    for row in rows:
        row["_origin"] = origin_of(row, sources)

    params, cfg, flat, paths, model = T.setup(args.device, 0, 16, 32.0)
    model.grad_ckpt = False
    model.eval()
    tokenizer = get_tokenizer(cfg.vocab_size)
    label = "base"
    if args.adapter:
        scale = load_adapter(model, args.adapter, torch.device(args.device))
        label = f"{os.path.basename(args.adapter)} (scale {scale})"

    frozen = freeze_kernels(model)
    print(f"  setup      {frozen} kernel groups frozen for the whole pass", flush=True)

    outcomes = []
    started = time.time()
    for index, row in enumerate(rows):
        prompt, _ = render_example({**row, "answers": []})
        ids = generate(model, tokenizer.encode(prompt), args.max_new_tokens,
                       torch.device(args.device))
        text = tokenizer.decode(ids)
        if index < args.show:
            print(f"  [{index}] {len(ids)} tok {text[:400]!r}", flush=True)
        want = row.get("answers") or []
        gold = (want[0].get("arguments") or {}).get("resolution") if want else None
        got, kind = predicted(text, tool_name)
        correct = (got == gold) if gold is not None else (got is None)
        outcomes.append({"gold": gold, "got": got, "kind": kind, "text": text,
                         "origin": row.get("_origin", "unknown"), "exact": correct})
        if index % 10 == 0:
            per = (time.time() - started) / (index + 1)
            print(f"  ... {index}/{len(rows)}  {per:.1f}s/row  "
                  f"eta {per * (len(rows) - index - 1) / 60:.1f}min", flush=True)

    report(outcomes, label)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(outcomes, fh, indent=1)
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
