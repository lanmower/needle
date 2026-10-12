#!/usr/bin/env python3
"""Score base and tuned needle on the untouched held-out file, per class.

`needle finetune` prints an accuracy at the end, but it is not a held-out
number: `_score_quantised` re-permutes the *entire* input file and takes the
first n_val of that permutation, while the training split removed its own
holdout by a different permutation of a differently-filtered list. So the
printed figure is mostly rows the model just trained on. This script scores
`data/pref_heldout.jsonl`, which `build_dataset.py` carved out before any
training file was written.

Aggregate accuracy is also the least useful number here. What matters is:

  * per-class recall -- a preference model that learns "proceed_now" for
    everything scores well on the majority class and is worthless
  * false calls on refusal rows -- the failure mode that makes the tuned
    model unusable in an agent loop
  * the predicted-class histogram -- collapse to one class is visible here
    and invisible in an aggregate

Usage:
  python -I scripts/eval_pref.py --data data/pref_heldout.jsonl
  python -I scripts/eval_pref.py --data data/pref_heldout.jsonl \
      --adapter checkpoints/needle_lora.safetensors
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def predicted(text, tool_name):
    """The resolution the model chose, or None when it made no usable call."""
    from needle.model.finetune import _calls_of

    calls = _calls_of(text)
    if not calls:
        return None, "no_call"
    if len(calls) != 1:
        return None, "multi_call"
    name, args = calls[0]
    if name != tool_name:
        return None, "wrong_tool"
    try:
        resolution = json.loads(args).get("resolution")
    except ValueError:
        return None, "bad_json"
    return resolution, "ok"


def score(model, params, tokenizer, rows, tool_name, dump):
    from needle.model.finetune import render_example
    from needle.model.run import generate

    outcomes = []
    for index, row in enumerate(rows):
        prompt, _ = render_example({**row, "answers": []})
        text = generate(model, params, tokenizer, prompt, max_new_tokens=96,
                        temperature=0.0, stream=False)
        want = row.get("answers") or []
        gold = (want[0].get("arguments") or {}).get("resolution") if want else None
        got, kind = predicted(text, tool_name)
        if dump and index < dump:
            print(f"\n--- row {index} gold={gold!r} got={got!r} ({kind})")
            print(f"    {text[:400]!r}")
        outcomes.append({"gold": gold, "got": got, "kind": kind,
                         "exact": kind == "ok" and got == gold and bool(want) is bool(got)})
    return outcomes


def report(label, outcomes):
    answers = [o for o in outcomes if o["gold"]]
    refusals = [o for o in outcomes if not o["gold"]]
    exact = sum(1 for o in outcomes if o["exact"])
    total = len(outcomes)

    print(f"\n===== {label} =====")
    print(f"exact calls      {exact}/{total}  ({exact / total:.1%})" if total else "no rows")

    if refusals:
        false_calls = sum(1 for o in refusals if o["got"] is not None)
        print(f"refusal rows     {len(refusals)}  false calls {false_calls} "
              f"({false_calls / len(refusals):.1%})")

    if answers:
        missed = sum(1 for o in answers if o["got"] is None)
        by_class = Counter(o["gold"] for o in answers)
        print(f"decision rows    {len(answers)}  emitted no call {missed} "
              f"({missed / len(answers):.1%})")
        print("per-class recall")
        for value, n in by_class.most_common():
            hit = sum(1 for o in answers if o["gold"] == value and o["got"] == value)
            print(f"    {hit:4d}/{n:<4d} {hit / n:6.1%}  {value}")

        hist = Counter(o["got"] for o in answers)
        top, top_n = hist.most_common(1)[0]
        print(f"predicted        {dict(hist.most_common())}")
        majority = by_class.most_common(1)[0][1] / len(answers)
        print(f"baselines        constant-{by_class.most_common(1)[0][0]} "
              f"{majority:.1%}   (a collapsed model scores this)")
        if top_n / len(answers) > 0.8:
            print(f"COLLAPSE         {top_n}/{len(answers)} rows predicted "
                  f"{top!r} -- the adapter learned one class, not the preference")
    return exact / total if total else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/pref_heldout.jsonl")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dump", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import jax
    import numpy as np
    from needle.model.architecture import SimpleAttentionNetwork
    from needle.model.checkpoints import read_adapter
    from needle.model.finetune import DEFAULT_BASE, merge_lora, read_examples
    from needle.model.quantize import cq_ste_params, configure_deploy, WEIGHT_BITS
    from needle.model.run import load_checkpoint
    from needle.model.tokenizer import get_tokenizer

    rows = list(read_examples(Path(args.data)))
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("no rows in " + args.data)
        return 1
    tool_name = (rows[0].get("tools") or [{}])[0].get("name")
    print(f"data {args.data}  {len(rows)} rows  tool {tool_name!r}")

    base_path = args.checkpoint
    adapter = read_adapter(args.adapter) if args.adapter else None
    if not base_path and adapter and adapter.get("base"):
        base_path = adapter["base"]
    base_path = base_path or DEFAULT_BASE
    print(f"base {base_path}")

    params, config = load_checkpoint(base_path)
    config.dtype = "float32"
    params = jax.device_put(
        jax.tree.map(lambda a: np.asarray(a).astype(np.float32), params))
    tokenizer = get_tokenizer(config.vocab_size)
    configure_deploy(act_bits=getattr(config, "act_bits", 8),
                     kv_bits=getattr(config, "kv_bits", 8))
    model = SimpleAttentionNetwork(config)

    results = {}
    merged = cq_ste_params(params, WEIGHT_BITS)
    results["base"] = report("BASE", score(model, merged, tokenizer, rows, tool_name, args.dump))

    if adapter:
        import jax.numpy as jnp

        lora = {tuple(k.split("/")): {"A": jnp.asarray(v["A"]), "B": jnp.asarray(v["B"])}
                for k, v in adapter["lora"].items()}
        merged = cq_ste_params(merge_lora(params, lora, adapter["scale"]), WEIGHT_BITS)
        print(f"\nmerged {len(lora)} weight groups  scale {adapter['scale']}")
        results["tuned"] = report("TUNED", score(model, merged, tokenizer, rows,
                                                 tool_name, args.dump))
        delta = results["tuned"] - results["base"]
        print(f"\ndelta {delta:+.1%}  base {results['base']:.1%} -> tuned {results['tuned']:.1%}")
        if delta <= 0:
            print("no gain over base on untouched rows -- the adapter is not carrying the preference")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
