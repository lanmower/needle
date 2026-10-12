#!/usr/bin/env python3
"""Report the token length of a fine-tune file and the seq_len it will bucket to.

`fit_max_len` rounds the longest example up to a power of two and silently
truncates anything past the cap, so a file whose p99 is 240 tokens still trains
at 1024 if one row is long -- paying 4x the compute -- or loses its tool call
if the cap is forced down. Run this before any training run: it prints both the
bucket you will get and how many rows each candidate cap would truncate.
"""

from __future__ import annotations

import argparse
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    import numpy as np
    from needle.model.finetune import fit_max_len, read_examples, render_example
    from needle.model.run import load_checkpoint
    from needle.model.tokenizer import get_tokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data")
    parser.add_argument("--checkpoint", default="checkpoints/needle3.safetensors")
    parser.add_argument("--cap", type=int, default=1024)
    args = parser.parse_args()

    _, config = load_checkpoint(args.checkpoint)
    tokenizer = get_tokenizer(config.vocab_size)

    lengths = []
    for example in read_examples(args.data):
        prompt, target = render_example(example)
        lengths.append(len(tokenizer.encode(prompt)) + len(tokenizer.encode(target)) + 2)
    if not lengths:
        print("no rows")
        return 1
    values = np.array(lengths)
    print(f"{args.data}  n {len(values)}")
    for label, value in (("max", values.max()), ("p99", np.percentile(values, 99)),
                         ("p90", np.percentile(values, 90)), ("p50", np.percentile(values, 50))):
        print(f"  {label}  {int(value)}")
    print(f"  fitted bucket at cap {args.cap}: {fit_max_len(args.data, tokenizer, args.cap)}")
    for cap in (256, 512, 1024):
        over = int((values > cap).sum())
        print(f"  --max-len {cap:<5} truncates {over} row(s)" if over
              else f"  --max-len {cap:<5} truncates 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
