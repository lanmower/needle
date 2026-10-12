#!/usr/bin/env python3
"""Summarise mined AskUserQuestion decisions and surface repeatable preferences.

Prints three things, in increasing order of usefulness:

  1. counts -- how many forms are real vs hook-satisfied
  2. repeated decision points -- the same question asked more than once, with
     every answer this user gave (the only place a preference can be *proven*
     rather than guessed)
  3. every real (question, options, answer) triple, so the preferences can be
     read rather than inferred from an aggregate
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

_PUNCT = re.compile(r"[^a-z0-9 ]+")
_SPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    return _SPACE.sub(" ", _PUNCT.sub(" ", (text or "").lower())).strip()


def load(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> int:
    # Transcripts are UTF-8; the Windows console codepage is not.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/form_answers.jsonl")
    parser.add_argument("--include-synthetic", action="store_true")
    parser.add_argument("--show-all", action="store_true", help="print every real decision")
    parser.add_argument("--custom-only", action="store_true", help="only free-text answers")
    args = parser.parse_args()

    records = load(Path(args.input))
    real = [r for r in records if args.include_synthetic or not r.get("synthetic")]

    print(f"forms: {len(records)}  real: {len(real)}  synthetic: {len(records) - len(real)}")

    # Flatten to per-question decisions.
    decisions = []
    for record in real:
        by_question = {o["question"]: o for o in record["options"]}
        for question, answer in record["answers"].items():
            offered = by_question.get(question, {})
            labels = [l for l in offered.get("labels", []) if l]
            values = answer if isinstance(answer, list) else [answer]
            decisions.append(
                {
                    "ts": record.get("ts"),
                    "project": record["project"],
                    "question": question,
                    "header": offered.get("header"),
                    "multi": offered.get("multiSelect", False),
                    "labels": labels,
                    "values": values,
                    "custom": any(str(v) not in labels for v in values) if labels else True,
                }
            )

    print(f"decisions: {len(decisions)}")
    print(f"free-text (not an offered option): {sum(1 for d in decisions if d['custom'])}")
    print(f"projects: {len({r['project'] for r in real})}")

    # Repeated decision points: the only place a preference is demonstrable.
    grouped: dict[str, list[dict]] = defaultdict(list)
    for decision in decisions:
        grouped[normalize(decision["question"])].append(decision)

    repeats = {k: v for k, v in grouped.items() if len(v) > 1}
    print(f"\n=== repeated decision points: {len(repeats)} ===")
    for key, items in sorted(repeats.items(), key=lambda kv: -len(kv[1]))[:20]:
        print(f"\n[{len(items)}x] {items[0]['question'][:160]}")
        for value, n in Counter(
            ", ".join(str(v) for v in d["values"]) for d in items
        ).most_common():
            print(f"    {n}x  {value[:120]}")
        print(f"    offered: {items[0]['labels']}")

    if args.custom_only:
        decisions = [d for d in decisions if d["custom"]]

    if args.show_all:
        print(f"\n=== every real decision ({len(decisions)}) ===")
        for decision in sorted(decisions, key=lambda d: d["ts"] or ""):
            answer = ", ".join(str(v) for v in decision["values"])
            tag = "free" if decision["custom"] else "opt "
            print(f"\n({tag}) [{decision['project']}] {decision['question'][:220]}")
            print(f"    -> {answer[:300]}")
            if decision["labels"]:
                print(f"    offered: {decision['labels']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
