#!/usr/bin/env python3
"""Merge the authored row files into a needle fine-tune set and validate it.

Three sources are combined:

  labeled  -- real historical decisions, the preference itself
  neutral  -- same schema, label forced by the wording, blunts catastrophic
              forgetting (the local path trains on your data only)
  refusal  -- answers: [] for inputs that are not decision points; without
              these the tuned model calls a tool on everything

It also carves out a held-out file. The CLI's own printed accuracy is NOT held
out (`_score_quantised` re-permutes the whole input file, so it scores mostly
rows the model just trained on), so a separate untouched file is the only
honest check.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

MAX_REASONING_WORDS = 20
_WORD = re.compile(r"\S+")
_QUOTE = re.compile(r'["“]([^"”]{6,})["”]')
_TOKEN = re.compile(r"[a-z0-9']+")


def orphan_quotes(row: dict) -> list[str]:
    """Spans the reasoning quotes that do not appear anywhere in the query.

    `reasoning` is supervised by the loss, so a line that quotes text the query
    never contained -- typically the operator's own answer -- trains the model
    to produce the answer from nothing instead of deriving it.
    """
    if not row.get("answers"):
        return []
    query = (row.get("query") or "").lower()
    return [
        span
        for span in (m.group(1).strip() for m in _QUOTE.finditer(row.get("reasoning") or ""))
        if span.lower() not in query
    ]


def group_rows(rows: list[dict], threshold: float = 0.5) -> list[list[int]]:
    """Bucket row indices so near-duplicate queries share a bucket.

    The authored rows are augmentations of a much smaller set of real
    decisions -- several wrappers around one decision. Splitting those at
    random puts one wrapper in train and another in the held-out file, so the
    held-out score partly measures recall of a decision the model already saw
    rather than generalisation. Rows carry no provenance field, so the grouping
    is recovered from the query text: boilerplate that most rows share is
    dropped first (it is wrapper vocabulary, not content), then rows are
    unioned when their remaining content words mostly overlap.
    """
    tokens = [{w for w in _TOKEN.findall((r.get("query") or "").lower()) if len(w) > 3}
              for r in rows]
    total = len(rows)
    if total < 2:
        return [[i] for i in range(total)]
    common = {w for w, n in Counter(w for t in tokens for w in t).items()
              if n > total * 0.3}
    tokens = [t - common for t in tokens]

    parent = list(range(total))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(total):
        left = tokens[i]
        if not left:
            continue
        for j in range(i + 1, total):
            right = tokens[j]
            overlap = len(left & right)
            if overlap and overlap / len(left | right) >= threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

    buckets: dict[int, list[int]] = {}
    for k in range(total):
        buckets.setdefault(find(k), []).append(k)
    return list(buckets.values())


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                print(f"  {path.name}:{number} unparseable: {exc}", file=sys.stderr)
    return rows


def enum_values(schema_path: Path) -> set[str]:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    values = set()
    for tool in schema:
        for prop in (tool.get("parameters") or {}).get("properties", {}).values():
            if "enum" in prop:
                values.update(prop["enum"])
    return values


def problems(row: dict, allowed: set[str], tool_name: str) -> list[str]:
    issues = []
    query = row.get("query")
    if not isinstance(query, str) or not query.strip():
        issues.append("missing/empty query (KeyError at render time)")
    if not isinstance(row.get("tools"), list) or not row["tools"]:
        issues.append("missing tools -- would train against an empty tool list")
    answers = row.get("answers")
    if not isinstance(answers, list):
        issues.append("missing answers list")
        return issues
    for call in answers:
        if call.get("name") != tool_name:
            issues.append(f"unexpected tool name {call.get('name')!r}")
        resolution = (call.get("arguments") or {}).get("resolution")
        if resolution not in allowed:
            issues.append(f"resolution {resolution!r} not in enum")
    reasoning = row.get("reasoning") or ""
    if len(_WORD.findall(reasoning)) > MAX_REASONING_WORDS:
        issues.append(f"reasoning over {MAX_REASONING_WORDS} words -- starves the call at 96 new tokens")
    return issues


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", default="data/schema.json")
    parser.add_argument("--labeled", default="data/rows_labeled.jsonl")
    parser.add_argument("--neutral", default="data/rows_neutral.jsonl")
    parser.add_argument("--neutral-extra", default="data/rows_neutral_extra.jsonl")
    parser.add_argument("--refusal", default="data/rows_refusal.jsonl")
    parser.add_argument("--out", default="data/pref_train.jsonl")
    parser.add_argument("--heldout", default="data/pref_heldout.jsonl")
    # 15% rather than 10%: the check that matters is per-class recall, and at
    # 10% each of the five classes holds ~13 rows, which cannot distinguish a
    # real gain from noise. The training side loses ~5% of its rows.
    parser.add_argument("--heldout-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-group-split", action="store_true",
                        help="split rows independently instead of by decision bucket")
    parser.add_argument("--group-threshold", type=float, default=0.5,
                        help="Jaccard overlap at which two queries count as one decision")
    args = parser.parse_args()

    schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
    allowed = enum_values(Path(args.schema))
    tool_name = schema[0]["name"]
    print(f"enum: {sorted(allowed)}")
    print(f"tool: {tool_name}")

    sources = {
        "labeled": load(Path(args.labeled)),
        "neutral": load(Path(args.neutral)) + load(Path(args.neutral_extra)),
        "refusal": load(Path(args.refusal)),
    }

    # Row authors should not have to embed the schema at all: an author who
    # wrote rows before the enum grew would otherwise ship a stale grammar,
    # and the model would see two different tool sets across the same file.
    bad_total = 0
    orphan_total = 0
    for name, rows in sources.items():
        restamped = 0
        bad = 0
        orphans = 0
        for index, row in enumerate(rows):
            if row.get("tools") != schema:
                row["tools"] = json.loads(json.dumps(schema))
                restamped += 1
            issues = problems(row, allowed, tool_name)
            if issues:
                bad += 1
                if bad <= 5:
                    print(f"  {name}[{index}] {'; '.join(issues)}")
            stray = orphan_quotes(row)
            if stray:
                orphans += 1
                if orphans <= 3:
                    print(f"  {name}[{index}] reasoning quotes off-query: {stray[0]!r}")
        print(f"{name:8s} {len(rows):5d} rows, {restamped} restamped, "
              f"{bad} invalid, {orphans} off-query reasoning")
        bad_total += bad
        orphan_total += orphans

    if orphan_total:
        print(f"\n{orphan_total} row(s) quote text the query never contained -- the model "
              "cannot derive those, and it learns to emit the answer from nothing. Fix before training.")

    if bad_total:
        print(f"\n{bad_total} invalid row(s) -- refusing to build.")
        return 1

    rng = random.Random(args.seed)
    merged = []
    for name, rows in sources.items():
        for row in rows:
            merged.append((name, row))

    seen = set()
    unique = []
    for name, row in merged:
        key = (row["query"].strip().lower(), json.dumps(row["answers"], sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        unique.append((name, row))
    print(f"\nmerged {len(merged)} -> {len(unique)} after dedup")

    rng.shuffle(unique)
    hold_count = max(1, int(len(unique) * args.heldout_frac))

    if args.no_group_split:
        held, train = unique[:hold_count], unique[hold_count:]
    else:
        buckets = group_rows([row for _, row in unique], args.group_threshold)
        rng.shuffle(buckets)
        largest = max(len(b) for b in buckets)
        print(f"grouped {len(unique)} rows -> {len(buckets)} decision buckets "
              f"(largest {largest})")
        if largest > len(unique) * 0.25:
            print("  WARNING: one bucket holds a quarter of the set -- the queries are "
                  "too uniform to separate, and the held-out split is optimistic")
        held_idx: set[int] = set()
        held_buckets = 0
        for bucket in buckets:
            if len(held_idx) < hold_count:
                held_buckets += 1
                held_idx.update(bucket)
        held = [row for k, row in enumerate(unique) if k in held_idx]
        train = [row for k, row in enumerate(unique) if k not in held_idx]
        print(f"held-out {len(held)} rows across {held_buckets} whole buckets; "
              "no wrapper of a held-out decision stays in train")

    for path, rows in ((Path(args.heldout), held), (Path(args.out), train)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for _, row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows):5d}  {path}")

    def report(rows, label):
        classes = Counter()
        refusals = 0
        for _, row in rows:
            calls = row.get("answers") or []
            if not calls:
                refusals += 1
            for call in calls:
                classes[(call.get("arguments") or {}).get("resolution")] += 1
        print(f"\n{label}: {len(rows)} rows, refusals {refusals}")
        for value, n in classes.most_common():
            print(f"    {n:5d}  {value}")

    report(train, "train")
    report(held, "heldout")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
