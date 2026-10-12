"""Can the untuned base match the operator's decisions by prompt alone?

The shipped 192-case suite is a forgetting guard, not the objective; the
objective is the `decision_choice` set built from real Claude Code questions.
This scores the base archive on that set under several system prompts, which is
the cheaper lever the preference skill says to exhaust before fine-tuning.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import needle  # noqa: E402

POLICY = """You resolve decision points for this operator. When a question asks
which way to go, call decision_choice with exactly one resolution:

- proceed_now: work can continue as it is, nothing is blocking.
- fix_root_cause: something is broken and the cause should be fixed first.
- widen_scope: the change should cover more than was first asked.
- ship_workaround: a workaround is acceptable and speed matters more than the
  real fix.
- stop_and_reassess: stop and re-evaluate before continuing.

Emit exactly one call, with the single argument resolution."""

POLICY_PRIOR = POLICY + """

Bias: prefer proceed_now when nothing is actually blocking. Prefer
fix_root_cause when a root cause is identifiable. Prefer stop_and_reassess when
the request is risky, unclear, or would waste effort. Prefer widen_scope only
when the question names a larger surface. Prefer ship_workaround only when the
question accepts a stopgap."""

def fewshot(path, per_class, include_reasoning=False):
    """One demonstration per class, round-robin, so the prior stays flat.

    Stating a policy in prose measured worse than saying nothing; showing the
    operator's own resolutions instead of describing them is the implicit form.
    Examples come from the training file, and the held-out split is
    group-disjoint, so no demonstration is a wrapper of a held-out question.
    """
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    by_class = {}
    for row in rows:
        answers = row.get("answers") or []
        if not answers:
            continue
        resolution = (answers[0].get("arguments") or {}).get("resolution")
        by_class.setdefault(resolution, []).append(row)
    picked = []
    for index in range(per_class):
        for resolution in sorted(by_class):
            bucket = by_class[resolution]
            if index < len(bucket):
                picked.append(bucket[index])
    lines = []
    for row in picked:
        answers = row.get("answers") or []
        resolution = (answers[0].get("arguments") or {}).get("resolution")
        reasoning = (row.get("reasoning") or "").strip()
        body = f"{reasoning} -> {resolution}" if include_reasoning else resolution
        lines.append(f"Q: {row['query']}\nA: decision_choice({body})")
    return ("This operator resolved these decision points as follows.\n\n"
            + "\n\n".join(lines))


def build_variants(args):
    variants = dict(VARIANTS)
    if args.fewshot:
        for per_class in args.fewshot.split(","):
            key = f"shot{per_class}"
            variants[key] = fewshot(args.train, int(per_class),
                                    include_reasoning=args.shot_reasoning)
    return variants


VARIANTS = {"none": "", "policy": POLICY, "prior": POLICY_PRIOR}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/pref_heldout.jsonl")
    parser.add_argument("--archive", default=None,
                        help="default is the cached untuned base .cact")
    parser.add_argument("--trigger", default="(options|pick|choose|which|should|want)")
    parser.add_argument("--variants", default="none,policy,prior")
    parser.add_argument("--fewshot", default="",
                        help="comma-separated demonstrations per class, added as "
                             "variants shot<N>; examples come from --train")
    parser.add_argument("--train", default="data/pref_train.jsonl")
    parser.add_argument("--shot-reasoning", action="store_true",
                        help="include the reasoning tail in each demonstration")
    parser.add_argument("--no-trigger", action="store_true",
                        help="omit the trigger: the omission arm, where the base "
                             "is free to make no call at all")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if not args.archive:
        from needle.agent import fetch

        args.archive = os.path.join(fetch.cache_dir(3), fetch.base_weights(3))

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    tools = rows[0]["tools"]
    tools = json.loads(tools) if isinstance(tools, str) else tools
    if args.trigger:
        for entry in tools:
            entry["triggers"] = [args.trigger]

    for variant in args.variants.split(","):
        system = build_variants(args)[variant]
        agent = needle.Needle(weights=args.archive, tools=tools, system=system)
        started = time.time()
        hit = 0
        answered = 0
        calls = 0
        preds = Counter()
        for row in rows:
            agent.reset()
            response = agent.complete(row["query"])
            got = response.get("function_calls") or []
            if not got and response.get("confidence") is None:
                got = response.get("suppressed_calls") or []
            usable = [c for c in got if isinstance(c, dict)
                      and str(c.get("name")) == "decision_choice"]
            want = row.get("answers") or []
            gold = (want[0].get("arguments") or {}).get("resolution") if want else None
            chosen = None
            if usable:
                calls += 1
                chosen = (usable[0].get("arguments") or {}).get("resolution")
                preds[chosen] += 1
            if gold is not None:
                answered += 1
                if chosen == gold:
                    hit += 1
        total = len(rows)
        print(f"{variant:<8} {hit}/{answered} decision rows = {hit/max(1,answered):6.1%}  "
              f"calls {calls}/{total}  {time.time()-started:.0f}s")
        print(f"         predicted {dict(preds.most_common())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
