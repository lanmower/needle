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

VARIANTS = {"none": "", "policy": POLICY, "prior": POLICY_PRIOR}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/pref_heldout.jsonl")
    parser.add_argument("--archive", default=None,
                        help="default is the cached untuned base .cact")
    parser.add_argument("--trigger", default="(options|pick|choose|which|should|want)")
    parser.add_argument("--variants", default="none,policy,prior")
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
        system = VARIANTS[variant]
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
