"""Does the untuned base pick the operator's option when asked for maximum progress?

This replaces the five-class `resolution` taxonomy, which the operator called a
weak description of what they want. The task is selection, not classification:
real question text and real option labels, one instruction, verbatim answer.
"""

import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import needle  # noqa: E402

MAX_PROGRESS = "Pick the one option that makes maximum progress."

NUMBERED = re.compile(r"^\s*(\d+)\)\s+(.+)$")


def option_count(query):
    return len([l for l in (query or "").splitlines()
                if NUMBERED.match(l)])


def normalize(text):
    return re.sub(r"\s+", " ", (text or "").strip().casefold())


def strip_parenthetical(text):
    return re.sub(r"\s*\([^)]*\)", " ", text or "")


def relaxed(gold, chosen):
    """Same option, allowing the model to drop a parenthetical suffix.

    "Merge clean branches (Recommended)" versus "Merge clean branches" is the
    same choice; a strict comparison counts it as a miss and understates the
    model.
    """
    left = normalize(strip_parenthetical(gold))
    right = normalize(strip_parenthetical(chosen))
    if not left or not right:
        return False
    return left == right or left in right or right in left


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/rows_grounded.jsonl")
    parser.add_argument("--schema", default="data/schema_grounded.json")
    parser.add_argument("--system", default=MAX_PROGRESS)
    parser.add_argument("--archive", default=None,
                        help="tuned .cact to score; default is the untuned base")
    parser.add_argument("--trigger", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--show", type=int, default=0)
    args = parser.parse_args()

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    tools = json.loads(open(args.schema, encoding="utf-8", newline="").read())
    if args.trigger:
        for entry in tools:
            entry["triggers"] = [args.trigger]

    agent = needle.Needle(tools=tools, system=args.system, stateless=True,
                          weights=args.archive)
    started = time.time()
    exact = 0
    near = 0
    calls = 0
    multi = 0
    misses = []
    chance = 0.0
    for index, row in enumerate(rows):
        agent.reset()
        response = agent.complete(row["query"])
        got = response.get("function_calls") or []
        if not got and response.get("confidence") is None:
            got = response.get("suppressed_calls") or []
        if got:
            calls += 1
        if len(got) > 1:
            multi += 1
        want = row.get("answers") or []
        gold = (want[0].get("arguments") or {}).get("choice") if want else None
        chosen = None
        for call in got:
            if str(call.get("name")) == "decision_choice":
                chosen = (call.get("arguments") or {}).get("choice")
                break
        options = option_count(row["query"]) or 1
        chance += 1.0 / options
        if chosen is not None and gold is not None and normalize(chosen) == normalize(gold):
            exact += 1
        elif chosen is not None and gold is not None and relaxed(gold, chosen):
            near += 1
        elif len(misses) < args.show:
            misses.append((gold, chosen))
    total = len(rows)
    print(f"rows {total}   {time.time() - started:.0f}s")
    print(f"archive       {args.archive or '(untuned base)'}")
    print(f"system        {args.system!r}")
    print(f"exact         {exact}/{total} = {exact / total:.1%}")
    print(f"near miss     +{near} -> {(exact + near) / total:.1%} "
          f"(parenthetical dropped or contained)")
    print(f"chance        1/options averaged = {chance / total:.1%}")
    print(f"call rate     {calls}/{total} = {calls / total:.1%}")
    print(f"multi-call    {multi}/{total}")
    for gold, chosen in misses:
        print(f"  miss  gold={gold!r}  got={chosen!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
