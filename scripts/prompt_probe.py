"""Can the untuned base be pushed up the shipped suite by prompt alone?

Fine-tuning is the last lever, so this measures what prompt engineering buys
before another adapter is trained. The base scores 176/192 with 6 critical
failures; the losses are concentrated in `irrelevant` (12/18), `invalid` (9/12)
and `parallel` (10/12), so the guidance blocks target exactly those three.
"""

import argparse
import os
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import needle  # noqa: E402
from needle import environments  # noqa: E402
from needle.environments import _harness  # noqa: E402

GUIDANCE = {
    "none": "",
    "refuse": ("If the request is not something these tools can do, make no "
               "call at all and reply in plain text."),
    "range": ("If the request names a value outside a tool's allowed range or "
              "enum, make no call at all."),
    "multi": ("If the request asks for more than one action, emit one separate "
              "call for each action."),
    "all": ("If the request is not something these tools can do, make no call "
            "at all and reply in plain text. If the request names a value "
            "outside a tool's allowed range or enum, make no call at all. If "
            "the request asks for more than one action, emit one separate call "
            "for each action."),
}


def score(module, system, weights):
    if weights:
        agent = needle.Needle(tools=module.TOOLS, system=system, weights=weights)
    else:
        agent = _harness.agent_for(module)
    passed = 0
    critical = 0
    per = Counter()
    totals = Counter()
    for case in module.TEST_CASES:
        agent.reset()
        response = agent.complete(case["query"])
        got = response.get("function_calls") or []
        if not got:
            got = response.get("suppressed_calls") or []
        validation = response.get("validation") or {}
        if got and (validation.get("ungrounded") or validation.get("negation")):
            got = []
        want = case["calls"]
        ok = (sorted(_harness._key(c) for c in got)
              == sorted(_harness._key(c) for c in want))
        totals[case["category"]] += 1
        if ok:
            passed += 1
            per[case["category"]] += 1
        elif case.get("critical"):
            critical += 1
    return passed, critical, per, totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--variants", default="none,refuse,range,multi,all")
    parser.add_argument("--surfaces", default="")
    args = parser.parse_args()

    names = [n for n in environments.ENVIRONMENTS
             if not args.surfaces or n in args.surfaces.split(",")]
    variants = args.variants.split(",")

    grand = {v: [0, 0, Counter(), Counter()] for v in variants}
    for name in names:
        module = environments.ENVIRONMENTS[name]
        for variant in variants:
            system = (module.SYSTEM or "").rstrip()
            extra = GUIDANCE[variant]
            if extra:
                system = system + "\n\n" + extra
            started = time.time()
            passed, critical, per, totals = score(module, system, args.weights)
            total = len(module.TEST_CASES)
            print(f"{name:<18} {variant:<7} {passed}/{total} = {passed/total:6.1%}  "
                  f"critical={critical}  {time.time()-started:.0f}s", flush=True)
            bucket = grand[variant]
            bucket[0] += passed
            bucket[1] += critical
            bucket[2].update(per)
            bucket[3].update(totals)

    print()
    for variant in variants:
        passed, critical, per, totals = grand[variant]
        total = sum(totals.values())
        print(f"TOTAL {variant:<7} {passed}/{total} = {passed/total:6.1%}  "
              f"critical={critical}")
        for category in sorted(totals):
            print(f"  {category:<12} {per[category]}/{totals[category]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
