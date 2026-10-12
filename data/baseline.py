"""Per-surface baseline pass rate for the UNTUNED base, using the project fixtures.

One engine per environment (`_harness.agent_for`) with `agent.reset()` before
every case, so no prior turn can leak into the next -- the same discipline the
project's own run_tests() uses. Prints per-surface and per-category rates.

Usage: python data/baseline.py [min_confidence] [weights.cact]

The confidence read is `(... or 0.0)`, not `response.get("confidence", 0.0)`:
`needle build` drops the confidence head for locally tuned archives, so
`confidence` is present-but-None and the shipped `_harness.run_tests` raises
`TypeError: '<' not supported between instances of 'NoneType' and 'float'` the
moment the tuned model emits any call.
"""

import json
import sys
import time

from needle.environments import _harness

NAMES = ("data_capture", "kitchen_appliance", "media_player",
         "productivity", "smart_home", "wearable")

MIN_CONFIDENCE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
WEIGHTS = sys.argv[2] if len(sys.argv) > 2 else None

# The fallback only applies when `confidence` is None, i.e. the archive has no
# confidence head. A model that still has one withholds on a real score below
# the floor, and honouring that is what makes `irrelevant` and `missing` pass --
# counting those withheld calls understated the untuned base by 13 cases.
COUNT_WITHHELD = True


def fold(value):
    if isinstance(value, str):
        return value.casefold()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {k: fold(v) for k, v in value.items()}
    if isinstance(value, list):
        return [fold(v) for v in value]
    return value


def key(call):
    return json.dumps(fold(call), sort_keys=True)


def run_one(module):
    agent = _harness.agent_for(module)
    if WEIGHTS:
        import needle
        agent = needle.Needle(tools=module.TOOLS, system=getattr(module, "SYSTEM", None),
                              weights=WEIGHTS)
    passed, critical_fails, per_category = 0, 0, {}
    started = time.time()
    for case in module.TEST_CASES:
        agent.reset()
        response = agent.complete(case["query"])
        got = response.get("function_calls") or []
        if (COUNT_WITHHELD and not got
                and response.get("confidence") is None):
            got = response.get("suppressed_calls") or []
        validation = response.get("validation") or {}
        if got and (validation.get("ungrounded") or validation.get("negation")):
            got = []
        if got and (response.get("confidence") or 0.0) < MIN_CONFIDENCE:
            got = []
        want = case["calls"]
        ok = sorted(key(c) for c in got) == sorted(key(c) for c in want)
        bucket = per_category.setdefault(case["category"], [0, 0])
        bucket[1] += 1
        if ok:
            passed += 1
            bucket[0] += 1
        elif case.get("critical"):
            critical_fails += 1
    return passed, len(module.TEST_CASES), critical_fails, per_category, time.time() - started


def main():
    total_passed = total_cases = total_critical = 0
    for name in NAMES:
        module = getattr(__import__("needle.environments", fromlist=[name]), name)
        passed, n, crit, per_cat, secs = run_one(module)
        total_passed += passed
        total_cases += n
        total_critical += crit
        print(f"{name}: {passed}/{n} = {passed / n:.1%}  critical_failures={crit}  "
              f"{secs:.1f}s", flush=True)
        print("    " + "  ".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(per_cat.items())), flush=True)
    print(f"TOTAL: {total_passed}/{total_cases} = {total_passed / total_cases:.1%}  "
          f"critical_failures={total_critical}  (gate {MIN_CONFIDENCE})")


if __name__ == "__main__":
    main()
