#!/usr/bin/env python3
"""Score a built `.cact` archive -- the artifact that ships, not the model.

`scripts/eval_pref.py` (JAX params) and `scripts/eval_torch.py` (torch port) both
score the model. Neither scores the archive `needle build` writes, and `build` is
lossy in two ways that can kill a preference silently: it quantises to CQ W4 / A8,
and it drops the confidence head so `complete()` reports `confidence: None`.

Usage:
  python scripts/validate_archive.py build/pref_gpu.cact
  python scripts/validate_archive.py build/pref_gpu.cact --baseline --suite
  python scripts/validate_archive.py build/pref_gpu.cact \
      --baseline build/base.cact --suite --limit 20 --dump 5
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
import time
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

NO_CALL = "<no call>"
COLLAPSE_FRACTION = 0.80

SURFACE_BASELINE = {
    "data_capture": (29, 1),
    "kitchen_appliance": (31, 0),
    "media_player": (28, 1),
    "productivity": (31, 1),
    "smart_home": (28, 1),
    "wearable": (29, 2),
}
SURFACE_FLOORS = {"media_player": 0.875, "smart_home": 0.875}
SUITE_YARDSTICK = (176, 192, 6)


def load_rows(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def tools_of(row):
    tools = row.get("tools") or []
    return json.loads(tools) if isinstance(tools, str) else tools


def gold_of(row):
    answers = row.get("answers") or []
    if not answers:
        return None
    return (answers[0].get("arguments") or {}).get("resolution")


def enum_of(rows):
    values = []
    for row in rows:
        for entry in tools_of(row):
            properties = (entry.get("parameters") or {}).get("properties") or {}
            for value in properties.get("resolution", {}).get("enum") or ():
                if value not in values:
                    values.append(value)
    return values


TRIGGER = None


def engine_key(row):
    tools = tools_of(row)
    if TRIGGER:
        for entry in tools:
            entry["triggers"] = [TRIGGER]
    return json.dumps([tools, row.get("system") or ""], sort_keys=True)


def open_engines(rows, archive):
    import needle

    engines = {}
    for row in rows:
        key = engine_key(row)
        if key not in engines:
            tools, system = json.loads(key)
            engines[key] = needle.Needle(weights=archive, tools=tools, system=system,
                                         stateless=True)
    return engines


def predicted(response, tool_name, enum, gate):
    calls = response.get("function_calls") or []
    withheld = response.get("suppressed_calls") or []
    if not calls and withheld:
        calls = withheld
    usable = [c for c in calls if isinstance(c, dict)
              and str(c.get("name")) == tool_name]
    if not usable:
        return None, "no_call" if not calls else "wrong_tool"
    call = usable[0]
    resolution = (call.get("arguments") or {}).get("resolution")
    if not isinstance(resolution, str):
        return None, "bad_args"
    if (response.get("confidence") or 0.0) < gate:
        return None, "below_gate"
    if enum and resolution not in enum:
        return resolution, "off_enum"
    return resolution, "ok" if len(usable) == 1 else "multi_call"


def score_archive(archive, rows, tool_name, enum, gate, limit, dump, max_new_tokens):
    engines = open_engines(rows, archive)
    scored = rows[:limit] if limit else rows
    print(f"\nscoring {archive}  {len(engines)} engine(s)  {len(scored)}/{len(rows)} rows  "
          f"tool {tool_name!r}  gate {gate}", flush=True)
    started = time.time()
    outcomes = []
    for index, row in enumerate(scored):
        response = engines[engine_key(row)].complete(row["query"],
                                                     max_new_tokens=max_new_tokens)
        got, kind = predicted(response, tool_name, enum, gate)
        gold = gold_of(row)
        outcomes.append({"index": index, "gold": gold, "got": got, "kind": kind,
                         "confidence": response.get("confidence"),
                         "executed": bool(response.get("function_calls")),
                         "withheld": bool(response.get("suppressed_calls")),
                         "error": response.get("error_code"),
                         "exact": got == gold})
        if index < dump:
            print(f"    row {index}  gold={gold!r}  got={got!r}  ({kind})  "
                  f"error={response.get('error_code')!r}")
            print(f"      {str(response.get('reasoning'))[:160]!r}")
        if (index + 1) % 10 == 0 or index + 1 == len(scored):
            hit = sum(1 for o in outcomes if o["exact"])
            print(f"  ... {index + 1}/{len(scored)}  exact {hit}  "
                  f"{time.time() - started:.0f}s", flush=True)
    for engine in engines.values():
        engine.close()
    return outcomes, time.time() - started


def report(label, outcomes, enum, gate, seconds):
    total = len(outcomes)
    answered = [o for o in outcomes if o["gold"] is not None]
    refusals = [o for o in outcomes if o["gold"] is None]
    exact = sum(1 for o in outcomes if o["exact"])

    print(f"\n===== {label} =====  ({seconds:.0f}s, confidence gate {gate})")
    print(f"  rows            {total}  ({len(answered)} decision, {len(refusals)} refusal)")
    print(f"  exact calls     {exact}/{total}  {exact / total:.1%}")
    if answered:
        hit = sum(1 for o in answered if o["exact"])
        print(f"  decision rows   {hit}/{len(answered)}  {hit / len(answered):.1%}")

    executed = sum(1 for o in outcomes if o.get("executed"))
    withheld = sum(1 for o in outcomes if o.get("withheld") and not o.get("executed"))
    if executed or withheld:
        print(f"  call delivery   {executed}/{total} reached function_calls, "
              f"{withheld}/{total} withheld into suppressed_calls -- this run "
              f"scores intent, so withheld calls count")

    if refusals:
        false_live = sum(1 for o in refusals
                         if o["got"] is not None and o.get("executed"))
        false_intent = sum(1 for o in refusals if o["got"] is not None)
        print(f"  false calls     {false_live}/{len(refusals)} "
              f"({false_live / len(refusals):.1%}) would have fired in deployment; "
              f"{false_intent}/{len(refusals)} ({false_intent / len(refusals):.1%}) "
              f"counting withheld calls as intent")

    if answered:
        print("  per-class recall (of rows whose gold is that class)")
        counts = Counter(o["gold"] for o in answered)
        order = [c for c in enum if c in counts] + [c for c in counts if c not in enum]
        for klass in sorted(order, key=lambda c: -counts[c]):
            hit = sum(1 for o in answered if o["gold"] == klass and o["got"] == klass)
            print(f"    {klass:<18} {hit:>4}/{counts[klass]:<4} {hit / counts[klass]:6.1%}")

    histogram = Counter(o["got"] if o["got"] is not None else NO_CALL for o in outcomes)
    print("  predicted histogram")
    for klass, n in histogram.most_common():
        print(f"    {str(klass):<18} {n:>4}  {n / total:6.1%}")

    collapsed = [k for k, n in histogram.items() if n / total > COLLAPSE_FRACTION]
    for klass in collapsed:
        print(f"  COLLAPSE        {histogram[klass]}/{total} rows predicted {klass!r} "
              f"({histogram[klass] / total:.1%}) -- one class learned, not the preference")

    if answered:
        counts = Counter(o["gold"] for o in answered)
        majority, majority_n = counts.most_common(1)[0]
        print(f"  baseline        constant-{majority} scores {majority_n}/{total} "
              f"{majority_n / total:.1%} on all rows, {majority_n}/{len(answered)} "
              f"{majority_n / len(answered):.1%} on decision rows")

    unusable = {k: v for k, v in Counter(o["kind"] for o in outcomes).items() if k != "ok"}
    if unusable:
        print(f"  unusable        {unusable}")
    errors = {k: v for k, v in Counter(o["error"] for o in outcomes).items() if k}
    if errors:
        print(f"  engine errors   {errors}")
    uncalibrated = sum(1 for o in outcomes if o["confidence"] is None)
    print(f"  confidence      {'absent' if uncalibrated else 'present'} "
          f"({uncalibrated}/{total} rows report None; read as "
          f"(x.get('confidence') or 0.0))")
    return exact / total if total else 0.0


def load_baseline_module(gate, archive):
    path = os.path.join(ROOT, "data", "baseline.py")
    saved = sys.argv
    sys.argv = [saved[0], str(gate), archive]
    try:
        spec = importlib.util.spec_from_file_location("needle_pref_baseline", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.argv = saved
    return module


def run_suite(archive, gate):
    baseline = load_baseline_module(gate, archive)
    print(f"\n===== shipped environment suite: {archive} (gate {gate}) =====")
    passed_total = cases_total = critical_total = 0
    regressed = []
    for name in baseline.NAMES:
        environment = importlib.import_module(f"needle.environments.{name}")
        print(f"  [suite] {name} ...", flush=True)
        passed, cases, critical, per_category, seconds = baseline.run_one(environment)
        passed_total += passed
        cases_total += cases
        critical_total += critical
        floor_passed, floor_critical = SURFACE_BASELINE.get(name, (None, None))
        verdict = ""
        if floor_passed is not None:
            if passed < floor_passed or critical > floor_critical:
                verdict = (f"  REGRESSED (base {floor_passed}/{cases}, "
                           f"{floor_critical} critical)")
                regressed.append(name)
            else:
                verdict = f"  holds (base {floor_passed}/{cases}, {floor_critical} critical)"
        rate = passed / cases if cases else 0.0
        floor = SURFACE_FLOORS.get(name)
        if floor is not None and rate < floor:
            verdict += f"  BELOW {floor:.1%} FLOOR"
            if name not in regressed:
                regressed.append(name)
        print(f"  {name:<17} {passed}/{cases} = {rate:.1%}  critical_failures={critical}  "
              f"{seconds:.0f}s{verdict}", flush=True)
        print("      " + "  ".join(f"{k} {v[0]}/{v[1]}"
                                   for k, v in sorted(per_category.items())), flush=True)

    yardstick_passed, yardstick_cases, yardstick_critical = SUITE_YARDSTICK
    print(f"  TOTAL           {passed_total}/{cases_total} = "
          f"{passed_total / cases_total:.1%}  critical_failures={critical_total}")
    if passed_total < yardstick_passed or critical_total > yardstick_critical:
        print(f"  REGRESSED       below the {yardstick_passed}/{yardstick_cases} = "
              f"{yardstick_passed / yardstick_cases:.1%}, {yardstick_critical} critical yardstick")
    else:
        print(f"  holds           at or above {yardstick_passed}/{yardstick_cases} = "
              f"{yardstick_passed / yardstick_cases:.1%}, {yardstick_critical} critical")
    if regressed:
        print(f"  regressed surfaces: {', '.join(regressed)}")
    return {"passed": passed_total, "cases": cases_total, "critical": critical_total,
            "regressed": regressed}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("archive")
    parser.add_argument("--data", default=os.path.join("data", "pref_heldout.jsonl"))
    parser.add_argument("--baseline", nargs="?", const="auto", default=None,
                        help="also score the untuned base .cact; bare --baseline uses "
                             "the cached base archive")
    parser.add_argument("--suite", action="store_true",
                        help="also run the 192-case shipped environment suite")
    parser.add_argument("--gate", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dump", type=int, default=0)
    parser.add_argument("--trigger", default=None,
                        help="regex added as `triggers` to every tool: a triggered "
                             "tool ships its call even below the confidence floor, "
                             "which is the only way a locally tuned archive (no "
                             "confidence head) delivers one")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    global TRIGGER
    TRIGGER = args.trigger

    baseline_archive = args.baseline
    if baseline_archive == "auto":
        from needle.agent import fetch

        candidate = os.path.join(fetch.cache_dir(3), fetch.base_weights(3))
        if not os.path.exists(candidate):
            print(f"no cached base archive at {candidate}; pass --baseline <path>")
            return 1
        baseline_archive = candidate

    rows = load_rows(args.data)
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        print(f"no rows in {args.data}")
        return 1
    tool_name = tools_of(rows[0])[0]["name"]
    enum = enum_of(rows)
    print(f"data {args.data}  {len(rows)} rows  tool {tool_name!r}  classes {enum}")

    results = {}

    outcomes, seconds = score_archive(args.archive, rows, tool_name, enum, args.gate,
                                      args.limit, args.dump, args.max_new_tokens)
    tuned_rate = report(f"TUNED {os.path.basename(args.archive)}", outcomes, enum,
                        args.gate, seconds)
    results["tuned"] = {"archive": args.archive, "exact_rate": tuned_rate,
                        "outcomes": outcomes}

    base_rate = None
    if baseline_archive:
        base_outcomes, base_seconds = score_archive(baseline_archive, rows, tool_name,
                                                    enum, args.gate, args.limit,
                                                    args.dump, args.max_new_tokens)
        base_rate = report(f"BASE {os.path.basename(baseline_archive)}", base_outcomes,
                           enum, args.gate, base_seconds)
        results["base"] = {"archive": baseline_archive, "exact_rate": base_rate,
                           "outcomes": base_outcomes}
        delta = tuned_rate - base_rate
        print(f"\ndelta {delta:+.1%}   base {base_rate:.1%} -> tuned {tuned_rate:.1%}")
        if delta <= 0:
            print("no gain over the untuned base on untouched rows -- the archive is not "
                  "carrying the preference")

    if args.suite:
        results["suite_tuned"] = run_suite(args.archive, args.gate)
        if baseline_archive:
            results["suite_base"] = run_suite(baseline_archive, args.gate)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=1)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
