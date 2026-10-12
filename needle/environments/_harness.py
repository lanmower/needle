import json

import needle

_agents = {}


def agent_for(module):
    key = module.__name__
    if key not in _agents:
        _agents[key] = needle.Needle(tools=module.TOOLS, system=module.SYSTEM)
    return _agents[key]


def _fold(value):
    if isinstance(value, str):
        return value.casefold()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {k: _fold(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_fold(v) for v in value]
    return value


def _key(call):
    return json.dumps(_fold(call), sort_keys=True)


def run_tests(module, min_confidence=0.0, verbose=True, count_withheld=False):
    """Run an environment's frozen acceptance suite against the shipped engine.
    The default scores raw model output; passing e.g. min_confidence=0.4 applies
    the production contract (act on a call only at or above the threshold,
    otherwise treat it as a refusal).

    `count_withheld` also reads `suppressed_calls`. `needle build --lora` drops
    the confidence head, so a locally tuned archive reports confidence None and
    the engine withholds every call it makes (a call below 0.1 is withheld), which
    scores a tuned archive at zero however much it learned. needle's own
    `extract()` reads both, so this matches the project's convention."""
    agent = agent_for(module)
    failures, critical_failures = [], []
    for case in module.TEST_CASES:
        agent.reset()
        response = agent.complete(case["query"])
        got = response.get("function_calls") or []
        if (count_withheld and not got
                and response.get("confidence") is None):
            got = response.get("suppressed_calls") or []
        validation = response.get("validation") or {}
        if got and (validation.get("ungrounded") or validation.get("negation")):
            got = []
        # `needle build --lora` drops the confidence head, so a locally tuned
        # archive reports confidence None; `.get(key, 0.0)` does not substitute
        # for a key that exists with value None, and None < 0.0 raises.
        if got and (response.get("confidence") or 0.0) < min_confidence:
            got = []
        want = case["calls"]
        ok = sorted(_key(c) for c in got) == sorted(_key(c) for c in want)
        if not ok:
            failures.append(case)
            if case.get("critical"):
                critical_failures.append(case)
            if verbose:
                print(f"FAIL [{case['category']}] {case['query']}")
                print(f"  want {json.dumps(want)}")
                print(f"  got  {json.dumps(got)}")
    passed = len(module.TEST_CASES) - len(failures)
    print(f"{passed}/{len(module.TEST_CASES)} passed, {len(critical_failures)} critical failures "
          f"(confidence gate {min_confidence})")
    return passed >= round(0.9 * len(module.TEST_CASES)) and not critical_failures
