"""Pick the next run's hyperparameters by measurement instead of guesswork.

Trains every grid point with scripts/torch_finetune.py, scores every adapter
with scripts/eval_torch.py on the same held-out subset, and ranks by exact-match
accuracy per GPU-minute so a config that is marginally better but much slower
loses.

    python scripts/sweep_gpu.py --dry-run
    python scripts/sweep_gpu.py --stub
    python scripts/sweep_gpu.py --budget-minutes 120

Launch from a plain shell, not through gm: gm clamps subprocess timeouts to
900 s and a sweep overruns that. Results are written after every config, so a
kill still leaves a partial answer behind.
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

TRAIN = os.path.join("scripts", "torch_finetune.py")
EVAL = os.path.join("scripts", "eval_torch.py")
HELD_OUT = os.path.join("data", "pref_heldout.jsonl")

MEASURED_SETUP_SECONDS = 16.0
MEASURED_SECONDS_PER_STEP = 5.5

# Measured 2026-10-12 by scripts/_prof.py: one decode step is 1154 ms at a
# 332-token prompt (full forward 1212 ms, of which engram_kv is 38 ms). A row
# costs roughly this times the tokens it emits, so ~46 s/row, not the 8 s this
# file first assumed. The cause is batch-1 underutilisation, not a hotspot:
# decode costs 3.6 ms/token against ~0.22 ms/token measured during training at
# batch 16, a 16x gap. Batching would recover it but needs left-padding, which
# shifts RoPE positions and would break the fp32 fidelity the gate established.
MEASURED_EVAL_SECONDS_PER_ROW = 46.0

ROW_FORMAT = ("{:<5} {:<8} {:<3} {:<5} {:<4} | {:<7} {:<6} {:<8} {:<7} {:<8} "
              "{:<12} {:<12}")


def grid(args):
    return [{"rank": rank, "lr": lr, "epochs": epochs, "batch": batch,
             "seed": seed}
            for rank, lr, epochs, batch, seed in itertools.product(
                args.ranks, args.lrs, args.epochs, args.batch_sizes, args.seeds)]


def slug(cfg):
    return "r{rank}_lr{lr:g}_e{epochs}_b{batch}_s{seed}".format(**cfg)


def steps_of(cfg, rows):
    return cfg["epochs"] * -(-rows // cfg["batch"])


def estimate_seconds(cfg, rows, eval_rows):
    return (2 * MEASURED_SETUP_SECONDS
            + steps_of(cfg, rows) * MEASURED_SECONDS_PER_STEP
            + eval_rows * MEASURED_EVAL_SECONDS_PER_ROW)


def train_command(cfg, rows, adapter, log, device, python_bin):
    mode = "probe" if rows else "train"
    return [python_bin, TRAIN, mode,
            "--device", device,
            "--rows", str(rows),
            "--epochs", str(cfg["epochs"]),
            "--batch-size", str(cfg["batch"]),
            "--lr", "{:g}".format(cfg["lr"]),
            "--lora-rank", str(cfg["rank"]),
            "--lora-alpha", "{:g}".format(2 * cfg["rank"]),
            "--seed", str(cfg["seed"]),
            "--out", adapter,
            "--log", log]


def eval_command(adapter, device, limit, tokens, out_json, python_bin):
    command = [python_bin, EVAL,
               "--adapter", adapter,
               "--device", device,
               "--limit", str(limit),
               "--out", out_json]
    if tokens:
        command[5:5] = ["--max-new-tokens", str(tokens)]
    return command


def full_run_command(cfg, device, python_bin):
    return " ".join([python_bin, TRAIN, "train",
                     "--device", device,
                     "--batch-size", str(cfg["batch"]),
                     "--epochs", str(cfg["epochs"]),
                     "--lr", "{:g}".format(cfg["lr"]),
                     "--lora-rank", str(cfg["rank"]),
                     "--lora-alpha", "{:g}".format(2 * cfg["rank"]),
                     "--seed", str(cfg["seed"]),
                     "--out", "checkpoints/pref_lora_gpu.safetensors",
                     "--log", "data/ft_gpu.log"])


def run(command, timeout_seconds, stub=None):
    if stub is not None:
        return stub
    started = time.time()
    done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                          timeout=timeout_seconds or None)
    return done.stdout + done.stderr, time.time() - started


def parse_exact(text):
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] == "exact":
            hit, total = fields[1].split("/")
            return int(hit), int(total)
    return None


def parse_final_loss(text):
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 4 and fields[0] == "first":
            return float(fields[3])
    return None


def held_out_golds(limit):
    path = os.path.join(ROOT, HELD_OUT)
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    if limit:
        rows = rows[:limit]
    golds = [((row.get("answers") or [{}])[0].get("arguments") or {}).get(
        "resolution") for row in rows]
    return len(rows), golds


def chance_baseline(golds):
    answered = [gold for gold in golds if gold is not None]
    majority, count = Counter(answered).most_common(1)[0]
    return majority, count / len(golds)


def stub_train_text(cfg, rows):
    steps = steps_of(cfg, rows)
    return "\n".join([
        "  setup     {:.1f}s  5 lora groups  seq_len 512  rows {}".format(
            MEASURED_SETUP_SECONDS, rows),
        "  device    cuda  NVIDIA GeForce RTX 3060 Laptop GPU",
        "  data      {} rows  seq_len 512  batch {}  {} steps  warmup 1  wd 0.0001".format(
            rows, cfg["batch"], steps),
        "  step      1/{}  loss 0.8266  lr 0.00e+00  5.23s/step".format(steps),
        "  first     0.8266 -> 0.7100   total {:.0f}s".format(
            steps * MEASURED_SECONDS_PER_STEP),
    ])


def stub_eval_text(cfg, rows):
    quality = (0.18 + 0.004 * cfg["rank"] + 0.03 * cfg["epochs"]
               - 40.0 * abs(cfg["lr"] - 1.2e-4))
    hit = max(0, min(rows, round(rows * quality)))
    return "\n".join([
        "  exact        {}/{}  {:.1%}".format(hit, rows, hit / rows),
        "  baseline     constant-widen_scope scores 8/{} {:.1%} on the answered rows".format(
            rows, 8 / rows),
    ])


def paths_for(cfg, args):
    stem = args.workdir.rstrip("/\\") + "/" + slug(cfg)
    return stem + ".safetensors", stem + ".log", stem + ".eval.json"


def measure_config(cfg, args, eval_rows, chance, stub=False):
    adapter, log, scored = paths_for(cfg, args)

    train_text, train_seconds = run(
        train_command(cfg, args.rows, adapter, log, args.device, args.python),
        args.max_run_seconds,
        stub=(stub_train_text(cfg, args.rows),
              MEASURED_SETUP_SECONDS
              + steps_of(cfg, args.rows) * MEASURED_SECONDS_PER_STEP)
        if stub else None)

    if not stub and not os.path.exists(adapter):
        return {"config": cfg, "status": "train_failed",
                "train_s": round(train_seconds, 1), "eval_s": 0.0,
                "gpu_min": round(train_seconds / 60.0, 3),
                "adapter": adapter, "log": log,
                "tail": train_text.strip().splitlines()[-3:]}

    eval_text, eval_seconds = run(
        eval_command(adapter, args.device, args.eval_limit, args.eval_tokens,
                     scored, args.python),
        args.max_run_seconds,
        stub=(stub_eval_text(cfg, eval_rows),
              MEASURED_SETUP_SECONDS + eval_rows * MEASURED_EVAL_SECONDS_PER_ROW)
        if stub else None)

    result = {"config": cfg, "status": "ok",
              "train_s": round(train_seconds, 1),
              "eval_s": round(eval_seconds, 1),
              "gpu_min": round((train_seconds + eval_seconds) / 60.0, 3),
              "final_loss": parse_final_loss(train_text),
              "collapse": "COLLAPSE" in eval_text,
              "adapter": adapter, "log": log, "scored": scored}
    counts = parse_exact(eval_text)
    if counts is None:
        result["status"] = "eval_failed"
        result["tail"] = eval_text.strip().splitlines()[-3:]
        return result
    hit, total = counts
    result["exact"] = hit
    result["scored_rows"] = total
    result["acc"] = hit / total
    result["chance"] = chance
    result["lift"] = hit / total - chance
    result["acc_per_gpu_min"] = result["acc"] / result["gpu_min"] if result["gpu_min"] else 0.0
    result["lift_per_gpu_min"] = result["lift"] / result["gpu_min"] if result["gpu_min"] else 0.0
    return result


def print_plan(plans, args, eval_rows):
    print("grid: {} configs   train rows {}   scored rows {}".format(
        len(plans), args.rows or "all 813", eval_rows))
    print("cost model, planning only: setup {:.0f}s x2 and {:.1f}s/step are "
          "MEASURED; {:.1f}s per scored row is ASSUMED".format(
              MEASURED_SETUP_SECONDS, MEASURED_SECONDS_PER_STEP,
              MEASURED_EVAL_SECONDS_PER_ROW))
    total = 0.0
    for cfg, seconds in plans:
        total += seconds
        adapter, log, scored = paths_for(cfg, args)
        print("")
        print("  [{}]  {} steps  est {:.0f}s".format(
            slug(cfg), steps_of(cfg, args.rows or 813), seconds))
        print("    " + " ".join(train_command(
            cfg, args.rows, adapter, log, args.device, args.python)))
        print("    " + " ".join(eval_command(
            adapter, args.device, args.eval_limit, args.eval_tokens, scored,
            args.python)))
    running = 0.0
    fits = 0
    for _, seconds in plans:
        if running + seconds > args.budget_minutes * 60.0:
            break
        running += seconds
        fits += 1
    print("")
    print("estimated total {:.0f}s = {:.1f} min   budget {:.0f} min".format(
        total, total / 60.0, args.budget_minutes))
    print("configs that fit the budget, cheapest first: {} of {} "
          "({:.1f} min)".format(fits, len(plans), running / 60.0))
    print("the pruning is driven by the ASSUMED seconds per scored row; once a "
          "config has been measured it is the measured time that counts")


def print_report(results, args, majority, chance, refusal_rows, spent_minutes):
    ok = [result for result in results if result["status"] == "ok"]
    print("")
    print("scored subset: the first {} held-out rows, identical for every "
          "config".format(args.eval_limit or "all"))
    print("  chance (constant-{:<10}) {:.1%} of the subset".format(
        majority, chance))
    print("  {} of those rows are refusals (gold is None); exact counts come "
          "straight from".format(refusal_rows))
    print("  eval_torch.py's own scorer and are not re-derived here, so a "
          "change to its exact rule needs no change in this file")
    print("  measured GPU time          {:.1f} min".format(spent_minutes))
    if not ok:
        print("")
        print("no config scored:")
        for result in results:
            if result["status"] != "ok":
                print("  {}  {}".format(slug(result["config"]), result["status"]))
                for line in result.get("tail", []):
                    print("    " + line)
        return None
    print("")
    print(ROW_FORMAT.format("rank", "lr", "ep", "batch", "seed", "exact",
                            "acc%", "train_s", "eval_s", "gpu_min",
                            "acc/gpu_min", "lift/gpu_min"))
    for result in sorted(ok, key=lambda item: -item["acc_per_gpu_min"]):
        cfg = result["config"]
        print(ROW_FORMAT.format(
            cfg["rank"], "{:g}".format(cfg["lr"]), cfg["epochs"], cfg["batch"],
            cfg["seed"],
            "{}/{}".format(result["exact"], result["scored_rows"]),
            "{:.1f}".format(100 * result["acc"]),
            "{:.0f}".format(result["train_s"]),
            "{:.0f}".format(result["eval_s"]),
            "{:.2f}".format(result["gpu_min"]),
            "{:.3f}".format(result["acc_per_gpu_min"]),
            "{:.3f}".format(result["lift_per_gpu_min"])))
    skipped = [item for item in results if item["status"] != "ok"]
    if skipped:
        print("")
        print("{} of {} configs did not score:".format(len(skipped), len(results)))
        for item in skipped:
            print("  {}  {}  est {}s".format(slug(item["config"]),
                                             item["status"],
                                             item.get("estimate_s", "-")))
    best = max(ok, key=lambda item: item["acc_per_gpu_min"])
    fastest = min(ok, key=lambda item: item["gpu_min"])
    most_accurate = max(ok, key=lambda item: item["acc"])
    print("")
    print("winner on accuracy per GPU-minute: {}".format(slug(best["config"])))
    print("  acc {:.1%} against chance {:.1%}, lift {:+.1%}, in {:.2f} "
          "GPU-min".format(best["acc"], best["chance"], best["lift"],
                           best["gpu_min"]))
    print("  most accurate overall: {} at {:.1%}, but {:.2f} GPU-min = {:.1f}x "
          "the winner".format(slug(most_accurate["config"]),
                              most_accurate["acc"], most_accurate["gpu_min"],
                              most_accurate["gpu_min"] / best["gpu_min"]))
    print("  cheapest: {} at {:.2f} GPU-min".format(slug(fastest["config"]),
                                                    fastest["gpu_min"]))
    if best["collapse"]:
        print("  WARNING the winner triggered eval_torch's COLLAPSE detector")
    print("")
    print("next full run (813 rows):")
    print("  " + full_run_command(best["config"], args.device, args.python))
    return best


def write_results(path, payload):
    target = os.path.join(ROOT, path)
    directory = os.path.dirname(target)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--lrs", type=float, nargs="+", default=[5e-5, 1e-4, 2e-4])
    ap.add_argument("--epochs", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[16])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--rows", type=int, default=256,
                    help="train rows per config (0 = all 813)")
    ap.add_argument("--eval-limit", type=int, default=32,
                    help="held-out rows scored per config")
    ap.add_argument("--eval-tokens", type=int, default=0,
                    help="0 = let eval_torch.py use its own default")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--workdir", default="data/sweep")
    ap.add_argument("--out", default="data/sweep_gpu.json")
    ap.add_argument("--budget-minutes", type=float, default=210.0)
    ap.add_argument("--max-run-seconds", type=float, default=0.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stub", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.join(ROOT, args.workdir), exist_ok=True)
    held_out_rows, golds = held_out_golds(args.eval_limit)
    majority, chance = chance_baseline(golds)
    refusal_rows = sum(1 for gold in golds if gold is None)

    eval_rows = args.eval_limit or held_out_rows
    plans = [(cfg, estimate_seconds(cfg, args.rows or 813, eval_rows))
             for cfg in grid(args)]
    plans.sort(key=lambda pair: pair[1])

    print("sweep: {} configs, {} train rows, {} scored rows, budget {:.0f} min".format(
        len(plans), args.rows or 813, eval_rows, args.budget_minutes))
    if args.stub:
        print("STUB: no subprocess is launched, canned text stands in for "
              "torch_finetune.py and eval_torch.py")

    if args.dry_run:
        print_plan(plans, args, eval_rows)
        return 0

    def save(winner=None):
        write_results(args.out, {
            "grid": {"ranks": args.ranks, "lrs": args.lrs,
                     "epochs": args.epochs, "batch_sizes": args.batch_sizes,
                     "seeds": args.seeds},
            "rows": args.rows, "eval_limit": args.eval_limit,
            "device": args.device, "budget_minutes": args.budget_minutes,
            "stub": args.stub, "chance": chance, "majority": majority,
            "refusal_rows": refusal_rows,
            "spent_minutes": round(spent / 60.0, 2), "winner": winner,
            "results": results})

    results = []
    spent = 0.0
    for cfg, estimate in plans:
        if spent + estimate > args.budget_minutes * 60.0:
            results.append({"config": cfg, "status": "over_budget",
                            "estimate_s": round(estimate)})
            save()
            continue
        print("")
        print("[{}] {}  est {:.0f}s  spent {:.1f} min".format(
            len(results) + 1, slug(cfg), estimate, spent / 60.0), flush=True)
        result = measure_config(cfg, args, eval_rows, chance, stub=args.stub)
        results.append(result)
        spent += result["train_s"] + result["eval_s"]
        if result["status"] == "ok":
            print("    exact {}/{}  {:.1%}  train {:.0f}s  eval {:.0f}s  "
                  "acc/gpu-min {:.3f}".format(
                      result["exact"], result["scored_rows"], result["acc"],
                      result["train_s"], result["eval_s"],
                      result["acc_per_gpu_min"]), flush=True)
        else:
            print("    {}".format(result["status"]), flush=True)
            for line in result.get("tail", []):
                print("      " + line, flush=True)
        save()

    best = print_report(results, args, majority, chance, refusal_rows,
                        spent / 60.0)
    save(best["config"] if best else None)
    print("")
    print("wrote " + args.out)
    print("reduced-scale evidence only: a winner at {} rows is not a guarantee "
          "at 813 rows x 3 epochs,".format(args.rows or 813))
    print("and it still has to clear 176/192 (91.7%) on the environment suites "
          "before it is adopted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
