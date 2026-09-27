---
name: needle-preference-tune
description: Rolls out a LoRA fine-tune of needle toward whatever preference the invoker states -- generates a training set that embodies it, trains an adapter, builds a .cact, and validates live that the preference actually shows up. Use when asked to fine-tune/train needle to prefer a specific tool-call choice, argument style, refusal behavior, or persona; the preference itself always comes from the invocation, never from this file. Hyperparameter/dataset-sizing guidance is grounded in cactuscompute.com/blog/finetuning-needle; schema/action-space design in cactuscompute.com/blog/designing-tools-for-needle and github.com/browser-use/jev-ultrafast; validation taxonomy in designing-tools-for-needle and needle-confidence. None of it is guessed.
---

# needle-preference-tune

The preference being trained is whatever the invoker describes this time --
this skill never assumes one. Its job is the pipeline: turn a stated
preference into a dataset, train it in, build it, and prove live that it
actually took, on this codebase's real CLI (`needle/cli.py`), not a
hypothetical one. Every number below is sourced from the project's own
`llms.txt` and its fine-tuning blog post, not inferred from first principles.

## 0. Read the preference

Restate it as a concrete, falsifiable rule before writing anything: given
situation X, the model should now choose Y over Z, where Z is what the
untuned base plausibly does today. Classify it, because it changes every
step downstream:

- **Tool/behavior selection** ("prefer calling A over B", "always refuse C")
  -- the cheap case. "A few hundred clean examples measurably improve which
  tool gets picked" (finetuning-needle).
- **Argument grounding/style** (how values get extracted or formatted) --
  the expensive case: needs "on the order of thousands of examples with
  reasoning lines and varied phrasings and values", and benefits from
  `--lora-rank 32` (doubles adapter capacity, adapter stays tiny).

If the invocation is vague, turn the ambiguity into a `prd-add`/stated
assumption and proceed -- don't stall on it.

**Fine-tuning is the last lever, not the first.** Per
`designing-tools-for-needle`: "When a suite passes on the base model and your
product needs more, the next lever is fine-tuning rather than a longer
description." Before generating a single training example, check whether
the stated preference is actually a schema problem -- a narrower tool, a
better per-argument description, an enum renamed to the words users say, or
a regex `trigger` for a phrasing no description enumerates all cost nothing
and ship instantly, where a fine-tune costs a training run and a rebuild
every time the preference is revisited. Reach for training only once those
are genuinely exhausted or the preference can't be expressed as a schema
change at all (e.g. it's about which of two already-correct tools to favor).

## 1. Turn the preference into schemas + a generation brief

A stated preference isn't only about which *action* tool gets called --
`structured-extraction-with-needle` is explicit that extraction and
classification are the same mechanism (a record is a tool with one call;
"classification is extraction with an enum"), and fine-tunes through the
identical JSONL shape, record as the tool and the passage as the query. If
the preference is really "classify/extract this field a certain way", write
it as a record schema with a `Literal`/enum field, not as a special case --
it's the same selection-over-generation principle again, a third independent
source agreeing with jev-ultrafast and designing-tools-for-needle.

Write the tool/record schemas (JSON) the preference is decided over. Match
the real deployment surface's tool count: above 5 declared tools, `Needle`'s
retrieval head renders only the top-5 per turn and constrains the grammar to
that subset, so a schema set larger than the target surface trains against a
grammar shape the deployed agent won't actually see.
`needle/model/finetune.py`'s `_GEN_TEMPLATE` already generates varied
`{query, reasoning, answers}` triples from schemas via OpenRouter -- reuse it
through `needle generate-data`, don't reimplement a generator.

The dataset must be a supervised demonstration set, not a labeled preference
pair set: needle has no DPO/RLHF path (`render_example` only ever supervises
the given `answers`), so bias comes from making the *answers you supply*
consistently embody the preference across many phrasings of the same
decision point -- including the ambiguous cases where the untuned base would
plausibly pick the other option. Cover the decision boundary, not just the
easy cases.

**Guard against catastrophic forgetting -- the local path has no built-in
mitigation for it.** Per the blog's own comparison table, the hosted
Platform path trains "your data reinforced with Needle's original dataset,
so nothing already learned is unlearned"; the local `needle finetune` path
keeps "your data only". Generate (or reserve) a second, smaller batch of
*unbiased* examples over the same schemas -- ordinary `generate-data` output
with no preference constraint -- and mix it into the training file alongside
the preference-slanted rows. Without this, a small preference-only dataset
is exactly the failure mode the blog's troubleshooting section names: "the
tuned model calls a tool on everything" / drifts off-brief on cases the
preference wasn't about. There's no published mixing ratio for this --
default to something like 70/30 preference-to-neutral and adjust based on
what step 7's regression check shows.

**Design principle, grounded in `browser-use/jev-ultrafast`: keep the
preference a selection, not a generation, and only offer what's currently
valid.** That project's whole speed/cost win (25% faster, ~10x fewer browser
round trips on its benchmark task) comes from one structural choice: almost
every decision is a closed-set pick from an indexed table of *currently
valid* operations/targets for that exact state, and its small LLM only
generates free text for the one operation (`TYPE_TEXT`) that truly needs it
-- everything else is selection, not generation. It also bundles the
operation and target decision into one model request instead of two
sequential calls, and its executor re-validates a selected target against
live DOM state (freshness, occlusion) before trusting it, rather than
trusting the model's output as ground truth.

The same structure already exists in needle (grammar-constrained
`{name, arguments}`, closed enum/`Literal` argument choices, the deterministic
repair step, `validation.ungrounded`) -- the lesson to actually apply when
authoring the preference's schemas:

- Wherever the stated preference can be expressed as choosing among a small,
  state-valid set of options (enum/`Literal`) instead of free-generating a
  value, do that. It's the same fork as step 0's classification: a selection
  preference is the "few hundred examples" case; forcing it into an
  open-ended argument makes it the "thousands of examples" case for no
  reason.
- If the preference is state-dependent (the right choice today depends on
  what's actually available/valid right now), put that state in the query or
  `system` facts so the training example is grounded in what was actually
  offered, not a context-free global rule the model can't apply consistently.
- Don't design a preference that needs a confirmatory second turn when the
  first call already has enough grounding to decide -- one request per
  decision, same as needle's existing single JSON-call turn.

This isn't jev-ultrafast's idea alone -- `designing-tools-for-needle` says
the identical thing independently, about needle specifically: "A narrow tool
with a plain description beats a broad one. The model is best at picking a
name; it is worst at inventing free-text values that stand in for a
decision." Two unrelated sources converging on the same structural rule is
why this section is load-bearing, not a stylistic preference. It adds two
more concrete authoring rules worth applying to the schemas:

- **Name enum options the way users would say them.** `action: ["increase",
  "decrease"]` matches "turn up", "louder", "raise" and their opposites
  because the engine knows those families; `["inc", "dec"]` does not. Never
  let an enum value collide with a common word from the opposite intent (a
  room named `office` poisons every request containing "off"). Ship a polar
  pair (`lock_door`/`unlock_door`) as two tools or one enum -- never leave
  the model to infer polarity from a description.
- **A description is a fact about the tool, not an instruction to the
  model** -- instructions placed there do not steer decoding. If a phrasing
  needs to reach a tool no description can enumerate, that's what a regex
  `trigger` is for (it forces a call past the confidence floor and the guess
  gates, though not the contradiction gates -- a negated or reported request
  still withholds). A catch-all trigger needs a negative lookahead excluding
  other tools' nouns and multi-action phrasing ("and"/"then"), or it
  misroutes. This is a cheaper fix than training data for a routing miss.

## 2. Generate the set

```
needle generate-data --tools schemas.json --num-samples 200 --output data/pref.jsonl
```

Needs `OPENROUTER_API_KEY` in the environment (see `OPENROUTER_URL`/
`DEFAULT_MODEL` in `finetune.py`). Size `--num-samples` off step 0's
classification: a few hundred for a selection preference, thousands for an
argument-grounding one. If the key is absent, or the preference is too
narrow/specific for synthetic diversity to hit reliably, hand-author
examples directly in the same JSONL shape instead of skipping generation:

```json
{"query": "...", "reasoning": "...", "answers": [{"name": "...", "arguments": {...}}]}
```

(`answers: []` for the off-topic/refusal rows `_GEN_TEMPLATE` also asks for
-- keep some; per troubleshooting below, skipping these is the #1 cause of a
model that calls a tool on everything.) `needle finetune --generate N` can
also expand a hand-authored seed file in the same run instead of a separate
`generate-data` pass.

## 3. Sanity-check before spending a training run on it

Read a real sample of `data/pref.jsonl` (not a summary of it) and confirm
every `answers` row actually encodes the target preference -- a generation
pass drifting off-brief is the single most common way this whole exercise
produces nothing. Check the refusal/off-topic rows and the neutral/rehearsal
rows from step 1 are both still present.

## 4. Train

One-time: `pip install -e ".[train]"` (pulls jax/flax/optax -- this is the
`[project.optional-dependencies] train` extra in `pyproject.toml`).

Documented defaults (don't restate these as flags unless deviating):
batch 16, learning rate 1e-4 with warmup and cosine decay, gradient clipping
at norm 1, LoRA rank 16 / alpha 32, max length 1024, val split 0.1, seed 0.

**Get the epoch count right -- this is the single most common way a run
looks fine and does nothing.** Steps per epoch = `ceil(N / batch_size)`. A
200-example set at the default batch 16 is 13 steps/epoch; the CLI's own
`--epochs 3` default is then only 39 steps total, which "barely moves a
rank-16 adapter at the default learning rate." For a few hundred examples,
plan on **10-30 epochs**, not the flag's default:

```
needle finetune data/pref.jsonl --epochs 20 --lora-rank 16 --lora-alpha 32 \
  --val-split 0.1 --out checkpoints/pref.safetensors
```

Omit `--checkpoint` to auto-download the Needle 3 base. Training always runs
at the full 20 layers regardless of what depth you'll export in step 5.

**Reading the loss:** it covers only the reasoning line and the JSON call,
not the whole sequence -- most of the call is boilerplate the base already
predicts, so a run starts near 1.0, not near random. Judge the *trend*, not
the level. A validation loss prints at each epoch end (from the `--val-split`
holdout): if it rises while training loss keeps falling, the run is
overfitting -- stop there, or add data, rather than training further.

## 5. Build

```
needle build --lora checkpoints/pref.safetensors --out build/pref.cact
```

Add `--layers N` for a smaller rung, or `--platform <target>` (one of
`needle.agent.fetch.PLATFORMS`) to also fetch that platform's engine binary
next to the archive for step 6. `build` always quantises to 4 bits locally
and merges the adapter into a fresh base download -- the archive it writes
is independent of anything cached from a previous build.

## 6. Validate live -- no test files, ever, but do reuse existing fixtures

Per this project's standing rule (see the `gm` skill): a test file is never
evidence and never gets written. Evidence here is running the actual
engine binary against real prompts and reading real output, same as the
`build-release` validation already done this session (`needle.exe --model
... --tools ... --prompt ...` / `needle run --checkpoint ...`).

Two tiers, cheapest first:

1. **Reuse, don't author.** `needle.environments` (`smart_home`,
   `media_player`, `productivity`, `wearable`, `kitchen_appliance`,
   `data_capture`) already ships `TOOLS`/`TEST_CASES`/`run_tests()` per
   surface -- a zero-authoring-cost regression baseline. Run it against the
   *base* model first to get today's pass rate (per `llms.txt` the shipped
   base already misses 5/6 suites at the default gate), then again with
   `weights="build/pref.cact"` after training. A preference fine-tune must
   not drop this baseline on surfaces the preference isn't targeting -- that
   delta is the regression signal, for free, and is exactly what step 1's
   rehearsal-mix rows are meant to protect.
2. **Author the boundary cases, against a real taxonomy, not a vibe.**
   `designing-tools-for-needle` names the exact six categories its own
   32-case suites cover: the exact call for a positive request; `[]` when a
   required value is missing, when no tool covers the request, when the
   request is negated, and when a stated value is out of bounds; two calls
   from one request, order-insensitive. `needle-confidence` adds the specific
   guess/grounding-gate edge cases worth throwing in too: a quoted command
   reported by someone else ("she said turn it off"), a required enum filled
   with an option the request never names, a required slot filled from a
   control word instead of request text, a required number with no default
   left unstated, an origin/destination swap, and a target the request
   explicitly excludes. Build the battery from these categories at the
   preference's specific decision boundary -- not just the easy cases -- and
   run it through the fine-tuned `.cact` and the untuned base side by side.
   Confirm two things, not one: the preference now shows up consistently,
   AND those existing suites still behave sanely.
3. **Ground the call against real state, not just parse it.** Per the
   `jev-ultrafast` lesson in step 1: a JSON call that parses and looks
   plausible is not the same as one that's actually valid against real
   current state. Where the tools represent real actions with real
   preconditions, execute the emitted call (or check it against the actual
   state it claims to act on) rather than stopping at "the JSON matches the
   schema" -- that's exactly where a preference can silently regress into
   a fluent-looking hallucination.

**Known, non-negotiable limitation:** local `needle finetune`/`needle build
--lora` never trains or carries the confidence-calibration head -- a
preference-tuned `.cact` built this way always reports `confidence: None`,
by design (the head is untouched). If the stated preference is itself about
confidence/refusal thresholds, say so up front: that specific preference
cannot be validated or shipped through this local path at all -- it needs
the hosted Platform (step 8).

## 7. Troubleshoot by documented signature, then iterate

The blog names four failure signatures directly -- match against these
before inventing a diagnosis:

- **Loss hovers at its starting value:** undertrained, not broken. Raise
  epochs first, then learning rate.
- **The tuned model calls a tool on everything:** the dataset has no `[]`
  (refusal/off-topic) examples. Add them.
- **Correct tool, wrong argument values:** more examples with reasoning
  lines and more varied values, then `--lora-rank 32`.
- **`failed to load weights`:** the `.cact` format is tied to the engine
  version; an archive built by an older package won't load. Rebuild with
  the current package.

Beyond these, treat a still-missing preference as a dataset problem: the
specific failing prompt shape is the next addition, sharper and at exactly
that decision point, not a blind epoch/lr increase.

## 8. The higher-quality path exists, but it spends money -- ask first

`needle platform finetune`/`needle platform generate` (the hosted Cactus
Platform, `NEEDLE_API_KEY`) trains every depth 2-20 layers on GPU, reinforces
your data with Needle's original dataset (so nothing already learned is
unlearned -- the rehearsal step 1 approximates by hand), scores each depth
against a held-out test file, keeps the calibrated confidence head, and
exports at the shipped model's real 2-bit precision instead of the local
path's 4-bit. Strictly higher-yield than the local path whenever it's
available. But per `llms.txt` and the blog: both `finetune` and `generate`
"spend allowance at submission." That is a world-scoped, money-spending
action (Section 4 of the `gm` skill) -- never submit one of these jobs
without the user explicitly asking for the hosted path first. Default to
the free local LoRA path above unless they do.

## Artifacts stay local

`data/`, `checkpoints/`, `*.cact` are already gitignored. Don't commit a
training run's outputs unless the user explicitly asks to ship a build.
