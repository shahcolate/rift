# Design: Agentic / Tool-Use Drift Detection

*Status: draft for review. Called for by STRATEGY.md P2 ("Design doc
first — pairing and scoring are genuinely harder here"). Nothing here
is implemented; file references describe the codebase as of 2026-07.*

## 1. Motivation

The frontier moved to agents. The unit of model behavior teams ship
is no longer "one prompt, one completion" — it is a loop: read the
task, call tools, read results, call more tools, answer. A model
upgrade that leaves single-shot accuracy flat can still break an
agent, and today Rift cannot see it.

The questions labs and engineering teams actually ask about a version
bump, roughly in the order they ask them:

1. **Task completion.** Does the new model still finish the
   multi-turn task? The agentic analogue of accuracy, and the only
   question with an obvious binary outcome.
2. **Tool-selection stability.** Given the same situation, does it
   still reach for the same tool? A shifted habit can mean slower and
   4× the tokens even when the task still succeeds — or new failures
   on tasks the old habit solved.
3. **Argument fidelity.** Right types, right fields, IDs copied
   rather than hallucinated — a known failure mode of model swaps,
   invisible to answer-only scoring when a retry loop papers over it.
4. **Trajectory economics.** Turns and tokens per completed task —
   cost-per-correct again, but the denominator can move 10× between
   versions.

Everything Rift's identity rests on — paired tests, CIs, BH pooling,
selftest null calibration, fingerprint provenance — applies unchanged
*if* we can get each case down to a paired observation. That "if" is
the entire design problem.

## 2. The pairing problem, and the core design decision

`compare_runs` (src/rift/comparator.py) takes two equal-length score
vectors, paired by case index: McNemar on binary scores, paired
t-test + bootstrap otherwise. Every downstream artifact — the gate's
exit code, the selftest false-positive rate, the Observatory's
`score_drift` events — assumes one scalar per case per model.

Agentic runs break the naive extensions of this:

- **Per-turn pairing is undefined.** Baseline finishes in 4 turns,
  challenger in 7; turn 3 of one is not "the same decision" as turn 3
  of the other. Sequence-alignment tricks (edit distance over
  tool-call sequences) produce a number, but not one with a
  paired-test interpretation, and they reward imitating the baseline
  rather than solving the task.
- **Trajectory content diverges immediately.** After the first
  differing tool call the two models are in different states and
  every subsequent token is conditioned on different context; no
  intermediate quantity is measured on comparable ground.

**Core design decision: the unit of pairing stays the case. Each
case's whole trajectory is collapsed to one scalar — task success by
default, optionally a weighted assertion rubric — and the existing
paired machinery applies unchanged.** McNemar on binary task success,
paired t-test + bootstrap on rubric scores, `test_used` recorded as
today. No new statistics.

This is the same move Rift already makes twice. The faithfulness
probe (src/rift/faithfulness.py) runs a multi-step protocol per case
and reduces it to a per-case scalar, paired on the intersection of
eligible cases; replication (`--trials k`) reduces k completions to
one per-case mean. The statistics layer never sees the protocol, only
the scalar — that boundary has kept the comparator small and
auditable, and we keep it.

What we give up: turn-level drift claims ("it regressed at step 3 of
refund flows") are out of reach as *gated* claims. They survive as
diagnostics (section 6) — where claims without a valid paired test
belong.

## 3. Provider surface

Today `Completion` (src/rift/providers/__init__.py) has no tool-call
representation and `BaseProvider.complete(prompt: str)` takes a bare
string — no messages, no tools. Two additions:

```python
@dataclass
class ToolCall:
    id: str          # provider id; synthesized "call_<turn>_<i>" for Gemini
    name: str
    arguments: dict | None   # parsed; None when the provider sent invalid JSON
    arguments_raw: str = ""  # verbatim, kept when parsing fails (scoreable)

@dataclass
class Completion:
    ...existing fields...
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None   # normalized: "tool_use" | "end" | other
```

Both default empty, so `Completion.from_cache`'s schema-drift
tolerance keeps old cache blobs loadable (it must additionally
reconstruct `ToolCall` objects from dicts — one small extension).

Providers gain a second method rather than an overload of `complete`:

```python
async def complete_turn(
    self, messages: list[dict], tools: list[dict] | None, **kwargs
) -> Completion: ...
```

`complete` stays untouched — every existing single-shot path, cache
entry, and provider keeps working. A provider without `complete_turn`
(riftlm, bare `local`) fails an agentic suite at preflight with a
clean ClickException, not per-case.

We accept one JSON-schema dialect in suites (the OpenAI/Gemini
`parameters` shape, which Anthropic's `input_schema` also takes) and
translate per provider — suites stay provider-agnostic (design
principle 4).

Per-provider mapping (the normalization `complete_turn` owes the
runner):

| Concept | Anthropic Messages | OpenAI Chat Completions | Gemini generateContent |
|---|---|---|---|
| Tools in request | `tools: [{name, description, input_schema}]` | `tools: [{type: "function", function: {name, description, parameters}}]` | `tools: [{functionDeclarations: [{name, description, parameters}]}]` |
| Model requests a call | `content[]` block `type: "tool_use"` (`id`, `name`, `input` dict) | `message.tool_calls[]` (`id`, `function.name`, `function.arguments` — a JSON **string**) | `parts[].functionCall` (`name`, `args` dict) — **no id** |
| Returning a result | user msg, `tool_result` block w/ `tool_use_id` | `role: "tool"` msg w/ `tool_call_id` | `parts[].functionResponse` (`name`, `response`) |
| Stop signal | `stop_reason: "tool_use"` | `finish_reason: "tool_calls"` | presence of `functionCall` parts |

Normalization landmines, handled in the provider layer so the runner
never sees them:

- OpenAI arguments arrive as a JSON string, occasionally invalid.
  That is model behavior, not a transport error: parse failures set
  `arguments=None`, keep `arguments_raw`, and flow to scoring
  (section 6) rather than raising.
- Gemini has no call id; we synthesize stable ids from (turn, index)
  and match results by name + order.
- All three providers can emit multiple (parallel) calls in one turn;
  the runner executes them sequentially in emitted order (section 5).
- Fable 5 / newer Anthropic models interleave thinking blocks with
  tool_use blocks; thinking text goes to `output_text`, not
  `tool_calls`.

## 4. Suite schema

```yaml
name: order_support_agent
description: Multi-turn customer-support tasks over a mocked order system
scoring: agentic
max_turns: 10                       # hard cap; suite-level, per-case override allowed
tools:                               # JSON-schema tool definitions, sent verbatim
  - name: lookup_order
    description: Fetch an order record by id.
    parameters: {type: object, properties: {order_id: {type: string}},
                 required: [order_id]}
  - name: issue_refund
    ...
environment: "./orders_env.py:make_env"   # module:fn — custom_scorer idiom
cases:
  - input: |
      Customer message: "Order 4521 arrived broken, I want my money back."
    environment_state:               # per-case fixture handed to make_env
      orders:
        "4521": {status: delivered, total: 1240.00, recipient: Acme Corp}
    expected:                        # task-outcome assertions, ALL must hold
      assertions:
        - {kind: tool_called, name: issue_refund,
           args_subset: {order_id: "4521"}}
        - {kind: env_state, path: orders.4521.refunded, equals: true}
        - {kind: final_answer_matches, regex: "(?i)refund"}
      reference_policy:              # optional; powers diagnostics only
        - lookup_order
        - issue_refund
    tags: [flow:refund]
```

Design notes:

- **`environment` reuses the `custom_scorer` loading idiom exactly**:
  a `module:fn` or `./file.py:fn` target, resolved against the suite
  file's directory via `SuiteConfig._source_dir` (which `_with_cases`
  in context_rot.py must keep preserving — already a documented
  invariant). The loader in src/rift/scoring/custom.py is hoisted to
  a shared module rather than duplicated. **Same trust boundary, same
  sentence**: loading an environment executes the target module; only
  run suites you trust. We add no isolation and claim none (§7).
- The environment protocol:

  ```python
  def make_env(state: dict) -> Environment
  class Environment(Protocol):
      def call(self, name: str, arguments: dict) -> dict | str: ...
      def snapshot(self) -> dict: ...   # final state, for env_state assertions
  ```

  One fresh instance per (case, model). `call` may mutate internal
  state (a refund marks the order refunded) but must be a
  deterministic function of `(environment_state, call history)` — the
  paired-determinism requirement, section 5.
- **`expected` is task-outcome assertions**, not an expected output
  string. v1 kinds: `tool_called` (name + optional `args_subset`,
  matched anywhere in the trajectory), `tool_not_called`, `env_state`
  (dotted path into the final snapshot), `final_answer_contains` /
  `final_answer_matches`. All-pass ⇒ task_success 1.0, else 0.0.
- `max_turns` is both a semantic bound (an agent that can't finish in
  N turns has failed) and the cost lever (section 9). Hitting it ends
  the loop; assertions are still evaluated (they almost always fail,
  which is the right verdict).
- Load-time validation mirrors the existing pattern: `scoring:
  agentic` requires `tools` + `environment`; `environment` without it
  is an error; unknown assertion kinds are a SuiteValidationError,
  not a runtime surprise.

## 5. Runner changes

`run_suite` grows an agentic path per case (the single-shot path is
untouched):

```
messages = [user(case.input)]
env = make_env(case.environment_state)
for turn in 1..max_turns:
    completion = complete_turn(messages, suite.tools, **model_params)
    if not completion.tool_calls: break          # final answer
    for call in completion.tool_calls:           # emitted order, sequential
        result = env.call(call.name, call.arguments or {})
        messages += tool_result(call.id, result)
score = evaluate_assertions(trajectory, env.snapshot(), case.expected)
```

**Paired determinism is the requirement everything above serves.**
The runner's contract (runner.py module docstring, property 1) is that
baseline and challenger see byte-identical prompts. For agents the
honest generalization is *counterfactual* identity: had the challenger
issued the same tool calls as the baseline, it would have seen
byte-identical results. Live tools break this three ways:

1. **Nondeterministic results.** A real search API answers Tuesday's
   query differently on Wednesday; the paired test is then comparing
   answers to different questions.
2. **Side effects.** The baseline's actions mutate shared state the
   challenger then observes — the runs contaminate each other.
3. **Availability noise.** A tool's flakiness becomes score variance
   attributed to the model.

Mocked environments eliminate all three: both models face the same
frozen world, so trajectory differences are attributable to the model
— the entire point of a paired test. The honest caveat: this measures
behavior in a simulated world, and a model could rank differently
against real flaky APIs. That is the same trade every fixed-prompt
eval makes (a suite is not production traffic either), and for drift
*attribution* it is the right side of the trade. State it in reports;
don't apologize for it.

Other runner mechanics:

- **Accounting.** `CaseResult` tokens/cost become sums over turns
  (single-shot: unchanged sums-of-one). New fields `turns: int = 1`
  and `trajectory: list[dict]` (the normalized message log, for
  diagnostics and audit). `--strip-io` must empty `trajectory` too —
  it is the largest IO surface in the file.
- **Fingerprints.** Every turn's `provider_fingerprint` feeds the
  run-level set (the pattern trials use), so a rollout *within a
  case* still trips the mid-run rollout warning.
- **Caching.** Per-*turn*, keyed on
  `(model, model_params, sha256(messages + tools))`. Because the
  environment is deterministic, a fully cached case replays the whole
  trajectory keylessly — preserving the keyless replay guarantee for
  demos and recorded benchmarks. The environment file's content
  digest folds into the cache key (the `resolve_model` checkpoint
  digest idiom), so editing the mock invalidates stale trajectories
  instead of silently replaying a world that no longer exists; the
  digest is also stamped into run metadata as provenance.
- **Errors and retries.** Transport failures retry with the existing
  backoff; a case that exhausts retries mid-trajectory errors with
  the partial trajectory preserved. Model-level protocol violations
  (unknown tool name, unparseable arguments) are NOT errors: the
  environment returns a structured error result and the loop
  continues — a model that recovers from its own malformed call has
  demonstrated something real, and one that doesn't will fail the
  assertions. Crashing the harness on bad model output would hide
  exactly the drift we want to catch.

## 6. Scoring: three tiers, one gate

**(a) `task_success` — binary, gates.** The all-assertions-pass bit.
Drives McNemar, the exit code, `cost_per_correct`, the CI action, and
Observatory `score_drift` events — the only number wired to
consequences. Suites opting into weighted assertions get continuous
scores and the paired-t path automatically (`_is_binary` already does
the selection); the default is binary, and a pre-registered primary
endpoint (preregistration.py) should pin the choice before running.

**(b) Tool-selection accuracy — diagnostic, never gates.** Where a
case supplies `reference_policy` (an ordered list of expected tool
names), we report the fraction of the reference matched *as a
subsequence* of the actual call sequence, per case, aggregated per
model, with the between-model delta. The reason this cannot gate is
also its honest limitation: a reference policy encodes one valid
strategy, and deviating from it while succeeding is not a regression.
This tier answers "did its habits change," not "did it get worse" —
the Observatory `notice` precedent: reported, never gated.

**(c) Argument fidelity — diagnostic, never gates.** For actual calls
whose name matches a `tool_called` assertion, a structured per-field
diff of arguments against `args_subset` — field present, type
correct, value exact — the field-wise treatment the extraction suites
give dict outputs. Reported as per-field mismatch rates and deltas,
plus a protocol-violation count (invalid JSON arguments, unknown tool
names) from the runner. This is where "the new model hallucinates
order IDs" becomes visible even when retries keep task_success flat.

If (b)/(c) deltas are ever shown with p-values, they pool through the
same BH correction the matrix and Observatory use; v1 renders them as
descriptive deltas with bootstrap CIs and no significance flag, to
keep the "only (a) gates" line unambiguous.

## 7. Explicitly out of scope for v1

- **Live tool execution.** Breaks paired determinism (section 5) and
  drags in secrets management and side-effect liability. Includes
  provider-side server tools (hosted web search, code execution) —
  live by definition.
- **Sandboxing guarantees.** The environment module runs in-process,
  like custom scorers. The exec_tests scorer already states the
  posture (src/rift/scoring/exec_tests.py): subprocess isolation is
  "NOT a sandbox — a determined adversarial model could still touch
  the filesystem or network." We extend exactly that stance, no
  stronger: environments are trusted suite code, and model *outputs*
  never execute in v1 (tool calls hit the mock, not an interpreter) —
  strictly less exposure than exec_tests already accepts.
- **Multi-agent** (agents calling agents, handoffs). The pairing
  story is unsolved even at one agent's remove.
- **Computer use.** Different provider surface, a pixel-level-state
  determinism problem, and no credible mock story yet.
- **Human-in-the-loop turns** beyond the initial prompt. (The
  sycophancy probe's scripted pushback covers the drift-relevant
  slice of this in the existing panel.)

## 8. Metrics and report shape

The headline needs no new rendering: task_success drift is a
`DriftResult` from binary vectors, and the reporter, the markdown
report, the GitHub Action gate, and the exit-code contract consume it
unchanged. Subgroup analysis via tags (`--subgroup flow:`) works
as-is.

New: a **trajectory diagnostics block** under the headline, rendered
in the visual style of the existing subgroup tables and explicitly
labeled diagnostic:

```
Trajectory diagnostics (not gated)
  mean turns            4.2 → 6.8   (+2.6)
  tool calls / case     3.9 → 7.1   (+3.2)
  ref-policy match      0.91 → 0.84 (−0.07)   [tier b]
  arg field mismatch    1.2% → 4.8% (+3.6pp)  [tier c]
  protocol violations   0 → 3
```

Observatory integration: an agentic suite in the panel is just a
suite — `index.jsonl` lines, `compare_runs` pairing, BH pooling, and
`score_drift`/`silent_swap` events work untouched. Two adjustments:
(1) `panel_version` currently hashes `(input, expected)` pairs; for
agentic suites it must also cover the tools block and the environment
digest, so editing the mock fires `panel_changed` (pairing restarts)
instead of a bogus week-over-week comparison against a different
world. (2) Trajectory summaries (turns, tool histogram, violation
counts) are computed pre-strip into the record's `derived` block,
like confidence parses and refusal flags today; their shifts surface
as `notice` events — reported, never gated.

## 9. Cost

Every turn resends the whole conversation, so input tokens grow
roughly quadratically in trajectory length: a 10-turn case with
chunky tool results can cost 20–50× a single-shot case, and an
agentic suite of 40 cases against two models is real money, not panel
pocket change.

Interaction with the budget guard (observatory.py):

- `estimate_stage_cost`'s cold-start fallback (chars/4 + 300 tokens
  per case) would underestimate an agentic stage by an order of
  magnitude. Agentic suites get their own fallback, an upper-bound
  shape of `max_turns × (max_turns+1)/2 × per-turn-prefix` — budget
  the cap, not the hope. Prior-observation actuals (the existing
  preferred path) take over automatically after the first pass.
- `max_turns` is the primary cost lever and defaults small (10);
  suite authors raising it are raising their bill knowingly.
- Per-turn caching makes iterating on assertions or the reporter
  free; only genuinely new trajectory prefixes hit the API. The
  environment-digest keying (section 5) is what makes this safe.
- `cost_per_correct` gets more interesting, not less: a challenger
  that succeeds equally often in twice the turns is a real regression
  the $/correct delta CI already knows how to price. Published
  agentic cost claims carry the same serving-configuration disclosure
  the existing benchmark analyses use.

## 10. Open questions

1. **Noise floor before gating.** Trajectories compound sampling
   noise: one different early token can fork the whole run. Is
   single-trial McNemar on task_success too twitchy to gate — should
   agentic gating require `--trials`-style replication, or a `rift
   selftest` pass on the agentic suite, before the exit code binds?
   Selftest is the honest arbiter: run it first, decide with data.
2. **Where do reference policies come from?** Hand-written is
   expensive; recording a "golden run" of the baseline biases tier
   (b) toward the incumbent by construction. Possibly report
   symmetrically (each model's match against the other's trajectory),
   or accept the bias and label it.
3. **Weighted rubrics vs forking paths.** Partial credit switches the
   test to paired-t and invites post-hoc endpoint shopping. Should
   weighted `scoring: agentic` require a `--preregister` spec?
4. **Stochastic-but-paired environments.** Flaky-API simulation wants
   a variable world; a per-case seeded RNG keeps pairing (same seed
   both sides). Worth the protocol surface in v1, or defer?
5. **Parallel tool calls.** We execute sequentially in emitted order;
   is parallelization itself a drift signal worth a diagnostic?
6. **Thinking blocks in trajectories.** Recording aids audit but
   bloats records and the `--strip-io` surface — and thinking content
   is the one part of a trajectory the faithfulness work says not to
   over-trust. Record, summarize, or drop?
7. **Seed suite and demo path.** What ships as
   `suites/agentic_*.yaml`? Candidate: the order-support environment
   from section 4 — small state, deterministic, judge-free
   assertions. And the keyless demo: is there a credible tiny
   tool-use task for RiftLM, or does the agentic demo rely on
   recorded trajectories replayed from cache (which section 5 makes
   possible)?
