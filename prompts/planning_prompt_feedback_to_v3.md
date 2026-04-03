# Round 2 Planning Prompt Feedback (50-query experiment)

## Updated Aggregate Results (GPT-OSS-120B, first 50 queries)

| Condition | Correct | Completed | vs Baseline |
|-----------|---------|-----------|-------------|
| No plan (baseline) | 27/50 (54%) | 49/50 | — |
| Self-plan v1 | 26/50 (52%) | 43/50 | -1 correct |
| Gemini plan v1 (start_ext) | 26/50 (52%) | 40/50 | -1 correct |
| **vanilla (v7)** | **21/50 (42%)** | **41/50** | **-6 correct** |
| **v8_prompt** | **24/50 (48%)** | **40/50** | **-3 correct** |
| reinject_v1 every 5 | 25/50 (50%) | 48/50 | -2 correct |
| **reinject_v6.1 every 5** | **28/50 (56%)** | **46/50** | **+1 correct** |

**The only condition that beats baseline is `reinject_v6.1_5`** (+1 correct, -3 completions).

---

## New Finding 1: Vanilla (v7) Fails Due to Empty Plans

**Root cause**: The v7 prompt is too minimal — the model outputs plan text without `<plan>` tags 10/50 times (20%). The parser returns an empty string. The executor receives "Here is the plan: " with nothing after it.

**Evidence**: All 10 empty-plan trajectories share the same structure — `result[2]` injection message is exactly 86 chars (just the wrapper, no content). The executor notices the missing plan in its reasoning: *"The user says 'Here is the planner's response' but they didn't show plan."*

**Why v8 fixes this**: 0/50 empty plans for v8. The longer, more structured prompt with an explicit example teaches the model the required output format reliably.

**Key lesson**: A planning prompt needs enough scaffolding to reliably produce parseable output. A 300-word "strategy brief" instruction is too sparse for the planner to know what it should generate.

---

## New Finding 2: v8 Plans Are 2.5x the Target Length

v8 target was ~600 chars. Actual average: **1521 chars**. The "under 15 lines" instruction is ignored.

Why: v8 uses free-form prose per step, which expands naturally. v6.1's structured fields (`Type:`, `Suggested query:`, `Depends on:`) actually constrain per-step verbosity because they force the model into a specific format. Paradoxically, **more structure = shorter output** because the model fills in defined slots rather than elaborating in prose.

v1 plans (1480 chars avg) and v8 plans (1521 chars avg) are nearly identical in length despite very different format instructions. The model's default plan length is ~1500 chars and it takes a hard token cap (or a very short example) to change this.

---

## New Finding 3: The "Pre-Committed Dependent Step" Bug

Plans fill in EXPECTED values for dependent constraints rather than waiting for actual upstream results.

**Example (qid=796)**: Question asks about a music artist from the same country as a footballer who was runner-up in the early-2000s World Cup. The plan correctly says "step 1: determine the footballer's nationality" — but then step 2 says "search for **Brazilian** music artists." The plan pre-committed to Brazil as the expected result of step 1, even though the actual answer is Germany (Germany was runner-up in 2002, not Brazil). The agent found Klose (German) in step 1 but then followed the plan's hardcoded "Brazilian" in step 2.

**The pattern**: The planner, having recognized the chain, uses its prior knowledge to fill in what step 1 "will find" — and sometimes gets it wrong. This creates a contradiction: the plan says "determine X first" but then uses an assumed value of X in downstream steps.

**Fix required**: Explicitly instruct the planner to leave genuine placeholders for dependent values, not expected values. "Search for music artists from **[the country identified in step 1]**" not "search for Brazilian music artists."

---

## New Finding 4: Re-Injection Fundamentally Changes the Trajectory Dynamics

| Condition | Avg tool calls | Avg context | Completions |
|-----------|---------------|-------------|-------------|
| No plan | 19.8 tc | 246K chars | 49/50 |
| v8 (static) | 21.3 tc | 259K chars | 40/50 |
| reinject_v1 every 5 | 26.8 tc | 327K chars | 48/50 |
| reinject_v6.1 every 5 | 25.8 tc | 333K chars | 46/50 |

**Counter-intuitive finding**: Re-injected conditions use 35% MORE context than static plans, yet have BETTER completion rates (48/50, 46/50 vs 40/50). Static plans consume context at the start without benefit; re-injected plans keep agents anchored throughout, preventing aimless searches that burn context unproductively.

**Why reinject_v6.1 beats baseline**: v6.1 plans have an "Answer sketch" field (hedged hypothesis). When correct (7/11 times), the re-injection keeps the hypothesis in the agent's working memory, enabling systematic verification. When wrong (4/11 times), re-injection reinforces the bad path. Net: +3.

**Implication**: The design should shift from "minimize plan length" to "design for re-injection." A plan intended to be re-read every 5 steps should be:
- Short enough to not bloat context per re-injection (~600 chars)
- Self-contained: each re-injection should make sense regardless of how far the agent has progressed
- No step-number references that become meaningless mid-trajectory

---

## New Finding 5: "Combined Constraint" Bug Persists

Despite the "one fact per query" rule in v8, plans still merge constraints:

**qid=809**: Plan suggests "December 2011 study rocks minerals history ancient life census petition" — multiple unrelated constraints combined into one query, diluting semantic signal. The no-plan agent searched for dinosaur-specific terms directly and worked through the chain faster.

---

## Priority Changes for v9

| Priority | Change | Evidence |
|----------|--------|----------|
| P0 | **Fix pre-committed dependent step**: instruct planner to use `[result from step N]` placeholders, never fill in expected values | qid=796 failure |
| P0 | **Enforce length via example**: the example plan must be ≤8 lines; add "Do not elaborate. Each step is one sentence." | avg 1521 chars despite 15-line cap |
| P0 | **Separate queries per constraint**: add anti-example of combined constraint query | qid=809, "combined constraint" bug |
| P1 | **Add back hedged hypothesis**: rename "Answer sketch" → "Best guess" with explicit hedging ("this may be wrong; verify before concluding") and 2-3 alternatives. Re-injection of correct hypothesis drives wins. | reinject_v6.1 +7 wins |
| P1 | **Design for re-injection**: write a 1-sentence "reminder" at the top of the plan that summarizes the key strategy. This is what gets most value from re-injection. | reinject dynamics |
| P2 | **Block `e.g.` entity loophole**: plans write "filmmaker most associated with dinosaur movies (e.g., Spielberg)" — the example sneak entity names past the no-entity rule | qid=786 |
