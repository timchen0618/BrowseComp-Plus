# Planning Prompt Feedback: Evidence-Based Analysis

## Executive Summary

Planning currently **hurts overall accuracy** across both models and both prompt versions (v1 and v6.1). The best planning condition (Gemini v6.1 + GPT-OSS-120B) achieves 43.6% vs 44.5% baseline. The analysis below identifies *why* and proposes concrete changes.

---

## 1. Quantitative Findings

### Accuracy Impact (GPT-OSS-120B)

| Condition | Accuracy | Completed |
|-----------|----------|-----------|
| No plan (baseline) | 44.5% | 774/830 |
| Self-plan, v1 | 41.6% | 646/830 |
| Gemini plan, v1 (start_ext) | 43.0% | 655/830 |
| Self-plan, v6.1 | 42.0% | 632/830 |
| Gemini plan, v6.1 (start_ext) | 43.6% | 642/830 |

### Accuracy Impact (Tongyi)

| Condition | Accuracy | Completed |
|-----------|----------|-----------|
| No plan (baseline) | 46.6% | 447/830 |
| Gemini plan, v1 (start_ext) | 33.7% | 475/830 |
| Gemini plan, v6.1 (start_ext) | 36.0% | 477/830 |
| Self-plan, v1 | 25.5% | 525/830 |

### Per-Query Win/Loss (GPT-OSS-120B, Gemini v1 plans)

- Plan wins (plan correct, no-plan wrong): **55**
- Plan loses (plan wrong, no-plan correct): **52**
- Plan caused incompletion (no-plan completed): **152 queries** (40 of which no-plan got correct)

### Token Budget Impact

- Plans add ~13K characters per trajectory on average
- v6.1 plans are **2x longer** than v1 plans (4179 vs 2048 avg chars; 54 vs 21 avg lines)
- Planned trajectories use ~1 more tool call on average but have ~120 fewer completions
- 10 planned trajectories (GPT) and 123 (Tongyi) had **zero tool calls** — agent never started executing

---

## 2. When Plans Help (55 wins, 7 deeply analyzed)

### Pattern A: Clue Triage — Plans Identify the Most Discriminating Starting Constraint
Plans analyze which clue is most specific/searchable and direct the agent there first. Without this, no-plan agents start with generic constraints ("born in 1920s") that match too many candidates.

**Example (qid=67):** Question contains cryptic cricket clues ("5.3 overs", "wicket keeper"). Plan correctly hypothesized "2002 ICC Champions Trophy" from the "unusual conclusion" clue → executor found answer in 14 calls vs 29 without plan. No-plan agent wasted searches on generic ODI stat queries.

**Example (qid=140):** Numbers "42:09 355 CS 2015" are meaningless without context. Plan added "League of Legends playoffs" → executor searched effectively. No-plan agent searched raw stat strings.

**Example (qid=165):** Plan correctly prioritized identifying the *country* from crime statistics (250K assaults → South Africa) before searching for movies. No-plan agent guessed "Lawrence of Arabia" (wrong country entirely).

**Example (qid=131):** Plan started with the CEO/company inspiration connection (highly searchable), found Fantastyka → Sapkowski → CD Projekt link, then the farewell piece for editor Parowski. No-plan agent gave up entirely.

### Pattern B: Phased Decomposition for Multi-Hop Questions
Plans that separate sequential sub-problems prevent conflation and ensure intermediate answers are resolved before the final question is attempted.

**Example (qid=83):** Phase 1 identifies the biological entity (yerba mate), Phase 2 finds the first documenter (Joseph Dalton Hooker). No-plan agent guessed "millet" and answered "Hou Ji."

**Example (qid=161):** Plan started with "album banned in 1970s" → Thomas Mapfumo → found track list. No-plan agent guessed Chico Buarque and fabricated a song title.

### Pattern C: Verification Checklists Prevent Premature Commitment
Plans with numbered verification steps that must ALL pass prevent anchoring on the first plausible candidate. No-plan agents lock onto wrong candidates (millet, WildTurtle, Claudio Villa) without cross-checking all constraints.

### Pattern D: Plans Prevent Fabrication Under Pressure
No-plan agents fabricate answers when stuck (qid=67: "ODI no #1666"; qid=161: fabricated "Pedra Dura" on album "Caravanas"). Plans transform "I'm stuck, let me guess" into "this candidate failed, try the next one."

### Search Efficiency Comparison (win cases)

| QID | No-Plan Searches | Plan Searches | No-Plan Result | Plan Result |
|-----|-----------------|---------------|----------------|-------------|
| 67  | 29              | ~15           | Wrong (guessed) | Correct (97%) |
| 83  | 19              | 26            | Wrong (millet)  | Correct (93%) |
| 131 | 15              | 20            | Gave up         | Correct (96%) |
| 140 | 27              | ~15           | Wrong           | Correct (85%) |
| 152 | 25              | ~20           | Wrong (62%)     | Correct (92%) |
| 161 | 22              | 24            | Wrong (fabricated) | Correct (87%) |
| 165 | ~20             | ~15           | Wrong           | Correct (92%) |
| 205 | 25              | 13            | Wrong           | Correct       |

### What Makes These Plans Useful
Plans help most on **longer queries** (mean 605 chars for wins vs 523 for losses) with multi-constraint, multi-hop structure where:
1. The search space is large with no obvious starting point
2. Multiple constraints must be verified in sequence
3. The most discriminating clue is not the most obvious one
4. There is risk of premature commitment to a wrong candidate

The four most valuable plan characteristics are:
- **Clue prioritization** (which constraint to search first)
- **Phased decomposition** (solve sub-problems in order)
- **Explicit verification checklists** (all constraints must pass)
- **Candidate branching** (if Candidate 1 fails, try Candidate 2)

---

## 3. When Plans Hurt (52 losses + 152 incompletions analyzed)

### Failure Mode A: Planner Hallucinated Hypothesis (MOST DAMAGING)
The planner generates a confident but wrong hypothesis, and the executor follows it into a dead end.

**Example (qid=78):** Plan hypothesized "Olav V" as the ruler; correct answer was about King Hussein. Agent spent all budget verifying the wrong entity.

**Example (qid=22):** Plan confidently stated "the sport is League of Legends" and started searching for "shortest game LoL World Championship" — completely wrong framing.

**Example (qid=72):** Plan started with "find the market in California that hosted a Thanksgiving Floral Arrangement Class" — a hallucinated detail that sent the agent on a wild goose chase.

### Failure Mode B: Over-Specification Locks Agent Into Wrong Path
The v1 plan format uses a rigid checklist with numbered phases and specific keyword suggestions. When the initial approach fails, the agent often follows the plan's fallback suggestions instead of thinking independently. The plan's structure becomes a cage rather than a guide.

**Quantitative evidence:** In lose cases, planned agents use MORE search calls than no-plan agents (avg +3-5 extra calls), suggesting the agent follows plan steps mechanically even when they're not working.

### Failure Mode C: Context Window Consumption
Plans consume ~2K chars (v1) or ~4K chars (v6.1) of context window at the start. This:
- Causes 119 extra incompletions (GPT) by reducing available context for search results
- Reduces effective tool calls by forcing earlier truncation
- v6.1's 2x length means even MORE context waste for marginal accuracy gain (43.0% → 43.6%)

### Failure Mode D: Format/Integration Issues
- 10 GPT and 123 Tongyi trajectories had zero tool calls — the agent received the plan but never started executing
- The plan injection format ("Here is the planner's response; please follow the plan") may confuse some models
- Tongyi is especially sensitive to this, explaining its dramatic accuracy drop (46.6% → 33.7%)

### Failure Mode E: Plans Suggest Keyword-Style Queries for a Semantic Retrieval System
The v1 prompt says nothing about the retrieval system. Plans from Gemini often suggest keyword-based queries with boolean operators and exact-match patterns (e.g., `"5.3 overs" "wicket" "ODI"`). But the actual retrieval is **dense semantic search** (Qwen3-Embedding-8B). This mismatch means the plan's suggested queries are suboptimal.

### Failure Mode F: Plans Encourage Speculation Over Search
Instructions like "identify candidate kingdoms" trigger massive parametric reasoning tangents instead of searching first. The agent spends tokens reasoning about what kingdoms existed rather than querying the retrieval system. (qid=3, 78)

### Failure Mode G: Confidence Degradation
Even when plan and no-plan agents reach the same answer, plan versions average **7.7% lower confidence**, and plans use **1.7x more searches on average** in lose cases. The plan's structure makes the agent second-guess itself.

---

## 4. Structural Analysis of Plans (Statistical)

### Plan Length vs Success
Point-biserial correlation between plan character count and correctness is **0.010 — essentially zero**. Plan length does NOT predict success. Incomplete trajectories also have statistically identical plan lengths to completed ones (mean 2058 vs 2051 chars).

### Optimal Plan Complexity
The empirical sweet spot is **4 phases and 12-16 action items** (61.5% and 63.2% success respectively). Plans with 6+ phases (42.1%) and very few action items <12 (46.3%) perform worst.

### Strongest Structural Signal
Plans with **explicit exploration AND verification labeling** show a **+6.2pp advantage** — the strongest structural feature. Plans that frame steps as "explore to find candidates" vs "verify this candidate" help the executor distinguish search modes.

### Template Bloat
Plans with `[Name TBD]` or `[To Be Identified]` placeholder templates perform **-2.4pp worse**. These add structure without actionable content.

### Key Insight
The dominant factor in success vs failure is NOT plan structure but rather: (a) whether the initial search strategy produces useful results from the retrieval backend, (b) whether the query is answerable given the index, and (c) inherent question difficulty. Plan structure explains very little of the outcome variance.

---

## 5. Prompt-Level Issues

### v1 Prompt Issues
1. **No query guidance** — doesn't tell planner about semantic retrieval; plans produce keyword-style queries
2. **Encourages keyword strings** — `Keywords: ...` in the checklist format invites boolean/keyword thinking
3. **No constraint on plan length** — some plans are 90 lines, consuming massive context
4. **Rigid checklist format** — `[ ] **Phase 1:**` pattern is a rigid waterfall that doesn't adapt to search results
5. **No hypothesis hedging** — plans state hypotheses as facts, and executors treat them as ground truth
6. **"Do NOT use the search tool" is the only executor awareness** — planner has no model of what the executor can/can't do, what retrieval works well/poorly, or what the token budget is

### v6.1 Prompt Issues
1. **Doubled plan length** (4K vs 2K chars) with marginal accuracy benefit (+0.6% GPT, +2.3% Tongyi)
2. **Excessive structural overhead** — Type annotations, Depends on, Suggested query, On failure for every substep adds tokens without proportional value
3. **Two lengthy examples** in the prompt itself (~3K chars) that are generated every time
4. **Analysis-to-Plan Mapping** (in experiment_v2) is a good idea but adds even more length
5. **Still no hard length constraint** on generated plans

---

## 6. Proposed Feedback for Prompt Improvement

### CRITICAL: Reduce Plan Length (Highest Priority)
The #1 issue is context consumption causing incompletions. Concrete proposals:
- **Hard cap: plans must be under 800 characters / 15 lines** (currently median ~2000 chars / 21 lines)
- Remove all structural boilerplate (Type annotations, Depends on, Budget fields)
- Plans should be **strategic guidance, not step-by-step instructions**
- One-line per major action, not multi-line substeps with metadata

### CRITICAL: Eliminate Hallucinated Hypotheses
The planner's confident-but-wrong guesses are the #1 quality issue. Proposals:
- **Never state a hypothesis as fact.** Use language like "likely candidates include..." or "start by checking if..."
- **Provide 2-3 alternative starting points**, not a single committed path
- **Explicitly instruct:** "You do not know the answer. Do not guess specific entities, names, or events. Instead, describe what to search for."
- Remove the "Answer sketch" field — it invites hallucination

### HIGH: Tell the Planner About the Retrieval System
The plan's search suggestions are calibrated for web search, not dense semantic retrieval. Add:
- "The executor uses dense semantic retrieval (embedding similarity), not keyword search. Write queries as natural language descriptions, not keyword strings. Boolean operators and exact-match patterns don't work."
- This is actually in v6.1 but NOT in v1, and the v1 plans are used in the start_ext condition

### HIGH: Make Plans Adaptive, Not Rigid
Replace the waterfall checklist with a more flexible format:
- **Strategy section:** 2-3 sentences describing the overall approach and constraint prioritization
- **Key insight:** What is the most distinctive/searchable constraint? What should the agent try first?
- **Pitfalls to avoid:** What dead ends might the agent fall into?
- **When to pivot:** Explicit criteria for abandoning the current approach

### MEDIUM: Hedge All Factual Claims
- Instruct: "If you recognize the answer or have a hypothesis, present it as one possibility among several. The executor must verify independently."
- Add: "Your plan will be wrong some of the time. Ensure the executor can recover by not committing to a single path."

### MEDIUM: Model-Specific Format Considerations
- Tongyi has severe issues parsing/executing plans (123 zero-tool-call trajectories)
- Consider a simpler injection format or model-specific plan formatting
- Test whether "system message" injection works better than "user message" for plan delivery

### LOW: Plan Complexity Scaling
- Simple questions (short query, few constraints) don't benefit from planning at all
- Consider a "plan/no-plan" classifier that only generates plans for complex questions (600+ char queries, 4+ constraints)

---

## 7. Radical Alternative: Ditch the Tree-Structured Checklist

Given that planning consistently hurts accuracy, consider a fundamentally different plan format:

### "Strategy Brief" Format (proposed)
Instead of a detailed action plan, generate a 3-5 sentence strategy brief:

```
STRATEGY: [1-2 sentences: overall approach]
START WITH: [the single most searchable constraint and why]
AVOID: [common pitfalls for this type of question]
PIVOT IF: [when to abandon current approach]
```

This format:
- Uses ~200-400 characters (vs 2000+)
- Doesn't lock the agent into specific queries
- Provides the "domain disambiguation" benefit (Pattern A) without the rigid structure
- Leaves the agent free to adapt based on search results
- Minimizes context consumption

### "Constraint Analysis Only" Format (alternative)
Just decompose the question into constraints with priority ranking:

```
CONSTRAINTS (search first → verify last):
1. [most distinctive/searchable constraint] — START HERE
2. [second constraint] — use to narrow candidates
3. [filter constraint] — apply as elimination criterion
4. [temporal dependency] — depends on result of #1
CAUTION: [any ambiguity or false assumption risk]
```

This is essentially the useful part of the current plan without the action steps.

---

## 8. Summary of Priority Changes

| Priority | Change | Expected Impact |
|----------|--------|-----------------|
| P0 | Hard cap plan length to <800 chars | Fix 119 extra incompletions |
| P0 | Eliminate confident hypotheses / entity guessing | Fix ~30% of plan-lose cases |
| P1 | Add retrieval system description (semantic, not keyword) | Better search queries |
| P1 | Replace checklist with strategy brief format | More adaptive execution |
| P2 | Hedge all factual claims, provide alternatives | Reduce single-path failures |
| P2 | Fix plan injection format for Tongyi | Fix 123 zero-tc trajectories |
| P3 | Add plan/no-plan classifier for question complexity | Skip planning for easy questions |
