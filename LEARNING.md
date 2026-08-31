# Understanding this project

A guide to the design for someone who has to *explain* it — in an interview, to
a colleague, or to yourself in six months.

It is organised around decisions rather than files, because a decision is what
someone will actually ask about. Every section has the same shape: what the
choice was between, why it went the way it did, and what would have happened
otherwise.

---

## Part 1 — What problem is this, really?

A procurement team at a Swedish municipality receives supplier invoices. Each
one is governed by a contract that states a rate, sometimes a volume ceiling,
sometimes an exclusion. Somebody checks them by hand.

Three properties of the problem shape everything downstream:

**The error tolerance is low in both directions.** A missed discrepancy is an
overpayment of public money. A false accusation wastes a caseworker's afternoon
and damages a supplier relationship. Most ML framing optimises one number;
here, the two errors are different in kind and never get averaged.

**The answer must be checkable.** A flag a caseworker cannot verify is worse
than no flag, because it costs them the time the system was supposed to save.
That single requirement forces clause-level citations, which forces
structure-aware chunking, which forces the ingestion design.

**The value is triage, not automation.** Nobody wants a robot approving
payments. They want the search narrowed from "read the contract" to "read this
sentence".

> **Say this out loud:** "It's a triage tool, not an approval tool. The output
> is always 'a human should look at this' or 'I found nothing wrong' — never
> 'pay it'. `requires_human_review` is hardcoded `True` on every path, and
> there's a test that asserts it."

---

## Part 2 — The decision that everything rests on

### LLM interprets, code decides

Look at what the model is allowed to return (`app/models.py`):

```python
class ClauseInterpretation(BaseModel):
    governing_clause_id: str | None
    contract_rate: float | None
    rate_unit: str
    quantity_cap: float | None
    ...
```

There is no `is_correct`. No `should_flag`. No `expected_total`. The model
reports **what the clause says**, and it is explicitly forbidden — in the system
prompt — from doing arithmetic or reaching a verdict.

Then `app/core/check.py` decides, in ordinary Python:

```python
expected_total = round(line.quantity * interp.contract_rate, 2)
if abs(line.unit_price - interp.contract_rate) > settings.rate_tolerance:
    ...
```

**Why it matters.** A hallucinated multiplication would approve an overpayment
and look exactly as confident as a correct one. There is no way to detect it
downstream, because the output format is identical. Arithmetic is free, exact
and testable in Python; delegating it to a system that is *merely usually right*
buys nothing and risks everything.

The practical dividend: `check.py` is a pure function, so `tests/test_check.py`
pins down every decision path with no API calls, no mocking, no flakiness. The
part that handles money is the part that is deterministic.

> **The question you will get:** *"How do you stop the LLM hallucinating a
> number?"*
>
> "I don't try to. I make sure it's never asked for one. It reports the rate it
> read from a clause; the multiplication happens in Python. And the clause it
> cites is checked against the set it was actually shown — if it cites something
> that wasn't retrieved, that's a hallucinated reference and the line fails
> closed to a human."

---

## Part 3 — Retrieval, and the bug that taught the most

### Two-stage search

Stage 1, a bi-encoder, embeds the query and fetches 20 candidates by vector
similarity — fast, high recall, mediocre precision. Stage 2, a cross-encoder,
reads query and clause *together* and reranks down to 4 — slow, high precision.
Running stage 2 over the whole corpus would be far too slow; running only stage
1 puts weak clauses in front of the model. The two-stage shape is the standard
answer and it is the right one.

### The bug: one query cannot find two different facts

The first version issued one query per invoice line, built from the line
description. Volume-ceiling violations were never caught. Not caught *wrongly* —
never caught at all.

The reason is obvious in hindsight. A line reads *"Config and go-live of
procured software"*. The clause that prices it says *"implementation… SEK 1 150
per hour"* — shared vocabulary, retrieved fine. The clause that caps it says
*"§5.1 Monthly ceiling — a maximum of 160 hours per calendar month"*. It shares
**no vocabulary with the work being described**, so it never ranked, and the
system silently never had the fact it needed to flag an over-cap invoice.

The fix is to split the query by the *fact being sought* rather than by the text
available (`app/core/interpret.py`):

```python
primary = retrieve(line.description, ...)   # what is this work?
rates   = retrieve(RATE_QUERY, ...)         # what does it cost?
caps    = retrieve(CAP_QUERY, ...)          # how much is allowed?
```

The same bug recurred later in a different disguise: `INV-007`'s rate clause
ranked *sixth* because the line description matched the scope clause (§3.1
"training… digital tools in teaching") far better than the fee clause (§4.1 "SEK
12 500 per training day"). Adding the rate-targeted query took retrieval recall
from 88.9% to 100%.

**The general lesson, which is the thing worth saying:** contracts organise
clauses by *type* — §3 scope, §4 fees, §5 volume — while an invoice line
describes *work*. So one query written from the line text systematically
retrieves one of the three clause families and misses the others.

> **Say this out loud:** "Retrieval failures are invisible. The system doesn't
> error — it just reports 'no clause governs this line' when the clause was
> right there at rank six. That's exactly why the eval scores retrieval recall
> separately from decision accuracy: end-to-end accuracy alone can't tell you
> which half is broken, and the two have opposite fixes."

### Chunking follows from citation

Chunks are clauses, split on `§` headings, never fixed-size windows. Three
reasons, in order of importance:

1. **Citations need clause IDs.** `C-2024-001-§4.2` is a thing a caseworker can
   look up. "Chunk 47" is not.
2. **A rate without its heading is a number without a meaning.** A window that
   splits "SEK 1 150 per hour" from the heading saying what it is *for* produces
   confident wrong answers, and nothing errors.
3. **Tables must stay whole.** Split a rate table and you get `895` with no row
   label — worse than useless, because it is plausible.

---

## Part 4 — Access control, and why the *location* is the point

Contracts belong to departments. A caseworker in Education must not see the IT
department's consultancy rates.

The filter is applied **during retrieval**, inside the SQL that does the search
(`app/store/sqlite_store.py`):

```sql
EXISTS (SELECT 1 FROM json_each(chunks.allowed_groups) AS g
        WHERE g.value IN (?, ?))
```

Not fetch-then-filter. Not filter-the-response. The unauthorised text is never
loaded into the process, so it cannot reach a prompt, a trace file, a log line,
or an error message — even if the code above it has a bug.

In Postgres there is a second, independent layer (`app/store/schema_pg.sql`):

```sql
CREATE POLICY chunks_group_read ON chunks FOR SELECT
    USING (allowed_groups && string_to_array(current_setting('app.user_groups', true), ','));
```

If the application forgets the `WHERE` clause entirely, the database still
returns nothing. And it is deny-by-default: unset the setting and
`current_setting(..., true)` is `NULL`, which is not `true`, so nothing matches.
The setting is bound per-transaction, so a pooled connection cannot leak one
user's groups into the next user's query.

Two details worth knowing you made deliberately:

**A forbidden contract returns 404, not 403.** "You may not read this" confirms
the contract exists — which supplier the municipality contracts with is itself
protected information. Both cases return the identical response, and a test
asserts they are byte-identical.

**Identity comes from headers, never from the request body.** `CheckRequest` has
a `principal` field for in-process callers, and `app/main.py` deliberately
ignores it. A caller who could name their own groups in the JSON they post would
have no access control at all. There is a test that posts privileged groups in
the body with an unprivileged header and asserts nothing leaks.

> **The question you will get:** *"Where do you enforce authorisation?"*
>
> "In the query predicate, and again in Postgres RLS. The distinction I'd draw
> is that post-filtering is a correctness bug waiting to happen — the data is
> already in memory, so any log line, error message or prompt built before the
> filter runs leaks it. Filtering during retrieval means it never arrives.
> Authentication, though, is *not* built — headers stand in for OIDC claims.
> That's one function."

---

## Part 5 — Parsing, or: the 4% that becomes a 500

A model asked for JSON returns JSON almost always. The residual few per cent is
where `json.loads(response)` becomes a production incident, and the failures are
boringly repetitive: markdown fences, a preamble, a trailing comma, `1 150`
written the Swedish way, truncation at `max_tokens`.

`app/llm/parsing.py` is a ladder, cheapest rung first: direct → strip fence →
scan for the first balanced block (respecting strings and escapes) → mechanical
repairs → close a truncated object. Every rung records what it did, and
`ParseResult.repairs` goes to the trace — because "the model returned malformed
JSON 4% of the time and we repaired it" is an operational fact worth measuring
rather than guessing.

**The last rung is the one that matters.** If nothing yields a valid object, it
returns a *failure*, not a half-populated object:

```python
def test_a_missing_rate_is_never_silently_zero():
    result = parse_model("not json at all", ClauseInterpretation, ...)
    assert result.value is None, "must not produce an object with contract_rate=0.0"
```

A `contract_rate` defaulted to `0.0` would flow into the arithmetic and produce
a confident, precise, entirely wrong statement about public money — and nothing
downstream would mark it as suspect. Failing closed sends it to a human, which
is the correct answer to "I don't know".

**A bug worth telling as a story.** The number parser handled `"1 150"` (Swedish
thousands space) and `"1 150,50"` (decimal comma), and I wrote a test for
`"1,150.50"` expecting it to pass. It returned **1.15** — a factor-of-1000 error
on a money value. The regex stopped the integer part at the first comma. The
fix was to capture the whole run of digits and separators and disambiguate
afterwards: if both separators appear, the *last* one is the decimal point; a
lone separator followed by exactly three digits is a thousands group. It is in
`_to_float`, with the reasoning in the docstring, and there are now nine
parametrised cases covering every convention a model might echo.

> **Say this out loud:** "Structured output is not a solved problem — it's a
> solved problem 96% of the time, and the other 4% is a 500 or, worse, a
> plausible wrong number. I treat parsing as a ladder that records what it
> repaired, and the terminal rung fails closed rather than defaulting."

---

## Part 6 — The agent loop, and what bounding actually means

`app/agent/loop.py` is an alternative to the fixed pipeline: instead of three
predetermined queries, the model chooses its own searches. It exists so the eval
can put a *number* on whether that is worth it.

The interesting part is not that it calls tools — that is twenty lines. It is
what happens when it will not stop. There are four bounds, because there are
four distinct ways to fail:

1. **Iteration cap.** The hard stop.
2. **Repeat detection.** A model asking the identical question twice has learned
   nothing from the first answer and will not learn from the third. Cheaper than
   waiting out the cap, and it catches the common two-call cycle much earlier —
   3 iterations instead of 6, verified by test.
3. **No-progress detection.** Several empty results mean it is searching for
   something that is not there; the honest answer is "not found".
4. **A forced final turn.** On hitting any bound, the model is asked once more
   *with no tools offered*, so it must answer from what it already has.

Running out of budget is **not an error** — it returns the best available answer
marked `exhausted`, which becomes `INTERPRETATION_FAILED`, which routes the line
to a human. The system is allowed to run out of ideas. It is not allowed to
invent a clause because it did.

Two things the tools deliberately do *not* have:

**No tool can decide anything.** `retrieve_clauses` and `read_clause`. No
`calculate`, no `flag`. Giving a model a tool is giving it an action, so a
`calculate` tool would move arithmetic back inside the model and undo Part 2. A
test asserts the tool list contains exactly those two names.

**The principal is bound at construction, not passed as an argument.** The model
chooses *what* to look up, never *who is asking*. This matters specifically
because contract text is supplier-supplied and lands directly in the model's
context — an injected instruction in a PDF is untrusted input the model may well
follow. Because groups aren't a parameter, the worst an injection can achieve is
retrieving something the caseworker could already read. There is a test that
passes `{"groups": ["it-dept"]}` into a tool call from an unprivileged user and
asserts it changes nothing.

> **The question you will get:** *"How do you keep an agent from looping
> forever?"*
>
> "An iteration cap is the floor, not the answer. I also detect repeated calls
> and lack of progress, because those catch the pathological cases earlier and
> more cheaply. The part I'd emphasise is the exit: on exhaustion it gets one
> tool-free turn to answer from what it has, and if it still can't, the line
> fails closed to a human review rather than returning a guess."

---

## Part 7 — Evaluation: the part that makes claims checkable

Every claim in the README came from `make eval`, and any of them can be
re-checked by running it again. The harness (`eval/`) is what turns "it seems to
work" into something defensible.

Three design choices worth explaining:

**Retrieval is scored separately from decisions.** Discussed in Part 3 — it is
the single most useful thing in the harness, because it tells you *which half*
to fix.

**There is a `ceiling` metric.** The accuracy achievable if the interpreter were
perfect, given the retrieval that actually happened. The gap between accuracy
and ceiling is the interpretation problem; the gap between ceiling and 100% is
the retrieval problem.

**False matches and false flags are never averaged.** They are reported as two
lists of invoice IDs. A false match is an overpayment nobody catches; a false
flag is a wasted afternoon. One "accuracy" number would hide the trade that
`ambiguity_confidence_threshold` exists to tune.

**The stub is a baseline, not a mock.** It is a lexical heuristic scored by the
same harness, which is what makes "what does the LLM buy us?" a measurable
question rather than an assertion. It currently scores 88.9% and its two
failures are both `ambiguous_mapping` cases needing a two-hop inference (§3.1
describes the work, §4.2 prices it) that lexical matching cannot make. That gap
*is* the answer.

And the harness immediately earned its keep: comparing the agent against the
fixed pipeline showed the agent **losing** — 77.8% vs 88.9% — and the retrieval
column showed why (recall 88.9% vs 100%). Worse, its errors were false
*matches*. Without separated metrics that would have looked like a model
problem; it was a search-strategy problem.

---

## Part 8 — Where the bodies are buried

Be first to say these. Volunteering a limitation reads as judgement; being
caught by one reads as overselling.

- **The data is synthetic.** I wrote the eight contracts and generated the
  invoices from a spec so the arithmetic per category is exact. 18 invoices is
  small — one error moves accuracy 5.6 points. The harness is the artefact, not
  the score.
- **Authentication is not implemented.** Authorisation is real and layered;
  identity is a header.
- **The eval numbers are the offline baseline.** No API key was used, by design,
  so they are a floor.
- **Everything is English.** I initially wrote the invoice lines in Swedish
  against English contracts and realised it created a cross-lingual retrieval
  problem the embedder could not solve — the eval would have been measuring the
  wrong thing. Rewrote them in English, kept Swedish supplier and place names.
  Real deployment needs a multilingual embedding model.
- **OCR is structural.** The `scanned` contract routes through the OCR branch
  and reports honestly that no recognition ran on a text fixture.
- **Terraform has never been applied.**
- **Single-line invoices dominate.** Multi-line invoices with interacting
  ceilings (a cap shared across lines) are not handled — the cap is checked
  per line, not per invoice.

---

## Part 9 — How to study this

**A reading order that builds up rather than sideways:**

1. `app/models.py` — the vocabulary. Note what `ClauseInterpretation` cannot say.
2. `app/core/check.py` — the decision, ~170 lines, no ML. Everything else feeds this.
3. `tests/test_check.py` — the decisions pinned down as examples.
4. `app/core/interpret.py` — the LLM boundary, and the three-query docstring.
5. `app/retrieval/retriever.py` + `app/store/sqlite_store.py` — where the access filter lives.
6. `app/llm/parsing.py` — the failure ladder.
7. `app/agent/loop.py` — bounding.
8. `eval/metrics.py` — why the numbers are split the way they are.

The module docstrings carry the reasoning, not just the description. That is
deliberate: the *what* is readable from the code, the *why* is not.

**Exercises that will teach you more than reading:**

- Break the cap query. Delete `CAP_QUERY` from `gather_clauses` and run
  `make eval`. Watch `unauthorized_quantity` collapse while everything else
  holds — that is what a silent retrieval failure looks like.
- Move the threshold. Set `AMBIGUITY_THRESHOLD=0.95` and watch false flags rise
  as false matches fall. That is the trade, and you can feel it.
- Run as the wrong user. `scripts/check_invoice.py INV-008 --groups education`.
- Read a trace. `var/traces.jsonl` has one JSON object per decision — every
  clause retrieved, both stage scores, the raw model output, what the parser
  repaired.
- Add a ninth contract and a discrepancy category the system has never seen.
  This is the real test of whether you understand the pipeline.

---

## Part 10 — Talking about it

**The two-sentence version.**

> "It checks supplier invoices against the contracts governing them and flags
> discrepancies for a caseworker — it never approves anything. The core design
> decision is that the LLM only interprets clauses; all arithmetic and every
> verdict happen in deterministic Python."

**The two-minute version.** Problem (manual checking, low error tolerance both
ways) → pipeline (validate, lookup, retrieve, interpret, check) → the split (LLM
reads, code decides, and why) → citations (every flag names a clause) →
evaluation (retrieval scored apart from decisions, false matches apart from
false flags) → honest scope (synthetic data, no auth).

**If you only remember four things:**

1. *LLM interprets, code decides* — and why hallucinated arithmetic is
   undetectable downstream.
2. *The access filter is in the query predicate* — unauthorised text never
   enters the process.
3. *One query cannot find two different facts* — the cap-clause bug, and how it
   failed silently.
4. *Failing closed beats defaulting* — a `0.0` rate is worse than an error,
   because nothing marks it as suspect.

**Do not claim:** production experience, real users, that the eval numbers
generalise from 18 synthetic invoices, or that authentication is built.

**Do claim:** you designed the LLM/deterministic boundary and can defend where
you drew it; you found and fixed a silent retrieval failure and built the metric
that would have caught it sooner; you put authorisation in the right layer and
know why the layer matters; you can explain what bounding an agent means beyond
a max-iterations constant.
