# Invoice–Contract Decision Assistant

Checks supplier invoices against the contracts that govern them, and **flags
discrepancies for a caseworker**. It never approves anything.

The setting is a Swedish municipality's procurement team. Someone currently
opens an invoice, finds the right contract, finds the clause that states the
rate, and does the arithmetic. That is the work this compresses — from reading
a contract to reading one sentence with a clause reference next to it.

```
  INV-011   Nordisk Konsult AB   contract C-2024-001
  period 2025-03   invoiced 224,250.00 SEK

  DECISION: FLAG   findings: unauthorized_quantity
  amount in question: 40,250.00 SEK
  requires human review: True

  [FLAG] line 1: Integration work, running hours March
         billed 1,150.00 / contract 1,150.00
         Quantity 195 exceeds the ceiling of 160 hour per calendar month
         set by C-2024-001-§5.1 (35 over, worth 40,250.00).
         source: C-2024-001-§4.2
```

The rate was right. The volume was not. That is the kind of thing that is
tedious to catch by hand and easy to catch here — provided the system also
finds the ceiling clause, which is most of the engineering.

---

## Run it

No database, no API key, no GPU. It works on a clean checkout because every
external dependency has a built-in fallback (see
[Backends](#backends-and-why-there-are-fallbacks)).

```bash
make install     # venv + runtime and dev dependencies
make test        # 91 tests
make eval        # score the system against the labelled set
make run         # API on http://localhost:8000
```

Check a single invoice and read the output as a caseworker would:

```bash
make check INVOICE=INV-011
.venv/bin/python scripts/check_invoice.py INV-008 --mode agent
.venv/bin/python scripts/check_invoice.py INV-008 --groups education   # denied
```

Over HTTP:

```bash
curl -s localhost:8000/health | jq          # which backends actually resolved

curl -s localhost:8000/check \
  -H 'Content-Type: application/json' \
  -H 'X-User-Id: anna' \
  -H 'X-User-Groups: procurement,it-dept' \
  -d "{\"invoice\": $(cat data/invoices/INV-008.json)}" | jq

# Same request, a user in the wrong department: no clauses, no rates, no leak.
curl -s localhost:8000/check \
  -H 'Content-Type: application/json' \
  -H 'X-User-Groups: education' \
  -d "{\"invoice\": $(cat data/invoices/INV-008.json)}" | jq '.findings'
```

With Postgres, so row-level security is genuinely exercised rather than
described:

```bash
docker compose up --build      # pgvector + the API
```

### Everything available

| Command | What it does |
|---|---|
| `make install` | Runtime + dev dependencies |
| `make install-all` | Adds the real model, embeddings and OCR extras |
| `make data` | Regenerate the synthetic invoices and labels |
| `make validate` | Prove the golden set is internally consistent |
| `make ingest` | Parse → chunk → embed → index the contracts |
| `make run` | Ingest, then serve on :8000 |
| `make check INVOICE=INV-008` | Check one invoice from the CLI |
| `make eval` | Score the fixed pipeline |
| `make eval-agent` | Score both retrieval strategies side by side |
| `make test` | Test suite |

---

## What it does, in order

```
invoice ──▶ validate ──▶ find contract ──▶ retrieve clauses ──▶ interpret ──▶ CHECK ──▶ flag
            (code)        (code)            (embeddings)         (LLM)        (code)
```

1. **Validate** — structure only. Missing quantity, negative price, no lines.
2. **Find the contract** — a metadata lookup, not a search. Deterministic, and
   the access filter applies here first.
3. **Retrieve clauses** — two-stage: a bi-encoder fetches 20 candidates, a
   cross-encoder reranks to 4. The caller's group filter is part of the SQL
   `WHERE`, so unreadable clauses are never retrieved at all.
4. **Interpret** — the only LLM call. It reads clauses and reports *what they
   say*: which clause governs, the rate, the unit, any ceiling. It is forbidden
   to do arithmetic or reach a verdict.
5. **Check** — plain Python. Compares the rate, recomputes the total, tests the
   quantity against the ceiling. Every number in the output comes from here.
6. **Flag** — with a clause citation, the amount in question, and the sentence
   explaining it.

**The split at steps 4 and 5 is the whole design.** A hallucinated
multiplication would approve an overpayment of public money and look exactly as
confident as a correct one. Arithmetic is free and exact in Python, so there is
no reason to delegate it to something that is merely usually right. The model
does the genuinely fuzzy job — mapping `"snr tech"` onto *"Senior technician"* in
a rate table — and nothing else.

---

## Results

18 labelled invoices, seven discrepancy categories, scored by `make eval`.
These numbers are from the **offline stub** — no API key — so they are a floor,
not a showcase.

```
decision accuracy            : 88.89%
category accuracy            : 88.89%

-- retrieval, scored independently of the model --
recall of governing clause   : 100.00%
MRR                          : 0.661
category ceiling given that  : 100.00%

-- the two errors, kept apart --
false MATCHES (missed money) : 2   [INV-015, INV-016]
false FLAGS (wasted time)    : 0
```

Retrieval and generation are scored **separately and on purpose**. A wrong
answer has two possible causes — the clause was never retrieved, or it was
retrieved and misread — and they have opposite fixes. Recall at 100% says every
remaining error belongs to the interpreter, so tuning the chunker would be
wasted effort.

Recall 100% next to MRR 0.661 is worth reading carefully: the governing clause
is always in the candidate set but is typically *second or third*, not first.
That is fine here — the model is shown all four and picks — but it is the exact
signature of a reranker doing less work than it appears to, and it is why both
numbers are reported. Recall alone would have looked perfect.

The two failures are both `ambiguous_mapping`, and both are *expected*: they
need a two-hop inference (§3.1 describes the work, §4.2 prices it) that a
lexical baseline cannot make. They are left in as the measurable gap that shows
what a real model buys. Set `ANTHROPIC_API_KEY` and re-run to close it.

Both are false *matches*, which is the dangerous direction — the system said
"no discrepancy found" when a human should have looked. That asymmetry is why
the report never averages the two error types into one accuracy figure.

---

## Backends, and why there are fallbacks

Every heavy dependency sits behind an interface with a working substitute, and
the substitute is selected automatically when the real one is unavailable:

| Component | Preferred | Fallback | Chosen by |
|---|---|---|---|
| Vector store | Postgres + pgvector | SQLite + numpy | `DATABASE_URL` set? |
| Embeddings | sentence-transformers | hashing embedder | is it installed? |
| Reranking | cross-encoder | BM25 lexical | is it installed? |
| Model | Anthropic API | deterministic stub | `ANTHROPIC_API_KEY` set? |

Three reasons this is a design decision and not a workaround:

- **The sovereignty constraint is real.** Municipal procurement data may need to
  stay inside the customer's environment. `LLMClient` exists so that swapping in
  a self-hosted model is one new class and one environment variable. Nothing in
  `app/core/` imports `anthropic`.
- **The stub is a baseline, not a mock.** It is a lexical heuristic scored by
  the same harness, so "what does the LLM actually buy us?" has a number.
- **The project runs anywhere**, including on a reviewer's laptop, and upgrades
  itself when infrastructure appears.

`GET /health` reports which ones actually resolved — worth checking, because the
service starts happily on fallbacks and the accuracy difference is large.

---

## Layout

```
app/
  ingest/      parse contracts (PDF/OCR/tables) → clause-shaped chunks
  embeddings/  bi-encoder + cross-encoder, each with a fallback
  store/       pgvector and SQLite behind one interface; schema_pg.sql has RLS
  retrieval/   two-stage search with the access filter applied *during* search
  llm/         client interface, prompts, defensive JSON parsing
  core/        validate → lookup → interpret → check → pipeline
  agent/       bounded tool-calling loop (tools read; they cannot decide)
  security/    access filter, PII redaction
  obs/         JSONL tracing, Prometheus metrics
  main.py      FastAPI surface
eval/          harness + metrics (retrieval scored apart from decisions)
data/          8 contracts, 18 invoices, labels.json, access.json
tests/         91 tests
deploy/        Helm chart, Terraform
```

The module docstrings carry the reasoning — why two queries per line, why
parsing fails closed, why the exclusion check needs a margin. Start with
`app/core/check.py`, which is short and is where every decision is actually
made.

---

## Notes on scope

Written as a portfolio project, so the honest boundaries:

- **The data is synthetic.** Eight contracts I wrote, invoices generated from a
  spec so the arithmetic per category is exact. Real municipal contracts are
  messier, and 18 invoices is small enough that one error moves accuracy by 5.6
  points. The harness matters more than the score.
- **Authentication is not built.** `X-User-Groups` stands in for OIDC claims.
  The *authorisation* is real — filtering during retrieval, plus RLS in
  Postgres — but identity is asserted, not verified. `app/main.py:_principal`
  is the one function that would change.
- **The OCR path is structural.** Contracts ship as Markdown; `C-2024-006` is
  flagged `scanned` and routed through the OCR branch, which reports itself
  honestly rather than pretending text was recognised.
- **Terraform has never been applied.** It shows the shape of the deployment,
  not operations experience.

If you are reading this to understand the design rather than to run it,
[`LEARNING.md`](LEARNING.md) is the walkthrough: what each decision was between,
what broke, and the questions worth being able to answer about it.
