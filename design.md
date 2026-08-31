# Invoice–Contract Decision Assistant — Design

> A realistic end-to-end AI deployment built to exercise the full production
> lifecycle: discovery → design → ingestion → retrieval → LLM-interpret +
> deterministic-check → access control → evaluation → observability → deploy.
> This is a learning/portfolio project, not a live production system with real
> users. Framed honestly on that basis.

---

## 1. Discovery (the "customer" brief)

- **Problem:** A mid-sized municipality's procurement team manually checks
  whether each supplier invoice matches the terms of the governing contract
  (rate, quantity, allowed items). It is slow, and they suspect overbilling.
- **Users:** Procurement caseworkers — non-technical, domain experts.
- **Desired outcome:** The system *flags* invoice–contract discrepancies for a
  human to review, cutting manual checking time. It does **not** auto-approve
  or auto-reject payments.
- **Error tolerance:** Low. A wrong "match" could approve an overpayment, and a
  wrong "mismatch" wastes the team's time and erodes trust. Therefore: a human
  confirms every flagged decision; the system never acts autonomously.
- **Data:** ~30–50 contracts (PDFs, some scanned) and a set of invoices
  (structured fields: supplier, contract ID, line items, quantity, unit price,
  total).
- **Boundary constraint (design as if):** Sensitive procurement data must not
  leave the customer's environment. Components must be self-hostable. For THIS
  build I will use the Anthropic API for the LLM to move fast, but I will
  document clearly which parts would swap to a self-hosted model in a real
  sovereign deployment (this is itself an interview talking point).

---

## 2. Scope for this build

- **In scope:** ingestion (incl. tables + OCR), two-stage retrieval,
  access-control-aware retrieval, LLM-interpret + deterministic-check core,
  tool-calling loop, evaluation harness, observability/tracing, containerized
  deploy with Helm + Terraform.
- **Out of scope (stated honestly):** real users, HA, production hardening,
  full ERP/DMS integration (simulated with local files + a mock API).
- **Framing:** "I built this to work through a realistic end-to-end AI
  deployment, exercising the full production lifecycle." Not "it serves real
  customers."

---

## 3. Architecture overview

Flow for a single invoice:

1. Invoice arrives (mock API / file drop) → **validate** structured fields.
2. **Deterministic contract lookup** — find the governing contract by supplier
   + contract ID (metadata filter, NOT semantic search).
3. **Semantic clause retrieval** — within that contract, retrieve the relevant
   clause(s) (pricing, quantity, allowed items) via bi-encoder candidate search
   + cross-encoder rerank.
4. **LLM interpretation** — read the rate, unit, and terms out of the
   natural-language clause; map messy invoice line descriptions to contract
   items. Returns STRUCTURED output (JSON).
5. **Deterministic check** — perform the arithmetic/rule comparison
   (rate × quantity == total? quantity within authorized limit?). The pass/fail
   decision is code, not the LLM.
6. **Flag for human** — surface discrepancies with the cited contract clause;
   never auto-act.
7. **Log/trace** everything for audit and evaluation.

---

## 4. Component boundaries — LLM vs deterministic (the important table)

| Component            | Responsibility                        | LLM / Deterministic          | Why                                             |
|----------------------|---------------------------------------|------------------------------|-------------------------------------------------|
| Field validation     | Check invoice schema is complete      | Deterministic                | Knowable rules                                  |
| Contract lookup      | Find the governing contract           | Deterministic (metadata)     | We have supplier + ID; no guessing              |
| Clause retrieval     | Find relevant clause in that contract | Semantic (bi + cross-encoder)| Language-shaped matching                         |
| Interpretation       | Read rate/terms; map descriptions     | **LLM**                      | Fuzzy natural-language understanding            |
| Match check          | Does the arithmetic/rule hold?        | **Deterministic**            | LLM unreliable at arithmetic; high error cost   |
| Decision             | Approve / flag                        | Human-in-the-loop            | High stakes; system flags, human decides        |
| Ambiguity handling   | >1 plausible clause                   | LLM proposes + confidence    | Surface both, human adjudicates                 |

**Rule of thumb enforced throughout:** deterministic wherever the answer is
knowable; LLM only where the path/meaning is genuinely fuzzy; human owns any
high-stakes decision.

---

## 5. Key design decisions (my reasoning = my interview answers)

- **pgvector over Chroma:** self-hostable, stores vectors alongside relational
  metadata so I filter + semantic-search in one query, and supports Postgres
  row-level security for defense-in-depth access control.
- **LLM interprets, code decides:** a hallucinated calculation could approve a
  payment. The safety-critical comparison must be deterministic.
- **Access control filtered DURING retrieval, not after:** forbidden chunks
  never enter the candidate set. App builds the filter from the user's groups;
  Postgres RLS enforces independently as a second layer (so a bug in app code
  can't leak documents).
- **Structured LLM output (JSON), defensively parsed:** the interpretation step
  returns a schema; I write the code that decouples/validates that response and
  handles malformed output — this is where production fragility hides.
- **"I don't know" / ambiguity is a feature:** if no clause clearly applies, or
  two are close, the system flags for a human rather than guessing.
- **Anthropic API now, swappable later:** the model is behind an interface so it
  can be replaced by a self-hosted model for a sovereign deployment. Documented
  as an explicit boundary.

---

## 6. Tech stack

- **Backend:** FastAPI (Python), async
- **Vector store:** pgvector (Postgres)
- **LLM:** Anthropic API (Claude) via a thin, swappable `LLMClient` interface
- **Embeddings:** a sentence-transformer bi-encoder + cross-encoder reranker
  (local, so retrieval works offline; keeps the "self-hostable" story intact)
- **Parsing:** PyMuPDF / pdfplumber for text + tables; an OCR step (Tesseract)
  for scanned PDFs
- **Evaluation:** golden set + retrieval metrics + generation/groundedness
  metrics + LLM-as-judge cross-checked against a deterministic baseline
- **Observability:** tracing (Langfuse or Arize Phoenix) + basic metrics
- **Deploy:** Docker → Helm chart → Terraform (local k8s / kind is fine)

---

## 7. Build phases (implement + learn incrementally — do NOT build all at once)

1. **Ingestion + storage** — parse contracts (incl. one scanned + one
   table-heavy), structure-aware chunk, attach metadata (contract ID, supplier,
   section, **access groups**), embed, store in pgvector.
2. **Retrieval** — two-stage (bi-encoder candidate → cross-encoder rerank) +
   **access-control filter applied during retrieval**.
3. **Invoice intake + validation** — mock API / file drop, schema validation,
   deterministic contract lookup.
4. **Interpret + check core** — LLM returns structured JSON (rate/terms/mapping);
   **I hand-write** the response decoupling/validation; deterministic code does
   the arithmetic decision; ambiguity → flag.
5. **Tool-calling loop (MCP)** — expose retrieval + lookup as tools; implement
   the bounded agent loop (max iterations, restricted tool set). **I write the
   loop skeleton myself.**
6. **Evaluation harness** — golden set; measure retrieval and generation
   separately; LLM-judge vs deterministic-baseline cross-check; wire as a
   regression gate.
7. **Access control hardening** — add Postgres **row-level security**; test that
   removing the app filter still doesn't leak (RLS holds).
8. **Observability** — tracing per request (chunks retrieved, model output,
   timings), metrics, a simple dashboard, an override/feedback signal.
9. **Deploy as code** — containerize, Helm chart, Terraform; package so it could
   ship as a self-contained bundle (air-gapped story).
10. **Guardrails** — PII detection/redaction (Presidio) + a groundedness check
    before returning.

---

## 8. Things I specifically want to understand deeply (teach-back targets)

- How the tool-calling loop bounds itself and decides the next tool.
- How to parse/validate/repair malformed LLM structured output.
- Exactly where the access filter lives (app vs DB) and how RLS backs it.
- How retrieval-vs-generation failure decomposition shows up in the eval numbers.
- What breaks with a table, a scanned PDF, an ambiguous clause, a malformed
  LLM response, and a permission edge case.

---

## 9. Honest-scoping reminders (for CV + interviews)

- Anthropic API used for speed; self-hosted swap documented, not implemented.
- Golden set is small; state the size and how I'd grow it from production data.
- No real users; framed as a lifecycle exercise. Do not inflate.