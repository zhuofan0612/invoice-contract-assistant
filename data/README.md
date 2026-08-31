# The dataset

Synthetic. A fictional Swedish municipality, "Sundhamn": 8 contracts, 18
invoices, machine-readable ground truth.

It is synthetic for the obvious reason — real municipal invoices and contracts
are not public — but it is *designed* rather than sampled, which turns out to be
the more useful property. Every discrepancy category is represented, the
arithmetic in each is exact, and the labels name the specific clause that
governs each invoice. That is what makes retrieval scoreable separately from
decisions.

```
contracts/     8 contracts as Markdown, clause-structured with § headings
invoices/      18 invoices as JSON
labels.json    ground truth: category, expected decision, governing clause
access.json    which groups may read which contract
```

## Contracts

Written by hand so the retrieval problem is realistic rather than trivial:
several contracts price similar work at different rates, and the clauses that
state rates, ceilings and exclusions are deliberately kept apart — which is how
real contracts are organised, and the reason one query per line is not enough.

| ID | Supplier | Dept | Notable |
|---|---|---|---|
| C-2024-001 | Nordisk Konsult AB | IT | Two rates (advisory 1 450, implementation 1 150); §3.2 excludes hardware; 160 h/month cap |
| C-2024-002 | Björkdal Städservice AB | Facilities | Per-occasion pricing; §3.3 excludes snow clearance; 22/month cap |
| C-2024-003 | Vasaberg Teknik AB | Facilities | **Rate table** by role (745 / 895 / 1 340) |
| C-2024-004 | Lindqvist Anläggning AB | Urban Dev | Priced per running metre — a unit that must not match hourly lines |
| C-2024-005 | Sundberg IT-Drift AB | IT | **Tiered table**: 1–50 → 210, 51–200 → 175, 201+ → 140 |
| C-2024-006 | Almgren Utbildning HB | Education | **Marked `scanned`** — routed through the OCR path |
| C-2024-007 | Kvarnholmen Livsmedel AB | Education | Per-portion; weekly rather than monthly cap |
| C-2024-008 | Trelleborg Säkerhet AB | Security | Two rates (guarding 520, consultancy 1 180) |

Three deliberate difficulties:

- **Two contracts contain tables** (a role/rate matrix and quantity-banded
  tiers), so chunking has to keep a table whole or a rate loses the row label
  that gives it meaning.
- **One is flagged `scanned: true`**, exercising the OCR ingestion branch.
- **Units vary** — hour, occasion, metre, portion, device, day — so a rate only
  governs a line if the units are compatible. An hourly rate does not price a
  line billed per metre, and the interpreter is told so explicitly.

## Invoices

18 invoices across seven categories:

| Category | n | What it tests |
|---|---|---|
| `clean_match` | 4 | No false flags on correct invoices |
| `overbilling_wrong_total` | 3 | `quantity × price ≠ line_total` — catchable with no contract at all |
| `wrong_rate` | 3 | Billed rate differs from the contract rate |
| `unauthorized_quantity` | 2 | Within rate, over the ceiling — needs the *cap* clause |
| `item_not_in_contract` | 2 | Positively excluded by a clause |
| `ambiguous_mapping` | 2 | Two clauses plausibly apply at different rates |
| `no_matching_clause` | 2 | Nothing in the contract governs this line |

Descriptions use realistic shorthand — `"snr tech"`, `"Config and go-live of
procured software"`, `"CPD delivery — digital tools in teaching"` — so mapping a
line onto legal prose is a genuine semantic task rather than string matching.
That mapping is the one job the LLM is given.

The last two categories are the important ones: `ambiguous_mapping` and
`no_matching_clause` have no correct *answer*, only a correct *behaviour*, which
is to say so and route to a human. A system that always produces a confident
verdict scores badly on them, as it should.

**Note on language.** Line descriptions are English; supplier and place names
are Swedish. An earlier version had Swedish descriptions against English
contracts, which created a cross-lingual retrieval problem the embedder could
not solve — the eval would have been measuring translation failure rather than
clause matching. Real deployment needs a multilingual embedding model.

## labels.json

```json
{
  "invoice_id": "INV-011",
  "category": "unauthorized_quantity",
  "expected_decision": "flag",
  "governing_clause_id": "C-2024-001-§4.2",
  "discrepancy": "195 hours invoiced against a ceiling of 160 h/month.",
  "notes": "Rate is correct; only the volume is wrong."
}
```

`governing_clause_id` is what makes retrieval independently scoreable: recall
and MRR are computed against it without reference to what the model concluded.

## access.json

Eight contracts, seven distinct group combinations — enough that a bug which
ignored groups would show up as cross-department results rather than passing by
coincidence.

| Contract | Department | Groups |
|---|---|---|
| C-2024-001, C-2024-005 | IT | `procurement`, `it-dept` |
| C-2024-002 | Facilities | `procurement`, `facilities` |
| C-2024-003 | Facilities | `procurement`, `facilities`, `property-mgmt` |
| C-2024-004 | Urban Development | `procurement`, `urban-dev` |
| C-2024-006 | Education | `procurement`, `education` |
| C-2024-007 | Education | `procurement`, `education`, `school-catering` |
| C-2024-008 | Security | `procurement`, `security-dept` |

`procurement` sees everything; department groups see their own. A contract with
no entry defaults to **no access**, not open access.

## Regenerating and trusting it

```bash
make data       # regenerate invoices, labels and access.json
make validate   # prove the golden set is internally consistent
```

`make validate` is the more interesting of the two. Ground truth that is itself
wrong is worse than no ground truth, so the validator resolves every
`governing_clause_id` **through the real chunker** and recomputes the arithmetic
for every category:

```
contracts ingested : 8 | clause chunks : 67 | distinct clause IDs : 67 | invoices : 18
note: 7 distinct access-group combinations across 8 contracts
OK: every governing_clause_id resolves to a real clause, every
invoice's arithmetic matches its labelled category, and the category
distribution is as specified.
```

If a change to chunking made a labelled clause ID stop existing, the eval would
otherwise keep running and silently score against clauses that can never be
retrieved. This fails loudly instead — and `tests/test_chunking.py` asserts the
same property.
