"""The deterministic checker.

This is the file that decides, so it is the file worth testing hardest, and it
is also the easiest to test hard: no LLM, no embeddings, no database. Given a
line and an interpretation, the verdict is a pure function. Every test here is
exact -- there is no `pytest.approx` on a decision about money.

The interpretation is constructed directly rather than obtained from a model,
which is the point of the split: the checker's behaviour can be pinned down
completely without a single API call.
"""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.core.check import check_line
from app.core.interpret import InterpretationOutcome
from app.models import ClauseInterpretation, Decision, Finding, LineItem

SETTINGS = get_settings()


def line(qty=10.0, price=1150.0, total=None, unit="hour", desc="Implementation work"):
    return LineItem(
        line_no=1, description=desc, quantity=qty, unit=unit,
        unit_price=price, line_total=qty * price if total is None else total,
    )


def interp(**kwargs) -> InterpretationOutcome:
    defaults = dict(
        line_no=1, governing_clause_id="C-2024-001-§4.2", covered_by_contract=True,
        contract_rate=1150.0, rate_unit="hour", confidence=0.9,
        reasoning="matched",
    )
    return InterpretationOutcome(
        line_no=1, interpretation=ClauseInterpretation(**{**defaults, **kwargs})
    )


# --- the clean case --------------------------------------------------------

def test_a_correct_line_matches_and_cites_its_clause():
    decision = check_line(line(), interp(), SETTINGS)
    assert decision.status is Decision.MATCH
    assert decision.finding is Finding.CLEAN_MATCH
    assert decision.delta == 0.0
    # A match must still be checkable by a human, so it carries a citation.
    assert decision.cited_clause_id == "C-2024-001-§4.2"


# --- the money arithmetic --------------------------------------------------

def test_wrong_rate_delta_is_the_overcharge_not_the_line_total():
    """`delta` is what the municipality would overpay, which is what gets triaged."""
    decision = check_line(line(qty=10, price=1450.0), interp(contract_rate=1150.0),
                          SETTINGS)
    assert decision.finding is Finding.WRONG_RATE
    assert decision.delta == 3000.0        # (1450 - 1150) * 10
    assert decision.expected_line_total == 11500.0


def test_a_line_total_that_contradicts_its_own_arithmetic_is_caught():
    """Detectable with no contract at all: quantity x price must equal the total."""
    decision = check_line(line(qty=6, price=12500.0, total=82500.0),
                          interp(contract_rate=12500.0), SETTINGS)
    assert decision.finding is Finding.OVERBILLING_WRONG_TOTAL
    assert decision.delta == 7500.0        # 82500 - 75000


def test_quantity_over_the_cap_is_priced_at_the_excess_only():
    """The overage is the unauthorised part, not the whole invoice."""
    decision = check_line(line(qty=180.0, price=1150.0),
                          interp(quantity_cap=160.0, cap_unit="hour",
                                 cap_clause_id="C-2024-001-§5.1"), SETTINGS)
    assert decision.finding is Finding.UNAUTHORIZED_QUANTITY
    assert decision.delta == 23000.0       # 20 hours over x 1150
    assert "C-2024-001-§5.1" in decision.explanation


def test_quantity_exactly_at_the_cap_is_allowed():
    """A ceiling of 160 permits 160. Off-by-one here is a false accusation."""
    decision = check_line(line(qty=160.0), interp(quantity_cap=160.0), SETTINGS)
    assert decision.status is Decision.MATCH


def test_several_problems_on_one_line_are_all_reported():
    """The headline finding is the most severe; the rest stay in the explanation."""
    decision = check_line(
        line(qty=180.0, price=1450.0, total=270000.0),
        interp(contract_rate=1150.0, quantity_cap=160.0), SETTINGS,
    )
    assert decision.finding is Finding.UNAUTHORIZED_QUANTITY   # by PRECEDENCE
    assert "does not equal" in decision.explanation            # the bad total
    assert "1,150.00" in decision.explanation                  # the wrong rate
    assert "ceiling" in decision.explanation                   # the cap


# --- uncertainty is an outcome, not an error -------------------------------

def test_an_excluded_item_flags_the_whole_line():
    """Nothing is payable under a contract that excludes the item, so delta is the total."""
    decision = check_line(
        line(qty=1, price=45000.0, desc="Server hardware"),
        interp(covered_by_contract=False, exclusion_clause_id="C-2024-001-§3.2",
               contract_rate=None, governing_clause_id=None),
        SETTINGS,
    )
    assert decision.finding is Finding.ITEM_NOT_IN_CONTRACT
    assert decision.delta == 45000.0
    assert decision.cited_clause_id == "C-2024-001-§3.2"


def test_no_governing_clause_flags_rather_than_assuming_the_price_is_fine():
    decision = check_line(line(), interp(governing_clause_id=None, contract_rate=None),
                          SETTINGS)
    assert decision.finding is Finding.NO_MATCHING_CLAUSE
    assert decision.status is Decision.FLAG


def test_low_confidence_with_alternatives_routes_to_a_human():
    decision = check_line(
        line(),
        interp(confidence=0.5, alternative_clause_ids=["C-2024-001-§4.1"]),
        SETTINGS,
    )
    assert decision.finding is Finding.AMBIGUOUS_MAPPING
    assert "C-2024-001-§4.1" in decision.explanation


def test_high_confidence_is_not_treated_as_ambiguous():
    """Otherwise every line with a plausible alternative would be flagged."""
    decision = check_line(
        line(),
        interp(confidence=0.95, alternative_clause_ids=["C-2024-001-§4.1"]),
        SETTINGS,
    )
    assert decision.status is Decision.MATCH


def test_a_failed_interpretation_flags_and_never_guesses():
    """The fail-closed path: no interpretation means a human looks, not a default rate."""
    decision = check_line(
        line(),
        InterpretationOutcome(line_no=1, error="could not parse model output"),
        SETTINGS,
    )
    assert decision.finding is Finding.INTERPRETATION_FAILED
    assert decision.status is Decision.FLAG
    assert decision.delta == 11500.0
    assert decision.contract_rate is None


# --- tolerances ------------------------------------------------------------

@pytest.mark.parametrize("price", [1150.0, 1150.005, 1149.995])
def test_floating_point_noise_does_not_create_findings(price):
    """Rounding to ore must not be reported as a discrepancy."""
    assert check_line(line(price=price), interp(), SETTINGS).status is Decision.MATCH


def test_a_real_one_krona_difference_is_still_caught():
    """The tolerance absorbs float error, not actual mispricing."""
    decision = check_line(line(qty=10, price=1151.0), interp(), SETTINGS)
    assert decision.finding is Finding.WRONG_RATE
    assert decision.delta == 10.0


# --- the invariant ---------------------------------------------------------

def test_the_checker_never_returns_an_approval():
    """`MATCH` means 'no discrepancy found', never 'pay this'.

    Encoded as a test because it is the system's central safety claim and the
    kind of thing a later refactor could quietly erode.
    """
    decision = check_line(line(), interp(), SETTINGS)
    assert decision.status in {Decision.MATCH, Decision.FLAG}
    assert not hasattr(decision, "approved")
