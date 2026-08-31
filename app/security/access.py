"""Turning an authenticated caller into a retrieval filter.

This is the only place the application decides what a user may see. Everything
downstream takes a `SearchFilter` and cannot widen it.

Scope note: authentication is out of scope for this build. `Principal` is
populated from a request header, which is fine for a local exercise and would
be replaced by a validated OIDC token in a real deployment -- the group claims
would come from the token, and nothing below this line would change.
"""

from __future__ import annotations

from app.config import Settings, get_settings
from app.models import Principal
from app.store.base import SearchFilter


class AccessDenied(Exception):
    """Raised when a caller has no route to a contract they asked about."""


def filter_for(
    principal: Principal,
    contract_id: str | None = None,
    settings: Settings | None = None,
    limit: int | None = None,
) -> SearchFilter:
    settings = settings or get_settings()
    return SearchFilter(
        contract_id=contract_id,
        allowed_groups=tuple(principal.groups),
        limit=limit if limit is not None else settings.candidate_k,
    )


def assert_can_read(principal: Principal, contract_allowed_groups: list[str]) -> None:
    """Explicit check used when a contract's ACL is already in hand.

    Deny by default: a caller with no groups is denied, and a contract with no
    allowed groups is readable by nobody.
    """
    if not set(principal.groups) & set(contract_allowed_groups):
        raise AccessDenied(
            f"user {principal.user_id!r} is in groups {sorted(principal.groups)}, "
            f"which does not intersect {sorted(contract_allowed_groups)}"
        )
