"""Document-level access control. This is the single definition of "may this user see this document"."""

from __future__ import annotations

from collections.abc import Iterable

from app.auth.users import ADMIN_ROLE, User


def can_access(user: User, allowed_roles: Iterable[str]) -> bool:
    """Admins see everything; everyone else needs at least one role in the document's ACL."""
    if user.is_admin:
        return True
    return bool(set(user.roles) & set(allowed_roles))


def assignable_roles(user: User, requested: Iterable[str]) -> list[str]:
    """Roles a user may put on a document they upload: admins any role, others only roles they hold."""
    requested = list(dict.fromkeys(requested))
    if user.is_admin:
        return requested
    return [r for r in requested if r in user.roles and r != ADMIN_ROLE]
