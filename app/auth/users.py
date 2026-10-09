"""Simulated user directory. Enterprise SSO is out of scope for the MVP (docs/PRD.md section 5)."""

from __future__ import annotations

from dataclasses import dataclass

ADMIN_ROLE = "admin"
ROLES = ("employee", "finance", "engineering", ADMIN_ROLE)


@dataclass(frozen=True)
class User:
    id: str
    name: str
    roles: tuple[str, ...]

    @property
    def is_admin(self) -> bool:
        return ADMIN_ROLE in self.roles

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "roles": list(self.roles)}


USERS: dict[str, User] = {
    u.id: u
    for u in (
        User("u_employee", "Erin (Employee)", ("employee",)),
        User("u_finance", "Farid (Finance)", ("employee", "finance")),
        User("u_engineer", "Noor (Engineering)", ("employee", "engineering")),
        User("u_admin", "Ada (Admin)", (ADMIN_ROLE,)),
    )
}
