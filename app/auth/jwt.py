"""JWT issuing/verification and the FastAPI auth dependencies."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt
from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.auth.users import USERS, User
from app.config import get_settings
from app.errors import ForbiddenError, UnauthorizedError

_bearer = HTTPBearer(auto_error=False)


def issue_token(user: User) -> tuple[str, int]:
    settings = get_settings()
    ttl = settings.jwt_ttl_minutes * 60
    now = datetime.now(UTC)
    claims = {
        "sub": user.id,
        "name": user.name,
        "roles": list(user.roles),
        "iat": now,
        "exp": now + timedelta(seconds=ttl),
    }
    return jwt.encode(claims, settings.jwt_secret, algorithm=settings.jwt_algorithm), ttl


def decode_token(token: str) -> User:
    settings = get_settings()
    try:
        claims = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
            options={"require": ["sub", "exp"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise UnauthorizedError("Token has expired") from exc
    except jwt.InvalidTokenError as exc:
        raise UnauthorizedError("Invalid token") from exc
    # The directory is the source of truth for roles: a token for a removed user, or one
    # carrying roles the user no longer holds, is not honoured.
    user = USERS.get(claims["sub"])
    if user is None:
        raise UnauthorizedError("Unknown user")
    return user


def get_current_user(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> User:
    if credentials is None:
        raise UnauthorizedError("Missing bearer token", headers={"WWW-Authenticate": "Bearer"})
    return decode_token(credentials.credentials)


def require_roles(*roles: str):
    """Dependency factory: user must hold one of `roles` (admins always pass)."""

    def dependency(user: User = Depends(get_current_user)) -> User:
        if not user.is_admin and not set(user.roles) & set(roles):
            raise ForbiddenError(f"Requires one of the roles: {', '.join(roles) or 'admin'}")
        return user

    return dependency
