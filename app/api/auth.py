from __future__ import annotations

from fastapi import APIRouter, Depends

from app.auth.jwt import get_current_user, issue_token
from app.auth.users import USERS, User
from app.errors import NotFoundError
from app.schemas import TokenRequest, TokenResponse, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/users", response_model=list[UserOut], summary="List the simulated demo users")
def list_users() -> list[dict]:
    return [user.to_dict() for user in USERS.values()]


@router.post("/token", response_model=TokenResponse, summary="Issue a JWT for a simulated user")
def create_token(body: TokenRequest) -> TokenResponse:
    """Simulated login (no password): the MVP stands in for SSO with a fixed user directory."""
    user = USERS.get(body.user_id)
    if user is None:
        raise NotFoundError(f"Unknown user '{body.user_id}'", details={"known_users": list(USERS)})
    token, ttl = issue_token(user)
    return TokenResponse(access_token=token, expires_in=ttl, user=UserOut(**user.to_dict()))


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)) -> dict:
    return user.to_dict()
