from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies.auth import get_current_user
from app.db.models import User
from app.db.session import get_db_session
from app.integrations.source_control.base import SourceControlError
from app.services.auth import (
    consume_oauth_flow,
    create_oauth_flow,
    create_session,
    delete_session,
    pkce_challenge,
    upsert_user,
)
from app.services.repositories import synchronize_installation

router = APIRouter(tags=["authentication"])


class CurrentUserResponse(BaseModel):
    id: uuid.UUID
    github_login: str
    avatar_url: str | None


@router.get("/auth/github/start")
async def github_auth_start(
    request: Request, session: AsyncSession = Depends(get_db_session)
) -> RedirectResponse:
    state, verifier = await create_oauth_flow(session, request.app.state.settings, "login")
    url = request.app.state.source_control.authorization_url(state, pkce_challenge(verifier))
    return RedirectResponse(url)


@router.get("/auth/github/callback")
async def github_auth_callback(
    request: Request,
    code: str = Query(min_length=1),
    state_value: str = Query(alias="state", min_length=1),
    session: AsyncSession = Depends(get_db_session),
) -> RedirectResponse:
    flow = await consume_oauth_flow(session, state_value)
    if flow is None:
        raise HTTPException(status_code=400, detail="OAuth state is invalid or expired")
    provider = request.app.state.source_control
    try:
        user_token = await provider.exchange_code(code, flow.code_verifier)
        source_user = await provider.get_user(user_token)
        if flow.purpose == "install_verify":
            if flow.user_id is None or flow.installation_id is None:
                raise HTTPException(status_code=400, detail="Installation flow is invalid")
            stored_user = await session.get(User, flow.user_id)
            if stored_user is None or stored_user.github_user_id != source_user.id:
                raise HTTPException(status_code=403, detail="GitHub account does not match session")
            installation = await provider.get_user_installation(user_token, flow.installation_id)
            repositories = await provider.list_user_repositories(user_token, flow.installation_id)
            await synchronize_installation(session, stored_user.id, installation, repositories)
            return RedirectResponse(
                f"{request.app.state.settings.web_public_url}/?github=installed"
            )
        user = await upsert_user(session, source_user.id, source_user.login, source_user.avatar_url)
        raw_session, expires_at = await create_session(session, request.app.state.settings, user.id)
        await session.commit()
    except SourceControlError as exc:
        await session.rollback()
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    response = RedirectResponse(f"{request.app.state.settings.web_public_url}/?github=connected")
    response.set_cookie(
        request.app.state.settings.session_cookie_name,
        raw_session,
        expires=int(expires_at.timestamp()),
        httponly=True,
        secure=request.app.state.settings.app_env == "production",
        samesite="lax",
        path="/",
    )
    return response


@router.get("/auth/me", response_model=CurrentUserResponse)
async def me(user: User = Depends(get_current_user)) -> CurrentUserResponse:
    return CurrentUserResponse(
        id=user.id, github_login=user.github_login, avatar_url=user.avatar_url
    )


@router.post("/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    request: Request,
    response: Response,
    session: AsyncSession = Depends(get_db_session),
) -> None:
    await delete_session(
        session, request.cookies.get(request.app.state.settings.session_cookie_name)
    )
    response.delete_cookie(request.app.state.settings.session_cookie_name, path="/")


@router.get("/github/install/start")
async def github_install_start(
    request: Request,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> RedirectResponse:
    state, _ = await create_oauth_flow(
        session, request.app.state.settings, "install_start", user_id=user.id
    )
    return RedirectResponse(request.app.state.source_control.installation_url(state))


@router.get("/github/install/setup")
async def github_install_setup(
    request: Request,
    installation_id: int,
    state_value: str = Query(alias="state", min_length=1),
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db_session),
) -> RedirectResponse:
    initial_flow = await consume_oauth_flow(session, state_value)
    if (
        initial_flow is None
        or initial_flow.purpose != "install_start"
        or initial_flow.user_id != user.id
    ):
        raise HTTPException(status_code=400, detail="Installation state is invalid or expired")
    state, verifier = await create_oauth_flow(
        session,
        request.app.state.settings,
        "install_verify",
        user_id=user.id,
        installation_id=installation_id,
    )
    return RedirectResponse(
        request.app.state.source_control.authorization_url(state, pkce_challenge(verifier))
    )
