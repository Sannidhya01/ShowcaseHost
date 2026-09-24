from __future__ import annotations

import base64
import hashlib
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from typing import cast

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.db.models import OAuthFlow, Session, User


def token_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def pkce_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


async def create_oauth_flow(
    session: AsyncSession,
    settings: Settings,
    purpose: str,
    *,
    user_id: uuid.UUID | None = None,
    installation_id: int | None = None,
) -> tuple[str, str]:
    state = secrets.token_urlsafe(32)
    verifier = secrets.token_urlsafe(64)
    session.add(
        OAuthFlow(
            state_hash=token_hash(state),
            code_verifier=verifier,
            purpose=purpose,
            user_id=user_id,
            installation_id=installation_id,
            expires_at=datetime.now(UTC) + timedelta(seconds=settings.oauth_flow_ttl_seconds),
        )
    )
    await session.commit()
    return state, verifier


async def consume_oauth_flow(session: AsyncSession, state: str) -> OAuthFlow | None:
    flow = await session.scalar(
        select(OAuthFlow).where(
            OAuthFlow.state_hash == token_hash(state), OAuthFlow.expires_at > datetime.now(UTC)
        )
    )
    if flow is None:
        return None
    await session.delete(flow)
    await session.flush()
    return flow


async def upsert_user(
    session: AsyncSession, github_user_id: int, login: str, avatar_url: str | None
) -> User:
    user = await session.scalar(select(User).where(User.github_user_id == github_user_id))
    if user is None:
        user = User(github_user_id=github_user_id, github_login=login, avatar_url=avatar_url)
        session.add(user)
    else:
        user.github_login = login
        user.avatar_url = avatar_url
    await session.flush()
    return user


async def create_session(
    session: AsyncSession, settings: Settings, user_id: uuid.UUID
) -> tuple[str, datetime]:
    raw_token = secrets.token_urlsafe(48)
    expires_at = datetime.now(UTC) + timedelta(seconds=settings.session_ttl_seconds)
    session.add(Session(user_id=user_id, token_hash=token_hash(raw_token), expires_at=expires_at))
    await session.flush()
    return raw_token, expires_at


async def find_user_for_session(session: AsyncSession, raw_token: str | None) -> User | None:
    if not raw_token:
        return None
    return cast(
        User | None,
        await session.scalar(
            select(User)
            .join(Session, Session.user_id == User.id)
            .where(
                Session.token_hash == token_hash(raw_token),
                Session.expires_at > datetime.now(UTC),
            )
        ),
    )


async def delete_session(session: AsyncSession, raw_token: str | None) -> None:
    if raw_token:
        await session.execute(delete(Session).where(Session.token_hash == token_hash(raw_token)))
        await session.commit()
