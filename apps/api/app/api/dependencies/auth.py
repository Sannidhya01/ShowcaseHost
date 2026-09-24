from fastapi import Cookie, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import User
from app.db.session import get_db_session
from app.services.auth import find_user_for_session


async def get_current_user(
    request: Request,
    session: AsyncSession = Depends(get_db_session),
    session_cookie: str | None = Cookie(default=None, alias="showcasehost_session"),
) -> User:
    cookie_name = request.app.state.settings.session_cookie_name
    raw_token = (
        request.cookies.get(cookie_name)
        if cookie_name != "showcasehost_session"
        else session_cookie
    )
    user = await find_user_for_session(session, raw_token)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required"
        )
    return user
