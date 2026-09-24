import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.integrations.source_control.base import SourceControlError

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(SourceControlError)
    async def source_control_exception_handler(
        request: Request, exc: SourceControlError
    ) -> JSONResponse:
        logger.warning(
            "Source-control request failed", extra={"path": request.url.path, "code": exc.code}
        )
        status_code = 503 if exc.code == "github_not_configured" else 502
        return JSONResponse(status_code=status_code, content={"detail": str(exc), "code": exc.code})

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled API exception", extra={"path": request.url.path})
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )
