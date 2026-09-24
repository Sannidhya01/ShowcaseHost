from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx
import jwt

from app.core.config import Settings
from app.integrations.source_control.base import (
    SourceControlError,
    SourceInstallation,
    SourceRepository,
    SourceUser,
)


class GitHubSourceControlProvider:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._client = client

    def _require_oauth(self) -> tuple[str, str]:
        if not self.settings.github_app_client_id or not self.settings.github_app_client_secret:
            raise SourceControlError("github_not_configured", "GitHub App OAuth is not configured")
        return (
            self.settings.github_app_client_id,
            self.settings.github_app_client_secret.get_secret_value(),
        )

    def _private_key(self) -> str:
        if self.settings.github_app_private_key:
            return self.settings.github_app_private_key.get_secret_value().replace("\\n", "\n")
        if self.settings.github_app_private_key_path:
            return self.settings.github_app_private_key_path.read_text(encoding="utf-8")
        raise SourceControlError(
            "github_not_configured", "GitHub App private key is not configured"
        )

    def _app_jwt(self) -> str:
        if self.settings.github_app_id is None:
            raise SourceControlError("github_not_configured", "GitHub App ID is not configured")
        now = int(time.time())
        return jwt.encode(
            {"iat": now - 60, "exp": now + 540, "iss": str(self.settings.github_app_id)},
            self._private_key(),
            algorithm="RS256",
        )

    def authorization_url(self, state: str, code_challenge: str) -> str:
        client_id, _ = self._require_oauth()
        query = urlencode(
            {
                "client_id": client_id,
                "redirect_uri": f"{self.settings.api_public_url}/auth/github/callback",
                "state": state,
                "code_challenge": code_challenge,
                "code_challenge_method": "S256",
            }
        )
        return f"{self.settings.github_web_url}/login/oauth/authorize?{query}"

    def installation_url(self, state: str) -> str:
        if not self.settings.github_app_slug:
            raise SourceControlError("github_not_configured", "GitHub App slug is not configured")
        return (
            f"{self.settings.github_web_url}/apps/{self.settings.github_app_slug}"
            f"/installations/new?{urlencode({'state': state})}"
        )

    def _headers(self, token: str) -> dict[str, str]:
        return {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": self.settings.github_api_version,
            "User-Agent": "ShowcaseHost/0.1",
        }

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        try:
            if self._client:
                response = await self._client.request(method, url, **kwargs)
            else:
                async with httpx.AsyncClient(timeout=30) as client:
                    response = await client.request(method, url, **kwargs)
        except httpx.HTTPError as exc:
            raise SourceControlError(
                "github_network_error", "GitHub request failed", transient=True
            ) from exc
        if response.status_code >= 400:
            transient = response.status_code in {403, 429} or response.status_code >= 500
            code = (
                "github_rate_limited" if response.status_code in {403, 429} else "github_api_error"
            )
            raise SourceControlError(
                code, f"GitHub returned HTTP {response.status_code}", transient=transient
            )
        return response

    async def exchange_code(self, code: str, code_verifier: str) -> str:
        client_id, client_secret = self._require_oauth()
        response = await self._request(
            "POST",
            f"{self.settings.github_web_url}/login/oauth/access_token",
            headers={"Accept": "application/json"},
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "code": code,
                "redirect_uri": f"{self.settings.api_public_url}/auth/github/callback",
                "code_verifier": code_verifier,
            },
        )
        token = response.json().get("access_token")
        if not token:
            raise SourceControlError("github_oauth_failed", "GitHub did not issue an access token")
        return str(token)

    async def get_user(self, user_token: str) -> SourceUser:
        response = await self._request(
            "GET", f"{self.settings.github_api_url}/user", headers=self._headers(user_token)
        )
        data = response.json()
        return SourceUser(
            id=int(data["id"]), login=data["login"], avatar_url=data.get("avatar_url")
        )

    async def _list_installations(self, user_token: str) -> list[dict[str, Any]]:
        return await self._paginate(
            f"{self.settings.github_api_url}/user/installations", user_token, "installations"
        )

    async def get_user_installation(
        self, user_token: str, installation_id: int
    ) -> SourceInstallation:
        installations = await self._list_installations(user_token)
        match = next((item for item in installations if int(item["id"]) == installation_id), None)
        if match is None:
            raise SourceControlError(
                "installation_forbidden", "GitHub installation is not accessible"
            )
        account = match["account"]
        return SourceInstallation(
            id=installation_id,
            account_id=int(account["id"]),
            account_login=account["login"],
            account_type=account["type"],
            repository_selection=match.get("repository_selection", "selected"),
        )

    async def _paginate(self, url: str, token: str, key: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        page = 1
        while True:
            response = await self._request(
                "GET", url, headers=self._headers(token), params={"per_page": 100, "page": page}
            )
            batch = response.json().get(key, [])
            items.extend(batch)
            if len(batch) < 100:
                return items
            page += 1

    async def list_user_repositories(
        self, user_token: str, installation_id: int
    ) -> list[SourceRepository]:
        items = await self._paginate(
            f"{self.settings.github_api_url}/user/installations/{installation_id}/repositories",
            user_token,
            "repositories",
        )
        return [self._repository(item, installation_id) for item in items]

    def _repository(self, data: dict[str, Any], installation_id: int) -> SourceRepository:
        return SourceRepository(
            id=int(data["id"]),
            installation_id=installation_id,
            owner=data["owner"]["login"],
            name=data["name"],
            full_name=data["full_name"],
            private=bool(data["private"]),
            default_branch=data["default_branch"],
        )

    async def create_installation_token(self, installation_id: int, repository_id: int) -> str:
        response = await self._request(
            "POST",
            f"{self.settings.github_api_url}/app/installations/{installation_id}/access_tokens",
            headers=self._headers(self._app_jwt()),
            json={"repository_ids": [repository_id], "permissions": {"contents": "read"}},
        )
        return str(response.json()["token"])

    async def get_repository(self, token: str, repository_id: int) -> SourceRepository:
        response = await self._request(
            "GET",
            f"{self.settings.github_api_url}/repositories/{repository_id}",
            headers=self._headers(token),
        )
        data = response.json()
        return self._repository(data, 0)

    async def resolve_default_branch_sha(self, token: str, repository: SourceRepository) -> str:
        response = await self._request(
            "GET",
            f"{self.settings.github_api_url}/repos/{repository.full_name}/commits/{repository.default_branch}",
            headers=self._headers(token),
        )
        sha = str(response.json()["sha"])
        if len(sha) != 40 or any(character not in "0123456789abcdef" for character in sha.lower()):
            raise SourceControlError("invalid_commit_sha", "GitHub returned an invalid commit SHA")
        return sha

    async def download_archive(
        self,
        token: str,
        repository: SourceRepository,
        commit_sha: str,
        destination: Path,
        max_bytes: int,
    ) -> int:
        response = await self._request(
            "GET",
            f"{self.settings.github_api_url}/repos/{repository.full_name}/tarball/{commit_sha}",
            headers=self._headers(token),
            follow_redirects=False,
        )
        location = response.headers.get("location")
        if response.status_code not in {301, 302, 303, 307, 308} or not location:
            raise SourceControlError(
                "archive_redirect_missing", "GitHub archive URL was not returned"
            )
        size = 0
        try:
            client = self._client or httpx.AsyncClient(timeout=httpx.Timeout(120, read=120))
            async with client.stream(
                "GET", location, headers={"User-Agent": "ShowcaseHost/0.1"}
            ) as stream:
                if stream.status_code >= 400:
                    raise SourceControlError(
                        "archive_download_failed",
                        f"GitHub archive download returned HTTP {stream.status_code}",
                        transient=stream.status_code >= 500,
                    )
                with destination.open("wb") as handle:
                    async for chunk in stream.aiter_bytes():
                        size += len(chunk)
                        if size > max_bytes:
                            raise SourceControlError(
                                "archive_too_large", "Repository archive exceeds limit"
                            )
                        handle.write(chunk)
            if self._client is None:
                await client.aclose()
        except httpx.HTTPError as exc:
            raise SourceControlError(
                "archive_download_failed", "Archive download failed", transient=True
            ) from exc
        return size

    async def revoke_installation_token(self, token: str) -> None:
        try:
            await self._request(
                "DELETE",
                f"{self.settings.github_api_url}/installation/token",
                headers=self._headers(token),
            )
        except SourceControlError:
            # Tokens are short-lived; revocation failure must not mask ingestion outcome.
            return
