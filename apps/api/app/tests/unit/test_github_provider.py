from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from pydantic import SecretStr

from app.core.config import Settings
from app.integrations.source_control.base import SourceControlError
from app.integrations.source_control.github import GitHubSourceControlProvider


def test_authorization_url_uses_state_and_pkce() -> None:
    provider = GitHubSourceControlProvider(
        Settings(
            github_app_client_id="client",
            github_app_client_secret=SecretStr("secret"),
            api_public_url="https://api.example.test",
        )
    )

    query = parse_qs(urlparse(provider.authorization_url("state", "challenge")).query)

    assert query["state"] == ["state"]
    assert query["code_challenge"] == ["challenge"]
    assert query["code_challenge_method"] == ["S256"]
    assert query["redirect_uri"] == ["https://api.example.test/auth/github/callback"]


async def test_installation_must_be_accessible_to_user() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"installations": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = GitHubSourceControlProvider(Settings(), client)
        with pytest.raises(SourceControlError) as error:
            await provider.get_user_installation("temporary-user-token", 999)

    assert error.value.code == "installation_forbidden"


async def test_repository_pagination_and_mapping() -> None:
    requests = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        page = int(request.url.params["page"])
        count = 100 if page == 1 else 1
        repositories = [
            {
                "id": page * 1000 + index,
                "owner": {"login": "owner"},
                "name": f"repo-{index}",
                "full_name": f"owner/repo-{index}",
                "private": False,
                "default_branch": "main",
            }
            for index in range(count)
        ]
        return httpx.Response(200, json={"repositories": repositories})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = GitHubSourceControlProvider(Settings(), client)
        repositories = await provider.list_user_repositories("temporary-user-token", 42)

    assert requests == 2
    assert len(repositories) == 101
    assert repositories[0].installation_id == 42
