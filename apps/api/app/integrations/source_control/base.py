from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class SourceUser:
    id: int
    login: str
    avatar_url: str | None


@dataclass(frozen=True)
class SourceInstallation:
    id: int
    account_id: int
    account_login: str
    account_type: str
    repository_selection: str


@dataclass(frozen=True)
class SourceRepository:
    id: int
    installation_id: int
    owner: str
    name: str
    full_name: str
    private: bool
    default_branch: str


class SourceControlError(Exception):
    def __init__(self, code: str, message: str, *, transient: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.transient = transient


class SourceControlProvider(Protocol):
    def authorization_url(self, state: str, code_challenge: str) -> str: ...

    async def exchange_code(self, code: str, code_verifier: str) -> str: ...

    async def get_user(self, user_token: str) -> SourceUser: ...

    async def get_user_installation(
        self, user_token: str, installation_id: int
    ) -> SourceInstallation: ...

    async def list_user_repositories(
        self, user_token: str, installation_id: int
    ) -> list[SourceRepository]: ...

    async def create_installation_token(self, installation_id: int, repository_id: int) -> str: ...

    async def get_repository(self, token: str, repository_id: int) -> SourceRepository: ...

    async def resolve_default_branch_sha(self, token: str, repository: SourceRepository) -> str: ...

    async def download_archive(
        self,
        token: str,
        repository: SourceRepository,
        commit_sha: str,
        destination: Path,
        max_bytes: int,
    ) -> int: ...

    async def revoke_installation_token(self, token: str) -> None: ...
