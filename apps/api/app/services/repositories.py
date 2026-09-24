from __future__ import annotations

import uuid
from typing import cast

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import GitHubInstallation, Repository, UserInstallation, UserRepository
from app.integrations.source_control.base import SourceInstallation, SourceRepository


async def synchronize_installation(
    session: AsyncSession,
    user_id: uuid.UUID,
    installation: SourceInstallation,
    repositories: list[SourceRepository],
) -> None:
    stored_installation = await session.get(GitHubInstallation, installation.id)
    if stored_installation is None:
        stored_installation = GitHubInstallation(
            id=installation.id,
            account_id=installation.account_id,
            account_login=installation.account_login,
            account_type=installation.account_type,
            repository_selection=installation.repository_selection,
        )
        session.add(stored_installation)
    else:
        stored_installation.account_id = installation.account_id
        stored_installation.account_login = installation.account_login
        stored_installation.account_type = installation.account_type
        stored_installation.repository_selection = installation.repository_selection
    installation_association = await session.get(UserInstallation, (user_id, installation.id))
    if installation_association is None:
        session.add(UserInstallation(user_id=user_id, installation_id=installation.id))
    accessible_repository_ids: set[uuid.UUID] = set()
    for source in repositories:
        repository = await session.scalar(
            select(Repository).where(Repository.github_repository_id == source.id)
        )
        if repository is None:
            repository = Repository(
                github_repository_id=source.id,
                installation_id=installation.id,
                owner=source.owner,
                name=source.name,
                full_name=source.full_name,
                private=source.private,
                default_branch=source.default_branch,
            )
            session.add(repository)
        else:
            if repository.installation_id != installation.id:
                await session.execute(
                    delete(UserRepository).where(UserRepository.repository_id == repository.id)
                )
            repository.installation_id = installation.id
            repository.owner = source.owner
            repository.name = source.name
            repository.full_name = source.full_name
            repository.private = source.private
            repository.default_branch = source.default_branch
        await session.flush()
        accessible_repository_ids.add(repository.id)
        repository_association = await session.get(UserRepository, (user_id, repository.id))
        if repository_association is None:
            session.add(UserRepository(user_id=user_id, repository_id=repository.id))
    existing_access = list(
        await session.scalars(
            select(UserRepository)
            .join(Repository, Repository.id == UserRepository.repository_id)
            .where(
                UserRepository.user_id == user_id,
                Repository.installation_id == installation.id,
            )
        )
    )
    for association in existing_access:
        if association.repository_id not in accessible_repository_ids:
            await session.delete(association)
    await session.commit()


async def list_user_repositories(session: AsyncSession, user_id: uuid.UUID) -> list[Repository]:
    result = await session.scalars(
        select(Repository)
        .join(UserRepository, UserRepository.repository_id == Repository.id)
        .where(UserRepository.user_id == user_id)
        .order_by(Repository.full_name)
    )
    return list(result)


async def get_user_repository(
    session: AsyncSession, user_id: uuid.UUID, repository_id: uuid.UUID
) -> Repository | None:
    return cast(
        Repository | None,
        await session.scalar(
            select(Repository)
            .join(UserRepository, UserRepository.repository_id == Repository.id)
            .where(Repository.id == repository_id, UserRepository.user_id == user_id)
        ),
    )
