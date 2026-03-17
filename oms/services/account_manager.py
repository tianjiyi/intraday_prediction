"""Multi-account credential management with Fernet encryption."""

import logging
from typing import Optional
from uuid import UUID

from cryptography.fernet import Fernet
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from oms.db.models import Account

logger = logging.getLogger(__name__)


class AccountManager:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession], encryption_key: str):
        self._session_factory = session_factory
        if encryption_key:
            self._fernet = Fernet(encryption_key.encode() if isinstance(encryption_key, str) else encryption_key)
        else:
            self._fernet = None
            logger.warning("OMS: No encryption key set — account credentials will not be encrypted")

    def _encrypt(self, value: str) -> str:
        if self._fernet:
            return self._fernet.encrypt(value.encode()).decode()
        return value

    def _decrypt(self, value: str) -> str:
        if self._fernet:
            return self._fernet.decrypt(value.encode()).decode()
        return value

    async def create_account(
        self,
        name: str,
        api_key: str,
        secret_key: str,
        base_url: str = "https://paper-api.alpaca.markets",
        is_default: bool = False,
        allowed_sources: list[str] | None = None,
        risk_limits: dict | None = None,
    ) -> Account:
        async with self._session_factory() as session:
            # If setting as default, clear other defaults
            if is_default:
                await session.execute(
                    update(Account).where(Account.is_default == True).values(is_default=False)
                )

            account = Account(
                name=name,
                api_key_encrypted=self._encrypt(api_key),
                secret_key_encrypted=self._encrypt(secret_key),
                base_url=base_url,
                is_default=is_default,
                allowed_sources=allowed_sources or [],
                risk_limits=risk_limits or {},
            )
            session.add(account)
            await session.commit()
            await session.refresh(account)
            logger.info(f"OMS: Created account '{name}' (id={account.id})")
            return account

    async def get_account_by_name(self, name: str) -> Optional[Account]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(Account).where(Account.name == name, Account.is_active == True)
            )
            return result.scalar_one_or_none()

    async def get_default_account(self) -> Optional[Account]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(Account).where(Account.is_default == True, Account.is_active == True)
            )
            account = result.scalar_one_or_none()
            if account:
                return account
            # Fallback: first active account
            result = await session.execute(
                select(Account).where(Account.is_active == True).limit(1)
            )
            return result.scalar_one_or_none()

    async def resolve_account(self, account_name: str | None) -> Optional[Account]:
        """Resolve account by name, or return default."""
        if account_name:
            return await self.get_account_by_name(account_name)
        return await self.get_default_account()

    async def get_all_accounts(self) -> list[Account]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(Account).where(Account.is_active == True).order_by(Account.name)
            )
            return list(result.scalars().all())

    async def update_account(self, account_id: UUID, **kwargs) -> Optional[Account]:
        async with self._session_factory() as session:
            # Handle is_default: clear others first
            if kwargs.get("is_default"):
                await session.execute(
                    update(Account).where(Account.is_default == True).values(is_default=False)
                )

            await session.execute(
                update(Account).where(Account.id == account_id).values(**kwargs)
            )
            await session.commit()

            result = await session.execute(select(Account).where(Account.id == account_id))
            return result.scalar_one_or_none()

    def get_credentials(self, account: Account) -> tuple[str, str]:
        """Decrypt and return (api_key, secret_key)."""
        return (
            self._decrypt(account.api_key_encrypted),
            self._decrypt(account.secret_key_encrypted),
        )
