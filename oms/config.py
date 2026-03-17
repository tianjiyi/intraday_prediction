"""OMS configuration from environment variables and config.yaml."""

import os
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database — either provide DATABASE_URL directly or individual vars
    database_url_override: str = ""  # Full asyncpg URL (takes priority)
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "kronos"
    postgres_password: str = "kronos_dev"
    oms_postgres_db: str = "oms"

    # Auth
    oms_webhook_secret: str = ""
    oms_api_key: str = ""
    oms_encryption_key: str = ""

    # Server
    host: str = "0.0.0.0"
    port: int = 8100

    # Background tasks
    position_sync_interval: int = 30
    equity_snapshot_interval: int = 300

    # Logging
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    @property
    def database_url(self) -> str:
        # Direct URL override (e.g. DATABASE_URL from Cloud Run)
        if self.database_url_override:
            return self.database_url_override

        # Cloud SQL Unix socket: host starts with /cloudsql/
        # asyncpg uses the directory as host param (it appends .s.PGSQL.5432)
        if self.postgres_host.startswith("/cloudsql/"):
            return (
                f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
                f"@/{self.oms_postgres_db}?host={self.postgres_host}"
            )

        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.oms_postgres_db}"
        )


def load_settings() -> Settings:
    """Load settings from env vars, .env file, and config.yaml."""
    # Try loading .env from project root
    root = Path(__file__).resolve().parent.parent
    env_path = root / ".env"

    settings = Settings(_env_file=str(env_path) if env_path.exists() else None)

    # Override with config.yaml if present
    config_path = Path(__file__).resolve().parent / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}

        server = cfg.get("server", {})
        if "host" in server:
            settings.host = server["host"]
        if "port" in server:
            settings.port = server["port"]

        if "position_sync_interval" in cfg:
            settings.position_sync_interval = cfg["position_sync_interval"]
        if "equity_snapshot_interval" in cfg:
            settings.equity_snapshot_interval = cfg["equity_snapshot_interval"]

        log_cfg = cfg.get("logging", {})
        if "level" in log_cfg:
            settings.log_level = log_cfg["level"]

    return settings
