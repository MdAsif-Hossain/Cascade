"""Application settings, loaded from the environment.

Every secret reaches the app through this module. Nothing else reads ``os.environ``,
so there is exactly one place to audit when asking "can a key leak from here".

Keys default to empty strings rather than being required, because the test suite
must run offline with no credentials present (CLAUDE.md section 13). A missing key
surfaces as an auth failure from the provider that needs it, at the moment it is
needed, rather than as an import-time crash that takes the whole service down.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # The backend runs from backend/ but .env lives at the repo root, so both
        # locations are searched. Later entries win, so a backend-local .env can
        # override the shared one during debugging.
        env_file=("../.env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    groq_api_key: str = ""
    google_ai_studio_api_key: str = ""
    openrouter_api_key: str = ""

    database_url: str = "sqlite:///./cascade.db"
    environment: str = "development"
    log_level: str = "INFO"

    def key_for(self, provider: str) -> str:
        """Return the configured key for a provider name.

        Centralised so the router can resolve credentials from a tier definition
        without hard-coding which attribute belongs to which provider.
        """
        keys = {
            "groq": self.groq_api_key,
            "gemini": self.google_ai_studio_api_key,
            "openrouter": self.openrouter_api_key,
        }
        if provider not in keys:
            raise KeyError(f"no API key configured for provider {provider!r}")
        return keys[provider]

    def configured_providers(self) -> list[str]:
        """Provider names that actually have a key, in no particular order.

        The router uses this to skip providers that cannot possibly succeed
        instead of spending a request to discover the credential is missing.
        """
        return [p for p in ("groq", "gemini", "openrouter") if self.key_for(p)]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings instance.

    Cached because ``BaseSettings`` re-reads and re-parses the env file on every
    instantiation, and this is called on every request path.
    """
    return Settings()
