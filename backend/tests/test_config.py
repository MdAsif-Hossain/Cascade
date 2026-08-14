"""Settings loading and credential resolution."""

from __future__ import annotations

import pytest

from app.core.config import Settings


def _settings(**overrides: str) -> Settings:
    """Build settings without reading any .env file, so tests never see real keys."""
    defaults = {
        "groq_api_key": "",
        "google_ai_studio_api_key": "",
        "openrouter_api_key": "",
    }
    return Settings(_env_file=None, **{**defaults, **overrides})  # type: ignore[arg-type]


class TestDefaults:
    def test_missing_keys_do_not_raise_at_construction(self):
        """CI runs with no credentials; a hard requirement here would break it."""
        assert _settings().groq_api_key == ""

    def test_database_defaults_to_local_sqlite(self):
        assert _settings().database_url.startswith("sqlite")


class TestCredentialResolution:
    def test_each_provider_resolves_to_its_own_key(self):
        settings = _settings(groq_api_key="g", google_ai_studio_api_key="m", openrouter_api_key="o")
        assert settings.key_for("groq") == "g"
        assert settings.key_for("gemini") == "m"
        assert settings.key_for("openrouter") == "o"

    def test_unknown_provider_is_rejected_rather_than_returning_blank(self):
        """A silent empty key would surface much later as a confusing auth failure."""
        with pytest.raises(KeyError):
            _settings().key_for("cerebras")

    def test_configured_providers_lists_only_those_with_keys(self):
        settings = _settings(groq_api_key="g", openrouter_api_key="o")
        assert set(settings.configured_providers()) == {"groq", "openrouter"}

    def test_no_keys_means_no_configured_providers(self):
        assert _settings().configured_providers() == []
