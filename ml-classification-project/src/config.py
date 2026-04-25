"""
Configuration Management
========================

This module handles all application configuration using environment variables
with sensible defaults.

STUDY NOTE: Configuration Management
------------------------------------
Why use environment variables?
1. Security: Secrets don't end up in code/git
2. Flexibility: Different configs for dev/staging/prod
3. 12-Factor App: Industry standard for cloud apps
4. Easy to change without code changes

Best Practices:
- Never hardcode secrets
- Always provide sensible defaults for non-secrets
- Validate configuration at startup
- Use typed configuration classes

Usage:
    from src.config import settings

    print(settings.API_PORT)  # 8000
    print(settings.MODEL_PATH)  # models/iris_classifier.joblib
"""

import os
from pathlib import Path
from typing import List, Optional
from functools import lru_cache


class Settings:
    """
    Application settings loaded from environment variables.

    STUDY NOTE:
    -----------
    This class follows the "Settings" pattern:
    - All config in one place
    - Type hints for documentation
    - Defaults for development
    - Validation on access
    """

    def __init__(self):
        # Base paths
        self.BASE_DIR = Path(__file__).parent.parent
        self.SRC_DIR = self.BASE_DIR / "src"

        # ---------------------------------------------------------------------
        # Application Settings
        # ---------------------------------------------------------------------
        self.APP_NAME: str = os.getenv("APP_NAME", "iris-classifier")
        self.APP_ENV: str = os.getenv("APP_ENV", "development")
        self.DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

        # ---------------------------------------------------------------------
        # API Settings
        # ---------------------------------------------------------------------
        self.API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
        self.API_PORT: int = int(os.getenv("API_PORT", "8000"))
        self.API_WORKERS: int = int(os.getenv("API_WORKERS", "1"))

        # CORS
        cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:8501")
        self.CORS_ORIGINS: List[str] = [
            origin.strip() for origin in cors_origins.split(",")
        ]

        # ---------------------------------------------------------------------
        # Model Settings
        # ---------------------------------------------------------------------
        self.MODEL_PATH: Path = self.BASE_DIR / os.getenv(
            "MODEL_PATH", "models/iris_classifier.joblib"
        )
        self.MODEL_VERSION: str = os.getenv("MODEL_VERSION", "1.0.0")

        # ---------------------------------------------------------------------
        # ML Training Settings
        # ---------------------------------------------------------------------
        self.RANDOM_STATE: int = int(os.getenv("RANDOM_STATE", "42"))
        self.TEST_SIZE: float = float(os.getenv("TEST_SIZE", "0.2"))
        self.CV_FOLDS: int = int(os.getenv("CV_FOLDS", "5"))

        # ---------------------------------------------------------------------
        # Security Settings
        # ---------------------------------------------------------------------
        self.API_KEY: Optional[str] = os.getenv("API_KEY")
        self.SECRET_KEY: Optional[str] = os.getenv("SECRET_KEY")

        # ---------------------------------------------------------------------
        # Monitoring Settings
        # ---------------------------------------------------------------------
        self.ENABLE_METRICS: bool = os.getenv("ENABLE_METRICS", "true").lower() == "true"
        self.METRICS_PORT: int = int(os.getenv("METRICS_PORT", "9090"))

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.APP_ENV.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.APP_ENV.lower() == "development"

    def validate(self) -> List[str]:
        """
        Validate configuration and return list of errors.

        STUDY NOTE:
        -----------
        Validation at startup catches misconfigurations early.
        Better to fail fast than have runtime errors later.
        """
        errors = []

        # Check required settings for production
        if self.is_production:
            if not self.API_KEY:
                errors.append("API_KEY is required in production")
            if not self.SECRET_KEY:
                errors.append("SECRET_KEY is required in production")
            if "*" in self.CORS_ORIGINS:
                errors.append("CORS_ORIGINS cannot be '*' in production")

        # Validate ranges
        if not 0 < self.TEST_SIZE < 1:
            errors.append(f"TEST_SIZE must be between 0 and 1, got {self.TEST_SIZE}")

        if self.CV_FOLDS < 2:
            errors.append(f"CV_FOLDS must be at least 2, got {self.CV_FOLDS}")

        if self.API_PORT < 1 or self.API_PORT > 65535:
            errors.append(f"API_PORT must be between 1-65535, got {self.API_PORT}")

        return errors

    def __repr__(self) -> str:
        """String representation (hides secrets)."""
        return (
            f"Settings("
            f"APP_NAME={self.APP_NAME}, "
            f"APP_ENV={self.APP_ENV}, "
            f"API_PORT={self.API_PORT}, "
            f"MODEL_PATH={self.MODEL_PATH}"
            f")"
        )


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    STUDY NOTE:
    -----------
    @lru_cache ensures settings are only loaded once.
    This is a form of the Singleton pattern.
    """
    return Settings()


# Global settings instance for easy import
settings = get_settings()


# =============================================================================
# Configuration validation at import time (optional)
# =============================================================================

def validate_settings_on_startup():
    """Validate settings and print warnings/errors."""
    errors = settings.validate()

    if errors:
        print("\n⚠️  Configuration Errors:")
        for error in errors:
            print(f"   - {error}")
        print()

        if settings.is_production:
            raise ValueError("Invalid configuration for production environment")


# Uncomment to validate on import:
# validate_settings_on_startup()


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Current Configuration")
    print("=" * 50)

    print(f"\nEnvironment: {settings.APP_ENV}")
    print(f"Debug Mode: {settings.DEBUG}")
    print(f"\nAPI Settings:")
    print(f"  Host: {settings.API_HOST}")
    print(f"  Port: {settings.API_PORT}")
    print(f"  CORS Origins: {settings.CORS_ORIGINS}")
    print(f"\nModel Settings:")
    print(f"  Path: {settings.MODEL_PATH}")
    print(f"  Version: {settings.MODEL_VERSION}")
    print(f"\nML Settings:")
    print(f"  Random State: {settings.RANDOM_STATE}")
    print(f"  Test Size: {settings.TEST_SIZE}")
    print(f"  CV Folds: {settings.CV_FOLDS}")

    print("\nValidation:")
    errors = settings.validate()
    if errors:
        for error in errors:
            print(f"  ❌ {error}")
    else:
        print("  ✅ All settings valid")

    print("=" * 50 + "\n")
