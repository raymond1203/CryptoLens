from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # LLM
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # Vector DB
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # Redis
    redis_url: str = "redis://localhost:6379"

    # On-chain
    alchemy_api_key: str = ""
    coingecko_api_url: str = "https://api.coingecko.com/api/v3"

    # Korean Sources
    upbit_api_key: str = ""
    bithumb_api_key: str = ""
    telegram_api_id: str = ""
    telegram_api_hash: str = ""

    # Legal
    law_oc: str = ""

    # Telegram Bot
    telegram_bot_token: str = ""


settings = Settings()
