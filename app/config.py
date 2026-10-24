from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    sample_db_path: str = "./sample.db"

    max_steps: int = 6
    max_total_tokens: int = 20000
    max_llm_retries: int = 3
    price_per_1m_input: float = 0.10
    price_per_1m_output: float = 0.40
    max_cost_usd: float = 0.05


settings = Settings()
