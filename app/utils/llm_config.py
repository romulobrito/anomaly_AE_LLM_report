from pydantic_settings import BaseSettings

class LLMSettings(BaseSettings):
    OPENROUTER_API_KEY: str
    BASE_URL: str = "https://openrouter.ai/api/v1"

    class Config:
        env_file = ".env"

llm_settings = LLMSettings() 