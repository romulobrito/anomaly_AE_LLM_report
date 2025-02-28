from pydantic_settings import BaseSettings
import os

class ModelSettings(BaseSettings):
    OPENROUTER_API_KEY: str
    BASE_URL: str = "https://openrouter.ai/api/v1"
    MODEL_PATH: str = os.path.join("checkpoints", "autoencoder.pt")
    
    class Config:
        env_file = "app/.env.model"
        extra = "allow"

model_settings = ModelSettings() 