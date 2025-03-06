from pydantic_settings import BaseSettings
from typing import List, Optional

class Settings(BaseSettings):
    # API
    PROJECT_NAME: str = "Pavimento API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "API para análise de pavimentos usando autoencoders"
    API_V1_STR: str = "/api/v1"
    
    # Segurança
    SECRET_KEY: str = "21d7e058970c86c444fa8d911c5d6785fa879d79f3093d41ef7186c877b35f78"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 dias
    
    # Usuário inicial (admin)
    FIRST_SUPERUSER: str = "admin@example.com"
    FIRST_SUPERUSER_PASSWORD: str = "admin123"
    
    # Banco de dados
    SQLALCHEMY_DATABASE_URI: str = "sqlite:///./sql_app.db"
    
    # CORS
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000"]
    
    # OpenRouter/Deepseek (opcional)
    OPENROUTER_API_KEY: str | None = None
    BASE_URL: str | None = None
    
    class Config:
        env_file = ".env"
        extra = "allow"  # Permite campos extras no .env
        case_sensitive = True

settings = Settings() 