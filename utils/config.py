from pydantic_settings import BaseSettings
from typing import List, Optional

class Config(BaseSettings):
    # Обязательные поля (переименованы в соответствии с .env)
    TELEGRAM_BOT_TOKEN: str  # было TELEGRAM_TOKEN
    OPENAI_API_KEY: str      # было OPENAI_KEY
    
    # Опциональные поля
    API_ID: Optional[str] = None
    API_HASH: Optional[str] = None
    SESSION_STRING: Optional[str] = None
    CHAT_LIMIT: int = 50
    GOOGLE_SHEET_URL: str = ""
    ADMINS: List[int] = []  # Добавьте ID админов в .env если нужно
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Игнорировать лишние поля