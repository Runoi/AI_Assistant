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
    DAYS_OFFSET: int = 30  # Количество дней для обновления базы знаний
    OPENAI_MARKUP_MODEL: str = "gpt-3.5-turbo-1106"  # Модель для разметки
    MARKUP_TEMPERATURE: float = 0.3  # Детерминированность разметки
    TELEGRAM_CHAT_ID: int = -10012345678  # ID основного чата
    MARKUP_MAX_TOKENS: int = 4096  # Лимит токенов для разметки
    BOT_ID: int = None
    OPENAI_PROXY: str = None  # Прокси для OpenAI, если требуется
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Игнорировать лишние поля