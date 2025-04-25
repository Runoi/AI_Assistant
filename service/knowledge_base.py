import os
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Set
from pathlib import Path
import logging
from pyrogram.types import Message, User
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from utils.config import Config

# Настройка логгирования
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class KnowledgeBase:
    def __init__(self, config: Config):
        """
        Централизованный класс для управления базой знаний
        
        Args:
            config (Config): Конфигурация приложения
        """
        self.config = config
        self.bot = None  # Будет установлен при вызове update_all_sources
        self._chat_id = None
        self._progress_message = None
        self._progress_message_id = None
        
        # Инициализация векторного хранилища
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=config.OPENAI_API_KEY,
            chunk_size=500
        )
        
        chroma_dir = Path("chroma_data").absolute()
        chroma_dir.mkdir(parents=True, exist_ok=True)
        
        self.vectorstore = Chroma(
            persist_directory=str(chroma_dir),
            embedding_function=self.embeddings,
            collection_metadata={"hnsw:space": "cosine"}
        )
        
        # Ленивая инициализация парсеров
        self._sheets_parser = None
        self._telegram_parser = None
        
        logger.info("База знаний инициализирована")

    @property
    def sheets_parser(self):
        """Ленивая инициализация парсера Google Sheets"""
        if self._sheets_parser is None:
            from service.google_sheet_utils import GoogleSheetsParser
            self._sheets_parser = GoogleSheetsParser(self.vectorstore)
            self._sheets_parser.knowledge_base = self
            logger.info("Инициализирован парсер Google Sheets")
        return self._sheets_parser

    @property
    def telegram_parser(self):
        """Ленивая инициализация парсера Telegram"""
        if self._telegram_parser is None:
            from service.chat_parser import TelegramParser
            self._telegram_parser = TelegramParser(self.vectorstore)
            self._telegram_parser.knowledge_base = self
            logger.info("Инициализирован парсер Telegram")
        return self._telegram_parser

    async def _update_progress(self, text: str):
        """
        Обновляет сообщение о прогрессе с автоматическим созданием нового при необходимости
        
        Args:
            text: Текст сообщения (автоматически обрезается до 4096 символов)
        """
        if not self.bot or not self._chat_id:
            return

        text = text[:4096]  # Ограничение длины для Telegram
        
        try:
            # Если сообщение уже существует, пробуем его отредактировать
            if self._progress_message:
                try:
                    await self._progress_message.edit_text(text)
                    return
                except Exception as edit_error:
                    logger.warning(f"Не удалось отредактировать сообщение: {str(edit_error)}")
            
            # Создаем новое сообщение
            self._progress_message = await self.bot.send_message(
                chat_id=self._chat_id,
                text=text
            )
            
        except Exception as e:
            logger.error(f"Ошибка обновления прогресса: {str(e)}", exc_info=True)

    async def search(
        self,
        query: str,
        k: int = 3,
        filters: Optional[Dict] = None
    ) -> List[Document]:
        """
        Поиск в векторной базе данных
        
        Args:
            query: Текст запроса
            k: Количество возвращаемых документов
            filters: Фильтры по метаданным
            
        Returns:
            Список релевантных документов
        """
        try:
            results = await self.vectorstore.asimilarity_search(
                query,
                k=k,
                filter=filters
            )
            logger.info(f"Найдено {len(results)} документов по запросу: '{query}'")
            return results
        except Exception as e:
            logger.error(f"Ошибка поиска: {str(e)}", exc_info=True)
            return []

    async def update_all_sources(
        self,
        bot,
        chat_id: int,
        telegram_days_offset: int = 30
    ) -> Dict[str, int]:
        """
        Полное обновление базы знаний с автоматическим отображением прогресса
        
        Args:
            bot: Экземпляр бота для отправки сообщений
            chat_id: ID чата для отправки сообщений прогресса
            telegram_days_offset: Парсинг последних N дней для Telegram
            
        Returns:
            Словарь с количеством добавленных документов по источникам
        """
        self.bot = bot
        self._chat_id = chat_id
        results = {"google_sheets": 0, "telegram": 0}
        
        try:
            # Этап 1: Начало обновления
            await self._update_progress("🔄 Начинаю обновление базы знаний...")
            
            # Этап 2: Обновление Google Sheets
            await self._update_progress("📊 Загружаю данные из Google Sheets...")
            results["google_sheets"] = await self._update_from_sheets()
            
            # Этап 3: Обновление Telegram
            status = f"📨 Загружаю данные из Telegram (последние {telegram_days_offset} дней)..."
            await self._update_progress(status)
            results["telegram"] = await self._update_from_telegram(telegram_days_offset)
            
            # Итоговый отчет
            total = sum(results.values())
            report = (
                "✅ Обновление завершено!\n"
                f"• Google Sheets: {results['google_sheets']}\n"
                f"• Telegram: {results['telegram']}\n"
                f"• Всего: {total}"
            )
            await self._update_progress(report)
            
            return results
            
        except Exception as e:
            error_msg = f"⚠️ Ошибка: {str(e)[:400]}"
            await self._update_progress(error_msg)
            logger.error(f"Ошибка обновления: {str(e)}", exc_info=True)
            return results
        finally:
            self._cleanup()

    async def _update_from_sheets(self) -> int:
        """Обновление данных из Google Sheets"""
        try:
            documents = await self.sheets_parser.parse_sheet(
                self.config.GOOGLE_SHEET_URL
            )
            
            if documents:
                ids = await self.vectorstore.aadd_documents(documents)
                await self._update_progress(f"✅ Google Sheets: {len(ids)} документов")
                return len(ids)
                
            await self._update_progress("ℹ️ Нет новых данных в Google Sheets")
            return 0
        except Exception as e:
            error_msg = f"⚠️ Ошибка Google Sheets: {str(e)[:300]}"
            await self._update_progress(error_msg)
            logger.error(f"Ошибка: {str(e)}", exc_info=True)
            return 0

    async def _update_from_telegram(self, days_offset: int) -> int:
        """Обновление данных из Telegram"""
        try:
            if not self.telegram_parser:
                raise ValueError("Парсер Telegram не инициализирован")
            
            documents = await self.telegram_parser.full_parse(days_offset=days_offset)
            
            if documents:
                ids = await self.vectorstore.aadd_documents(documents)
                msg = f"✅ Telegram: {len(ids)} сообщений (последние {days_offset} дней)"
                await self._update_progress(msg)
                return len(ids)
                
            await self._update_progress(f"ℹ️ Нет новых сообщений в Telegram за {days_offset} дней")
            return 0
            
        except Exception as e:
            error_msg = f"⚠️ Ошибка Telegram: {str(e)[:300]}"
            await self._update_progress(error_msg)
            logger.error(f"Ошибка: {str(e)}", exc_info=True)
            return 0

    def _cleanup(self):
        """Очистка ресурсов и сброс состояния"""
        self.bot = None
        self._chat_id = None
        self._progress_message = None
        logger.info("Ресурсы KnowledgeBase очищены")

    async def close(self):
        """Корректное закрытие всех ресурсов"""
        try:
            if hasattr(self, 'vectorstore'):
                del self.vectorstore
            self._cleanup()
            logger.info("База знаний закрыта корректно")
        except Exception as e:
            logger.error(f"Ошибка при закрытии: {str(e)}")