import os
import asyncio
import time
import math
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set, AsyncGenerator
from pyrogram import Client
from pyrogram.types import Message
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import logging
from dotenv import load_dotenv

load_dotenv('.env')

logger = logging.getLogger(__name__)

class TelegramParser:
    def __init__(self, vectorstore: Chroma):
        """
        Парсер Telegram чатов с интеграцией в базу знаний
        
        Args:
            vectorstore: Векторное хранилище Chroma
        """
        self.vectorstore = vectorstore
        self.knowledge_base = None  # Будет установлено KnowledgeBase
        self.client = None
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.existing_message_ids: Set[str] = set()
        self.processed_chats: Dict[str, int] = {}
        self.progress_message = None  # Храним ссылку на сообщение прогресса
        self.progress_message_id = None

    async def initialize(self):
        """Инициализация клиента Telegram"""
        try:
            await self._load_existing_messages()
            self.client = Client(
                "my_account",
                api_id=int(os.getenv("API_ID")),
                api_hash=str(os.getenv("API_HASH")),
                session_string=await self._get_session_string()
            )
            await self.client.start()
            logger.info("Клиент Telegram инициализирован")
        except Exception as e:
            logger.error(f"Ошибка инициализации: {str(e)}")
            raise

    async def _load_existing_messages(self):
        """Загрузка существующих сообщений для дедупликации"""
        try:
            existing = self.vectorstore.get(
                where={"source": "telegram"},
                include=["metadatas"]
            )
            if existing and "metadatas" in existing:
                self.existing_message_ids = {
                    meta["message_id"] for meta in existing["metadatas"]
                    if "message_id" in meta
                }
            logger.info(f"Загружено {len(self.existing_message_ids)} существующих сообщений")
        except Exception as e:
            logger.error(f"Ошибка загрузки сообщений: {str(e)}")
            self.existing_message_ids = set()

    async def _get_session_string(self) -> str:
        """Получение строки сессии"""
        async with Client(
            "temp_session",
            api_id=int(os.getenv("API_ID")),
            api_hash=str(os.getenv("API_HASH"))
        ) as temp_client:
            return await temp_client.export_session_string()

    async def _update_progress(self, text: str, force_new: bool = False) -> bool:
        """
        Умное обновление сообщения прогресса
        Возвращает True, если сообщение было обновлено
        """
        if not self.knowledge_base or not hasattr(self.knowledge_base, 'bot'):
            return False

        try:
            # Если есть существующее сообщение и не требуется новое
            if self.progress_message and not force_new:
                try:
                    await self.progress_message.edit_text(text)
                    return True
                except Exception as edit_error:
                    logger.warning(f"Не удалось отредактировать сообщение: {edit_error}")
            
            # Создаем новое сообщение
            new_msg = await self.knowledge_base.bot.send_message(
                chat_id=self.knowledge_base._chat_id,
                text=text
            )
            
            # Удаляем старое сообщение, если оно есть
            if self.progress_message:
                try:
                    await self.progress_message.delete()
                except Exception as delete_error:
                    logger.warning(f"Не удалось удалить старое сообщение: {delete_error}")
            
            self.progress_message = new_msg
            return True
            
        except Exception as e:
            logger.error(f"Ошибка обновления прогресса: {e}")
            return False

    async def extract_text(self, message: Message) -> Optional[str]:
        """Извлечение текста из сообщения"""
        try:
            if message.text:
                return message.text
            if message.caption:
                return message.caption
            if message.media:
                return f"[Медиа: {message.media.__class__.__name__}]"
            return None
        except Exception as e:
            logger.error(f"Ошибка извлечения текста: {str(e)}")
            return None

    async def process_message(self, message: Message) -> Optional[Document]:
        """Обработка сообщения в формат Document"""
        try:
            text = await self.extract_text(message)
            if not text:
                return None

            return Document(
                page_content=text,
                metadata={
                    "chat_id": str(message.chat.id),
                    "chat_title": str(getattr(message.chat, 'title', 'private')),
                    "message_id": str(message.id),
                    "date": message.date.isoformat() if message.date else "",
                    "sender": str(getattr(message.from_user, 'first_name', '')),
                    "source": "telegram"
                }
            )
        except Exception as e:
            logger.error(f"Ошибка обработки сообщения: {str(e)}")
            return None

    def _generate_progress_bar(self, percent: int, length: int = 10) -> str:
        """Генерация текстового прогресс-бара"""
        filled = math.ceil(length * percent / 100)
        empty = length - filled
        return f"[{'█' * filled}{'░' * empty}] {percent}%"

    async def parse_chat(
        self,
        chat_id: int,
        chat_title: str,
        days_offset: int = 30
    ) -> AsyncGenerator[Document, None]:
        """Парсинг чата с улучшенным индикатором прогресса"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_offset)

        # Получаем приблизительное количество сообщений
        approx_count = 0
        try:
            chat = await self.client.get_chat(chat_id)
            if hasattr(chat, 'messages'):
                approx_count = min(chat.messages, 1000)
        except Exception as e:
            logger.warning(f"Не удалось получить количество сообщений: {str(e)}")

        await self._update_progress(
            f"🔍 Начало парсинга: {chat_title}\n"
            f"⏳ Период: {start_date.date()} - {end_date.date()}\n"
            f"📝 Примерное количество сообщений: {approx_count if approx_count > 0 else 'неизвестно'}"
        )

        total = saved = 0
        last_update = 0
        progress_emoji = "🔄"
        progress_interval = max(1, approx_count // 10) if approx_count > 0 else 10
        start_time = time.time()

        async for message in self.client.get_chat_history(chat_id, limit=1000):
            if message.date < start_date:
                continue

            if str(message.id) in self.existing_message_ids:
                continue

            doc = await self.process_message(message)
            if doc:
                self.existing_message_ids.add(str(message.id))
                saved += 1
                yield doc

            total += 1
            
            # Обновляем прогресс
            if total % progress_interval == 0 or total == 1 or time.time() - last_update > 5:
                progress_percent = min(100, int(total / approx_count * 100)) if approx_count > 0 else 0
                
                # Расчет оставшегося времени
                time_elapsed = time.time() - start_time
                if total > 0 and approx_count > 0:
                    time_remaining = (time_elapsed / total) * (approx_count - total)
                    time_str = f"⏱️ ~{timedelta(seconds=int(time_remaining))} осталось"
                else:
                    time_str = ""

                status = (
                    f"{progress_emoji} {chat_title}\n"
                    f"{self._generate_progress_bar(progress_percent)}\n"
                    f"📊 Прогресс: {progress_percent}% ({total}/{approx_count if approx_count > 0 else '?'})\n"
                    f"💾 Сохранено: {saved}\n"
                    f"{time_str}"
                )
                
                await self._update_progress(status)
                last_update = time.time()
                progress_emoji = "⏳" if progress_emoji == "🔄" else "🔄"

        # Финальный отчет
        time_elapsed = timedelta(seconds=int(time.time() - start_time))
        await self._update_progress(
            f"✅ Парсинг завершен: {chat_title}\n"
            f"• Обработано сообщений: {total}\n"
            f"• Новых сохранено: {saved}\n"
            f"• Время выполнения: {time_elapsed}"
        )
        logger.info(f"Чат {chat_title}: {saved}/{total} сообщений за {time_elapsed}")

    async def full_parse(self, days_offset: int = 30) -> List[Document]:
        """Полный парсинг с управлением прогрессом"""
        if not self.client:
            await self.initialize()

        documents = []
        dialogs = [dialog async for dialog in self.client.get_dialogs()]
        
        await self._update_progress(
            f"🚀 Начинаю парсинг {len(dialogs)} чатов\n"
            f"⏳ Период: последние {days_offset} дней"
        )

        for i, dialog in enumerate(dialogs, 1):
            chat_title = getattr(dialog.chat, 'title', 'private')
            chat_docs = []
            
            async for doc in self.parse_chat(dialog.chat.id, chat_title, days_offset):
                chat_docs.append(doc)
                if len(chat_docs) % 20 == 0:
                    await self._update_progress(
                        f"📊 {i}/{len(dialogs)} чатов\n"
                        f"💬 {chat_title}: {len(chat_docs)} сообщений\n"
                        f"📝 Всего: {len(documents) + len(chat_docs)}"
                    )
            
            documents.extend(chat_docs)
            await asyncio.sleep(1)  # Небольшая пауза между чатами

        await self._update_progress(
            f"🎉 Парсинг завершен!\n"
            f"• Чатов: {len(dialogs)}\n"
            f"• Сообщений: {len(documents)}"
        )
        
        return documents