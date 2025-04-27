import os
import json
import logging
import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Set, AsyncGenerator, Optional
from tiktoken import encoding_for_model

from pyrogram import Client
from pyrogram.types import Message
from openai import AsyncOpenAI
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

from utils.config import Config

load_dotenv()
logger = logging.getLogger(__name__)

class TelegramParser:
    def __init__(self, knowledge_base, client: Client = None):
        """
        Усовершенствованный парсер Telegram чатов с:
        - LLM-разметкой вопросов/ответов
        - Автоматической обработкой больших чатов
        - Поддержкой JSON response_format
        """
        self.kb = knowledge_base
        self.client = client
        self.vectorstore = knowledge_base.vectorstore
        self.embeddings = knowledge_base.embeddings
        self.existing_message_ids: Set[str] = set()
        self.processed_chats: Dict[str, int] = {}
        
        # Инициализация OpenAI клиента с настройками
        self.openai_client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=30.0,
            max_retries=3
        )
        
        # Настройки токенизации и чанкинга
        self.encoder = encoding_for_model("gpt-3.5-turbo")
        self.MAX_TOKENS = 12000  # С запасом для промпта и ответа
        self.MESSAGE_CHUNK_SIZE = 30  # Оптимальный размер чанка
        self.MESSAGE_LENGTH_LIMIT = 500  # Лимит символов на сообщение

    async def initialize(self):
        """Инициализация клиента Telegram"""
        try:
            await self._load_existing_messages()
            if not self.client:
                self.client = Client(
                    "my_account",
                    api_id=int(os.getenv("API_ID")),
                    api_hash=str(os.getenv("API_HASH")),
                    session_string=await self._get_session_string()
                )
                await self.client.start()
            logger.info("Telegram client initialized")
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise

    async def _markup_messages(self, messages: List[Message]) -> List[Dict]:
        """
        Разметка сообщений через OpenAI API с:
        - Четким указанием формата JSON
        - Контролем размера чанков
        - Обработкой ошибок
        """
        SYSTEM_PROMPT = """Ты ассистент для разметки чатов. Проанализируй сообщения и верни JSON:
        {
            "messages": [
                {
                    "text": "текст сообщения",
                    "type": "question|answer|noise",
                    "theme": "onboarding|payments|technical|other"
                }
            ]
        }
        Важно: верни только JSON-объект без дополнительного текста!"""

        all_marked = []
        
        for chunk_idx in range(0, len(messages), self.MESSAGE_CHUNK_SIZE):
            chunk = messages[chunk_idx:chunk_idx + self.MESSAGE_CHUNK_SIZE]
            chunk_texts = []
            current_tokens = 0
            
            # Формирование чанка с контролем токенов
            for msg in chunk:
                if not msg.text:
                    continue
                    
                text = msg.text[:self.MESSAGE_LENGTH_LIMIT].strip()
                if not text:
                    continue
                    
                msg_text = f"{len(chunk_texts)+1}. {text}"
                msg_tokens = len(self.encoder.encode(msg_text))
                
                if current_tokens + msg_tokens > self.MAX_TOKENS:
                    break
                    
                chunk_texts.append(msg_text)
                current_tokens += msg_tokens

            if not chunk_texts:
                continue

            try:
                # Явное указание необходимости JSON в пользовательском сообщении
                user_content = "Разметь следующие сообщения в JSON формате:\n" + "\n".join(chunk_texts)
                
                response = await self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system", 
                            "content": SYSTEM_PROMPT
                        },
                        {
                            "role": "user",
                            "content": user_content
                        }
                    ],
                    temperature=0.5,
                    response_format={"type": "json_object"}
                )
                
                # Усиленная валидация ответа
                try:
                    result = json.loads(response.choices[0].message.content)
                    if isinstance(result, dict) and "messages" in result:
                        all_marked.extend(
                            msg for msg in result["messages"] 
                            if isinstance(msg, dict) and "text" in msg
                        )
                except (json.JSONDecodeError, AttributeError) as e:
                    logger.error(f"Invalid JSON in chunk {chunk_idx}: {e}")
                    
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_idx}: {str(e)}")
                continue

        return all_marked

    async def _create_documents(self, marked_messages: List[Dict]) -> List[Document]:
        """Создание документов из размеченных сообщений"""
        documents = []
        current_question = None
        
        for msg in marked_messages:
            if not isinstance(msg, dict):
                continue
                
            text = msg.get("text", "").strip()
            msg_type = msg.get("type", "")
            theme = msg.get("theme", "other")

            if not text or not msg_type:
                continue

            if msg_type == "question":
                current_question = {
                    "text": text,
                    "theme": theme
                }
            elif msg_type == "answer" and current_question:
                # Создаем документ только для пар вопрос-ответ
                content_hash = hashlib.md5(
                    f"{current_question['text']}{text}".encode()
                ).hexdigest()
                
                documents.append(Document(
                    page_content=current_question["text"],
                    metadata={
                        "answer": text,
                        "source": "telegram",
                        "theme": current_question["theme"],
                        "date": datetime.now().isoformat(),
                        "content_hash": content_hash,
                        "message_type": "qa_pair"
                    }
                ))
                current_question = None
                
        return documents

    async def parse_chat(
        self,
        chat_id: int,
        chat_title: str,
        days_offset: int = 30
    ) -> List[Document]:
        """Основной метод парсинга чата"""
        try:
            if not self.client:
                await self.initialize()

            # Сбор сообщений с фильтрацией по дате и дедупликацией
            messages = [
                msg async for msg in self.client.get_chat_history(chat_id, limit=1500)
                if (msg.date > datetime.now() - timedelta(days=days_offset)) and
                (str(msg.id) not in self.existing_message_ids)
            ]

            # LLM-разметка и создание документов
            marked_messages = await self._markup_messages(messages)
            if not marked_messages:
                logger.warning(f"No valid marked messages in {chat_title}")
                return []

            documents = await self._create_documents(marked_messages)
            logger.info(f"Created {len(documents)} QA documents from {chat_title}")
            return documents

        except Exception as e:
            logger.error(f"Failed to parse chat {chat_title}: {str(e)}")
            return []

    async def full_parse(self, days_offset: int = 30) -> List[Document]:
        """Парсинг всех доступных чатов"""
        if not self.client:
            await self.initialize()

        documents = []
        dialogs = [dialog async for dialog in self.client.get_dialogs()][:100]  # Лимит 100 чатов

        for dialog in dialogs:
            chat_title = getattr(dialog.chat, 'title', 'private')
            try:
                chat_docs = await self.parse_chat(
                    dialog.chat.id,
                    chat_title,
                    days_offset
                )
                documents.extend(chat_docs)
                await asyncio.sleep(1)  # Пауза между чатами
            except Exception as e:
                logger.error(f"Error parsing {chat_title}: {str(e)}")

        return documents

    async def _load_existing_messages(self):
        """Загрузка существующих сообщений для дедупликации"""
        try:
            existing = self.vectorstore.get(
                where={"source": "telegram"},
                include=["metadatas"]
            )
            if existing and "metadatas" in existing:
                self.existing_message_ids = {
                    str(meta.get("message_id", ""))
                    for meta in existing["metadatas"]
                    if meta.get("message_id")
                }
            logger.info(f"Loaded {len(self.existing_message_ids)} existing message IDs")
        except Exception as e:
            logger.error(f"Error loading existing messages: {str(e)}")
            self.existing_message_ids = set()

    async def _get_session_string(self) -> str:
        """Получение строки сессии Telegram"""
        async with Client(
            "temp_session",
            api_id=int(os.getenv("API_ID")),
            api_hash=str(os.getenv("API_HASH"))
        ) as temp_client:
            return await temp_client.export_session_string()