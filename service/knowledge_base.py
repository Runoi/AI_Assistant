import os
import logging
import asyncio
from datetime import datetime
import time
from typing import List, Dict, Optional, Set, Tuple, Union
from pathlib import Path

import numpy as np
from pyrogram.types import Message
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

from utils.config import Config

load_dotenv()
logger = logging.getLogger(__name__)

class KnowledgeBase:
    def __init__(self, config: Config, client=None):
        """
        Улучшенная база знаний с поддержкой LLM-разметки
        
        Args:
            config: Конфигурация приложения
            client: Клиент Pyrogram (опционально)
        """
        self.config = config
        self.client = client
        self.bot = None
        self._chat_id = None
        
        # Инициализация векторного хранилища
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=config.OPENAI_API_KEY,
            chunk_size=500,
            openai_proxy=config.OPENAI_PROXY,
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

    
    

    @property
    def sheets_parser(self):
        """Ленивая инициализация парсера Google Sheets"""
        if self._sheets_parser is None:
            from service.google_sheet_utils import GoogleSheetsParser
            self._sheets_parser = GoogleSheetsParser(self.vectorstore)
            logger.info("Initialized Google Sheets parser")
        return self._sheets_parser

    @property
    def telegram_parser(self):
        """Ленивая инициализация парсера Telegram с LLM"""
        if self._telegram_parser is None:
            from service.chat_parser import TelegramParser
            self._telegram_parser = TelegramParser(self, client=self.client)
            logger.info("Initialized Telegram parser with LLM markup")
        return self._telegram_parser

    async def update_all_sources(
        self,
        bot,
        chat_id: int,
        telegram_days_offset: int = 30,
        regime: str = "all"
    ) -> Dict[str, int]:
        """
        Обновление всех источников с LLM-разметкой чатов
        Возвращает: {"google_sheets": count, "telegram": count}
        """
        self.bot = bot
        self._chat_id = chat_id
        results = {"google_sheets": 0, "telegram": 0}
        
        try:
            if regime == "sheets":
                # 1. Обновление Google Sheets
                sheets_count = await self._update_from_sheets()
                results["google_sheets"] = sheets_count
                total = sum(results.values())
                report = (
                "✅ Обновление завершено\n"
                f"• Google Sheets: {sheets_count}\n"
                f"• Всего: {total}"
            )
                await self._update_progress(report)
                return results
            elif regime == "telegram":
                # 2. Обновление Telegram с LLM-разметкой
                telegram_count = await self._update_from_telegram_llm(telegram_days_offset)
                results["telegram"] = telegram_count
                total = sum(results.values())
                report = (
                "✅ Обновление завершено\n"
                
                f"• Telegram (LLM): {telegram_count}\n"
                f"• Всего: {total}"
            )
                await self._update_progress(report)
                return results
            elif regime == "all":
                sheets_count = await self._update_from_sheets()
                telegram_count = await self._update_from_telegram_llm(telegram_days_offset)
                results["google_sheets"] = sheets_count
                total = sum(results.values())
                report = (
                "✅ Обновление завершено\n"
                f"• Google Sheets: {sheets_count}\n"
                f"• Telegram (LLM): {telegram_count}\n"
                f"• Всего: {total}"
            )
                await self._update_progress(report)
                return results
            
            
            
            
        except Exception as e:
            error_msg = f"⚠️ Ошибка: {str(e)[:400]}"
            await self._update_progress(error_msg)
            logger.error(f"Update error: {e}", exc_info=True)
            return results

    async def _update_from_sheets(self) -> int:
        try:
            # Получаем документы из Google Sheets
            documents = await self.sheets_parser.parse_sheet(self.config.GOOGLE_SHEET_URL)
            if not documents:
                logger.warning("No documents parsed from Google Sheets")
                return 0

            # Получаем существующие вопросы из векторного хранилища
            existing_questions = await self._get_existing_questions()
            
            # Нормализация и дедупликация
            unique_docs = {}
            duplicate_count = 0
            
            for doc in documents:
                norm_text = self._normalize_text(doc.page_content)
                
                # Проверяем по нормализованному тексту вопроса
                if norm_text in existing_questions:
                    duplicate_count += 1
                    continue
                    
                if norm_text not in unique_docs:
                    unique_docs[norm_text] = doc
                    existing_questions.add(norm_text)  # Добавляем в существующие

            if duplicate_count > 0:
                logger.info(f"Пропущено {duplicate_count} дубликатов")

            if not unique_docs:
                logger.warning("No new unique documents after deduplication")
                return 0

            # Создаем эмбеддинги только для новых документов
            texts = [doc.page_content for doc in unique_docs.values()]
            embeddings = await self.embeddings.aembed_documents(texts)

            # Добавляем в векторное хранилище
            ids = await self.vectorstore.aadd_documents(
                documents=list(unique_docs.values()),
                embeddings=embeddings
            )
            
            if not ids:
                logger.error("Failed to add documents to vectorstore")
                return 0

            logger.info(f"Added {len(ids)} new documents from Google Sheets")
            return len(ids)
        
        except Exception as e:
            logger.error(f"Ошибка обновления: {e}", exc_info=True)
            return 0

    async def _get_existing_questions(self) -> Set[str]:
        """Получает множество нормализованных текстов существующих вопросов"""
        existing = self.vectorstore.get(include=["documents"])
        if existing and "documents" in existing:
            return {
                self._normalize_text(doc)
                for doc in existing["documents"]
            }
        return set()

    def _normalize_text(self, text: str) -> str:
        """Улучшенная нормализация текста для сравнения"""
        if not isinstance(text, str):
            return ""
        
        # Приведение к нижнему регистру, удаление пунктуации и лишних пробелов
        text = text.lower().strip()
        for char in "?!.,:;\"'":
            text = text.replace(char, "")
        return " ".join(text.split())  # Удаление множественных пробелов

    
    async def _update_from_telegram_llm(self, days_offset: int) -> int:
        """Обновление Telegram чатов с LLM-разметкой"""
        try:
            documents = await self.telegram_parser.full_parse(days_offset=days_offset)
            
            if documents:
                # Фильтрация дубликатов перед добавлением
                existing_hashes = await self._get_existing_hashes()
                new_docs = [
                    doc for doc in documents 
                    if doc.metadata["content_hash"] not in existing_hashes
                ]
                
                if new_docs:
                    await self.vectorstore.aadd_documents(new_docs)
                    logger.info(f"Added {len(new_docs)} LLM-marked telegram messages")
                    return len(new_docs)
                    
            logger.info("No new telegram messages")
            return 0
        except Exception as e:
            logger.error(f"Telegram LLM update error: {e}", exc_info=True)
            return 0
    
    async def get_prompt_from_sheets(self) -> Optional[str]:
        """Получает промпт из Google Sheets"""
        try:
            return await self.sheets_parser.parse_prompt_from_sheet(
                self.config.GOOGLE_SHEET_URL
            )
        except Exception as e:
            logger.error(f"Error getting prompt from sheets: {e}")
            return None

    async def _get_existing_hashes(self) -> Set[str]:
        """Получение хешей существующих документов"""
        existing = self.vectorstore.get(include=["metadatas"])
        if existing and "metadatas" in existing:
            return {
                meta["content_hash"]
                for meta in existing["metadatas"]
                if "content_hash" in meta
            }
        return set()
    async def export_to_json(self) -> Optional[str]:
        """Экспортирует всю базу знаний в JSON файл и возвращает путь к нему."""
        try:
            # Получаем все данные из векторного хранилища
            data = self.vectorstore.get(include=["metadatas", "documents"])
            if not data or "documents" not in data:
                return None

            # Формируем структуру для экспорта
            export_data = []
            for doc, meta in zip(data["documents"], data["metadatas"]):
                export_data.append({
                    "question": doc,
                    "answer": meta.get("answer", ""),
                    "source": meta.get("source", "unknown"),
                    "theme": meta.get("theme", ""),
                    "content_hash": meta.get("content_hash", ""),
                    "date": meta.get("date", "")
                })

            # Сохраняем во временный файл с указанием кодировки UTF-8
            import tempfile
            import json
            import os

            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
                file_path = f.name

            return file_path

        except Exception as e:
            logger.error(f"Export error: {e}")
            return None
        

    async def remove_duplicates(self) -> dict:
        """
        Удаляет дубликаты из базы знаний.
        Возвращает: {"total": int, "removed": int, "remaining": int}
        """
        try:
            # Получаем все документы (без явного запроса ids)
            data = self.vectorstore.get(include=["metadatas", "documents"])
            if not data or not data.get("documents"):
                return {"total": 0, "removed": 0, "remaining": 0}

            # Получаем ids отдельным запросом
            ids = self.vectorstore.get()["ids"]
            
            # Собираем уникальные записи
            unique_entries = {}
            duplicates_ids = []

            for idx, (meta, doc_text) in enumerate(zip(data["metadatas"], data["documents"])):
                content_hash = meta.get("content_hash", doc_text)  # Используем текст как fallback
                
                if content_hash in unique_entries:
                    duplicates_ids.append(ids[idx])
                else:
                    unique_entries[content_hash] = ids[idx]

            # Удаляем дубликаты
            if duplicates_ids:
                await self.vectorstore.adelete(ids=duplicates_ids)
            
            return {
                "total": len(ids),
                "removed": len(duplicates_ids),
                "remaining": len(ids) - len(duplicates_ids)
            }

        except Exception as e:
            logger.error(f"Ошибка удаления дубликатов: {e}", exc_info=True)
            return {"error": str(e)}

    async def get_all_sources(self) -> List[Dict]:
        """Получение статистики по источникам с примерами"""
        try:
            results = self.vectorstore.get(include=["metadatas", "documents"])
            if not results or "metadatas" not in results:
                return []
                
            # Группировка по источникам
            sources = {}
            for meta, content in zip(results["metadatas"], results["documents"]):
                source = meta.get("source", "unknown")
                if source not in sources:
                    sources[source] = {
                        "count": 0,
                        "examples": [],
                        "themes": set()
                    }
                sources[source]["count"] += 1
                if len(sources[source]["examples"]) < 5:
                    sources[source]["examples"].append(content[:200])
                if "theme" in meta:
                    sources[source]["themes"].add(meta["theme"])
            
            # Форматирование результата
            return [
                {
                    "source": source,
                    "count": data["count"],
                    "examples": data["examples"],
                    "themes": list(data["themes"])[:5]  # Ограничение количества тем
                }
                for source, data in sources.items()
            ]
        except Exception as e:
            logger.error(f"Sources stats error: {e}", exc_info=True)
            return []

    async def search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        try:
            norm_query = self._normalize_text(query)
            query_embedding = await self.embeddings.aembed_query(norm_query)
            
            # Получаем raw результаты от Chroma
            results = self.vectorstore._collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["metadatas", "documents", "distances"]
            )
            
            # Преобразуем в нужный формат
            output = []
            for i in range(len(results["ids"][0])):
                doc = Document(
                    page_content=results["documents"][0][i],
                    metadata=results["metadatas"][0][i]
                )
                score = 1.0 - results["distances"][0][i]  # Конвертируем расстояние в схожесть
                output.append((doc, score))
            
            logger.debug(f"Поиск: '{query}' -> Лучший score: {output[0][1]:.2f}")
            return output
            
        except Exception as e:
            logger.error(f"Ошибка поиска: {e}")
            return []

    def _normalize_query(self, query: str) -> str:
        """Нормализация поискового запроса"""
        return query.strip().capitalize()

    async def _update_progress(self, text: str):
        """Обновление сообщения о прогрессе"""
        if not self.bot or not self._chat_id:
            return
            
        try:
            text = text[:4000]  # Ограничение длины
            if hasattr(self, "_progress_message"):
                await self._progress_message.edit_text(text)
            else:
                self._progress_message = await self.bot.send_message(
                    chat_id=self._chat_id,
                    text=text
                )
        except Exception as e:
            logger.error(f"Progress update failed: {e}")

    

    async def close(self):
        """Корректное закрытие ресурсов"""
        try:
            if hasattr(self, "vectorstore"):
                del self.vectorstore
            
            logger.info("KnowledgeBase closed")
        except Exception as e:
            logger.error(f"Close error: {e}")