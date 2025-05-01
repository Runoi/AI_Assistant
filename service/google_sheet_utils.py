import hashlib
import os
from pathlib import Path
import requests
import csv
from io import StringIO
from datetime import datetime
from typing import List, Optional, Set
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import asyncio
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv('.env')

OPENAI_API_KEY = str(os.getenv("OPENAI_API_KEY"))


class ChromaVectorStore:
    def __init__(self, persist_directory: str = "chroma_data"):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=OPENAI_API_KEY,
            openai_proxy=os.getenv("OPENAI_PROXY"),
        )
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
    
    async def aadd_documents(self, documents: List[Document]) -> int:
        """Асинхронно добавляет документы в ChromaDB"""
        if documents:
            await self.vectorstore.aadd_documents(documents)
            return len(documents)
        return 0
    
    async def get_existing_hashes(self) -> Set[str]:
        """Возвращает хеши существующих документов"""
        existing = self.vectorstore._collection.get(include=["metadatas"])
        if existing and "metadatas" in existing:
            return {
                meta["content_hash"]
                for meta in existing["metadatas"]
                if "content_hash" in meta
            }
        return set()
    
    async def get_existing_questions(self) -> Set[str]:
        """Возвращает нормализованные тексты существующих вопросов"""
        existing = self.vectorstore._collection.get(include=["documents"])
        if existing and "documents" in existing:
            return {
                self._normalize_question(doc)
                for doc in existing["documents"]
            }
        return set()
    def _normalize_question(self, text: str) -> str:
        """Нормализация вопроса для сравнения"""
        return text.strip().lower().replace("?", "").replace("!", "").replace(".", "")


class GoogleSheetsParser:
    def __init__(self, vectorstore: ChromaVectorStore):
        self.vectorstore = vectorstore
        self.processed_hashes: Set[str] = set()
        self.existing_questions: Set[str] = set()  # Добавляем инициализацию

    
    async def initialize(self):
        """Загружает существующие вопросы при инициализации"""
        self.processed_hashes = await self.vectorstore.get_existing_hashes()
        self.existing_questions = await self.vectorstore.get_existing_questions()  # Новая функция
        print(f"🔍 Загружено {len(self.existing_questions)} существующих вопросов")

    @staticmethod
    def _get_export_url(sheet_url: str) -> str:
        """Генерирует корректный URL для экспорта CSV"""
        try:
            if '/spreadsheets/d/' in sheet_url:
                sheet_id = sheet_url.split('/spreadsheets/d/')[1].split('/')[0]
            else:
                sheet_id = sheet_url.split('/d/')[1].split('/')[0]
            return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
        except IndexError:
            raise ValueError("Неверный формат URL Google Таблицы")

    def _generate_hash(self, content: str) -> str:
        """Генерирует MD5 хеш содержимого"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    async def parse_sheet(self, sheet_url: str) -> List[Document]:
        """Парсит Google Таблицу и возвращает только новые документы"""
        try:
            export_url = self._get_export_url(sheet_url)
            print(f"🔗 Загружаем данные по URL: {export_url}")
            
            response = requests.get(export_url, timeout=30)
            response.encoding = 'utf-8'
            response.raise_for_status()
            
            all_documents = self._process_csv(response.text)
            
            # Фильтруем по нормализованному тексту вопроса
            new_documents = []
            duplicate_count = 0
            
            for doc in all_documents:
                norm_question = self._normalize_question(doc.page_content)
                if norm_question in self.existing_questions:
                    duplicate_count += 1
                    continue
                    
                new_documents.append(doc)
                self.existing_questions.add(norm_question)
            
            print(f"📊 Найдено {len(all_documents)} записей, из них новых: {len(new_documents)}, дубликатов: {duplicate_count}")
            return new_documents
        except Exception as e:
            print(f"⚠️ Ошибка: {e}")
            return []

    def _process_csv(self, csv_content: str) -> List[Document]:
        seen = set()
        documents = []
        last_valid_answer = None  # Храним последний непустой ответ
        
        reader = csv.reader(StringIO(csv_content))
        headers = [h.strip().lower() for h in next(reader)]
        
        try:
            q_col = headers.index("вопрос")
            a_col = headers.index("ответ")
        except ValueError:
            return []
        
        for row in reader:
            if len(row) <= max(q_col, a_col):
                continue
                
            question = self._normalize_question(row[q_col])
            answer = self._normalize_answer(row[a_col])
            
            # Если ответ пустой, используем последний валидный
            if not answer and last_valid_answer:
                answer = last_valid_answer
            elif answer:  # Запоминаем новый валидный ответ
                last_valid_answer = answer
                
            if not question or not answer:
                continue
                
            content_key = hashlib.md5(f"{question.lower()}{answer.lower()}".encode()).hexdigest()
            if content_key in seen:
                continue
                
            seen.add(content_key)
            documents.append(Document(
                page_content=question,
                metadata={
                    "answer": answer,
                    "source": "google_sheets",
                    "content_hash": content_key
                }
            ))
        
        return documents

    def _normalize_question(self, text: str) -> str:
        """Приведение вопроса к стандартному виду"""
        if not isinstance(text, str):
            return ""
        return text.strip().capitalize()

    def _normalize_answer(self, text: str) -> str:
        """Приведение ответа к стандартному виду"""
        if not isinstance(text, str):
            return ""
        return text.strip()
        
    async def parse_prompt_from_sheet(self, sheet_url: str) -> Optional[str]:
        """Парсит промпт из листа 'Prompt' в Google Sheets"""
        try:
            # Получаем ID таблицы из URL
            sheet_id = sheet_url.split('/d/')[1].split('/')[0]
            
            # URL для экспорта листа 'Prompt' в CSV
            prompt_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet=Prompt"
            
            response = requests.get(prompt_url, timeout=30)
            response.raise_for_status()
            
            # Читаем CSV, ищем первый непустой промпт
            reader = csv.reader(StringIO(response.text))
            for row in reader:
                if row and row[0].strip():  # Проверяем первую колонку
                    return row[0].strip()
                    
            return None
        except Exception as e:
            
            return None
