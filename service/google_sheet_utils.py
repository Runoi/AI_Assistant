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

class GoogleSheetsParser:
    def __init__(self, vectorstore: ChromaVectorStore):
        self.vectorstore = vectorstore
        self.processed_hashes: Set[str] = set()

    async def initialize(self):
        """Загружает существующие хеши при инициализации"""
        self.processed_hashes = await self.vectorstore.get_existing_hashes()
        print(f"🔍 Загружено {len(self.processed_hashes)} существующих хешей")

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
            
            return self._process_csv(response.text)
        except Exception as e:
            print(f"⚠️ Ошибка: {e}")
            return []

    def _process_csv(self, csv_content: str) -> List[Document]:
        """Обрабатывает CSV контент с фильтрацией дубликатов"""
        documents = []
        current_answer = ""
        
        reader = csv.reader(StringIO(csv_content))
        next(reader)  # Пропускаем заголовок
        
        for row in reader:
            if len(row) < 2:
                continue
                
            question = row[0].strip()
            answer = row[1].strip() or current_answer
            
            if not question:
                continue
                
            # Генерируем уникальный хеш
            content = f"{question}{answer}"
            content_hash = self._generate_hash(content)
            
            if content_hash not in self.processed_hashes:
                documents.append(Document(
                    page_content=question,
                    metadata={
                        "answer": answer,
                        "source": "google_sheets",
                        "content_hash": content_hash
                    }
                ))
                self.processed_hashes.add(content_hash)
            
            if row[1].strip():
                current_answer = row[1].strip()
        
        return documents
    
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
            logger.error(f"Ошибка парсинга промпта: {e}")
            return None
