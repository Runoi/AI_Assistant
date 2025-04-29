from typing import Dict, List, Optional, Union
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from service.knowledge_base import KnowledgeBase
from utils.config import Config
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class TerraChatAI:
    def __init__(self, config: Config, client=None):
        self.config = config
        self.client = client
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.3,
            api_key=config.OPENAI_API_KEY,
            max_retries=3,
            timeout=30,
            openai_proxy=config.OPENAI_PROXY
        )
        self.kb = KnowledgeBase(config, client=client)
        #self.setup_rag_chain()
        self.message_history: Dict[int, List[Union[HumanMessage, AIMessage]]] = {}
        self.user_sessions: Dict[int, datetime] = {}
        self.current_prompt = self._get_default_prompt()
        self.setup_rag_chain()  # Создаем цепочку с дефолтным промптом
        
        # База общих вопросов
        self.common_answers = {
            "кто ты": "Я - AI-ассистент TERRA. Помогаю с вопросами о наставниках и заявках.",
            "что ты умеешь": "Могу отвечать на вопросы, искать информацию в базе знаний TERRA.",
            "помощь": "Задайте вопрос о наставниках, заявках или другой информации TERRA.",
        }

    def _get_default_prompt(self) -> str:
        """Возвращает промпт по умолчанию."""
        return """Ты — ассистент-человек по имени TERRA. Отвечай на вопросы, используя только предоставленный контекст.

                Твоя логика ответа:
                1. Прочти вопрос и пойми его суть
                2. Найди в контексте прямой ответ
                3. Если прямого ответа нет, найди похожие вопросы
                4. На основе аналогии сделай вывод
                5. Если даже по аналогиям ответить нельзя, скажи, что недостаточно данных.

                Правила:
                - Не привязывай ответ только к отдельным людям
                - Отвечай на суть вопроса
                - Формулируй ответ полезно для пользователя

                Контекст: {context}
                История диалога: {history}
                Вопрос: {question}

                Ответ (ясный, логичный и полезный для пользователя):"""

    def get_current_prompt(self) -> str:
        """Возвращает текущий промпт"""
        return self.current_prompt
        
    def update_prompt(self, new_prompt: str) -> bool:
        """Обновляет промпт и пересоздает цепочку RAG"""
        try:
            self.current_prompt = new_prompt
            self.setup_rag_chain()  # Пересоздаем цепочку с новым промптом
            return True
        except Exception as e:
            logger.error(f"Ошибка обновления промпта: {e}")
            return False
        
    def setup_rag_chain(self):
        """Настройка цепочки RAG с текущим промптом."""
        prompt = ChatPromptTemplate.from_template(self.get_current_prompt())
        self.rag_chain = (
            {
                "context": self._retrieve_context,
                "history": self._get_chat_history,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

    async def _retrieve_context(self, input_data: Dict) -> str:
        """Поиск контекста с защитой от неправильных типов"""
        try:
            question = input_data.get("question", "")
            if not isinstance(question, str):
                logger.error(f"Некорректный тип вопроса: {type(question)}")
                return ""
                
            docs = await self.kb.search(question, k=5)
            return "\n".join(
                f"Вопрос: {doc.page_content}\nОтвет: {doc.metadata.get('answer', '...')}"
                for doc in docs
            )
        except Exception as e:
            logger.error(f"Ошибка поиска контекста: {e}")
            return ""

    async def _get_chat_history(self, input_data: Dict) -> str:
        """Получение истории чата"""
        user_id = input_data.get("user_id")
        if not user_id or user_id not in self.message_history:
            return ""
        return "\n".join(
            f"{'User' if isinstance(msg, HumanMessage) else 'Bot'}: {msg.content}"
            for msg in self.message_history[user_id][-3:]
        )

    async def generate_answer(self, user_id: int, question: Union[str, Dict]) -> Optional[str]:
        """Основной метод генерации ответа"""
        try:
            # Нормализация входа
            question_text = self._normalize_question(question)
            if not question_text:
                return None
                
            # Проверка общих вопросов
            if answer := self._check_common_questions(question_text):
                print(f"Ответ на общий вопрос: {answer}")
                return answer
                
            # Генерация ответа
            return await self._generate_rag_answer(user_id, question_text)
            
        except Exception as e:
            logger.error(f"Ошибка генерации ответа: {e}")
            return None

    def _normalize_question(self, question: Union[str, Dict]) -> Optional[str]:
        """Приведение вопроса к строке"""
        if isinstance(question, dict):
            question = question.get("text", "")
        elif not isinstance(question, str):
            question = str(question)
            
        question = question.strip()
        return question if len(question) >= 2 else None

    def _check_common_questions(self, question: str) -> Optional[str]:
        """Проверка общих вопросов"""
        lower_q = question.lower().strip()
        for q, answer in self.common_answers.items():
            if q in lower_q:
                return answer
        return None

    async def _generate_rag_answer(self, user_id: int, question: str) -> Optional[str]:
        """Генерация ответа через RAG"""
        try:
            result = await self.rag_chain.ainvoke({
                "question": question,
                "user_id": user_id
            })
            
            if not self._is_valid_answer(result):
                return None
                
            self._update_history(user_id, question, result)
            return result
        except Exception as e:
            logger.error(f"Ошибка RAG цепи: {e}")
            return None

    def _update_history(self, user_id: int, question: str, answer: str):
        """Обновление истории сообщений"""
        if user_id not in self.message_history:
            self.message_history[user_id] = []
            
        self.message_history[user_id].extend([
            HumanMessage(content=question),
            AIMessage(content=answer)
        ])
        
        # Ограничение истории
        if len(self.message_history[user_id]) > 10:
            self.message_history[user_id] = self.message_history[user_id][-10:]

    def _is_valid_answer(self, text: str) -> bool:
        """Проверка валидности ответа"""
        if not text or len(text.strip()) < 3:
            return False
            
        text = text.lower()
        invalid = {"не знаю", "нет информации", "не нашел"}
        return not any(phrase in text for phrase in invalid)

    async def cleanup_inactive_sessions(self, hours=24):
        """Очистка неактивных сессий"""
        now = datetime.now()
        inactive = [
            uid for uid, last in self.user_sessions.items()
            if (now - last).total_seconds() > hours * 3600
        ]
        
        for uid in inactive:
            self.message_history.pop(uid, None)
            self.user_sessions.pop(uid, None)