import asyncio
from typing import Dict, List, Optional, Union
from collections import Counter
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
            model= config.MODEL,
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
        self._init_task = asyncio.create_task(self._async_init())
        # База общих вопросов
        self.common_answers = {
            
            "что ты умеешь": "Могу отвечать на вопросы, искать информацию в базе знаний TERRA.",
            "помощь": "Задайте вопрос о наставниках, заявках или другой информации TERRA.",
        }

    async def _async_init(self):
        """Асинхронная часть инициализации"""
        try:
            await self._load_prompt_from_sheets()
        except Exception as e:
            logger.error(f"Error in async init: {e}")

    async def _load_prompt_from_sheets(self):
        """Загружает и устанавливает промпт из Google Sheets"""
        try:
            prompt = await self.kb.get_prompt_from_sheets()
            if prompt and prompt.strip():
                # Обновляем текущий промпт
                self.current_prompt = prompt.strip()
                # Пересоздаём цепочку RAG с новым промптом
                self.setup_rag_chain()
                logger.info("Prompt updated from Google Sheets")
        except Exception as e:
            logger.error(f"Failed to load prompt from sheets: {e}")

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
        async def prepare_context(input_data: Dict) -> Dict:
            """Подготовка контекста с учетом релевантности"""
            question = input_data["question"]
            user_id = input_data.get("user_id")
            
            # Получаем результаты с оценкой релевантности
            results = await self.kb.search(question, k=5)
            
            # Фильтруем и форматируем контекст
            context_parts = []
            for doc, score in results:
                if score > 0.5:  # Порог релевантности
                    context_parts.append(
                        f"[Релевантность: {score:.2f}]\n"
                        f"В: {doc.page_content}\n"
                        f"О: {doc.metadata.get('answer', '...')}\n"
                    )
            
            # Получаем историю чата
            history = await self._get_chat_history({"user_id": user_id}) if user_id else ""
            
            return {
                "question": question,
                "context": "\n".join(context_parts) if context_parts else "Нет релевантной информации",
                "history": history
            }
    
        self.rag_chain = (
            RunnablePassthrough()
            | prepare_context
            | prompt
            | self.llm
            | StrOutputParser()
        )
    async def _retrieve_context(self, input_data: Dict) -> str:
        question = input_data["question"]
        docs = await self.kb.search(question, k=5)
        
        if not docs:
            return ""
        
        context = []
        for doc in docs:
            # Добавляем ответ ТОЛЬКО если он есть в metadata
            answer = doc.metadata.get("answer")
            if answer:
                context.append(
                    f"Вопрос: {doc.page_content}\nОтвет: {answer}\n"
                    f"Источник: {doc.metadata.get('source', 'unknown')}\n"
                    f"Score: {doc.metadata.get('score', 0):.2f}"
                )
        
        return "\n".join(context) if context else ""

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
        """Генерация ответа через RAG, с приоритетом прямых совпадений"""
        try:
            results = await self.kb.search(question, k=5)
            if not results:
                logger.warning(f"No results found for question: {question}")
                return None

            filtered_results = [(doc, score) for doc, score in results if score > 0.5]
            if not filtered_results:
                logger.warning(f"No relevant results (all scores <= 0.7) for: {question}")
                return None

            # Логируем top-3 ответа для отладки
            for i, (doc, score) in enumerate(filtered_results[:3], 1):
                raw = doc.metadata.get("answer", "").strip()
                logger.info(f"Топ-{i}: score={score:.2f} | answer='{raw}'")

            # Попробуем найти наиболее часто встречающийся валидный ответ среди топ-3
            top_answers = [
                doc.metadata.get("answer", "").strip()
                for doc, score in filtered_results[:3]
                if self._is_valid_answer(doc.metadata.get("answer", "").strip())
            ]

            if top_answers:
                counter = Counter(top_answers)
                most_common_answer, count = counter.most_common(1)[0]
                if count >= 2:
                    logger.info("Ответ подтверждён несколькими источниками")
                    self._update_history(user_id, question, most_common_answer)
                    return most_common_answer

            best_doc, best_score = filtered_results[0]
            logger.info(f"Best match score: {best_score:.2f} for: '{question}'")

            raw_answer = best_doc.metadata.get("answer", "...").strip()
            if best_score > 0.75 and self._is_valid_answer(raw_answer):
                logger.info("Ответ напрямую из базы знаний")
                self._update_history(user_id, question, raw_answer)
                return raw_answer

            context = f"В: {best_doc.page_content}\nО: {raw_answer}"
            history = await self._get_chat_history({"user_id": user_id})

            prompt_template = ChatPromptTemplate.from_template(self.get_current_prompt())
            formatted_prompt = await prompt_template.ainvoke({
                "question": question,
                "context": context,
                "history": history
            })

            result_msg = await self.llm.ainvoke(formatted_prompt)
            result = result_msg.content

            logger.info(f"Raw RAG result: {result}")

            if self._is_valid_answer(result):
                self._update_history(user_id, question, result)
                return result
            else:
                return None

        except Exception as e:
            logger.error(f"RAG chain error: {e}", exc_info=True)
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
        if not text or len(text.strip()) < 5:  # Увеличить минимальную длину
            return False
        text = text.lower()
        invalid_phrases = {
            "не знаю", "нет информации", "не нашел", 
            "нет ответа", "не могу ответить", "у меня нет ответа"
        }
        # Запретить ответы, содержащие invalid_phrases, даже если есть "нет"
        return not any(phrase in text for phrase in invalid_phrases)

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