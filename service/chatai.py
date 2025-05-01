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
        """Улучшенная генерация ответа с интегрированным перефразированием"""
        try:
            # 1. Поиск в базе знаний
            results = await self.kb.search(question, k=3)
            if not results:
                return None

            best_doc, best_score = results[0]
            raw_answer = best_doc.metadata.get("answer", "").strip()

            # 2. Проверка валидности базового ответа
            if not self._is_valid_answer(raw_answer):
                # Полноценная RAG-цепочка если ответ невалидный
                context = f"В: {best_doc.page_content}\nО: {raw_answer}"
                history = await self._get_chat_history({"user_id": user_id})
                
                rag_prompt = ChatPromptTemplate.from_template(self.get_current_prompt())
                formatted_prompt = await rag_prompt.ainvoke({
                    "question": question,
                    "context": context,
                    "history": history
                })
                result_msg = await self.llm.ainvoke(formatted_prompt)
                return result_msg.content if self._is_valid_answer(result_msg.content) else None

            # 3. Улучшение ответа через ИИ (встроено в основной поток)
            if best_score > 0.75:  # Порог для улучшения
                try:
                    # Минимальное улучшение для точных ответов
                    if best_score > 0.9:
                        enhance_prompt = f"""
                        Перефразируй ответ более естественно, сохраняя точность:
                        Вопрос: {question}
                        Текущий ответ: {raw_answer}
                        Улучшенный вариант (максимум 1 предложение):
                        """
                    else:
                        # Полное улучшение с контекстом
                        enhance_prompt = f"""Твоя задача — развернуть ответ,если ответ ответ в два-три слова, используя контекст вопроса, но не добавляя новых фактов.Иначе оставь его каким он есть. 

                                                Правила:
                                                1. Сохрани 100% оригинального смысла
                                                2. Используй только информацию из предоставленного контекста
                                                3. Добавляй не более 5-7 слов сверх исходного ответа
                                                4. Следи за грамматической связностью

                                                Формат:
                                                Контекст: [релевантный фрагмент базы знаний]
                                                Вопрос: [вопрос пользователя] 
                                                Текущий ответ: [краткий ответ]

                                                Пример:
                                                Контекст: Для оплаты доступны карты Visa и Mastercard
                                                Вопрос: Какие карты принимаются?
                                                Текущий ответ: Visa и Mastercard
                                                Развернутый: Да, мы принимаем Visa и Mastercard

                                                Обработай:
                                                Контекст: {best_doc.page_content[:200]}...
                                                Вопрос: {question}
                                                Текущий ответ: {raw_answer}

                                                Развернутый ответ: """
                    
                    enhanced_msg = await self.llm.ainvoke(enhance_prompt)
                    final_answer = enhanced_msg.content if self._is_valid_answer(enhanced_msg.content) else raw_answer
                except Exception as e:
                    logger.warning(f"Answer enhancement failed, using original: {e}")
                    final_answer = raw_answer
            else:
                final_answer = raw_answer

            # 4. Обновление истории и возврат
            self._update_history(user_id, question, final_answer)
            return final_answer

        except Exception as e:
            logger.error(f"Full RAG generation error: {e}")
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