import logging
from aiogram import Dispatcher, types, F

from aiogram.filters import Command
from aiogram.types import Message
from typing import Callable, Dict, Any, Awaitable
from service.chatai import TerraChatAI
from service.knowledge_base import KnowledgeBase
from utils.config import Config
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

async def cmd_start(message: types.Message):
    await message.answer(
        "🤖 Я AI-ассистент TERRA. Задайте вопрос о наставниках или заявках."
    )

async def cmd_update(message: types.Message, config: Config):
    """Обработчик команды обновления"""
    if message.from_user.id not in config.ADMINS:
        return await message.answer("⛔ Недостаточно прав")
    
    kb = KnowledgeBase(config)
    try:
        # Передаем бота и chat_id для управления сообщениями
        results = await kb.update_all_sources(
            bot=message.bot,
            chat_id=message.chat.id
        )
        
        # await message.answer(
        #     f"✅ Обновление завершено!\n"
        #     f"Google Sheets: {results['google_sheets']}\n"
        #     f"Telegram: {results['telegram']}"
        # )
    except Exception as e:
        await message.answer(f"⚠️ Ошибка: {str(e)[:400]}")
    finally:
        await kb.close()

async def handle_question(message: Message, chat_ai: TerraChatAI):
    try:
        # Правильный способ отправить действие "печатает"
        await message.bot.send_chat_action(
            chat_id=message.chat.id,
            action="typing"
        )
        
        answer = await chat_ai.generate_answer(message.text)
        await message.answer(answer)
    except Exception as e:
        await message.answer("⚠️ Ошибка обработки запроса")
        logging.error(f"Error: {str(e)}")

class DependencyMiddleware:
    def __init__(self, config: Config, chat_ai: TerraChatAI):
        self.config = config
        self.chat_ai = chat_ai

    async def __call__(self, handler, event, data):
        data["config"] = self.config
        data["chat_ai"] = self.chat_ai
        return await handler(event, data)

def register_handlers(dp: Dispatcher, config: Config):
    # Инициализация сервисов
    chat_ai = TerraChatAI(config)
    
    # Установка middleware
    dp.message.middleware(DependencyMiddleware(config, chat_ai))
    
    # Регистрация обработчиков
    dp.message.register(cmd_start, Command("start"))
    dp.message.register(cmd_update, Command("update"))
    dp.message.register(
    handle_question, 
    F.text,
    ~Command("start"),
    ~Command("update")
)