import os
import logging
from aiogram import Bot, Dispatcher
from handlers.commands import register_handlers
from utils.config import Config

# Настройка логгирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    # Загрузка конфигурации
    config = Config()
    
    # Инициализация бота
    bot = Bot(token=config.TELEGRAM_BOT_TOKEN)
    dp = Dispatcher()
    
    # Регистрация обработчиков
    register_handlers(dp, config)
    
    # Запуск бота
    await dp.start_polling(bot)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())