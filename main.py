import logging
from pyrogram import Client
from handlers.commands import register_handlers
from utils.config import Config

# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bot.log", encoding='utf-8'),
        logging.StreamHandler()
    ],
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

async def main():
    config = Config()
    app = Client("terra_bot", api_id=config.API_ID, api_hash=config.API_HASH)
    
    # Регистрация обработчиков
    register_handlers(app, config)
    
    # Запуск бота
    await app.start()
    await asyncio.Event().wait()  # Бесконечное ожидание

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())