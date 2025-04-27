import asyncio
import logging
from typing import Optional
from pyrogram import filters
from pyrogram.types import Message
from pyrogram.client import Client
from pyrogram import enums
from service.chatai import TerraChatAI
from service.knowledge_base import KnowledgeBase
from utils.config import Config
from pyrogram.enums import ChatType

logger = logging.getLogger(__name__)

async def mark_as_read(client: Client, message: Message):
    """Пометить сообщение как прочитанное"""
    try:
        await client.read_chat_history(message.chat.id)
        # Или для конкретного сообщения (если поддерживается API):
        # await client.view_messages(message.chat.id, message.id)
    except Exception as e:
        logger.warning(f"Не удалось пометить сообщение как прочитанное: {e}")

def register_handlers(app: Client, config: Config):
    chat_ai = TerraChatAI(config,client=app)
    kb = KnowledgeBase(config,app)

    @app.on_message(filters.command("start"))
    async def start(client: Client, message: Message):
        await mark_as_read(client, message)
        await message.reply("🤖 Я AI-ассистент TERRA. Задайте вопрос о наставниках или заявках.")

    @app.on_message(filters.command("update") & filters.user(config.ADMINS))
    async def update(client: Client, message: Message):
        await mark_as_read(client, message)
        kb = KnowledgeBase(config,client=client)
        try:
            await kb.update_all_sources(
                bot=client,
                chat_id=message.chat.id,
                telegram_days_offset=config.DAYS_OFFSET,
            )
        except Exception as e:
            await message.reply(f"⚠️ Ошибка: {str(e)[:400]}")
        finally:
            await kb.close()

    @app.on_message(filters.command("update_llm") & filters.user(config.ADMINS))
    async def update_with_llm(client: Client, message: Message):
        """Обновление с использованием LLM-разметки"""
        try:
            kb = KnowledgeBase(config, client=client)
            count = await kb.update_from_telegram(days_offset=30)
            
            await message.reply(
                f"🔄 Обновлено с LLM-разметкой\n"
                f"• Добавлено документов: {count}\n"
                f"• Источники: Telegram"
            )
        except Exception as e:
            logger.error(f"LLM update failed: {e}")
            await message.reply(f"⚠️ Ошибка: {str(e)[:200]}")

    @app.on_message(filters.command("kb_stats") & filters.user(config.ADMINS))
    async def kb_stats_handler(client: Client, message: Message):
        """Показывает расширенную статистику базы знаний с примерами"""
        def truncate_example(text: str, max_length: int = 80) -> str:
            """Обрезает пример для отображения в телеграме (вспомогательная функция)"""
            text = text.replace('\n', ' ')
            if len(text) > max_length:
                return text[:max_length]
            return text

        try:
            # Получаем данные с примерами
            sources = await kb.get_all_sources()
            if not sources:
                await message.reply("📭 База знаний пуста")
                return

            # Формируем красивый ответ с Markdown-разметкой
            response = [
                "📚 <b>Статистика базы знаний</b>",
                "",
                f"<b>Всего источников:</b> {len(sources)}",
                f"<b>Общее количество записей:</b> {sum(src['count'] for src in sources)}",
                ""
            ]

            # Добавляем информацию по каждому источнику
            for src in sorted(sources, key=lambda x: x['count'], reverse=True):
                response.extend([
                    "━━━━━━━━━━━━━━━━━",
                    f"🔹 <b>{src['source'].upper()}</b> [<code>{src['count']}</code>]",
                    ""
                ])
                
                # Добавляем примеры (первые 3 из 5)
                for example in src['examples'][:5]:
                    response.append(f"▪️ {truncate_example(example)}")  # Используем локальную функцию
                
                response.append("")

            # Разбиваем сообщение если слишком длинное
            message_text = "\n".join(response)
            if len(message_text) > 4000:
                parts = [message_text[i:i+4000] for i in range(0, len(message_text), 4000)]
                for part in parts:
                    await message.reply(part)
                    await asyncio.sleep(1)  # Пауза между сообщениями
            else:
                await message.reply(message_text)

        except Exception as e:
            logger.error(f"Ошибка в kb_stats: {e}", exc_info=True)
            await message.reply("⚠️ Ошибка получения статистики. Подробности в логах.")

    @app.on_message(
        filters.text 
        & ~filters.command("start") 
        & ~filters.command("update")
    )
    async def handle_question(client: Client, message: Message):
        """Обработчик вопросов с полной защитой от пустых сообщений"""
        try:
            # 1. Проверка чата и пользователя
            if message.chat.id != -1001945870336:
                return

            if not message.from_user or not message.text:
                await _safe_reply(message, "Пожалуйста, отправьте текстовый вопрос")
                return
            await mark_as_read(client, message)
            # 2. Подготовка вопроса
            user_id = message.from_user.id
            question = message.text.strip()
            
            if len(question) < 2:
                await _safe_reply(message, "Вопрос слишком короткий")
                return

            logger.info(f"Обработка вопроса от {user_id}: {question[:100]}...")

            # 3. Отправка действия "печатает"
            await client.send_chat_action(message.chat.id, enums.ChatAction.TYPING)
            await asyncio.sleep(5)  # Задержка для имитации печати

            # 4. Генерация ответа
            try:
                answer = await asyncio.wait_for(
                    chat_ai.generate_answer(user_id, question),
                    timeout=20.0
                )
                answer = _validate_message(answer)
            except asyncio.TimeoutError:
                logger.warning(f"Таймаут для {user_id}")
                answer = "Извините, не успел обработать запрос. Попробуйте позже."
            except Exception as e:
                logger.error(f"Ошибка генерации: {e}")
                answer = "Произошла ошибка при обработке вопроса"

            # 5. Гарантированная отправка ответа
            await _safe_reply(message, answer)
            logger.info(f"Ответ пользователю {user_id} отправлен")

        except Exception as e:
            logger.error(f"Критическая ошибка: {e}", exc_info=True)
            await _safe_reply(message, "Произошла непредвиденная ошибка")

    async def _safe_reply(message: Message, text: str) -> bool:
        """Безопасная отправка сообщения с обработкой всех ошибок"""
        try:
            text = _validate_message(text)
            if not text:
                return False
                
            await message.reply(text)
            return True
        except Exception as e:
            logger.error(f"Ошибка отправки сообщения: {e}")
            return False

    def _validate_message(text: Optional[str]) -> str:
        """Проверка и очистка текста сообщения"""
        if not text:
            return "Извините, не получилось сформировать ответ"
            
        # Удаляем непечатаемые символы
        cleaned = "".join(c for c in str(text) if c.isprintable() or c in "\n\r\t")
        cleaned = cleaned.strip()
        
        # Заменяем полностью пустые сообщения
        if not cleaned:
            return "Ответ не содержит текста"
            
        # Обрезаем слишком длинные сообщения
        return cleaned[:4000]


    