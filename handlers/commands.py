import asyncio
from datetime import datetime
import logging
import os
from typing import Optional
from dotenv import load_dotenv
from pyrogram import filters
from pyrogram.types import Message
from pyrogram.client import Client
from pyrogram import enums
from service.chatai import TerraChatAI
from service.knowledge_base import KnowledgeBase
from utils.config import Config
from pyrogram.enums import ChatType
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class ResponseMode:
    TEST = "test"          # Только тестовый чат
    WHITELIST = "whitelist" # Только чаты из белого списка
    PRIVATE = "private"    # Только личные сообщения
    ALL = "all"            # Все чаты (осторожно!)

async def mark_as_read(client: Client, message: Message):
    """Пометить сообщение как прочитанное"""
    try:
        await client.read_chat_history(message.chat.id)
    except Exception as e:
        logger.warning(f"Не удалось пометить сообщение как прочитанное: {e}")

def register_handlers(app: Client, config: Config):
    chat_ai = TerraChatAI(config,client=app)
    kb = KnowledgeBase(config,app)

    def should_respond(message: Message) -> bool:
        """Определяет, должен ли бот отвечать на сообщение в зависимости от режима"""
        load_dotenv('.env')
        regime = os.getenv('REGIME', ResponseMode.TEST)
        chat_id = message.chat.id
        
        if regime == ResponseMode.TEST:
            return chat_id == -1001945870336  # Только тестовый чат
            
        elif regime == ResponseMode.WHITELIST:
            # Проверка по ID чатов из конфига
            if chat_id in config.ALLOWED_CHAT_IDS:
                return True
            # Проверка по username/ссылке если есть
            if hasattr(message.chat, 'username') and message.chat.username:
                return message.chat.username in config.ALLOWED_CHAT_USERNAMES
            return False
            
        elif regime == ResponseMode.PRIVATE:
            return message.chat.type == ChatType.PRIVATE
            
        elif regime == ResponseMode.ALL:
            return True
            
        return False

    @app.on_message(filters.command("start"))
    async def start(client: Client, message: Message):
        await mark_as_read(client, message)
        await message.reply("🤖 Я AI-ассистент TERRA. Задайте вопрос о наставниках или заявках.")

    @app.on_message(filters.command("update") & filters.user(config.ADMINS))
    async def update(client: Client, message: Message):
        await mark_as_read(client, message)
        kb = KnowledgeBase(config,client=client)
        chat_ai = TerraChatAI(config, client=client)  # Добавляем инициализацию chat_ai
        
        try:
            parts = message.text.split(maxsplit=1)
            if len(parts) < 2:
                await message.reply("Используйте: /update (regime)\nДоступные режимы: all, sheets, telegram")
                return
                
            regime = parts[1].strip().lower()
            if regime not in ["all", "sheets", "telegram"]:
                await message.reply("Недопустимый режим. Используйте: all, sheets, telegram")
                return

            
                       

            # Обновление базы знаний
            await kb.update_all_sources(
                bot=client,
                chat_id=message.chat.id,
                telegram_days_offset=config.DAYS_OFFSET,
                regime=regime
            )
            
            # Если режим all или sheets - обновляем промпт из таблицы
            if regime in ["all", "sheets"]:
                try:
                    await message.reply("🔄 Проверяю обновления промпта в таблице...")
                    prompt = await kb.get_prompt_from_sheets()
                    logger.info(prompt)
                    if prompt and prompt.strip():
                        if chat_ai.update_prompt(prompt.strip()):
                            await message.reply("✅ Промпт успешно обновлен из таблицы!")
                        else:
                            await message.reply("⚠️ Промпт найден, но не удалось его обновить")
                    else:
                        await message.reply("ℹ️ В таблице не найден промпт для обновления")
                except Exception as e:
                    await message.reply(f"⚠️ Ошибка при обновлении промпта: {str(e)[:200]}")
                    
        except Exception as e:
            await message.reply(f"⚠️ Ошибка: {str(e)[:400]}")
        finally:
            await asyncio.sleep(3)
            await kb.close()

    @app.on_message(filters.command("export_kb") & filters.user(config.ADMINS))
    async def export_knowledge_base(client: Client, message: Message):
        """Экспортирует базу знаний в JSON файл и отправляет его администратору."""
        try:
            await message.reply("🔄 Начинаю экспорт базы знаний...")
            
            kb = KnowledgeBase(config, client=client)
            file_path = await kb.export_to_json()
            
            if not file_path or not os.path.exists(file_path):
                await message.reply("❌ Не удалось создать файл экспорта")
                return

            # Отправляем файл
            await client.send_document(
                chat_id=message.chat.id,
                document=file_path,
                caption=f"📚 Экспорт базы знаний ({datetime.now().strftime('%Y-%m-%d')})"
            )
            
            # Удаляем временный файл
            os.unlink(file_path)
            
        except Exception as e:
            logger.error(f"Export KB error: {e}")
            await message.reply(f"⚠️ Ошибка экспорта: {str(e)[:200]}")        

    @app.on_message(filters.command("remove_duplicates") & filters.user(config.ADMINS))
    async def handle_remove_duplicates(client: Client, message: Message):
        """Удаляет дубликаты из базы знаний."""
        try:
            await message.reply("🔍 Поиск дубликатов... (это может занять время)")
            
            kb = KnowledgeBase(config, client=client)
            result = await kb.remove_duplicates()
            
            if "error" in result:
                await message.reply(f"❌ Ошибка: {result['error']}")
                return
            
            report = (
                "✅ Дубликаты удалены:\n"
                f"• Всего записей: {result['total']}\n"
                f"• Удалено дубликатов: {result['removed']}\n"
                f"• Осталось уникальных: {result['remaining']}"
            )
            await message.reply(report)
            
        except Exception as e:
            logger.error(f"Ошибка в /remove_duplicates: {e}")
            await message.reply("⚠️ Ошибка. Подробности в логах.")

    @app.on_message(filters.command("help"))
    async def help(client: Client, message: Message):
        help_text = """
        📌 **Доступные команды**:
        - `/start` - Начало работы/тестовое сообщение.
        - `/help` - Эта справка.
        Для администраторов:
        - `/update (regime)` - Обновить базу знаний из: all - всех источников, sheets - таблиц, telegram - чатов.
        - `/kb_stats` - Статистика базы знаний.
        - `/add_qa` - Добавить Вопрос:Ответ вручную.
        - `/export_kb` - экспорт в JSON
        - `/remove_duplicates` — удалить дубликаты
         - `/get_prompt` - показать текущий промпт
        - `/set_prompt` - установить новый промпт (ответьте на сообщение с промптом)
        """
        await message.reply(help_text)

    @app.on_message(filters.command("add_qa") & filters.user(config.ADMINS))
    async def add_qa(client: Client, message: Message):
        # Пример: /add_qa Вопрос? Ответ!
        parts = message.text.split(maxsplit=1)
        if len(parts) < 2:
            await message.reply("Используйте: /add_qa Вопрос? Ответ!")
            return
        
        q, a = parts[1].split("?", maxsplit=1)
        doc = Document(
            page_content=q.strip(),
            metadata={"answer": a.strip(), "source": "manual"}
        )
        await kb.vectorstore.aadd_documents([doc])
        await message.reply("✅ Q/A добавлено в базу знаний!")

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

    @app.on_message(filters.command("get_prompt") & filters.user(config.ADMINS))
    async def get_prompt(client: Client, message: Message):
        """Показывает текущий промпт"""
        try:
            prompt = await kb.get_prompt_from_sheets()
            if prompt:
                chat_ai.update_prompt(prompt.strip())
            if len(prompt) > 4000:
                parts = [prompt[i:i+4000] for i in range(0, len(prompt), 4000)]
                for part in parts:
                    await message.reply(f"<code>{part}</code>", parse_mode=enums.ParseMode.HTML)
                    await asyncio.sleep(1)
            else:
                await message.reply(f"<code>{prompt}</code>", parse_mode=enums.ParseMode.HTML)
        except Exception as e:
            await message.reply(f"Ошибка: {str(e)}")

   
    @app.on_message(filters.command("debug_search") & filters.user(config.ADMINS))
    async def debug_search(client: Client, message: Message):
        try:
            query = message.text.split(maxsplit=1)[1]
            # Используем новый параметр with_scores вместо include_scores
            results = await kb.search(query, k=5)
            
            if not results:
                await message.reply("🔍 Ничего не найдено")
                return
                
            response = ["🔍 Результаты поиска в векторной БД:"]
            for doc, score in results:
                source = doc.metadata.get("source", "unknown")
                response.append(
                    f"• [{source}] {doc.page_content[:80]}... (score: {score:.2f})"
                )
            
            await message.reply("\n".join(response)[:4000])
        except Exception as e:
            await message.reply(f"⚠️ Ошибка: {str(e)}")

    

    @app.on_message(filters.command("set_prompt") & filters.user(config.ADMINS))
    async def set_prompt(client: Client, message: Message):
        """Устанавливает новый промпт"""
        try:
            message_text = message.text.split(maxsplit=1)
                
            new_prompt = message_text[1]
            if chat_ai.update_prompt(new_prompt):
                await message.reply("✅ Промпт успешно обновлен!")
            else:
                await message.reply("❌ Не удалось обновить промпт")
        except Exception as e:
            await message.reply(f"Ошибка: {str(e)}")


    @app.on_message(
        filters.text 
        & ~filters.command("start") 
        & ~filters.command("update")
        & ~filters.command("get_prompt")
        & ~filters.command("set_prompt")
    )
    async def handle_question(client: Client, message: Message):
        """Обработчик вопросов с полной защитой от пустых сообщений"""
        try:
            load_dotenv('.env')
            regime = os.getenv('REGIME', 'test')
            # if regime == 'test':
            #     if message.chat.id != -1001945870336:
            #         return
            # Проверяем, должен ли бот отвечать в этом чате
            if not should_respond(message):
                return

            if not message.from_user or not message.text:
                await _safe_reply(message, "Пожалуйста, отправьте текстовый вопрос")
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
            await asyncio.sleep(15)  # Задержка для имитации печати
            await client.send_chat_action(message.chat.id, enums.ChatAction.TYPING)
            

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

    def _validate_message(text: Optional[str]) -> Optional[str]:
        """Проверка и очистка текста сообщения с полным игнорированием стоп-слов"""
        if not text:
            return None
            
        # Удаляем непечатаемые символы
        cleaned = "".join(c for c in str(text) if c.isprintable() or c in "\n\r\t")
        cleaned = cleaned.strip()
        
        if not cleaned:
            return None
        
        # Список стоп-слов для полного игнорирования ответа
        STOP_PHRASES = {
            "извините",
            "не знаю", 
            "не могу",
            "нет информации",
            "не удалось",
            "я не",
            "мне неизвестно",
            "не понимаю",
            "не имею данных",
            "не располагаю информацией"
        }

        # Проверка на стоп-фразы (без учета регистра)
        lower_text = cleaned.lower()
        for phrase in STOP_PHRASES:
            if phrase in lower_text:
                return None  # Полное игнорирование ответа
        
        # Дополнительная проверка "пустых" ответов
        if len(cleaned.split()) < 3 and any(w in cleaned.lower() for w in ["нет", "не", "ничего"]):
            return None

        # Обрезаем слишком длинные сообщения
        return cleaned[:4000]


    