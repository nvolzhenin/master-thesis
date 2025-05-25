import logging
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from predictor import MatchPredictor


# Токен Telegram-бота
TELEGRAM_TOKEN = "my-token"

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Команда /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Отправь двух игроков в формате: Имя1 vs Имя2"
    )

# Обработка текстовых сообщений
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()

    if "vs" not in text:
        await update.message.reply_text("Введите двух игроков в формате: `Игрок1 vs Игрок2`")
        return

    p1_name, p2_name = map(str.strip, text.split("vs", 1))
    model = MatchPredictor()
    result = model.predict(p1_name, p2_name)

    # Формируем сообщение
    msg = f"🎾 Предсказание матча:\n"
    msg += f"👤 {result['player_1']} vs {result['player_2']}\n"
    msg += f"📊 Счёт: {result['predicted_score']}\n"
    msg += f"🕒 Время: {result['predicted_minutes']} мин\n"

    # Добавим примечание, если есть
    if result.get("note"):
        msg += f"\n📝 {result['note']}"

    await update.message.reply_text(msg)


# Обработка ошибок
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(msg="Произошла ошибка:", exc_info=context.error)

# Главная точка входа
def main() -> None:
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)

    print("✅ Бот запущен. Ожидаю сообщения...")
    app.run_polling()

if __name__ == "__main__":
    main()