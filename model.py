import os
import telebot
from telebot.types import Message, ChatPermissions
import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from datetime import datetime
import time
import threading
import re

# Определяем пути к файлам
#BASE_DIR = "/home/basicx1000/Рабочий стол/бот_ан_ind/nadya2"
BASE_DIR = "/root/nadya2"
LOG_FILE = os.path.join(BASE_DIR, "history1.csv")
MODEL_PATH = os.path.join(BASE_DIR, "spam_classifier_weights.pth")

# Убеждаемся, что директория существует
os.makedirs(BASE_DIR, exist_ok=True)

# Загрузка модели и токенизатора DeepPavlov
class SpamClassifier(nn.Module):
    def __init__(self, bert_model, dropout=0.3):
        super(SpamClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.fc(output)

# Используем DeepPavlov BERT
bert_model = AutoModel.from_pretrained('DeepPavlov/rubert-base-cased')
model = SpamClassifier(bert_model)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Инициализация токенизатора
tokenizer = AutoTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')

# Telegram Bot API Token
API_TOKEN = "7701845953:AAG_rayKNbYuzDRtnbcSpMGMOrBteTHurcI"
bot = telebot.TeleBot(API_TOKEN)

# Логирование сообщений
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["date", "chat_id", "chat_name", "user_id", "username", "message", "is_spam", "contains_tg_link", "deleted", "banned"]).to_csv(LOG_FILE, index=False)

def log_message(data):
    df = pd.DataFrame([data])
    df.to_csv(LOG_FILE, mode="a", header=False, index=False)

# Проверка сообщения на спам
def check_spam(message_text):
    encoding = tokenizer.encode_plus(
        message_text,
        max_length=128,
        truncation=True,
        padding='max_length',
        add_special_tokens=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    outputs = model(input_ids, attention_mask)
    _, preds = torch.max(outputs, dim=1)
    return preds[0].item() == 1  # Возвращает True, если спам

import re

def contains_telegram_link(text):
    return bool(re.search(r'(@\w+|t\.me/\S+|\bлс\b|\bлc\b|\bличное сообщение\b|\bличные сообщения\b|\bличным сообщением\b|личных сообщениях)', text, re.IGNORECASE))
#def contains_telegram_link(text):
    #return bool(re.search(r'(@\w+|t\.me/\S+|лс)', text, re.IGNORECASE))

# Проверка, является ли пользователь администратором
def is_admin(chat_id, user_id):
    try:
        chat_admins = bot.get_chat_administrators(chat_id)
        return any(admin.user.id == user_id for admin in chat_admins)
    except Exception as e:
        print(f"Ошибка при проверке администратора: {e}")
        return False

# Блокировка пользователя в чате
def ban_user(chat_id, user_id):
    try:
        bot.restrict_chat_member(
            chat_id,
            user_id,
            ChatPermissions(
                can_send_messages=False,
                can_send_media_messages=False,
                can_send_other_messages=False,
                can_add_web_page_previews=False
            )
        )
    except Exception as e:
        print(f"Ошибка блокировки пользователя {user_id} в чате {chat_id}: {e}")

import threading

# Функция удаления предупреждения без задержки основной работы бота
def delete_warning_later(chat_id, message_id, delay=12):
    def delayed_deletion():
        time.sleep(delay)
        try:
            bot.delete_message(chat_id, message_id)
        except Exception as e:
            print(f"Ошибка удаления предупреждения: {e}")

    threading.Thread(target=delayed_deletion, daemon=True).start()





import os
import telebot
from telebot.types import Message
import pandas as pd
import time


ADMIN_ID = 5518423134  # Только этот админ может отправлять сообщения во все чаты



# Функция для получения уникальных чатов, где используется бот
def get_unique_chat_ids():
    try:
        df = pd.read_csv(LOG_FILE)
        unique_chat_ids = df["chat_id"].dropna().unique().tolist()
        return [int(chat_id) for chat_id in unique_chat_ids]
    except Exception as e:
        print(f"Ошибка при загрузке chat_id: {e}")
        return []

# Команда для рассылки сообщений во все чаты
@bot.message_handler(commands=["broadcast"])
def broadcast_message(message: Message):
    if message.from_user.id != ADMIN_ID:
        bot.reply_to(message, "❌ У вас нет прав для выполнения этой команды.")
        return

    try:
        text_to_send = message.text.replace("/broadcast", "").strip()
        if not text_to_send:
            bot.reply_to(message, "⚠️ Пожалуйста, укажите текст сообщения после команды.")
            return

        chat_ids = get_unique_chat_ids()
        if not chat_ids:
            bot.reply_to(message, "⚠️ Не найдено чатов для рассылки.")
            return

        for chat_id in chat_ids:
            try:
                bot.send_message(chat_id, text_to_send)
                print(f"✅ Сообщение отправлено в чат {chat_id}")
                time.sleep(0.5)  # Задержка для избежания ограничений Telegram
            except Exception as e:
                print(f"❌ Ошибка при отправке в чат {chat_id}: {e}")

        bot.reply_to(message, "✅ Сообщение успешно отправлено во все чаты.")
    except Exception as e:
        print(f"❌ Ошибка при массовой рассылке в чаты: {e}")
        bot.reply_to(message, "❌ Произошла ошибка при отправке сообщений.")


# Обработка текстовых сообщений
@bot.message_handler(content_types=["text"])
def handle_messages(message: Message):
    try:
        chat_id = message.chat.id
        user_id = message.from_user.id
        username = message.from_user.username or "Unknown"
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        chat_name = message.chat.title if message.chat.type != "private" else "Private"

        # Проверяем, является ли отправитель администратором чата
        if is_admin(chat_id, user_id):
            return  # Не фильтруем администраторов

        is_spam = check_spam(message.text)  # Проверяем сообщение на спам
        has_telegram_link = contains_telegram_link(message.text)  # Проверяем наличие Telegram-ссылки

        # Новая логика:
        should_delete = is_spam  # Удаляем, если это спам
        should_ban = is_spam and has_telegram_link  # Баним, если спам + Telegram-ссылка

        # Логируем сообщение
        log_entry = {
            "date": date,
            "chat_id": chat_id,
            "chat_name": chat_name,
            "user_id": user_id,
            "username": username,
            "message": message.text,
            "is_spam": is_spam,
            "contains_tg_link": has_telegram_link,
            "deleted": should_delete,
            "banned": should_ban
        }
        log_message(log_entry)

        # Удаляем запрещенные сообщения
        if should_delete:
            bot.delete_message(chat_id, message.message_id)

            warn_text = f"🚨 @{username}, ваше сообщение было удалено!\n"
            if is_spam:
                warn_text += "❗ Причина: обнаружен СПАМ.\n"

            warn_text += "\n🤖 Бот для 10-го шага.\n📝 Ежедневный самоанализ: @stepna12_bot"

            warn_msg = bot.send_message(chat_id, warn_text)

            # Запускаем поток для удаления предупреждения
            threading.Thread(target=delete_warning_later, args=(chat_id, warn_msg.message_id), daemon=True).start()

        # Блокируем пользователя, если он отправил спам со ссылкой
        # Блокируем пользователя, если он отправил спам со ссылкой
        if should_ban:
            ban_user(chat_id, user_id)
            ban_msg = bot.send_message(
                chat_id, 
                f"🚫 @{username}, вы **заблокированы** в этом чате за отправку СПАМА !\n"
                "🔒 Для разблокировки свяжитесь с администратором.\n"
                "🤖 Бот для 10-го шага.\n📝 Ежедневный самоанализ: @stepna12_bot"
            )
            # Удаляем предупреждение через 60 секунд
            threading.Thread(target=delete_warning_later, args=(chat_id, ban_msg.message_id, 60), daemon=True).start()

    except Exception as e:
        print(f"Ошибка обработки сообщения: {e}")




# Запуск бота
print("Бот запущен...")
bot.polling(none_stop=True)

