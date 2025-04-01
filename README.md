---
# Telegram Anti-Spam Bot

## Описание проекта
Этот проект представляет собой антиспам-бота для Telegram, использующего нейросетевую модель на основе BERT для детекции спама. Бот анализирует текстовые сообщения и определяет, являются ли они спамом или нет.

## Возможности
- Классификация сообщений как "спам" или "не спам"
- Обучение модели на основе предобученного `DeepPavlov/rubert-base-cased`
- Работа с Telegram API
- Поддержка GPU для ускоренного обучения

## Установка и запуск
### 1. Клонирование репозитория
```bash
git clone https://github.com/se2112/telegram-antispam-bot.git
cd telegram-antispam-bot

2. Установка зависимостей

pip install -r requirements.txt


3. Запуск бота

python bot_telegram.py

Структура проекта

telegram-antispam-bot/
│── bot_telegram.py         # Основной код бота
│── model.py                # Код модели
│── requirements.txt        # Зависимости проекта
│── filtered_spam_messages.csv # Пример спам сообщений
│── not_spam_messages.csv   # Пример не спам сообщений
│── spam_classifier_weights.pth # Веса обученной модели
│── README.md               # Описание проекта
│── TRAINING.md             # Документация по обучению

Проверка работы бота

Для проверки работы бота установите @addictedNadya_bot в качестве администратора в чат и предоставьте доступ к сообщениям.

Метрики обучения

После 3-х эпох обучения модель показала следующие результаты:

Epoch 3/3

    Train Loss: 0.0408

    Train Acc: 0.9865

    Val Loss: 0.0525

    Val Acc: 0.9872


##  Проверка сообщения на спам
После обучения можно использовать функцию:
```python
from model.inference import check_spam_bert
text = "Вы выиграли бесплатный iPhone!"
print(check_spam_bert(text))  # Спам
```

##  Лицензия
Этот проект распространяется по лицензии MIT. Подробнее см. [LICENSE](LICENSE).




---
