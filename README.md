---
# Telegram Anti-Spam Bot

##  Описание проекта
Этот проект представляет собой антиспам-бота для Telegram, использующего нейросетевую модель на основе BERT для детекции спама. Бот анализирует текстовые сообщения и определяет, являются ли они спамом или нет.

## 🚀 Возможности
- Классификация сообщений как "спам" или "не спам"
- Обучение модели на основе предобученного `DeepPavlov/rubert-base-cased`
- Работа с Telegram API
- Поддержка GPU для ускоренного обучения

##  Установка и запуск
### 1. Клонирование репозитория
```bash
git clone https://github.com/ВАШ_GITHUB/telegram-antispam-bot.git
cd telegram-antispam-bot
```

### 2. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 3. Запуск бота
```bash
python bot/main.py
```

## 📂 Структура проекта
```
telegram-antispam-bot/
│── bot/                 # Код Telegram-бота
│── model/               # Код обучения/инференса модели
│── data/                # Датасет (спам/не-спам примеры)
│── weights/             # Веса модели (не хранятся в репозитории)
│── requirements.txt     # Зависимости проекта
│── README.md            # Описание проекта
│── .gitignore           # Исключения для Git
│── LICENSE              # Лицензия
```

##   Обучение модели
### 1. Подготовка данных
Датасет хранится в `data/spam_examples.csv` и `data/non_spam_examples.csv`.

Пример структуры CSV:
```csv
message,label
"Вы выиграли 1000$!",spam
"Привет, как дела?",not_spam
```

### 2. Запуск обучения
```bash
python model/train.py
```
После обучения веса сохраняются в `weights/spam_classifier_model.pth`.

## 🔍 Проверка сообщения на спам
После обучения можно использовать функцию:
```python
from model.inference import check_spam_bert
text = "Вы выиграли бесплатный iPhone!"
print(check_spam_bert(text))  # Спам
```

##  Лицензия
Этот проект распространяется по лицензии MIT. Подробнее см. [LICENSE](LICENSE).




---
