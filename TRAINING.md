# Обучение модели для фильтрации спама

Этот проект использует модель BERT для классификации сообщений как спам или не спам. Модель обучается на данных о сообщениях, включая метки "spam" и "not spam". Ниже приведено описание процесса обучения модели и шагов, которые необходимо выполнить для запуска и обучения модели.

## 1. Загрузка данных

Для обучения модели используются два набора данных:
- **spam_data**: Сообщения, которые были помечены как спам.
- **not_spam_data**: Сообщения, которые не являются спамом.

Данные загружаются с использованием библиотеки `pandas` из CSV-файлов.

```python
spam_data = pd.read_csv('/path/to/spam/filtered_spam_messages.csv')
not_spam_data = pd.read_csv('/path/to/spam/not_spam_messages.csv')

После загрузки данные объединяются и метки spam заменяются на 1, а метки not spam — на 0.

2. Предобработка данных

Перед обучением данные проходят предобработку:

    Пропущенные значения в столбце с текстом заполняются пустыми строками.

    Все данные токенизируются с использованием модели DeepPavlov/rubert-base-cased.
    
3. Создание кастомного датасета

Для обучения и тестирования создается класс SpamDataset, который использует токенизатор BERT для преобразования текста в формат, подходящий для модели.

class SpamDataset(Dataset):
    def __init__(self, messages, labels, tokenizer, max_len):
        self.messages = messages
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx):
        message = str(self.messages[idx])
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            message,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

4. Определение модели

Модель представляет собой класс SpamClassifier, который использует предобученную модель BERT для классификации сообщений как спам или не спам.

class SpamClassifier(nn.Module):
    def __init__(self, bert_model, dropout=0.3):
        super(SpamClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # Используем [CLS] токен
        output = self.dropout(pooled_output)
        return self.fc(output)


5. Обучение модели

Обучение проводится с использованием оптимизатора AdamW и функции потерь CrossEntropyLoss. Для каждого эпоха выводятся показатели потерь и точности на обучающей и валидационной выборках.
Результаты обучения:

    Epoch 1/3: Train Loss: 0.1033, Train Acc: 0.9651, Val Loss: 0.0432, Val Acc: 0.9872

    Epoch 2/3: Train Loss: 0.0367, Train Acc: 0.9891, Val Loss: 0.0708, Val Acc: 0.9793

    Epoch 3/3: Train Loss: 0.0235, Train Acc: 0.9933, Val Loss: 0.0526, Val Acc: 0.9846
    
    
6. Сохранение модели

После завершения обучения веса модели сохраняются в файл spam_classifier_weights.pth.

7. Использование модели для предсказания

После обучения модель можно использовать для предсказания, является ли текст сообщением спамом или не спамом:

def check_spam_bert(text):
    model.eval()
    encoding = tokenizer.encode_plus(
        text,
        max_length=max_len,
        truncation=True,
        padding='max_length',
        add_special_tokens=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    outputs = model(input_ids, attention_mask)
    _, preds = torch.max(outputs, dim=1)
    return "Спам" if preds[0] == 1 else "Не спам"


8. Запуск проверки текста

while True:
    input_text = input("Введите текст для проверки (или введите 'exit' для выхода): ")
    if input_text.lower() == 'exit':
        print("Выход из программы.")
        break
    result = check_spam_bert(input_text)
    print(f"Результат: {result}")


Требования

Для успешного выполнения обучения и использования модели необходимо установить следующие библиотеки:

    torch

    transformers

    sklearn

    pandas

    numpy

Убедитесь, что у вас установлен GPU для ускорения обучения.

Лицензия

Проект распространяется под лицензией BSD 3-Clause License. См. файл LICENSE для подробностей.


    
