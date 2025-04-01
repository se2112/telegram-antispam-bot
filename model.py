import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Загрузка данных
spam_data = pd.read_csv('/home/basicx1000/spam/filtered_spam_messages.csv')
not_spam_data = pd.read_csv('/home/basicx1000/spam/not_spam_messages.csv')

# Объединение данных
data = pd.concat([spam_data, not_spam_data], ignore_index=True)
data['label'] = data['label'].map({'spam': 1, 'not spam': 0})

# Проверка NaN
data['message'] = data['message'].fillna("")

# Инициализация русской модели BERT
tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
bert_model = BertModel.from_pretrained('DeepPavlov/rubert-base-cased')

# Кастомный датасет
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

# Подготовка данных
max_len = 128
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)
train_dataset = SpamDataset(X_train.values, y_train.values, tokenizer, max_len)
test_dataset = SpamDataset(X_test.values, y_test.values, tokenizer, max_len)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Определение модели
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

# Инициализация модели, оптимизатора и функции потерь
model = SpamClassifier(bert_model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# Функция обучения
def train_epoch(model, data_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        _, preds = torch.max(outputs, dim=1)
        correct += torch.sum(preds == labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return total_loss / len(data_loader), correct.double() / len(data_loader.dataset)

# Функция оценки
def eval_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)
            correct += torch.sum(preds == labels)
    
    return total_loss / len(data_loader), correct.double() / len(data_loader.dataset)

# Обучение модели
epochs = 3
for epoch in range(epochs):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = eval_model(model, test_loader, criterion)
    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

# Сохранение модели
torch.save(model.state_dict(), "spam_classifier_weights.pth")  # Сохранение весов
torch.save(model, "spam_classifier_model.pth")  # Сохранение всей модели

# Функция проверки текста
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

# Проверка текста с циклом
while True:
    input_text = input("Введите текст для проверки (или введите 'exit' для выхода): ")
    if input_text.lower() == 'exit':
        print("Выход из программы.")
        break
    result = check_spam_bert(input_text)
    print(f"Результат: {result}")
