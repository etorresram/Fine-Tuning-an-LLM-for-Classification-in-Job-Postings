# Optimized and cleaner version of your DistilBERT job sector classification script

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)

# Config
PRE_TRAINED_MODEL_NAME = 'distilbert-base-cased'
MAX_LEN = 200
BATCH_SIZE = 16
EPOCHS = 5
RANDOM_SEED = 42
DATASET_PATH = '/Users/etorresram/Desktop/gitFolder/jobs_sectors.csv'

# Reproducibility
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Load and preprocess data
df = pd.read_csv(DATASET_PATH)
df['label'] = LabelEncoder().fit_transform(df['naics_code'])

# Train-test split
df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=RANDOM_SEED)

# Tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME, num_labels=21).to(device)

# Dataset class
class JobDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = [str(t) if pd.notnull(t) else "" for t in texts]
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, i):
        enc = self.tokenizer(
            self.texts[i],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[i], dtype=torch.long)
        }

# DataLoaders
train_loader = DataLoader(JobDataset(df_train['description'].values, df_train['label'].values, tokenizer, MAX_LEN), batch_size=BATCH_SIZE)
test_loader = DataLoader(JobDataset(df_test['description'].values, df_test['label'].values, tokenizer, MAX_LEN), batch_size=BATCH_SIZE)

# Optimizer, scheduler, loss
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
loss_fn = torch.nn.CrossEntropyLoss().to(device)

# Training loop
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler):
    model.train()
    total_loss, correct = 0, 0
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        loss = loss_fn(logits, labels)
        _, preds = torch.max(logits, dim=1)
        correct += torch.sum(preds == labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct.double() / len(data_loader.dataset), total_loss / len(data_loader)

# Evaluation loop
def eval_model(model, data_loader, loss_fn, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = loss_fn(logits, labels)
            _, preds = torch.max(logits, dim=1)
            correct += torch.sum(preds == labels)
            total_loss += loss.item()

    return correct.double() / len(data_loader.dataset), total_loss / len(data_loader)

# Train
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    train_acc, train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, scheduler)
    test_acc, test_loss = eval_model(model, test_loader, loss_fn, device)

    print(f"Train loss: {train_loss:.4f}, accuracy: {train_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}, accuracy: {test_acc:.4f}\n")
