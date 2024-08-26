import torch
import torch.package
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sentiment import SentimentLSTM

BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
HIDDEN_SIZE = 512
EMBEDDING_SIZE = 200
DROPOUT = 0.3
GRAD_CLIP = 0.5
NUM_CLASSES = 28


class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]


def accuracy(outputs, labels):
    preds = outputs > 0.5
    corrects = (preds == labels).float().sum()
    return corrects / labels.numel()


datasets = [
    "datasets/goemotion_1.csv",
    "datasets/goemotion_2.csv",
    "datasets/goemotion_3.csv",
]
df = pd.concat([pd.read_csv(file_path, header=None) for file_path in datasets])
texts = df[0].values
labels = df.iloc[:, 1:].values
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2
)

vocab = {}
for text in train_texts:
    for word in text.split():
        if word not in vocab:
            vocab[word] = len(vocab) + 1

train_sequences = []
for text in train_texts:
    sequence = []
    for word in text.split():
        sequence.append(vocab[word])
    train_sequences.append(sequence)

val_sequences = []
for text in val_texts:
    sequence = []
    for word in text.split():
        sequence.append(vocab.get(word, 0))
    val_sequences.append(sequence)

max_len = max(len(s) for s in train_sequences)
train_sequences = [s + [0] * (max_len - len(s)) for s in train_sequences]
val_sequences = [s + [0] * (max_len - len(s)) for s in val_sequences]

train_sequences = pad_sequence([torch.tensor(seq) for seq in train_sequences], True)
train_labels = torch.tensor(train_labels)
val_sequences = pad_sequence([torch.tensor(seq) for seq in val_sequences], True)
val_labels = torch.tensor(val_labels)
train_dataset = SentimentDataset(train_sequences, train_labels)
train_loader = DataLoader(train_dataset, BATCH_SIZE, True)
val_dataset = SentimentDataset(val_sequences, val_labels)
val_loader = DataLoader(val_dataset, BATCH_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentimentLSTM(
    len(vocab) + 1, EMBEDDING_SIZE, HIDDEN_SIZE, NUM_CLASSES, vocab, DROPOUT
).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, fused=True
)

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    train_acc = 0.0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        train_loss += loss.item()
        train_acc += accuracy(outputs, labels)
    train_loss = train_loss / len(train_loader)
    train_acc = train_acc / len(train_loader)

    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            val_loss += loss.item()
            val_acc += accuracy(outputs, labels)
    val_loss = val_loss / len(val_loader)
    val_acc = val_acc / len(val_loader)

    print(f"Epoch: {epoch + 1}/{EPOCHS}")
    print(f"Training Loss: {train_loss:.4f}")
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    print("-" * 50)

torch.save(model, "sentiment_model.pt")
