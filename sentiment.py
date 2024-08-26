import torch


class SentimentLSTM(torch.nn.Module):
    def __init__(
        self, vocab_size, embedding_size, hidden_size, num_classes, vocab, dropout
    ):
        super(SentimentLSTM, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size)
        self.lstm = torch.nn.LSTM(
            embedding_size, hidden_size, batch_first=True, bidirectional=True
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_size * 2, num_classes)
        self.vocab = vocab

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

    def infer(self, sentence):
        sequence = []
        for word in sentence.split():
            sequence.append(self.vocab.get(word, 0))
        sequence = sequence + [0] * (50 - len(sequence))
        sequence = (
            torch.tensor(sequence).unsqueeze(0).to(next(self.parameters()).device)
        )
        with torch.no_grad():
            output = self(sequence)
        probs = torch.sigmoid(output).squeeze().tolist()
        labels = [
            "admiration",
            "amusement",
            "anger",
            "annoyance",
            "approval",
            "caring",
            "confusion",
            "curiosity",
            "desire",
            "disappointment",
            "disapproval",
            "disgust",
            "embarrassment",
            "excitement",
            "fear",
            "gratitude",
            "grief",
            "joy",
            "love",
            "nervousness",
            "optimism",
            "pride",
            "realization",
            "relief",
            "remorse",
            "sadness",
            "surprise",
            "neutral",
        ]
        results = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)
        return results[:1]
