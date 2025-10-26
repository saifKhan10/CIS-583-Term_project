import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    """
    Simple BiLSTM sentiment classifier.

    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        pad_idx: int,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Embedding matrix: [vocab_size, embedding_dim]
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx 
        )

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # Classifier head
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        """
        input_ids: [batch_size, seq_len] (each entry is a token ID)
        returns logits: [batch_size, output_dim]
        """

        # 1. Embed tokens
        embedded = self.embedding(input_ids)

        # outputs: [batch_size, seq_len, hidden_dim * 2] because bidirectional
        outputs, _ = self.lstm(embedded)

        #    This is a simple but common pooling strategy
        last_timestep = outputs[:, -1, :]  # [batch_size, hidden_dim * 2]

        # 4. Dropout for regularization
        dropped = self.dropout(last_timestep)

        # 5. Classifier head
        logits = self.fc(dropped)  # [batch_size, output_dim]

        return logits
