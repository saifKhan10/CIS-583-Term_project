import re
import random
from collections import Counter
from typing import List, Dict, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from src.config import CONFIG

############################################################
# 1. Basic tokenizer for LSTM
############################################################

def simple_tokenize(text: str) -> List[str]:
    """
    Lowercase, remove non-letters/numbers except basic punctuation,
    and split on whitespace.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9'?!.,]+", " ", text)
    tokens = text.strip().split()
    return tokens


############################################################
# 2. Vocab building
############################################################

def build_vocab(texts: List[str], min_freq: int = 2, max_size: int = 20000) -> Dict[str, int]:
    """
    Build a word -> index mapping.
    """
    counter = Counter()
    for t in texts:
        counter.update(simple_tokenize(t))

    # start with special tokens
    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
    }
    # add most common tokens
    for word, freq in counter.most_common():
        if freq < min_freq:
            continue
        if len(vocab) >= max_size:
            break
        vocab[word] = len(vocab)

    return vocab


def encode_tokens(tokens: List[str], vocab: Dict[str, int]) -> List[int]:
    """
    Convert list of tokens into list of vocab indices.
    Unknown words -> <UNK> index.
    """
    unk_idx = vocab["<UNK>"]
    return [vocab.get(tok, unk_idx) for tok in tokens]


def pad_or_truncate(ids: List[int], max_len: int, pad_idx: int) -> List[int]:
    """
    Make every sequence exactly max_len long.
    Shorter -> pad with pad_idx
    Longer  -> cut off the end
    """
    if len(ids) < max_len:
        return ids + [pad_idx] * (max_len - len(ids))
    else:
        return ids[:max_len]


############################################################
# 3. Torch Dataset for LSTM
############################################################

class LSTMSentimentDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], vocab: Dict[str, int], max_len: int):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        self.pad_idx = vocab["<PAD>"]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        label = self.labels[idx]

        tokens = simple_tokenize(text)
        token_ids = encode_tokens(tokens, self.vocab)
        padded_ids = pad_or_truncate(token_ids, self.max_len, self.pad_idx)

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }


############################################################
# 4. High-level loader: returns loaders for train/val/test
############################################################

def load_lstm_dataloaders():
    """
    Loads IMDB with Hugging Face,
    builds vocab from training text,
    tokenizes to integer ID sequences,
    and returns train/val/test DataLoaders for the LSTM model.
    """
    print("Loading IMDB dataset for LSTM...")
    dataset = load_dataset("imdb")

    # Hugging Face IMDB structure:
    # dataset["train"]   -> 25k examples
    # dataset["test"]    -> 25k examples
    # each item: {"text": str, "label": 0 or 1}

    # We'll split train into (train/val)
    all_train_texts = [ex["text"] for ex in dataset["train"]]
    all_train_labels = [ex["label"] for ex in dataset["train"]]

    # deterministic split: 90% train / 10% val
    random.seed(CONFIG["seed"])
    indices = list(range(len(all_train_texts)))
    random.shuffle(indices)

    split_point = int(0.9 * len(indices))
    train_idx = indices[:split_point]
    val_idx = indices[split_point:]

    train_texts = [all_train_texts[i] for i in train_idx]
    train_labels = [all_train_labels[i] for i in train_idx]

    val_texts = [all_train_texts[i] for i in val_idx]
    val_labels = [all_train_labels[i] for i in val_idx]

    test_texts = [ex["text"] for ex in dataset["test"]]
    test_labels = [ex["label"] for ex in dataset["test"]]

    # 1) Build vocab ONLY on training text (important for realism)
    print("Building vocab...")
    vocab = build_vocab(train_texts,
                        min_freq=2,
                        max_size=20000)

    pad_idx = vocab["<PAD>"]
    max_len = CONFIG["max_len"] 

    # 2) Create Dataset objects
    print("Encoding splits...")
    train_dataset = LSTMSentimentDataset(train_texts, train_labels, vocab, max_len)
    val_dataset   = LSTMSentimentDataset(val_texts,   val_labels,   vocab, max_len)
    test_dataset  = LSTMSentimentDataset(test_texts,  test_labels,  vocab, max_len)

    # 3) Wrap with DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False
    )

    print(f"LSTM data ready! | Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    print(f"Vocab size: {len(vocab)} | Pad idx: {pad_idx} | Max len: {max_len}")

    return train_loader, val_loader, test_loader, vocab
