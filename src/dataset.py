"""
dataset.py
------------
Loads and preprocesses the IMDB dataset for sentiment analysis.
Uses Hugging Face Datasets + BERT Tokenizer.
Returns PyTorch-ready DataLoaders for training, validation, and testing.
"""

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import BertTokenizer
from src.config import CONFIG


def load_and_preprocess_data():
    """
    Loads IMDB dataset from Hugging Face and tokenizes it for BERT.
    Returns train, validation, and test DataLoaders.
    """
    # 1. Load dataset
    print("Loading IMDB dataset...")
    dataset = load_dataset("imdb")

    # 2. Initialize BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(CONFIG["bert_model_name"])

    # 3. Tokenization function
    def tokenize_function(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=CONFIG["max_len"]
        )

    # 4. Apply tokenizer to dataset
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # 5. Set PyTorch format (only keep needed columns)
    tokenized_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )

    # 6. Split training into train + validation (90/10 split)
    print("Splitting into train/val/test sets...")
    train_valid = tokenized_dataset["train"].train_test_split(test_size=0.1, seed=CONFIG["seed"])
    train_dataset = train_valid["train"]
    val_dataset = train_valid["test"]
    test_dataset = tokenized_dataset["test"]

    # 7. Create PyTorch DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"])

    print(f"Dataset ready! | Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Debug mode: Run this file directly to test data loading
    train_loader, val_loader, test_loader = load_and_preprocess_data()
    batch = next(iter(train_loader))
    print(f"\nBatch Keys: {batch.keys()}")
    print(f"Input IDs Shape: {batch['input_ids'].shape}")
    print(f"Attention Mask Shape: {batch['attention_mask'].shape}")
    print(f"Labels Shape: {batch['label'].shape}")
