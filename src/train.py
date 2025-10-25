# train.py
import torch
from torch.optim import AdamW, Adam
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

from src.config import CONFIG


def _epoch(model, data_loader, device, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    phase = "TRAIN" if train_mode else "VAL"

    for batch in tqdm(data_loader, desc=phase, total=len(data_loader)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        if train_mode:
            optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        logits = outputs.logits

        if train_mode:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(1, len(data_loader))
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def train_bert(model, train_loader, val_loader, device, epochs: int, lr: float):
    optimizer = AdamW(model.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = _epoch(model, train_loader, device, optimizer)
        with torch.no_grad():
            val_loss, val_acc = _epoch(model, val_loader, device, optimizer=None)
        print(f"Epoch {ep}/{epochs} | train_loss={tr_loss:.4f} acc={tr_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f}")
    return model

def test_bert(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    print(f"Test | loss={total_loss/len(test_loader):.4f} acc={acc:.4f} f1={f1:.4f}")
    print(classification_report(all_labels, all_preds, digits=4))

############################################
# LSTM-specific training helpers
############################################

def _epoch_lstm(model, data_loader, device, criterion, optimizer=None):
    """
    Train or eval one epoch for the LSTM model.
    Returns (avg_loss, accuracy).
    NOTE: LSTM model returns logits directly, so we compute loss manually.
    """
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    phase = "TRAIN_LSTM" if train_mode else "VAL_LSTM"

    for batch in tqdm(data_loader, desc=phase, total=len(data_loader)):
        input_ids = batch["input_ids"].to(device)  # [batch, seq_len] token IDs
        labels = batch["label"].to(device)

        if train_mode:
            optimizer.zero_grad()

        logits = model(input_ids)                  # [batch, num_classes]
        loss = criterion(logits, labels)

        if train_mode:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / max(1, len(data_loader))
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def train_lstm(model, train_loader, val_loader, device):
    """
    Train the LSTM sentiment classifier.
    Returns the trained model.
    """
    lstm_cfg = CONFIG["lstm"]
    optimizer = Adam(model.parameters(), lr=lstm_cfg["lr"])
    criterion = nn.CrossEntropyLoss()

    last_val_loss = None
    last_val_acc = None

    for ep in range(1, CONFIG["epochs"] + 1):
        tr_loss, tr_acc = _epoch_lstm(model, train_loader, device, criterion, optimizer)
        with torch.no_grad():
            val_loss, val_acc = _epoch_lstm(model, val_loader, device, criterion, optimizer=None)

        print(f"[LSTM] Epoch {ep}/{CONFIG['epochs']} | "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f}")

        last_val_loss = val_loss
        last_val_acc = val_acc

    return model, last_val_loss, last_val_acc


def test_lstm(model, test_loader, device):
    """
    Evaluate the LSTM model on the test set.
    Returns (test_acc, test_f1)
    """
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="TEST_LSTM", total=len(test_loader)):
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"[LSTM TEST] loss={total_loss/len(test_loader):.4f} acc={acc:.4f} f1={f1:.4f}")
    print(classification_report(all_labels, all_preds, digits=4))

    return acc, f1
