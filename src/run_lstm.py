import torch
from pathlib import Path
import json
from datetime import datetime

from src.config import CONFIG
from src.dataset_ltsm import load_lstm_dataloaders
from src.lstm import LSTMClassifier
from src.train import train_lstm, test_lstm


def save_metrics(metrics: dict, out_path="results/lstm_metrics.json"):
    """
    Save final LSTM metrics (val acc, test acc, etc.) to a JSON file
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metrics["timestamp"] = datetime.now().isoformat(timespec="seconds")

    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved metrics to {out_path}")


def run_lstm():
    # pick device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # data loaders + vocab
    train_loader, val_loader, test_loader, vocab = load_lstm_dataloaders()

    # build the model
    lstm_cfg = CONFIG["lstm"]
    model = LSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=lstm_cfg["embedding_dim"],
        hidden_dim=lstm_cfg["hidden_dim"],
        output_dim=2,
        pad_idx=vocab["<PAD>"],
        dropout=lstm_cfg["dropout"]
    ).to(device)

    # train
    model, val_loss, val_acc = train_lstm(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )

    # test
    test_acc, test_f1 = test_lstm(model, test_loader, device)

    # print summary
    print("==== FINAL LSTM RESULTS ====")
    print(f"Val Acc:  {val_acc:.4f}")
    print(f"Test Acc: {test_acc:.4f}")
    print(f"Test F1:  {test_f1:.4f}")

    # save metrics for report/README
    save_metrics({
        "model": "LSTMClassifier",
        "epochs": CONFIG["epochs"],
        "batch_size": CONFIG["batch_size"],
        "max_len": CONFIG["max_len"],
        "learning_rate": lstm_cfg["lr"],
        "embedding_dim": lstm_cfg["embedding_dim"],
        "hidden_dim": lstm_cfg["hidden_dim"],
        "dropout": lstm_cfg["dropout"],
        "val_acc_last_epoch": val_acc,
        "test_acc": test_acc,
        "test_f1": test_f1,
    })


if __name__ == "__main__":
    run_lstm()
