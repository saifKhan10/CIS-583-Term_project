from asyncio import subprocess
import torch
import json
from transformers import BertTokenizer
from pathlib import Path
from torch.optim import AdamW


from src.config import CONFIG
from src.dataset import load_and_preprocess_data
from src.bert import load_bert_model
from src.train import train_bert
from src.train import test_bert

def ensure_lstm_metrics():
    lstm_metrics_path = Path("results/lstm_metrics.json")

    if lstm_metrics_path.exists():
        print("LSTM metrics already exist, skipping LSTM training.")
        return

    print("No LSTM metrics found. Running LSTM pipeline (src.run_lstm)...")
    subprocess.run(["python", "-m", "src.run_lstm"], check=True)

    if lstm_metrics_path.exists():
        print("LSTM metrics created successfully.")
    else:
        print("LSTM metrics still missing after run_lstm. Check run_lstm.py for errors.")

def save_json(obj, out_path):
    """
    Utility: write a dict to JSON on disk (indent=2).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"💾 Saved {out_path}")

def save_metrics(metrics: dict, out_path):
    """Save metrics (bert_metrics.json, comparison.json, etc.)"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Saved {out_path}")

def load_or_run_lstm():
    path = Path("results/lstm_metrics.json")
    if not path.exists():
        print("⚠ Could not find results/lstm_metrics.json after ensure_lstm_metrics()")
        return None
    with open(path, "r") as f:
        return json.load(f)

def main():

    load_or_run_lstm()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # data
    train_loader, val_loader, test_loader = load_and_preprocess_data()

    # model
    model = load_bert_model(num_labels=2).to(device)

    # train
    model = train_bert(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=CONFIG["epochs"],
        lr=CONFIG["bert_lr"],
    )

    # evaluate
    test_bert(model, test_loader, device)

    # save
    out_dir = Path(CONFIG["save_dir"]) / "bert_imdb"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Saving to:", out_dir)
    model.save_pretrained(out_dir)
    BertTokenizer.from_pretrained(CONFIG["bert_model_name"]).save_pretrained(out_dir)

    bert_metrics = {
        "model": "BERT (bert-base-uncased)",
        "epochs": CONFIG["epochs"],
        "batch_size": CONFIG["batch_size"],
        "max_len": CONFIG["max_len"],
        "learning_rate": CONFIG["bert_lr"],
        # Accuracy / F1 are printed by test_bert() but not returned.
        # If you later modify test_bert to return them, we can add them here.
    }
    save_json(bert_metrics, "results/bert_metrics.json")

    lstm_metrics = load_or_run_lstm()

    if lstm_metrics is not None:
        print("\n================== MODEL COMPARISON ==================")
        print(f"{'Metric':<18} {'LSTM':<12} {'BERT':<12}")
        print("------------------------------------------------------")

        # LSTM: these come from run_lstm.py output
        lstm_val_acc = lstm_metrics.get("val_acc_last_epoch", "n/a")
        lstm_test_acc = lstm_metrics.get("test_acc", "n/a")
        lstm_test_f1 = lstm_metrics.get("test_f1", "n/a")

        # BERT: since test_bert only prints, we don't have the exact numbers here
        # unless you modify test_bert to return them.
        # We'll leave placeholders for now.
        bert_val_acc = "see console"
        bert_test_acc = "see console"
        bert_test_f1 = "see console"

        print(f"{'Val Acc':<18} {lstm_val_acc:<12} {bert_val_acc:<12}")
        print(f"{'Test Acc':<18} {lstm_test_acc:<12} {bert_test_acc:<12}")
        print(f"{'Test F1':<18} {lstm_test_f1:<12} {bert_test_f1:<12}")
        print("======================================================\n")

        # 10. Save comparison.json for report/slides
        comparison = {
            "lstm": {
                "val_acc_last_epoch": lstm_val_acc,
                "test_acc": lstm_test_acc,
                "test_f1": lstm_test_f1,
                "embedding_dim": lstm_metrics.get("embedding_dim", None),
                "hidden_dim": lstm_metrics.get("hidden_dim", None),
                "epochs": lstm_metrics.get("epochs", None),
            },
            "bert": {
                "epochs": CONFIG["epochs"],
                "batch_size": CONFIG["batch_size"],
                "max_len": CONFIG["max_len"],
                "lr": CONFIG["bert_lr"],
                "note": "BERT test acc/f1 printed in console by test_bert()",
            },
        }
        save_json(comparison, "results/comparison.json")
    else:
        print("⚠ Skipping comparison output because LSTM metrics could not be loaded.")

if __name__ == "__main__":
    main()
