import torch
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score

from src.config import CONFIG
from src.dataset import load_and_preprocess_data
from src.dataset_ltsm import load_lstm_dataloaders
from src.bert import load_bert_model
from src.lstm import LSTMClassifier


############################################
# Utility: save a confusion matrix plot
############################################
def plot_confusion_matrix(y_true, y_pred, title, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["neg", "pos"])

    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


############################################
# Utility: run model on a dataloader and collect preds/labels
############################################
def collect_preds_bert(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return all_labels, all_preds


def collect_preds_lstm(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            # dataset_lstm loaders only give "input_ids" and "label"
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return all_labels, all_preds


############################################
# Step 1. Load fine-tuned BERT from disk
############################################
def load_finetuned_bert(device):
    bert_dir = Path(CONFIG["save_dir"]) / "bert_imdb"

    model = load_bert_model(num_labels=2).to(device)

    # model.save_pretrained() wrote out a HuggingFace-style checkpoint into bert_dir.
    from transformers import BertForSequenceClassification
    finetuned = BertForSequenceClassification.from_pretrained(str(bert_dir)).to(device)
    return finetuned


############################################
# Step 2. Load trained LSTM
############################################
def load_trained_lstm(device):

    lstm_model_path = Path(CONFIG["lstm_model_file"])

    # load_lstm_dataloaders() returns loaders + vocab
    train_loader, val_loader, test_loader, vocab = load_lstm_dataloaders()

    lstm_cfg = CONFIG["lstm"]
    model = LSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=lstm_cfg["embedding_dim"],
        hidden_dim=lstm_cfg["hidden_dim"],
        output_dim=2,
        pad_idx=vocab["<PAD>"],
        dropout=lstm_cfg["dropout"]
    ).to(device)

    if lstm_model_path.exists():
        print(f"Loading saved LSTM weights from {lstm_model_path}")
        state = torch.load(lstm_model_path, map_location=device)
        model.load_state_dict(state)
    else:
        print("LSTM weights file not found. Using randomly initialized LSTM.")
        print("This means your confusion matrix for LSTM may not match the final metrics exactly.")

    return model, test_loader


############################################
# Step 3. Plot accuracy bar chart
############################################
def plot_accuracy_bar(lstm_acc, bert_acc, out_path):
    labels = ["LSTM", "BERT"]
    values = [lstm_acc, bert_acc]

    fig, ax = plt.subplots(figsize=(4,4))
    ax.bar(labels, values)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Sentiment Classification Accuracy (IMDB)")
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom')
    plt.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


############################################
# main() for evaluation pipeline
############################################
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    ########################################
    # BERT evaluation + confusion matrix
    ########################################
    print("\n=== Evaluating fine-tuned BERT on test set ===")
    # reuse the same dataloader code you use in main.py
    _, _, bert_test_loader = load_and_preprocess_data()

    bert_model = load_finetuned_bert(device)
    bert_true, bert_pred = collect_preds_bert(bert_model, bert_test_loader, device)

    bert_acc = accuracy_score(bert_true, bert_pred)
    bert_f1 = f1_score(bert_true, bert_pred)

    print(f"BERT Test Accuracy: {bert_acc:.4f}")
    print(f"BERT Test F1:       {bert_f1:.4f}")

    plot_confusion_matrix(
        bert_true,
        bert_pred,
        title="BERT Confusion Matrix (IMDB)",
        out_path="results/confusion_matrix_bert.png"
    )

    ########################################
    # LSTM evaluation + confusion matrix
    ########################################
    print("\n=== Evaluating LSTM on test set ===")
    lstm_model, lstm_test_loader = load_trained_lstm(device)

    lstm_true, lstm_pred = collect_preds_lstm(lstm_model, lstm_test_loader, device)

    lstm_acc = accuracy_score(lstm_true, lstm_pred)
    lstm_f1 = f1_score(lstm_true, lstm_pred, zero_division=0)

    print(f"LSTM Test Accuracy: {lstm_acc:.4f}")
    print(f"LSTM Test F1:       {lstm_f1:.4f}")

    plot_confusion_matrix(
        lstm_true,
        lstm_pred,
        title="LSTM Confusion Matrix (IMDB)",
        out_path="results/confusion_matrix_lstm.png"
    )

    ########################################
    # Accuracy bar chart (LSTM vs BERT)
    ########################################
    print("\n=== Generating comparison bar chart ===")
    plot_accuracy_bar(
        lstm_acc=lstm_acc,
        bert_acc=bert_acc,
        out_path="results/accuracy_comparison.png"
    )

    ########################################
    # Save summary metrics to disk
    ########################################
    summary = {
        "bert": {
            "test_acc": bert_acc,
            "test_f1": bert_f1,
        },
        "lstm": {
            "test_acc": lstm_acc,
            "test_f1": lstm_f1,
        }
    }
    with open("results/eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved results/eval_summary.json")

    print("\nEvaluation complete. Figures in results/ :")
    print(" - confusion_matrix_bert.png")
    print(" - confusion_matrix_lstm.png")
    print(" - accuracy_comparison.png")
    print(" - eval_summary.json")


if __name__ == "__main__":
    main()
