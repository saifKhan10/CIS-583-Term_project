# main.py (top)
import torch
from transformers import BertTokenizer
from pathlib import Path
from src.config import CONFIG
from src.dataset import load_and_preprocess_data
from src.bert import load_bert_model
from src.train import train_bert
from src.train import test_bert



def main():
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

if __name__ == "__main__":
    main()
