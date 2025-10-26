"""
config.py
----------
Central configuration for sentiment analysis project.
Contains dataset paths, model hyperparameters, and training settings.
"""

CONFIG = {
    # Dataset settings
    "dataset_name": "imdb",
    "dataset_path": "data/raw/imdb.csv",  # Only used if you download manually
    "max_len": 256,                       # Max tokens per review
    "batch_size": 16,                     # Reviews per batch

    # Training settings
    "epochs": 3,
    "seed": 42,

    # BERT model settings
    "bert_model_name": "bert-base-uncased",
    "bert_lr": 2e-5,

    # LSTM model settings (for later enhancement)
    "lstm": {
        "embedding_dim": 200,
        "hidden_dim": 256,
        "dropout": 0.3,
        "lr": 1e-3
    },

    # Paths for saving models and results
    "save_dir": "results/",
    "bert_model_file": "results/bert_model.pt",
    "lstm_model_file": "results/lstm_model.pt"
}
