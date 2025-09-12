# READ ME
Hello This is the Term Project Repository for CIS 583

## Code Structure  
sentiment_analysis/  
│── src/  
│   ├── __init__.py  
│   ├── config.py                 # Central configs, hyperparameters, paths  
│   ├── dataset.py                # Dataset loading, tokenization, preprocessing  
│   ├── lstm_model.py             # Bi-LSTM architecture  
│   ├── bert_model.py            # Transformer/BERT implementation  
│   ├── train.py                 # Training loop for both models  
│   ├── evaluate.py              # Metrics, confusion matrix, plots  
│   ├── utils.py                 # Logging, visualization, helper functions  
│  
│── notebooks/                   # Jupyter notebooks for experimentation  
│── results/                     # Store logs, graphs, trained models  
│── reports/                     # IEEE report, drafts, figures  
│── requirements.txt             # Dependencies  
│── main.py                      # Entry point — orchestrates the pipeline  
│── README.md                    # Project overview  
