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
│   ├── main.py                      # Entry point — orchestrates the pipeline  
│   
│── results/                     # Store logs, graphs, trained models  
│── reports/                     # IEEE report, drafts, figures  
│── requirements.txt             # Dependencies  
│── README.md                    # Project overview  
