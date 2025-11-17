# READ ME
Hello This is the Term Project Repository for CIS 583

## Code Structure  
sentiment_analysis/  
│── src/  
│   ├── __init__.py  
│   ├── config.py                # Central configs, hyperparameters, paths  
│   ├── dataset.py               # Dataset loading, tokenization, preprocessing
│   ├── dataset_lstm.py          # LSTM Dataset
│   ├── lstm.py                  # Bi-LSTM architecture
│   ├── run_lstm.py              # Runs the LSTM Model
│   ├── bert.py                  # Transformer/BERT implementation  
│   ├── train.py                 # Training loop for both models  
│   ├── evaluate.py              # Metrics, confusion matrix, plots  
│   ├── main.py                      # Entry point — orchestrates the pipeline  
│   
│── results/                     # Store logs, graphs, trained models  
│── README.md                    # Project overview  
