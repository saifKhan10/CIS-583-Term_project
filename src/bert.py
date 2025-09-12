from transformers import BertForSequenceClassification
from src.config import CONFIG

def load_bert_model(num_labels: int = 2):
    return BertForSequenceClassification.from_pretrained(
        CONFIG["bert_model_name"],
        num_labels=num_labels
    )
