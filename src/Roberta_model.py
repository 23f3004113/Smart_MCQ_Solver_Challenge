from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


MODEL_NAME="roberta-base"



def load_roberta():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )


    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=5
    )


    return tokenizer,model
